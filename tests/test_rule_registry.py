"""Tests for the Rule Registry module (FORGE Modulo 4)."""
import numpy as np
import pandas as pd
import pytest

from forgedge.event_discovery.models import (
    ActivationStats,
    EventCandidate,
    EventComponent,
    GateResult,
)
from forgedge.rule_discovery import BacktestParams, RuleDiscoveryConfig, run_backtest
from forgedge.rule_discovery.models import (
    RuleDiscoveryResponse,
    ValidatedRule,
)
from forgedge.resolver import PipelineContext, resolve, resolve_config
from forgedge.rule_registry import (
    RegistryConfig,
    RuleRegistry,
    RuleSubmission,
    classify_genericity,
    flat_table,
    jaccard_by_date,
    recalibrate_candidate,
)
from forgedge.rule_registry.correlation import (
    correlation_matrices,
    gain_correlation_by_date,
)
from forgedge.rule_registry.cross_ticker import transfer_bar
from forgedge.rule_registry.dedup import mark_duplicates
from forgedge.rule_registry.models import RuleDocument
from forgedge.unset import UNSET


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _profitable_frame(seed: int, n: int = 1500, feat_lo: float = 0.0,
                      feat_hi: float = 1.0, drift: float = 0.05) -> pd.DataFrame:
    """OHLC table where a low ``feat`` precedes a real up-move.

    ``feat`` is drawn on the per-ticker range ``[feat_lo, feat_hi]`` so the
    distribution differs across tickers (exercising threshold recalibration),
    while the *pattern* — low feat → up-move — holds on every ticker.
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(feat_lo, feat_hi, n)
    q20 = np.quantile(feat, 0.20)
    close = np.empty(n)
    close[0] = 100.0
    boost = np.zeros(n)
    for i in range(n):
        if feat[i] < q20:
            for j in range(i + 1, min(i + 16, n)):
                boost[j] += drift / 15.0
    noise = rng.normal(0.0, 0.002, n)
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + boost[i] + noise[i])
    high = close * (1.0 + 0.01)
    low = close * (1.0 - 0.01)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": open_, "high": high, "low": low, "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)), "feat": feat,
        }
    )


def _make_candidate(event_id: str, threshold: float) -> EventCandidate:
    """A one-component identity event: ``feat < threshold``."""
    comp = EventComponent(
        source_feature="feat",
        transform="identity",
        transform_params={},
        transformed_col="feat",
        threshold=threshold,
        threshold_type="distributional_p20",
        direction="below",
        event_type="threshold",
        expression=f"feat < {threshold:.6g}",
        source_cols=[],
    )
    stats = ActivationStats(
        n_activations=100, n_active_months=12, zero_months=0,
        max_monthly_share=0.2, mean_tpm=8.0,
    )
    gate = GateResult(
        passed=True, n_activations=100, n_active_months=12,
        max_monthly_share=0.2, mean_tpm=8.0,
    )
    return EventCandidate(
        event_id=event_id, status="CANDIDATE", components=[comp],
        expression=comp.expression, activation_stats=stats, consistency_gate=gate,
    )


def _make_response(frame, candidate, params, *, verdict="EDGE",
                   alpha_id="ALPHA-001", asset="ADAUSDC") -> RuleDiscoveryResponse:
    """Build a Rule Discovery response with a real in-sample summary."""
    f = frame.copy()
    f.index = pd.DatetimeIndex(pd.to_datetime(f["open_dt"]))
    sig = candidate.apply(f)
    f["__s__"] = sig.fillna(0).to_numpy()
    summary = run_backtest(f, "__s__", params, timestamp_col="open_dt")
    return RuleDiscoveryResponse(
        date="2026-06-13", verdict=verdict, alpha_id=alpha_id,
        asset=asset, timeframe="1H",
        validated_rule=ValidatedRule(
            expression=candidate.expression,
            event_candidate_id=candidate.event_id, params=params,
        ),
        in_sample_summary=summary, walk_forward=None,
        statistical_validation=None, regime_analysis=None,
    )


def _submission(ticker, frame, threshold, *, verdict="EDGE", event_id=None,
                alpha_id=None) -> RuleSubmission:
    event_id = event_id or f"EVT-{ticker}"
    alpha_id = alpha_id or f"ALPHA-{ticker}"
    cand = _make_candidate(event_id, threshold)
    params = BacktestParams(buy_type="limit", buy_drop_pct=0.005,
                            buy_delay_bar=6, sell_pct=0.03, target_h=12)
    resp = _make_response(frame, cand, params, verdict=verdict,
                          alpha_id=alpha_id, asset=ticker)
    return RuleSubmission(ticker=ticker, response=resp, candidate=cand)


def _doc(rule_id, dates, gains, pf, ticker="ADAUSDC"):
    """A bare RuleDocument for correlation / dedup unit tests."""
    return RuleDocument(
        rule_id=rule_id, expression="feat < 0.2", source_ticker=ticker,
        source_alpha_id="A", verdict="EDGE", grade="B",
        activation_idx=list(range(len(dates))), activation_dates=list(dates),
        gains=list(gains),
        params={"entry_mode": "limit", "buy_drop_pct": 0.005, "sell_pct": 0.03,
                "target_h": 12, "fee": 0.002, "direction": "long"},
        stats={"pf": pf, "win_rate": 0.7, "total_trades": len(dates),
               "expectancy": 0.01, "zero_months": 0, "dsr": 1.2, "grade": "B",
               "monthly_gains": [0.1, -0.05, 0.2]},
        regime={"type": "agnostic", "avoid_in": []},
    )


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_jaccard_basic(self):
        assert jaccard_by_date(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)
        assert jaccard_by_date(["a"], ["a"]) == 1.0
        assert jaccard_by_date([], []) == 0.0
        assert jaccard_by_date(["a", "b"], ["c", "d"]) == 0.0

    def test_gain_correlation_identical_rules_is_one(self):
        dates = [f"2024-01-{d:02d}T00:00:00" for d in range(1, 21)]
        gains = list(np.linspace(-0.1, 0.1, 20))
        a = _doc("R1", dates, gains, pf=3.0)
        b = _doc("R2", dates, gains, pf=2.0)
        assert gain_correlation_by_date(a, b, min_active=10) == pytest.approx(1.0)

    def test_gain_correlation_few_points_is_zero(self):
        dates = ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
        a = _doc("R1", dates, [0.1, -0.1], pf=3.0)
        b = _doc("R2", dates, [0.1, -0.1], pf=2.0)
        assert gain_correlation_by_date(a, b, min_active=10) == 0.0

    def test_matrices_symmetric_with_unit_diagonal(self):
        dates = [f"2024-01-{d:02d}T00:00:00" for d in range(1, 21)]
        g = list(np.linspace(-0.1, 0.1, 20))
        docs = [_doc("R1", dates, g, 3.0), _doc("R2", dates[5:], g[5:], 2.0),
                _doc("R3", dates[::2], g[::2], 1.5)]
        m = correlation_matrices(docs, min_active=5)
        assert np.allclose(m.jaccard.values, m.jaccard.values.T)
        assert np.allclose(m.spearman.values, m.spearman.values.T)
        assert np.allclose(np.diag(m.jaccard.values), 1.0)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_weaker_rule_flagged_duplicate(self):
        dates = [f"2024-01-{d:02d}T00:00:00" for d in range(1, 21)]
        docs = [_doc("R1", dates, [0.01] * 20, pf=3.0),
                _doc("R2", dates, [0.01] * 20, pf=2.0)]
        m = correlation_matrices(docs, min_active=5)
        mark_duplicates(docs, m, overlap_threshold=0.70)
        assert docs[0].is_duplicate is False
        assert docs[1].is_duplicate is True
        assert docs[1].duplicate_of == "R1"

    def test_chain_points_at_immediate_dominant(self):
        dates = [f"2024-01-{d:02d}T00:00:00" for d in range(1, 21)]
        # All three fully overlap → chain by PF: R1 > R2 > R3.
        docs = [_doc("R1", dates, [0.01] * 20, pf=3.0),
                _doc("R2", dates, [0.01] * 20, pf=2.5),
                _doc("R3", dates, [0.01] * 20, pf=2.0)]
        m = correlation_matrices(docs, min_active=5)
        mark_duplicates(docs, m, overlap_threshold=0.70)
        assert docs[0].duplicate_of is None
        assert docs[1].duplicate_of == "R1"
        assert docs[2].duplicate_of == "R2"  # immediate dominant, not the root

    def test_independent_rules_not_flagged(self):
        a = [f"2024-01-{d:02d}T00:00:00" for d in range(1, 11)]
        b = [f"2024-02-{d:02d}T00:00:00" for d in range(1, 11)]
        docs = [_doc("R1", a, [0.01] * 10, 3.0), _doc("R2", b, [0.01] * 10, 2.0)]
        m = correlation_matrices(docs, min_active=5)
        mark_duplicates(docs, m, overlap_threshold=0.70)
        assert all(d.is_duplicate is False for d in docs)


# ---------------------------------------------------------------------------
# Genericity classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_badges(self):
        thr = 2.0 / 3.0  # the default: exactly two-thirds
        assert classify_genericity(3, 3, thr) == "GENERIC"
        assert classify_genericity(2, 3, thr) == "PARTIAL"
        assert classify_genericity(1, 3, thr) == "SPECIFIC"
        assert classify_genericity(0, 3, thr) == "ISOLATED"
        assert classify_genericity(2, 2, thr) == "GENERIC"
        assert classify_genericity(1, 2, thr) == "SPECIFIC"


# ---------------------------------------------------------------------------
# The transfer criterion — the F8 case
# ---------------------------------------------------------------------------

class TestTransferBar:
    """`PASS ⟺ pf_other >= floor AND pf_other >= retention · pf_home`.

    The bar is computed from a resolved config, so these run against the values
    a real session would use (floor 1.5, retention 0.8) rather than literals
    pinned here.
    """

    @staticmethod
    def _cfg(**overrides):
        return resolve_config(RegistryConfig(**overrides), "registry")

    def _passes(self, pf_home: float, pf_other: float, **overrides) -> bool:
        return pf_other >= transfer_bar(pf_home, self._cfg(**overrides))

    def test_a_rule_that_holds_its_edge_elsewhere_is_generic(self):
        """The row a single 2.0 bar got backwards: a PARTIAL-EDGE rule admitted
        at 1.5, keeping the same PF away from home, is the *definition* of a
        pattern that transfers — and was failed everywhere, on every ticker,
        making the whole PARTIAL-EDGE class ISOLATED by construction."""
        assert self._passes(pf_home=1.6, pf_other=1.6)

    def test_a_rule_that_gave_up_a_third_of_its_edge_is_not(self):
        """The other row it got backwards.  2.05 clears an absolute 2.0, so this
        used to PASS — while the rule had lost 32% of what it was worth at
        home, which is what "does not transfer" looks like."""
        assert not self._passes(pf_home=3.0, pf_other=2.05)
        assert self._passes(pf_home=3.0, pf_other=2.6)      # −13%: it holds

    def test_the_absolute_floor_still_bites(self):
        """Retention alone would let a rule decay with its home PF forever:
        1.2 retains 75% of 1.6, but 1.2 is not a tradeable profit factor.  The
        floor is the half that says "good enough *here*", independent of how
        good it was there."""
        assert not self._passes(pf_home=1.6, pf_other=1.2)

    def test_a_weak_rule_can_be_generic_without_the_test_getting_easier(self):
        """The objection to deriving the bar from the home PF: weaker rules get
        a lower bar.  They do — and it is still bounded below by the floor, so
        the easiest possible test is the one M3 already used to admit the rule.

        The rule may read GENERIC while its M3 verdict stays PARTIAL-EDGE and
        its grade stays what it is.  Transfer and quality are different axes
        and the registry records both; this one is not asked to carry the other.
        """
        assert self._passes(pf_home=1.5, pf_other=1.5)
        assert not self._passes(pf_home=1.5, pf_other=1.4)

    def test_the_bar_is_the_higher_half_and_is_recorded(self):
        cfg = self._cfg()
        assert transfer_bar(1.5, cfg) == pytest.approx(1.5)    # floor wins
        assert transfer_bar(4.0, cfg) == pytest.approx(3.2)    # retention wins

    def test_an_undefined_home_pf_leaves_only_the_floor(self):
        """A rule with no losing trade at home has an infinite PF; 80% of
        infinity is not a bar anyone can clear, and treating it as one would
        silently mark the best rules ISOLATED."""
        cfg = self._cfg()
        assert transfer_bar(float("inf"), cfg) == pytest.approx(1.5)
        assert transfer_bar(float("nan"), cfg) == pytest.approx(1.5)

    def test_the_floor_follows_m3_rather_than_being_a_second_opinion(self):
        """Both halves are resolved, so moving M3's admission bar moves the
        registry's — which is the point: `cross_pf_threshold` was an
        independent copy that happened to be stricter."""
        assert self._cfg().cross_pf_threshold == pytest.approx(1.5)

        bundle = {"rule_discovery": RuleDiscoveryConfig(), "registry": RegistryConfig()}
        bundle["rule_discovery"].criteria.partial_min_profit_factor = 1.8
        out, _trace, _v = resolve(bundle, PipelineContext())
        assert out["registry"].cross_pf_threshold == pytest.approx(1.8)

    def test_an_explicit_threshold_is_still_honoured(self):
        assert self._cfg(cross_pf_threshold=2.5).cross_pf_threshold == pytest.approx(2.5)
        assert not self._passes(pf_home=3.0, pf_other=2.4, cross_pf_threshold=2.5)


# ---------------------------------------------------------------------------
# Threshold recalibration
# ---------------------------------------------------------------------------

class TestRecalibration:
    def test_identity_threshold_shifts_to_target_percentile(self):
        src = _profitable_frame(seed=1, feat_lo=0.0, feat_hi=1.0)
        tgt = _profitable_frame(seed=2, feat_lo=0.0, feat_hi=2.0)  # 2× the range
        # Source threshold = 20th percentile of feat on the source.
        thr = float(np.quantile(src["feat"], 0.20))
        cand = _make_candidate("EVT", thr)
        adapted = recalibrate_candidate(cand, src, tgt)
        new_thr = adapted.components[0].threshold
        expected = float(np.quantile(tgt["feat"], 0.20))
        # Recalibrated threshold lands on the target's matching percentile…
        assert new_thr == pytest.approx(expected, rel=0.05)
        # …which is higher because the target range is wider.
        assert new_thr > thr
        assert "feat" in adapted.expression

    def test_original_candidate_not_mutated(self):
        src = _profitable_frame(seed=1)
        tgt = _profitable_frame(seed=2, feat_hi=2.0)
        thr = float(np.quantile(src["feat"], 0.20))
        cand = _make_candidate("EVT", thr)
        recalibrate_candidate(cand, src, tgt)
        assert cand.components[0].threshold == thr  # untouched


# ---------------------------------------------------------------------------
# End-to-end registry
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    frames = {
        "ADAUSDC": _profitable_frame(seed=1, feat_lo=0.0, feat_hi=1.0),
        "SOLUSDC": _profitable_frame(seed=2, feat_lo=0.0, feat_hi=2.0),
        "BTCUSDC": _profitable_frame(seed=3, feat_lo=0.0, feat_hi=0.5),
    }
    subs = []
    for t, f in frames.items():
        thr = float(np.quantile(f["feat"], 0.20))
        subs.append(_submission(t, f, thr))
    return RuleRegistry(subs, frames, RegistryConfig()).run()


class TestRuleRegistry:
    def test_ingestion_assigns_rule_ids(self, registry):
        ids = [d.rule_id for d in registry.documents]
        assert ids == ["RULE_ADA_01", "RULE_SOL_01", "RULE_BTC_01"]

    def test_non_edge_not_ingested(self):
        f = _profitable_frame(seed=1)
        thr = float(np.quantile(f["feat"], 0.20))
        sub = _submission("ADAUSDC", f, thr, verdict="NON-EDGE")
        reg = RuleRegistry([sub], {"ADAUSDC": f}).run()
        assert reg.documents == []

    def test_cross_ticker_scored(self, registry):
        for d in registry.documents:
            assert d.cross_ticker_total == 2  # two other tickers
            assert 0 <= d.cross_ticker_score <= 2
            assert d.classification in {"GENERIC", "PARTIAL", "SPECIFIC", "ISOLATED"}
            # every other ticker produced a result with an adapted expression
            assert set(d.cross_ticker.keys()) <= {"ADAUSDC", "SOLUSDC", "BTCUSDC"}
            for res in d.cross_ticker.values():
                assert res.verdict in {"PASS", "FAIL"}

    def test_profitable_pattern_generalises(self, registry):
        # The low-feat→up-move pattern holds on every ticker, so the rules
        # should clear the cross-ticker PF on the others and read GENERIC.
        ada = next(d for d in registry.documents if d.rule_id == "RULE_ADA_01")
        assert ada.stats["pf"] > 1.0
        assert ada.cross_ticker_score >= 1

    def test_flat_table_has_cross_ticker_columns(self, registry):
        df = registry.flat_table()
        assert len(df) == 3
        for col in ("rule_id", "expression", "source_ticker", "is_duplicate",
                    "cross_ticker_score", "is_generic", "classification"):
            assert col in df.columns
        # cross-ticker block: one column group per ticker
        assert "pf_SOLUSDC" in df.columns
        assert "verdict_BTCUSDC" in df.columns

    def test_html_report_is_self_contained(self, registry):
        html = registry.html_report(timeframe="1H")
        assert html.startswith("<!doctype html>")
        assert "FORGE Report" in html
        assert "Cross-ticker summary" in html
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        for d in registry.documents:
            assert d.rule_id in html

    def test_missing_frame_raises(self):
        f = _profitable_frame(seed=1)
        thr = float(np.quantile(f["feat"], 0.20))
        sub = _submission("ADAUSDC", f, thr)
        with pytest.raises(ValueError):
            RuleRegistry([sub], {"SOLUSDC": f})  # wrong ticker frame

    def test_a_catalogued_rule_records_values_not_sentinels(self):
        """A submission assembled by hand — a caller replaying a rule, a test —
        carries whatever `BacktestParams` it was given, unresolved fields
        included.  Ingestion is where configuration becomes a *catalogue
        record*, and a record that reads `fee=UNSET` is not a record of
        anything: the flat table and the HTML report both render this dict.
        """
        f = _profitable_frame(seed=1)
        thr = float(np.quantile(f["feat"], 0.20))
        sub = _submission("ADAUSDC", f, thr)
        assert sub.response.validated_rule.params.fee is UNSET   # as handed in

        reg = RuleRegistry([sub], {"ADAUSDC": f}).run()
        assert reg.documents[0].params["fee"] == pytest.approx(0.002)
        assert flat_table(reg.documents).iloc[0]["fee"] == pytest.approx(0.002)

    def test_export_csv(self, tmp_path):
        f = _profitable_frame(seed=1)
        thr = float(np.quantile(f["feat"], 0.20))
        sub = _submission("ADAUSDC", f, thr)
        reg = RuleRegistry([sub], {"ADAUSDC": f},
                           RegistryConfig(export_format="csv")).run()
        path = reg.export(str(tmp_path / "out.csv"))
        assert path.endswith(".csv")
        loaded = pd.read_csv(path)
        assert loaded.iloc[0]["rule_id"] == "RULE_ADA_01"


# ---------------------------------------------------------------------------
# Event-signal reconstruction (no blind reindex of a stale cache)
# ---------------------------------------------------------------------------

class TestSignalReconstruction:
    """Ingestion must evaluate the event on the candles it observes, not blindly
    reindex the candidate's cached ``event_series`` (which would collapse the
    signal to all-zeros when the index differs)."""

    def test_reevaluated_when_source_index_differs(self):
        f = _profitable_frame(seed=1)
        thr = float(np.quantile(f["feat"], 0.20))
        cand = _make_candidate("EVT-ADA", thr)
        # Cache the activations on the original timestamps…
        base = f.copy()
        base.index = pd.DatetimeIndex(pd.to_datetime(base["open_dt"]))
        cand.event_series = cand.apply(base)
        # …then hand the registry the SAME data shifted 10 years forward, so the
        # prepared index no longer matches the cache. A blind reindex would map
        # every bar to NaN→inactive and ledger a rule that never fires.
        shifted = f.copy()
        shifted["open_dt"] = pd.to_datetime(shifted["open_dt"]) + pd.Timedelta(days=3650)
        params = BacktestParams(buy_type="limit", buy_drop_pct=0.005,
                                buy_delay_bar=6, sell_pct=0.03, target_h=12)
        resp = _make_response(shifted, cand, params, alpha_id="A-ADA", asset="ADAUSDC")
        sub = RuleSubmission(ticker="ADAUSDC", response=resp, candidate=cand)

        with pytest.warns(UserWarning, match="differs from"):
            reg = RuleRegistry([sub], {"ADAUSDC": shifted}).run()

        # The recomputed ledger (activation arrays) must not collapse to zero —
        # a blind reindex would have produced an empty ledger.
        d = reg.documents[0]
        assert len(d.gains) > 0
        assert len(d.activation_dates) == len(d.gains)

    def test_stale_cache_matches_clean_recompute(self):
        # A stale (mismatched) cache must yield exactly the same ledger as a
        # candidate that carries no cache at all — i.e. the cache is ignored.
        f = _profitable_frame(seed=3)
        thr = float(np.quantile(f["feat"], 0.20))

        clean = _submission("ADAUSDC", f, thr, event_id="EVT-clean", alpha_id="A-clean")
        stale = _submission("ADAUSDC", f, thr, event_id="EVT-stale", alpha_id="A-stale")
        # An all-ones cache on a positional index that won't match the prepared
        # DatetimeIndex; if trusted it would fire on every bar.
        stale.candidate.event_series = pd.Series(
            np.ones(len(f)), index=pd.RangeIndex(len(f))
        )

        reg = RuleRegistry([clean, stale], {"ADAUSDC": f}).run()
        d_clean, d_stale = reg.documents
        assert len(d_stale.gains) > 0
        assert d_stale.activation_dates == d_clean.activation_dates
        assert d_stale.gains == d_clean.gains

    def test_matching_cache_used_as_fast_path(self):
        # When the cache index matches the observed frame exactly it is trusted
        # verbatim (no warning, no recompute) — proven by a deliberately empty
        # cache that yields an empty ledger.  The frame already carries the
        # DatetimeIndex so the registry keeps it as-is.
        f = _profitable_frame(seed=2)
        f.index = pd.DatetimeIndex(pd.to_datetime(f["open_dt"]))
        thr = float(np.quantile(f["feat"], 0.20))
        sub = _submission("ADAUSDC", f, thr)
        sub.candidate.event_series = pd.Series(0.0, index=f.index)  # matches prepared index

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            reg = RuleRegistry([sub], {"ADAUSDC": f}).run()

        # The empty cache is trusted verbatim → the ledger is empty (no recompute).
        assert len(reg.documents[0].gains) == 0
