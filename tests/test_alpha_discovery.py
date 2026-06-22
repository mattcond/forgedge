"""Tests for the Alpha Discovery module (FORGE Modulo 2)."""
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    AlphaDiscovery,
    DiscoveryConfig,
    EventDiscovery,
    PromotionThresholds,
    TargetConfig,
)
from forgedge.alpha_discovery import stats
from forgedge.alpha_discovery.market_structure import analyse_market_structure
from forgedge.alpha_discovery.models import (
    DerivedTarget,
    EventStats,
    ICResult,
    OOSValidation,
    RegimeAnalysis,
)
from forgedge.alpha_discovery.target import (
    binary_target,
    forward_log_returns,
    forward_returns,
)
from forgedge.event_discovery.models import GateParams


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _predictive_kpi_table(
    n: int = 6000, seed: int = 7, include_noise: bool = False
) -> pd.DataFrame:
    """KPI table with a genuine mean-reversion signal in ``feat``.

    The one-step-ahead return is driven by ``-k * (feat - 0.5)``, so a low
    ``feat`` predicts a positive forward return.  Event Discovery should pick
    up ``feat < <low>`` events; Alpha Discovery should derive a *long* target
    at a short horizon for them (the advantage is injected at lag 1, so the
    statistical separation decays as the horizon grows) and confirm it OOS —
    the signal is stationary across the whole table.
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)

    # Return at t+1 depends on feat[t]: low feat -> positive next return.
    k = 0.02
    noise = rng.normal(0.0, 0.004, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -k * (feat[:-1] - 0.5) + noise[1:]
    close = 100.0 * np.exp(np.cumsum(r))

    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    data = {
        "open_dt": dates,
        "close": close,
        "volume": np.abs(rng.normal(1e6, 1e5, n)),
        "feat": feat,
    }
    if include_noise:
        # Same distribution as 'feat' but with no effect on returns: its
        # events pass Event Discovery's structural gates yet carry no alpha.
        data["nfeat"] = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame(data)


def _make_candidates(df: pd.DataFrame):
    ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
    return ed, ed.run()


# ---------------------------------------------------------------------------
# stats — pure-numpy primitives
# ---------------------------------------------------------------------------

class TestStats:
    def test_spearman_monotonic_is_one(self):
        x = np.arange(50.0)
        rho, p = stats.spearmanr(x, x ** 2)  # strictly increasing
        assert rho == pytest.approx(1.0)
        assert p == pytest.approx(0.0)

    def test_spearman_sign_and_significance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        y = -0.4 * x + rng.normal(size=300)
        rho, p = stats.spearmanr(x, y)
        assert rho < 0
        assert p < 0.01

    def test_spearman_drops_nan_pairs(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([2.0, 1.0, 9.0, np.nan, 6.0])
        rho, p = stats.spearmanr(x, y)
        assert math.isfinite(rho)

    def test_spearman_too_few_points(self):
        rho, p = stats.spearmanr([1.0, 2.0], [2.0, 1.0])
        assert math.isnan(rho) and math.isnan(p)

    def test_ttest_greater_detects_shift(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.5, 1.0, 200)
        b = rng.normal(0.0, 1.0, 200)
        t, p = stats.ttest_ind(a, b, alternative="greater")
        assert t > 0 and p < 0.001

    def test_ttest_greater_one_sided_complement(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.0, 1.0, 150)
        b = rng.normal(0.5, 1.0, 150)  # a < b, so 'greater' should be ~1
        _, p = stats.ttest_ind(a, b, alternative="greater")
        assert p > 0.9

    def test_cohens_d_zero_for_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert stats.cohens_d(a, a) == pytest.approx(0.0)

    def test_benjamini_hochberg_basic(self):
        p = [0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9]
        mask = stats.benjamini_hochberg(p, 0.10)
        assert mask.tolist() == [True, True, True, True, False, False, False]

    def test_benjamini_hochberg_ignores_nan(self):
        p = [0.001, np.nan, 0.9]
        mask = stats.benjamini_hochberg(p, 0.10)
        assert mask[0] and not mask[1] and not mask[2]

    def test_benjamini_hochberg_none_significant(self):
        mask = stats.benjamini_hochberg([0.6, 0.7, 0.8], 0.10)
        assert not mask.any()

    def test_betai_symmetry(self):
        # I_x(a,b) + I_{1-x}(b,a) == 1
        assert stats.betai(2.0, 3.0, 0.3) + stats.betai(3.0, 2.0, 0.7) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Step 1 — forward returns and the binary target
# ---------------------------------------------------------------------------

class TestTargetPrimitives:
    def test_forward_returns_columns_match_grid(self):
        close = pd.Series(np.linspace(100, 110, 200))
        fwd = forward_returns(close, [1, 6, 24])
        assert list(fwd.columns) == [1, 6, 24]

    def test_forward_returns_no_lookahead_tail_is_nan(self):
        close = pd.Series(np.linspace(100, 110, 100))
        fwd = forward_returns(close, [10])
        assert fwd[10].iloc[-10:].isna().all()
        assert fwd[10].iloc[:-10].notna().all()

    def test_forward_returns_values(self):
        close = pd.Series([100.0, 110.0, 121.0])
        fwd = forward_returns(close, [1])
        assert fwd[1].iloc[0] == pytest.approx(0.10)
        assert fwd[1].iloc[1] == pytest.approx(0.10)

    def test_forward_returns_rejects_nonpositive_horizon(self):
        with pytest.raises(ValueError):
            forward_returns(pd.Series([1.0, 2.0]), [0])

    def test_binary_target_no_lookahead_tail_is_nan(self):
        close = pd.Series(np.linspace(100, 110, 100))
        tgt, _ = binary_target(close, 10, 0.01, "long")
        assert tgt.iloc[-10:].isna().all()

    def test_long_target_triggers_on_rise(self):
        # Monotonic +2%/bar rise: every early bar reaches +4% within 24 bars.
        close = pd.Series(100.0 * (1.02 ** np.arange(60)))
        tgt, base = binary_target(close, 24, 0.04, "long", target_mode="abs")
        assert tgt.iloc[0] == 1.0
        assert base > 0.9

    def test_short_direction_triggers_on_fall(self):
        close = pd.Series(100.0 * (0.98 ** np.arange(60)))
        tgt, base = binary_target(close, 24, 0.04, "short", target_mode="abs")
        assert tgt.iloc[0] == 1.0
        assert base > 0.9

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            binary_target(pd.Series([1.0, 2.0, 3.0]), 24, 0.04, "sideways")

    def test_invalid_target_mode_raises(self):
        with pytest.raises(ValueError):
            binary_target(pd.Series(100.0 * (1.01 ** np.arange(40))),
                          5, 0.04, "long", target_mode="relative")

    def test_long_proj_log_no_excess_on_uniform_trend(self):
        """A uniform +2%/bar trend is pure trend, no excess: ABS fires on every
        bar (rides the trend) but PROJ_LOG fires on none (price == trend)."""
        close = pd.Series(100.0 * (1.02 ** np.arange(120)))
        _, base_abs = binary_target(close, 10, 0.04, "long", target_mode="abs")
        _, base_proj = binary_target(close, 10, 0.04, "long", target_mode="proj")
        assert base_abs > 0.9          # ABS credits the trend
        assert base_proj == 0.0        # PROJ strips it — no genuine excess

    def test_proj_log_long_excess_above_trend(self):
        """A counter-trend bounce above the prevailing trend triggers PROJ_LOG
        where it reflects genuine excess (and the series has the 3*h warmup)."""
        rng = np.random.default_rng(0)
        n = 200
        h = 10
        # gentle uptrend + a sharp temporary spike at t=120 (excess over trend)
        close = pd.Series(100.0 * (1.002 ** np.arange(n)))
        close.iloc[121:126] *= 1.20    # forward window of t≈120 beats the trend
        tgt, _ = binary_target(close, h, 0.05, "long", target_mode="proj")
        assert tgt.iloc[110:121].max() == 1.0   # the pre-spike bars see the excess

    def test_proj_log_short_reverts_to_abs(self):
        close = pd.Series(100.0 * (0.99 ** np.arange(120)) +
                          np.sin(np.arange(120)))
        abs_t, _ = binary_target(close, 10, 0.04, "short", target_mode="abs")
        proj_t, _ = binary_target(close, 10, 0.04, "short", target_mode="proj")
        pd.testing.assert_series_equal(abs_t, proj_t)

    def test_proj_log_warmup_warning_on_short_series(self, caplog):
        """n < 3*h: PROJ reverts to ABS with a warning."""
        import logging
        close = pd.Series(100.0 * (1.01 ** np.arange(20)))   # 20 < 3*10
        with caplog.at_level(logging.WARNING):
            proj_t, _ = binary_target(close, 10, 0.04, "long", target_mode="proj")
        abs_t, _ = binary_target(close, 10, 0.04, "long", target_mode="abs")
        pd.testing.assert_series_equal(proj_t, abs_t)
        assert any("PROJ needs" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 2 — market structure
# ---------------------------------------------------------------------------

class TestMarketStructure:
    def test_mean_reverting_series_detected(self):
        # OU process around a constant level → Hurst well below 0.5.
        rng = np.random.default_rng(3)
        x = [np.log(100.0)]
        for _ in range(3000):
            x.append(x[-1] + 0.1 * (np.log(100.0) - x[-1]) + 0.01 * rng.standard_normal())
        close = pd.Series(np.exp(x))
        ms = analyse_market_structure(close, close.pct_change())
        assert ms.hurst < 0.5
        assert ms.expected_family == "mean_reversion"

    def test_autocorr_lags_present(self):
        df = _predictive_kpi_table(n=2000)
        close = df.set_index("open_dt")["close"]
        ms = analyse_market_structure(close, close.pct_change(), acf_lags=[1, 6, 24])
        assert set(ms.autocorr) == {1, 6, 24}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_invalid_train_ratio_raises(self):
        with pytest.raises(ValueError):
            AlphaConfig(train_ratio=0.0)
        with pytest.raises(ValueError):
            AlphaConfig(train_ratio=1.2)

    def test_invalid_horizon_grid_raises(self):
        with pytest.raises(ValueError):
            AlphaConfig(horizon_grid=())
        with pytest.raises(ValueError):
            AlphaConfig(horizon_grid=(0, 4))

    def test_horizon_grid_sorted_and_deduplicated(self):
        cfg = AlphaConfig(horizon_grid=(24, 1, 24, 6))
        assert cfg.horizon_grid == (1, 6, 24)


# ---------------------------------------------------------------------------
# End-to-end — Alpha Discovery over Event Discovery output
# ---------------------------------------------------------------------------

class TestAlphaDiscoveryEndToEnd:
    @pytest.fixture(scope="class")
    def fitted(self):
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig(asset="SYN"))
        contracts = ad.run()
        return df, cands, ad, contracts

    def test_runs_and_returns_one_contract_per_candidate(self, fitted):
        _, cands, ad, contracts = fitted
        assert len(contracts) == len(cands)

    def test_finds_at_least_one_promoted_alpha(self, fitted):
        _, _, ad, _ = fitted
        promoted = ad.promoted_contracts()
        assert len(promoted) >= 1
        assert all(c.status == "HYPOTHESIS" for c in promoted)

    def test_promoted_alpha_is_built_on_feat(self, fitted):
        _, _, ad, _ = fitted
        feats = {c.underlying_feature.feature for c in ad.promoted_contracts()}
        # The injected signal lives in 'feat' (or features derived from it).
        assert any("feat" in f for f in feats)

    def test_derived_target_matches_injected_signal(self, fitted):
        """The signal is injected at lag 1, so the derived horizon must be
        short and the direction must follow the sign of the advantage."""
        _, _, ad, _ = fitted
        long_promoted = [c for c in ad.promoted_contracts() if c.direction == "long"]
        assert long_promoted
        for c in long_promoted:
            dt = c.derived_target
            assert dt.holding_period_h <= 8
            assert dt.sell_pct > 0
            assert dt.mean_advantage > 0

    def test_direction_is_sign_of_mean_advantage(self, fitted):
        _, _, _, contracts = fitted
        for c in contracts:
            dt = c.derived_target
            if dt.direction == "long":
                assert dt.mean_advantage > 0
                assert dt.sell_pct > 0          # MFE-based, always positive
            elif dt.direction == "short":
                assert dt.mean_advantage < 0
                assert dt.sell_pct > 0          # MFE-based, always positive

    def test_target_profile_covers_grid(self, fitted):
        _, _, ad, contracts = fitted
        grid = set(ad.config.horizon_grid)
        for c in contracts[:10]:
            assert set(c.derived_target.advantage_by_h) == grid
            assert set(c.derived_target.t_stat_by_h) == grid
            assert set(c.derived_target.score_by_h) == grid
            assert c.derived_target.holding_period_h in grid

    def test_promoted_alpha_has_lift_and_oos_confirmation(self, fitted):
        _, _, ad, _ = fitted
        # Best-scoring contract (by composite) should have strong evidence.
        best = max(ad.promoted_contracts(), key=lambda c: c.alpha_score.composite_score)
        assert best.event_stats.win_rate > best.event_stats.base_rate
        assert best.oos_validation is not None
        assert best.oos_validation.mean_advantage > 0

    def test_summary_sorted_by_score(self, fitted):
        _, _, ad, _ = fitted
        s = ad.summary()
        assert list(s.columns)[:3] == ["alpha_id", "status", "promoted"]
        assert {"holding_period_h", "sell_pct", "direction", "oos_passed"} <= set(s.columns)
        scores = s["composite_score"].to_numpy()
        assert np.all(np.diff(scores) <= 1e-9)       # descending

    def test_contract_dict_serialises(self, fitted):
        _, _, ad, _ = fitted
        c = ad.promoted_contracts()[0]
        d = c.to_contract_dict()
        assert d["status"] == "HYPOTHESIS"
        assert d["event_expression"]
        assert "statistical_evidence" in d
        assert d["derived_target"]["holding_period_h"] == c.derived_target.holding_period_h
        assert d["derived_target"]["base_rate"] == pytest.approx(c.base_rate)
        # OOS recorded but not required to pass (non-blocking gate).
        assert d["oos_validation"] is not None

    def test_noise_candidates_get_low_scores_with_diagnostics(self):
        """Candidates built on a feature with no effect on returns are promoted
        (A-D grading; all directed contracts go to Rule Discovery) but must
        score in the C/D range and carry non-blocking diagnostic notes."""
        df = _predictive_kpi_table(include_noise=True)
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig())
        contracts = ad.run()

        noise = [c for c in contracts if "nfeat" in c.underlying_feature.feature]
        assert noise
        # All directed noise candidates are promoted (non-blocking design).
        directed = [c for c in noise if c.derived_target.direction in ("long", "short")]
        assert all(c.promoted for c in directed)
        # They must score poorly — noise has near-zero IC, lift, Cohen's d.
        scores = [c.alpha_score.composite_score for c in noise]
        assert np.mean(scores) < 0.35
        # Diagnostics should flag weak evidence on most noise candidates.
        with_diagnostics = [c for c in noise if c.rejection_reasons]
        assert len(with_diagnostics) >= 0.5 * len(noise)

    def test_expression_is_propagated_unchanged(self, fitted):
        _, cands, ad, contracts = fitted
        by_id = {c.event_id: c.expression for c in cands}
        for contract in contracts:
            assert contract.event_expression == by_id[contract.event_candidate_id]

    def test_split_respects_train_ratio(self, fitted):
        df, _, ad, _ = fitted
        assert ad.split_idx == int(round(len(df) * ad.config.train_ratio))


# ---------------------------------------------------------------------------
# Promotion / FDR / OOS behaviour
# ---------------------------------------------------------------------------

class TestPromotionGates:
    def test_raw_pvalue_mode_is_more_permissive_than_fdr(self):
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)

        fdr = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            thresholds=PromotionThresholds(use_fdr=True, fdr_q=0.05)))
        raw = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            thresholds=PromotionThresholds(use_fdr=False, max_p_value=0.05)))
        fdr.run(); raw.run()
        assert len(raw.promoted_contracts()) >= len(fdr.promoted_contracts())

    def test_strict_thresholds_become_diagnostics(self):
        """Strict IC/lift/Cohen's d thresholds are non-blocking: all directed
        contracts are promoted but carry diagnostic notes."""
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            thresholds=PromotionThresholds(min_lift=0.95, min_cohens_d=5.0)))
        ad.run()
        directed = [c for c in ad._contracts
                    if c.derived_target.direction in ("long", "short")]
        assert all(c.promoted for c in directed)
        has_diagnostic = any(
            any("[diagnostic]" in r for r in c.rejection_reasons)
            for c in directed
        )
        assert has_diagnostic

    def test_train_ratio_one_disables_oos(self):
        """With no held-out tail there is no OOS validation — contracts carry
        ``oos_validation=None`` and the OOS gate is skipped."""
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig(train_ratio=1.0))
        contracts = ad.run()
        assert all(c.oos_validation is None for c in contracts)
        assert ad.promoted_contracts()  # promotion still possible

    def test_impossible_oos_threshold_is_diagnostic_only(self):
        """oos_max_p=0 makes every OOS confirmation fail; with non-blocking
        design, all directed contracts are still promoted but carry OOS notes."""
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            thresholds=PromotionThresholds(oos_max_p=0.0)))
        ad.run()
        directed = [c for c in ad._contracts
                    if c.derived_target.direction in ("long", "short")]
        assert all(c.promoted for c in directed)
        with_oos = [c for c in ad._contracts if c.oos_validation is not None]
        assert any(
            any("OOS" in r for r in c.rejection_reasons) for c in with_oos
        )


# ---------------------------------------------------------------------------
# Scope metadata is traceability-only
# ---------------------------------------------------------------------------

class TestScopeMetadataInert:
    def test_metadata_does_not_affect_measurements(self):
        """asset/exchange/timeframe/fee are stamped into the contract but must
        leave every statistical measure untouched."""
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)

        a = AlphaDiscovery(ed.df, cands, AlphaConfig(
            asset="AAA", exchange="ex1", timeframe="1H", fee_per_side=0.001))
        b = AlphaDiscovery(ed.df, cands, AlphaConfig(
            asset="ZZZ", exchange="ex2", timeframe="4H", fee_per_side=0.009))
        a.run(); b.run()

        sa = a.summary().drop(columns=["alpha_id"])
        sb = b.summary().drop(columns=["alpha_id"])
        pd.testing.assert_frame_equal(sa, sb)

        ca, cb = a._contracts[0], b._contracts[0]
        assert (ca.asset, ca.timeframe, ca.fee_per_side) == ("AAA", "1H", 0.001)
        assert (cb.asset, cb.timeframe, cb.fee_per_side) == ("ZZZ", "4H", 0.009)


# ---------------------------------------------------------------------------
# Regime sensitivity
# ---------------------------------------------------------------------------

class TestRegimeSensitivity:
    def test_no_regime_column_yields_unknown_dependency(self):
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig())
        ad.run()
        c = ad.summary().iloc[0]
        assert c["regime_dependency"] == "unknown"
        # Score still computed (breadth term renormalised away).
        assert 0.0 <= c["composite_score"] <= 1.0

    def test_regime_column_is_used(self):
        df = _predictive_kpi_table()
        # Inject a simple two-regime column.
        regime = pd.Series(
            pd.Categorical(
                np.where(np.arange(len(df)) % 2 == 0, "BEAR", "BULL"),
                categories=["BEAR", "BULL"], ordered=True,
            )
        )
        df = df.copy()
        df["regime"] = regime.values
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df, cands, AlphaConfig())
        ad.run()
        promoted = ad.promoted_contracts()
        assert promoted
        ra = promoted[0].regime_analysis
        assert ra.dependency_type in ("agnostic", "conditional", "specific", "broken")
        assert {r.regime for r in ra.per_regime} <= {"BEAR", "BULL"}


# ---------------------------------------------------------------------------
# No-recompute contract — events and regimes are consumed, never re-derived
# ---------------------------------------------------------------------------

class TestNoRecompute:
    def test_events_come_from_stored_series_not_apply(self, monkeypatch):
        """Fast path: when the observed candles are identical to the event's
        (the sequential ``ed.df`` case), the cached ``event_series`` is reused
        verbatim and ``EventCandidate.apply`` must never run."""
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        assert all(c.event_series is not None for c in cands)

        def _boom(self, frame):
            raise AssertionError("Alpha Discovery recomputed an event via apply()")

        from forgedge.event_discovery.models import EventCandidate
        monkeypatch.setattr(EventCandidate, "apply", _boom)

        ad = AlphaDiscovery(ed.df, cands, AlphaConfig())
        contracts = ad.run()
        assert len(contracts) == len(cands)

    def test_features_read_from_table_when_present(self, monkeypatch):
        """With ed.df as input every feature column exists — no replay needed."""
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)

        import forgedge.alpha_discovery.discovery as disc

        def _boom(comp, frame):
            raise AssertionError(
                f"Alpha Discovery rebuilt feature '{comp.source_feature}' "
                "despite it being available in the table"
            )

        monkeypatch.setattr(disc, "build_feature_series", _boom)

        ad = AlphaDiscovery(ed.df, cands, AlphaConfig())
        contracts = ad.run()
        assert len(contracts) == len(cands)

    def test_apply_fallback_when_series_missing(self):
        """Candidates serialised without event_series still work via replay."""
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        stripped = cands[:5]
        for c in stripped:
            c.event_series = None

        ad = AlphaDiscovery(ed.df, stripped, AlphaConfig())
        contracts = ad.run()
        assert len(contracts) == 5

    def test_activation_counts_match_event_discovery(self):
        """The activations Alpha Discovery sees are Event Discovery's, bar for
        bar, restricted to the in-sample window."""
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig())
        ad.run()
        split = ad.split_idx
        n = len(df)
        for cand, contract in zip(cands, ad._contracts):
            stored = cand.event_series.fillna(0).astype(bool)
            # The forward window of every IS bar is complete (split + h < n),
            # so n_activations equals the stored IS activation count exactly.
            assert split + contract.derived_target.holding_period_h < n
            assert contract.event_stats.n_activations == int(stored.iloc[:split].sum())


# ---------------------------------------------------------------------------
# The event is consumed as an activation function on the observed candles
# ---------------------------------------------------------------------------

class TestEventAppliedAsFunction:
    """Regression guard for the event/Alpha temporal-consistency fix.

    When Event Discovery builds an event on one candle set and Alpha Discovery
    observes a *different* one, Alpha must evaluate the event as an activation
    function on the candles it observes — not reindex the cached activations of
    a foreign candle set (which would silently force every non-overlapping bar
    to "inactive").  The fast path (identical candles) still reuses the cache.
    """

    def _identity_candidate(self, cands):
        """A single-component, identity-transform, level (non-crossing) event.

        For such an event the activation function is a pure pointwise threshold,
        so the expected activations can be computed by an oracle that does not
        go through ``apply`` — the comparison is not tautological.
        """
        for c in cands:
            comp = c.components[0]
            if (len(c.components) == 1 and comp.transform == "identity"
                    and comp.event_type != "crossing"):
                return c
        return None

    @staticmethod
    def _identity_oracle(frame, comp):
        s = frame[comp.source_feature]
        hit = (s < comp.threshold) if comp.direction == "below" else (s > comp.threshold)
        return hit.astype(float).where(s.notna(), np.nan)

    @pytest.fixture(scope="class")
    def split_setup(self):
        # Event on the first half (set A); Alpha will observe the full table (B).
        full = _predictive_kpi_table(n=6000)
        A = full.iloc[:3000].copy()
        ed = EventDiscovery(A, DiscoveryConfig(
            timestamp_col="open_dt",
            gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0),
        ))
        cands = ed.run()
        cand = self._identity_candidate(cands)
        assert cand is not None, "expected a single-component identity candidate"
        return full, cand

    def test_event_series_equals_function_on_observed_candles(self, split_setup):
        full, cand = split_setup
        ad = AlphaDiscovery(full.copy(), [cand], AlphaConfig())
        used = ad._event_series(cand).reindex(ad._frame.index)
        oracle = self._identity_oracle(ad._frame, cand.components[0])
        pd.testing.assert_series_equal(
            used.fillna(0.0), oracle.fillna(0.0), check_names=False
        )

    def test_event_fires_on_non_common_tail(self, split_setup):
        full, cand = split_setup
        ad = AlphaDiscovery(full.copy(), [cand], AlphaConfig())
        used = ad._event_series(cand).reindex(ad._frame.index)
        tail = ad._frame.index.difference(cand.event_series.index)
        # Old reindex behaviour forced this to 0; the event genuinely fires here.
        assert used.reindex(tail).fillna(0).sum() > 0

    def test_overlap_still_matches_stored(self, split_setup):
        full, cand = split_setup
        ad = AlphaDiscovery(full.copy(), [cand], AlphaConfig())
        used = ad._event_series(cand).reindex(ad._frame.index)
        common = ad._frame.index.intersection(cand.event_series.index)
        pd.testing.assert_series_equal(
            used.reindex(common).fillna(0.0),
            cand.event_series.reindex(common).fillna(0.0),
            check_names=False,
        )

    def test_contract_counts_include_tail_activations(self, split_setup):
        full, cand = split_setup
        ad = AlphaDiscovery(full.copy(), [cand], AlphaConfig())
        ad.run()
        split = ad.split_idx
        stored_is = int(
            cand.event_series.reindex(ad._frame.index).fillna(0).iloc[:split].sum()
        )
        # The IS window now sees the tail activations the old code dropped.
        assert ad._contracts[0].event_stats.n_activations > stored_is

    def test_warns_on_candle_mismatch(self, split_setup):
        full, cand = split_setup
        ad = AlphaDiscovery(full.copy(), [cand], AlphaConfig())
        with pytest.warns(UserWarning, match="activation function"):
            ad._event_series(cand)

    def test_fast_path_identical_candles_uses_cache_silently(self):
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig())
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # no warning on the fast path
            used = ad._event_series(cands[0])
        pd.testing.assert_series_equal(used, cands[0].event_series)


# ---------------------------------------------------------------------------
# Persistence — the Alpha Contract artifact round-trips through a pickle
# ---------------------------------------------------------------------------

class TestPersist:
    @pytest.fixture(scope="class")
    def contract(self):
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN"))
        ad.run()
        return ad._contracts[0]

    def test_persist_writes_pickle(self, contract, tmp_path):
        path = tmp_path / "alpha.pkl"
        assert contract.persist(path) is None
        assert path.exists() and path.stat().st_size > 0

    def test_persist_roundtrip_preserves_contract(self, contract, tmp_path):
        path = tmp_path / "alpha.pkl"
        contract.persist(path)
        loaded = pickle.loads(path.read_bytes())

        assert loaded.alpha_id == contract.alpha_id
        assert loaded.status == contract.status
        assert loaded.promoted == contract.promoted
        assert loaded.event_expression == contract.event_expression
        assert loaded.direction == contract.direction
        dt, ldt = contract.derived_target, loaded.derived_target
        assert ldt.holding_period_h == dt.holding_period_h
        assert ldt.direction == dt.direction
        assert ldt.advantage_by_h == dt.advantage_by_h
        assert loaded.alpha_score.grade == contract.alpha_score.grade
        assert loaded.rejection_reasons == contract.rejection_reasons

    def test_persist_accepts_str_path(self, contract, tmp_path):
        path = str(tmp_path / "as_str.pkl")
        contract.persist(path)
        loaded = pickle.loads(open(path, "rb").read())
        assert loaded.alpha_id == contract.alpha_id


# ---------------------------------------------------------------------------
# Direction from excess log-return — the drift must not dictate direction
# (issue #88)
# ---------------------------------------------------------------------------

def _log_sufficient_stats(close: pd.Series, horizons):
    """Mirror ``AlphaDiscovery.run`` log-space prep for a unit test on
    ``_derive_target`` — the whole series is treated as in-sample."""
    L = forward_log_returns(close, horizons).to_numpy()
    F = forward_returns(close, horizons).to_numpy()
    valid = np.isfinite(L) & np.isfinite(F)
    L0 = np.where(valid, L, 0.0)
    cnt_t = valid.sum(axis=0).astype(float)
    sum_t = L0.sum(axis=0)
    return valid, L0, cnt_t, sum_t, F


def _naive_excess_t(active, valid, L0, cnt_t, sum_t):
    """The pre-fix naive excess t-stat ``Δ_h / (σ_cond/√n)`` — assumes the
    overlapping forward windows are independent, so it inflates on long
    horizons for clustered events.  Used only to demonstrate the rotation
    null deflates that inflation."""
    af = active.astype(float)
    cnt_a = af @ valid
    sum_a = af @ L0
    sumsq_a = af @ (L0 * L0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_c = np.where(cnt_a > 0, sum_a / np.maximum(cnt_a, 1), np.nan)
        mu_b = sum_t / np.maximum(cnt_t, 1)
        var = (sumsq_a - cnt_a * mu_c ** 2) / np.maximum(cnt_a - 1, 1)
        se = np.sqrt(var / np.maximum(cnt_a, 1))
        return (mu_c - mu_b) / se


def _trending_reversal_table(n=4000, seed=11, drift=0.004, shock=0.012):
    """Strong upward drift with an event that predicts a *short-term reversal*.

    ``feat > 0.8`` marks the event; the bar after an active bar gets the drift
    minus ``shock`` (a one-step underperformance).  The excess log-return is
    therefore negative at every horizon (drift cancels in the excess), but the
    *raw* conditional return is dominated by the drift on long horizons — the
    exact configuration that made the old ``|mean|/√h`` criterion derive a
    spurious *long*.
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    active = feat > 0.8
    noise = rng.normal(0.0, 0.002, n)
    r = np.full(n, drift) + noise
    r[1:][active[:-1]] -= shock          # reversal on the bar after the event
    r[0] = 0.0
    close = pd.Series(100.0 * np.exp(np.cumsum(r)))
    return close, active


def _drift_series(n=4000, seed=5, drift=0.004, vol=0.01):
    rng = np.random.default_rng(seed)
    r = np.full(n, drift) + rng.normal(0.0, vol, n)
    return pd.Series(100.0 * np.exp(np.cumsum(r))), rng


def _clustered_mask(rng, n, rate=0.05, burst=30):
    """Activations in bursts of consecutive bars — maximal forward-window
    overlap, independent of the returns."""
    m = np.zeros(n, bool)
    i = 0
    while i < n:
        if rng.random() < rate / burst:
            m[i:i + burst] = True
            i += burst
        else:
            i += 1
    return m


class TestExcessLogReturnDirection:
    horizons = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48]

    def test_direction_is_excess_not_drift(self):
        """The reversal event must derive *short* from the negative excess,
        even though the asset's drift makes the raw conditional return positive
        on the horizon the old criterion would have selected."""
        close, active = _trending_reversal_table()
        valid, L0, cnt_t, sum_t, F = _log_sufficient_stats(close, self.horizons)

        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005,
        )

        # Excess log-return is negative -> short, by construction.
        assert dt.direction == "short"
        assert dt.mean_advantage < 0
        assert all(v <= 1e-9 for v in dt.advantage_by_h.values())

        # The bug it fixes: the *raw* (simple) conditional return is dominated
        # by the drift at long horizons, so |mean|/√h would have chosen a long
        # horizon with a positive mean -> a spurious "long".
        af = active.astype(float)
        raw_mean = (af @ np.where(valid, F, 0.0)) / np.maximum(af @ valid, 1)
        h_arr = np.array(self.horizons, dtype=float)
        old_score = np.abs(raw_mean) / np.sqrt(h_arr)
        j_old = int(np.nanargmax(old_score))
        assert raw_mean[j_old] > 0          # old criterion -> spurious long

    def test_pure_drift_excess_is_negligible_vs_real_edge(self):
        """An event independent of returns on a drifting series carries (almost)
        no excess: its |Δ_h*| is an order of magnitude below the reversal
        event's, so the drift never manufactures a strong spurious edge."""
        # require_significant=False so the raw excess at h* is exposed for the
        # comparison (the significance gate is exercised separately below).
        close, rng = _drift_series(seed=3)
        active = rng.random(len(close)) < 0.05   # fires independently of returns
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        noise_dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005, require_significant=False,
        )

        close_r, active_r = _trending_reversal_table()
        v, l0, ct, st, _ = _log_sufficient_stats(close_r, self.horizons)
        real_dt = AlphaDiscovery._derive_target(
            active_r, v, l0, ct, st, self.horizons,
            close_r.to_numpy(), 0.5, 0.005, require_significant=False,
        )
        assert abs(noise_dt.mean_advantage) < 0.2 * abs(real_dt.mean_advantage)

    def _weak_event(self):
        """A bursty no-edge event on a drifting series for which no horizon
        clears Benjamini-Hochberg (statistically_weak) — found deterministically
        under a fixed seed."""
        close, rng = _drift_series(seed=5)
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        for _ in range(30):
            active = _clustered_mask(rng, len(close))
            if active.sum() < 30:
                continue
            dt = AlphaDiscovery._derive_target(
                active, valid, L0, cnt_t, sum_t, self.horizons,
                close.to_numpy(), 0.5, 0.005, require_significant=False,
            )
            if dt.statistically_weak:
                return close, active, valid, L0, cnt_t, sum_t
        raise AssertionError("no statistically_weak clustered event found")

    def test_statistically_weak_yields_undetermined(self):
        """Default gate (require_significant_direction): when no horizon clears
        BH, the direction is 'undetermined' instead of a coin-flip read off
        argmax|z| — the RSI>80 (DOGE) behaviour the user expects."""
        close, active, valid, L0, cnt_t, sum_t = self._weak_event()
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005,   # require_significant defaults True
        )
        assert dt.statistically_weak
        assert dt.h_sig == ()
        assert dt.direction == "undetermined"
        assert math.isnan(dt.mean_advantage)

    def test_require_significant_false_restores_legacy_direction(self):
        """The toggle re-enables the non-blocking behaviour: the same weak event
        is assigned a direction (subject only to min_direction_t)."""
        close, active, valid, L0, cnt_t, sum_t = self._weak_event()
        legacy = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005,
            require_significant=False, min_direction_t=0.0,
        )
        assert legacy.statistically_weak          # still flagged as weak…
        assert legacy.direction in ("long", "short")   # …but a direction stands

    def test_significant_event_keeps_direction(self):
        """A genuine edge (BH-significant horizon) is unaffected by the gate."""
        close, active = _trending_reversal_table()
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005,
        )
        assert not dt.statistically_weak
        assert dt.holding_period_h in dt.h_sig
        assert dt.direction == "short"

    def test_clustered_noise_not_falsely_significant(self):
        """Regression for issue #88's clustered-event pathology: a bursty event
        with no real edge would, under a naive t-stat, be pinned to the long
        grid edge and flagged significant.  The rotation null — which keeps the
        event's clustering — deflates that inflation, so it is (almost always)
        marked statistically_weak."""
        close, rng = _drift_series(seed=5)
        n = len(close)
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)

        not_weak = 0
        naive_significant = 0
        deflation = []
        runs = 0
        for _ in range(60):
            active = _clustered_mask(rng, n)
            if active.sum() < 30:
                continue
            runs += 1
            dt = AlphaDiscovery._derive_target(
                active, valid, L0, cnt_t, sum_t, self.horizons,
                close.to_numpy(), 0.5, 0.005,
            )
            if not dt.statistically_weak:
                not_weak += 1
            nt = _naive_excess_t(active, valid, L0, cnt_t, sum_t)
            if abs(nt[int(np.nanargmax(np.abs(nt)))]) > 2.5:
                naive_significant += 1
            # rotation z at the longest horizon is heavily deflated vs naive t
            deflation.append(abs(dt.t_stat_by_h[48]) / (abs(nt[-1]) + 1e-9))

        assert runs >= 30
        # The naive statistic is spuriously significant on most clustered events…
        assert naive_significant / runs > 0.5
        # …while the rotation null keeps the spurious-significant rate low.
        assert not_weak / runs < 0.30
        # And it deflates the inflated long-horizon statistic by a large factor.
        assert np.median(deflation) < 0.5

    def test_direction_floor_forces_undetermined(self):
        """A ``min_direction_t`` floor above |z_h*| refuses to assign a
        direction — the contract path for an edge too weak to call."""
        close, active = _trending_reversal_table()
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        base = dict(close_is=close.to_numpy())
        derived = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons, **base
        )
        floor = abs(derived.t_stat_by_h[derived.holding_period_h]) + 1.0
        gated = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            min_direction_t=floor, **base
        )
        assert derived.direction == "short"          # without the floor
        assert gated.direction == "undetermined"     # with it
        assert math.isnan(gated.mean_advantage)
        # The diagnostic profile is still populated for transparency.
        assert set(gated.advantage_by_h) == set(self.horizons)

    def test_real_excess_clears_gate_and_is_significant(self):
        """The reversal event has a real edge: its selected horizon is in
        ``h_sig`` (FDR passes) and the rotation p-value at ``h*`` is small."""
        close, active = _trending_reversal_table()
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005,
        )
        assert not dt.statistically_weak
        assert dt.holding_period_h in dt.h_sig
        assert dt.p_value_by_h[dt.holding_period_h] < 0.10

    def test_h_star_maximises_abs_z_over_h_sig(self):
        """``h*`` is the argmax of |z_h| restricted to the significant set."""
        close, active = _trending_reversal_table()
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons,
            close.to_numpy(), 0.5, 0.005,
        )
        # score_by_h holds |z_h|; h* maximises it over the significant horizons.
        best = max(dt.h_sig, key=lambda h: dt.score_by_h[h])
        assert dt.holding_period_h == best
        assert dt.score_by_h[dt.holding_period_h] == pytest.approx(
            abs(dt.t_stat_by_h[dt.holding_period_h])
        )

    def test_rotation_null_is_deterministic(self):
        """The rotation null evaluates every circular shift exactly (FFT), so it
        is reproducible without a seed."""
        close, active = _trending_reversal_table()
        valid, L0, cnt_t, sum_t, _ = _log_sufficient_stats(close, self.horizons)
        kw = dict(close_is=close.to_numpy())
        a = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons, **kw
        )
        b = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, self.horizons, **kw
        )
        assert a.p_value_by_h == b.p_value_by_h
        assert a.t_stat_by_h == b.t_stat_by_h


# ---------------------------------------------------------------------------
# Composite score / grade — significance must shape the grade (issue #91)
# ---------------------------------------------------------------------------

def _score_inputs(ic=0.05, lift=0.15, cohens_d=0.40, z=1.5, breadth=float("nan"),
                  weak=False, h=3):
    """Minimal (ICResult, EventStats, RegimeAnalysis, DerivedTarget, OOS) tuple
    for a direct ``_score`` call.  ``breadth=nan`` drops the regime term."""
    ic_res = ICResult("feat", ic, 0.01, 400, True, ic, 0.8, True)
    ev = EventStats(100, 0.40, 0.25, lift, 0.01, cohens_d, 2.0, 0.02)
    regime = RegimeAnalysis([], "unknown", [], [], breadth)
    derived = DerivedTarget(
        h, 0.02, "long", 0.005, {h: 0.005}, {h: z}, {h: abs(z)}, {h: 0.04},
        (h,) if not weak else (), weak,
    )
    return ic_res, ev, regime, derived


def _oos(passed):
    return OOSValidation(100, 30, 0.01, 2.0, 0.02, 0.40, 0.25, 0.15, passed)


class TestScoreGrading:
    @pytest.fixture(scope="class")
    def ad(self):
        df = _predictive_kpi_table(n=400)
        return AlphaDiscovery(df, [], AlphaConfig())

    def test_negative_cohens_d_penalises_not_neutral(self, ad):
        """A negative Cohen's d (conditioned group worse than background) must
        drag the score *below* the d=0 case — not be clipped to neutral."""
        pos = ad._score(*_score_inputs(cohens_d=+0.5), None)
        zero = ad._score(*_score_inputs(cohens_d=0.0), None)
        neg = ad._score(*_score_inputs(cohens_d=-0.5), None)
        assert pos.composite_score > zero.composite_score > neg.composite_score

    def test_z_star_rewards_high_edge_to_noise(self, ad):
        """A higher |z*| (rotation-null edge-to-noise) lifts the score."""
        lo = ad._score(*_score_inputs(z=0.3), None)
        hi = ad._score(*_score_inputs(z=3.0), None)
        assert hi.composite_score > lo.composite_score

    def test_statistically_weak_applies_penalty(self, ad):
        """A statistically_weak target has its composite multiplied by the
        configured penalty (0.6 by default)."""
        strong = ad._score(*_score_inputs(weak=False), None)
        weak = ad._score(*_score_inputs(weak=True), None)
        assert weak.composite_score < strong.composite_score
        assert weak.composite_score == pytest.approx(
            strong.composite_score * 0.6, abs=1e-3
        )

    def test_oos_pass_adds_bonus(self, ad):
        """A passing OOS confirmation adds the configured bonus."""
        without = ad._score(*_score_inputs(), None)
        with_oos = ad._score(*_score_inputs(), _oos(passed=True))
        assert with_oos.composite_score == pytest.approx(
            without.composite_score + 0.05, abs=1e-3
        )

    def test_weak_adverse_signal_grades_below_strong_clean(self, ad):
        """The issue's headline: a weak signal with cohens_d < 0 must grade
        strictly below a confirmed signal with cohens_d > 0."""
        adverse = ad._score(
            *_score_inputs(cohens_d=-0.3, z=0.6, weak=True), _oos(passed=False)
        )
        clean = ad._score(
            *_score_inputs(cohens_d=+0.6, z=2.5, weak=False), _oos(passed=True)
        )
        assert clean.composite_score > adverse.composite_score
        order = {"D": 0, "C": 1, "B": 2, "A": 3}
        assert order[clean.grade] >= order[adverse.grade]

    def test_composite_stays_in_unit_interval(self, ad):
        """Even with a strongly negative Cohen's d the stored composite is
        clamped to [0, 1]."""
        s = ad._score(*_score_inputs(ic=0.0, lift=0.0, cohens_d=-1.0, z=0.0), None)
        assert 0.0 <= s.composite_score <= 1.0

    def test_legacy_four_tuple_weights_still_work(self):
        """A legacy 4-tuple ``score_weights`` is upgraded and scores normally."""
        df = _predictive_kpi_table(n=400)
        ad = AlphaDiscovery(df, [], AlphaConfig(score_weights=(0.25, 0.30, 0.25, 0.20)))
        assert len(ad.config.score_weights) == 5
        s = ad._score(*_score_inputs(), None)
        assert 0.0 <= s.composite_score <= 1.0


# ---------------------------------------------------------------------------
# Fixed-target mode — user-specified (horizon, min_return, side) (issue #99)
# ---------------------------------------------------------------------------

class TestFixedTarget:
    @pytest.fixture(scope="class")
    def base(self):
        df = _predictive_kpi_table(n=4000)
        return _make_candidates(df)        # (ed, candidates)

    def _run(self, base, diagnostic=True, **ft_kwargs):
        ed, cands = base
        cfg = AlphaConfig(
            asset="SYN",
            fixed_target=TargetConfig(**ft_kwargs),
            fixed_target_diagnostic=diagnostic,
        )
        return AlphaDiscovery(ed.df, cands, cfg).run()

    def test_target_overrides_derivation(self, base):
        contracts = self._run(base, horizon=6, min_return=0.03, side="short")
        assert contracts
        for c in contracts:
            dt = c.derived_target
            assert dt.fixed_target is True
            assert dt.holding_period_h == 6
            assert dt.sell_pct == pytest.approx(0.03)
            assert dt.direction == "short"          # never overwritten by the data
            assert math.isnan(dt.mean_advantage)    # not a derived quantity

    def test_never_undetermined_all_promoted(self, base):
        """In fixed-target mode the direction is user-given, so no candidate is
        lost to 'undetermined'."""
        contracts = self._run(base, horizon=6, min_return=0.03, side="long")
        assert contracts
        assert all(c.derived_target.direction == "long" for c in contracts)
        assert all(c.status == "HYPOTHESIS" and c.promoted for c in contracts)

    def test_diagnostic_populates_convergence_fields(self, base):
        contracts = self._run(base, horizon=6, min_return=0.03, side="long")
        dt = contracts[0].derived_target
        # read-only derivation fills the per-horizon profile + convergence fields
        assert dt.advantage_by_h            # non-empty diagnostic profile
        assert dt.data_derived_horizon_h is not None
        assert dt.data_derived_horizon_h > 0

    def test_diagnostic_off_skips_derivation(self, base):
        contracts = self._run(base, diagnostic=False,
                              horizon=6, min_return=0.03, side="long")
        dt = contracts[0].derived_target
        assert dt.fixed_target is True
        assert dt.holding_period_h == 6 and dt.direction == "long"
        assert dt.advantage_by_h == {}            # no read-only derivation
        assert dt.data_derived_horizon_h is None
        assert dt.data_derived_sell_pct is None

    def test_horizon_outside_grid_is_supported(self, base):
        """A horizon absent from horizon_grid is added to the forward-return
        grid so all downstream measures can be read at it."""
        contracts = self._run(base, horizon=7, min_return=0.03, side="long")
        assert 7 not in AlphaConfig().horizon_grid
        assert all(c.derived_target.holding_period_h == 7 for c in contracts)

    def test_downstream_measures_run(self, base):
        """A fixed-target contract is structurally complete: event stats, OOS
        and grade are all populated against the user's target."""
        contracts = self._run(base, horizon=6, min_return=0.03, side="long")
        c = contracts[0]
        assert c.event_stats.n_activations >= 0
        assert c.oos_validation is not None
        assert c.alpha_score.grade in ("A", "B", "C", "D")

    def test_targetconfig_validation(self):
        with pytest.raises(ValueError):
            TargetConfig(horizon=0, min_return=0.03, side="long")
        with pytest.raises(ValueError):
            TargetConfig(horizon=6, min_return=0.0, side="long")
        with pytest.raises(ValueError):
            TargetConfig(horizon=6, min_return=0.03, side="sideways")
        # side is normalised
        assert TargetConfig(horizon=6, min_return=0.03, side="LONG").side == "long"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_missing_timestamp_raises(self):
        df = _predictive_kpi_table(n=500).drop(columns=["open_dt"])
        _, cands = _make_candidates(_predictive_kpi_table(n=500))
        with pytest.raises(ValueError):
            AlphaDiscovery(df, cands, AlphaConfig(timestamp_col="open_dt"))

    def test_accepts_datetime_index(self):
        df = _predictive_kpi_table(n=2000)
        _, cands = _make_candidates(df)
        indexed = df.set_index("open_dt")
        ad = AlphaDiscovery(indexed, cands, AlphaConfig())
        contracts = ad.run()
        assert len(contracts) == len(cands)
