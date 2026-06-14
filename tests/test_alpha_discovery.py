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
)
from forgedge.alpha_discovery import stats
from forgedge.alpha_discovery.market_structure import analyse_market_structure
from forgedge.alpha_discovery.target import binary_target, forward_returns
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
        tgt, base = binary_target(close, 24, 0.04, "long")
        assert tgt.iloc[0] == 1.0
        assert base > 0.9

    def test_short_direction_triggers_on_fall(self):
        close = pd.Series(100.0 * (0.98 ** np.arange(60)))
        tgt, base = binary_target(close, 24, 0.04, "short")
        assert tgt.iloc[0] == 1.0
        assert base > 0.9

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            binary_target(pd.Series([1.0, 2.0, 3.0]), 24, 0.04, "sideways")


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
            gate_params=GateParams(min_act=30, min_months=2, max_conc=0.6, min_tpm=1.0),
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
