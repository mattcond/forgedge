"""Tests for the Alpha Discovery module (FORGE Modulo 2)."""
import math

import numpy as np
import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    AlphaDiscovery,
    DiscoveryConfig,
    EventDiscovery,
    PromotionThresholds,
    TargetDefinition,
)
from forgedge.alpha_discovery import stats
from forgedge.alpha_discovery.market_structure import analyse_market_structure
from forgedge.alpha_discovery.target import build_target


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _predictive_kpi_table(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """KPI table with a genuine mean-reversion signal in ``feat``.

    The one-step-ahead return is driven by ``-k * (feat - 0.5)``, so a low
    ``feat`` predicts a positive forward return.  Event Discovery should pick
    up ``feat < <low>`` events and Alpha Discovery should measure a strong
    negative IC and a high win rate on them.
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
    return pd.DataFrame(
        {
            "open_dt": dates,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": feat,
        }
    )


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
# Step 1 — target
# ---------------------------------------------------------------------------

class TestTarget:
    def test_base_rate_in_unit_interval(self):
        df = _predictive_kpi_table()
        fwd, tgt, base = build_target(
            df.set_index("open_dt")["close"], TargetDefinition(holding_period_h=24, sell_pct=0.04)
        )
        assert 0.0 <= base <= 1.0

    def test_no_lookahead_tail_is_nan(self):
        close = pd.Series(np.linspace(100, 110, 100))
        fwd, tgt, _ = build_target(close, TargetDefinition(holding_period_h=10, sell_pct=0.01))
        # Last h bars cannot see a full forward window.
        assert tgt.iloc[-10:].isna().all()
        assert fwd.iloc[-10:].isna().all()

    def test_long_target_triggers_on_rise(self):
        # Monotonic +2%/bar rise: every early bar reaches +4% within 24 bars.
        close = pd.Series(100.0 * (1.02 ** np.arange(60)))
        _, tgt, base = build_target(close, TargetDefinition(holding_period_h=24, sell_pct=0.04))
        assert tgt.iloc[0] == 1.0
        assert base > 0.9

    def test_short_direction_triggers_on_fall(self):
        close = pd.Series(100.0 * (0.98 ** np.arange(60)))
        _, tgt, base = build_target(
            close, TargetDefinition(holding_period_h=24, sell_pct=0.04, direction="short")
        )
        assert tgt.iloc[0] == 1.0
        assert base > 0.9

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            build_target(pd.Series([1.0, 2.0, 3.0]), TargetDefinition(direction="sideways"))


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
# End-to-end — Alpha Discovery over Event Discovery output
# ---------------------------------------------------------------------------

class TestAlphaDiscoveryEndToEnd:
    @pytest.fixture(scope="class")
    def fitted(self):
        df = _predictive_kpi_table()
        ed, cands = _make_candidates(df)
        ad = AlphaDiscovery(
            df.copy(),
            cands,
            AlphaConfig(
                target=TargetDefinition(holding_period_h=12, sell_pct=0.01, asset="SYN"),
            ),
        )
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

    def test_promoted_alpha_has_negative_ic_and_lift(self, fitted):
        _, _, ad, _ = fitted
        c = ad.promoted_contracts()[0]
        assert c.underlying_feature.ic < 0          # mean-reversion signal
        assert c.event_stats.lift >= 0.08
        assert c.event_stats.cohens_d >= 0.15
        assert c.event_stats.win_rate > c.event_stats.base_rate

    def test_summary_sorted_by_score(self, fitted):
        _, _, ad, _ = fitted
        s = ad.summary()
        assert list(s.columns)[:3] == ["alpha_id", "status", "promoted"]
        scores = s["composite_score"].to_numpy()
        assert np.all(np.diff(scores) <= 1e-9)       # descending

    def test_contract_dict_serialises(self, fitted):
        _, _, ad, _ = fitted
        d = ad.promoted_contracts()[0].to_contract_dict()
        assert d["status"] == "HYPOTHESIS"
        assert d["event_expression"]
        assert "statistical_evidence" in d
        assert d["target_definition"]["base_rate"] == pytest.approx(ad.base_rate)

    def test_rejected_contracts_carry_reasons(self, fitted):
        _, _, ad, contracts = fitted
        rejected = [c for c in contracts if not c.promoted]
        assert rejected  # the predictive table also yields many weak candidates
        assert all(c.rejection_reasons for c in rejected)

    def test_expression_is_propagated_unchanged(self, fitted):
        _, cands, ad, contracts = fitted
        by_id = {c.event_id: c.expression for c in cands}
        for contract in contracts:
            assert contract.event_expression == by_id[contract.event_candidate_id]


# ---------------------------------------------------------------------------
# Promotion / FDR behaviour
# ---------------------------------------------------------------------------

class TestPromotionGates:
    def test_raw_pvalue_mode_is_more_permissive_than_fdr(self):
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        target = TargetDefinition(holding_period_h=12, sell_pct=0.01)

        fdr = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            target=target, thresholds=PromotionThresholds(use_fdr=True, fdr_q=0.05)))
        raw = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            target=target, thresholds=PromotionThresholds(use_fdr=False, max_p_value=0.05)))
        fdr.run(); raw.run()
        assert len(raw.promoted_contracts()) >= len(fdr.promoted_contracts())

    def test_strict_thresholds_reject_everything(self):
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            target=TargetDefinition(holding_period_h=12, sell_pct=0.01),
            thresholds=PromotionThresholds(min_lift=0.95, min_cohens_d=5.0)))
        ad.run()
        assert ad.promoted_contracts() == []


# ---------------------------------------------------------------------------
# Regime sensitivity
# ---------------------------------------------------------------------------

class TestRegimeSensitivity:
    def test_no_regime_column_yields_unknown_dependency(self):
        df = _predictive_kpi_table()
        _, cands = _make_candidates(df)
        ad = AlphaDiscovery(df.copy(), cands, AlphaConfig(
            target=TargetDefinition(holding_period_h=12, sell_pct=0.01)))
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
        ad = AlphaDiscovery(df, cands, AlphaConfig(
            target=TargetDefinition(holding_period_h=12, sell_pct=0.01)))
        ad.run()
        promoted = ad.promoted_contracts()
        assert promoted
        ra = promoted[0].regime_analysis
        assert ra.dependency_type in ("agnostic", "conditional", "specific", "broken")
        assert {r.regime for r in ra.per_regime} <= {"BEAR", "BULL"}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_missing_timestamp_raises(self):
        df = _predictive_kpi_table(n=500).drop(columns=["open_dt"])
        _, cands = _make_candidates(_predictive_kpi_table(n=500))
        with pytest.raises(ValueError):
            AlphaDiscovery(df, cands, AlphaConfig(
                target=TargetDefinition(), timestamp_col="open_dt"))

    def test_accepts_datetime_index(self):
        df = _predictive_kpi_table(n=2000)
        _, cands = _make_candidates(df)
        indexed = df.set_index("open_dt")
        ad = AlphaDiscovery(indexed, cands, AlphaConfig(
            target=TargetDefinition(holding_period_h=12, sell_pct=0.01)))
        contracts = ad.run()
        assert len(contracts) == len(cands)
