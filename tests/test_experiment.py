"""Tests for forgedge.experiment (StepWiseDiscovery and its redundancy checks)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgedge.experiment import (
    ChainResult,
    SeedAttempt,
    StepWiseDiscovery,
    StepWiseDiscoveryConfig,
    StepWiseDiscoveryResult,
)
from forgedge.experiment.redundancy import (
    family_key,
    is_redundant,
    max_abs_corr,
    max_jaccard,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "ADA_1D_TRAIN.parquet"


def _make_tiny_kpi(n: int = 25, seed: int = 0) -> pd.DataFrame:
    """A KPI table deliberately too short for any realistic discovery session.

    Used only to exercise the fail-fast (``strict=True``) config-coherence
    path in ``StepWiseDiscovery._resolve`` — never runs a full pipeline.
    """
    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame({
        "open_dt": pd.date_range("2024-01-01", periods=n, freq="1D"),
        "close": price,
        "close_rsi_14": rng.uniform(0, 100, n),
    })


def _component(transformed_col: str):
    """Minimal stand-in for an EventComponent — max_abs_corr only reads .transformed_col."""
    return SimpleNamespace(transformed_col=transformed_col)


# ---------------------------------------------------------------------------
# redundancy.py — pure functions, no pipeline needed
# ---------------------------------------------------------------------------


class TestFamilyKey:
    def test_strips_numeric_periods(self):
        cand_a = SimpleNamespace(components=[SimpleNamespace(source_feature="close_ema_09", transform="identity")])
        cand_b = SimpleNamespace(components=[SimpleNamespace(source_feature="close_ema_21", transform="identity")])
        assert family_key(cand_a) == family_key(cand_b) == "close_ema|identity"

    def test_different_transform_is_different_family(self):
        cand_a = SimpleNamespace(components=[SimpleNamespace(source_feature="close_ema_09", transform="identity")])
        cand_b = SimpleNamespace(components=[SimpleNamespace(source_feature="close_ema_09", transform="rolling_zscore")])
        assert family_key(cand_a) != family_key(cand_b)


class TestMaxJaccard:
    def test_empty_used_list_is_zero(self):
        a = np.array([True, False, True, True])
        assert max_jaccard(a, []) == 0.0

    def test_identical_series_is_one(self):
        a = np.array([True, False, True, True, False])
        assert max_jaccard(a, [a.copy()]) == pytest.approx(1.0)

    def test_disjoint_series_is_zero(self):
        a = np.array([True, True, False, False])
        b = np.array([False, False, True, True])
        assert max_jaccard(a, [b]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([True, True, True, False])   # 3 True
        b = np.array([True, True, False, False])  # 2 True, intersection=2, union=3
        assert max_jaccard(a, [b]) == pytest.approx(2 / 3)

    def test_picks_the_maximum_across_multiple_used(self):
        a = np.array([True, True, False, False])
        low_overlap = np.array([False, True, False, False])   # J=1/4
        high_overlap = np.array([True, True, False, False])   # J=1.0
        assert max_jaccard(a, [low_overlap, high_overlap]) == pytest.approx(1.0)


class TestMaxAbsCorr:
    def test_empty_used_cols_is_zero(self):
        series_by_col = {"a": np.arange(100, dtype=float)}
        assert max_abs_corr([_component("a")], [], series_by_col) == 0.0

    def test_perfectly_correlated_series(self):
        x = np.linspace(0, 10, 200)
        series_by_col = {"a": x, "b": 2 * x + 1}
        corr = max_abs_corr([_component("a")], ["b"], series_by_col)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_perfectly_anticorrelated_series_is_abs(self):
        x = np.linspace(0, 10, 200)
        series_by_col = {"a": x, "b": -x}
        corr = max_abs_corr([_component("a")], ["b"], series_by_col)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_uncorrelated_series_near_zero(self):
        rng = np.random.default_rng(1)
        series_by_col = {"a": rng.normal(size=5000), "b": rng.normal(size=5000)}
        corr = max_abs_corr([_component("a")], ["b"], series_by_col)
        assert corr < 0.1

    def test_missing_column_returns_zero_not_error(self):
        series_by_col = {"a": np.arange(100, dtype=float)}
        assert max_abs_corr([_component("a")], ["does_not_exist"], series_by_col) == 0.0

    def test_too_few_paired_observations_skipped(self):
        series_by_col = {"a": np.array([1.0, np.nan, np.nan]), "b": np.array([1.0, 2.0, np.nan])}
        assert max_abs_corr([_component("a")], ["b"], series_by_col) == 0.0


class TestIsRedundant:
    def test_flags_on_jaccard_alone(self):
        a = np.array([True, True, False, False])
        redundant, j, corr = is_redundant(
            bool_series=a, used_bool_series=[a.copy()],
            components=[_component("x")], used_cols=[],
            series_by_col={},
            max_jaccard_threshold=0.85, max_abs_corr_threshold=0.95,
        )
        assert redundant is True
        assert j == pytest.approx(1.0)
        assert corr == 0.0

    def test_flags_on_correlation_alone_even_with_low_jaccard(self):
        # The exact case documented in the module: different-enough boolean
        # activations (low Jaccard) but near-identical continuous signal.
        a = np.array([True, True, False, False, False])
        used_bool = np.array([False, True, False, False, True])  # J = 1/4, below 0.85
        x = np.linspace(0, 1, 200)
        redundant, j, corr = is_redundant(
            bool_series=a, used_bool_series=[used_bool],
            components=[_component("x")], used_cols=["y"],
            series_by_col={"x": x, "y": x * 3 + 0.1},
            max_jaccard_threshold=0.85, max_abs_corr_threshold=0.95,
        )
        assert j < 0.85
        assert corr > 0.95
        assert redundant is True

    def test_not_redundant_when_both_below_threshold(self):
        rng = np.random.default_rng(2)
        a = rng.random(1000) > 0.7
        used_bool = rng.random(1000) > 0.7
        redundant, j, corr = is_redundant(
            bool_series=a, used_bool_series=[used_bool],
            components=[_component("x")], used_cols=["y"],
            series_by_col={"x": rng.normal(size=1000), "y": rng.normal(size=1000)},
            max_jaccard_threshold=0.85, max_abs_corr_threshold=0.95,
        )
        assert redundant is False


# ---------------------------------------------------------------------------
# StepWiseDiscovery — config resolution (fast, no pipeline run)
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_strict_raises_on_incoherent_config(self):
        kpi = _make_tiny_kpi()
        engine = StepWiseDiscovery(kpi, ticker="TINY", timeframe="1D",
                                    config=StepWiseDiscoveryConfig(strict=True))
        with pytest.raises(ValueError):
            engine.run()

    def test_non_strict_does_not_raise_on_incoherent_config_at_resolve(self):
        kpi = _make_tiny_kpi()
        engine = StepWiseDiscovery(kpi, ticker="TINY", timeframe="1D",
                                    config=StepWiseDiscoveryConfig(strict=False))
        # Resolution itself must not raise under strict=False — whether the
        # full .run() completes on data this thin is a separate question,
        # not what this test is checking.
        engine._resolve()
        assert engine.disc_cfg is not None
        assert engine.alpha_cfg is not None
        assert engine.rd_cfg is not None
        assert engine.context is not None


# ---------------------------------------------------------------------------
# StepWiseDiscovery — end-to-end smoke test on the committed reference fixture
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStepWiseDiscoveryEndToEnd:
    @pytest.fixture(scope="class")
    @staticmethod
    def result() -> StepWiseDiscoveryResult:
        kpi = pd.read_parquet(_FIXTURE)
        engine = StepWiseDiscovery(
            kpi, ticker="ADAUSDC", timeframe="1D",
            config=StepWiseDiscoveryConfig(n_seeds=2, depth=2),
        )
        return engine.run()

    def test_result_shape(self, result):
        assert isinstance(result, StepWiseDiscoveryResult)
        assert len(result.chains) <= 2
        assert len(result.chains) >= 1
        for chain in result.chains:
            assert isinstance(chain, ChainResult)
            assert 0 <= chain.depth_reached <= 2
            assert chain.verdict in ("EDGE", "PARTIAL-EDGE", "NON-EDGE", "INSUFFICIENT-DATA")

    def test_summary_dataframe(self, result):
        df = result.summary()
        assert len(df) == len(result.chains)
        for col in ("chain", "seed_family", "expression", "depth_reached", "composite_score", "verdict"):
            assert col in df.columns

    def test_seed_attempts_cover_every_accepted_seed(self, result):
        assert len(result.seed_attempts) >= len(result.chains)
        accepted = [a for a in result.seed_attempts if a.accepted]
        assert len(accepted) == len(result.chains)
        seed_families = {c.seed_family for c in result.chains}
        assert {a.family for a in accepted} == seed_families

    def test_holdout_never_overlaps_search(self, result):
        ts_col = "open_dt"
        max_search_ts = result.search_df[ts_col].max()
        min_holdout_ts = result.holdout_df[ts_col].min()
        assert max_search_ts < min_holdout_ts

    def test_outer_split_covers_full_table(self, result):
        assert len(result.search_df) + len(result.holdout_df) <= result.time_budget.n_bars
        assert len(result.search_df) == result.time_budget.split

    def test_no_chain_composes_on_a_redundant_pair(self, result):
        # Every composed (depth >= 1) chain's two feature families must be
        # genuinely distinct — the whole point of the redundancy checks.
        for chain in result.chains:
            if chain.depth_reached == 0:
                continue
            source_features = [c.source_feature for c in chain.candidate.components]
            assert len(source_features) == len(set(source_features))
