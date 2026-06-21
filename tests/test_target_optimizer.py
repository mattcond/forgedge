"""Tests for the ``TargetOptimizer`` standalone module (issue #100)."""
import numpy as np
import pandas as pd
import pytest

from forgedge import TargetConfig, TargetOptimizer
from forgedge.target_optimizer import _RESULT_COLUMNS, _lift_score


def _ohlc_table(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """OHLCV + a mean-reverting feature, mirroring the forge test fixture.

    Low ``feat`` predicts a positive next-bar return, so events on ``feat`` (and
    its deltas) lift the win rate of a short-horizon long target above the base
    rate — giving the optimizer a real signal to rank.
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -0.02 * (feat[:-1] - 0.5) + rng.normal(0.0, 0.01, n - 1)
    close = 100.0 * np.exp(np.cumsum(r))
    op = close * (1 + rng.normal(0.0, 0.001, n))
    high = np.maximum(op, close) * (1 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(op, close) * (1 - np.abs(rng.normal(0.0, 0.003, n)))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="4h"),
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": feat,
        }
    )


class TestTargetConfig:
    """The optimizer-specific knobs added to the shared TargetConfig."""

    def test_defaults(self):
        cfg = TargetConfig(horizon=20, min_return=0.10, side="long")
        assert cfg.min_activations == 10
        assert cfg.min_lift == 1.0

    def test_side_normalised(self):
        assert TargetConfig(horizon=1, min_return=0.01, side="SHORT").side == "short"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon": 0, "min_return": 0.1, "side": "long"},
            {"horizon": 1, "min_return": 0.0, "side": "long"},
            {"horizon": 1, "min_return": 0.1, "side": "sideways"},
            {"horizon": 1, "min_return": 0.1, "side": "long", "min_activations": 0},
            {"horizon": 1, "min_return": 0.1, "side": "long", "min_lift": -0.5},
        ],
    )
    def test_invalid_raises(self, kwargs):
        with pytest.raises(ValueError):
            TargetConfig(**kwargs)


class TestLiftScore:
    """The two-proportion lift helper underpinning every scoring pass."""

    def test_skips_below_min_activations(self):
        event = pd.Series([1, 0, 0, 0, 0], dtype=float)
        target = pd.Series([1, 0, 1, 0, 1], dtype=float)
        assert _lift_score(event, target, min_activations=2) is None

    def test_degenerate_base_rate_skipped(self):
        event = pd.Series([1, 1, 0, 0], dtype=float)
        target = pd.Series([1, 1, 1, 1], dtype=float)  # base rate 1.0
        assert _lift_score(event, target, min_activations=1) is None

    def test_positive_lift(self):
        # Event fires exactly on the winning bars → lift > 1.
        event = pd.Series([1, 1, 0, 0, 0, 0], dtype=float)
        target = pd.Series([1, 1, 0, 0, 1, 0], dtype=float)
        score = _lift_score(event, target, min_activations=2)
        assert score["win_rate_event"] == 1.0
        assert score["lift"] > 1.0
        assert score["z_score"] > 0.0


class TestTargetOptimizerRun:
    @pytest.fixture(scope="class")
    @classmethod
    def opt_and_results(cls):
        df = _ohlc_table()
        opt = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long")
        )
        results = opt.run()
        return opt, results, df

    def test_schema_and_non_empty(self, opt_and_results):
        _, results, _ = opt_and_results
        assert list(results.columns) == _RESULT_COLUMNS
        assert len(results) > 0

    def test_ranked_by_lift_descending(self, opt_and_results):
        _, results, _ = opt_and_results
        lifts = results["lift"].tolist()
        assert lifts == sorted(lifts, reverse=True)

    def test_every_survivor_beats_min_lift(self, opt_and_results):
        opt, results, _ = opt_and_results
        assert (results["lift"] >= opt.target_cfg.min_lift).all()

    def test_base_rate_recorded(self, opt_and_results):
        opt, _, _ = opt_and_results
        assert 0.0 < opt.base_rate < 1.0

    def test_candidates_aligned_with_results(self, opt_and_results):
        opt, results, _ = opt_and_results
        assert [c.event_id for c in opt.candidates] == results["event_id"].tolist()

    def test_higher_min_lift_is_stricter(self):
        df = _ohlc_table()
        loose = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift=1.0)
        ).run()
        strict = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift=1.5)
        ).run()
        assert len(strict) <= len(loose)
        assert (strict["lift"] >= 1.5).all()


class TestTargetOptimizerOOS:
    def test_validate_oos_merges_in_and_out_of_sample(self):
        df = _ohlc_table(n=1200)
        train = df.iloc[:900]
        opt = TargetOptimizer(
            train, TargetConfig(horizon=6, min_return=0.02, side="long")
        )
        opt.run()
        oos = opt.validate_oos(df, top_k=5)
        assert len(oos) <= 5
        # IS columns preserved, OOS columns appended.
        for col in ("lift", "lift_oos", "z_oos", "n_activations_oos"):
            assert col in oos.columns
        # Same events, re-scored on the full set.
        assert set(oos["event_id"]).issubset({c.event_id for c in opt.candidates})

    def test_validate_oos_before_run_raises(self):
        opt = TargetOptimizer(
            _ohlc_table(), TargetConfig(horizon=6, min_return=0.02, side="long")
        )
        with pytest.raises(RuntimeError):
            opt.validate_oos(_ohlc_table())


class TestTargetOptimizerEmpty:
    def test_unreachable_target_returns_empty_schema(self):
        df = _ohlc_table()
        # A +500% move in one bar never happens → base rate 0 → nothing scores.
        opt = TargetOptimizer(
            df, TargetConfig(horizon=1, min_return=5.0, side="long")
        )
        results = opt.run()
        assert results.empty
        assert list(results.columns) == _RESULT_COLUMNS
        assert opt.candidates == []


class TestTargetOptimizerAlphaHandoff:
    def test_discover_alpha_uses_fixed_target(self):
        df = _ohlc_table()
        opt = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift=1.5)
        )
        results = opt.run()
        contracts = opt.discover_alpha()
        # One contract per surviving candidate, all at the fixed horizon/side.
        assert len(contracts) == len(results)
        for c in contracts:
            assert c.derived_target.holding_period_h == 6
            assert c.derived_target.direction == "long"

    def test_discover_alpha_before_run_raises(self):
        opt = TargetOptimizer(
            _ohlc_table(), TargetConfig(horizon=6, min_return=0.02, side="long")
        )
        with pytest.raises(RuntimeError):
            opt.discover_alpha()
