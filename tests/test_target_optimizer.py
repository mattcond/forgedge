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
        assert cfg.min_lift_atoms == 1.0
        assert cfg.min_lift_result == 1.0

    def test_side_normalised(self):
        assert TargetConfig(horizon=1, min_return=0.01, side="SHORT").side == "short"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon": 0, "min_return": 0.1, "side": "long"},
            {"horizon": 1, "min_return": 0.0, "side": "long"},
            {"horizon": 1, "min_return": 0.1, "side": "sideways"},
            {"horizon": 1, "min_return": 0.1, "side": "long", "min_activations": 0},
            {"horizon": 1, "min_return": 0.1, "side": "long", "min_lift_atoms": -0.5},
            {"horizon": 1, "min_return": 0.1, "side": "long", "min_lift_result": -0.5},
        ],
    )
    def test_invalid_raises(self, kwargs):
        with pytest.raises(ValueError):
            TargetConfig(**kwargs)

    def test_deprecated_min_lift_warns_and_maps_to_both(self):
        # The legacy single threshold is applied to both passes, with a warning.
        with pytest.warns(DeprecationWarning):
            cfg = TargetConfig(horizon=6, min_return=0.02, side="long", min_lift=1.3)
        assert cfg.min_lift_atoms == 1.3
        assert cfg.min_lift_result == 1.3

    def test_deprecated_min_lift_still_validates(self):
        # A negative legacy value maps through and trips the new validation.
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                TargetConfig(horizon=1, min_return=0.1, side="long", min_lift=-0.5)


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
    pytestmark = pytest.mark.slow
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
        assert (results["lift"] >= opt.target_cfg.min_lift_result).all()

    def test_base_rate_recorded(self, opt_and_results):
        opt, _, _ = opt_and_results
        assert 0.0 < opt.base_rate < 1.0

    def test_candidates_aligned_with_results(self, opt_and_results):
        opt, results, _ = opt_and_results
        assert [c.event_id for c in opt.candidates] == results["event_id"].tolist()

    def test_higher_min_lift_result_is_stricter(self):
        df = _ohlc_table()
        loose = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift_result=1.0)
        ).run()
        strict = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift_result=1.5)
        ).run()
        assert len(strict) <= len(loose)
        assert (strict["lift"] >= 1.5).all()


class TestTargetOptimizerLiftSplit:
    """min_lift_atoms / min_lift_result decouple AND discovery from result filtering (issue #107)."""
    pytestmark = pytest.mark.slow

    def test_atoms_floor_preserves_emergent_and(self):
        """A moderately-positive atom (individual lift in [1.0, 1.4)) can
        combine into a pair whose *emergent* lift clears 1.4 -- something a
        single min_lift=1.4 pre-filter (legacy behaviour) would never let
        into composition at all, since that atom alone wouldn't survive it.

        Verified directly by checking which atom(s) contributed to a
        surviving pair, not via a raw pair-count comparison against a
        stricter-floor run (issue #230): `ANDComposer` caps compositions at
        `_MAX_PAIRS` and (since #230) samples them representatively rather
        than always from the same few atoms, so a smaller/stronger atom pool
        can produce a *higher* survival rate among its own capped sample than
        a larger/more heterogeneous one -- a raw count comparison between two
        differently-sized, both-capped configs is not a reliable signal of
        which pre-filter "preserves more emergent ANDs".
        """
        df = _ohlc_table()

        # Capture each lift-positive atom's own (individual) lift alongside
        # the min_lift_atoms=1.0 pass, so "moderate" atoms (lift in
        # [1.0, 1.4), which the final min_lift_result=1.4 floor would reject
        # standalone) can be identified by expression.
        captured: list[tuple[str, float]] = []
        orig_prune = TargetOptimizer._prune_by_lift

        def _spying_prune(self, events, target, min_lift):
            if min_lift == 1.0:
                for ev in events:
                    score = _lift_score(ev.series, target, self.target_cfg.min_activations)
                    if score is not None:
                        captured.append((ev.component.expression, score["lift"]))
            return orig_prune(self, events, target, min_lift)

        TargetOptimizer._prune_by_lift = _spying_prune
        try:
            keep_atoms = TargetOptimizer(
                df,
                TargetConfig(
                    horizon=6, min_return=0.02, side="long",
                    min_lift_atoms=1.0, min_lift_result=1.4,
                ),
            ).run()
        finally:
            TargetOptimizer._prune_by_lift = orig_prune

        moderate_exprs = {expr for expr, lift in captured if 1.0 <= lift < 1.4}
        assert moderate_exprs, "fixture sanity: some atoms must have moderate, non-final-floor lift"

        n2 = keep_atoms.loc[keep_atoms["n_components"] == 2]
        assert len(n2) > 0
        # Those AND pairs genuinely clear the result floor.
        assert (n2["lift"] >= 1.4).all()
        # And at least one of them is a genuinely emergent pair: composed
        # from a moderate atom that would never have survived a single
        # min_lift=1.4 pre-filter on its own.
        assert any(
            any(expr in composed_expr for expr in moderate_exprs)
            for composed_expr in n2["expression"]
        ), "at least one surviving pair must involve a moderate-lift (not individually-1.4+) atom"

    def test_legacy_single_threshold_matches_symmetric_split(self):
        # min_lift=X (deprecated) must equal min_lift_atoms=X, min_lift_result=X.
        df = _ohlc_table()
        with pytest.warns(DeprecationWarning):
            legacy = TargetOptimizer(
                df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift=1.3)
            ).run()
        split = TargetOptimizer(
            df,
            TargetConfig(
                horizon=6, min_return=0.02, side="long",
                min_lift_atoms=1.3, min_lift_result=1.3,
            ),
        ).run()
        assert legacy["event_id"].tolist() == split["event_id"].tolist()


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

    def test_validate_oos_low_sample_flag_and_no_nan_suppression(self):
        """OOS events below min_activations are still scored; low_sample_oos=True."""
        # Small OOS window (50 bars) with high min_activations to force violations.
        df = _ohlc_table(n=1200)
        train = df.iloc[:1150]
        oos = df.iloc[1150:]  # 50 bars — very few activations expected
        opt = TargetOptimizer(
            train,
            TargetConfig(horizon=6, min_return=0.02, side="long", min_activations=20),
        )
        opt.run()
        result = opt.validate_oos(oos, top_k=5)

        # low_sample_oos column must exist.
        assert "low_sample_oos" in result.columns

        # Events with n_activations_oos < 20 must have low_sample_oos=True.
        flagged = result[result["low_sample_oos"] == True]
        if not flagged.empty:
            assert flagged["n_activations_oos"].lt(20).all()
            # lift_oos is NaN only when the base rate in the OOS window is
            # degenerate (all 0 or all 1 targets), not merely because the sample
            # is small — n_activations_oos is always populated.
            assert flagged["n_activations_oos"].notna().all()

        # n_activations_oos is always populated (never NaN), even for flagged rows.
        assert result["n_activations_oos"].notna().all()

        # Events with sufficient OOS activations must have low_sample_oos=False.
        ok = result[result["n_activations_oos"] >= 20]
        if not ok.empty:
            assert (ok["low_sample_oos"] == False).all()


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
    pytestmark = pytest.mark.slow
    def test_discover_alpha_uses_fixed_target(self):
        df = _ohlc_table()
        opt = TargetOptimizer(
            df, TargetConfig(horizon=6, min_return=0.02, side="long", min_lift_result=1.5)
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
