"""Tests for the session TimeBudget (single purged/embargoed time axis)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    AlphaDiscovery,
    CustomEvent,
    DiscoveryConfig,
    EventDiscovery,
    TimeBudget,
    forge,
)
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.rule_discovery import BacktestParams, GridSpec, WalkForwardConfig, walk_forward


# ---------------------------------------------------------------------------
# Unit — the budget arithmetic
# ---------------------------------------------------------------------------

class TestTimeBudgetUnit:
    def test_build_defaults_purge_to_horizon(self):
        tb = TimeBudget.build(1000, 0.7, horizon_bars=12)
        assert tb.split == 700
        assert tb.purge_bars == 12
        assert tb.embargo_bars == 0
        assert tb.oos_start == 700

    def test_purge_slice_is_horizon_capped(self):
        tb = TimeBudget.build(1000, 0.7, horizon_bars=10)
        assert tb.purge_slice(3) == (697, 700)
        assert tb.purge_slice(48) == (690, 700)  # capped at purge_bars

    def test_purge_disabled(self):
        tb = TimeBudget.build(1000, 0.7, horizon_bars=10, purge_bars=0)
        lo, hi = tb.purge_slice(10)
        assert lo == hi

    def test_no_oos_when_train_ratio_one(self):
        tb = TimeBudget.build(1000, 1.0, horizon_bars=10)
        assert not tb.has_oos
        lo, hi = tb.purge_slice(10)
        assert lo == hi  # crossing rows are off the end and already NaN

    def test_embargo_shifts_oos_start(self):
        tb = TimeBudget.build(1000, 0.7, horizon_bars=10, embargo_bars=5)
        assert tb.oos_start == 705
        assert "embargo 5" in tb.describe()


# ---------------------------------------------------------------------------
# M2 — purge and embargo in Alpha Discovery
# ---------------------------------------------------------------------------

def _frame_with_flags(n=600, split_ratio=0.7, h_max=8, seed=5):
    """Synthetic frame with two marker features.

    ``purge_flag``  — active only in the purge zone ``[split - h_max, split)``,
    each activation followed by a violent up-move (a fake edge that exists
    *only* across the IS/OOS boundary).
    ``mixed_flag``  — active at a few interior IS bars and at the first 5 OOS
    bars (for the embargo test).
    """
    rng = np.random.default_rng(seed)
    split = int(round(n * split_ratio))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))

    # Activations inside [split - 2, split): with the horizon grid (2, 4, 8)
    # every horizon's purge slice covers them, so the default budget removes
    # them from every measure.
    purge_flag = np.zeros(n)
    purge_flag[split - 2 : split] = 1.0
    # Violent up-move right after the boundary → a boundary-crossing "edge".
    close[split : split + 3 * h_max] *= np.linspace(1.0, 1.5, 3 * h_max)

    mixed_flag = np.zeros(n)
    mixed_flag[100:140:4] = 1.0          # interior IS activations
    mixed_flag[split : split + 5] = 1.0  # early-OOS activations

    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close,
            "purge_flag": purge_flag,
            "mixed_flag": mixed_flag,
        }
    ), split


def _alpha_cfg(**kw):
    return AlphaConfig(
        horizon_grid=(2, 4, 8),
        train_ratio=0.7,
        thresholds=PromotionThresholds(
            require_significant_direction=False, min_direction_t=0.0
        ),
        **kw,
    )


def _candidate(frame, formula, name):
    return CustomEvent(name=name, formula=formula).to_event_candidate(frame)


class TestAlphaDiscoveryPurge:
    def test_boundary_only_edge_is_purged(self):
        # Every activation sits in the purge zone: with the default budget the
        # candidate has no usable IS bar left → direction undetermined.
        df, _ = _frame_with_flags()
        cand = _candidate(df, "purge_flag > 0.5", "purge_only")
        ad = AlphaDiscovery(df, [cand], _alpha_cfg())
        (contract,) = ad.run()
        assert contract.direction == "undetermined"

    def test_purge_disabled_reads_the_boundary_edge(self):
        # Same data, purge explicitly off → the boundary-crossing up-move is
        # read as a (spurious) long edge.  This is the pre-TimeBudget
        # behaviour, kept reachable for comparison.
        df, split = _frame_with_flags()
        n = len(df)
        cand = _candidate(df, "purge_flag > 0.5", "purge_only")
        tb = TimeBudget.build(n, 0.7, horizon_bars=8, purge_bars=0)
        ad = AlphaDiscovery(df, [cand], _alpha_cfg(), time_budget=tb)
        (contract,) = ad.run()
        assert contract.direction == "long"

    def test_embargo_excludes_early_oos_activations(self):
        df, split = _frame_with_flags()
        n = len(df)
        cand0 = _candidate(df, "mixed_flag > 0.5", "mixed")
        ad0 = AlphaDiscovery(
            df, [cand0], _alpha_cfg(),
            time_budget=TimeBudget.build(n, 0.7, horizon_bars=8, embargo_bars=0),
        )
        (c0,) = ad0.run()

        cand5 = _candidate(df, "mixed_flag > 0.5", "mixed")
        ad5 = AlphaDiscovery(
            df, [cand5], _alpha_cfg(),
            time_budget=TimeBudget.build(n, 0.7, horizon_bars=8, embargo_bars=5),
        )
        (c5,) = ad5.run()

        if c0.direction == "undetermined":
            pytest.skip("fixture produced no directed contract")
        assert c0.oos_validation.n_activations > 0
        assert c5.oos_validation.n_activations == 0
        assert c5.oos_validation.n_bars == c0.oos_validation.n_bars - 5

    def test_explicit_budget_overrides_train_ratio(self):
        df, _ = _frame_with_flags()
        n = len(df)
        tb = TimeBudget.build(n, 0.5, horizon_bars=8)
        ad = AlphaDiscovery(df, [_candidate(df, "mixed_flag > 0.5", "m")],
                            _alpha_cfg(), time_budget=tb)
        ad.run()
        assert ad.split_idx == tb.split == n // 2


# ---------------------------------------------------------------------------
# M1 — boundary alignment
# ---------------------------------------------------------------------------

class TestEventDiscoveryAlignment:
    def test_budget_split_overrides_config(self):
        df, _ = _frame_with_flags(n=6000)
        n = len(df)
        tb = TimeBudget.build(n, 0.6, horizon_bars=8)
        ed = EventDiscovery(
            df, DiscoveryConfig(timestamp_col="open_dt"), time_budget=tb
        )
        ed.run()
        assert ed._split_idx == tb.split

    def test_standalone_default_unchanged(self):
        df, _ = _frame_with_flags(n=6000)
        ed = EventDiscovery(df, DiscoveryConfig(timestamp_col="open_dt"))
        ed.run()
        assert ed._split_idx == len(df)  # train_ratio=1.0 → no split


# ---------------------------------------------------------------------------
# M3 — walk-forward purge / embargo
# ---------------------------------------------------------------------------

def _wf_candle(n=12000, seed=3):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n)))
    sig = np.zeros(n, dtype=int)
    sig[::40] = 1
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "__sig__": sig,
        }
    )


class TestWalkForwardPurge:
    SPEC = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])

    def test_train_windows_are_purged(self):
        df = _wf_candle()
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4)
        wf = walk_forward(df, "__sig__", BacktestParams(), self.SPEC, cfg)
        assert wf is not None
        # auto purge = max target_h (24) + max buy_delay + 1 bars of 1h each
        for s in wf.splits:
            gap = pd.Timestamp(s.test_from) - pd.Timestamp(s.train_to)
            assert gap >= pd.Timedelta(hours=24)

    def test_purge_zero_restores_contiguous_windows(self):
        df = _wf_candle()
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4, purge_bars=0)
        wf = walk_forward(df, "__sig__", BacktestParams(), self.SPEC, cfg)
        for s in wf.splits:
            assert s.train_to == s.test_from

    def test_embargo_shifts_test_start(self):
        df = _wf_candle()
        cfg = WalkForwardConfig(
            n_splits=3, min_train_months=4, purge_bars=0, embargo_bars=12
        )
        wf = walk_forward(df, "__sig__", BacktestParams(), self.SPEC, cfg)
        for s in wf.splits:
            gap = pd.Timestamp(s.test_from) - pd.Timestamp(s.train_to)
            assert gap == pd.Timedelta(hours=12)


# ---------------------------------------------------------------------------
# forge() wiring
# ---------------------------------------------------------------------------

def _forge_kpi(n=2600, seed=7):
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -0.02 * (feat[:-1] - 0.5) + rng.normal(0.0, 0.004, n - 1)
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "feat": feat,
        }
    )


class TestForgeWiring:
    pytestmark = pytest.mark.slow
    def test_effective_budget_exposed_by_default(self):
        res = forge(
            _forge_kpi(),
            timeframe="4H",
            run_rule_discovery=False,
            fast_null=False,
            progress=False,
        )
        tb = res.time_budget
        assert tb is not None
        assert tb.split == res.alpha_discovery.split_idx
        # The purge width covers the largest scanned horizon — at least the
        # base grid's max, possibly more when the enrichment added horizons.
        assert tb.purge_bars >= max(res.alpha_discovery.config.horizon_grid)
        scanned = {
            h for c in res.contracts for h in c.derived_target.t_stat_by_h
        }
        assert tb.purge_bars == max(scanned)

    def test_explicit_budget_drives_m1_and_m2(self):
        kpi = _forge_kpi()
        n = len(kpi)
        tb = TimeBudget.build(n, 0.6, horizon_bars=48)
        res = forge(
            kpi,
            timeframe="4H",
            time_budget=tb,
            run_rule_discovery=False,
            fast_null=True,
            progress=False,
        )
        assert res.time_budget is tb
        assert res.alpha_discovery.split_idx == tb.split
        assert res.event_discovery._split_idx == tb.split


# ---------------------------------------------------------------------------
# F6 (#180) — one axis, three modules that use it differently
# ---------------------------------------------------------------------------

class TestOneAxisReported:
    """`ForgeResult.time_budget` used to be M2's axis wearing the session's name.

    `forge()` forwarded only what the caller had passed — `None` by default —
    so each module cut its own timeline and the budget reported for the run
    was Alpha Discovery's. Under `forge_preset()` that meant announcing a 70%
    split for a session in which Event Discovery had used 100% of the span.
    """
    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def result(self):
        """forge() is a pure function of its inputs, and the three tests below
        that read a full run all call it with identical arguments — cache the
        one run (mirrors test_golden.py's session-scoped forge_result)."""
        kpi = _forge_kpi()
        res = forge(kpi, timeframe="4H", run_rule_discovery=False,
                    fast_null=False, progress=False)
        return kpi, res

    def test_the_budget_is_built_even_when_none_is_passed(self, result):
        _, res = result
        assert res.time_budget is not None

    def test_it_states_m1s_axis_instead_of_leaving_it_inferred(self, result):
        """M1's whole-span run is a decision (invariant #1: it never observes
        the forward return), so the budget says so rather than letting the
        reader deduce it from a split M1 does not use."""
        kpi, res = result
        tb = res.time_budget

        assert tb.event_split == len(kpi)          # DiscoveryConfig default 1.0
        assert tb.event_split_idx == res.event_discovery._split_idx
        assert tb.split == res.alpha_discovery.split_idx
        assert tb.event_split_idx != tb.split      # the two really do differ
        assert "M1 whole span, by choice" in tb.describe()
        assert "invariant #1" in tb.describe()

    def test_an_unset_event_split_still_means_follow_the_split(self):
        """Backwards compatibility: a budget built before this field existed,
        or handed in by a caller, drives M1 at `split` exactly as before."""
        tb = TimeBudget.build(1000, 0.6, horizon_bars=8)
        assert tb.event_split is None
        assert tb.event_split_idx == tb.split == 600
        assert "follows the split" in tb.describe()

    def test_a_declared_event_ratio_is_recorded(self):
        tb = TimeBudget.build(1000, 0.6, event_train_ratio=0.9)
        assert tb.event_split == 900
        assert TimeBudget.build(1000, 0.6, event_train_ratio=1.0).event_split == 1000

    def test_the_purge_widens_for_enriched_horizons_and_never_narrows(self, result):
        """The session budget is sized on the configured grid, but per-event
        enrichment can scan further. Purging less than the horizon actually
        read would put the look-ahead back, so M2 widens rather than obeys."""
        _, res = result
        tb = res.time_budget
        scanned = {h for c in res.contracts for h in c.derived_target.t_stat_by_h}
        assert tb.purge_bars >= max(res.alpha_discovery.config.horizon_grid)
        assert tb.purge_bars == max(scanned)
        # Widening the purge must not have moved the axis itself.
        assert tb.split == res.alpha_discovery.split_idx


class TestFoldOverlapIsVisible:
    """M3's walk-forward folds are OOS for M3's own selection, but a fold whose
    test window sits inside M2's IS is scoring the contract's target on the
    span that target was fit on. That was invisible.

    The origin is deliberately *not* moved to the session split: on the
    reference 28-month fixture that leaves the `balanced` preset with zero
    folds, removing the gate invariant #5 makes the verdict depend on.
    """
    pytestmark = pytest.mark.slow

    def test_folds_report_whether_they_test_in_sample(self):
        kpi = _forge_kpi()
        res = forge(kpi, timeframe="4H", fast_null=False, progress=False,
                    run_registry=False)
        wfs = [rr.walk_forward for _c, rr in res.rule_responses if rr.walk_forward]
        assert wfs, "expected at least one rule with a walk-forward"
        for wf in wfs:
            flags = [s.tests_in_sample for s in wf.splits]
            assert all(f is not None for f in flags)
            assert wf.n_splits_in_sample == sum(1 for f in flags if f)
            # The early folds are the ones inside; the flag must be monotone.
            assert flags == sorted(flags, reverse=True)

    def test_without_a_session_axis_it_is_none_never_false(self):
        """A standalone `RuleDiscovery` has nothing to compare against, and
        `None` says that. `False` would claim a clean fold on no evidence."""
        from forgedge.rule_discovery.walkforward import _split_timestamp

        candle = _wf_candle(n=3000)
        assert _split_timestamp(candle, "open_dt", None) is None
        # A split covering the whole span leaves nothing out-of-sample.
        assert _split_timestamp(
            candle, "open_dt", TimeBudget.build(len(candle), 1.0)
        ) is None

    def test_the_boundary_is_read_off_the_frame_the_folds_use(self):
        from forgedge.rule_discovery.walkforward import _split_timestamp

        candle = _wf_candle(n=3000)
        tb = TimeBudget.build(len(candle), 0.6)
        assert _split_timestamp(candle, "open_dt", tb) == candle["open_dt"].iloc[1800]
        # A budget longer than the frame is not extrapolated onto it.
        assert _split_timestamp(candle.head(100), "open_dt", tb) is None


class TestQuarantineIsOnePolicy:
    """The embargo is one policy on two boundaries; the purges are two
    quantities wearing one name."""

    def test_the_fold_embargo_follows_the_session_embargo(self):
        from forgedge import AlphaConfig, RuleDiscoveryConfig, config_report
        from forgedge.resolver import PipelineContext

        rep = config_report(None, AlphaConfig(embargo_bars=7), RuleDiscoveryConfig(),
                            ctx=PipelineContext(timeframe="1D"))
        assert rep.configs["rule_discovery"].walk_forward.embargo_bars == 7

    def test_an_explicit_fold_embargo_still_wins(self):
        from forgedge import AlphaConfig, RuleDiscoveryConfig, config_report
        from forgedge.rule_discovery.models import RuleWalkForwardConfig
        from forgedge.resolver import PipelineContext

        rd = RuleDiscoveryConfig(walk_forward=RuleWalkForwardConfig(embargo_bars=2))
        rep = config_report(None, AlphaConfig(embargo_bars=7), rd,
                            ctx=PipelineContext(timeframe="1D"))
        assert rep.configs["rule_discovery"].walk_forward.embargo_bars == 2

    def test_the_default_stays_zero_so_the_change_is_additive(self):
        from forgedge import AlphaConfig, RuleDiscoveryConfig, config_report
        from forgedge.resolver import PipelineContext

        rep = config_report(None, AlphaConfig(), RuleDiscoveryConfig(),
                            ctx=PipelineContext(timeframe="1D"))
        assert rep.configs["rule_discovery"].walk_forward.embargo_bars == 0
