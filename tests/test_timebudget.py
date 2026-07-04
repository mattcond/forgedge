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
