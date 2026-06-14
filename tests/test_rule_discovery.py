"""Tests for the Rule Discovery module (FORGE Modulo 3)."""
import math

import numpy as np
import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    AlphaDiscovery,
    DiscoveryConfig,
    EventDiscovery,
    RuleDiscovery,
    RuleDiscoveryConfig,
    RuleDiscoveryResponse,
)
from forgedge.rule_discovery import (
    BacktestParams,
    GridSpec,
    ScoringParams,
    SelectionCriteria,
    WalkForwardConfig,
    build_grid,
    deflated_sharpe,
    html_report,
    run_backtest,
    run_grid,
    select_best,
    text_report,
    validate,
    walk_forward,
)
from forgedge.rule_discovery import excursion_stats, execution_envelope
from forgedge.rule_discovery.validation import _ttest_1samp_greater


# ---------------------------------------------------------------------------
# Synthetic candle helpers
# ---------------------------------------------------------------------------

def _candle_with_signal(
    n: int = 4000,
    seed: int = 11,
    signal_every: int = 40,
    drift_after_signal: float = 0.05,
    intrabar: float = 0.01,
):
    """Build an OHLC table where a periodic signal precedes a real up-move.

    After each signal bar the price drifts up by ~``drift_after_signal`` over the
    next ~20 bars, so a limit buy at a small discount fills and the take-profit
    is reached — a genuinely profitable, regularly-spaced pattern.
    """
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    close[0] = 100.0
    signal = np.zeros(n, dtype=int)
    boost = np.zeros(n)
    for i in range(0, n, signal_every):
        signal[i] = 1
        # inject upward drift over the following 20 bars
        for j in range(i + 1, min(i + 21, n)):
            boost[j] += drift_after_signal / 20.0
    noise = rng.normal(0.0, 0.002, n)
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + boost[i] + noise[i])
    high = close * (1.0 + intrabar)
    low = close * (1.0 - intrabar)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open_dt": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "__sig__": signal,
        }
    )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_market_buy_fills_every_signal(self):
        df = _candle_with_signal(n=2000, signal_every=50)
        params = BacktestParams(buy_type="market", sell_pct=0.03, target_h=24)
        s = run_backtest(df, "__sig__", params)
        # Market orders always fill → fill_rate is 1.0.
        assert s.fill_rate == pytest.approx(1.0)
        assert s.total_trades == s.total_signals

    def test_limit_fill_rate_below_one(self):
        df = _candle_with_signal(n=2000, signal_every=50, intrabar=0.003)
        params = BacktestParams(buy_type="limit", buy_drop_pct=0.02, buy_delay_bar=4)
        s = run_backtest(df, "__sig__", params)
        # A 2% discount with only 0.3% intrabar range rarely fills.
        assert 0.0 <= s.fill_rate < 1.0

    def test_profitable_pattern_has_pf_above_one(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        params = BacktestParams(
            buy_type="limit", buy_drop_pct=0.005, buy_delay_bar=6,
            sell_pct=0.03, target_h=24,
        )
        s = run_backtest(df, "__sig__", params)
        assert s.total_trades > 20
        assert s.profit_factor > 1.0
        assert s.win_rate_pct > 0.5

    def test_early_stopping_changes_exit(self):
        df = _candle_with_signal(n=2000, signal_every=40, drift_after_signal=0.06)
        p_es = BacktestParams(buy_drop_pct=0.005, sell_pct=0.03, early_stopping=True)
        p_no = BacktestParams(buy_drop_pct=0.005, sell_pct=0.03, early_stopping=False)
        s_es = run_backtest(df, "__sig__", p_es)
        s_no = run_backtest(df, "__sig__", p_no)
        # Early stopping caps gains at the target → higher target-hit rate.
        assert s_es.target_hit_rate_pct >= s_no.target_hit_rate_pct

    def test_fee_reduces_expectancy(self):
        df = _candle_with_signal(n=2000, signal_every=40, drift_after_signal=0.06)
        lo = run_backtest(df, "__sig__", BacktestParams(fee=0.0, buy_drop_pct=0.005))
        hi = run_backtest(df, "__sig__", BacktestParams(fee=0.01, buy_drop_pct=0.005))
        assert lo.expectancy > hi.expectancy

    def test_timerange_restricts_signals(self):
        df = _candle_with_signal(n=4000, signal_every=40)
        full = run_backtest(df, "__sig__", BacktestParams(buy_type="market"))
        half = run_backtest(
            df, "__sig__", BacktestParams(buy_type="market"),
            timerange_from="2023-01-01", timerange_to="2023-03-01",
        )
        assert half.total_signals < full.total_signals

    def test_no_signal_returns_empty_summary(self):
        df = _candle_with_signal(n=500)
        df["__sig__"] = 0
        s = run_backtest(df, "__sig__", BacktestParams())
        assert s.total_trades == 0
        assert s.pf_score_tpm == 0.0

    def test_invalid_buy_type_raises(self):
        df = _candle_with_signal(n=200)
        with pytest.raises(ValueError):
            run_backtest(df, "__sig__", BacktestParams(buy_type="stop"))

    def test_exit_convention_close_vs_high_handcrafted(self):
        """Deterministic case isolating the take-profit detection column.

        Signal on bar 0 (close=100); limit buy at -2% = 98 fills on bar 1
        (low=97). target_h=3, sell_pct=6% → sell_price=103.88. In the exit
        window one bar's *high* reaches 104 but no bar *closes* above 103.88,
        so 'high' books a TARGET_HIT while 'close' closes at the horizon.
        """
        dates = pd.date_range("2023-01-01", periods=6, freq="1h")
        df = pd.DataFrame({
            "open_dt": dates,
            "open":  [100, 98.5, 101, 100, 100, 100.0],
            "high":  [100, 99.0, 104, 102,  101, 100.0],
            "low":   [100, 97.0, 100, 99.0, 99.0, 100.0],
            "close": [100, 98.5, 101, 99.0, 100, 100.0],
            "__sig__": [1, 0, 0, 0, 0, 0],
        })
        common = dict(buy_type="limit", buy_drop_pct=0.02, buy_delay_bar=4,
                      sell_pct=0.06, target_h=3, fee=0.002, early_stopping=True)
        s_hi, t_hi = run_backtest(
            df, "__sig__", BacktestParams(target_hit_col="high", **common),
            return_trades=True,
        )
        s_cl, t_cl = run_backtest(
            df, "__sig__", BacktestParams(target_hit_col="close", **common),
            return_trades=True,
        )
        assert s_hi.total_trades == 1 and s_cl.total_trades == 1
        # high convention → target hit at sell_price; close → close at horizon.
        assert bool(t_hi["target_hit"].iloc[0]) is True
        assert bool(t_cl["target_hit"].iloc[0]) is False
        assert t_hi["exit_price"].iloc[0] == pytest.approx(98 * 1.06)
        assert t_cl["exit_price"].iloc[0] == pytest.approx(100.0)  # horizon close

    def test_high_hit_rate_at_least_close_hit_rate(self):
        df = _candle_with_signal(n=3000, signal_every=40, drift_after_signal=0.05)
        common = dict(buy_drop_pct=0.005, sell_pct=0.03, target_h=24)
        hi = run_backtest(df, "__sig__", BacktestParams(target_hit_col="high", **common))
        cl = run_backtest(df, "__sig__", BacktestParams(target_hit_col="close", **common))
        # Intrabar touch can only book *more* target hits than a close check.
        assert hi.target_hit_rate_pct >= cl.target_hit_rate_pct

    def test_return_trades_shape(self):
        df = _candle_with_signal(n=2000, signal_every=40, drift_after_signal=0.06)
        s, trades = run_backtest(
            df, "__sig__", BacktestParams(buy_drop_pct=0.005), return_trades=True
        )
        assert len(trades) == s.total_trades
        assert {"net_pct_gain", "fill_dt", "exit_dt", "target_hit"} <= set(trades.columns)


# ---------------------------------------------------------------------------
# Short direction
# ---------------------------------------------------------------------------

def _candle_with_short_signal(n=4000, seed=21, signal_every=40,
                              drop_after_signal=0.06, intrabar=0.01):
    """Mirror of ``_candle_with_signal``: a signal precedes a real *down* move,
    so a short entered at a small premium fills and the downside target is hit."""
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    close[0] = 100.0
    signal = np.zeros(n, dtype=int)
    drag = np.zeros(n)
    for i in range(0, n, signal_every):
        signal[i] = 1
        for j in range(i + 1, min(i + 21, n)):
            drag[j] -= drop_after_signal / 20.0
    noise = rng.normal(0.0, 0.002, n)
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + drag[i] + noise[i])
    high = close * (1.0 + intrabar)
    low = close * (1.0 - intrabar)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open_dt": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": np.abs(rng.normal(1e6, 1e5, n)), "__sig__": signal,
    })


class TestShortDirection:
    def test_default_is_long(self):
        assert BacktestParams().direction == "long"

    def test_invalid_direction_raises(self):
        df = _candle_with_signal(n=200)
        with pytest.raises(ValueError):
            run_backtest(df, "__sig__", BacktestParams(direction="sideways"))

    def test_short_profits_on_downmove(self):
        df = _candle_with_short_signal(n=4000, signal_every=40, drop_after_signal=0.06)
        params = BacktestParams(direction="short", buy_type="limit",
                                buy_drop_pct=0.005, buy_delay_bar=6,
                                sell_pct=0.03, target_h=24)
        s = run_backtest(df, "__sig__", params)
        assert s.total_trades > 20
        assert s.profit_factor > 1.0
        assert s.win_rate_pct > 0.5

    def test_short_market_fills_every_signal(self):
        df = _candle_with_short_signal(n=2000, signal_every=50)
        s = run_backtest(df, "__sig__", BacktestParams(direction="short", buy_type="market"))
        assert s.fill_rate == pytest.approx(1.0)
        assert s.total_trades == s.total_signals

    def test_short_limit_fill_mirrors_long_on_symmetric_data(self):
        """On a driftless symmetric random walk the short's limit fill rate must
        mirror the long's. Guards against the short reusing the long fill
        comparison (which pins the short fill near 1.0)."""
        rng = np.random.default_rng(3)
        n = 6000
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        intr = np.abs(rng.normal(0.0, 0.005, n))
        df = pd.DataFrame({
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "open": close, "high": close * (1 + intr), "low": close * (1 - intr),
            "close": close, "volume": 1.0,
            "__sig__": (rng.random(n) < 0.1).astype(int),
        })
        common = dict(buy_type="limit", buy_drop_pct=0.02, buy_delay_bar=4, sell_pct=0.06)
        fl = run_backtest(df, "__sig__", BacktestParams(direction="long", **common)).fill_rate
        fs = run_backtest(df, "__sig__", BacktestParams(direction="short", **common)).fill_rate
        assert abs(fl - fs) < 0.10        # symmetric fills
        assert fs < 0.6                   # NOT pinned near 1.0 (the bug symptom)

    def test_short_envelope_and_excursion(self):
        df = _candle_with_short_signal(n=4000, signal_every=40, drop_after_signal=0.06)
        params = BacktestParams(direction="short", buy_drop_pct=0.005,
                                sell_pct=0.03, target_h=24)
        env = execution_envelope(df, "__sig__", params)
        # Optimistic short hits the target via low → at least as many hits.
        assert env.optimistic.target_hit_rate_pct >= env.conservative.target_hit_rate_pct
        _, trades = run_backtest(df, "__sig__", params, return_trades=True)
        ex = excursion_stats(df, trades)
        assert ex is not None
        # P&L-framed: adverse ≤ favourable for the short too.
        assert ex.mae_mean <= ex.mfe_mean

    def test_short_handcrafted_fill_and_target(self):
        """Short entry at +2% premium fills on a high spike; target at -6% is hit
        when a later bar's low reaches it."""
        dates = pd.date_range("2023-01-01", periods=6, freq="1h")
        # entry = 100*1.02 = 102 (short). target = 102*0.94 = 95.88.
        df = pd.DataFrame({
            "open_dt": dates,
            "open":  [100, 101, 100, 97, 96, 96.0],
            "high":  [100, 103, 101, 98, 97, 96.0],   # bar1 high=103 ≥ 102 → fill
            "low":   [100, 101, 96, 95, 95, 96.0],     # bar3 low=95 ≤ 95.88 → TP (high conv)
            "close": [100, 102, 99, 96, 96, 96.0],
            "__sig__": [1, 0, 0, 0, 0, 0],
        })
        params = BacktestParams(direction="short", buy_type="limit",
                                buy_drop_pct=0.02, buy_delay_bar=4,
                                sell_pct=0.06, target_h=3, fee=0.0,
                                target_hit_col="low")
        s, t = run_backtest(df, "__sig__", params, return_trades=True)
        assert s.total_trades == 1
        assert t["buy_price"].iloc[0] == pytest.approx(102.0)
        # Fill is the first bar whose HIGH reaches the +2% premium: bar 1 (high=103).
        # (Before the is_short fix this filled on bar 2 via the long comparison.)
        assert int(t["fill_rn"].iloc[0]) == 1
        assert bool(t["target_hit"].iloc[0]) is True
        assert t["exit_price"].iloc[0] == pytest.approx(102 * 0.94)
        # net = (entry - exit)/entry = 6%.
        assert t["net_pct_gain"].iloc[0] == pytest.approx(0.06, abs=1e-9)


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------

class TestScoring:
    def test_pf_score_tpm_in_range(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        s = run_backtest(
            df, "__sig__", BacktestParams(buy_drop_pct=0.005),
            scoring=ScoringParams(pf_tpm_target=3),
        )
        assert 0.0 <= s.pf_score_tpm <= max(s.profit_factor, 0.0) + 1e-9
        assert 0.0 <= s.c_norm <= 1.0

    def test_uniform_distribution_higher_cnorm(self):
        # Regular signal → near-uniform monthly distribution → decent c_norm.
        df = _candle_with_signal(n=6000, signal_every=30, drift_after_signal=0.06)
        s = run_backtest(
            df, "__sig__", BacktestParams(buy_type="market", sell_pct=0.03),
            scoring=ScoringParams(pf_tpm_target=3),
        )
        assert s.c_norm > 0.0


# ---------------------------------------------------------------------------
# Execution envelope & MAE/MFE
# ---------------------------------------------------------------------------

class TestRangeOfAction:
    def test_envelope_brackets_performance(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        params = BacktestParams(buy_drop_pct=0.005, sell_pct=0.03, target_h=24)
        env = execution_envelope(df, "__sig__", params)
        # Same fills, same trade count — only the exit detection differs.
        assert env.conservative.total_trades == env.optimistic.total_trades
        # Optimistic (high) can only book at least as many target hits as close.
        assert env.optimistic.target_hit_rate_pct >= env.conservative.target_hit_rate_pct

    def test_excursion_signs_and_bounds(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        params = BacktestParams(buy_drop_pct=0.005, sell_pct=0.03, target_h=24)
        _, trades = run_backtest(df, "__sig__", params, return_trades=True)
        ex = excursion_stats(df, trades)
        assert ex is not None and ex.n_trades > 0
        # MAE (deepest point) never exceeds MFE (highest point) of the window.
        assert ex.mae_worst <= ex.mae_mean
        assert ex.mfe_mean <= ex.mfe_best
        assert ex.mae_worst <= ex.mfe_best
        assert ex.mae_mean <= ex.mfe_mean
        assert 0.0 <= ex.mfe_reached_target_pct <= 100.0

    def test_excursion_negative_mae_when_underwater(self):
        """A trade that dips below the fill price must show a negative MAE."""
        dates = pd.date_range("2023-01-01", periods=6, freq="1h")
        # Fill at 98 on bar 1; bar 2 dips to low=94 (underwater), then recovers.
        df = pd.DataFrame({
            "open_dt": dates,
            "open":  [100, 98.5, 96, 99, 100, 100.0],
            "high":  [100, 99.0, 97, 101, 101, 100.0],
            "low":   [100, 97.0, 94, 98, 99, 100.0],
            "close": [100, 98.5, 96, 100, 100, 100.0],
            "__sig__": [1, 0, 0, 0, 0, 0],
        })
        params = BacktestParams(buy_type="limit", buy_drop_pct=0.02, buy_delay_bar=4,
                                sell_pct=0.06, target_h=3, fee=0.002)
        _, trades = run_backtest(df, "__sig__", params, return_trades=True)
        ex = excursion_stats(df, trades)
        assert ex is not None and ex.n_trades == 1
        # buy_price=98, deepest low=94 → MAE = (94-98)/98 ≈ -4.08%.
        assert ex.mae_worst == pytest.approx((94 - 98) / 98, abs=1e-6)

    def test_excursion_none_when_no_trades(self):
        df = _candle_with_signal(n=300)
        df["__sig__"] = 0
        _, trades = run_backtest(df, "__sig__", BacktestParams(), return_trades=True)
        assert excursion_stats(df, trades) is None

    def test_response_carries_envelope_and_excursion(self):
        df = _predictive_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        c = promoted[0]
        resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()
        if resp.in_sample_summary.total_trades > 0:
            assert resp.execution_envelope is not None
            assert resp.excursion is not None
            d = resp.to_dict()
            assert "execution_envelope" in d and "excursion" in d


# ---------------------------------------------------------------------------
# Grid screening
# ---------------------------------------------------------------------------

class TestGrid:
    def test_build_grid_fills_unset_axes(self):
        spec = build_grid(GridSpec(), BacktestParams(buy_drop_pct=0.01, sell_pct=0.04))
        assert spec.buy_drop_pct and spec.sell_pct and spec.target_h and spec.buy_delay_bar
        assert 0.01 in spec.buy_drop_pct

    def test_run_grid_sorted_by_score(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        spec = GridSpec(buy_drop_pct=[0.003, 0.005], sell_pct=[0.02, 0.03], target_h=[24])
        results = run_grid(df, "__sig__", BacktestParams(), spec)
        assert len(results) == 4
        scores = [r.summary.pf_score_tpm for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_prefers_qualified(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        results = run_grid(df, "__sig__", BacktestParams(), spec)
        best = select_best(results, SelectionCriteria(min_trades=5, min_profit_factor=1.0))
        assert best is not None


# ---------------------------------------------------------------------------
# Statistical validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_deflated_sharpe_haircut(self):
        # More trials → larger haircut → lower DSR.
        dsr_few = deflated_sharpe(2.0, n_trials=5, n_obs=100)
        dsr_many = deflated_sharpe(2.0, n_trials=500, n_obs=100)
        assert dsr_few > dsr_many
        assert dsr_many < 2.0

    def test_deflated_sharpe_identity_when_single_trial(self):
        assert deflated_sharpe(1.5, n_trials=1, n_obs=100) == pytest.approx(1.5)

    def test_ttest_detects_positive_mean(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0.5, 1.0, 300)
        t, p = _ttest_1samp_greater(sample, 0.0)
        assert t > 0 and p < 0.001

    def test_ttest_high_p_when_below_null(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(0.0, 1.0, 200)
        _, p = _ttest_1samp_greater(sample, 0.5)
        assert p > 0.9

    def test_validate_on_profitable_trades(self):
        df = _candle_with_signal(n=4000, signal_every=40, drift_after_signal=0.06)
        _, trades = run_backtest(
            df, "__sig__", BacktestParams(buy_drop_pct=0.005), return_trades=True
        )
        sv = validate(trades, base_rate=0.3, n_trials=40)
        assert sv.n_trials_tested == 40
        assert sv.temporal_stability in ("PASS", "WARN", "FAIL")
        assert math.isfinite(sv.ttest_expectancy_p)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_splits_are_chronological_and_non_overlapping(self):
        df = _candle_with_signal(n=12000, signal_every=30, drift_after_signal=0.05)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4)
        wf = walk_forward(df, "__sig__", BacktestParams(), spec, cfg)
        assert wf is not None
        assert len(wf.splits) >= 2
        for a, b in zip(wf.splits, wf.splits[1:]):
            assert a.test_to <= b.test_from  # no overlap, ordered
            assert a.train_from <= a.test_from

    def test_anchored_train_starts_at_origin(self):
        df = _candle_with_signal(n=12000, signal_every=30)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4, train_span_months=None)
        wf = walk_forward(df, "__sig__", BacktestParams(), spec, cfg)
        origins = {s.train_from for s in wf.splits}
        assert len(origins) == 1  # anchored → all train windows share an origin

    def test_rolling_train_window_moves(self):
        df = _candle_with_signal(n=14000, signal_every=30)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4, train_span_months=3)
        wf = walk_forward(df, "__sig__", BacktestParams(), spec, cfg)
        origins = [s.train_from for s in wf.splits]
        assert len(set(origins)) > 1  # rolling → train origin advances

    def test_short_span_returns_none(self):
        df = _candle_with_signal(n=300, signal_every=20)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        wf = walk_forward(
            df, "__sig__", BacktestParams(), spec,
            WalkForwardConfig(n_splits=4, min_train_months=6),
        )
        assert wf is None

    def test_oos_diagnostics_present_and_bracketed(self):
        df = _candle_with_signal(n=12000, signal_every=30, drift_after_signal=0.05)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4)
        wf = walk_forward(df, "__sig__", BacktestParams(), spec, cfg, base_rate=0.3)
        assert wf is not None
        # OOS twins of the in-sample diagnostics are populated.
        assert wf.oos_envelope is not None
        assert wf.oos_excursion is not None
        assert wf.oos_validation is not None
        # The conservative OOS envelope equals the headline OOS summary (close).
        assert wf.oos_envelope.conservative.profit_factor == wf.oos_summary.profit_factor
        # Optimistic (high) books at least as many target hits as conservative.
        assert (wf.oos_envelope.optimistic.target_hit_rate_pct
                >= wf.oos_envelope.conservative.target_hit_rate_pct)
        # OOS Sharpe is not deflated (no selection bias on OOS data).
        assert wf.oos_validation.n_trials_tested == 1

    def test_oos_trades_match_concatenated_splits(self):
        df = _candle_with_signal(n=12000, signal_every=30, drift_after_signal=0.05)
        spec = GridSpec(buy_drop_pct=[0.005], sell_pct=[0.03], target_h=[24])
        cfg = WalkForwardConfig(n_splits=3, min_train_months=4)
        wf = walk_forward(df, "__sig__", BacktestParams(), spec, cfg)
        total = sum(s.test_summary.total_trades for s in wf.splits)
        assert wf.oos_trades is not None
        assert len(wf.oos_trades) == total


# ---------------------------------------------------------------------------
# End-to-end over the FORGE pipeline
# ---------------------------------------------------------------------------

def _predictive_kpi_table(n=8000, seed=7):
    """Mean-reversion table (mirrors the Alpha Discovery fixture) with OHLC."""
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    k = 0.02
    noise = rng.normal(0.0, 0.004, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -k * (feat[:-1] - 0.5) + noise[1:]
    close = 100.0 * np.exp(np.cumsum(r))
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open_dt": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": feat,
        }
    )


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def pipeline(self):
        df = _predictive_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN", timeframe="1H"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        return ed, cands, promoted, by_id

    def test_pipeline_produces_response(self, pipeline):
        ed, _, promoted, by_id = pipeline
        assert promoted
        c = promoted[0]
        rd = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id])
        resp = rd.run()
        assert resp.verdict in ("EDGE", "PARTIAL-EDGE", "NON-EDGE")
        assert resp.alpha_id == c.alpha_id
        assert resp.in_sample_summary.total_signals >= 0

    def test_signal_matches_event_discovery_activations(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        cand = by_id[c.event_candidate_id]
        rd = RuleDiscovery(ed.df, c, cand)
        rd.run()
        injected = rd._frame[rd.config.signal_col].to_numpy()
        expected = cand.event_series.reindex(rd._frame.index).fillna(0).to_numpy()
        assert np.array_equal(injected, expected)

    def test_signal_reevaluated_when_index_differs(self, pipeline):
        # When the observed candles carry a timestamp index disjoint from the
        # candidate's stored activation series, the event must be re-evaluated as
        # an activation function (EventCandidate.apply), not reindexed: a blind
        # reindex would map every bar to NaN→inactive and backtest a rule that
        # never fires.  Features are unchanged here — only the timestamps move —
        # so re-evaluation recovers the genuine activations.
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        cand = by_id[c.event_candidate_id]
        shifted = ed.df.copy()
        shifted.index = shifted.index + pd.Timedelta(days=3650)
        assert not cand.event_series.index.equals(shifted.index)

        with pytest.warns(UserWarning, match="differs from the"):
            rd = RuleDiscovery(shifted, c, cand)
            rd._inject_signal()
        injected = rd._frame[rd.config.signal_col].to_numpy()

        # The old blind-reindex path would have collapsed this to all-zeros.
        assert injected.sum() > 0
        expected = cand.apply(rd._frame).fillna(0).to_numpy()
        assert np.array_equal(injected, expected)

    def test_mismatched_candidate_raises(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        other = next(cand for cid, cand in by_id.items() if cid != c.event_candidate_id)
        with pytest.raises(ValueError):
            RuleDiscovery(ed.df, c, other)

    def test_edge_response_carries_validated_rule(self, pipeline):
        ed, _, promoted, by_id = pipeline
        # Find any non-NON-EDGE verdict; otherwise assert all carry reasons.
        for c in promoted[:8]:
            rd = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id])
            resp = rd.run()
            if resp.is_edge:
                assert resp.validated_rule is not None
                assert resp.validated_rule.expression == c.event_expression
            else:
                assert resp.rejection_reasons

    def test_reports_render(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()
        txt = text_report(resp)
        assert resp.verdict in txt
        htm = html_report(resp)
        assert "<html" in htm and resp.verdict in htm

    def test_response_serialises_to_dict(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()
        d = resp.to_dict()
        assert d["verdict"] == resp.verdict
        assert "in_sample_results" in d

    def test_expression_propagated_unchanged(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()
        if resp.validated_rule:
            assert resp.validated_rule.expression == c.event_expression

    def test_accepts_datetime_index(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        indexed = ed.df.copy()
        rd = RuleDiscovery(indexed, c, by_id[c.event_candidate_id])
        resp = rd.run()
        assert resp.verdict in ("EDGE", "PARTIAL-EDGE", "NON-EDGE")

    def test_response_persist_roundtrip(self, pipeline, tmp_path):
        """RuleDiscoveryResponse.persist pickles the contract; it reloads identically."""
        import pickle
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()

        path = tmp_path / "rule_contract.pkl"
        assert resp.persist(path) is None        # mirrors EventCandidate.persist
        assert path.exists()

        reloaded = pickle.loads(path.read_bytes())
        assert isinstance(reloaded, RuleDiscoveryResponse)
        assert reloaded.verdict == resp.verdict
        assert reloaded.alpha_id == resp.alpha_id
        assert reloaded.to_dict() == resp.to_dict()


# ---------------------------------------------------------------------------
# Config seeding
# ---------------------------------------------------------------------------

class TestConfig:
    def test_contract_target_seeds_params(self):
        df = _predictive_kpi_table(n=6000)
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        c = promoted[0]
        rd = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id],
                           RuleDiscoveryConfig(use_contract_target=True))
        base = rd._seed_base_params([])
        assert base.target_h == int(c.derived_target.holding_period_h)

    def test_early_elimination_toggle(self):
        """A rule that fails the fast screen is short-circuited by default
        (no diagnostics); with early_elimination=False the full pipeline runs."""
        df = _predictive_kpi_table(n=8000)
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        c = promoted[0]

        # A 30% limit discount never fills (1% intrabar) → fast-screen NON-EDGE.
        def make_cfg(early):
            return RuleDiscoveryConfig(
                grid=GridSpec(buy_drop_pct=[0.30], sell_pct=[0.05], target_h=[12]),
                walk_forward=WalkForwardConfig(n_splits=2, min_train_months=3),
                criteria=SelectionCriteria(early_elimination=early),
            )

        # Default: short-circuit → diagnostics skipped.
        r_on = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id], make_cfg(True)).run()
        assert r_on.verdict == "NON-EDGE"
        assert r_on.walk_forward is None
        assert r_on.regime_analysis is None
        assert r_on.execution_envelope is None
        assert r_on.rejection_reasons

        # Disabled: full pipeline runs; still NON-EDGE but diagnostics populated.
        r_off = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id], make_cfg(False)).run()
        assert r_off.verdict == "NON-EDGE"
        assert r_off.execution_envelope is not None
        assert r_off.regime_analysis is not None
        assert r_off.walk_forward is not None
        assert r_off.rejection_reasons
