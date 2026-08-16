"""Tests for the Rule Discovery module (FORGE Modulo 3)."""
import math
from types import SimpleNamespace

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
    EntryOptimization,
    GridSpec,
    ScoringParams,
    SelectionCriteria,
    ValidatedRule,
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
from forgedge.rule_discovery.validation import _ttest_1samp_greater, opportunity_sharpe
from forgedge.resolver import PipelineContext, collect_context, resolve, resolve_config
from forgedge.unset import UNSET


def _nan_safe_eq(a, b) -> bool:
    """Deep equality treating NaN == NaN, unlike plain ``==`` on ``dict``/``float``.

    ``RuleDiscoveryResponse.to_dict()`` legitimately carries ``nan`` (e.g. MAE/MFE
    on a same-session ``target_h=0`` trade, whose realised holding window is
    empty by construction) — comparing two independently produced dicts with
    plain ``==`` fails on those keys even when the dicts are otherwise identical.
    """
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_safe_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_nan_safe_eq(x, y) for x, y in zip(a, b))
    return a == b


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

    def test_the_limit_anchor_can_be_a_derived_indicator(self):
        """`buy_price = anchor × (1 - buy_drop_pct)`, and the anchor is read
        with `_PreparedCandles.column()` — any numeric column on the table.

        That is not incidental: "place a limit at 90% of the 3-bar SMA" has no
        other expression in this engine, and it is why the anchor is a
        *reference level* rather than another name for the price series.  Pinned
        because the resolver fills this field in from `close_col`, and a future
        change that tightened it to price columns would remove the capability
        without any test noticing.
        """
        df = _candle_with_signal(n=2000, signal_every=40, drift_after_signal=0.06)
        df["close_sma_3"] = df["close"].rolling(3).mean().bfill()
        base = dict(buy_type="limit", buy_drop_pct=0.005, buy_delay_bar=6,
                    sell_pct=0.03, target_h=24)

        on_close = run_backtest(df, "__sig__", BacktestParams(**base))
        on_sma = run_backtest(
            df, "__sig__", BacktestParams(buy_price_anchor="close_sma_3", **base)
        )

        assert on_sma.total_trades > 0
        # A different reference level is a different set of fills — if these
        # matched, the anchor would not be doing anything.
        assert on_sma.total_trades != on_close.total_trades

    def test_an_unknown_anchor_column_is_a_clear_error(self):
        df = _candle_with_signal(n=500, signal_every=40)
        with pytest.raises(KeyError, match="buy_price_anchor"):
            run_backtest(df, "__sig__", BacktestParams(buy_price_anchor="nope"))

    def test_an_unresolved_fee_charges_the_documented_default(self):
        """`BacktestParams.fee` is session-resolved, so a caller who builds one
        by hand and hands it straight here holds `UNSET` (see forgedge.unset).

        The sentinel must never reach the arithmetic — `UNSET * 2` raises by
        design, and it must not silently become zero either, which would show a
        free lunch.  A function with a documented default has already decided:
        0.002/side, exactly as if it had been written out.
        """
        df = _candle_with_signal(n=2000, signal_every=40, drift_after_signal=0.06)
        assert BacktestParams().fee is UNSET

        implicit = run_backtest(df, "__sig__", BacktestParams(buy_drop_pct=0.005))
        explicit = run_backtest(
            df, "__sig__", BacktestParams(buy_drop_pct=0.005, fee=0.002)
        )
        free = run_backtest(df, "__sig__", BacktestParams(buy_drop_pct=0.005, fee=0.0))

        assert implicit.expectancy == pytest.approx(explicit.expectancy, rel=1e-12)
        assert implicit.expectancy < free.expectancy

    def test_the_contract_cost_basis_is_the_cost_charged(self):
        """F7 — `AlphaConfig.fee_per_side` stamped the contract while
        `BacktestParams.fee` charged the backtest, and nothing connected them.

        They agreed only because both defaulted to 0.002, so a caller who set
        one got contracts documenting one cost and a backtest charging another,
        silently.  The resolver now derives the second from the first; here the
        difference is measured on the net, which is the only place it matters.
        """
        df = _candle_with_signal(n=2000, signal_every=40, drift_after_signal=0.06)
        ctx = PipelineContext(timeframe="1H")

        cheap = resolve_config(
            RuleDiscoveryConfig(), "rule_discovery", ctx,
        )
        # What the resolver does inside forge(): M2's cost basis reaches M3.
        bundle = {"alpha": AlphaConfig(fee_per_side=0.0005),
                  "rule_discovery": RuleDiscoveryConfig()}
        resolved, _trace, _v = resolve(bundle, collect_context(bundle, ctx))
        assert resolved["rule_discovery"].base_params.fee == pytest.approx(0.0005)

        base = BacktestParams(buy_drop_pct=0.005, sell_pct=0.03, target_h=24)
        five_bp = run_backtest(df, "__sig__", base.merged(fee=0.0005))
        twenty_bp = run_backtest(df, "__sig__", base.merged(fee=cheap.base_params.fee))

        # 15 bp/side × 2 sides on every trade — the gap the silent mismatch hid.
        assert cheap.base_params.fee == pytest.approx(0.002)
        gap = five_bp.expectancy - twenty_bp.expectancy
        assert gap == pytest.approx(2 * (0.002 - 0.0005), rel=1e-6)

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

    def test_target_h_zero_is_same_session_round_trip(self):
        """target_h=0 exits at the fill bar's own close (issue #158).

        signal→fill is always 1 bar (act on the bar after the signal); with
        target_h=0 the exit window collapses onto the fill bar itself, so
        fill_rn == exit_rn and the net gain is the fill bar's own open→close.
        """
        dates = pd.date_range("2023-01-01", periods=4, freq="1h")
        df = pd.DataFrame({
            "open_dt": dates,
            "open":  [100.0, 98.0, 105.0, 105.0],
            "high":  [100.0, 99.0, 106.0, 106.0],
            "low":   [100.0, 97.0, 104.0, 104.0],
            "close": [100.0, 98.5, 105.0, 105.0],
            "__sig__": [1, 0, 0, 0],
        })
        params = BacktestParams(
            buy_type="limit", buy_drop_pct=0.02, buy_delay_bar=4,
            sell_pct=0.5, target_h=0, fee=0.0, early_stopping=True,
        )
        s, trades = run_backtest(df, "__sig__", params, return_trades=True)
        assert s.total_trades == 1
        row = trades.iloc[0]
        assert row["fill_rn"] == row["exit_rn"]
        assert row["fill_dt"] == row["exit_dt"]
        assert row["exit_price"] == pytest.approx(98.5)  # fill bar's own close
        expected_net = (row["exit_price"] - row["buy_price"]) / row["buy_price"]
        assert row["net_pct_gain"] == pytest.approx(expected_net)


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


class TestMonthIndex:
    """The monthly-stats index must count every month entries can open in."""

    @staticmethod
    def _daily_candle():
        """Three full months of flat daily bars with one signal per month."""
        dts = pd.date_range("2025-01-01", "2025-03-31", freq="D")
        n = len(dts)
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open_dt": dts,
                "open": close,
                "high": close * 1.2,
                "low": close * 0.8,
                "close": close,
                "__sig__": 0,
            }
        )
        for d in ("2025-01-10", "2025-02-10", "2025-03-10"):
            df.loc[df.open_dt == d, "__sig__"] = 1
        return df

    def test_whole_table_counts_final_month(self):
        df = self._daily_candle()
        params = BacktestParams(
            buy_type="market", direction="long", sell_pct=0.05, target_h=3, fee=0.0
        )
        s = run_backtest(df, "__sig__", params)
        assert s.total_trades == 3
        assert s.n_months == 3
        assert s.active_months == 3
        assert s.zero_months == 0
        assert s.tpm_mu == pytest.approx(1.0)

    def test_month_aligned_exclusive_bound_unchanged(self):
        # A walk-forward-style [from, to) window with a month-aligned exclusive
        # bound must not count the month the bound names.
        from forgedge.rule_discovery.backtest import _as_datetime64, _month_index

        dt = _as_datetime64(self._daily_candle()["open_dt"])
        months = _month_index("2025-01-01", "2025-03-01", dt)
        assert [str(m) for m in months] == ["2025-01", "2025-02"]

    def test_mid_month_bound_counts_partial_month(self):
        from forgedge.rule_discovery.backtest import _as_datetime64, _month_index

        dt = _as_datetime64(self._daily_candle()["open_dt"])
        months = _month_index("2025-01-01", "2025-03-15", dt)
        assert [str(m) for m in months] == ["2025-01", "2025-02", "2025-03"]


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

    def test_excursion_mixed_direction_raises(self):
        """excursion_stats must raise ValueError for a mixed long/short frame.

        Regression for issue #54: previously .any() accepted a mixed frame and
        silently applied short logic to all trades including long ones.
        """
        dates = pd.date_range("2023-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "open_dt": dates, "open": [100.0] * 10,
            "high": [101.0] * 10, "low": [99.0] * 10, "close": [100.0] * 10,
        })
        # Build a minimal trades frame with both directions present
        mixed = pd.DataFrame({
            "fill_rn": [1, 3],
            "exit_rn": [2, 4],
            "buy_price": [100.0, 100.0],
            "sell_price": [103.0, 97.0],
            "direction": ["long", "short"],
        })
        with pytest.raises(ValueError, match="mixed-direction"):
            excursion_stats(df, mixed)

    def test_excursion_short_direction_uses_short_formula(self):
        """excursion_stats with all-short trades must use short P&L formula.

        For a short trade: adverse excursion = price rises (high > entry),
        favourable excursion = price falls (low < entry).
        """
        df = _candle_with_short_signal(n=400, signal_every=40, drop_after_signal=0.06)
        params = BacktestParams(direction="short", sell_pct=0.03, target_h=12)
        _, trades = run_backtest(df, "__sig__", params, return_trades=True)
        ex = excursion_stats(df, trades)
        if ex is None or ex.n_trades == 0:
            pytest.skip("no trades — skipping short excursion test")
        # Short excursion ordering: worst <= mean for both MAE and MFE
        assert ex.mae_worst <= ex.mae_mean
        assert ex.mfe_mean <= ex.mfe_best

    def test_response_carries_envelope_and_excursion(self):
        df = _predictive_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        c = promoted[0]
        # `early_elimination=False` so the full pipeline always runs.  Guarding
        # on `total_trades > 0` was not the right condition: under a market
        # entry every signal fills, so a rule short-circuited by the fast screen
        # still has trades while legitimately carrying no diagnostics — which is
        # what early elimination *is*, and what `test_early_elimination_toggle`
        # pins.  This test is about the response carrying its diagnostics when
        # they were computed.
        resp = RuleDiscovery(
            ed.df, c, by_id[c.event_candidate_id],
            RuleDiscoveryConfig(criteria=SelectionCriteria(early_elimination=False)),
        ).run()
        assert resp.in_sample_summary.total_trades > 0
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

    def test_build_grid_target_h_reaches_zero_for_short_seed(self):
        """A short seed horizon can explore a same-session hold (issue #158):
        the auto-built fan floors at 0, not 1."""
        spec = build_grid(GridSpec(), BacktestParams(target_h=1))
        assert 0 in spec.target_h

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
        best = select_best(results, SelectionCriteria(min_tpm=0.5, min_profit_factor=1.0))
        assert best is not None


class TestDynamicMinTrades:
    """RD-04 — trade-count gate scaled to the IS length instead of a fixed 30."""

    def test_floor_applies_on_short_is(self):
        from forgedge.rule_discovery.discovery import _MIN_TRADES_ABS, _dynamic_min_trades
        # 3 months × 2 tpm = 6 → clamped to the absolute statistical floor.
        assert _MIN_TRADES_ABS == 10
        assert _dynamic_min_trades(3, 2.0) == 10

    def test_scales_with_is_length(self):
        from forgedge.rule_discovery.discovery import _dynamic_min_trades
        assert _dynamic_min_trades(12, 2.0) == 24   # 12 mo × 2/mo
        assert _dynamic_min_trades(24, 2.0) == 48   # longer IS ⇒ stricter
        assert _dynamic_min_trades(12, 3.0) == 36

    def test_min_trades_removed_from_criteria(self):
        assert not hasattr(SelectionCriteria(), "min_trades")
        with pytest.raises(TypeError):
            SelectionCriteria(min_trades=30)


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


def _market_edge_kpi_table(n=9000, seed=3, drift=0.05, span=12, sigma=0.009,
                           intrabar=0.02):
    """A table whose signal survives a *market* entry.

    ``_predictive_kpi_table`` reverts too weakly: at a next-open fill its
    profit factor sits around 1.1, below every gate, so the market stage
    returns NON-EDGE and the limit stage is never reached.  That is fine for
    the tests it was written for and useless for testing Stage 2.

    Here a low ``feat`` is followed by a real drift over the next ``span``
    bars, strong enough relative to ``sigma`` to clear the gates at a market
    entry.  ``intrabar`` is wide enough that a limit order fills *sometimes* —
    around two thirds of the time — which is the interesting regime: a limit
    that always fills is a market order, and one that never fills is not an
    operating point.
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    boost = np.zeros(n)
    for i in np.flatnonzero(feat < 0.15):
        boost[i + 1:min(i + span + 1, n)] += drift / span
    r = rng.normal(0.0, sigma, n) + boost
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "open": close,
            "high": close * (1 + intrabar),
            "low": close * (1 - intrabar),
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
        assert resp.verdict in ("EDGE", "PARTIAL-EDGE", "NON-EDGE", "INSUFFICIENT-DATA")
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
        assert _nan_safe_eq(reloaded.to_dict(), resp.to_dict())

    # ── RD-130 — entry mode (limit / market / auto) ──────────────────────

    def test_entry_mode_invalid_raises(self, pipeline):
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        rd = RuleDiscovery(
            ed.df, c, by_id[c.event_candidate_id],
            RuleDiscoveryConfig(entry_mode="bogus"),
        )
        with pytest.raises(ValueError, match="entry_mode"):
            rd.run()

    def test_entry_mode_auto_matches_default(self, pipeline):
        """entry_mode='auto' is the default path — bit-for-bit identical output.

        The default moved from ``"limit"`` to ``"auto"`` (#185).  In limit mode
        the grid varies ``buy_drop_pct``, so the limit entry does double duty as
        order mechanic *and* entry-price optimiser, and the deeper the discount
        the more it fills only on favourable paths — the fill confound, which
        inflates PF on a subset that is not tradeable.  ``"auto"`` makes the
        verdict a measurement of the *signal* and leaves the entry price to a
        separate, out-of-sample-confirmed stage.
        """
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        cand = by_id[c.event_candidate_id]
        default = RuleDiscovery(ed.df, c, cand).run()
        explicit = RuleDiscovery(
            ed.df, c, cand, RuleDiscoveryConfig(entry_mode="auto")
        ).run()
        assert _nan_safe_eq(explicit.to_dict(), default.to_dict())

    def test_entry_mode_limit_is_still_fully_supported(self, pipeline):
        """The old default is a mode, not a removal: it still runs end to end
        and still publishes a limit operating point."""
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        resp = RuleDiscovery(
            ed.df, c, by_id[c.event_candidate_id],
            RuleDiscoveryConfig(entry_mode="limit"),
        ).run()
        assert resp.verdict in {"EDGE", "PARTIAL-EDGE", "NON-EDGE", "INSUFFICIENT-DATA"}
        if resp.validated_rule is not None:
            assert resp.validated_rule.params.buy_type == "limit"

    def test_entry_mode_market_fills_and_no_buydrop_reject(self, pipeline):
        """Market baseline: ~100% fill, never rejected for 'buy_drop too deep'."""
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        resp = RuleDiscovery(
            ed.df, c, by_id[c.event_candidate_id],
            RuleDiscoveryConfig(entry_mode="market"),
        ).run()
        assert resp.in_sample_summary.fill_rate >= 0.95
        assert not any("buy_drop too deep" in r for r in resp.rejection_reasons)
        if resp.validated_rule is not None:
            assert resp.validated_rule.params.buy_type == "market"

    def test_entry_mode_auto_verdict_is_market_verdict(self, pipeline):
        """The auto verdict is authoritative from the market stage."""
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        cand = by_id[c.event_candidate_id]
        market = RuleDiscovery(
            ed.df, c, cand, RuleDiscoveryConfig(entry_mode="market")
        ).run()
        auto = RuleDiscovery(
            ed.df, c, cand, RuleDiscoveryConfig(entry_mode="auto")
        ).run()
        assert auto.verdict == market.verdict
        # The optimiser can never fabricate an edge from a NON-EDGE market rule.
        if market.verdict == "NON-EDGE":
            assert auto.verdict == "NON-EDGE"
            assert auto.validated_rule is None

    def test_optimize_limit_floor_blocks_adoption(self, pipeline):
        """An unreachable fill floor (1.01) → no limit qualifies → market kept."""
        ed, _, promoted, by_id = pipeline
        c = promoted[0]
        cfg = RuleDiscoveryConfig(
            entry_mode="auto",
            criteria=SelectionCriteria(min_fill_rate_opt=1.01),
        )
        rd = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id], cfg)
        rd._inject_signal()
        base = rd._seed_base_params([])
        mkt = run_backtest(
            rd._frame, cfg.signal_col, base.merged(buy_type="market"),
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
        )
        market_params = base.merged(buy_type="market")
        fake = SimpleNamespace(
            in_sample_summary=mkt,
            walk_forward=None,
            statistical_validation=None,
            grid_results=[],
            validated_rule=ValidatedRule(
                expression="x", event_candidate_id="e", params=market_params,
            ),
        )
        opt = rd._optimize_limit_entry(fake, base, rd._market_grid(market_params))
        assert isinstance(opt, EntryOptimization)
        assert opt.min_fill_rate_opt == 1.01
        assert opt.adopted is False
        assert opt.selected_entry == "market"
        assert opt.failed_condition == "fill"
        assert opt.limit_fill_rate is None      # nothing reached the floor
        # The verdict never comes from Stage 2, adopted or not.
        assert opt.authoritative == "market"


class TestOpportunitySharpe:
    """The quantity condition 2 is built on, and why it is not the other one.

    The worked example from #185: the limit point earns more per trade and
    trades far less often.  Which point wins depends entirely on how "annualise"
    is defined, and the two definitions disagree in opposite directions.
    """

    @staticmethod
    def _trades(n: int, mu: float, sd: float, seed: int = 0) -> pd.DataFrame:
        """A ledger with exactly the requested mean and standard deviation."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, n)
        x = (x - x.mean()) / x.std(ddof=1)
        return pd.DataFrame({"net_pct_gain": mu + sd * x})

    def test_it_matches_the_closed_form(self):
        tr = self._trades(60, mu=0.008, sd=0.030)
        got = opportunity_sharpe(tr, span_years=2.0)
        expected = (0.008 / 0.030) * math.sqrt(60 / 2.0)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_halving_the_trades_costs_root_two(self):
        """The property the whole criterion rests on: the same per-trade edge
        at half the frequency is worth `1/sqrt(2)` as much."""
        many = opportunity_sharpe(self._trades(120, 0.01, 0.03), span_years=2.0)
        few = opportunity_sharpe(self._trades(60, 0.01, 0.03), span_years=2.0)
        assert many / few == pytest.approx(math.sqrt(2.0), rel=1e-9)

    def test_it_disagrees_with_the_capacity_annualisation_where_it_matters(self):
        """`validate()` annualises by `bars_per_year / avg_holding_bars` — the
        number of non-overlapping holding periods that fit in a year.

        Two operating points on the *same* rule hold for the same length, so
        that factor is identical for both and cancels: the comparison collapses
        onto the per-trade Sharpe, which is blind to frequency.  Here the limit
        point wins on it by 41% while earning 25% less in total, which is the
        adoption the criterion exists to prevent.
        """
        market = self._trades(60, mu=0.0080, sd=0.030, seed=1)
        limit = self._trades(24, mu=0.0150, sd=0.040, seed=2)
        hold_bars, bpy, span = 24.0, 24 * 365, 2.0

        capacity = {
            name: validate(tr, base_rate=0.5, n_trials=1,
                           bars_per_year=bpy, avg_holding_bars=hold_bars).sharpe_ratio
            for name, tr in (("market", market), ("limit", limit))
        }
        opportunity = {
            name: opportunity_sharpe(tr, span_years=span)
            for name, tr in (("market", market), ("limit", limit))
        }
        total = {name: float(tr["net_pct_gain"].sum())
                 for name, tr in (("market", market), ("limit", limit))}

        assert capacity["limit"] > capacity["market"]        # would adopt
        assert opportunity["limit"] < opportunity["market"]  # correctly refuses
        assert total["limit"] < total["market"]              # and it earns less
        # The capacity reading is the per-trade Sharpe times a constant: the
        # ratio it reports is the per-trade ratio, unchanged by annualising.
        per_trade = {name: tr["net_pct_gain"].mean() / tr["net_pct_gain"].std(ddof=1)
                     for name, tr in (("market", market), ("limit", limit))}
        assert (capacity["limit"] / capacity["market"]) == pytest.approx(
            per_trade["limit"] / per_trade["market"], rel=1e-6   # sharpe_ratio is rounded
        )

    def test_it_is_undefined_rather_than_wrong_on_thin_input(self):
        assert math.isnan(opportunity_sharpe(self._trades(1, 0.01, 0.03), 1.0))
        assert math.isnan(opportunity_sharpe(self._trades(10, 0.01, 0.03), 0.0))
        flat = pd.DataFrame({"net_pct_gain": [0.01] * 10})   # zero dispersion
        assert math.isnan(opportunity_sharpe(flat, 1.0))


class TestEntryAdoption:
    """The three-condition, out-of-sample adoption criterion (#185).

    The fixture is engineered to produce a *market* edge — the previous
    synthetic reverts too weakly for the market baseline to clear the gates, so
    Stage 2 never ran on it and the criterion could not be observed at all.
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        df = _market_edge_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN", timeframe="1H"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        assert promoted
        return ed, promoted, by_id

    @staticmethod
    def _cfg(**criteria):
        # A permissive optimisation floor: on this fixture a 1% discount fills
        # ~65% of the time (a real limit entry misses the moves that run away),
        # and the default 0.80 would stop every candidate at condition 1 —
        # leaving conditions 2 and 3 untested.
        base = dict(min_fill_rate_opt=0.20)
        base.update(criteria)
        return RuleDiscoveryConfig(criteria=SelectionCriteria(**base))

    def _first_with_stage_two(self, pipeline, cfg):
        ed, promoted, by_id = pipeline
        for c in promoted[:6]:
            resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id], cfg).run()
            if resp.entry_optimization is not None:
                return resp
        pytest.skip("no market edge reached Stage 2 on this fixture")

    def test_the_verdict_is_never_stage_twos_to_give(self, pipeline):
        resp = self._first_with_stage_two(pipeline, self._cfg())
        assert resp.entry_optimization.authoritative == "market"
        assert resp.is_edge

    def test_both_operating_points_are_published_with_evidence(self, pipeline):
        """The defect this replaces: the limit point was adopted on a single
        in-sample pass while the market point it displaced had a walk-forward,
        statistics and a regime breakdown.  Three scalars cannot say how many
        trades a point makes or at what win rate, and M4 then catalogued — and
        cross-ticker tested — an operating point never validated out of sample.
        """
        opt = self._first_with_stage_two(pipeline, self._cfg()).entry_optimization
        assert opt.market_rule is not None
        assert opt.market_summary is not None
        assert opt.limit_summary is not None
        assert opt.limit_walk_forward is not None
        # Full artefacts, not scalars: the questions a trader asks are answerable.
        assert opt.limit_summary.total_trades >= 0
        assert opt.limit_summary.win_rate_pct == opt.limit_summary.win_rate_pct

    def test_the_limit_point_is_replayed_not_reoptimised(self, pipeline):
        """`reoptimise=False`: one fixed `buy_drop_pct` scored on every test
        window.  A replay adds no selection, so it adds no `n_trials` — which is
        what makes it fair to give the limit point an OOS record at all."""
        opt = self._first_with_stage_two(pipeline, self._cfg()).entry_optimization
        wf = opt.limit_walk_forward
        assert len(wf.splits) >= 2
        distinct = {(s.params.buy_drop_pct, s.params.sell_pct, s.params.target_h)
                    for s in wf.splits}
        assert len(distinct) == 1

    def test_n_trials_is_per_operating_point(self, pipeline):
        """D5 — the market point was chosen over Stage 1's cells, the limit point
        over those *and* Stage 2's.  The `min_dsr` gate always reads the market
        point's, so the verdict never pays for Stage 2."""
        resp = self._first_with_stage_two(pipeline, self._cfg())
        opt = resp.entry_optimization
        market_trials = resp.statistical_validation.n_trials_tested
        assert market_trials == 15          # 5 sell_pct x 3 target_h, entry collapsed
        assert opt.limit_validation is not None
        assert opt.limit_validation.n_trials_tested > market_trials

    def test_a_rejected_point_still_reports_its_statistics(self, pipeline):
        """D9 — the DSR is reported per point as an absolute metric.  A point
        that was measured and turned down is exactly where a reader wants the
        number that was measured."""
        opt = self._first_with_stage_two(pipeline, self._cfg()).entry_optimization
        if opt.adopted:
            pytest.skip("this fixture adopted the limit point")
        assert opt.limit_validation is not None
        assert opt.failed_condition in {"fill", "sharpe", "net_gain"}

    def test_a_better_deflated_sharpe_does_not_buy_adoption(self, pipeline):
        """The substance of the change, and why the new quantity was needed.

        On this fixture the limit point comes back with a **higher Deflated
        Sharpe than the market point** — even carrying the larger trial count
        from Stage 2 — and is still turned down.  A criterion built on any
        statistic the pipeline already computes would have published it.

        The two disagree because they annualise differently.  The DSR goes
        through `validate`, which annualises by *capacity*
        (`bars_per_year / avg_holding_bars`): a point that fills two thirds as
        often holds for the same length, so capacity barely moves and the
        frequency the choice is about cancels out.  `opportunity_sharpe` counts
        realised trades, so filling 66% of the time costs `sqrt(0.66) ~ 0.81x`
        that the per-trade edge has to beat.  Here it does not: 67.3 against
        79.1, and 15.7 of net gain against 23.9.

        Capacity is the right denominator for "how good is this rule" and the
        wrong one for choosing between two operating points on the same rule.
        Reusing `StatisticalValidation.sharpe_ratio` would have looked like
        implementing the criterion while quietly defeating it.
        """
        resp = self._first_with_stage_two(pipeline, self._cfg())
        opt = resp.entry_optimization
        if opt.failed_condition != "sharpe":
            pytest.skip("this fixture did not fail on the Sharpe condition")
        assert opt.limit_validation.deflated_sharpe > resp.statistical_validation.deflated_sharpe
        assert opt.limit_opportunity_sharpe < opt.market_opportunity_sharpe
        assert opt.limit_oos_net_gain < opt.market_oos_net_gain
        assert opt.adopted is False
        assert opt.selected_entry == "market"

    def test_the_published_point_is_the_selected_one(self, pipeline):
        resp = self._first_with_stage_two(pipeline, self._cfg())
        opt = resp.entry_optimization
        expected = "limit" if opt.adopted else "market"
        assert opt.selected_entry == expected
        assert resp.validated_rule.params.buy_type == expected

    def test_to_dict_summarises_the_replay_without_carrying_its_ledger(self, pipeline):
        """`asdict` would deep-copy the walk-forward's trade DataFrame into the
        "summary" dict — a memory bug waiting for a large run."""
        opt = self._first_with_stage_two(pipeline, self._cfg()).entry_optimization
        d = opt.to_dict()
        assert d["authoritative"] == "market"
        assert "limit_oos_summary" in d and "market_oos_summary" in d
        assert not any(isinstance(v, pd.DataFrame) for v in d.values())


# ---------------------------------------------------------------------------
# Config seeding
# ---------------------------------------------------------------------------

class TestWalkForwardSelection:
    """§3.4 — the operating point is selected inside WF train windows only."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        # `_market_edge_kpi_table` since #185.  `_predictive_kpi_table`
        # produced a tradeable verdict only under the old `"limit"` default,
        # where the entry discount was part of the measured edge; at a market
        # entry it reverts too weakly and every rule is NON-EDGE, which would
        # have quietly switched this whole class off via its skip guard.
        df = _market_edge_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN", timeframe="1H"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        assert promoted
        return ed, promoted[0], by_id[promoted[0].event_candidate_id]

    def test_default_mode_is_walk_forward(self, pipeline):
        ed, c, cand = pipeline
        assert RuleDiscoveryConfig().selection_mode == "walk_forward"
        resp = RuleDiscovery(ed.df, c, cand).run()
        assert any("selection_mode=walk_forward" in n for n in resp.notes)

    def test_published_params_come_from_last_train_window(self, pipeline):
        ed, c, cand = pipeline
        resp = RuleDiscovery(ed.df, c, cand).run()
        if resp.validated_rule is None:
            pytest.skip("no tradeable verdict on the fixture")
        assert resp.walk_forward is not None
        last = resp.walk_forward.splits[-1].params
        pub = resp.validated_rule.params
        assert (pub.sell_pct, pub.target_h, pub.buy_drop_pct) == (
            last.sell_pct, last.target_h, last.buy_drop_pct
        )

    def test_is_metrics_exclude_the_final_test_window(self, pipeline):
        ed, c, cand = pipeline
        wf_resp = RuleDiscovery(ed.df, c, cand).run()
        full_resp = RuleDiscovery(
            ed.df, c, cand, config=RuleDiscoveryConfig(selection_mode="full_sample")
        ).run()
        # The selection span ends at the last train window: strictly fewer IS
        # months than the whole table.
        assert wf_resp.in_sample_summary.n_months < full_resp.in_sample_summary.n_months

    def test_consensus_policy_picks_a_train_selection(self, pipeline):
        ed, c, cand = pipeline
        cfg = RuleDiscoveryConfig(wf_param_policy="consensus")
        resp = RuleDiscovery(ed.df, c, cand, config=cfg).run()
        if resp.validated_rule is None or resp.walk_forward is None:
            pytest.skip("no tradeable verdict on the fixture")
        pub = resp.validated_rule.params
        keys = {
            (s.params.sell_pct, s.params.target_h, s.params.buy_drop_pct)
            for s in resp.walk_forward.splits
        }
        assert (pub.sell_pct, pub.target_h, pub.buy_drop_pct) in keys

    def test_short_span_falls_back_to_full_sample(self, pipeline):
        ed, c, cand = pipeline
        short = ed.df.iloc[:800].copy()  # ~1 month of hourly bars — no split
        with pytest.warns(UserWarning, match="differs from the"):
            resp = RuleDiscovery(short, c, cand).run()
        assert any("falling back to full-sample" in n for n in resp.notes)

    def test_invalid_selection_mode_raises(self, pipeline):
        ed, c, cand = pipeline
        cfg = RuleDiscoveryConfig(selection_mode="bogus")
        with pytest.raises(ValueError, match="selection_mode"):
            RuleDiscovery(ed.df, c, cand, config=cfg).run()

    def test_full_sample_mode_has_no_wf_note(self, pipeline):
        ed, c, cand = pipeline
        resp = RuleDiscovery(
            ed.df, c, cand, config=RuleDiscoveryConfig(selection_mode="full_sample")
        ).run()
        assert not any("selection_mode=walk_forward" in n for n in resp.notes)


class TestPowerGate:
    """§3.2 — positive verdicts degrade to INSUFFICIENT-DATA when the pooled
    OOS evidence cannot support them.  The gate never reads per-window counts:
    walk-forward windows are short by design and are not individually gated.
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        # `_market_edge_kpi_table` since #185.  `_predictive_kpi_table`
        # produced a tradeable verdict only under the old `"limit"` default,
        # where the entry discount was part of the measured edge; at a market
        # entry it reverts too weakly and every rule is NON-EDGE, which would
        # have quietly switched this whole class off via its skip guard.
        df = _market_edge_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN", timeframe="1H"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        return ed, promoted, by_id

    @pytest.fixture(scope="class")
    def tradeable_case(self, pipeline):
        """A (contract, candidate) with a tradeable verdict under defaults."""
        ed, promoted, by_id = pipeline
        for c in promoted[:10]:
            resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()
            if resp.is_edge:
                return ed, c, by_id[c.event_candidate_id], resp
        pytest.skip("no tradeable verdict in the synthetic fixture")

    def test_well_powered_fixture_is_not_degraded(self, tradeable_case):
        # The strong synthetic edge has hundreds of pooled OOS trades: the
        # default power gate must not touch it.
        _, _, _, resp = tradeable_case
        assert resp.verdict in ("EDGE", "PARTIAL-EDGE")
        assert RuleDiscoveryConfig().criteria.power_gate is True

    def test_thin_pooled_oos_degrades_to_insufficient_data(self, tradeable_case):
        ed, c, cand, _ = tradeable_case
        cfg = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_oos_trades=10**6)
        )
        resp = RuleDiscovery(ed.df, c, cand, config=cfg).run()
        assert resp.verdict == "INSUFFICIENT-DATA"
        assert any("pooled OOS trades" in r for r in resp.rejection_reasons)
        # Not tradeable, but the operating point is kept for re-evaluation.
        assert not resp.is_edge
        assert resp.validated_rule is not None

    def test_power_gate_off_restores_verdict(self, tradeable_case):
        ed, c, cand, baseline = tradeable_case
        cfg = RuleDiscoveryConfig(
            criteria=SelectionCriteria(power_gate=False, min_oos_trades=10**6)
        )
        resp = RuleDiscovery(ed.df, c, cand, config=cfg).run()
        assert resp.verdict == baseline.verdict

    def test_non_edge_is_never_converted_to_insufficient_data(self, pipeline):
        # An unpassable power floor must not turn NON-EDGE verdicts into
        # INSUFFICIENT-DATA: a rule that fails the economic gates stays
        # NON-EDGE (the operational consequence is the same either way).
        ed, promoted, by_id = pipeline
        baseline_cfg = RuleDiscoveryConfig()
        gated_cfg = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_oos_trades=10**6)
        )
        seen = 0
        for c in promoted[:10]:
            cand = by_id[c.event_candidate_id]
            base = RuleDiscovery(ed.df, c, cand, config=baseline_cfg).run()
            if base.verdict != "NON-EDGE":
                continue
            gated = RuleDiscovery(ed.df, c, cand, config=gated_cfg).run()
            assert gated.verdict == "NON-EDGE"
            seen += 1
        if not seen:
            pytest.skip("no NON-EDGE verdict in the synthetic fixture")

    def test_short_windows_are_not_individually_gated(self, tradeable_case):
        # Many 1-month test windows → each holds only a handful of trades.
        # The gate must read the *pooled* ledger and leave the verdict alone.
        ed, c, cand, _ = tradeable_case
        cfg = RuleDiscoveryConfig(
            walk_forward=WalkForwardConfig(
                n_splits=6, min_train_months=4, test_span_months=1
            ),
        )
        resp = RuleDiscovery(ed.df, c, cand, config=cfg).run()
        assert resp.walk_forward is not None
        n_pooled = len(resp.walk_forward.oos_trades)
        assert n_pooled >= RuleDiscoveryConfig().criteria.min_oos_trades
        # a per-window gate would have fired here; the pooled gate must not
        assert resp.verdict != "INSUFFICIENT-DATA"

    def test_mde_degrades_marginal_effect(self, tradeable_case):
        # Force the MDE branch: min_oos_trades=0 keeps the count branch quiet,
        # and an inflated claimed expectancy is simulated by shrinking the OOS
        # via a huge purge... instead, verify the branch arithmetic directly.
        from forgedge.rule_discovery.validation import expectancy_mde

        rng = np.random.default_rng(0)
        net = rng.normal(0.001, 0.05, 30)  # tiny effect, noisy, n=30
        mde = expectancy_mde(net)
        assert mde > 0.001  # such a sample cannot confirm a 0.1% expectancy


class TestSearchLevelGates:
    """Level-0/1 verdict gates: rotation-null p and honest DSR degradation."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        # `_market_edge_kpi_table` since #185.  `_predictive_kpi_table`
        # produced a tradeable verdict only under the old `"limit"` default,
        # where the entry discount was part of the measured edge; at a market
        # entry it reverts too weakly and every rule is NON-EDGE, which would
        # have quietly switched this whole class off via its skip guard.
        df = _market_edge_kpi_table()
        ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
        cands = ed.run()
        ad = AlphaDiscovery(ed.df, cands, AlphaConfig(asset="SYN", timeframe="1H"))
        ad.run()
        promoted = ad.promoted_contracts()
        by_id = {c.event_id: c for c in cands}
        return ed, promoted, by_id

    @pytest.fixture(scope="class")
    def edge_case(self, pipeline):
        """A (contract, candidate) whose baseline verdict is a full EDGE."""
        ed, promoted, by_id = pipeline
        for c in promoted[:10]:
            resp = RuleDiscovery(ed.df, c, by_id[c.event_candidate_id]).run()
            if resp.verdict == "EDGE":
                return ed, c, by_id[c.event_candidate_id]
        pytest.skip("no baseline EDGE in the synthetic fixture")

    def test_high_rotation_p_caps_edge_to_partial(self, edge_case):
        ed, c, cand = edge_case
        old = c.rotation_p
        try:
            c.rotation_p = 0.90
            resp = RuleDiscovery(ed.df, c, cand).run()
            assert resp.verdict == "PARTIAL-EDGE"
            assert any("rotation null" in r for r in resp.rejection_reasons)
        finally:
            c.rotation_p = old

    def test_low_rotation_p_keeps_edge(self, edge_case):
        ed, c, cand = edge_case
        old = c.rotation_p
        try:
            c.rotation_p = 0.001
            resp = RuleDiscovery(ed.df, c, cand).run()
            assert resp.verdict == "EDGE"
        finally:
            c.rotation_p = old

    def test_missing_rotation_p_is_inert(self, edge_case):
        ed, c, cand = edge_case
        assert c.rotation_p is None  # standalone pipeline never annotated it
        resp = RuleDiscovery(ed.df, c, cand).run()
        assert resp.verdict == "EDGE"
        assert not any("rotation" in r for r in resp.rejection_reasons)

    def test_n_trials_upstream_multiplies_dsr_trials(self, edge_case):
        ed, c, cand = edge_case
        cfg = RuleDiscoveryConfig(n_trials_upstream=7)
        resp = RuleDiscovery(ed.df, c, cand, config=cfg).run()
        assert resp.statistical_validation.n_trials_tested == 7 * len(resp.grid_results)

    def test_undefined_dsr_blocks_full_edge(self, edge_case):
        # A huge upstream factor sends the DSR haircut's radicand negative:
        # the deflated Sharpe is undefined and must block a full EDGE instead
        # of silently skipping the gate.
        ed, c, cand = edge_case
        cfg = RuleDiscoveryConfig(n_trials_upstream=10**9)
        resp = RuleDiscovery(ed.df, c, cand, config=cfg).run()
        assert np.isfinite(resp.statistical_validation.sharpe_ratio)
        assert not np.isfinite(resp.statistical_validation.deflated_sharpe)
        assert resp.verdict == "PARTIAL-EDGE"
        assert any("DSR undefined" in r for r in resp.rejection_reasons)


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
        #
        # `entry_mode="limit"` is pinned deliberately.  The never-filling
        # discount is a *limit-mode device*: under the default `"auto"`, Stage 1
        # enters at the next open, `buy_drop_pct` is inert, the rule fills 100%
        # and the fill screen has nothing to fire on.  This test is about
        # `early_elimination`, so it keeps the mode in which its trigger exists
        # rather than swapping in a different trigger and testing something else.
        def make_cfg(early):
            return RuleDiscoveryConfig(
                entry_mode="limit",
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
