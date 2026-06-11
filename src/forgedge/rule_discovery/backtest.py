"""Rule-based backtest engine for Rule Discovery (pure pandas / numpy).

This is a dependency-light re-implementation of the limit-order backtest used to
stress-test a rule's operability.  It keeps the same semantics as the reference
``backtest_module`` (signal → limit fill → discrete take-profit / horizon exit →
round-trip fee) and the composite scoring metrics of ``backtest_scoring.md``
(``pf_score_tpm``, ``exp_score_tpm`` …), but drops the DuckDB / SciPy
dependencies so it lives inside the ``numpy``/``pandas``-only FORGE runtime.

The entry signal is **not** parsed from a string here: Rule Discovery
reconstructs the Event Candidate's boolean activation series with the exact
parameters stored on the candidate (Event Discovery's own replay path) and
injects it as a column.  ``run_backtest`` only reads that column — so the
activations are bit-for-bit identical to Event/Alpha Discovery.

Fill / exit mechanics (Rule Discovery spec, Section 2.1)
-------------------------------------------------------
* bar ``t`` — signal active → ``buy_price = anchor_t * (1 - buy_drop_pct)``
  (``limit``) or next open (``market``);
* bars ``t+1 .. t+buy_delay_bar`` — fill window: filled on the first bar whose
  ``low <= buy_price`` (``limit``); ``market`` always fills at ``t+1`` open;
* bars ``fill+1 .. fill+target_h`` — exit window: with ``early_stopping`` the
  position closes at ``sell_price`` on the first bar whose ``high >= sell_price``
  (a realistic limit-sell fill), otherwise at the horizon bar's ``close``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .models import BacktestParams, BacktestSummary, ScoringParams

# Sigmoid steepness on the trade count (hardcoded, see backtest_scoring.md).
_K_TRADES = 0.15


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    candle: pd.DataFrame,
    signal_col: str,
    params: BacktestParams,
    timerange_from: Optional[str] = None,
    timerange_to: Optional[str] = None,
    scoring: Optional[ScoringParams] = None,
    timestamp_col: str = "open_dt",
    return_trades: bool = False,
):
    """Backtest a reconstructed entry signal over a (sub)range of candles.

    Parameters
    ----------
    candle : pd.DataFrame
        Candle table with at least ``open``/``high``/``low``/``close``, the
        timestamp column and the boolean ``signal_col``.  Must be chronologically
        sorted.  The full table is always passed so the fill / exit windows can
        reach past ``timerange_to``; the date filter only restricts which signals
        are *opened*.
    signal_col : str
        Name of the 0/1 entry-signal column (reconstructed by Rule Discovery).
    params : BacktestParams
        Order mechanics.
    timerange_from, timerange_to : str, optional
        ``[from, to)`` window (inclusive / exclusive) on the entry bar.  ``None``
        uses the whole table.
    scoring : ScoringParams, optional
        Composite-scoring knobs.  Defaults to :class:`ScoringParams`.
    timestamp_col : str
        Datetime column name.
    return_trades : bool
        When ``True`` also return the per-trade DataFrame.

    Returns
    -------
    BacktestSummary
        Aggregated metrics (and the per-trade frame when ``return_trades``).
    """
    if params.buy_type not in ("limit", "market"):
        raise ValueError(f"buy_type must be 'limit' or 'market', got {params.buy_type!r}")

    scoring = scoring or ScoringParams()

    c = candle
    n = len(c)
    if timestamp_col not in c.columns:
        raise KeyError(f"timestamp column {timestamp_col!r} not found on candle table")

    dt = pd.to_datetime(c[timestamp_col]).to_numpy()
    low = c["low"].to_numpy(dtype=float)
    high = c["high"].to_numpy(dtype=float)
    open_ = c["open"].to_numpy(dtype=float)
    close = c["close"].to_numpy(dtype=float)
    target_arr = c[params.target_col].to_numpy(dtype=float)
    signal = c[signal_col].fillna(0).to_numpy()

    if params.buy_type == "limit" and params.buy_price_anchor not in c.columns:
        raise KeyError(
            f"buy_price_anchor {params.buy_price_anchor!r} not found on candle table"
        )
    anchor = (
        c[params.buy_price_anchor].to_numpy(dtype=float)
        if params.buy_type == "limit"
        else close
    )

    # ── entry bars (HIT) restricted to the entry window ──────────────────
    ts_from = pd.Timestamp(timerange_from) if timerange_from else None
    ts_to = pd.Timestamp(timerange_to) if timerange_to else None

    active = signal.astype(bool)
    # An entry "opens" on the *next* bar (we act on the bar after the signal).
    entry_rn = np.where(active)[0]
    entry_rn = entry_rn[entry_rn + 1 < n]  # need a next bar to act on
    open_rn = entry_rn + 1
    open_dt = dt[open_rn]

    in_window = np.ones(len(open_rn), dtype=bool)
    if ts_from is not None:
        in_window &= open_dt >= np.datetime64(ts_from)
    if ts_to is not None:
        in_window &= open_dt < np.datetime64(ts_to)

    signal_rn = entry_rn[in_window]
    total_signals = int(signal_rn.size)

    months = _month_index(ts_from, ts_to, dt)
    n_months = max(len(months), 1)

    if total_signals == 0:
        empty = _empty_summary(n_months)
        return (empty, _empty_trades()) if return_trades else empty

    # ── buy price ────────────────────────────────────────────────────────
    if params.buy_type == "limit":
        buy_price = anchor[signal_rn] * (1.0 - params.buy_drop_pct)
    else:  # market: fill at the next bar's open
        buy_price = open_[np.minimum(signal_rn + 1, n - 1)]

    # ── fill scan ────────────────────────────────────────────────────────
    fill_rn = _scan_fill(signal_rn, buy_price, low, n, params)

    valid_fill = fill_rn >= 0
    target_rn = np.where(valid_fill, fill_rn + params.target_h, -1)
    valid = valid_fill & (target_rn < n)

    # ── exit scan ────────────────────────────────────────────────────────
    sell_price = buy_price * (1.0 + params.sell_pct)
    exit_price, exit_rn, target_hit = _scan_exit(
        fill_rn, target_rn, sell_price, high, close, target_arr, valid, params
    )

    # ── per-trade frame ──────────────────────────────────────────────────
    fee_rt = params.fee * 2.0
    bp = buy_price[valid]
    ep = exit_price[valid]
    net = (ep - bp) / bp - fee_rt

    trades = pd.DataFrame(
        {
            "signal_rn": signal_rn[valid],
            "fill_rn": fill_rn[valid],
            "target_rn": target_rn[valid],
            "exit_rn": exit_rn[valid],
            "fill_dt": dt[fill_rn[valid]],
            "exit_dt": dt[exit_rn[valid]],
            "buy_price": bp,
            "sell_price": sell_price[valid],
            "exit_price": ep,
            "net_pct_gain": net,
            "target_hit": target_hit[valid],
        }
    )

    summary = _summarise(trades, total_signals, months, scoring)
    return (summary, trades) if return_trades else summary


# ---------------------------------------------------------------------------
# Fill / exit scans
# ---------------------------------------------------------------------------

def _scan_fill(
    signal_rn: np.ndarray,
    buy_price: np.ndarray,
    low: np.ndarray,
    n: int,
    params: BacktestParams,
) -> np.ndarray:
    """Return the fill row index per signal (``-1`` when never filled)."""
    fill_rn = np.full(signal_rn.size, -1, dtype=np.int64)

    if params.buy_type == "market":
        cand = signal_rn + 1
        ok = cand < n
        fill_rn[ok] = cand[ok]
        return fill_rn

    delay = params.buy_delay_bar
    for i in range(signal_rn.size):
        bp = buy_price[i]
        if not np.isfinite(bp):
            continue
        srn = signal_rn[i]
        lo = srn + 1
        hi = min(srn + delay, n - 1)
        if lo > hi:
            continue
        window = low[lo : hi + 1]
        mask = window <= bp
        if mask.any():
            fill_rn[i] = lo + int(np.argmax(mask))
    return fill_rn


def _scan_exit(
    fill_rn: np.ndarray,
    target_rn: np.ndarray,
    sell_price: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    target_arr: np.ndarray,
    valid: np.ndarray,
    params: BacktestParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve exit price / bar / hit flag for every valid trade."""
    size = fill_rn.size
    exit_price = np.full(size, np.nan)
    exit_rn = np.full(size, -1, dtype=np.int64)
    target_hit = np.zeros(size, dtype=bool)

    idx = np.where(valid)[0]
    # Default: close-at-horizon exit.
    exit_price[idx] = target_arr[target_rn[idx]]
    exit_rn[idx] = target_rn[idx]

    if not params.early_stopping:
        return exit_price, exit_rn, target_hit

    for i in idx:
        frn = int(fill_rn[i])
        trn = int(target_rn[i])
        sp = sell_price[i]
        lo = frn + 1
        hi = trn + 1  # exclusive
        if lo >= hi:
            continue
        window = high[lo:hi]
        mask = window >= sp
        if mask.any():
            hit_rn = lo + int(np.argmax(mask))
            exit_price[i] = sp
            exit_rn[i] = hit_rn
            target_hit[i] = True
    return exit_price, exit_rn, target_hit


# ---------------------------------------------------------------------------
# Summary & scoring
# ---------------------------------------------------------------------------

def _summarise(
    trades: pd.DataFrame,
    total_signals: int,
    months: pd.PeriodIndex,
    scoring: ScoringParams,
) -> BacktestSummary:
    """Aggregate per-trade results into a :class:`BacktestSummary`."""
    n_months = max(len(months), 1)
    n = len(trades)
    net = trades["net_pct_gain"].to_numpy()

    wins = net[net > 0]
    losses = net[net < 0]
    n_win = int(wins.size)
    n_loss = int(losses.size)

    pos = float(wins.sum())
    neg = float(-losses.sum())
    if neg == 0:
        profit_factor = 9999.0 if pos > 0 else 0.0
    else:
        profit_factor = pos / neg

    win_rate = n_win / n if n else float("nan")
    expectancy = float(net.mean()) if n else float("nan")
    std = float(net.std(ddof=1)) if n > 1 else 0.0

    # ── monthly distribution (on the fill bar) ───────────────────────────
    fill_months = pd.to_datetime(trades["fill_dt"]).dt.to_period("M")
    monthly_counts = fill_months.value_counts().reindex(months, fill_value=0)
    active_months = int((monthly_counts > 0).sum())
    zero_months = n_months - active_months
    mu = float(monthly_counts.mean())
    sigma = float(monthly_counts.std(ddof=0))

    # ── composite scores (backtest_scoring.md) ───────────────────────────
    min_trades_dyn = max(scoring.pf_min_trades, n_months * scoring.pf_min_tpm)
    sig_trades = 1.0 / (1.0 + math.exp(-_K_TRADES * (n - min_trades_dyn)))
    pf_score = round(max(profit_factor * sig_trades, 0.0), 6)

    if n_months > 0 and scoring.pf_tpm_target > 0 and mu > 0:
        f_r = min(scoring.pf_tpm_target / mu, 1.0)
        C = (mu / (sigma + 1.0)) * f_r
        c_norm = max(0.0, min(1.0, C / scoring.pf_tpm_target))
    else:
        c_norm = 0.0

    pf_score_tpm = round(max(profit_factor * c_norm, 0.0), 6)
    exp = expectancy if (n and math.isfinite(expectancy)) else 0.0
    exp_score_tpm = round(max(exp * c_norm, 0.0), 6)

    # Sharpe per trade: floor the denominator at the round-trip cost proxy so a
    # near-zero std (overfit, near-identical trades) cannot inflate it.
    sharpe_raw = exp / max(std, 1e-9)

    return BacktestSummary(
        total_signals=total_signals,
        total_trades=n,
        fill_rate=round(n / total_signals, 6) if total_signals else float("nan"),
        win_rate_pct=round(win_rate, 6) if math.isfinite(win_rate) else float("nan"),
        winning_trades=n_win,
        losing_trades=n_loss,
        total_net_gain=round(float(net.sum()), 6) if n else 0.0,
        expectancy=round(expectancy, 6) if math.isfinite(expectancy) else float("nan"),
        std_net_gain=round(std, 6),
        profit_factor=round(profit_factor, 4),
        best_trade=round(float(net.max()), 6) if n else float("nan"),
        worst_trade=round(float(net.min()), 6) if n else float("nan"),
        target_hit_rate_pct=round(float(trades["target_hit"].mean()) * 100, 2) if n else 0.0,
        n_months=n_months,
        active_months=active_months,
        zero_months=zero_months,
        tpm_mu=round(mu, 4),
        tpm_sigma=round(sigma, 4),
        c_norm=round(c_norm, 6),
        pf_score=pf_score,
        pf_score_tpm=pf_score_tpm,
        exp_score_tpm=exp_score_tpm,
        sharpe_raw=round(sharpe_raw, 6),
    )


def _month_index(ts_from, ts_to, dt: np.ndarray) -> pd.PeriodIndex:
    """Months spanned by the entry window (defaults to the candle span)."""
    if ts_from is None:
        ts_from = pd.Timestamp(dt.min())
    if ts_to is None:
        ts_to = pd.Timestamp(dt.max())
    start = pd.Timestamp(ts_from).to_period("M")
    end = pd.Timestamp(ts_to).to_period("M")
    n = max((end.year - start.year) * 12 + (end.month - start.month), 1)
    return pd.period_range(start, periods=n, freq="M")


def _empty_summary(n_months: int) -> BacktestSummary:
    nan = float("nan")
    return BacktestSummary(
        total_signals=0, total_trades=0, fill_rate=nan, win_rate_pct=nan,
        winning_trades=0, losing_trades=0, total_net_gain=0.0, expectancy=nan,
        std_net_gain=0.0, profit_factor=0.0, best_trade=nan, worst_trade=nan,
        target_hit_rate_pct=0.0, n_months=n_months, active_months=0,
        zero_months=n_months, tpm_mu=0.0, tpm_sigma=0.0, c_norm=0.0,
        pf_score=0.0, pf_score_tpm=0.0, exp_score_tpm=0.0, sharpe_raw=0.0,
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_rn", "fill_rn", "target_rn", "exit_rn", "fill_dt", "exit_dt",
            "buy_price", "sell_price", "exit_price", "net_pct_gain", "target_hit",
        ]
    )
