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
  position closes at ``sell_price`` on the first bar whose
  ``target_hit_col >= sell_price``, otherwise at the horizon bar's ``close``.

The take-profit detection column is configurable (``BacktestParams.target_hit_col``):
``"close"`` (default) reproduces the certified reference ``backtest_module`` — a
bar must *close* at or above the target — and ``"high"`` uses the optimistic
intrabar touch of the Rule Discovery spec.  This engine reproduces the reference
``run_backtest`` summary (total_signals, total_trades, win_rate, profit_factor,
expectancy, total_net_gain, best/worst trade, target_hit_rate) bit-for-bit with
``target_hit_col="close"``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..episodes import ConcurrencyStats, concurrency, episode_ids
from ..unset import coalesce
from .models import BacktestParams, BacktestSummary, ScoringParams

# Sigmoid steepness on the trade count (hardcoded, see backtest_scoring.md).
_K_TRADES = 0.15

#: Gap bridged when grouping active bars into episodes for the trade ledger.
#: Matches ``GateParams.episode_gap``'s default so M1's episode count and M3's
#: describe the same thing — a one-bar interruption is not a new event.
_EPISODE_GAP = 1


# ---------------------------------------------------------------------------
# Datetime fast paths
# ---------------------------------------------------------------------------
# ``pd.to_datetime`` on an *already* datetime64 column is not a no-op: pandas
# runs ``should_cache``, which iterates the whole array to decide whether to
# build a lookup cache.  Inside the grid / walk-forward screening the timestamp
# column is converted hundreds of times over the full candle table, so this scan
# dominated the profile (~48% of total runtime).  These helpers skip the scan
# whenever the data is already datetime64 and only fall back to ``pd.to_datetime``
# for genuinely unparsed input.

def _as_datetime64(values) -> np.ndarray:
    """Datetime64 numpy array, skipping ``pd.to_datetime`` when already parsed."""
    arr = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    if not np.issubdtype(arr.dtype, np.datetime64):
        arr = pd.to_datetime(values).to_numpy()
    return arr


def _months_m(values) -> pd.PeriodIndex:
    """Month ``PeriodIndex`` from datetime-like values without the cache scan."""
    return pd.DatetimeIndex(_as_datetime64(values)).to_period("M")


# ---------------------------------------------------------------------------
# Prepared candles (extract per-bar arrays once)
# ---------------------------------------------------------------------------
# ``run_backtest`` is invoked ~210 times per ``RuleDiscovery.run`` over the same
# candle table.  Parsing the timestamps and materialising every price column
# (``to_numpy``) is fixed work that does not depend on the operational
# parameters, yet it was repeated on every call.  ``_PreparedCandles`` extracts
# those arrays once; the grid and walk-forward screening build it a single time
# and thread it through.  Columns whose name varies with the parameters
# (``target_col`` / ``target_hit_col`` / ``buy_price_anchor``) are resolved
# lazily and memoised — across a whole run these collapse to ``close``/``high``/
# ``low``.

class _PreparedCandles:
    """Immutable per-bar numpy arrays extracted once from a candle table."""

    __slots__ = ("n", "dt", "low", "high", "open", "close", "signal", "_frame", "_cache")

    def __init__(self, candle: pd.DataFrame, signal_col: str, timestamp_col: str):
        # A caller may forward an unresolved config field (see forgedge.unset):
        # UNSET means "you decide", and "open_dt" is this module's decision.
        timestamp_col = coalesce(timestamp_col, default="open_dt")
        if timestamp_col not in candle.columns:
            raise KeyError(f"timestamp column {timestamp_col!r} not found on candle table")
        self.n = len(candle)
        self.dt = _as_datetime64(candle[timestamp_col])
        self.low = candle["low"].to_numpy(dtype=float)
        self.high = candle["high"].to_numpy(dtype=float)
        self.open = candle["open"].to_numpy(dtype=float)
        self.close = candle["close"].to_numpy(dtype=float)
        self.signal = candle[signal_col].fillna(0).to_numpy()
        self._frame = candle
        self._cache = {
            "low": self.low, "high": self.high,
            "open": self.open, "close": self.close,
        }

    def has_column(self, name: str) -> bool:
        return name in self._cache or name in self._frame.columns

    def column(self, name: str) -> np.ndarray:
        """Return a numeric column by name, materialising and memoising on first use."""
        arr = self._cache.get(name)
        if arr is None:
            if name not in self._frame.columns:
                raise KeyError(f"column {name!r} not found on candle table")
            arr = self._frame[name].to_numpy(dtype=float)
            self._cache[name] = arr
        return arr


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
    _prepared: Optional["_PreparedCandles"] = None,
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
    # A caller may forward unresolved config fields (see forgedge.unset); this
    # is the one place they are turned into values, so no read site below has
    # to know the sentinel exists.
    params = params.resolved()
    if params.buy_type not in ("limit", "market"):
        raise ValueError(f"buy_type must be 'limit' or 'market', got {params.buy_type!r}")
    if params.direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {params.direction!r}")

    # Resolved here for the same reason `params` is: the caller may hold
    # unresolved fields, and this is the one place they become values.
    scoring = (scoring or ScoringParams()).resolved()
    is_short = params.direction == "short"

    prep = _prepared if _prepared is not None else _PreparedCandles(
        candle, signal_col, timestamp_col
    )
    n = prep.n
    dt = prep.dt
    low = prep.low
    high = prep.high
    open_ = prep.open
    close = prep.close
    target_arr = prep.column(params.target_col)
    hit_arr = prep.column(params.target_hit_col)
    signal = prep.signal

    if params.buy_type == "limit":
        if not prep.has_column(params.buy_price_anchor):
            raise KeyError(
                f"buy_price_anchor {params.buy_price_anchor!r} not found on candle table"
            )
        anchor = prep.column(params.buy_price_anchor)
    else:
        anchor = close

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

    # ── entry (buy/short) price ──────────────────────────────────────────
    # Long: limit below the anchor (discount); short: above it (premium).
    offset = (1.0 + params.buy_drop_pct) if is_short else (1.0 - params.buy_drop_pct)
    if params.buy_type == "limit":
        buy_price = anchor[signal_rn] * offset
    else:  # market: fill at the next bar's open
        buy_price = open_[np.minimum(signal_rn + 1, n - 1)]

    # ── fill scan ────────────────────────────────────────────────────────
    # Long limit fills when low ≤ entry; short limit fills when high ≥ entry.
    fill_probe = high if is_short else low
    fill_rn = _scan_fill(signal_rn, buy_price, fill_probe, n, params, is_short=is_short)

    valid_fill = fill_rn >= 0
    target_rn = np.where(valid_fill, fill_rn + params.target_h, -1)
    valid = valid_fill & (target_rn < n)

    # ── exit scan ────────────────────────────────────────────────────────
    # Take-profit below the entry for a short, above for a long.
    sell_price = buy_price * ((1.0 - params.sell_pct) if is_short else (1.0 + params.sell_pct))
    exit_price, exit_rn, target_hit = _scan_exit(
        fill_rn, target_rn, sell_price, hit_arr, target_arr, valid, is_short, params
    )

    # ── per-trade results (numpy) ────────────────────────────────────────
    fee_rt = params.fee * 2.0
    bp = buy_price[valid]
    ep = exit_price[valid]
    # Long gains when price rises, short when it falls.
    net = ((bp - ep) if is_short else (ep - bp)) / bp - fee_rt
    fill_dt = dt[fill_rn[valid]]
    hit = target_hit[valid]

    # The summary is computed straight from the numpy arrays; the per-trade
    # DataFrame — needed only when ``return_trades`` (the winning config's
    # ledger, the walk-forward OOS trades) — is materialised on demand.  This
    # keeps the ~200 grid/walk-forward screening calls free of DataFrame
    # construction.
    # ── overlap (issue #168) ─────────────────────────────────────────────
    # A trade opens on every active bar, with no flat-state check, so a
    # persistent event produces a stack of positions on the same price path.
    # Measured here, on the ledger the parameters actually produce, rather than
    # inferred from M1's discovery-window episode counts: the grid can pick a
    # different target_h than M1 ever saw, so those numbers would answer a
    # different question.
    open_rn, close_rn = fill_rn[valid], exit_rn[valid]
    conc = concurrency(open_rn, close_rn, n_bars=n)
    ep_ids = episode_ids(active, _EPISODE_GAP)
    trade_episodes = ep_ids[signal_rn[valid]]
    n_episodes = int(np.unique(trade_episodes).size) if trade_episodes.size else 0

    summary = _summarise_arrays(
        net, fill_dt, hit, total_signals, months, scoring,
        n_episodes=n_episodes, concurrency_stats=conc,
    )
    if not return_trades:
        return summary

    trades = pd.DataFrame(
        {
            "signal_rn": signal_rn[valid],
            "fill_rn": open_rn,
            "target_rn": target_rn[valid],
            "exit_rn": close_rn,
            "episode_id": trade_episodes,
            "fill_dt": fill_dt,
            "exit_dt": dt[exit_rn[valid]],
            "direction": params.direction,
            "buy_price": bp,
            "sell_price": sell_price[valid],
            "exit_price": ep,
            "net_pct_gain": net,
            "target_hit": hit,
        }
    )
    return summary, trades


# ---------------------------------------------------------------------------
# Fill / exit scans
# ---------------------------------------------------------------------------

def _scan_fill(
    signal_rn: np.ndarray,
    buy_price: np.ndarray,
    probe: np.ndarray,
    n: int,
    params: BacktestParams,
    is_short: bool = False,
) -> np.ndarray:
    """Return the fill row index per signal (``-1`` when never filled).

    ``probe`` is ``low`` for a long (fills when ``low ≤ entry``) and ``high``
    for a short (fills when ``high ≥ entry``).
    """
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
        window = probe[lo : hi + 1]
        mask = (window >= bp) if is_short else (window <= bp)
        if mask.any():
            fill_rn[i] = lo + int(np.argmax(mask))
    return fill_rn


def _scan_exit(
    fill_rn: np.ndarray,
    target_rn: np.ndarray,
    sell_price: np.ndarray,
    hit_arr: np.ndarray,
    target_arr: np.ndarray,
    valid: np.ndarray,
    is_short: bool,
    params: BacktestParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve exit price / bar / hit flag for every valid trade.

    The take-profit is detected on ``hit_arr`` (``params.target_hit_col``).  A
    long hits when the price rises to ``sell_price`` (``hit_arr ≥ sell_price``);
    a short hits when it falls to it (``hit_arr ≤ sell_price``).
    """
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
        window = hit_arr[lo:hi]
        mask = (window <= sp) if is_short else (window >= sp)
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
    """Aggregate a per-trade DataFrame into a :class:`BacktestSummary`.

    Thin adapter over :func:`_summarise_arrays`; kept for callers that already
    hold a materialised trades frame (e.g. the walk-forward OOS concatenation).

    The overlap fields are recomputed from the frame's own bar indices when it
    carries them.  For the walk-forward that is the right thing rather than an
    approximation: the concatenated OOS ledger is the record a user would have
    traded across all test windows, so its concurrency is the concurrency they
    would have had to fund.
    """
    # The second way into `_summarise_arrays`: the walk-forward hands its
    # concatenated ledger straight here, never passing through `run_backtest`,
    # so the sentinel has to be spent on this path too.
    scoring = scoring.resolved()
    has_geometry = {"fill_rn", "exit_rn", "episode_id"} <= set(trades.columns)
    conc = (
        concurrency(trades["fill_rn"].to_numpy(), trades["exit_rn"].to_numpy())
        if has_geometry and len(trades)
        else None
    )
    n_episodes = (
        int(trades["episode_id"].nunique()) if has_geometry and len(trades) else 0
    )
    return _summarise_arrays(
        trades["net_pct_gain"].to_numpy(),
        trades["fill_dt"].to_numpy(),
        trades["target_hit"].to_numpy(),
        total_signals,
        months,
        scoring,
        n_episodes=n_episodes,
        concurrency_stats=conc,
    )


def _summarise_arrays(
    net: np.ndarray,
    fill_dt: np.ndarray,
    target_hit: np.ndarray,
    total_signals: int,
    months: pd.PeriodIndex,
    scoring: ScoringParams,
    *,
    n_episodes: int = 0,
    concurrency_stats: Optional[ConcurrencyStats] = None,
) -> BacktestSummary:
    """Aggregate per-trade numpy arrays into a :class:`BacktestSummary`.

    ``n_episodes`` and ``concurrency_stats`` are keyword-only and optional: the
    overlap measurements need the bar-index geometry, which the caller has and
    this function does not.  Omitting them leaves the overlap fields at their
    defaults rather than inventing values.
    """
    n_months = max(len(months), 1)
    n = int(net.size)

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
    fill_months = _months_m(fill_dt)
    monthly_counts = fill_months.value_counts().reindex(months, fill_value=0)
    active_months = int((monthly_counts > 0).sum())
    zero_months = n_months - active_months
    mu = float(monthly_counts.mean())
    sigma = float(monthly_counts.std(ddof=0))

    # ── composite scores (backtest_scoring.md) ───────────────────────────
    min_trades_dyn = max(scoring.pf_min_trades, n_months * scoring.pf_min_tpm)
    sig_trades = 1.0 / (1.0 + math.exp(-_K_TRADES * (n - min_trades_dyn)))
    pf_score = round(max(profit_factor * sig_trades, 0.0), 6)

    # ── consistency term: excess dispersion only (F3, issue #178) ────────
    #
    # `c_norm` used to be level-dependent:
    #
    #     f_r = min(target / mu, 1);  C = (mu / (sigma + 1)) * f_r
    #     c_norm = C / target
    #
    # which above the target froze the numerator while `sigma` kept growing.
    # For a pure Poisson process (`sigma = sqrt(mu)`, index of dispersion 1,
    # perfect regularity) that scored 0.366 at mu=3, 0.240 at mu=10 and 0.154
    # at mu=30 — the *same* process, penalised 2.4x for trading more often.
    # It was measuring frequency dressed up as irregularity, and on 15m data,
    # where the preset's gate demands 76.8 trades/month, the objective it feeds
    # preferred the cells that traded *less*: gate and objective pointing in
    # opposite directions.
    #
    # The index of dispersion is scale-free — 1 for Poisson at any rate — so
    # only variance *in excess* of what the rate necessarily produces is
    # penalised.  This is the correction M1's ConsistencyGate already applies
    # via its Poisson chi-squared floor; the same mistake lived one module
    # downstream.
    #
    # `c_norm` now measures one thing.  "Enough trades" is `criteria.min_tpm`
    # and `_dynamic_min_trades`, which already do it and are also in
    # `_passes` — so the guard against "three fantastic trades" is not lost.
    if n_months > 0 and mu > 0:
        dispersion = (sigma * sigma) / mu
        c_norm = min(1.0, 1.0 / max(dispersion, 1.0))
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
        target_hit_rate_pct=round(float(target_hit.mean()) * 100, 2) if n else 0.0,
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
        n_episodes=int(n_episodes),
        mean_concurrent_positions=(
            round(concurrency_stats.mean, 4)
            if concurrency_stats is not None and math.isfinite(concurrency_stats.mean)
            else float("nan")
        ),
        max_concurrent_positions=(
            concurrency_stats.peak if concurrency_stats is not None else 0
        ),
    )


def _month_index(ts_from, ts_to, dt: np.ndarray) -> pd.PeriodIndex:
    """Months spanned by the entry window (defaults to the candle span).

    The entry window is ``[from, to)``: when ``ts_to`` is given, the last
    countable month is the one containing ``ts_to - 1ns`` — so a month-aligned
    exclusive bound (e.g. ``"2025-04-01"``) does not add the month it names,
    while a mid-month bound still counts its (partial) final month.  When
    ``ts_to`` is ``None`` the window closes at the last candle, whose month is
    a real, countable month.  Dropping that final month (the previous
    behaviour) silently excluded its trades from every monthly statistic
    (``tpm_mu``, ``zero_months``, ``c_norm``) on whole-table backtests.
    """
    start = (
        pd.Timestamp(ts_from) if ts_from is not None else pd.Timestamp(dt.min())
    ).to_period("M")
    if ts_to is not None:
        end = (pd.Timestamp(ts_to) - pd.Timedelta(1, "ns")).to_period("M")
    else:
        end = pd.Timestamp(dt.max()).to_period("M")
    n = max((end.year - start.year) * 12 + (end.month - start.month) + 1, 1)
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
            "signal_rn", "fill_rn", "target_rn", "exit_rn", "episode_id",
            "fill_dt", "exit_dt", "direction", "buy_price", "sell_price",
            "exit_price", "net_pct_gain", "target_hit",
        ]
    )


def optimistic_hit_col(direction: str) -> str:
    """Resolve the optimistic intrabar take-profit column for a direction.

    A long take-profit fills when ``high`` reaches it; a short when ``low`` does.
    The conservative convention is always ``close`` for both.
    """
    return "low" if direction == "short" else "high"
