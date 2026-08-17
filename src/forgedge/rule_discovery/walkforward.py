"""Walk-forward out-of-sample validation for Rule Discovery (Step 4).

The in-sample grid screening always overstates performance — it picks the best
of many configurations on data it has seen.  Walk-forward validation removes
that bias: the timeline is cut into consecutive test windows; for each, the
operational parameters are re-selected on the *preceding* train window and then
evaluated **once** on the untouched test window.  Concatenating the per-window
test trades yields the rule's honest out-of-sample track record.

Two modes (``RuleWalkForwardConfig.train_span_months``):

* **anchored** (``None``) — the train window always starts at the data origin
  and grows; each test window is a fresh, later slice.
* **rolling** (an integer) — the train window is a fixed-length span that slides
  forward with the test window.

``reoptimise=False`` keeps a single fixed parameter set and only replays it
out-of-sample (pure OOS replay, no per-fold optimisation).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..unset import coalesce
from .analysis import excursion_stats
from .backtest import _as_datetime64, _PreparedCandles, optimistic_hit_col, run_backtest
from .grid import build_grid, run_grid, select_best
from .models import (
    BacktestParams,
    ExecutionEnvelope,
    GridSpec,
    ScoringParams,
    SelectionCriteria,
    RuleWalkForwardConfig,
    WalkForwardResult,
    WalkForwardSplit,
)
from .validation import validate


def _month_bounds(candle: pd.DataFrame, timestamp_col: str):
    timestamp_col = coalesce(timestamp_col, default="open_dt")
    dt = pd.DatetimeIndex(_as_datetime64(candle[timestamp_col]))
    start = dt.min().to_period("M").to_timestamp()
    # Exclusive upper bound: first day of the month after the last candle.
    end = (dt.max().to_period("M") + 1).to_timestamp()
    return start, end


def _build_splits(start: pd.Timestamp, end: pd.Timestamp, cfg: RuleWalkForwardConfig):
    """Return a list of ``(train_from, train_to, test_from, test_to)`` timestamps."""
    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    # Session-resolved (#177): a caller who never went through the resolver
    # holds the sentinel, and 6 is this module's documented decision.
    min_train = max(int(coalesce(cfg.min_train_months, default=6)), 1)
    if total_months <= min_train:
        return []

    # Determine test-window length.
    usable = total_months - min_train
    if cfg.test_span_months:
        test_span = cfg.test_span_months
        n_splits = min(cfg.n_splits, max(usable // test_span, 1))
    else:
        n_splits = max(cfg.n_splits, 1)
        test_span = max(usable // n_splits, 1)

    splits = []
    for i in range(n_splits):
        test_start_off = min_train + i * test_span
        test_end_off = test_start_off + test_span
        if i == n_splits - 1:
            test_end_off = total_months  # last window absorbs the remainder
        if test_start_off >= total_months:
            break
        test_end_off = min(test_end_off, total_months)
        if test_end_off <= test_start_off:
            break

        if cfg.train_span_months:  # rolling
            train_start_off = max(0, test_start_off - cfg.train_span_months)
        else:  # anchored
            train_start_off = 0

        splits.append(
            (
                start + pd.DateOffset(months=train_start_off),
                start + pd.DateOffset(months=test_start_off),
                start + pd.DateOffset(months=test_start_off),
                start + pd.DateOffset(months=test_end_off),
            )
        )
    return splits


def walk_forward(
    candle: pd.DataFrame,
    signal_col: str,
    base: BacktestParams,
    spec: GridSpec,
    cfg: RuleWalkForwardConfig,
    scoring: Optional[ScoringParams] = None,
    criteria: Optional[SelectionCriteria] = None,
    timestamp_col: str = "open_dt",
    base_rate: float = 0.0,
) -> Optional[WalkForwardResult]:
    """Run the walk-forward validation and return the aggregated OOS result.

    For every split the parameters are (re-)selected on the train window and the
    test window is then scored once under both take-profit conventions (``close``
    and ``high``).  The concatenated test trades feed the OOS twins of the
    in-sample diagnostics: execution envelope, MAE/MFE, and the win-rate /
    expectancy t-tests.  ``base_rate`` is the contract's target base rate, used
    as the null for the OOS win-rate t-test.

    Returns ``None`` when the data span is too short to form a single
    train/test split.
    """
    scoring = scoring or ScoringParams()
    criteria = criteria or SelectionCriteria()

    windows = selection_windows(candle, spec, base, cfg, timestamp_col)
    if not windows:
        return None

    # Extract the per-bar candle arrays once for the whole walk-forward — every
    # train grid screening and test-window backtest below reuses them.
    prep = _PreparedCandles(candle, signal_col, timestamp_col)

    splits: List[WalkForwardSplit] = []
    close_trades: List[pd.DataFrame] = []
    high_trades: List[pd.DataFrame] = []
    last_train_grid = None

    for idx, (tr_from, tr_to_eff, te_from_eff, te_to) in enumerate(windows):
        tr_from_s, tr_to_s = _fmt(tr_from), _fmt(tr_to_eff)
        te_from_s, te_to_s = _fmt(te_from_eff), _fmt(te_to)

        # ── select parameters on the train window ──
        if cfg.reoptimise:
            grid_res = run_grid(
                candle, signal_col, base, spec, scoring=scoring,
                timerange_from=tr_from_s, timerange_to=tr_to_s,
                timestamp_col=timestamp_col, _prepared=prep,
            )
            last_train_grid = grid_res
            best = select_best(grid_res, criteria)
            params = best.params if best else base
            train_summary = best.summary if best else run_backtest(
                candle, signal_col, base, tr_from_s, tr_to_s, scoring, timestamp_col,
                _prepared=prep,
            )
        else:
            params = base
            train_summary = run_backtest(
                candle, signal_col, params, tr_from_s, tr_to_s, scoring, timestamp_col,
                _prepared=prep,
            )

        # ── evaluate once on the untouched test window (both conventions) ──
        test_summary, test_tr = run_backtest(
            candle, signal_col, params.merged(target_hit_col="close"),
            te_from_s, te_to_s, scoring, timestamp_col, return_trades=True,
            _prepared=prep,
        )
        _, test_tr_high = run_backtest(
            candle, signal_col, params.merged(target_hit_col=optimistic_hit_col(params.direction)),
            te_from_s, te_to_s, scoring, timestamp_col, return_trades=True,
            _prepared=prep,
        )
        close_trades.append(test_tr)
        high_trades.append(test_tr_high)

        splits.append(
            WalkForwardSplit(
                split_idx=idx,
                train_from=tr_from_s, train_to=tr_to_s,
                test_from=te_from_s, test_to=te_to_s,
                params=params,
                train_summary=train_summary,
                test_summary=test_summary,
            )
        )

    # ── aggregate OOS metrics over the concatenated test trades ──
    total_signals = sum(s.test_summary.total_signals for s in splits)
    months = _oos_months(splits)

    oos_summary = _summarise_concat(close_trades, total_signals, months, scoring)
    oos_high_summary = _summarise_concat(high_trades, total_signals, months, scoring)
    oos_envelope = ExecutionEnvelope(conservative=oos_summary, optimistic=oos_high_summary)

    oos_concat = _concat(close_trades)
    oos_excursion = excursion_stats(candle, oos_concat) if oos_concat is not None else None
    oos_validation = (
        validate(
            oos_concat, base_rate=base_rate, n_trials=1,
            bars_per_year=_bars_per_year(candle, timestamp_col),
            avg_holding_bars=_avg_holding(oos_concat),
        )
        if oos_concat is not None and len(oos_concat) >= 2
        else None
    )

    n_profitable = sum(1 for s in splits if s.test_summary.total_net_gain > 0)
    consistency = n_profitable / len(splits) if splits else 0.0

    return WalkForwardResult(
        splits=splits,
        oos_summary=oos_summary,
        n_profitable_splits=n_profitable,
        consistency=round(consistency, 4),
        oos_envelope=oos_envelope,
        oos_excursion=oos_excursion,
        oos_validation=oos_validation,
        oos_trades=oos_concat,
        last_train_grid=last_train_grid,
    )


def selection_windows(
    candle: pd.DataFrame,
    spec: GridSpec,
    base: BacktestParams,
    cfg: RuleWalkForwardConfig,
    timestamp_col: str = "open_dt",
) -> List[tuple]:
    """Purged/embargoed walk-forward windows as Timestamp tuples.

    Returns one ``(train_from, train_to, test_from, test_to)`` tuple per
    split — ``train_to`` already pulled back by the purge and ``test_from``
    pushed forward by the embargo — or ``[]`` when the data span is too short
    for a single split.  This is the single source of the walk-forward's
    temporal geometry, shared by :func:`walk_forward` and by Rule Discovery's
    walk-forward selection mode (§3.4), so the two can never disagree on
    where selection is allowed to look.
    """
    start, end = _month_bounds(candle, timestamp_col)
    bounds = _build_splits(start, end, cfg)
    if not bounds:
        return []

    # Purge / embargo widths (in bars → wall-clock via the bar duration).
    # A trade entered in the last bars of a train window fills and exits
    # *inside* the adjacent test window (run_backtest deliberately lets the
    # fill/exit scans reach past timerange_to), so the parameter selection
    # would be scored on test prices.  Purge the train tail by the worst-case
    # trade span of the resolved grid; optionally embargo the test head.
    resolved = build_grid(spec, base)
    if cfg.purge_bars is not None:
        purge_bars = max(int(cfg.purge_bars), 0)
    else:
        purge_bars = (
            max(resolved.target_h) + max(resolved.buy_delay_bar) + 1
        )  # +1: the entry acts on the bar after the signal
    embargo_bars = max(int(getattr(cfg, "embargo_bars", 0)), 0)
    delta = _bar_delta(candle, timestamp_col)

    windows = []
    for tr_from, tr_to, te_from, te_to in bounds:
        tr_to_eff = max(pd.Timestamp(tr_from), pd.Timestamp(tr_to) - purge_bars * delta)
        te_from_eff = min(pd.Timestamp(te_to), pd.Timestamp(te_from) + embargo_bars * delta)
        windows.append((pd.Timestamp(tr_from), tr_to_eff, te_from_eff, pd.Timestamp(te_to)))
    return windows


def _oos_months(splits) -> pd.PeriodIndex:
    """Months covered by the union of the test windows (non-overlapping)."""
    return pd.period_range(
        pd.Timestamp(splits[0].test_from).to_period("M"),
        pd.Timestamp(splits[-1].test_to).to_period("M") - 1,
        freq="M",
    )


def _concat(frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    nonempty = [t for t in frames if t is not None and not t.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else None


def _summarise_concat(frames, total_signals, months, scoring):
    """Summarise concatenated test trades, reusing the backtest summariser."""
    from .backtest import _empty_summary, _summarise

    concat = _concat(frames)
    if concat is None:
        return _empty_summary(max(len(months), 1))
    return _summarise(concat, total_signals, months, scoring)


def _bars_per_year(candle: pd.DataFrame, timestamp_col: str) -> float:
    dt = pd.Series(_as_datetime64(candle[timestamp_col]))
    if len(dt) > 1:
        delta = dt.diff().dt.total_seconds().median()
        if delta and delta > 0:
            return 365.25 * 86400.0 / float(delta)
    return 24 * 365.25


def _avg_holding(trades: Optional[pd.DataFrame]) -> Optional[float]:
    if trades is None or trades.empty or "exit_rn" not in trades.columns:
        return None
    hold = (trades["exit_rn"] - trades["fill_rn"]).to_numpy(dtype=float)
    hold = hold[(hold > 0)]
    return float(hold.mean()) if hold.size else None


def _bar_delta(candle: pd.DataFrame, timestamp_col: str) -> pd.Timedelta:
    """Median bar duration, for converting bar counts to wall-clock spans."""
    dt = pd.Series(_as_datetime64(candle[timestamp_col]))
    if len(dt) > 1:
        med = dt.diff().median()
        if pd.notna(med) and med > pd.Timedelta(0):
            return med
    return pd.Timedelta(hours=1)


def _fmt(ts: pd.Timestamp) -> str:
    # Full resolution: purge/embargo boundaries are generally not
    # midnight-aligned (they are bar-count offsets from a month boundary).
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
