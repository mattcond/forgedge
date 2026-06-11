"""Walk-forward out-of-sample validation for Rule Discovery (Step 4).

The in-sample grid screening always overstates performance — it picks the best
of many configurations on data it has seen.  Walk-forward validation removes
that bias: the timeline is cut into consecutive test windows; for each, the
operational parameters are re-selected on the *preceding* train window and then
evaluated **once** on the untouched test window.  Concatenating the per-window
test trades yields the rule's honest out-of-sample track record.

Two modes (``WalkForwardConfig.train_span_months``):

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

from .backtest import run_backtest
from .grid import run_grid, select_best
from .models import (
    BacktestParams,
    GridSpec,
    ScoringParams,
    SelectionCriteria,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSplit,
)


def _month_bounds(candle: pd.DataFrame, timestamp_col: str):
    dt = pd.to_datetime(candle[timestamp_col])
    start = dt.min().to_period("M").to_timestamp()
    # Exclusive upper bound: first day of the month after the last candle.
    end = (dt.max().to_period("M") + 1).to_timestamp()
    return start, end


def _build_splits(start: pd.Timestamp, end: pd.Timestamp, cfg: WalkForwardConfig):
    """Return a list of ``(train_from, train_to, test_from, test_to)`` timestamps."""
    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    min_train = max(cfg.min_train_months, 1)
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
    cfg: WalkForwardConfig,
    scoring: Optional[ScoringParams] = None,
    criteria: Optional[SelectionCriteria] = None,
    timestamp_col: str = "open_dt",
) -> Optional[WalkForwardResult]:
    """Run the walk-forward validation and return the aggregated OOS result.

    Returns ``None`` when the data span is too short to form a single
    train/test split.
    """
    scoring = scoring or ScoringParams()
    criteria = criteria or SelectionCriteria()

    start, end = _month_bounds(candle, timestamp_col)
    bounds = _build_splits(start, end, cfg)
    if not bounds:
        return None

    splits: List[WalkForwardSplit] = []
    oos_trades: List[pd.DataFrame] = []

    for idx, (tr_from, tr_to, te_from, te_to) in enumerate(bounds):
        tr_from_s, tr_to_s = _fmt(tr_from), _fmt(tr_to)
        te_from_s, te_to_s = _fmt(te_from), _fmt(te_to)

        # ── select parameters on the train window ──
        if cfg.reoptimise:
            grid_res = run_grid(
                candle, signal_col, base, spec, scoring=scoring,
                timerange_from=tr_from_s, timerange_to=tr_to_s,
                timestamp_col=timestamp_col,
            )
            best = select_best(grid_res, criteria)
            params = best.params if best else base
            train_summary = best.summary if best else run_backtest(
                candle, signal_col, base, tr_from_s, tr_to_s, scoring, timestamp_col
            )
        else:
            params = base
            train_summary = run_backtest(
                candle, signal_col, params, tr_from_s, tr_to_s, scoring, timestamp_col
            )

        # ── evaluate once on the untouched test window ──
        test_summary, test_tr = run_backtest(
            candle, signal_col, params, te_from_s, te_to_s, scoring,
            timestamp_col, return_trades=True,
        )
        oos_trades.append(test_tr)

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
    oos_summary = _aggregate_oos(oos_trades, splits, scoring, timestamp_col)
    n_profitable = sum(1 for s in splits if s.test_summary.total_net_gain > 0)
    consistency = n_profitable / len(splits) if splits else 0.0

    return WalkForwardResult(
        splits=splits,
        oos_summary=oos_summary,
        n_profitable_splits=n_profitable,
        consistency=round(consistency, 4),
    )


def _aggregate_oos(oos_trades, splits, scoring, timestamp_col):
    """Summarise the concatenation of every test-window trade.

    Reuses the backtest summariser so the OOS track record carries the same
    metric set as any single run.
    """
    from .backtest import _summarise

    frames = [t for t in oos_trades if not t.empty]
    if not frames:
        from .backtest import _empty_summary
        n_months = sum(_span_months(s.test_from, s.test_to) for s in splits)
        return _empty_summary(max(n_months, 1))

    all_trades = pd.concat(frames, ignore_index=True)
    total_signals = sum(s.test_summary.total_signals for s in splits)

    # Months covered by the union of the test windows (no overlap by construction).
    months = pd.period_range(
        pd.Timestamp(splits[0].test_from).to_period("M"),
        pd.Timestamp(splits[-1].test_to).to_period("M") - 1,
        freq="M",
    )
    summary = _summarise(all_trades, total_signals, months, scoring)
    return summary


def _span_months(a: str, b: str) -> int:
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    return max((tb.year - ta.year) * 12 + (tb.month - ta.month), 0)


def _fmt(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")
