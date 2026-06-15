"""Execution envelope and MAE/MFE excursion analysis for Rule Discovery.

Two diagnostics that describe a rule's "range of action" without committing to a
single exit convention:

* **Execution envelope** — the selected configuration backtested under both
  meaningful take-profit conventions (``close`` = conservative, ``high`` =
  optimistic).  The true performance is bracketed between the two; no single
  optimistic/pessimistic choice is forced on the verdict.
* **MAE / MFE** — per-trade Maximum Adverse / Favourable Excursion over the
  realised holding window.  This is the natural home of the ``low``/``high``
  intrabar range (mirrors ``strategy_report_lib``'s ``target_low/high_return``):
  it characterises how far each trade swings before it closes, independently of
  the exit rule.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .backtest import optimistic_hit_col, run_backtest
from .models import (
    BacktestParams,
    ExcursionStats,
    ExecutionEnvelope,
    ScoringParams,
)


def execution_envelope(
    candle: pd.DataFrame,
    signal_col: str,
    params: BacktestParams,
    scoring: Optional[ScoringParams] = None,
    timestamp_col: str = "open_dt",
    timerange_from: Optional[str] = None,
    timerange_to: Optional[str] = None,
) -> ExecutionEnvelope:
    """Backtest ``params`` under the conservative (``close``) and optimistic
    (intrabar) take-profit conventions and return both summaries as a bracket.

    The optimistic column is direction-aware — ``high`` for a long, ``low`` for
    a short.  Two backtests total — independent of the grid size — so it is
    cheap to run once on the selected configuration.
    """
    scoring = scoring or ScoringParams()
    conservative = run_backtest(
        candle, signal_col, params.merged(target_hit_col="close"),
        timerange_from=timerange_from, timerange_to=timerange_to,
        scoring=scoring, timestamp_col=timestamp_col,
    )
    optimistic = run_backtest(
        candle, signal_col, params.merged(target_hit_col=optimistic_hit_col(params.direction)),
        timerange_from=timerange_from, timerange_to=timerange_to,
        scoring=scoring, timestamp_col=timestamp_col,
    )
    return ExecutionEnvelope(conservative=conservative, optimistic=optimistic)


def excursion_stats(
    candle: pd.DataFrame,
    trades: pd.DataFrame,
) -> Optional[ExcursionStats]:
    """Compute MAE/MFE over the realised holding window of each trade.

    Parameters
    ----------
    candle : pd.DataFrame
        The same candle table the trades were produced from — the trade
        ``fill_rn`` / ``exit_rn`` are positional indices into it.
    trades : pd.DataFrame
        Executed trades from :func:`run_backtest` (``return_trades=True``).
        The per-trade take-profit is read from ``sell_price`` / ``buy_price`` so
        the favourable-excursion-reached-target metric is correct even for a
        concatenation of trades with different ``sell_pct`` (e.g. walk-forward
        OOS windows that each re-selected their parameters).

    Returns
    -------
    ExcursionStats or None
        ``None`` when there are no trades.
    """
    if trades is None or trades.empty:
        return None

    low = candle["low"].to_numpy(dtype=float)
    high = candle["high"].to_numpy(dtype=float)
    n = len(candle)

    fill = trades["fill_rn"].to_numpy(dtype=np.int64)
    exit_ = trades["exit_rn"].to_numpy(dtype=np.int64)
    bp = trades["buy_price"].to_numpy(dtype=float)
    # Per-trade take-profit target as a (positive) fraction of the entry price.
    target = np.abs(trades["sell_price"].to_numpy(dtype=float) / np.where(bp > 0, bp, np.nan) - 1.0)
    # Excursion is expressed in P&L terms, so MAE ≤ MFE for both directions.
    if "direction" in trades.columns:
        directions = trades["direction"].dropna().unique()
        if len(directions) > 1:
            raise ValueError(
                f"excursion_stats received a mixed-direction trades frame "
                f"({directions.tolist()}); each call must contain a single "
                "direction (long or short)."
            )
        is_short = len(directions) == 1 and directions[0] == "short"
    else:
        is_short = False

    mae = np.full(fill.size, np.nan)
    mfe = np.full(fill.size, np.nan)
    for i in range(fill.size):
        lo = fill[i] + 1
        hi = exit_[i]  # inclusive of the exit bar
        if lo > hi or hi >= n or bp[i] <= 0:
            continue
        wl = low[lo : hi + 1]
        wh = high[lo : hi + 1]
        if wl.size == 0:
            continue
        if is_short:
            # Short P&L = (entry - price)/entry: adverse when price rises (high),
            # favourable when it falls (low).
            mae[i] = (bp[i] - wh.max()) / bp[i]
            mfe[i] = (bp[i] - wl.min()) / bp[i]
        else:
            mae[i] = (wl.min() - bp[i]) / bp[i]
            mfe[i] = (wh.max() - bp[i]) / bp[i]

    finite = np.isfinite(mae)
    mae = mae[finite]
    mfe = mfe[finite]
    tgt = target[finite]
    if mae.size == 0:
        return ExcursionStats(0, *(float("nan"),) * 6, 0.0)

    reached = float((mfe >= tgt).mean()) * 100.0
    return ExcursionStats(
        n_trades=int(mae.size),
        mae_mean=round(float(mae.mean()), 6),
        mae_median=round(float(np.median(mae)), 6),
        mae_worst=round(float(mae.min()), 6),
        mfe_mean=round(float(mfe.mean()), 6),
        mfe_median=round(float(np.median(mfe)), 6),
        mfe_best=round(float(mfe.max()), 6),
        mfe_reached_target_pct=round(reached, 2),
    )
