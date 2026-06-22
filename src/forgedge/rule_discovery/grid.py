"""Operational-parameter grid screening for Rule Discovery (Step 2.4 / 3.1).

The grid varies only the order mechanics — ``buy_drop_pct``, ``sell_pct``,
``target_h``, ``buy_delay_bar`` — never the event expression, which is fixed by
the Alpha Contract.  Each combination is backtested and ranked by
``pf_score_tpm`` (PF balanced by monthly consistency), the criterion the
specification mandates to avoid selecting an overfit raw-PF winner.
"""
from __future__ import annotations

import itertools
from typing import List, Optional

import numpy as np
import pandas as pd

from .backtest import _PreparedCandles, run_backtest
from .models import (
    BacktestParams,
    GridResult,
    GridSpec,
    ScoringParams,
)


def build_grid(spec: GridSpec, base: BacktestParams) -> GridSpec:
    """Fill the unset grid axes with a sensible default around ``base``.

    A swept axis left as ``None`` is replaced by a small symmetric fan around
    the base value; ``buy_delay_bar`` defaults to the single base value.
    """
    drop = spec.buy_drop_pct
    sell = spec.sell_pct
    th = spec.target_h
    delay = spec.buy_delay_bar

    if drop is None:
        d = base.buy_drop_pct
        drop = [round(max(0.001, d + k), 4) for k in (-0.005, -0.002, 0.0, 0.002, 0.005)]
    if sell is None:
        s = base.sell_pct
        sell = [round(max(0.005, s + k), 4) for k in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    if th is None:
        h = base.target_h
        th = sorted({max(1, int(round(h * f))) for f in (0.5, 1.0, 2.0)})
    if delay is None:
        delay = [base.buy_delay_bar]

    # Deduplicate while preserving order.
    return GridSpec(
        buy_drop_pct=list(dict.fromkeys(drop)),
        sell_pct=list(dict.fromkeys(sell)),
        target_h=list(dict.fromkeys(th)),
        buy_delay_bar=list(dict.fromkeys(delay)),
    )


def run_grid(
    candle: pd.DataFrame,
    signal_col: str,
    base: BacktestParams,
    spec: GridSpec,
    scoring: Optional[ScoringParams] = None,
    timerange_from: Optional[str] = None,
    timerange_to: Optional[str] = None,
    timestamp_col: str = "open_dt",
    _prepared: Optional[_PreparedCandles] = None,
) -> List[GridResult]:
    """Backtest every combination of the resolved grid.

    Returns the results sorted by ``pf_score_tpm`` (descending) so the best
    operating point is first.  The per-bar candle arrays are extracted once and
    shared across every combination (and, when ``_prepared`` is supplied by the
    caller, across the whole walk-forward).
    """
    spec = build_grid(spec, base)
    scoring = scoring or ScoringParams()
    prep = _prepared if _prepared is not None else _PreparedCandles(
        candle, signal_col, timestamp_col
    )

    axes = {
        "buy_drop_pct": spec.buy_drop_pct,
        "sell_pct": spec.sell_pct,
        "target_h": spec.target_h,
        "buy_delay_bar": spec.buy_delay_bar,
    }
    keys = list(axes)
    results: List[GridResult] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        params = base.merged(**dict(zip(keys, combo)))
        summary = run_backtest(
            candle, signal_col, params,
            timerange_from=timerange_from, timerange_to=timerange_to,
            scoring=scoring, timestamp_col=timestamp_col, _prepared=prep,
        )
        results.append(GridResult(params=params, summary=summary))

    results.sort(key=lambda r: _sort_key(r.summary.pf_score_tpm), reverse=True)
    return results


def grid_dataframe(results: List[GridResult]) -> pd.DataFrame:
    """Flat, sortable DataFrame of a grid run (one row per combination)."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.row() for r in results])


def select_best(
    results: List[GridResult],
    criteria=None,
) -> Optional[GridResult]:
    """Pick the highest ``pf_score_tpm`` combination, preferring those that pass
    the acceptance criteria.

    When ``criteria`` is given, combinations meeting every gate are preferred;
    if none qualify the global ``pf_score_tpm`` leader is returned so callers
    can still inspect the best attempt (and reject it downstream).
    """
    if not results:
        return None
    if criteria is None:
        return results[0]

    qualified = [r for r in results if _passes(r.summary, criteria)]
    pool = qualified or results
    return max(pool, key=lambda r: _sort_key(r.summary.pf_score_tpm))


def _passes(summary, criteria) -> bool:
    s = summary
    return (
        np.isfinite(s.profit_factor) and s.profit_factor >= criteria.min_profit_factor
        and np.isfinite(s.win_rate_pct) and s.win_rate_pct >= criteria.min_win_rate
        # No absolute trade-count gate here — frequency is enforced relative to the
        # IS period via ``min_tpm`` (the dynamic floor lives in the verdict gates).
        and s.tpm_mu >= criteria.min_tpm
        and s.pf_score_tpm >= criteria.min_pf_score_tpm
        and np.isfinite(s.fill_rate) and s.fill_rate >= criteria.min_fill_rate
    )


def _sort_key(x: float) -> float:
    return x if (x is not None and np.isfinite(x)) else -np.inf
