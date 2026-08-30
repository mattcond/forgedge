"""Playground use cases anchored on Modulo 0 — Market Context.

Two use cases from the tracking issue:

1. ``regime_transitions`` — how "nervous" are regime boundaries (how often,
   and after how short a run, the classifier flips label).
2. ``regime_time_share`` — how much of its history each asset spends in each
   regime, to spot assets effectively "prigionieri" of a single regime.

Both read ``ForgeResult.enriched`` — the KPI Table after Market Context, with
``regime``/``regime_stable`` appended — and skip any result where Market
Context was disabled (no ``regime`` column) rather than raising.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from ..forge import ForgeResult

__all__ = ["regime_transitions", "regime_time_share"]


def _timestamp_at(df: pd.DataFrame, position: int):
    """Best-effort timestamp for a row position: DatetimeIndex, then 'open_dt', else the position itself."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[position]
    if "open_dt" in df.columns:
        return df["open_dt"].iloc[position]
    return position


def regime_transitions(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format log of every regime flip, with the run length that preceded it.

    A short ``run_length_before`` right at a flip is what "nervous" boundaries
    look like: the classifier oscillating between labels rather than settling
    into a stable state — exactly the condition ``regime_stable`` (with its
    ``stable_window``) is meant to filter out downstream. Skips any result
    whose ``enriched`` frame has no ``regime`` column (Market Context
    disabled).

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``bar_index``, ``timestamp``, ``from_regime``,
        ``to_regime``, ``run_length_before``. Downstream, e.g.
        ``df[df["run_length_before"] <= 2].groupby("ticker").size()`` to rank
        assets by boundary nervousness.
    """
    rows: List[dict] = []

    for result in results:
        df = result.enriched
        if "regime" not in df.columns or df.empty:
            continue

        regime = df["regime"].astype("object")
        prev_regime = regime.shift(1)
        changed = regime.ne(prev_regime)
        run_id = changed.cumsum()
        run_length = regime.groupby(run_id).cumcount() + 1

        is_flip = changed.to_numpy() & prev_regime.notna().to_numpy()
        for position in np.flatnonzero(is_flip):
            rows.append(
                {
                    "ticker": result.ticker,
                    "bar_index": int(position),
                    "timestamp": _timestamp_at(df, position),
                    "from_regime": prev_regime.iloc[position],
                    "to_regime": regime.iloc[position],
                    "run_length_before": int(run_length.iloc[position - 1]),
                }
            )

    columns = [
        "ticker",
        "bar_index",
        "timestamp",
        "from_regime",
        "to_regime",
        "run_length_before",
    ]
    return pd.DataFrame(rows, columns=columns)


def regime_time_share(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format share of bars each asset spends in each regime.

    An asset with one regime dominating its whole history is a candidate for
    "rules discovered on it look generic but are actually regime-specific" —
    there was never enough of another regime present to prove otherwise.
    Skips any result whose ``enriched`` frame has no ``regime`` column.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``regime``, ``n_bars``, ``share`` (0-1, over the
        classified — non-NaN — bars of that result). Downstream, e.g.
        ``df.sort_values("share", ascending=False).groupby("ticker").head(1)``
        to find each asset's dominant regime.
    """
    rows: List[dict] = []

    for result in results:
        df = result.enriched
        if "regime" not in df.columns:
            continue

        counts = df["regime"].value_counts(dropna=True)
        total = int(counts.sum())
        for regime_label, n_bars in counts.items():
            rows.append(
                {
                    "ticker": result.ticker,
                    "regime": regime_label,
                    "n_bars": int(n_bars),
                    "share": (float(n_bars) / total) if total else float("nan"),
                }
            )

    columns = ["ticker", "regime", "n_bars", "share"]
    return pd.DataFrame(rows, columns=columns)
