"""Step 1 — target construction and the base rate.

Builds the two forward-looking series Alpha Discovery measures against:

* ``fwd_return`` — the point-to-point return over the holding horizon,
  ``close[t+h] / close[t] - 1`` (used for the IC, Cohen's d and the t-test);
* ``target``     — the binary economic target: did the favourable extreme
  within the next ``h`` bars reach ``sell_pct``?  For a long this is the
  forward *maximum*; for a short, the forward *minimum*.

All shifts use ``shift(-h)`` so no future information leaks into bar ``t``'s
features — the look-ahead lives only in the target/forward-return columns,
exactly where it belongs.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .models import TargetDefinition


def build_target(
    close: pd.Series, target: TargetDefinition
) -> Tuple[pd.Series, pd.Series, float]:
    """Construct the forward return, the binary target and the base rate.

    Parameters
    ----------
    close : pd.Series
        Close-price series, chronologically ordered, aligned to the KPI table.
    target : TargetDefinition
        Holding period, sell percentage and direction.

    Returns
    -------
    fwd_return : pd.Series
        ``close[t+h] / close[t] - 1`` for a long; the negated return for a
        short (so that "good for the trade" is always positive).  NaN in the
        last ``h`` bars where the horizon runs off the end.
    target_binary : pd.Series
        ``1.0`` when the favourable forward extreme reaches ``sell_pct``,
        ``0.0`` otherwise, NaN where the horizon is incomplete.
    base_rate : float
        ``target_binary.mean()`` — the unconditional win rate, the benchmark
        every candidate's win rate is compared against.

    Raises
    ------
    ValueError
        If ``direction`` is neither ``"long"`` nor ``"short"``, or the horizon
        is not positive.
    """
    h = int(target.holding_period_h)
    if h <= 0:
        raise ValueError(f"holding_period_h must be positive, got {h}.")
    direction = target.direction.lower().strip()
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}.")

    c = close.astype(float)
    # Forward window [t+1 .. t+h]: shift the rolling extreme back by h so it is
    # attached to the originating bar t.  min_periods=h drops the tail where the
    # window is incomplete, preventing a truncated (look-ahead-safe) estimate.
    fwd_max = c.rolling(h, min_periods=h).max().shift(-h)
    fwd_min = c.rolling(h, min_periods=h).min().shift(-h)
    fwd_close = c.shift(-h)

    if direction == "long":
        fwd_return = fwd_close / c - 1.0
        target_binary = (fwd_max / c - 1.0 >= target.sell_pct)
    else:
        # Short: profit when price falls; express both as "good = positive".
        fwd_return = -(fwd_close / c - 1.0)
        target_binary = (fwd_min / c - 1.0 <= -target.sell_pct)

    horizon_valid = fwd_close.notna() & fwd_max.notna() & fwd_min.notna()
    target_binary = target_binary.astype(float).where(horizon_valid, np.nan)
    fwd_return = fwd_return.where(horizon_valid, np.nan)

    base_rate = float(target_binary.mean())
    return fwd_return, target_binary, base_rate
