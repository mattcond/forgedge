"""Step 1 — forward returns and the (derived) binary target.

Alpha Discovery receives no economic parameters: the holding horizon, the
sell percentage and the direction are *derived* per Event Candidate.  This
module provides the two look-ahead-safe building blocks of that derivation:

* :func:`forward_returns` — the point-to-point return over each candidate
  horizon, ``close[t+h] / close[t] - 1`` (signed, long convention).  The IC,
  the MFE-based ``sell_pct`` and the binary-target measures read from here;
* :func:`forward_log_returns` — the log-space counterpart
  ``log(close[t+h] / close[t])``, used by the direction/horizon derivation so
  the conditional **excess** return is robust to the extreme outliers of
  crypto forward returns;
* :func:`binary_target`  — once ``(h*, sell_pct*, direction*)`` have been
  derived, the binary economic target: did the favourable extreme within the
  next ``h`` bars reach ``sell_pct``?  For a long this is the forward
  *maximum*; for a short, the forward *minimum*.

All shifts use ``shift(-h)`` so no future information leaks into bar ``t``'s
features — the look-ahead lives only in the target/forward-return columns,
exactly where it belongs.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def forward_returns(close: pd.Series, horizons: Sequence[int]) -> pd.DataFrame:
    """Point-to-point forward return per candidate horizon.

    Parameters
    ----------
    close : pd.Series
        Close-price series, chronologically ordered, aligned to the KPI table.
    horizons : sequence of int
        Candidate holding horizons, in bars (each must be positive).

    Returns
    -------
    pd.DataFrame
        One column per horizon (``int`` column labels):
        ``close[t+h] / close[t] - 1``, NaN in the last ``h`` bars where the
        horizon runs off the end.  Signed with the **long** convention —
        direction is derived downstream from the sign of the advantage.
    """
    c = close.astype(float)
    out = {}
    for h in horizons:
        h = int(h)
        if h <= 0:
            raise ValueError(f"horizons must be positive, got {h}.")
        out[h] = c.shift(-h) / c - 1.0
    return pd.DataFrame(out, index=close.index)


def forward_log_returns(close: pd.Series, horizons: Sequence[int]) -> pd.DataFrame:
    """Point-to-point forward **log**-return per candidate horizon.

    The direction/horizon derivation (Step 1) operates in log space:
    ``r_h(t) = log(close[t+h] / close[t])``.  Crypto forward returns carry
    extreme right-skewed outliers (single bars of +70× are not unheard of on
    illiquid pairs); on a *simple* return the unconditional baseline is then
    dominated by a handful of jumps and the excess return ``μ_cond − μ_base``
    collapses to noise.  The log transform compresses those outliers
    (``log(71) ≈ 4.26``) without winsorising — winsorising would clip the
    baseline asymmetrically and bias the direction — so the conditional edge
    of a signal can be separated from the asset's drift.

    Parameters
    ----------
    close : pd.Series
        Close-price series, chronologically ordered, aligned to the KPI table.
    horizons : sequence of int
        Candidate holding horizons, in bars (each must be positive).

    Returns
    -------
    pd.DataFrame
        One column per horizon (``int`` column labels):
        ``log(close[t+h] / close[t])``, NaN in the last ``h`` bars where the
        horizon runs off the end and wherever a non-positive close makes the
        log undefined.  Signed with the **long** convention — direction is
        derived downstream from the sign of the *excess* log-return.
    """
    c = close.astype(float)
    safe = c.where(c > 0)
    out = {}
    for h in horizons:
        h = int(h)
        if h <= 0:
            raise ValueError(f"horizons must be positive, got {h}.")
        out[h] = np.log(safe.shift(-h) / safe)
    return pd.DataFrame(out, index=close.index)


def binary_target(
    close: pd.Series, holding_period_h: int, sell_pct: float, direction: str
) -> Tuple[pd.Series, float]:
    """Binary economic target at a derived ``(h, sell_pct, direction)``.

    Parameters
    ----------
    close : pd.Series
        Close-price series, chronologically ordered.
    holding_period_h : int
        Forward horizon in bars.
    sell_pct : float
        Return threshold the favourable extreme must reach (positive).
    direction : {'long', 'short'}
        Selects which forward extreme defines the target: the forward
        *maximum* for a long, the forward *minimum* for a short.

    Returns
    -------
    target_binary : pd.Series
        ``1.0`` when the favourable forward extreme reaches ``sell_pct``,
        ``0.0`` otherwise, NaN where the horizon is incomplete.
    base_rate : float
        ``target_binary.mean()`` — the unconditional win rate at these
        parameters, the benchmark the event's win rate is compared against.

    Raises
    ------
    ValueError
        If ``direction`` is neither ``"long"`` nor ``"short"``, or the horizon
        is not positive.
    """
    h = int(holding_period_h)
    if h <= 0:
        raise ValueError(f"holding_period_h must be positive, got {h}.")
    direction = direction.lower().strip()
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}.")

    c = close.astype(float)
    # Forward window [t+1 .. t+h]: shift the rolling extreme back by h so it is
    # attached to the originating bar t.  min_periods=h drops the tail where the
    # window is incomplete, preventing a truncated (look-ahead-safe) estimate.
    if direction == "long":
        fwd_ext = c.rolling(h, min_periods=h).max().shift(-h)
        hit = fwd_ext / c - 1.0 >= sell_pct
    else:
        fwd_ext = c.rolling(h, min_periods=h).min().shift(-h)
        hit = fwd_ext / c - 1.0 <= -sell_pct

    target_binary = hit.astype(float).where(fwd_ext.notna(), np.nan)
    base_rate = float(target_binary.mean())
    return target_binary, base_rate
