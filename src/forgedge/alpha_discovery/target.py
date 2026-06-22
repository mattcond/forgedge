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
  *maximum*; for a short, the forward *minimum*.  ``target_mode="proj"`` scores
  the forward return in **excess of the local trend** (PROJ_LOG, long only) so a
  trend-riding long is not credited with the trend premium;
* :func:`proj_log_excess` — the PROJ_LOG core: ``log(fwd_ext/close) −
  log(SMA_2h/SMA_2h[−h])``.

All shifts use ``shift(-h)`` so no future information leaks into bar ``t``'s
features — the look-ahead lives only in the target/forward-return columns,
exactly where it belongs.
"""
from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def proj_log_excess(
    close: pd.Series, fwd_ext: pd.Series, h: int
) -> Tuple[pd.Series, pd.Series]:
    """Forward log-return in **excess of the local trend** (PROJ_LOG core).

    ``log(fwd_ext / close) − log(SMA_2h[t] / SMA_2h[t−h])`` — the log-ratio of
    the price's forward performance to the trend's own performance over the same
    span.  Subtracting the trend stops a long event that merely rides a bull
    move from being credited with that move; the log form is symmetric,
    time-additive and compresses the extreme IS bars that otherwise inflate the
    lift without generalising.  The trend lookback is fixed to ``h`` (not exposed).

    Returns ``(log_excess, valid)`` where ``valid`` masks the bars with a
    complete forward window *and* a full ``2h`` trend window (needs ``≈3h`` bars
    of history).
    """
    c = close.astype(float)
    sma = c.rolling(2 * h, min_periods=2 * h).mean()
    sma_prev = sma.shift(h)
    valid = fwd_ext.notna() & sma.notna() & sma_prev.notna() & (c > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_excess = np.log(fwd_ext / c) - np.log(sma / sma_prev)
    return log_excess, valid


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
    close: pd.Series,
    holding_period_h: int,
    sell_pct: float,
    direction: str,
    target_mode: str = "abs",
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
    target_mode : {'abs', 'proj'}
        ``"abs"`` (default, back-compatible): absolute return from close —
        ``fwd_ext / close − 1 ≥ sell_pct``.  ``"proj"``: PROJ_LOG — the forward
        return in **excess of the local trend**, ``log(fwd_max/close) −
        log(SMA_2h/SMA_2h[−h]) ≥ log(1+sell_pct)``, so a long event that merely
        rides a bull trend is not credited with the trend premium.  PROJ applies
        to **long only**; ``"short"`` reverts to ``"abs"`` (the bear trend *is*
        the alpha to capture).  PROJ also reverts to ``"abs"`` when history is
        shorter than ``3·h`` (insufficient trend warmup), with a warning.

    Returns
    -------
    target_binary : pd.Series
        ``1.0`` when the (mode-dependent) target is met, ``0.0`` otherwise, NaN
        where the forward (or trend) window is incomplete.
    base_rate : float
        ``target_binary.mean()`` — the unconditional win rate at these
        parameters, the benchmark the event's win rate is compared against.

    Raises
    ------
    ValueError
        If ``direction`` is not ``"long"``/``"short"``, ``target_mode`` is not
        ``"abs"``/``"proj"``, or the horizon is not positive.
    """
    h = int(holding_period_h)
    if h <= 0:
        raise ValueError(f"holding_period_h must be positive, got {h}.")
    direction = direction.lower().strip()
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}.")
    target_mode = str(target_mode).lower().strip()
    if target_mode not in ("abs", "proj"):
        raise ValueError(f"target_mode must be 'abs' or 'proj', got {target_mode!r}.")

    c = close.astype(float)
    # Forward window [t+1 .. t+h]: shift the rolling extreme back by h so it is
    # attached to the originating bar t.  min_periods=h drops the tail where the
    # window is incomplete, preventing a truncated (look-ahead-safe) estimate.
    if direction == "long":
        fwd_ext = c.rolling(h, min_periods=h).max().shift(-h)
        if target_mode == "proj":
            if len(c) < 3 * h:
                logger.warning(
                    "binary_target: PROJ needs >= 3*h=%d bars for the trend "
                    "warmup, got %d — reverting to ABS.", 3 * h, len(c),
                )
            else:
                log_excess, valid = proj_log_excess(c, fwd_ext, h)
                hit = log_excess >= np.log(1.0 + sell_pct)
                target_binary = hit.astype(float).where(valid, np.nan)
                return target_binary, float(target_binary.mean())
        hit = fwd_ext / c - 1.0 >= sell_pct
    else:
        if target_mode == "proj":
            logger.debug(
                "binary_target: PROJ requested for a short target — reverting to "
                "ABS (the bear trend is the alpha to capture, not noise)."
            )
        fwd_ext = c.rolling(h, min_periods=h).min().shift(-h)
        hit = fwd_ext / c - 1.0 <= -sell_pct

    target_binary = hit.astype(float).where(fwd_ext.notna(), np.nan)
    base_rate = float(target_binary.mean())
    return target_binary, base_rate
