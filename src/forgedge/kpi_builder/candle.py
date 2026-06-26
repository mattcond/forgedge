"""Per-bar candlestick *geometry* features (continuous, scale-free).

Instead of pre-baking named patterns (``is_hammer`` …) with hardcoded human
thresholds — which works against FORGE deriving its own distributional
thresholds — this exposes the raw geometric primitives of each candle as
continuous, scale-free measures.  FORGE's Event Discovery can then discover
pattern-like conditions itself with asset-adaptive thresholds, e.g. a bullish
hammer becomes ``lower_wick > p90 AND close_pos > p80``.

Features (all in ``[-1, 1]`` or ``[0, 1]``, invariant to a global price scale):

==============  ===========================================  ===============
column          formula                                       meaning
==============  ===========================================  ===============
``body``        ``(close - open) / (high - low)``             signed body size
``upper_wick``  ``(high - max(open, close)) / (high - low)``  upper shadow
``lower_wick``  ``(min(open, close) - low) / (high - low)``   lower shadow
``close_pos``   ``(close - low) / (high - low)``              close in range
``range_pct``   ``(high - low) / close``                      bar amplitude
``gap``         ``(open - prev_close) / prev_close``          opening gap
==============  ===========================================  ===============

By construction ``abs(body) + upper_wick + lower_wick == 1`` on non-flat bars.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLC = ("open", "high", "low", "close")


def candle_features(
    df: pd.DataFrame,
    *,
    order_on: str = "open_dt",
    add_gap: bool = True,
    round_to: int = 5,
) -> pd.DataFrame:
    """Add per-bar candlestick geometry features to ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        A frame with ``open``/``high``/``low``/``close`` (e.g. the output of
        :func:`~forgedge.kpi_builder.builder.build_features`).
    order_on : str
        Column used to order the series before computing ``gap`` (default
        ``open_dt``).  Falls back to a ``DatetimeIndex``; otherwise assumes the
        existing row order is chronological.
    add_gap : bool
        Compute the opening ``gap`` vs the previous bar's close (needs ordering).
    round_to : int
        Decimal places for the produced columns.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the geometry columns appended.  Flat bars
        (``high == low``) yield ``NaN`` for the ratio features rather than
        dividing by zero.
    """
    missing = [c for c in _OHLC if c not in df.columns]
    if missing:
        raise KeyError(
            f"candle_features richiede le colonne {list(_OHLC)}; mancano: {missing}"
        )

    out = df.copy()
    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)
    rng = h - l

    def _sdiv(num: pd.Series, den: pd.Series) -> pd.Series:
        """Safe divide: ±inf (division by a zero range) → NaN."""
        return (num / den).replace([np.inf, -np.inf], np.nan)

    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    out["body"] = _sdiv(c - o, rng).round(round_to)
    out["upper_wick"] = _sdiv(h - body_top, rng).round(round_to)
    out["lower_wick"] = _sdiv(body_bot - l, rng).round(round_to)
    out["close_pos"] = _sdiv(c - l, rng).round(round_to)
    out["range_pct"] = _sdiv(rng, c).round(round_to)

    if add_gap:
        if order_on in out.columns:
            base = out.sort_values(order_on)
        elif isinstance(out.index, pd.DatetimeIndex):
            base = out.sort_index()
        else:
            logger.debug("candle_features: '%s' assente e nessun DatetimeIndex; assumo già ordinato", order_on)
            base = out
        prev_close = base["close"].astype(float).shift(1)
        gap = (base["open"].astype(float) - prev_close) / prev_close
        # index-aligned back to `out`
        out["gap"] = gap.replace([np.inf, -np.inf], np.nan).round(round_to)

    return out
