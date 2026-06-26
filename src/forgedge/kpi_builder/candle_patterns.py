"""Named candlestick patterns as **categorical** features (opt-in).

This module is intentionally **not** part of the recommended pipeline
(``build_features`` → ``candle_features`` → ``lag_features``).  Named patterns
encode *fixed human thresholds* (the definition of a "hammer"), whereas FORGE is
at its best deriving its own distributional thresholds from the continuous
geometry in :mod:`~forgedge.kpi_builder.candle`.  It is provided for users who
explicitly want ready-made pattern events.

Categorical, not boolean
-------------------------
Each detector returns the **pattern name** (e.g. ``"HAMMER"``) on bars where the
pattern is present and ``None`` elsewhere — *never* ``1``/``0``.  Returning a
string makes FORGE's type classifier treat the value as **categorical**, so it
generates ``== "HAMMER"`` events instead of performing algebraic / continuous
transforms on it (which would be meaningless for a pattern flag).

One column, not one-per-pattern
-------------------------------
:func:`pattern_features` coalesces the detectors into a **single** categorical
column (default ``candle_pattern``) holding the matched pattern name (or
``None``).  This is deliberate: FORGE's ``TypeClassifier`` skips columns with
≤ 1 distinct non-null value, so a per-pattern ``{"HAMMER", None}`` column would
be **ignored**.  A single column carrying many pattern names is classified
CATEGORICAL and one-hot expanded into a ``== "HAMMER"`` / ``== "DOJI"`` … event
per value — the same effect, but visible to Event Discovery.  When several
patterns match the same bar, the first by precedence wins (see ``PATTERNS``).

    from forgedge.kpi_builder import pattern_features, is_hammer

    s = is_hammer(candles)                 # Series of "HAMMER" / None
    kpi = pattern_features(kpi)            # single 'candle_pattern' categorical column

Scoring through ``forge()``
---------------------------
Event Discovery (M1) classifies ``candle_pattern`` as categorical and mines
``== "HAMMER"`` events from it; Alpha Discovery (M2) scores them end-to-end via
a point-biserial IC on the binarised one-hot, so the column flows through a full
``forge()`` run.  It stays **opt-in on quality grounds, not technical ones**:
named patterns encode fixed human thresholds, so the continuous
:mod:`~forgedge.kpi_builder.candle` geometry — which lets FORGE derive its own
asset-adaptive thresholds — is preferred for automatic discovery.

Patterns implemented (precedence order): ``MORNING_STAR``, ``EVENING_STAR``,
``BULLISH_ENGULFING``, ``BEARISH_ENGULFING``, ``BULLISH_HARAMI``,
``BEARISH_HARAMI``, ``MARUBOZU``, ``HAMMER``, ``SHOOTING_STAR``, ``DOJI``.
Multi-bar patterns (2-/3-bar) read the previous bar(s) via ``shift``, so they
need chronologically ordered candles: both the detectors and ``pattern_features``
order by the ``order_on`` timestamp column (default ``open_dt``); when it is
missing ``pattern_features`` raises rather than silently assuming sorted input.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLC = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlc(df: pd.DataFrame):
    missing = [c for c in _OHLC if c not in df.columns]
    if missing:
        raise KeyError(f"i pattern richiedono {list(_OHLC)}; mancano: {missing}")
    return (df["open"].astype(float), df["high"].astype(float),
            df["low"].astype(float), df["close"].astype(float))


def _fractions(df: pd.DataFrame):
    """Per-bar body / upper-wick / lower-wick as fractions of the range."""
    o, h, l, c = _ohlc(df)
    rng = (h - l).replace(0, np.nan)
    f_body = (c - o).abs() / rng
    f_up = (h - np.maximum(o, c)) / rng
    f_low = (np.minimum(o, c) - l) / rng
    return o, h, l, c, f_body, f_up, f_low


def _named(mask: pd.Series, name: str) -> pd.Series:
    """Boolean mask → categorical Series of ``name`` / ``None``."""
    arr = np.where(np.asarray(mask, dtype=bool), name, None)
    return pd.Series(arr, index=mask.index, dtype=object)


def _order(df: pd.DataFrame, order_on: str) -> pd.DataFrame:
    """Return ``df`` ordered chronologically for ``.shift()``-based patterns.

    Sorts by the ``order_on`` timestamp column when present, else by a
    ``DatetimeIndex``; otherwise the existing row order is assumed chronological.
    The index is preserved so callers can ``reindex`` back to the input order.
    """
    if order_on in df.columns:
        return df.sort_values(order_on)
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Single-bar patterns
# ---------------------------------------------------------------------------

def is_doji(df: pd.DataFrame, *, body_max: float = 0.1) -> pd.Series:
    """Open ≈ close: body is at most ``body_max`` of the range → ``"DOJI"``."""
    *_, f_body, _, _ = _fractions(df)
    return _named(f_body <= body_max, "DOJI")


def is_hammer(df: pd.DataFrame, *, wick_mult: float = 2.0,
              body_max: float = 0.35, opp_max: float = 0.15) -> pd.Series:
    """Small body, long lower wick, tiny upper wick → ``"HAMMER"``."""
    *_, f_body, f_up, f_low = _fractions(df)
    mask = (f_body > 0) & (f_body <= body_max) & (f_low >= wick_mult * f_body) & (f_up <= opp_max)
    return _named(mask, "HAMMER")


def is_shooting_star(df: pd.DataFrame, *, wick_mult: float = 2.0,
                     body_max: float = 0.35, opp_max: float = 0.15) -> pd.Series:
    """Small body, long upper wick, tiny lower wick → ``"SHOOTING_STAR"``."""
    *_, f_body, f_up, f_low = _fractions(df)
    mask = (f_body > 0) & (f_body <= body_max) & (f_up >= wick_mult * f_body) & (f_low <= opp_max)
    return _named(mask, "SHOOTING_STAR")


def is_marubozu(df: pd.DataFrame, *, body_min: float = 0.9) -> pd.Series:
    """Body fills ≥ ``body_min`` of the range (almost no wicks) → ``"MARUBOZU"``."""
    *_, f_body, _, _ = _fractions(df)
    return _named(f_body >= body_min, "MARUBOZU")


# ---------------------------------------------------------------------------
# Two-bar patterns — ordered by `order_on` so "previous bar" is the real one
# ---------------------------------------------------------------------------

def is_bullish_engulfing(df: pd.DataFrame, *, order_on: str = "open_dt") -> pd.Series:
    """Down bar followed by an up bar whose body engulfs it → ``"BULLISH_ENGULFING"``."""
    o, _, _, c = _ohlc(_order(df, order_on))
    po, pc = o.shift(1), c.shift(1)
    mask = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    return _named(mask, "BULLISH_ENGULFING").reindex(df.index)


def is_bearish_engulfing(df: pd.DataFrame, *, order_on: str = "open_dt") -> pd.Series:
    """Up bar followed by a down bar whose body engulfs it → ``"BEARISH_ENGULFING"``."""
    o, _, _, c = _ohlc(_order(df, order_on))
    po, pc = o.shift(1), c.shift(1)
    mask = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    return _named(mask, "BEARISH_ENGULFING").reindex(df.index)


def is_bullish_harami(df: pd.DataFrame, *, order_on: str = "open_dt") -> pd.Series:
    """Large down bar then a small up bar contained in its body → ``"BULLISH_HARAMI"``."""
    o, _, _, c = _ohlc(_order(df, order_on))
    po, pc = o.shift(1), c.shift(1)
    top, bot = np.maximum(o, c), np.minimum(o, c)
    mask = (pc < po) & (c > o) & (top <= po) & (bot >= pc)
    return _named(mask, "BULLISH_HARAMI").reindex(df.index)


def is_bearish_harami(df: pd.DataFrame, *, order_on: str = "open_dt") -> pd.Series:
    """Large up bar then a small down bar contained in its body → ``"BEARISH_HARAMI"``."""
    o, _, _, c = _ohlc(_order(df, order_on))
    po, pc = o.shift(1), c.shift(1)
    top, bot = np.maximum(o, c), np.minimum(o, c)
    mask = (pc > po) & (c < o) & (top <= pc) & (bot >= po)
    return _named(mask, "BEARISH_HARAMI").reindex(df.index)


# ---------------------------------------------------------------------------
# Three-bar patterns (simplified: gap not required; reclaim of bar-1 body)
# ---------------------------------------------------------------------------

def is_morning_star(df: pd.DataFrame, *, order_on: str = "open_dt",
                    star_body_max: float = 0.3) -> pd.Series:
    """Bearish bar, small-bodied star, bullish bar reclaiming bar-1 → ``"MORNING_STAR"``."""
    o, h, l, c = _ohlc(_order(df, order_on))
    o1, c1 = o.shift(2), c.shift(2)
    o2, c2, h2, l2 = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    f_body2 = (c2 - o2).abs() / (h2 - l2).replace(0, np.nan)
    mid1 = (o1 + c1) / 2.0
    mask = (c1 < o1) & (f_body2 <= star_body_max) & (c > o) & (c >= mid1)
    return _named(mask, "MORNING_STAR").reindex(df.index)


def is_evening_star(df: pd.DataFrame, *, order_on: str = "open_dt",
                    star_body_max: float = 0.3) -> pd.Series:
    """Bullish bar, small-bodied star, bearish bar breaking bar-1 → ``"EVENING_STAR"``."""
    o, h, l, c = _ohlc(_order(df, order_on))
    o1, c1 = o.shift(2), c.shift(2)
    o2, c2, h2, l2 = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    f_body2 = (c2 - o2).abs() / (h2 - l2).replace(0, np.nan)
    mid1 = (o1 + c1) / 2.0
    mask = (c1 > o1) & (f_body2 <= star_body_max) & (c < o) & (c <= mid1)
    return _named(mask, "EVENING_STAR").reindex(df.index)


# ---------------------------------------------------------------------------
# Registry + step
# ---------------------------------------------------------------------------

# Ordered by precedence (most specific / informative first): when several
# patterns match the same bar, the first one here wins in the combined column.
PATTERNS = {
    "morning_star": is_morning_star,
    "evening_star": is_evening_star,
    "bullish_engulfing": is_bullish_engulfing,
    "bearish_engulfing": is_bearish_engulfing,
    "bullish_harami": is_bullish_harami,
    "bearish_harami": is_bearish_harami,
    "marubozu": is_marubozu,
    "hammer": is_hammer,
    "shooting_star": is_shooting_star,
    "doji": is_doji,
}

# Patterns that look at the previous bar(s): they need the candles ordered by the
# timestamp column, and receive ``order_on``.  The single-bar patterns are
# order-independent.
_MULTIBAR = frozenset({
    "morning_star", "evening_star",
    "bullish_engulfing", "bearish_engulfing",
    "bullish_harami", "bearish_harami",
})


def pattern_features(
    df: pd.DataFrame,
    *,
    patterns: Optional[Sequence[str]] = None,
    order_on: str = "open_dt",
    col: str = "candle_pattern",
) -> pd.DataFrame:
    """Add a single **categorical** column of named candlestick patterns (opt-in).

    Not part of the recommended pipeline — prefer the continuous
    :func:`~forgedge.kpi_builder.candle.candle_features` so FORGE derives its own
    thresholds.  The added column ``col`` holds the matched pattern name (e.g.
    ``"HAMMER"``) or ``None``; FORGE classifies it CATEGORICAL and one-hots it
    into one ``== "<PATTERN>"`` event per value.

    A *single* column (rather than one boolean per pattern) is used on purpose:
    FORGE's classifier skips columns with ≤ 1 distinct non-null value, so a
    ``{"HAMMER", None}`` column would be ignored.  When several patterns match a
    bar, the first by :data:`PATTERNS` precedence wins.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with ``open``/``high``/``low``/``close`` (e.g. the output of
        :func:`~forgedge.kpi_builder.builder.build_features`).
    patterns : sequence of str, optional
        Subset of :data:`PATTERNS` keys to consider (in precedence order).
        ``None`` uses all.
    order_on : str
        Timestamp column used to **order the candles** before evaluating the
        2-/3-bar patterns (default ``open_dt``); a ``DatetimeIndex`` is accepted
        as a fallback.  When a multi-bar pattern is requested and neither is
        available, a ``KeyError`` is raised — the previous-bar relationship can
        only be guaranteed on chronologically ordered candles.
    col : str
        Name of the produced categorical column (default ``"candle_pattern"``).

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the categorical pattern column appended.
    """
    keys = list(patterns) if patterns is not None else list(PATTERNS)
    unknown = [k for k in keys if k not in PATTERNS]
    if unknown:
        raise KeyError(f"pattern sconosciuti: {unknown}. Disponibili: {list(PATTERNS)}")
    _ohlc(df)  # validate OHLC presence up front

    # Guarantee chronological order for 2-/3-bar patterns (they read the previous
    # bar via shift): require the timestamp column (or a DatetimeIndex).
    if any(k in _MULTIBAR for k in keys):
        if order_on not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError(
                f"pattern_features: i pattern a 2-3 candele richiedono la colonna "
                f"timestamp '{order_on}' (o un DatetimeIndex) per ordinare le candele. "
                f"Passa order_on=<colonna> oppure costruisci la tabella con build_features()."
            )

    # Coalesce detectors in precedence order: first non-null name wins.  Multi-bar
    # detectors order internally via order_on and realign to df's index.
    result = pd.Series(None, index=df.index, dtype=object)
    for key in keys:
        det = PATTERNS[key]
        series = det(df, order_on=order_on) if key in _MULTIBAR else det(df)
        result = result.where(result.notna(), series)

    out = df.copy()
    out[col] = result  # index-aligned
    return out
