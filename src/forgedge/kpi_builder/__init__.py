"""KPI Builder — candles → KPI Table for forge() (FORGE "Modulo -1").

Standalone, external to :func:`forge`: it produces a FORGE-ready ``DataFrame``
from raw OHLCV candles and a KPI configuration.  The user passes :func:`forge`
either their own table or the output of these composable steps.

Recommended pipeline
--------------------
    build_features(candles, timestamp_col=...)  # base indicators
        → candle_features(df)                   # continuous candle geometry
        → lag_features(df, *cols)               # lags of real columns

Opt-in (not part of the recommended pipeline)
---------------------------------------------
:func:`pattern_features` adds *named* candlestick patterns as a single
**categorical** column (``candle_pattern``: ``"HAMMER"`` / ``"DOJI"`` / … /
``None``).  Prefer the continuous geometry of :func:`candle_features` so FORGE
derives its own distributional thresholds; use named patterns only when you
explicitly want ready-made pattern events.
"""
from .builder import build_features, lag_features, INDICATORS
from .candle import candle_features
from .candle_patterns import (
    PATTERNS,
    pattern_features,
    is_doji,
    is_hammer,
    is_shooting_star,
    is_marubozu,
    is_bullish_engulfing,
    is_bearish_engulfing,
    is_bullish_harami,
    is_bearish_harami,
    is_morning_star,
    is_evening_star,
)
from .config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, load_kpi_config

__all__ = [
    "build_features",
    "lag_features",
    "candle_features",
    "pattern_features",
    "PATTERNS",
    "is_doji",
    "is_hammer",
    "is_shooting_star",
    "is_marubozu",
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_bullish_harami",
    "is_bearish_harami",
    "is_morning_star",
    "is_evening_star",
    "INDICATORS",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "load_kpi_config",
]
