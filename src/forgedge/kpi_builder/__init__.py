"""KPI Builder — candles → KPI Table for forge() (FORGE "Modulo -1").

Standalone, external to :func:`forge`: it produces a FORGE-ready ``DataFrame``
from raw OHLCV candles and a KPI configuration.  The user passes :func:`forge`
either their own table or the output of :func:`build_features` / :func:`lag_features`.
"""
from .builder import build_features, lag_features, INDICATORS
from .candle import candle_features
from .config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, load_kpi_config

__all__ = [
    "build_features",
    "lag_features",
    "candle_features",
    "INDICATORS",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "load_kpi_config",
]
