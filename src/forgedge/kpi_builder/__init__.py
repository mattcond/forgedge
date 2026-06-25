"""KPI Builder — candles → KPI Table for forge() (FORGE "Modulo -1").

Standalone, external to :func:`forge`: it produces a FORGE-ready ``DataFrame``
from raw OHLCV candles and a KPI configuration.  The user passes :func:`forge`
either their own table or the output of :func:`build_kpi_table`.
"""
from .builder import build_kpi_table, INDICATORS
from .config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, load_kpi_config

__all__ = [
    "build_kpi_table",
    "INDICATORS",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "load_kpi_config",
]
