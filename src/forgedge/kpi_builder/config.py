"""Configuration for the KPI Builder.

The configuration is a plain mapping ``{indicator_name: {"enabled": bool,
"params": {"periods": [...], "columns": [...]}}}``.  It is intentionally a
Python ``dict`` so the default works with **no extra dependency** (``forgedge``
ships only ``pandas``/``numpy``); YAML files are supported as an opt-in via
:func:`load_kpi_config`, which imports ``PyYAML`` lazily.

The ``time_column`` and ``suffix`` keys of the original *enricher* YAML are
intentionally dropped: the timestamp column is now a top-level argument of
:func:`~forgedge.kpi_builder.builder.build_features`, and the output column
names are fixed by the indicator functions (and must stay so for FORGE's
``FeatureGenerator`` to recognise them).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

# Reference YAML shipped alongside the package (an editable example for users);
# the runtime default below does not require it.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_enricher.yaml")

# Default KPI configuration (mirrors the reference YAML).  Indicators whose
# ``columns`` are absent in the input candles are skipped with a warning, so a
# config referencing ``volume`` is safe on OHLC-only data.
DEFAULT_CONFIG: Mapping = {
    "moving_average": {"enabled": True, "params": {"periods": [3, 9, 12, 25, 96, 168],
                                                   "columns": ["close", "high", "low", "volume"]}},
    "ema":            {"enabled": True, "params": {"periods": [3, 9, 12, 25, 96, 168],
                                                   "columns": ["close", "high", "low"]}},
    "volatility":     {"enabled": True, "params": {"periods": [5, 12, 24, 96, 168],
                                                   "columns": ["close"]}},
    "min":            {"enabled": True, "params": {"periods": [3, 6, 12, 24, 48, 96, 168],
                                                   "columns": ["close", "high", "low"]}},
    "max":            {"enabled": True, "params": {"periods": [3, 6, 12, 24, 48, 96, 168],
                                                   "columns": ["close", "high", "low"]}},
    "return":         {"enabled": True, "params": {"periods": [1, 3, 6, 12, 24, 48, 96, 168],
                                                   "columns": ["close"]}},
    "rsi":            {"enabled": True, "params": {"periods": [14, 25], "columns": ["close"]}},
    "bollinger_bands": {"enabled": True, "params": {"periods": [20], "columns": ["close"]}},
    "max_drawdown":   {"enabled": True, "params": {"periods": [12, 24, 48, 120, 240],
                                                   "columns": ["close"]}},
    # ATR requires `high`/`low` in addition to `columns` (normally just
    # "close"); disabled by default so OHLC-only consumers stay unaffected.
    # Two periods (not one) so FeatureGenerator can pair atr14/atr28 (and
    # natr14/natr28) into a same-family ratio — like EMA, ATR is unbounded
    # and not reliably scale-free standalone; see indicators.multiple_atr.
    "atr":            {"enabled": False, "params": {"periods": [14, 28],
                                                   "columns": ["close"]}},
    # MACD's `periods` is read as a flat list of (fast, slow, signal) triples,
    # not independent window lengths — see indicators.multiple_macd.
    "macd":           {"enabled": False, "params": {"periods": [12, 26, 9],
                                                   "columns": ["close"]}},
}
# NB: il lag non è nel config. Si applica dopo build_features() con lag_features(),
# così si ritardano colonne reali (anche derivate) senza conoscerne i nomi a priori.
#
# NB2: "indicator_lag_cross" (Event Discovery, non KPI Builder) non è qui né
# in default_enricher.yaml. FeatureGenerator genera automaticamente confronti
# indicatore-vs-base-OHLC sfasata nel tempo (es. close_sma_12[t] > low[t-3]),
# limitati alle famiglie price-scale (SMA/EMA/WMA/HMA); default lag = (1, 3),
# sovrascrivibile con DiscoveryConfig(indicator_lag_cross_lags=(...)) —
# see event_discovery.feature_generator._generate_indicator_lag_cross.


def load_kpi_config(path: "str | Path") -> dict:
    """Load a KPI configuration from a YAML file (opt-in, requires PyYAML).

    Parameters
    ----------
    path : str or Path
        Path to a YAML file with the same structure as :data:`DEFAULT_CONFIG`.

    Raises
    ------
    ImportError
        If ``PyYAML`` is not installed — pass a ``dict`` config instead.
    FileNotFoundError
        If ``path`` does not exist.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyYAML è richiesto per caricare config YAML. "
            "Installa 'pyyaml' oppure passa direttamente un dict come config."
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {p}")
    with open(p, "r") as fh:
        return yaml.safe_load(fh)
