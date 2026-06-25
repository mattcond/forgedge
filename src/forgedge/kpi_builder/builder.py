"""``build_kpi_table`` — turn raw candles into a FORGE-ready KPI Table.

This module is **external to** :func:`forge`: it produces a ``DataFrame`` that
:func:`forge` can consume, but never calls it.  The user is free to feed
:func:`forge` their own table or the output of :func:`build_kpi_table`.

Pipeline
--------
1. Resolve the KPI configuration (a ``dict`` or a YAML path; the packaged
   default is used when ``None``).
2. Derive the output datetime column (``output_timestamp_col``, default
   ``open_dt`` — what FORGE expects) from the user-declared ``timestamp_col``
   and sort chronologically.
3. **Phase 1 — base indicators**: every enabled indicator is dispatched to its
   ``ta.multiple_*`` function over each configured column.  Columns referenced
   by the config but absent in the candles are skipped with a warning.
4. **Phase 2 — lagging**: computed last, so it can lag columns *derived* in
   phase 1 (e.g. ``close_ema_03`` → ``close_ema_03_prev_01``).
5. ``NaN`` warm-up values are preserved (never coerced to ``None``) and the
   column names are exactly those FORGE's ``FeatureGenerator`` recognises.

Example
-------
    from forgedge import build_kpi_table, forge

    kpi = build_kpi_table(candles, timestamp_col="open_time")   # ms epoch
    result = forge(kpi, ticker="ADAUSDC", timeframe="1H")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd

from . import indicators as ta
from .config import DEFAULT_CONFIG, load_kpi_config

logger = logging.getLogger(__name__)

# Dispatch table: config indicator name → ta.multiple_* function.  Adding an
# indicator is one function in ``indicators.py`` plus one entry here.
INDICATORS = {
    "moving_average":  ta.multiple_moving_average,
    "ema":             ta.multiple_ema,
    "volatility":      ta.multiple_rolling_volatility,
    "min":             ta.multiple_rolling_min,
    "max":             ta.multiple_rolling_max,
    "return":          ta.multiple_returns,
    "rsi":             ta.multiple_rsi,
    "bollinger_bands": ta.multiple_bollinger,
    "max_drawdown":    ta.multiple_max_drawdown,
}
_LAGGING = "lagging"

ConfigType = Union[Mapping, str, Path, None]


def build_kpi_table(
    candles: pd.DataFrame,
    config: ConfigType = None,
    *,
    timestamp_col: str,
    output_timestamp_col: str = "open_dt",
    timestamp_unit: str = "ms",
    add_color: bool = True,
    sort_output: bool = True,
) -> pd.DataFrame:
    """Build a KPI Table from raw candles, ready to pass to :func:`forge`.

    Parameters
    ----------
    candles : pd.DataFrame
        Raw OHLCV candles.  Must contain ``timestamp_col`` and the price/volume
        columns referenced by ``config`` (typically ``open``/``high``/``low``/
        ``close``/``volume``).  Columns referenced by the config but missing here
        are skipped with a warning (so OHLC-only data is fine).
    config : dict | str | Path | None
        KPI configuration.  ``dict`` is used as-is; ``str``/``Path`` is loaded as
        YAML (requires PyYAML); ``None`` uses the packaged default
        (:data:`~forgedge.kpi_builder.config.DEFAULT_CONFIG`).
    timestamp_col : str, keyword-only, required
        Name of the column holding each candle's timestamp.  Used both to order
        the series and to derive ``output_timestamp_col``.
    output_timestamp_col : str
        Name of the datetime column produced for FORGE (default ``open_dt`` — the
        column :func:`forge` looks for).
    timestamp_unit : str
        Epoch unit used when ``timestamp_col`` is numeric (default ``"ms"``).
        Ignored when the column is already datetime or a parseable string.
    add_color : bool
        Add the built-in ``color`` column (+1 green / -1 red / 0 doji), available
        to phase-2 lagging.  Requires ``open`` and ``close``.
    sort_output : bool
        Return the table sorted chronologically by ``output_timestamp_col``.

    Returns
    -------
    pd.DataFrame
        The original candle columns plus all configured indicators (and lags),
        with ``output_timestamp_col`` as a datetime column.  ``NaN`` warm-up
        values are preserved.
    """
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("`candles` deve essere un pandas.DataFrame")
    if timestamp_col not in candles.columns:
        raise KeyError(
            f"timestamp_col '{timestamp_col}' non presente nelle candele. "
            f"Colonne disponibili: {list(candles.columns)}"
        )

    cfg = _resolve_config(config)
    df = candles.copy()

    # ── Output datetime column (also used as the ordering key) ────────────────
    df[output_timestamp_col] = _to_datetime(df[timestamp_col], timestamp_unit)
    df = df.sort_values(output_timestamp_col).reset_index(drop=True)
    order_on = output_timestamp_col

    # ── Phase 1 — base indicators ─────────────────────────────────────────────
    base_outputs = []
    for name, conf in cfg.items():
        if name == _LAGGING:
            continue
        if not _enabled(conf):
            logger.info("indicatore '%s' disabilitato, salto", name)
            continue
        fn = INDICATORS.get(name)
        if fn is None:
            logger.warning("indicatore '%s' non implementato, salto", name)
            continue
        periods, columns = _params(conf)
        for col in columns:
            if col not in df.columns:
                logger.warning("indicatore '%s': colonna assente '%s', salto", name, col)
                continue
            base_outputs.append(fn(df, periods, on=col, order_on=order_on))

    if add_color:
        if {"open", "close"}.issubset(df.columns):
            df["color"] = np.where(df["close"] > df["open"], 1,
                           np.where(df["close"] < df["open"], -1, 0))
        else:
            logger.warning("color richiede 'open' e 'close': salto")

    for res in base_outputs:
        df = df.join(res, how="left")

    # ── Phase 2 — lagging (può riferirsi a colonne DERIVATE in fase 1) ─────────
    lag_conf = cfg.get(_LAGGING)
    if lag_conf and _enabled(lag_conf):
        periods, columns = _params(lag_conf)
        lag_outputs = []
        for col in columns:
            if col not in df.columns:
                logger.warning("lagging: colonna assente '%s', salto", col)
                continue
            lag_outputs.append(ta.multiple_lagging(df, periods, on=col, order_on=order_on))
        for res in lag_outputs:
            df = df.join(res, how="left")

    if sort_output:
        df = df.sort_values(output_timestamp_col).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_config(config: ConfigType) -> Mapping:
    """Return a config mapping from a dict, a YAML path, or the default."""
    if config is None:
        return DEFAULT_CONFIG
    if isinstance(config, Mapping):
        return config
    if isinstance(config, (str, Path)):
        return load_kpi_config(config)
    raise TypeError(
        "`config` deve essere un dict, un percorso YAML (str/Path) o None; "
        f"ricevuto {type(config).__name__}"
    )


def _enabled(conf: Mapping) -> bool:
    """Whether an indicator block is enabled (missing flag → enabled)."""
    return bool(conf.get("enabled", True))


def _params(conf: Mapping) -> "tuple[list, list]":
    """Extract ``(periods, columns)`` from an indicator block."""
    params = conf.get("params", {}) or {}
    return list(params.get("periods", [])), list(params.get("columns", []))


def _to_datetime(series: pd.Series, unit: str) -> pd.Series:
    """Coerce a timestamp column to datetime.

    Datetime columns are returned unchanged; numeric columns are treated as
    epochs in ``unit`` (default milliseconds); anything else is parsed.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit=unit)
    return pd.to_datetime(series)
