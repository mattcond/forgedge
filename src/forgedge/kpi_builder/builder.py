"""``build_features`` + ``lag_features`` — candles → KPI Table for forge().

This module is **external to** :func:`forge`: it produces a ``DataFrame`` that
:func:`forge` can consume, but never calls it.  The user feeds :func:`forge`
either their own table or the output here.

Two composable steps
--------------------
1. :func:`build_features` — from raw candles + a KPI config, compute the base
   indicators (SMA, EMA, RSI, Bollinger, rolling min/max, returns, volatility,
   max-drawdown) plus the built-in ``color``.  It also derives the ``open_dt``
   datetime column FORGE expects and sorts chronologically.

2. :func:`lag_features` — lag *columns that already exist* in the built table.
   Because it runs after the build, the caller selects real, inspectable columns
   (by name or by substring via ``like=``) instead of having to know the derived
   names up front:

       kpi = build_features(candles, config, timestamp_col="open_time")
       kpi = lag_features(kpi, "close_ema_03", like="_rsi_", periods=[1, 2, 3])
       result = forge(kpi, ticker="ADAUSDC", timeframe="1H")

This makes lagging derived features trivial (e.g. lag every EMA:
``lag_features(kpi, *[c for c in kpi if "_ema_" in c])``) without coupling the
caller to the internal naming convention.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

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
_LAG_SUFFIX = "prev"

ConfigType = Union[Mapping, str, Path, None]


def build_features(
    candles: pd.DataFrame,
    config: ConfigType = None,
    *,
    timestamp_col: str,
    output_timestamp_col: str = "open_dt",
    timestamp_unit: str = "ms",
    add_color: bool = True,
    sort_output: bool = True,
) -> pd.DataFrame:
    """Compute the base KPI features from raw candles.

    Lagging is **not** done here — use :func:`lag_features` on the result so you
    can lag real, inspectable columns by name or pattern.

    Parameters
    ----------
    candles : pd.DataFrame
        Raw OHLCV candles.  Must contain ``timestamp_col`` and the price/volume
        columns referenced by ``config``.  Columns referenced by the config but
        missing here are skipped with a warning (so OHLC-only data is fine).
    config : dict | str | Path | None
        KPI configuration.  ``dict`` is used as-is; ``str``/``Path`` is loaded as
        YAML (requires PyYAML); ``None`` uses the packaged default
        (:data:`~forgedge.kpi_builder.config.DEFAULT_CONFIG`).  Any ``lagging``
        block is ignored — lagging now lives in :func:`lag_features`.
    timestamp_col : str, keyword-only, required
        Column holding each candle's timestamp.  Used to order the series and to
        derive ``output_timestamp_col``.
    output_timestamp_col : str
        Datetime column produced for FORGE (default ``open_dt``).
    timestamp_unit : str
        Epoch unit when ``timestamp_col`` is numeric (default ``"ms"``).
    add_color : bool
        Add the built-in ``color`` column (+1 green / -1 red / 0 doji).
    sort_output : bool
        Return sorted chronologically by ``output_timestamp_col``.

    Returns
    -------
    pd.DataFrame
        Candle columns + configured base indicators (+ ``color`` + ``open_dt``).
        ``NaN`` warm-up values are preserved.
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

    # ── Output datetime column (also the ordering key) ────────────────────────
    df[output_timestamp_col] = _to_datetime(df[timestamp_col], timestamp_unit)
    df = df.sort_values(output_timestamp_col).reset_index(drop=True)
    order_on = output_timestamp_col

    # ── Base indicators (dispatch) ────────────────────────────────────────────
    outputs = []
    for name, conf in cfg.items():
        if name == "lagging":
            logger.info("config: blocco 'lagging' ignorato — usa lag_features() dopo build_features()")
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
            outputs.append(fn(df, periods, on=col, order_on=order_on))

    if add_color:
        if {"open", "close"}.issubset(df.columns):
            df["color"] = np.where(df["close"] > df["open"], 1,
                           np.where(df["close"] < df["open"], -1, 0))
        else:
            logger.warning("color richiede 'open' e 'close': salto")

    for res in outputs:
        df = df.join(res, how="left")

    if sort_output:
        df = df.sort_values(output_timestamp_col).reset_index(drop=True)
    return df


def lag_features(
    df: pd.DataFrame,
    *cols: str,
    periods: Union[int, Sequence[int]] = (1, 2, 3),
    like: Optional[str] = None,
    order_on: str = "open_dt",
) -> pd.DataFrame:
    """Add lagged copies of existing columns to a built feature table.

    Runs *after* :func:`build_features`, so columns are real and inspectable —
    no need to know the derived names up front.  Each requested column ``c`` and
    period ``w`` produces ``{c}_prev_{w:02d}`` (the value ``w`` bars earlier).

    Parameters
    ----------
    df : pd.DataFrame
        A table produced by :func:`build_features` (or any frame ordered by
        ``order_on``).
    *cols : str
        Explicit column names to lag.  A name not present raises ``KeyError``
        (it is almost always a typo — the columns are known at this point).
    periods : int or sequence of int
        Lag lengths in bars (default ``(1, 2, 3)``).
    like : str, optional
        Also lag every column whose name *contains* this substring (e.g.
        ``like="_ema_"`` lags all EMAs).  The ``order_on`` column is never
        selected this way.
    order_on : str
        Column used to order the series before shifting (default ``open_dt``).
        When absent and the index is a ``DatetimeIndex`` the index is used;
        otherwise the existing row order is assumed chronological.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the lag columns appended.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` deve essere un pandas.DataFrame")

    selected: list[str] = []
    for c in cols:
        if c not in df.columns:
            sample = list(df.columns)[:12]
            raise KeyError(
                f"lag_features: colonna '{c}' non presente nel DataFrame. "
                f"Colonne (prime {len(sample)}): {sample}…"
            )
        if c not in selected:
            selected.append(c)
    if like is not None:
        for c in df.columns:
            if like in c and c != order_on and c not in selected:
                selected.append(c)
    if not selected:
        raise ValueError(
            "lag_features: nessuna colonna selezionata. "
            "Passa nomi espliciti (lag_features(df, 'close_ema_03', …)) o like=…"
        )

    if isinstance(periods, int):
        periods = (periods,)

    out = df.copy()
    if order_on in out.columns:
        base = out.sort_values(order_on)
    elif isinstance(out.index, pd.DatetimeIndex):
        base = out.sort_index()
    else:
        logger.debug("lag_features: '%s' assente e nessun DatetimeIndex; assumo già ordinato", order_on)
        base = out

    for col in selected:
        s = base[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        for w in periods:
            shifted = s.shift(w)
            if is_num:
                shifted = shifted.round(5)
            out[f"{col}_{_LAG_SUFFIX}_{int(w):02d}"] = shifted  # index-aligned back to out
    return out


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
