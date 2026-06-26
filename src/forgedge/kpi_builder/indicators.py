"""Technical-analysis primitives used by the KPI Builder.

Pure ``pandas``/``numpy`` indicator functions, reused (almost verbatim) from the
original *enricher* ``ta.py``.  Every ``multiple_*`` function shares the same
signature ``(df, windows, on, order_on)`` and returns a DataFrame of one column
per window, named with the convention ``{base}_{indicator}_{period:02d}`` that
FORGE's ``FeatureGenerator`` recognises (e.g. ``close_ema_25``,
``close_bb_lower_20``, ``close_min_168``).

The functions sort by ``order_on`` internally but preserve the original index,
so the caller can ``join`` the results back by index.
"""
import numpy as np
import pandas as pd


def moving_average(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Simple moving average (SMA) of a time series."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .rolling(window=window, center=False)
            .mean()
            .round(5))


def multiple_moving_average(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """SMA over several windows → one column per window (``{on}_sma_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = moving_average(df, w, on, order_on)
        tmp.name = f"{on.lower()}_sma_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def lagging(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Lagged (shifted) version of a time series."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .shift(window)
            .round(5))


def multiple_lagging(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Lag over several windows → one column per window (``{on}_prev_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = lagging(df, w, on, order_on)
        tmp.name = f"{on.lower()}_prev_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def rolling_volatility(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Rolling standard deviation of pct-changes (volatility)."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .pct_change()
            .rolling(window=window, center=False)
            .std()
            .round(5))


def multiple_rolling_volatility(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Volatility over several windows (``{on}_vol_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = rolling_volatility(df, w, on, order_on)
        tmp.name = f"{on.lower()}_vol_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def rolling_min(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Rolling minimum."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .rolling(window=window, center=False)
            .min()
            .round(5))


def multiple_rolling_min(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Rolling minimum over several windows (``{on}_min_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = rolling_min(df, w, on, order_on)
        tmp.name = f"{on.lower()}_min_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def rolling_max(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Rolling maximum."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .rolling(window=window, center=False)
            .max()
            .round(5))


def multiple_rolling_max(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Rolling maximum over several windows (``{on}_max_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = rolling_max(df, w, on, order_on)
        tmp.name = f"{on.lower()}_max_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def returns(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Percentage return over ``window`` bars."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .astype(float)
            .pct_change(window)
            .round(5))


def multiple_returns(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Returns over several windows (``{on}_ret_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = returns(df, w, on, order_on)
        tmp.name = f"{on.lower()}_ret_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def rsi(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Relative Strength Index (Wilder smoothing via EWM)."""
    delta = (df
             .sort_values(order_on, ascending=True)[on]
             .diff())
    gain = (delta.where(delta > 0, 0)
            .ewm(alpha=1 / window, adjust=False)
            .mean())
    loss = (-delta.where(delta < 0, 0)
            .ewm(alpha=1 / window, adjust=False)
            .mean())
    rs = gain / (loss + 1e-10)  # piccolo epsilon per evitare la divisione per zero
    return (100 - (100 / (1 + rs))).round(5)


def multiple_rsi(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """RSI over several windows (``{on}_rsi_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = rsi(df, w, on, order_on)
        tmp.name = f"{on.lower()}_rsi_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def bollinger(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.DataFrame:
    """Bollinger bands (mid/upper/lower/width) at ±2 standard deviations."""
    sorted_df = df.sort_values(order_on, ascending=True)
    sma = sorted_df[on].rolling(window=window, center=False).mean()
    sma.name = f"{on.lower()}_bb_mid_{window:02d}"

    std = sorted_df[on].rolling(window=window, center=False).std()

    upper_band = (sma + (std * 2)).round(5)
    upper_band.name = f"{on.lower()}_bb_upper_{window:02d}"

    lower_band = (sma - (std * 2)).round(5)
    lower_band.name = f"{on.lower()}_bb_lower_{window:02d}"

    bb_width = (upper_band - lower_band) / sma
    bb_width.name = f"{on.lower()}_bb_width_{window:02d}"

    return pd.concat([sma, upper_band, lower_band, bb_width], axis=1)


def multiple_bollinger(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Bollinger bands over several windows."""
    out = [bollinger(df, w, on, order_on) for w in windows]
    return pd.concat(out, axis=1)


def max_drawdown(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Rolling maximum drawdown (absolute value)."""
    sorted_df = df.sort_values(order_on, ascending=True)
    roll_max = sorted_df[on].rolling(window=window, center=False).max().astype(float)
    realized_drawdown = sorted_df[on].astype(float) / roll_max - 1.0
    return realized_drawdown.rolling(window=window, center=False).min().abs().round(5)


def multiple_max_drawdown(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """Max drawdown over several windows (``{on}_mdd_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = max_drawdown(df, w, on, order_on)
        tmp.name = f"{on.lower()}_mdd_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)


def ema(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Exponential moving average (EMA)."""
    return (df
            .sort_values(order_on, ascending=True)[on]
            .ewm(span=window, adjust=False)
            .mean()
            .round(5))


def multiple_ema(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """EMA over several windows (``{on}_ema_{w:02d}``)."""
    out = []
    for w in windows:
        tmp = ema(df, w, on, order_on)
        tmp.name = f"{on.lower()}_ema_{w:02d}"
        out.append(tmp)
    return pd.concat(out, axis=1)
