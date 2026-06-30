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


def atr(df: pd.DataFrame, window: int, on: str, order_on: str) -> pd.Series:
    """Average True Range (Wilder smoothing), in the price units of ``on``.

    True range combines the current bar's range with any gap versus the
    previous close, so it needs ``high``/``low`` in addition to ``on`` (the
    close-equivalent column). ``on`` is normally ``"close"``.
    """
    sorted_df = df.sort_values(order_on, ascending=True)
    high = sorted_df["high"].astype(float)
    low = sorted_df["low"].astype(float)
    prev_close = sorted_df[on].astype(float).shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean().round(5)


def multiple_atr(df: pd.DataFrame, windows: list, on: str, order_on: str) -> pd.DataFrame:
    """ATR (raw) and NATR (normalised) over several windows.

    Two columns per window: ``{on}_atr_{w:02d}`` (raw, in price units — handy
    for stop-loss sizing) and ``{on}_natr_{w:02d}`` (= atr / ``on``, a
    dimensionless fraction).

    Unlike bounded oscillators (RSI, %B), neither is *guaranteed* scale-free
    standalone: both track volatility, which can drift across market regimes
    over a long history, so FORGE's empirical scale-free heuristic may still
    reject a single configured period. The robust path — the same one EMA
    and SMA already rely on — is configuring **two or more periods**: both
    ``atr`` and ``natr`` are recognised by ``parse_feature`` (family ``atr``/
    ``natr``), so FeatureGenerator auto-derives a same-family ratio (e.g.
    ``ratio_close_atr14_atr28``) that's scale-free by construction regardless
    of the inputs' own classification.
    """
    out = []
    close = df.sort_values(order_on, ascending=True)[on].astype(float)
    for w in windows:
        raw = atr(df, w, on, order_on)
        raw.name = f"{on.lower()}_atr_{w:02d}"
        natr = (raw / close).round(5)
        natr.name = f"{on.lower()}_natr_{w:02d}"
        out.append(raw)
        out.append(natr)
    return pd.concat(out, axis=1)


def macd(df: pd.DataFrame, fast: int, slow: int, signal: int, on: str, order_on: str) -> pd.DataFrame:
    """MACD line, signal line and histogram for one ``(fast, slow, signal)`` triple.

    The line is normalised by ``on`` (``(ema_fast - ema_slow) / on``) instead
    of the textbook raw price-unit difference, which improves (but, being
    momentum rather than a bounded oscillator, does not guarantee) the odds
    of passing FORGE's empirical scale-free check. Unlike ATR, MACD's column
    name carries two period parameters (``fast``/``slow``) and isn't
    recognised by ``parse_feature``, so there is no same-family ratio to
    rescue it if a given asset/period combination is classified non
    scale-free — configuring multiple ``(fast, slow, signal)`` triples does
    not help here the way extra periods help ATR/MDD.
    """
    sorted_series = df.sort_values(order_on, ascending=True)[on].astype(float)
    ema_fast = sorted_series.ewm(span=fast, adjust=False).mean()
    ema_slow = sorted_series.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / sorted_series
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    base = f"{on.lower()}_macd_{fast:02d}_{slow:02d}"
    macd_line = macd_line.round(5)
    macd_line.name = base
    signal_line = signal_line.round(5)
    signal_line.name = f"{base}_signal_{signal:02d}"
    hist = hist.round(5)
    hist.name = f"{base}_hist_{signal:02d}"
    return pd.concat([macd_line, signal_line, hist], axis=1)


def multiple_macd(df: pd.DataFrame, periods: list, on: str, order_on: str) -> pd.DataFrame:
    """MACD over one or more ``(fast, slow, signal)`` triples.

    ``periods`` is read as a flat list of triples, e.g. ``[12, 26, 9]`` for a
    single MACD(12,26,9), or ``[12, 26, 9, 5, 35, 5]`` for two configurations.
    Each triple produces three columns: the MACD line
    (``{on}_macd_{fast:02d}_{slow:02d}``), the signal line
    (``..._signal_{signal:02d}``) and the histogram (``..._hist_{signal:02d}``).
    """
    if len(periods) % 3 != 0:
        raise ValueError(
            f"macd: 'periods' deve contenere triple (fast, slow, signal); "
            f"ricevuti {len(periods)} valori: {periods}"
        )
    out = []
    for i in range(0, len(periods), 3):
        fast, slow, signal = periods[i:i + 3]
        out.append(macd(df, fast, slow, signal, on, order_on))
    return pd.concat(out, axis=1)
