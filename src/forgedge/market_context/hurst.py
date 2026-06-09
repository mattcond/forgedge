"""Hurst / Ornstein-Uhlenbeck analysis — tooling to choose the EMA windows.

This module is *not* part of the runtime regime classification.  It is the
offline analysis used to pick the fast/slow EMA spans of the
:class:`~forgedge.market_context.ema_proxy.EMAProxyClassifier`.

It ports the analysis from the ``hurst.ipynb`` notebook with two changes:

* it depends only on ``numpy``/``pandas`` (the OU fit uses ``numpy.linalg``
  instead of ``statsmodels``), so it adds no dependency to the package;
* it adds :func:`rolling_halflife`, a *local* OU half-life that is robust to
  the long-term price drift which contaminates a single global OU fit.

Rationale for the v1.0 defaults (``short_period=9``, ``long_period=25``)
-----------------------------------------------------------------------
On the supplied 1H data (ADAUSDC, DOGEUSDC) the rolling Hurst exponent is
well below 0.5 (mean-reverting) and the local OU half-life — estimated on a
~1-week window — is ~20h, stable across both assets.  Following the
convention "slow EMA span ≈ half-life, fast EMA span ≈ half-life / 2-3":

* ``long_period = 25``  ≈ the local OU half-life → the slow EMA tracks the
  mean-reversion level;
* ``short_period = 9``  ≈ half-life / 2.3        → the fast EMA tracks the
  short-term deviation from that level.

The EMA ratio ``ema_short / ema_long`` is therefore a proxy for "how far is
price above/below its mean-reversion level", which is exactly the trend
signal the regime classifier discretises.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timeframe utilities
# ---------------------------------------------------------------------------

TIMEFRAME_HOURS: Dict[str, float] = {
    "1m": 1 / 60, "5m": 5 / 60, "15m": 15 / 60, "30m": 0.5,
    "1h": 1.0, "2h": 2.0, "4h": 4.0, "6h": 6.0, "8h": 8.0,
    "12h": 12.0, "1d": 24.0, "3d": 72.0, "1w": 168.0,
}


def tf_hours(timeframe: str) -> float:
    """Return the number of hours per candle for ``timeframe``."""
    tf = timeframe.lower().strip()
    if tf not in TIMEFRAME_HOURS:
        raise ValueError(
            f"Timeframe '{timeframe}' not recognised. "
            f"Valid values: {list(TIMEFRAME_HOURS)}"
        )
    return TIMEFRAME_HOURS[tf]


def hours_to_candles(hours: float, timeframe: str) -> int:
    """Convert a number of hours into candles for ``timeframe`` (min 1)."""
    return max(1, int(np.round(hours / tf_hours(timeframe))))


# ---------------------------------------------------------------------------
# Hurst exponent — Detrended Fluctuation Analysis
# ---------------------------------------------------------------------------

def hurst_dfa(
    series: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> float:
    """Estimate the Hurst exponent via Detrended Fluctuation Analysis.

    Parameters
    ----------
    series : np.ndarray
        Price series (levels, not returns).
    min_window, max_window : int
        Inner DFA segment bounds, in candles.  ``max_window`` defaults to
        ``len(returns) // 4``.

    Returns
    -------
    float
        ``< 0.5`` mean-reverting, ``= 0.5`` random walk, ``> 0.5`` trending.
        ``NaN`` if the series is too short to fit the scaling law.
    """
    series = np.asarray(series, dtype=float)
    log_ret = np.diff(np.log(series))
    n = len(log_ret)

    if max_window is None:
        max_window = n // 4
    if max_window < min_window * 2:
        return np.nan

    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=20, dtype=int)
    )

    fluct: List[float] = []
    for w in windows:
        segments = [log_ret[i:i + w] for i in range(0, n - w + 1, w)]
        rms = []
        for seg in segments:
            if len(seg) < 4:
                continue
            x = np.arange(len(seg), dtype=float)
            coeff = np.polyfit(x, seg, 1)
            det = seg - np.polyval(coeff, x)
            rms.append(np.sqrt(np.mean(det ** 2)))
        fluct.append(float(np.mean(rms)) if rms else np.nan)

    valid = [
        (np.log(w), np.log(f))
        for w, f in zip(windows, fluct)
        if not np.isnan(f) and f > 0
    ]
    if len(valid) < 3:
        return np.nan

    log_w = np.array([v[0] for v in valid])
    log_f = np.array([v[1] for v in valid])
    return float(np.polyfit(log_w, log_f, 1)[0])


# ---------------------------------------------------------------------------
# Half-life — discrete Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------

def ou_halflife(series: np.ndarray) -> Optional[float]:
    """Estimate the OU mean-reversion half-life (in candles).

    Fits the discrete OU regression ``dP_t = const + kappa * P_(t-1) + eps``
    by ordinary least squares (``numpy.linalg.lstsq``) and returns
    ``-log(2) / log(1 + kappa)``.

    Returns
    -------
    float or None
        ``None`` when the process is not mean-reverting (``kappa >= 0``) or
        the resulting half-life is non-physical.
    """
    series = np.asarray(series, dtype=float)
    log_p = np.log(series)
    delta = np.diff(log_p)
    lag = log_p[:-1]

    design = np.vstack([np.ones_like(lag), lag]).T
    beta, *_ = np.linalg.lstsq(design, delta, rcond=None)
    kappa = float(beta[1])

    if kappa >= 0:
        return None
    hl = -np.log(2) / np.log(1 + kappa)
    if hl <= 0 or hl > len(series):
        return None
    return float(hl)


def rolling_halflife(
    prices: pd.Series,
    window_candles: int,
    stride_candles: int = 24,
) -> pd.Series:
    """Estimate the OU half-life on a rolling window.

    A single global OU fit on a trending crypto series is dominated by the
    long-term drift and yields an unusably large half-life.  Estimating the
    half-life locally — on a window of a few hundred candles — recovers the
    intraday mean-reversion timescale that actually drives the EMA-proxy
    regime signal.

    Parameters
    ----------
    prices : pd.Series
        Price levels.
    window_candles : int
        Width of the rolling estimation window, in candles.
    stride_candles : int
        Step between successive estimates, in candles.

    Returns
    -------
    pd.Series
        Half-life (in candles) indexed by the timestamp at the end of each
        window.  ``NaN`` where the local fit is not mean-reverting.
    """
    prices = prices.sort_index()
    values = prices.values
    times = prices.index
    out = {}
    for i in range(window_candles, len(values) + 1, stride_candles):
        hl = ou_halflife(values[i - window_candles:i])
        out[times[i - 1]] = np.nan if hl is None else hl
    return pd.Series(out, name="half_life")


# ---------------------------------------------------------------------------
# Variance Ratio profile
# ---------------------------------------------------------------------------

def variance_ratio_profile(
    series: np.ndarray,
    lags_candles: List[int],
) -> Dict[int, float]:
    """Variance Ratio for each lag (in candles).

    ``VR > 1`` momentum, ``VR = 1`` random walk, ``VR < 1`` mean reversion.
    """
    series = np.asarray(series, dtype=float)
    log_ret = np.diff(np.log(series))
    var_1 = np.var(log_ret, ddof=1)
    if var_1 == 0:
        return {lag: np.nan for lag in lags_candles}

    out: Dict[int, float] = {}
    for lag in lags_candles:
        if lag >= len(log_ret):
            out[lag] = np.nan
            continue
        ret_lag = np.array(
            [log_ret[i:i + lag].sum() for i in range(len(log_ret) - lag + 1)]
        )
        out[lag] = float(np.var(ret_lag, ddof=1) / (lag * var_1))
    return out


# ---------------------------------------------------------------------------
# Window suggestion
# ---------------------------------------------------------------------------

def suggest_ema_windows(
    prices: pd.Series,
    timeframe: str = "1h",
    estimation_window_hours: float = 168.0,
    stride_hours: float = 24.0,
    fast_ratio: float = 1 / 2.3,
) -> dict:
    """Suggest fast/slow EMA spans from the local OU half-life.

    The slow span is set to the median local half-life; the fast span to a
    fraction (``fast_ratio``) of it.  This is the procedure behind the v1.0
    defaults (``short_period=9``, ``long_period=25``).

    Parameters
    ----------
    prices : pd.Series
        Price levels, chronologically ordered.
    timeframe : str
        Candle timeframe (e.g. ``"1h"``), used to convert the estimation
        window from hours to candles.
    estimation_window_hours : float
        Width of the local half-life estimation window, in hours.
    stride_hours : float
        Step between successive estimates, in hours.
    fast_ratio : float
        Fast span as a fraction of the slow span (default ``1/2.3``).

    Returns
    -------
    dict
        ``half_life_candles``, ``half_life_hours``, ``suggested_short_period``,
        ``suggested_long_period``, ``n_estimates`` and ``hurst_median``.
    """
    win_c = hours_to_candles(estimation_window_hours, timeframe)
    stride_c = hours_to_candles(stride_hours, timeframe)

    hl_series = rolling_halflife(prices, win_c, stride_c).dropna()
    if hl_series.empty:
        raise ValueError(
            "No mean-reverting window found — cannot suggest EMA spans. "
            "The series may be purely trending over this estimation window."
        )

    hl_candles = float(hl_series.median())
    long_period = max(2, int(round(hl_candles)))
    short_period = max(1, int(round(hl_candles * fast_ratio)))

    h_series = _rolling_hurst(prices, win_c, stride_c).dropna()
    hurst_median = float(h_series.median()) if not h_series.empty else np.nan

    return {
        "half_life_candles": round(hl_candles, 1),
        "half_life_hours": round(hl_candles * tf_hours(timeframe), 1),
        "suggested_short_period": short_period,
        "suggested_long_period": long_period,
        "n_estimates": int(len(hl_series)),
        "hurst_median": round(hurst_median, 3) if not np.isnan(hurst_median) else np.nan,
    }


def _rolling_hurst(prices: pd.Series, window_candles: int, stride_candles: int) -> pd.Series:
    """Rolling Hurst exponent — used internally by :func:`suggest_ema_windows`."""
    prices = prices.sort_index()
    values = prices.values
    times = prices.index
    min_win = max(4, window_candles // 10)
    out = {}
    for i in range(window_candles, len(values) + 1, stride_candles):
        out[times[i - 1]] = hurst_dfa(values[i - window_candles:i], min_window=min_win)
    return pd.Series(out, name="H")
