"""Step 2 — market structure analysis (interpretive context).

This step does not select or generate events.  It verifies that the market has
the structural properties compatible with the alpha family being evaluated, so
the downstream measures can be read correctly: a mean-reversion candidate is
only expected to show positive lift when the price series is itself
mean-reverting (Hurst < 0.5).

The Hurst exponent reuses the DFA implementation already shipped for the
Market Context module — no second copy of the estimator.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..market_context.hurst import hurst_dfa
from .models import MarketStructure

# Default ACF lags (in bars) probed on the horizon return.
DEFAULT_ACF_LAGS: List[int] = [1, 2, 3, 6, 12, 24]


def analyse_market_structure(
    close: pd.Series,
    return_series: pd.Series,
    acf_lags: List[int] = DEFAULT_ACF_LAGS,
    hurst_band: float = 0.5,
    hurst_margin: float = 0.02,
) -> MarketStructure:
    """Compute the Hurst exponent and the return autocorrelation profile.

    Parameters
    ----------
    close : pd.Series
        Close-price levels (strictly positive) used for the Hurst estimate.
    return_series : pd.Series
        A return series whose autocorrelation locates the lag at which the
        market reverts (typically the horizon return).
    acf_lags : list[int]
        Lags, in bars, at which to report the autocorrelation.
    hurst_band : float
        Centre of the random-walk band (0.5).
    hurst_margin : float
        Half-width of the random-walk band: ``|H - 0.5| <= margin`` is read as
        a random walk, below as mean-reverting, above as trending.

    Returns
    -------
    MarketStructure
        Hurst value, its interpretation, the favoured alpha family, and the
        ACF at the requested lags.
    """
    prices = close.astype(float).dropna().to_numpy()
    h = hurst_dfa(prices)

    if not np.isfinite(h):
        interpretation = "undetermined"
        family = "none"
    elif h < hurst_band - hurst_margin:
        interpretation = "mean_reverting"
        family = "mean_reversion"
    elif h > hurst_band + hurst_margin:
        interpretation = "trending"
        family = "momentum"
    else:
        interpretation = "random_walk"
        family = "none"

    rets = return_series.astype(float).dropna()
    autocorr: Dict[int, float] = {}
    for lag in acf_lags:
        if lag < len(rets):
            autocorr[lag] = float(rets.autocorr(lag=lag))
        else:
            autocorr[lag] = float("nan")

    return MarketStructure(
        hurst=float(h) if np.isfinite(h) else float("nan"),
        hurst_interpretation=interpretation,
        expected_family=family,
        autocorr=autocorr,
    )
