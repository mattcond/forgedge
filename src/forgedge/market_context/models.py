"""Core data structures and interface for the Market Context module.

The Market Context Module (Modulo 0 of the FORGE pipeline) classifies every
bar of the KPI Table by market regime.  It does not implement the classification
logic itself — it delegates to an object implementing the :class:`RegimeClassifier`
interface.  This is the extensibility point that lets the classifier be swapped
in future versions (HMM, KMeans, custom) without touching any downstream module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Default regime labels (ordered from most bearish to most bullish)
# ---------------------------------------------------------------------------

DEFAULT_LABELS: List[str] = [
    "STRONG_BEAR",
    "BEAR",
    "NEUTRAL",
    "BULL",
    "STRONG_BULL",
]

# Output column names added to the KPI Table.  Part of the interface contract:
# every RegimeClassifier implementation must ultimately produce these.
REGIME_COL = "regime"
REGIME_STABLE_COL = "regime_stable"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EMAProxyConfig:
    """Configuration for the v1.0 :class:`EMAProxyClassifier`.

    Attributes
    ----------
    source_col : str
        OHLCV column on which the EMAs are computed (default ``close``).
        Only this column is required in the user's KPI Table — the EMA
        indicators themselves do not need to be present.
    auto_window : bool
        When ``True`` (default) the Market Context Module derives the fast and
        slow EMA spans from the loaded data via the Hurst / Ornstein-Uhlenbeck
        analysis (the local mean-reversion half-life).  ``short_period`` and
        ``long_period`` are then used **only as a fallback** when the analysis
        does not converge (no mean-reverting half-life on the series).  Set to
        ``False`` to force the configured ``short_period`` / ``long_period``.
    short_period : int
        Fallback span of the fast EMA, used when ``auto_window`` is ``False``
        or the OU half-life does not converge.  The default of ``9`` ≈ the
        observed intraday half-life / 2.3 on crypto 1H data.
    long_period : int
        Fallback span of the slow EMA.  The default of ``25`` ≈ the observed
        intraday OU half-life on crypto 1H data.
    thresholds : list[float]
        Ascending cut points applied to the ``ema_short / ema_long`` ratio.
        Their number must be exactly ``len(labels) - 1``.  The defaults
        ``[0.975, 0.990, 1.010, 1.025]`` are calibrated empirically on
        crypto 1H data.
    window_estimation_bars : int
        Width, in bars, of the rolling window used to estimate the local OU
        half-life when ``auto_window`` is ``True``.
    window_stride_bars : int
        Step, in bars, between successive local half-life estimates.
    fast_ratio : float
        Fast span as a fraction of the slow span when auto-deriving
        (default ``1 / 2.3``).
    min_window_estimates : int
        Minimum number of converging local half-life estimates required to
        trust the auto-derivation.  Below this the module falls back to
        ``short_period`` / ``long_period``.
    """

    source_col: str = "close"
    auto_window: bool = True
    short_period: int = 9
    long_period: int = 25
    thresholds: List[float] = field(
        default_factory=lambda: [0.975, 0.990, 1.010, 1.025]
    )
    window_estimation_bars: int = 168
    window_stride_bars: int = 24
    fast_ratio: float = 1 / 2.3
    min_window_estimates: int = 10



@dataclass
class MarketContextConfig:
    """Configuration for the Market Context Module.

    Mirrors the ``market_context`` block of ``forge_config.yaml``.

    Attributes
    ----------
    classifier : str
        Which :class:`RegimeClassifier` implementation to use.  In v1.0 only
        ``"ema_proxy"`` is available; v2.0+ adds ``"hmm"``, ``"kmeans"`` and
        ``"custom"``.
    ema_proxy : EMAProxyConfig
        Parameters for the EMA-proxy classifier.
    labels : list[str]
        Regime labels, ordered from most bearish to most bullish.  These
        stay the same across classifier implementations.
    stable_window : int
        Number of consecutive identical bars required for ``regime_stable``
        to be ``True``.  Used downstream to exclude transition bars from
        regime analysis.
    """

    classifier: str = "ema_proxy"
    ema_proxy: EMAProxyConfig = field(default_factory=EMAProxyConfig)
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    stable_window: int = 12


# ---------------------------------------------------------------------------
# RegimeClassifier interface
# ---------------------------------------------------------------------------

class RegimeClassifier(ABC):
    """Pluggable interface for regime classification.

    Any implementation that respects this contract can be plugged into the
    Market Context Module without changes to downstream modules.  An
    implementation must:

    * classify each bar into one of the configured labels,
    * expose the ordered label list,
    * expose its configuration for traceability in the report.
    """

    @abstractmethod
    def classify(self, kpi_table: pd.DataFrame) -> pd.Series:
        """Classify each bar of the KPI Table.

        Parameters
        ----------
        kpi_table : pd.DataFrame
            The full KPI Table.

        Returns
        -------
        pd.Series
            Ordered categorical labels, index aligned to ``kpi_table``,
            e.g. ``STRONG_BEAR | BEAR | NEUTRAL | BULL | STRONG_BULL``.
        """

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Return the ordered list of possible labels (most bearish first)."""

    @abstractmethod
    def get_config(self) -> dict:
        """Return the configuration used — for traceability in the report."""
