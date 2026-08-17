"""Core data structures and interface for the Market Context module.

The Market Context Module (Modulo 0 of the FORGE pipeline) classifies every
bar of the KPI Table by market regime.  It does not implement the classification
logic itself — it delegates to an object implementing the :class:`RegimeClassifier`
interface.  This is the extensibility point that lets the classifier be swapped
in future versions (HMM, KMeans, custom) without touching any downstream module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import List, Optional

import pandas as pd

from ..unset import UNSET, is_set


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
        Ascending cut points applied to the ``ema_short / ema_long`` ratio when
        ``threshold_mode == "fixed"`` (and the fallback for ``"balanced"``).
        Their number must be exactly ``len(labels) - 1``.  The defaults
        ``[0.975, 0.990, 1.010, 1.025]`` are calibrated empirically on
        crypto 1H data.
    threshold_mode : str
        How the ratio is cut into regimes:

        * ``"fixed"`` (default) — the absolute ``thresholds`` above (e.g. a
          ``STRONG_BULL`` is literally fast EMA > 2.5% above the slow EMA).
        * ``"balanced"`` — the thresholds are recomputed per asset as the
          quantiles of the EMA ratio at the cumulative boundaries of
          ``target_distribution``, so the regime frequencies match that target
          (two populated tails).  The span is left untouched, so its
          mean-reversion meaning is preserved; only the cut adapts.  Falls back
          to ``"fixed"`` if the ratio is degenerate.
    target_distribution : list[float]
        Target regime frequencies for ``threshold_mode == "balanced"``, one per
        label.  Need not sum to 1 (interpreted as relative weights and
        normalised); all entries must be > 0.  Default
        ``[0.10, 0.20, 0.40, 0.20, 0.10]`` (a bell with two 10% tails).
    threshold_basis : str
        How the ``"balanced"`` quantile thresholds are estimated over time:

        * ``"global"`` (default) — the quantiles are computed **once** over the
          whole ratio series.  This hits the target distribution exactly but is
          *not* causal: a bar's label depends on the full sample, including
          future bars (look-ahead).  Consistent with how FORGE calibrates its
          distributional thresholds elsewhere; appropriate for one-shot
          in-sample labelling.
        * ``"expanding"`` — the quantiles at bar *t* are computed over the
          history ``[0..t]`` only, so the labelling is causal (no look-ahead).
          The target distribution then holds only approximately, and the first
          ``threshold_warmup`` bars fall back to the fixed ``thresholds``.
    threshold_warmup : int
        Number of leading bars that use the fixed ``thresholds`` before enough
        history is available for ``threshold_basis == "expanding"`` quantiles.
    window_unit : str
        Unit in which the estimation window/stride (``window_estimation`` and
        ``window_stride``) are expressed:

        * ``"day"`` (default) — measured in calendar days and converted to bars
          using the candle duration.  ``W = 168`` means 168 days on *every*
          timeframe (even 1H), so the estimation horizon — and the derived
          spans — line up in wall-clock terms across timeframes.  Requires a
          DatetimeIndex / datetime column (or an explicit ``bar_hours``) and
          enough history (> ``window_estimation`` days); otherwise the module
          falls back to ``short_period`` / ``long_period``.
        * ``"bar"`` — measured in candles (timeframe-agnostic).  ``W = 168``
          means 168 bars regardless of timeframe: one week on 1H, 168 days on
          1D.  The derived spans are *not* comparable in wall-clock terms across
          timeframes, but no time information is needed.

        The same single value of ``window_estimation`` is simply reinterpreted
        from days to bars by this flag.
    window_estimation : float
        Width of the rolling estimation window, expressed in the unit selected
        by ``window_unit`` (days when ``"day"``, bars when ``"bar"``).
    window_stride : float
        Step between successive estimates, in the same unit as
        ``window_estimation``.
    bar_hours : float or None
        Explicit candle duration in hours, used to convert days↔bars in
        ``"day"`` mode.  When left unset it is inferred from the table's
        DatetimeIndex (or a datetime column).

        Session-resolved from the declared ``timeframe`` when the config goes
        through :func:`forgedge.forge`, which is what stops M0 from answering
        "how long is a bar" on its own while three other modules answer it
        elsewhere (F5, #179).  Standalone use is unchanged: no session, no
        declared timeframe, so the inference still runs — and ``None`` remains
        an accepted way to ask for it explicitly.
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
    threshold_mode: str = "fixed"
    target_distribution: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.40, 0.20, 0.10]
    )
    threshold_basis: str = "global"
    threshold_warmup: int = 200
    window_unit: str = "day"
    window_estimation: float = 168
    window_stride: float = 1
    bar_hours: Optional[float] = UNSET
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

        Session-resolved: the default is **12 hours** of an unchanged regime,
        converted at the session's bar duration and floored at 2 bars — 12 on
        1H (unchanged), 3 on 4H, 2 on 1D, 48 on 15m.  The floor exists because
        ``stable_window=1`` marks every bar stable and the field stops meaning
        anything.  It used to be a flat 12 bars, which on daily candles asked
        for twelve *days* of unchanged regime and left 31.9% of the reference
        fixture's bars stable, against 85.7% at the converted value (F5, #179).
    """

    classifier: str = "ema_proxy"
    ema_proxy: EMAProxyConfig = field(default_factory=EMAProxyConfig)
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    stable_window: int = UNSET

    def resolved(self) -> "MarketContextConfig":
        """Return a copy with every ``UNSET`` field at its documented default.

        ``MarketContext`` is usable standalone — with no session and therefore
        no declared timeframe — so the sentinel has to be spent somewhere.
        Here it falls back to the **hourly** calibration, i.e. exactly the
        value the field carried before it became session-resolved.  Under
        :func:`forgedge.forge` the resolver has already written the converted
        value and this is a no-op.

        ``ema_proxy.bar_hours`` is deliberately *not* filled in: unset there
        means "measure it from the timestamps", which is a real behaviour and
        the right one without a declared timeframe.
        """
        if is_set(self.stable_window):
            return self
        return replace(self, stable_window=12)


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
