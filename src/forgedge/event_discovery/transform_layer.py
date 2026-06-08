"""Step 2 — Transform Layer.

Applies four temporal transformations to every feature in the extended catalog:
  Identity    — raw value (only for scale-free features)
  Pctrank     — rolling percentile rank  (windows: 48, 96, 168)
  Z-score     — rolling standardisation  (windows: 48, 96, 168)
  Delta       — lag difference            (lags:    1, 3, 6, 12)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PCTRANK_WINDOWS: list[int] = [48, 96, 168]
ZSCORE_WINDOWS: list[int] = [48, 96, 168]
DELTA_LAGS: list[int] = [1, 3, 6, 12]


@dataclass
class TransformedSeries:
    """Container for one transformed version of a feature.

    Produced by TransformLayer and consumed by EventGenerator.  Each
    TransformedSeries gives rise to a set of threshold events (and crossing
    events for identity transforms).

    Attributes
    ----------
    col : str
        Name for this transformed column (e.g. ``pr_close_rsi_25_96``).
    series : pd.Series
        The transformed numeric values, aligned to the original DataFrame index.
    transform : str
        Transform identifier: ``"identity"``, ``"rolling_pctrank"``,
        ``"rolling_zscore"``, or ``"delta"``.
    transform_params : dict
        Parameters used (e.g. ``{"window": 96}`` or ``{"lag": 3}``).
        Empty dict for the identity transform.
    source_feature : str
        Name of the FeatureGenerator output column this was computed from.
    is_zscore : bool
        True only for ``rolling_zscore`` transforms.  When True, the
        EventGenerator uses fixed theoretical z-score thresholds (±1.0,
        ±1.5, ±2.0) instead of empirical quantile thresholds.
    """

    col: str
    series: pd.Series
    transform: str          # "identity" | "rolling_pctrank" | "rolling_zscore" | "delta"
    transform_params: dict
    source_feature: str
    is_zscore: bool = False


class TransformLayer:
    """Applies all transforms to a set of features and returns the results."""

    def transform_all(
        self,
        df: pd.DataFrame,
        feature_meta: dict,   # col → DerivedFeature (from FeatureGenerator)
    ) -> list[TransformedSeries]:
        """Apply all transforms to every feature in ``feature_meta``.

        Iterates over the derived feature catalog and calls ``transform_one``
        for each column that exists in ``df``.  Columns that are in
        ``feature_meta`` but missing from ``df`` are silently skipped (this
        can happen if a derived column failed to be created due to all-NaN
        inputs).

        Parameters
        ----------
        df : pd.DataFrame
            Extended DataFrame produced by FeatureGenerator.
        feature_meta : dict
            Mapping col → DerivedFeature from FeatureGenerator.generate().

        Returns
        -------
        list[TransformedSeries]
            Flat list of all transformed versions of all features.
            Length = sum over features of (1 if scale_free else 0) +
            len(PCTRANK_WINDOWS) + len(ZSCORE_WINDOWS) + len(DELTA_LAGS).
        """
        results: list[TransformedSeries] = []
        for col, meta in feature_meta.items():
            if col not in df.columns:
                continue
            series = df[col]
            results.extend(self.transform_one(series, col, meta.is_scale_free))
        return results

    def transform_one(
        self,
        series: pd.Series,
        col: str,
        is_scale_free: bool,
    ) -> list[TransformedSeries]:
        """Apply all four transforms to a single feature series.

        Transform logic
        ---------------
        **Identity**: Emitted only when ``is_scale_free=True``.  The raw
        values are passed through unchanged.  Thresholds are then set at
        empirical quantiles of the series itself, which only makes sense
        when the series has a stable level (scale-free property).

        **Rolling Pctrank**: For each window in ``PCTRANK_WINDOWS`` (48, 96,
        168 bars) the value is ranked within the rolling window and expressed
        as a percentile in [0, 1].  Output is scale-free regardless of the
        source feature.  Suitable for detecting extreme values relative to
        recent history.

        **Rolling Z-score**: For each window in ``ZSCORE_WINDOWS`` (48, 96,
        168 bars) the rolling mean is subtracted and divided by the rolling
        standard deviation.  NaN is returned when std == 0 (flat window).
        The ``is_zscore=True`` flag signals the EventGenerator to use fixed
        theoretical thresholds (±1, ±1.5, ±2 sigma) rather than quantiles.

        **Delta**: First-difference over each lag in ``DELTA_LAGS``
        (1, 3, 6, 12 bars).  Captures momentum / rate-of-change.  Always
        computed regardless of scale-free status.

        Parameters
        ----------
        series : pd.Series
            Numeric values for the feature column.
        col : str
            Name of the source feature (used to construct output column names).
        is_scale_free : bool
            Whether to include the identity transform.

        Returns
        -------
        list[TransformedSeries]
            Between 10 (non-scale-free) and 11 (scale-free) TransformedSeries
            objects per feature.
        """
        out: list[TransformedSeries] = []

        # Identity — only for scale-free series
        if is_scale_free:
            out.append(TransformedSeries(
                col=col,
                series=series,
                transform="identity",
                transform_params={},
                source_feature=col,
                is_zscore=False,
            ))

        # Pctrank
        for w in PCTRANK_WINDOWS:
            t_col = f"pr_{col}_{w}"
            t_series = _rolling_pctrank(series, w)
            out.append(TransformedSeries(
                col=t_col,
                series=t_series,
                transform="rolling_pctrank",
                transform_params={"window": w},
                source_feature=col,
                is_zscore=False,
            ))

        # Z-score
        for w in ZSCORE_WINDOWS:
            t_col = f"zs_{col}_{w}"
            t_series = _rolling_zscore(series, w)
            out.append(TransformedSeries(
                col=t_col,
                series=t_series,
                transform="rolling_zscore",
                transform_params={"window": w},
                source_feature=col,
                is_zscore=True,
            ))

        # Delta
        for lag in DELTA_LAGS:
            t_col = f"delta_{col}_{lag}"
            t_series = series.diff(lag)
            out.append(TransformedSeries(
                col=t_col,
                series=t_series,
                transform="delta",
                transform_params={"lag": lag},
                source_feature=col,
                is_zscore=False,
            ))

        return out


# ---------------------------------------------------------------------------
# Rolling computation helpers
# ---------------------------------------------------------------------------

def _rolling_pctrank(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling percentile rank in [0, 1].

    Uses ``min_periods = max(2, window // 2)`` so that values near the start
    of the series are not discarded entirely, while still requiring at least
    half the window to be filled before emitting a rank.

    Parameters
    ----------
    series : pd.Series
        Input numeric series (NaNs are allowed).
    window : int
        Rolling window size in bars.

    Returns
    -------
    pd.Series
        Values in [0, 1]; NaN where fewer than ``min_periods`` non-null
        observations are available.
    """
    min_p = max(2, window // 2)
    return series.rolling(window, min_periods=min_p).rank(pct=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score: ``(x - mean) / std`` over a trailing window.

    Uses ``min_periods = max(2, window // 2)`` for the same reason as
    ``_rolling_pctrank``.  Bars where the rolling standard deviation is
    zero (flat market) are set to NaN via ``.where(std > 0)`` to avoid
    propagating +inf or 0/0=NaN artefacts.

    Parameters
    ----------
    series : pd.Series
        Input numeric series.
    window : int
        Rolling window size in bars.

    Returns
    -------
    pd.Series
        Z-score values; NaN where std == 0 or insufficient data.
    """
    min_p = max(2, window // 2)
    roll = series.rolling(window, min_periods=min_p)
    mean = roll.mean()
    std = roll.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mean) / std
    return z.where(std > 0)
