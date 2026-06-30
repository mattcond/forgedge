"""Step 0 — Type classification and scale-free detection."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .models import ColumnClassification, ColumnType

_SKIP_COLS = {
    "open_dt", "close_time", "timestamp", "date", "datetime",
    "regime", "regime_stable", "open", "high", "low", "volume",
}


class TypeClassifier:
    """Classifies each DataFrame column as continuous, binary, or categorical.

    Continuous columns are further assessed for scale-free behaviour.
    The scale-free check is conservative: when in doubt, returns False.
    Users may override via ``scale_free_overrides``.

    Parameters
    ----------
    max_categorical_classes : int
        Columns classified as CATEGORICAL with more distinct values than this
        limit are still stored in the classification dict but are excluded from
        the event generation pipeline.
    scale_free_overrides : dict[str, bool] or None
        Manual overrides for specific columns.  Useful when the automatic
        heuristic produces a false negative (e.g. a short history for RSI).
    skip_cols : set[str] or None
        Additional column names to skip, merged with the built-in ``_SKIP_COLS``
        set (datetime, OHLCV raw columns, regime labels).
    support_overlap_threshold : float
        Minimum support overlap for the scale-free heuristic.  A column passes
        when the robust support (``[q05, q95]``) of its first half overlaps
        with that of its second half by at least this fraction (intersection /
        union).  Higher values are stricter; default 0.5.
    scale_free_drift_threshold : float, optional
        **Deprecated** and ignored.  The scale-free heuristic no longer uses a
        windowed drift ratio (see issue #136).  Passing this argument emits a
        ``DeprecationWarning`` and has no effect.
    """

    def __init__(
        self,
        max_categorical_classes: int = 20,
        scale_free_overrides: Optional[dict[str, bool]] = None,
        skip_cols: Optional[set[str]] = None,
        support_overlap_threshold: float = 0.5,
        scale_free_drift_threshold: Optional[float] = None,
    ):
        if scale_free_drift_threshold is not None:
            warnings.warn(
                "scale_free_drift_threshold is deprecated and ignored; the "
                "scale-free heuristic now uses a support-overlap test "
                "(support_overlap_threshold). See issue #136.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.max_categorical_classes = max_categorical_classes
        self.scale_free_overrides = scale_free_overrides or {}
        self.skip_cols = (skip_cols or set()) | _SKIP_COLS
        self.support_overlap_threshold = support_overlap_threshold

    def fit(self, df: pd.DataFrame) -> dict[str, ColumnClassification]:
        """Classify every column in ``df`` and return the results dict.

        Columns listed in ``skip_cols`` and all-NaN columns are silently
        ignored.  The returned dict maps column name → ColumnClassification
        and is passed directly to FeatureGenerator and EventGenerator.

        Parameters
        ----------
        df : pd.DataFrame
            The raw KPI table.  Must include a datetime column (skipped
            automatically via ``_SKIP_COLS``).

        Returns
        -------
        dict[str, ColumnClassification]
            One entry per non-skipped, non-empty column.
        """
        results: dict[str, ColumnClassification] = {}
        for col in df.columns:
            if col in self.skip_cols:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if series.nunique() <= 1:
                continue
            results[col] = self._classify(col, series)
        return results

    def _classify(self, col: str, series: pd.Series) -> ColumnClassification:
        """Determine the type of a single non-null series.

        Decision tree:
        1. Non-numeric dtype → CATEGORICAL (strings, objects).
        2. Exactly 2 distinct values → BINARY (boolean flags, 0/1 indicators).
        3. Otherwise → CONTINUOUS; run scale-free detection and apply overrides.

        Parameters
        ----------
        col : str
            Column name (used to look up any user override).
        series : pd.Series
            Already dropna'd values for the column.

        Returns
        -------
        ColumnClassification
        """
        n_distinct = int(series.nunique())

        if not pd.api.types.is_numeric_dtype(series):
            return ColumnClassification(
                col_name=col,
                col_type=ColumnType.CATEGORICAL,
                n_distinct=n_distinct,
            )

        if n_distinct == 2:
            return ColumnClassification(
                col_name=col,
                col_type=ColumnType.BINARY,
                n_distinct=n_distinct,
            )

        is_sf = self._is_scale_free(series)
        override = self.scale_free_overrides.get(col)
        return ColumnClassification(
            col_name=col,
            col_type=ColumnType.CONTINUOUS,
            n_distinct=n_distinct,
            is_scale_free=is_sf,
            scale_free_override=override,
        )

    def _is_scale_free(self, series: pd.Series) -> bool:
        """Return True only if the series support is stationary (period-invariant).

        The heuristic splits the series into two halves and measures how much
        their robust supports (``[q05, q95]``) overlap::

            lo1, hi1 = q05(first_half),  q95(first_half)
            lo2, hi2 = q05(second_half), q95(second_half)
            overlap = intersection([lo1,hi1], [lo2,hi2])
                      / union([lo1,hi1], [lo2,hi2])
            scale_free = overlap >= support_overlap_threshold

        A series is scale-free when the two halves revisit the same value
        range (``overlap >= threshold``, default 0.5).

        Why support overlap, not windowed drift
        ---------------------------------------
        A bounded, stationary oscillator (RSI, Stochastic, %B) keeps visiting
        the same support throughout the sample, so the two halves overlap
        almost completely (overlap ≈ 0.94) regardless of the indicator period
        — RSI14 and RSI25 score the same.  A trending or random-walk series
        (price, EMA) makes new extremes in its second half that the first half
        never reached, so the supports barely overlap (overlap ≈ 0.16).

        This fixes two earlier defects:

        * The original mean-drift test produced a false negative on RSI/%B
          because their *mean* rises with an uptrend even though their domain
          is fixed (issue #133).
        * The four-window quantile-drift test (issue #133's fix) was not
          period-invariant: a slower oscillator (RSI25) drags the regime
          through its windows longer and drifted just over the threshold,
          giving a false negative (issue #136).  Support overlap is computed
          once over the whole series and is regime- and period-invariant.

        Design rationale
        ----------------
        The check stays conservative.  A false positive (declaring a trending
        price series scale-free) would corrupt the identity-transform
        thresholds by anchoring them to a level that no longer holds out-of-
        sample.  A false negative only costs some identity events — far less
        damaging.

        Minimum data requirements
        -------------------------
        * ``n >= 48`` total non-null observations (each half ≥ 24 bars).

        Columns with a degenerate support (``union == 0``) always return False
        because they carry no level information.

        Parameters
        ----------
        series : pd.Series
            Numeric series (NaNs already dropped by the caller).

        Returns
        -------
        bool
            True if the two halves' supports overlap by at least the
            threshold; False otherwise or when there is insufficient data.
        """
        series = series.dropna()
        n = len(series)
        if n < 48:
            return False

        h = n // 2
        first = series.iloc[:h]
        second = series.iloc[h:]

        lo1, hi1 = float(first.quantile(0.05)), float(first.quantile(0.95))
        lo2, hi2 = float(second.quantile(0.05)), float(second.quantile(0.95))

        intersection = max(0.0, min(hi1, hi2) - max(lo1, lo2))
        union = max(hi1, hi2) - min(lo1, lo2)
        if union == 0:
            return False

        overlap = intersection / union
        return overlap >= self.support_overlap_threshold
