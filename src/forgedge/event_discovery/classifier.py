"""Step 0 — Type classification and scale-free detection."""
from __future__ import annotations

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
    scale_free_drift_threshold : float
        Maximum allowed support-drift ratio for the scale-free heuristic.  A
        column passes if the drift of its extreme quantiles (q10, q90) across
        rolling windows, normalised by the global inter-quantile range, stays
        ``<= threshold``.  Lower values are stricter; default 0.20.
    """

    def __init__(
        self,
        max_categorical_classes: int = 20,
        scale_free_overrides: Optional[dict[str, bool]] = None,
        skip_cols: Optional[set[str]] = None,
        scale_free_drift_threshold: float = 0.20,
    ):
        self.max_categorical_classes = max_categorical_classes
        self.scale_free_overrides = scale_free_overrides or {}
        self.skip_cols = (skip_cols or set()) | _SKIP_COLS
        self.scale_free_drift_threshold = scale_free_drift_threshold

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
        """Return True only if the series *support* is stable across rolling windows.

        The heuristic splits the series into four equal-length windows and
        measures how much the extreme quantiles (q10 and q90) drift between
        windows, normalised by the global inter-quantile range::

            range_global = q90(series) - q10(series)
            drift_lo = std(q10 per window) / range_global
            drift_hi = std(q90 per window) / range_global
            scale_free = max(drift_lo, drift_hi) <= threshold

        A series is considered scale-free when the worst of the two drifts is
        within ``scale_free_drift_threshold`` (default 0.20).

        Why support-stability, not mean-drift
        -------------------------------------
        The previous heuristic measured the drift of the *window mean*
        relative to the overall standard deviation.  On a trending market the
        mean of a bounded oscillator (RSI, %B) moves with the regime — RSI is
        higher on average in an uptrend — so the mean-drift test produced a
        systematic false negative on RSI and %B even though their domain is
        fixed (issue #133).  The extreme quantiles of a bounded oscillator are
        regime-invariant (q10 of RSI stays around ~30, q90 around ~70), so the
        support-stability test classifies them correctly as scale-free while
        still rejecting trending price/EMA series whose quantiles drift.

        Design rationale
        ----------------
        The check stays conservative.  A false positive (declaring a trending
        price series scale-free) would corrupt the identity-transform
        thresholds by anchoring them to a level that no longer holds out-of-
        sample.  A false negative only costs some identity events — far less
        damaging.

        Minimum data requirements
        -------------------------
        * ``n >= 48`` total non-null observations.
        * Each window must be at least 12 bars (``w = n // 4 >= 12``).

        Columns with a degenerate support (``range_global == 0``) always return
        False because they carry no level information.

        Parameters
        ----------
        series : pd.Series
            Numeric series (NaNs already dropped by the caller).

        Returns
        -------
        bool
            True if the worst support-drift is within the threshold; False
            otherwise or when there is insufficient data.
        """
        series = series.dropna()
        n = len(series)
        if n < 48:
            return False

        n_windows = 4
        w = n // n_windows
        if w < 12:
            return False

        range_global = float(series.quantile(0.90) - series.quantile(0.10))
        if range_global == 0:
            return False

        q_lo = np.array(
            [series.iloc[i * w: (i + 1) * w].quantile(0.10) for i in range(n_windows)],
            dtype=float,
        )
        q_hi = np.array(
            [series.iloc[i * w: (i + 1) * w].quantile(0.90) for i in range(n_windows)],
            dtype=float,
        )
        drift_lo = float(np.std(q_lo, ddof=1)) / range_global
        drift_hi = float(np.std(q_hi, ddof=1)) / range_global
        return max(drift_lo, drift_hi) <= self.scale_free_drift_threshold
