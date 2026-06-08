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
        Maximum allowed drift ratio for the scale-free heuristic.  A column
        passes if ``std(window_means) / overall_std <= threshold``.
        Lower values are stricter; default 0.25 is conservative.
    """

    def __init__(
        self,
        max_categorical_classes: int = 20,
        scale_free_overrides: Optional[dict[str, bool]] = None,
        skip_cols: Optional[set[str]] = None,
        scale_free_drift_threshold: float = 0.25,
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

        if n_distinct <= 2:
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
        """Return True only if the series level is stable across rolling windows.

        The heuristic splits the series into four equal-length windows and
        measures how much the window means drift relative to the overall
        series standard deviation::

            drift_ratio = std(window_means) / overall_std

        A series is considered scale-free when ``drift_ratio <=
        scale_free_drift_threshold`` (default 0.25).

        Design rationale
        ----------------
        The check is deliberately conservative.  A false positive (declaring
        a trending price series scale-free) would corrupt the identity-transform
        thresholds by anchoring them to a level that no longer holds out-of-
        sample.  A false negative (missing that RSI is scale-free) only costs
        some identity events — far less damaging.

        Minimum data requirements
        -------------------------
        * ``n >= 48`` total non-null observations.
        * Each window must be at least 12 bars (``w = n // 4 >= 12``).

        Columns with zero standard deviation (constant series) always return
        False because they carry no information.

        Parameters
        ----------
        series : pd.Series
            Numeric series (NaNs already dropped by the caller).

        Returns
        -------
        bool
            True if the drift ratio is within the threshold; False otherwise
            or when there is insufficient data.
        """
        series = series.dropna()
        n = len(series)
        if n < 48:
            return False

        n_windows = 4
        w = n // n_windows
        if w < 12:
            return False

        overall_std = float(series.std())
        if overall_std == 0:
            return False

        window_means = np.array(
            [series.iloc[i * w: (i + 1) * w].mean() for i in range(n_windows)],
            dtype=float,
        )
        drift_ratio = float(np.std(window_means)) / overall_std
        return drift_ratio <= self.scale_free_drift_threshold
