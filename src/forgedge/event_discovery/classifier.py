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

        Splits the series into four equal windows and checks whether the
        standard deviation of the window means (relative to the overall series
        std) is below ``scale_free_drift_threshold``.  In case of doubt the
        method returns False (conservative: a false positive would corrupt
        identity thresholds).
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
