"""Step 3 — Event Generation.

Converts continuous transformed series into boolean event series by applying
threshold and crossing conditions.  Also handles binary and categorical columns
that bypass Steps 1-2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import EventComponent, RawEvent
from .transform_layer import TransformedSeries

# ---------------------------------------------------------------------------
# Threshold catalogs
# ---------------------------------------------------------------------------

DIST_QUANTILES_LOW: list[float] = [0.03, 0.05, 0.08, 0.10, 0.15]
DIST_QUANTILES_HIGH: list[float] = [0.85, 0.90, 0.92, 0.95, 0.97]

ZSCORE_THRESHOLDS_LOW: list[float] = [-2.0, -1.5, -1.0]
ZSCORE_THRESHOLDS_HIGH: list[float] = [1.0, 1.5, 2.0]


class EventGenerator:
    """Generates raw boolean events from transformed series and native columns."""

    def generate_from_transformed(
        self, ts: TransformedSeries
    ) -> list[RawEvent]:
        """Generate threshold and crossing events for one TransformedSeries."""
        series = ts.series.dropna()
        if len(series) < 10:
            return []

        thresholds = self._compute_thresholds(series, ts.is_zscore)
        events: list[RawEvent] = []

        for threshold, t_type, direction in thresholds:
            # Threshold event
            bool_series = _apply_threshold(ts.series, threshold, direction)
            comp = EventComponent(
                source_feature=ts.source_feature,
                transform=ts.transform,
                transform_params=ts.transform_params,
                transformed_col=ts.col,
                threshold=threshold,
                threshold_type=t_type,
                direction=direction,
                event_type="threshold",
                expression=_make_expr(ts.col, direction, threshold),
            )
            events.append(RawEvent(series=bool_series, component=comp))

            # Crossing event — only makes sense for identity transform (absolute thresholds)
            if ts.transform == "identity":
                cross_series = _apply_crossing(ts.series, threshold, direction)
                cross_comp = EventComponent(
                    source_feature=ts.source_feature,
                    transform=ts.transform,
                    transform_params=ts.transform_params,
                    transformed_col=ts.col,
                    threshold=threshold,
                    threshold_type=t_type,
                    direction=direction,
                    event_type="crossing",
                    expression=_make_crossing_expr(ts.col, direction, threshold),
                )
                events.append(RawEvent(series=cross_series, component=cross_comp))

        return events

    def generate_from_binary(
        self, series: pd.Series, col_name: str
    ) -> list[RawEvent]:
        """A binary column is already a boolean event — no thresholds needed."""
        vals = sorted(series.dropna().unique())
        if len(vals) != 2:
            return []
        high_val = vals[1]
        bool_series = (series == high_val).astype(float)
        bool_series[series.isna()] = float("nan")
        comp = EventComponent(
            source_feature=col_name,
            transform="binary_native",
            transform_params={},
            transformed_col=col_name,
            threshold=float(high_val),
            threshold_type="binary_native",
            direction="above",
            event_type="threshold",
            expression=f"{col_name} == {high_val}",
        )
        return [RawEvent(series=bool_series, component=comp)]

    def generate_from_categorical(
        self, series: pd.Series, col_name: str, max_classes: int = 20
    ) -> list[RawEvent]:
        """One-hot expand a categorical column into N boolean events."""
        classes = series.dropna().unique()
        if len(classes) > max_classes:
            return []
        events: list[RawEvent] = []
        for cls in classes:
            bool_series = (series == cls).astype(float)
            bool_series[series.isna()] = float("nan")
            safe_cls = str(cls).replace(" ", "_").replace(".", "")
            derived_col = f"is_{col_name}_{safe_cls}"
            comp = EventComponent(
                source_feature=col_name,
                transform="categorical_onehot",
                transform_params={"class": str(cls)},
                transformed_col=derived_col,
                threshold=1.0,
                threshold_type="categorical_onehot",
                direction="above",
                event_type="threshold",
                expression=f"{col_name} == '{cls}'",
            )
            events.append(RawEvent(series=bool_series, component=comp))
        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_thresholds(
        self, series: pd.Series, is_zscore: bool
    ) -> list[tuple[float, str, str]]:
        """Return list of (threshold_value, threshold_type_label, direction)."""
        thresholds: list[tuple[float, str, str]] = []

        if is_zscore:
            for z in ZSCORE_THRESHOLDS_LOW:
                thresholds.append((z, f"theoretical_z{z:.1f}", "below"))
            for z in ZSCORE_THRESHOLDS_HIGH:
                thresholds.append((z, f"theoretical_z{z:.1f}", "above"))
        else:
            for q in DIST_QUANTILES_LOW:
                val = float(series.quantile(q))
                label = f"distributional_p{int(round(q * 100)):02d}"
                thresholds.append((val, label, "below"))
            for q in DIST_QUANTILES_HIGH:
                val = float(series.quantile(q))
                label = f"distributional_p{int(round(q * 100)):02d}"
                thresholds.append((val, label, "above"))

        return thresholds


# ---------------------------------------------------------------------------
# Boolean series constructors
# ---------------------------------------------------------------------------

def _apply_threshold(
    series: pd.Series, threshold: float, direction: str
) -> pd.Series:
    if direction == "below":
        return (series < threshold).astype(float).where(series.notna(), other=float("nan"))
    return (series > threshold).astype(float).where(series.notna(), other=float("nan"))


def _apply_crossing(
    series: pd.Series, threshold: float, direction: str
) -> pd.Series:
    """True only at the bar where the series first crosses the threshold."""
    if direction == "below":
        current = series < threshold
        previous = series.shift(1) >= threshold
    else:
        current = series > threshold
        previous = series.shift(1) <= threshold
    cross = (current & previous).astype(float)
    cross[series.isna() | series.shift(1).isna()] = float("nan")
    return cross


# ---------------------------------------------------------------------------
# Expression string builders
# ---------------------------------------------------------------------------

def _make_expr(col: str, direction: str, threshold: float) -> str:
    op = "<" if direction == "below" else ">"
    return f"{col} {op} {threshold:.6g}"


def _make_crossing_expr(col: str, direction: str, threshold: float) -> str:
    direction_word = "crosses_below" if direction == "below" else "crosses_above"
    return f"{col} {direction_word} {threshold:.6g}"
