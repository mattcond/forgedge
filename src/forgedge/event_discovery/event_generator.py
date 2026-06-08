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
        """Convert one TransformedSeries into a list of RawEvent objects.

        For each threshold in the catalog (computed via ``_compute_thresholds``),
        two event types are created:

        * **Threshold event**: a persistent boolean that is True for every bar
          where the series is above (or below) the threshold.  Produced for all
          transform types.

        * **Crossing event**: a one-bar pulse that fires only on the bar where
          the series *transitions* across the threshold (from the other side).
          Crossing events are only generated for the ``identity`` transform
          because absolute thresholds are meaningless for rolling pctrank/zscore
          (which are re-anchored every bar) and for delta (which oscillates
          around zero).

        Series with fewer than 10 non-null values are skipped to avoid
        degenerate threshold computations.

        Parameters
        ----------
        ts : TransformedSeries
            One output from TransformLayer.transform_one().

        Returns
        -------
        list[RawEvent]
            Between 0 and ``2 * len(thresholds)`` events (the factor of 2
            applies only to identity transforms due to crossing events).
        """
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
        """Wrap a binary column as a single threshold RawEvent.

        Binary columns (exactly two distinct values) are already boolean
        signals — no Transform Layer step is needed.  The higher of the two
        unique values is treated as the "active" state (e.g. value 1 for a
        0/1 flag).

        Parameters
        ----------
        series : pd.Series
            Raw binary column from the KPI table.
        col_name : str
            Column name, used to build the EventComponent.

        Returns
        -------
        list[RawEvent]
            A single-element list, or empty if ``series`` has != 2 distinct
            non-null values.
        """
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
        """One-hot expand a categorical column into N boolean RawEvents.

        Each unique class value becomes a separate event:
        ``is_{col_name}_{class} == True``.  This allows the pipeline to
        discover, for example, that a specific candlestick pattern label
        (e.g. ``shape == 'hammer'``) is a consistent event.

        If the column has more than ``max_classes`` distinct values it is
        skipped entirely (high-cardinality columns are likely identifiers
        rather than signals).

        Special characters in class names (spaces, dots) are sanitised for
        use in column names.

        Parameters
        ----------
        series : pd.Series
            Raw categorical column from the KPI table.
        col_name : str
            Column name used to build the EventComponent.
        max_classes : int
            Maximum number of unique values before the column is discarded.

        Returns
        -------
        list[RawEvent]
            One RawEvent per distinct class value, or empty list if the
            column exceeds ``max_classes``.
        """
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
        """Build the threshold catalog for one transformed series.

        Two threshold strategies are used depending on whether the series
        comes from a z-score transform:

        **Distributional thresholds** (non-z-score):
        Quantiles of the full series are computed at the levels defined in
        ``DIST_QUANTILES_LOW`` and ``DIST_QUANTILES_HIGH``.  This approach
        is data-driven and adapts to the empirical distribution; it works
        well for percentile-rank and delta series where the distribution
        shape varies.

        **Theoretical z-score thresholds** (z-score transforms):
        Fixed values from ``ZSCORE_THRESHOLDS_LOW`` and
        ``ZSCORE_THRESHOLDS_HIGH`` (±1.0, ±1.5, ±2.0 sigma) are used.
        These are parameter-free and comparable across different features,
        making them appropriate when the series is already standardised.

        Parameters
        ----------
        series : pd.Series
            Already dropna'd non-null values for quantile computation.
        is_zscore : bool
            True if the series comes from ``rolling_zscore`` transform.

        Returns
        -------
        list[tuple[float, str, str]]
            Each tuple is ``(threshold_value, label, direction)`` where
            direction is ``"below"`` or ``"above"``.
        """
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
    """Build a persistent boolean series from a threshold condition.

    Returns a float series (0.0 / 1.0 / NaN) where 1.0 indicates
    the threshold condition is active at that bar, and NaN propagates
    wherever the input is NaN.

    Parameters
    ----------
    series : pd.Series
        Input numeric series (may contain NaN).
    threshold : float
        Numerical threshold value.
    direction : str
        ``"below"`` → event active when ``series < threshold``;
        ``"above"`` → event active when ``series > threshold``.

    Returns
    -------
    pd.Series
        Float series with values in {0.0, 1.0, NaN}.
    """
    if direction == "below":
        return (series < threshold).astype(float).where(series.notna(), other=float("nan"))
    return (series > threshold).astype(float).where(series.notna(), other=float("nan"))


def _apply_crossing(
    series: pd.Series, threshold: float, direction: str
) -> pd.Series:
    """Build a one-bar pulse series that fires only on threshold crossings.

    A crossing is detected when the series moves from one side of the
    threshold to the other between consecutive bars:

    * direction ``"below"``: fires at bar *t* when ``series[t] < threshold``
      AND ``series[t-1] >= threshold`` (downward crossing).
    * direction ``"above"``: fires at bar *t* when ``series[t] > threshold``
      AND ``series[t-1] <= threshold`` (upward crossing).

    NaN is propagated when either the current or previous bar is NaN.

    This is more selective than a threshold event — it fires only at the
    moment of entry, not throughout the duration of the condition.

    Parameters
    ----------
    series : pd.Series
        Input numeric series (may contain NaN).
    threshold : float
        Threshold to cross.
    direction : str
        ``"below"`` for downward crossings, ``"above"`` for upward.

    Returns
    -------
    pd.Series
        Float series with values in {0.0, 1.0, NaN}.
    """
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
    """Build a human-readable threshold expression string.

    Example: ``"pr_close_rsi_25_96 > 0.85"``

    Parameters
    ----------
    col : str
        Transformed column name.
    direction : str
        ``"below"`` or ``"above"``.
    threshold : float
        Threshold value.

    Returns
    -------
    str
    """
    op = "<" if direction == "below" else ">"
    return f"{col} {op} {threshold:.6g}"


def _make_crossing_expr(col: str, direction: str, threshold: float) -> str:
    """Build a human-readable crossing expression string.

    Example: ``"close_rsi_25 crosses_above 70.0"``

    Parameters
    ----------
    col : str
        Transformed column name.
    direction : str
        ``"below"`` or ``"above"``.
    threshold : float
        Threshold value.

    Returns
    -------
    str
    """
    direction_word = "crosses_below" if direction == "below" else "crosses_above"
    return f"{col} {direction_word} {threshold:.6g}"
