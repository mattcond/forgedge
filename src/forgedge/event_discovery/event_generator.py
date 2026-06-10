"""Step 3 — Event Generation.

Converts continuous transformed series into boolean event series by applying
threshold and crossing conditions.  Also handles binary and categorical columns
that bypass Steps 1-2.
"""
from __future__ import annotations

from decimal import Decimal
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
        self, ts: TransformedSeries, ts_col: str = "open_dt"
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
        ts_col : str
            Name of the timestamp/ordering column passed through to the SQL
            expression builder.  Must match ``DiscoveryConfig.timestamp_col``
            of the originating pipeline.  Defaults to ``"open_dt"``.

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

        # Merge feature-level params (e.g. diffnorm_std) into transform_params
        # so that EventComponent.transform_params is self-contained for replay.
        merged_params = {**ts.transform_params, **ts.feature_params}

        for threshold, t_type, direction in thresholds:
            # Threshold event
            bool_series = _apply_threshold(ts.series, threshold, direction)
            comp = EventComponent(
                source_feature=ts.source_feature,
                transform=ts.transform,
                transform_params=merged_params,
                transformed_col=ts.col,
                threshold=threshold,
                threshold_type=t_type,
                direction=direction,
                event_type="threshold",
                expression=_make_expr(ts.col, direction, threshold),
                source_cols=ts.source_cols,
                event_formula=_build_event_formula(
                    ts.source_feature, ts.source_cols, ts.transform,
                    merged_params, threshold, direction, "threshold",
                ),
                sql_expression=_build_sql_expression(
                    ts.source_feature, ts.source_cols, ts.transform,
                    merged_params, threshold, direction, "threshold",
                    ts_col=ts_col,
                ),
            )
            events.append(RawEvent(series=bool_series, component=comp))

            # Crossing event — only for identity transform on low-direction thresholds.
            # The spec defines only the downward crossing formula (serie_t < soglia AND
            # serie_{t-1} >= soglia).  High-direction crossings are not meaningful for
            # the oversold/extreme-low detection use case and are omitted.
            if ts.transform == "identity" and direction == "below":
                cross_series = _apply_crossing(ts.series, threshold, direction)
                cross_comp = EventComponent(
                    source_feature=ts.source_feature,
                    transform=ts.transform,
                    transform_params=merged_params,
                    transformed_col=ts.col,
                    threshold=threshold,
                    threshold_type=t_type,
                    direction=direction,
                    event_type="crossing",
                    expression=_make_crossing_expr(ts.col, direction, threshold),
                    source_cols=ts.source_cols,
                    event_formula=_build_event_formula(
                        ts.source_feature, ts.source_cols, ts.transform,
                        merged_params, threshold, direction, "crossing",
                    ),
                    sql_expression=_build_sql_expression(
                        ts.source_feature, ts.source_cols, ts.transform,
                        merged_params, threshold, direction, "crossing",
                        ts_col=ts_col,
                    ),
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
            event_formula=f"{col_name} = {high_val}",
            sql_expression=f'"{col_name}" = {high_val}',
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
                event_formula=f"{col_name} = '{cls}'",
                sql_expression=f'"{col_name}" = \'{cls}\'',
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


# ---------------------------------------------------------------------------
# SQL expression builders (DuckDB-compatible)
# ---------------------------------------------------------------------------

def _build_sql_expression(
    source_feature: str,
    source_cols: list,
    transform: str,
    transform_params: dict,
    threshold: float,
    direction: str,
    event_type: str,
    ts_col: str = "open_dt",
) -> str:
    """Build a DuckDB-compatible SQL boolean expression for one event component.

    Parameters
    ----------
    source_feature : str
        Name of the derived feature (e.g. ``"close_rsi_25"`` or
        ``"diffnorm_close_rsi14_rsi25"``).
    source_cols : list
        Original native columns used to build the feature (empty for arity-1).
    transform : str
        Transform identifier (``"identity"``, ``"rolling_pctrank"``,
        ``"rolling_zscore"``, ``"delta"``).
    transform_params : dict
        Transform parameters including ``window``, ``lag``, and
        ``diffnorm_std`` when applicable.
    threshold : float
        Numerical threshold for the boolean condition.
    direction : str
        ``"below"`` or ``"above"``.
    event_type : str
        ``"threshold"`` or ``"crossing"``.
    ts_col : str
        Name of the timestamp/ordering column in the target table.
        Defaults to ``"open_dt"``.

    Returns
    -------
    str
        SQL boolean expression string, valid as a SELECT or WHERE clause
        when used against a table with the same schema as
        ``EventDiscovery.df``.
    """
    feat_sql = _sql_feature(source_feature, source_cols, transform_params)
    t_sql = _sql_transform(feat_sql, transform, transform_params, ts_col)
    return _sql_condition(t_sql, threshold, direction, event_type, ts_col)


def _sql_feature(source_feature: str, source_cols: list, transform_params: dict) -> str:
    """Return SQL expression for the raw (pre-transform) feature value."""
    sf = source_feature
    sc = source_cols
    if sf.startswith("ratio_") and len(sc) == 2:
        return f'"{sc[0]}" / NULLIF("{sc[1]}", 0)'
    if sf.startswith("spread_") and len(sc) == 2:
        return f'("{sc[0]}" - "{sc[1]}") / NULLIF("{sc[1]}", 0)'
    if sf.startswith("diffnorm_") and len(sc) == 2:
        std = transform_params.get("diffnorm_std", 1.0)
        return f'("{sc[0]}" - "{sc[1]}") / {std:.6g}'
    if sf.startswith("bb_pct_b_") and len(sc) == 3:
        val, lower, upper = sc
        return f'("{val}" - "{lower}") / NULLIF("{upper}" - "{lower}", 0)'
    if sf.startswith("pos_") and len(sc) == 3:
        val, mn, mx = sc
        return f'("{val}" - "{mn}") / NULLIF("{mx}" - "{mn}", 0)'
    return f'"{sf}"'


def _sql_transform(feat_sql: str, transform: str, transform_params: dict, ts_col: str) -> str:
    """Wrap a feature SQL expression with the temporal transform.

    The rolling transforms reproduce ``min_periods = max(2, window // 2)``
    from :func:`TransformLayer` via a ``CASE WHEN COUNT(*) OVER win >= min_p``
    guard, so warmup bars evaluate to NULL exactly as they do in pandas.
    """
    if transform == "identity":
        return feat_sql
    if transform == "rolling_pctrank":
        w = transform_params["window"]
        min_p = max(2, w // 2)
        win = f"ORDER BY {ts_col} ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        # Non-null count drives both the min_periods guard and the denominator,
        # matching pandas which ignores NaN in rolling().rank(pct=True).
        cnt = f"COUNT({feat_sql}) OVER ({win})"
        arr = f"list({feat_sql}) OVER ({win})"
        # Average-method rank (pandas default): values strictly below plus half
        # of the tied group.  avg_rank = (#less) + (#equal + 1) / 2, then / count.
        less = f"length(list_filter({arr}, v -> v < {feat_sql}))"
        eq = f"length(list_filter({arr}, v -> v = {feat_sql}))"
        pct = f"({less} + ({eq} + 1) / 2.0)::DOUBLE / NULLIF({cnt}, 0)"
        return f"CASE WHEN {cnt} >= {min_p} AND {feat_sql} IS NOT NULL THEN {pct} END"
    if transform == "rolling_zscore":
        w = transform_params["window"]
        min_p = max(2, w // 2)
        win = f"ORDER BY {ts_col} ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        cnt = f"COUNT({feat_sql}) OVER ({win})"
        avg = f"AVG({feat_sql}) OVER ({win})"
        std = f"STDDEV_SAMP({feat_sql}) OVER ({win})"
        z = f"({feat_sql} - {avg}) / NULLIF({std}, 0)"
        return f"CASE WHEN {cnt} >= {min_p} THEN {z} END"
    if transform == "delta":
        lag = transform_params["lag"]
        return f"{feat_sql} - LAG({feat_sql}, {lag}) OVER (ORDER BY {ts_col})"
    return feat_sql


def _sql_threshold_literal(threshold: float) -> str:
    """Render a threshold as a DuckDB DOUBLE literal that round-trips exactly.

    The pipeline binarises with the full-precision Python ``float`` threshold,
    so the SQL comparison must use the *identical* IEEE-754 double — otherwise
    values sitting exactly on a discrete boundary (e.g. a rolling pctrank of
    ``k/window``) flip between active and inactive.

    A plain ``repr(float)`` is **not** safe here: DuckDB's text-to-double
    parser is not always correctly rounded and can land on the adjacent double
    for a 17-significant-digit string.  Emitting the *exact* decimal expansion
    of the double via :class:`decimal.Decimal` removes the ambiguity — the
    value is representable exactly, so any parser must round it back to the
    same double — and the explicit ``::DOUBLE`` cast keeps the comparison in
    double space (the exact expansion overflows DuckDB's DECIMAL precision).
    """
    return f"{Decimal(float(threshold))}::DOUBLE"


def _sql_condition(
    t_sql: str,
    threshold: float,
    direction: str,
    event_type: str,
    ts_col: str,
) -> str:
    """Apply threshold/crossing condition to a transformed SQL expression."""
    op = "<" if direction == "below" else ">"
    opp = ">=" if direction == "below" else "<="
    thr = _sql_threshold_literal(threshold)
    if event_type == "crossing":
        # Crossing events are only generated for identity transform, so t_sql
        # contains no nested window functions — LAG(t_sql) is valid DuckDB SQL.
        return (
            f"(({t_sql}) {op} {thr}"
            f" AND LAG(({t_sql})) OVER (ORDER BY {ts_col}) {opp} {thr})"
        )
    return f"({t_sql}) {op} {thr}"


# ---------------------------------------------------------------------------
# Human-readable formula builders
# ---------------------------------------------------------------------------

def _build_event_formula(
    source_feature: str,
    source_cols: list,
    transform: str,
    transform_params: dict,
    threshold: float,
    direction: str,
    event_type: str,
) -> str:
    """Build a human-readable mathematical formula for one event component."""
    feat = _formula_feature(source_feature, source_cols, transform_params)
    t = _formula_transform(feat, transform, transform_params)
    return _formula_condition(t, threshold, direction, event_type)


def _formula_feature(source_feature: str, source_cols: list, transform_params: dict) -> str:
    sf = source_feature
    sc = source_cols
    if sf.startswith("ratio_") and len(sc) == 2:
        return f"{sc[0]} / {sc[1]}"
    if sf.startswith("spread_") and len(sc) == 2:
        return f"({sc[0]} - {sc[1]}) / {sc[1]}"
    if sf.startswith("diffnorm_") and len(sc) == 2:
        std = transform_params.get("diffnorm_std", 1.0)
        if std != 1.0:
            return f"({sc[0]} - {sc[1]}) / {std:.4g}"
        return f"{sc[0]} - {sc[1]}"
    if sf.startswith("bb_pct_b_") and len(sc) == 3:
        val, lower, upper = sc
        return f"bb_pct_b({val}, {lower}, {upper})"
    if sf.startswith("pos_") and len(sc) == 3:
        val, mn, mx = sc
        return f"pos({val}, {mn}, {mx})"
    # arity-1: native column
    return sc[0] if sc else sf


def _formula_transform(feat: str, transform: str, transform_params: dict) -> str:
    if transform == "identity":
        return feat
    if transform == "rolling_pctrank":
        w = transform_params.get("window", "?")
        return f"pctrank({feat}, w={w})"
    if transform == "rolling_zscore":
        w = transform_params.get("window", "?")
        return f"zscore({feat}, w={w})"
    if transform == "delta":
        lag = transform_params.get("lag", 1)
        return f"Δ({feat}, lag={lag})"
    return feat


def _formula_condition(t: str, threshold: float, direction: str, event_type: str) -> str:
    thr = f"{threshold:.4g}"
    if event_type == "crossing":
        arrow = "↓" if direction == "below" else "↑"
        return f"{t} crosses {arrow} {thr}"
    op = "<" if direction == "below" else ">"
    return f"{t} {op} {thr}"
