"""Core data structures for the Event Discovery module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class ColumnType(str, Enum):
    """Enumeration of the three column types recognised by the TypeClassifier.

    CONTINUOUS   — numeric series with more than two distinct values.
    BINARY       — numeric series with exactly two distinct values (0/1 flags).
    CATEGORICAL  — non-numeric or low-cardinality string series.
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass
class ColumnClassification:
    """Result of classifying a single DataFrame column (Step 0).

    Attributes
    ----------
    col_name : str
        Name of the original column.
    col_type : ColumnType
        Inferred type (CONTINUOUS / BINARY / CATEGORICAL).
    n_distinct : int
        Number of unique non-null values observed.
    is_scale_free : bool or None
        Result of the automatic scale-free heuristic.  None if not applicable
        (non-continuous columns are not tested).
    scale_free_override : bool or None
        User-supplied override.  When set, it takes precedence over
        ``is_scale_free`` in ``effective_scale_free``.
    """

    col_name: str
    col_type: ColumnType
    n_distinct: int
    is_scale_free: Optional[bool] = None
    scale_free_override: Optional[bool] = None

    @property
    def effective_scale_free(self) -> bool:
        """Return the scale-free flag that should be used by downstream steps.

        Resolution order:
        1. Non-continuous columns are never scale-free (False).
        2. If a user override exists, return it.
        3. Otherwise return the automatic detection result.
        """
        if self.col_type != ColumnType.CONTINUOUS:
            return False
        if self.scale_free_override is not None:
            return self.scale_free_override
        return bool(self.is_scale_free)

    @property
    def scale_free_overridden(self) -> bool:
        """True when a user override contradicts the automatic detection result.

        Useful for auditing which columns were forced into a different regime
        than the heuristic would have chosen.
        """
        return (
            self.scale_free_override is not None
            and self.scale_free_override != self.is_scale_free
        )


@dataclass
class GateParams:
    """Thresholds that govern the Consistency Gate (Step 4).

    Attributes
    ----------
    min_act : int
        Minimum total number of bar-level activations across the full dataset.
        Events with fewer activations are discarded immediately (volume check).
    min_months : int
        Minimum number of calendar months in which the event must fire at least
        once.  Ensures the event is not limited to a short burst.
    max_conc : float
        Maximum allowed share of total activations concentrated in a single
        month (0–1).  Prevents events that fire densely in one period only.
    min_tpm : float
        Minimum average activations per month (total_activations / n_months).
        Guards against events that pass volume/coverage but are too sparse on
        a per-month basis.
    """

    min_act: int = 50
    min_months: int = 8
    max_conc: float = 0.40
    min_tpm: float = 2.0


@dataclass
class GateResult:
    """Outcome of one ConsistencyGate evaluation.

    Attributes
    ----------
    passed : bool
        True if all four gate criteria were satisfied.
    n_activations : int
        Total number of True bars in the event series.
    n_active_months : int
        Number of calendar months that contain at least one activation.
    max_monthly_share : float
        Fraction of total activations belonging to the most concentrated month
        (max_month_count / n_activations).
    mean_tpm : float
        Average activations per calendar month (n_activations / n_total_months).
    fail_reason : str or None
        Human-readable explanation of the first criterion that caused failure,
        or None when the gate passed.
    """

    passed: bool
    n_activations: int
    n_active_months: int
    max_monthly_share: float
    mean_tpm: float
    fail_reason: Optional[str] = None


@dataclass
class EventComponent:
    """Metadata describing a single boolean condition within an event.

    For a simple event this is the only component.  For AND-composed events
    the parent ``EventCandidate`` holds a list of components, one per
    constituent condition.

    Attributes
    ----------
    source_feature : str
        Name of the original KPI column (e.g. ``close_rsi_25``).
    transform : str
        Transform applied: ``identity``, ``rolling_pctrank``,
        ``rolling_zscore``, ``delta``, ``binary_native``,
        ``categorical_onehot``, or ``and_composition``.
    transform_params : dict
        Parameters of the transform (e.g. ``{"window": 96}`` for pctrank).
    transformed_col : str
        Name of the column *after* the transform was applied.
    threshold : float
        Numerical threshold used to binarise the transformed series.
    threshold_type : str
        Label describing how the threshold was chosen (e.g.
        ``distributional_p05``, ``theoretical_z-2.0``).
    direction : str
        ``"below"`` (series < threshold) or ``"above"`` (series > threshold).
    event_type : str
        ``"threshold"`` for a persistent boolean, ``"crossing"`` for a
        one-bar transition signal.
    expression : str
        Human-readable string representation of the condition, e.g.
        ``"pr_close_rsi_25_96 > 0.85"``.
    """

    source_feature: str
    transform: str
    transform_params: dict
    transformed_col: str
    threshold: float
    threshold_type: str
    direction: str   # "below" | "above"
    event_type: str  # "threshold" | "crossing"
    expression: str
    source_cols: list = field(default_factory=list)
    """Original native column names used to build ``source_feature``.

    Empty for arity-1 (native) features.  For arity-2 features, contains
    ``[col_a, col_b]``; for arity-3, ``[value_col, lower_col, upper_col]``.
    Used by ``EventCandidate.apply()`` to reconstruct the feature series on
    new data without requiring the full FeatureGenerator to run.
    """
    sql_expression: str = ""
    """DuckDB-compatible SQL boolean expression that replicates this component.

    Can be used directly in a SELECT or WHERE clause against a table that
    contains the same source columns as the original KPI DataFrame (i.e.
    any table with the same schema as ``EventDiscovery.df``).

    For pctrank, ``list_grade_up`` + ``list_position`` window functions are
    used (DuckDB ≥ 0.8 required).  For zscore and delta, standard window
    functions (``AVG``, ``STDDEV_SAMP``, ``LAG``) are used.  The
    ``ORDER BY open_dt`` clause assumes the timestamp column is named
    ``open_dt``; substitute as needed.

    Example usage in DuckDB::

        import duckdb
        rel = duckdb.from_df(ed.df.reset_index())
        rel.query("df", f"SELECT *, ({comp.sql_expression})::INT AS event_active FROM df")
    """


@dataclass
class ActivationStats:
    """Aggregated statistics about an event's activation pattern.

    These are computed after the Consistency Gate to provide a richer
    picture than the gate metrics alone.

    Attributes
    ----------
    n_activations : int
        Total bar-level activations over the full dataset.
    n_active_months : int
        Number of calendar months with at least one activation.
    zero_months : int
        Number of calendar months with zero activations (complement of
        n_active_months relative to the full date range).
    max_monthly_share : float
        Fraction of activations concentrated in the busiest month.
    mean_tpm : float
        Average activations per calendar month.
    """

    n_activations: int
    n_active_months: int
    zero_months: int
    max_monthly_share: float
    mean_tpm: float


@dataclass
class RawEvent:
    """Intermediate event representation used inside the pipeline.

    Created by Step 3 (EventGenerator), annotated with a GateResult by
    Step 4 (ConsistencyGate), and consumed by Step 5 (ANDComposer) before
    being promoted to EventCandidate.

    Attributes
    ----------
    series : pd.Series
        Boolean (0/1/NaN) activation series aligned to the KPI table index.
    component : EventComponent
        Full metadata describing the condition that produced this event.
    gate_result : GateResult or None
        Populated by ConsistencyGate.filter(); None before gate evaluation.
    """

    series: pd.Series
    component: EventComponent
    gate_result: Optional[GateResult] = None

    @property
    def key(self) -> str:
        """Unique string key identifying this specific event (feature + transform + threshold).

        Used for deduplication and logging.  Format:
        ``<transformed_col>__<event_type>__<direction>__<threshold:.8f>``
        """
        c = self.component
        return f"{c.transformed_col}__{c.event_type}__{c.direction}__{c.threshold:.8f}"

    @property
    def transform_key(self) -> str:
        """String key identifying the transform slot, independent of threshold.

        Events sharing the same transform_key differ only in their threshold
        value — the AND composer uses this to avoid pairing redundant events
        (one threshold is a superset of another within the same slot).
        Format: ``<source_feature>__<transform>__<sorted_temporal_params>``.

        Only temporal parameters (``window``, ``lag``) are included; feature-level
        parameters such as ``diffnorm_std`` are excluded to keep the key stable
        and comparable across different assets.
        """
        c = self.component
        _temporal = {"window", "lag"}
        params_str = "_".join(
            f"{k}{v}" for k, v in sorted(c.transform_params.items())
            if k in _temporal
        )
        return f"{c.source_feature}__{c.transform}__{params_str}"


@dataclass
class EventCandidate:
    """Final output artifact produced by the Event Discovery pipeline.

    Consumed by the downstream Alpha Discovery module to evaluate whether
    the event's activations carry predictive power for future returns.

    Attributes
    ----------
    event_id : str
        Human-readable identifier encoding the feature, transform, and index
        (e.g. ``EVT-close_rsi_25-PR-0042``).
    status : str
        Pipeline status tag; always ``"CANDIDATE"`` at this stage.
    components : list[EventComponent]
        One component for simple events, two or three for AND compositions.
    expression : str
        Full human-readable boolean expression (ANDs joined with `` AND ``).
    activation_stats : ActivationStats
        Aggregated statistics about the event's temporal distribution.
    consistency_gate : GateResult
        Gate evaluation result; always ``passed=True`` for candidates
        returned from EventDiscovery.run().
    event_series : pd.Series or None
        The raw 0/1/NaN boolean series.  Stored for downstream use;
        excluded from repr to keep logs readable.
    """

    event_id: str
    status: str
    components: list[EventComponent]
    expression: str
    activation_stats: ActivationStats
    consistency_gate: GateResult
    event_series: Optional[pd.Series] = field(default=None, repr=False)

    @property
    def sql_expression(self) -> str:
        """DuckDB-compatible SQL boolean expression for the full event.

        For single-component events, returns the component's ``sql_expression``
        directly.  For AND-composed events, joins each component's expression
        with ``AND``, wrapping each in parentheses for clarity.

        Returns an empty string if no component has a populated ``sql_expression``
        (e.g. legacy candidates created before this field was added).

        Example usage in DuckDB::

            import duckdb
            rel = duckdb.from_df(ed.df.reset_index())
            query = f"SELECT *, ({candidate.sql_expression})::INT AS active FROM df"
            rel.query("df", query)
        """
        parts = [c.sql_expression for c in self.components if c.sql_expression]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return " AND ".join(f"({p})" for p in parts)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Reconstruct the event boolean series on new (out-of-sample) data.

        Given a DataFrame that contains the same source columns used during
        training, this method replicates all pipeline steps — feature
        construction, temporal transform, threshold application — using the
        parameters stored in each ``EventComponent``.

        For ``diffnorm`` features, the in-sample standard deviation stored in
        ``transform_params["diffnorm_std"]`` is used as the normaliser, so the
        event preserves exactly the same scale as the training period.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the same column names as the original KPI table.
            Must contain all columns referenced in ``self.components``.

        Returns
        -------
        pd.Series
            Float series (0.0 / 1.0 / NaN) with the same index as ``df``.
            1.0 means the event is active on that bar.

        Raises
        ------
        KeyError
            If a required source column is missing from ``df``.
        """
        result: pd.Series | None = None
        for comp in self.components:
            part = _apply_component(comp, df)
            if result is None:
                result = part
            else:
                # Match pipeline AND behavior: NaN is treated as inactive (0),
                # consistent with the uint8 bitwise AND used in ANDComposer.
                result = (result.fillna(0).astype(bool) & part.fillna(0).astype(bool)).astype(float)
        if result is None:
            return pd.Series(float("nan"), index=df.index)
        return result

    def to_dict(self) -> dict:
        """Serialise the candidate to a flat dictionary for DataFrame construction.

        The ``components`` key contains a list of per-component dicts.
        All other keys are scalar values suitable for a summary DataFrame row.
        """
        return {
            "event_id": self.event_id,
            "status": self.status,
            "expression": self.expression,
            "n_activations": self.activation_stats.n_activations,
            "n_active_months": self.activation_stats.n_active_months,
            "zero_months": self.activation_stats.zero_months,
            "max_monthly_share": self.activation_stats.max_monthly_share,
            "mean_tpm": self.activation_stats.mean_tpm,
            "gate_passed": self.consistency_gate.passed,
            "components": [
                {
                    "source_feature": c.source_feature,
                    "transform": c.transform,
                    "transform_params": c.transform_params,
                    "threshold": c.threshold,
                    "threshold_type": c.threshold_type,
                    "direction": c.direction,
                    "event_type": c.event_type,
                    "expression": c.expression,
                    "sql_expression": c.sql_expression,
                }
                for c in self.components
            ],
        }


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _apply_component(comp: "EventComponent", df: pd.DataFrame) -> pd.Series:
    """Reconstruct a single EventComponent's boolean series on ``df``.

    Steps
    -----
    1. Build the feature series (native, ratio, spread_pct, diffnorm, …).
    2. Apply the temporal transform (identity, pctrank, zscore, delta).
    3. Apply the threshold / crossing condition.

    Parameters
    ----------
    comp : EventComponent
        A fully populated component (source_cols and transform_params
        must contain all values stored during training).
    df : pd.DataFrame
        DataFrame with native source columns.

    Returns
    -------
    pd.Series
        Float 0/1/NaN series aligned to ``df.index``.
    """
    sf = comp.source_feature

    # ── 1. Feature series ────────────────────────────────────────────────
    if comp.transform in ("binary_native",):
        # Binary: active when column equals the stored threshold value
        raw = df[sf]
        return (raw == comp.threshold).astype(float).where(raw.notna(), float("nan"))

    if comp.transform == "categorical_onehot":
        cls = comp.transform_params.get("class")
        raw = df[sf]
        return (raw == cls).astype(float).where(raw.notna(), float("nan"))

    sc = comp.source_cols
    if sf.startswith("ratio_") and len(sc) == 2:
        series = df[sc[0]] / df[sc[1]]
    elif sf.startswith("spread_") and len(sc) == 2:
        series = (df[sc[0]] - df[sc[1]]) / df[sc[1]]
    elif sf.startswith("diffnorm_") and len(sc) == 2:
        std = comp.transform_params.get("diffnorm_std")
        if not std:
            raise KeyError(
                f"'diffnorm_std' missing from transform_params of component "
                f"'{comp.expression}'. Was this candidate created with an older "
                "version of EventDiscovery?"
            )
        series = (df[sc[0]] - df[sc[1]]) / std
    elif sf.startswith("bb_pct_b_") and len(sc) == 3:
        val, lower, upper = sc
        denom = df[upper] - df[lower]
        series = (df[val] - df[lower]) / denom.replace(0, float("nan"))
    elif sf.startswith("pos_") and len(sc) == 3:
        val, mn, mx = sc
        denom = df[mx] - df[mn]
        series = (df[val] - df[mn]) / denom.replace(0, float("nan"))
    else:
        # Arity-1 native feature — use source_feature name directly
        series = df[sf]

    # ── 2. Temporal transform ────────────────────────────────────────────
    # Use the same min_periods heuristic as TransformLayer to match NaN positions.
    t = comp.transform
    if t == "rolling_pctrank":
        w = comp.transform_params["window"]
        min_p = max(2, w // 2)
        series = series.rolling(w, min_periods=min_p).rank(pct=True)
    elif t == "rolling_zscore":
        w = comp.transform_params["window"]
        min_p = max(2, w // 2)
        roll = series.rolling(w, min_periods=min_p)
        mu = roll.mean()
        sd = roll.std()
        series = (series - mu) / sd.replace(0, float("nan"))
    elif t == "delta":
        series = series.diff(comp.transform_params["lag"])
    # identity: no transform

    # ── 3. Threshold / crossing ──────────────────────────────────────────
    thr = comp.threshold
    if comp.event_type == "crossing":
        if comp.direction == "below":
            current  = series < thr
            previous = series.shift(1) >= thr
        else:
            current  = series > thr
            previous = series.shift(1) <= thr
        result = (current & previous).astype(float)
        result[series.isna() | series.shift(1).isna()] = float("nan")
        return result
    else:
        if comp.direction == "below":
            return (series < thr).astype(float).where(series.notna(), float("nan"))
        return (series > thr).astype(float).where(series.notna(), float("nan"))
