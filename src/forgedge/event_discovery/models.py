"""Core data structures for the Event Discovery module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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
        Format: ``<source_feature>__<transform>__<sorted_params>``.
        """
        c = self.component
        params_str = "_".join(f"{k}{v}" for k, v in sorted(c.transform_params.items()))
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
                }
                for c in self.components
            ],
        }
