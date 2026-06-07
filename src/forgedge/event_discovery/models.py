"""Core data structures for the Event Discovery module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class ColumnType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass
class ColumnClassification:
    col_name: str
    col_type: ColumnType
    n_distinct: int
    is_scale_free: Optional[bool] = None
    scale_free_override: Optional[bool] = None

    @property
    def effective_scale_free(self) -> bool:
        if self.col_type != ColumnType.CONTINUOUS:
            return False
        if self.scale_free_override is not None:
            return self.scale_free_override
        return bool(self.is_scale_free)

    @property
    def scale_free_overridden(self) -> bool:
        return (
            self.scale_free_override is not None
            and self.scale_free_override != self.is_scale_free
        )


@dataclass
class GateParams:
    min_act: int = 50
    min_months: int = 8
    max_conc: float = 0.40
    min_tpm: float = 2.0


@dataclass
class GateResult:
    passed: bool
    n_activations: int
    n_active_months: int
    max_monthly_share: float
    mean_tpm: float
    fail_reason: Optional[str] = None


@dataclass
class EventComponent:
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
    n_activations: int
    n_active_months: int
    zero_months: int
    max_monthly_share: float
    mean_tpm: float


@dataclass
class RawEvent:
    """Internal event used during pipeline processing."""
    series: pd.Series
    component: EventComponent
    gate_result: Optional[GateResult] = None

    @property
    def key(self) -> str:
        c = self.component
        return f"{c.transformed_col}__{c.event_type}__{c.direction}__{c.threshold:.8f}"

    @property
    def transform_key(self) -> str:
        """Identifies the transform type (independent of threshold)."""
        c = self.component
        params_str = "_".join(f"{k}{v}" for k, v in sorted(c.transform_params.items()))
        return f"{c.source_feature}__{c.transform}__{params_str}"


@dataclass
class EventCandidate:
    event_id: str
    status: str
    components: list[EventComponent]
    expression: str
    activation_stats: ActivationStats
    consistency_gate: GateResult
    event_series: Optional[pd.Series] = field(default=None, repr=False)

    def to_dict(self) -> dict:
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
