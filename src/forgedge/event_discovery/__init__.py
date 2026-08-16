"""Event Discovery module — FORGE pipeline step 1."""
from .discovery import DiscoveryConfig, EventDiscovery
from .diversity_gate import apply_diversity_gate
from .models import (
    ActivationStats,
    ColumnClassification,
    ColumnType,
    CustomEvent,
    EventCandidate,
    EventComponent,
    FoldResult,
    GateParams,
    GateResult,
    RawEvent,
    ValidationResult,
    EventWalkForwardConfig,
    WalkForwardConfig,
)

__all__ = [
    "EventDiscovery",
    "DiscoveryConfig",
    "EventWalkForwardConfig",
    "WalkForwardConfig",
    "GateParams",
    "CustomEvent",
    "EventCandidate",
    "EventComponent",
    "ActivationStats",
    "GateResult",
    "ValidationResult",
    "FoldResult",
    "ColumnClassification",
    "ColumnType",
    "RawEvent",
    "apply_diversity_gate",
]
