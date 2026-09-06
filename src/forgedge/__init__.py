"""forgedge — Feature-Oriented Rule Generation Engine."""

__version__ = "0.2.0"

from .alpha_discovery import (
    AlphaConfig,
    AlphaContract,
    AlphaDiscovery,
    DerivedTarget,
    OOSValidation,
    PromotionThresholds,
    TargetConfig,
)
from .calibration import (
    CalibrationReport,
    FastRotationNull,
    RotationCalibrator,
    RotationConfig,
)
from .event_discovery import (
    CustomEvent,
    DiscoveryConfig,
    EventDiscovery,
    EventWalkForwardConfig,
)
from .forge import ForgeResult, forge, forge_multi, lookup_by_id
from .ledger import HypothesisLedger
from .resolver import (
    Constraint,
    Derivation,
    PipelineContext,
    ResolutionTrace,
    Violation,
    collect_context,
    resolve,
    resolve_config,
)
from .unset import UNSET
from .rule_report import RuleSpec, rule_performance_report
from .timebudget import TimeBudget
from .market_context import (
    EMAProxyClassifier,
    EMAProxyConfig,
    MarketContext,
    MarketContextConfig,
    RegimeClassifier,
)
from .rule_discovery import (
    BacktestParams,
    RuleDiscovery,
    RuleDiscoveryConfig,
    RuleDiscoveryResponse,
    RuleWalkForwardConfig,
    SelectionCriteria,
    ValidatedRule,
    WalkForwardConfig,
    html_report,
    rule_summary_report,
    text_report,
)
from .rule_registry import (
    CrossTickerResult,
    RegistryConfig,
    RuleDocument,
    RuleRegistry,
    RuleSubmission,
)
from .target_optimizer import TargetOptimizer
from .presets import forge_preset, preset_info, PRESETS
from .summary_report import summary_report, DataQualityReport, Finding
from .config_report import config_report, ConfigReport
from .episodes import episode_starts, episode_ids, concurrency, ConcurrencyStats
from .kpi_builder import build_features, lag_features, candle_features, pattern_features

__all__ = [
    "forge",
    "forge_multi",
    "ForgeResult",
    "lookup_by_id",
    "RotationCalibrator",
    "RotationConfig",
    "CalibrationReport",
    "FastRotationNull",
    "HypothesisLedger",
    "RuleSpec",
    "rule_performance_report",
    "TimeBudget",
    "UNSET",
    "PipelineContext",
    "ResolutionTrace",
    "Derivation",
    "Constraint",
    "Violation",
    "resolve",
    "resolve_config",
    "collect_context",
    "config_report",
    "ConfigReport",
    "EventDiscovery",
    "DiscoveryConfig",
    "CustomEvent",
    "MarketContext",
    "MarketContextConfig",
    "EMAProxyConfig",
    "EMAProxyClassifier",
    "RegimeClassifier",
    "AlphaDiscovery",
    "AlphaConfig",
    "TargetConfig",
    "TargetOptimizer",
    "DerivedTarget",
    "OOSValidation",
    "PromotionThresholds",
    "AlphaContract",
    "RuleDiscovery",
    "RuleDiscoveryConfig",
    "RuleDiscoveryResponse",
    "BacktestParams",
    "SelectionCriteria",
    "EventWalkForwardConfig",
    "RuleWalkForwardConfig",
    # Ambiguous legacy alias — resolves to RuleWalkForwardConfig.
    "WalkForwardConfig",
    "ValidatedRule",
    "text_report",
    "html_report",
    "rule_summary_report",
    "RuleRegistry",
    "RuleSubmission",
    "RegistryConfig",
    "RuleDocument",
    "CrossTickerResult",
    "forge_preset",
    "preset_info",
    "PRESETS",
    "summary_report",
    "DataQualityReport",
    "Finding",
    "episode_starts",
    "episode_ids",
    "concurrency",
    "ConcurrencyStats",
    "build_features",
    "lag_features",
    "candle_features",
    "pattern_features",
]
