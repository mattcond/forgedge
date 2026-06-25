"""forgedge — Feature-Oriented Rule Generation Engine."""

__version__ = "0.1.0"

from .alpha_discovery import (
    AlphaConfig,
    AlphaContract,
    AlphaDiscovery,
    DerivedTarget,
    OOSValidation,
    PromotionThresholds,
    TargetConfig,
)
from .calibration import CalibrationReport, RotationCalibrator, RotationConfig
from .event_discovery import CustomEvent, DiscoveryConfig, EventDiscovery
from .forge import ForgeResult, forge, forge_multi
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
    SelectionCriteria,
    ValidatedRule,
    WalkForwardConfig,
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
from .kpi_builder import build_kpi_table

__all__ = [
    "forge",
    "forge_multi",
    "ForgeResult",
    "RotationCalibrator",
    "RotationConfig",
    "CalibrationReport",
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
    "WalkForwardConfig",
    "ValidatedRule",
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
    "build_kpi_table",
]
