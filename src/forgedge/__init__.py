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

__all__ = [
    "forge",
    "forge_multi",
    "ForgeResult",
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
]
