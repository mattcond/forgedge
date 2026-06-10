"""forgedge — Feature-Oriented Rule Generation Engine."""

__version__ = "0.1.0"

from .alpha_discovery import (
    AlphaConfig,
    AlphaContract,
    AlphaDiscovery,
    DerivedTarget,
    OOSValidation,
    PromotionThresholds,
)
from .event_discovery import DiscoveryConfig, EventDiscovery
from .market_context import (
    EMAProxyClassifier,
    EMAProxyConfig,
    MarketContext,
    MarketContextConfig,
    RegimeClassifier,
)

__all__ = [
    "EventDiscovery",
    "DiscoveryConfig",
    "MarketContext",
    "MarketContextConfig",
    "EMAProxyConfig",
    "EMAProxyClassifier",
    "RegimeClassifier",
    "AlphaDiscovery",
    "AlphaConfig",
    "DerivedTarget",
    "OOSValidation",
    "PromotionThresholds",
    "AlphaContract",
]
