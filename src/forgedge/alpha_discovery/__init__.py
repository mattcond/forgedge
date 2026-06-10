"""Alpha Discovery module — FORGE pipeline Modulo 2.

Derives an economic target per Event Candidate from the data, confirms it
out-of-sample, measures predictive power and formalises the survivors into
Alpha Contracts.
"""
from .discovery import AlphaDiscovery
from .market_structure import analyse_market_structure
from .models import (
    AlphaConfig,
    AlphaContract,
    AlphaScore,
    DerivedTarget,
    EventStats,
    ICResult,
    MarketStructure,
    OOSValidation,
    PromotionThresholds,
    RegimeAnalysis,
    RegimeStat,
)
from .target import binary_target, forward_returns

__all__ = [
    "AlphaDiscovery",
    "AlphaConfig",
    "DerivedTarget",
    "OOSValidation",
    "PromotionThresholds",
    "AlphaContract",
    "ICResult",
    "EventStats",
    "RegimeStat",
    "RegimeAnalysis",
    "MarketStructure",
    "AlphaScore",
    "forward_returns",
    "binary_target",
    "analyse_market_structure",
]
