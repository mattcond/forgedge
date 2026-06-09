"""Alpha Discovery module — FORGE pipeline Modulo 2.

Measures the predictive power of Event Candidates against an economic target
and formalises the survivors into Alpha Contracts.
"""
from .discovery import AlphaDiscovery
from .market_structure import analyse_market_structure
from .models import (
    AlphaConfig,
    AlphaContract,
    AlphaScore,
    EventStats,
    ICResult,
    MarketStructure,
    PromotionThresholds,
    RegimeAnalysis,
    RegimeStat,
    TargetDefinition,
)
from .target import build_target

__all__ = [
    "AlphaDiscovery",
    "AlphaConfig",
    "TargetDefinition",
    "PromotionThresholds",
    "AlphaContract",
    "ICResult",
    "EventStats",
    "RegimeStat",
    "RegimeAnalysis",
    "MarketStructure",
    "AlphaScore",
    "build_target",
    "analyse_market_structure",
]
