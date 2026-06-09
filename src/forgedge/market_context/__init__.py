"""Market Context module — FORGE pipeline Modulo 0."""
from .context import MarketContext, build_classifier
from .ema_proxy import EMAProxyClassifier
from .models import (
    DEFAULT_LABELS,
    REGIME_COL,
    REGIME_STABLE_COL,
    EMAProxyConfig,
    MarketContextConfig,
    RegimeClassifier,
)

__all__ = [
    "MarketContext",
    "MarketContextConfig",
    "EMAProxyConfig",
    "EMAProxyClassifier",
    "RegimeClassifier",
    "build_classifier",
    "DEFAULT_LABELS",
    "REGIME_COL",
    "REGIME_STABLE_COL",
]
