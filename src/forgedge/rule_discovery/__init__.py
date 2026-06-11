"""Rule Discovery module — FORGE pipeline Modulo 3.

Validates the operability of an Alpha Contract: reconstructs the event,
parametrises realistic limit-order mechanics, backtests them, confirms the edge
out-of-sample with a walk-forward scheme, and emits a verdict
(``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE``).
"""
from .backtest import run_backtest
from .discovery import RuleDiscovery
from .grid import build_grid, grid_dataframe, run_grid, select_best
from .models import (
    BacktestParams,
    BacktestSummary,
    GridResult,
    GridSpec,
    RegimeBreakdown,
    RuleDiscoveryConfig,
    RuleDiscoveryResponse,
    ScoringParams,
    SelectionCriteria,
    StatisticalValidation,
    ValidatedRule,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSplit,
)
from .report import html_report, text_report
from .validation import deflated_sharpe, validate
from .walkforward import walk_forward

__all__ = [
    "RuleDiscovery",
    "RuleDiscoveryConfig",
    "RuleDiscoveryResponse",
    "BacktestParams",
    "BacktestSummary",
    "ScoringParams",
    "GridSpec",
    "GridResult",
    "SelectionCriteria",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardSplit",
    "StatisticalValidation",
    "RegimeBreakdown",
    "ValidatedRule",
    "run_backtest",
    "run_grid",
    "build_grid",
    "grid_dataframe",
    "select_best",
    "walk_forward",
    "validate",
    "deflated_sharpe",
    "text_report",
    "html_report",
]
