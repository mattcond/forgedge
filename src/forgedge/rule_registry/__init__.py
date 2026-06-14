"""Rule Registry module — FORGE pipeline Modulo 4.

Collects the rules validated by Rule Discovery into a stateless, in-memory
registry for the current session, computes the Jaccard (activation overlap) and
Spearman (gain correlation) matrices, flags duplicates, runs the cross-ticker
backtest with thresholds recalibrated on each target ticker's distribution, and
exports the final artefacts — a flat table (CSV / Excel) and a self-contained
HTML report.
"""
from .correlation import (
    correlation_matrices,
    gain_correlation_by_date,
    jaccard_by_date,
)
from .cross_ticker import classify_genericity, cross_ticker_backtest
from .dedup import mark_duplicates
from .export import export_table, flat_table, select_for_export
from .ingestion import build_document, is_ingestable
from .models import (
    CorrelationMatrices,
    CrossTickerResult,
    RegistryConfig,
    RuleDocument,
    RuleSubmission,
)
from .recalibrate import recalibrate_candidate
from .registry import RuleRegistry
from .report import html_report

__all__ = [
    "RuleRegistry",
    "RuleSubmission",
    "RegistryConfig",
    "RuleDocument",
    "CrossTickerResult",
    "CorrelationMatrices",
    "build_document",
    "is_ingestable",
    "correlation_matrices",
    "jaccard_by_date",
    "gain_correlation_by_date",
    "mark_duplicates",
    "cross_ticker_backtest",
    "classify_genericity",
    "recalibrate_candidate",
    "flat_table",
    "export_table",
    "select_for_export",
    "html_report",
]
