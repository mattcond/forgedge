"""Core data structures for the Rule Registry module (FORGE Modulo 4).

Rule Registry is the fourth and last module of the FORGE pipeline.  It receives
the rules validated by Rule Discovery — one pool per ticker present in the input
CSV — collects them into an in-memory registry for the current session,
computes the correlation matrices, marks duplicates, runs the cross-ticker
backtest and produces the final output: a flat table and a self-contained HTML
report.

The dataclasses here mirror, field by field, the document schema documented in
``docs/modules/RuleRegistry.md`` (Section 4).  Rule Registry never re-evaluates
the quality of a single rule on its source ticker — Rule Discovery already did
that.  It evaluates the *relations* between rules (overlap, gain correlation,
duplication) and their *generalisability* across the other tickers of the
session.

Stateless by design (spec Section 3): the registry is rebuilt from scratch on
every FORGE session — there is no persistent catalog.  The exported flat table
is the only persistence artefact; the user manages it as they see fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..event_discovery.models import EventCandidate
from ..rule_discovery.models import BacktestParams, RuleDiscoveryResponse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegistryConfig:
    """Top-level configuration for the Rule Registry pipeline.

    Every knob of the four registry steps is exposed here.  The thresholds left
    ``TBD`` in the specification are given the defaults calibrated against the
    worked example of ``docs/modules/RuleRegistry.md`` (Section 12): the
    ``OVERLAP_THRESHOLD`` of 0.70 reproduces the deduplication of the example,
    and ``CROSS_PF_THRESHOLD`` of 2.0 reproduces every PASS/FAIL verdict of the
    cross-ticker section.

    Attributes
    ----------
    overlap_threshold : float
        Jaccard similarity above which two rules are considered duplicates
        (Step 3).  The weaker of the pair (lower in-sample PF) is flagged.
    gain_corr_threshold : float
        Spearman correlation used purely for reporting the "same regime
        exposure" reading of Section 6 — does not drive deduplication.
    cross_pf_threshold : float
        Minimum profit factor for a ``PASS`` verdict in the cross-ticker
        backtest (Step 4).
    generic_ratio_threshold : float
        Minimum fraction of ``PASS`` verdicts for the ``is_generic`` flag and
        the ``GENERIC``/``PARTIAL`` badge boundary (Step 4).  Defaults to exactly
        two-thirds (``2/3 ≈ 0.667``) so a 2-of-3-ticker rule reads ``PARTIAL`` as
        the spec intends — a literal ``0.67`` would just miss it.
    cross_min_active : int
        Minimum aligned active bars required before a Spearman gain correlation
        is computed; below it the correlation is reported as ``0.0`` (spec
        Section 6, Matrix B).
    export_format : str
        Flat-table format: ``"excel"`` or ``"csv"``.
    export_duplicates : bool
        Include rules flagged ``is_duplicate`` in the export (Section 9).
    export_non_generic : bool
        Include rules that are not generic in the export (Section 9).
    html_include_tradelog : bool
        Append the cross-session trade log to the HTML report (Section 11).
    html_charts : bool
        Embed the inline SVG charts (equity curves, correlation heatmaps).
    timestamp_col : str
        Datetime column on every ticker KPI table (default ``"open_dt"``).
    session_date : str or None
        ISO date stamped onto the report; ``None`` → today.
    """

    overlap_threshold: float = 0.70
    gain_corr_threshold: float = 0.70
    cross_pf_threshold: float = 2.0
    generic_ratio_threshold: float = 2.0 / 3.0
    cross_min_active: int = 10
    export_format: str = "excel"
    export_duplicates: bool = True
    export_non_generic: bool = True
    html_include_tradelog: bool = True
    html_charts: bool = True
    timestamp_col: str = "open_dt"
    session_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class RuleSubmission:
    """One validated rule handed to the registry, with its reconstruction context.

    Rule Discovery emits a :class:`RuleDiscoveryResponse`; the registry needs a
    little more to do its job — the source ticker label and the originating
    :class:`EventCandidate` (so the rule's boolean signal can be replayed on the
    *other* tickers with thresholds recalibrated on the local distribution).

    Attributes
    ----------
    ticker : str
        Source ticker the rule was extracted on (e.g. ``"ADAUSDC"``).
    response : RuleDiscoveryResponse
        The Rule Discovery verdict.  Only ``EDGE`` / ``PARTIAL-EDGE`` responses
        enter the registry (Step 1); ``NON-EDGE`` submissions are ignored.
    candidate : EventCandidate
        The Event Candidate referenced by the validated rule.  Provides the
        deterministic replay path (``apply``) used by the cross-ticker backtest.
    grade : str or None
        Optional letter grade carried over from upstream (e.g. the Alpha
        Discovery grade).  When ``None`` the registry derives a grade from the
        Rule Discovery evidence (see :func:`forgedge.rule_registry.ingestion`).
    """

    ticker: str
    response: RuleDiscoveryResponse
    candidate: EventCandidate
    grade: Optional[str] = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CrossTickerResult:
    """Outcome of replaying one rule on one alternative ticker (Step 4).

    The rule's logical structure is unchanged; only the absolute thresholds are
    recalibrated onto the target ticker's distribution (spec Section 8).

    Attributes
    ----------
    ticker : str
        The target ticker this result refers to.
    expression_adapted : str
        The rule expression with its absolute thresholds recalibrated on the
        target distribution.  The relative (pctrank/zscore) thresholds are
        approximately invariant by construction.
    pf : float
        Profit factor on the target ticker.
    win_rate : float
        Win rate on the target ticker (0–1).
    total_trades : int
        Number of executed trades on the target ticker.
    zero_months : int
        Months with no trades on the target ticker.
    verdict : str
        ``"PASS"`` when ``pf >= cross_pf_threshold``, else ``"FAIL"``.
    """

    ticker: str
    expression_adapted: str
    pf: float
    win_rate: float
    total_trades: int
    zero_months: int
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RuleDocument:
    """One in-memory registry document — a validated rule on a specific ticker.

    Mirrors the ``RuleDocument`` schema of the spec (Section 4).  Fields are
    grouped into: identification, the parallel activation arrays that feed the
    correlation matrices, operational parameters, source statistics, regime
    metadata, the correlation/deduplication fields populated by Steps 2–3, and
    the cross-ticker fields populated by Step 4.

    The ``activation_idx`` / ``activation_dates`` / ``gains`` arrays are the key
    data structure for the correlation matrices; they are parallel and indexed
    on the executed trades of the source ticker.
    """

    # ── identification ───────────────────────────────────────────────────
    rule_id: str
    expression: str
    source_ticker: str
    source_alpha_id: str
    verdict: str
    grade: str

    # ── parallel activation arrays (length = n executed trades) ──────────
    activation_idx: List[int]
    activation_dates: List[str]
    gains: List[float]

    # ── operational parameters ───────────────────────────────────────────
    params: Dict[str, object]

    # ── source-ticker statistics (from Rule Discovery) ───────────────────
    stats: Dict[str, object]

    # ── regime ───────────────────────────────────────────────────────────
    regime: Dict[str, object]

    # ── populated by Step 2 / 3 (initially None) ─────────────────────────
    overlap_max: Optional[float] = None
    gain_corr_max: Optional[float] = None
    is_duplicate: Optional[bool] = None
    duplicate_of: Optional[str] = None

    # ── populated by Step 4 (initially empty) ────────────────────────────
    cross_ticker: Dict[str, CrossTickerResult] = field(default_factory=dict)
    cross_ticker_score: Optional[int] = None
    cross_ticker_total: Optional[int] = None
    is_generic: Optional[bool] = None
    classification: Optional[str] = None  # GENERIC | PARTIAL | SPECIFIC | ISOLATED

    # ── internal handles (excluded from any serialisation) ───────────────
    _candidate: Optional[EventCandidate] = field(default=None, repr=False)
    _bt_params: Optional[BacktestParams] = field(default=None, repr=False)
    _trades: Optional[pd.DataFrame] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def pf(self) -> float:
        """Source-ticker profit factor (convenience for deduplication ranking)."""
        return float(self.stats.get("pf", float("nan")))

    def to_dict(self) -> dict:
        """Nested dictionary mirroring the document schema (spec Section 4)."""
        return {
            "rule_id": self.rule_id,
            "expression": self.expression,
            "source_ticker": self.source_ticker,
            "source_alpha_id": self.source_alpha_id,
            "verdict": self.verdict,
            "grade": self.grade,
            "activation_idx": list(self.activation_idx),
            "activation_dates": list(self.activation_dates),
            "gains": list(self.gains),
            "params": dict(self.params),
            "stats": dict(self.stats),
            "regime": dict(self.regime),
            "overlap_max": self.overlap_max,
            "gain_corr_max": self.gain_corr_max,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "cross_ticker": {k: v.to_dict() for k, v in self.cross_ticker.items()},
            "cross_ticker_score": self.cross_ticker_score,
            "cross_ticker_total": self.cross_ticker_total,
            "is_generic": self.is_generic,
            "classification": self.classification,
        }


@dataclass
class CorrelationMatrices:
    """The two correlation matrices computed at the end of ingestion (Step 2).

    Attributes
    ----------
    rule_ids : list[str]
        Row/column labels, in registry order.
    jaccard : pd.DataFrame
        Symmetric Jaccard matrix on the activation dates (temporal overlap).
    spearman : pd.DataFrame
        Symmetric Spearman matrix on the date-aligned gains (return
        correlation).  Bars with no trade contribute a gain of ``0``.
    """

    rule_ids: List[str]
    jaccard: pd.DataFrame
    spearman: pd.DataFrame
