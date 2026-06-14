"""Main RuleRegistry orchestrator — FORGE pipeline Modulo 4.

Collects the rules validated by Rule Discovery — one pool per ticker of the
session — into an in-memory registry, computes the correlation matrices, marks
duplicates, runs the cross-ticker backtest and exposes the final output: a flat
table and a self-contained HTML report.

Flow (Rule Registry spec, Steps 1–5):

1. **Ingestion** — one document per ``EDGE`` / ``PARTIAL-EDGE`` rule.
2. **Correlation matrices** — Jaccard on the activation dates, Spearman on the
   date-aligned gains.
3. **Deduplication** — flag (never delete) the weaker rule of every overlapping
   pair.
4. **Cross-ticker backtest** — replay each rule on every other ticker with
   thresholds recalibrated on the local distribution; classify genericity.
5. **Export** — flat table (CSV / Excel) and HTML report.

The registry is stateless across sessions (spec Section 3): it is rebuilt from
scratch every time; the exported flat table is the only persistence artefact.

Usage
-----
    from forgedge.rule_registry import RuleRegistry, RuleSubmission

    submissions = [
        RuleSubmission(ticker="ADAUSDC", response=resp, candidate=cand),
        ...
    ]
    frames = {"ADAUSDC": ada_kpi, "SOLUSDC": sol_kpi, "BTCUSDC": btc_kpi}

    registry = RuleRegistry(submissions, frames).run()
    df   = registry.flat_table()
    html = registry.html_report()

Or straight from per-ticker :class:`~forgedge.forge.ForgeResult` runs::

    registry = RuleRegistry.from_forge_results(
        {"ADAUSDC": ada_result, "SOLUSDC": sol_result}
    ).run()
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from ..event_discovery.discovery import _infer_timestamp_unit
from . import correlation, dedup, export, ingestion, report
from .cross_ticker import cross_ticker_backtest
from .models import (
    CorrelationMatrices,
    RegistryConfig,
    RuleDocument,
    RuleSubmission,
)

# Quote currencies stripped to build the short ticker label of a rule_id.
_QUOTE_SUFFIXES = ("USDC", "USDT", "BUSD", "USD", "EUR", "BTC", "ETH")


class RuleRegistry:
    """FORGE Rule Registry module (Modulo 4).

    Parameters
    ----------
    submissions : list[RuleSubmission]
        Validated rules from Rule Discovery, each with its source ticker and
        Event Candidate.  Only ``EDGE`` / ``PARTIAL-EDGE`` submissions are
        ingested.
    frames : dict[str, pd.DataFrame]
        ``ticker -> KPI table`` for every ticker of the session.  Each table is
        the Event Discovery post-pipeline frame (carries the native price
        columns and every feature the candidates reference).  Every submission's
        ticker must be present.
    config : RegistryConfig, optional
        Thresholds, export and report settings.
    """

    def __init__(
        self,
        submissions: List[RuleSubmission],
        frames: Dict[str, pd.DataFrame],
        config: Optional[RegistryConfig] = None,
    ):
        self.config = config or RegistryConfig()
        self.submissions = list(submissions)
        self.frames: Dict[str, pd.DataFrame] = {
            t: self._prepare_frame(df) for t, df in frames.items()
        }

        missing = {s.ticker for s in self.submissions if s.ticker not in self.frames}
        if missing:
            raise ValueError(
                f"No KPI frame provided for ticker(s): {sorted(missing)}. "
                "Pass a frame for every submission's source ticker."
            )

        self.documents: List[RuleDocument] = []
        self.matrices: Optional[CorrelationMatrices] = None
        self._ran = False

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self) -> "RuleRegistry":
        """Execute Steps 1–4 and return ``self`` (export is on demand)."""
        self.ingest()
        self.compute_correlations()
        self.deduplicate()
        self.cross_ticker()
        self._ran = True
        return self

    def ingest(self) -> List[RuleDocument]:
        """Step 1 — build one document per tradeable submission."""
        self.documents = []
        counters: Dict[str, int] = {}
        for sub in self.submissions:
            if not ingestion.is_ingestable(sub):
                continue
            short = _short_ticker(sub.ticker)
            counters[short] = counters.get(short, 0) + 1
            rule_id = f"RULE_{short}_{counters[short]:02d}"
            doc = ingestion.build_document(
                sub, self.frames[sub.ticker], rule_id, self.config
            )
            self.documents.append(doc)
        return self.documents

    def compute_correlations(self) -> CorrelationMatrices:
        """Step 2 — Jaccard / Spearman matrices and per-rule maxima."""
        self.matrices = correlation.correlation_matrices(
            self.documents, min_active=self.config.cross_min_active
        )
        correlation.annotate_correlation_maxima(self.documents, self.matrices)
        return self.matrices

    def deduplicate(self) -> None:
        """Step 3 — flag duplicates (never deletes)."""
        if self.matrices is None:
            self.compute_correlations()
        dedup.mark_duplicates(
            self.documents, self.matrices, self.config.overlap_threshold
        )

    def cross_ticker(self) -> None:
        """Step 4 — cross-ticker backtest and genericity classification."""
        cross_ticker_backtest(self.documents, self.frames, self.config)

    # ------------------------------------------------------------------
    # Step 5 — outputs
    # ------------------------------------------------------------------

    def flat_table(self, *, apply_filters: bool = False) -> pd.DataFrame:
        """Flat registry table (spec Section 10).

        ``apply_filters`` honours ``export_duplicates`` / ``export_non_generic``;
        the default keeps every rule (FORGE shows complexity rather than hiding
        it).
        """
        cfg = self.config if apply_filters else None
        return export.flat_table(self.documents, cfg)

    def export(self, path: str) -> str:
        """Write the flat table to ``path`` (CSV / Excel) and return the path."""
        return export.export_table(self.documents, path, self.config)

    def html_report(self, **kwargs) -> str:
        """Render the self-contained HTML report (spec Section 11)."""
        if self.matrices is None:
            self.compute_correlations()
        kwargs.setdefault("tickers", list(self.frames.keys()))
        kwargs.setdefault("session_date", self._session_date())
        return report.html_report(
            self.documents, self.matrices, self.config, **kwargs
        )

    def summary(self) -> pd.DataFrame:
        """Compact one-row-per-rule overview (rule_id, source, PF, cross, class)."""
        rows = []
        for d in self.documents:
            rows.append({
                "rule_id": d.rule_id,
                "source_ticker": d.source_ticker,
                "grade": d.grade,
                "pf": d.stats.get("pf"),
                "win_rate": d.stats.get("win_rate"),
                "total_trades": d.stats.get("total_trades"),
                "overlap_max": d.overlap_max,
                "gain_corr_max": d.gain_corr_max,
                "is_duplicate": d.is_duplicate,
                "duplicate_of": d.duplicate_of,
                "cross_ticker_score": d.cross_ticker_score,
                "cross_ticker_total": d.cross_ticker_total,
                "is_generic": d.is_generic,
                "classification": d.classification,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_forge_results(
        cls,
        results_by_ticker: Dict[str, "object"],
        config: Optional[RegistryConfig] = None,
    ) -> "RuleRegistry":
        """Build a registry directly from per-ticker :class:`ForgeResult` runs.

        Each value is the :class:`~forgedge.forge.ForgeResult` of running the
        FORGE pipeline on one ticker.  Every tradeable rule
        (``response.is_edge``) becomes a submission whose Event Candidate and
        Alpha grade are pulled from the same run; the enriched KPI frame of each
        run is registered for the cross-ticker backtest.
        """
        submissions: List[RuleSubmission] = []
        frames: Dict[str, pd.DataFrame] = {}
        for ticker, result in results_by_ticker.items():
            # Prefer the Event Discovery post-pipeline frame (native columns +
            # every derived feature + regime) so the cross-ticker backtest reads
            # exactly the data the rule was validated on; fall back to the
            # Market-Context-enriched table.
            frames[ticker] = getattr(result, "event_frame", None)
            if frames[ticker] is None:
                frames[ticker] = result.enriched
            if hasattr(result, "submissions"):
                submissions.extend(result.submissions())
                continue
            by_id = {c.event_id: c for c in result.candidates}
            for contract, response in result.rule_responses:
                if not response.is_edge or response.validated_rule is None:
                    continue
                cand = by_id.get(response.validated_rule.event_candidate_id)
                if cand is None:
                    continue
                grade = None
                if getattr(contract, "alpha_score", None) is not None:
                    grade = contract.alpha_score.grade
                submissions.append(
                    RuleSubmission(
                        ticker=ticker, response=response, candidate=cand, grade=grade
                    )
                )
        return cls(submissions, frames, config=config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_date(self) -> str:
        return self.config.session_date or date.today().isoformat()

    def _prepare_frame(self, kpi_table: pd.DataFrame) -> pd.DataFrame:
        """Chronologically-sorted copy with a DatetimeIndex and timestamp column.

        Mirrors the contract of Rule Discovery's frame preparation so the
        cross-ticker backtest reads exactly the same data shape Event/Alpha/Rule
        Discovery produced.
        """
        df = kpi_table.copy()
        ts_col = self.config.timestamp_col

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            if ts_col not in df.columns:
                df[ts_col] = df.index
            return df

        if ts_col not in df.columns:
            raise ValueError(
                f"Timestamp column {ts_col!r} not found and index is not a "
                "DatetimeIndex. Set RegistryConfig(timestamp_col='...')."
            )

        raw = df[ts_col]
        if pd.api.types.is_datetime64_any_dtype(raw):
            parsed = pd.to_datetime(raw)
        elif pd.api.types.is_numeric_dtype(raw):
            parsed = pd.to_datetime(raw, unit=_infer_timestamp_unit(raw))
        else:
            parsed = pd.to_datetime(raw)

        df.index = pd.DatetimeIndex(parsed.to_numpy(), name=ts_col)
        df[ts_col] = df.index
        return df.sort_index()


def _short_ticker(ticker: str) -> str:
    """Short label for a rule_id — the base asset, quote currency stripped.

    ``ADAUSDC`` → ``ADA``; falls back to the upper-cased ticker when no known
    quote suffix matches.
    """
    up = ticker.upper()
    for suffix in _QUOTE_SUFFIXES:
        if up.endswith(suffix) and len(up) > len(suffix):
            return up[: -len(suffix)]
    return up
