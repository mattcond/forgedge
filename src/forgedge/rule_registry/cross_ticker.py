"""Step 4 — cross-ticker backtest for the Rule Registry (FORGE Modulo 4).

Every rule in the registry is replayed on all tickers other than the one it was
extracted on (spec Section 8).  The rule's logical structure is unchanged; its
absolute thresholds are recalibrated on each target ticker's local distribution
(see :mod:`forgedge.rule_registry.recalibrate`) and the recalibrated signal is
backtested with the *same* operational parameters that Rule Discovery validated.

A rule that clears ``cross_pf_threshold`` on a target ticker scores a ``PASS``;
the fraction of PASS verdicts drives the genericity classification:

    3/3 (100%)  GENERIC    — works on every tested ticker
    2/3  (67%)  PARTIAL    — works on the majority
    1/3  (33%)  SPECIFIC   — works only on (close to) the source
    0/3   (0%)  ISOLATED   — does not generalise
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..rule_discovery.backtest import run_backtest
from .models import CrossTickerResult, RegistryConfig, RuleDocument
from .recalibrate import recalibrate_candidate

_SIGNAL_COL = "__registry_xticker_signal__"


def classify_genericity(score: int, total: int, ratio_threshold: float) -> str:
    """Map a cross-ticker score to a badge (spec Section 8 table).

    ``GENERIC`` requires a clean sweep; ``PARTIAL`` clears the ratio threshold
    without sweeping; ``SPECIFIC`` passes somewhere but below threshold;
    ``ISOLATED`` never passes.
    """
    if total <= 0:
        return "ISOLATED"
    ratio = score / total
    if ratio >= 1.0:
        return "GENERIC"
    if ratio >= ratio_threshold:
        return "PARTIAL"
    if ratio > 0.0:
        return "SPECIFIC"
    return "ISOLATED"


def cross_ticker_backtest(
    docs: List[RuleDocument],
    frames: Dict[str, pd.DataFrame],
    config: RegistryConfig,
) -> None:
    """Run the cross-ticker backtest for every document, in place (spec Step 4).

    Parameters
    ----------
    docs : list[RuleDocument]
        Registry documents with their ``_candidate`` / ``_bt_params`` handles
        populated by ingestion.
    frames : dict[str, pd.DataFrame]
        ``ticker -> KPI table``, already prepared (sorted, DatetimeIndex and a
        timestamp column).  Must contain every ticker referenced by a document.
    config : RegistryConfig
        Thresholds and column names.
    """
    tickers = list(frames.keys())

    for doc in docs:
        source = doc.source_ticker
        candidate = doc._candidate
        params = doc._bt_params
        source_df = frames.get(source)

        results: Dict[str, CrossTickerResult] = {}
        score = 0
        total = 0
        if candidate is not None and params is not None and source_df is not None:
            for target in tickers:
                if target == source:
                    continue
                target_df = frames[target]
                total += 1
                results[target] = _one_target(
                    candidate, params, source_df, target_df, target, config
                )
                if results[target].verdict == "PASS":
                    score += 1

        doc.cross_ticker = results
        doc.cross_ticker_score = score
        doc.cross_ticker_total = total
        if total > 0:
            ratio = score / total
            doc.is_generic = ratio >= config.generic_ratio_threshold
            doc.classification = classify_genericity(
                score, total, config.generic_ratio_threshold
            )
        else:
            doc.is_generic = None
            doc.classification = None


def _one_target(
    candidate,
    params,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target: str,
    config: RegistryConfig,
) -> CrossTickerResult:
    """Backtest one rule on one target ticker with recalibrated thresholds."""
    adapted = recalibrate_candidate(candidate, source_df, target_df)
    signal = adapted.apply(target_df).reindex(target_df.index)

    frame = target_df.copy()
    frame[_SIGNAL_COL] = signal.fillna(0).to_numpy()

    summary = run_backtest(
        frame, _SIGNAL_COL, params, timestamp_col=config.timestamp_col
    )

    pf = float(summary.profit_factor)
    verdict = "PASS" if (pf == pf and pf >= config.cross_pf_threshold) else "FAIL"
    wr = float(summary.win_rate_pct)
    return CrossTickerResult(
        ticker=target,
        expression_adapted=adapted.expression,
        pf=round(pf, 4),
        win_rate=round(wr, 4) if wr == wr else float("nan"),
        total_trades=int(summary.total_trades),
        zero_months=int(summary.zero_months),
        verdict=verdict,
    )
