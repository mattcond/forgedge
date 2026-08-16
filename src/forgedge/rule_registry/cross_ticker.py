"""Step 4 — cross-ticker backtest for the Rule Registry (FORGE Modulo 4).

Every rule in the registry is replayed on all tickers other than the one it was
extracted on (spec Section 8).  The rule's logical structure is unchanged; its
absolute thresholds are recalibrated on each target ticker's local distribution
(see :mod:`forgedge.rule_registry.recalibrate`) and the recalibrated signal is
backtested with the *same* operational parameters that Rule Discovery validated.

A rule scores a ``PASS`` on a target ticker when it clears **both** halves of
the transfer criterion::

    PASS  ⟺  pf_other >= cross_pf_threshold           (it is tradeable there)
             AND  pf_other >= retention · pf_home      (it actually transfers)

and the fraction of PASS verdicts drives the genericity classification:

    3/3 (100%)  GENERIC    — works on every tested ticker
    2/3  (67%)  PARTIAL    — works on the majority
    1/3  (33%)  SPECIFIC   — works only on (close to) the source
    0/3   (0%)  ISOLATED   — does not generalise

Why two halves (F8)
-------------------
A single absolute bar answers *"is it good elsewhere?"*, while every word in
the ``GENERIC``/``PARTIAL``/``SPECIFIC``/``ISOLATED`` vocabulary asks *"does it
transfer?"*.  Those come apart in both directions:

===========================  =========  ==========  =====================
rule                         PF home    PF away     single absolute bar
===========================  =========  ==========  =====================
transfers perfectly          1.6        1.6         FAIL at 2.0 — wrong
degraded by a third          3.0        2.05        PASS at 2.0 — wrong
===========================  =========  ==========  =====================

Lowering the single bar to the home PF fixes the first row and makes the
second worse still: the *weakest* rules would get the *easiest* genericity
test.  Transfer and quality are orthogonal axes, and the registry already has
a place for each — this classification for transfer, the M3 verdict and the
grade for quality.  One number cannot carry both, so it carries one.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..rule_discovery.backtest import run_backtest
from ..unset import coalesce
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
                    candidate, params, source_df, target_df, target, config,
                    pf_home=doc.pf,
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


def transfer_bar(pf_home: float, config: RegistryConfig) -> float:
    """The profit factor a rule must reach away from home to score a ``PASS``.

    The higher of the two halves of the criterion, so one number can be shown
    in a report and compared against ``pf`` directly.  A non-finite
    ``pf_home`` — no home trades, or an infinite PF because nothing lost —
    leaves only the absolute floor, since there is no meaningful fraction of
    infinity to retain.
    """
    floor = float(coalesce(config.cross_pf_threshold, default=1.5))
    retention = float(coalesce(config.min_cross_pf_retention, default=0.8))
    if pf_home is None or not np.isfinite(pf_home):
        return floor
    return max(floor, retention * float(pf_home))


def _one_target(
    candidate,
    params,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target: str,
    config: RegistryConfig,
    pf_home: float = float("nan"),
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
    bar = transfer_bar(pf_home, config)
    verdict = "PASS" if (pf == pf and pf >= bar) else "FAIL"
    wr = float(summary.win_rate_pct)
    return CrossTickerResult(
        ticker=target,
        expression_adapted=adapted.expression,
        pf=round(pf, 4),
        win_rate=round(wr, 4) if wr == wr else float("nan"),
        total_trades=int(summary.total_trades),
        zero_months=int(summary.zero_months),
        verdict=verdict,
        bar=round(bar, 4),
    )
