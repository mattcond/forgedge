"""Playground use cases anchored on Modulo 2 — Alpha Discovery.

Two use cases from the tracking issue:

1. ``discard_reasons_by_grade`` — why does Rule Discovery (M3) verdict
   ``NON-EDGE`` on alpha contracts of a given grade.
2. ``undetermined_direction_by_family`` — which source-feature families feed
   contracts that Alpha Discovery (M2) could not orient
   (``direction == "undetermined"``).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd

from ..forge import ForgeResult

__all__ = ["discard_reasons_by_grade", "undetermined_direction_by_family"]

_FAMILY_RE = re.compile(r"^(?:open|high|low|close|volume)_([a-z0-9]+)_\d+$")


def _contract_grade(contract) -> Optional[str]:
    """Upper-case letter grade of a contract, or ``None`` if ungraded.

    Mirrors ``forgedge.forge._contract_grade`` — kept local since that helper
    is private to the orchestrator module.
    """
    score = getattr(contract, "alpha_score", None)
    if score is None or getattr(score, "grade", None) is None:
        return None
    return str(score.grade).strip().upper()


def discard_reasons_by_grade(
    results: Iterable[ForgeResult],
    grade: str = "A",
) -> pd.DataFrame:
    """Long-format breakdown of why M3 verdicts ``NON-EDGE`` on ``grade`` contracts.

    Reads ``result.rule_responses`` (every promoted contract paired with its
    Rule Discovery verdict, not just the tradeable ones from ``.edges()``),
    keeps the contracts of the requested grade whose ``response.verdict ==
    "NON-EDGE"``, and explodes ``response.rejection_reasons`` one row per
    reason so the caller can aggregate freely (``groupby("reason").size()``,
    cross-tab against ``failed_condition``, etc.).

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.
    grade : str, default "A"
        Letter grade to filter on (case-insensitive).

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``alpha_id``, ``event_candidate_id``, ``reason``,
        ``failed_condition``. Empty (with these columns) if nothing matches.
    """
    target_grade = grade.strip().upper()
    rows: List[dict] = []

    for result in results:
        for contract, response in result.rule_responses:
            if _contract_grade(contract) != target_grade:
                continue
            if response.verdict != "NON-EDGE":
                continue

            entry_opt = getattr(response, "entry_optimization", None)
            failed_condition = getattr(entry_opt, "failed_condition", None)
            reasons = response.rejection_reasons or [None]

            for reason in reasons:
                rows.append(
                    {
                        "ticker": result.ticker,
                        "alpha_id": contract.alpha_id,
                        "event_candidate_id": contract.event_candidate_id,
                        "reason": reason,
                        "failed_condition": failed_condition,
                    }
                )

    columns = ["ticker", "alpha_id", "event_candidate_id", "reason", "failed_condition"]
    return pd.DataFrame(rows, columns=columns)


def _feature_family(source_feature: str, source_cols: Optional[list]) -> str:
    """Semantic family of a component's source feature.

    Native columns follow ``{base}_{indicator}_{period}`` (e.g.
    ``close_rsi_25`` -> ``"rsi"``). Arity-2/3 paired features (``source_cols``
    populated — cross-OHLC, MACD-vs-signal, price-vs-volume, etc.) don't carry
    that naming convention on their synthetic ``source_feature``, so they are
    bucketed by arity instead. Anything else falls back to ``"other"``.
    """
    if source_cols:
        n = len(source_cols)
        if n == 2:
            return "cross_pair"
        if n == 3:
            return "cross_triple"
        return "other"
    match = _FAMILY_RE.match(source_feature)
    return match.group(1) if match else "other"


def undetermined_direction_by_family(
    results: Iterable[ForgeResult],
) -> pd.DataFrame:
    """Long-format link between source-feature family and M2's derived direction.

    For every evaluated contract (``result.contracts`` — promoted and
    rejected alike), resolves its originating ``EventCandidate`` via
    ``event_candidate_id`` and emits one row per component with the
    component's semantic family and the contract's ``direction`` (including
    ``"undetermined"``). Composed (AND) events contribute one row per
    constituent component, so a family that only ever appears inside
    composed events is still counted.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``alpha_id``, ``event_candidate_id``, ``family``,
        ``direction``. Downstream: e.g.
        ``df.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())``.
    """
    rows: List[dict] = []

    for result in results:
        candidates_by_id = {c.event_id: c for c in result.candidates}

        for contract in result.contracts:
            candidate = candidates_by_id.get(contract.event_candidate_id)
            if candidate is None:
                continue

            for component in candidate.components:
                family = _feature_family(component.source_feature, component.source_cols)
                rows.append(
                    {
                        "ticker": result.ticker,
                        "alpha_id": contract.alpha_id,
                        "event_candidate_id": contract.event_candidate_id,
                        "family": family,
                        "direction": contract.direction,
                    }
                )

    columns = ["ticker", "alpha_id", "event_candidate_id", "family", "direction"]
    return pd.DataFrame(rows, columns=columns)
