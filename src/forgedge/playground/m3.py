"""Playground use cases anchored on Modulo 3 — Rule Discovery.

Two use cases from the tracking issue:

1. ``diagnostics_vs_verdict`` — which non-blocking M2 diagnostics correlate
   with a later M3 verdict, stratified by grade.
2. ``lottery_only_winners`` — contracts whose only obstacle to a full EDGE
   was the search-level rotation null, not any of the economic/statistical
   gates.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from ..forge import ForgeResult

__all__ = ["diagnostics_vs_verdict", "lottery_only_winners"]

_ROTATION_REASON_PREFIX = "search-level rotation null not cleared"


def _contract_grade(contract):
    score = getattr(contract, "alpha_score", None)
    if score is None or getattr(score, "grade", None) is None:
        return None
    return str(score.grade).strip().upper()


def diagnostics_vs_verdict(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format link between M2's non-blocking diagnostics and the M3 verdict.

    Explodes ``AlphaContract.diagnostics`` — observations that inform the
    alpha grade but gate nothing in M2 — against the ``RuleDiscoveryResponse
    .verdict`` M3 later assigned the same contract. A diagnostic that shows
    up disproportionately often on ``NON-EDGE`` rows is a candidate to
    promote from an FYI into an actual M2 gate.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``alpha_id``, ``grade``, ``diagnostic``,
        ``verdict``. Downstream, e.g.
        ``pd.crosstab(df["diagnostic"], df["verdict"], normalize="index")``,
        optionally filtered to one ``grade`` first.
    """
    rows: List[dict] = []

    for result in results:
        for contract, response in result.rule_responses:
            diagnostics = contract.diagnostics or [None]
            for diagnostic in diagnostics:
                rows.append(
                    {
                        "ticker": result.ticker,
                        "alpha_id": contract.alpha_id,
                        "grade": _contract_grade(contract),
                        "diagnostic": diagnostic,
                        "verdict": response.verdict,
                    }
                )

    columns = ["ticker", "alpha_id", "grade", "diagnostic", "verdict"]
    return pd.DataFrame(rows, columns=columns)


def lottery_only_winners(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format flag for PARTIAL-EDGE contracts blocked only by the rotation null.

    A ``PARTIAL-EDGE`` verdict means at least one entry landed in
    ``response.rejection_reasons`` (the ``edge_block`` that would otherwise
    make it a full ``EDGE``). This isolates the case where the *only* such
    entry is the search-level rotation null
    (``"search-level rotation null not cleared (rotation_p=... > ...)"``,
    annotated via ``AlphaContract.rotation_p``/``rotation_threshold``) — a
    contract that cleared every economic/statistical gate and only lost the
    multiple-testing lottery, as opposed to one still failing on PF, DSR,
    OOS consistency, etc. Useful to judge how much a permissive preset's
    ``PARTIAL-EDGE`` pile is "close misses" versus genuinely weak.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``alpha_id``, ``grade``, ``rotation_p``,
        ``rotation_threshold``, ``n_reasons``, ``rotation_only``. Filtered to
        ``verdict == "PARTIAL-EDGE"`` contracts only. Downstream, e.g.
        ``df.groupby("grade")["rotation_only"].mean()``.
    """
    rows: List[dict] = []

    for result in results:
        for contract, response in result.rule_responses:
            if response.verdict != "PARTIAL-EDGE":
                continue

            reasons = response.rejection_reasons or []
            rotation_only = len(reasons) == 1 and reasons[0].startswith(_ROTATION_REASON_PREFIX)

            rows.append(
                {
                    "ticker": result.ticker,
                    "alpha_id": contract.alpha_id,
                    "grade": _contract_grade(contract),
                    "rotation_p": getattr(contract, "rotation_p", None),
                    "rotation_threshold": getattr(contract, "rotation_threshold", None),
                    "n_reasons": len(reasons),
                    "rotation_only": rotation_only,
                }
            )

    columns = [
        "ticker",
        "alpha_id",
        "grade",
        "rotation_p",
        "rotation_threshold",
        "n_reasons",
        "rotation_only",
    ]
    return pd.DataFrame(rows, columns=columns)
