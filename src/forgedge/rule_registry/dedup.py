"""Step 3 — deduplication for the Rule Registry (FORGE Modulo 4).

After the correlation matrices are computed, the registry *marks* duplicates —
it never deletes a rule (spec Section 3 and Section 7).  For every pair whose
Jaccard overlap exceeds ``overlap_threshold`` the weaker rule (lower in-sample
profit factor) is flagged ``is_duplicate = True`` and its ``duplicate_of`` is
set to the dominant rule.

Chains are preserved as the spec requires (Section 7): if ``R01 ← R02 ← R03``
all overlap pairwise, the registry records ``R02.duplicate_of = R01`` and
``R03.duplicate_of = R02`` — each rule points at its *immediate* dominant (the
weakest rule that still dominates it), so the user can read the chain and decide
for themselves.
"""
from __future__ import annotations

from typing import List

from .models import CorrelationMatrices, RuleDocument


def mark_duplicates(
    docs: List[RuleDocument],
    matrices: CorrelationMatrices,
    overlap_threshold: float,
) -> None:
    """Flag duplicates in place (spec Section 7).

    For each document, look at every *other* document that overlaps it above the
    threshold and has a strictly higher profit factor; the immediate dominant is
    the one with the **smallest** such PF (the closest stronger neighbour).  A
    document with no dominant keeps ``is_duplicate = False`` / ``duplicate_of =
    None``.
    """
    ids = matrices.rule_ids
    jac = matrices.jaccard

    for i, doc in enumerate(docs):
        dominant = None
        dominant_pf = float("inf")
        for j, other in enumerate(docs):
            if i == j:
                continue
            if float(jac.iat[i, j]) <= overlap_threshold:
                continue
            # The dominant must be strictly stronger on its own ticker.
            if not (other.pf > doc.pf):
                continue
            # Pick the closest stronger neighbour (smallest PF above this rule)
            # so a PF-ordered chain points at the immediate dominant, not the
            # global root.
            if other.pf < dominant_pf:
                dominant = other
                dominant_pf = other.pf

        if dominant is not None:
            doc.is_duplicate = True
            doc.duplicate_of = dominant.rule_id
        else:
            doc.is_duplicate = False
            doc.duplicate_of = None
