"""Playground use cases anchored on Modulo 4 — Rule Registry.

Two use cases from the tracking issue:

1. ``classification_by_grade`` — does a higher alpha grade actually mean
   better cross-ticker generalisation?
2. ``duplicate_clusters`` — how much dedup weight is there, and which
   survivors absorb it.

Unlike the other playground modules, these take ``RuleRegistry`` objects
directly rather than ``ForgeResult``: the registry that matters for
cross-ticker classification is the pooled one ``forge_multi()`` returns
separately (``ForgeResult.registry`` is ``None`` on each per-ticker result in
that path — see its docstring), not something reachable from ``ForgeResult``
alone. Pass ``[result.registry]`` for a single-ticker ``forge()`` run (trivial
cross-ticker) or ``[registry]`` for a ``forge_multi()`` pooled registry.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from ..rule_registry import RuleRegistry

__all__ = ["classification_by_grade", "duplicate_clusters"]


def classification_by_grade(registries: Iterable[RuleRegistry]) -> pd.DataFrame:
    """Long-format link between a rule's originating alpha grade and its cross-ticker classification.

    One row per ``RuleDocument`` with a populated ``classification``
    (``GENERIC``/``PARTIAL``/``SPECIFIC``/``ISOLATED`` — ``None`` when Step 4
    never ran, e.g. a single-ticker session with nothing to replay against).

    Parameters
    ----------
    registries : Iterable[RuleRegistry]
        One or more already-``run()`` Rule Registries (see module docstring
        for how to obtain one).

    Returns
    -------
    pd.DataFrame
        Columns: ``rule_id``, ``source_ticker``, ``grade``,
        ``classification``. Downstream, e.g.
        ``pd.crosstab(df["grade"], df["classification"], normalize="index")``
        to check whether grade A rules skew more ``GENERIC`` than grade C.
    """
    rows: List[dict] = []

    for registry in registries:
        for doc in registry.documents:
            if doc.classification is None:
                continue
            rows.append(
                {
                    "rule_id": doc.rule_id,
                    "source_ticker": doc.source_ticker,
                    "grade": doc.grade,
                    "classification": doc.classification,
                }
            )

    columns = ["rule_id", "source_ticker", "grade", "classification"]
    return pd.DataFrame(rows, columns=columns)


def duplicate_clusters(registries: Iterable[RuleRegistry]) -> pd.DataFrame:
    """Long-format dedup outcome for every rule in the registry.

    One row per ``RuleDocument``, flagging whether it was marked a duplicate
    (Step 2-3) and, if so, which surviving ``rule_id`` it was folded into.
    Grouping by ``duplicate_of`` sizes each survivor's absorption cluster —
    a large cluster is a sign M1/M2 rediscovered the same underlying idea
    under different thresholds/expressions many times over.

    Parameters
    ----------
    registries : Iterable[RuleRegistry]
        One or more already-``run()`` Rule Registries (see module docstring
        for how to obtain one).

    Returns
    -------
    pd.DataFrame
        Columns: ``rule_id``, ``source_ticker``, ``grade``, ``is_duplicate``,
        ``duplicate_of``. Downstream, e.g. ``df["is_duplicate"].mean()`` for
        the overall dedup rate, or
        ``df[df["is_duplicate"]].groupby("duplicate_of").size().sort_values(ascending=False)``
        for the largest absorption clusters.
    """
    rows: List[dict] = []

    for registry in registries:
        for doc in registry.documents:
            rows.append(
                {
                    "rule_id": doc.rule_id,
                    "source_ticker": doc.source_ticker,
                    "grade": doc.grade,
                    "is_duplicate": doc.is_duplicate,
                    "duplicate_of": doc.duplicate_of,
                }
            )

    columns = ["rule_id", "source_ticker", "grade", "is_duplicate", "duplicate_of"]
    return pd.DataFrame(rows, columns=columns)
