"""Redundancy checks for StepWiseDiscovery — Jaccard + continuous correlation.

Two components (or two whole seeds) can be redundant in three progressively
harder-to-catch ways:

1. **Same name** — literally the same ``source_feature`` re-selected at a
   later depth with a different threshold/transform. Cheapest check,
   name-based (see ``StepWiseDiscovery`` — tracked as a plain ``set`` of
   ``source_feature`` strings, not implemented here).
2. **Same information, different name, similar activation** — e.g. an
   ATR-ratio and its NATR analogue (``NATR = ATR / close``) chosen with
   thresholds that still produce overlapping boolean activation sets.
   Caught by :func:`max_jaccard`, the same ``J(A, B) = |A∩B| / |A∪B|``
   definition and default threshold (``0.85``) as
   :func:`forgedge.event_discovery.diversity_gate.apply_diversity_gate` /
   ``ANDComposer(max_constituent_jaccard=...)``.
3. **Same information, different name, DIFFERENT activation** — the gap
   Jaccard alone cannot close: two components whose *continuous*
   pre-threshold signal is near-identical can still end up with divergent
   boolean activation sets purely because of where each one's threshold
   happened to land. Caught by :func:`max_abs_corr`, comparing the
   continuous series directly. Deliberately a stricter default threshold
   (``0.95``) than Jaccard's — it must only catch near-duplicate
   information, not merely-correlated-but-distinct signals (e.g. two
   different-period RSIs).

Both checks are pure functions over already-computed series — no discovery
logic lives here. :class:`~forgedge.experiment.step_wise_discovery.StepWiseDiscovery`
applies them at two points: within a single chain (a child must be diverse
from every component already in its parent chain) and across accepted seeds
(a new seed must be diverse from every seed already accepted) — the same
two thresholds, the same two functions, just a different "already used"
pool passed in.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np

DEFAULT_MAX_JACCARD = 0.85
"""Same default as ``forgedge.event_discovery.diversity_gate``/``ANDComposer``."""

DEFAULT_MAX_ABS_CORR = 0.95
"""Deliberately stricter than ``DEFAULT_MAX_JACCARD`` — see module docstring."""

__all__ = [
    "DEFAULT_MAX_JACCARD",
    "DEFAULT_MAX_ABS_CORR",
    "family_key",
    "max_jaccard",
    "max_abs_corr",
    "is_redundant",
]


def family_key(candidate) -> str:
    """Group a candidate by feature family, stripping numeric periods.

    ``ratio_close_ema_09_close_lag3`` and ``ratio_close_ema_21_close_lag3``
    both reduce to ``ratio_close_ema_close_lag|identity`` — the same
    structural pattern at a different period, which the top-level seed
    selection treats as one "family" so it doesn't spend more than one seed
    slot on it. This is a **name-based** heuristic (fast, no series
    comparison) and is intentionally coarser than :func:`max_jaccard`/
    :func:`max_abs_corr` — it catches "same pattern, different period", not
    "different pattern, same information" (that's what the two similarity
    checks below exist for).
    """
    parts = []
    for c in candidate.components:
        base = re.sub(r"_?\d+", "", c.source_feature)
        parts.append(f"{base}|{c.transform}")
    return "+".join(parts)


def max_jaccard(bool_series: np.ndarray, used_series_list: Sequence[np.ndarray]) -> float:
    """Max Jaccard similarity of ``bool_series`` against each series in ``used_series_list``.

    ``J(A, B) = |A ∩ B| / |A ∪ B|``. Returns ``0.0`` if ``used_series_list``
    is empty (nothing to compare against yet).
    """
    if not used_series_list:
        return 0.0
    best = 0.0
    a_sum = int(np.asarray(bool_series).sum())
    for used in used_series_list:
        used = np.asarray(used)
        inter = int(np.logical_and(bool_series, used).sum())
        union = a_sum + int(used.sum()) - inter
        j = inter / union if union > 0 else 0.0
        if j > best:
            best = j
    return float(best)


def max_abs_corr(
    components: Iterable,
    used_cols: Sequence[str],
    series_by_col: Dict[str, np.ndarray],
) -> float:
    """Max ``|Pearson correlation|`` of each component's CONTINUOUS
    pre-threshold series against every column in ``used_cols``.

    ``series_by_col`` maps a ``transformed_col`` name to its full-length
    continuous series (identity **and** rolling transforms alike) — the
    caller is expected to have computed it once, up front, from
    ``TransformLayer``/``FeatureGenerator`` output (see
    :class:`~forgedge.experiment.step_wise_discovery.StepWiseDiscovery`).
    A component built independently on a partition still resolves to the
    same key here as long as it was generated from the same feature/
    transform parameters, since the column-naming logic is deterministic —
    it does not need to come from the same DataFrame.

    Returns ``0.0`` if a column can't be found (never blocks a candidate on
    missing data — Jaccard remains the backstop) or if ``used_cols`` is
    empty. Requires at least 30 finite, paired observations to compute a
    correlation for a given pair; pairs with fewer are skipped.
    """
    if not used_cols:
        return 0.0
    best = 0.0
    for comp in components:
        s = series_by_col.get(comp.transformed_col)
        if s is None:
            continue
        s = np.asarray(s, dtype=float)
        for used_col in used_cols:
            u = series_by_col.get(used_col)
            if u is None:
                continue
            u = np.asarray(u, dtype=float)
            valid = np.isfinite(s) & np.isfinite(u)
            if valid.sum() < 30:
                continue
            corr = np.corrcoef(s[valid], u[valid])[0, 1]
            if np.isfinite(corr) and abs(corr) > best:
                best = abs(corr)
    return float(best)


def is_redundant(
    *,
    bool_series: np.ndarray,
    used_bool_series: Sequence[np.ndarray],
    components: Iterable,
    used_cols: Sequence[str],
    series_by_col: Dict[str, np.ndarray],
    max_jaccard_threshold: float = DEFAULT_MAX_JACCARD,
    max_abs_corr_threshold: float = DEFAULT_MAX_ABS_CORR,
):
    """Combined Jaccard + continuous-correlation redundancy check.

    Returns a ``(redundant: bool, jaccard: float, abs_corr: float)`` tuple —
    the two scores are returned even when neither trips the threshold, so
    callers can log them for diagnostics regardless of the outcome.
    """
    j = max_jaccard(bool_series, used_bool_series)
    corr = max_abs_corr(components, used_cols, series_by_col)
    redundant = bool(j >= max_jaccard_threshold or corr >= max_abs_corr_threshold)
    return redundant, j, corr
