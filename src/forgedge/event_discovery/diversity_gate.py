"""Diversity Gate — Jaccard deduplication of post-ConsistencyGate events.

Removes near-duplicate events before AND composition, keeping the event
with the highest activation count when two events overlap beyond the
configured threshold.
"""
from __future__ import annotations

import numpy as np

from .models import RawEvent


def apply_diversity_gate(
    events: list[RawEvent],
    threshold: float = 0.85,
) -> list[RawEvent]:
    """Greedy Jaccard deduplication of a post-ConsistencyGate event list.

    For each event (ordered by activation count descending) the function
    checks whether it is "too similar" to any already-kept event.  If the
    Jaccard similarity with any kept event meets or exceeds ``threshold``,
    the event is discarded; otherwise it is added to the kept set.

    Jaccard is computed on the True-valued bars of each activation series::

        J(Ei, Ej) = |Ei ∩ Ej| / |Ei ∪ Ej|

    Complexity is O(n × k) per event where k is the number of kept events
    so far.  In practice k ≪ n for thresholds ≥ 0.85, making the loop
    fast even for n ~ 2 000.

    Parameters
    ----------
    events : list[RawEvent]
        Events that have already passed ConsistencyGate, each with a
        populated ``.series`` (boolean 0/1/NaN) and ``.gate_result``.
        The function re-orders them internally by activation count.
    threshold : float
        Maximum tolerated Jaccard similarity between any two kept events.
        A pair with ``J >= threshold`` is considered a near-duplicate and
        the lower-priority event (fewer activations) is discarded.
        Use ``1.0`` to remove only exact duplicates.

    Returns
    -------
    list[RawEvent]
        Deduplicated subset in descending activation-count order.

    Notes
    -----
    *uint8 overflow*: raw dot-products of uint8 boolean vectors can exceed
    255 for series longer than 255 bars.  All arithmetic is performed in
    int32 to avoid silent overflow.

    *Incremental allocation*: the kept-event matrix is pre-allocated to the
    worst-case size and trimmed at the end, avoiding the O(n²) copy cost of
    ``np.vstack`` inside a loop.
    """
    if not events:
        return events

    events = sorted(
        events,
        key=lambda e: e.gate_result.n_activations if e.gate_result else 0,
        reverse=True,
    )

    n = len(events)
    n_bars = len(events[0].series)

    # Pre-allocate matrix: rows = kept events (worst case = n), cols = bars.
    # int32 avoids overflow for series up to 2^31 bars.
    matrix = np.empty((n, n_bars), dtype=np.int32)
    n_kept = 0
    kept_indices: list[int] = []

    for i, ev in enumerate(events):
        col = ev.series.fillna(0).values.astype(np.int32)

        if n_kept == 0:
            matrix[0] = col
            n_kept = 1
            kept_indices.append(i)
            continue

        kept = matrix[:n_kept]           # (n_kept, n_bars)
        intersections = kept @ col       # (n_kept,)  — dot product
        col_sum = int(col.sum())
        kept_sums = kept.sum(axis=1)     # (n_kept,)
        unions = kept_sums + col_sum - intersections
        jaccard = np.where(unions > 0, intersections / unions, 0.0)

        if jaccard.max() < threshold:
            matrix[n_kept] = col
            n_kept += 1
            kept_indices.append(i)

    return [events[i] for i in kept_indices]
