"""Step 5 — AND Composition.

Combines pairs (or triples) of passing events with logical AND and
re-applies the Consistency Gate on the composed result.

Composition rules
-----------------
ALLOWED
  • Different transforms on the same source feature
    (identity × pctrank, identity × zscore, pctrank × delta, …)
  • Semantically distinct source features
    (close_rsi_25 AND ratio_ema09_ema25)

NOT ALLOWED
  • Same transform + same source feature with different thresholds
    (redundant: one is a superset of the other)
  • More than three events in AND (structural overfitting)
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .consistency_gate import ConsistencyGate, _build_month_index
from .models import EventComponent, GateResult, RawEvent

# Maximum number of events to retain per (feature, transform) slot
# before cross-feature AND composition.
_MAX_PER_SLOT = 3

# Hard cap on AND-composition pool to keep O(n^2) tractable.
_MAX_POOL = 300

# Maximum composed events returned (avoid memory explosion).
_MAX_COMPOSED = 2000


class ANDComposer:
    """Composes passing single events into AND-combined multi-component events.

    Parameters
    ----------
    gate : ConsistencyGate or None
        Gate instance reused from Step 4 so that composed events are
        evaluated against the same thresholds as single events.
        Defaults to a new ``ConsistencyGate()`` with default params.
    """

    def __init__(self, gate: Optional[ConsistencyGate] = None):
        self.gate = gate or ConsistencyGate()

    def compose(
        self,
        passing_events: list[RawEvent],
        timestamps: pd.Series,
        max_components: int = 2,
    ) -> list[RawEvent]:
        """Generate valid AND compositions and return those passing the gate.

        Algorithm
        ---------
        1. **Pool construction**: ``_build_composition_pool`` reduces the
           full set of passing events to a tractable subset by keeping at
           most ``_MAX_PER_SLOT`` events per (source_feature, transform,
           params) slot and capping the total pool at ``_MAX_POOL``.  This
           keeps the O(n²) pair enumeration manageable.

        2. **Boolean array cache**: Each event's series is pre-converted to
           a uint8 numpy array once, so the bitwise AND of any pair requires
           only a single vectorised operation.

        3. **Pair enumeration** (max_components >= 2): All ``combinations(pool, 2)``
           are checked.  Invalid pairs (same transform + same params + same
           feature, or same native type on the same source) are skipped via
           ``_is_valid_pair``.  The gate is evaluated on the AND result; only
           passing pairs are added to ``composed``.  The full pd.Series for
           the AND result is only constructed after the gate passes, avoiding
           memory allocation for the ~95% of pairs that fail.

        4. **Triple enumeration** (max_components >= 3, only if pool <= 80):
           Same logic applied to ``combinations(pool, 3)``.  The pool size
           cap prevents O(n³) blowup.

        5. **Hard cap**: ``_MAX_COMPOSED`` limits the total number of composed
           events returned to avoid memory explosions when many pairs pass.

        Parameters
        ----------
        passing_events : list[RawEvent]
            All single events that passed the gate in Step 4.
        timestamps : pd.Series
            Datetime series aligned to the KPI table rows.
        max_components : int
            Maximum number of components per composed event (2 or 3).

        Returns
        -------
        list[RawEvent]
            AND-composed events that passed the Consistency Gate.
        """
        if not passing_events:
            return []

        month_index, n_total_months = _build_month_index(timestamps)

        # Build the composition pool from two sub-pools:
        #   A) within-feature pool: all passing events, capped per feature×transform slot
        #   B) cross-feature pool: top events per feature (by n_activations)
        pool = _build_composition_pool(passing_events)

        # Pre-cache uint8 boolean arrays for fast numpy AND
        bool_arrays: dict[int, np.ndarray] = {
            id(ev): ev.series.fillna(0).values.astype(np.uint8)
            for ev in pool
        }

        composed: list[RawEvent] = []

        for ev_a, ev_b in itertools.combinations(pool, 2):
            if len(composed) >= _MAX_COMPOSED:
                break
            if not self._is_valid_pair(ev_a, ev_b):
                continue

            a_arr = bool_arrays[id(ev_a)]
            b_arr = bool_arrays[id(ev_b)]
            and_arr = a_arr & b_arr

            counts = np.zeros(n_total_months, dtype=np.int32)
            np.add.at(counts, month_index, and_arr.astype(np.int32))

            result = self.gate.evaluate(and_arr.astype(bool), counts, n_total_months)
            if not result.passed:
                continue

            and_series = pd.Series(and_arr.astype(float), index=ev_a.series.index)
            composed.append(_make_composed_event(ev_a, ev_b, and_series, result))

        if max_components >= 3 and len(pool) <= 80:
            for ev_a, ev_b, ev_c in itertools.combinations(pool, 3):
                if len(composed) >= _MAX_COMPOSED:
                    break
                if not self._is_valid_triple(ev_a, ev_b, ev_c):
                    continue
                a_arr = bool_arrays[id(ev_a)]
                b_arr = bool_arrays[id(ev_b)]
                c_arr = bool_arrays[id(ev_c)]
                and_arr = a_arr & b_arr & c_arr
                counts = np.zeros(n_total_months, dtype=np.int32)
                np.add.at(counts, month_index, and_arr.astype(np.int32))
                result = self.gate.evaluate(and_arr.astype(bool), counts, n_total_months)
                if result.passed:
                    and_series = pd.Series(and_arr.astype(float), index=ev_a.series.index)
                    composed.append(_make_composed_event(ev_a, ev_b, and_series, result, third=ev_c))

        return composed

    # ------------------------------------------------------------------
    # Validity checks
    # ------------------------------------------------------------------

    def _is_valid_pair(self, a: RawEvent, b: RawEvent) -> bool:
        """Return True if the pair ``(a, b)`` is a valid AND composition candidate.

        Two rejection rules are applied:

        1. **Same native type on same source**: if both events come from
           ``binary_native`` or ``categorical_onehot`` and share the same
           source feature, the AND would be either trivially false (two
           different one-hot classes) or redundant.

        2. **Same transform + same window/lag + same source**: the event with
           the stricter threshold is a subset of the event with the looser
           threshold, making the AND equivalent to the stricter one alone.
           Example: ``pctrank > 0.95 AND pctrank > 0.90`` reduces to
           ``pctrank > 0.95``.

        Parameters
        ----------
        a : RawEvent
        b : RawEvent

        Returns
        -------
        bool
        """
        ca, cb = a.component, b.component
        # Both native-type on same source → skip
        if (ca.transform in ("binary_native", "categorical_onehot")
                and cb.transform in ("binary_native", "categorical_onehot")
                and ca.source_feature == cb.source_feature):
            return False
        # Same source + same transform + same window/lag params → subset
        if (ca.source_feature == cb.source_feature
                and ca.transform == cb.transform
                and ca.transform_params == cb.transform_params):
            return False
        return True

    def _is_valid_triple(self, a: RawEvent, b: RawEvent, c: RawEvent) -> bool:
        """Return True if all three pairwise combinations of ``(a, b, c)`` are valid.

        A triple is valid only when every constituent pair passes
        ``_is_valid_pair``.  This is a sufficient (though not necessary)
        condition for the triple to be non-redundant.

        Parameters
        ----------
        a : RawEvent
        b : RawEvent
        c : RawEvent

        Returns
        -------
        bool
        """
        return (
            self._is_valid_pair(a, b)
            and self._is_valid_pair(a, c)
            and self._is_valid_pair(b, c)
        )


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _build_composition_pool(events: list[RawEvent]) -> list[RawEvent]:
    """Reduce the full set of passing events to a tractable composition pool.

    Grouping strategy
    -----------------
    Events are grouped by a *slot key* composed of
    ``(source_feature, transform, sorted_params)``.  Within each slot,
    events differ only in their threshold value.  Keeping more than
    ``_MAX_PER_SLOT`` (default 3) events per slot adds diminishing diversity
    while rapidly inflating the O(n²) pair count.

    The top events within each slot are selected by ``n_activations``
    (descending) to favour events that fire more often and are therefore
    more likely to survive the AND's intersection.

    After slot-level pruning, the overall pool is capped at ``_MAX_POOL``
    (default 300) using the same ``n_activations`` ranking.

    Parameters
    ----------
    events : list[RawEvent]
        All events that passed the Consistency Gate (Step 4).

    Returns
    -------
    list[RawEvent]
        Subset of ``events`` of length <= ``_MAX_POOL``.
    """
    slots: dict[str, list[RawEvent]] = defaultdict(list)
    for ev in events:
        c = ev.component
        p_str = "_".join(f"{k}{v}" for k, v in sorted(c.transform_params.items()))
        slot_key = f"{c.source_feature}__{c.transform}__{p_str}"
        slots[slot_key].append(ev)

    pool: list[RawEvent] = []
    for slot_events in slots.values():
        # Keep top events per slot, sorted descending by n_activations
        top = sorted(
            slot_events,
            key=lambda e: e.gate_result.n_activations if e.gate_result else 0,
            reverse=True,
        )[:_MAX_PER_SLOT]
        pool.extend(top)

    # Final cap
    if len(pool) > _MAX_POOL:
        pool = sorted(
            pool,
            key=lambda e: e.gate_result.n_activations if e.gate_result else 0,
            reverse=True,
        )[:_MAX_POOL]

    return pool


# ---------------------------------------------------------------------------
# Composed event constructor
# ---------------------------------------------------------------------------

def _make_composed_event(
    a: RawEvent,
    b: RawEvent,
    and_series: pd.Series,
    gate_result: GateResult,
    third: Optional[RawEvent] = None,
) -> RawEvent:
    """Construct a RawEvent representing the AND composition of two or three events.

    The new event's ``EventComponent`` uses ``transform="and_composition"``
    and stores the constituent components in a ``_components`` attribute
    (a dynamic attribute, not part of the dataclass definition) so that
    ``EventDiscovery._to_candidate`` can recover the full component list.

    The ``source_feature``, ``transformed_col``, and ``expression`` fields
    are formed by joining the corresponding fields of the constituent
    components with `` AND ``.

    Parameters
    ----------
    a : RawEvent
        First constituent event.
    b : RawEvent
        Second constituent event.
    and_series : pd.Series
        Pre-computed boolean AND of ``a.series`` and ``b.series``
        (and ``third.series`` if provided).
    gate_result : GateResult
        Already-evaluated gate result for this composed event.
    third : RawEvent or None
        Optional third constituent for triple compositions.

    Returns
    -------
    RawEvent
        The composed event with ``gate_result`` already set.
    """
    components = [a.component, b.component]
    if third is not None:
        components.append(third.component)

    expr = " AND ".join(c.expression for c in components)
    source = " AND ".join(c.source_feature for c in components)
    t_cols = " AND ".join(c.transformed_col for c in components)

    comp = EventComponent(
        source_feature=source,
        transform="and_composition",
        transform_params={"n_components": len(components)},
        transformed_col=t_cols,
        threshold=float("nan"),
        threshold_type="composed",
        direction="n/a",
        event_type="threshold",
        expression=expr,
    )
    comp._components = components  # type: ignore[attr-defined]

    ev = RawEvent(series=and_series, component=comp)
    ev.gate_result = gate_result
    return ev
