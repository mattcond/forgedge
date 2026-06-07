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
    def __init__(self, gate: Optional[ConsistencyGate] = None):
        self.gate = gate or ConsistencyGate()

    def compose(
        self,
        passing_events: list[RawEvent],
        timestamps: pd.Series,
        max_components: int = 2,
    ) -> list[RawEvent]:
        """Generate valid AND compositions and return those passing the gate."""
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
        return (
            self._is_valid_pair(a, b)
            and self._is_valid_pair(a, c)
            and self._is_valid_pair(b, c)
        )


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _build_composition_pool(events: list[RawEvent]) -> list[RawEvent]:
    """Return a manageable subset of passing events for AND composition.

    Groups events by (source_feature, transform, transform_params) slot.
    Within each slot, keeps the _MAX_PER_SLOT most activated events.
    Overall pool is capped at _MAX_POOL.
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
