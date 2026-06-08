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

Performance
-----------
Pair evaluation is fully vectorized: all event series are stacked into a
boolean matrix and the gate criteria are computed in a single batch of
numpy/BLAS operations rather than per-pair Python loops.  This removes the
need for an arbitrary pool cap and makes evaluation of tens of thousands of
pairs practical on typical hardware.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .consistency_gate import ConsistencyGate, _build_month_index
from .models import EventComponent, GateParams, GateResult, RawEvent

# Number of (i, j) pairs processed per vectorized batch.  Larger values
# consume more peak memory; smaller values add Python loop overhead.
# At 5 000 pairs with n_rows ≈ 8 760 (1 year 1H) the float32 cast of
# and_chunk costs ~175 MB — a comfortable headroom on modern hardware.
_CHUNK_SIZE = 5_000

# Maximum number of events to retain per (feature, transform) slot
# before cross-feature AND composition.  Within-slot events differ only
# in their threshold; keeping at most 3 preserves diversity while
# containing the total pair count.
_MAX_PER_SLOT = 3

# Pool size cap for triple enumeration.  O(n³) blowup makes uncapped
# triples infeasible; pairs are fine because the vectorized batch loop
# scales well.
_MAX_POOL_TRIPLES = 80

# Maximum composed events returned (pairs + triples combined).
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
           params) slot.  No global pool cap is applied for pair
           evaluation — the vectorized batch approach makes the full pool
           tractable regardless of size.

        2. **Validity pre-filter**: ``_validity_mask`` computes a boolean
           matrix of shape (n_pool, n_pool) in pure numpy by comparing
           string arrays for source_feature and transform.  The upper
           triangle of valid pairs is extracted as two index arrays
           ``(ii, jj)`` so no per-pair Python logic is needed.

        3. **One-hot month matrix**: ``_build_one_hot_f32`` constructs a
           (n_rows, n_months) float32 matrix where ``M[r, m] = 1`` if row
           ``r`` belongs to month ``m``.  A left-multiply of a boolean row
           vector against this matrix sums activations by month in a single
           BLAS ``gemm`` call.

        4. **Vectorized pair gate** (max_components >= 2): The pool is
           stacked into a uint8 boolean matrix of shape (n_pool, n_rows).
           Pairs are processed in chunks of ``_CHUNK_SIZE``:

           * ``and_chunk = bool_matrix[ii_chunk] & bool_matrix[jj_chunk]``
             — bitwise AND, shape (K, n_rows)
           * ``counts = and_chunk.astype(float32) @ one_hot``
             — monthly activation counts, shape (K, n_months), via BLAS
           * Gate criteria (volume, coverage, concentration, frequency)
             are evaluated as vectorized numpy comparisons over the entire
             chunk, producing a boolean mask of passing pairs.

           Only the passing pairs require the overhead of constructing a
           pandas Series and a ``RawEvent`` object.

        5. **Triple enumeration** (max_components >= 3, pool <= 80):
           Same loop logic applied to ``combinations(pool[:80], 3)``.
           The pool is capped at ``_MAX_POOL_TRIPLES`` to prevent O(n³)
           blowup.

        6. **Hard cap**: ``_MAX_COMPOSED`` limits the total number of
           composed events returned to avoid memory explosions.

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

        pool = _build_composition_pool(passing_events)
        if len(pool) < 2:
            return []

        # Pre-compute shared structures used by all batch evaluations
        one_hot = _build_one_hot_f32(month_index, n_total_months)
        bool_matrix = np.stack(
            [ev.series.fillna(0).values.astype(np.uint8) for ev in pool]
        )  # (n_pool, n_rows)

        composed: list[RawEvent] = []
        p = self.gate.params

        # ----------------------------------------------------------------
        # Pair enumeration — fully vectorized
        # ----------------------------------------------------------------
        # Build validity mask and extract upper-triangle index pairs
        valid_mask = _validity_mask(pool)
        ii_all, jj_all = np.where(np.triu(valid_mask, k=1))

        n_pairs = len(ii_all)
        for chunk_start in range(0, n_pairs, _CHUNK_SIZE):
            if len(composed) >= _MAX_COMPOSED:
                break
            chunk_end = min(chunk_start + _CHUNK_SIZE, n_pairs)
            ii = ii_all[chunk_start:chunk_end]
            jj = jj_all[chunk_start:chunk_end]

            # Shape: (K, n_rows)
            and_chunk = bool_matrix[ii] & bool_matrix[jj]

            # Monthly counts via BLAS matmul: (K, n_months)
            counts_chunk = and_chunk.astype(np.float32) @ one_hot

            # Gate criteria — vectorized
            n_act = and_chunk.sum(axis=1).astype(np.int32)     # (K,)
            n_active = (counts_chunk > 0).sum(axis=1)          # (K,)
            safe_n_act = np.maximum(n_act, 1).astype(np.float32)
            max_conc = counts_chunk.max(axis=1) / safe_n_act   # (K,)
            mean_tpm = n_act / n_total_months                  # (K,)

            passing_mask = (
                (n_act >= p.min_act)
                & (n_active >= p.min_months)
                & (max_conc <= p.max_conc)
                & (mean_tpm >= p.min_tpm)
            )

            passing_indices = np.where(passing_mask)[0]
            remaining = _MAX_COMPOSED - len(composed)
            if len(passing_indices) > remaining:
                passing_indices = passing_indices[:remaining]

            for k in passing_indices:
                ev_a = pool[ii[k]]
                ev_b = pool[jj[k]]
                gate_result = GateResult(
                    passed=True,
                    n_activations=int(n_act[k]),
                    n_active_months=int(n_active[k]),
                    max_monthly_share=float(max_conc[k]),
                    mean_tpm=float(mean_tpm[k]),
                )
                and_series = pd.Series(
                    and_chunk[k].astype(float), index=ev_a.series.index
                )
                composed.append(_make_composed_event(ev_a, ev_b, and_series, gate_result))

        # ----------------------------------------------------------------
        # Triple enumeration — capped pool, per-triple loop
        # ----------------------------------------------------------------
        if max_components >= 3 and len(pool) <= _MAX_POOL_TRIPLES:
            for ev_a, ev_b, ev_c in itertools.combinations(pool, 3):
                if len(composed) >= _MAX_COMPOSED:
                    break
                if not self._is_valid_triple(ev_a, ev_b, ev_c):
                    continue
                a_arr = bool_matrix[pool.index(ev_a)]
                b_arr = bool_matrix[pool.index(ev_b)]
                c_arr = bool_matrix[pool.index(ev_c)]
                and_arr = a_arr & b_arr & c_arr
                counts = (and_arr.astype(np.float32) @ one_hot).astype(np.int32)
                result = self.gate.evaluate(and_arr.astype(bool), counts, n_total_months)
                if result.passed:
                    and_series = pd.Series(
                        and_arr.astype(float), index=ev_a.series.index
                    )
                    composed.append(
                        _make_composed_event(ev_a, ev_b, and_series, result, third=ev_c)
                    )

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
    while rapidly inflating the total pair count.

    The top events within each slot are selected by ``n_activations``
    (descending) to favour events that fire more often and are therefore
    more likely to survive the AND's intersection.

    Unlike the previous implementation, no global pool cap is applied.
    The vectorized batch gate in ``ANDComposer.compose`` handles arbitrarily
    large pools efficiently through chunked numpy/BLAS operations.

    Parameters
    ----------
    events : list[RawEvent]
        All events that passed the Consistency Gate (Step 4).

    Returns
    -------
    list[RawEvent]
        Subset of ``events``, at most ``_MAX_PER_SLOT`` per
        (feature, transform, params) slot.
    """
    slots: dict[str, list[RawEvent]] = defaultdict(list)
    for ev in events:
        c = ev.component
        p_str = "_".join(f"{k}{v}" for k, v in sorted(c.transform_params.items()))
        slot_key = f"{c.source_feature}__{c.transform}__{p_str}"
        slots[slot_key].append(ev)

    pool: list[RawEvent] = []
    for slot_events in slots.values():
        top = sorted(
            slot_events,
            key=lambda e: e.gate_result.n_activations if e.gate_result else 0,
            reverse=True,
        )[:_MAX_PER_SLOT]
        pool.extend(top)

    return pool


# ---------------------------------------------------------------------------
# Vectorized helpers
# ---------------------------------------------------------------------------

def _build_one_hot_f32(month_index: np.ndarray, n_months: int) -> np.ndarray:
    """Build a float32 one-hot month matrix for vectorized monthly counting.

    Constructs a matrix ``M`` of shape ``(n_rows, n_months)`` where
    ``M[r, m] = 1.0`` if row ``r`` belongs to calendar month ``m``.
    Pre-multiplying a boolean activation row vector by ``M`` produces
    per-month activation counts as a single BLAS ``gemv``/``gemm`` call,
    making batch counting of thousands of pairs fast.

    Parameters
    ----------
    month_index : np.ndarray
        Integer array of shape (n_rows,), as returned by
        ``_build_month_index``.  Each value is the zero-based month index
        for that row.
    n_months : int
        Total number of distinct calendar months.

    Returns
    -------
    np.ndarray
        Float32 matrix of shape (n_rows, n_months).
    """
    n_rows = len(month_index)
    one_hot = np.zeros((n_rows, n_months), dtype=np.float32)
    one_hot[np.arange(n_rows), month_index] = 1.0
    return one_hot


def _validity_mask(pool: list[RawEvent]) -> np.ndarray:
    """Build a boolean validity matrix for all pool pairs in pure numpy.

    Encodes the same two rejection rules as ``_is_valid_pair`` but
    operates over all N² pairs simultaneously via broadcasting, avoiding
    per-pair Python logic:

    1. **Native-type same source**: both transforms are in
       ``{binary_native, categorical_onehot}`` and source features match.
    2. **Same transform + same params + same source**: the AND would be
       equivalent to the stricter threshold alone.

    Parameters
    ----------
    pool : list[RawEvent]
        Composition candidate pool produced by ``_build_composition_pool``.

    Returns
    -------
    np.ndarray
        Boolean matrix of shape (n_pool, n_pool).  Entry ``[i, j]`` is
        True when the pair ``(pool[i], pool[j])`` is a valid composition
        candidate.  The upper triangle is what ``compose`` extracts with
        ``np.triu(mask, k=1)``.
    """
    n = len(pool)
    sources = np.array([ev.component.source_feature for ev in pool])
    transforms = np.array([ev.component.transform for ev in pool])
    # Params encoded as a single string for equality comparison
    params_str = np.array([
        "_".join(f"{k}{v}" for k, v in sorted(ev.component.transform_params.items()))
        for ev in pool
    ])

    native_types = {"binary_native", "categorical_onehot"}
    is_native = np.array([ev.component.transform in native_types for ev in pool])

    # Rule 1: both native + same source → invalid
    same_source = sources[:, None] == sources[None, :]  # (n, n)
    both_native = is_native[:, None] & is_native[None, :]
    native_same_source = both_native & same_source

    # Rule 2: same source + same transform + same params → invalid
    same_transform = transforms[:, None] == transforms[None, :]
    same_params = params_str[:, None] == params_str[None, :]
    subset_pair = same_source & same_transform & same_params

    invalid = native_same_source | subset_pair
    # Diagonal is also invalid (pair with itself)
    np.fill_diagonal(invalid, True)

    return ~invalid


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
