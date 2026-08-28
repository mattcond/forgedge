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
Both pair and triple evaluation are fully vectorized with no arbitrary
pool cap.

Pairs: all valid pairs are chunked through a numpy/BLAS batch loop.
Within each chunk an *early volume pre-filter* computes
``n_act = and_chunk.sum(axis=1)`` on the cheap uint8 representation and
skips the expensive float32 matmul entirely for chunks where no pair
exceeds ``min_act_floor`` (derived as ``max(int(min_tpm * n_total_months), 1)``).
For sparse financial signals (~3 % activation rate) this saves ~85 % of
the matmul work.  When volume-passing pairs exist the matmul is performed
only on the sub-chunk ``and_chunk[vol_mask]``.

Triples: enumeration uses *pair-seeded pruning*.  Since
``and(i,j,k) ⊆ and(i,j)`` pointwise, any triple whose seed pair already
fails the volume criterion is provably impossible to pass the gate.  Only
pairs where ``n_act(and(i,j)) >= min_act_floor`` are used as seeds.  For
each seed pair the inner loop over all valid third events is vectorized with
the same early-volume + sub-chunk matmul pattern.  This reduces the
effective search space from ``O(n³)`` to ``O(n_vol_pairs × n_pool)`` and
makes uncapped triple evaluation practical.

Chunk size (#228): the volume pre-filter's "~85% of the matmul work" saving
above assumes a sparse activation rate, which permissive gate params (a low
``min_tpm``) invalidate — most of a chunk can pass, and the episode-mode
computation (#226) then runs on it at full chunk size. ``_pair_chunk_size()``
bounds the chunk size itself against ``n_rows`` so that worst case stays
within a fixed memory budget regardless of dataset length, rather than
relying on the pre-filter alone to keep sub-chunks small.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from ..episodes import episode_starts
from .consistency_gate import ConsistencyGate, _build_month_index, _eff_max_dispersion, _gate_pass
from .models import EventComponent, GateResult, RawEvent

# Sentinel used as the default for the ``gate`` parameter of
# ``ANDComposer.compose()``.  Distinguishes "caller did not pass gate"
# (→ use self.gate, full gate applied) from "caller explicitly passed None"
# (→ skip full gate, return all volume-passing compositions).
_COMPOSE_DEFAULT: "ConsistencyGate" = object()  # type: ignore[assignment]

# Number of (i, j) pairs processed per vectorized batch.  Larger values
# consume more peak memory; smaller values add Python loop overhead.
# At 5 000 pairs with n_rows ≈ 8 760 (1 year 1H) the uint8 AND costs ~44 MB
# and the float32 matmul on volume-passing pairs (typically ≪ 5000) is negligible.
#
# That "typically ≪ 5000" assumption is exactly what breaks under permissive
# gate params (#228): the episode-mode computation #226 added
# (`_episode_stats`/`episode_starts`) runs on the *volume-passing* sub-chunk,
# and a low `min_tpm` (e.g. the library's own `GateParams()` default) lets
# most of a chunk pass that pre-filter, so the sub-chunk can reach full
# `_CHUNK_SIZE` — measured ~2.2 GB peak per chunk at `_CHUNK_SIZE=5000` on a
# realistic 23 352-row (~2.7y 1H) dataset, enough to OOM a real
# `ANDComposer.compose()` call.  `_pair_chunk_size()` below is the actual
# per-call chunk size — it bounds this worst case against a fixed memory
# budget instead of assuming a small sub-chunk; `_CHUNK_SIZE` is now only its
# upper bound (unchanged behaviour on the short histories the number above
# was calibrated against).
_CHUNK_SIZE = 5_000

# Target peak bytes for one chunk's worst-case (full sub-chunk) episode-mode
# computation (#228).  ~20 bytes/cell was measured for `_episode_stats` on a
# realistic chunk after the #228 dtype/cleanup fix in `episodes.py`; the
# ~25% margin covers the pair/triple loops' own smaller (K, n_rows) and
# (K, n_months) temporaries alongside it.
_EPISODE_CHUNK_BUDGET_BYTES = 1_000_000_000  # ~1 GB
_EPISODE_BYTES_PER_CELL = 25

# Maximum number of events to retain per (feature, transform) slot
# before cross-feature AND composition.  Within-slot events differ only
# in their threshold; keeping at most 3 preserves diversity while
# containing the total pair count.
_MAX_PER_SLOT = 3

# Maximum pair events returned (2-component AND compositions).
_MAX_PAIRS = 2000

# Maximum triple events returned (3-component AND compositions).
# Independent of _MAX_PAIRS so pairs never starve triples.
_MAX_TRIPLES = 500


def _pair_chunk_size(n_rows: int) -> int:
    """Chunk size (``K``, candidates per vectorized batch) for
    ``ANDComposer.compose()``'s pair and triple loops (#228).

    Bounded so a full chunk's worst-case episode-mode computation
    (``_episode_stats``, #226) stays within ``_EPISODE_CHUNK_BUDGET_BYTES``
    regardless of ``n_rows`` — a fixed ``_CHUNK_SIZE`` implicitly assumed the
    volume pre-filter always thins a chunk down before that computation runs,
    which permissive gate params (low ``min_tpm``) make false.  Equals
    ``_CHUNK_SIZE`` for the short histories that constant was calibrated
    against and shrinks for longer ones.

    Parameters
    ----------
    n_rows : int
        Number of bars in the series being composed (``len(timestamps)``).

    Returns
    -------
    int
        Chunk size to use, ``1 <= result <= _CHUNK_SIZE``.
    """
    if n_rows <= 0:
        return _CHUNK_SIZE
    budget_chunk = _EPISODE_CHUNK_BUDGET_BYTES // (n_rows * _EPISODE_BYTES_PER_CELL)
    return min(_CHUNK_SIZE, max(1, budget_chunk))


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
        *,
        gate: Optional[ConsistencyGate] = _COMPOSE_DEFAULT,
    ) -> list[RawEvent]:
        """Generate valid AND compositions and return those passing the gate.

        Algorithm
        ---------
        1. **Pool construction**: ``_build_composition_pool`` reduces the
           full set of passing events to a tractable subset by keeping at
           most ``_MAX_PER_SLOT`` events per (source_feature, transform,
           params) slot.  No global pool cap is applied.

        2. **Validity pre-filter**: ``_validity_mask`` builds a boolean
           matrix of shape (n_pool, n_pool) in pure numpy via broadcasting.
           The upper triangle of valid pairs is extracted as index arrays
           ``(ii_all, jj_all)`` with no per-pair Python logic.

        3. **One-hot month matrix**: ``_build_one_hot_f32`` constructs a
           (n_rows, n_months) float32 matrix where ``M[r, m] = 1`` if row
           ``r`` belongs to month ``m``.  Used in the BLAS matmul for
           monthly counting.

        4. **Vectorized pair gate** (max_components >= 2): Pairs are
           processed in chunks of ``_pair_chunk_size(n_rows)`` (#228) — at
           most ``_CHUNK_SIZE``, shrunk for long histories so a chunk's
           worst-case episode-mode computation stays within a fixed memory
           budget.  Each chunk uses a two-pass evaluation:

           * **Pass 1** (cheap): ``and_chunk = bool_matrix[ii] & bool_matrix[jj]``
             (uint8 AND), then ``n_act = and_chunk.sum(axis=1)`` to identify
             volume-passing pairs.  If no pair exceeds ``min_act_floor``, the
             entire chunk skips the matmul.

           * **Pass 2** (when needed): the matmul
             ``and_chunk[vol_mask].astype(float32) @ one_hot``
             is performed only on the volume-passing sub-chunk, making it
             cheap when volume-passing pairs are rare (sparse signals).

           While processing pair chunks, any pair where
           ``n_act >= min_act_floor`` is also recorded as a *volume-passing seed*
           for triple enumeration.

        5. **Pair-seeded triple gate** (max_components >= 3): For each
           volume-passing pair ``(idx_a, idx_b)``, all valid third indices
           ``k > idx_b`` are identified via the validity mask and processed
           with the same early-volume + sub-chunk matmul pattern:

           * ``and_ijk = and_ij[None,:] & bool_matrix[valid_k_chunk]``
           * Volume pre-filter on ``n_act_t``; skip matmul if nothing passes.
           * Full gate on sub-chunk where ``n_act_t >= min_act_floor``.

           This reduces the search space from ``O(n³)`` to
           ``O(n_vol_pairs × n_pool)`` with no pool cap.

        6. **Hard caps**: ``_MAX_PAIRS`` limits pair compositions and
           ``_MAX_TRIPLES`` limits triple compositions independently, so
           pairs never starve triples.

        Parameters
        ----------
        passing_events : list[RawEvent]
            Atoms to compose.  In the standard FORGE pipeline these are
            events that passed the ConsistencyGate; in TargetOptimizer they
            may be pre-filtered by lift score instead.
        timestamps : pd.Series
            Datetime series aligned to the KPI table rows.
        max_components : int
            Maximum number of components per composed event.
            ``1`` disables AND composition entirely (returns ``[]``).
            ``2`` generates pairs only.  ``3`` generates pairs and triples.
        gate : ConsistencyGate or None, keyword-only
            Controls the full ConsistencyGate evaluation during composition.

            - **Omitted** (default): uses ``self.gate``; composed events are
              returned only when they pass all four criteria (volume, coverage,
              concentration, frequency).  Standard FORGE pipeline behaviour.
            - **``None``**: the cheap volume pre-filter (``min_act`` from
              ``self.gate.params``) still runs, but the full gate check is
              skipped.  All volume-passing compositions are returned with
              ``gate_result.passed=False``; the caller is responsible for a
              subsequent ``ConsistencyGate.filter()`` pass.
            - **``ConsistencyGate`` instance**: uses that gate for the full
              check, overriding ``self.gate`` for this call only.

        Returns
        -------
        list[RawEvent]
            AND-composed events.  When *gate* is provided (or defaulted to
            ``self.gate``), only events that passed the ConsistencyGate are
            returned.  When *gate* is ``None``, all volume-passing events are
            returned with ``gate_result.passed=False``.
        """
        if not passing_events or max_components < 2:
            return []

        _gate = self.gate if gate is _COMPOSE_DEFAULT else gate
        # When gate=None, still use self.gate.params for the volume pre-filter
        p = (_gate or self.gate).params
        month_index, n_total_months = _build_month_index(timestamps)
        n_months_dof = max(n_total_months - 1, 1)
        min_act_floor = max(int(p.min_tpm * n_total_months), 1)
        # Episode-mode dispersion threshold — one number for this whole call
        # (depends only on n_total_months), shared with ConsistencyGate.evaluate()
        # via the same function so a composed event is judged by the same
        # criteria as a single one, mode-aware (#226, the tail of #205).
        eff_max_dispersion = _eff_max_dispersion(n_total_months, p.dispersion_margin)

        pool = _build_composition_pool(passing_events)
        if len(pool) < 2:
            return []

        one_hot = _build_one_hot_f32(month_index, n_total_months)
        bool_matrix = np.stack(
            [ev.series.fillna(0).values.astype(np.uint8) for ev in pool]
        )  # (n_pool, n_rows)
        # Bounded against n_rows so a full chunk's worst-case episode-mode
        # computation can't OOM regardless of dataset length (#228).
        chunk_size = _pair_chunk_size(bool_matrix.shape[1])

        pairs: list[RawEvent] = []
        triples: list[RawEvent] = []

        valid_mask = _validity_mask(pool)
        ii_all, jj_all = np.where(np.triu(valid_mask, k=1))
        n_pairs = len(ii_all)

        vol_passing_ii: list[int] = []
        vol_passing_jj: list[int] = []

        # ----------------------------------------------------------------
        # Pair enumeration — two-pass: cheap volume filter, then matmul
        # ----------------------------------------------------------------
        for chunk_start in range(0, n_pairs, chunk_size):
            if len(pairs) >= _MAX_PAIRS:
                break
            chunk_end = min(chunk_start + chunk_size, n_pairs)
            ii = ii_all[chunk_start:chunk_end]
            jj = jj_all[chunk_start:chunk_end]

            # Pass 1: cheap uint8 AND + activation count
            and_chunk = bool_matrix[ii] & bool_matrix[jj]      # (K, n_rows)
            n_act = and_chunk.sum(axis=1).astype(np.int32)     # (K,)

            # Collect volume-passing seeds for triple enumeration
            if max_components >= 3:
                vol_seed = n_act >= min_act_floor
                vol_passing_ii.extend(ii[vol_seed].tolist())
                vol_passing_jj.extend(jj[vol_seed].tolist())

            # Pass 2: full gate only where volume passes — skip matmul otherwise
            sub_idx = np.where(n_act >= min_act_floor)[0]
            if len(sub_idx) == 0:
                continue

            counts_sub = and_chunk[sub_idx].astype(np.float64) @ one_hot  # (S, n_months)
            n_act_sub = n_act[sub_idx].astype(np.float64)
            n_active_sub = (counts_sub > 0).sum(axis=1)  # diagnostic only
            max_conc_sub = counts_sub.max(axis=1) / n_act_sub  # diagnostic only
            mean_tpm_sub = n_act_sub / n_total_months
            # Per-bar Index of Dispersion — diagnostic in "episode" mode, the
            # gating quantity in "bar" mode.
            sum_sq_dev = ((counts_sub - mean_tpm_sub[:, None]) ** 2).sum(axis=1)
            var_sub = sum_sq_dev / n_months_dof
            id_sub = np.where(mean_tpm_sub > 0, var_sub / mean_tpm_sub, np.inf)
            # Episode-level stats — the "episode"-mode gating quantities
            # (#226): same episode_starts() bridging ConsistencyGate uses for
            # single events, batched across this sub-chunk.
            n_episodes_sub, episode_tpm_sub, episode_id_sub = _episode_stats(
                and_chunk[sub_idx].astype(bool), p.episode_gap, n_total_months, one_hot
            )

            if _gate is not None:
                gate_pass = _gate_pass(
                    p,
                    mean_tpm=mean_tpm_sub, id_score=id_sub,
                    episode_tpm=episode_tpm_sub, n_episodes=n_episodes_sub,
                    episode_id=episode_id_sub, eff_max_dispersion=eff_max_dispersion,
                )
                passing = np.where(gate_pass)[0]
            else:
                passing = np.arange(len(sub_idx))

            remaining = _MAX_PAIRS - len(pairs)
            if len(passing) > remaining:
                passing = passing[:remaining]

            for k in passing:
                orig = sub_idx[k]
                gate_result = GateResult(
                    passed=(_gate is not None),
                    n_activations=int(n_act_sub[k]),
                    n_active_months=int(n_active_sub[k]),
                    max_monthly_share=float(max_conc_sub[k]),
                    mean_tpm=float(mean_tpm_sub[k]),
                    index_of_dispersion=float(id_sub[k]),
                    n_episodes=int(n_episodes_sub[k]),
                    episode_index_of_dispersion=float(episode_id_sub[k]),
                )
                and_series = pd.Series(
                    and_chunk[orig].astype(float), index=pool[ii[orig]].series.index
                )
                pairs.append(
                    _make_composed_event(pool[ii[orig]], pool[jj[orig]], and_series, gate_result)
                )

        # ----------------------------------------------------------------
        # Triple enumeration — pair-seeded, two-pass, no pool cap
        # ----------------------------------------------------------------
        if max_components >= 3 and vol_passing_ii:
            for idx_a, idx_b in zip(vol_passing_ii, vol_passing_jj):
                if len(triples) >= _MAX_TRIPLES:
                    break

                and_ij = bool_matrix[idx_a] & bool_matrix[idx_b]  # (n_rows,)

                # k > idx_b ensures unique triples (seed enumerated as i < j)
                valid_k = np.where(valid_mask[idx_a] & valid_mask[idx_b])[0]
                valid_k = valid_k[valid_k > idx_b]
                if len(valid_k) == 0:
                    continue

                for k_start in range(0, len(valid_k), chunk_size):
                    if len(triples) >= _MAX_TRIPLES:
                        break
                    k_end = min(k_start + chunk_size, len(valid_k))
                    k_chunk = valid_k[k_start:k_end]

                    # Pass 1: cheap volume check
                    and_ijk = and_ij[None, :] & bool_matrix[k_chunk]  # (K, n_rows)
                    n_act_t = and_ijk.sum(axis=1).astype(np.int32)

                    sub_t = np.where(n_act_t >= min_act_floor)[0]
                    if len(sub_t) == 0:
                        continue

                    # Pass 2: full gate on volume-passing subset
                    counts_t = and_ijk[sub_t].astype(np.float64) @ one_hot
                    n_act_s = n_act_t[sub_t].astype(np.float64)
                    n_active_s = (counts_t > 0).sum(axis=1)  # diagnostic only
                    max_conc_s = counts_t.max(axis=1) / n_act_s  # diagnostic only
                    mean_tpm_s = n_act_s / n_total_months
                    sum_sq_dev_t = ((counts_t - mean_tpm_s[:, None]) ** 2).sum(axis=1)
                    var_t = sum_sq_dev_t / n_months_dof
                    id_t = np.where(mean_tpm_s > 0, var_t / mean_tpm_s, np.inf)
                    # Episode-level stats (#226) — same treatment as the pair loop.
                    n_episodes_t, episode_tpm_t, episode_id_t = _episode_stats(
                        and_ijk[sub_t].astype(bool), p.episode_gap, n_total_months, one_hot
                    )

                    if _gate is not None:
                        gate_t = _gate_pass(
                            p,
                            mean_tpm=mean_tpm_s, id_score=id_t,
                            episode_tpm=episode_tpm_t, n_episodes=n_episodes_t,
                            episode_id=episode_id_t, eff_max_dispersion=eff_max_dispersion,
                        )
                        passing_t = np.where(gate_t)[0]
                    else:
                        passing_t = np.arange(len(sub_t))

                    remaining = _MAX_TRIPLES - len(triples)
                    if len(passing_t) > remaining:
                        passing_t = passing_t[:remaining]

                    for m in passing_t:
                        orig_m = sub_t[m]
                        gate_result = GateResult(
                            passed=(_gate is not None),
                            n_activations=int(n_act_s[m]),
                            n_active_months=int(n_active_s[m]),
                            max_monthly_share=float(max_conc_s[m]),
                            mean_tpm=float(mean_tpm_s[m]),
                            index_of_dispersion=float(id_t[m]),
                            n_episodes=int(n_episodes_t[m]),
                            episode_index_of_dispersion=float(episode_id_t[m]),
                        )
                        and_series = pd.Series(
                            and_ijk[orig_m].astype(float),
                            index=pool[idx_a].series.index,
                        )
                        triples.append(
                            _make_composed_event(
                                pool[idx_a],
                                pool[idx_b],
                                and_series,
                                gate_result,
                                third=pool[k_chunk[orig_m]],
                            )
                        )

        return pairs + triples

    # ------------------------------------------------------------------
    # Validity checks (kept for external / testing use)
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
        if (ca.transform in ("binary_native", "categorical_onehot")
                and cb.transform in ("binary_native", "categorical_onehot")
                and ca.source_feature == cb.source_feature):
            return False
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

    No global pool cap is applied.  The vectorized batch gate in
    ``ANDComposer.compose`` handles arbitrarily large pools efficiently
    through chunked numpy operations with an early volume pre-filter.

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

def _episode_stats(
    and_bool: np.ndarray, gap: int, n_total_months: int, one_hot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched episode rate/count/dispersion for a ``(K, n_rows)`` boolean
    AND matrix — the same episode-counting semantics
    ``ConsistencyGate.evaluate`` uses for single events (``episode_starts``'s
    gap-bridging, monthly episode counts via the same one-hot matmul already
    used for bar counts), vectorized across the whole chunk (#226).

    NaN in ``episode_id`` reproduces ``ConsistencyGate.evaluate``'s own NaN
    cases exactly — ``n_total_months <= 1`` or zero episodes — so
    ``_gate_pass``'s NaN handling (skip, not fail, the dispersion criterion)
    applies identically whether the caller is this function or the
    single-event path.

    Parameters
    ----------
    and_bool : np.ndarray
        Boolean AND-composition matrix, shape ``(K, n_rows)``.
    gap : int
        ``GateParams.episode_gap``.
    n_total_months : int
        Total calendar months spanned by the dataset (one value for the
        whole ``compose()`` call, not per-candidate).
    one_hot : np.ndarray
        ``(n_rows, n_total_months)`` one-hot month matrix from
        ``_build_one_hot_f32``, reused for the episode-count matmul.

    Returns
    -------
    n_episodes : np.ndarray[int64], shape (K,)
    episode_tpm : np.ndarray[float64], shape (K,)
    episode_id : np.ndarray[float64], shape (K,)
        NaN where not evaluated (``n_total_months <= 1`` or zero episodes).
    """
    starts = episode_starts(and_bool, gap)
    n_episodes = starts.sum(axis=1).astype(np.int64)
    episode_tpm = (
        n_episodes.astype(np.float64) / n_total_months
        if n_total_months > 0 else np.zeros(and_bool.shape[0], dtype=np.float64)
    )

    if n_total_months > 1:
        epi_counts = starts.astype(np.float64) @ one_hot
        emu = epi_counts.mean(axis=1)
        evar = epi_counts.var(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            episode_id = np.where(emu > 0, evar / emu, np.inf)
        episode_id = np.where(n_episodes > 0, episode_id, np.nan)
    else:
        episode_id = np.full(and_bool.shape[0], np.nan)

    return n_episodes, episode_tpm, episode_id


def _build_one_hot_f32(month_index: np.ndarray, n_months: int) -> np.ndarray:
    """Build a float32 one-hot month matrix for vectorized monthly counting.

    Constructs a matrix ``M`` of shape ``(n_rows, n_months)`` where
    ``M[r, m] = 1.0`` if row ``r`` belongs to calendar month ``m``.
    Pre-multiplying a boolean activation sub-chunk by ``M`` produces
    per-month activation counts as a single BLAS ``gemm`` call.  The matrix
    is only used when at least one pair in a chunk exceeds the volume
    threshold (early volume pre-filter).

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

    The resulting matrix is reused for both pair enumeration (upper triangle
    extraction) and triple extension (row lookup for valid k values).

    Parameters
    ----------
    pool : list[RawEvent]
        Composition candidate pool produced by ``_build_composition_pool``.

    Returns
    -------
    np.ndarray
        Boolean matrix of shape (n_pool, n_pool).  Entry ``[i, j]`` is
        True when the pair ``(pool[i], pool[j])`` is a valid composition
        candidate.
    """
    sources = np.array([ev.component.source_feature for ev in pool])
    transforms = np.array([ev.component.transform for ev in pool])
    params_str = np.array([
        "_".join(f"{k}{v}" for k, v in sorted(ev.component.transform_params.items()))
        for ev in pool
    ])

    native_types = {"binary_native", "categorical_onehot"}
    is_native = np.array([ev.component.transform in native_types for ev in pool])

    same_source = sources[:, None] == sources[None, :]
    both_native = is_native[:, None] & is_native[None, :]
    native_same_source = both_native & same_source

    same_transform = transforms[:, None] == transforms[None, :]
    same_params = params_str[:, None] == params_str[None, :]
    subset_pair = same_source & same_transform & same_params

    invalid = native_same_source | subset_pair
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
    and stores the constituent components in the ``components`` field so that
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

    sql_parts = [c.sql_expression for c in components if c.sql_expression]
    sql_expr = " AND ".join(f"({s})" for s in sql_parts) if sql_parts else ""

    formula_parts = [c.event_formula for c in components if c.event_formula]
    formula_expr = " AND ".join(f"({f})" for f in formula_parts) if formula_parts else ""

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
        event_formula=formula_expr,
        sql_expression=sql_expr,
        components=components,
    )

    ev = RawEvent(series=and_series, component=comp)
    ev.gate_result = gate_result
    return ev
