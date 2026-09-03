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

Enumeration order (#230): "no arbitrary pool cap" above is about the search
space, not the result — ``_MAX_PAIRS``/``_MAX_TRIPLES`` still cap what's
*returned*, and which candidates fill that cap depends on traversal order.
``np.where(np.triu(...))`` enumerates row-major: under permissive gate
params, where most examined pairs pass, the "stop once the cap is reached"
loop can exhaust every pair involving the first pool index before a second
one is ever tried — measured on a realistic pool: every one of 2000 kept
pairs sharing the same single component. ``_shuffle_order()`` permutes the
traversal (fixed seed — deterministic, not random per run) so the same
early-exit logic samples the whole pool instead; raising the caps was
evaluated as an alternative and rejected (real, non-trivial time/memory cost
without fixing the underlying skew — see issue #230).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..episodes import episode_starts
from .consistency_gate import ConsistencyGate, _build_month_index, _eff_max_dispersion, _gate_pass
from .models import EventCandidate, EventComponent, GateResult, RawEvent

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

# Fixed, deterministic seed used to permute pair/triple enumeration order
# (#230) — an arbitrary constant, not derived from wall-clock or any other
# non-deterministic source, so the same KPI table + config always produces
# the same composed events. See `_shuffle_order` for why this matters.
_PAIR_ORDER_SHUFFLE_SEED = 20260828


def _shuffle_order(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic permutation of ``range(n)`` (#230).

    ``ANDComposer.compose()``'s pair enumeration (row-major from
    ``np.where(np.triu(...))``) and each triple seed's third-candidate
    enumeration (ascending pool index) are otherwise structurally biased:
    combined with the "stop once ``_MAX_PAIRS``/``_MAX_TRIPLES`` found" early
    exit, the loop exhausts every combination involving the *first*
    enumerated index before ever trying a second one. Measured on a
    realistic pool (SUIUSDC, ~7000-10000 events post-gate) under permissive
    gate params: all 2000 kept pairs shared the same single "root" event.
    Permuting the traversal order — with a fixed seed, so still fully
    reproducible run to run for the same input — lets the same "first N
    found" logic sample the whole pool instead. Verified empirically:
    distinct source features touched went from 129 to 420 on one regime,
    102 to 503 on another, with no change to per-chunk cost.
    """
    if n <= 1:
        return np.arange(n)
    return rng.permutation(n)


def _stratified_pair_order(
    pool: list[RawEvent],
    ii_all: np.ndarray,
    jj_all: np.ndarray,
    stratify_fn: Callable[[RawEvent, RawEvent], Optional[str]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder valid pairs by ``stratify_fn``'s stratum, round-robin interleaved.

    Used by ``ANDComposer.compose()`` when a caller passes ``stratify_fn`` in
    place of the default flat ``_shuffle_order`` traversal (issue #254). Each
    valid pair ``(pool[i], pool[j])`` is assigned a stratum key via
    ``stratify_fn``; pairs whose key is ``None`` are dropped. Each stratum is
    then shuffled independently (same deterministic ``rng``, so still fully
    reproducible for a given input), and the strata are interleaved
    round-robin — one pair from each stratum in turn — rather than
    concatenated and truncated.

    This is deliberate, not cosmetic: concatenating strata and then applying
    a flat cap downstream systematically starves whichever stratum is
    smaller or later in iteration order, independent of (and in addition to)
    the single-root-domination bug #230 already fixed for the unstratified
    case.

    Parameters
    ----------
    pool : list[RawEvent]
        The composition pool ``ii_all``/``jj_all`` index into.
    ii_all, jj_all : np.ndarray
        Row/column indices of every structurally valid pair, as produced by
        ``np.where(np.triu(valid_mask, k=1))``.
    stratify_fn : callable
        ``(RawEvent, RawEvent) -> Optional[str]``, called once per valid
        pair.
    rng : np.random.Generator
        Shared, seeded generator — its state advances with each stratum's
        shuffle, in stratum-insertion order, so the result is deterministic
        for a given pool + stratify_fn.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Reordered (and possibly shortened) ``(ii, jj)``, ready to feed the
        same chunked pair-evaluation loop ``compose()`` already runs.
    """
    strata: dict[str, list[int]] = defaultdict(list)
    for pos in range(len(ii_all)):
        i, j = int(ii_all[pos]), int(jj_all[pos])
        key = stratify_fn(pool[i], pool[j])
        if key is not None:
            strata[key].append(pos)

    for positions in strata.values():
        perm = _shuffle_order(len(positions), rng)
        positions[:] = [positions[p] for p in perm]

    interleaved = [
        pos for group in zip_longest(*strata.values()) for pos in group if pos is not None
    ]
    order = np.array(interleaved, dtype=int)
    return ii_all[order], jj_all[order]


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
        max_constituent_jaccard: Optional[float] = None,
        pool_selector: Optional[Callable[[list[RawEvent]], list[RawEvent]]] = None,
        stratify_fn: Optional[Callable[[RawEvent, RawEvent], Optional[str]]] = None,
        max_pairs: Optional[int] = None,
        max_triples: Optional[int] = None,
        triple_third_filter: Optional[Callable[[RawEvent, RawEvent, RawEvent], bool]] = None,
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
           ``(ii_all, jj_all)``, then permuted with a fixed seed
           (``_shuffle_order``, #230) so "stop once ``_MAX_PAIRS`` found"
           samples the whole pool instead of exhausting one pool index's
           row before ever trying a second one — a real effect under
           permissive gate params, not a hypothetical one (measured: every
           kept pair sharing the same single component, pre-fix).

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
           ``k > idx_b`` are identified via the validity mask, permuted the
           same way as step 2 (#230), and processed with the same
           early-volume + sub-chunk matmul pattern:

           * ``and_ijk = and_ij[None,:] & bool_matrix[valid_k_chunk]``
           * Volume pre-filter on ``n_act_t``; skip matmul if nothing passes.
           * Full gate on sub-chunk where ``n_act_t >= min_act_floor``.

           This reduces the search space from ``O(n³)`` to
           ``O(n_vol_pairs × n_pool)`` with no pool cap.

        6. **Hard caps**: ``_MAX_PAIRS`` limits pair compositions and
           ``_MAX_TRIPLES`` limits triple compositions independently, so
           pairs never starve triples. Which specific pairs/triples fill
           those caps is what steps 2 and 5's shuffle (#230) — and the
           optional ``max_constituent_jaccard`` — control.

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
        max_constituent_jaccard : float or None, keyword-only
            Opt-in redundancy filter (#230), disabled (``None``, default) to
            preserve existing behaviour. When set, a pair whose two
            constituents' Jaccard similarity — ``|E1∩E2| / |E1∪E2|`` on their
            activation series, the same formula ``DiversityGate`` already
            uses for single events — exceeds this threshold is rejected
            before the (expensive) full gate check, at no extra cost: the
            intersection count is already computed by the cheap volume
            pre-filter. A triple's seed pair is rejected the same way (its
            own Jaccard checked once per seed); the third component is not
            separately checked against either seed constituent. Only
            meaningfully reduces redundancy once the traversal samples
            broadly — measured negligible effect on its own (see the
            enumeration-order fix above), a small real one layered on top of
            it (roughly 1-2% of examined pairs at a threshold of 0.5 on a
            realistic pool).
        pool_selector : callable or None, keyword-only
            Overrides pool construction (step 1). ``None`` (default) uses
            ``_build_composition_pool`` — today's structural slot-grouping
            (top ``_MAX_PER_SLOT`` events per (feature, transform, params)
            slot). A caller (e.g. a grade-guided composition stage running
            after Alpha Discovery, issue #254) can pass a different
            ``list[RawEvent] -> list[RawEvent]`` reduction instead. Every
            other step — validity, chunked gate, caps — is unaffected by
            this choice.
        stratify_fn : callable or None, keyword-only
            Overrides the *priority order* pairs are evaluated/kept in
            (issue #254). ``None`` (default) reproduces today's behaviour: a
            single flat, seed-deterministic shuffle of every valid pair
            (``_shuffle_order``, #230), so "stop once ``max_pairs`` is
            reached" samples the whole pool rather than exhausting one
            index's row first.

            When provided, ``stratify_fn(pool[i], pool[j])`` is called once
            per valid pair and must return a stratum key (any hashable, e.g.
            ``"A_same"``) or ``None`` to drop that pair entirely. Pairs are
            then grouped by key, each group independently shuffled with the
            same deterministic seed, and the groups **round-robin
            interleaved** — never concatenated-then-truncated — before the
            ``max_pairs``/``_MAX_PAIRS`` cap is applied downstream. This is
            deliberate: concatenating groups and truncating at a flat cap
            systematically starves whichever stratum happens to be smaller
            or later in iteration order, independent of and in addition to
            the single-root-domination bug #230 already fixed for the
            unstratified case.

            Does not change triple enumeration's own third-candidate order
            (each seed pair's ``valid_k`` is still shuffled by
            ``_shuffle_order`` alone) — only which pairs become seeds, and in
            what order, is affected.
        max_pairs, max_triples : int or None, keyword-only
            Override ``_MAX_PAIRS``/``_MAX_TRIPLES`` for this call. ``None``
            (default) keeps today's module-level caps (2000 / 500). A
            pairing policy that evaluates a much larger candidate pool (e.g.
            grade-guided composition, issue #254) may need to raise these.
        triple_third_filter : callable or None, keyword-only
            Constrains a triple's *third* component beyond ``_validity_mask``
            (issue #254 Phase 4). ``None`` (default) reproduces today's
            behaviour: any structurally-valid third candidate is eligible,
            in an order fixed only by ``_shuffle_order`` (unaffected by
            ``stratify_fn``, which governs pair order/seeding only, not the
            third-candidate search within one seed).

            When provided, ``triple_third_filter(pool[idx_a], pool[idx_b],
            pool[k])`` is called once per structurally-valid third candidate
            of each seed pair; a third failing the filter (returning falsy)
            is dropped before the shuffle, exactly like ``stratify_fn``
            returning ``None`` drops a pair. A caller building a grade-guided
            triple policy would key this off the seed pair's root grade, not
            the third's own pairwise relation to each seed member
            individually — see ``forgedge.composition.grade_guided_compose``.

            Does not change which (seed-pair, third) triples are
            *structurally* reachable in the first place: a unique index
            triple ``{p, q, r}`` (``p < q < r``) is only ever generated from
            seed ``(p, q)`` with third ``r`` — the ``k > idx_b`` ordering
            constraint combined with seeds always satisfying ``idx_a <
            idx_b`` already rules out any other seed producing the same
            triple, so no deduplication step is needed here or in a filter
            built on top of this hook.

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

        select_pool = pool_selector if pool_selector is not None else _build_composition_pool
        pool = select_pool(passing_events)
        if len(pool) < 2:
            return []

        effective_max_pairs = _MAX_PAIRS if max_pairs is None else max_pairs
        effective_max_triples = _MAX_TRIPLES if max_triples is None else max_triples

        one_hot = _build_one_hot_f32(month_index, n_total_months)
        bool_matrix = np.stack(
            [ev.series.fillna(0).values.astype(np.uint8) for ev in pool]
        )  # (n_pool, n_rows)
        # Only materialized to support max_constituent_jaccard (#230); the
        # per-pair sum it enables costs nothing extra there (reuses the Pass 1
        # intersection count), but this reduction itself is O(n_pool x n_rows)
        # so it's cheap regardless of whether the filter is active.
        row_sums = bool_matrix.sum(axis=1).astype(np.float64)
        # Bounded against n_rows so a full chunk's worst-case episode-mode
        # computation can't OOM regardless of dataset length (#228).
        chunk_size = _pair_chunk_size(bool_matrix.shape[1])
        # Reused (state advances) across the pair loop and every triple
        # seed's third-candidate shuffle below — deterministic given the
        # fixed seed and the fixed call order for a given input (#230).
        shuffle_rng = np.random.default_rng(_PAIR_ORDER_SHUFFLE_SEED)

        pairs: list[RawEvent] = []
        triples: list[RawEvent] = []

        valid_mask = _validity_mask(pool)
        ii_all, jj_all = np.where(np.triu(valid_mask, k=1))
        if stratify_fn is None:
            n_pairs = len(ii_all)
            pair_perm = _shuffle_order(n_pairs, shuffle_rng)
            ii_all, jj_all = ii_all[pair_perm], jj_all[pair_perm]
        else:
            ii_all, jj_all = _stratified_pair_order(pool, ii_all, jj_all, stratify_fn, shuffle_rng)
        n_pairs = len(ii_all)

        vol_passing_ii: list[int] = []
        vol_passing_jj: list[int] = []

        # ----------------------------------------------------------------
        # Pair enumeration — two-pass: cheap volume filter, then matmul
        # ----------------------------------------------------------------
        for chunk_start in range(0, n_pairs, chunk_size):
            if len(pairs) >= effective_max_pairs:
                break
            chunk_end = min(chunk_start + chunk_size, n_pairs)
            ii = ii_all[chunk_start:chunk_end]
            jj = jj_all[chunk_start:chunk_end]

            # Pass 1: cheap uint8 AND + activation count
            and_chunk = bool_matrix[ii] & bool_matrix[jj]      # (K, n_rows)
            n_act = and_chunk.sum(axis=1).astype(np.int32)     # (K,)
            vol_ok = n_act >= min_act_floor

            if max_constituent_jaccard is not None:
                union = row_sums[ii] + row_sums[jj] - n_act
                jaccard = np.where(union > 0, n_act / union, 0.0)
                vol_ok = vol_ok & (jaccard <= max_constituent_jaccard)

            # Collect volume-passing seeds for triple enumeration
            if max_components >= 3:
                vol_passing_ii.extend(ii[vol_ok].tolist())
                vol_passing_jj.extend(jj[vol_ok].tolist())

            # Pass 2: full gate only where volume (+ Jaccard, if set) passes
            sub_idx = np.where(vol_ok)[0]
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

            remaining = effective_max_pairs - len(pairs)
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
                if len(triples) >= effective_max_triples:
                    break

                and_ij = bool_matrix[idx_a] & bool_matrix[idx_b]  # (n_rows,)

                # k > idx_b ensures unique triples (seed enumerated as i < j)
                valid_k = np.where(valid_mask[idx_a] & valid_mask[idx_b])[0]
                valid_k = valid_k[valid_k > idx_b]
                if len(valid_k) == 0:
                    continue
                if triple_third_filter is not None:
                    valid_k = np.array(
                        [k for k in valid_k
                         if triple_third_filter(pool[idx_a], pool[idx_b], pool[k])],
                        dtype=valid_k.dtype,
                    )
                    if len(valid_k) == 0:
                        continue
                # Same enumeration-order fix as the pair loop, scoped to this
                # seed's own third-candidates (#230) — a single seed's
                # ascending-index valid_k can otherwise fill all of
                # _MAX_TRIPLES on its own before a second seed is ever tried.
                valid_k = valid_k[_shuffle_order(len(valid_k), shuffle_rng)]

                for k_start in range(0, len(valid_k), chunk_size):
                    if len(triples) >= effective_max_triples:
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

                    remaining = effective_max_triples - len(triples)
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


def raw_event_from_candidate(candidate: EventCandidate) -> RawEvent:
    """Adapt a single-component ``EventCandidate`` back into a ``RawEvent``.

    The standalone entry point for invoking ``ANDComposer.compose()`` on a
    list of ``EventCandidate`` objects rather than only on the ``RawEvent``s
    ``EventDiscovery.run()``'s own Step 5 produces internally — e.g. a
    grade-guided composition stage running after Alpha Discovery's first
    pass (issue #254), where the input is M1's promoted ``EventCandidate``
    pool, not raw pre-promotion events.

    Parameters
    ----------
    candidate : EventCandidate
        A single-component candidate (``len(candidate.components) == 1``).
        An already-composed candidate cannot be composed further this way —
        see Raises.

    Returns
    -------
    RawEvent
        ``series=candidate.event_series``,
        ``component=candidate.components[0]``,
        ``gate_result=candidate.consistency_gate``.

    Raises
    ------
    ValueError
        If ``candidate`` has more than one component (i.e. is itself already
        an AND composition).
    """
    if len(candidate.components) != 1:
        raise ValueError(
            "raw_event_from_candidate expects a single-component candidate, "
            f"got {len(candidate.components)} components on "
            f"{candidate.event_id!r}"
        )
    return RawEvent(
        series=candidate.event_series,
        component=candidate.components[0],
        gate_result=candidate.consistency_gate,
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
