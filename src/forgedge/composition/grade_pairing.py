"""Grade-guided pairing/composition (issue #254, Phases 2 and 4).

Replaces M1's purely structural AND-pairing criterion (tpm, dispersion,
``transform_key``) with the A-D letter grade Alpha Discovery's first pass
already assigns to each 1D event — empirically a much stronger pairing
signal than structural correlation (see the issue's own AMZN 1D experiment:
122 PARTIAL-EDGE/EDGE contracts vs. 5 from structural pairing on the same
data).

Pairing scheme: same grade first (``A_same``, ``B_same``, ...), then
adjacent grade via a root+partner scheme read off
``GradePairingConfig.adjacency`` (default A<->{A,B}, B<->{B,C}, C<->{C,D} —
D is never a root, only ever reached as B/C's partner).

Each grade stratum is composed via its **own, independent**
``ANDComposer.compose()`` call (one call per ``(stratum_key, grade_set)``
from :func:`_enumerate_strata`), rather than one shared call across the
whole pool with a stratify hook approximating fairness through round-robin
interleaving. Two problems measured on a real pipeline run (DAAX 1H,
``docs/analysis/exhaustive_and_composition_coverage_proposal.md``) motivate
this:

* **Cross-process non-determinism.** ``ANDComposer``'s own pair/triple
  traversal shuffle uses a fixed seed (issue #230), which only guarantees
  reproducibility *given* a fixed pool order — and the pool's incoming
  order depends on ``FeatureGenerator``'s internal ``set()`` iteration,
  which is hash-seed-randomized per Python process (issue #24). Two
  identical runs of the same config on the same data produced two
  different "best" composed rules purely because the process restarted.
  Sorting the pool by a canonical key (``RawEvent.component.expression``,
  a deterministic string) before any further processing removes this
  dependency entirely — same input, same output, independent of process.
* **Selection diversity collapses without an explicit constraint.** A
  shared cap across a stratify-keyed round-robin lets whichever
  feature-pair combinations happen to be visited first (or score best,
  if ranked) dominate — measured: a plain merit ranking (tpm-closeness +
  low dispersion + episode count) kept only 49/303 distinct
  source-feature-pair combinations (16%) versus 149/303 (49%) from random
  sampling of the same size, the same "single dominant component" failure
  mode issue #230 already had to fix once for the raw pair-index shuffle.
  Stratifying **within** each grade stratum's own ``compose()`` call by
  ``(source_feature_a, source_feature_b)`` — reusing
  ``ANDComposer.compose()``'s existing ``stratify_fn`` hook, unmodified —
  fixes this the same way #230 fixed root-index domination: round-robin
  across feature-pair groups before the per-stratum cap is reached, so a
  single feature combination's threshold variants can no longer crowd out
  every other combination. Measured with the deduplication in place: 231
  /276 distinct feature-pair combinations (84%) at comparable or better
  quality (dispersion 0.673 vs 0.987 for the random baseline).

One more refinement falls out of the same measurements for free: grades
``A``/``B`` are rare by construction (the letter grade is a strict ranking,
and A/B are the top of it), so any stratum rooted only in
``GradePairingConfig.exhaustive_grades`` (default ``{A, B}``) has, in
practice, far fewer admissible pairs than the cap — "exhaustive" for those
strata costs nothing extra; it's what ``min(cap, available)`` already gives
once each stratum gets its own independent budget instead of sharing one
global cap. The strata that actually need sampling (anything touching C or
D — ``D_same`` alone was 97.6% of the admissible pair space measured on
DAAX) get ``per_stratum_pair_cap``/``per_stratum_triple_cap`` as before, now
combined with the feature-pair stratification above.

See ``docs/analysis/exhaustive_and_composition_coverage_proposal.md`` for
the full empirical writeup this design is based on.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

from ..alpha_discovery.models import AlphaContract
from ..event_discovery.and_composer import ANDComposer, raw_event_from_candidate
from ..event_discovery.consistency_gate import ConsistencyGate
from ..event_discovery.discovery import raw_event_to_candidate
from ..event_discovery.models import EventCandidate, RawEvent

__all__ = ["GradePairingConfig", "grade_guided_compose"]


@dataclass
class GradePairingConfig:
    """Policy for :func:`grade_guided_compose`.

    Attributes
    ----------
    max_components : int
        ``2`` (default) composes pairs only; ``3`` also composes triples
        (issue #254 Phase 4) — a triple's third component is constrained by
        the seed pair's own *root* grade (the better, alphabetically-first
        of the two — A<B<C<D), via the same adjacency relation that governs
        pairs, not merely by structural admissibility. Passed straight
        through to ``ANDComposer.compose()``.
    adjacency : dict[str, tuple[str, ...]]
        For each root grade, which grades (including itself) it may pair
        with. Default ``{"A": ("A","B"), "B": ("B","C"), "C": ("C","D")}`` —
        the scheme the issue's own experiment used. Symmetric: an ``(X, Y)``
        pair is allowed whenever ``Y`` is listed under ``X`` *or* ``X`` is
        listed under ``Y`` (so ``"B": ("B","C")`` alone is enough to also
        allow an A-graded event to pair with B, via A's own entry).
    per_stratum_pair_cap : int
        Maximum pairs composed from a single grade stratum (e.g. ``B_C``),
        applied independently per stratum — a stratum's budget is never
        shared with, or reduced by, any other stratum's population. Ignored
        for a stratum whose grades are all in ``exhaustive_grades`` (see
        below). Default ``100``.
    per_stratum_triple_cap : int
        Same, for triples (only relevant when ``max_components >= 3``).
        Default ``50``.
    include_singles_in_pass2 : bool
        Whether the caller (``forge()``'s two-pass orchestration, Phase 3)
        should pool the original 1D candidates alongside the composed ones
        for Alpha Discovery's second pass, matching today's implicit
        ``all_passing = passing_single + passing_composed`` in M1's own
        Step 5. Not read by :func:`grade_guided_compose` itself — it always
        returns only the newly composed candidates; this flag is here so
        the whole two-pass policy lives in one config object. Default
        ``True``.
    max_constituent_jaccard : float or None
        Forwarded to ``ANDComposer.compose()`` — see its own docstring.
        Default ``None`` (disabled).
    exhaustive_grades : frozenset[str]
        A stratum whose grade set is a subset of this (default ``{"A",
        "B"}``) is composed without a pair/triple cap (bounded only by
        ``exhaustive_safety_cap``, never by ``per_stratum_pair_cap``) — see
        the module docstring for why this is safe in practice: these grades
        are rare by construction, so the natural pair count already sits
        far below any reasonable cap.
    exhaustive_safety_cap : int
        Defensive upper bound applied to a stratum in ``exhaustive_grades``
        instead of ``per_stratum_pair_cap``/``per_stratum_triple_cap`` — not
        expected to bind given how grades are distributed in practice, but
        guards against a pathological input (e.g. a mis-tuned promotion
        threshold that makes ``A``/``B`` common) silently reproducing the
        same uncapped-materialization cost this design otherwise avoids.
        Default ``5000``.
    """

    max_components: int = 2
    adjacency: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {"A": ("A", "B"), "B": ("B", "C"), "C": ("C", "D")}
    )
    per_stratum_pair_cap: int = 100
    per_stratum_triple_cap: int = 50
    include_singles_in_pass2: bool = True
    max_constituent_jaccard: Optional[float] = None
    exhaustive_grades: FrozenSet[str] = field(default_factory=lambda: frozenset({"A", "B"}))
    exhaustive_safety_cap: int = 5000


def _enumerate_strata(
    config: GradePairingConfig, grades_present: set,
) -> List[Tuple[str, FrozenSet[str]]]:
    """List every ``(stratum_key, grade_set)`` reachable from the grades
    actually present in this pool: same-grade for each grade, plus
    adjacent-grade pairs per ``config.adjacency``.

    One entry here becomes one independent ``ANDComposer.compose()`` call
    in :func:`grade_guided_compose` — see the module docstring for why
    "independent" (rather than one shared call with a grade-keyed
    stratify hook) is the fix, not just a refactor.
    """
    strata: List[Tuple[str, FrozenSet[str]]] = []
    presents = sorted(grades_present)
    for g in presents:
        strata.append((f"{g}_same", frozenset({g})))
    for i, g1 in enumerate(presents):
        for g2 in presents[i + 1:]:
            if g2 in config.adjacency.get(g1, ()) or g1 in config.adjacency.get(g2, ()):
                strata.append((f"{g1}_{g2}", frozenset({g1, g2})))
    return strata


def _pair_matches_stratum(ga: str, gb: str, grades: FrozenSet[str]) -> bool:
    """Does a pair graded ``(ga, gb)`` belong to this ``grades`` stratum?

    A same-grade stratum (``len(grades) == 1``) needs both sides equal to
    that grade; an adjacent-grade stratum (``len(grades) == 2``) needs one
    side each — excluding a same-grade pair, which belongs to its own
    ``X_same`` stratum instead.
    """
    if len(grades) == 1:
        g = next(iter(grades))
        return ga == g and gb == g
    return {ga, gb} == set(grades)


def _stratum_root(grades: FrozenSet[str]) -> str:
    """The stratum's root grade: the single grade itself for a same-grade
    stratum, otherwise the alphabetically-first (better) of the two."""
    return next(iter(grades)) if len(grades) == 1 else min(grades)


def _reachable_from_root(config: "GradePairingConfig", root: str) -> FrozenSet[str]:
    """Grades a triple's third component may have, given the seed pair's
    root grade: the root itself, plus whatever ``config.adjacency`` lists
    under it. Forward-only (does not also admit a grade whose *own* entry
    happens to list ``root`` as a partner) — the same relation
    ``_enumerate_strata`` already uses to decide which adjacent-grade
    strata exist, applied to a single grade instead of a pair.
    """
    return frozenset({root, *config.adjacency.get(root, ())})


def grade_guided_compose(
    candidates: List[EventCandidate],
    contracts: List[AlphaContract],
    timestamps: pd.Series,
    config: GradePairingConfig,
    gate: ConsistencyGate,
) -> List[EventCandidate]:
    """Compose pairs (and, at ``max_components=3``, triples) guided by grade.

    Reads each candidate's A-D grade off ``contracts`` (matched by
    ``AlphaContract.event_candidate_id == EventCandidate.event_id`` —
    ``contracts`` is expected to be Alpha Discovery's *first-pass* output,
    i.e. every candidate it graded, not only ``.promoted_contracts()``: an
    "undetermined direction" event still carries a usable
    ``alpha_score.grade``), sorts the resulting pool by a canonical key
    (deterministic across processes — see the module docstring), and runs
    one independent ``ANDComposer.compose()`` call per grade stratum
    (:func:`_enumerate_strata`). Within each stratum's own call, pairs are
    additionally stratified by their two source features
    (``ANDComposer.compose()``'s ``stratify_fn`` hook, unmodified) so the
    per-stratum cap is spent across distinct feature combinations before a
    single combination's threshold variants can dominate it — the same
    round-robin-over-concatenation fix issue #230 made for the raw pair
    index shuffle, applied one level up. At ``max_components >= 3`` each
    stratum's own call also gets a ``triple_third_filter`` constraining the
    third component by the stratum's root grade (Phase 4, #254).

    A stratum whose grades are all in ``config.exhaustive_grades`` (default
    ``{"A", "B"}``) is composed without ``per_stratum_pair_cap`` /
    ``per_stratum_triple_cap`` (bounded only by ``exhaustive_safety_cap``)
    — see the module docstring for why this is cheap in practice.

    Returned candidates are fresh: new ``event_id``s from
    ``raw_event_to_candidate`` (Phase 1, #254), never inheriting the grade,
    target, or any other Alpha Discovery pass-1 state from their
    constituents — the caller's second Alpha Discovery pass derives their
    target from scratch, per the issue's own requirement.

    Parameters
    ----------
    candidates : list[EventCandidate]
        M1's 1D candidate pool (or any single-component candidate list).
        Multi-component (already-composed) candidates are silently skipped
        — composing an already-composed event further is out of scope here.
    contracts : list[AlphaContract]
        Alpha Discovery's first-pass output over ``candidates`` — every
        contract it built, used only to look up
        ``alpha_score.grade`` per ``event_candidate_id``.
    timestamps : pd.Series
        Datetime series aligned to the KPI table rows, as ``ANDComposer
        .compose()``/``raw_event_to_candidate`` require.
    config : GradePairingConfig
        Pairing scheme and budget.
    gate : ConsistencyGate
        Reused from M1 so composed candidates are evaluated against the
        same thresholds as the 1D pool — the same principle
        ``EventDiscovery.run()``'s own Step 5 already follows.

    Returns
    -------
    list[EventCandidate]
        Newly composed candidates only (never the untouched singles —
        pooling those in for a second Alpha Discovery pass is the caller's
        decision, per ``config.include_singles_in_pass2``).
    """
    grade_by_id: Dict[str, str] = {
        c.event_candidate_id: c.alpha_score.grade
        for c in contracts
        if c.alpha_score is not None and c.alpha_score.grade
    }

    raw_events: List[RawEvent] = []
    grade_of: Dict[int, str] = {}
    for cand in candidates:
        grade = grade_by_id.get(cand.event_id)
        if grade is None or len(cand.components) != 1:
            continue
        raw = raw_event_from_candidate(cand)
        raw_events.append(raw)
        grade_of[id(raw)] = grade

    if len(raw_events) < 2:
        return []

    # Canonical, deterministic order (see module docstring): removes the
    # dependency on FeatureGenerator's hash-seed-randomized set() iteration
    # order (issue #24) from which pairs get composed and returned.
    raw_events.sort(key=lambda r: r.component.expression)

    by_grade: Dict[str, List[RawEvent]] = defaultdict(list)
    for r in raw_events:
        by_grade[grade_of[id(r)]].append(r)

    composed: List[RawEvent] = []
    for stratum_key, grades in _enumerate_strata(config, set(by_grade)):
        root = _stratum_root(grades)
        # The pool a triple's third component is drawn from must extend
        # beyond this stratum's own two grades whenever the root's adjacency
        # reaches further (e.g. "A_same" has grades={"A"} but a third
        # component graded "B" is still admissible, since ANDComposer's
        # triple search only ever looks within the *same* pool array passed
        # to this compose() call -- see and_composer.py's valid_k lookup).
        # Pair formation itself stays restricted to `grades` via
        # feature_pair_stratify below, so this extension never lets a
        # mismatched pair (e.g. A-B) slip into the "A_same" stratum's own
        # pair output.
        pool_grades = grades | _reachable_from_root(config, root) if config.max_components >= 3 else grades
        pool = [r for g in pool_grades for r in by_grade[g]]
        if len(pool) < 2:
            continue

        exhaustive = grades.issubset(config.exhaustive_grades)
        pair_cap = config.exhaustive_safety_cap if exhaustive else config.per_stratum_pair_cap
        triple_cap = 0
        if config.max_components >= 3:
            triple_cap = config.exhaustive_safety_cap if exhaustive else config.per_stratum_triple_cap

        def feature_pair_stratify(
            a: RawEvent, b: RawEvent, _grades: FrozenSet[str] = grades,
        ) -> Optional[str]:
            ga, gb = grade_of.get(id(a)), grade_of.get(id(b))
            if ga is None or gb is None or not _pair_matches_stratum(ga, gb, _grades):
                return None
            return "|".join(sorted((a.component.source_feature, b.component.source_feature)))

        def triple_third_filter(
            a: RawEvent, b: RawEvent, c: RawEvent, _root: str = root,
        ) -> bool:
            # The seed pair (a, b) already matched this stratum's grades via
            # feature_pair_stratify, so the root computed above applies.
            g3 = grade_of.get(id(c))
            return g3 is not None and g3 in _reachable_from_root(config, _root)

        composer = ANDComposer(gate)
        composed.extend(composer.compose(
            pool, timestamps, max_components=config.max_components,
            gate=gate,
            pool_selector=lambda p: p,
            stratify_fn=feature_pair_stratify,
            max_pairs=pair_cap,
            max_triples=triple_cap,
            triple_third_filter=triple_third_filter if config.max_components >= 3 else None,
            max_constituent_jaccard=config.max_constituent_jaccard,
        ))

    # Extending a same-grade stratum's pool to reach third components (above)
    # means the same logical triple can, in principle, be independently
    # discovered from two different stratum calls (e.g. an "A,A,B" triple
    # from both "A_same" seeded on the A-A pair, and "A_B" seeded on an A-B
    # pair with the second A as third) -- dedupe by constituent set, not by
    # expression string, since join order (and therefore the string) depends
    # on which call found it first.
    seen: set = set()
    deduped: List[RawEvent] = []
    for ev in composed:
        key = frozenset(c.expression for c in ev.component.components)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    timestamp_col = (
        candidates[0].event_series.index.name
        if candidates and candidates[0].event_series is not None
        else (timestamps.name or "timestamp")
    )
    return [
        raw_event_to_candidate(
            ev, idx, timestamps,
            timestamp_col=timestamp_col, gate_params=gate.params,
        )
        for idx, ev in enumerate(deduped)
    ]
