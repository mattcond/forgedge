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
D is never a root, only ever reached as B/C's partner). Reuses
``ANDComposer.compose()``'s Phase 1 hooks (``forgedge.event_discovery
.and_composer``, #254 Phase 1) rather than re-deriving any pairing or gate
logic: one ``compose()`` call over the whole eligible pool, with a
``stratify_fn`` that keys each valid pair by its grade stratum so the
round-robin interleaving fixes the exact under-sampling bug the issue
reports for a single shared cap (a small stratum's sole pair getting
crowded out entirely by a much larger stratum before the cap is reached).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
        Guaranteed minimum representation (and the unit :func:`grade_guided_compose`
        scales the total pair budget by) for each grade stratum that
        actually appears in a given pool. Not a hard per-stratum ceiling —
        see :func:`grade_guided_compose`'s docstring for the precise
        fairness property this delivers and why. Default ``100``.
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
    """

    max_components: int = 2
    adjacency: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {"A": ("A", "B"), "B": ("B", "C"), "C": ("C", "D")}
    )
    per_stratum_pair_cap: int = 100
    per_stratum_triple_cap: int = 50
    include_singles_in_pass2: bool = True
    max_constituent_jaccard: Optional[float] = None


def _stratum_key(config: GradePairingConfig, grade_of: Dict[int, str], a: RawEvent, b: RawEvent) -> Optional[str]:
    """Grade-based stratum key for one valid pair, or ``None`` to exclude it.

    ``None`` for a pair whose grades aren't in ``config.adjacency``'s allowed
    set drops it from composition entirely — e.g. an A-D pair under the
    default scheme, which is neither same-grade nor listed as adjacent by
    either grade's own entry.
    """
    ga, gb = grade_of.get(id(a)), grade_of.get(id(b))
    if ga is None or gb is None:
        return None
    if ga == gb:
        return f"{ga}_same"
    lo, hi = sorted((ga, gb))
    if hi in config.adjacency.get(lo, ()) or lo in config.adjacency.get(hi, ()):
        return f"{lo}_{hi}"
    return None


def _triple_third_grade_ok(
    config: GradePairingConfig, grade_of: Dict[int, str], root_grade: str, third: RawEvent
) -> bool:
    """Is ``third``'s grade admissible as a triple's third component, given
    the seed pair's own root grade (issue #254 Phase 4)?

    Reuses the exact same same-grade/adjacency relation ``_stratum_key``
    applies to pairs — a triple's third component must relate to the
    *root* grade the same way a valid pairing partner would, not just be
    structurally distinct (``_validity_mask``) or reachable from either seed
    member individually via two separate pairwise checks.
    """
    g3 = grade_of.get(id(third))
    if g3 is None:
        return False
    if g3 == root_grade:
        return True
    lo, hi = sorted((root_grade, g3))
    return hi in config.adjacency.get(lo, ()) or lo in config.adjacency.get(hi, ())


def _count_strata(config: GradePairingConfig, grades_present: set) -> int:
    """Number of distinct grade strata the pool in front of us can produce.

    Used to scale the total pair/triple budget handed to ``compose()`` —
    see :func:`grade_guided_compose`'s docstring for what this buys.
    """
    same = len(grades_present)
    adjacent = 0
    presents = sorted(grades_present)
    for i, g1 in enumerate(presents):
        for g2 in presents[i + 1:]:
            if g2 in config.adjacency.get(g1, ()) or g1 in config.adjacency.get(g2, ()):
                adjacent += 1
    return same + adjacent


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
    ``alpha_score.grade``), builds a single composable pool of every
    single-component, graded candidate, and calls
    ``ANDComposer.compose()`` once with a grade-derived ``stratify_fn``
    (Phase 1, #254) so round-robin interleaving — not concatenation — decides
    traversal order across strata. At ``max_components >= 3`` a
    ``triple_third_filter`` (Phase 1's other composition hook) additionally
    constrains each triple's third component by the seed pair's own root
    grade (Phase 4, #254) — see ``_triple_third_grade_ok``.

    Fairness property of the pair/triple budget
    ---------------------------------------------
    The total budget handed to ``compose()`` is
    ``n_strata * config.per_stratum_pair_cap`` (similarly for triples), where
    ``n_strata`` is the number of grade strata actually present in this
    pool. Round-robin interleaving (``ANDComposer._stratified_pair_order``)
    guarantees every stratum contributes at least one candidate near the
    front of the traversal, before the budget can be exhausted by a single
    dominant stratum — this is the concrete fix for the failure mode the
    issue reports (a shared flat cap silently starving a small stratum to
    zero). It is **not** a hard per-stratum ceiling: once a small stratum's
    own valid pairs are exhausted, later rounds hand its "slot" to whichever
    strata remain, so a large stratum can end up contributing more than its
    nominal ``per_stratum_pair_cap`` share. A hard ceiling would need either
    N independent ``compose()`` calls (which reintroduces pool contamination
    across adjacent strata — see the design plan) or truncating each
    stratum's *pre-gate* candidate list before the gate has actually run,
    which trades one fairness problem for another (a stratum whose early
    candidates happen to fail the gate would be under-represented even
    though it has other valid candidates deeper in its own list). Guarding
    against complete starvation, which is what the issue's own v1-vs-v2
    numbers describe, is the property this design targets.

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

    grades_present = set(grade_of.values())
    n_strata = max(1, _count_strata(config, grades_present))
    effective_max_pairs = n_strata * config.per_stratum_pair_cap
    effective_max_triples = (
        n_strata * config.per_stratum_triple_cap if config.max_components >= 3 else 0
    )

    def stratify(a: RawEvent, b: RawEvent) -> Optional[str]:
        return _stratum_key(config, grade_of, a, b)

    def triple_third_filter(a: RawEvent, b: RawEvent, c: RawEvent) -> bool:
        # The seed pair (a, b) already passed `stratify` (same-grade or
        # adjacent), so both grades are known; the pair's "root" is the
        # better (alphabetically-first, A<B<C<D) of the two — the same
        # grade `_stratum_key`'s own sorted(lo, hi) treats as the
        # adjacency-dict key that validated the pair in the first place.
        ga, gb = grade_of.get(id(a)), grade_of.get(id(b))
        if ga is None or gb is None:
            return False
        root_grade = min(ga, gb)
        return _triple_third_grade_ok(config, grade_of, root_grade, c)

    composer = ANDComposer(gate)
    composed_raw = composer.compose(
        raw_events, timestamps, max_components=config.max_components,
        gate=gate,
        pool_selector=lambda pool: pool,
        stratify_fn=stratify,
        max_pairs=effective_max_pairs,
        max_triples=effective_max_triples,
        triple_third_filter=triple_third_filter if config.max_components >= 3 else None,
        max_constituent_jaccard=config.max_constituent_jaccard,
    )

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
        for idx, ev in enumerate(composed_raw)
    ]
