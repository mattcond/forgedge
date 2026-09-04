# Two-pass, grade-guided event composition — implementation plan

> **Status: fully implemented (Phases 1-8 shipped).** Phases 1-4 (this
> document's mechanical core) landed first, one PR per phase. Phase 5
> (multi-asset/timeframe validation) confirmed the improvement generalises
> on 1D (8/8 assets tested, 1.44x-3.05x more EDGE/PARTIAL-EDGE contracts)
> and is non-worse on 1H. Phase 6 measured real cost (+44%-75% total wall
> time over single-pass at realistic 1D scale, dominated by the composition
> search itself). Given a real, generalising 1D win and a *non-destructive*
> (not negative) 1H result, the project made `two_pass_composition=True`
> and `DiscoveryConfig.retain_raw_events=False` **the new defaults** in
> Phase 8 — a stronger outcome than this document's own §5 originally
> scoped ("no default changes until Phases 5-6 confirm"; they did, and the
> defaults changed). Phase 7 (this document's own listed doc surface)
> is complete. `forge(two_pass_composition=False)` still reproduces the
> pre-#254 single-pass behaviour this plan calls "the regression anchor"
> exactly. The rest of this document is kept as the historical design
> record — see `docs/manual-en.md`/`manuale-it.md` §8/§10, the `forgedge`
> skill, and `src/forgedge/docs/specs/modulo_1_{en,it}.md`/
> `modulo_2_{en,it}.md` for the current, defaults-accurate description of
> the shipped behaviour.

**Starting point:** issue [#254](https://github.com/mattcond/forgedge/issues/254) —
`ANDComposer` (`event_discovery/and_composer.py`) composes pairs/triples of events
today entirely inside `EventDiscovery.run()` ("Step 5"), using only structural
pairing criteria (tpm, dispersion, `transform_key`). Empirically (AMZN 1D, `balanced`
preset), that produces 256 candidates and 5 PARTIAL-EDGE/EDGE (1.95%). Pairing by
the A–D grade Alpha Discovery (M2) assigns to each event — same grade first, then
adjacent grade with a root+partner scheme (A↔{A,B}, B↔{B,C}, C↔{C,D}) — instead
produces ~1400+ evaluated candidates and 122 PARTIAL-EDGE/EDGE (~8-9%), with the
search-level rotation-null p-value (0.0497) crossing significance for the first time
in the whole experimental session. Structural correlation (phi) between events turned
out to be a poor proxy for a pair's economic quality; M2's grade, which already
encodes return information, is a much better pairing signal.

**Question this plan answers:** the issue's own 8-step sketch is directionally right,
but which parts are a mechanical, testable refactor doable now, and which parts need
real data and empirical judgment before any default changes? And: what exactly breaks
downstream if M1's output stops being "the final N-dimensional event pool"?

**Method:** three parallel code explorations (M1/`ANDComposer`, M2/grading,
`forge()` orchestration + every downstream consumer) followed by a design pass, then
hand-verification of the design's load-bearing claims directly against source. That
verification pass found one concrete wiring bug the design missed — see §4 — which is
exactly the kind of thing this document exists to catch before code is written, not
after.

This document is the plan; it does not itself change any pipeline behavior.

---

## 1. Why this is not an invariant change, but is a real orchestration change

M1 stays returns-blind (produces only the 1D event pool); M2 remains the first stage
that ever sees the forward return. What changes is *when* composition happens: today
it happens inside M1 using only structural criteria; after this refactor it happens
between two M2 calls, using the grade the first call assigns. Two facts make this
tractable rather than invasive:

- **`AlphaDiscovery` is stateless per call and has zero coupling to how its candidate
  list was produced.** `AlphaDiscovery.__init__(kpi_table, event_candidates, config,
  time_budget)` treats every `EventCandidate` purely as an activation function. A
  "second M2 pass on composed candidates" is therefore just a second
  `AlphaDiscovery(...)` construction over a different candidate list — no change
  needed inside `AlphaDiscovery` itself.
- **`EventCandidate` already carries everything needed to reconstruct a `RawEvent`.**
  `.event_series`, `.consistency_gate`, `.components` are ordinary dataclass fields
  populated by `EventDiscovery._to_candidate()`. A grade-guided composer can work
  directly off M2 pass 1's input list via a thin adapter, with no need to keep M1
  internals alive past `EventDiscovery.run()`.

The structural pairing criterion itself lives in exactly two functions of
`and_composer.py`: `_build_composition_pool()` (slot grouping) and the pair/triple
priority order (`_shuffle_order()` + the `_MAX_PAIRS`/`_MAX_TRIPLES` caps).
`_validity_mask()` (admissibility: no same-source-native pairs, no same-transform
"subset" pairs) is content-agnostic and stays shared by both modes. Everything
downstream of pair/triple *selection* — the chunked gate evaluation, `episode_starts`,
`_gate_pass`/`_eff_max_dispersion` — is shared machinery that must not be
re-derived: this is the direct lesson of issues #226 (composed events silently judged
by different pass/fail criteria than single events) and #124, both cited by #254
itself.

## 2. Design decisions (recommended defaults, not open alternatives)

1. **`TargetOptimizer` stays out of scope.** It composes atoms *before* any
   `AlphaDiscovery` call, against a user-fixed target — it cannot structurally host a
   grade-derived criterion. Its docstring gets one sentence noting this refactor
   deliberately does not touch it.
2. **`_grade()`'s cutoffs (0.75/0.50/0.25, hardcoded in `alpha_discovery/discovery.py`)
   stay where they are.** The composer only needs the letter label already on
   `AlphaScore.grade`, never the numeric cutoffs — no dependency either way.
3. **Per-stratum pairing caps are a flat scalar applied independently per stratum**
   (`per_stratum_pair_cap`/`per_stratum_triple_cap: int`), not a dict keyed by
   grade-combination — this satisfies the issue's requirement without inventing
   dict-valued territory the resolver has never had. Critical implementation detail:
   strata must be **round-robin interleaved, not concatenated then globally capped** —
   concatenate-then-truncate reproduces the exact v1 under-sampling bug the issue
   itself reports (51 edges vs. 76 from a tighter baseline, a sampling artifact fixed
   in v2, which then produced 122).
4. **M2 pass 2 evaluates the pooled set (1D + composed) by default**, matching today's
   implicit `all_passing = passing_single + passing_composed` — so promoted-contract
   counts don't silently shrink relative to today. Configurable via
   `include_singles_in_pass2: bool = True`.
5. **`ForgeResult` exposes pass 1 as a first-class artifact**: new optional
   `grading_candidates`/`grading_contracts` fields, populated only in two-pass mode.
   `.candidates`/`.contracts`/`.promoted` keep meaning "what M3 and the rotation null
   actually operated on" (pass 2's pooled output) — consistent with the invariant that
   a later module never reaches back into an earlier one's internals by inference.

## 3. Phases implementable now (mechanical, testable on existing synthetic fixtures)

### Phase 1 — Decouple `ANDComposer` (issue step 1), zero behavior change

Files: `event_discovery/and_composer.py`, `event_discovery/discovery.py`.

- `compose()` gains optional `pool_selector`, `stratify_fn`, `max_pairs`,
  `max_triples` — all `None` by default, reproducing today's behavior exactly
  (`_build_composition_pool`, flat shuffle, `_MAX_PAIRS=2000`/`_MAX_TRIPLES=500`). No
  change to the shared gate-evaluation machinery.
- New `raw_event_from_candidate(candidate: EventCandidate) -> RawEvent` — the
  "invoke standalone on a list of `EventCandidate`" entry point the issue's step 1
  asks for.
- `EventDiscovery._to_candidate()` extracted into a module-level
  `raw_event_to_candidate(...)` free function; the instance method becomes a
  one-line wrapper. Reused by Phase 2 to turn newly-composed `RawEvent`s back into
  fresh `EventCandidate`s with brand-new `event_id`s (never inheriting from
  components, per the issue's requirement).
- M1's own Step 5 call site is untouched — it already calls `compose()` with every
  new parameter at its default.

Tests: extend `tests/test_event_discovery.py::TestBugRegressions` — (i) `compose()`
with no new parameters reproduces the existing #226/#228/#230/#124 fixtures exactly;
(ii) a synthetic `stratify_fn` case proving round-robin interleaving instead of
concatenate-then-truncate, sized so a flat cap would bite unevenly under the naive
approach (same construction style as #230's own regression test). `test_golden.py`:
no re-pin needed.

### Phase 2 — Grade-informed pairing config (issue step 2)

New package `src/forgedge/composition/` (mirrors `calibration/` — this stage sits
between M1 and M2, owned by neither):

- `GradePairingConfig`: `max_components: int = 2`,
  `adjacency: dict = {"A": ("A","B"), "B": ("B","C"), "C": ("C","D")}`,
  `per_stratum_pair_cap: int`, `per_stratum_triple_cap: int`,
  `include_singles_in_pass2: bool = True`, `max_constituent_jaccard: Optional[float] = None`.
- `grade_guided_compose(candidates, contracts, timestamps, config, gate) ->
  list[EventCandidate]`: builds the `event_id → grade` lookup from `contracts`,
  converts eligible candidates via `raw_event_from_candidate`, builds a grade-aware
  `pool_selector`/`stratify_fn` keyed on `(root_grade, "same"|"adjacent")`, calls
  `ANDComposer.compose(...)`, converts survivors back via `raw_event_to_candidate`
  with fresh ids.
- `resolver.py`: new `Constraint` (`STRUCTURAL`, `WARN`)
  `grade_pairing_cap_incoherent`, following the existing `Constraint` pattern.
- `presets.py`: unchanged in this phase — a per-preset `GradePairingConfig` mapping
  is deferred to Phase 8 (post-validation).

Tests: new `tests/composition/test_grade_pairing.py` (house style: seeded synthetic
KPI table, `pytest.approx`, no mocking beyond tripwires) — same-grade and
adjacent-grade strata, `per_stratum_pair_cap` honored per stratum not globally, no
duplicate pairs, fresh ids never reused from pass 1. Extend `tests/test_config_report.py`
for the new `Constraint`.

### Phase 3 — Two-pass orchestration in `forge()`, behind an off-by-default flag (issue step 3)

Files: `forge.py`, `ledger.py`.

- New `forge()` parameters: `two_pass_composition: bool = False`,
  `grade_pairing_config: Optional[GradePairingConfig] = None`.
- Validation: `two_pass_composition=True` requires
  `event_discovery_config.max_and_components <= 1`, else `ValueError` (same pattern
  as the existing `manual_events`/`event_discovery_config` mutual exclusivity — fail
  loudly, not silently, per invariant #9).
- `two_pass_composition=False` → byte-identical to today (the regression anchor).
  `two_pass_composition=True`:
  1. M2 pass 1 (grading): `AlphaDiscovery(alpha_frame, alpha_candidates, cfg,
     time_budget=time_budget).run()`.
  2. `grade_guided_compose(...)` over pass 1's output.
  3. `pass2_candidates = alpha_candidates + composed` (or `composed` alone if
     `include_singles_in_pass2=False`).
  4. M2 pass 2: fresh `AlphaDiscovery` over `pass2_candidates`.
  5. Rebind for the rest of the function to run unmodified.
  6. `ledger`, rotation null, M3, `ForgeResult` assembly proceed as today, now over
     pass 2's output.

#### 4. A wiring detail the design pass missed, found during source verification

The design's first draft said "rebind `alpha_candidates` to pass 2's output before
the rotation-null block runs." Direct verification against `forge.py` found this is
necessary but **not sufficient**: Module 3's lookup,
`by_id = {c.event_id: c for c in candidates}`, is built from the **`candidates`**
variable — M1's raw output, also what `ForgeResult.candidates` is populated from —
**not** from `alpha_candidates`. If only `alpha_candidates` were rebound, every
composed contract from pass 2 would carry an `event_candidate_id` absent from
`by_id`, and `RuleDiscovery` would silently skip it
(`cand = by_id.get(...); if cand is None: continue`) — M3 would backtest zero
composed contracts, and the refactor would measurably produce 0 extra edges instead
of the ~122 the issue's own experiment found, with no error or warning anywhere.
This is precisely the silent-failure shape invariant #9 exists to rule out.

**Fix, folded into Phase 3**: rebind **both** `candidates` and `alpha_candidates` to
`pass2_candidates` before the ledger/rotation-null/M3 block runs. Save pass 1's pool
and contracts to separate variables (`grading_candidates`, `grading_contracts`)
*before* this rebind, to populate `ForgeResult`'s new fields (§2.5).

Also added in Phase 3: `ForgeResult.grading_candidates`/`.grading_contracts`
(Optional, two-pass only); zero-cost wall-clock instrumentation around
pass1/composition/pass2 (`ForgeResult.composition_timing: Optional[dict]`) — this
absorbs the cheap part of issue step 6, leaving the real cost measurement for the
Phase 6 follow-up; `HypothesisLedger.m2_pass1_candidates: int = 0`, a descriptive
field so the ledger's multiple-testing surface accounting stays honest under
two-pass mode.

Tests: extend `tests/test_forge.py` — (1) a monkeypatch "must-not-be-called"
tripwire proving `grade_guided_compose` is never invoked when the flag is `False`;
(2) a synthetic two-pass run asserting two `AlphaDiscovery.run()` calls,
`grading_candidates`/`grading_contracts` populated and distinct from
`.candidates`/`.contracts`, **and explicit proof that a composed contract reaches
M3** (i.e. the rebinding fix in §4 has a regression test, not just a code fix); (3)
the `ValueError` on flag+incompatible `max_and_components`. Exit criterion: the full
existing suite, `test_golden.py` included, passes with zero change to expected
output (the golden fixture calls `forge()` with defaults, so `two_pass_composition`
stays `False` there).

### Phase 4 — Extend to triples (issue step 4)

`GradePairingConfig.max_components=3`; the triple's stratum is keyed on the
**root's** grade only, per the issue's "root + two partners" phrasing. Verify
during implementation whether `and_composer.py`'s existing pair-seeded triple
search already dedups `(idx_a, idx_b, k)` before assuming it needs adding.

Tests: extend `tests/composition/test_grade_pairing.py` with a `max_components=3`
case — no duplicate `(root, partner1, partner2)` id-sets, `per_stratum_triple_cap`
honored independently of the pair cap.

## 5. Phases explicitly deferred to follow-up work (need real data / empirical judgment)

- **Phase 5 — Multi-asset/timeframe validation (issue step 5).** The issue's own
  numbers come from one asset/timeframe (AMZN 1D). Before any default changes, repeat
  the baseline-vs-new comparison on 2-3 additional combinations to confirm 122 vs. 5
  edges (p=0.0497) generalizes. This is the issue's own explicit gate before touching
  any default — not a coding task. **Follow-up issue, opened once Phases 1-4 are
  merged.**
- **Phase 6 — Real cost measurement.** Phase 3 ships the free instrumentation; the
  actual wall-time measurement at realistic scale (`balanced` preset) and the
  decision on whether a pre-filter is needed before pass 1 need a real run.
  **Follow-up.**
- **Phase 7 — Docs and public surface (issue step 7).** Sequenced after Phases 1-4
  land, so docs describe shipped API rather than a moving target:
  `docs/specs/modulo_1_{en,it}.md`, `modulo_2_{en,it}.md`,
  `src/forgedge/docs/modules/EventDiscovery.md`/`AlphaDiscovery.md`,
  `docs/specs/configuration_{en,it}.md`, `how_to_use_{en,it}.md`, both manuals, the
  `forgedge` skill (`SKILL.md` + `api-reference.md`), the two tutorial notebooks, and
  the 8 `max_and_components=` usages across `examples/*.py`.
- **Phase 8 — Deprecate the legacy structural path (issue step 8).** No default
  changes, no `DeprecationWarning`, no `presets.py` remapping until Phases 5-6
  confirm the improvement generalizes and the cost is acceptable. Explicitly
  conditioned on both.

## 6. Critical files

- `src/forgedge/event_discovery/and_composer.py` — pluggable
  `pool_selector`/`stratify_fn`/`max_pairs`/`max_triples` hooks on `compose()`, plus
  `raw_event_from_candidate()`.
- `src/forgedge/event_discovery/discovery.py` — `_to_candidate` extraction into
  reusable `raw_event_to_candidate()`.
- `src/forgedge/composition/grade_pairing.py` (new) — `GradePairingConfig` and
  `grade_guided_compose()`.
- `src/forgedge/forge.py` — the two-pass branch, and specifically the **double
  rebind of `candidates` and `alpha_candidates`** before the
  ledger/rotation-null/M3 block (§4), plus the new `ForgeResult` fields.
- `src/forgedge/ledger.py` — `m2_pass1_candidates`.
- `tests/test_forge.py`, `tests/test_golden.py` — the regression anchors that must
  stay green (golden unchanged) through Phases 1-4, plus the new two-pass coverage
  in Phase 3, including the explicit "a composed contract reaches M3" test.

## 7. End-to-end verification

1. Full suite: `pytest -q -m "not slow"` stays green at every phase, `test_golden.py`
   unchanged through Phases 1-4.
2. End of Phase 3: an ad hoc script (style of `examples/pipeline_coherence_audit.py`)
   calling `forge(kpi_table, ..., two_pass_composition=True,
   grade_pairing_config=GradePairingConfig(...))` on a synthetic fixture, manually
   confirming `ForgeResult.grading_candidates` (1D pool),
   `ForgeResult.candidates` (pooled pass-2), and `ForgeResult.rule_responses`
   actually contain contracts derived from composed events — the concrete proof the
   §4 rebinding fix works, not just that unit tests pass.
3. Only once Phases 1-4 are in `main`: open the Phase 5 follow-up issue, referencing
   this document and the original AMZN 1D numbers from #254 as the baseline to
   reproduce on 2-3 additional combinations before touching any `presets.py` default.
