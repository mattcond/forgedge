"""Tests for forgedge.composition.grade_pairing (issue #254, Phases 2 and 4)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgedge.composition import GradePairingConfig, grade_guided_compose
from forgedge.event_discovery.consistency_gate import ConsistencyGate, _build_month_index
from forgedge.event_discovery.models import (
    ActivationStats,
    EventCandidate,
    EventComponent,
    GateParams,
)


def _component(feature: str) -> EventComponent:
    return EventComponent(
        source_feature=feature, transform="identity", transform_params={},
        transformed_col=feature, threshold=0.5, threshold_type="test",
        direction="above", event_type="threshold", expression=f"{feature} > 0.5",
        source_cols=[], sql_expression="",
    )


def _candidate(event_id: str, feature: str, series: pd.Series, gate: ConsistencyGate,
               month_index, n_months) -> EventCandidate:
    """Build a real single-component EventCandidate whose GateResult/ActivationStats
    reflect the actual activation series, so ANDComposer's gate evaluates it for real
    (an all-zero or all-mismatched series would trivially fail the volume floor and
    the tests below would prove nothing)."""
    g = gate.evaluate_series(series, month_index, n_months)
    comp = _component(feature)
    stats = ActivationStats(
        n_activations=g.n_activations, n_active_months=g.n_active_months, zero_months=0,
        max_monthly_share=g.max_monthly_share, mean_tpm=g.mean_tpm,
        index_of_dispersion=g.index_of_dispersion, n_episodes=g.n_episodes,
        episode_index_of_dispersion=g.episode_index_of_dispersion,
    )
    return EventCandidate(
        event_id=event_id, status="CANDIDATE", components=[comp], expression=comp.expression,
        activation_stats=stats, consistency_gate=g, event_series=series, gate_params=gate.params,
    )


def _contract(event_candidate_id: str, grade: str) -> SimpleNamespace:
    """A duck-typed AlphaContract stand-in -- grade_guided_compose only ever reads
    .event_candidate_id and .alpha_score.grade, the same minimal-fixture approach
    tests/test_deployment.py already uses for AlphaContract."""
    return SimpleNamespace(
        event_candidate_id=event_candidate_id,
        alpha_score=SimpleNamespace(grade=grade),
    )


def _gate_and_ts(n=2000, min_tpm=0.5, dispersion_margin=3.0, min_episodes=1):
    """n=2000 hourly bars (~3 months) matches ANDComposer's own Phase 1 test
    fixtures (tests/test_event_discovery.py) -- a longer synthetic span
    tightens the episode-dispersion floor enough that even a strongly
    overlapping pair can fail it, which would make these fixtures prove
    nothing about grade_guided_compose itself."""
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
    gate = ConsistencyGate(GateParams(min_tpm=min_tpm, dispersion_margin=dispersion_margin, min_episodes=min_episodes))
    month_index, n_months = _build_month_index(ts)
    return gate, ts, month_index, n_months


def _overlapping_pair(rng, n, p_base=0.30, p_thin=0.85):
    """Two independently-thinned draws from a shared random base.

    Reliably passes ConsistencyGate both individually and AND-composed (the
    same construction ANDComposer's own Phase 1 regression tests use) --
    unlike a deterministic periodic pattern (e.g. ``idx % 5 < 3``), whose
    unnaturally regular clustering trips the episode-dispersion check even
    under a loose ``dispersion_margin``.
    """
    base = rng.random(n) < p_base
    s1 = pd.Series((base & (rng.random(n) < p_thin)).astype(float))
    s2 = pd.Series((base & (rng.random(n) < p_thin)).astype(float))
    return s1, s2


def _overlapping_group(rng, n, k, p_base=0.30, p_thin=0.85):
    """``k`` independently-thinned draws from a shared random base -- the
    same construction as ``_overlapping_pair``, generalised so a k-way AND
    (needed for triple tests, issue #254 Phase 4) reliably clears the gate
    too, not just each pairwise AND."""
    base = rng.random(n) < p_base
    return [pd.Series((base & (rng.random(n) < p_thin)).astype(float)) for _ in range(k)]


def _make_kpi_table(n: int = 4380, seed: int = 42) -> pd.DataFrame:
    """Synthetic KPI table -- same generator as tests/test_event_discovery.py's
    own helper, duplicated here rather than cross-imported so this file's
    fixtures stay self-contained (house style: each test module owns its
    fixtures). Default n=4380 ~= 6 months of 1H data."""
    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(0.0002, 0.005, n))
    vol = np.abs(rng.normal(1e6, 2e5, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    def sma(s, w):
        return pd.Series(s).rolling(w, min_periods=1).mean().values

    def ema(s, w):
        return pd.Series(s).ewm(span=w, adjust=False).mean().values

    def rsi(s, w=14):
        d = pd.Series(s).diff()
        g = d.clip(lower=0).rolling(w, min_periods=1).mean()
        l = (-d.clip(upper=0)).rolling(w, min_periods=1).mean()
        return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50).values

    return pd.DataFrame(
        {
            "open_dt": dates,
            "close": price,
            "volume": vol,
            "close_rsi_14": rsi(price, 14),
            "close_rsi_25": rsi(price, 25),
            "close_ema_09": ema(price, 9),
            "close_ema_25": ema(price, 25),
            "close_sma_25": sma(price, 25),
            "volume_sma_25": sma(vol, 25),
            "close_bb_lower_20": pd.Series(sma(price, 20))
            - 2 * pd.Series(price).rolling(20, min_periods=1).std().values,
            "close_bb_upper_20": pd.Series(sma(price, 20))
            + 2 * pd.Series(price).rolling(20, min_periods=1).std().values,
            "close_min_24": pd.Series(price).rolling(24, min_periods=1).min().values,
            "close_max_24": pd.Series(price).rolling(24, min_periods=1).max().values,
        }
    )


def _grade_lookup_by_expression(candidates, contracts):
    """Map a single-component candidate's expression -> its grade, so a
    composed candidate's constituents can be traced back to the grades that
    produced them (composed candidates get fresh event_ids, so identity has
    to go through the expression instead)."""
    grade_by_id = {c.event_candidate_id: c.alpha_score.grade for c in contracts}
    return {
        cand.components[0].expression: grade_by_id.get(cand.event_id)
        for cand in candidates if len(cand.components) == 1
    }


def _strata_of(composed, grade_of_expr):
    strata = set()
    for ev in composed:
        grades = tuple(sorted(grade_of_expr[c.expression] for c in ev.components))
        strata.add(grades)
    return strata


class TestGradeGuidedCompose:
    def test_same_grade_pairing_produces_the_expected_stratum(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)

        candidates, contracts = [], []
        # Two strongly-overlapping A-graded events -> guaranteed A-A composable pair.
        s_a1, s_a2 = _overlapping_pair(rng, n)
        for i, s in enumerate([s_a1, s_a2]):
            eid = f"EVT-a{i}"
            candidates.append(_candidate(eid, f"feat_a{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, "A"))
        # A pool of B/C/D-graded noise so the A-A stratum has to compete for budget.
        for i in range(15):
            p_act = rng.uniform(0.15, 0.35)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            eid = f"EVT-noise{i}"
            candidates.append(_candidate(eid, f"feat_noise{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, rng.choice(["B", "C", "D"])))

        config = GradePairingConfig(per_stratum_pair_cap=10, per_stratum_triple_cap=5)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        grade_of_expr = _grade_lookup_by_expression(candidates, contracts)
        strata = _strata_of(composed, grade_of_expr)
        assert ("A", "A") in strata, "the guaranteed A-A pair must survive composition"

    def test_adjacency_scheme_excludes_non_adjacent_grades(self):
        """Default adjacency is A<->{A,B}, B<->{B,C}, C<->{C,D} -- an A-D pair
        is neither same-grade nor adjacent under either grade's own entry, and
        must never appear in the composed output."""
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)

        # s_d is literally s_a -- AND(s_a, s_d) == s_a, which independently
        # passes the gate, so this pair WOULD easily compose if the adjacency
        # filter didn't block it: a false negative here is a fixture bug, not
        # a real pass.
        s_a, _ = _overlapping_pair(rng, n)
        s_d = s_a.copy()

        candidates = [
            _candidate("EVT-a", "feat_a", s_a, gate, month_idx, n_months),
            _candidate("EVT-d", "feat_d", s_d, gate, month_idx, n_months),
        ]
        contracts = [_contract("EVT-a", "A"), _contract("EVT-d", "D")]

        config = GradePairingConfig(per_stratum_pair_cap=10)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        assert composed == [], "an A-D pair must be excluded by the default adjacency scheme"

    def test_adjacent_grade_root_partner_pairing_is_allowed(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s_b, s_c = _overlapping_pair(rng, n)

        candidates = [
            _candidate("EVT-b", "feat_b", s_b, gate, month_idx, n_months),
            _candidate("EVT-c", "feat_c", s_c, gate, month_idx, n_months),
        ]
        contracts = [_contract("EVT-b", "B"), _contract("EVT-c", "C")]

        config = GradePairingConfig(per_stratum_pair_cap=10)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        assert composed, "fixture must produce the B-C pair"
        grade_of_expr = _grade_lookup_by_expression(candidates, contracts)
        assert _strata_of(composed, grade_of_expr) == {("B", "C")}

    def test_small_stratum_not_starved_by_a_much_larger_one(self):
        """The concrete fix for the issue's own v1 bug: a tiny stratum's sole
        pair must not be silently excluded just because a much larger stratum
        shares the same overall pair budget."""
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)

        candidates, contracts = [], []
        # The sole A-A pair -- tiny stratum, guaranteed composable.
        s_a1, s_a2 = _overlapping_pair(rng, n)
        for i, s in enumerate([s_a1, s_a2]):
            eid = f"EVT-a{i}"
            candidates.append(_candidate(eid, f"feat_a{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, "A"))
        # A much larger D-D stratum: many more valid pairs than the pair budget.
        for i in range(15):
            p_act = rng.uniform(0.15, 0.35)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            eid = f"EVT-d{i}"
            candidates.append(_candidate(eid, f"feat_d{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, "D"))

        # Two strata present (A_same, D_same) -> total budget = 2 * per_stratum_pair_cap.
        # A flat cap under naive concatenation (D enumerated first) would spend
        # the whole budget on D-D pairs alone, exactly the v1 bug the issue reports.
        config = GradePairingConfig(per_stratum_pair_cap=2)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        grade_of_expr = _grade_lookup_by_expression(candidates, contracts)
        strata = _strata_of(composed, grade_of_expr)
        assert ("D", "D") in strata, "fixture must produce real competition from the big stratum"
        assert ("A", "A") in strata, "small stratum's sole pair must not be starved"

    def test_fresh_event_ids_never_reused_from_pass1(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s1, s2 = _overlapping_pair(rng, n)
        candidates = [
            _candidate("EVT-1", "feat_1", s1, gate, month_idx, n_months),
            _candidate("EVT-2", "feat_2", s2, gate, month_idx, n_months),
        ]
        contracts = [_contract("EVT-1", "A"), _contract("EVT-2", "A")]

        config = GradePairingConfig(per_stratum_pair_cap=10)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        assert composed, "fixture must produce at least one composed candidate"
        input_ids = {c.event_id for c in candidates}
        composed_ids = {c.event_id for c in composed}
        assert composed_ids.isdisjoint(input_ids)
        assert len(composed_ids) == len(composed), "no duplicate composed event_ids"

    def test_multi_component_candidates_are_skipped(self):
        """A candidate that is itself already an AND composition (2+ components)
        must never be re-composed further -- grade_guided_compose silently
        excludes it rather than raising (raw_event_from_candidate would raise
        on it, so this proves the skip actually happens upstream of that call).
        Two genuinely eligible singles are included too, so the pool has real
        composable pairs and the exclusion is actually exercised."""
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s1, s2 = _overlapping_pair(rng, n)
        single1 = _candidate("EVT-single1", "feat_single1", s1, gate, month_idx, n_months)
        single2 = _candidate("EVT-single2", "feat_single2", s2, gate, month_idx, n_months)

        _, s3 = _overlapping_pair(rng, n)
        already_composed = _candidate("EVT-composed", "feat_x", s3, gate, month_idx, n_months)
        already_composed.components = [_component("feat_x"), _component("feat_y")]

        contracts = [
            _contract("EVT-single1", "A"),
            _contract("EVT-single2", "A"),
            _contract("EVT-composed", "A"),
        ]
        config = GradePairingConfig(per_stratum_pair_cap=10)

        composed = grade_guided_compose(
            [single1, single2, already_composed], contracts, ts, config, gate,
        )
        assert composed, "the two genuinely eligible singles must still compose"
        for ev in composed:
            for c in ev.components:
                assert c.source_feature not in ("feat_x", "feat_y")

    def test_ungraded_candidates_are_skipped(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s1, s2 = _overlapping_pair(rng, n)
        graded1 = _candidate("EVT-graded1", "feat_graded1", s1, gate, month_idx, n_months)
        graded2 = _candidate("EVT-graded2", "feat_graded2", s2, gate, month_idx, n_months)

        _, s3 = _overlapping_pair(rng, n)
        ungraded = _candidate("EVT-ungraded", "feat_ungraded", s3, gate, month_idx, n_months)

        # "ungraded" has no matching contract.
        contracts = [_contract("EVT-graded1", "A"), _contract("EVT-graded2", "A")]
        config = GradePairingConfig(per_stratum_pair_cap=10)

        composed = grade_guided_compose([graded1, graded2, ungraded], contracts, ts, config, gate)
        assert composed, "the two graded singles must still compose"
        for ev in composed:
            for c in ev.components:
                assert c.source_feature != "feat_ungraded"

    def test_fewer_than_two_eligible_candidates_returns_empty(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        s1, _ = _overlapping_pair(rng, len(ts))
        only = _candidate("EVT-only", "feat_only", s1, gate, month_idx, n_months)
        contracts = [_contract("EVT-only", "A")]
        config = GradePairingConfig(per_stratum_pair_cap=10)

        assert grade_guided_compose([only], contracts, ts, config, gate) == []
        assert grade_guided_compose([], [], ts, config, gate) == []


class TestGradeGuidedComposeTriples:
    """max_components=3 (issue #254 Phase 4): triple composition keyed on
    the seed pair's root grade, not just pairwise structural admissibility."""

    def test_max_components_2_produces_no_triples_by_default(self):
        """Phase 4 is opt-in -- a pool that COULD produce triples must not,
        unless max_components=3 is set explicitly."""
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s0, s1, s2 = _overlapping_group(rng, n, 3)

        candidates, contracts = [], []
        for i, s in enumerate([s0, s1, s2]):
            eid = f"EVT-a{i}"
            candidates.append(_candidate(eid, f"feat_a{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, "A"))

        config = GradePairingConfig(per_stratum_pair_cap=10, per_stratum_triple_cap=10)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        assert composed, "fixture must produce at least the pairs"
        assert all(len(ev.components) == 2 for ev in composed), (
            "max_components=2 (the default) must never produce a 3-component candidate"
        )

    def test_max_components_3_composes_a_root_adjacent_triple(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s_a0, s_a1, s_b0 = _overlapping_group(rng, n, 3)

        candidates = [
            _candidate("EVT-a0", "feat_a0", s_a0, gate, month_idx, n_months),
            _candidate("EVT-a1", "feat_a1", s_a1, gate, month_idx, n_months),
            _candidate("EVT-b0", "feat_b0", s_b0, gate, month_idx, n_months),
        ]
        contracts = [_contract("EVT-a0", "A"), _contract("EVT-a1", "A"), _contract("EVT-b0", "B")]

        config = GradePairingConfig(max_components=3, per_stratum_pair_cap=10, per_stratum_triple_cap=10)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        triples = [ev for ev in composed if len(ev.components) == 3]
        assert triples, "the A-A-B triple must survive composition (root=A admits B)"
        grade_of_expr = _grade_lookup_by_expression(candidates, contracts)
        for ev in triples:
            grades = tuple(sorted(grade_of_expr[c.expression] for c in ev.components))
            assert grades == ("A", "A", "B")

    def test_triple_third_component_is_constrained_by_the_root_grade(self):
        """A third component reachable from neither seed member under the
        adjacency scheme's ROOT-grade relation must never appear, even when
        the underlying 3-way AND would otherwise pass the gate easily."""
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s_a0, s_a1, s_d0 = _overlapping_group(rng, n, 3)

        candidates = [
            _candidate("EVT-a0", "feat_a0", s_a0, gate, month_idx, n_months),
            _candidate("EVT-a1", "feat_a1", s_a1, gate, month_idx, n_months),
            _candidate("EVT-d0", "feat_d0", s_d0, gate, month_idx, n_months),
        ]
        # root of the (A, A) seed pair is "A"; "D" is unreachable from "A"
        # under the default adjacency (only B is) -- fixture setup deliberately
        # gives the D event strong 3-way overlap so a false negative here
        # would be a filter bug, not a fixture that never had a chance.
        contracts = [_contract("EVT-a0", "A"), _contract("EVT-a1", "A"), _contract("EVT-d0", "D")]

        config = GradePairingConfig(max_components=3, per_stratum_pair_cap=10, per_stratum_triple_cap=10)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        triples = [ev for ev in composed if len(ev.components) == 3]
        assert triples == [], (
            "an A-A-D triple must never form: D is not reachable from root grade A"
        )

    def test_no_duplicate_triple_constituent_sets(self):
        """The and_composer.py invariant Phase 4 relies on (k > idx_b combined
        with seeds always satisfying idx_a < idx_b means each unique index
        triple is generated from exactly one seed) -- verified here through
        grade_guided_compose's own output on a pool large enough to produce
        several distinct triples."""
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        group = _overlapping_group(rng, n, 6)

        candidates, contracts = [], []
        for i, s in enumerate(group):
            eid = f"EVT-a{i}"
            candidates.append(_candidate(eid, f"feat_a{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, "A"))

        config = GradePairingConfig(max_components=3, per_stratum_pair_cap=50, per_stratum_triple_cap=50)
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)

        triples = [ev for ev in composed if len(ev.components) == 3]
        assert len(triples) >= 2, "fixture must produce enough triples to meaningfully check for duplicates"
        constituent_sets = [frozenset(c.expression for c in ev.components) for ev in triples]
        assert len(constituent_sets) == len(set(constituent_sets)), (
            "no two composed triples may share the same constituent set"
        )

    def test_per_stratum_triple_cap_bounds_the_total_triple_budget(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        group = _overlapping_group(rng, n, 8)

        candidates, contracts = [], []
        for i, s in enumerate(group):
            eid = f"EVT-a{i}"
            candidates.append(_candidate(eid, f"feat_a{i}", s, gate, month_idx, n_months))
            contracts.append(_contract(eid, "A"))

        # Only one stratum present ("A_same") -> effective_max_triples ==
        # 1 * per_stratum_triple_cap exactly.
        config = GradePairingConfig(
            max_components=3, per_stratum_pair_cap=50, per_stratum_triple_cap=3,
        )
        composed = grade_guided_compose(candidates, contracts, ts, config, gate)
        triples = [ev for ev in composed if len(ev.components) == 3]
        assert len(triples) <= 3

    def test_fresh_triple_event_ids_never_reused(self):
        gate, ts, month_idx, n_months = _gate_and_ts()
        rng = np.random.default_rng(254)
        n = len(ts)
        s_a0, s_a1, s_b0 = _overlapping_group(rng, n, 3)
        candidates = [
            _candidate("EVT-a0", "feat_a0", s_a0, gate, month_idx, n_months),
            _candidate("EVT-a1", "feat_a1", s_a1, gate, month_idx, n_months),
            _candidate("EVT-b0", "feat_b0", s_b0, gate, month_idx, n_months),
        ]
        contracts = [_contract("EVT-a0", "A"), _contract("EVT-a1", "A"), _contract("EVT-b0", "B")]
        config = GradePairingConfig(max_components=3, per_stratum_pair_cap=10, per_stratum_triple_cap=10)

        composed = grade_guided_compose(candidates, contracts, ts, config, gate)
        input_ids = {c.event_id for c in candidates}
        composed_ids = {c.event_id for c in composed}
        assert composed_ids.isdisjoint(input_ids)
        assert len(composed_ids) == len(composed)


class TestGradeGuidedComposeRealPipeline:
    def test_end_to_end_against_real_event_and_alpha_discovery(self):
        """'Verified'-style integration test: run the real M1 (EventDiscovery)
        and M2-pass-1 (AlphaDiscovery) pipeline on a synthetic KPI table, then
        grade_guided_compose over the real output -- proves the module's
        assumptions about AlphaContract/EventCandidate shapes hold against the
        actual pipeline, not just hand-built fixtures."""
        from forgedge.alpha_discovery import AlphaConfig, AlphaDiscovery
        from forgedge.event_discovery import DiscoveryConfig, EventDiscovery

        df = _make_kpi_table(n=4380, seed=7)
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1),
            max_and_components=1,  # M1 stays returns-blind, 1D-only, per the design
        )
        ed = EventDiscovery(df, cfg)
        candidates = ed.run()
        assert len(candidates) > 50, "fixture must produce a real pool to compose from"

        ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="TEST", timeframe="1H"))
        contracts = ad.run()
        assert len(contracts) == len(candidates)

        gate = ConsistencyGate(cfg.gate_params)
        timestamps = ed.df.index.to_series().reset_index(drop=True)
        config = GradePairingConfig(per_stratum_pair_cap=50, per_stratum_triple_cap=20)
        composed = grade_guided_compose(candidates, contracts, timestamps, config, gate)

        assert composed, "real pipeline output must produce at least one composed candidate"

        grade_of_expr = _grade_lookup_by_expression(candidates, contracts)
        allowed = {("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"),
                   ("A", "B"), ("B", "C"), ("C", "D")}
        strata = _strata_of(composed, grade_of_expr)
        assert strata <= allowed, f"composed pairs outside the allowed adjacency scheme: {strata - allowed}"

        composed_ids = {c.event_id for c in composed}
        input_ids = {c.event_id for c in candidates}
        assert composed_ids.isdisjoint(input_ids)
