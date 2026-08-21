"""Golden regression tests — ADA 1D known pipeline outputs must never change.

A single session-scoped fixture runs the full forge() pipeline once on the
committed ADA_1D_TRAIN fixture.  Three test classes assert on deterministic
outputs at each pipeline stage:

  TestGoldenEventDiscovery  — EventDiscovery: known atomic candidate stats
  TestGoldenAlphaDiscovery  — AlphaDiscovery: known contract derived-target
  TestGoldenRuleDiscovery   — RuleDiscovery:  known EDGE verdict and PF

If any surgical modification to EventDiscovery, ANDComposer, AlphaDiscovery
or RuleDiscovery changes these values, the relevant test class breaks,
pinpointing exactly which stage was affected.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forgedge import forge
from forgedge.forge import ForgeResult

_FIXTURE = Path(__file__).parent / "fixtures" / "ADA_1D_TRAIN.parquet"

# ---------------------------------------------------------------------------
# Expected values — captured from first deterministic run, never changed
# ---------------------------------------------------------------------------

# Golden values re-pinned after fix #134 (episode counting as default, PR #138).
# Event IDs changed because episode-mode gate reorders candidate numbering;
# the underlying expressions, signals, and RD metrics are unchanged.
#
# AD/RD golden values re-pinned again after the timeframe-scaled default
# horizon grid: forge(timeframe="1D") now scans the daily-calibrated grid
# (1, 2, 3, 5, 7, 10) instead of the hourly default (1..48), so every derived
# target on this 1D fixture changed (the previous golden contracts selected
# h*=16, which no longer exists in the grid).  The whole-table monthly stats
# fix in rule_discovery.backtest._month_index (final month no longer dropped)
# also shifts the RD in-sample summary.  Event Discovery goldens are
# untouched — Modulo 1 never sees the horizon grid.
_ED_GOLDEN_ID = "EVT-close_ret_03-ID-0000"
_ED_GOLDEN = {
    "expression":           "close_ret_03 < -0.138425",
    "n_activations":        27,
    "mean_tpm":             pytest.approx(0.931034, rel=1e-4),
    "zero_months":          15,
    "index_of_dispersion":  pytest.approx(1.605820, rel=1e-4),
    "gate_passed":          True,
    "n_components":         1,
}

_AD_GOLDEN_CANDIDATE_ID = "EVT-ratio_close_ret03_re-PR-0527"
_AD_GOLDEN = {
    "expression":   "pr_ratio_close_ret03_ret96_96 < 0.166667",
    "direction":    "short",
    "h_star":       5,
    "sell_pct":     pytest.approx(0.059684, rel=1e-4),
    # lift re-pinned 0.072515 → 0.069682 with the TimeBudget purge: the last
    # h* IS bars are excluded from the win-rate/base-rate (their forward
    # window crosses into the OOS tail).  Every other pin is unchanged.
    "lift":         pytest.approx(0.069682, rel=1e-3),
    "n_activations": 93,
    "promoted":     True,
    "status":       "HYPOTHESIS",
}

# verdict re-pinned EDGE → PARTIAL-EDGE with the default search-level rotation
# null (FastRotationNull): on this 1D fixture the whole discovery session does
# not beat its own rotation null (search p ≈ 0.7 on the full ADA set), so the
# golden rule is capped — economics unchanged (PF / trades / expectancy are
# identical; only the verdict gate moved).
#
# Economics re-pinned again under walk-forward selection (§3.4,
# selection_mode="walk_forward" default): the published operating point now
# comes from the last walk-forward train window and the IS summary is
# computed on the selection span [start, last train window end) — the final
# test window no longer feeds any selection or metric.  Previous full-sample
# values: PF 2.0042, 105 trades, expectancy 0.019196.
#
# Verdict re-pinned PARTIAL-EDGE → INSUFFICIENT-DATA with the §3.2 power gate:
# the golden rule's pooled OOS ledger (96 trades — the count floor passes) has
# a minimum detectable expectancy of 0.0160 vs a claimed IS expectancy of
# 0.0114, so the OOS cannot confirm an effect of the claimed size.  Economics
# pins unchanged; the rule keeps its ValidatedRule but is not tradeable.
#
# ── Re-pinned again with entry_mode="auto" as the default (issue #185) ──────
#
# Values under the "limit" default:
#     INSUFFICIENT-DATA · PF 1.7337 · 82 trades · expectancy 0.011362
#
# Those moved because the *measurement* changed, not because the rule did.  In
# "limit" mode the grid varies buy_drop_pct, so the limit entry doubles as an
# entry-price optimiser: a deeper discount fills only on the paths that came
# back down to it, and the PF is computed on that favourable subset.  Under
# "auto" the verdict comes from a market entry (fill 1.00 against 0.88), so it
# measures the signal.  PF fell, trades rose, and the pooled OOS became able to
# confirm the effect, lifting the §3.2 power gate's demotion.
#
# ── Re-pinned again at step 7 (issue #177) ─────────────────────────────────
#
# Values under "auto" before this step:
#     PARTIAL-EDGE · PF 1.5818 · 93 trades · expectancy 0.016959
#
# Two changes reach this rule, and the second is the one that flips it.
#
# 1. `min_train_months` is derived from the rate now (20 months, not a fixed
#    6), so the selection span is longer: 93 -> 98 trades, PF 1.58 -> 1.67.
#
# 2. The verdict is NON-EDGE on `expectancy not significant (p=0.069)`, and
#    that is F16 doing exactly what it was built for:
#
#        nominal trades          98
#        mean concurrency         1.88
#        effective sample        52.09
#        t  = 2.0623 nominal  ->  1.5035 effective   (ratio sqrt(98/52.09))
#        p  = below 0.05      ->  0.069438
#
#    The rule opens a position on every active bar, so 98 trades are not 98
#    independent observations — they overlap on one price path.  Counting them
#    as independent overstated the t statistic by 37%, and `max_ttest_p` is one
#    of the three hard gates.  This rule was previously admitted on overstated
#    significance; it is not newly bad, it was never significant.
#
#    The economics are untouched and stay nominal: PF 1.6746 and expectancy
#    0.014758 are computed on all 98 trades, because they are reproducible in
#    production given the capital to fund the concurrent positions.
#
# Across the fixture the same correction moves 9 rules from PARTIAL-EDGE to
# NON-EDGE (44 -> 35), with NON-EDGE 325 -> 333.
#
# Event and Alpha Discovery goldens are untouched: none of this is visible to
# M1 or M2.
#
# ── Step 8 (issue #178, F3): checked, and the pins below do NOT move ────────
#
# `c_norm` was rebuilt on the index of dispersion and `scoring.pf_min_tpm` now
# resolves from `criteria.min_tpm`, both of which feed `pf_score_tpm` — the
# grid's ranking key *and* a gate in `_passes`.  The issue expected the
# published operating point to shift.  On this fixture it does not: the golden
# rule's cell wins under both scorings (its `c_norm` rises 0.2401 -> 0.4253 and
# its `pf_score_tpm` 0.4020 -> 0.7122, but so does every competitor's, and the
# ordering survives).  Measured across the fixture the hidden PF threshold
# `min_pf_score_tpm / c_norm` falls from a median of 1.67 to 0.47, and stops
# exceeding the declared `min_profit_factor=2.0` for 34.3% -> 0.8% of rules.
#
# Left unpinned deliberately: this is the result, not an omission.
#
# ── Step 9 (issue #179, F5): checked, and the pins below do NOT move ────────
#
# Every field meaning "N bars" is now converted at the session's bar duration:
# on this 1D fixture `buy_delay_bar` 6 -> 1 (six *days* of resting limit order
# -> one), `target_h`'s class default 24 -> 10, `stable_window` 12 -> 2.  Two
# reasons the goldens survive it:
#
#  - `target_h` is seeded from the contract's `holding_period_h` on every rule
#    this fixture produces, so its class default reached 0% of them — the
#    conversion is correctness hygiene there, not a live change.
#  - `stable_window` changes the `regime_stable` column (31.9% -> 85.7% of
#    bars marked stable), and that column is read only under
#    `AlphaConfig.use_stable_regime_only`, which defaults to False.  The
#    column's *meaning* was wrong on daily data; nothing on the default path
#    was consuming it.
#
# `buy_delay_bar` does move: across the fixture the published operating points
# go 37 -> 38, one rule surviving that the six-day order had been failing.
# The golden rule is not among them.
#
# ── Step 10 (issue #180, F6): checked, and the pins below do NOT move ───────
#
# `forge()` now builds one temporal axis instead of forwarding whatever the
# caller passed, and threads it through all three stages. Nothing moves here
# because the axis it makes explicit is the one the run already had:
#
#  - M1 keeps `train_ratio=1.0` (D13 — invariant #1, it never observes the
#    forward return), so the candidate set is untouched. The budget now
#    *records* that choice instead of leaving it to be inferred from a split
#    M1 does not use.
#  - M2's split and purge are unchanged. The session budget is sized on the
#    configured grid and M2 widens it when enrichment scans further, so the
#    purge width is what it always was — widening, never narrowing.
#  - M3's walk-forward origin is deliberately unchanged (see
#    `WalkForwardSplit.tests_in_sample`), so its folds are the same folds.
#
# What is new is measurable rather than numeric: on this fixture 3 of every
# rule's 4 walk-forward folds test inside M2's in-sample region — 117 rules out
# of 117 — which is now reported on `WalkForwardResult.n_splits_in_sample`
# rather than being invisible.
#
# ── Step 11 (issue #182, F9): checked, and the pins below do NOT move ───────
#
# The five per-hypothesis significance levels — `max_p_value`, `ic_max_p`,
# `max_ttest_p`, `max_rotation_p`, `RotationConfig.alpha` — now derive from
# `PipelineContext.alpha` instead of being five independent literals. They
# were all already 0.05, agreeing by coincidence rather than by construction,
# so this buys coherence and changes no number: the same five values reach the
# same gates. D16 predicted exactly that, and it holds.
#
# `fdr_q`, `oos_max_p` and `min_pass_rate` are deliberately not tied to alpha
# (a false-discovery rate, a confirmation level, and a vote), so they could not
# have moved either.
#
# ── #196 (tail of F5): checked, and the pins below do NOT move ──────────────
#
# `AlphaConfig.horizon_grid` is session-resolved now, on every path rather than
# only when `forge()` built the config itself.  This fixture runs `forge()`
# without an `alpha_config`, which is precisely the path that already got the
# substituted daily grid `(1, 2, 3, 5, 7, 10)` — so the value reaching M2 is
# identical and every downstream pin with it.
#
# The gap this closes is the *other* path: an explicit `AlphaConfig` on daily
# candles used to keep the hourly grid and scan up to 48 days.  No golden
# exercises that path, which is why nothing here moves and why the coverage
# for it is in `TestTheHorizonGridFollowsTheSession` instead.
_RD_GOLDEN = {
    "verdict":        "NON-EDGE",
    "profit_factor":  pytest.approx(1.6746, rel=1e-3),
    "total_trades":   98,
    "expectancy":     pytest.approx(0.014758, rel=1e-3),
}

# A deterministic *long* contract under the default PROJ_LOG target_mode (issue
# #109).  The golden short contract above reverts to ABS (short → ABS), so this
# pins the PROJ_LOG long path.
_AD_GOLDEN_LONG_CANDIDATE_ID = "EVT-close_ret_03-DL-0062"
_AD_GOLDEN_LONG = {
    "expression":    "delta_close_ret_03_1 > 0.07038",
    "direction":     "long",
    "h_star":        3,
    "sell_pct":      pytest.approx(0.056136, rel=1e-4),
    "lift":          pytest.approx(0.160894, rel=1e-3),
    "n_activations": 64,
    "promoted":      True,
    "status":        "HYPOTHESIS",
}

# ---------------------------------------------------------------------------
# Session fixtures — pipeline runs once, derived fixtures reuse the result
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def forge_result() -> ForgeResult:
    kpi = pd.read_parquet(_FIXTURE)
    return forge(
        kpi,
        ticker="ADAUSDC",
        timeframe="1D",
        run_rule_discovery=True,
        run_registry=False,
        progress=False,
    )


@pytest.fixture(scope="session")
def golden_candidate(forge_result):
    by_id = {c.event_id: c for c in forge_result.candidates}
    cand = by_id.get(_ED_GOLDEN_ID)
    assert cand is not None, (
        f"Golden event {_ED_GOLDEN_ID!r} not found in candidates. "
        "EventDiscovery output changed."
    )
    return cand


@pytest.fixture(scope="session")
def golden_contract(forge_result):
    by_cand_id = {c.event_candidate_id: c for c in forge_result.contracts}
    contract = by_cand_id.get(_AD_GOLDEN_CANDIDATE_ID)
    assert contract is not None, (
        f"Golden candidate {_AD_GOLDEN_CANDIDATE_ID!r} not found in contracts. "
        "AlphaDiscovery output changed."
    )
    return contract


@pytest.fixture(scope="session")
def golden_long_contract(forge_result):
    by_cand_id = {c.event_candidate_id: c for c in forge_result.contracts}
    contract = by_cand_id.get(_AD_GOLDEN_LONG_CANDIDATE_ID)
    assert contract is not None, (
        f"Golden long candidate {_AD_GOLDEN_LONG_CANDIDATE_ID!r} not found in "
        "contracts. AlphaDiscovery PROJ_LOG long output changed."
    )
    return contract


@pytest.fixture(scope="session")
def golden_response(forge_result):
    by_cand_id = {
        contract.event_candidate_id: response
        for contract, response in forge_result.rule_responses
    }
    response = by_cand_id.get(_AD_GOLDEN_CANDIDATE_ID)
    assert response is not None, (
        f"No RuleDiscovery response for {_AD_GOLDEN_CANDIDATE_ID!r}. "
        "RuleDiscovery output changed."
    )
    return response


# ---------------------------------------------------------------------------
# TestGoldenEventDiscovery
# ---------------------------------------------------------------------------

class TestGoldenEventDiscovery:
    """EventDiscovery: first atomic candidate must always have the same stats."""

    def test_expression(self, golden_candidate):
        assert golden_candidate.expression == _ED_GOLDEN["expression"]

    def test_n_activations(self, golden_candidate):
        assert golden_candidate.activation_stats.n_activations == _ED_GOLDEN["n_activations"]

    def test_mean_tpm(self, golden_candidate):
        assert golden_candidate.activation_stats.mean_tpm == _ED_GOLDEN["mean_tpm"]

    def test_zero_months(self, golden_candidate):
        assert golden_candidate.activation_stats.zero_months == _ED_GOLDEN["zero_months"]

    def test_index_of_dispersion(self, golden_candidate):
        assert golden_candidate.activation_stats.index_of_dispersion == _ED_GOLDEN["index_of_dispersion"]

    def test_gate_passed(self, golden_candidate):
        assert golden_candidate.consistency_gate.passed == _ED_GOLDEN["gate_passed"]

    def test_n_components(self, golden_candidate):
        assert len(golden_candidate.components) == _ED_GOLDEN["n_components"]


# ---------------------------------------------------------------------------
# TestGoldenAlphaDiscovery
# ---------------------------------------------------------------------------

class TestGoldenAlphaDiscovery:
    """AlphaDiscovery: known contract derived-target and promotion status."""

    def test_expression(self, golden_contract):
        assert golden_contract.event_expression == _AD_GOLDEN["expression"]

    def test_direction(self, golden_contract):
        assert golden_contract.direction == _AD_GOLDEN["direction"]

    def test_h_star(self, golden_contract):
        assert golden_contract.derived_target.holding_period_h == _AD_GOLDEN["h_star"]

    def test_sell_pct(self, golden_contract):
        assert golden_contract.derived_target.sell_pct == _AD_GOLDEN["sell_pct"]

    def test_lift(self, golden_contract):
        assert golden_contract.event_stats.lift == _AD_GOLDEN["lift"]

    def test_n_activations(self, golden_contract):
        assert golden_contract.event_stats.n_activations == _AD_GOLDEN["n_activations"]

    def test_promoted(self, golden_contract):
        assert golden_contract.promoted == _AD_GOLDEN["promoted"]

    def test_status(self, golden_contract):
        assert golden_contract.status == _AD_GOLDEN["status"]


class TestGoldenAlphaDiscoveryProjLong:
    """AlphaDiscovery PROJ_LOG long path: a known long contract under the
    default ``target_mode="proj"`` (issue #109)."""

    def test_expression(self, golden_long_contract):
        assert golden_long_contract.event_expression == _AD_GOLDEN_LONG["expression"]

    def test_direction(self, golden_long_contract):
        assert golden_long_contract.direction == _AD_GOLDEN_LONG["direction"]

    def test_h_star(self, golden_long_contract):
        assert golden_long_contract.derived_target.holding_period_h == _AD_GOLDEN_LONG["h_star"]

    def test_sell_pct(self, golden_long_contract):
        assert golden_long_contract.derived_target.sell_pct == _AD_GOLDEN_LONG["sell_pct"]

    def test_lift(self, golden_long_contract):
        assert golden_long_contract.event_stats.lift == _AD_GOLDEN_LONG["lift"]

    def test_n_activations(self, golden_long_contract):
        assert golden_long_contract.event_stats.n_activations == _AD_GOLDEN_LONG["n_activations"]

    def test_promoted(self, golden_long_contract):
        assert golden_long_contract.promoted == _AD_GOLDEN_LONG["promoted"]

    def test_status(self, golden_long_contract):
        assert golden_long_contract.status == _AD_GOLDEN_LONG["status"]


# ---------------------------------------------------------------------------
# TestGoldenRuleDiscovery
# ---------------------------------------------------------------------------

class TestGoldenRuleDiscovery:
    """RuleDiscovery: known EDGE verdict and profit factor for golden contract."""

    def test_verdict(self, golden_response):
        assert golden_response.verdict == _RD_GOLDEN["verdict"]

    def test_profit_factor(self, golden_response):
        assert golden_response.in_sample_summary.profit_factor == _RD_GOLDEN["profit_factor"]

    def test_total_trades(self, golden_response):
        assert golden_response.in_sample_summary.total_trades == _RD_GOLDEN["total_trades"]

    def test_expectancy(self, golden_response):
        assert golden_response.in_sample_summary.expectancy == _RD_GOLDEN["expectancy"]
