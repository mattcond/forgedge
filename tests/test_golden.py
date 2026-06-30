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

_AD_GOLDEN_CANDIDATE_ID = "EVT-close_vol_05-DL-0281"
_AD_GOLDEN = {
    "expression":   "delta_close_vol_05_12 < -0.0257995",
    "direction":    "short",
    "h_star":       16,
    "sell_pct":     pytest.approx(0.1403, rel=1e-4),
    "lift":         pytest.approx(0.139807, rel=1e-3),
    "n_activations": 92,
    "promoted":     True,
    "status":       "HYPOTHESIS",
}

_RD_GOLDEN = {
    "verdict":        "EDGE",
    "profit_factor":  pytest.approx(2.7587, rel=1e-3),
    "total_trades":   109,
    "expectancy":     pytest.approx(0.057913, rel=1e-3),
}

# A deterministic *long* contract under the default PROJ_LOG target_mode (issue
# #109).  The golden short contract above reverts to ABS (short → ABS), so this
# pins the PROJ_LOG long path.
_AD_GOLDEN_LONG_CANDIDATE_ID = "EVT-ratio_close_vol05_vo-ZS-1160"
_AD_GOLDEN_LONG = {
    "expression":    "zs_ratio_close_vol05_vol24_96 > 1",
    "direction":     "long",
    "h_star":        16,
    "sell_pct":      pytest.approx(0.198542, rel=1e-4),
    "lift":          pytest.approx(0.144720, rel=1e-3),
    "n_activations": 106,
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
