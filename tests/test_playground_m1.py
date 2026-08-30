"""Tests for forgedge.playground.m1 — dead events and gate-survival stats.

Builds minimal stand-ins (plain attribute access only, no full dataclasses)
for ``ForgeResult``, ``EventCandidate``, ``AlphaContract``, ``EventDiscovery``,
``RawEvent`` and ``GateResult``/``GateParams``.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from forgedge.playground import dead_event_candidates, gate_survival_observed


def _candidate(event_id, expression="expr"):
    return SimpleNamespace(event_id=event_id, expression=expression)


def _contract(event_candidate_id, direction="long"):
    return SimpleNamespace(event_candidate_id=event_candidate_id, direction=direction)


def _result(ticker, candidates=(), contracts=(), event_discovery=None):
    return SimpleNamespace(
        ticker=ticker,
        candidates=list(candidates),
        contracts=list(contracts),
        event_discovery=event_discovery,
    )


def _gate_result(passed, mean_tpm=1.0, index_of_dispersion=1.2, episode_index_of_dispersion=1.1,
                  n_episodes=5, fail_reason=None):
    return SimpleNamespace(
        passed=passed,
        mean_tpm=mean_tpm,
        index_of_dispersion=index_of_dispersion,
        episode_index_of_dispersion=episode_index_of_dispersion,
        n_episodes=n_episodes,
        fail_reason=fail_reason,
    )


def _raw_event(gate_result):
    return SimpleNamespace(gate_result=gate_result)


def _gate_params(min_tpm=0.5, max_dispersion=1.5, dispersion_margin=1.3, event_counting="episode"):
    return SimpleNamespace(
        min_tpm=min_tpm,
        max_dispersion=max_dispersion,
        dispersion_margin=dispersion_margin,
        event_counting=event_counting,
    )


def _event_discovery(raw_events, gate_params=None):
    return SimpleNamespace(
        raw_events=raw_events,
        config=SimpleNamespace(gate_params=gate_params or _gate_params()),
    )


class TestDeadEventCandidates:
    def test_candidate_with_no_contracts_is_dead(self):
        candidate = _candidate("EVT-1")
        result = _result("T", candidates=[candidate], contracts=[])

        df = dead_event_candidates([result])

        assert len(df) == 1
        assert df.iloc[0]["status"] == "dead"
        assert df.iloc[0]["n_contracts"] == 0

    def test_candidate_with_only_undetermined_contracts(self):
        candidate = _candidate("EVT-1")
        contracts = [
            _contract("EVT-1", direction="undetermined"),
            _contract("EVT-1", direction="undetermined"),
        ]
        result = _result("T", candidates=[candidate], contracts=contracts)

        df = dead_event_candidates([result])

        assert df.iloc[0]["status"] == "undetermined_only"
        assert df.iloc[0]["n_contracts"] == 2
        assert df.iloc[0]["n_undetermined"] == 2

    def test_candidate_with_at_least_one_oriented_contract_is_actionable(self):
        candidate = _candidate("EVT-1")
        contracts = [
            _contract("EVT-1", direction="undetermined"),
            _contract("EVT-1", direction="long"),
        ]
        result = _result("T", candidates=[candidate], contracts=contracts)

        df = dead_event_candidates([result])

        assert df.iloc[0]["status"] == "actionable"

    def test_contracts_from_other_candidates_do_not_count(self):
        candidate = _candidate("EVT-1")
        contracts = [_contract("EVT-OTHER", direction="long")]
        result = _result("T", candidates=[candidate], contracts=contracts)

        df = dead_event_candidates([result])

        assert df.iloc[0]["status"] == "dead"

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = dead_event_candidates([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "event_candidate_id",
            "expression",
            "n_contracts",
            "n_undetermined",
            "status",
        ]

    def test_aggregates_across_multiple_results(self):
        r1 = _result("T1", candidates=[_candidate("EVT-1")], contracts=[])
        r2 = _result("T2", candidates=[_candidate("EVT-2")], contracts=[_contract("EVT-2", "long")])

        df = dead_event_candidates([r1, r2])

        assert set(df["ticker"]) == {"T1", "T2"}
        waste = df[df["status"] != "actionable"]
        assert list(waste["ticker"]) == ["T1"]


class TestGateSurvivalObserved:
    def test_emits_one_row_per_raw_event_with_configured_thresholds(self):
        gp = _gate_params(min_tpm=0.7)
        ed = _event_discovery(
            raw_events=[_raw_event(_gate_result(True)), _raw_event(_gate_result(False, fail_reason="low_tpm"))],
            gate_params=gp,
        )
        result = _result("T", event_discovery=ed)

        df = gate_survival_observed([result])

        assert len(df) == 2
        assert (df["min_tpm"] == 0.7).all()
        assert set(df["passed"]) == {True, False}
        assert df.loc[~df["passed"], "fail_reason"].iloc[0] == "low_tpm"

    def test_skips_raw_events_without_gate_result(self):
        ed = _event_discovery(raw_events=[_raw_event(None)])
        result = _result("T", event_discovery=ed)

        df = gate_survival_observed([result])

        assert df.empty

    def test_skips_result_without_event_discovery(self):
        result = _result("T", event_discovery=None)

        df = gate_survival_observed([result])

        assert df.empty

    def test_skips_result_with_raw_events_not_retained(self):
        ed = _event_discovery(raw_events=None)
        result = _result("T", event_discovery=ed)

        df = gate_survival_observed([result])

        assert df.empty

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = gate_survival_observed([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "mean_tpm",
            "index_of_dispersion",
            "episode_index_of_dispersion",
            "n_episodes",
            "passed",
            "fail_reason",
            "min_tpm",
            "max_dispersion",
            "dispersion_margin",
            "event_counting",
        ]

    def test_downstream_survival_rate_by_ticker(self):
        ed1 = _event_discovery(raw_events=[_raw_event(_gate_result(True)), _raw_event(_gate_result(False))])
        ed2 = _event_discovery(raw_events=[_raw_event(_gate_result(True)), _raw_event(_gate_result(True))])
        r1 = _result("T1", event_discovery=ed1)
        r2 = _result("T2", event_discovery=ed2)

        df = gate_survival_observed([r1, r2])
        rate = df.groupby("ticker")["passed"].mean()

        assert rate["T1"] == pytest.approx(0.5)
        assert rate["T2"] == pytest.approx(1.0)
