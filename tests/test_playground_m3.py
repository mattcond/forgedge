"""Tests for forgedge.playground.m3 — diagnostics vs verdict, lottery winners."""

from types import SimpleNamespace

import pandas as pd
import pytest

from forgedge.playground import diagnostics_vs_verdict, lottery_only_winners


def _contract(alpha_id, grade="A", diagnostics=None, rotation_p=None, rotation_threshold=None):
    alpha_score = SimpleNamespace(grade=grade) if grade is not None else None
    return SimpleNamespace(
        alpha_id=alpha_id,
        alpha_score=alpha_score,
        diagnostics=diagnostics or [],
        rotation_p=rotation_p,
        rotation_threshold=rotation_threshold,
    )


def _response(verdict, rejection_reasons=None):
    return SimpleNamespace(verdict=verdict, rejection_reasons=rejection_reasons or [])


def _result(ticker, rule_responses=()):
    return SimpleNamespace(ticker=ticker, rule_responses=list(rule_responses))


class TestDiagnosticsVsVerdict:
    def test_explodes_multiple_diagnostics(self):
        contract = _contract("A-1", diagnostics=["thin_activation", "short_history"])
        result = _result("T", rule_responses=[(contract, _response("NON-EDGE"))])

        df = diagnostics_vs_verdict([result])

        assert len(df) == 2
        assert set(df["diagnostic"]) == {"thin_activation", "short_history"}
        assert (df["verdict"] == "NON-EDGE").all()

    def test_no_diagnostics_still_yields_one_row(self):
        contract = _contract("A-1", diagnostics=[])
        result = _result("T", rule_responses=[(contract, _response("EDGE"))])

        df = diagnostics_vs_verdict([result])

        assert len(df) == 1
        assert df.iloc[0]["diagnostic"] is None

    def test_grade_carried_through(self):
        contract = _contract("A-1", grade="b", diagnostics=["x"])
        result = _result("T", rule_responses=[(contract, _response("EDGE"))])

        df = diagnostics_vs_verdict([result])

        assert df.iloc[0]["grade"] == "B"

    def test_ungraded_contract_has_none_grade(self):
        contract = _contract("A-1", grade=None, diagnostics=["x"])
        result = _result("T", rule_responses=[(contract, _response("EDGE"))])

        df = diagnostics_vs_verdict([result])

        assert df.iloc[0]["grade"] is None

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = diagnostics_vs_verdict([])

        assert df.empty
        assert list(df.columns) == ["ticker", "alpha_id", "grade", "diagnostic", "verdict"]

    def test_downstream_crosstab(self):
        c1 = _contract("A-1", diagnostics=["thin_activation"])
        c2 = _contract("A-2", diagnostics=["thin_activation"])
        c3 = _contract("A-3", diagnostics=["thin_activation"])
        result = _result(
            "T",
            rule_responses=[
                (c1, _response("NON-EDGE")),
                (c2, _response("NON-EDGE")),
                (c3, _response("EDGE")),
            ],
        )

        df = diagnostics_vs_verdict([result])
        rate = df.groupby("diagnostic")["verdict"].apply(lambda s: (s == "NON-EDGE").mean())

        assert rate["thin_activation"] == pytest.approx(2 / 3)


class TestLotteryOnlyWinners:
    def test_flags_rotation_only_partial_edge(self):
        contract = _contract("A-1", rotation_p=0.08, rotation_threshold=0.05)
        response = _response(
            "PARTIAL-EDGE",
            rejection_reasons=["search-level rotation null not cleared (rotation_p=0.0800 > 0.0500)"],
        )
        result = _result("T", rule_responses=[(contract, response)])

        df = lottery_only_winners([result])

        assert len(df) == 1
        assert bool(df.iloc[0]["rotation_only"]) is True
        assert df.iloc[0]["n_reasons"] == 1

    def test_partial_edge_with_other_reasons_is_not_rotation_only(self):
        contract = _contract("A-1")
        response = _response(
            "PARTIAL-EDGE",
            rejection_reasons=["PF 1.02 < 1.2", "search-level rotation null not cleared (rotation_p=0.08 > 0.05)"],
        )
        result = _result("T", rule_responses=[(contract, response)])

        df = lottery_only_winners([result])

        assert bool(df.iloc[0]["rotation_only"]) is False
        assert df.iloc[0]["n_reasons"] == 2

    def test_partial_edge_without_rotation_reason_is_not_rotation_only(self):
        contract = _contract("A-1")
        response = _response("PARTIAL-EDGE", rejection_reasons=["PF 1.02 < 1.2"])
        result = _result("T", rule_responses=[(contract, response)])

        df = lottery_only_winners([result])

        assert bool(df.iloc[0]["rotation_only"]) is False

    def test_edge_and_non_edge_verdicts_are_excluded(self):
        c1 = _contract("A-1")
        c2 = _contract("A-2")
        result = _result(
            "T",
            rule_responses=[
                (c1, _response("EDGE")),
                (c2, _response("NON-EDGE", rejection_reasons=["PF 1.0 < 1.2"])),
            ],
        )

        df = lottery_only_winners([result])

        assert df.empty

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = lottery_only_winners([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "alpha_id",
            "grade",
            "rotation_p",
            "rotation_threshold",
            "n_reasons",
            "rotation_only",
        ]

    def test_downstream_rate_by_grade(self):
        c1 = _contract("A-1", grade="A")
        c2 = _contract("A-2", grade="A")
        response_rotation_only = _response(
            "PARTIAL-EDGE",
            rejection_reasons=["search-level rotation null not cleared (rotation_p=0.08 > 0.05)"],
        )
        response_other = _response("PARTIAL-EDGE", rejection_reasons=["PF 1.0 < 1.2"])
        result = _result("T", rule_responses=[(c1, response_rotation_only), (c2, response_other)])

        df = lottery_only_winners([result])
        rate = df.groupby("grade")["rotation_only"].mean()

        assert rate["A"] == pytest.approx(0.5)
