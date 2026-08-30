"""Tests for forgedge.playground.m2 — pure data-shape transforms.

These build the minimal stand-ins the two functions actually read (via plain
attribute access) rather than the full ``AlphaContract``/``RuleDiscoveryResponse``
dataclasses, whose nested required fields (event_stats, market_structure,
walk_forward, ...) are irrelevant to this module's logic.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from forgedge.playground import discard_reasons_by_grade, undetermined_direction_by_family
from forgedge.playground.m2 import _feature_family


def _contract(alpha_id, grade, event_candidate_id="EVT-1", direction="long"):
    alpha_score = SimpleNamespace(grade=grade) if grade is not None else None
    return SimpleNamespace(
        alpha_id=alpha_id,
        alpha_score=alpha_score,
        event_candidate_id=event_candidate_id,
        direction=direction,
    )


def _response(verdict, rejection_reasons=None, failed_condition=None):
    entry_optimization = SimpleNamespace(failed_condition=failed_condition)
    return SimpleNamespace(
        verdict=verdict,
        rejection_reasons=rejection_reasons or [],
        entry_optimization=entry_optimization,
    )


def _component(source_feature, source_cols=None):
    return SimpleNamespace(source_feature=source_feature, source_cols=source_cols or [])


def _candidate(event_id, components):
    return SimpleNamespace(event_id=event_id, components=components)


def _result(ticker, rule_responses=(), candidates=(), contracts=()):
    return SimpleNamespace(
        ticker=ticker,
        rule_responses=list(rule_responses),
        candidates=list(candidates),
        contracts=list(contracts),
    )


class TestDiscardReasonsByGrade:
    def test_keeps_only_non_edge_of_requested_grade(self):
        a_rejected = _contract("A-1", grade="A")
        a_accepted = _contract("A-2", grade="A")
        b_rejected = _contract("B-1", grade="B")

        result = _result(
            "BTCUSDC",
            rule_responses=[
                (a_rejected, _response("NON-EDGE", rejection_reasons=["pf_below_floor"])),
                (a_accepted, _response("EDGE")),
                (b_rejected, _response("NON-EDGE", rejection_reasons=["pf_below_floor"])),
            ],
        )

        df = discard_reasons_by_grade([result], grade="A")

        assert list(df["alpha_id"]) == ["A-1"]
        assert list(df["reason"]) == ["pf_below_floor"]
        assert df.iloc[0]["ticker"] == "BTCUSDC"

    def test_grade_filter_is_case_insensitive(self):
        contract = _contract("A-1", grade="a")
        result = _result(
            "T",
            rule_responses=[(contract, _response("NON-EDGE", rejection_reasons=["x"]))],
        )

        df = discard_reasons_by_grade([result], grade="A")

        assert len(df) == 1

    def test_ungraded_contract_never_matches(self):
        contract = _contract("A-1", grade=None)
        result = _result(
            "T",
            rule_responses=[(contract, _response("NON-EDGE", rejection_reasons=["x"]))],
        )

        df = discard_reasons_by_grade([result], grade="A")

        assert df.empty

    def test_explodes_multiple_reasons_into_multiple_rows(self):
        contract = _contract("A-1", grade="A")
        result = _result(
            "T",
            rule_responses=[
                (
                    contract,
                    _response(
                        "NON-EDGE",
                        rejection_reasons=["pf_below_floor", "insufficient_power"],
                        failed_condition="sharpe",
                    ),
                )
            ],
        )

        df = discard_reasons_by_grade([result], grade="A")

        assert len(df) == 2
        assert set(df["reason"]) == {"pf_below_floor", "insufficient_power"}
        assert (df["failed_condition"] == "sharpe").all()

    def test_empty_rejection_reasons_still_yields_one_row(self):
        contract = _contract("A-1", grade="A")
        result = _result("T", rule_responses=[(contract, _response("NON-EDGE"))])

        df = discard_reasons_by_grade([result], grade="A")

        assert len(df) == 1
        assert df.iloc[0]["reason"] is None

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = discard_reasons_by_grade([], grade="A")

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "alpha_id",
            "event_candidate_id",
            "reason",
            "failed_condition",
        ]

    def test_aggregates_across_multiple_results(self):
        c1 = _contract("A-1", grade="A")
        c2 = _contract("A-2", grade="A")
        r1 = _result("T1", rule_responses=[(c1, _response("NON-EDGE", rejection_reasons=["x"]))])
        r2 = _result("T2", rule_responses=[(c2, _response("NON-EDGE", rejection_reasons=["x"]))])

        df = discard_reasons_by_grade([r1, r2], grade="A")

        assert set(df["ticker"]) == {"T1", "T2"}


class TestFeatureFamily:
    @pytest.mark.parametrize(
        "source_feature, expected",
        [
            ("close_rsi_25", "rsi"),
            ("close_ema_09", "ema"),
            ("volume_zscore_48", "zscore"),
            ("not_a_conforming_name", "other"),
        ],
    )
    def test_native_columns(self, source_feature, expected):
        assert _feature_family(source_feature, []) == expected

    def test_pair_features_bucketed_by_arity(self):
        assert _feature_family("macd_vs_signal", ["macd", "macd_signal"]) == "cross_pair"

    def test_triple_features_bucketed_by_arity(self):
        assert _feature_family("band_position", ["close", "lower", "upper"]) == "cross_triple"

    def test_native_feature_with_nonempty_length_one_source_cols_still_resolves_by_name(self):
        # Regression: EventComponent.source_cols is documented "empty for
        # arity-1 (native) features" but observed non-empty (length 1) on
        # real candidates (e.g. "close_ret_03") — truthiness alone must not
        # route a native feature into the cross-feature buckets.
        assert _feature_family("close_ret_03", ["close"]) == "ret"


class TestUndeterminedDirectionByFamily:
    def test_emits_one_row_per_component(self):
        candidate = _candidate(
            "EVT-1",
            components=[_component("close_rsi_25"), _component("close_ema_09")],
        )
        contract = _contract("A-1", grade="A", event_candidate_id="EVT-1", direction="undetermined")
        result = _result("T", candidates=[candidate], contracts=[contract])

        df = undetermined_direction_by_family([result])

        assert len(df) == 2
        assert set(df["family"]) == {"rsi", "ema"}
        assert (df["direction"] == "undetermined").all()

    def test_contract_without_matching_candidate_is_skipped(self):
        contract = _contract("A-1", grade="A", event_candidate_id="EVT-MISSING")
        result = _result("T", candidates=[], contracts=[contract])

        df = undetermined_direction_by_family([result])

        assert df.empty

    def test_downstream_undetermined_rate_by_family(self):
        rsi_candidate = _candidate("EVT-RSI", components=[_component("close_rsi_25")])
        ema_candidate = _candidate("EVT-EMA", components=[_component("close_ema_09")])
        contracts = [
            _contract("A-1", grade="A", event_candidate_id="EVT-RSI", direction="undetermined"),
            _contract("A-2", grade="A", event_candidate_id="EVT-RSI", direction="long"),
            _contract("A-3", grade="A", event_candidate_id="EVT-EMA", direction="long"),
        ]
        result = _result("T", candidates=[rsi_candidate, ema_candidate], contracts=contracts)

        df = undetermined_direction_by_family([result])
        rate = df.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())

        assert rate["rsi"] == pytest.approx(0.5)
        assert rate["ema"] == pytest.approx(0.0)

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = undetermined_direction_by_family([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "alpha_id",
            "event_candidate_id",
            "family",
            "direction",
        ]
