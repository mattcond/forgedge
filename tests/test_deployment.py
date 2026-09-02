"""Tests for forgedge.deployment — promotion gate, export, monitoring manifest."""

import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from forgedge.deployment import (
    PromotionGateConfig,
    export_rules,
    monitoring_manifest,
    promotion_gate,
)


def _candidate(event_id):
    return SimpleNamespace(event_id=event_id)


def _validated_rule(expression="expr", event_candidate_id="EVT-1", direction="long"):
    params = SimpleNamespace(
        direction=direction, buy_type="market", buy_drop_pct=0.0,
        buy_delay_bar=1, sell_pct=0.02, target_h=5, fee=0.001,
    )
    return SimpleNamespace(
        expression=expression,
        event_candidate_id=event_candidate_id,
        params=params,
        to_dict=lambda: {
            "expression": expression,
            "event_candidate_id": event_candidate_id,
            "direction": direction,
            "entry_mode": "market",
            "buy_drop_pct": 0.0,
            "buy_delay_bar": 1,
            "sell_pct": 0.02,
            "target_h": 5,
            "fee": 0.001,
        },
    )


def _contract(alpha_id, grade="A", event_candidate_id="EVT-1"):
    alpha_score = SimpleNamespace(grade=grade) if grade is not None else None
    return SimpleNamespace(alpha_id=alpha_id, alpha_score=alpha_score, event_candidate_id=event_candidate_id)


def _split(profit_factor):
    return SimpleNamespace(test_summary=SimpleNamespace(profit_factor=profit_factor))


def _walk_forward(consistency, fold_pfs=None):
    if consistency is None:
        return None
    splits = [_split(pf) for pf in fold_pfs] if fold_pfs is not None else []
    return SimpleNamespace(consistency=consistency, splits=splits)


def _response(verdict, rejection_reasons=None, consistency=0.8, validated_rule=None, fold_pfs=None):
    return SimpleNamespace(
        verdict=verdict,
        is_edge=verdict in ("EDGE", "PARTIAL-EDGE"),
        rejection_reasons=rejection_reasons or [],
        walk_forward=_walk_forward(consistency, fold_pfs=fold_pfs),
        validated_rule=validated_rule if validated_rule is not None else _validated_rule(),
    )


def _result(ticker, candidates=(), rule_responses=()):
    return SimpleNamespace(ticker=ticker, candidates=list(candidates), rule_responses=list(rule_responses))


def _doc(source_alpha_id, is_duplicate=None, classification=None):
    return SimpleNamespace(source_alpha_id=source_alpha_id, is_duplicate=is_duplicate, classification=classification)


def _registry(documents):
    return SimpleNamespace(documents=list(documents))


class TestPromotionGate:
    def test_only_edge_and_partial_edge_included(self):
        c1, c2 = _contract("A-1"), _contract("A-2")
        result = _result(
            "T",
            rule_responses=[
                (c1, _response("EDGE")),
                (c2, _response("NON-EDGE")),
            ],
        )

        df = promotion_gate([result])

        assert list(df["alpha_id"]) == ["A-1"]

    def test_rotation_only_flag(self):
        contract = _contract("A-1")
        response = _response(
            "PARTIAL-EDGE",
            rejection_reasons=["search-level rotation null not cleared (rotation_p=0.08 > 0.05)"],
        )
        result = _result("T", rule_responses=[(contract, response)])

        df = promotion_gate([result])

        assert bool(df.iloc[0]["rotation_only"]) is True
        # default config does not block on rotation_only
        assert bool(df.iloc[0]["promotable"]) is True

    def test_blocks_duplicate_by_default(self):
        contract = _contract("A-1")
        response = _response("EDGE")
        result = _result("T", rule_responses=[(contract, response)])
        registry = _registry([_doc("A-1", is_duplicate=True)])

        df = promotion_gate([result], registries=[registry])

        assert bool(df.iloc[0]["is_duplicate"]) is True
        assert bool(df.iloc[0]["promotable"]) is False

    def test_blocks_isolated_by_default(self):
        contract = _contract("A-1")
        response = _response("EDGE")
        result = _result("T", rule_responses=[(contract, response)])
        registry = _registry([_doc("A-1", classification="ISOLATED")])

        df = promotion_gate([result], registries=[registry])

        assert bool(df.iloc[0]["is_isolated"]) is True
        assert bool(df.iloc[0]["promotable"]) is False

    def test_no_registries_leaves_duplicate_and_isolated_none(self):
        contract = _contract("A-1")
        response = _response("EDGE")
        result = _result("T", rule_responses=[(contract, response)])

        df = promotion_gate([result])

        assert df.iloc[0]["is_duplicate"] is None
        assert df.iloc[0]["is_isolated"] is None
        assert bool(df.iloc[0]["promotable"]) is True

    def test_low_consistency_blocks_by_default(self):
        contract = _contract("A-1")
        response = _response("EDGE", consistency=0.3)
        result = _result("T", rule_responses=[(contract, response)])

        df = promotion_gate([result])

        assert bool(df.iloc[0]["promotable"]) is False

    def test_consistency_check_disabled(self):
        contract = _contract("A-1")
        response = _response("EDGE", consistency=0.3)
        result = _result("T", rule_responses=[(contract, response)])
        config = PromotionGateConfig(require_consistency=False)

        df = promotion_gate([result], config=config)

        assert bool(df.iloc[0]["promotable"]) is True

    def test_block_rotation_only_when_configured(self):
        contract = _contract("A-1")
        response = _response(
            "PARTIAL-EDGE",
            rejection_reasons=["search-level rotation null not cleared (rotation_p=0.08 > 0.05)"],
        )
        result = _result("T", rule_responses=[(contract, response)])
        config = PromotionGateConfig(block_rotation_only=True)

        df = promotion_gate([result], config=config)

        assert bool(df.iloc[0]["promotable"]) is False

    def test_fold_stability_score_computed_but_off_by_default(self):
        contract = _contract("A-1")
        # mean=2.0, std(ddof=1)=sqrt(2) -> score=2-sqrt(2); gate off by
        # default, so it doesn't block regardless of the score's value.
        response = _response("EDGE", fold_pfs=[1.0, 3.0])
        result = _result("T", rule_responses=[(contract, response)])

        df = promotion_gate([result])

        assert df.iloc[0]["fold_stability_score"] == pytest.approx(2.0 - 2**0.5)
        assert bool(df.iloc[0]["promotable"]) is True

    def test_fold_stability_score_none_below_two_folds(self):
        contract = _contract("A-1")
        response = _response("EDGE", fold_pfs=[2.0])
        result = _result("T", rule_responses=[(contract, response)])

        df = promotion_gate([result])

        assert df.iloc[0]["fold_stability_score"] is None

    def test_fold_stability_score_caps_sentinel_profit_factor(self):
        contract = _contract("A-1")
        # Without capping, 9999.0 would blow up the mean/std; with the
        # default 10.0 cap: fold_pfs -> [1.0, 10.0], mean=5.5, std=6.3639...
        response = _response("EDGE", fold_pfs=[1.0, 9999.0])
        result = _result("T", rule_responses=[(contract, response)])

        df = promotion_gate([result])

        expected_mean = (1.0 + 10.0) / 2
        expected_std = pd.Series([1.0, 10.0]).std(ddof=1)
        assert df.iloc[0]["fold_stability_score"] == pytest.approx(expected_mean - expected_std)

    def test_low_fold_stability_score_blocks_when_configured(self):
        contract = _contract("A-1")
        # One lucky fold (capped 10.0) plus a weak one (0.5): high mean, high
        # variance -> low stability score despite a strong pooled-looking PF.
        response = _response("EDGE", fold_pfs=[0.5, 9999.0])
        result = _result("T", rule_responses=[(contract, response)])
        config = PromotionGateConfig(min_fold_stability_score=5.0)

        df = promotion_gate([result], config=config)

        assert bool(df.iloc[0]["promotable"]) is False

    def test_fold_stability_gate_does_not_block_when_score_is_none(self):
        contract = _contract("A-1")
        response = _response("EDGE", fold_pfs=[2.0])  # single fold -> None
        result = _result("T", rule_responses=[(contract, response)])
        config = PromotionGateConfig(min_fold_stability_score=5.0)

        df = promotion_gate([result], config=config)

        assert df.iloc[0]["fold_stability_score"] is None
        assert bool(df.iloc[0]["promotable"]) is True

    def test_consistent_folds_pass_fold_stability_gate(self):
        contract = _contract("A-1")
        response = _response("EDGE", fold_pfs=[2.43, 2.15, 1.94, 1.18])
        result = _result("T", rule_responses=[(contract, response)])
        config = PromotionGateConfig(min_fold_stability_score=1.0)

        df = promotion_gate([result], config=config)

        assert bool(df.iloc[0]["promotable"]) is True

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = promotion_gate([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "alpha_id",
            "grade",
            "verdict",
            "rotation_only",
            "is_duplicate",
            "is_isolated",
            "consistency",
            "fold_stability_score",
            "promotable",
        ]


class TestExportRules:
    def test_writes_pkl_and_yaml_for_promotable_contract(self, tmp_path):
        candidate = _candidate("EVT-1")
        contract = _contract("A-1", event_candidate_id="EVT-1")
        response = _response("EDGE")
        result = _result("T", candidates=[candidate], rule_responses=[(contract, response)])

        manifest = export_rules([result], tmp_path)

        assert len(manifest) == 1
        pkl_path = tmp_path / "A-1.pkl"
        yaml_path = tmp_path / "A-1.yaml"
        assert pkl_path.exists()
        assert yaml_path.exists()
        with open(pkl_path, "rb") as fh:
            loaded = pickle.load(fh)
        assert loaded.event_id == "EVT-1"
        yaml_text = yaml_path.read_text()
        assert "alpha_id: A-1" in yaml_text
        assert "ticker: T" in yaml_text

    def test_skips_non_promotable_by_default(self, tmp_path):
        candidate = _candidate("EVT-1")
        contract = _contract("A-1", event_candidate_id="EVT-1")
        response = _response("EDGE", consistency=0.1)  # below default floor
        result = _result("T", candidates=[candidate], rule_responses=[(contract, response)])

        manifest = export_rules([result], tmp_path)

        assert manifest.empty
        assert not (tmp_path / "A-1.pkl").exists()

    def test_promotable_only_false_exports_everything(self, tmp_path):
        candidate = _candidate("EVT-1")
        contract = _contract("A-1", event_candidate_id="EVT-1")
        response = _response("EDGE", consistency=0.1)
        result = _result("T", candidates=[candidate], rule_responses=[(contract, response)])

        manifest = export_rules([result], tmp_path, promotable_only=False)

        assert len(manifest) == 1
        assert bool(manifest.iloc[0]["promotable"]) is False

    def test_creates_output_dir(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        candidate = _candidate("EVT-1")
        contract = _contract("A-1", event_candidate_id="EVT-1")
        result = _result("T", candidates=[candidate], rule_responses=[(contract, _response("EDGE"))])

        export_rules([result], target)

        assert target.is_dir()

    def test_no_matches_returns_empty_frame_with_expected_columns(self, tmp_path):
        manifest = export_rules([], tmp_path)

        assert manifest.empty
        assert list(manifest.columns) == [
            "ticker",
            "alpha_id",
            "event_candidate_id",
            "verdict",
            "promotable",
            "pkl_path",
            "yaml_path",
        ]


class TestMonitoringManifest:
    def test_reflects_rule_spec_from_forge_result(self, monkeypatch, tmp_path):
        candidate = _candidate("EVT-1")
        contract = _contract("A-1", event_candidate_id="EVT-1")
        response = _response("EDGE")
        result = _result("T", candidates=[candidate], rule_responses=[(contract, response)])

        fake_spec = SimpleNamespace(
            name="A-1", candidate=candidate, is_end=None, verdict="EDGE", oos_expectancy=0.01
        )
        monkeypatch.setattr(
            "forgedge.deployment.rules.RuleSpec.from_forge_result",
            lambda r: [fake_spec],
        )

        df = monitoring_manifest([result])

        assert len(df) == 1
        assert df.iloc[0]["event_candidate_id"] == "EVT-1"
        assert df.iloc[0]["oos_expectancy"] == pytest.approx(0.01)

    def test_no_matches_returns_empty_frame_with_expected_columns(self, monkeypatch):
        monkeypatch.setattr(
            "forgedge.deployment.rules.RuleSpec.from_forge_result",
            lambda r: [],
        )

        df = monitoring_manifest([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "rule_name",
            "event_candidate_id",
            "is_end",
            "verdict",
            "oos_expectancy",
        ]
