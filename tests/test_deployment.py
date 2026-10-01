"""Tests for forgedge.deployment — promotion gate, export, monitoring manifest."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgedge.deployment import (
    KpiRecipe,
    PromotionGateConfig,
    export_rules,
    monitoring_manifest,
    promotion_gate,
)
from forgedge.event_discovery.models import (
    ActivationStats,
    EventCandidate,
    EventComponent,
    GateResult,
)
from forgedge.kpi_builder import build_features


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
            "kpi_recipe_path",
            "kpi_recipe_verified",
        ]

    def test_stub_candidate_without_components_gets_no_recipe(self, tmp_path):
        candidate = _candidate("EVT-1")
        contract = _contract("A-1", event_candidate_id="EVT-1")
        result = _result("T", candidates=[candidate], rule_responses=[(contract, _response("EDGE"))])

        manifest = export_rules([result], tmp_path)

        assert manifest.iloc[0]["kpi_recipe_path"] is None
        assert manifest.iloc[0]["kpi_recipe_verified"] is None
        assert not (tmp_path / "A-1.kpi_recipe.json").exists()


def _kpi_frame(n=300, seed=0):
    """Real KPI Table indexed by timestamp, like ``ForgeResult.event_frame``."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    candles = pd.DataFrame(
        {
            "open_time": 1_600_000_000_000 + np.arange(n) * 3_600_000,
            "open": openp,
            "high": np.maximum(openp, close) * 1.002,
            "low": np.minimum(openp, close) * 0.998,
            "close": close,
        }
    )
    kpi = build_features(candles, timestamp_col="open_time")
    return kpi.drop(columns="open_time").set_index("open_dt")


def _event_candidate(event_id="EVT-1", source_feature="close_rsi_14"):
    comp = EventComponent(
        source_feature=source_feature,
        transform="rolling_pctrank",
        transform_params={"window": 24},
        transformed_col=f"pr_{source_feature}_24",
        threshold=0.8,
        threshold_type="test",
        direction="above",
        event_type="threshold",
        expression=f"pr_{source_feature}_24 > 0.8",
    )
    return EventCandidate(
        event_id=event_id,
        status="CANDIDATE",
        components=[comp],
        expression=comp.expression,
        activation_stats=ActivationStats(
            n_activations=0, n_active_months=0, zero_months=0,
            max_monthly_share=float("nan"), mean_tpm=float("nan"),
        ),
        consistency_gate=GateResult(
            passed=True, n_activations=0, n_active_months=0,
            max_monthly_share=float("nan"), mean_tpm=float("nan"),
        ),
    )


class TestExportKpiRecipe:
    def _result(self, event_frame=None, source_feature="close_rsi_14"):
        candidate = _event_candidate(source_feature=source_feature)
        contracts = [
            (_contract("A-1", event_candidate_id="EVT-1"), _response("EDGE")),
            (_contract("A-2", event_candidate_id="EVT-1"), _response("EDGE")),
        ]
        result = _result("T", candidates=[candidate], rule_responses=contracts)
        result.event_frame = event_frame
        return result

    def test_writes_verified_recipe_next_to_pickle(self, tmp_path):
        result = self._result(event_frame=_kpi_frame())

        manifest = export_rules([result], tmp_path)

        assert list(manifest["kpi_recipe_verified"]) == [True, True]
        recipe_path = tmp_path / "A-1.kpi_recipe.json"
        assert manifest.iloc[0]["kpi_recipe_path"] == str(recipe_path)
        assert (tmp_path / "A-1.pkl").exists()  # the pickle fallback is always kept
        payload = json.loads(recipe_path.read_text())
        assert payload["verification"]["matches"] is True
        assert payload["verification"]["n_compared_bars"] == 300
        recipe = KpiRecipe.from_dict(payload)
        assert recipe.build_features_config == {
            "rsi": {"enabled": True, "params": {"periods": [14], "columns": ["close"]}}
        }

    def test_without_event_frame_recipe_is_written_but_unverified(self, tmp_path):
        result = self._result(event_frame=None)

        manifest = export_rules([result], tmp_path)

        assert pd.isna(manifest.iloc[0]["kpi_recipe_verified"])
        payload = json.loads((tmp_path / "A-1.kpi_recipe.json").read_text())
        assert payload["verification"] is None

    def test_unresolved_recipe_is_exported_as_unverified(self, tmp_path):
        frame = _kpi_frame().copy()
        frame["proprietary_signal"] = frame["close_rsi_14"]
        result = self._result(event_frame=frame, source_feature="proprietary_signal")

        manifest = export_rules([result], tmp_path)

        assert bool(manifest.iloc[0]["kpi_recipe_verified"]) is False
        payload = json.loads((tmp_path / "A-1.kpi_recipe.json").read_text())
        assert payload["unresolved_columns"] == ["proprietary_signal"]
        assert "unresolved columns" in payload["verification"]["error"]

    def test_warmup_bars_restrict_the_verification_window(self, tmp_path):
        result = self._result(event_frame=_kpi_frame())

        export_rules([result], tmp_path, kpi_recipe_warmup_bars=50)

        payload = json.loads((tmp_path / "A-1.kpi_recipe.json").read_text())
        assert payload["verification"]["n_compared_bars"] == 250

    def test_include_kpi_recipe_false_skips_it(self, tmp_path):
        result = self._result(event_frame=_kpi_frame())

        manifest = export_rules([result], tmp_path, include_kpi_recipe=False)

        assert not list(tmp_path.glob("*.kpi_recipe.json"))
        assert manifest["kpi_recipe_path"].isna().all()


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
            "kpi_recipe_path",
            "kpi_recipe_verified",
        ]

    def test_joins_kpi_recipe_from_export_manifest(self, monkeypatch):
        specs = [
            SimpleNamespace(name="A-1", candidate=_candidate("EVT-1"), is_end=None,
                            verdict="EDGE", oos_expectancy=0.01),
            SimpleNamespace(name="A-9", candidate=_candidate("EVT-9"), is_end=None,
                            verdict="EDGE", oos_expectancy=0.02),
        ]
        monkeypatch.setattr(
            "forgedge.deployment.rules.RuleSpec.from_forge_result",
            lambda r: specs,
        )
        exported = pd.DataFrame(
            [{"ticker": "T", "event_candidate_id": "EVT-1",
              "kpi_recipe_path": "/x/A-1.kpi_recipe.json", "kpi_recipe_verified": True}]
        )

        df = monitoring_manifest([_result("T")], exported=exported)

        assert df.iloc[0]["kpi_recipe_path"] == "/x/A-1.kpi_recipe.json"
        assert bool(df.iloc[0]["kpi_recipe_verified"]) is True
        assert pd.isna(df.iloc[1]["kpi_recipe_path"])  # not exported: kept, not filtered
