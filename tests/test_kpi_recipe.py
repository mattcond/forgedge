"""Tests for forgedge.deployment.kpi_recipe — minimal KPI recipe per rule (#296).

Every test builds a real KPI Table with ``kpi_builder`` from seeded synthetic
candles and replays hand-built ``EventComponent``s on it, so each recipe
feature (arity-1/2/3 features, lags, positional MACD periods, candle
geometry, unresolved columns) is pinned on exactly the column it is about,
instead of depending on which candidates Event Discovery happens to mine.
"""

import json

import numpy as np
import pandas as pd
import pytest

import forgedge.deployment.kpi_recipe as kpi_recipe_mod
from forgedge.deployment import (
    KpiRecipe,
    KpiRecipeVerification,
    minimal_kpi_recipe,
    verify_kpi_recipe,
)
from forgedge.event_discovery.models import (
    ActivationStats,
    CustomEvent,
    EventCandidate,
    EventComponent,
    GateResult,
)
from forgedge.kpi_builder import (
    DEFAULT_CONFIG,
    build_features,
    candle_features,
    lag_features,
    pattern_features,
)


def _candles(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Geometric random walk with a realistic OHLC envelope (hourly, ms epochs)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    span = np.abs(rng.normal(0, 0.004, n)) * close
    return pd.DataFrame(
        {
            "open_time": 1_600_000_000_000 + np.arange(n) * 3_600_000,
            "open": openp,
            "high": np.maximum(openp, close) + span,
            "low": np.minimum(openp, close) - span,
            "close": close,
            "volume": rng.uniform(10, 100, n),
        }
    )


def _config_with(**extra):
    cfg = {name: dict(conf) for name, conf in DEFAULT_CONFIG.items()}
    for name, params in extra.items():
        cfg[name] = {"enabled": True, "params": params}
    return cfg


def _kpi(candles=None, config=None) -> pd.DataFrame:
    """Full KPI Table: build_features + candle_features + patterns + lags."""
    candles = _candles() if candles is None else candles
    kpi = build_features(candles, config=config, timestamp_col="open_time")
    kpi = candle_features(kpi)
    kpi = pattern_features(kpi)
    kpi = lag_features(kpi, "close_ema_12", "body", periods=[1, 3])
    kpi = lag_features(kpi, "close_ema_12_prev_01", periods=[2])
    return kpi


def _component(source_feature, *, source_cols=(), transform="rolling_pctrank",
               transform_params=None, threshold=0.8, direction="above"):
    params = {"window": 24} if transform_params is None else transform_params
    return EventComponent(
        source_feature=source_feature,
        transform=transform,
        transform_params=params,
        transformed_col=f"t_{source_feature}",
        threshold=threshold,
        threshold_type="test",
        direction=direction,
        event_type="threshold",
        expression=f"{source_feature} {direction} {threshold}",
        source_cols=list(source_cols),
    )


def _candidate(*components) -> EventCandidate:
    return EventCandidate(
        event_id="EVT-TEST",
        status="CANDIDATE",
        components=list(components),
        expression=" AND ".join(c.expression for c in components),
        activation_stats=ActivationStats(
            n_activations=0, n_active_months=0, zero_months=0,
            max_monthly_share=float("nan"), mean_tpm=float("nan"),
        ),
        consistency_gate=GateResult(
            passed=True, n_activations=0, n_active_months=0,
            max_monthly_share=float("nan"), mean_tpm=float("nan"),
        ),
    )


# ---------------------------------------------------------------------------
# Recipe extraction
# ---------------------------------------------------------------------------

class TestMinimalKpiRecipe:
    def test_arity1_feature_resolves_to_single_indicator_entry(self):
        recipe = minimal_kpi_recipe(_candidate(_component("close_ema_12")))

        assert recipe.build_features_config == {
            "ema": {"enabled": True, "params": {"periods": [12], "columns": ["close"]}}
        }
        assert recipe.lag_features == []
        assert recipe.candle_features is False
        assert recipe.pattern_features is False
        assert recipe.color is False
        assert recipe.unresolved_columns == []
        assert recipe.is_complete

    def test_arity2_feature_reads_source_cols_not_source_feature(self):
        comp = _component(
            "ratio_close_ema03_sma25", source_cols=["close_ema_03", "low_sma_25"]
        )
        recipe = minimal_kpi_recipe(_candidate(comp))

        assert recipe.build_features_config["ema"]["params"] == {"periods": [3], "columns": ["close"]}
        assert recipe.build_features_config["moving_average"]["params"] == {
            "periods": [25], "columns": ["low"],
        }
        assert recipe.unresolved_columns == []

    def test_arity3_feature_and_base_columns_are_never_rebuilt(self):
        comp = _component(
            "bb_pct_b_close_20",
            source_cols=["close", "close_bb_lower_20", "close_bb_upper_20"],
        )
        recipe = minimal_kpi_recipe(_candidate(comp))

        # `close` is a base column; both band columns come from one entry.
        assert recipe.build_features_config == {
            "bollinger_bands": {"enabled": True, "params": {"periods": [20], "columns": ["close"]}}
        }

    def test_and_composition_unions_component_requirements(self):
        cand = _candidate(
            _component("close_rsi_14"),
            _component("close_vol_24"),
            _component("close_rsi_25"),
        )
        recipe = minimal_kpi_recipe(cand)

        assert recipe.build_features_config["rsi"]["params"] == {"periods": [14, 25], "columns": ["close"]}
        assert recipe.build_features_config["volatility"]["params"] == {"periods": [24], "columns": ["close"]}

    def test_lag_and_lag_of_lag_resolve_their_parent_first(self):
        cand = _candidate(
            _component("close_ema_12_prev_01_prev_02"),
            _component("close_ema_12_prev_03"),
        )
        recipe = minimal_kpi_recipe(cand)

        assert recipe.build_features_config == {
            "ema": {"enabled": True, "params": {"periods": [12], "columns": ["close"]}}
        }
        # The lag-of-lag reads close_ema_12_prev_01, so that lag comes first.
        assert recipe.lag_features == [
            {"column": "close_ema_12", "periods": [1, 3]},
            {"column": "close_ema_12_prev_01", "periods": [2]},
        ]

    def test_lag_of_base_column(self):
        recipe = minimal_kpi_recipe(_candidate(_component("close_prev_02")))

        assert recipe.build_features_config == {}
        assert recipe.lag_features == [{"column": "close", "periods": [2]}]

    def test_lag_spelling_lag_features_never_writes_is_unresolved(self):
        # lag_features always zero-pads: `close_prev_2` is not its output.
        recipe = minimal_kpi_recipe(_candidate(_component("close_prev_2")))

        assert recipe.unresolved_columns == ["close_prev_2"]

    def test_macd_positional_triple_is_kept_intact_and_in_order(self):
        # Sorting {12, 26, 9} into [9, 12, 26] builds a different MACD whose
        # columns the event can't find — the bug reproduced in the #295
        # prototype. The triple must survive verbatim.
        cfg = _config_with(macd={"periods": [12, 26, 9, 5, 35, 5], "columns": ["close"]})
        cand = _candidate(
            _component("close_macd_12_26_signal_09"),
            _component("close_macd_05_35_hist_05"),
        )
        recipe = minimal_kpi_recipe(cand, kpi_config=cfg)

        assert recipe.build_features_config["macd"]["params"]["periods"] == [12, 26, 9, 5, 35, 5]

    def test_macd_only_needed_triple_is_kept(self):
        cfg = _config_with(macd={"periods": [12, 26, 9, 5, 35, 5], "columns": ["close"]})
        recipe = minimal_kpi_recipe(_candidate(_component("close_macd_05_35")), kpi_config=cfg)

        assert recipe.build_features_config["macd"]["params"]["periods"] == [5, 35, 5]

    def test_opt_in_indicator_disabled_in_config_still_resolves(self):
        # ATR is `enabled: False` in DEFAULT_CONFIG; a KPI Table built with it
        # turned on must still get a recipe for its columns.
        assert DEFAULT_CONFIG["atr"]["enabled"] is False
        recipe = minimal_kpi_recipe(_candidate(_component("close_natr_28")))

        assert recipe.build_features_config == {
            "atr": {"enabled": True, "params": {"periods": [28], "columns": ["close"]}}
        }

    def test_indicator_without_column_prefix(self):
        # Aroon writes `aroon_up_{w}` — no `{on}_` prefix. Resolved by
        # construction, not by naming-convention parsing.
        recipe = minimal_kpi_recipe(_candidate(_component("aroon_up_25")))

        assert recipe.build_features_config == {
            "aroon": {"enabled": True, "params": {"periods": [25], "columns": ["close"]}}
        }

    def test_candle_pattern_and_color_flags(self):
        cand = _candidate(
            _component("body"),
            _component("candle_pattern", transform="categorical_onehot",
                       transform_params={"class": "DOJI"}),
            _component("color", transform="binary_native", transform_params={}, threshold=1.0),
        )
        recipe = minimal_kpi_recipe(cand)

        assert recipe.candle_features is True
        assert recipe.pattern_features is True
        assert recipe.color is True
        assert recipe.build_features_config == {}

    def test_custom_event_and_unknown_columns_are_unresolved(self):
        custom = CustomEvent("my_signal > 0", name="mine").to_event_candidate(
            pd.DataFrame({"my_signal": [1.0, -1.0]})
        )
        cand = _candidate(custom.components[0], _component("regime_score"), _component("close_ema_12"))
        recipe = minimal_kpi_recipe(cand)

        assert recipe.unresolved_columns == ["my_signal > 0", "regime_score"]
        assert not recipe.is_complete
        # The resolvable part is still reported.
        assert "ema" in recipe.build_features_config

    def test_column_index_is_built_once_per_config(self, monkeypatch):
        kpi_recipe_mod._column_index_cached.cache_clear()
        calls = []
        real = kpi_recipe_mod.build_features

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(kpi_recipe_mod, "build_features", counting)

        minimal_kpi_recipe(_candidate(_component("close_ema_12")))
        first = len(calls)
        for feature in ("close_rsi_14", "close_vol_24", "low_min_48", "close_ema_12_prev_01"):
            minimal_kpi_recipe(_candidate(_component(feature)))

        assert first > 0
        assert len(calls) == first  # every later candidate is lookups only
        kpi_recipe_mod._column_index_cached.cache_clear()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestKpiRecipeSerialisation:
    def test_json_round_trip(self):
        cfg = _config_with(macd={"periods": [12, 26, 9], "columns": ["close"]})
        cand = _candidate(
            _component("close_macd_12_26_signal_09"),
            _component("close_ema_12_prev_01_prev_02"),
            _component("body"),
        )
        recipe = minimal_kpi_recipe(cand, kpi_config=cfg)

        text = recipe.to_json()
        restored = KpiRecipe.from_json(text)

        assert restored == recipe
        assert json.loads(text) == recipe.to_dict()

    def test_from_dict_ignores_unknown_keys(self):
        payload = minimal_kpi_recipe(_candidate(_component("close_ema_12"))).to_dict()
        payload["verification"] = {"matches": True}

        assert KpiRecipe.from_dict(payload).build_features_config == payload["build_features_config"]


# ---------------------------------------------------------------------------
# Rebuild + round-trip verification
# ---------------------------------------------------------------------------

class TestRebuildAndVerify:
    @pytest.mark.parametrize(
        "components",
        [
            [_component("close_ema_12")],
            [_component("ratio_close_ema03_sma25", source_cols=["close_ema_03", "low_sma_25"])],
            [_component("bb_pct_b_close_20", source_cols=["close", "close_bb_lower_20", "close_bb_upper_20"])],
            [_component("close_ema_12_prev_01_prev_02"), _component("body_prev_03", threshold=0.5)],
            [_component("close_macd_12_26_hist_09", transform="rolling_zscore", threshold=0.5)],
            [_component("close_natr_14", transform="delta", transform_params={"lag": 1}, threshold=0.0)],
            [_component("candle_pattern", transform="categorical_onehot", transform_params={"class": "DOJI"})],
        ],
        ids=["arity1", "arity2", "arity3", "lags", "macd", "atr", "pattern"],
    )
    def test_round_trip_is_exact_when_reference_built_from_same_candles(self, components):
        cfg = _config_with(
            macd={"periods": [12, 26, 9], "columns": ["close"]},
            atr={"periods": [14, 28], "columns": ["close"]},
        )
        reference = _kpi(config=cfg)
        cand = _candidate(*components)
        recipe = minimal_kpi_recipe(cand, kpi_config=cfg)

        check = verify_kpi_recipe(cand, recipe, reference)

        assert check.error is None
        assert check.matches, check
        assert check.n_compared_bars == len(reference)
        assert check.n_mismatched_bars == 0
        assert cand.apply(reference).fillna(0).sum() > 0  # not vacuously equal

    def test_rebuild_computes_only_what_the_recipe_needs(self):
        reference = _kpi()
        recipe = minimal_kpi_recipe(_candidate(_component("close_ema_12_prev_03")))

        reduced = recipe.rebuild(reference)

        assert set(reduced.columns) == {
            "open_dt", "open", "high", "low", "close", "volume",
            "close_ema_12", "close_ema_12_prev_03",
        }

    def test_rebuild_accepts_datetime_index(self):
        # EventDiscovery.df / ForgeResult.event_frame carry the timestamp as
        # the index, not as a column.
        reference = _kpi().set_index("open_dt")
        cand = _candidate(_component("close_rsi_14"))
        recipe = minimal_kpi_recipe(cand)

        check = verify_kpi_recipe(cand, recipe, reference)

        assert check.matches, check

    def test_rebuild_refuses_incomplete_recipe(self):
        recipe = minimal_kpi_recipe(_candidate(_component("regime_score")))

        with pytest.raises(ValueError, match="unresolved columns"):
            recipe.rebuild(_kpi())

    def test_incomplete_recipe_is_reported_not_raised(self):
        cand = _candidate(_component("regime_score"))
        check = verify_kpi_recipe(cand, minimal_kpi_recipe(cand), _kpi())

        assert isinstance(check, KpiRecipeVerification)
        assert check.matches is False
        assert "unresolved columns" in check.error

    def test_wrong_recipe_is_caught(self):
        reference = _kpi()
        cand = _candidate(_component("close_ema_12"))
        wrong = KpiRecipe(
            build_features_config={"ema": {"enabled": True, "params": {"periods": [25], "columns": ["close"]}}}
        )

        check = verify_kpi_recipe(cand, wrong, reference)

        assert check.matches is False
        assert check.error.startswith("KeyError")

    def test_value_mismatch_is_located(self):
        reference = _kpi()
        cand = _candidate(_component("close_rsi_14", transform="identity", threshold=50.0))
        recipe = minimal_kpi_recipe(cand)
        tampered = reference.copy()
        rows = tampered.index[200:205]
        tampered.loc[rows, "close_rsi_14"] = np.where(
            tampered.loc[rows, "close_rsi_14"] > 50.0, 0.0, 100.0
        )

        check = verify_kpi_recipe(cand, recipe, tampered)

        assert check.matches is False
        assert check.n_mismatched_bars == 5
        assert check.first_mismatch_at == tampered.loc[rows[0], "open_dt"]
        assert check.last_mismatch_at == tampered.loc[rows[-1], "open_dt"]

    def test_warm_up_mismatch_is_confined_to_start_and_masked_out(self):
        # The reference KPI Table is built on a longer history then truncated
        # (like tests/fixtures/ADA_1D_TRAIN.parquet): the rebuilt recursive EMA
        # starts from a different state and only converges after a transient.
        long_kpi = build_features(_candles(n=800, seed=3), timestamp_col="open_time")
        reference = long_kpi.iloc[400:]
        rebuilt_ema = build_features(reference[["open_time", "open", "high", "low", "close"]],
                                     config={"ema": DEFAULT_CONFIG["ema"]},
                                     timestamp_col="open_time")["close_ema_25"]
        first_ref, first_new = reference["close_ema_25"].iloc[0], rebuilt_ema.iloc[0]
        assert first_ref != first_new
        # Threshold between the two first values guarantees a bar-0 mismatch.
        threshold = (first_ref + first_new) / 2
        direction = "above" if first_ref > first_new else "below"
        cand = _candidate(_component("close_ema_25", transform="identity",
                                     threshold=threshold, direction=direction))
        recipe = minimal_kpi_recipe(cand)

        whole = verify_kpi_recipe(cand, recipe, reference)
        mask = pd.Series(np.arange(len(reference)) >= 200, index=reference.index)
        masked = verify_kpi_recipe(cand, recipe, reference, evaluation_mask=mask)

        assert whole.matches is False
        assert whole.first_mismatch_at == reference["open_dt"].iloc[0]
        assert whole.last_mismatch_at < reference["open_dt"].iloc[200]
        assert masked.matches, masked
        assert masked.n_compared_bars == len(reference) - 200

    def test_verification_to_dict_is_json_serialisable(self):
        check = KpiRecipeVerification(
            matches=False, n_compared_bars=10, n_mismatched_bars=1,
            first_mismatch_at=pd.Timestamp("2024-01-01"),
            last_mismatch_at=pd.Timestamp("2024-01-01"),
        )

        payload = json.loads(json.dumps(check.to_dict()))

        assert payload["first_mismatch_at"] == "2024-01-01T00:00:00"
        assert payload["n_mismatched_bars"] == 1
