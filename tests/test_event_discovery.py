"""Tests for the Event Discovery module."""
import numpy as np
import pandas as pd
import pytest

from forgedge.event_discovery import (
    DiscoveryConfig,
    EventDiscovery,
    GateParams,
)
from forgedge.event_discovery.classifier import TypeClassifier
from forgedge.event_discovery.consistency_gate import (
    ConsistencyGate,
    _build_month_index,
    _count_by_month,
)
from forgedge.event_discovery.event_generator import EventGenerator
from forgedge.event_discovery.feature_generator import FeatureGenerator, parse_feature
from forgedge.event_discovery.models import ColumnType, GateParams, RawEvent
from forgedge.event_discovery.transform_layer import TransformLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_kpi_table(n: int = 4380, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic KPI table.  Default n=4380 ≈ 6 months of 1H data."""
    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
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


# ---------------------------------------------------------------------------
# Step 0 — TypeClassifier
# ---------------------------------------------------------------------------


class TestTypeClassifier:
    def test_rsi_is_continuous_and_scale_free(self):
        # Scale-free detection needs enough data (≥6 months) to be reliable.
        df = _make_kpi_table(n=8760)
        cls = TypeClassifier().fit(df)
        assert cls["close_rsi_14"].col_type == ColumnType.CONTINUOUS
        assert cls["close_rsi_14"].effective_scale_free is True

    def test_raw_price_is_continuous_not_scale_free(self):
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        assert cls["close"].col_type == ColumnType.CONTINUOUS
        assert cls["close"].effective_scale_free is False

    def test_binary_column(self):
        df = pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=100, freq="1h"), "flag": [0, 1] * 50}
        )
        cls = TypeClassifier().fit(df)
        assert cls["flag"].col_type == ColumnType.BINARY

    def test_categorical_string_column(self):
        df = pd.DataFrame(
            {
                "open_dt": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "shape": ["doji", "hammer", "spinning"] * 33 + ["doji"],
            }
        )
        cls = TypeClassifier().fit(df)
        assert cls["shape"].col_type == ColumnType.CATEGORICAL

    def test_high_cardinality_categorical_discarded(self):
        df = pd.DataFrame(
            {
                "open_dt": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "cat": [f"class_{i}" for i in range(100)],
            }
        )
        cls = TypeClassifier(max_categorical_classes=20).fit(df)
        # High-cardinality categorical should be classified but excluded from pipeline
        assert cls["cat"].col_type == ColumnType.CATEGORICAL
        assert cls["cat"].n_distinct > 20

    def test_scale_free_override_respected(self):
        df = _make_kpi_table()
        cls = TypeClassifier(scale_free_overrides={"close": True}).fit(df)
        assert cls["close"].effective_scale_free is True
        assert cls["close"].scale_free_overridden is True


# ---------------------------------------------------------------------------
# Step 1 — FeatureGenerator
# ---------------------------------------------------------------------------


class TestFeatureGenerator:
    def test_arity1_scale_free_passthrough(self):
        # RSI needs enough history to be detected as scale-free
        df = _make_kpi_table(n=8760)
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        # RSI should pass through as arity-1 identity
        assert "close_rsi_14" in meta
        assert meta["close_rsi_14"].arity == 1
        assert meta["close_rsi_14"].operation == "identity"

    def test_arity2_ema_ratio_generated(self):
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        ext_df, meta = FeatureGenerator().generate(df, cls)
        arity2_cols = [k for k, v in meta.items() if v.arity == 2]
        assert any("ratio" in c for c in arity2_cols)

    def test_arity3_bb_position_generated(self):
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        ext_df, meta = FeatureGenerator().generate(df, cls)
        arity3_cols = [k for k, v in meta.items() if v.arity == 3]
        assert any("bb_pct_b" in c for c in arity3_cols), f"No bb_pct_b in {arity3_cols}"

    def test_parse_feature_ema(self):
        pf = parse_feature("close_ema_09")
        assert pf is not None
        assert pf.base == "close"
        assert pf.indicator == "ema"
        assert pf.params == [9]
        assert pf.family == "ema"

    def test_parse_feature_bb_lower(self):
        pf = parse_feature("close_bb_lower_20")
        assert pf is not None
        assert pf.family == "bollinger"

    def test_derived_features_are_scale_free(self):
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        derived = {k: v for k, v in meta.items() if v.arity >= 2}
        assert all(v.is_scale_free for v in derived.values())


# ---------------------------------------------------------------------------
# Step 2 — TransformLayer
# ---------------------------------------------------------------------------


class TestTransformLayer:
    def test_pctrank_bounded(self):
        series = pd.Series(np.random.randn(500))
        t = TransformLayer()
        results = t.transform_one(series, "test_feat", is_scale_free=True)
        pr = next(r for r in results if r.transform == "rolling_pctrank")
        valid = pr.series.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_zscore_mean_near_zero(self):
        series = pd.Series(np.random.randn(500))
        t = TransformLayer()
        results = t.transform_one(series, "test_feat", is_scale_free=False)
        zs = next(r for r in results if r.transform == "rolling_zscore")
        assert abs(float(zs.series.dropna().mean())) < 0.3

    def test_identity_only_for_scale_free(self):
        series = pd.Series(np.arange(500, dtype=float))
        t = TransformLayer()
        results_sf = t.transform_one(series, "feat", is_scale_free=True)
        results_nf = t.transform_one(series, "feat", is_scale_free=False)
        assert any(r.transform == "identity" for r in results_sf)
        assert not any(r.transform == "identity" for r in results_nf)

    def test_delta_length_preserved(self):
        series = pd.Series(np.random.randn(200))
        t = TransformLayer()
        results = t.transform_one(series, "feat", is_scale_free=False)
        deltas = [r for r in results if r.transform == "delta"]
        assert len(deltas) == 4  # lags 1, 3, 6, 12
        for d in deltas:
            assert len(d.series) == len(series)


# ---------------------------------------------------------------------------
# Step 4 — ConsistencyGate
# ---------------------------------------------------------------------------


class TestConsistencyGate:
    def _make_gate(self):
        return ConsistencyGate(GateParams(min_act=30, min_months=4, max_conc=0.40, min_tpm=1.0))

    def _timestamps(self, n: int = 720) -> pd.Series:
        return pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))

    def test_sparse_event_fails_volume(self):
        ts = self._timestamps()
        month_idx, n_m = _build_month_index(ts)
        event = pd.Series([1.0] * 10 + [0.0] * 710)
        active = event.fillna(0).values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = self._make_gate().evaluate(active, counts, n_m)
        assert not result.passed
        assert "volume" in result.fail_reason

    def test_concentrated_event_fails_concentration(self):
        ts = self._timestamps()
        month_idx, n_m = _build_month_index(ts)
        # All activations in first month
        event = pd.Series([1.0] * 50 + [0.0] * 670)
        active = event.fillna(0).values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = ConsistencyGate(GateParams(min_act=30, min_months=1, max_conc=0.40, min_tpm=1.0)).evaluate(
            active, counts, n_m
        )
        assert not result.passed
        assert "concentration" in result.fail_reason

    def test_uniform_event_passes(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        rng = np.random.default_rng(0)
        event = pd.Series((rng.random(8760) < 0.10).astype(float))
        active = event.values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = ConsistencyGate(GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)).evaluate(
            active, counts, n_m
        )
        assert result.passed


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEventDiscoveryE2E:
    def test_run_returns_candidates(self):
        df = _make_kpi_table(n=2000)
        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                gate_params=GateParams(min_act=20, min_months=2, max_conc=0.60, min_tpm=1.0),
                max_and_components=2,
            ),
        )
        candidates = ed.run()
        assert len(candidates) > 0

    def test_all_candidates_have_passed_gate(self):
        df = _make_kpi_table(n=2000)
        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                gate_params=GateParams(min_act=20, min_months=2, max_conc=0.60, min_tpm=1.0)
            ),
        )
        candidates = ed.run()
        for c in candidates:
            assert c.consistency_gate.passed, f"{c.event_id} gate not passed"

    def test_candidates_have_expressions(self):
        df = _make_kpi_table(n=2000)
        ed = EventDiscovery(df)
        candidates = ed.run()
        for c in candidates:
            assert c.expression
            assert len(c.components) >= 1

    def test_summary_dataframe_shape(self):
        df = _make_kpi_table()
        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                gate_params=GateParams(min_act=20, min_months=4, max_conc=0.60, min_tpm=1.0)
            ),
        )
        candidates = ed.run()
        summary = ed.summary()
        assert len(summary) == len(candidates)
        assert "event_id" in summary.columns
        assert "expression" in summary.columns
        assert "n_activations" in summary.columns

    def test_summary_empty_dataframe_has_columns(self):
        """summary() on empty candidate set returns a DataFrame with correct columns."""
        df = _make_kpi_table(n=200)
        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                # Impossible gate: nothing will pass
                gate_params=GateParams(min_act=99999, min_months=8, max_conc=0.01, min_tpm=100.0)
            ),
        )
        ed.run()
        summary = ed.summary()
        assert len(summary) == 0
        assert "event_id" in summary.columns

    def test_classifications_available_after_run(self):
        df = _make_kpi_table()
        ed = EventDiscovery(df)
        ed.run()
        cls = ed.get_classifications()
        assert cls is not None
        assert "close_rsi_14" in cls

    def test_datetimeindex_input(self):
        df = _make_kpi_table()
        df = df.set_index("open_dt")
        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                timestamp_col="open_dt",
                gate_params=GateParams(min_act=20, min_months=4, max_conc=0.60, min_tpm=1.0),
            ),
        )
        candidates = ed.run()
        assert len(candidates) > 0
