"""Tests for the Event Discovery module."""
import numpy as np
import pandas as pd
import pytest

from forgedge.event_discovery import (
    DiscoveryConfig,
    EventDiscovery,
    FoldResult,
    GateParams,
    ValidationResult,
    WalkForwardConfig,
)
from forgedge.event_discovery.and_composer import ANDComposer
from forgedge.event_discovery.classifier import TypeClassifier
from forgedge.event_discovery.consistency_gate import (
    ConsistencyGate,
    _build_month_index,
    _count_by_month,
    _monthly_counts,
)
from forgedge.event_discovery.discovery import _count_zero_months
from forgedge.event_discovery.event_generator import EventGenerator
from forgedge.event_discovery.feature_generator import FeatureGenerator, parse_feature
from forgedge.event_discovery.models import (
    ColumnType,
    EventCandidate,
    EventComponent,
    GateParams,
    RawEvent,
    _apply_component,
)
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

    def test_bb_pct_b_uses_matching_base_not_close(self):
        """bb_pct_b_{base} must use the {base} column as numerator, not close.

        Regression for issue #51: previously close_col was selected once outside
        the loop, so bb_pct_b_high_20 used close as numerator instead of high.
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + np.abs(rng.normal(0, 0.5, n))
        sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
        std20 = pd.Series(close).rolling(20, min_periods=1).std().fillna(0).values

        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": close,
            "high": high,
            "close_bb_lower_20": sma20 - 2 * std20,
            "close_bb_upper_20": sma20 + 2 * std20,
            "high_bb_lower_20": sma20 - 2 * std20 + 0.5,
            "high_bb_upper_20": sma20 + 2 * std20 + 0.5,
        })

        cls = TypeClassifier().fit(df)
        ext_df, meta = FeatureGenerator().generate(df, cls)

        assert "bb_pct_b_close_20" in ext_df.columns
        assert "bb_pct_b_high_20" in ext_df.columns

        # source_cols must record the correct base column (primary assertion)
        assert meta["bb_pct_b_high_20"].source_cols[0] == "high"
        assert meta["bb_pct_b_close_20"].source_cols[0] == "close"

        # Verify that bb_pct_b_high_20 uses high as numerator and not close.
        # Pick a bar where high != close (all bars after index 0) and check
        # that the computed value matches (high - lower) / (upper - lower)
        # rather than (close - lower) / (upper - lower).
        i = 10  # arbitrary mid-series bar with stable rolling window
        lower = df["high_bb_lower_20"].iloc[i]
        upper = df["high_bb_upper_20"].iloc[i]
        width = upper - lower
        val_high = ext_df["bb_pct_b_high_20"].iloc[i]
        assert abs(val_high - (df["high"].iloc[i] - lower) / width) < 1e-10, (
            "bb_pct_b_high_20 should use 'high' as numerator"
        )
        assert abs(val_high - (df["close"].iloc[i] - lower) / width) > 1e-6, (
            "bb_pct_b_high_20 must NOT use 'close' as numerator"
        )

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
        return ConsistencyGate(GateParams(min_tpm=1.0, max_dispersion=2.5))

    def _timestamps(self, n: int = 720) -> pd.Series:
        return pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))

    def test_sparse_event_fails_volume(self):
        # Use a 12-month series with only 2 activations → mean_tpm = 2/12 < min_tpm=1.0
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        event = pd.Series([1.0] * 2 + [0.0] * 8758)
        active = event.fillna(0).values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = self._make_gate().evaluate(active, counts, n_m)
        assert not result.passed
        assert "rate" in result.fail_reason

    def test_concentrated_event_fails_concentration(self):
        # 12-month series, 50 activations all in first month → high dispersion
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        # All 50 activations in first month (first 744 hours), none after
        active = np.array([True] * 50 + [False] * (8760 - 50), dtype=bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = ConsistencyGate(GateParams(min_tpm=1.0, max_dispersion=2.5)).evaluate(
            active, counts, n_m
        )
        assert not result.passed
        assert "dispersion" in result.fail_reason

    def test_uniform_event_passes(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        rng = np.random.default_rng(0)
        event = pd.Series((rng.random(8760) < 0.10).astype(float))
        active = event.values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = ConsistencyGate(GateParams(min_tpm=2.0, max_dispersion=2.5)).evaluate(
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
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0),
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
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0)
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
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0)
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
                gate_params=GateParams(min_tpm=100.0, max_dispersion=0.001)
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
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0),
            ),
        )
        candidates = ed.run()
        assert len(candidates) > 0


    def test_unsorted_input_produces_same_results_as_sorted(self):
        """Rolling windows must be computed in chronological order regardless of input row order."""
        df_sorted = _make_kpi_table(n=2000)
        # Shuffle the input rows (preserves all data, breaks row order)
        df_shuffled = df_sorted.sample(frac=1, random_state=0).reset_index(drop=True)

        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0)
        )
        candidates_sorted = EventDiscovery(df_sorted, config=cfg).run()
        candidates_shuffled = EventDiscovery(df_shuffled, config=cfg).run()

        ids_sorted = {c.event_id for c in candidates_sorted}
        ids_shuffled = {c.event_id for c in candidates_shuffled}
        assert ids_sorted == ids_shuffled, (
            f"Sorted input produced {ids_sorted - ids_shuffled} extra events and "
            f"missed {ids_shuffled - ids_sorted} compared to shuffled input"
        )

    def test_unsorted_datetimeindex_input_is_sorted(self):
        """DatetimeIndex path also sorts rows before rolling calculations."""
        df = _make_kpi_table(n=2000)
        df_sorted = df.set_index("open_dt")
        df_shuffled = df_sorted.sample(frac=1, random_state=0)

        cfg = DiscoveryConfig(
            timestamp_col="open_dt",
            gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0),
        )
        candidates_sorted = EventDiscovery(df_sorted, config=cfg).run()
        candidates_shuffled = EventDiscovery(df_shuffled, config=cfg).run()

        ids_sorted = {c.event_id for c in candidates_sorted}
        ids_shuffled = {c.event_id for c in candidates_shuffled}
        assert ids_sorted == ids_shuffled


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


class TestWalkForward:
    """Tests for train/test split and walk-forward OOS validation."""

    @pytest.fixture(scope="class")
    @classmethod
    def long_df(cls):
        """8 760-bar (≈12 months, 1H) table shared across all walk-forward
        tests.  Each test builds its own EventDiscovery with its own config,
        so sharing the immutable input DataFrame is safe."""
        return _make_kpi_table(n=8760, seed=7)

    def test_no_split_leaves_validation_none(self, long_df):
        cfg = DiscoveryConfig(gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0))
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        assert ed.oos_period is None
        assert all(c.validation is None for c in cands)

    def test_split_without_wf_has_correct_periods(self, long_df):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
        )
        ed = EventDiscovery(long_df, cfg)
        ed.run()
        assert ed.is_period is not None
        assert ed.oos_period is not None
        assert ed.is_period[1] < ed.oos_period[0]
        is_start, is_end = ed.is_period
        assert isinstance(is_start, pd.Timestamp)

    def test_split_without_wf_validation_is_none(self, long_df):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
        )
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        assert all(c.validation is None for c in cands)

    def test_raw_events_none_before_run(self, long_df):
        ed = EventDiscovery(long_df)
        assert ed.raw_events is None

    def test_raw_events_populated_after_run(self, long_df):
        ed = EventDiscovery(long_df)
        cands = ed.run()
        assert ed.raw_events is not None
        assert len(ed.raw_events) > 0

    def test_raw_events_count_exceeds_single_event_candidates(self, long_df):
        cfg = DiscoveryConfig(gate_params=GateParams(min_act=20, min_months=4, max_conc=0.60, min_tpm=1.0))
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        # raw_events are pre-gate atomic events; passing single-event candidates are a filtered subset
        single_cands = [c for c in cands if len(c.components) == 1]
        assert len(ed.raw_events) >= len(single_cands)

    def test_walk_forward_populates_validation(self, long_df):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
            walk_forward=WalkForwardConfig(n_splits=3, min_pass_rate=0.5),
        )
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        assert len(cands) > 0
        assert all(c.validation is not None for c in cands)
        v = cands[0].validation
        assert isinstance(v, ValidationResult)
        assert v.n_folds == 3
        assert len(v.fold_results) == 3
        assert 0.0 <= v.pass_rate <= 1.0
        assert v.passed == (v.pass_rate >= 0.5)

    def test_fold_results_structure(self, long_df):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
            walk_forward=WalkForwardConfig(n_splits=2, min_pass_rate=0.5),
        )
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        for cand in cands:
            v = cand.validation
            for i, fr in enumerate(v.fold_results):
                assert isinstance(fr, FoldResult)
                assert fr.fold_idx == i
                assert fr.n_rows > 0
                assert fr.passed == fr.gate_result.passed

    def test_validated_candidates_filters_correctly(self, long_df):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
            walk_forward=WalkForwardConfig(n_splits=3, min_pass_rate=0.5),
        )
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        stable = ed.validated_candidates()
        assert all(c.validation.passed for c in stable)
        assert len(stable) <= len(cands)

    def test_validated_candidates_raises_without_config(self, long_df):
        cfg = DiscoveryConfig(gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0))
        ed = EventDiscovery(long_df, cfg)
        ed.run()
        with pytest.raises(RuntimeError, match="Walk-forward validation was not configured"):
            ed.validated_candidates()

    def test_summary_includes_oos_columns_when_wf_set(self, long_df):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
            walk_forward=WalkForwardConfig(n_splits=2, min_pass_rate=0.5),
        )
        ed = EventDiscovery(long_df, cfg)
        ed.run()
        s = ed.summary()
        for col in ("oos_pass_rate", "oos_n_passed", "oos_n_folds", "oos_stable"):
            assert col in s.columns, f"Missing column: {col}"

    def test_summary_no_oos_columns_without_wf(self, long_df):
        cfg = DiscoveryConfig(gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0))
        ed = EventDiscovery(long_df, cfg)
        ed.run()
        s = ed.summary()
        assert "oos_pass_rate" not in s.columns

    def test_custom_oos_gate_params(self, long_df):
        strict_oos = GateParams(min_tpm=999.0, max_dispersion=0.001)
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.8, max_dispersion=10.0),
            train_ratio=0.70,
            walk_forward=WalkForwardConfig(n_splits=2, min_pass_rate=0.5, oos_gate_params=strict_oos),
        )
        ed = EventDiscovery(long_df, cfg)
        cands = ed.run()
        stable = ed.validated_candidates()
        assert len(stable) == 0


# ---------------------------------------------------------------------------
# Bug-regression tests
# ---------------------------------------------------------------------------

def _make_component(
    source_feature: str,
    transform: str,
    transform_params: dict,
    source_cols: list,
    threshold: float = 0.5,
    direction: str = "above",
    event_type: str = "threshold",
) -> EventComponent:
    return EventComponent(
        source_feature=source_feature,
        transform=transform,
        transform_params=transform_params,
        transformed_col=source_feature,
        threshold=threshold,
        threshold_type="test",
        direction=direction,
        event_type=event_type,
        expression=f"{source_feature} > {threshold}",
        source_cols=source_cols,
        sql_expression="",
    )


class TestBugRegressions:
    # ── Bug #1: _apply_component leaves ±inf in ratio_ / spread_ series ────

    def test_ratio_zero_denominator_gives_nan_not_inf(self):
        """ratio_ replay must produce NaN (not ±inf) where denominator == 0."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 0.0, 2.0]})
        comp = _make_component(
            "ratio_a_b", "identity", {}, ["a", "b"], threshold=0.5
        )
        series = _apply_component(comp, df)
        assert not np.isinf(series).any(), "±inf found in ratio_ replay result"
        assert pd.isna(series.iloc[1]), "zero-denominator bar should be NaN"

    def test_spread_zero_denominator_gives_nan_not_inf(self):
        """spread_ replay must produce NaN (not ±inf) where denominator == 0."""
        df = pd.DataFrame({"price": [100.0, 100.0, 102.0], "ma": [100.0, 0.0, 100.0]})
        comp = _make_component(
            "spread_price_ma", "identity", {}, ["price", "ma"], threshold=0.0
        )
        series = _apply_component(comp, df)
        assert not np.isinf(series).any(), "±inf found in spread_ replay result"
        assert pd.isna(series.iloc[1])

    def test_ratio_replay_matches_training_path(self):
        """End-to-end: apply() on out-of-sample data with zero-denominator bars
        must not produce ±inf that would corrupt rolling window statistics."""
        rng = np.random.default_rng(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.005, 500))
        ema = pd.Series(close).ewm(span=25).mean().values.copy()
        ema[50] = 0.0  # inject a zero-denominator bar
        df = pd.DataFrame({"close": close, "close_ema_25": ema,
                           "open_dt": pd.date_range("2024-01-01", periods=500, freq="1h")})

        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                gate_params=GateParams(min_tpm=0.1, max_dispersion=100.0)
            ),
        )
        ed.run()
        for cand in ed._candidates or []:
            result = cand.apply(df.drop(columns=["open_dt"]))
            assert not np.isinf(result.fillna(0)).any(), (
                f"±inf found in apply() result for {cand.event_id}"
            )

    # ── Bug #2: ANDComposer float32 → float64 boundary consistency ──────────

    def test_and_composer_max_conc_matches_single_gate(self):
        """max_monthly_share in AND-composed GateResult must match float64
        ConsistencyGate.evaluate() for the same activation array."""
        # Use a 1-year hourly series where each event fires ~15% of bars
        # (≈1314 activations). Gate is loose so the AND pair passes easily.
        rng = np.random.default_rng(7)
        n = 8760
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        month_idx, n_months = _build_month_index(ts)

        gate = ConsistencyGate(GateParams(min_tpm=2.0, max_dispersion=2.5))

        # Build two highly correlated events so their AND fires often enough
        base = rng.random(n) < 0.30
        s1 = pd.Series((base & (rng.random(n) < 0.80)).astype(float))
        s2 = pd.Series((base & (rng.random(n) < 0.80)).astype(float))

        def _make_ev(s: pd.Series, name: str) -> RawEvent:
            comp = _make_component(name, "identity", {}, [])
            ev = RawEvent(series=s, component=comp)
            ev.gate_result = gate.evaluate_series(s, month_idx, n_months)
            return ev

        ev1 = _make_ev(s1, "feat_a")
        ev2 = _make_ev(s2, "feat_b")
        assert ev1.gate_result.passed and ev2.gate_result.passed, (
            "Fixture setup failed: single events must pass gate"
        )

        composed = ANDComposer(gate).compose([ev1, ev2], ts, max_components=2)
        assert composed, "Fixture setup failed: AND composed event must pass gate"

        for ev in composed:
            and_arr = (ev1.series.fillna(0).values.astype(bool) &
                       ev2.series.fillna(0).values.astype(bool))
            counts = _count_by_month(and_arr, month_idx, n_months)
            scalar = gate.evaluate(and_arr, counts, n_months)
            assert abs(ev.gate_result.max_monthly_share - scalar.max_monthly_share) < 1e-9, (
                "float64 mismatch between ANDComposer and ConsistencyGate.evaluate"
            )

    # ── Bug #3: diffnorm_std == 0 must return all-NaN, not KeyError ─────────

    def test_diffnorm_std_zero_returns_all_nan(self):
        """_apply_component with diffnorm_std=0 must return NaN series, not raise."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        comp = _make_component(
            "diffnorm_a_b", "identity", {"diffnorm_std": 0.0}, ["a", "b"]
        )
        result = _apply_component(comp, df)
        assert result.isna().all(), "diffnorm_std=0 should produce all-NaN"

    def test_diffnorm_std_none_raises_key_error(self):
        """_apply_component with missing diffnorm_std must raise KeyError."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.5, 1.0]})
        comp = _make_component(
            "diffnorm_a_b", "identity", {}, ["a", "b"]
        )
        with pytest.raises(KeyError, match="diffnorm_std"):
            _apply_component(comp, df)

    # ── Bug #4: _monthly_counts index-alignment safety ───────────────────────

    def test_monthly_counts_mismatched_indices(self):
        """_monthly_counts must produce correct results when active has a
        DatetimeIndex but timestamps has a RangeIndex."""
        dates = pd.date_range("2024-01-01", periods=60, freq="1D")
        active_dt = pd.Series(
            [1.0] * 30 + [0.0] * 30,
            index=dates,
        )
        timestamps_ri = pd.Series(dates)

        result = _monthly_counts(active_dt, timestamps_ri)
        assert result[pd.Period("2024-01", freq="M")] == 30
        assert result[pd.Period("2024-02", freq="M")] == 0

    def test_monthly_counts_rangeindex_active(self):
        """_monthly_counts with RangeIndex active and RangeIndex timestamps."""
        dates = pd.date_range("2024-01-01", periods=60, freq="1D")
        active_ri = pd.Series([1.0] * 30 + [0.0] * 30)
        timestamps_ri = pd.Series(dates)

        result = _monthly_counts(active_ri, timestamps_ri)
        assert result[pd.Period("2024-01", freq="M")] == 30
        assert result[pd.Period("2024-02", freq="M")] == 0

    # ── Bug #5: NaT in period range must not crash ───────────────────────────

    def test_count_zero_months_nat_index_returns_zero(self):
        """_count_zero_months must return 0 (not crash) when index contains NaT."""
        nat_index = pd.DatetimeIndex([pd.NaT, pd.NaT, pd.NaT])
        series = pd.Series([1.0, 0.0, 1.0], index=nat_index)
        ts = pd.Series([pd.NaT, pd.NaT, pd.NaT])
        result = _count_zero_months(series, ts)
        assert result == 0

    def test_monthly_counts_nat_timestamps_returns_empty(self):
        """_monthly_counts must return empty Series (not crash) on all-NaT timestamps."""
        active = pd.Series([1.0, 0.0, 1.0])
        timestamps = pd.Series(pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]))
        result = _monthly_counts(active, timestamps)
        assert len(result) == 0


    # ── Bug #6: EventComponent._components dynamic attr lost after deepcopy ──

    def test_and_composition_components_survive_deepcopy(self):
        """EventComponent.components must be a declared field, not a dynamic
        attribute, so it survives deepcopy (and any serialisation round-trip).

        Regression for issue #52: previously comp._components was set as a
        dynamic attr which is dropped by dataclasses serialisation helpers.
        """
        import copy

        df = _make_kpi_table(n=4380, seed=42)
        cfg = DiscoveryConfig(max_and_components=2, gate_params=GateParams(
            min_tpm=0.5, max_dispersion=15.0
        ))
        ed = EventDiscovery(df, config=cfg)
        candidates = ed.run()

        and_candidates = [c for c in candidates if len(c.components) == 2]
        assert and_candidates, "need at least one AND-composed candidate for this test"

        original = and_candidates[0]
        restored = copy.deepcopy(original)

        assert len(restored.components) == 2
        for orig_comp, rest_comp in zip(original.components, restored.components):
            assert orig_comp.source_feature == rest_comp.source_feature
            assert orig_comp.transform == rest_comp.transform
            assert orig_comp.threshold == rest_comp.threshold

    # ── Bug #7: _build_month_index KeyError on NaT timestamps ───────────────

    def test_build_month_index_skips_nat_timestamps(self):
        """_build_month_index must not raise KeyError when timestamps contain NaT.

        Regression for issue #53: pandas Period.unique() excludes NaT, so the
        dict lookup crashed when iterating NaT periods.  NaT rows are now
        assigned sentinel -1 and skipped in _count_by_month.
        """
        dates = pd.date_range("2024-01-01", periods=6, freq="1ME")
        timestamps = pd.Series([dates[0], pd.NaT, dates[1], pd.NaT, dates[2], dates[3]])

        month_index, n_months = _build_month_index(timestamps)

        assert n_months == 4
        assert len(month_index) == len(timestamps)
        # NaT rows (indices 1 and 3) must be -1
        assert month_index[1] == -1
        assert month_index[3] == -1
        # Valid rows must have non-negative indices
        assert all(month_index[i] >= 0 for i in [0, 2, 4, 5])

    def test_count_by_month_ignores_sentinel_indices(self):
        """_count_by_month must skip month_index == -1 without IndexError."""
        # 6 rows: 3 valid months, 2 NaT (sentinel -1)
        month_index = np.array([0, -1, 1, -1, 2, 2], dtype=np.int32)
        active = np.array([True, True, True, True, True, False], dtype=bool)

        counts = _count_by_month(active, month_index, n_months=3)

        assert counts.tolist() == [1, 1, 1]  # NaT rows ignored

    def test_consistency_gate_filter_survives_nat_timestamps(self):
        """ConsistencyGate.filter must not crash when the timestamp series
        contains NaT values (issue #53)."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        timestamps = pd.Series(dates, dtype="datetime64[ns]")
        # Inject NaT at a few positions
        timestamps.iloc[0] = pd.NaT
        timestamps.iloc[50] = pd.NaT

        rng = np.random.default_rng(42)
        event_series = pd.Series(
            (rng.random(n) > 0.8).astype(float), index=dates
        )

        from forgedge.event_discovery.models import EventComponent, RawEvent
        comp = EventComponent(
            source_feature="close",
            transform="identity",
            transform_params={},
            transformed_col="close",
            threshold=0.5,
            threshold_type="distributional_p50",
            direction="above",
            event_type="threshold",
            expression="close > 0.5",
        )
        raw = RawEvent(series=event_series, component=comp)

        gate = ConsistencyGate()
        # Must not raise — result can pass or fail, but no exception
        result = gate.filter([raw], timestamps)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Diversity Gate
# ---------------------------------------------------------------------------

from forgedge.event_discovery.diversity_gate import apply_diversity_gate
from forgedge.event_discovery.models import GateResult


def _make_raw_event(series: pd.Series, n_activations: int) -> RawEvent:
    """Build a minimal RawEvent with a populated gate_result."""
    gate = GateResult(
        passed=True,
        n_activations=n_activations,
        n_active_months=1,
        max_monthly_share=0.5,
        mean_tpm=1.0,
    )
    from forgedge.event_discovery.models import EventComponent
    comp = EventComponent(
        source_feature="x",
        transform="identity",
        transform_params={},
        transformed_col="x",
        threshold=0.5,
        threshold_type="distributional_p50",
        direction="below",
        event_type="threshold",
        expression="x < 0.5",
    )
    ev = RawEvent(series=series, component=comp)
    ev.gate_result = gate
    return ev


class TestDiversityGate:
    """Unit tests for apply_diversity_gate()."""

    def _bool_series(self, indices: list[int], length: int = 100) -> pd.Series:
        """Build a 0/1 Series with 1s at the given indices."""
        s = pd.Series(0.0, index=range(length))
        for i in indices:
            s.iloc[i] = 1.0
        return s

    def test_identical_events_deduplicated(self):
        """Two events with identical series (J=1.0) → one is discarded."""
        s = self._bool_series(list(range(10)))
        ev1 = _make_raw_event(s.copy(), n_activations=10)
        ev2 = _make_raw_event(s.copy(), n_activations=10)
        result = apply_diversity_gate([ev1, ev2], threshold=0.85)
        assert len(result) == 1

    def test_diverse_events_preserved(self):
        """Two events with J < threshold → both kept."""
        s1 = self._bool_series(list(range(10)))        # indices 0-9
        s2 = self._bool_series(list(range(50, 60)))    # indices 50-59, no overlap
        ev1 = _make_raw_event(s1, n_activations=10)
        ev2 = _make_raw_event(s2, n_activations=10)
        result = apply_diversity_gate([ev1, ev2], threshold=0.85)
        assert len(result) == 2

    def test_priority_keeps_more_frequent_event(self):
        """With near-identical events, the one with more activations is kept."""
        # ev_big has 20 activations; ev_small has 10, all within ev_big's set.
        big_indices = list(range(20))
        small_indices = list(range(10))  # subset → J = 10/20 = 0.5
        # Make them identical (J=1.0) to guarantee deduplication
        s_big = self._bool_series(big_indices)
        s_small = self._bool_series(big_indices)   # identical series
        ev_big   = _make_raw_event(s_big,   n_activations=20)
        ev_small = _make_raw_event(s_small, n_activations=10)
        result = apply_diversity_gate([ev_big, ev_small], threshold=0.85)
        assert len(result) == 1
        assert result[0].gate_result.n_activations == 20

    def test_disabled_via_discovery_config_is_noop(self):
        """diversity_gate_enabled=False → same candidates as baseline."""
        df = _make_kpi_table(n=4380)
        gate_p = GateParams(min_tpm=1.0, max_dispersion=10.0)
        base = EventDiscovery(df, DiscoveryConfig(gate_params=gate_p)).run()
        with_gate = EventDiscovery(
            df,
            DiscoveryConfig(gate_params=gate_p, diversity_gate_enabled=False),
        ).run()
        assert len(base) == len(with_gate)

    def test_threshold_boundary_exact_duplicate_removed(self):
        """J == 1.0 >= threshold=0.85 → duplicate is discarded."""
        s = self._bool_series(list(range(20)))
        ev1 = _make_raw_event(s.copy(), n_activations=20)
        ev2 = _make_raw_event(s.copy(), n_activations=20)
        result = apply_diversity_gate([ev1, ev2], threshold=0.85)
        assert len(result) == 1

    def test_threshold_1_0_keeps_near_duplicates(self):
        """threshold=1.0 → only exact duplicates removed; overlapping events kept."""
        s1 = self._bool_series(list(range(10)))
        s2 = self._bool_series(list(range(5, 15)))  # J = 5/15 = 0.33
        # Force high overlap: 9 shared out of 11 → J = 9/11 ≈ 0.818 < 1.0
        s1 = self._bool_series(list(range(10)))
        s2 = self._bool_series(list(range(1, 11)))  # J = 9/11 ≈ 0.818
        ev1 = _make_raw_event(s1, n_activations=10)
        ev2 = _make_raw_event(s2, n_activations=10)
        result = apply_diversity_gate([ev1, ev2], threshold=1.0)
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        result = apply_diversity_gate([], threshold=0.85)
        assert result == []

    def test_diversity_gate_reduces_pool_in_pipeline(self):
        """diversity_gate_enabled=True yields ≤ candidates than baseline."""
        df = _make_kpi_table(n=4380)
        gate_p = GateParams(min_tpm=1.0, max_dispersion=10.0)
        base = EventDiscovery(df, DiscoveryConfig(gate_params=gate_p)).run()
        filtered = EventDiscovery(
            df,
            DiscoveryConfig(
                gate_params=gate_p,
                diversity_gate_enabled=True,
                diversity_threshold=0.85,
            ),
        ).run()
        # After deduplication the single-event pool is smaller or equal;
        # the AND pool may differ, so total candidates can only decrease or stay equal.
        assert len(filtered) <= len(base)


# ---------------------------------------------------------------------------
# Custom Event Injection (issue #77)
# ---------------------------------------------------------------------------

from forgedge.event_discovery.models import CustomEvent


class TestCustomEvent:
    """Unit tests for CustomEvent — user-defined formula events."""

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open_dt": pd.date_range("2024-01-01", periods=6, freq="1h"),
                "close_adj_v2": [120.0, 95.0, 100.0, 80.0, 110.0, 90.0],
                "volume": [1, 2, 3, 4, 5, 6],
            }
        )

    def test_formula_evaluated_correctly(self):
        """apply() produces the expected boolean Series."""
        df = self._frame()
        ev = CustomEvent("close_adj_v2 < 100", name="below_100")
        series = ev.apply(df)
        assert series.dtype == bool
        assert series.tolist() == [False, True, False, True, False, True]

    def test_invalid_formula_raises_value_error(self):
        """A formula referencing an unknown column raises a readable ValueError."""
        df = self._frame()
        ev = CustomEvent("nonexistent_col < 100")
        with pytest.raises(ValueError, match="failed to evaluate"):
            ev.apply(df)

    def test_name_defaults_to_formula(self):
        ev = CustomEvent("close_adj_v2 < 100")
        assert ev.name == "close_adj_v2 < 100"

    def test_to_event_candidate_apply_roundtrip(self):
        """EventCandidate.apply() re-evaluates the custom formula on new data."""
        df = self._frame()
        ev = CustomEvent("close_adj_v2 < 100", name="below_100")
        cand = ev.to_event_candidate(df)

        # The cached series matches the direct evaluation.
        assert cand.event_series.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        # apply() on a fresh frame with different values recomputes correctly.
        df2 = df.copy()
        df2["close_adj_v2"] = [50.0, 200.0, 50.0, 200.0, 50.0, 200.0]
        replayed = cand.apply(df2)
        assert replayed.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def test_candidate_carries_formula_in_expressions(self):
        """The custom formula surfaces in expression / sql / event_formula."""
        df = self._frame()
        ev = CustomEvent("close_adj_v2 < 100", name="below_100")
        cand = ev.to_event_candidate(df)
        assert cand.expression == "close_adj_v2 < 100"
        assert cand.sql_expression == "close_adj_v2 < 100"
        assert cand.event_formula == "close_adj_v2 < 100"
        assert cand.event_id == "CUSTOM-below_100"

    def test_compound_formula(self):
        """A multi-condition formula evaluates with pandas-eval semantics."""
        df = self._frame()
        ev = CustomEvent("close_adj_v2 < 100 and volume > 3")
        series = ev.apply(df)
        # close<100 → [F,T,F,T,F,T]; volume>3 → [F,F,F,T,T,T]; AND → [F,F,F,T,F,T]
        assert series.tolist() == [False, False, False, True, False, True]

    def test_nan_treated_as_inactive(self):
        """NaN comparison results are coerced to inactive (False)."""
        df = self._frame()
        df.loc[2, "close_adj_v2"] = np.nan
        ev = CustomEvent("close_adj_v2 < 100")
        series = ev.apply(df)
        assert series.iloc[2] == False  # noqa: E712 — NaN < 100 → inactive
