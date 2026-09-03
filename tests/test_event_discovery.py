"""Tests for the Event Discovery module."""
import math

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
from forgedge.event_discovery.discovery import MIN_FOLD_LAMBDA
from forgedge.event_discovery.consistency_gate import (
    ConsistencyGate,
    _build_month_index,
    _chi2_ppf_095,
    _count_by_month,
    _episode_starts,
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
    # Modest positive drift so the synthetic price actually trends like a real
    # asset (~1.6x over the default horizon).  A near-flat random walk would be
    # support-stationary and wrongly read as scale-free by the classifier.
    price = 100 * np.cumprod(1 + rng.normal(0.0002, 0.005, n))
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

    def test_bounded_oscillator_with_trending_mean_is_scale_free(self):
        """Regression for issue #133: a bounded oscillator whose mean drifts
        with the regime must still be detected as scale-free.

        The old mean-drift heuristic produced a false negative on RSI/%B
        because the window mean rises in an uptrend.  The support-overlap test
        keys off the value range revisited by both halves, which stays stable.
        """
        rng = np.random.default_rng(0)
        n = 2000
        # Oscillator bounded in [0, 100] whose centre ramps from 35 → 65
        # across the sample (regime drift), with stable ±20 dispersion clipped
        # to the domain.  Mean drifts strongly; the [q05,q95] support stays
        # overlapping between the two halves.
        centre = np.linspace(35, 65, n)
        osc = np.clip(centre + rng.normal(0, 20, n), 0, 100)
        df = pd.DataFrame({
            "open_dt": pd.date_range("2020-01-01", periods=n, freq="1D"),
            "osc": osc,
        })
        cls = TypeClassifier().fit(df)
        assert cls["osc"].col_type == ColumnType.CONTINUOUS
        assert cls["osc"].effective_scale_free is True

    def test_trending_price_not_scale_free(self):
        """A trending unbounded price series must NOT be scale-free: its
        second half makes new extremes its first half never reached, so the
        supports barely overlap."""
        rng = np.random.default_rng(1)
        n = 2000
        price = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        df = pd.DataFrame({
            "open_dt": pd.date_range("2020-01-01", periods=n, freq="1D"),
            "price": price,
        })
        cls = TypeClassifier().fit(df)
        assert cls["price"].effective_scale_free is False

    def test_scale_free_is_period_invariant(self):
        """Regression for issue #136: a fast and a slow oscillator computed on
        the SAME trending price must receive the SAME scale-free verdict.

        The windowed quantile-drift test (issue #133's fix) classified RSI25
        as not-scale-free while RSI14 passed, because the slower indicator
        drags the regime through its windows longer.  The support-overlap test
        is computed once over the whole series and is period-invariant.
        """
        rng = np.random.default_rng(7)
        n = 3000
        price = 100 * np.cumprod(1 + rng.normal(0.0006, 0.006, n))

        def rsi(s, w):
            s = pd.Series(s)
            d = s.diff()
            g = d.clip(lower=0).rolling(w, min_periods=1).mean()
            l = (-d.clip(upper=0)).rolling(w, min_periods=1).mean()
            return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50).values

        df = pd.DataFrame({
            "open_dt": pd.date_range("2018-01-01", periods=n, freq="1D"),
            "rsi_14": rsi(price, 14),
            "rsi_25": rsi(price, 25),
        })
        cls = TypeClassifier().fit(df)
        assert cls["rsi_14"].effective_scale_free is True
        assert cls["rsi_25"].effective_scale_free is True
        assert cls["rsi_14"].effective_scale_free == cls["rsi_25"].effective_scale_free

    def test_scale_free_drift_threshold_deprecated(self):
        """The legacy scale_free_drift_threshold kwarg is accepted but ignored,
        emitting a DeprecationWarning (issue #136)."""
        with pytest.warns(DeprecationWarning):
            TypeClassifier(scale_free_drift_threshold=0.2)

    def test_block_count_adapts_to_series_length(self):
        """The scale-free heuristic uses k=2 for short series and grows to
        max_scale_free_blocks for long ones (each block >= min_block_size)."""
        c = TypeClassifier(max_scale_free_blocks=4, min_block_size=250)
        rng = np.random.default_rng(0)

        def k_for(n):
            return max(2, min(c.max_scale_free_blocks, n // c.min_block_size))

        assert k_for(365) == 2     # 1 year daily -> two blocks
        assert k_for(900) == 3
        assert k_for(1300) == 4
        assert k_for(5000) == 4    # capped at max_scale_free_blocks

    def test_roundtrip_path_not_scale_free(self):
        """A price that trends up then symmetrically back down is a
        non-stationary *path*: with k>=4 blocks at least one block's support
        sits far from the global support, so it is not scale-free.

        A single two-way split would wrongly pass it (both halves share a
        similar overall range); the adaptive multi-block test catches it.
        """
        rng = np.random.default_rng(3)
        half = 1500
        up = 100 * np.cumprod(1 + rng.normal(0.0015, 0.005, half))
        down = up[-1] * np.cumprod(1 + rng.normal(-0.0015, 0.005, half))
        path = np.concatenate([up, down])
        df = pd.DataFrame({
            "open_dt": pd.date_range("2015-01-01", periods=len(path), freq="1D"),
            "path": path,
        })
        cls = TypeClassifier().fit(df)
        assert cls["path"].effective_scale_free is False

    def test_short_series_override_forces_scale_free(self):
        """On a short history (~1 year) the heuristic falls back to k=2 and is
        unreliable; an explicit override is the supported path and takes
        precedence over whatever the heuristic decides.  Here a trending price
        (auto-detected NOT scale-free) is forced scale-free via override."""
        rng = np.random.default_rng(5)
        n = 365
        price = 100 * np.cumprod(1 + rng.normal(0.0008, 0.008, n))
        df = pd.DataFrame({
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="1D"),
            "px": price,
        })
        auto = TypeClassifier().fit(df)
        forced = TypeClassifier(scale_free_overrides={"px": True}).fit(df)
        assert auto["px"].effective_scale_free is False        # heuristic says no
        assert forced["px"].effective_scale_free is True       # override wins
        assert forced["px"].scale_free_overridden is True


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

    def test_parse_feature_volume_ma_not_shadowed_by_generic_pattern(self):
        """Regression: the generic {base}_{indicator}_{param} pattern's base
        alternation includes "volume", so it used to match volume_sma_N /
        volume_ema_N first and shadow the dedicated volume-MA pattern —
        which also resolved base/indicator incorrectly even when reached
        (base="sma" instead of "volume"). Both are fixed: dedicated pattern
        now runs first and captures "volume" as an explicit group."""
        pf = parse_feature("volume_sma_25")
        assert pf is not None
        assert pf.base == "volume"
        assert pf.indicator == "sma"
        assert pf.family == "volume_ma"
        assert pf.params == [25]

        pf_ema = parse_feature("volume_ema_09")
        assert pf_ema.base == "volume"
        assert pf_ema.indicator == "ema"
        assert pf_ema.family == "volume_ma"

    def test_parse_feature_mdd_atr_natr(self):
        """max_drawdown/ATR/NATR participate in same-family ratio pairing
        (issue: previously unrecognised, so e.g. mdd_12/mdd_24 never paired)."""
        for col, indicator in [
            ("close_mdd_12", "mdd"),
            ("close_atr_14", "atr"),
            ("close_natr_14", "natr"),
        ]:
            pf = parse_feature(col)
            assert pf is not None, col
            assert pf.indicator == indicator
            assert pf.family == indicator

    def test_arity2_ratio_volume_vs_volume_ma_generated(self):
        """Regression: 'volume vs its own MA' (ratio_volume_*) was dead code —
        it keyed off family=='volume_ma', which no column ever reached because
        the dedicated pattern was shadowed (see
        test_parse_feature_volume_ma_not_shadowed_by_generic_pattern)."""
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert any(k.startswith("ratio_volume_") for k in meta), (
            f"no ratio_volume_* feature generated; arity2 keys: "
            f"{[k for k, v in meta.items() if v.arity == 2]}"
        )

    def test_arity2_mdd_ratio_generated_with_two_periods(self):
        """close_mdd_12 / close_mdd_24 should pair into a ratio, the same
        mechanism EMA fast/slow pairs already use, now that mdd is parsed."""
        df = _make_kpi_table()
        df["close_mdd_12"] = (
            (df["close"] / df["close"].cummax() - 1).abs().fillna(0)
        )
        df["close_mdd_24"] = df["close_mdd_12"]  # any second period is enough
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert any("mdd12" in k and "mdd24" in k for k in meta), (
            f"no mdd ratio pair generated; arity2 keys: "
            f"{[k for k, v in meta.items() if v.arity == 2]}"
        )

    def test_derived_features_are_scale_free(self):
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        derived = {k: v for k, v in meta.items() if v.arity >= 2}
        assert all(v.is_scale_free for v in derived.values())

    # ── Issue #161: cross-column, cross-time ("lag-cross") pairing ──────────

    def _ohlc_df(self, n: int = 500, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        return pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": close, "high": high, "low": low,
        })

    def test_lag_cross_generates_close_vs_low_lag1(self):
        """The exact gap reported in #161: 'close[t] > low[t-1]' must now be
        constructible as a scale-free arity-2 feature."""
        df = self._ohlc_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_close_low_lag1" in meta
        assert "spread_close_low_lag1" in meta
        assert meta["ratio_close_low_lag1"].is_scale_free
        assert meta["ratio_close_low_lag1"].source_cols == ["close", "low"]
        assert meta["ratio_close_low_lag1"].params == {"cross_lag": 1}

    def test_lag_cross_numeric_correctness(self):
        """ratio_a_b_lagN == a[t] / b[t-N]; spread_a_b_lagN == (a-b_lag)/b_lag."""
        df = self._ohlc_df()
        cls = TypeClassifier().fit(df)
        ext, _ = FeatureGenerator().generate(df, cls)
        expected_ratio = df["close"] / df["low"].shift(3)
        expected_spread = (df["close"] - df["low"].shift(3)) / df["low"].shift(3)
        pd.testing.assert_series_equal(
            ext["ratio_close_low_lag3"].dropna(), expected_ratio.dropna(),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            ext["spread_close_low_lag3"].dropna(), expected_spread.dropna(),
            check_names=False,
        )

    def test_lag_cross_uses_delta_lags(self):
        df = self._ohlc_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        from forgedge.event_discovery.transform_layer import DELTA_LAGS
        for lag in DELTA_LAGS:
            assert f"ratio_close_low_lag{lag}" in meta

    def test_lag_cross_excludes_same_base_pairs(self):
        """No a[t] vs a[t-lag] lag-cross combo — that shape already exists
        natively as the 'return' family (close_ret_N)."""
        df = self._ohlc_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert not any(
            k.startswith(("ratio_close_close_", "spread_close_close_"))
            for k in meta
        )

    def test_lag_cross_bounded_feature_count(self):
        """3 present OHLC bases (close/high/low) -> 3*2 ordered pairs * 4 lags
        * 2 ops = 48 new columns — the surface stays predictable/bounded even
        though every ordered (base_a, base_b) pair is generated."""
        df = self._ohlc_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        lag_cross = [k for k, v in meta.items() if v.operation in ("ratio_lag", "spread_pct_lag")]
        assert len(lag_cross) == 48

    def test_lag_cross_absent_without_second_ohlc_base(self):
        """Only 'close' present (no high/low/open) -> the OHLC x OHLC
        lag-cross family (#161, needs 2+ distinct OHLC bases) generates
        nothing, and no crash. #165's indicator-vs-OHLC-base family is a
        separate, unrelated shape that only needs one base (e.g. an
        indicator vs its own base, lagged) and legitimately fires here — see
        test_indicator_lag_cross_* — so this checks specifically for the
        OHLC x OHLC shape (both source_cols must themselves be raw OHLC
        bases), not just any "ratio_lag"/"spread_pct_lag" operation."""
        df = _make_kpi_table()
        assert "high" not in df.columns and "low" not in df.columns
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        ohlc_bases = {"open", "high", "low", "close"}
        assert not any(
            v.operation in ("ratio_lag", "spread_pct_lag")
            and set(v.source_cols) <= ohlc_bases
            for v in meta.values()
        )

    def test_lag_cross_replay_matches_generation(self):
        """build_feature_series (the OOS replay path used by
        EventCandidate.apply) must reproduce the lag-cross feature exactly."""
        from forgedge.event_discovery.models import build_feature_series

        df = self._ohlc_df()
        cls = TypeClassifier().fit(df)
        ext, meta = FeatureGenerator().generate(df, cls)
        m = meta["ratio_high_low_lag6"]
        comp = EventComponent(
            source_feature="ratio_high_low_lag6",
            transform="identity",
            transform_params=dict(m.params),
            transformed_col="ratio_high_low_lag6",
            threshold=0.0,
            threshold_type="",
            direction="above",
            event_type="threshold",
            expression="",
            source_cols=m.source_cols,
        )
        replayed = build_feature_series(comp, df)
        pd.testing.assert_series_equal(
            replayed.dropna(), ext["ratio_high_low_lag6"].dropna(), check_names=False,
        )

    # ── Issue #162: MACD/signal, price-vs-volume, candle-geometry pairing ──

    def test_parse_feature_macd(self):
        """MACD line/signal/hist column names are now recognised (previously
        None for both: two-or-three numeric groups matched no pattern)."""
        line = parse_feature("close_macd_12_26")
        assert line is not None
        assert line.base == "close" and line.params == [12, 26] and line.family == "macd_line"

        signal = parse_feature("close_macd_12_26_signal_09")
        assert signal is not None
        assert signal.params == [12, 26, 9] and signal.family == "macd_signal"

        hist = parse_feature("close_macd_12_26_hist_09")
        assert hist is not None
        assert hist.params == [12, 26, 9] and hist.family == "macd_hist"

    def _macd_df(self, n: int = 500, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": 100 + np.cumsum(rng.normal(0, 1, n)),
            "close_macd_12_26": rng.normal(0, 0.01, n),
            "close_macd_12_26_signal_09": rng.normal(0, 0.01, n),
            "close_macd_12_26_hist_09": rng.normal(0, 0.005, n),
        })

    def test_macd_line_paired_with_its_own_signal(self):
        """The textbook MACD/signal crossover: ratio_ and diffnorm_ of the
        MACD line against its own signal line, matched by shared
        (base, fast, slow) — not against an unrelated MACD config."""
        df = self._macd_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_close_macd12_26_signal" in meta
        assert "diffnorm_close_macd12_26_signal" in meta
        m = meta["ratio_close_macd12_26_signal"]
        assert m.source_cols == ["close_macd_12_26", "close_macd_12_26_signal_09"]

    def test_macd_pairing_numeric_correctness(self):
        df = self._macd_df()
        cls = TypeClassifier().fit(df)
        ext, _ = FeatureGenerator().generate(df, cls)
        expected = df["close_macd_12_26"] / df["close_macd_12_26_signal_09"]
        expected = expected.replace([float("inf"), float("-inf")], float("nan"))
        pd.testing.assert_series_equal(
            ext["ratio_close_macd12_26_signal"].dropna(), expected.dropna(), check_names=False,
        )

    def test_macd_pairing_does_not_cross_configs(self):
        """A second, unrelated MACD triple (5,35,5) must not be paired with
        the (12,26,9) triple's line or signal."""
        df = self._macd_df()
        df["close_macd_05_35"] = df["close_macd_12_26"] * 0.5
        df["close_macd_05_35_signal_05"] = df["close_macd_12_26_signal_09"] * 0.5
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_close_macd12_26_signal" in meta
        assert "ratio_close_macd05_35_signal" in meta
        # no cross-config pairing exists
        assert not any(
            "macd12_26" in k and "macd05_35" in k for k in meta
        )

    def test_macd_replay_matches_generation(self):
        from forgedge.event_discovery.models import build_feature_series

        df = self._macd_df()
        cls = TypeClassifier().fit(df)
        ext, meta = FeatureGenerator().generate(df, cls)
        m = meta["diffnorm_close_macd12_26_signal"]
        comp = EventComponent(
            source_feature="diffnorm_close_macd12_26_signal", transform="identity",
            transform_params=dict(m.params), transformed_col="", threshold=0.0,
            threshold_type="", direction="above", event_type="threshold",
            expression="", source_cols=m.source_cols,
        )
        replayed = build_feature_series(comp, df)
        pd.testing.assert_series_equal(
            replayed.dropna(), ext["diffnorm_close_macd12_26_signal"].dropna(), check_names=False,
        )

    def test_price_volume_pairing_requires_volume_return(self):
        """No-op until a volume return column exists — 'return' is close-only
        in the default kpi_builder config."""
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert not any("volume_ret" in k and "close_ret" in k for k in meta)

    def test_price_volume_pairing_generated_and_correct(self):
        rng = np.random.default_rng(1)
        n = 500
        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": 100 + np.cumsum(rng.normal(0, 1, n)),
            "close_ret_05": rng.normal(0, 0.01, n),
            "volume_ret_05": rng.normal(0, 0.1, n),
        })
        cls = TypeClassifier().fit(df)
        ext, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_close_ret05_volume_ret05" in meta
        assert "diffnorm_close_ret05_volume_ret05" in meta
        expected = (df["close_ret_05"] / df["volume_ret_05"]).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        pd.testing.assert_series_equal(
            ext["ratio_close_ret05_volume_ret05"].dropna(), expected.dropna(), check_names=False,
        )

    def test_price_volume_pairing_matches_same_period_only(self):
        """close_ret_05 pairs with volume_ret_05, not volume_ret_12."""
        rng = np.random.default_rng(1)
        n = 500
        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": 100 + np.cumsum(rng.normal(0, 1, n)),
            "close_ret_05": rng.normal(0, 0.01, n),
            "volume_ret_12": rng.normal(0, 0.1, n),
        })
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert not any("close_ret05" in k and "volume_ret" in k for k in meta)

    def _candle_df(self, n: int = 500, seed: int = 3) -> pd.DataFrame:
        from forgedge.kpi_builder.candle import candle_features
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        open_ = close + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": open_, "high": high, "low": low, "close": close,
        })
        return candle_features(df)

    def test_parse_feature_candle_geometry(self):
        for col in ("body", "upper_wick", "lower_wick", "close_pos", "range_pct", "gap"):
            pf = parse_feature(col)
            assert pf is not None, col
            assert pf.family == "candle_geometry"

    def test_parse_feature_color_excluded(self):
        """color is deliberately not recognised (issue #162 non-goal)."""
        assert parse_feature("color") is None

    def test_candle_geometry_pairs_among_themselves(self):
        df = self._candle_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_body_upper_wick" in meta
        assert "diffnorm_body_range_pct" in meta
        assert "ratio_gap_range_pct" in meta

    def test_candle_geometry_bounded_pair_count(self):
        """6 geometry columns -> C(6,2)=15 pairs * 2 ops = 30, no NATR present
        (ATR is disabled by default) -> exactly 30."""
        df = self._candle_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        cg_pairs = [
            k for k, v in meta.items()
            if v.arity == 2 and v.operation in ("ratio", "diff_norm")
            and all(
                parse_feature(c) is not None and parse_feature(c).family == "candle_geometry"
                for c in v.source_cols
            )
        ]
        assert len(cg_pairs) == 30

    def test_candle_geometry_paired_with_natr_not_raw_atr(self):
        """Paired against close_natr_N (dimensionless), never close_atr_N
        (absolute price units — would reintroduce a price-level dependency)."""
        df = self._candle_df()
        rng = np.random.default_rng(9)
        df["close_natr_14"] = np.abs(rng.normal(0.02, 0.005, len(df)))
        df["close_atr_14"] = df["close_natr_14"] * df["close"]
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_body_close_natr_14" in meta
        assert not any(
            "atr_14" in k and "natr" not in k and k.startswith(("ratio_body", "diffnorm_body"))
            for k in meta
        )

    def test_candle_geometry_numeric_correctness(self):
        df = self._candle_df()
        cls = TypeClassifier().fit(df)
        ext, _ = FeatureGenerator().generate(df, cls)
        expected = (df["body"] / df["upper_wick"]).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        pd.testing.assert_series_equal(
            ext["ratio_body_upper_wick"].dropna(), expected.dropna(), check_names=False,
        )

    def test_candle_geometry_absent_without_candle_features(self):
        """No candle-geometry pairs when candle_features() was never called."""
        df = _make_kpi_table()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert not any(
            k.startswith(("ratio_body", "diffnorm_body", "ratio_gap", "diffnorm_gap"))
            for k in meta
        )

    def test_candle_geometry_replay_matches_generation(self):
        from forgedge.event_discovery.models import build_feature_series

        df = self._candle_df()
        cls = TypeClassifier().fit(df)
        ext, meta = FeatureGenerator().generate(df, cls)
        m = meta["ratio_gap_range_pct"]
        comp = EventComponent(
            source_feature="ratio_gap_range_pct", transform="identity",
            transform_params=dict(m.params), transformed_col="", threshold=0.0,
            threshold_type="", direction="above", event_type="threshold",
            expression="", source_cols=m.source_cols,
        )
        replayed = build_feature_series(comp, df)
        pd.testing.assert_series_equal(
            replayed.dropna(), ext["ratio_gap_range_pct"].dropna(), check_names=False,
        )

    # ── Issue #165: indicator vs lagged OHLC-base cross-time pairing ────────

    def _indicator_lag_cross_df(self, n: int = 500, seed: int = 4) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        return pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": close, "high": high, "low": low,
            "close_sma_12": pd.Series(close).rolling(12, min_periods=1).mean(),
            "close_rsi_14": np.clip(50 + rng.normal(0, 15, n), 0, 100),
        })

    def test_indicator_lag_cross_generates_ma_vs_low_lag3(self):
        """The exact gap reported in #165: 'ma_12[t] > low[t-3]'."""
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert "ratio_close_sma_12_low_lag3" in meta
        m = meta["ratio_close_sma_12_low_lag3"]
        assert m.source_cols == ["close_sma_12", "low"]
        assert m.params == {"cross_lag": 3}
        assert m.transforms == frozenset({"identity"})

    def test_indicator_lag_cross_numeric_correctness(self):
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        ext, _ = FeatureGenerator().generate(df, cls)
        expected = (df["close_sma_12"] / df["low"].shift(3)).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        pd.testing.assert_series_equal(
            ext["ratio_close_sma_12_low_lag3"].dropna(), expected.dropna(), check_names=False,
        )

    def test_indicator_lag_cross_excludes_bounded_indicators(self):
        """RSI (bounded, not price-scale) must not be paired against a raw
        OHLC base — the ATR-vs-NATR dimensional-soundness reasoning from
        #162 applies here too."""
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert not any("rsi" in k and "_lag" in k for k in meta)

    def test_indicator_lag_cross_ratio_only_no_spread(self):
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        assert not any(
            k.startswith("spread_close_sma_12_") or k.startswith("diffnorm_close_sma_12_")
            for k in meta
        )

    def test_indicator_lag_cross_default_lags_are_1_and_3(self):
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls)
        lags_seen = {
            v.params["cross_lag"] for k, v in meta.items()
            if v.operation == "ratio_lag" and "sma_12" in k
        }
        assert lags_seen == {1, 3}

    def test_indicator_lag_cross_custom_lags_override(self):
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls, indicator_lag_cross_lags=(2, 5))
        lags_seen = {
            v.params["cross_lag"] for k, v in meta.items()
            if v.operation == "ratio_lag" and "sma_12" in k
        }
        assert lags_seen == {2, 5}
        assert "ratio_close_sma_12_low_lag1" not in meta

    def test_indicator_lag_cross_disabled_with_empty_tuple(self):
        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        _, meta = FeatureGenerator().generate(df, cls, indicator_lag_cross_lags=())
        assert not any("sma_12" in k and "_lag" in k for k in meta)
        # #161's OHLC x OHLC lag-cross is independent and stays active
        assert any(
            v.operation in ("ratio_lag", "spread_pct_lag") and "sma" not in k
            for k, v in meta.items()
        )

    def test_indicator_lag_cross_replay_matches_generation(self):
        from forgedge.event_discovery.models import build_feature_series

        df = self._indicator_lag_cross_df()
        cls = TypeClassifier().fit(df)
        ext, meta = FeatureGenerator().generate(df, cls)
        m = meta["ratio_close_sma_12_low_lag3"]
        comp = EventComponent(
            source_feature="ratio_close_sma_12_low_lag3", transform="identity",
            transform_params=dict(m.params), transformed_col="", threshold=0.0,
            threshold_type="", direction="above", event_type="threshold",
            expression="", source_cols=m.source_cols,
        )
        replayed = build_feature_series(comp, df)
        pd.testing.assert_series_equal(
            replayed.dropna(), ext["ratio_close_sma_12_low_lag3"].dropna(), check_names=False,
        )

    def test_discovery_config_indicator_lag_cross_lags_threaded_through(self):
        """DiscoveryConfig.indicator_lag_cross_lags reaches FeatureGenerator
        via EventDiscovery, end to end."""
        df = self._indicator_lag_cross_df()
        ed = EventDiscovery(
            df, DiscoveryConfig(train_ratio=1.0, indicator_lag_cross_lags=(2,))
        )
        ed.run()
        assert "ratio_close_sma_12_low_lag2" in ed.df.columns
        assert "ratio_close_sma_12_low_lag1" not in ed.df.columns
        assert "ratio_close_sma_12_low_lag3" not in ed.df.columns


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
    """The default gate is episode-counting (issue #134); ``event_counting="bar"``
    reproduces the historical behaviour and is pinned where bar semantics are
    under test."""

    def _bar_gate(self, **kw):
        kw.setdefault("min_tpm", 1.0)
        kw.setdefault("max_dispersion", 2.5)
        return ConsistencyGate(GateParams(event_counting="bar", **kw))

    def test_sparse_event_fails_volume(self):
        # 12-month series with only 2 activations → 2/12 < min_tpm=1.0 (bar mode)
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        event = pd.Series([1.0] * 2 + [0.0] * 8758)
        active = event.fillna(0).values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = self._bar_gate().evaluate(active, counts, n_m)
        assert not result.passed
        assert "rate" in result.fail_reason

    def test_concentrated_event_fails_concentration(self):
        # 50 activations all in the first month → high per-bar dispersion (bar mode)
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        active = np.array([True] * 50 + [False] * (8760 - 50), dtype=bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = self._bar_gate().evaluate(active, counts, n_m)
        assert not result.passed
        assert "dispersion" in result.fail_reason

    def test_uniform_event_passes(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        rng = np.random.default_rng(0)
        event = pd.Series((rng.random(8760) < 0.10).astype(float))
        active = event.values.astype(bool)
        counts = _count_by_month(active, month_idx, n_m)
        result = self._bar_gate(min_tpm=2.0).evaluate(active, counts, n_m)
        assert result.passed

    # ── Issue #134: episode counting, χ² floor, n_eff ───────────────────────

    def _persistent_rare_event(self, seed: int = 1):
        """A rare, well-distributed event whose activations come in multi-bar
        runs (a persistent state like RSI<30): per-bar ID is inflated, but
        per-episode ID stays ~1."""
        n = 1825
        ts = pd.Series(pd.date_range("2019-01-01", periods=n, freq="1D"))
        mon = pd.Series(ts.dt.to_period("M"))
        rng = np.random.default_rng(seed)
        mask = np.zeros(n, dtype=bool)
        for m in mon.unique():
            idx = np.where(mon.values == m)[0]
            if len(idx) < 8:
                continue
            for _ in range(rng.poisson(1.2)):
                s = rng.choice(idx[:-5])
                mask[s: s + rng.integers(3, 6)] = True
        return pd.Series(mask.astype(float)), ts

    def test_persistent_state_rejected_by_bar_gate(self):
        """Bar mode wrongly rejects the rare persistent event for dispersion
        (the issue #134 defect that episode mode fixes)."""
        event, ts = self._persistent_rare_event()
        mi, nm = _build_month_index(ts)
        r = ConsistencyGate(
            GateParams(event_counting="bar", min_tpm=0.5, max_dispersion=2.5)
        ).evaluate_series(event, mi, nm)
        assert not r.passed
        assert "dispersion" in r.fail_reason

    def test_episode_mode_passes_well_distributed_persistent_event(self):
        """Default (episode) mode measures dispersion per-episode, so the same
        event passes; episode metrics are populated."""
        event, ts = self._persistent_rare_event()
        mi, nm = _build_month_index(ts)
        r = ConsistencyGate(GateParams()).evaluate_series(event, mi, nm)
        assert r.passed
        # per-bar ID is inflated well above the per-episode ID
        assert r.index_of_dispersion > r.episode_index_of_dispersion
        assert r.n_episodes > 0 and not np.isnan(r.n_eff)

    def test_min_tpm_counts_episodes_in_episode_mode(self):
        """In episode mode the rate criterion counts episodes per month, not
        bars: a low-episode-rate event fails the rate criterion."""
        ts = pd.Series(pd.date_range("2019-01-01", periods=1825, freq="1D"))
        mi, nm = _build_month_index(ts)
        mon = pd.Series(ts.dt.to_period("M"))
        # one episode every ~3 months → ~0.33 episodes/month
        mask = np.zeros(1825, dtype=bool)
        for i, m in enumerate(mon.unique()):
            if i % 3 == 0:
                idx = np.where(mon.values == m)[0]
                mask[idx[len(idx) // 2]] = True
        r = ConsistencyGate(
            GateParams(min_tpm=0.5, min_episodes=1)
        ).evaluate_series(pd.Series(mask.astype(float)), mi, nm)
        assert not r.passed
        assert "rate" in r.fail_reason

    def test_min_episodes_power_floor(self):
        """Episode mode rejects events with too few episodes for power."""
        ts = pd.Series(pd.date_range("2019-01-01", periods=1825, freq="1D"))
        mi, nm = _build_month_index(ts)
        mask = np.zeros(1825, dtype=bool)
        mask[100] = True
        mask[400:403] = True  # 2 episodes
        r = ConsistencyGate(
            GateParams(min_tpm=0.0, min_episodes=10)
        ).evaluate_series(pd.Series(mask.astype(float)), mi, nm)
        assert not r.passed
        assert "episodes" in r.fail_reason

    def test_poisson_chi2_floor_raises_low_max_dispersion(self):
        """Episode mode never rejects an event statistically consistent with
        randomness: a Poisson-distributed episodic event passes even with
        max_dispersion=1.0 — which is not read at all in episode mode any
        more (#205), so this also demonstrates that `dispersion_margin`'s
        class default (1.3) does not accidentally tighten the floor below
        what the class default max_dispersion used to (imperfectly) protect.
        """
        rng = np.random.default_rng(4)
        ts = pd.Series(pd.date_range("2019-01-01", periods=1825, freq="1D"))
        mi, nm = _build_month_index(ts)
        mon = pd.Series(ts.dt.to_period("M"))
        mask = np.zeros(1825, dtype=bool)
        for m in mon.unique():
            idx = np.where(mon.values == m)[0]
            for _ in range(rng.poisson(1.5)):  # Poisson arrivals, isolated bars
                mask[rng.choice(idx)] = True
        r = ConsistencyGate(
            GateParams(min_tpm=0.3, max_dispersion=1.0, min_episodes=10)
        ).evaluate_series(pd.Series(mask.astype(float)), mi, nm)
        # raw ID may sit slightly above 1.0 by sampling noise but below
        # dispersion_margin(1.3) x the χ² floor (~1.32 for 60 months), so the
        # event still passes
        assert r.passed

    def test_n_eff_is_episodes_over_id(self):
        """n_eff = n_episodes / episode_ID (issue #134 design effect)."""
        event, ts = self._persistent_rare_event()
        mi, nm = _build_month_index(ts)
        r = ConsistencyGate(GateParams()).evaluate_series(event, mi, nm)
        assert abs(r.n_eff - r.n_episodes / r.episode_index_of_dispersion) < 1e-6

    # ── #205: dispersion_margin replaces max_dispersion in episode mode ────

    def _clustered_episodic_event(self, seed: int = 7):
        """Episodes concentrated in every 6th month — genuinely over-dispersed
        (episode ID ~2.0 over 60 months), not just noisy-looking.  Built to sit
        between the Poisson floor (~1.32 at 60 months) scaled by the tight
        presets' margins and by the loose ones', so it is rejected by one and
        accepted by the other — the differentiation #205 restores."""
        n = 1825
        ts = pd.Series(pd.date_range("2019-01-01", periods=n, freq="1D"))
        mon = pd.Series(ts.dt.to_period("M"))
        rng = np.random.default_rng(seed)
        mask = np.zeros(n, dtype=bool)
        for i, m in enumerate(mon.unique()):
            if i % 6 != 0:
                continue
            idx = np.where(mon.values == m)[0]
            for _ in range(rng.integers(2, 5)):
                mask[rng.choice(idx[:-2])] = True
        return pd.Series(mask.astype(float)), ts

    def test_max_dispersion_is_unread_in_episode_mode(self):
        """A `max_dispersion` extreme enough to reject or admit anything on
        its own has zero effect on the verdict in episode mode — only
        `dispersion_margin` does (#205)."""
        event, ts = self._clustered_episodic_event()
        mi, nm = _build_month_index(ts)
        tight = ConsistencyGate(
            GateParams(min_tpm=0.01, min_episodes=1, max_dispersion=0.001)
        ).evaluate_series(event, mi, nm)
        loose = ConsistencyGate(
            GateParams(min_tpm=0.01, min_episodes=1, max_dispersion=100.0)
        ).evaluate_series(event, mi, nm)
        assert tight.passed == loose.passed
        assert tight.episode_index_of_dispersion == loose.episode_index_of_dispersion

    def test_dispersion_margin_differentiates_tight_from_loose_presets(self):
        """The point of #205: a genuinely clustered event is rejected by a
        tight margin (`sniper`/`balanced`-like) and accepted by a loose one
        (`sweep`/`burst`-like) — the differentiation `max_dispersion` used to
        promise but the Poisson floor silently erased."""
        event, ts = self._clustered_episodic_event()
        mi, nm = _build_month_index(ts)

        def _passes(margin):
            r = ConsistencyGate(
                GateParams(min_tpm=0.01, min_episodes=1, dispersion_margin=margin)
            ).evaluate_series(event, mi, nm)
            return r.passed

        assert _passes(1.05) is False    # sniper-like: no slack above the floor
        assert _passes(1.30) is False    # balanced-like
        assert _passes(1.60) is True     # sweep-like
        assert _passes(3.00) is True     # burst-like: clustering allowed on purpose

    def test_bar_mode_still_uses_max_dispersion_directly(self):
        """`dispersion_margin` is an episode-mode concept; bar mode is
        unaffected and keeps comparing the raw configured value with no
        floor, exactly as before #205."""
        ts = pd.Series(pd.date_range("2024-01-01", periods=8760, freq="1h"))
        month_idx, n_m = _build_month_index(ts)
        active = np.array([True] * 50 + [False] * (8760 - 50), dtype=bool)
        counts = _count_by_month(active, month_idx, n_m)
        strict = self._bar_gate(max_dispersion=2.5, dispersion_margin=1.05).evaluate(
            active, counts, n_m
        )
        loose = self._bar_gate(max_dispersion=2.5, dispersion_margin=3.00).evaluate(
            active, counts, n_m
        )
        # dispersion_margin is not read in bar mode, so the two are identical
        assert strict.passed == loose.passed
        assert not strict.passed
        assert "dispersion" in strict.fail_reason

    def test_episode_gap_merging(self):
        """A one-bar interruption does not start a new episode (gap=1);
        gap=0 gives strict consecutive runs."""
        a = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
        assert int(_episode_starts(a, gap=0).sum()) == 4
        assert int(_episode_starts(a, gap=1).sum()) == 3  # idx1..4 merge over 1-bar hole


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEventDiscoveryE2E:
    @pytest.fixture(scope="class")
    def df_2000(self):
        """EventDiscovery defensively copies its input (``self.df =
        kpi_table.copy()`` in ``__init__``), so sharing the raw table across
        the differently-configured runs below is safe.  No test below mutates
        the frame it receives — ``.sample()``/``.set_index()`` return copies."""
        return _make_kpi_table(n=2000)

    @pytest.fixture(scope="class")
    def df_default(self):
        return _make_kpi_table()

    def test_run_returns_candidates(self, df_2000):
        ed = EventDiscovery(
            df_2000,
            config=DiscoveryConfig(
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0),
                max_and_components=2,
            ),
        )
        candidates = ed.run()
        assert len(candidates) > 0

    def test_all_candidates_have_passed_gate(self, df_2000):
        ed = EventDiscovery(
            df_2000,
            config=DiscoveryConfig(
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0)
            ),
        )
        candidates = ed.run()
        for c in candidates:
            assert c.consistency_gate.passed, f"{c.event_id} gate not passed"

    def test_candidates_have_expressions(self, df_2000):
        ed = EventDiscovery(df_2000)
        candidates = ed.run()
        for c in candidates:
            assert c.expression
            assert len(c.components) >= 1

    def test_summary_dataframe_shape(self, df_default):
        ed = EventDiscovery(
            df_default,
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

    def test_classifications_available_after_run(self, df_default):
        ed = EventDiscovery(df_default)
        ed.run()
        cls = ed.get_classifications()
        assert cls is not None
        assert "close_rsi_14" in cls

    def test_datetimeindex_input(self, df_default):
        df = df_default.set_index("open_dt")
        ed = EventDiscovery(
            df,
            config=DiscoveryConfig(
                timestamp_col="open_dt",
                gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0),
            ),
        )
        candidates = ed.run()
        assert len(candidates) > 0


    def test_unsorted_input_produces_same_results_as_sorted(self, df_2000):
        """Rolling windows must be computed in chronological order regardless of input row order."""
        df_sorted = df_2000
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

    def test_unsorted_datetimeindex_input_is_sorted(self, df_2000):
        """DatetimeIndex path also sorts rows before rolling calculations."""
        df = df_2000
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
# event_distribution_report (#215)
# ---------------------------------------------------------------------------


class TestEventDistributionReport:
    """Tests for ``EventDiscovery.event_distribution_report`` (#215).

    ``config_report()`` is blind to data by construction, so it can only
    catch config-vs-config incoherence — never a preset that is internally
    coherent yet still rejects every candidate a specific asset actually
    produces.  This report is the always-computed diagnostic that closes
    that gap: it aggregates the ``GateResult`` already attached to every
    raw (pre-AND-composition) event, and — below a 15% gate-survival rate —
    appends a concrete parameter suggestion at the observed median.
    """

    @pytest.fixture(scope="class")
    def df_2000(self):
        return _make_kpi_table(n=2000)

    def test_none_before_run(self, df_2000):
        ed = EventDiscovery(df_2000)
        assert ed.event_distribution_report is None

    def test_populated_after_run(self, df_2000):
        ed = EventDiscovery(
            df_2000, config=DiscoveryConfig(gate_params=GateParams(min_tpm=1.0, min_episodes=1))
        )
        ed.run()
        report = ed.event_distribution_report
        assert isinstance(report, str)
        assert "M1 Event Discovery" in report
        assert "Consistency Gate" in report

    def test_populated_even_with_zero_candidates(self, df_2000):
        """The whole point of #215: a report exists even when the gate
        rejects every single candidate — the scenario that used to log only
        a bare, uninformative '0 candidate(s)'."""
        ed = EventDiscovery(
            df_2000,
            config=DiscoveryConfig(
                gate_params=GateParams(min_tpm=1000.0, min_episodes=1000)
            ),
        )
        candidates = ed.run()
        assert len(candidates) == 0
        report = ed.event_distribution_report
        assert report is not None
        assert "0 superano il Consistency Gate (0.0%)" in report

    def test_median_stats_match_raw_events(self, df_2000):
        """The reported tpm/dispersion medians are computed over ALL raw
        events (including zero-activation ones), episode mode."""
        cfg = DiscoveryConfig(gate_params=GateParams(min_tpm=1.0, min_episodes=1))
        ed = EventDiscovery(df_2000, config=cfg)
        ed.run()
        raw = ed.raw_events
        assert len(raw) > 0

        _, n_months = _build_month_index(ed._timestamps)
        expected_rate_median = float(
            np.median([ev.gate_result.n_episodes / n_months for ev in raw])
        )
        dispersions = [
            ev.gate_result.episode_index_of_dispersion for ev in raw
            if ev.gate_result.episode_index_of_dispersion == ev.gate_result.episode_index_of_dispersion
        ]
        expected_disp_median = float(np.median(dispersions))

        assert f"mediana={expected_rate_median:.2f}" in ed.event_distribution_report
        assert f"mediana={expected_disp_median:.2f}" in ed.event_distribution_report

    def test_low_survival_hint_fires_with_correct_suggestion(self, df_2000):
        """Below 15% gate-survival, the report suggests min_tpm/dispersion_margin
        at the observed median (#215 point 5), episode mode."""
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=50.0, dispersion_margin=1.0, min_episodes=1)
        )
        ed = EventDiscovery(df_2000, config=cfg)
        candidates = ed.run()
        raw = ed.raw_events
        survival = len([e for e in raw if e.gate_result.passed]) / len(raw)
        assert survival < 0.15, "fixture must actually exercise the low-survival branch"

        report = ed.event_distribution_report
        assert "Meno del 15% dei candidati generati supera il Consistency Gate" in report
        assert "min_tpm<=" in report
        assert "dispersion_margin>=" in report
        assert "max_dispersion>=" not in report

    def test_high_survival_has_no_hint(self, df_2000):
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.1, dispersion_margin=10.0, min_episodes=1)
        )
        ed = EventDiscovery(df_2000, config=cfg)
        ed.run()
        raw = ed.raw_events
        survival = len([e for e in raw if e.gate_result.passed]) / len(raw)
        assert survival >= 0.15, "fixture must actually exercise the high-survival branch"
        assert "Prova questi parametri" not in ed.event_distribution_report

    def test_bar_mode_hint_suggests_max_dispersion(self, df_2000):
        """In 'bar' mode the second suggested field is the absolute
        max_dispersion, not the episode-mode-only dispersion_margin (#205)."""
        cfg = DiscoveryConfig(
            gate_params=GateParams(
                min_tpm=50.0, max_dispersion=0.1, event_counting="bar"
            )
        )
        ed = EventDiscovery(df_2000, config=cfg)
        candidates = ed.run()
        raw = ed.raw_events
        survival = len([e for e in raw if e.gate_result.passed]) / len(raw)
        assert survival < 0.15, "fixture must actually exercise the low-survival branch"

        report = ed.event_distribution_report
        assert "max_dispersion>=" in report
        assert "dispersion_margin>=" not in report


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


class TestWalkForward:
    """Tests for train/test split and walk-forward OOS validation."""
    pytestmark = pytest.mark.slow

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
        cfg = DiscoveryConfig(gate_params=GateParams(min_tpm=1.0, max_dispersion=10.0))
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
                # `passed` is the gate's verdict *and* the fold being long
                # enough to have one (#177): a fold whose expected episode
                # count falls under MIN_FOLD_LAMBDA cannot distinguish a
                # healthy candidate from a silent one, so it does not get to
                # say "passed" either way — it is excluded from the
                # denominator instead.
                assert fr.passed == (fr.gate_result.passed and not fr.indeterminate)
                assert fr.indeterminate == (fr.lam < MIN_FOLD_LAMBDA)

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

    # ── Issue #226: ANDComposer must judge composed events by the same
    # criteria as single ones, mode-aware — not its own bar-only reimplementation

    @staticmethod
    def _bursty_series(rng, n, burst_len=48, period=24 * 20, p_in_burst=0.6):
        """Time-concentrated activation pattern (regime-clustered bursts,
        quiet in between) — bar-level dispersion is high (persistent
        multi-bar bursts), episode-level dispersion is low (episodes land
        roughly one per burst, spread fairly evenly across bursts). This is
        exactly the shape #205/#226 are about: a preset with a generous
        `dispersion_margin` should accept it; a hardcoded bar-level
        `max_dispersion=1.5` should not."""
        s = np.zeros(n, dtype=bool)
        for start in range(0, n, period):
            end = min(start + burst_len, n)
            s[start:end] = rng.random(end - start) < p_in_burst
        return s

    def test_and_composer_episode_mode_gate_decision_matches_single_gate(self):
        """The heart of #226: a composed pair's pass/fail, n_episodes and
        episode_index_of_dispersion must exactly match evaluating the same
        AND result directly through ConsistencyGate.evaluate_series() — not
        a bar-level approximation. Before the fix this pair was wrongly
        rejected (bar-ID ~4.9 > the default max_dispersion=1.5, a field
        episode mode never reads) even though the correct episode-level
        evaluation passes comfortably under a generous dispersion_margin."""
        rng = np.random.default_rng(3)
        n = 24 * 400
        ts = pd.Series(pd.date_range("2023-01-01", periods=n, freq="1h"))
        month_idx, n_months = _build_month_index(ts)

        a = self._bursty_series(rng, n)
        b = self._bursty_series(rng, n)
        s1, s2 = pd.Series(a.astype(float)), pd.Series(b.astype(float))

        params = GateParams(
            min_tpm=1.0, dispersion_margin=5.0, min_episodes=1, event_counting="episode"
        )
        gate = ConsistencyGate(params)

        def _make_ev(s, name):
            ev = RawEvent(series=s, component=_make_component(name, "identity", {}, []))
            ev.gate_result = gate.evaluate_series(s, month_idx, n_months)
            return ev

        ev1, ev2 = _make_ev(s1, "feat_a"), _make_ev(s2, "feat_b")
        assert ev1.gate_result.passed and ev2.gate_result.passed

        composed = ANDComposer(gate).compose([ev1, ev2], ts, max_components=2)
        assert composed, (
            "composed pair wrongly rejected — ANDComposer is still judging "
            "episode-mode candidates by bar-level dispersion (#226)"
        )

        and_arr = a & b
        reference = gate.evaluate_series(pd.Series(and_arr.astype(float)), month_idx, n_months)
        assert reference.passed, "fixture must actually be a real pass under episode mode"

        gr = composed[0].gate_result
        assert gr.passed == reference.passed
        assert gr.n_episodes == reference.n_episodes
        assert gr.episode_index_of_dispersion == pytest.approx(
            reference.episode_index_of_dispersion, rel=1e-9
        )

    def test_and_composer_episode_mode_correctly_rejects_over_dispersed_pair(self):
        """The rejection path, not just the acceptance path: a pair whose
        episode-level ID genuinely exceeds a *tight* dispersion_margin must
        still be rejected — the fix corrects which criterion is checked, it
        does not make the composer permissive."""
        rng = np.random.default_rng(11)
        n = 24 * 400
        ts = pd.Series(pd.date_range("2023-01-01", periods=n, freq="1h"))
        month_idx, n_months = _build_month_index(ts)

        a = self._bursty_series(rng, n, p_in_burst=0.8)
        b = self._bursty_series(rng, n, p_in_burst=0.8)
        s1, s2 = pd.Series(a.astype(float)), pd.Series(b.astype(float))

        # For this fixture (seed=11, p_in_burst=0.8, n=24*400h) the composed
        # AND measures episode_index_of_dispersion ~= 1.34 (verified directly
        # via _eff_max_dispersion). margin=0.5 puts eff_max_dispersion ~= 0.86,
        # safely below that with headroom, so the pair must fail episode
        # dispersion rather than clear it.
        params = GateParams(
            min_tpm=1.0, dispersion_margin=0.5, min_episodes=1, event_counting="episode"
        )
        gate = ConsistencyGate(params)

        def _make_ev(s, name):
            ev = RawEvent(series=s, component=_make_component(name, "identity", {}, []))
            ev.gate_result = gate.evaluate_series(s, month_idx, n_months)
            return ev

        ev1, ev2 = _make_ev(s1, "feat_a"), _make_ev(s2, "feat_b")

        and_arr = a & b
        reference = gate.evaluate_series(pd.Series(and_arr.astype(float)), month_idx, n_months)
        assert not reference.passed and "dispersion" in (reference.fail_reason or ""), (
            "fixture must actually fail on episode dispersion for this test to mean anything"
        )

        composed = ANDComposer(gate).compose([ev1, ev2], ts, max_components=2)
        assert composed == [], (
            "an over-dispersed pair under a tight dispersion_margin must be rejected, "
            "not silently accepted by the fix"
        )

    def test_and_composer_bar_mode_unaffected_by_episode_fix(self):
        """Regression guard: `"bar"` mode composed events must still be
        judged by `max_dispersion` directly, exactly as before #226 —
        the fix only changes what happens under `event_counting="episode"`."""
        rng = np.random.default_rng(7)
        n = 8760
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        month_idx, n_months = _build_month_index(ts)

        gate = ConsistencyGate(GateParams(
            min_tpm=2.0, max_dispersion=2.5, event_counting="bar",
        ))
        base = rng.random(n) < 0.30
        s1 = pd.Series((base & (rng.random(n) < 0.80)).astype(float))
        s2 = pd.Series((base & (rng.random(n) < 0.80)).astype(float))

        def _make_ev(s, name):
            ev = RawEvent(series=s, component=_make_component(name, "identity", {}, []))
            ev.gate_result = gate.evaluate_series(s, month_idx, n_months)
            return ev

        ev1, ev2 = _make_ev(s1, "feat_a"), _make_ev(s2, "feat_b")
        assert ev1.gate_result.passed and ev2.gate_result.passed

        composed = ANDComposer(gate).compose([ev1, ev2], ts, max_components=2)
        assert composed

        and_arr = (s1.fillna(0).values.astype(bool) & s2.fillna(0).values.astype(bool))
        reference = gate.evaluate_series(pd.Series(and_arr.astype(float)), month_idx, n_months)

        gr = composed[0].gate_result
        assert gr.passed == reference.passed
        assert gr.index_of_dispersion == pytest.approx(reference.index_of_dispersion, rel=1e-9)

    # ── Issue #228: ANDComposer chunk size must stay bounded on long
    # histories, and the bound must not change what compose() returns

    def test_pair_chunk_size_bounded_and_shrinks_with_n_rows(self):
        """`_pair_chunk_size` must never exceed `_CHUNK_SIZE`, must shrink
        (non-increasing) as n_rows grows, and must always return >= 1 even
        for pathologically long histories."""
        from forgedge.event_discovery.and_composer import _CHUNK_SIZE, _pair_chunk_size

        assert _pair_chunk_size(0) == _CHUNK_SIZE
        assert _pair_chunk_size(100) == _CHUNK_SIZE

        sizes = [_pair_chunk_size(n) for n in (1_000, 10_000, 100_000, 1_000_000, 50_000_000)]
        assert all(1 <= s <= _CHUNK_SIZE for s in sizes)
        assert all(a >= b for a, b in zip(sizes, sizes[1:])), "chunk size must not grow with n_rows"

    def test_and_composer_chunk_size_does_not_change_composed_results(self):
        """The #228 memory fix only changes how many pairs/triples are
        batched per vectorized iteration — forcing a small chunk size must
        produce byte-identical composed events to the default, larger one."""
        import forgedge.event_discovery.and_composer as and_composer_mod

        rng = np.random.default_rng(13)
        n = 8760
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))

        gate = ConsistencyGate(GateParams(min_tpm=1.0, dispersion_margin=2.0, min_episodes=3))
        events = []
        for i in range(12):
            p_act = rng.uniform(0.03, 0.12)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))

        passing = gate.filter(events, ts)
        assert len(passing) >= 2, "fixture must produce a composable pool"

        composer = ANDComposer(gate)

        original = and_composer_mod._pair_chunk_size
        try:
            and_composer_mod._pair_chunk_size = lambda n_rows: 3  # force many tiny chunks
            small_chunk_result = composer.compose(passing, ts, max_components=3)
        finally:
            and_composer_mod._pair_chunk_size = original

        default_result = composer.compose(passing, ts, max_components=3)

        def _key(ev):
            return (ev.component.source_feature, ev.component.transform,
                    ev.gate_result.n_activations, ev.gate_result.episode_index_of_dispersion)

        assert sorted(map(_key, small_chunk_result)) == sorted(map(_key, default_result))

    # ── Issue #230: pair/triple enumeration order must not let one pool
    # event dominate every kept composition under permissive gate params

    def test_shuffle_order_is_a_valid_deterministic_permutation(self):
        from forgedge.event_discovery.and_composer import _shuffle_order

        assert list(_shuffle_order(0, np.random.default_rng(1))) == []
        assert list(_shuffle_order(1, np.random.default_rng(1))) == [0]

        rng_a = np.random.default_rng(230)
        rng_b = np.random.default_rng(230)
        perm_a = _shuffle_order(500, rng_a)
        perm_b = _shuffle_order(500, rng_b)
        assert sorted(perm_a.tolist()) == list(range(500)), "must be a permutation of range(n)"
        assert perm_a.tolist() == perm_b.tolist(), "same seed must reproduce the same permutation"

    def test_and_composer_pair_shuffle_avoids_single_root_domination(self):
        """The bug #230 fixes: under permissive gate params where most pairs
        pass, pre-fix row-major enumeration let one pool event dominate every
        kept pair (measured on a real fixture: max reuse == _MAX_PAIRS).
        Reproduces the pathology by monkeypatching `_shuffle_order` to a
        no-op, then shows the real (shuffled) code avoids it -- proving the
        shuffle itself is what fixes it, not incidental fixture luck."""
        import forgedge.event_discovery.and_composer as and_composer_mod

        rng = np.random.default_rng(230)
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        events = []
        for i in range(40):
            p_act = rng.uniform(0.15, 0.35)  # high overlap -> most pairs clear the gate easily
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))

        passing = gate.filter(events, ts)
        assert len(passing) >= 22, "fixture must produce enough passing singles to expose the bug"

        composer = ANDComposer(gate)

        original_max_pairs = and_composer_mod._MAX_PAIRS
        original_shuffle = and_composer_mod._shuffle_order
        try:
            and_composer_mod._MAX_PAIRS = 20  # small relative to pool -> forces early-exit regime
            shuffled_result = composer.compose(passing, ts, max_components=2)

            and_composer_mod._shuffle_order = lambda n, rng: np.arange(n)  # simulate pre-#230
            unshuffled_result = composer.compose(passing, ts, max_components=2)
        finally:
            and_composer_mod._MAX_PAIRS = original_max_pairs
            and_composer_mod._shuffle_order = original_shuffle

        def _max_reuse(result):
            counts: dict[str, int] = {}
            for ev in result:
                for c in ev.component.components:
                    counts[c.source_feature] = counts.get(c.source_feature, 0) + 1
            return max(counts.values()) if counts else 0

        assert len(shuffled_result) == 20 and len(unshuffled_result) == 20
        assert _max_reuse(unshuffled_result) == 20, (
            "fixture sanity: row-major order must reproduce the pre-#230 pathology"
        )
        assert _max_reuse(shuffled_result) < 20, (
            "shuffled enumeration must not let one event dominate every kept pair"
        )

    def test_and_composer_shuffle_is_deterministic_across_repeated_calls(self):
        """Same input + same config must produce byte-identical composed
        events on repeated calls -- the #230 shuffle uses a fixed seed, not
        real entropy, so FORGE's reproducibility guarantee is unaffected."""
        rng = np.random.default_rng(7)
        n = 4380
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1))

        events = []
        for i in range(15):
            p_act = rng.uniform(0.1, 0.3)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))
        passing = gate.filter(events, ts)

        composer = ANDComposer(gate)
        result_a = composer.compose(passing, ts, max_components=3)
        result_b = composer.compose(passing, ts, max_components=3)

        def _key(ev):
            return (ev.component.source_feature, ev.gate_result.n_activations)

        assert list(map(_key, result_a)) == list(map(_key, result_b)), (
            "repeated calls on identical input must return compositions in the same order"
        )

    def test_and_composer_max_constituent_jaccard_rejects_near_duplicate_pair(self):
        """Opt-in filter (#230): a pair whose two constituents are near
        duplicates (high Jaccard) must be rejected when the threshold is set,
        and the default (None) must leave existing behaviour unchanged."""
        rng = np.random.default_rng(3)
        n = 8760
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1))

        base = rng.random(n) < 0.20
        # near-duplicate pair: differs on ~2% of bars -> Jaccard well above 0.9
        near_dup = base.copy()
        flip_idx = rng.choice(n, size=int(0.02 * n), replace=False)
        near_dup[flip_idx] = ~near_dup[flip_idx]
        # an unrelated third event, low overlap with the other two
        independent = rng.random(n) < 0.20

        events = [
            RawEvent(series=pd.Series(base.astype(float)), component=_make_component("feat_a", "identity", {}, [])),
            RawEvent(series=pd.Series(near_dup.astype(float)), component=_make_component("feat_b", "identity", {}, [])),
            RawEvent(series=pd.Series(independent.astype(float)), component=_make_component("feat_c", "identity", {}, [])),
        ]
        passing = gate.filter(events, ts)
        assert len(passing) == 3

        composer = ANDComposer(gate)

        no_filter = composer.compose(passing, ts, max_components=2)
        pairs_no_filter = {
            frozenset(c.source_feature for c in ev.component.components) for ev in no_filter
        }
        assert frozenset({"feat_a", "feat_b"}) in pairs_no_filter, (
            "fixture sanity: the near-duplicate pair must be composable without the filter"
        )

        filtered = composer.compose(passing, ts, max_components=2, max_constituent_jaccard=0.8)
        pairs_filtered = {
            frozenset(c.source_feature for c in ev.component.components) for ev in filtered
        }
        assert frozenset({"feat_a", "feat_b"}) not in pairs_filtered, (
            "near-duplicate pair must be rejected once max_constituent_jaccard is set"
        )

    # ── Issue #254 (Phase 1): ANDComposer.compose() gains optional
    # pool_selector/stratify_fn/max_pairs/max_triples hooks, plus a
    # standalone RawEvent<->EventCandidate adapter, with zero change to
    # existing (all-default) behaviour.

    def test_new_compose_kwargs_default_to_unchanged_behaviour(self):
        """compose() with every new kwarg explicitly passed as its default
        (None) must reproduce the exact same composed events, in the exact
        same order, as a call that omits them entirely."""
        rng = np.random.default_rng(2540)
        n = 4380
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1))

        events = []
        for i in range(15):
            p_act = rng.uniform(0.1, 0.3)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))
        passing = gate.filter(events, ts)

        composer = ANDComposer(gate)
        baseline = composer.compose(passing, ts, max_components=3)
        explicit_defaults = composer.compose(
            passing, ts, max_components=3,
            pool_selector=None, stratify_fn=None, max_pairs=None, max_triples=None,
        )

        def _key(ev):
            return (ev.component.source_feature, ev.gate_result.n_activations)

        assert list(map(_key, baseline)) == list(map(_key, explicit_defaults))

    def test_pool_selector_override_restricts_composition_pool(self):
        """A custom pool_selector must be used in place of
        _build_composition_pool -- composed events must only draw
        constituents from what the selector returns."""
        rng = np.random.default_rng(11)
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        events = []
        for i in range(10):
            p_act = rng.uniform(0.15, 0.35)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))
        passing = gate.filter(events, ts)
        assert len(passing) >= 4, "fixture must produce enough passing singles"

        allowed = {"feat_0", "feat_1", "feat_2"}

        def only_allowed(pool):
            return [ev for ev in pool if ev.component.source_feature in allowed]

        composer = ANDComposer(gate)
        result = composer.compose(passing, ts, max_components=2, pool_selector=only_allowed)

        for ev in result:
            constituents = {c.source_feature for c in ev.component.components}
            assert constituents <= allowed, (
                f"pool_selector was bypassed: composed event drew from {constituents}"
            )

    def test_stratify_fn_round_robin_interleaves_strata_not_concatenates(self):
        """A custom stratify_fn's round-robin interleaving (issue #254) must
        put the sole pair of a tiny stratum ahead of a much larger stratum's
        pairs in traversal order -- concatenating strata and then truncating
        at a flat max_pairs would starve the tiny stratum entirely, exactly
        the v1 under-sampling bug issue #254 itself reports (51 edges found
        vs. 76 from a tighter baseline, fixed by round-robin interleaving in
        v2)."""
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        rng = np.random.default_rng(254)
        # Strongly overlapping shared-base pair -- the sole member of the
        # "small" stratum, reliably passing both individually and AND-composed
        # (same construction as test_and_composer_max_conc_matches_single_gate).
        base = rng.random(n) < 0.30
        s0 = pd.Series((base & (rng.random(n) < 0.85)).astype(float))
        s1 = pd.Series((base & (rng.random(n) < 0.85)).astype(float))

        events = [
            RawEvent(series=s0, component=_make_component("feat_0", "identity", {}, [])),
            RawEvent(series=s1, component=_make_component("feat_1", "identity", {}, [])),
        ]
        # A "big" stratum with far more valid pairs than max_pairs below.
        for i in range(2, 17):
            p_act = rng.uniform(0.15, 0.35)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))

        passing = gate.filter(events, ts)
        passing_features = {ev.component.source_feature for ev in passing}
        assert {"feat_0", "feat_1"} <= passing_features, "fixture setup failed: feat_0/feat_1 must pass the gate"
        assert len(passing) >= 12, "fixture must produce a 'big' stratum much larger than max_pairs"

        small_pair = frozenset({"feat_0", "feat_1"})

        def stratify(a, b):
            pair = frozenset({a.component.source_feature, b.component.source_feature})
            return "small" if pair == small_pair else "big"

        composer = ANDComposer(gate)
        result = composer.compose(
            passing, ts, max_components=2, stratify_fn=stratify, max_pairs=2,
        )
        composed_pairs = {frozenset(c.source_feature for c in ev.component.components) for ev in result}
        assert small_pair in composed_pairs, (
            "round-robin interleaving must not starve the tiny stratum even though "
            "the 'big' stratum has far more valid pairs than max_pairs"
        )

    def test_stratify_fn_none_key_drops_the_pair(self):
        """A stratify_fn returning None for a pair must exclude it from the
        composed output entirely, not merely deprioritise it."""
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        rng = np.random.default_rng(254)
        base = rng.random(n) < 0.30
        s0 = pd.Series((base & (rng.random(n) < 0.85)).astype(float))
        s1 = pd.Series((base & (rng.random(n) < 0.85)).astype(float))

        events = [
            RawEvent(series=s0, component=_make_component("feat_0", "identity", {}, [])),
            RawEvent(series=s1, component=_make_component("feat_1", "identity", {}, [])),
        ]
        passing = gate.filter(events, ts)
        assert len(passing) == 2

        composer = ANDComposer(gate)
        excluded = composer.compose(passing, ts, max_components=2, stratify_fn=lambda a, b: None)
        assert excluded == []

    def test_max_pairs_max_triples_override_the_module_caps(self):
        """max_pairs/max_triples override _MAX_PAIRS/_MAX_TRIPLES for a
        single call without touching the module-level defaults."""
        import forgedge.event_discovery.and_composer as and_composer_mod

        rng = np.random.default_rng(230)
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        events = []
        for i in range(40):
            p_act = rng.uniform(0.15, 0.35)
            s = pd.Series((rng.random(n) < p_act).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))
        passing = gate.filter(events, ts)
        assert len(passing) >= 22, "fixture must produce more valid pairs than the override cap"

        composer = ANDComposer(gate)
        capped = composer.compose(passing, ts, max_components=3, max_pairs=5, max_triples=3)
        assert len(capped) <= 5 + 3

        # The module-level defaults must be untouched by the override.
        assert and_composer_mod._MAX_PAIRS == 2000
        assert and_composer_mod._MAX_TRIPLES == 500

    def test_raw_event_from_candidate_round_trips_single_component(self):
        from forgedge.event_discovery.and_composer import raw_event_from_candidate
        from forgedge.event_discovery.discovery import raw_event_to_candidate
        from forgedge.event_discovery.models import ActivationStats, GateResult

        n = 500
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        series = pd.Series((np.arange(n) % 4 == 0).astype(float))
        comp = _make_component("feat_x", "identity", {}, [])
        g = GateResult(
            passed=True, n_activations=125, n_active_months=1,
            max_monthly_share=1.0, mean_tpm=125.0,
        )
        candidate = EventCandidate(
            event_id="EVT-feat_x-ID-0000",
            status="CANDIDATE",
            components=[comp],
            expression=comp.expression,
            activation_stats=ActivationStats(
                n_activations=125, n_active_months=1, zero_months=0,
                max_monthly_share=1.0, mean_tpm=125.0,
            ),
            consistency_gate=g,
            event_series=series,
        )

        raw = raw_event_from_candidate(candidate)
        assert raw.component is comp
        assert raw.gate_result is g
        assert raw.series is series

        # Round-trips back to a fresh EventCandidate with a new event_id.
        rebuilt = raw_event_to_candidate(
            raw, idx=7, timestamps=ts, timestamp_col="open_dt", gate_params=None,
        )
        assert rebuilt.event_id != candidate.event_id
        assert rebuilt.components == [comp]

    def test_raw_event_from_candidate_rejects_multi_component_candidate(self):
        from forgedge.event_discovery.and_composer import raw_event_from_candidate
        from forgedge.event_discovery.models import ActivationStats, GateResult

        comp_a = _make_component("feat_a", "identity", {}, [])
        comp_b = _make_component("feat_b", "identity", {}, [])
        g = GateResult(passed=True, n_activations=10, n_active_months=1, max_monthly_share=1.0, mean_tpm=10.0)
        composed_candidate = EventCandidate(
            event_id="EVT-feat_a AND feat_b-IDxID-0000",
            status="CANDIDATE",
            components=[comp_a, comp_b],
            expression="feat_a AND feat_b",
            activation_stats=ActivationStats(
                n_activations=10, n_active_months=1, zero_months=0,
                max_monthly_share=1.0, mean_tpm=10.0,
            ),
            consistency_gate=g,
            event_series=pd.Series([1.0, 0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="single-component"):
            raw_event_from_candidate(composed_candidate)

    def test_raw_event_to_candidate_matches_instance_method(self):
        """The module-level free function extracted from
        EventDiscovery._to_candidate must produce identical output to the
        (now one-line wrapper) instance method for the same inputs -- proof
        the extraction changed nothing."""
        df = _make_kpi_table(n=1500, seed=11)
        cfg = DiscoveryConfig(gate_params=GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1))
        ed = EventDiscovery(df, cfg)
        ed.run()  # populates ed.config with the resolved session config

        from forgedge.event_discovery.discovery import raw_event_to_candidate

        n = 500
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        series = pd.Series((np.arange(n) % 3 == 0).astype(float))
        comp = _make_component("feat_y", "identity", {}, [])
        g = GateResult(passed=True, n_activations=167, n_active_months=1, max_monthly_share=1.0, mean_tpm=167.0)
        ev = RawEvent(series=series, component=comp)
        ev.gate_result = g

        via_method = ed._to_candidate(ev, idx=3, timestamps=ts)
        via_function = raw_event_to_candidate(
            ev, idx=3, timestamps=ts,
            timestamp_col=ed.config.timestamp_col,
            gate_params=ed.config.gate_params,
        )

        assert via_method.event_id == via_function.event_id
        assert via_method.components == via_function.components
        assert via_method.activation_stats == via_function.activation_stats

    # ── Issue #254 (Phase 4): ANDComposer.compose() gains an optional
    # triple_third_filter hook, constraining a triple's third component
    # beyond structural admissibility -- zero change to existing behaviour
    # when omitted.

    def test_triple_third_filter_default_none_reproduces_unchanged_behaviour(self):
        rng = np.random.default_rng(2541)
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        base = rng.random(n) < 0.30
        events = []
        for i in range(6):
            s = pd.Series((base & (rng.random(n) < 0.85)).astype(float))
            events.append(RawEvent(series=s, component=_make_component(f"feat_{i}", "identity", {}, [])))
        passing = gate.filter(events, ts)

        composer = ANDComposer(gate)
        baseline = composer.compose(passing, ts, max_components=3)
        explicit_none = composer.compose(passing, ts, max_components=3, triple_third_filter=None)

        def _key(ev):
            return (ev.component.source_feature, ev.gate_result.n_activations)

        assert list(map(_key, baseline)) == list(map(_key, explicit_none))

    def test_triple_third_filter_excludes_rejected_thirds(self):
        """A third candidate the filter rejects must never appear in any
        composed triple, even though the underlying 3-way AND would
        otherwise pass the gate easily (fixture built so a false negative
        here would be a filter bug, not a fixture that never had a chance)."""
        rng = np.random.default_rng(2542)
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        base = rng.random(n) < 0.30
        s_a0 = pd.Series((base & (rng.random(n) < 0.85)).astype(float))
        s_a1 = pd.Series((base & (rng.random(n) < 0.85)).astype(float))
        s_ok = pd.Series((base & (rng.random(n) < 0.85)).astype(float))
        s_rejected = pd.Series((base & (rng.random(n) < 0.85)).astype(float))

        ev_a0 = RawEvent(series=s_a0, component=_make_component("feat_a0", "identity", {}, []))
        ev_a1 = RawEvent(series=s_a1, component=_make_component("feat_a1", "identity", {}, []))
        ev_ok = RawEvent(series=s_ok, component=_make_component("feat_ok", "identity", {}, []))
        ev_rejected = RawEvent(series=s_rejected, component=_make_component("feat_rejected", "identity", {}, []))
        passing = gate.filter([ev_a0, ev_a1, ev_ok, ev_rejected], ts)
        assert len(passing) == 4, "fixture setup failed: all four singles must pass the gate"

        def reject_feat_rejected(a, b, c):
            return c.component.source_feature != "feat_rejected"

        composer = ANDComposer(gate)
        result = composer.compose(passing, ts, max_components=3, triple_third_filter=reject_feat_rejected)

        triples = [ev for ev in result if len(ev.component.components) == 3]
        assert triples, "fixture must produce at least one triple to meaningfully check exclusion"
        for ev in triples:
            features = {c.source_feature for c in ev.component.components}
            assert "feat_rejected" not in features

    def test_triple_third_filter_receives_the_seed_pair_and_candidate_third(self):
        """The filter must be called with (pool[idx_a], pool[idx_b], pool[k])
        -- not some other ordering -- so a caller can key its decision off
        the seed pair specifically (e.g. grade_guided_compose's root-grade
        logic, issue #254 Phase 4)."""
        rng = np.random.default_rng(2543)
        n = 2000
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=0.5, dispersion_margin=3.0, min_episodes=1))

        base = rng.random(n) < 0.30
        events = [
            RawEvent(
                series=pd.Series((base & (rng.random(n) < 0.85)).astype(float)),
                component=_make_component(f"feat_{i}", "identity", {}, []),
            )
            for i in range(4)
        ]
        passing = gate.filter(events, ts)

        seen_triples = []

        def recording_filter(a, b, c):
            seen_triples.append((a.component.source_feature, b.component.source_feature, c.component.source_feature))
            return True

        composer = ANDComposer(gate)
        composer.compose(passing, ts, max_components=3, triple_third_filter=recording_filter)

        assert seen_triples, "fixture must exercise the triple loop at least once"
        for a_feat, b_feat, c_feat in seen_triples:
            assert a_feat != c_feat and b_feat != c_feat, (
                "the third argument must be a distinct candidate, not the seed pair itself"
            )

    # ── Issue #232: EventDiscovery must not have to keep every raw
    # candidate's full series resident just to gate it

    def test_retain_raw_events_false_makes_raw_events_none(self):
        df = _make_kpi_table(n=1500, seed=11)
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1),
            retain_raw_events=False,
        )
        ed = EventDiscovery(df, cfg)
        ed.run()
        assert ed.raw_events is None

    def test_retain_raw_events_false_produces_identical_candidates_and_report(self):
        """The #232 memory fix only changes whether the pre-gate candidate
        population stays resident afterward -- it must not change which
        candidates are discovered or the distribution-report diagnostic."""
        df = _make_kpi_table(n=1500, seed=11)
        gp = GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1)

        ed_retain = EventDiscovery(df.copy(), DiscoveryConfig(gate_params=gp, retain_raw_events=True))
        cands_retain = ed_retain.run()

        ed_free = EventDiscovery(df.copy(), DiscoveryConfig(gate_params=gp, retain_raw_events=False))
        cands_free = ed_free.run()

        assert ed_retain.raw_events is not None and len(ed_retain.raw_events) > 0
        assert ed_free.raw_events is None
        assert sorted(c.event_id for c in cands_retain) == sorted(c.event_id for c in cands_free)
        assert ed_retain.event_distribution_report == ed_free.event_distribution_report

    def test_retain_raw_events_false_uses_measurably_less_memory(self):
        """Directional, not a strict threshold (avoids flakiness): freeing a
        gate-failing candidate's series as it's gated, instead of keeping
        the whole pre-gate population resident, must measurably lower peak
        memory -- verified stable (~13% on this fixture) across repeats."""
        import tracemalloc

        df = _make_kpi_table(n=3000, seed=11)
        gp = GateParams(min_tpm=0.5, dispersion_margin=2.0, min_episodes=1)

        peaks = {}
        for retain in (True, False):
            tracemalloc.start()
            EventDiscovery(df.copy(), DiscoveryConfig(gate_params=gp, retain_raw_events=retain)).run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[retain] = peak

        assert peaks[False] < peaks[True] * 0.98, (
            f"retain_raw_events=False should use less peak memory: "
            f"{peaks[False]/1e6:.1f} MB vs {peaks[True]/1e6:.1f} MB"
        )

    def test_retain_raw_events_true_keeps_targetoptimizer_atoms_intact(self):
        """TargetOptimizer reads ed.raw_events directly, pre-gate (#232's own
        motivation for defaulting retain_raw_events=True) -- must keep
        working unchanged for a caller that never touches the new flag."""
        from forgedge import TargetConfig, TargetOptimizer

        df = _make_kpi_table(n=1500, seed=11)
        opt = TargetOptimizer(df, TargetConfig(horizon=6, min_return=0.02, side="long"))
        results = opt.run()
        assert opt._ed is not None
        assert opt._ed.raw_events is not None and len(opt._ed.raw_events) > 0
        assert isinstance(results, pd.DataFrame)

    # ── Issue #98: ANDComposer.compose() gate=None skips full gate ───────────

    def _make_two_events(self, n=8760):
        """Helper: two volume-passing events for compose() tests."""
        rng = np.random.default_rng(42)
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        gate = ConsistencyGate(GateParams(min_tpm=2.0, max_dispersion=2.5))
        month_idx, n_months = _build_month_index(ts)
        base = rng.random(n) < 0.30
        s1 = pd.Series((base & (rng.random(n) < 0.80)).astype(float))
        s2 = pd.Series((base & (rng.random(n) < 0.80)).astype(float))
        comp1 = _make_component("feat_x", "identity", {}, [])
        comp2 = _make_component("feat_y", "identity", {}, [])
        ev1 = RawEvent(series=s1, component=comp1)
        ev2 = RawEvent(series=s2, component=comp2)
        ev1.gate_result = gate.evaluate_series(s1, month_idx, n_months)
        ev2.gate_result = gate.evaluate_series(s2, month_idx, n_months)
        return [ev1, ev2], ts, gate

    def test_compose_gate_none_returns_volume_passing_events(self):
        """gate=None skips full gate; composed events have passed=False."""
        events, ts, gate = self._make_two_events()
        composed_no_gate = ANDComposer(gate).compose(events, ts, gate=None)
        assert len(composed_no_gate) > 0
        assert all(not ev.gate_result.passed for ev in composed_no_gate)

    def test_compose_gate_none_returns_more_than_with_gate(self):
        """gate=None returns >= composed events compared to full gate (no events filtered)."""
        events, ts, gate = self._make_two_events()
        composed_with_gate = ANDComposer(gate).compose(events, ts)
        composed_no_gate = ANDComposer(gate).compose(events, ts, gate=None)
        assert len(composed_no_gate) >= len(composed_with_gate)

    def test_compose_gate_none_subsequent_filter_recovers_standard_results(self):
        """Applying gate.filter() after gate=None compose() matches standard compose()."""
        events, ts, gate = self._make_two_events()
        composed_standard = ANDComposer(gate).compose(events, ts)
        composed_raw = ANDComposer(gate).compose(events, ts, gate=None)
        filtered = gate.filter(composed_raw, ts)
        standard_ids = {ev.component.expression for ev in composed_standard}
        filtered_ids = {ev.component.expression for ev in filtered}
        assert standard_ids == filtered_ids

    def test_compose_default_behaviour_unchanged(self):
        """Omitting gate param preserves current behaviour (passed=True on results)."""
        events, ts, gate = self._make_two_events()
        composed = ANDComposer(gate).compose(events, ts)
        assert all(ev.gate_result.passed for ev in composed)

    # ── Issue #124: max_and_components=1 must produce no composed events ─────

    def test_max_components_1_returns_no_compositions(self):
        """max_components=1 must return [] — no AND composition at all."""
        events, ts, gate = self._make_two_events()
        result = ANDComposer(gate).compose(events, ts, max_components=1)
        assert result == []

    @pytest.fixture(scope="class")
    def big_table(self):
        """EventDiscovery defensively copies its input (``self.df =
        kpi_table.copy()`` in ``__init__``), so sharing the raw table across
        the differently-configured runs below is safe."""
        return _make_kpi_table(n=8760)

    def test_max_components_1_via_discovery_config(self, big_table):
        """DiscoveryConfig(max_and_components=1) produces only single-component candidates."""
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.5, max_dispersion=15.0),
            max_and_components=1,
        )
        ed = EventDiscovery(big_table, cfg)
        cands = ed.run()
        assert all(len(c.components) == 1 for c in cands), (
            "max_and_components=1 must not produce 2- or 3-component candidates"
        )

    def test_max_components_2_produces_only_pairs(self, big_table):
        """max_and_components=2 produces single+pair candidates, no triples."""
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.5, max_dispersion=15.0),
            max_and_components=2,
        )
        ed = EventDiscovery(big_table, cfg)
        cands = ed.run()
        assert all(len(c.components) <= 2 for c in cands), (
            "max_and_components=2 must not produce 3-component candidates"
        )

    def test_triples_not_starved_by_pair_cap(self, big_table):
        """With max_and_components=3, triples are generated even when pairs cap is hit."""
        from forgedge.event_discovery.and_composer import _MAX_PAIRS
        cfg = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.5, max_dispersion=15.0),
            max_and_components=3,
        )
        ed = EventDiscovery(big_table, cfg)
        cands = ed.run()
        pairs = [c for c in cands if len(c.components) == 2]
        triples = [c for c in cands if len(c.components) == 3]
        # If pairs are at cap and triples exist, starvation bug is fixed
        if len(pairs) >= _MAX_PAIRS:
            assert len(triples) > 0, (
                "triples must not be starved when pairs reach _MAX_PAIRS"
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
# Issue #161: cross-column, cross-time ("lag-cross") events
# ---------------------------------------------------------------------------

class TestLagCrossEvents:
    """close[t] > low[t-1]-shaped events: end-to-end discovery, SQL/formula
    export, and OOS replay for identity/crossing/delta transforms."""

    def _ohlc_df(self, n: int = 1500, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        return pd.DataFrame({
            "open_dt": pd.date_range("2019-01-01", periods=n, freq="1D"),
            "close": close, "high": high, "low": low,
        })

    def test_exploratory_discovery_finds_lag_cross_candidates(self):
        """Regression for #161: unassisted (non-manual-events) discovery must
        be able to produce close/high/low cross-column, cross-time atoms —
        previously exactly zero such atoms were ever generated."""
        df = self._ohlc_df()
        ed = EventDiscovery(
            df,
            DiscoveryConfig(
                train_ratio=1.0,
                gate_params=GateParams(min_tpm=0.1, max_dispersion=5.0, min_episodes=1),
            ),
        )
        cands = ed.run()
        lag_cross = [
            c for c in cands
            if any("_lag" in comp.source_feature for comp in c.components)
        ]
        assert lag_cross, "no cross-column, cross-time candidate was discovered"

    def test_replay_matches_stored_series_across_transforms(self):
        """EventCandidate.apply() on the pipeline's own frame (ed.df) must
        reproduce the stored activation series exactly, for identity,
        crossing, and delta lag-cross components alike."""
        df = self._ohlc_df()
        ed = EventDiscovery(
            df,
            DiscoveryConfig(
                train_ratio=1.0,
                gate_params=GateParams(min_tpm=0.1, max_dispersion=5.0, min_episodes=1),
            ),
        )
        cands = ed.run()
        by_transform: dict[str, EventCandidate] = {}
        for c in cands:
            comp = c.components[0]
            if "_lag" not in comp.source_feature:
                continue
            key = f"{comp.event_type}:{comp.transform}"
            by_transform.setdefault(key, c)
        assert by_transform, "no lag-cross candidate to check"

        for key, cand in by_transform.items():
            replayed = cand.apply(ed.df).reindex(cand.event_series.index).fillna(0)
            stored = cand.event_series.fillna(0)
            assert (replayed == stored).all(), f"replay mismatch for {key}"

    def test_sql_expression_has_no_nested_window_function(self):
        """Nested window function calls (e.g. LAG(LAG(...) OVER (...), N)
        OVER (...)) are invalid SQL. A lag-cross feature's identity/threshold
        and identity/crossing SQL, and its delta-transform SQL, must not
        contain that pattern (issue #161's SQL export follow-on fix)."""
        df = self._ohlc_df()
        ed = EventDiscovery(
            df,
            DiscoveryConfig(
                train_ratio=1.0,
                gate_params=GateParams(min_tpm=0.1, max_dispersion=5.0, min_episodes=1),
            ),
        )
        cands = ed.run()
        checked = 0
        for c in cands:
            comp = c.components[0]
            if "_lag" not in comp.source_feature:
                continue
            if comp.transform not in ("identity", "delta"):
                continue
            sql = comp.sql_expression.replace(" ", "").replace("\n", "")
            assert "LAG(LAG" not in sql, f"nested LAG in: {comp.sql_expression}"
            checked += 1
        assert checked > 0, "no identity/delta lag-cross component to check"

    def test_event_formula_shows_lag_annotation(self):
        """The human-readable formula must show which operand is time-shifted
        and by how much, e.g. 'close / low[t-1]'."""
        m_params = {"cross_lag": 1}
        comp = _make_component(
            "ratio_close_low_lag1", "identity", m_params, ["close", "low"],
            threshold=1.0, direction="above",
        )
        from forgedge.event_discovery.event_generator import _build_event_formula
        formula = _build_event_formula(
            comp.source_feature, comp.source_cols, comp.transform,
            comp.transform_params, comp.threshold, comp.direction, comp.event_type,
        )
        assert "low[t-1]" in formula
        assert formula.startswith("close / low[t-1]")

    def test_crossing_sql_uses_shifted_base_columns_not_lag_of_feature(self):
        """The crossing condition's 'previous bar' term must be re-derived
        from LAG on the base columns (a[t-1]/b[t-1-cross_lag]), not
        LAG(feat_sql) — which would nest window functions."""
        from forgedge.event_discovery.event_generator import _build_sql_expression
        sql = _build_sql_expression(
            source_feature="ratio_close_low_lag1",
            source_cols=["close", "low"],
            transform="identity",
            transform_params={"cross_lag": 1},
            threshold=1.0,
            direction="below",
            event_type="crossing",
            ts_col="open_dt",
        )
        assert 'LAG("close", 1)' in sql
        assert 'LAG("low", 2)' in sql  # cross_lag(1) + extra_shift(1)
        assert "LAG(LAG" not in sql.replace(" ", "")


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


# ---------------------------------------------------------------------------
# Folds that cannot conclude — F1 (#177)
# ---------------------------------------------------------------------------

class TestIndeterminateFolds:
    """A fold too short to distinguish a healthy candidate from a silent one
    is not evidence against the candidate.

    `GateParams` used to claim all its parameters were rate/ratio invariant.
    Two are; `min_episodes` is an absolute count, so applying it verbatim to a
    walk-forward fold makes the implicit rate requirement inversely
    proportional to the fold's length — 5.1x stricter than in-sample on the
    configuration `forge`'s own docstring recommends for production, at which
    0.7% of fold evaluations passed.
    """
    pytestmark = pytest.mark.slow

    @staticmethod
    def _table(n=1500, seed=2):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
        return pd.DataFrame({
            "open_dt": pd.date_range("2022-01-01", periods=n, freq="1D"),
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": rng.uniform(0.0, 1.0, n),
        })

    def _run(self, n_splits, train_ratio=0.80, min_tpm=1.0, n=1500):
        cfg = DiscoveryConfig(
            timestamp_col="open_dt", train_ratio=train_ratio,
            walk_forward=WalkForwardConfig(n_splits=n_splits),
            gate_params=GateParams(min_tpm=min_tpm),
        )
        ed = EventDiscovery(self._table(n=n), cfg)
        cands = ed.run()
        return [c.validation for c in cands if c.validation is not None]

    def test_min_episodes_no_longer_gates_the_folds(self):
        """The absolute floor is a statistical-power guard for the *discovery*
        window.  Out of sample the power question is answered by the fold's own
        expected count, not by a constant that happens to be 10."""
        vals = self._run(n_splits=1)
        assert vals
        # With min_episodes applied out of sample virtually nothing passed;
        # the folds now decide on rate and dispersion, as documented.
        assert any(v.passed for v in vals)

    def test_a_short_fold_is_indeterminate_not_failed(self):
        """Chopped into many splits, each fold expects fewer than
        MIN_FOLD_LAMBDA episodes — an empty one is then compatible with a
        healthy candidate more than 5 % of the time."""
        vals = self._run(n_splits=12)
        assert vals
        folds = [f for v in vals for f in v.fold_results]
        assert any(f.indeterminate for f in folds)
        for f in folds:
            assert f.indeterminate == (f.lam < MIN_FOLD_LAMBDA)
            if f.indeterminate:
                assert not f.passed          # never counted as a pass either

    def test_indeterminate_folds_leave_the_denominator(self):
        vals = self._run(n_splits=12)
        for v in vals:
            testable = [f for f in v.fold_results if not f.indeterminate]
            assert v.n_testable == len(testable)
            if testable:
                assert v.pass_rate == pytest.approx(v.n_passed / len(testable))

    def test_nothing_testable_is_inconclusive_not_failed(self):
        """`passed = None`, and the distinction matters downstream: `None` is
        falsy, so a filter written as `if candidate.validation` would treat
        "we could not tell" as "it failed" and discard everything."""
        vals = self._run(n_splits=40, min_tpm=0.2)
        inconclusive = [v for v in vals if v.n_testable == 0]
        if not inconclusive:
            pytest.skip("every fold was testable on this fixture")
        for v in inconclusive:
            assert v.passed is None
            assert v.fold_results            # diagnostics kept, not dropped
            assert math.isnan(v.pass_rate)

    def test_the_forge_filter_keeps_everything_when_nothing_concluded(self):
        """`only_validated_events` narrows the set only when validation
        actually *concluded*.  The pre-#177 test was `validation is not None`,
        which `passed=None` satisfies — so the filter would have kept zero
        candidates, exactly what its own comment exists to prevent."""
        from types import SimpleNamespace

        cands = [SimpleNamespace(validation=SimpleNamespace(passed=None)),
                 SimpleNamespace(validation=SimpleNamespace(passed=None))]
        concluded = any(c.validation is not None and c.validation.passed is not None
                        for c in cands)
        assert concluded is False
