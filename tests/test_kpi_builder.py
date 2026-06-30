"""Tests for forgedge.kpi_builder — build_features + lag_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import build_features, lag_features, candle_features, pattern_features
from forgedge.kpi_builder import (
    DEFAULT_CONFIG, PATTERNS, is_hammer, is_doji, is_bullish_engulfing,
)


def _candles(n: int = 300, with_volume: bool = True, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(openp, close) + span
    low = np.minimum(openp, close) - span
    start_ms = 1_600_000_000_000
    data = {
        "open_time": start_ms + np.arange(n) * 3_600_000,  # 1H in ms
        "open": openp, "high": high, "low": low, "close": close,
    }
    if with_volume:
        data["volume"] = rng.uniform(10, 100, n)
    return pd.DataFrame(data)


# ── build_features ────────────────────────────────────────────────────────────

def test_build_features_produces_expected_columns():
    kpi = build_features(_candles(), timestamp_col="open_time")
    assert "open_dt" in kpi.columns
    assert pd.api.types.is_datetime64_any_dtype(kpi["open_dt"])
    for col in ("close_sma_03", "close_ema_25", "close_rsi_14",
                "close_bb_lower_20", "close_max_168", "close_mdd_12",
                "volume_sma_03", "color"):
        assert col in kpi.columns, col


def test_build_features_does_not_lag():
    """Lagging is no longer a build step — no *_prev_* columns are produced."""
    kpi = build_features(_candles(), timestamp_col="open_time")
    assert not any("_prev_" in c for c in kpi.columns)


def test_column_names_match_feature_generator():
    from forgedge.event_discovery.feature_generator import parse_feature
    kpi = build_features(_candles(), timestamp_col="open_time")
    for col in ("close_ema_09", "close_sma_12", "close_bb_lower_20", "close_min_24"):
        assert parse_feature(col) is not None, col


def test_nan_warmup_preserved_no_object_dtype():
    kpi = build_features(_candles(), timestamp_col="open_time")
    feature_cols = [c for c in kpi.columns
                    if c not in ("open_time", "open_dt", "open", "high", "low", "close", "volume")]
    assert kpi["close_sma_168"].isna().any()
    assert not any(kpi[c].dtype == object for c in feature_cols)


def test_missing_columns_are_skipped():
    kpi = build_features(_candles(with_volume=False), timestamp_col="open_time")
    assert not any(c.startswith("volume_") for c in kpi.columns)
    assert "close_ema_03" in kpi.columns


# ── atr / macd (opt-in, disabled in DEFAULT_CONFIG) ────────────────────────────

def test_atr_disabled_by_default():
    kpi = build_features(_candles(), timestamp_col="open_time")
    assert not any("_atr_" in c for c in kpi.columns)


def test_atr_enabled_produces_column():
    cfg = {"atr": {"enabled": True, "params": {"periods": [14], "columns": ["close"]}}}
    kpi = build_features(_candles(), cfg, timestamp_col="open_time")
    assert "close_atr_14" in kpi.columns
    assert kpi["close_atr_14"].dropna().ge(0).all()


def test_atr_skipped_without_high_low():
    candles = _candles().drop(columns=["high", "low"])
    cfg = {"atr": {"enabled": True, "params": {"periods": [14], "columns": ["close"]}}}
    kpi = build_features(candles, cfg, timestamp_col="open_time")
    assert not any("_atr_" in c for c in kpi.columns)


def test_macd_disabled_by_default():
    kpi = build_features(_candles(), timestamp_col="open_time")
    assert not any("_macd_" in c for c in kpi.columns)


def test_macd_enabled_produces_line_signal_hist():
    cfg = {"macd": {"enabled": True, "params": {"periods": [12, 26, 9], "columns": ["close"]}}}
    kpi = build_features(_candles(), cfg, timestamp_col="open_time")
    assert "close_macd_12_26" in kpi.columns
    assert "close_macd_12_26_signal_09" in kpi.columns
    assert "close_macd_12_26_hist_09" in kpi.columns


def test_macd_multiple_triples():
    cfg = {"macd": {"enabled": True,
                    "params": {"periods": [12, 26, 9, 5, 35, 5], "columns": ["close"]}}}
    kpi = build_features(_candles(), cfg, timestamp_col="open_time")
    assert "close_macd_12_26" in kpi.columns
    assert "close_macd_05_35" in kpi.columns


def test_macd_invalid_periods_length_raises():
    cfg = {"macd": {"enabled": True, "params": {"periods": [12, 26], "columns": ["close"]}}}
    with pytest.raises(ValueError):
        build_features(_candles(), cfg, timestamp_col="open_time")


def test_lagging_block_in_config_is_ignored():
    """A stray 'lagging' block (legacy config) must not error and must not lag."""
    cfg = {
        "ema": {"enabled": True, "params": {"periods": [9], "columns": ["close"]}},
        "lagging": {"enabled": True, "params": {"periods": [1], "columns": ["close_ema_09"]}},
    }
    kpi = build_features(_candles(), cfg, timestamp_col="open_time")
    assert "close_ema_09" in kpi.columns
    assert not any("_prev_" in c for c in kpi.columns)


def test_timestamp_col_is_required_and_validated():
    with pytest.raises(TypeError):
        build_features(_candles())  # missing required keyword
    with pytest.raises(KeyError):
        build_features(_candles(), timestamp_col="does_not_exist")


def test_datetime_timestamp_input():
    df = _candles()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    kpi = build_features(df.drop(columns=["open_time"]), timestamp_col="ts")
    assert pd.api.types.is_datetime64_any_dtype(kpi["open_dt"])
    assert "close_ema_09" in kpi.columns


def test_output_is_sorted_even_if_input_shuffled():
    df = _candles().sample(frac=1.0, random_state=1).reset_index(drop=True)
    kpi = build_features(df, timestamp_col="open_time")
    assert kpi["open_dt"].is_monotonic_increasing


def test_default_config_used_when_none_and_has_no_lagging():
    kpi = build_features(_candles(), None, timestamp_col="open_time")
    assert "close_mdd_240" in kpi.columns
    assert "lagging" not in DEFAULT_CONFIG


# ── lag_features ──────────────────────────────────────────────────────────────

def test_lag_features_explicit_columns():
    kpi = build_features(_candles(), timestamp_col="open_time")
    out = lag_features(kpi, "close_ema_03", "color", periods=[1, 2, 3])
    for col in ("close_ema_03_prev_01", "close_ema_03_prev_02", "close_ema_03_prev_03",
                "color_prev_01"):
        assert col in out.columns, col


def test_lag_features_is_previous_value_of_derived_column():
    kpi = build_features(_candles(), timestamp_col="open_time")
    out = lag_features(kpi, "close_ema_03", periods=[1])
    np.testing.assert_allclose(
        out["close_ema_03_prev_01"].to_numpy()[1:],
        out["close_ema_03"].round(5).to_numpy()[:-1],
        equal_nan=True,
    )


def test_lag_features_like_pattern():
    kpi = build_features(_candles(), timestamp_col="open_time")
    out = lag_features(kpi, like="_ema_", periods=[1])
    ema_cols = [c for c in kpi.columns if "_ema_" in c]
    assert ema_cols  # there are EMAs to lag
    for c in ema_cols:
        assert f"{c}_prev_01" in out.columns


def test_lag_features_periods_int():
    kpi = build_features(_candles(), timestamp_col="open_time")
    out = lag_features(kpi, "close", periods=2)
    assert "close_prev_02" in out.columns
    assert "close_prev_01" not in out.columns


def test_lag_features_missing_column_raises():
    kpi = build_features(_candles(), timestamp_col="open_time")
    with pytest.raises(KeyError):
        lag_features(kpi, "not_a_column")


def test_lag_features_no_selection_raises():
    kpi = build_features(_candles(), timestamp_col="open_time")
    with pytest.raises(ValueError):
        lag_features(kpi)  # neither explicit cols nor like=


def test_lag_features_excludes_order_on_from_like():
    kpi = build_features(_candles(), timestamp_col="open_time")
    # 'open' matches open, open_time and open_dt; order_on (open_dt) is excluded.
    out = lag_features(kpi, like="open", periods=[1])
    assert "open_prev_01" in out.columns
    assert "open_dt_prev_01" not in out.columns


def test_lag_features_preserves_row_order_and_count():
    kpi = build_features(_candles(n=250), timestamp_col="open_time")
    out = lag_features(kpi, "close", periods=[1])
    assert len(out) == len(kpi)
    assert out["open_dt"].is_monotonic_increasing


# ── candle_features ───────────────────────────────────────────────────────────

_GEOMETRY = ("body", "upper_wick", "lower_wick", "close_pos", "range_pct", "gap")


def test_candle_features_adds_geometry_columns():
    out = candle_features(_candles())
    for col in _GEOMETRY:
        assert col in out.columns, col


def test_candle_features_requires_ohlc():
    with pytest.raises(KeyError):
        candle_features(_candles().drop(columns=["high"]))


def test_candle_features_value_ranges_and_invariant():
    out = candle_features(_candles(n=500))
    g = out.dropna(subset=["body", "upper_wick", "lower_wick", "close_pos"])
    assert (g["body"].abs() <= 1 + 1e-9).all()
    assert ((g["upper_wick"] >= -1e-9) & (g["upper_wick"] <= 1 + 1e-9)).all()
    assert ((g["lower_wick"] >= -1e-9) & (g["lower_wick"] <= 1 + 1e-9)).all()
    assert ((g["close_pos"] >= -1e-9) & (g["close_pos"] <= 1 + 1e-9)).all()
    # geometric identity: |body| + upper_wick + lower_wick == 1 on non-flat bars
    total = g["body"].abs() + g["upper_wick"] + g["lower_wick"]
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-4)


def test_candle_features_are_scale_free():
    base = _candles(n=300)
    scaled = base.copy()
    for c in ("open", "high", "low", "close"):
        scaled[c] = scaled[c] * 1000.0
    a = candle_features(base)[list(_GEOMETRY)]
    b = candle_features(scaled)[list(_GEOMETRY)]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-4, equal_nan=True)


def test_candle_features_flat_bar_is_nan_not_inf():
    df = _candles(n=50)
    df.loc[10, ["open", "high", "low", "close"]] = 100.0  # flat bar, range == 0
    out = candle_features(df)
    row = out.loc[10]
    assert np.isnan(row["body"]) and np.isnan(row["close_pos"])
    assert not np.isinf(out[list(_GEOMETRY)].to_numpy(dtype="float64")).any()


def test_candle_features_gap_is_pct_change_of_open_vs_prev_close():
    df = _candles(n=100)
    out = candle_features(df)
    expected = (df["open"].to_numpy()[1:] / df["close"].to_numpy()[:-1]) - 1.0
    np.testing.assert_allclose(out["gap"].to_numpy()[1:], np.round(expected, 5), atol=1e-4)


def test_candle_features_compose_with_build_and_lag():
    kpi = build_features(_candles(), timestamp_col="open_time")
    kpi = candle_features(kpi)
    kpi = lag_features(kpi, like="wick", periods=[1])
    assert "upper_wick_prev_01" in kpi.columns and "lower_wick_prev_01" in kpi.columns
    # no column was upcast to object
    assert not any(kpi[c].dtype == object for c in _GEOMETRY)


# ── pattern_features (named, categorical, opt-in) ─────────────────────────────

def _pattern_frame() -> pd.DataFrame:
    # row0 bearish, row1 bullish-engulfs-row0, row2 hammer, row3 doji, row4 plain
    rows = [
        (10.0, 10.1, 8.9, 9.0),    # bearish
        (8.9, 10.2, 8.8, 10.1),    # bullish engulfing of row0
        (10.0, 10.5, 8.5, 10.4),   # hammer (long lower wick, small upper)
        (10.0, 10.5, 9.5, 10.01),  # doji (tiny body)
        (10.0, 10.3, 9.8, 10.2),   # plain
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["open_dt"] = pd.date_range("2022-01-01", periods=len(df), freq="D")
    return df


def test_pattern_features_single_categorical_column():
    out = pattern_features(_pattern_frame())
    assert "candle_pattern" in out.columns
    assert out["candle_pattern"].dtype == object          # categorical, not numeric
    # only pattern names or None — never 1/0
    assert set(out["candle_pattern"].dropna().unique()) <= set(
        {"DOJI", "HAMMER", "SHOOTING_STAR", "MARUBOZU",
         "BULLISH_ENGULFING", "BEARISH_ENGULFING", "BULLISH_HARAMI",
         "BEARISH_HARAMI", "MORNING_STAR", "EVENING_STAR"})


def test_pattern_features_is_classifiable_by_forge():
    """The single column must have ≥2 distinct values so FORGE keeps it."""
    from forgedge.event_discovery.classifier import TypeClassifier
    from forgedge.event_discovery.models import ColumnType
    out = pattern_features(_pattern_frame())
    cls = TypeClassifier().fit(out[["candle_pattern"]])
    assert "candle_pattern" in cls  # not skipped as constant
    assert cls["candle_pattern"].col_type == ColumnType.CATEGORICAL


def test_pattern_detectors_return_name_or_none():
    df = _pattern_frame()
    assert is_hammer(df).iloc[2] == "HAMMER"
    assert is_hammer(df).iloc[3] is None        # the doji bar is not a hammer
    assert is_doji(df).iloc[3] == "DOJI"


def test_pattern_features_detects_engulfing_hammer_doji():
    out = pattern_features(_pattern_frame())
    assert out["candle_pattern"].iloc[1] == "BULLISH_ENGULFING"
    assert out["candle_pattern"].iloc[2] == "HAMMER"
    assert out["candle_pattern"].iloc[3] == "DOJI"


def test_pattern_features_precedence_hammer_over_doji():
    # a dragonfly-doji bar matches both doji and hammer; hammer wins by precedence
    df = pd.DataFrame(
        [(10.0, 10.05, 8.5, 10.02)], columns=["open", "high", "low", "close"])
    df["open_dt"] = pd.date_range("2022-01-01", periods=1, freq="D")
    assert is_doji(df).iloc[0] == "DOJI"
    assert is_hammer(df).iloc[0] == "HAMMER"
    assert pattern_features(df)["candle_pattern"].iloc[0] == "HAMMER"


def test_pattern_features_subset_and_unknown():
    out = pattern_features(_pattern_frame(), patterns=["doji"])
    assert out["candle_pattern"].iloc[2] is None    # hammer not considered
    assert out["candle_pattern"].iloc[3] == "DOJI"
    with pytest.raises(KeyError):
        pattern_features(_pattern_frame(), patterns=["not_a_pattern"])


def test_pattern_features_requires_ohlc():
    with pytest.raises(KeyError):
        pattern_features(_pattern_frame().drop(columns=["high"]))


def test_pattern_features_not_in_default_build():
    """Named patterns must be opt-in — build_features never emits the column."""
    kpi = build_features(_candles(), timestamp_col="open_time")
    assert "candle_pattern" not in kpi.columns


def test_pattern_features_orders_shuffled_candles():
    """2-bar patterns must be correct even if rows arrive out of order."""
    frame = _pattern_frame()
    engulf_dt = frame["open_dt"].iloc[1]            # the bullish-engulfing bar
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = pattern_features(shuffled)                # order_on='open_dt' present
    got = out.loc[out["open_dt"] == engulf_dt, "candle_pattern"].iloc[0]
    assert got == "BULLISH_ENGULFING"


def test_multibar_detector_orders_standalone():
    """A multi-bar detector sorts by order_on internally (standalone-safe)."""
    frame = _pattern_frame()
    engulf_dt = frame["open_dt"].iloc[1]
    shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
    s = is_bullish_engulfing(shuffled, order_on="open_dt")
    assert s[shuffled["open_dt"] == engulf_dt].iloc[0] == "BULLISH_ENGULFING"


def test_pattern_features_multibar_requires_timestamp():
    """Without a timestamp column / DatetimeIndex, multi-bar patterns raise."""
    frame = _pattern_frame().drop(columns=["open_dt"]).reset_index(drop=True)
    with pytest.raises(KeyError):
        pattern_features(frame, patterns=["bullish_engulfing"])


def test_pattern_features_singlebar_ok_without_timestamp():
    """Single-bar patterns are order-independent → no timestamp required."""
    frame = _pattern_frame().drop(columns=["open_dt"]).reset_index(drop=True)
    out = pattern_features(frame, patterns=["hammer", "doji"])
    assert "candle_pattern" in out.columns


def test_pattern_features_datetimeindex_fallback():
    """A DatetimeIndex (no open_dt column) is accepted for ordering."""
    frame = _pattern_frame().set_index("open_dt")
    out = pattern_features(frame)
    assert "candle_pattern" in out.columns
