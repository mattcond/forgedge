"""Tests for forgedge.kpi_builder — build_features + lag_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import build_features, lag_features, candle_features
from forgedge.kpi_builder import DEFAULT_CONFIG


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
