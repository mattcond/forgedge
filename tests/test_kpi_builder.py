"""Tests for forgedge.kpi_builder — candles → KPI Table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import build_kpi_table
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


def test_basic_build_produces_expected_columns():
    kpi = build_kpi_table(_candles(), timestamp_col="open_time")
    # output datetime column for forge
    assert "open_dt" in kpi.columns
    assert pd.api.types.is_datetime64_any_dtype(kpi["open_dt"])
    # representative indicators across families (FeatureGenerator naming)
    for col in ("close_sma_03", "close_ema_25", "close_rsi_14",
                "close_bb_lower_20", "close_max_168", "close_mdd_12",
                "volume_sma_03", "color"):
        assert col in kpi.columns, col


def test_column_names_match_feature_generator():
    """The produced names must be recognised by FORGE's FeatureGenerator."""
    from forgedge.event_discovery.feature_generator import parse_feature
    kpi = build_kpi_table(_candles(), timestamp_col="open_time")
    for col in ("close_ema_09", "close_sma_12", "close_bb_lower_20", "close_min_24"):
        assert parse_feature(col) is not None, col


def test_nan_warmup_preserved_no_object_dtype():
    kpi = build_kpi_table(_candles(), timestamp_col="open_time")
    feature_cols = [c for c in kpi.columns
                    if c not in ("open_time", "open_dt", "open", "high", "low", "close", "volume")]
    # leading warm-up NaNs exist …
    assert kpi["close_sma_168"].isna().any()
    # … but no feature column was upcast to object (would break FORGE)
    assert not any(kpi[c].dtype == object for c in feature_cols)


def test_lagging_of_derived_columns():
    """Phase 2 lags columns derived in phase 1 (the two-phase contract)."""
    kpi = build_kpi_table(_candles(), timestamp_col="open_time")
    for col in ("close_ema_03_prev_01", "close_ema_25_prev_03", "color_prev_01"):
        assert col in kpi.columns, col
    # the lag is the previous bar's value of the derived column
    np.testing.assert_allclose(
        kpi["close_ema_03_prev_01"].to_numpy()[1:],
        kpi["close_ema_03"].round(5).to_numpy()[:-1],
        equal_nan=True,
    )


def test_missing_columns_are_skipped():
    """OHLC-only data (no volume) builds fine; volume features are skipped."""
    kpi = build_kpi_table(_candles(with_volume=False), timestamp_col="open_time")
    assert not any(c.startswith("volume_") for c in kpi.columns)
    assert "close_ema_03" in kpi.columns  # the rest is unaffected


def test_timestamp_col_is_required_and_validated():
    with pytest.raises(TypeError):
        build_kpi_table(_candles())  # missing required keyword
    with pytest.raises(KeyError):
        build_kpi_table(_candles(), timestamp_col="does_not_exist")


def test_datetime_timestamp_input():
    df = _candles()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    kpi = build_kpi_table(df.drop(columns=["open_time"]), timestamp_col="ts")
    assert pd.api.types.is_datetime64_any_dtype(kpi["open_dt"])
    assert "close_ema_09" in kpi.columns


def test_output_is_sorted_even_if_input_shuffled():
    df = _candles().sample(frac=1.0, random_state=1).reset_index(drop=True)
    kpi = build_kpi_table(df, timestamp_col="open_time")
    assert kpi["open_dt"].is_monotonic_increasing


def test_dict_config_subset():
    cfg = {"ema": {"enabled": True, "params": {"periods": [9, 25], "columns": ["close"]}}}
    kpi = build_kpi_table(_candles(), cfg, timestamp_col="open_time")
    assert "close_ema_09" in kpi.columns and "close_ema_25" in kpi.columns
    assert "close_rsi_14" not in kpi.columns  # nothing else computed


def test_disabled_indicator_skipped():
    cfg = {
        "ema": {"enabled": False, "params": {"periods": [9], "columns": ["close"]}},
        "rsi": {"enabled": True, "params": {"periods": [14], "columns": ["close"]}},
    }
    kpi = build_kpi_table(_candles(), cfg, timestamp_col="open_time")
    assert "close_ema_09" not in kpi.columns
    assert "close_rsi_14" in kpi.columns


def test_row_count_preserved():
    candles = _candles(n=250)
    kpi = build_kpi_table(candles, timestamp_col="open_time")
    assert len(kpi) == len(candles)


def test_default_config_is_used_when_none():
    kpi = build_kpi_table(_candles(), None, timestamp_col="open_time")
    # a column from each phase-1 family in the default config
    assert "close_mdd_240" in kpi.columns
    assert set(DEFAULT_CONFIG).issuperset({"ema", "rsi", "lagging"})
