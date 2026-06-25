"""Tests for forgedge.summary_report — data-quality diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import summary_report, DataQualityReport


def _clean_ohlc(n: int = 400, start: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """A structurally valid OHLC daily frame (gentle random walk)."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.empty(n)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    span = np.abs(rng.normal(0, 0.005, n)) * close
    high = np.maximum(openp, close) + span
    low = np.minimum(openp, close) - span
    return pd.DataFrame({
        "open_dt": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": openp, "high": high, "low": low, "close": close,
    })


def _codes(rep: DataQualityReport) -> set:
    return {f.code for f in rep.findings}


def test_returns_none_by_default(capsys):
    df = _clean_ohlc()
    out = summary_report(df)
    assert out is None
    printed = capsys.readouterr().out
    assert "DATA QUALITY SUMMARY" in printed


def test_clean_data_not_failed():
    rep = summary_report(_clean_ohlc(), verbose=False, return_report=True)
    assert not rep.has_critical
    assert "scale_mixed" not in _codes(rep)


def test_parity_crossing_is_not_mixed_scale():
    """A series straddling 1.0 (adjacent 10^-1 / 10^0 buckets) is NOT a scale bug."""
    df = _clean_ohlc(start=1.0)
    # squeeze prices into ~0.9..1.2 so both magnitude buckets are occupied
    for c in ("open", "high", "low", "close"):
        df[c] = 0.9 + (df[c] - df[c].min()) / (df[c].max() - df[c].min()) * 0.3
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    rep = summary_report(df, verbose=False, return_report=True)
    assert "scale_mixed" not in _codes(rep)


def test_mixed_scale_detected():
    """Sub-set of bars stored ×10000 → magnitudes 2+ orders apart → FAIL."""
    df = _clean_ohlc(start=1.1)
    bad = slice(50, 90)
    for c in ("open", "high", "low", "close"):
        df.loc[bad, c] = df.loc[bad, c] * 10000.0
    rep = summary_report(df, verbose=False, return_report=True)
    assert rep.has_critical
    assert "scale_mixed" in _codes(rep)


def test_high_less_than_low_fails():
    df = _clean_ohlc()
    df.loc[10, "high"] = df.loc[10, "low"] - 1.0
    rep = summary_report(df, verbose=False, return_report=True)
    assert rep.has_critical
    assert "ohlc_hl" in _codes(rep)


def test_close_out_of_range_fails():
    df = _clean_ohlc()
    df.loc[20, "close"] = df.loc[20, "high"] * 2.0  # close above high
    rep = summary_report(df, verbose=False, return_report=True)
    assert rep.has_critical
    assert "ohlc_close" in _codes(rep)


def test_nonpositive_price_fails():
    df = _clean_ohlc()
    df.loc[5, "low"] = -1.0
    rep = summary_report(df, verbose=False, return_report=True)
    assert rep.has_critical
    assert "nonpos" in _codes(rep)


def test_extreme_return_is_warn_not_fail():
    """A single +80% bar (crypto-plausible) is WARN, never a structural FAIL."""
    df = _clean_ohlc()
    i = 200
    df.loc[i, "close"] = df.loc[i - 1, "close"] * 1.8
    df.loc[i, "high"] = df.loc[i, "close"]
    df.loc[i, "open"] = df.loc[i - 1, "close"]
    df.loc[i, "low"] = df.loc[i, "open"]
    rep = summary_report(df, verbose=False, return_report=True)
    codes = _codes(rep)
    assert "ret_extreme" in codes
    # the extreme move alone must not produce any FAIL finding
    assert not any(f.level == "FAIL" for f in rep.findings)


def test_duplicate_timestamps_fail():
    df = _clean_ohlc(n=100)
    df.loc[50, "open_dt"] = df.loc[49, "open_dt"]
    rep = summary_report(df, verbose=False, return_report=True)
    assert rep.has_critical
    assert "dup_time" in _codes(rep)


def test_missing_column_fails():
    df = _clean_ohlc().drop(columns=["high"])
    rep = summary_report(df, verbose=False, return_report=True)
    assert rep.has_critical
    assert "missing_cols" in _codes(rep)


def test_datetime_index_is_used_for_gaps():
    """Timestamps taken from a DatetimeIndex; an injected gap → WARN."""
    df = _clean_ohlc(n=120).set_index("open_dt")
    # drop a contiguous week to create a gap
    df = df.drop(df.index[60:67])
    rep = summary_report(df, verbose=False, return_report=True)
    assert "gaps" in _codes(rep)


def test_empty_frame():
    rep = summary_report(pd.DataFrame(columns=["open", "high", "low", "close"]),
                         verbose=False, return_report=True)
    assert rep.has_critical
    assert "empty" in _codes(rep)


def test_one_line_and_worst():
    rep = summary_report(_clean_ohlc(), verbose=False, return_report=True)
    assert isinstance(rep.one_line(), str)
    assert rep.worst in ("OK", "WARN", "FAIL")
