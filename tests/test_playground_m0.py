"""Tests for forgedge.playground.m0 — regime transitions and time share.

Builds a minimal ``ForgeResult`` stand-in (only ``.ticker``/``.enriched`` are
read) over a small synthetic ``enriched`` frame with a hand-picked ``regime``
sequence, so expected transitions/shares can be computed by hand.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgedge.playground import regime_time_share, regime_transitions


def _result(ticker, regime_sequence, use_datetime_index=False, open_dt=None):
    df = pd.DataFrame({"regime": pd.Categorical(regime_sequence)})
    if use_datetime_index:
        df.index = pd.date_range("2024-01-01", periods=len(regime_sequence), freq="1h")
    elif open_dt is not None:
        df["open_dt"] = open_dt
    return SimpleNamespace(ticker=ticker, enriched=df)


class TestRegimeTransitions:
    def test_no_transitions_for_constant_regime(self):
        result = _result("T", ["bull"] * 5)

        df = regime_transitions([result])

        assert df.empty

    def test_counts_each_flip_with_preceding_run_length(self):
        # bull(3) -> bear(2) -> bull(1)
        result = _result("T", ["bull", "bull", "bull", "bear", "bear", "bull"])

        df = regime_transitions([result])

        assert len(df) == 2
        first, second = df.iloc[0], df.iloc[1]
        assert first["from_regime"] == "bull" and first["to_regime"] == "bear"
        assert first["run_length_before"] == 3
        assert second["from_regime"] == "bear" and second["to_regime"] == "bull"
        assert second["run_length_before"] == 2

    def test_leading_nan_regime_is_not_a_transition(self):
        result = _result("T", [np.nan, np.nan, "bull", "bull"])

        df = regime_transitions([result])

        assert df.empty

    def test_nervous_boundary_has_short_run_length(self):
        # bull(1) -> bear(1) -> bull(1) -> bear(1): maximally nervous
        result = _result("T", ["bull", "bear", "bull", "bear"])

        df = regime_transitions([result])

        assert len(df) == 3
        assert (df["run_length_before"] == 1).all()

    def test_uses_datetime_index_when_available(self):
        result = _result("T", ["bull", "bear"], use_datetime_index=True)

        df = regime_transitions([result])

        assert isinstance(df.iloc[0]["timestamp"], pd.Timestamp)

    def test_falls_back_to_open_dt_column(self):
        open_dt = pd.date_range("2024-01-01", periods=2, freq="1h")
        result = _result("T", ["bull", "bear"], open_dt=open_dt)

        df = regime_transitions([result])

        assert df.iloc[0]["timestamp"] == open_dt[1]

    def test_falls_back_to_bar_index_without_timestamp_info(self):
        result = _result("T", ["bull", "bear"])

        df = regime_transitions([result])

        assert df.iloc[0]["timestamp"] == 1

    def test_skips_results_without_regime_column(self):
        result = SimpleNamespace(ticker="T", enriched=pd.DataFrame({"close": [1, 2, 3]}))

        df = regime_transitions([result])

        assert df.empty

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = regime_transitions([])

        assert df.empty
        assert list(df.columns) == [
            "ticker",
            "bar_index",
            "timestamp",
            "from_regime",
            "to_regime",
            "run_length_before",
        ]

    def test_aggregates_across_multiple_results(self):
        r1 = _result("T1", ["bull", "bear"])
        r2 = _result("T2", ["bear", "bull"])

        df = regime_transitions([r1, r2])

        assert set(df["ticker"]) == {"T1", "T2"}


class TestRegimeTimeShare:
    def test_shares_sum_to_one(self):
        result = _result("T", ["bull", "bull", "bull", "bear"])

        df = regime_time_share([result])

        assert df.set_index("regime")["share"].to_dict() == pytest.approx(
            {"bull": 0.75, "bear": 0.25}
        )

    def test_dominant_regime_via_downstream_groupby(self):
        result = _result("T", ["bull", "bull", "bull", "bear"])

        df = regime_time_share([result])
        top = df.sort_values("share", ascending=False).groupby("ticker").head(1)

        assert top.iloc[0]["regime"] == "bull"

    def test_nan_regime_excluded_from_share(self):
        result = _result("T", ["bull", np.nan, "bull"])

        df = regime_time_share([result])

        assert df["n_bars"].sum() == 2
        assert df.set_index("regime")["share"].to_dict() == {"bull": 1.0}

    def test_skips_results_without_regime_column(self):
        result = SimpleNamespace(ticker="T", enriched=pd.DataFrame({"close": [1, 2, 3]}))

        df = regime_time_share([result])

        assert df.empty

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = regime_time_share([])

        assert df.empty
        assert list(df.columns) == ["ticker", "regime", "n_bars", "share"]

    def test_aggregates_across_multiple_results(self):
        r1 = _result("T1", ["bull", "bull"])
        r2 = _result("T2", ["bear", "bear"])

        df = regime_time_share([r1, r2])

        assert set(df["ticker"]) == {"T1", "T2"}
