"""Tests for forgedge.playground.funnel — end-to-end conversion funnel."""

from types import SimpleNamespace

import pandas as pd
import pytest

from forgedge.playground import conversion_funnel


def _result(ticker, n_candidates=0, n_contracts=0, n_promoted=0, n_edges=0):
    return SimpleNamespace(
        ticker=ticker,
        candidates=list(range(n_candidates)),
        contracts=list(range(n_contracts)),
        promoted=list(range(n_promoted)),
        edges=lambda: list(range(n_edges)),
    )


class TestConversionFunnel:
    def test_emits_four_stage_rows_per_result(self):
        result = _result("T", n_candidates=10, n_contracts=8, n_promoted=4, n_edges=2)

        df = conversion_funnel([result])

        assert len(df) == 4
        assert set(df["stage"]) == {"candidates", "contracts", "promoted", "edges"}
        counts = df.set_index("stage")["n"].to_dict()
        assert counts == {"candidates": 10, "contracts": 8, "promoted": 4, "edges": 2}

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = conversion_funnel([])

        assert df.empty
        assert list(df.columns) == ["ticker", "stage", "n"]

    def test_downstream_pivot_per_asset(self):
        r1 = _result("T1", n_candidates=10, n_contracts=5, n_promoted=2, n_edges=1)
        r2 = _result("T2", n_candidates=20, n_contracts=15, n_promoted=8, n_edges=4)

        df = conversion_funnel([r1, r2])
        pivot = df.pivot(index="ticker", columns="stage", values="n")

        assert pivot.loc["T1", "candidates"] == 10
        assert pivot.loc["T2", "edges"] == 4

    def test_zero_counts_are_preserved(self):
        result = _result("T", n_candidates=5, n_contracts=0, n_promoted=0, n_edges=0)

        df = conversion_funnel([result])

        assert (df[df["stage"] != "candidates"]["n"] == 0).all()
