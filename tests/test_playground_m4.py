"""Tests for forgedge.playground.m4 — classification by grade, duplicate clusters.

These take RuleRegistry-like stand-ins directly (see m4.py's module
docstring for why: the registry that matters lives outside ForgeResult in
the forge_multi() path).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from forgedge.playground import classification_by_grade, duplicate_clusters


def _doc(rule_id, source_ticker="T", grade="A", classification=None, is_duplicate=None, duplicate_of=None):
    return SimpleNamespace(
        rule_id=rule_id,
        source_ticker=source_ticker,
        grade=grade,
        classification=classification,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
    )


def _registry(documents):
    return SimpleNamespace(documents=list(documents))


class TestClassificationByGrade:
    def test_includes_only_documents_with_classification(self):
        d1 = _doc("R1", classification="GENERIC")
        d2 = _doc("R2", classification=None)
        registry = _registry([d1, d2])

        df = classification_by_grade([registry])

        assert len(df) == 1
        assert df.iloc[0]["rule_id"] == "R1"

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = classification_by_grade([])

        assert df.empty
        assert list(df.columns) == ["rule_id", "source_ticker", "grade", "classification"]

    def test_aggregates_across_multiple_registries(self):
        r1 = _registry([_doc("R1", classification="GENERIC")])
        r2 = _registry([_doc("R2", classification="ISOLATED")])

        df = classification_by_grade([r1, r2])

        assert set(df["rule_id"]) == {"R1", "R2"}

    def test_downstream_crosstab_by_grade(self):
        docs = [
            _doc("R1", grade="A", classification="GENERIC"),
            _doc("R2", grade="A", classification="GENERIC"),
            _doc("R3", grade="C", classification="ISOLATED"),
        ]
        registry = _registry(docs)

        df = classification_by_grade([registry])
        crosstab = pd.crosstab(df["grade"], df["classification"])

        assert crosstab.loc["A", "GENERIC"] == 2
        assert crosstab.loc["C", "ISOLATED"] == 1


class TestDuplicateClusters:
    def test_flags_duplicate_and_survivor(self):
        survivor = _doc("R1", is_duplicate=False, duplicate_of=None)
        dup = _doc("R2", is_duplicate=True, duplicate_of="R1")
        registry = _registry([survivor, dup])

        df = duplicate_clusters([registry])

        assert set(df["is_duplicate"]) == {False, True}
        assert df[df["rule_id"] == "R2"]["duplicate_of"].iloc[0] == "R1"

    def test_unresolved_duplicate_status_stays_none(self):
        doc = _doc("R1", is_duplicate=None, duplicate_of=None)
        registry = _registry([doc])

        df = duplicate_clusters([registry])

        assert df.iloc[0]["is_duplicate"] is None

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        df = duplicate_clusters([])

        assert df.empty
        assert list(df.columns) == [
            "rule_id",
            "source_ticker",
            "grade",
            "is_duplicate",
            "duplicate_of",
        ]

    def test_downstream_cluster_sizes(self):
        docs = [
            _doc("R1", is_duplicate=False),
            _doc("R2", is_duplicate=True, duplicate_of="R1"),
            _doc("R3", is_duplicate=True, duplicate_of="R1"),
            _doc("R4", is_duplicate=False),
        ]
        registry = _registry(docs)

        df = duplicate_clusters([registry])
        sizes = df[df["is_duplicate"]].groupby("duplicate_of").size()

        assert sizes["R1"] == 2
        assert df["is_duplicate"].mean() == pytest.approx(0.5)
