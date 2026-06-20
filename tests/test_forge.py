"""Tests for the ``forge`` end-to-end orchestrator."""
import numpy as np
import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    CustomEvent,
    DiscoveryConfig,
    ForgeResult,
    RuleDiscoveryConfig,
    RuleRegistry,
    forge,
    forge_multi,
)
from forgedge.event_discovery.models import GateParams
from forgedge.market_context.models import REGIME_COL
from forgedge.rule_discovery.models import GridSpec, WalkForwardConfig


# A deliberately strict, single-component Event Discovery config: the table
# spans enough calendar months for the default gate, and these thresholds keep
# the candidate count (and therefore the Rule Discovery work) small so the
# end-to-end tests stay fast.
_FAST_ED_CONFIG = DiscoveryConfig(
    max_and_components=1,
    gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
)

# Minimal Rule Discovery config — a single-cell grid and a light, non-reoptimised
# walk-forward — so each Modulo 3 run is cheap.  The verdicts are not the point
# of these orchestration tests; the wiring is.
_FAST_RD_CONFIG = RuleDiscoveryConfig(
    grid=GridSpec(buy_drop_pct=[0.0], buy_delay_bar=[0]),
    walk_forward=WalkForwardConfig(n_splits=2, reoptimise=False),
)


def _ohlc_kpi_table(n: int = 2600, seed: int = 7) -> pd.DataFrame:
    """KPI table with OHLCV + a mean-reverting feature, enough for every module.

    Sampled at 4H so ~2,600 bars already span more than a year (clearing the
    8-month gate) while keeping the pipeline fast.  Low ``feat`` predicts a
    positive next-bar return, so Event Discovery finds ``feat < x`` events and
    the full pipeline has something to chew on through Rule Discovery (which
    needs the high/low columns for its backtest).
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    k = 0.02
    noise = rng.normal(0.0, 0.004, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -k * (feat[:-1] - 0.5) + noise[1:]
    close = 100.0 * np.exp(np.cumsum(r))
    op = close * (1 + rng.normal(0.0, 0.001, n))
    high = np.maximum(op, close) * (1 + np.abs(rng.normal(0.0, 0.002, n)))
    low = np.minimum(op, close) * (1 - np.abs(rng.normal(0.0, 0.002, n)))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": feat,
        }
    )


class TestForge:
    @pytest.fixture(scope="class")
    @classmethod
    def kpi(cls):
        return _ohlc_kpi_table()

    @pytest.fixture(scope="class")
    @classmethod
    def full_result(cls, kpi):
        """Full M0→M3 pipeline, shared by tests that assert on the same run."""
        return forge(
            kpi,
            asset="TEST",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
        )

    def test_end_to_end_runs_every_module(self, full_result):
        result = full_result
        assert isinstance(result, ForgeResult)
        # Modulo 0 enriched the table.
        assert REGIME_COL in result.enriched.columns
        assert result.market_context is not None
        assert len(result.candidates) > 0
        # event_frame is the Event Discovery post-pipeline frame.
        assert result.event_frame is result.event_discovery.df
        # Modulo 2 produced one contract per candidate fed in.
        assert len(result.contracts) == len(result.candidates)
        assert all(c in result.contracts for c in result.promoted)
        assert len(result.rule_responses) == len(result.promoted)
        for contract, response in result.rule_responses:
            assert response.verdict in {"EDGE", "PARTIAL-EDGE", "NON-EDGE"}
            assert contract.alpha_id == response.alpha_id
        # Modulo 4 — Rule Registry built from this run's tradeable rules.
        assert isinstance(result.registry, RuleRegistry)
        assert len(result.registry.documents) == len(result.submissions())

    def test_summary_carries_rule_verdict(self, full_result):
        summary = full_result.summary()
        assert "rule_verdict" in summary.columns
        assert len(summary) == len(full_result.contracts)

    def test_skips_market_context_when_regime_present(self, kpi):
        enriched = forge(
            kpi, event_discovery_config=_FAST_ED_CONFIG, run_rule_discovery=False
        ).enriched
        result = forge(
            enriched,
            asset="TEST",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.market_context is None
        assert REGIME_COL in result.enriched.columns

    def test_run_market_context_false_skips_module_zero(self, kpi):
        result = forge(
            kpi,
            run_market_context=False,
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.market_context is None
        assert REGIME_COL not in result.enriched.columns

    def test_run_rule_discovery_false_stops_after_alpha(self, kpi):
        result = forge(
            kpi, event_discovery_config=_FAST_ED_CONFIG, run_rule_discovery=False
        )
        assert result.rule_responses == []
        assert len(result.candidates) > 0
        assert result.alpha_discovery is not None
        # Modulo 4 is skipped when Rule Discovery did not run.
        assert result.registry is None

    def test_run_registry_false_skips_module_four(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            run_registry=False,
        )
        assert result.registry is None
        # Rule Discovery still ran.
        assert len(result.rule_responses) == len(result.promoted)

    def test_default_alpha_config_carries_metadata(self, kpi):
        result = forge(
            kpi,
            asset="MYCOIN",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.alpha_discovery.config.asset == "MYCOIN"
        assert result.alpha_discovery.config.timeframe == "4H"

    def test_explicit_alpha_config_is_respected(self, kpi):
        cfg = AlphaConfig(asset="EXPLICIT", timeframe="1D")
        result = forge(
            kpi,
            asset="IGNORED",
            alpha_config=cfg,
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.alpha_discovery.config.asset == "EXPLICIT"
        assert result.alpha_discovery.config.timeframe == "1D"

    def test_edges_and_validated_rules_are_consistent(self, full_result):
        for contract, response in full_result.edges():
            assert response.is_edge
        for response in full_result.validated_rules():
            assert response.validated_rule is not None


class TestForgeGradeFilter:
    """rule_discovery_grades restricts which alphas reach Rule Discovery."""

    def test_filter_limits_rule_discovery_to_selected_grades(self):
        kpi = _ohlc_kpi_table()
        # Baseline: every promoted contract is backtested.
        baseline = forge(
            kpi,
            asset="TEST",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            progress=False,
        )
        assert len(baseline.rule_responses) == len(baseline.promoted)

        grades = {c.alpha_score.grade for c in baseline.promoted}
        keep = sorted(grades)[:1]  # keep just one of the present grades
        expected = [c for c in baseline.promoted if c.alpha_score.grade in keep]

        filtered = forge(
            kpi,
            asset="TEST",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            rule_discovery_grades=keep,
            progress=False,
        )
        # Only the selected-grade contracts were backtested…
        assert len(filtered.rule_responses) == len(expected)
        backtested = {c.alpha_id for c, _ in filtered.rule_responses}
        assert backtested == {c.alpha_id for c in expected}
        # …yet the full promoted/contracts lists are preserved for audit.
        assert len(filtered.promoted) == len(baseline.promoted)
        # Nothing of an excluded grade slipped through to a response.
        for contract, _ in filtered.rule_responses:
            assert contract.alpha_score.grade in keep

    def test_filter_is_case_insensitive(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            asset="TEST",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            rule_discovery_grades=["a", "b", "c", "d"],
            progress=False,
        )
        # Every grade accepted (any case) ⇒ same as no filter.
        assert len(result.rule_responses) == len(result.promoted)

    def test_empty_grade_set_skips_all_rule_discovery(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            asset="TEST",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            rule_discovery_grades=[],
            progress=False,
        )
        assert result.rule_responses == []
        # Registry still ran, just with nothing tradeable to catalogue.
        assert isinstance(result.registry, RuleRegistry)
        assert result.registry.documents == []


class TestForgeMulti:
    def test_pools_tickers_into_one_cross_ticker_registry(self):
        frames = {
            "BTCUSDC": _ohlc_kpi_table(seed=7),
            "ETHUSDC": _ohlc_kpi_table(seed=11),
        }
        results, registry = forge_multi(
            frames,
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
        )

        assert set(results) == {"BTCUSDC", "ETHUSDC"}
        # Per-ticker registries are skipped in favour of the pooled one.
        for res in results.values():
            assert res.registry is None
            assert res.ticker in frames
        # The pooled registry knows about every ticker's frame for cross-ticker.
        assert isinstance(registry, RuleRegistry)
        assert set(registry.frames) == {"BTCUSDC", "ETHUSDC"}
        # Every ingested document traces back to a session ticker.
        for doc in registry.documents:
            assert doc.source_ticker in frames


class TestForgeManualEvents:
    """Custom Event Injection — forge(manual_events=...) (issue #77)."""

    def test_mutual_exclusion_raises(self):
        """Passing both manual_events and event_discovery_config → ValueError."""
        kpi = _ohlc_kpi_table()
        with pytest.raises(ValueError, match="mutually exclusive"):
            forge(
                kpi,
                manual_events=[CustomEvent("close < 100")],
                event_discovery_config=_FAST_ED_CONFIG,
            )

    def test_manual_event_end_to_end(self):
        """forge(manual_events=[...]) skips M1 and runs M2/M3 on the injected event."""
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            asset="TEST",
            timeframe="4H",
            manual_events=[CustomEvent("feat < 0.5", name="feat_low")],
            rule_discovery_config=_FAST_RD_CONFIG,
        )
        assert isinstance(result, ForgeResult)
        # M1 was skipped — no EventDiscovery instance.
        assert result.event_discovery is None
        # The injected event became the sole candidate, carrying the formula.
        assert len(result.candidates) == 1
        cand = result.candidates[0]
        assert cand.event_id == "CUSTOM-feat_low"
        assert cand.expression == "feat < 0.5"
        # M2 evaluated it (one contract per candidate).
        assert len(result.contracts) == 1
        assert result.alpha_discovery is not None

    def test_gate_failure_does_not_block(self, caplog):
        """An event that fails the Consistency Gate still reaches M2, with a warning."""
        import logging

        kpi = _ohlc_kpi_table()
        # An almost-never-true formula fails the gate's volume criterion.
        with caplog.at_level(logging.WARNING, logger="forgedge.forge"):
            result = forge(
                kpi,
                asset="TEST",
                timeframe="4H",
                manual_events=[CustomEvent("feat < 0.0001", name="rare")],
                rule_discovery_config=_FAST_RD_CONFIG,
            )
        assert len(result.candidates) == 1
        assert not result.candidates[0].consistency_gate.passed
        assert any("failed ConsistencyGate" in rec.message for rec in caplog.records)


class TestForgeProgress:
    """Status logging and the optional stderr progress output."""

    _MANUAL = [CustomEvent("feat < 0.4", name="feat_low")]

    def test_logs_stage_milestones_at_info(self, caplog):
        import logging

        kpi = _ohlc_kpi_table()
        with caplog.at_level(logging.INFO, logger="forgedge.forge"):
            forge(kpi, manual_events=self._MANUAL, rule_discovery_config=_FAST_RD_CONFIG)
        msgs = [r.getMessage() for r in caplog.records]
        for needle in ("M0 Market Context", "M2 Alpha Discovery", "M3 Rule Discovery", "done"):
            assert any(needle in m for m in msgs), needle

    def test_progress_false_is_silent_on_stderr(self, capsys):
        kpi = _ohlc_kpi_table()
        forge(
            kpi,
            manual_events=self._MANUAL,
            rule_discovery_config=_FAST_RD_CONFIG,
            progress=False,
        )
        assert "[forge" not in capsys.readouterr().err

    def test_progress_true_prints_stages_to_stderr(self, capsys):
        kpi = _ohlc_kpi_table()
        forge(
            kpi,
            ticker="BTCUSDC",
            manual_events=self._MANUAL,
            rule_discovery_config=_FAST_RD_CONFIG,
            progress=True,
        )
        err = capsys.readouterr().err
        assert "forge:BTCUSDC" in err
        assert "M3 Rule Discovery" in err
        assert "done" in err
