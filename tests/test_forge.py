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
    pytestmark = pytest.mark.slow

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
        # 4H matches the fixture's own spacing — the timeframe is incidental
        # to what this test checks, and declaring it honestly keeps the
        # declared-vs-measured check (#179) quiet.
        cfg = AlphaConfig(asset="EXPLICIT", timeframe="4H")
        # What the caller *set* is respected verbatim; what they left unset is
        # resolved from the session. Before #196 an explicit config on a slow
        # timeframe kept the hourly grid and merely warned — the config was
        # respected a little too literally.
        result = forge(
            kpi,
            asset="IGNORED",
            timeframe="4H",          # declared, and matching cfg.timeframe
            alpha_config=cfg,
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.alpha_discovery.config.asset == "EXPLICIT"
        assert result.alpha_discovery.config.timeframe == "4H"
        assert result.alpha_discovery.config.horizon_grid == (1, 2, 4, 8, 12, 24)

    def test_edges_and_validated_rules_are_consistent(self, full_result):
        for contract, response in full_result.edges():
            assert response.is_edge
        for response in full_result.validated_rules():
            assert response.validated_rule is not None


class TestForgeTimeframeScaledHorizons:
    """The horizon grid is calibrated to the session's bar class.

    The old class default was calibrated on ~hourly bars; using it verbatim on
    daily data means holding periods of up to 48 days (the "silent footgun" of
    docs/analysis/lowfreq_robustness.md).  `forge()` used to substitute the
    daily grid only when no explicit `alpha_config` was passed; since #196 the
    resolver derives it on every path, so the substitution — and the warning
    that stood in for it on the other path — are both gone.
    """
    pytestmark = pytest.mark.slow

    @staticmethod
    def _daily_kpi(n: int = 1000) -> pd.DataFrame:
        df = _ohlc_kpi_table(n=n)
        df["open_dt"] = pd.date_range("2022-01-01", periods=n, freq="D")
        return df

    def test_default_config_on_daily_uses_scaled_grid(self):
        from forgedge.presets import default_horizon_grid

        result = forge(
            self._daily_kpi(),
            ticker="DAILY",
            timeframe="1D",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        grid = result.alpha_discovery.config.horizon_grid
        assert grid == default_horizon_grid("1D")
        assert max(grid) <= 10  # days, not the 48-bar hourly default

    def test_intraday_gets_the_intraday_class_grid(self):
        """4H is in the same class as 1H, so it scans the same horizons — up to
        24 bars. That used to be "the default, left untouched"; it is now the
        class grid, derived rather than inherited."""
        from forgedge.presets import _TFClass

        result = forge(
            _ohlc_kpi_table(),
            ticker="INTRA",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        grid = result.alpha_discovery.config.horizon_grid
        assert grid == _TFClass("4H").horizon_grid == (1, 2, 4, 8, 12, 24)

    def test_explicit_custom_grid_on_daily_does_not_warn(self):
        import warnings as _warnings

        cfg = AlphaConfig(asset="X", timeframe="1D", horizon_grid=(1, 2, 3, 5))
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = forge(
                self._daily_kpi(),
                alpha_config=cfg,
                event_discovery_config=_FAST_ED_CONFIG,
                run_rule_discovery=False,
            )
        assert not [w for w in caught if "horizon_grid" in str(w.message)]
        assert result.alpha_discovery.config.horizon_grid == (1, 2, 3, 5)

    def test_default_horizon_grid_helper(self):
        from forgedge.presets import default_horizon_grid

        assert default_horizon_grid("1D") == (1, 2, 3, 5, 7, 10)
        assert default_horizon_grid("3D") == (1, 2, 3, 5, 7, 10)
        assert default_horizon_grid("1W") == (1, 2, 3, 5, 7, 10)
        assert default_horizon_grid("1H") is None
        assert default_horizon_grid("15m") is None
        assert default_horizon_grid("junk") is None


class TestForgeFastNullAndLedger:
    """Default fast rotation null + hypothesis ledger wiring."""
    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def result(self):
        return forge(
            _ohlc_kpi_table(),
            ticker="SYN",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            progress=False,
        )

    def test_fast_null_runs_by_default(self, result):
        assert result.calibration is not None
        assert result.calibration.tippett_best_stat == "abs_z"
        for c in result.promoted:
            assert c.rotation_p is not None
            assert c.rotation_threshold is not None

    def test_fast_null_off_skips_annotation(self):
        res = forge(
            _ohlc_kpi_table(),
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
            fast_null=False,
            progress=False,
        )
        assert res.calibration is None
        assert all(c.rotation_p is None for c in res.promoted)

    def test_ledger_records_surface(self, result):
        led = result.ledger
        assert led is not None
        assert led.m1_candidates == len(result.candidates)
        assert led.m2_horizons == len(result.alpha_discovery.config.horizon_grid)
        assert led.m2_promoted == len(result.promoted)
        if result.rule_responses:
            assert led.m3_grid_cells == len(result.rule_responses[0][1].grid_results)
        # With the per-event horizon enrichment the exact return-test count is
        # recorded and is at least the uniform candidates × base-grid product.
        assert led.m2_return_tests == result.alpha_discovery.n_return_tests
        assert led.m2_surface >= led.m1_candidates * led.m2_horizons
        assert str(led.m1_candidates) in led.describe()

    def test_full_edge_requires_clearing_the_null(self, result):
        # Any full-EDGE verdict must come from a contract that cleared the
        # search-level null bar; capped rules carry the reason.
        for contract, resp in result.rule_responses:
            if resp.verdict == "EDGE":
                assert contract.rotation_p <= 0.05
            elif any("rotation null" in r for r in resp.rejection_reasons):
                assert contract.rotation_p > 0.05


class TestForgeGradeFilter:
    """rule_discovery_grades restricts which alphas reach Rule Discovery."""
    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def kpi(self):
        """forge() is a pure function of its inputs (never mutates the KPI
        table), so the identical default table below is safe to share."""
        return _ohlc_kpi_table()

    def test_filter_limits_rule_discovery_to_selected_grades(self, kpi):
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

    def test_filter_is_case_insensitive(self, kpi):
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

    def test_empty_grade_set_skips_all_rule_discovery(self, kpi):
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


class TestForgeTwoPassComposition:
    """two_pass_composition (issue #254): grade-guided event composition."""
    pytestmark = pytest.mark.slow

    def test_default_off_never_calls_grade_guided_compose(self, monkeypatch):
        """Off-by-default: with two_pass_composition left at its default
        (False), forge() must not even import/invoke the composition stage,
        and the two-pass-only ForgeResult fields must stay None."""
        import importlib

        # `import forgedge.forge as forge_module` would silently bind to the
        # `forge()` *function* instead of the module: forgedge/__init__.py's
        # `from .forge import forge` rebinds the `forge` attribute on the
        # `forgedge` package to the function, and `pkg.submodule` attribute
        # access is what `import pkg.submodule as x` resolves through.
        forge_module = importlib.import_module("forgedge.forge")

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError(
                "grade_guided_compose must not be called when two_pass_composition=False"
            )
        monkeypatch.setattr(forge_module, "grade_guided_compose", _must_not_be_called)

        result = forge(
            _ohlc_kpi_table(),
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            progress=False,
        )
        assert result.grading_candidates is None
        assert result.grading_contracts is None
        assert result.composition_timing is None

    def test_max_and_components_conflict_raises(self):
        with pytest.raises(ValueError, match="max_and_components"):
            forge(
                _ohlc_kpi_table(),
                timeframe="4H",
                event_discovery_config=DiscoveryConfig(max_and_components=2),
                two_pass_composition=True,
                progress=False,
            )

    def test_max_and_components_conflict_raises_on_omitted_event_discovery_config(self):
        """An OMITTED event_discovery_config still resolves to DiscoveryConfig()'s
        own class default (max_and_components=2), which must be rejected too --
        two_pass_composition can't silently coexist with the legacy structural
        composer just because the caller never set the field explicitly."""
        with pytest.raises(ValueError, match="max_and_components"):
            forge(_ohlc_kpi_table(), timeframe="4H", two_pass_composition=True, progress=False)

    def test_manual_events_mode_is_exempt_from_the_max_and_components_check(self):
        """manual_events and event_discovery_config are already mutually
        exclusive -- the max_and_components validation must not fire on a
        None event_discovery_config that only exists because manual events
        were used instead of automatic discovery."""
        kpi = _ohlc_kpi_table(n=600)
        events = [CustomEvent("feat < 0.3")]
        # Must not raise -- reaches AlphaDiscovery's grading pass instead.
        result = forge(
            kpi, timeframe="4H", manual_events=events,
            two_pass_composition=True, run_rule_discovery=False, progress=False,
        )
        assert result.grading_candidates is not None

    def test_composed_contracts_reach_m3_and_pass_two_pools_correctly(self):
        """The concrete regression test for the wiring bug found during
        design review (docs/analysis/issue_254_two_pass_composition_plan.md):
        M3's candidate lookup is built from `candidates`, not
        `alpha_candidates` -- rebinding only the latter would leave every
        composed contract's event_candidate_id unresolvable, and
        RuleDiscovery would silently skip it. This proves at least one
        composed contract actually gets backtested, not just promoted."""
        from forgedge.composition import GradePairingConfig

        kpi = _ohlc_kpi_table(n=2600, seed=11)
        result = forge(
            kpi,
            timeframe="4H",
            event_discovery_config=DiscoveryConfig(
                max_and_components=1, gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
            ),
            rule_discovery_config=_FAST_RD_CONFIG,
            two_pass_composition=True,
            grade_pairing_config=GradePairingConfig(per_stratum_pair_cap=20, per_stratum_triple_cap=10),
            progress=False,
        )

        # Pass-1 artefacts preserved for audit.
        assert result.grading_candidates is not None
        assert result.grading_contracts is not None
        assert len(result.grading_candidates) == len(result.grading_contracts)
        grading_ids = {c.event_id for c in result.grading_candidates}
        assert grading_ids <= {c.event_id for c in result.candidates}

        # Pass 2's pool is strictly larger: composition actually added candidates.
        composed_ids = {c.event_id for c in result.candidates} - grading_ids
        assert composed_ids, "grade-guided composition must add candidates on this fixture"

        assert result.composition_timing is not None
        assert set(result.composition_timing) == {
            "pass1_seconds", "composition_seconds", "pass2_seconds",
        }
        assert all(v >= 0 for v in result.composition_timing.values())

        # The rebinding fix itself: a composed contract must reach Rule
        # Discovery, not be silently dropped.
        composed_responses = [
            (c, r) for c, r in result.rule_responses if c.event_candidate_id in composed_ids
        ]
        assert composed_responses, "at least one composed contract must reach Rule Discovery"

    def test_ledger_reports_two_pass_surface(self):
        kpi = _ohlc_kpi_table(n=2600, seed=11)
        result = forge(
            kpi,
            timeframe="4H",
            event_discovery_config=DiscoveryConfig(
                max_and_components=1, gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
            ),
            run_rule_discovery=False,
            two_pass_composition=True,
            progress=False,
        )
        assert result.ledger.m2_pass1_candidates == len(result.grading_candidates)
        assert result.ledger.m1_candidates == len(result.candidates)
        assert "two-pass" in result.ledger.describe()

    def test_include_singles_in_pass2_false_drops_the_originals(self):
        from forgedge.composition import GradePairingConfig

        kpi = _ohlc_kpi_table(n=2600, seed=11)
        result = forge(
            kpi,
            timeframe="4H",
            event_discovery_config=DiscoveryConfig(
                max_and_components=1, gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
            ),
            run_rule_discovery=False,
            two_pass_composition=True,
            grade_pairing_config=GradePairingConfig(
                per_stratum_pair_cap=20, include_singles_in_pass2=False,
            ),
            progress=False,
        )
        grading_ids = {c.event_id for c in result.grading_candidates}
        pooled_ids = {c.event_id for c in result.candidates}
        assert pooled_ids.isdisjoint(grading_ids), (
            "include_singles_in_pass2=False must exclude the original 1D pool "
            "from pass 2 entirely"
        )


class TestForgeMulti:
    pytestmark = pytest.mark.slow
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
    pytestmark = pytest.mark.slow

    @pytest.fixture(scope="class")
    def kpi(self):
        """forge() is a pure function of its inputs (never mutates the KPI
        table), so the identical default table below is safe to share."""
        return _ohlc_kpi_table()

    def test_mutual_exclusion_raises(self, kpi):
        """Passing both manual_events and event_discovery_config → ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            forge(
                kpi,
                manual_events=[CustomEvent("close < 100")],
                event_discovery_config=_FAST_ED_CONFIG,
            )

    def test_manual_event_end_to_end(self, kpi):
        """forge(manual_events=[...]) skips M1 and runs M2/M3 on the injected event."""
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

    def test_gate_failure_does_not_block(self, kpi, caplog):
        """An event that fails the Consistency Gate still reaches M2, with a warning."""
        import logging

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
    pytestmark = pytest.mark.slow

    _MANUAL = [CustomEvent("feat < 0.4", name="feat_low")]

    @pytest.fixture(scope="class")
    def kpi(self):
        """forge() is a pure function of its inputs (never mutates the KPI
        table), so the identical default table below is safe to share."""
        return _ohlc_kpi_table()

    def test_logs_stage_milestones_at_info(self, kpi, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="forgedge.forge"):
            forge(kpi, manual_events=self._MANUAL, rule_discovery_config=_FAST_RD_CONFIG)
        msgs = [r.getMessage() for r in caplog.records]
        for needle in ("M0 Market Context", "M2 Alpha Discovery", "M3 Rule Discovery", "done"):
            assert any(needle in m for m in msgs), needle

    def test_progress_false_is_silent_on_stderr(self, kpi, capsys):
        forge(
            kpi,
            manual_events=self._MANUAL,
            rule_discovery_config=_FAST_RD_CONFIG,
            progress=False,
        )
        assert "[forge" not in capsys.readouterr().err

    def test_progress_true_prints_stages_to_stderr(self, kpi, capsys):
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

    def test_m1_log_uses_event_distribution_report(self, kpi, caplog):
        """#215 — the M1 stage line carries the distribution diagnostic, not a
        bare count, whenever Event Discovery actually ran (not manual injection)."""
        import logging

        with caplog.at_level(logging.INFO, logger="forgedge.forge"):
            result = forge(
                kpi,
                event_discovery_config=_FAST_ED_CONFIG,
                rule_discovery_config=_FAST_RD_CONFIG,
            )
        report = result.event_discovery.event_distribution_report
        assert report is not None
        msgs = [r.getMessage() for r in caplog.records]
        assert any(report in m for m in msgs)

    def test_manual_events_m1_log_is_bare_count(self, kpi, caplog):
        """Manual event injection skips EventDiscovery.run() entirely (#215's
        aggregation only exists there), so the M1 line keeps its original,
        bare-count form instead of a distribution report."""
        import logging

        with caplog.at_level(logging.INFO, logger="forgedge.forge"):
            forge(kpi, manual_events=self._MANUAL, rule_discovery_config=_FAST_RD_CONFIG)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("M1 Event Discovery — 1 candidate(s)" in m for m in msgs)
        assert not any("Consistency Gate (" in m for m in msgs)


class TestWalkForwardConfigNaming:
    """The two ``WalkForwardConfig`` classes are distinct and both reachable.

    Until they were renamed, Event Discovery and Rule Discovery each exported a
    dataclass called ``WalkForwardConfig``.  They are different types with a
    ``n_splits`` field carrying different semantics — OOS *validation* windows
    in M1, walk-forward *test* windows in M3 — yet ``forge()``'s own docstring
    example imported one while the top-level ``forgedge.WalkForwardConfig``
    silently resolved to the other.  These tests pin the disambiguation and the
    backwards-compatible aliases.
    """

    def test_the_two_configs_are_distinct_types(self):
        from forgedge import EventWalkForwardConfig, RuleWalkForwardConfig

        assert EventWalkForwardConfig is not RuleWalkForwardConfig
        # The field that made the collision dangerous: same name, different
        # meaning, different default.
        assert EventWalkForwardConfig().n_splits == 3
        assert RuleWalkForwardConfig().n_splits == 4
        assert hasattr(EventWalkForwardConfig(), "min_pass_rate")
        assert hasattr(RuleWalkForwardConfig(), "min_train_months")

    def test_legacy_module_aliases_still_resolve(self):
        """Old imports keep working, each to its own module's class."""
        from forgedge.event_discovery.models import (
            EventWalkForwardConfig,
            WalkForwardConfig as LegacyEventWF,
        )
        from forgedge.rule_discovery.models import (
            RuleWalkForwardConfig,
            WalkForwardConfig as LegacyRuleWF,
        )

        assert LegacyEventWF is EventWalkForwardConfig
        assert LegacyRuleWF is RuleWalkForwardConfig

    def test_top_level_alias_keeps_resolving_to_rule_discovery(self):
        """``forgedge.WalkForwardConfig`` has always been the M3 one — it stays
        that way, so existing top-level imports do not silently change type."""
        import forgedge

        assert forgedge.WalkForwardConfig is forgedge.RuleWalkForwardConfig

    def test_legacy_names_still_configure_the_pipeline(self):
        """A config built with the legacy names behaves identically."""
        from forgedge.event_discovery.models import WalkForwardConfig as LegacyEventWF
        from forgedge.rule_discovery.models import WalkForwardConfig as LegacyRuleWF

        disc = DiscoveryConfig(
            train_ratio=0.8, walk_forward=LegacyEventWF(n_splits=2, min_pass_rate=0.5)
        )
        rd = RuleDiscoveryConfig(walk_forward=LegacyRuleWF(n_splits=2, reoptimise=False))
        assert disc.walk_forward.n_splits == 2
        assert rd.walk_forward.reoptimise is False


class TestForgeResolution:
    """``forge()`` resolves the configuration once, and says what it did.

    The session context is seeded from whatever the caller set explicitly, so a
    schema chosen on one module reaches the others — and the trace makes the
    decision auditable after the fact, next to ``ledger.describe()``.
    """
    pytestmark = pytest.mark.slow

    def test_result_carries_the_context_and_the_trace(self):
        # `_ohlc_kpi_table` is 4H-spaced, and since #179 declaring anything
        # else here would (correctly) raise `timeframe_mismatch`.
        kpi = _ohlc_kpi_table(n=900, seed=3)
        result = forge(
            kpi, ticker="X", timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False, run_registry=False,
            fast_null=False, progress=False,
        )
        assert result.context is not None
        assert result.context.timeframe == "4H"
        assert result.context.timeframe_declared
        assert result.context.inferred_bar_hours == pytest.approx(4.0)
        assert result.context.n_bars == len(kpi)
        assert result.context.span_months > 0
        assert result.resolution is not None
        assert "derived" in result.resolution.describe()
        # `coherence` is the report produced by the same resolver call the
        # pipeline ran with — so `coherence.configs` is what executed.
        assert result.coherence is not None
        assert not result.coherence.has_critical
        # The module re-resolves defensively on construction, and resolve()
        # returns copies — so the guarantee is equality of what will run,
        # not object identity.
        assert (result.coherence.configs["alpha"].timestamp_col
                == result.alpha_discovery.config.timestamp_col)

    def test_a_schema_set_on_one_module_reaches_the_others(self):
        """The F10 case, end to end: Event Discovery's column is collected into
        the session context and distributed to Alpha and Rule Discovery."""
        kpi = _ohlc_kpi_table(n=900, seed=3).rename(columns={"open_dt": "ts"})
        disc = DiscoveryConfig(
            max_and_components=1,
            gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
            timestamp_col="ts",
        )
        result = forge(
            kpi, ticker="X", timeframe="1D",
            event_discovery_config=disc,
            rule_discovery_config=_FAST_RD_CONFIG,
            run_registry=False, fast_null=False, progress=False,
        )
        assert result.alpha_discovery.config.timestamp_col == "ts"
        derived = {d.field for d in result.resolution.effective}
        assert "alpha.timestamp_col" in derived
        assert "rule_discovery.timestamp_col" in derived
