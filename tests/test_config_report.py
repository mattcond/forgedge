"""Tests for ``config_report`` — the coherence half of the resolver.

One test per constraint, each pinning both sides: the coherent configuration
stays silent, the incoherent one is named with the value to set. Plus the
property the whole thing rests on — the report and ``forge()`` resolve through
the same call, so what is shown is what runs.

No pipeline runs here: every check is a pure function of the configs plus the
session context, which is exactly why they are cheap enough to run before
anything starts.
"""
import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    ConfigReport,
    DiscoveryConfig,
    PipelineContext,
    RegistryConfig,
    RuleDiscoveryConfig,
    config_report,
)
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams
from forgedge.rule_discovery.models import (
    BacktestParams,
    RuleWalkForwardConfig,
    ScoringParams,
    SelectionCriteria,
)


def _ctx(span_months: float = 36.0, timeframe: str = "1D", **kw) -> PipelineContext:
    """A context with a realistic span — the FAIL checks compare against it."""
    return PipelineContext(timeframe=timeframe, span_months=span_months,
                           n_bars=int(span_months * 30), **kw)


def _codes(report: ConfigReport) -> set:
    return {f.code for f in report.findings}


def _message(report: ConfigReport, code: str) -> str:
    return next(f.message for f in report.findings if f.code == code)


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------

class TestSameResolverGuarantee:
    """What the report shows is what the pipeline will execute — verified, not
    assumed. The report and ``forge()`` go through the same resolver call, so
    there are not two implementations to keep in step."""

    def test_the_report_carries_the_configs_that_will_run(self):
        disc = DiscoveryConfig(timestamp_col="ts")
        rep = config_report(disc, AlphaConfig(), RuleDiscoveryConfig(),
                            RegistryConfig(), ctx=None, timeframe="1D")
        # Resolved copies, not the caller's objects, and not placeholders.
        assert rep.configs["alpha"].timestamp_col == "ts"
        assert rep.configs["rule_discovery"].timestamp_col == "ts"
        assert rep.configs["alpha"] is not None

    def test_inspecting_a_config_does_not_modify_it(self):
        alpha = AlphaConfig()
        config_report(None, alpha, None, None, ctx=_ctx())
        from forgedge import UNSET

        assert alpha.timestamp_col is UNSET

    def test_the_stock_preset_on_daily_data_is_coherent(self):
        """The inverse of what this test asserted between steps 3 and 6.

        `forge_preset("balanced", "1D")` used to be flagged, and that was the
        finding rather than a bug in the check (F2): at a fixed
        `min_train_months=6` the floor `max(10, 6 × 0.80) = 10` implied 1.67
        trades/month against a configured `criteria.min_tpm=0.80` — 2.1x the
        rate the preset asked for, inverting the ordering `presets.py`
        documents, and the reason daily-data users saw mass early elimination.

        Step 7 derives the window from the rate instead: 20 months at this rate,
        with a 95 % Poisson margin. The naive `10 / 0.80 = 12.5` would satisfy
        the floor *in expectation* and come up short about 44 % of the time —
        #173 again, in milder form, after having been "fixed".
        """
        disc, alpha, rd = _preset_triple()
        rep = config_report(disc, alpha, rd, RegistryConfig(), ctx=_ctx())
        resolved = rep.configs["rule_discovery"]

        assert resolved.walk_forward.min_train_months == 20
        assert resolved.walk_forward.min_train_months * resolved.criteria.min_tpm >= 10
        assert "wf_bucket_too_short" not in _codes(rep)


def _preset_triple():
    from forgedge.presets import forge_preset

    return forge_preset("balanced", "1D", asset="X")


# ---------------------------------------------------------------------------
# One test per constraint
# ---------------------------------------------------------------------------

class TestFailConstraints:
    """FAIL is reserved for a configuration that makes a stage structurally
    incapable of producing a verdict."""

    def test_wf_bucket_too_short_is_issue_173(self):
        """The reported case, now reachable only by writing it out.

        Left alone, `min_train_months` is derived from the rate (#177) and the
        pair cannot disagree. A caller who pins a window *and* a rate that do
        not fit still gets the FAIL — the check half of the same constraint.
        """
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=0.20),
            walk_forward=RuleWalkForwardConfig(min_train_months=6),
        )
        rep = config_report(None, None, rd, ctx=_ctx(span_months=36))

        assert "wf_bucket_too_short" in _codes(rep)
        assert rep.has_critical
        msg = _message(rep, "wf_bucket_too_short")
        assert "#173" in msg
        # The message must carry the value to set, not just the failure.
        assert "0.28" in msg          # 10 / 36 months of history
        assert "36" in msg

    def test_the_window_is_derived_from_the_rate(self):
        """The fix, not the check: an unset window is sized to the rate it is
        about to demand, with a 95 % Poisson margin rather than the naive
        `floor / rate` — which is satisfied in expectation and short about
        44 % of the time."""
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=0.80))
        rep = config_report(None, None, rd, ctx=_ctx(span_months=36))

        assert rep.configs["rule_discovery"].walk_forward.min_train_months == 20
        assert "wf_bucket_too_short" not in _codes(rep)

    def test_an_explicit_window_that_fits_is_left_alone(self):
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=2.0),
            walk_forward=RuleWalkForwardConfig(min_train_months=6),
        )
        rep = config_report(None, None, rd, ctx=_ctx())
        assert rep.configs["rule_discovery"].walk_forward.min_train_months == 6
        assert "wf_bucket_too_short" not in _codes(rep)

    def test_wf_bucket_ignores_full_sample_selection(self):
        """The bucket only exists in walk-forward mode; in full_sample the
        floor is measured against the whole in-sample span."""
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=0.20), selection_mode="full_sample",
        )
        rep = config_report(None, None, rd, ctx=_ctx())
        assert "wf_bucket_too_short" not in _codes(rep)

    def test_m1_oos_fold_too_short(self):
        """F1 — folds that cannot conclude anything at the configured rate.

        The configuration `forge`'s own docstring recommends for production:
        `train_ratio=0.80, n_splits=3` leaves ~2-month folds, and at 1.0
        episodes/month an empty fold has probability `e**-2 = 14%` even for a
        healthy candidate.  Every fold comes back INDETERMINATE and the
        walk-forward concludes nothing — a property of the configuration, said
        once before running rather than discovered as thousands of fold
        failures.
        """
        disc = DiscoveryConfig(
            train_ratio=0.80,
            walk_forward=EventWalkForwardConfig(n_splits=3),
            gate_params=GateParams(min_tpm=1.0),
        )
        rep = config_report(disc, None, None, ctx=_ctx(span_months=29))

        assert "m1_oos_fold_too_short" in _codes(rep)
        msg = _message(rep, "m1_oos_fold_too_short")
        assert "n_splits" in msg          # names the value to set
        assert "INDETERMINATO" in msg
        assert "%" in msg                 # states P(empty fold)

    def test_m1_oos_fold_is_silent_when_the_folds_can_conclude(self):
        """Fewer splits, longer folds: at 1.0 episodes/month a single fold over
        a 5.8-month OOS expects 5.8 episodes, comfortably testable."""
        disc = DiscoveryConfig(
            train_ratio=0.80,
            walk_forward=EventWalkForwardConfig(n_splits=1),
            gate_params=GateParams(min_tpm=1.0),
        )
        rep = config_report(disc, None, None, ctx=_ctx(span_months=29))
        assert "m1_oos_fold_too_short" not in _codes(rep)

    def test_m1_oos_fold_is_silent_without_walk_forward(self):
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=1.0))
        rep = config_report(disc, None, None, ctx=_ctx())
        assert "m1_oos_fold_too_short" not in _codes(rep)

    def test_oos_span_too_short(self):
        """F4 — masked by #173 until step 7, because NON-EDGE is never rescued.

        The pooled test span cannot supply `min_oos_trades`, so every positive
        verdict is degraded rather than the candidate being blamed.
        """
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=0.20, min_oos_trades=10),
            walk_forward=RuleWalkForwardConfig(min_train_months=6),
        )
        rep = config_report(None, None, rd, ctx=_ctx(span_months=36))

        assert "oos_span_too_short" in _codes(rep)
        assert "INSUFFICIENT-DATA" in _message(rep, "oos_span_too_short")

    def test_a_derived_window_longer_than_the_history_says_so(self):
        """Where the #173 configuration lands once the window is derived: a
        permissive rate now asks for a *selection span* longer than the data,
        which is the same problem stated more accurately — an unreachable floor
        is an insufficient window, not a rejected candidate.

        The message has to carry the way out, like every other one.
        """
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=0.20))
        rep = config_report(None, None, rd, ctx=_ctx(span_months=36))
        msg = _message(rep, "oos_span_too_short")

        assert rep.configs["rule_discovery"].walk_forward.min_train_months > 36
        assert "nessuna finestra di test" in msg
        assert "#173" in msg
        assert "min_tpm" in msg

    def test_oos_span_flags_a_train_window_that_eats_the_history(self):
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=3.0),
            walk_forward=RuleWalkForwardConfig(min_train_months=40),
        )
        rep = config_report(None, None, rd, ctx=_ctx(span_months=36))
        assert "nessuna finestra di test" in _message(rep, "oos_span_too_short")


class TestWarnConstraints:
    def test_m3_stricter_than_m1(self):
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=1.0))
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=3.0))
        rep = config_report(disc, None, rd, ctx=_ctx())

        assert "m3_stricter_than_m1" in _codes(rep)
        # Silent the other way round: forge_preset deliberately keeps M3 looser.
        rd_ok = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=0.8))
        assert "m3_stricter_than_m1" not in _codes(
            config_report(disc, None, rd_ok, ctx=_ctx()))

    def test_scoring_uncalibrated(self):
        """F3 — pf_score_tpm is the grid's selection criterion, so a target
        calibrated on a different frequency steers which operating point gets
        published."""
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=0.25),
            scoring=ScoringParams(pf_min_tpm=2, pf_tpm_target=3),
        )
        rep = config_report(None, None, rd, ctx=_ctx())

        assert "scoring_uncalibrated" in _codes(rep)
        msg = _message(rep, "scoring_uncalibrated")
        assert "pf_min_tpm" in msg and "pf_tpm_target" in msg

    def test_scoring_is_silent_when_the_knobs_track_the_rate(self):
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=2.5),
            scoring=ScoringParams(pf_min_trades=15, pf_min_tpm=2, pf_tpm_target=3),
        )
        rep = config_report(None, None, rd, ctx=_ctx())
        assert "scoring_uncalibrated" not in _codes(rep)

    def test_timeframe_mismatch_catches_the_declared_label(self):
        alpha = AlphaConfig(timeframe="1H", horizon_grid=(1, 2, 3))
        rep = config_report(None, alpha, None, ctx=_ctx(timeframe="1D"))
        assert "timeframe_mismatch" in _codes(rep)

    def test_timeframe_mismatch_catches_hourly_bar_counts_on_daily_data(self):
        """The footgun already fixed for horizon_grid, left unfixed one module
        later: target_h=24 means 24 *days* on 1D bars.

        Only the untouched default is flagged — a caller who wrote a bar count
        deliberately does not need telling."""
        alpha = AlphaConfig(timeframe="1D", horizon_grid=(1, 2, 3))
        rep = config_report(None, alpha, RuleDiscoveryConfig(),
                            ctx=_ctx(timeframe="1D"))
        msg = _message(rep, "timeframe_mismatch")
        assert "target_h" in msg and "giorni" in msg

        # Chosen values are the caller's business.
        chosen = RuleDiscoveryConfig(
            base_params=BacktestParams(target_h=3, buy_delay_bar=1))
        assert "timeframe_mismatch" not in _codes(
            config_report(None, alpha, chosen, ctx=_ctx(timeframe="1D")))

    def test_split_disagreement(self):
        disc = DiscoveryConfig(train_ratio=0.8)
        alpha = AlphaConfig(train_ratio=0.7)
        rep = config_report(disc, alpha, None, ctx=_ctx())
        assert "split_disagreement" in _codes(rep)

    def test_split_is_silent_when_m1_deliberately_sees_everything(self):
        """train_ratio=1.0 on M1 is a documented choice, not a mismatch: M1
        never observes the forward return, so seeing the whole table leaks no
        return information (D13)."""
        disc = DiscoveryConfig(train_ratio=1.0)
        alpha = AlphaConfig(train_ratio=0.7)
        rep = config_report(disc, alpha, None, ctx=_ctx())
        assert "split_disagreement" not in _codes(rep)

    def test_registry_stricter_than_m3(self):
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(partial_min_profit_factor=1.5))
        reg = RegistryConfig(cross_pf_threshold=2.0)
        rep = config_report(None, None, rd, reg, ctx=_ctx())

        assert "registry_stricter_than_m3" in _codes(rep)
        assert "PARTIAL-EDGE" in _message(rep, "registry_stricter_than_m3")

    def test_alpha_level_drift(self):
        from forgedge.alpha_discovery.models import PromotionThresholds

        alpha = AlphaConfig(thresholds=PromotionThresholds(max_p_value=0.20))
        rep = config_report(None, alpha, None, ctx=_ctx())
        assert "alpha_level_drift" in _codes(rep)

    def test_the_take_profit_floor_follows_m2(self):
        """M3's clamp was a hardcoded 0.01 against M2's 0.005, so the binding
        constraint was the one the caller could not configure and — on intraday
        bars, where a median-MFE target routinely sits under 1 % — it replaced
        a *derived* target with a constant (F11).  Now it is derived."""
        alpha = AlphaConfig(mfe_floor=0.004)
        rep = config_report(None, alpha, RuleDiscoveryConfig(), ctx=_ctx(timeframe="15m"))

        assert rep.configs["rule_discovery"].criteria.min_sell_pct == pytest.approx(0.004)
        assert "tp_floor_conflict" not in _codes(rep)

    def test_tp_floor_conflict_is_an_explicit_disagreement(self):
        """The only case left: the caller set M3's floor above M2's, so a
        derived target between the two is still replaced by a constant."""
        alpha = AlphaConfig(mfe_floor=0.005)
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_sell_pct=0.02))
        assert "tp_floor_conflict" in _codes(
            config_report(None, alpha, rd, ctx=_ctx(timeframe="15m")))
        # Timeframe no longer decides: the disagreement is the finding.
        assert "tp_floor_conflict" in _codes(
            config_report(None, alpha, rd, ctx=_ctx(timeframe="1D")))

    def test_entry_mode_inert_gate(self):
        """A fill gate the caller *tuned*, that the entry mode makes inert."""
        rd = RuleDiscoveryConfig(
            entry_mode="market", criteria=SelectionCriteria(min_fill_rate=0.65))
        rep = config_report(None, None, rd, ctx=_ctx())

        assert "entry_mode_inert_gate" in _codes(rep)
        # Fully meaningful in limit mode, so silent there.
        rd_limit = RuleDiscoveryConfig(
            entry_mode="limit", criteria=SelectionCriteria(min_fill_rate=0.65))
        assert "entry_mode_inert_gate" not in _codes(
            config_report(None, None, rd_limit, ctx=_ctx()))

    def test_the_inert_gate_is_silent_on_an_untouched_default(self):
        """`entry_mode` now defaults to "auto" (#185), where Stage 1 fills
        ≈ 100%.  A check that fired whenever the mode is not "limit" would fire
        on *every* default configuration — the always-on warning F14 is about,
        and the one users learn to scroll past on their way to real findings.

        The value that was inherited is not a finding; the value that was moved
        is.  Same rule as `timeframe_mismatch`.
        """
        rep = config_report(None, None, RuleDiscoveryConfig(), ctx=_ctx())
        assert "entry_mode_inert_gate" not in _codes(rep)

    def test_schema_and_fee_mismatches_are_reported(self):
        disc = DiscoveryConfig(timestamp_col="a")
        alpha = AlphaConfig(timestamp_col="b", fee_per_side=0.0005)
        rd = RuleDiscoveryConfig(base_params=BacktestParams(fee=0.002))
        rep = config_report(disc, alpha, rd, ctx=_ctx())

        assert "schema_mismatch" in _codes(rep)
        assert "fee_mismatch" in _codes(rep)
        assert "never reads" in _message(rep, "fee_mismatch") or \
               "contract" in _message(rep, "fee_mismatch")


# ---------------------------------------------------------------------------
# Partial input
# ---------------------------------------------------------------------------

class TestPartialInput:
    def test_a_constraint_with_missing_inputs_says_nothing(self):
        """A partial bundle is a supported case, not a violation."""
        rep = config_report(None, AlphaConfig(), None, ctx=_ctx())
        assert "wf_bucket_too_short" not in _codes(rep)
        assert "m3_stricter_than_m1" not in _codes(rep)

    def test_span_dependent_checks_stay_silent_without_data(self):
        """Without a span there is nothing to compare against — silence is the
        honest answer, not a pass."""
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=0.20))
        rep = config_report(None, None, rd, ctx=PipelineContext(timeframe="1D"))
        assert "oos_span_too_short" not in _codes(rep)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_to_text_shows_the_trace_then_the_diagnostics(self):
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=0.20))
        text = config_report(DiscoveryConfig(), AlphaConfig(), rd,
                             ctx=_ctx()).to_text()

        assert "RESOLUTION TRACE" in text
        assert "DIAGNOSTICA" in text
        assert text.index("RESOLUTION TRACE") < text.index("DIAGNOSTICA")
        assert "CRITICO" in text

    def test_one_line_is_compact_and_names_the_codes(self):
        # The window is pinned so the #173 constraint is the one reported:
        # left unset it is derived from the rate, and the FAIL that fires is
        # `oos_span_too_short` instead (see TestFailConstraints).
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=0.20),
            walk_forward=RuleWalkForwardConfig(min_train_months=6),
        )
        line = config_report(None, None, rd, ctx=_ctx()).one_line()
        assert "\n" not in line
        assert "wf_bucket_too_short" in line

    def test_a_coherent_configuration_says_so(self):
        """A report that always fires is one users learn to ignore, so a
        configuration with nothing wrong with it must come back clean."""
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=3.0))
        alpha = AlphaConfig(timeframe="1D", horizon_grid=(1, 2, 3, 5))
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=2.5, partial_min_profit_factor=2.0),
            scoring=ScoringParams(pf_min_trades=15, pf_min_tpm=2, pf_tpm_target=3),
            base_params=BacktestParams(target_h=3, buy_delay_bar=1),
        )
        rep = config_report(disc, alpha, rd, RegistryConfig(), ctx=_ctx())

        assert not rep.has_critical, rep.one_line()
        assert not rep.has_warnings, rep.one_line()
        assert "coerente" in rep.to_text()


# ---------------------------------------------------------------------------
# strict
# ---------------------------------------------------------------------------

class TestForgeStrict:
    """``strict=True`` is the default: a run that cannot produce a verdict
    should not start, because its wall of rejections is indistinguishable from
    "the signal is bad" — which is what the caller was trying to measure."""

    @staticmethod
    def _kpi(n=900):
        import numpy as np

        rng = np.random.default_rng(5)
        close = 100 + np.cumsum(rng.normal(0, 0.6, n))
        return pd.DataFrame({
            "open_dt": pd.date_range("2022-01-01", periods=n, freq="D"),
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": rng.uniform(1e5, 1e6, n),
            "feat": rng.uniform(0, 1, n),
        })

    def _incoherent(self):
        return dict(
            event_discovery_config=DiscoveryConfig(
                max_and_components=1, gate_params=GateParams(min_tpm=0.30)),
            alpha_config=AlphaConfig(asset="X", timeframe="1D",
                                     horizon_grid=(1, 2, 3)),
            # `min_train_months` pinned: since #177 an unset window is derived
            # from the rate, and this helper's job is to produce the *#173*
            # incoherence — a window and a rate that do not fit each other.
            rule_discovery_config=RuleDiscoveryConfig(
                criteria=SelectionCriteria(min_tpm=0.25),
                walk_forward=RuleWalkForwardConfig(min_train_months=6)),
        )

    def test_strict_stops_a_run_that_cannot_produce_a_verdict(self):
        from forgedge import forge

        with pytest.raises(ValueError, match="cannot produce a verdict"):
            forge(self._kpi(), ticker="X", timeframe="1D", progress=False,
                  **self._incoherent())

    def test_the_error_names_the_constraint_and_the_value_to_set(self):
        from forgedge import forge

        with pytest.raises(ValueError, match="wf_bucket_too_short"):
            forge(self._kpi(), ticker="X", timeframe="1D", progress=False,
                  **self._incoherent())

    def test_strict_false_downgrades_to_warnings_and_runs(self):
        from forgedge import forge

        with pytest.warns(UserWarning, match="wf_bucket_too_short"):
            result = forge(self._kpi(), ticker="X", timeframe="1D",
                           strict=False, run_rule_discovery=False,
                           run_registry=False, fast_null=False, progress=False,
                           **self._incoherent())
        assert result.coherence.has_critical
