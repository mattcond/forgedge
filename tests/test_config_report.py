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
    MarketContextConfig,
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

        Re-pinned again by #204: `criteria.min_tpm` was 0.80 above because the
        preset's own `_episode`-suffixed M3 spec key applied a fill ratio
        straight to M1's *episode* rate, which M3 does not count in — M3 opens
        a trade on every active bar, so a declared episode rate has to be
        converted to bars first (`bars_per_episode`, ~1.76 measured). The
        stock preset's M3 rate is now 1.4667, not 0.80, so the window this
        test pins moved from 20 months to 11.
        """
        disc, alpha, rd = _preset_triple()
        rep = config_report(disc, alpha, rd, RegistryConfig(), ctx=_ctx())
        resolved = rep.configs["rule_discovery"]

        assert resolved.walk_forward.min_train_months == 11
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
    def test_m1_is_window_too_short(self):
        """#206 — the reported case: `sniper`'s own stock rate (0.3
        episodes/month) needs 53.3 months at 95% Poisson confidence to reach
        `min_episodes=10`, not the "≥2 anni" the preset's description used to
        claim before this was measured."""
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=0.3, min_episodes=10))
        rep = config_report(disc, None, None, ctx=_ctx(span_months=29))

        assert "m1_is_window_too_short" in _codes(rep)
        msg = _message(rep, "m1_is_window_too_short")
        assert "min_episodes" in msg      # names the field
        assert "53.3" in msg              # the measured window, not a naive one
        assert "min_episodes" in msg and "min_tpm" in msg  # names both fixes

    def test_m1_is_window_is_silent_when_the_window_can_conclude(self):
        """Same rate, a span long enough to actually reach the floor."""
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=0.3, min_episodes=10))
        rep = config_report(disc, None, None, ctx=_ctx(span_months=60))
        assert "m1_is_window_too_short" not in _codes(rep)

    def test_m1_is_window_is_silent_in_bar_mode(self):
        """`min_episodes` is an episode-mode-only criterion (ignored in
        `"bar"` mode by `ConsistencyGate`), so the check has nothing to say
        there regardless of the span."""
        disc = DiscoveryConfig(
            gate_params=GateParams(min_tpm=0.3, min_episodes=10, event_counting="bar")
        )
        rep = config_report(disc, None, None, ctx=_ctx(span_months=29))
        assert "m1_is_window_too_short" not in _codes(rep)

    def test_m1_is_window_is_silent_without_a_span(self):
        """Same invariant as its OOS sibling: the resolver's derive half
        never reads the data, and this check needs `span_months` to say
        anything at all."""
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=0.3, min_episodes=10))
        rep = config_report(disc, None, None, ctx=PipelineContext(timeframe="1D"))
        assert "m1_is_window_too_short" not in _codes(rep)

    def test_m1_is_window_never_fails_strict_mode(self):
        """WARN, not FAIL (#206) — a candidate with a higher realised rate
        than the configured floor can still clear `min_episodes` on a shorter
        history, so this is worth knowing, not a reason to refuse to start."""
        disc = DiscoveryConfig(gate_params=GateParams(min_tpm=0.3, min_episodes=10))
        rep = config_report(disc, None, None, ctx=_ctx(span_months=29))
        assert "m1_is_window_too_short" in _codes(rep)
        assert not rep.has_critical

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
        """F3 — `pf_score_tpm` is what the grid maximises and a gate in
        `_passes`, so a rate floor calibrated differently from the gate's steers
        which operating point gets published."""
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=0.25),
            scoring=ScoringParams(pf_min_tpm=2),
        )
        rep = config_report(None, None, rd, ctx=_ctx())

        assert "scoring_uncalibrated" in _codes(rep)
        msg = _message(rep, "scoring_uncalibrated")
        assert "pf_min_tpm" in msg and "min_tpm" in msg

    def test_the_scoring_rate_floor_follows_the_gate(self):
        """Left alone it is derived, so the two cannot disagree — it used to be
        a fixed 2 while the gate ran from 0.8 on daily bars to 76.8 on 15m."""
        rd = RuleDiscoveryConfig(criteria=SelectionCriteria(min_tpm=0.25))
        rep = config_report(None, None, rd, ctx=_ctx())

        assert rep.configs["rule_discovery"].scoring.pf_min_tpm == pytest.approx(0.25)
        assert "scoring_uncalibrated" not in _codes(rep)

    def test_scoring_is_silent_when_the_knobs_track_the_rate(self):
        rd = RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=2.5),
            scoring=ScoringParams(pf_min_trades=15, pf_min_tpm=2),
        )
        rep = config_report(None, None, rd, ctx=_ctx())
        assert "scoring_uncalibrated" not in _codes(rep)

    def test_timeframe_mismatch_catches_the_declared_label(self):
        alpha = AlphaConfig(timeframe="1H", horizon_grid=(1, 2, 3))
        rep = config_report(None, alpha, None, ctx=_ctx(timeframe="1D"))
        assert "timeframe_mismatch" in _codes(rep)

    def test_the_m3_bar_counts_are_converted_not_warned_about(self):
        """#179 turned this warning into a derive.

        `target_h=24` used to mean 24 *days* on daily bars and the report said
        so; now the session's bar duration decides the value and there is
        nothing left to warn about. A converted value is not a mismatch.
        """
        alpha = AlphaConfig(timeframe="1D", horizon_grid=(1, 2, 3))
        rep = config_report(None, alpha, RuleDiscoveryConfig(),
                            ctx=_ctx(timeframe="1D"))
        assert "timeframe_mismatch" not in _codes(rep)

        bp = rep.configs["rule_discovery"].base_params
        assert bp.target_h == 10        # top of the daily horizon class
        assert bp.buy_delay_bar == 1    # 6 hours of live limit order, in days

        # Chosen values are still the caller's business.
        chosen = RuleDiscoveryConfig(
            base_params=BacktestParams(target_h=3, buy_delay_bar=2))
        out = config_report(None, alpha, chosen, ctx=_ctx(timeframe="1D"))
        assert out.configs["rule_discovery"].base_params.target_h == 3
        assert out.configs["rule_discovery"].base_params.buy_delay_bar == 2

    def test_hourly_sessions_keep_the_values_they_always_had(self):
        """The conversion is calibrated *on* 1H, so an hourly session is a
        no-op — which is what makes the change safe to land."""
        rep = config_report(None, AlphaConfig(timeframe="1H"),
                            RuleDiscoveryConfig(),
                            market_context=MarketContextConfig(),
                            ctx=_ctx(timeframe="1H"))
        bp = rep.configs["rule_discovery"].base_params
        assert (bp.target_h, bp.buy_delay_bar) == (24, 6)
        assert rep.configs["market_context"].stable_window == 12
        assert rep.configs["alpha"].bars_per_day == pytest.approx(24.0)

    def test_declared_timeframe_versus_measured_spacing(self):
        """The third disagreement F5 names: the label says one thing and the
        timestamps say another, with M2 scaling on the label and M0 on the
        spacing."""
        rep = config_report(None, None, None,
                            ctx=_ctx(timeframe="1D", inferred_bar_hours=1.0))
        msg = _message(rep, "timeframe_mismatch")
        assert "24" in msg and "1h" in msg.lower()

        # Agreement is silent, and so is a spacing that merely wobbles.
        assert "timeframe_mismatch" not in _codes(
            config_report(None, None, None,
                          ctx=_ctx(timeframe="1D", inferred_bar_hours=24.0)))
        assert "timeframe_mismatch" not in _codes(
            config_report(None, None, None,
                          ctx=_ctx(timeframe="1D", inferred_bar_hours=24.4)))

    def test_an_undeclared_timeframe_is_not_a_contradiction(self):
        """`forge(kpi)` with no timeframe inherits `"1H"`. Reporting that
        against the data would be flagging a default nobody chose — the same
        rule `_untouched_default` applies to the bar counts."""
        assert "timeframe_mismatch" not in _codes(
            config_report(None, None, None,
                          ctx=_ctx(timeframe="1H", inferred_bar_hours=4.0,
                                   timeframe_declared=False)))
        # Declared, and it still disagrees: that is worth saying.
        assert "timeframe_mismatch" in _codes(
            config_report(None, None, None,
                          ctx=_ctx(timeframe="1H", inferred_bar_hours=4.0)))

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
            scoring=ScoringParams(pf_min_trades=15, pf_min_tpm=2),
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


class TestBarDurationHasOneSource:
    """F5 (#179) — the conversion and the measurement, one each."""

    def test_the_conversion_is_not_reimplemented_by_the_presets(self):
        """`_TFClass` used to carry its own `bars_per_month` / `bars_per_day`
        arithmetic next to the context's."""
        from forgedge.presets import _TFClass

        for tf in ("1D", "4H", "1H", "15m"):
            ctx = PipelineContext(timeframe=tf)
            t = _TFClass(tf)
            assert t.bars_per_month == pytest.approx(ctx.bars_per_month)
            assert t.bars_per_day == max(1, round(ctx.bars_per_day))

    def test_the_measurement_is_shared_by_the_modules_that_need_it(self):
        """M0 and M2 each measured the spacing themselves, with the context
        making a third copy. They now call the same function."""
        import pandas as pd
        from forgedge.resolver import measure_bar_hours

        for freq, hours in (("1h", 1.0), ("4h", 4.0), ("D", 24.0), ("15min", 0.25)):
            df = pd.DataFrame({
                "open_dt": pd.date_range("2024-01-01", periods=50, freq=freq),
                "close": range(50),
            })
            assert measure_bar_hours(df) == pytest.approx(hours)
            # ...and off a DatetimeIndex, which is the other accepted shape.
            assert measure_bar_hours(df.set_index("open_dt")) == pytest.approx(hours)

    def test_no_measurement_reads_as_none_never_as_zero(self):
        import pandas as pd
        from forgedge.resolver import measure_bar_hours

        assert measure_bar_hours(pd.DataFrame({"close": [1, 2, 3]})) is None
        assert measure_bar_hours(pd.DataFrame({"open_dt": pd.to_datetime(["2024-01-01"])})) is None

    def test_a_gap_does_not_move_the_measurement(self):
        """Median, not mean: exchange downtime leaves the answer alone."""
        import pandas as pd
        from forgedge.resolver import measure_bar_hours

        stamps = list(pd.date_range("2024-01-01", periods=40, freq="1h"))
        stamps += list(pd.date_range("2024-02-01", periods=40, freq="1h"))
        df = pd.DataFrame({"open_dt": stamps, "close": range(80)})
        assert measure_bar_hours(df) == pytest.approx(1.0)


class TestOneSignificanceLevel:
    """F9 (#182) — seven thresholds, and only five of them are an alpha.

    They were five independent `0.05` literals that agreed by coincidence:
    nothing related them, so they could drift apart by inattention, and a
    caller wanting a different regime had to find all five.
    """

    _ALPHA_PATHS = (
        ("alpha", "thresholds", "max_p_value"),
        ("alpha", "thresholds", "ic_max_p"),
        ("rule_discovery", "criteria", "max_ttest_p"),
        ("rule_discovery", "criteria", "max_rotation_p"),
    )

    @staticmethod
    def _resolve(**ctx_kw):
        return config_report(None, AlphaConfig(), RuleDiscoveryConfig(),
                             ctx=_ctx(**ctx_kw))

    def _values(self, rep):
        out = []
        for kind, group, field in self._ALPHA_PATHS:
            out.append(getattr(getattr(rep.configs[kind], group), field))
        return out

    def test_the_defaults_do_not_move(self):
        """The whole point: no behavioural change. All five were already
        0.05, so this step buys coherence, not a different pipeline."""
        assert self._values(self._resolve()) == [0.05] * 4

    def test_one_number_moves_all_of_them(self):
        rep = self._resolve(alpha=0.01)
        assert self._values(rep) == [0.01] * 4

    def test_a_pinned_threshold_wins_and_is_reported(self):
        """Legal — it is the caller's session — but it means two
        per-hypothesis error rates are running, which is worth saying."""
        from forgedge.alpha_discovery.models import PromotionThresholds

        alpha = AlphaConfig(thresholds=PromotionThresholds(max_p_value=0.20))
        rep = config_report(None, alpha, RuleDiscoveryConfig(), ctx=_ctx())
        assert rep.configs["alpha"].thresholds.max_p_value == 0.20
        assert "alpha_level_drift" in _codes(rep)
        assert "max_p_value" in _message(rep, "alpha_level_drift")

    def test_the_two_that_are_not_an_alpha_stay_put(self):
        """`fdr_q` is a false-discovery rate over a family and `oos_max_p` a
        confirmation level; tying either to a per-test alpha is a category
        error, not a tidy-up."""
        rep = self._resolve(alpha=0.01)
        th = rep.configs["alpha"].thresholds
        assert th.fdr_q == 0.10
        assert th.oos_max_p == 0.10

    def test_min_pass_rate_is_a_vote_not_a_probability(self):
        disc = DiscoveryConfig(
            train_ratio=0.8, walk_forward=EventWalkForwardConfig(n_splits=3))
        rep = config_report(disc, AlphaConfig(), RuleDiscoveryConfig(),
                            ctx=_ctx(alpha=0.01))
        assert rep.configs["event_discovery"].walk_forward.min_pass_rate == 0.6

    def test_the_rotation_config_follows_the_same_level(self):
        """Not a module config, so it is resolved at its own chokepoint."""
        from forgedge.calibration.models import RotationConfig
        from forgedge import UNSET

        assert RotationConfig().alpha is UNSET
        assert RotationConfig().resolved().alpha == 0.05
        assert RotationConfig().resolved(0.01).alpha == 0.01
        assert RotationConfig(alpha=0.2).resolved(0.01).alpha == 0.2


class TestTheHorizonGridFollowsTheSession:
    """#196 — the tail of F5: the last "N bars" field on the old mechanism.

    `horizon_grid` was substituted by `forge()` *only* when no `AlphaConfig`
    was passed at all; a caller who built one to change something else kept
    the hourly grid and got a warning instead of a conversion. On daily
    candles that meant scanning holding periods of up to 48 days.
    """

    @staticmethod
    def _grid(timeframe, alpha=None):
        rep = config_report(None, alpha or AlphaConfig(timeframe=timeframe), None,
                            ctx=_ctx(timeframe=timeframe))
        return rep.configs["alpha"].horizon_grid

    def test_it_is_class_calibrated_like_target_h(self):
        """Not wall-clock converted: 24 hours in daily bars is 1, which asks a
        different question. An hourly session scans to 24 hours, a daily one to
        10 days."""
        assert self._grid("1H") == (1, 2, 4, 8, 12, 24)
        assert self._grid("4H") == (1, 2, 4, 8, 12, 24)
        assert self._grid("1D") == (1, 2, 3, 5, 7, 10)
        assert self._grid("15m") == (1, 2, 5, 10, 20, 50)

    def test_it_agrees_with_the_horizon_that_m3_falls_back_to(self):
        """One calibration, read by both — `target_h`'s class default is the
        top of this grid, and that is not a coincidence to be maintained by
        hand."""
        for tf in ("1H", "1D", "15m"):
            rep = config_report(None, AlphaConfig(timeframe=tf), RuleDiscoveryConfig(),
                                ctx=_ctx(timeframe=tf))
            grid = rep.configs["alpha"].horizon_grid
            assert rep.configs["rule_discovery"].base_params.target_h == max(grid)

    def test_the_case_the_issue_was_opened_for(self):
        """An `AlphaConfig` built to change `train_ratio` — nothing to do with
        horizons — used to keep the hourly grid on daily candles."""
        grid = self._grid("1D", AlphaConfig(timeframe="1D", train_ratio=0.8))
        assert max(grid) == 10          # days, not 48
        assert "timeframe_mismatch" not in _codes(
            config_report(None, AlphaConfig(timeframe="1D", train_ratio=0.8), None,
                          ctx=_ctx(timeframe="1D")))

    def test_an_explicit_grid_is_still_the_callers_business(self):
        assert self._grid("1D", AlphaConfig(timeframe="1D",
                                            horizon_grid=(3, 9, 27))) == (3, 9, 27)

    def test_an_hourly_session_is_unchanged_where_it_matters(self):
        """The calibration point. The *literal* default changed — the old one
        reached 48 bars — but what an hourly session scans is the hourly class
        grid, which is what `forge()` already substituted on that path."""
        from forgedge.presets import _TFClass

        assert self._grid("1H") == _TFClass("1H").horizon_grid

    def test_the_sentinel_is_spent_at_the_standalone_chokepoint(self):
        """No session, no declared timeframe: `resolved()` falls back to the
        hourly calibration rather than leaving a sentinel to be iterated."""
        from forgedge import UNSET

        raw = AlphaConfig()
        assert raw.horizon_grid is UNSET
        assert raw.resolved().horizon_grid == (1, 2, 4, 8, 12, 24)
        # ...and the rest of __post_init__ still ran on the copy.
        assert raw.resolved().target_mode == "proj"
        assert len(raw.resolved().score_weights) == 5

    def test_unset_is_not_an_empty_grid(self):
        """`__post_init__` validates the grid; the sentinel must not be read as
        an empty one and rejected."""
        AlphaConfig()                                  # must not raise
        with pytest.raises(ValueError, match="positive horizons"):
            AlphaConfig(horizon_grid=())
        with pytest.raises(ValueError, match="positive horizons"):
            AlphaConfig(horizon_grid=(1, 0, 3))


class TestTheSessionRateReachesM3:
    """#200 — `criteria.min_tpm` is the root of a chain, not a lone threshold.

    `min_train_months` is sized from it with a Poisson margin (#177) and
    `scoring.pf_min_tpm` tracks it (#178). When only M1's rate moved, the
    walk-forward stayed sized for the rate the session no longer had — raising
    the session's rate *degraded* the walk-forward instead of tightening it.
    """

    @staticmethod
    def _run(m1_rate=None, rd=None, event_counting="episode"):
        disc = DiscoveryConfig()
        if m1_rate is not None:
            disc = DiscoveryConfig(
                gate_params=GateParams(min_tpm=m1_rate, event_counting=event_counting)
            )
        return config_report(disc, None, rd or RuleDiscoveryConfig(), timeframe="1D")

    def test_a_declared_rate_reaches_m3(self):
        """In *bar* mode M1 and M3 already share a unit, so the declared rate
        reaches M3 unconverted — the isolated `rate_retention` case, kept free
        of the episode→bar factor #204 adds (see `TestM3CountsBarsNotEpisodes`
        for that one)."""
        rep = self._run(1.0, event_counting="bar")
        rd = rep.configs["rule_discovery"]
        assert rd.criteria.min_tpm == pytest.approx(1.0)     # the declared rate
        assert "m3_stricter_than_m1" not in _codes(rep)

    def test_the_whole_chain_follows(self):
        """The point of the fix: one declaration, three coherent values."""
        rd = self._run(4.0, event_counting="bar").configs["rule_discovery"]
        assert rd.criteria.min_tpm == pytest.approx(4.0)
        assert rd.scoring.pf_min_tpm == pytest.approx(4.0)   # #178 tracks it
        assert rd.walk_forward.min_train_months == 4         # #177 sizes from it

    def test_the_rate_is_propagated_unchanged_and_that_was_measured(self):
        """`rate_retention` is 1.0 because a margin below 1 costs history
        twice: it lengthens `min_train_months` (Poisson margin) *and* shrinks
        the pooled OOS trade count (`test_months × min_tpm`). At 2.0 a session
        needs 13 months to produce a verdict; at 1.6 it needs 16.2 — so a 25%
        cut in the floor demands 25% more data, and a 14-month session that
        worked stops working.

        Checked in *bar* mode: episode mode has its own, separate factor
        (`bars_per_episode`, #204) that is not what this test is about."""
        from forgedge.resolver import poisson_min_window

        def span_needed(rate):
            return poisson_min_window(10, rate) + 10 / rate

        assert span_needed(2.0) == pytest.approx(13.0, abs=0.1)
        assert span_needed(1.6) == pytest.approx(16.2, abs=0.1)
        # ...which is why the resolver does not apply one.
        rd = self._run(2.0, event_counting="bar").configs["rule_discovery"]
        assert rd.criteria.min_tpm == 2.0

    def test_an_inherited_default_is_not_a_declaration(self):
        """The measurement that shaped this: `GateParams.min_tpm` defaults to
        0.5 and `SelectionCriteria.min_tpm` to 2.0 — two class defaults that
        disagree by 4x and were never designed to relate. Propagating the
        inherited one would drop M3's floor to 0.4 and inflate
        `min_train_months` from 8 to 40, more than a 29-month history can
        supply: the walk-forward would vanish entirely."""
        rd = self._run(None).configs["rule_discovery"]
        assert rd.criteria.min_tpm == 2.0                    # documented default
        assert rd.walk_forward.min_train_months == 8

    def test_an_explicit_m3_rate_still_wins(self):
        from forgedge.rule_discovery.models import SelectionCriteria

        rep = self._run(2.0, RuleDiscoveryConfig(
            criteria=SelectionCriteria(min_tpm=1.9)))
        assert rep.configs["rule_discovery"].criteria.min_tpm == pytest.approx(1.9)

    def test_m3_never_ends_up_stricter_than_m1(self):
        """`m3_stricter_than_m1` compares the two rates in the *same* unit
        (#204) — M1's declared rate converted to bars via `bars_per_episode`
        when it counts episodes — so it stays silent in both counting modes,
        not just the bar-mode case where no conversion is needed."""
        for rate in (1.0, 2.0, 5.0):
            for counting in ("bar", "episode"):
                rep = self._run(rate, event_counting=counting)
                assert "m3_stricter_than_m1" not in _codes(rep), counting


class TestThePresetScalesBothRates:
    """The reported case: `forge_preset(min_tpm=2)` moved M1 and left M3.

    Checked in *bar* mode — M1 and M3 already share a unit there, so this
    isolates the #200 property (M3 follows M1 at the preset's own fill
    ratio) from the episode→bar conversion #204 adds on top of it.  See
    `TestM3CountsBarsNotEpisodes` for the episode-mode numbers.
    """

    @staticmethod
    def _rates(preset="balanced", **kw):
        from forgedge.presets import forge_preset

        kw.setdefault("event_counting", "bar")
        d, _a, r = forge_preset(preset, "1D", asset="X", **kw)
        return d.gate_params.min_tpm, r.criteria.min_tpm

    def test_overriding_the_rate_moves_both(self):
        from forgedge.resolver import poisson_min_window

        m1, m3 = self._rates(min_tpm=2)
        assert (m1, m3) == (2, pytest.approx(1.6667, rel=1e-3))
        assert poisson_min_window(10, m3) == pytest.approx(9.6, abs=0.1)

    def test_the_untouched_presets_are_bit_for_bit_unchanged(self):
        """The ratio is read from each spec, not flattened to one number —
        the presets disagree about it (1.00 on sniper/sweep, 0.83 on
        balanced, 0.80 on burst) and flattening would move them."""
        assert self._rates("sniper") == (1.0, pytest.approx(1.0))
        assert self._rates("balanced") == (3.0, pytest.approx(2.5))
        assert self._rates("sweep") == (1.0, pytest.approx(1.0))
        assert self._rates("burst") == (2.5, pytest.approx(2.0))

    def test_an_explicit_rd_rate_still_wins(self):
        assert self._rates(min_tpm=2, rd_min_tpm=1.9) == (2, pytest.approx(1.9))

    def test_the_ratio_is_preserved_across_presets(self):
        for preset, ratio in (("sniper", 1.0), ("balanced", 2.5 / 3.0),
                              ("sweep", 1.0), ("burst", 0.8)):
            m1, m3 = self._rates(preset, min_tpm=3)
            assert m3 == pytest.approx(3 * ratio), preset


class TestM3CountsBarsNotEpisodes:
    """#204 — the fill ratio alone understated M3's floor in episode mode.

    M3 has no notion of episodes (it opens a trade on every active bar, no
    flat-state check), so a declared *episode* rate has to be converted to
    the *bar* rate M3 actually counts before the fill ratio applies.  The
    `_episode`-suffixed M3 spec keys applied the ratio straight to the
    episode rate instead — correct in bar mode (M1 and M3 already share a
    unit there), silently wrong in episode mode, where it understated M3's
    floor by `bars_per_episode` (~1.76, measured median on `ADA_1D_TRAIN`)
    and inflated `min_train_months` for no reason.
    """

    @staticmethod
    def _rates(preset="balanced", **kw):
        from forgedge.presets import forge_preset

        kw.setdefault("event_counting", "episode")
        d, _a, r = forge_preset(preset, "1D", asset="X", **kw)
        return d.gate_params.min_tpm, r.criteria.min_tpm

    def test_the_reported_case_is_fixed(self):
        """`forge_preset("balanced", min_tpm=2)` — M3 used to stay at the
        spec's literal 0.8 whatever M1 was told; it now follows M1 through
        the same bar-rate conversion the resolver's `_derive_m3_rate` uses."""
        m1, m3 = self._rates(min_tpm=2)
        assert (m1, m3) == (2, pytest.approx(2.9333, rel=1e-3))

    def test_bars_per_episode_times_the_bar_mode_fill_ratio(self):
        """Each preset's episode-mode M3 rate is `M1 x bars_per_episode x
        fill_ratio`, where `fill_ratio` is the *bar*-mode pair
        (`daily_rd_min_tpm / daily_min_tpm`) — the only ratio that is a pure
        fill margin, same unit on both sides."""
        from forgedge.resolver import PipelineContext

        bpe = PipelineContext().bars_per_episode
        for preset, fill_ratio in (("sniper", 1.0), ("balanced", 2.5 / 3.0),
                                    ("sweep", 1.0), ("burst", 0.8)):
            m1, m3 = self._rates(preset)
            assert m3 == pytest.approx(m1 * bpe * fill_ratio, rel=1e-3), preset

    def test_no_preset_timeframe_counting_combination_falsely_flags_m3_stricter(self):
        """The naive comparison (`_check_m3_vs_m1` before #204) would flag
        every one of these as M3 demanding a frequency nobody asked for — it
        was comparing an episode count to a bar count.  None of them should
        fire once the check converts units the same way the derive does."""
        from forgedge import config_report
        from forgedge.presets import forge_preset as _fp

        for preset in ("sniper", "balanced", "sweep", "burst"):
            for tf in ("1D", "4H", "1H", "15m"):
                for counting in ("episode", "bar"):
                    d, a, r = _fp(preset, tf, asset="X", event_counting=counting)
                    rep = config_report(d, a, r, timeframe=tf)
                    assert "m3_stricter_than_m1" not in _codes(rep), (preset, tf, counting)

    def test_bar_mode_is_unaffected(self):
        """In bar mode M1 and M3 already share a unit, so the conversion
        factor is 1 and every value this test class exists for stays exactly
        what `TestThePresetScalesBothRates` already pins."""
        m1, m3 = self._rates(event_counting="bar")
        assert (m1, m3) == (3.0, pytest.approx(2.5))


class TestPresetsDifferentiateMinEpisodes:
    """#206 — `min_episodes` is now preset-parametrized, not a flat class
    default: `sniper`/`balanced`/`burst` keep 10 (already coherent with their
    own rate, or the point of the preset), `sweep` lowers it to 5 because it
    is permissive by design and defers rigor to the RotationCalibrator.
    """

    def test_the_stock_values(self):
        from forgedge.presets import forge_preset

        for preset, expected in (("sniper", 10), ("balanced", 10),
                                  ("sweep", 5), ("burst", 10)):
            d, _a, _r = forge_preset(preset, "1D", asset="X")
            assert d.gate_params.min_episodes == expected, preset

    def test_an_explicit_override_still_wins(self):
        from forgedge.presets import forge_preset

        d, _a, _r = forge_preset("sweep", "1D", asset="X", min_episodes=8)
        assert d.gate_params.min_episodes == 8

    def test_not_scaled_by_timeframe(self):
        """An absolute episode count, unlike `min_tpm` — the timeframe's own
        effect on how long that takes is already carried by the rate."""
        from forgedge.presets import forge_preset

        for tf in ("1D", "4H", "1H", "15m"):
            d, _a, _r = forge_preset("sweep", tf, asset="X")
            assert d.gate_params.min_episodes == 5, tf

    def test_sweeps_lower_floor_is_reachable_where_sniper_and_sweeps_old_floor_was_not(self):
        """The measured point of lowering it: at 31 months of history `sweep`'s
        own rate now clears `min_episodes=5` at 95% confidence (needs 30.8);
        it did not before this issue, when both shared min_episodes=10 and
        needed 53.3 months regardless of the span."""
        from forgedge import config_report
        from forgedge.presets import forge_preset

        d, a, r = forge_preset("sweep", "1D", asset="X")
        rep = config_report(d, a, r, ctx=_ctx(span_months=31))
        assert "m1_is_window_too_short" not in _codes(rep)

        d, a, r = forge_preset("sniper", "1D", asset="X")
        rep = config_report(d, a, r, ctx=_ctx(span_months=31))
        assert "m1_is_window_too_short" in _codes(rep)
