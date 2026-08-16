"""Tests for the central parameter resolver (``forgedge.resolver``).

The resolver's whole value is that it is *predictable*: it fills only what the
caller left unset, never touches what they chose, derives with a margin rather
than in expectation, and records what it did. Each of those is a property the
rest of the plan builds on, so each gets pinned here.
"""
import copy
import pickle

import pandas as pd
import pytest

from forgedge import (
    AlphaConfig,
    DiscoveryConfig,
    PipelineContext,
    RegistryConfig,
    RuleDiscoveryConfig,
    UNSET,
    collect_context,
    resolve,
    resolve_config,
)
from forgedge.market_context.models import MarketContextConfig
from forgedge.resolver import (
    PROPAGATION,
    STATISTICAL,
    Constraint,
    _topological,
    poisson_min_window,
    timeframe_minutes,
)
from forgedge.unset import is_set


def _bundle(**over):
    b = {
        "market_context": MarketContextConfig(),
        "event_discovery": DiscoveryConfig(),
        "alpha": AlphaConfig(),
        "rule_discovery": RuleDiscoveryConfig(),
        "registry": RegistryConfig(),
    }
    b.update(over)
    return b


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------

class TestUnsetSentinel:
    def test_is_falsy_and_readable(self):
        assert not UNSET
        assert repr(UNSET) == "UNSET"
        assert not is_set(UNSET)
        assert is_set("open_dt")

    def test_arithmetic_fails_loudly(self):
        """An unresolved value that reaches a computation must not silently
        behave like a number — it must raise at the point of use."""
        with pytest.raises(TypeError):
            UNSET * 2
        with pytest.raises(TypeError):
            UNSET < 3

    def test_identity_survives_copying(self):
        """``resolve()`` works on copies, so ``is UNSET`` has to hold after one."""
        assert copy.copy(UNSET) is UNSET
        assert copy.deepcopy(UNSET) is UNSET
        assert pickle.loads(pickle.dumps(UNSET)) is UNSET


# ---------------------------------------------------------------------------
# The core contract
# ---------------------------------------------------------------------------

class TestResolveContract:
    def test_returns_copies_and_never_mutates_the_caller(self):
        """``config_report()`` calls this on a user's live configs; mutating
        them would make inspection a side effect."""
        bundle = _bundle()
        out, _trace, _viol = resolve(bundle, PipelineContext())

        assert out["alpha"] is not bundle["alpha"]
        assert bundle["alpha"].timestamp_col is UNSET       # caller untouched
        assert out["alpha"].timestamp_col == "open_dt"      # copy resolved

    def test_an_explicit_value_is_never_overwritten(self):
        """The non-negotiable rule: a field the caller set stays as written,
        even when it contradicts the session context."""
        bundle = _bundle(
            event_discovery=DiscoveryConfig(timestamp_col="mine"),
            alpha=AlphaConfig(timestamp_col="mine"),
            rule_discovery=RuleDiscoveryConfig(timestamp_col="mine"),
            registry=RegistryConfig(timestamp_col="mine"),
        )
        out, trace, _ = resolve(bundle, PipelineContext(timestamp_col="theirs"))

        for key in ("event_discovery", "alpha", "rule_discovery", "registry"):
            assert out[key].timestamp_col == "mine"
        # Not a derivation *anywhere* — the bundle's other unset fields are
        # resolved as usual — but not one that touched the field the caller
        # wrote.  Silence on that field is what "never overwritten" means.
        assert [d for d in trace.effective if d.field.endswith("timestamp_col")] == []

    def test_is_idempotent(self):
        out1, trace1, _ = resolve(_bundle(), PipelineContext())
        out2, trace2, _ = resolve(out1, PipelineContext())

        assert len(trace1) > 0
        assert len(trace2) == 0
        assert out2["alpha"].timestamp_col == out1["alpha"].timestamp_col

    def test_derive_mode_never_reads_the_data(self):
        """The invariant that makes the config report trustworthy: what it
        shows is computable without the frame, so it is exactly what runs."""
        blind = PipelineContext(timestamp_col="ts")            # n_bars=0, span=0
        informed = PipelineContext(timestamp_col="ts", n_bars=5000, span_months=60)

        out_blind, trace_blind, _ = resolve(_bundle(), blind)
        out_informed, trace_informed, _ = resolve(_bundle(), informed)

        assert out_blind["alpha"].timestamp_col == out_informed["alpha"].timestamp_col
        assert [d.resolved for d in trace_blind] == [d.resolved for d in trace_informed]

    def test_partial_bundles_resolve(self):
        """A caller with only one config is a supported case, not an error."""
        out, _trace, _ = resolve({"alpha": AlphaConfig()}, PipelineContext())
        assert out["alpha"].timestamp_col == "open_dt"


# ---------------------------------------------------------------------------
# Propagation — the F10 case
# ---------------------------------------------------------------------------

class TestSchemaPropagation:
    def test_a_column_set_on_one_module_reaches_the_others(self):
        """`forge_preset(timestamp_col=...)` used to configure M1 alone, and
        M2/M3/M4 then failed asking for a value the user believed they had
        already given (F10)."""
        bundle = _bundle(event_discovery=DiscoveryConfig(timestamp_col="ts"))
        ctx = collect_context(bundle)
        assert ctx.timestamp_col == "ts"

        out, trace, violations = resolve(bundle, ctx)
        assert out["alpha"].timestamp_col == "ts"
        assert out["rule_discovery"].timestamp_col == "ts"
        assert out["registry"].timestamp_col == "ts"
        # Nothing to report about the schema: one value, propagated.  (Other
        # constraints may well fire on a bundle of raw class defaults — the
        # M1/M3 rate gates disagree out of the box — but that is not this
        # test's business.)
        assert "schema_mismatch" not in {v.code for v in violations}
        # Exactly the three that were missing it — M1 named it, so M1 is not
        # re-derived.  Scoped to this field: the same bundle legitimately
        # derives the price column, the regime columns, the fee and M4's
        # genericity floor, and none of that is this test's subject.
        assert {d.field for d in trace.effective if d.field.endswith("timestamp_col")} == {
            "alpha.timestamp_col",
            "rule_discovery.timestamp_col",
            "registry.timestamp_col",
        }

    def test_disagreeing_explicit_columns_are_reported_not_reconciled(self):
        """Two explicit values that conflict are a `Finding`, never a silent
        fix — correcting them quietly would reintroduce exactly the failure
        mode the plan removes."""
        bundle = _bundle(
            event_discovery=DiscoveryConfig(timestamp_col="a"),
            alpha=AlphaConfig(timestamp_col="b"),
        )
        out, _trace, violations = resolve(bundle, collect_context(bundle))

        assert out["event_discovery"].timestamp_col == "a"
        assert out["alpha"].timestamp_col == "b"
        codes = [v.code for v in violations]
        assert "schema_mismatch" in codes

    def test_an_explicit_context_outranks_the_configs(self):
        """Ladder: an explicit PipelineContext beats collection from configs."""
        bundle = _bundle(event_discovery=DiscoveryConfig(timestamp_col="from_cfg"))
        ctx = collect_context(bundle, PipelineContext(timestamp_col="from_ctx"))
        assert ctx.timestamp_col == "from_ctx"

    def test_the_price_column_reaches_the_backtest(self):
        """"The price series" had four names with independent defaults (F10):
        `AlphaConfig.close_col` against `BacktestParams.{target_col,
        buy_price_anchor}`.  M2 measured returns on one column while M3 priced
        the exit on another, and nothing said so."""
        bundle = _bundle(alpha=AlphaConfig(close_col="px"))
        ctx = collect_context(bundle)
        assert ctx.close_col == "px"

        out, _trace, violations = resolve(bundle, ctx)
        assert out["rule_discovery"].base_params.target_col == "px"
        assert out["rule_discovery"].base_params.buy_price_anchor == "px"
        assert "schema_mismatch" not in {v.code for v in violations}

    def test_the_regime_columns_travel_with_the_schema(self):
        bundle = _bundle(alpha=AlphaConfig())
        out, _trace, _v = resolve(bundle, PipelineContext(
            regime_col="rg", regime_stable_col="rg_ok"))
        assert out["alpha"].regime_col == "rg"
        assert out["alpha"].regime_stable_col == "rg_ok"

    def test_an_explicit_limit_anchor_is_mechanics_not_schema(self):
        """The one asymmetry in the schema table, and the reason it exists.

        `buy_price_anchor` is *filled in* from the session's price column, but
        it does not *define* it: a caller who anchors the limit order on
        "open" has chosen order mechanics.  Seeding the context from it would
        push "open" back out into `alpha.close_col` and change the series M2
        measures forward returns on — a config the caller never asked for.

        It is out of the equality check for the same reason: disagreement here
        is a choice, and a warning that fires on a legitimate choice is one
        users learn to ignore.
        """
        bundle = _bundle(alpha=AlphaConfig())
        bundle["rule_discovery"].base_params.buy_price_anchor = "open"
        ctx = collect_context(bundle)

        assert ctx.close_col == "close"          # unmoved by the anchor
        out, _trace, violations = resolve(bundle, ctx)
        assert out["alpha"].close_col == "close"
        assert out["rule_discovery"].base_params.buy_price_anchor == "open"
        assert out["rule_discovery"].base_params.target_col == "close"
        assert "schema_mismatch" not in {v.code for v in violations}

    def test_a_disagreeing_exit_column_is_still_reported(self):
        """`target_col` *is* in the equality check: the horizon exit has to be
        priced on the series M2 measured returns on, or the two modules are
        describing different instruments."""
        bundle = _bundle(alpha=AlphaConfig(close_col="px"))
        bundle["rule_discovery"].base_params.target_col = "close"
        out, _trace, violations = resolve(bundle, collect_context(bundle))

        assert out["alpha"].close_col == "px"
        assert out["rule_discovery"].base_params.target_col == "close"
        assert "schema_mismatch" in {v.code for v in violations}


# ---------------------------------------------------------------------------
# Economics — the F7 case
# ---------------------------------------------------------------------------

class TestFeePropagation:
    def test_the_documented_cost_becomes_the_charged_cost(self):
        """`_seed_base_params` seeded direction, target_h and sell_pct from the
        contract but never the fee, and no code path read
        `contract.fee_per_side` at all.  The two agreed only because both
        defaulted to 0.002."""
        bundle = _bundle(alpha=AlphaConfig(fee_per_side=0.0005))
        out, _trace, violations = resolve(bundle, collect_context(bundle))

        assert out["rule_discovery"].base_params.fee == pytest.approx(0.0005)
        assert "fee_mismatch" not in {v.code for v in violations}

    def test_two_explicit_fees_are_reported_not_reconciled(self):
        bundle = _bundle(alpha=AlphaConfig(fee_per_side=0.0005))
        bundle["rule_discovery"].base_params.fee = 0.002
        out, _trace, violations = resolve(bundle, collect_context(bundle))

        assert out["alpha"].fee_per_side == pytest.approx(0.0005)
        assert out["rule_discovery"].base_params.fee == pytest.approx(0.002)
        assert "fee_mismatch" in {v.code for v in violations}

    def test_a_fee_set_only_on_the_backtest_reaches_the_contract(self):
        """Propagation is not one-directional: whichever side names the cost
        basis, one value ends up on both."""
        bundle = _bundle()
        bundle["rule_discovery"].base_params.fee = 0.0005
        out, _trace, violations = resolve(bundle, collect_context(bundle))

        assert out["alpha"].fee_per_side == pytest.approx(0.0005)
        assert "fee_mismatch" not in {v.code for v in violations}


# ---------------------------------------------------------------------------
# Derivation mechanics
# ---------------------------------------------------------------------------

class TestDerivationMechanics:
    def test_only_the_propagation_stage_derives_today(self):
        """Statistical constraints are registered and checked, but their
        `derive` is switched on by the issue that owns the fix — which is what
        keeps this change additive."""
        from forgedge.resolver import CONSTRAINTS

        deriving = [c for c in CONSTRAINTS if c.derived and c.derive]
        assert deriving
        assert all(c.stage == PROPAGATION for c in deriving)

    def test_conflicting_derivations_keep_the_conservative_value(self):
        """Every derived quantity is a *requirement*, so demanding more is the
        safe direction; the losing derivation stays in the trace."""
        from forgedge.resolver import _more_conservative

        assert _more_conservative(6, 21) == 21
        assert _more_conservative(21, 6) == 21
        # Non-numeric values keep the first derivation rather than guessing.
        assert _more_conservative("a", "b") == "a"

    def test_topological_order_puts_producers_first(self):
        produced = Constraint(code="p", level="WARN", stage=PROPAGATION,
                              free=(), derived="alpha.timestamp_col",
                              derive=lambda v, c: ("x", "why"))
        consumer = Constraint(code="c", level="WARN", stage=PROPAGATION,
                              free=("alpha.timestamp_col",), derived=None,
                              check=lambda v, c: None)
        assert [c.code for c in _topological([consumer, produced])] == ["p", "c"]

    def test_a_cyclic_table_is_a_loud_error(self):
        """A cycle is a bug in the constraint table, not in the user's config,
        so it must not degrade into a silently partial resolution."""
        a = Constraint(code="a", level="WARN", stage=PROPAGATION,
                       free=("y",), derived="x", derive=lambda v, c: (1, ""))
        b = Constraint(code="b", level="WARN", stage=PROPAGATION,
                       free=("x",), derived="y", derive=lambda v, c: (1, ""))
        with pytest.raises(RuntimeError, match="Cyclic constraint dependency"):
            _topological([a, b])


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

class TestResolutionTrace:
    def test_records_the_default_to_resolved_diff_with_a_reason(self):
        """A resolver that adapts parameters behind the caller's back has to be
        auditable: the diff is the debugging question, the reason keeps the
        number from being magic."""
        bundle = _bundle(event_discovery=DiscoveryConfig(timestamp_col="ts"))
        _out, trace, _ = resolve(bundle, collect_context(bundle))

        d = next(d for d in trace if d.field == "alpha.timestamp_col")
        assert d.default == "open_dt"
        assert d.resolved == "ts"
        assert d.rule == "schema_mismatch"
        assert "ts" in d.reason
        assert d.order >= 1

    def test_describe_and_to_text_are_renderable(self):
        _out, trace, _ = resolve(_bundle(), PipelineContext())
        assert "derived" in trace.describe()
        text = trace.to_text()
        assert "RESOLUTION TRACE" in text
        assert "timestamp_col" in text


# ---------------------------------------------------------------------------
# Poisson margin — D2.2
# ---------------------------------------------------------------------------

class TestPoissonMargin:
    def test_the_margin_is_meaningfully_above_the_naive_inversion(self):
        """`floor / rate` satisfies the constraint in expectation only: at
        floor=10, rate=0.8 it yields lambda=10.4, under which fewer than 10
        events are observed ~44 % of the time.  A config derived that way fails
        about one run in two — #173 in milder form, after being "fixed"."""
        naive = 10 / 0.8
        with_margin = poisson_min_window(10, 0.8)

        assert naive == pytest.approx(12.5)
        assert with_margin > naive * 1.4
        assert with_margin == pytest.approx(20.0, rel=0.1)

    def test_scales_inversely_with_the_rate(self):
        assert poisson_min_window(10, 8.0) < poisson_min_window(10, 0.8)

    def test_an_unreachable_floor_returns_infinity(self):
        """No window is enough — the honest answer, not a capped one."""
        assert poisson_min_window(10, 0.0) == float("inf")
        assert poisson_min_window(10, 1e-6) == float("inf")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class TestPipelineContext:
    @pytest.mark.parametrize(
        "timeframe,minutes,bars_per_day",
        [("1D", 1440, 1.0), ("4H", 240, 6.0), ("1H", 60, 24.0), ("15m", 15, 96.0)],
    )
    def test_bar_arithmetic(self, timeframe, minutes, bars_per_day):
        ctx = PipelineContext(timeframe=timeframe)
        assert timeframe_minutes(timeframe) == minutes
        assert ctx.bars_per_day == pytest.approx(bars_per_day)
        assert ctx.bars_per_month == pytest.approx(bars_per_day * 30)

    def test_round_trips_bars_and_months(self):
        ctx = PipelineContext(timeframe="1D")
        assert ctx.months_of(ctx.bars_of(6)) == pytest.approx(6.0)

    def test_from_frame_fills_only_the_data_facts(self):
        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=300, freq="D"),
            "close": range(300),
        })
        ctx = PipelineContext.from_frame(df, timeframe="1D")
        assert ctx.n_bars == 300
        assert ctx.span_months == pytest.approx(299 / 30, rel=0.01)
        assert ctx.timeframe == "1D"

    def test_rejects_an_unparseable_timeframe(self):
        with pytest.raises(ValueError, match="Unrecognised timeframe"):
            timeframe_minutes("weekly")
        with pytest.raises(ValueError, match="Unknown timeframe unit"):
            timeframe_minutes("3y")


# ---------------------------------------------------------------------------
# Standalone module use
# ---------------------------------------------------------------------------

class TestStandaloneModules:
    @pytest.mark.parametrize(
        "cfg,kind",
        [
            (DiscoveryConfig(), "event_discovery"),
            (AlphaConfig(), "alpha"),
            (RuleDiscoveryConfig(), "rule_discovery"),
            (RegistryConfig(), "registry"),
        ],
    )
    def test_a_config_resolved_with_no_context_keeps_the_historical_default(self, cfg, kind):
        """Standalone use must behave exactly as before: the fix applies where
        there is an orchestrator that knows enough to apply it."""
        assert resolve_config(cfg, kind).timestamp_col == "open_dt"

    def test_modules_resolve_their_own_config_on_construction(self):
        """`what you inspect is what runs` has to hold for hand-built pipelines,
        not only for forge()."""
        from forgedge import EventDiscovery

        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=200, freq="D"),
            "close": range(200),
        })
        ed = EventDiscovery(df, config=DiscoveryConfig())
        assert ed.config.timestamp_col == "open_dt"


class TestUnresolvedFieldsAtPublicBoundaries:
    """A public function that receives an unresolved config field falls back to
    its own documented default.

    Forwarding a config field straight into a helper is an ordinary, documented
    pattern::

        cfg = RuleDiscoveryConfig(...)
        run_backtest(frame, cfg.signal_col, params, timestamp_col=cfg.timestamp_col)

    Once a field's class default becomes ``UNSET``, that pattern hands the
    sentinel to a function whose signature already declares a default. ``UNSET``
    means "you decide", so the function's default *is* the decision — anything
    else turns a resolvable state into a ``KeyError`` deep inside a column
    lookup.  This will keep mattering as later steps convert more fields.
    """

    def test_run_backtest_accepts_an_unresolved_timestamp_col(self):
        import numpy as np
        from forgedge.rule_discovery.backtest import run_backtest
        from forgedge.rule_discovery.models import BacktestParams

        rng = np.random.default_rng(11)
        n = 300
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        frame = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "sig": (rng.random(n) < 0.1).astype(float),
        })
        cfg = RuleDiscoveryConfig()
        assert cfg.timestamp_col is UNSET       # the caller never resolved it

        summary = run_backtest(
            frame, "sig", BacktestParams(target_h=3),
            timestamp_col=cfg.timestamp_col,
        )
        assert summary.total_signals > 0
