"""The central parameter resolver — one place where the pipeline's latent
parameters are named, propagated and checked.

FORGE has a handful of **latent parameters**: quantities with exactly one
meaning for the whole pipeline — *how often do we expect this to fire*, *how
many observations make a statistic credible*, *how long is one bar*, *where does
in-sample end*, *what does a round trip cost*, *what counts as significant*.
None of them is represented anywhere.  Each is materialised as several
independent config fields, in different modules, each with its own class
default, and nothing verifies that the copies are jointly satisfiable.  Issue
#173 is what that looks like from the outside: a configuration that cannot
produce a verdict, failing silently as a wall of rejections.

This module is the answer.  It is **not** a defaulting mechanism: it is a
constraint propagator.  The caller sets what they care about, and every field
left at :data:`~forgedge.unset.UNSET` is derived from the constraints that
relate it to what was set.

Two modes, one table
--------------------
Every constraint is a relation between materialisations of one latent
parameter, and it is read in one of two modes depending on the state of the
field it governs:

============  ==========  ==========================================
field state   mode        outcome
============  ==========  ==========================================
``UNSET``     **derive**  the resolver computes it — no conflict possible
set           **check**   left alone; the relation is verified instead
============  ==========  ==========================================

The check mode is what :func:`forgedge.config_report` consumes.  One definition
per constraint, two uses — the resolver and the coherence report can never
disagree, because they are the same table.

Directed, not symmetric
-----------------------
``min_train_months × min_tpm >= 10`` relates three quantities; to be solvable it
must declare which one is **free** (typically the profile choice, ``min_tpm``)
and which is **derived** (``min_train_months``).  Every constraint carries that
direction.  Some declare no derived field at all: deriving them would mean
weakening a statistical guard until it passes, which is exactly the silent
failure this plan exists to remove.  Those are check-only.

Staged activation
-----------------
Constraints carry a :attr:`Constraint.stage`.  Only ``"propagation"`` — moving
an identical value between configs — derives today; ``"statistical"``
constraints are registered and checked here, and each has its ``derive`` turned
on by the issue that owns the corresponding fix (#177, #178, #181, #182).  That
staging is what keeps this change additive: propagation is a no-op at default
values, so no verdict moves.

The resolver never reads the data
---------------------------------
Derivation uses the timeframe, the schema and the configs' own values.  It never
reads ``n_bars`` or ``span_months`` — those are visible to *check* mode only.
Besides being unnecessary (deriving a window from a rate is arithmetic on the
rate), it is a design guard: a resolver that could see the available span would
be tempted to *cap* a requirement to fit it, silently weakening the gate the
caller asked for.  A window that cannot support a test must be reported, not
accommodated.  The invariant also makes ``resolve()`` total and deterministic
without the data, which is what lets the config report show exactly what will
run.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .unset import UNSET, is_set

__all__ = [
    "PipelineContext",
    "Derivation",
    "ResolutionTrace",
    "Constraint",
    "Violation",
    "CONSTRAINTS",
    "resolve",
    "resolve_config",
    "collect_context",
    "timeframe_minutes",
    "poisson_min_window",
]


# ---------------------------------------------------------------------------
# Timeframe arithmetic
# ---------------------------------------------------------------------------

_TF_RE = re.compile(r"^(\d+)\s*([a-zA-Z]+)$")

_UNIT_MINUTES = {
    "m": 1, "min": 1,
    "h": 60, "H": 60,
    "d": 1440, "D": 1440,
    "w": 10080, "W": 10080,
}


def timeframe_minutes(timeframe: str) -> int:
    """Parse a timeframe string (``"1D"``, ``"4H"``, ``"15m"``) into minutes.

    The single implementation of this conversion in the package;
    :mod:`forgedge.presets` reuses it rather than keeping its own copy.
    """
    m = _TF_RE.match(str(timeframe).strip())
    if m is None:
        raise ValueError(
            f"Unrecognised timeframe {timeframe!r}. "
            "Expected format: '1D', '4H', '15m', '1W', etc."
        )
    n, unit = int(m.group(1)), m.group(2)
    if unit not in _UNIT_MINUTES:
        raise ValueError(
            f"Unknown timeframe unit {unit!r} in {timeframe!r}. "
            f"Supported: {list(_UNIT_MINUTES)}"
        )
    return n * _UNIT_MINUTES[unit]


# ---------------------------------------------------------------------------
# Statistical primitives shared by the constraints
# ---------------------------------------------------------------------------

def _poisson_sf(floor: int, lam: float) -> float:
    """``P(X >= floor)`` for ``X ~ Poisson(lam)``.

    Pure-Python recursion over the pmf — the package deliberately carries no
    scipy dependency.  Underflow at very large ``lam`` is harmless: the answer
    there is ``1.0`` to every digit that matters, and the short-circuit says so.
    """
    if floor <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    if lam > 700.0:  # e**-lam underflows; P(X >= floor) is 1.0 for any sane floor
        return 1.0 if lam > floor else 0.0
    term = math.exp(-lam)      # pmf(0)
    cdf = term
    for i in range(1, int(floor)):
        term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def poisson_min_window(floor: int, rate: float, confidence: float = 0.95,
                       cap: float = 1200.0) -> float:
    """Smallest window (in the units of ``rate``) that reaches ``floor`` with
    probability ``confidence`` under Poisson arrivals.

    The naive inversion ``floor / rate`` satisfies the constraint *in
    expectation*, which is not the same as satisfying it.  At ``floor=10`` and
    ``rate=0.8`` it returns 12.5 months — λ = 10.4 expected events, of which
    fewer than 10 are observed about 44 % of the time.  A configuration derived
    that way fails roughly one run in two, which is issue #173 in milder form
    *after* having been "fixed".

    This returns the smallest ``n`` with ``P(Poisson(n · rate) >= floor) >=
    confidence`` — about ``1.65 × floor / rate`` for the default confidence.

    Returns ``inf`` when no window under ``cap`` suffices (a rate so low the
    floor is unreachable in any plausible history).
    """
    if rate <= 0:
        return float("inf")
    if floor <= 0:
        return 0.0
    lo = floor / rate                      # expectation-only lower bound
    n = lo
    step = max(lo * 0.05, 1e-3)
    while n <= cap:
        if _poisson_sf(int(floor), n * rate) >= confidence:
            return n
        n += step
    return float("inf")


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineContext:
    """Session facts that no single module owns.

    Built once per :func:`forgedge.forge` run and threaded through the
    resolver.  Splits deliberately into two halves:

    * **derivable facts** — timeframe, schema, economics, statistical policy.
      These are what *derive* mode may read.
    * **data facts** — ``n_bars`` and ``span_months``.  These are readable by
      *check* mode only; see the module docstring for why the resolver must not
      see them.

    Attributes
    ----------
    timeframe : str
        Bar size, e.g. ``"1D"``, ``"4H"``, ``"15m"``.  The single declared
        source of bar duration for the session.
    timestamp_col, close_col, regime_col, regime_stable_col : str
        The KPI table's schema, propagated to every module that needs it.
    fee_per_side : float
        Round-trip cost basis, one per session.
    alpha : float
        Per-hypothesis significance level (see #182 — the seven p-thresholds
        derive from this one).
    min_sample : int
        Minimum observation count below which a statistic is not credible.
    target_rate_tpm : float or None
        The session's arrival rate.  Note the direction: this is an *output* of
        the preset (its most characterising choice), collected into the context
        — not an input to it.
    cross_pf_retention : float
        Fraction of a rule's home profit factor it must retain elsewhere to
        count as transferring (M4, F8).  Policy rather than data, which is why
        it lives here next to ``alpha`` and ``min_sample`` instead of being a
        literal buried in the cross-ticker loop.
    net_gain_retention : float
        Fraction of the market point's OOS net gain the limit point must retain
        to be adopted (M3, #185).  The same shape as ``cross_pf_retention`` and
        here for the same reason.
    n_bars, span_months : int, float
        Data facts.  **Read by check mode only.**
    """

    timeframe: str = "1H"
    # schema
    timestamp_col: str = "open_dt"
    close_col: str = "close"
    regime_col: str = "regime"
    regime_stable_col: str = "regime_stable"
    # economics
    fee_per_side: float = 0.002
    # statistical policy
    alpha: float = 0.05
    min_sample: int = 10
    target_rate_tpm: Optional[float] = None
    cross_pf_retention: float = 0.8
    net_gain_retention: float = 0.5
    # data facts — check mode only
    n_bars: int = 0
    span_months: float = 0.0

    # -- bar arithmetic ------------------------------------------------

    @property
    def bar_minutes(self) -> int:
        return timeframe_minutes(self.timeframe)

    @property
    def bar_hours(self) -> float:
        return self.bar_minutes / 60.0

    @property
    def bars_per_day(self) -> float:
        return 1440.0 / self.bar_minutes

    @property
    def bars_per_month(self) -> float:
        return self.bars_per_day * 30.0

    def months_of(self, n_bars: int) -> float:
        """Convert a bar count to calendar months at this timeframe."""
        return float(n_bars) / self.bars_per_month

    def bars_of(self, months: float) -> int:
        """Convert calendar months to a bar count at this timeframe."""
        return int(round(float(months) * self.bars_per_month))

    # -- construction --------------------------------------------------

    @classmethod
    def from_frame(cls, frame, *, timeframe: str = "1H", **overrides) -> "PipelineContext":
        """Build a context from a KPI table, filling the data facts.

        Reads the timestamp source once — a ``DatetimeIndex`` or the
        ``timestamp_col`` column — to populate ``n_bars`` and ``span_months``.
        Every other field comes from ``overrides`` or the class default.
        """
        import pandas as pd

        timestamp_col = overrides.get("timestamp_col", cls.timestamp_col)
        n_bars = int(len(frame))
        span_months = 0.0
        stamps = None
        if isinstance(getattr(frame, "index", None), pd.DatetimeIndex):
            stamps = pd.Series(frame.index)
        elif timestamp_col in getattr(frame, "columns", ()):
            stamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
        if stamps is not None and len(stamps.dropna()) >= 2:
            s = stamps.dropna()
            span_days = (s.iloc[-1] - s.iloc[0]).total_seconds() / 86400.0
            span_months = max(span_days / 30.0, 0.0)
        return cls(timeframe=timeframe, n_bars=n_bars, span_months=span_months,
                   **overrides)


# ---------------------------------------------------------------------------
# Resolution trace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Derivation:
    """One field the resolver filled in, and why.

    Attributes
    ----------
    order : int
        Position in the topological order — the numbering is the chain, so
        ``min_tpm → min_train_months → pf_min_tpm`` reads top to bottom.
    field : str
        Dotted path of the field written, e.g.
        ``"rule_discovery.walk_forward.min_train_months"``.
    default : Any
        What the field would have been without the resolver.  The
        ``default → resolved`` diff is the actual debugging question.
    resolved : Any
        What the pipeline will use.
    rule : str
        Code of the constraint that fired.
    reason : str
        Human-readable derivation, with the input values inlined — so the
        number is never a magic number.
    inputs : tuple of str
        Dotted paths read.  Reconstructs the dependency chain.
    superseded : bool
        ``True`` when another constraint produced a more conservative value for
        the same field and won.  The losing derivation is kept, not discarded.
    """

    order: int
    field: str
    default: Any
    resolved: Any
    rule: str
    reason: str
    inputs: Tuple[str, ...] = ()
    superseded: bool = False


@dataclass
class ResolutionTrace:
    """Ordered record of everything :func:`resolve` decided.

    A resolver that adapts parameters behind the caller's back is powerful and
    can surprise.  The trace is what keeps it auditable: every derived value
    carries the rule that produced it and the inputs it read, in the order they
    were applied.
    """

    derivations: List[Derivation] = field(default_factory=list)
    untouched: int = 0

    def __len__(self) -> int:
        return len(self.derivations)

    def __iter__(self):
        return iter(self.derivations)

    @property
    def effective(self) -> List[Derivation]:
        """Derivations that actually took effect (losers of a conflict excluded)."""
        return [d for d in self.derivations if not d.superseded]

    def describe(self) -> str:
        """One-line summary, in the idiom of ``ledger.describe()``."""
        n = len(self.effective)
        return (f"resolution: {n} field(s) derived, "
                f"{self.untouched} left at their documented default")

    def to_text(self, verbose: bool = False) -> str:
        """Full table — the ``default → resolved`` diff with the reason."""
        lines = [
            f"RESOLUTION TRACE  ({len(self.effective)} derived, "
            f"{self.untouched} left at default)"
        ]
        if not self.derivations:
            lines.append("  (nothing to derive — every field was set explicitly)")
            return "\n".join(lines)
        lines.append(f" {'#':>2}  {'field':<44} {'default':>12} → {'resolved':<12} rule")
        for d in self.derivations:
            mark = "  (superseded)" if d.superseded else ""
            lines.append(
                f" {d.order:>2}  {d.field:<44} {str(d.default):>12} → "
                f"{str(d.resolved):<12} {d.rule}{mark}"
            )
            if verbose or d.reason:
                lines.append(f"        ← {d.reason}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Violation:
    """A constraint that failed in *check* mode.

    Returned by :func:`resolve` for :func:`forgedge.config_report` to render;
    this module never prints, warns or raises on one.
    """

    code: str
    level: str      # "FAIL" | "WARN"
    message: str
    fields: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

# Stages — see the module docstring.  Only PROPAGATION derives today.
PROPAGATION = "propagation"    # moves an identical value between configs
STATISTICAL = "statistical"    # derive turned on by the issue owning the fix
STRUCTURAL = "structural"      # check-only, always


@dataclass(frozen=True)
class Constraint:
    """One relation between materialisations of a latent parameter.

    Attributes
    ----------
    code : str
        Stable identifier, shared with the coherence report.
    level : {"FAIL", "WARN"}
        Severity when violated in check mode.  ``FAIL`` is reserved for
        configurations that make a stage *structurally incapable* of producing
        a verdict.
    stage : str
        :data:`PROPAGATION`, :data:`STATISTICAL` or :data:`STRUCTURAL`.
    free : tuple of str
        Dotted paths read.  Also the edges of the dependency graph.
    derived : str or None
        Dotted path written.  ``None`` marks a check-only constraint: deriving
        it would mean weakening a statistical guard until it passes.
    derive : callable or None
        ``(values, ctx) -> (value, reason)``.  Called only when ``derived`` is
        currently ``UNSET`` and the stage is active.
    check : callable or None
        ``(values, ctx) -> str | None``.  Returns a message when the relation
        is violated.
    """

    code: str
    level: str
    stage: str
    free: Tuple[str, ...]
    derived: Optional[str] = None
    derive: Optional[Callable[[Dict[str, Any], PipelineContext], Tuple[Any, str]]] = None
    check: Optional[Callable[[Dict[str, Any], PipelineContext], Optional[str]]] = None


# -- path access -------------------------------------------------------

_MISSING = object()


def _get_path(bundle: Dict[str, Any], path: str) -> Any:
    """Read a dotted path out of the config bundle, or ``_MISSING``."""
    head, *rest = path.split(".")
    node = bundle.get(head, None)
    if node is None:
        return _MISSING
    for part in rest:
        if not hasattr(node, part):
            return _MISSING
        node = getattr(node, part)
    return node


def _set_path(bundle: Dict[str, Any], path: str, value: Any) -> bool:
    """Write a dotted path into the bundle.  ``False`` when the target is absent."""
    head, *rest = path.split(".")
    node = bundle.get(head, None)
    if node is None or not rest:
        return False
    for part in rest[:-1]:
        if not hasattr(node, part):
            return False
        node = getattr(node, part)
    if not hasattr(node, rest[-1]):
        return False
    setattr(node, rest[-1], value)
    return True


# -- the registry ------------------------------------------------------

def _from_context(attr: str):
    """Derive a schema/economics/policy field straight from the context."""
    def _derive(values: Dict[str, Any], ctx: PipelineContext):
        return getattr(ctx, attr), f"session {attr}={getattr(ctx, attr)!r}"
    return _derive


def _check_all_equal(paths: Sequence[str], label: str):
    def _check(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
        seen = {p: v for p, v in values.items() if v is not _MISSING and is_set(v)}
        distinct = set(seen.values())
        if len(distinct) <= 1:
            return None
        detail = ", ".join(f"{p}={v!r}" for p, v in sorted(seen.items()))
        return (f"{label} disagree across modules ({detail}); the pipeline will "
                f"read a different column in each, and the mismatch surfaces "
                f"later as a missing-column error")
    return _check


#: Schema propagation, per context field: every path it fills in, and the
#: subset over which disagreement is a *bug* rather than a choice.
#:
#: The two lists differ in exactly one place, and the difference is the whole
#: point.  ``buy_price_anchor`` is **not a schema field at all**: it names the
#: *reference level* the limit offset is applied to
#: (``buy_price = anchor × (1 ∓ buy_drop_pct)``), and any numeric column on the
#: candle table is legal there — including a derived indicator, which is how
#: "place a limit at 90% of the 3-bar SMA" is expressed
#: (``buy_price_anchor="close_sma_3"``).  It is in the propagation list only
#: because its *default* reference level happens to be the session's close, so
#: renaming the price column has to carry it along.  Checking it for equality
#: against the price column would flag a whole category of legitimate strategy.
#:
#: ``target_col`` is checked: the horizon exit has to be priced on the same
#: series M2 measured its forward returns on, or the two modules are describing
#: different instruments.
#:
#: ``target_hit_col`` appears in neither.  It names an exit *convention*
#: (``"close"`` conservative, ``"high"``/``"low"`` optimistic) and the
#: walk-forward sets it per pass.
_SCHEMA_GROUPS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "timestamp_col": (
        ("event_discovery.timestamp_col", "alpha.timestamp_col",
         "rule_discovery.timestamp_col", "registry.timestamp_col"),
        ("event_discovery.timestamp_col", "alpha.timestamp_col",
         "rule_discovery.timestamp_col", "registry.timestamp_col"),
    ),
    "close_col": (
        ("alpha.close_col", "rule_discovery.base_params.target_col",
         "rule_discovery.base_params.buy_price_anchor"),
        ("alpha.close_col", "rule_discovery.base_params.target_col"),
    ),
    "regime_col": (("alpha.regime_col",), ()),
    "regime_stable_col": (("alpha.regime_stable_col",), ()),
}


def _schema_constraints() -> List[Constraint]:
    """Propagate the KPI table's schema to every module that reads it."""
    out: List[Constraint] = []
    for attr, (paths, checked) in _SCHEMA_GROUPS.items():
        for path in paths:
            out.append(Constraint(
                code="schema_mismatch",
                level="WARN",
                stage=PROPAGATION,
                free=(),
                derived=path,
                derive=_from_context(attr),
            ))
        if len(checked) > 1:
            out.append(Constraint(
                code="schema_mismatch",
                level="WARN",
                stage=PROPAGATION,
                free=tuple(checked),
                derived=None,
                check=_check_all_equal(checked, f"`{attr}`"),
            ))
    return out


def _check_fee(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    alpha = values.get("alpha.fee_per_side", _MISSING)
    rd = values.get("rule_discovery.base_params.fee", _MISSING)
    if alpha is _MISSING or rd is _MISSING:
        return None
    if not (is_set(alpha) and is_set(rd)):
        return None
    if abs(float(alpha) - float(rd)) < 1e-12:
        return None
    return (f"AlphaConfig.fee_per_side={alpha} but BacktestParams.fee={rd}: the "
            f"contracts document one cost basis and the backtest charges "
            f"another. Rule Discovery never reads the contract's fee")


def _derive_backtest_fee(values: Dict[str, Any], ctx: PipelineContext):
    """M3's fee follows the contract's cost basis when the caller set one.

    `_seed_base_params` seeds direction, target_h and sell_pct from the
    contract but never the fee, so a caller who sets `AlphaConfig.fee_per_side`
    gets contracts documenting one cost and backtests charging another (F7).
    """
    upstream = values.get("alpha.fee_per_side", _MISSING)
    if upstream is not _MISSING and is_set(upstream):
        return upstream, f"alpha.fee_per_side={upstream}"
    return ctx.fee_per_side, f"session fee_per_side={ctx.fee_per_side}"


def _derive_cross_pf_floor(values: Dict[str, Any], ctx: PipelineContext):
    """M4's absolute cross-ticker floor is the bar that admitted the rule.

    ``cross_pf_threshold`` used to default to 2.0 independently of M3, which
    made the entire ``PARTIAL-EDGE`` class — admitted at 1.5 — structurally
    incapable of ever being ``GENERIC`` (F8).
    """
    upstream = values.get("rule_discovery.criteria.partial_min_profit_factor", _MISSING)
    if upstream is not _MISSING and is_set(upstream):
        value = float(upstream)
        return value, (f"criteria.partial_min_profit_factor={value:g} — the bar "
                       f"that admitted the rule at home")
    return 1.5, "documented criteria.partial_min_profit_factor default 1.5"


def _derive_tp_floor(values: Dict[str, Any], ctx: PipelineContext):
    """M3's take-profit floor is M2's floor, not an independent stricter one.

    ``_seed_base_params`` clamped with a hardcoded ``max(0.01, sell_pct)`` while
    ``AlphaConfig.mfe_floor`` was 0.005 — so the binding constraint was the one
    the caller could not configure, and on intraday bars it replaced a *derived*
    target with a constant (F11).
    """
    upstream = values.get("alpha.mfe_floor", _MISSING)
    if upstream is not _MISSING and is_set(upstream):
        value = float(upstream)
        return value, f"alpha.mfe_floor={value:g}"
    return 0.005, "documented alpha.mfe_floor default 0.005"


def _derive_min_train_months(values: Dict[str, Any], ctx: PipelineContext):
    """#173 — the selection window sized to the rate it is about to demand.

    Under ``selection_mode="walk_forward"`` the early-elimination screen runs on
    the first train window, whose length is ``min_train_months``.  That window
    then has to produce ``_MIN_TRADES_ABS`` trades at ``criteria.min_tpm``, and
    nothing related the two: at the stock preset values on daily bars the floor
    implied 1.67 trades/month against a configured 0.80, so every candidate was
    eliminated at the first fold regardless of signal quality.

    Sized with a **Poisson 95 % margin**, not the naive ``floor / rate``.  The
    naive inversion satisfies the constraint in expectation, which is not the
    same as satisfying it: at floor 10 and rate 0.8 it gives 12.5 months, and
    fewer than 10 trades arrive about 44 % of the time — #173 in milder form
    after having been "fixed".

    Never reads the data: the window this asks for is what the test *needs*, and
    capping it to the history available would hide the mismatch instead of
    reporting it.  The check half says when the data cannot supply it.
    """
    rate = values.get("rule_discovery.criteria.min_tpm", _MISSING)
    if rate is _MISSING or not is_set(rate) or float(rate) <= 0:
        return 6, "documented walk_forward.min_train_months default 6"
    floor = _min_trades_abs()
    need = poisson_min_window(floor, float(rate))
    if need == float("inf"):
        return 6, (f"criteria.min_tpm={float(rate):g} cannot reach the floor of "
                   f"{floor} at any window; kept the documented default 6")
    months = max(int(math.ceil(need)), 1)
    return months, (f"{floor} trades at criteria.min_tpm={float(rate):g} with a "
                    f"95 % Poisson margin")


CONSTRAINTS: List[Constraint] = [
    *_schema_constraints(),
    # ── genericity: floor + retention ────────────────────────────────
    Constraint(
        code="registry_stricter_than_m3",
        level="WARN",
        stage=PROPAGATION,
        free=("rule_discovery.criteria.partial_min_profit_factor",),
        derived="registry.cross_pf_threshold",
        derive=_derive_cross_pf_floor,
    ),
    Constraint(
        code="registry_stricter_than_m3",
        level="WARN",
        stage=PROPAGATION,
        free=(),
        derived="registry.min_cross_pf_retention",
        derive=_from_context("cross_pf_retention"),
    ),
    Constraint(
        code="entry_adoption_policy",
        level="WARN",
        stage=PROPAGATION,
        free=(),
        derived="rule_discovery.criteria.min_net_gain_retention",
        derive=_from_context("net_gain_retention"),
    ),
    # ── the take-profit floor is M2's, not a second opinion (F11) ─────
    Constraint(
        code="tp_floor_conflict",
        level="WARN",
        stage=PROPAGATION,
        free=("alpha.mfe_floor",),
        derived="rule_discovery.criteria.min_sell_pct",
        derive=_derive_tp_floor,
    ),
    # ── #173: the selection window has to fit the rate it demands ─────
    Constraint(
        code="wf_bucket_too_short",
        level="FAIL",
        stage=PROPAGATION,
        free=("rule_discovery.criteria.min_tpm",),
        derived="rule_discovery.walk_forward.min_train_months",
        derive=_derive_min_train_months,
    ),
    # ── economics ────────────────────────────────────────────────────
    Constraint(
        code="fee_mismatch",
        level="WARN",
        stage=PROPAGATION,
        free=("alpha.fee_per_side",),
        derived="rule_discovery.base_params.fee",
        derive=_derive_backtest_fee,
    ),
    Constraint(
        code="fee_mismatch",
        level="WARN",
        stage=PROPAGATION,
        free=(),
        derived="alpha.fee_per_side",
        derive=lambda v, ctx: (ctx.fee_per_side,
                               f"session fee_per_side={ctx.fee_per_side}"),
    ),
    Constraint(
        code="fee_mismatch",
        level="WARN",
        stage=PROPAGATION,
        free=("alpha.fee_per_side", "rule_discovery.base_params.fee"),
        derived=None,
        check=_check_fee,
    ),
]


# ---------------------------------------------------------------------------
# Statistical and structural constraints — check-only for now
# ---------------------------------------------------------------------------
#
# Every one of these relates materialisations of a latent parameter that the
# audit found mutually unsatisfiable in practice.  They are registered here and
# evaluated in *check* mode; each has its `derive` switched on by the issue that
# owns the corresponding fix (#177, #178, #181, #182).  Until then the
# configuration can still be built wrong — it simply is no longer built wrong in
# silence.

def _need(values: Dict[str, Any], *paths: str) -> Optional[Tuple[Any, ...]]:
    """Return the values at ``paths``, or ``None`` if any is absent or unset.

    A constraint whose inputs are not all present has nothing to say — a
    partial bundle is a supported case, not a violation.
    """
    out = []
    for path in paths:
        v = values.get(path, _MISSING)
        if v is _MISSING or not is_set(v) or v is None:
            return None
        out.append(v)
    return tuple(out)


def _min_trades_abs() -> int:
    """M3's absolute trade floor.  Imported lazily: rule_discovery imports this
    module, so a top-level import would close the cycle."""
    from .rule_discovery.discovery import _MIN_TRADES_ABS

    return int(_MIN_TRADES_ABS)


def _check_wf_bucket(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """#173 — the first walk-forward train window versus the configured rate.

    Under ``selection_mode="walk_forward"`` the early-elimination screen runs on
    the *first train window*, whose length is ``min_train_months`` — a parameter
    nobody relates to ``min_tpm``.  When the product falls under the absolute
    floor, every candidate is eliminated at the first fold regardless of signal
    quality, and nothing says so.
    """
    got = _need(values,
                "rule_discovery.walk_forward.min_train_months",
                "rule_discovery.criteria.min_tpm")
    if got is None:
        return None
    months, rate = float(got[0]), float(got[1])
    mode = values.get("rule_discovery.selection_mode", _MISSING)
    if mode is not _MISSING and is_set(mode) and mode != "walk_forward":
        return None
    floor = _min_trades_abs()
    expected = months * rate
    if expected >= floor:
        return None

    need = poisson_min_window(floor, rate)
    msg = (f"walk_forward.min_train_months={months:g} × criteria.min_tpm={rate:g} "
           f"= {expected:.1f} trade attesi, sotto il floor assoluto di {floor}: "
           f"ogni candidato verrà eliminato al primo fold, indipendentemente dalla "
           f"qualità del segnale (#173). ")
    if need == float("inf"):
        return msg + (f"A questo tasso il floor non è raggiungibile con nessuna "
                      f"finestra: alzare criteria.min_tpm.")
    span = ctx.span_months
    if span and need > span:
        return msg + (f"Servono min_train_months >= {need:.0f} mesi ma i dati ne "
                      f"coprono {span:.0f}: su questi dati non è raggiungibile con "
                      f"nessun valore. Alzare criteria.min_tpm a >= "
                      f"{floor / max(span, 1e-9):.2f}, oppure allungare la storia.")
    return msg + (f"Servono min_train_months >= {need:.0f} (margine di Poisson 95 %), "
                  f"oppure criteria.min_tpm >= {floor / months:.2f}.")


def _check_m1_fold(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F1 — can the walk-forward folds conclude anything at the configured rate?

    Folds are equal-length by construction (``n_oos // n_splits``), so "too
    short" is never a property of one candidate: it is a property of the
    configuration, and it should be said once, before running, rather than
    discovered as 5112 fold failures.

    A fold is testable when its expected episode count ``lambda`` reaches
    :data:`~forgedge.event_discovery.discovery.MIN_FOLD_LAMBDA` (3.0).  Below
    it, ``P(empty fold) = e**-lambda`` exceeds the 5 % convention the rest of
    the plan uses, so an empty fold is compatible with a healthy candidate and
    the fold is measuring its own brevity.

    This check reads ``span_months``, which is why it is a *check* and not a
    derivation: ``n_splits`` could be solved from the span, but the resolver's
    derive half is deliberately blind to the data (see the module docstring) —
    a derivation that could see the available history would be one that fits
    the requirement to the data instead of reporting that they disagree.
    """
    got = _need(values,
                "event_discovery.train_ratio",
                "event_discovery.walk_forward.n_splits",
                "event_discovery.gate_params.min_tpm")
    if got is None:
        return None
    train_ratio, n_splits, rate = float(got[0]), int(got[1]), float(got[2])
    counting = values.get("event_discovery.gate_params.event_counting", "episode")
    if counting != "episode" or train_ratio >= 1.0 or n_splits < 1 or rate <= 0:
        return None
    span = ctx.span_months
    if not span:
        return None

    oos_months = span * (1.0 - train_ratio)
    fold_months = oos_months / n_splits
    lam = fold_months * rate
    if lam >= _MIN_FOLD_LAMBDA:
        return None
    max_splits = int(oos_months * rate / _MIN_FOLD_LAMBDA)
    fix = (f"usare n_splits <= {max_splits}" if max_splits >= 1
           else (f"a questo tasso nemmeno un fold unico è testabile "
                 f"({oos_months:.1f} mesi × {rate:g} = {oos_months * rate:.1f} "
                 f"episodi attesi): abbassare train_ratio o alzare "
                 f"gate_params.min_tpm"))
    return (f"i fold OOS durano {fold_months:.1f} mesi (train_ratio={train_ratio:g}, "
            f"n_splits={n_splits}) e a min_tpm={rate:g} ne attendono {lam:.1f} "
            f"episodi, sotto la soglia di testabilità di {_MIN_FOLD_LAMBDA:g}: "
            f"P(fold vuoto) = {math.exp(-lam):.0%} anche per un candidato sano, "
            f"quindi ogni fold risulterà INDETERMINATO e il walk-forward non "
            f"potrà concludere nulla. {fix}.")


def _check_oos_span(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F4 — the pooled walk-forward test span versus ``min_oos_trades``."""
    got = _need(values,
                "rule_discovery.criteria.min_tpm",
                "rule_discovery.criteria.min_oos_trades",
                "rule_discovery.walk_forward.min_train_months")
    if got is None:
        return None
    rate, min_oos, train_months = float(got[0]), int(got[1]), float(got[2])
    span = ctx.span_months
    if not span:
        return None
    test_months = span - train_months
    if test_months <= 0:
        # Since #177 this is the usual landing place for the #173 configuration:
        # the window is no longer a fixed 6 but derived from the rate, so a
        # permissive `min_tpm` asks for a selection span longer than the data.
        # That is a more accurate statement of the same problem — an unreachable
        # floor is an insufficient window, not a rejected candidate — but it is
        # only useful if it says what to do about it.
        floor = _min_trades_abs()
        rate_needed = poisson_min_window(floor, 1.0) / max(span * 0.5, 1e-9)
        return (f"walk_forward.min_train_months={train_months:g} copre l'intero span "
                f"disponibile ({span:.0f} mesi): non resta nessuna finestra di test. "
                f"La finestra è derivata da criteria.min_tpm={rate:g} — servono "
                f"{floor} trade con margine di Poisson 95 % (#173). Alzare min_tpm a "
                f">= {rate_needed:.2f} perché selezione e test stiano entrambi nello "
                f"span, oppure allungare la storia.")
    expected = test_months * rate
    if expected >= min_oos:
        return None
    return (f"le finestre di test aggregate coprono {test_months:.0f} mesi e a "
            f"criteria.min_tpm={rate:g} producono {expected:.1f} trade attesi, sotto "
            f"criteria.min_oos_trades={min_oos}: ogni verdetto positivo verrà "
            f"degradato a INSUFFICIENT-DATA. Alzare min_tpm a >= "
            f"{min_oos / test_months:.2f}, o abbassare min_oos_trades.")


def _check_m3_vs_m1(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F2 — M3 must not demand a higher rate than M1 admitted."""
    got = _need(values,
                "event_discovery.gate_params.min_tpm",
                "rule_discovery.criteria.min_tpm")
    if got is None:
        return None
    m1, m3 = float(got[0]), float(got[1])
    if m3 <= m1:
        return None
    return (f"criteria.min_tpm={m3:g} (M3) supera gate_params.min_tpm={m1:g} (M1): "
            f"gli eventi ammessi da Event Discovery vengono rifiutati da Rule "
            f"Discovery per una frequenza che nessuno ha mai richiesto a monte. "
            f"forge_preset() tiene M3 leggermente più permissivo di M1 proprio per "
            f"evitarlo.")


def _check_scoring(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F3 — the scoring knobs versus the frequency the gate demands."""
    got = _need(values,
                "rule_discovery.criteria.min_tpm",
                "rule_discovery.scoring.pf_min_tpm")
    if got is None:
        return None
    rate, pf_min_tpm = float(got[0]), float(got[1])
    problems = []
    if pf_min_tpm > rate:
        problems.append(
            f"scoring.pf_min_tpm={pf_min_tpm:g} supera criteria.min_tpm={rate:g}, "
            f"quindi il punteggio composito penalizza regole che il gate ammette")
    target = values.get("rule_discovery.scoring.pf_tpm_target", _MISSING)
    if target is not _MISSING and is_set(target) and float(target) > 0:
        t = float(target)
        if not (0.5 * rate <= t <= 2.0 * rate):
            problems.append(
                f"scoring.pf_tpm_target={t:g} è fuori dalla banda [{0.5 * rate:.2f}, "
                f"{2.0 * rate:.2f}] attorno a criteria.min_tpm={rate:g}: c_norm — e "
                f"quindi pf_score_tpm, che è il criterio di selezione della griglia — "
                f"è calibrato su una frequenza diversa da quella richiesta dal gate")
    return "; ".join(problems) or None


def _untouched_default(path: str, value: Any) -> bool:
    """``True`` when the field still holds its class default.

    The same question ``_warn_if_hourly_grid_on_slow_timeframe`` asks: a value
    the caller chose is their business, but an hourly-calibrated default that
    reached daily data is the footgun.  These fields are plain defaults rather
    than ``UNSET``, so the class is the only place to read them from.
    """
    lazy = {
        "rule_discovery.base_params.target_h": ("rule_discovery.models", "BacktestParams", "target_h"),
        "rule_discovery.base_params.buy_delay_bar": ("rule_discovery.models", "BacktestParams", "buy_delay_bar"),
        "alpha.horizon_grid": ("alpha_discovery.models", "AlphaConfig", "horizon_grid"),
    }
    spec = lazy.get(path)
    if spec is None:
        return False
    import importlib

    mod = importlib.import_module(f".{spec[0]}", package="forgedge")
    cls = getattr(mod, spec[1])
    return cls.__dataclass_fields__[spec[2]].default == value


def _check_timeframe(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F5 — the declared timeframe versus the bar counts configured against it.

    Bar-count fields are only flagged when they still hold their **hourly**
    class default on a slower timeframe: ``target_h=24`` means 24 hours on 1H
    and 24 *days* on 1D, and a caller who wrote 24 deliberately does not need
    telling.  This is the check ``forge()`` already performs for
    ``horizon_grid``, generalised to the two M3 fields that never got it.
    """
    problems = []
    declared = values.get("alpha.timeframe", _MISSING)
    if declared is not _MISSING and is_set(declared) and declared != ctx.timeframe:
        problems.append(
            f"AlphaConfig.timeframe={declared!r} ma la sessione dichiara "
            f"{ctx.timeframe!r}: i due scalano parametri diversi")

    if ctx.bar_minutes > 60:
        bars_per_day = ctx.bars_per_day
        for path, label in (
            ("rule_discovery.base_params.target_h", "BacktestParams.target_h"),
            ("rule_discovery.base_params.buy_delay_bar", "BacktestParams.buy_delay_bar"),
        ):
            v = values.get(path, _MISSING)
            if v is _MISSING or not is_set(v) or not _untouched_default(path, v):
                continue
            problems.append(
                f"{label}={v:g} è il default tarato su barre orarie, e a "
                f"{ctx.timeframe} vale {float(v) / bars_per_day:.0f} giorni")
        grid = values.get("alpha.horizon_grid", _MISSING)
        if (grid is not _MISSING and is_set(grid) and len(grid)
                and _untouched_default("alpha.horizon_grid", grid)):
            problems.append(
                f"max(horizon_grid)={max(grid)} è il default orario e a "
                f"{ctx.timeframe} vale {max(grid) / bars_per_day:.0f} giorni")
    return "; ".join(problems) or None


def _check_split(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F6 — M1 and M2 cutting the timeline at different points."""
    got = _need(values, "event_discovery.train_ratio", "alpha.train_ratio")
    if got is None:
        return None
    m1, m2 = float(got[0]), float(got[1])
    if abs(m1 - m2) < 1e-9 or m1 >= 1.0:
        # train_ratio=1.0 on M1 is a deliberate, documented choice: M1 never
        # observes the forward return, so seeing the whole table leaks no
        # return information (see #180).
        return None
    return (f"DiscoveryConfig.train_ratio={m1:g} e AlphaConfig.train_ratio={m2:g} "
            f"tagliano la timeline in punti diversi, e ForgeResult.time_budget "
            f"riporta solo quello di M2. Passare un TimeBudget esplicito per "
            f"condividere un asse solo.")


def _check_registry_pf(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F8 — the cross-ticker bar versus the bar that admitted the rule."""
    got = _need(values,
                "registry.cross_pf_threshold",
                "rule_discovery.criteria.partial_min_profit_factor")
    if got is None:
        return None
    cross, partial = float(got[0]), float(got[1])
    if cross <= partial:
        return None
    return (f"RegistryConfig.cross_pf_threshold={cross:g} supera "
            f"criteria.partial_min_profit_factor={partial:g}: una regola ammessa a "
            f"PF {partial:g} in casa dovrebbe fare meglio altrove per essere "
            f"GENERIC, quindi l'intera classe PARTIAL-EDGE è strutturalmente esclusa "
            f"dalla genericità.")


def _check_alpha_drift(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F9 — the per-hypothesis thresholds versus the session's alpha."""
    drifted = []
    for path in ("alpha.thresholds.max_p_value", "alpha.thresholds.ic_max_p",
                 "rule_discovery.criteria.max_ttest_p",
                 "rule_discovery.criteria.max_rotation_p"):
        v = values.get(path, _MISSING)
        if v is _MISSING or not is_set(v):
            continue
        if abs(float(v) - ctx.alpha) > 1e-12:
            drifted.append(f"{path.split('.')[-1]}={v:g}")
    if not drifted:
        return None
    return (f"soglie per-ipotesi diverse dall'alpha di sessione ({ctx.alpha:g}): "
            f"{', '.join(drifted)}. Non è un errore — fdr_q e oos_max_p sono "
            f"quantità diverse e restano scelte dal preset — ma le soglie che *sono* "
            f"un alpha dovrebbero derivare da uno solo.")


def _check_tp_floor(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """F11 — M3's hardcoded take-profit floor versus M2's own.

    M3's floor is now a configurable field derived from ``mfe_floor`` (#177), so
    the two agree unless the caller made them disagree — which is the only case
    left worth reporting.  Until then the clamp was a hardcoded 0.01 against
    M2's 0.005 and this check had to guess where it bit (intraday only), because
    firing on every default configuration is the F14 failure mode.
    """
    got = _need(values, "alpha.mfe_floor", "rule_discovery.criteria.min_sell_pct")
    if got is None:
        return None
    mfe_floor, m3_floor = float(got[0]), float(got[1])
    if m3_floor <= mfe_floor:
        return None
    return (f"criteria.min_sell_pct={m3_floor:g} supera AlphaConfig.mfe_floor="
            f"{mfe_floor:g}: un target derivato fra i due viene sostituito da una "
            f"costante in Rule Discovery, e il floor più severo è quello che M2 non "
            f"ha scelto. Il target economico è derivato per evento (invariante 3), "
            f"quindi il floor di M3 dovrebbe seguire mfe_floor, non superarlo.")


#: Mirror of ``forgedge.event_discovery.discovery.MIN_FOLD_LAMBDA``.  Duplicated
#: rather than imported: the resolver is imported *by* every module and must not
#: import them back.  A test pins the two together.
_MIN_FOLD_LAMBDA: float = 3.0


#: `SelectionCriteria.min_fill_rate`'s class default — see `_check_entry_mode_gate`.
_DEFAULT_MIN_FILL_RATE = 0.40


def _check_entry_mode_gate(values: Dict[str, Any], ctx: PipelineContext) -> Optional[str]:
    """A fill-rate gate the caller tuned, that the entry mode makes inert.

    Silent on the untouched class default.  Since ``entry_mode`` defaults to
    ``"auto"`` (#185), where Stage 1 fills ≈ 100%, a check that fired whenever
    the mode is not ``"limit"`` would fire on *every* default configuration —
    the always-on warning that F14 is about, and the one users learn to skip
    past on their way to the real findings.

    The same technique as ``_check_timeframe``: flag the value that was moved,
    not the value that was inherited.  A caller who deliberately sets the
    default back gets silence — the cost of not carrying an ``UNSET`` sentinel
    on this field, and the same limitation its sibling accepts.
    """
    got = _need(values, "rule_discovery.entry_mode", "rule_discovery.criteria.min_fill_rate")
    if got is None:
        return None
    mode, min_fill = got[0], float(got[1])
    if mode == "limit" or min_fill <= 0:
        return None
    if abs(min_fill - _DEFAULT_MIN_FILL_RATE) < 1e-12:
        return None
    return (f"entry_mode={mode!r} riempie ~100 %, quindi criteria.min_fill_rate="
            f"{min_fill:g} non scatta mai e uno dei tre criteri di early "
            f"elimination sparisce. In questa modalità la soglia che conta è "
            f"min_fill_rate_opt.")


def _statistical_constraints() -> List[Constraint]:
    return [
        Constraint(code="wf_bucket_too_short", level="FAIL", stage=STATISTICAL,
                   free=("rule_discovery.walk_forward.min_train_months",
                         "rule_discovery.criteria.min_tpm",
                         "rule_discovery.selection_mode"),
                   check=_check_wf_bucket),
        Constraint(code="m1_oos_fold_too_short", level="FAIL", stage=STATISTICAL,
                   free=("event_discovery.train_ratio",
                         "event_discovery.walk_forward.n_splits",
                         "event_discovery.gate_params.min_tpm",
                         "event_discovery.gate_params.min_episodes",
                         "event_discovery.gate_params.event_counting"),
                   check=_check_m1_fold),
        Constraint(code="oos_span_too_short", level="FAIL", stage=STRUCTURAL,
                   free=("rule_discovery.criteria.min_tpm",
                         "rule_discovery.criteria.min_oos_trades",
                         "rule_discovery.walk_forward.min_train_months"),
                   check=_check_oos_span),
        Constraint(code="m3_stricter_than_m1", level="WARN", stage=STRUCTURAL,
                   free=("event_discovery.gate_params.min_tpm",
                         "rule_discovery.criteria.min_tpm"),
                   check=_check_m3_vs_m1),
        Constraint(code="scoring_uncalibrated", level="WARN", stage=STATISTICAL,
                   free=("rule_discovery.criteria.min_tpm",
                         "rule_discovery.scoring.pf_min_tpm",
                         "rule_discovery.scoring.pf_tpm_target"),
                   check=_check_scoring),
        Constraint(code="timeframe_mismatch", level="WARN", stage=STATISTICAL,
                   free=("alpha.timeframe", "alpha.horizon_grid",
                         "rule_discovery.base_params.target_h",
                         "rule_discovery.base_params.buy_delay_bar"),
                   check=_check_timeframe),
        Constraint(code="split_disagreement", level="WARN", stage=STRUCTURAL,
                   free=("event_discovery.train_ratio", "alpha.train_ratio"),
                   check=_check_split),
        Constraint(code="registry_stricter_than_m3", level="WARN", stage=STATISTICAL,
                   free=("registry.cross_pf_threshold",
                         "rule_discovery.criteria.partial_min_profit_factor"),
                   check=_check_registry_pf),
        Constraint(code="alpha_level_drift", level="WARN", stage=STATISTICAL,
                   free=("alpha.thresholds.max_p_value", "alpha.thresholds.ic_max_p",
                         "rule_discovery.criteria.max_ttest_p",
                         "rule_discovery.criteria.max_rotation_p"),
                   check=_check_alpha_drift),
        Constraint(code="tp_floor_conflict", level="WARN", stage=STATISTICAL,
                   free=("alpha.mfe_floor",
                         "rule_discovery.criteria.min_sell_pct"),
                   check=_check_tp_floor),
        Constraint(code="entry_mode_inert_gate", level="WARN", stage=STRUCTURAL,
                   free=("rule_discovery.entry_mode",
                         "rule_discovery.criteria.min_fill_rate"),
                   check=_check_entry_mode_gate),
    ]


CONSTRAINTS.extend(_statistical_constraints())


#: What each resolvable field would be without the resolver.  The trace's
#: ``default → resolved`` column is the actual debugging question, and once a
#: field's class default becomes ``UNSET`` the historical value is no longer
#: machine-readable — so it is recorded here, next to the constraints that
#: derive it.
_DOCUMENTED_DEFAULTS: Dict[str, Any] = {
    "event_discovery.timestamp_col": "open_dt",
    "alpha.timestamp_col": "open_dt",
    "rule_discovery.timestamp_col": "open_dt",
    "registry.timestamp_col": "open_dt",
    "alpha.close_col": "close",
    "alpha.regime_col": "regime",
    "alpha.regime_stable_col": "regime_stable",
    "rule_discovery.base_params.target_col": "close",
    "rule_discovery.base_params.buy_price_anchor": "close",
    "alpha.fee_per_side": 0.002,
    "rule_discovery.base_params.fee": 0.002,
    "registry.cross_pf_threshold": 2.0,
    # `registry.min_cross_pf_retention` is deliberately absent: the retention
    # half of the criterion had no value before this table entry would have
    # described one, and the fallback (`UNSET → 0.8`) says exactly that.
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _topological(constraints: Sequence[Constraint]) -> List[Constraint]:
    """Order constraints so a derived field is written before it is read.

    Kahn's algorithm over the field-dependency graph, with a deterministic
    tie-break on declaration order.  A cycle is a configuration error in the
    table itself, not in the user's config, so it raises.
    """
    indexed = list(enumerate(constraints))
    producers: Dict[str, List[int]] = {}
    for i, c in indexed:
        if c.derived:
            producers.setdefault(c.derived, []).append(i)

    edges: Dict[int, set] = {i: set() for i, _ in indexed}
    indegree: Dict[int, int] = {i: 0 for i, _ in indexed}
    for i, c in indexed:
        for path in c.free:
            for j in producers.get(path, ()):
                if j != i and i not in edges[j]:
                    edges[j].add(i)
                    indegree[i] += 1

    queue = sorted(i for i, d in indegree.items() if d == 0)
    order: List[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j in sorted(edges[i]):
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)
        queue.sort()

    if len(order) != len(indexed):
        stuck = sorted(constraints[i].code for i in indegree if indegree[i] > 0)
        raise RuntimeError(
            f"Cyclic constraint dependency among {stuck}. The constraint table "
            f"is malformed — a derived field is, transitively, its own input."
        )
    return [constraints[i] for i in order]


def _copy_configs(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-ish copy: dataclasses are replaced, so the caller's objects are never
    mutated.  ``resolve()`` returning copies is what makes it safe for
    ``config_report()`` to call on a user's live configs."""
    import copy

    return {k: (copy.deepcopy(v) if v is not None else None) for k, v in bundle.items()}


def resolve(
    bundle: Dict[str, Any],
    ctx: Optional[PipelineContext] = None,
    *,
    active_stages: Sequence[str] = (PROPAGATION,),
) -> Tuple[Dict[str, Any], ResolutionTrace, List[Violation]]:
    """Fill every ``UNSET`` field the constraints can derive, and check the rest.

    Parameters
    ----------
    bundle : dict
        ``{"market_context": …, "event_discovery": …, "alpha": …,
        "rule_discovery": …, "registry": …}``.  Missing or ``None`` entries are
        skipped, so a partial bundle resolves fine.
    ctx : PipelineContext, optional
        Session facts.  Defaults to :class:`PipelineContext` — which reproduces
        today's class defaults exactly, so standalone use is unchanged.
    active_stages : sequence of str
        Which constraint stages may **derive**.  Defaults to propagation only;
        the statistical stages are switched on by the issues that own them
        (#177, #178, #181, #182).  Check mode runs for every stage regardless.

    Returns
    -------
    (resolved_bundle, trace, violations)
        The bundle is a **copy** — the caller's configs are never mutated.

    Notes
    -----
    Idempotent: resolving an already-resolved bundle derives nothing and
    returns an empty trace.
    """
    ctx = ctx or PipelineContext()
    out = _copy_configs(bundle)
    trace = ResolutionTrace()
    violations: List[Violation] = []
    winners: Dict[str, Tuple[Any, str]] = {}
    order = 0

    for c in _topological(CONSTRAINTS):
        values = {p: _get_path(out, p) for p in c.free}

        # ── derive ────────────────────────────────────────────────────
        if c.derived and c.derive and c.stage in active_stages:
            current = _get_path(out, c.derived)
            if current is not _MISSING and not is_set(current):
                value, reason = c.derive(values, ctx)
                if is_set(value) and value is not None:
                    order += 1
                    prior = winners.get(c.derived)
                    superseded = False
                    if prior is not None:
                        # Conflict on the same field: the more conservative
                        # value wins, and the loser stays in the trace.
                        keep = _more_conservative(prior[0], value)
                        superseded = keep is not value
                        if not superseded:
                            _set_path(out, c.derived, value)
                            winners[c.derived] = (value, c.code)
                    else:
                        _set_path(out, c.derived, value)
                        winners[c.derived] = (value, c.code)
                    trace.derivations.append(Derivation(
                        order=order,
                        field=c.derived,
                        default=_documented_default(c.derived),
                        resolved=value,
                        rule=c.code,
                        reason=reason,
                        inputs=c.free,
                        superseded=superseded,
                    ))

        # ── check ─────────────────────────────────────────────────────
        if c.check is not None:
            values = {p: _get_path(out, p) for p in c.free}
            message = c.check(values, ctx)
            if message:
                violations.append(Violation(
                    code=c.code, level=c.level, message=message, fields=c.free
                ))

    trace.untouched = _count_unset(out)
    return out, trace, violations


#: Where each context field may be *collected from* — level 3 of the precedence
#: ladder.  A value the caller wrote into any config is a choice, and the
#: session adopts it rather than overwriting it with a class default.  This is
#: what makes ``forge_preset(timestamp_col="ts")`` reach M2/M3/M4: the preset
#: sets M1's, the context collects it, and the resolver distributes it.
_CONTEXT_SOURCES: Dict[str, Tuple[str, ...]] = {
    "timestamp_col": (
        "event_discovery.timestamp_col",
        "alpha.timestamp_col",
        "rule_discovery.timestamp_col",
        "registry.timestamp_col",
    ),
    # Note the absence of `rule_discovery.base_params.buy_price_anchor`, which
    # `_SCHEMA_GROUPS` does fill in from this field.  Propagation and seeding
    # are not symmetric, because the anchor is a *level*, not a name for the
    # price series: `buy_price_anchor="close_sma_3"` says where to put the
    # limit, not what the KPI table calls its close.  Seeding from it would
    # push "close_sma_3" back out into `alpha.close_col` and have M2 measure
    # forward returns on a moving average.  A field seeds the context only when
    # setting it means "this is what the column is called".
    "close_col": (
        "alpha.close_col",
        "rule_discovery.base_params.target_col",
    ),
    "regime_col": ("alpha.regime_col",),
    "regime_stable_col": ("alpha.regime_stable_col",),
    "fee_per_side": (
        "alpha.fee_per_side",
        "rule_discovery.base_params.fee",
    ),
}


def collect_context(
    bundle: Dict[str, Any],
    base: Optional[PipelineContext] = None,
    **overrides: Any,
) -> PipelineContext:
    """Build the session context, seeding it from explicitly-set config fields.

    Precedence, strongest first:

    1. ``overrides`` — ``forge()``'s own arguments;
    2. ``base`` — an explicit :class:`PipelineContext` from the caller;
    3. fields the caller set in any config (this function's job);
    4. the :class:`PipelineContext` class default.

    When two configs disagree the first source in :data:`_CONTEXT_SOURCES` wins;
    the disagreement itself is reported by the corresponding check constraint,
    never silently reconciled.
    """
    base = base or PipelineContext()
    collected: Dict[str, Any] = {}
    for attr, paths in _CONTEXT_SOURCES.items():
        if attr in overrides:
            continue
        if getattr(base, attr) != getattr(PipelineContext, attr, None):
            continue  # the caller's explicit context outranks the configs
        for path in paths:
            value = _get_path(bundle, path)
            if value is not _MISSING and is_set(value) and value is not None:
                collected[attr] = value
                break
    return replace(base, **{**collected, **overrides})


def resolve_config(cfg: Any, kind: str, ctx: Optional[PipelineContext] = None,
                   **kwargs) -> Any:
    """Resolve a single config, for standalone module use.

    Every module's ``run()`` opens with this, so ``what you inspect is what
    runs`` holds for hand-built pipelines too — not only for :func:`forge`.
    Cross-config constraints simply find nothing to read and stay inert.
    """
    if cfg is None:
        return cfg
    out, _trace, _violations = resolve({kind: cfg}, ctx, **kwargs)
    return out[kind]


def _more_conservative(a: Any, b: Any) -> Any:
    """Pick the safer of two derivations for the same field.

    Numerically that is the larger value: every derived quantity in this table
    is a *requirement* (a window length, a floor, a count), so demanding more is
    the conservative direction.  Non-numeric values keep the first derivation.
    """
    try:
        return a if float(a) >= float(b) else b
    except (TypeError, ValueError):
        return a


def _documented_default(path: str) -> Any:
    """What the field would have been without the resolver (see
    :data:`_DOCUMENTED_DEFAULTS`)."""
    return _DOCUMENTED_DEFAULTS.get(path, UNSET)


def _count_unset(bundle: Dict[str, Any]) -> int:
    """Count fields still at ``UNSET`` after resolution (the documented-default
    population)."""
    total = 0

    def _walk(obj: Any, depth: int = 0) -> None:
        nonlocal total
        if depth > 3 or obj is None:
            return
        fields = getattr(type(obj), "__dataclass_fields__", None)
        if not fields:
            return
        for name in fields:
            value = getattr(obj, name, None)
            if value is UNSET:
                total += 1
            elif hasattr(type(value), "__dataclass_fields__"):
                _walk(value, depth + 1)

    for cfg in bundle.values():
        _walk(cfg)
    return total
