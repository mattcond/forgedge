"""Core data structures for the Rule Discovery module (FORGE Modulo 3).

Rule Discovery receives an :class:`~forgedge.alpha_discovery.models.AlphaContract`
from Alpha Discovery and answers an operational question: *is the statistically
flagged pattern exploitable in a realistic backtest — with fees, a finite fill
rate, limit orders and a discrete target?*

The dataclasses here mirror, field by field, the structures documented in
``docs/modules/RuleDiscovery.md``.  Rule Discovery never alters the event
expression: it reconstructs the event with the parameters stored on the Event
Candidate, parametrises the order mechanics, backtests them, validates the
result out-of-sample with a walk-forward scheme, and emits a verdict
(``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ..unset import UNSET, coalesce, is_set


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestParams:
    """Operational parameters of a single backtest run.

    These are the only degrees of freedom Rule Discovery explores — the event
    expression itself is fixed.  Defaults follow the ``BASE_CONFIG`` of the
    Rule Discovery specification (Section 3.2).

    Attributes
    ----------
    direction : str
        ``"long"`` (default) or ``"short"``.  The short side is the exact mirror
        of the long: the limit entry sits *above* the anchor at
        ``anchor * (1 + buy_drop_pct)`` and fills when ``high`` reaches it; the
        take-profit sits *below* at ``entry * (1 - sell_pct)`` and is reached
        when the price falls to it; the net gain is ``(entry - exit) / entry``.
    buy_type : str
        ``"limit"`` — place a limit order at ``anchor * (1 ∓ buy_drop_pct)``
        (below for long, above for short), filled if touched within
        ``buy_delay_bar`` bars; ``"market"`` — enter at the next bar's open.
    buy_drop_pct : float
        Magnitude of the limit offset from the anchor (e.g. ``0.01`` = 1%): a
        discount for a long entry, a premium for a short entry.  Ignored when
        ``buy_type == "market"``.
    buy_delay_bar : int
        Number of bars the limit order stays live.  Ignored for market orders.
    buy_price_anchor : str
        Column the limit offset is applied to:
        ``buy_price = anchor × (1 ∓ buy_drop_pct)``.  **Any numeric column on
        the candle table is legal**, not only a price column — anchoring on a
        derived indicator is how "place a limit at 90% of the 3-bar SMA" is
        expressed (``buy_price_anchor="close_sma_3", buy_drop_pct=0.10``).

        Session-resolved (default ``"close"``) so that renaming the price
        column carries the *default* anchor along; an explicit anchor is a
        reference level of its own and is never checked against ``close_col``.
    sell_pct : float
        Take-profit target as a fraction of the fill price (e.g. ``0.04`` = 4%):
        above the entry for a long, below it for a short.
    target_h : int
        Number of bars held *after* the fill bar before the horizon exit; the
        position is closed at the target bar's close if the take-profit was
        never reached. The total signal→exit span is always
        ``1 (signal→fill, unavoidable — you cannot act on a bar's own close
        before it happens) + target_h (fill→exit)``, so "hold for N bars from
        entry" maps to ``target_h = N - 1``. ``target_h=0`` is a legal,
        meaningful value: it exits at the fill bar's own close (a same-session
        round-trip, ``fill_rn == exit_rn``) — it is not a "no horizon"
        placeholder.
    target_col : str
        Price column used for the close-at-horizon exit.  Session-resolved from
        the KPI table's ``close_col``; default ``"close"``.  Its sibling
        ``target_hit_col`` deliberately stays a literal: it selects an exit
        *convention* (conservative vs optimistic), not a schema fact.
    target_hit_col : str
        Price column scanned to detect the take-profit during the exit window.
        ``"close"`` (default) reproduces the certified reference engine — the
        target is hit only when a bar *closes* past ``sell_price`` (conservative).
        The optimistic intrabar convention is ``"high"`` for a long and ``"low"``
        for a short (resolve it with :func:`forgedge.rule_discovery.optimistic_hit_col`).
    fee : float
        Fee per side (e.g. ``0.002``).  The round-trip cost applied is
        ``fee * 2``.  Session-resolved from ``AlphaConfig.fee_per_side`` — the
        contract's cost basis and the cost actually charged are one value, not
        two independent copies that happened to share a default (F7).  Default
        ``0.002``.
    early_stopping : bool
        ``True`` — scan ``target_hit_col`` in the exit window and exit at
        ``sell_price`` on the first bar that reaches the take-profit.
        ``False`` — always exit at the target bar's close.
    """

    direction: str = "long"
    buy_type: str = "limit"
    buy_drop_pct: float = 0.010
    buy_delay_bar: int = 6
    buy_price_anchor: str = UNSET
    sell_pct: float = 0.040
    target_h: int = 24
    target_col: str = UNSET
    target_hit_col: str = "close"
    fee: float = UNSET
    early_stopping: bool = True

    def merged(self, **overrides) -> "BacktestParams":
        """Return a copy with ``overrides`` applied."""
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return BacktestParams(**data)

    def resolved(self) -> "BacktestParams":
        """Return a copy with every ``UNSET`` field at its documented default.

        Three fields (``fee``, ``target_col``, ``buy_price_anchor``) are
        session-resolved, so a caller who never went through
        :func:`forgedge.resolve` — a hand-built ``BacktestParams`` handed
        straight to :func:`run_backtest` — holds the sentinel rather than a
        value.  ``UNSET`` means *you decide*, and a function that documents a
        default has already decided; this is where that decision is applied,
        once, instead of being rediscovered at each read site.

        Returns ``self`` unchanged when there is nothing to resolve, so the
        grid's ~210 backtests per run do not each pay for a copy.
        """
        if is_set(self.fee) and is_set(self.target_col) and is_set(self.buy_price_anchor):
            return self
        return self.merged(
            fee=coalesce(self.fee, default=0.002),
            target_col=coalesce(self.target_col, default="close"),
            buy_price_anchor=coalesce(self.buy_price_anchor, default="close"),
        )


@dataclass
class ScoringParams:
    """Parameters of the composite scoring metrics (``pf_score_tpm`` & co).

    Mirrors the exposed knobs of ``backtest_scoring.md``.  The dynamic
    trade-count threshold is ``max(pf_min_trades, n_months * pf_min_tpm)``;
    ``pf_tpm_target`` is the target trades-per-month for the consistency term
    ``C_norm`` and **must be calibrated** against the realised ``tpm_mu``.

    Attributes
    ----------
    pf_min_trades : int
        Absolute floor of the dynamic trade-count threshold.
    pf_min_tpm : int
        Minimum trades/month feeding the dynamic threshold.
    pf_tpm_target : int
        Target trades/month for the monthly-consistency term ``C_norm``.
    """

    pf_min_trades: int = 15
    pf_min_tpm: int = 2
    pf_tpm_target: int = 3


@dataclass
class GridSpec:
    """Search grid over the operational parameters (Step 2.4).

    Each attribute is a list of candidate values; the screening evaluates the
    full cartesian product.  ``None`` means "do not vary — use the value from
    :class:`BacktestParams`".  When the grid is left empty Rule Discovery
    builds a sensible default centred on the Alpha Contract's derived target
    — including ``target_h``, whose auto-built fan now reaches down to ``0``
    (a same-session hold) instead of floored at ``1``.  See
    :attr:`BacktestParams.target_h` for what the value counts.
    """

    buy_drop_pct: Optional[List[float]] = None
    sell_pct: Optional[List[float]] = None
    target_h: Optional[List[int]] = None
    buy_delay_bar: Optional[List[int]] = None

    def is_empty(self) -> bool:
        return not any(
            getattr(self, f) for f in ("buy_drop_pct", "sell_pct", "target_h", "buy_delay_bar")
        )


@dataclass
class RuleWalkForwardConfig:
    """Walk-forward out-of-sample validation parameters (Step 4).

    The timeline is divided into ``n_splits`` consecutive **test** windows.
    For every split the operational parameters are re-selected on the train
    window that precedes it (a grid screening optimising ``pf_score_tpm``), then
    evaluated *once* on the untouched test window.  Concatenating the per-split
    test trades gives the honest out-of-sample track record of the rule.

    Attributes
    ----------
    n_splits : int
        Number of out-of-sample test windows.
    train_span_months : int or None
        Length of each train window, in calendar months.  ``None`` → anchored
        walk-forward: the train window always starts at the beginning of the
        data and grows.  An integer → rolling walk-forward with a fixed-length
        train window.
    test_span_months : int or None
        Length of each test window, in months.  ``None`` → the post-train span
        is divided equally into ``n_splits`` windows.
    min_train_months : int
        Minimum train length required before the first test window.
    reoptimise : bool
        ``True`` (default) — re-run the grid screening on every train window
        (true walk-forward optimisation).  ``False`` — keep the in-sample best
        parameters fixed and only replay them out-of-sample.
    purge_bars : int or None
        Purge width, in bars, at the end of every **train** window: entries
        opened in the last ``purge_bars`` bars of a train window have their
        fill/exit windows inside the adjacent test window, so the parameter
        selection would be scored on test prices.  ``None`` (default) sizes
        the purge automatically from the resolved grid — the largest
        ``target_h`` plus the fill delay.  ``0`` disables purging (the
        pre-TimeBudget behaviour).
    embargo_bars : int
        Extra quarantine at the start of every **test** window, in bars.
        Default ``0`` (the purge already removes the mechanical overlap).
    """

    n_splits: int = 4
    train_span_months: Optional[int] = None
    test_span_months: Optional[int] = None
    min_train_months: int = 6
    reoptimise: bool = True
    purge_bars: Optional[int] = None
    embargo_bars: int = 0


#: Backwards-compatible alias.  Until it was renamed, this class shared the
#: name ``WalkForwardConfig`` with
#: :class:`forgedge.event_discovery.models.EventWalkForwardConfig` — a
#: different dataclass, with a ``n_splits`` field carrying different semantics
#: (walk-forward *test* windows here, OOS *validation* windows there).  The
#: top-level ``forgedge.WalkForwardConfig`` resolves to *this* one, while
#: ``forge()``'s own docstring example imports the other from
#: ``forgedge.event_discovery.models``.  Prefer the explicit name.
WalkForwardConfig = RuleWalkForwardConfig


@dataclass
class SelectionCriteria:
    """Acceptance thresholds for the in-sample selection and the verdict (Step 3/8).

    Attributes
    ----------
    min_profit_factor : float
        Minimum profit factor for ``EDGE``.
    min_win_rate : float
        Minimum win rate (0–1).
    min_tpm : float
        Minimum trades/month.  This is the sole frequency gate: the minimum
        executed-trade count is **not** a fixed absolute but is derived from it
        as ``max(10, n_months * min_tpm)`` (see ``_dynamic_min_trades``), so the
        requirement scales with the in-sample length instead of penalising short
        IS periods or under-demanding long ones (spec RD-04).
    min_pf_score_tpm : float
        Minimum composite ``pf_score_tpm``.
    min_fill_rate : float
        Minimum fill rate — below it the limit discount is too deep.

        **Inert under the default ``entry_mode="auto"``**, where Stage 1 is a
        market entry and fills ≈ 100%.  The field is kept, not removed: it is
        fully meaningful in ``entry_mode="limit"``, and under ``"auto"`` the
        floor that actually bites is ``min_fill_rate_opt``.  ``config_report()``
        raises ``entry_mode_inert_gate`` when this has been *moved off its
        default* and the mode makes it inert, rather than letting a
        deliberately-tuned gate pass silently unused.
    min_fill_rate_opt : float
        Fill-rate floor for the ``entry_mode="auto"`` limit-optimisation stage.
        Distinct from ``min_fill_rate`` (the general gate): the limit optimiser
        may only improve the operating point of an already-real market edge if
        the optimised entry still fills at ``>= min_fill_rate_opt`` — high by
        design (default ``0.80``), so a deep, rarely-filled limit cannot inflate
        PF on a non-tradeable subset of trades (the "fill confound").
    min_net_gain_retention : float
        Fraction of the market point's out-of-sample **net gain** the limit
        point must retain to be adopted — the third and last of the adoption
        conditions, and deliberately the loosest.  Its job is not to co-decide
        with the annualised Sharpe but to backstop the one case the Sharpe
        cannot see: a tiny mu with a tiny sigma gives an excellent Sharpe while
        producing almost nothing, because the Sharpe is scale-free in mu.  At
        the default ``0.5`` the Sharpe decides and this floor only intervenes
        once half the return has been burnt.

        Session-resolved.  Same shape as ``RegistryConfig.min_cross_pf_retention``
        (#181): absolute condition plus relative retention.
    partial_min_profit_factor : float
        Lower PF bound for a ``PARTIAL-EDGE`` (between this and
        ``min_profit_factor``).
    min_active_month_rate : float
        Minimum fraction of IS months that must contain at least one trade
        for a full ``EDGE`` verdict.  Computed as
        ``active_months / n_months >= min_active_month_rate``.  A rate-based
        threshold is timeframe-agnostic: on 1H data (tpm ≫ 1) the rate is
        naturally near 1.0 without any special calibration; on 1D data
        (tpm ~ 1.5–4) a Poisson process with dispersion up to
        ``max_dispersion`` (ConsistencyGate) yields an active rate in the
        0.75–0.95 range, which the default 0.80 accommodates correctly.
    max_regime_dependency : float
        Maximum regime-dependency score for a full ``EDGE``.
    min_dsr : float
        Minimum Deflated Sharpe Ratio for a full ``EDGE``.  A DSR that is
        *undefined* because the selection haircut's radicand went negative
        (``n_trials`` too large for ``n_obs`` — "selection bias so severe the
        Sharpe is not credible") also blocks a full ``EDGE``.
    max_ttest_p : float
        Maximum p-value for the win-rate / expectancy significance tests.
    max_rotation_p : float
        Maximum search-level rotation-null p-value (``AlphaContract.rotation_p``,
        annotated by ``FastRotationNull`` / ``RotationCalibrator``) for a full
        ``EDGE``.  The rotation p prices the *whole* discovery surface — the
        probability that a search over the same candidates on rotation-decoupled
        returns produces a statistic at least as good — so requiring it caps
        rules that merely won the multiple-testing lottery at ``PARTIAL-EDGE``.
        Inert when the contract carries no annotation (standalone Rule
        Discovery, or ``forge(fast_null=False)``).
    power_gate : bool
        §3.2 — power-aware verdicts.  When ``True`` (default) a verdict that
        would be ``EDGE`` / ``PARTIAL-EDGE`` is degraded to
        ``INSUFFICIENT-DATA`` when the out-of-sample evidence cannot support
        it: no walk-forward was possible, the **pooled** OOS trade count is
        below ``min_oos_trades``, or the pooled OOS sample's minimum
        detectable expectancy exceeds the in-sample expectancy being claimed
        (the OOS could not confirm an effect of that size even if it were
        real).  The assessment reads only the *concatenated* test-window
        ledger — never per-window counts: walk-forward windows are short by
        design and are not individually gated.  ``NON-EDGE`` verdicts are
        never rescued (the operational consequence — do not trade — is the
        same either way).  ``INSUFFICIENT-DATA`` keeps its ``ValidatedRule``
        for future re-evaluation but is not tradeable (``is_edge`` is
        ``False``) and never reaches the Rule Registry.
    min_oos_trades : int
        Minimum pooled walk-forward test trades (across **all** test windows)
        for a confident positive verdict.  Below it the verdict is
        ``INSUFFICIENT-DATA``.  Never applied per window.
    early_elimination : bool
        When ``True`` (default), a rule that fails the fast in-sample screen
        (Step 2.3 — too few trades, losing in-sample, or unfillable) is rejected
        as ``NON-EDGE`` immediately, *before* the expensive walk-forward and the
        regime / envelope / MAE-MFE diagnostics, to save compute.  When
        ``False`` the full pipeline always runs: the rule still gets a
        ``NON-EDGE`` verdict carrying the same reasons, but with the walk-forward
        and every diagnostic populated — useful for uniform reporting and to see
        how a weak rule would have behaved out-of-sample.
    """

    min_profit_factor: float = 2.0
    min_win_rate: float = 0.55
    min_tpm: float = 2.0
    min_pf_score_tpm: float = 0.30
    min_fill_rate: float = 0.40
    min_fill_rate_opt: float = 0.80
    min_net_gain_retention: float = UNSET
    partial_min_profit_factor: float = 1.5
    min_active_month_rate: float = 0.80
    max_regime_dependency: float = 0.30
    min_dsr: float = 1.0
    max_ttest_p: float = 0.05
    max_rotation_p: float = 0.05
    power_gate: bool = True
    min_oos_trades: int = 10
    early_elimination: bool = True


@dataclass
class RuleDiscoveryConfig:
    """Top-level configuration for the Rule Discovery pipeline.

    Attributes
    ----------
    base_params : BacktestParams
        Fixed parameters and grid centre.  Rule Discovery overrides
        ``buy_type``/``sell_pct``/``target_h`` from the Alpha Contract when
        ``use_contract_target`` is set.
    scoring : ScoringParams
        Composite-scoring knobs.
    grid : GridSpec
        Operational grid; auto-built around the contract target when empty.
    walk_forward : RuleWalkForwardConfig
        OOS walk-forward settings.
    criteria : SelectionCriteria
        Acceptance / verdict thresholds.
    entry_mode : str
        How the entry order is evaluated — ``"auto"`` (default), ``"market"``
        or ``"limit"``.

        * ``"auto"`` — two stages.  Stage 1 evaluates the rule at a **market**
          entry and that verdict is authoritative.  Stage 2 sweeps
          ``buy_drop_pct`` on the survivors, replays the winner out-of-sample,
          and publishes it only if it clears all three adoption conditions (see
          :class:`EntryOptimization`).  Stage 2 chooses *which parameters* get
          published, never *whether* the rule is an edge.
        * ``"market"`` — the baseline alone: enter at the next bar's open (fill
          ≈ 100%), no entry optimiser.  Isolates the *signal's* edge; the
          ``fill_rate`` gate is effectively inert.
        * ``"limit"`` — the pre-#185 default: the grid varies ``buy_drop_pct``
          and the limit entry doubles as an entry-price optimiser.  Fully
          supported, and the right choice when the limit order *is* the strategy
          rather than an execution refinement — but be aware of what it mixes
          together (below).

        **Why the default moved.**  In ``"limit"`` mode the entry does double
        duty: order mechanic *and* entry-price optimiser.  A deeper discount
        fills more rarely and, crucially, **only on the paths that came back to
        it** — so the profit factor rises on a subset of trades that is not the
        tradeable population.  The verdict then measures the entry price rather
        than the signal.  ``"auto"`` keeps both readings and separates them.

        Both operating points and the adoption decision are reported on
        ``RuleDiscoveryResponse.entry_optimization``.
    use_contract_target : bool
        Seed ``sell_pct``/``target_h`` from the contract's derived target.
    timestamp_col : str
        Datetime column on the KPI table (default ``"open_dt"``).
    signal_col : str
        Name of the reconstructed boolean signal column injected into the
        candle table.
    discovery_date : str or None
        ISO date stamped onto the response; ``None`` → today.
    selection_mode : {"walk_forward", "full_sample"}
        Where the published operating point is *selected* (§3.4 of the FORGE
        2.0 functional analysis).

        * ``"walk_forward"`` (default) — the operating point comes from the
          walk-forward's train windows only (per ``wf_param_policy``); the
          in-sample summary, the early-elimination screen, the statistical
          validation and every diagnostic are computed on the **selection
          span** ``[start, last train window end)``, so no metric that feeds
          the verdict or the published ``ValidatedRule`` ever reads the final
          test window.  The early-elimination pre-screen runs on the *first*
          train window (selection-side data only).  Falls back to
          ``"full_sample"`` (with a note) when the data span is too short for
          a single walk-forward split.
        * ``"full_sample"`` — the legacy behaviour: grid screening, selection
          and IS metrics on the whole table.  The walk-forward still gates the
          verdict, but the published parameters were exposed to its test
          windows.
    wf_param_policy : {"last", "consensus"}
        How ``selection_mode="walk_forward"`` picks the published operating
        point from the per-split train selections: ``"last"`` (default) — the
        most recent train window's winner (what you would trade next);
        ``"consensus"`` — the most frequent parameter set across splits,
        ties broken toward the most recent.
    n_trials_upstream : int
        Multiplier folded into the Deflated-Sharpe ``n_trials`` on top of the
        grid-cell count, for callers who want the analytic haircut to include
        an explicit upstream search factor (e.g. the number of sibling
        contracts receiving a verdict).  Default ``1`` — grid cells only, the
        historical behaviour.  Note the stock haircut becomes undefined
        (blocking a full ``EDGE``) roughly beyond ``n_trials > n_obs^1.73``;
        the correlation-aware correction for the full discovery surface is the
        rotation-null gate (``SelectionCriteria.max_rotation_p``), not this
        knob — see ``forgedge.ledger``.
    """

    base_params: BacktestParams = field(default_factory=BacktestParams)
    scoring: ScoringParams = field(default_factory=ScoringParams)
    grid: GridSpec = field(default_factory=GridSpec)
    walk_forward: RuleWalkForwardConfig = field(default_factory=RuleWalkForwardConfig)
    criteria: SelectionCriteria = field(default_factory=SelectionCriteria)
    entry_mode: str = "auto"
    use_contract_target: bool = True
    timestamp_col: str = UNSET
    signal_col: str = "__rule_signal__"
    discovery_date: Optional[str] = None
    selection_mode: str = "walk_forward"
    wf_param_policy: str = "last"
    n_trials_upstream: int = 1


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BacktestSummary:
    """Aggregated metrics of a single backtest run.

    Holds the standard performance metrics plus the composite scores defined in
    ``backtest_scoring.md``.  All attributes carry ``nan`` when there are no
    executed trades.

    Overlap
    -------
    ``run_backtest`` opens a trade on **every** active bar, with no flat-state
    check.  That is a deliberate, capital-permitting policy — the reported
    economics are reproducible in production *given enough capital to fund the
    concurrent positions* — but until issue #168 there was no supported way to
    find out how much capital that is.  Three fields answer it:

    n_episodes : int
        Activation *episodes* behind the trades — consecutive (gap-bridged)
        active bars counted as one thing happening.  Answers "how often does
        this signal fire".
    mean_concurrent_positions : float
        Average open positions over the bars where at least one is open.
        Answers "when this rule is working, how many positions am I funding" —
        and ``total_trades / mean_concurrent_positions`` is the sample size the
        overlap actually supports.
    max_concurrent_positions : int
        The peak.  Decides whether the rule is deployable at all on a given
        account.

    Episodes and concurrency are **different measures and generally disagree**:
    episodes group by signal, concurrency by price path, and trades from
    distinct episodes still overlap whenever the holding period outruns the gap
    between them.  On the case in #168: 76 episodes, mean concurrency 3.71.
    """

    total_signals: int
    total_trades: int
    fill_rate: float
    win_rate_pct: float
    winning_trades: int
    losing_trades: int
    total_net_gain: float
    expectancy: float
    std_net_gain: float
    profit_factor: float
    best_trade: float
    worst_trade: float
    target_hit_rate_pct: float
    # temporal distribution
    n_months: int
    active_months: int
    zero_months: int
    tpm_mu: float
    tpm_sigma: float
    c_norm: float
    # composite scores
    pf_score: float
    pf_score_tpm: float
    exp_score_tpm: float
    sharpe_raw: float
    # overlap (issue #168) — defaulted so a summary built before these existed
    # still constructs; every summary the engine produces fills them in.
    n_episodes: int = 0
    mean_concurrent_positions: float = float("nan")
    max_concurrent_positions: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionEnvelope:
    """Best/worst-case execution bracket for the selected configuration.

    The same rule is backtested under the two meaningful take-profit
    conventions, bracketing realistic execution without forcing a single
    choice:

    * ``conservative`` — ``target_hit_col="close"``: the target counts only when
      a bar *closes* at or above ``sell_price`` (understates — a real limit sell
      would fill intrabar).  Matches the certified reference engine.
    * ``optimistic`` — ``target_hit_col="high"``: the target fills on the first
      intrabar touch (overstates — assumes the limit sell always fills).

    The true performance of the rule lies between the two.
    """

    conservative: "BacktestSummary"
    optimistic: "BacktestSummary"

    def to_dict(self) -> dict:
        return {
            "conservative_close": self.conservative.to_dict(),
            "optimistic_high": self.optimistic.to_dict(),
        }


@dataclass
class ExcursionStats:
    """MAE / MFE — intra-trade adverse and favourable excursion.

    For every executed trade, over its realised holding window
    ``[fill+1 .. exit]``:

    * **MAE** (Maximum Adverse Excursion) — the deepest drawdown reached,
      ``(min low - buy_price) / buy_price`` (negative);
    * **MFE** (Maximum Favourable Excursion) — the highest run-up reached,
      ``(max high - buy_price) / buy_price`` (positive).

    These describe the rule's "range of action" — how far each trade swings
    against and in favour of the position before it closes — independently of
    which exit convention is chosen.  All values are fractions of the buy price.
    """

    n_trades: int
    mae_mean: float
    mae_median: float
    mae_worst: float
    mfe_mean: float
    mfe_median: float
    mfe_best: float
    mfe_reached_target_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GridResult:
    """A single grid combination and the metrics it produced."""

    params: BacktestParams
    summary: BacktestSummary

    def row(self) -> dict:
        """Flat dict combining the swept parameters and key metrics."""
        p = self.params
        s = self.summary
        return {
            "buy_drop_pct": p.buy_drop_pct,
            "sell_pct": p.sell_pct,
            "target_h": p.target_h,
            "buy_delay_bar": p.buy_delay_bar,
            "profit_factor": s.profit_factor,
            "win_rate_pct": s.win_rate_pct,
            "total_trades": s.total_trades,
            "expectancy": s.expectancy,
            "tpm_mu": s.tpm_mu,
            "fill_rate": s.fill_rate,
            "zero_months": s.zero_months,
            "pf_score_tpm": s.pf_score_tpm,
        }


@dataclass
class StatisticalValidation:
    """Statistical validation of the selected configuration (Step 4)."""

    ttest_winrate_t: float
    ttest_winrate_p: float
    ttest_expectancy_t: float
    ttest_expectancy_p: float
    deflated_sharpe: float
    sharpe_ratio: float
    n_trials_tested: int
    temporal_stability: str  # "PASS" | "WARN" | "FAIL"
    pf_first_half: float
    pf_second_half: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WalkForwardSplit:
    """One train→test split of the walk-forward validation."""

    split_idx: int
    train_from: str
    train_to: str
    test_from: str
    test_to: str
    params: BacktestParams
    train_summary: BacktestSummary
    test_summary: BacktestSummary


@dataclass
class WalkForwardResult:
    """Aggregated out-of-sample walk-forward result.

    Attributes
    ----------
    splits : list[WalkForwardSplit]
        Per-split detail, chronologically ordered.
    oos_summary : BacktestSummary
        Metrics computed on the concatenation of every test-window trade — the
        honest out-of-sample track record.
    n_profitable_splits : int
        Number of test windows with a positive net gain.
    consistency : float
        ``n_profitable_splits / n_splits`` — fraction of OOS windows in profit.
    oos_envelope : ExecutionEnvelope or None
        Execution bracket (close↔high) over the concatenated OOS trades — the
        out-of-sample twin of the in-sample envelope.
    oos_excursion : ExcursionStats or None
        MAE/MFE over the concatenated OOS trades.
    oos_validation : StatisticalValidation or None
        Win-rate / expectancy t-tests and Sharpe on the OOS trades.  The Sharpe
        is **not** deflated (``n_trials=1``): out-of-sample data played no part
        in the parameter selection, so no multiple-testing haircut applies.
    oos_trades : pd.DataFrame or None
        The concatenated per-trade ledger of every test window (close
        convention), excluded from ``repr`` to keep logs readable.
    """

    splits: List[WalkForwardSplit]
    oos_summary: BacktestSummary
    n_profitable_splits: int
    consistency: float
    oos_envelope: Optional["ExecutionEnvelope"] = None
    oos_excursion: Optional["ExcursionStats"] = None
    oos_validation: Optional["StatisticalValidation"] = None
    oos_trades: Optional["pd.DataFrame"] = field(default=None, repr=False)
    # Grid screening of the *last* train window (populated when
    # ``reoptimise=True``).  Walk-forward selection mode reads the published
    # operating point's selection surface from here instead of re-running it.
    last_train_grid: Optional[List["GridResult"]] = field(default=None, repr=False)


@dataclass
class RegimeBreakdown:
    """Per-regime performance and the aggregate dependency score (Step 5)."""

    per_regime: List[dict]
    dependency_score: float
    zero_months: int
    avoid_in: List[str]


@dataclass
class ValidatedRule:
    """The operational rule emitted by Rule Discovery for a ``(PARTIAL-)EDGE``."""

    expression: str
    event_candidate_id: str
    params: BacktestParams

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "event_candidate_id": self.event_candidate_id,
            "direction": self.params.direction,
            "entry_mode": self.params.buy_type,
            "buy_drop_pct": self.params.buy_drop_pct,
            "buy_delay_bar": self.params.buy_delay_bar,
            "sell_pct": self.params.sell_pct,
            "target_h": self.params.target_h,
            "fee": self.params.fee,
        }


@dataclass
class EntryOptimization:
    """Both operating points of the ``entry_mode="auto"`` evaluation, with evidence.

    Stage 1 evaluates the rule at a **market** entry and that verdict is
    authoritative — Stage 2 can never turn a NON-EDGE into an edge, only choose
    which parameters get published.  Stage 2 sweeps ``buy_drop_pct`` on the
    survivor, replays the winner out-of-sample, and adopts it only if it clears
    all three conditions below.

    Both points are carried as full artefacts rather than as three scalars each.
    Three scalars cannot answer "how many trades, at what win rate, with what
    expectancy" about a point you are being asked to trade, and the limit point
    used to be published on strictly less evidence than the market point it
    replaced.

    The adoption criterion, all three evaluated **out-of-sample**
    ----------------------------------------------------------------
    1. ``fill_rate >= min_fill_rate_opt`` — no PF inflated by rare fills.
    2. ``opportunity_sharpe >= market's`` — risk-adjusted return per unit of
       *time*, so a point that trades less often must earn more per trade to
       compensate.  See :func:`forgedge.rule_discovery.opportunity_sharpe` for
       why this is not ``StatisticalValidation.sharpe_ratio``.
    3. ``net_gain >= min_net_gain_retention × market's`` — a backstop against
       the one case the Sharpe cannot see, a tiny mu with a tiny sigma.

    Evaluating out-of-sample is the conservative choice and the only defensible
    one: adopting the limit point *because it is better* means it has to be
    better where selection did not reach.

    Attributes
    ----------
    selected_entry : str
        Entry mechanism of the published point — ``"market"`` or ``"limit"``.
    authoritative : str
        Where the **verdict** came from.  Always ``"market"``.
    adopted : bool
        ``True`` when the limit point cleared all three conditions.
    failed_condition : str or None
        ``"fill"`` / ``"sharpe"`` / ``"net_gain"`` — which condition stopped the
        adoption, or ``None`` when adopted (or when there was no candidate at
        all).  This is the field that separates *"the limit point was better but
        did not survive out-of-sample"* from *"the limit point was not better"*,
        two cases the previous implementation could not tell apart because the
        information that distinguishes them was never computed.
    min_fill_rate_opt, min_net_gain_retention : float
        The thresholds actually applied.
    market_rule, limit_rule : ValidatedRule or None
        The two operating points as publishable rules.
    market_summary, limit_summary : BacktestSummary or None
        Their **out-of-sample** summaries, over identical test windows.
    limit_walk_forward : WalkForwardResult or None
        The limit point's own walk-forward — a *replay* of a fixed
        ``buy_drop_pct`` (``reoptimise=False``), not a re-optimisation, so it
        adds no selection and therefore no ``n_trials``.
    limit_validation : StatisticalValidation or None
        The limit point's in-sample statistics, carrying **its own** trial count
        (Stage 1 cells + Stage 2 cells).  Reported as an absolute metric; the
        ``min_dsr`` gate always reads the market point's.
    market_opportunity_sharpe, limit_opportunity_sharpe : float or None
        Condition 2's two sides.
    market_oos_net_gain, limit_oos_net_gain : float or None
        Condition 3's two sides.
    limit_oos_fill_rate : float or None
        Condition 1's measured value.
    limit_buy_drop_pct : float or None
        The swept parameter's winning value.
    market_profit_factor, market_fill_rate, market_pf_score_tpm : float
        Stage-1 **in-sample** metrics (unchanged meaning).
    limit_profit_factor, limit_fill_rate, limit_pf_score_tpm : float or None
        The best in-sample limit candidate's metrics.  Retained for continuity;
        ``pf_score_tpm`` is no longer the adoption criterion.
    reason : str
        Human-readable explanation of the decision.
    """

    selected_entry: str
    adopted: bool
    min_fill_rate_opt: float
    market_profit_factor: float
    market_fill_rate: float
    market_pf_score_tpm: float
    limit_profit_factor: Optional[float]
    limit_fill_rate: Optional[float]
    limit_pf_score_tpm: Optional[float]
    limit_buy_drop_pct: Optional[float]
    reason: str
    # -- the published-evidence half (#185) --
    authoritative: str = "market"
    failed_condition: Optional[str] = None
    min_net_gain_retention: float = 0.5
    market_rule: Optional["ValidatedRule"] = None
    limit_rule: Optional["ValidatedRule"] = None
    market_summary: Optional[BacktestSummary] = None
    limit_summary: Optional[BacktestSummary] = None
    limit_walk_forward: Optional["WalkForwardResult"] = field(default=None, repr=False)
    limit_validation: Optional["StatisticalValidation"] = None
    market_opportunity_sharpe: Optional[float] = None
    limit_opportunity_sharpe: Optional[float] = None
    market_oos_net_gain: Optional[float] = None
    limit_oos_net_gain: Optional[float] = None
    limit_oos_fill_rate: Optional[float] = None

    def to_dict(self) -> dict:
        """Flat, JSON-friendly view.

        Hand-written rather than ``asdict``: the walk-forward carries a trade
        ledger, and a serialiser that deep-copies a DataFrame into a "summary"
        dict is a memory bug waiting for a large run.  The replay is summarised
        by the numbers a reader needs; the object itself stays reachable on the
        attribute.
        """
        out = {
            "selected_entry": self.selected_entry,
            "authoritative": self.authoritative,
            "adopted": self.adopted,
            "failed_condition": self.failed_condition,
            "min_fill_rate_opt": self.min_fill_rate_opt,
            "min_net_gain_retention": self.min_net_gain_retention,
            "market_profit_factor": self.market_profit_factor,
            "market_fill_rate": self.market_fill_rate,
            "market_pf_score_tpm": self.market_pf_score_tpm,
            "limit_profit_factor": self.limit_profit_factor,
            "limit_fill_rate": self.limit_fill_rate,
            "limit_pf_score_tpm": self.limit_pf_score_tpm,
            "limit_buy_drop_pct": self.limit_buy_drop_pct,
            "market_opportunity_sharpe": self.market_opportunity_sharpe,
            "limit_opportunity_sharpe": self.limit_opportunity_sharpe,
            "market_oos_net_gain": self.market_oos_net_gain,
            "limit_oos_net_gain": self.limit_oos_net_gain,
            "limit_oos_fill_rate": self.limit_oos_fill_rate,
            "reason": self.reason,
        }
        for name, point in (("market", self.market_summary), ("limit", self.limit_summary)):
            out[f"{name}_oos_summary"] = asdict(point) if point is not None else None
        out["market_params"] = (asdict(self.market_rule.params)
                                if self.market_rule is not None else None)
        out["limit_params"] = (asdict(self.limit_rule.params)
                               if self.limit_rule is not None else None)
        out["limit_validation"] = (asdict(self.limit_validation)
                                   if self.limit_validation is not None else None)
        return out


@dataclass
class RuleDiscoveryResponse:
    """Verdict and full evidence produced by Rule Discovery (Section 8).

    This is the artifact handed to Rule Registry.  ``verdict`` is one of
    ``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE`` / ``INSUFFICIENT-DATA``.
    ``INSUFFICIENT-DATA`` (§3.2) is a would-be positive verdict whose pooled
    out-of-sample evidence is too thin to confirm it: the rule is **not
    tradeable** (``is_edge`` is ``False``, it never reaches the Rule
    Registry) but keeps its ``validated_rule`` so it can be re-evaluated when
    more data exists.  ``validated_rule`` is ``None`` only for ``NON-EDGE``.
    """

    date: str
    verdict: str  # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE" | "INSUFFICIENT-DATA"
    alpha_id: str
    asset: str
    timeframe: str
    validated_rule: Optional[ValidatedRule]
    in_sample_summary: BacktestSummary
    walk_forward: Optional[WalkForwardResult]
    statistical_validation: Optional[StatisticalValidation]
    regime_analysis: Optional[RegimeBreakdown]
    execution_envelope: Optional[ExecutionEnvelope] = None
    excursion: Optional[ExcursionStats] = None
    entry_optimization: Optional[EntryOptimization] = None
    grid_results: List[GridResult] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def is_edge(self) -> bool:
        """Tradeable verdict — ``INSUFFICIENT-DATA`` is deliberately *not*
        tradeable (unconfirmed ≠ confirmed)."""
        return self.verdict in ("EDGE", "PARTIAL-EDGE")

    def to_dict(self) -> dict:
        """Nested dictionary mirroring the YAML layout of the spec (Section 8)."""
        wf = self.walk_forward
        sv = self.statistical_validation
        ra = self.regime_analysis
        return {
            "date": self.date,
            "verdict": self.verdict,
            "alpha_id": self.alpha_id,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "validated_rule": self.validated_rule.to_dict() if self.validated_rule else None,
            "in_sample_results": self.in_sample_summary.to_dict(),
            "walk_forward": (
                {
                    "n_splits": len(wf.splits),
                    "consistency": wf.consistency,
                    "n_profitable_splits": wf.n_profitable_splits,
                    "oos_results": wf.oos_summary.to_dict(),
                    "oos_execution_envelope": (
                        wf.oos_envelope.to_dict() if wf.oos_envelope else None
                    ),
                    "oos_excursion": (
                        wf.oos_excursion.to_dict() if wf.oos_excursion else None
                    ),
                    "oos_statistical_validation": (
                        wf.oos_validation.to_dict() if wf.oos_validation else None
                    ),
                    "splits": [
                        {
                            "split_idx": s.split_idx,
                            "train": [s.train_from, s.train_to],
                            "test": [s.test_from, s.test_to],
                            "test_profit_factor": s.test_summary.profit_factor,
                            "test_win_rate": s.test_summary.win_rate_pct,
                            "test_trades": s.test_summary.total_trades,
                            "test_net_gain": s.test_summary.total_net_gain,
                        }
                        for s in wf.splits
                    ],
                }
                if wf is not None
                else None
            ),
            "statistical_validation": sv.to_dict() if sv else None,
            "execution_envelope": (
                self.execution_envelope.to_dict() if self.execution_envelope else None
            ),
            "excursion": self.excursion.to_dict() if self.excursion else None,
            "entry_optimization": (
                self.entry_optimization.to_dict() if self.entry_optimization else None
            ),
            "regime_analysis": (
                {
                    "dependency_score": ra.dependency_score,
                    "zero_months": ra.zero_months,
                    "avoid_in": ra.avoid_in,
                    "per_regime": ra.per_regime,
                }
                if ra is not None
                else None
            ),
            "rejection_reasons": self.rejection_reasons or None,
            "notes": self.notes,
        }

    def persist(self, path) -> None:
        """Save this Rule Discovery contract to disk as a pickle file.

        Reload with::

            import pickle
            response = pickle.load(open(path, "rb"))
            response.verdict
        """
        import pickle
        import pathlib
        pathlib.Path(path).write_bytes(pickle.dumps(self))
