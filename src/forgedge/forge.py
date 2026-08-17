"""``forge`` — the end-to-end FORGE pipeline orchestrator.

A single entry point that chains the five FORGE modules — Market Context
(Modulo 0), Event Discovery (Modulo 1), Alpha Discovery (Modulo 2), Rule
Discovery (Modulo 3) and Rule Registry (Modulo 4) — into one call that turns a
raw KPI Table into validated, catalogued trading rules.

Usage
-----
    from forgedge import forge

    kpi = pd.read_parquet("btc_1h.parquet")
    result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

    print(result.summary())                 # one row per evaluated candidate
    for contract, response in result.edges():
        print(contract.alpha_id, response.verdict)
    print(result.registry.summary())        # Modulo 4 — catalogued rules

Every stage accepts the configuration object of its module, so the whole
pipeline is tunable through one call::

    from forgedge import (
        forge, MarketContextConfig, EMAProxyConfig,
        DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig, RegistryConfig,
    )
    from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams
    from forgedge.alpha_discovery.models import PromotionThresholds

    result = forge(
        kpi,
        ticker="BTCUSDC",
        timeframe="1H",
        market_context_config=MarketContextConfig(
            ema_proxy=EMAProxyConfig(auto_window=True, window_unit="day", bar_hours=1.0),
        ),
        event_discovery_config=DiscoveryConfig(
            train_ratio=0.80,
            walk_forward=EventWalkForwardConfig(n_splits=4, min_pass_rate=0.75),
            gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
        ),
        alpha_config=AlphaConfig(
            train_ratio=0.70,
            thresholds=PromotionThresholds(min_lift=0.08, min_cohens_d=0.15),
        ),
        rule_discovery_config=RuleDiscoveryConfig(),
        registry_config=RegistryConfig(),
    )

The Rule Registry's cross-ticker backtest only has other tickers to replay
against in a multi-ticker session.  Use :func:`forge_multi` to run the pipeline
on several tickers and pool their rules into one cross-ticker registry.
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

from .alpha_discovery.discovery import AlphaDiscovery
from .alpha_discovery.models import AlphaConfig, AlphaContract
from .calibration import FastRotationNull, RotationCalibrator
from .calibration.models import CalibrationReport, RotationConfig
from .ledger import HypothesisLedger
from .event_discovery.discovery import DiscoveryConfig, EventDiscovery
from .event_discovery.models import (
    ActivationStats,
    CustomEvent,
    EventCandidate,
    GateParams,
)
from .market_context.context import MarketContext
from .market_context.models import REGIME_COL, MarketContextConfig
from .presets import default_horizon_grid
from .config_report import ConfigReport, config_report
from .resolver import PipelineContext, ResolutionTrace, collect_context
from .unset import UNSET, coalesce, is_set
from .rule_discovery.discovery import RuleDiscovery
from .rule_discovery.models import RuleDiscoveryConfig, RuleDiscoveryResponse
from .rule_registry.models import RegistryConfig, RuleSubmission
from .rule_registry.registry import RuleRegistry
from .timebudget import TimeBudget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

class _Reporter:
    """Routes pipeline progress to ``logging`` and, on demand, to a live bar.

    Every stage milestone is always emitted at ``INFO`` on the package logger
    (silent unless the host application configures logging).  When ``enabled``
    is set, the same milestones are also printed to ``stderr`` with a running
    elapsed-time stamp, and long loops are wrapped in a progress bar — ``tqdm``
    when it is installed, otherwise a dependency-free textual fallback.
    """

    def __init__(self, enabled: bool, label: str = ""):
        self.enabled = enabled
        self.label = label
        self._t0 = time.perf_counter()

    @property
    def _prefix(self) -> str:
        return f"forge:{self.label}" if self.label else "forge"

    def stage(self, msg: str) -> None:
        """Announce a pipeline milestone."""
        logger.info("[%s] %s", self._prefix, msg)
        if self.enabled:
            elapsed = time.perf_counter() - self._t0
            print(f"[{self._prefix} +{elapsed:6.1f}s] {msg}", file=sys.stderr, flush=True)

    def track(
        self, iterable: Iterable, desc: str, total: Optional[int] = None
    ) -> Iterable:
        """Wrap ``iterable`` in a progress bar when reporting is enabled."""
        if not self.enabled or not total:
            return iterable
        tqdm = _get_tqdm()
        if tqdm is not None:
            return tqdm(iterable, desc=f"[{self._prefix}] {desc}", total=total, leave=False)
        return _TextBar(iterable, desc=f"[{self._prefix}] {desc}", total=total)


def _get_tqdm():
    """Return ``tqdm.auto.tqdm`` if installed, else ``None`` (optional dep)."""
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    except Exception:  # pragma: no cover - tqdm is an optional convenience
        return None


class _TextBar:
    """Minimal dependency-free progress bar printed in place on ``stderr``."""

    def __init__(self, iterable: Iterable, desc: str, total: int, width: int = 28):
        self._iterable = iterable
        self._desc = desc
        self._total = total
        self._width = width
        self._t0 = time.perf_counter()

    def __iter__(self) -> Iterator:
        for i, item in enumerate(self._iterable, start=1):
            yield item
            self._draw(i)
        # Newline so the finished bar is not overwritten by later output.
        print("", file=sys.stderr, flush=True)

    def _draw(self, done: int) -> None:
        frac = done / self._total
        filled = int(self._width * frac)
        bar = "█" * filled + "·" * (self._width - filled)
        elapsed = time.perf_counter() - self._t0
        print(
            f"\r{self._desc} |{bar}| {done}/{self._total} ({frac:4.0%}) {elapsed:5.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )


@dataclass
class ForgeResult:
    """The complete output of a :func:`forge` run.

    Holds every artefact produced along the pipeline so callers can both read
    the final verdicts *and* drill back into any intermediate stage for audit.

    Attributes
    ----------
    enriched : pd.DataFrame
        KPI Table after Market Context — the original columns plus ``regime``
        and ``regime_stable``.  When Market Context is skipped this is the
        table actually fed to Event Discovery.
    candidates : list[EventCandidate]
        Every Event Candidate that passed the Consistency Gate (Modulo 1).
    contracts : list[AlphaContract]
        Every evaluated Alpha Contract (promoted *and* rejected), so rejections
        can be audited via ``contract.rejection_reasons``.
    promoted : list[AlphaContract]
        The subset of ``contracts`` that cleared every promotion gate
        (``status == "HYPOTHESIS"``) and were handed to Rule Discovery.
    rule_responses : list[tuple[AlphaContract, RuleDiscoveryResponse]]
        One ``(contract, response)`` pair per promoted contract, in promotion
        order.  The verdict lives on ``response.verdict``.
    ticker : str
        Label used for this run's pool in the Rule Registry (and for the
        traceability metadata of the Alpha Contracts).
    event_frame : pd.DataFrame
        Event Discovery's post-pipeline frame (``ed.df``) — the native price
        columns, every derived feature and the ``regime`` column.  This is the
        frame Alpha/Rule Discovery and the Rule Registry actually read.
    registry : RuleRegistry or None
        Modulo 4 — the in-memory Rule Registry built from this run's tradeable
        rules (``None`` when Rule Discovery or the registry stage was skipped).
        Single-ticker, so its cross-ticker backtest is trivial; for a real
        cross-ticker catalogue use :func:`forge_multi`.
    market_context, event_discovery, alpha_discovery : module instances
        The live module objects, for introspection — e.g.
        ``market_context.distribution()``, ``event_discovery.summary()``,
        ``alpha_discovery.summary()``.  ``market_context`` is ``None`` when the
        stage was skipped.
    calibration : CalibrationReport or None
        The search-level rotation-null report — from :class:`FastRotationNull`
        (default) or the full :class:`RotationCalibrator` when
        ``rotation_calibration`` was passed.  ``None`` when both were skipped.
    ledger : HypothesisLedger or None
        The session's hypothesis ledger — how many candidates, horizons and
        grid cells the run consumed (see :mod:`forgedge.ledger`).
    time_budget : TimeBudget or None
        The effective temporal axis of Alpha Discovery — split, purge and
        embargo (see :mod:`forgedge.timebudget`).  Built from the alpha config
        unless an explicit budget was passed to :func:`forge`.
    context : PipelineContext or None
        The session's resolved facts — timeframe, schema, economics,
        statistical policy (see :mod:`forgedge.resolver`).
    resolution : ResolutionTrace or None
        Ordered record of every field the resolver derived, with the rule that
        produced it and the inputs it read.  Log it alongside
        ``ledger.describe()``: together they say what was searched and with
        what configuration.
    coherence : ConfigReport or None
        The configuration report — the resolved configs plus every constraint
        violation found in *check* mode.  Produced by the **same** resolver call
        the pipeline ran with, so ``coherence.configs`` is literally what
        executed, not a reconstruction.
    """

    enriched: pd.DataFrame
    candidates: List[EventCandidate]
    contracts: List[AlphaContract]
    promoted: List[AlphaContract]
    rule_responses: List[Tuple[AlphaContract, RuleDiscoveryResponse]]
    ticker: str = "ASSET"
    event_frame: Optional[pd.DataFrame] = None
    registry: Optional[RuleRegistry] = None
    market_context: Optional[MarketContext] = None
    event_discovery: Optional[EventDiscovery] = None
    alpha_discovery: Optional[AlphaDiscovery] = None
    calibration: Optional[CalibrationReport] = None
    ledger: Optional[HypothesisLedger] = None
    time_budget: Optional[TimeBudget] = None
    context: Optional[PipelineContext] = None
    resolution: Optional[ResolutionTrace] = None
    coherence: Optional[ConfigReport] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def edges(self) -> List[Tuple[AlphaContract, RuleDiscoveryResponse]]:
        """Return ``(contract, response)`` pairs whose verdict is tradeable.

        Tradeable means ``EDGE`` or ``PARTIAL-EDGE`` (``response.is_edge``).
        """
        return [(c, r) for c, r in self.rule_responses if r.is_edge]

    def validated_rules(self) -> List[RuleDiscoveryResponse]:
        """Return the responses that carry a non-null ``validated_rule``."""
        return [r for _, r in self.rule_responses if r.validated_rule is not None]

    def submissions(self) -> List[RuleSubmission]:
        """Build the Rule Registry submissions for this run's tradeable rules.

        One :class:`RuleSubmission` per ``EDGE`` / ``PARTIAL-EDGE`` response that
        carries a validated rule, tagged with this run's ``ticker`` and the Alpha
        grade.  This is what feeds the Rule Registry (Modulo 4), both for the
        single-ticker ``result.registry`` and for :func:`forge_multi`.
        """
        by_id = {c.event_id: c for c in self.candidates}
        subs: List[RuleSubmission] = []
        for contract, response in self.rule_responses:
            if not response.is_edge or response.validated_rule is None:
                continue
            cand = by_id.get(response.validated_rule.event_candidate_id)
            if cand is None:
                continue
            grade = None
            if getattr(contract, "alpha_score", None) is not None:
                grade = contract.alpha_score.grade
            subs.append(
                RuleSubmission(
                    ticker=self.ticker, response=response, candidate=cand, grade=grade
                )
            )
        return subs

    def summary(self) -> pd.DataFrame:
        """Return Alpha Discovery's flat candidate summary, with the rule verdict.

        One row per evaluated candidate (sorted by composite score), augmented
        with a ``rule_verdict`` column for the promoted contracts that reached
        Rule Discovery.  Returns an empty frame when no candidates were
        evaluated.
        """
        if self.alpha_discovery is None:
            return pd.DataFrame()
        df = self.alpha_discovery.summary()
        if df.empty:
            return df
        verdicts = {c.alpha_id: r.verdict for c, r in self.rule_responses}
        df = df.copy()
        df["rule_verdict"] = df["alpha_id"].map(verdicts)
        return df


def forge(
    kpi_table: pd.DataFrame,
    *,
    ticker: Optional[str] = None,
    asset: str = "ASSET",
    timeframe: str = UNSET,
    market_context_config: Optional[MarketContextConfig] = None,
    event_discovery_config: Optional[DiscoveryConfig] = None,
    alpha_config: Optional[AlphaConfig] = None,
    rotation_calibration: Optional[RotationConfig] = None,
    fast_null: bool = True,
    time_budget: Optional[TimeBudget] = None,
    rule_discovery_config: Optional[RuleDiscoveryConfig] = None,
    registry_config: Optional[RegistryConfig] = None,
    manual_events: Optional[List[CustomEvent]] = None,
    run_market_context: bool = True,
    run_rule_discovery: bool = True,
    run_registry: bool = True,
    only_validated_events: bool = False,
    strict: bool = True,
    rule_discovery_grades: Optional[Iterable[str]] = None,
    progress: bool = True,
) -> ForgeResult:
    """Run the full FORGE rule-extraction pipeline end to end.

    Chains the five modules and returns every artefact in a :class:`ForgeResult`:

    0. **Market Context** classifies each bar by regime and appends ``regime`` /
       ``regime_stable`` (skipped when ``run_market_context=False`` or the
       table already carries a ``regime`` column).
    1. **Event Discovery** mines structurally-consistent boolean events.
    2. **Alpha Discovery** derives a target per event, measures its edge and
       promotes the candidates that confirm out-of-sample.
    3. **Rule Discovery** backtests each promoted contract under realistic order
       mechanics and emits an ``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE`` verdict.
    4. **Rule Registry** collects the tradeable rules into an in-memory catalogue,
       computes the correlation matrices and flags duplicates.

    Parameters
    ----------
    kpi_table : pd.DataFrame
        Raw KPI Table — a ``close`` column plus any technical features, and a
        datetime source (``open_dt`` column or a DatetimeIndex).
    ticker : str, optional
        Ticker label for the Rule Registry pool and the Alpha Contract metadata
        (e.g. ``"BTCUSDC"``).  When omitted it falls back to ``alpha_config.asset``
        (if given) or ``asset``.
    asset : str
        Traceability metadata for the Alpha Contracts.  Used only when neither
        ``ticker`` nor an explicit ``alpha_config`` sets it.
    timeframe : str
        Bar size, e.g. ``"1D"``, ``"4H"``, ``"15m"``.  The session's single
        source of bar duration: it picks M2's horizon grid and, since #179,
        every field that counts bars — ``BacktestParams.buy_delay_bar``,
        ``MarketContextConfig.stable_window`` and the rest (F5).  Defaults to
        ``"1H"``.

        Leaving it out is not the same as passing ``"1H"``: an omitted
        timeframe is an inherited default, so ``config_report()`` will not
        report it as disagreeing with the table's own timestamp spacing.
        Declare it and a real disagreement is flagged.
    market_context_config : MarketContextConfig, optional
        Modulo 0 configuration.  Defaults to the production EMA-proxy settings.
    event_discovery_config : DiscoveryConfig, optional
        Modulo 1 configuration.  Defaults to a no-split, single-component run;
        for production pass ``train_ratio < 1`` with a ``walk_forward`` block.
    alpha_config : AlphaConfig, optional
        Modulo 2 configuration.  When omitted, a default is built carrying the
        resolved ticker as ``asset`` and ``timeframe``; on a daily-or-slower
        ``timeframe`` the default ``horizon_grid`` (calibrated on ~hourly bars)
        is replaced by the timeframe-scaled grid from
        :func:`forgedge.presets.default_horizon_grid`.  When an explicit config
        keeps the hourly default grid on such a timeframe a ``UserWarning`` is
        emitted (the grid would scan holding periods of 48+ days).
    rotation_calibration : RotationConfig, optional
        When set, run the search-level rotation null calibrator inline after
        Alpha Discovery.  Rotates only the ``close`` column K times and re-runs
        AlphaDiscovery on each draw, producing a Tippett (min-p) p-value that
        controls for the multiple-testing surface of the full search.  Results
        are stored as ``AlphaContract.rotation_p`` / ``rotation_threshold`` on
        each promoted contract.  Default ``None`` — skip calibration, zero extra
        cost, backward-compatible.

        For large K (≥ 100 draws) consider the standalone / disaccoppiato mode
        instead: run ``forge()`` without this parameter, then call
        ``RotationCalibrator(result.event_frame, result.candidates, alpha_config)``
        separately — same results, no impact on the main pipeline runtime.
        When set, it supersedes the default ``fast_null`` pass.
    fast_null : bool, default True
        Run the :class:`FastRotationNull` after Alpha Discovery — the exact
        search-level rotation null over every circular offset, computed via FFT
        cross-correlation in roughly the cost of one Alpha Discovery pass (no
        K, no seed).  Annotates each promoted contract with ``rotation_p`` /
        ``rotation_threshold`` and stores the report on
        ``ForgeResult.calibration``.  Rule Discovery then requires
        ``rotation_p <= criteria.max_rotation_p`` for a full ``EDGE`` verdict
        (rules that only won the multiple-testing lottery are capped at
        ``PARTIAL-EDGE``).  Set ``False`` to skip (pre-#116 behaviour);
        ignored when ``rotation_calibration`` is passed (the full calibrator
        supersedes it).
    time_budget : TimeBudget, optional
        A single temporal axis for the whole session (see
        :mod:`forgedge.timebudget`): its ``split`` becomes the IS boundary of
        Event Discovery *and* Alpha Discovery, and its purge / embargo widths
        quarantine the boundary crossings.  ``None`` (default) keeps each
        module's config-driven split; Alpha Discovery still builds its own
        purged budget internally (purge = ``max(horizon_grid)`` — the removal
        of a mechanical look-ahead, on by default), and the Rule Discovery
        walk-forward purges its train windows from its own resolved grid.
        The effective Alpha Discovery budget is exposed on
        ``ForgeResult.time_budget``.
    rule_discovery_config : RuleDiscoveryConfig, optional
        Modulo 3 configuration.  Defaults to the standard grid and acceptance
        gates.
    registry_config : RegistryConfig, optional
        Modulo 4 configuration.  Defaults to the calibrated dedup / cross-ticker
        thresholds.
    manual_events : list[CustomEvent], optional
        Skip automatic Event Discovery (Modulo 1) and inject these user-defined
        formula events straight into Alpha/Rule Discovery.  Mutually exclusive
        with ``event_discovery_config`` — passing both raises ``ValueError``.
        Each event still crosses the Consistency Gate, but a failure only emits
        a ``logger.warning`` and does not drop the event.  AND composition is
        not performed in this mode.
    run_market_context : bool, default True
        Run Modulo 0.  Set ``False`` to feed a table that already carries the
        ``regime`` columns (or to skip regime analysis entirely).
    run_rule_discovery : bool, default True
        Run Modulo 3.  Set ``False`` to stop after Alpha Discovery and obtain
        the promoted contracts (hypotheses) without backtesting them; the
        returned ``rule_responses`` is then empty and Modulo 4 is skipped too.
    run_registry : bool, default True
        Run Modulo 4 on this run's tradeable rules.  No effect when Rule
        Discovery was skipped.  Single-ticker, so the cross-ticker backtest is
        trivial — use :func:`forge_multi` for a real cross-ticker catalogue.
    only_validated_events : bool, default False
        When Event Discovery ran with walk-forward validation, hand Alpha
        Discovery only the candidates with ``validation.passed == True``.  Has
        no effect when walk-forward was not configured.
    rule_discovery_grades : iterable of str, optional
        Restrict the (expensive) Rule Discovery backtest to the promoted Alpha
        Contracts whose letter grade (``A`` / ``B`` / ``C`` / ``D``) is in this
        set — e.g. ``("A", "B")`` skips the weaker grade-C/D alphas that rarely
        survive validation, cutting pipeline time.  Comparison is
        case-insensitive.  When omitted every promoted contract is backtested
        (the previous behaviour).  Contracts filtered out here still appear in
        ``contracts`` / ``promoted`` for audit; they simply get no rule
        response and never reach the Rule Registry.
    strict : bool, default True
        Stop on a configuration that makes a stage **structurally incapable** of
        producing a verdict — a `FAIL` finding from
        :func:`forgedge.config_report`, such as issue #173's walk-forward
        bucket.  Such a run cannot tell you anything: every candidate is
        eliminated for configuration reasons, and the wall of rejections is
        indistinguishable from "the signal is bad".  Failing fast is the point.
        ``strict=False`` degrades every finding to a ``UserWarning`` and runs
        anyway.  Non-critical incoherences are always warnings, never errors.
    progress : bool, default True
        Print per-stage status and a Rule Discovery progress bar to ``stderr``.
        Independently of this flag every milestone is emitted at ``INFO`` on the
        ``forgedge.forge`` logger, so configuring logging surfaces the same
        information without the ``stderr`` output.  Set ``False`` to silence all
        ``stderr`` output (e.g. in batch / CI contexts).
        The progress bar uses ``tqdm`` when installed, else a built-in textual
        fallback.

    Returns
    -------
    ForgeResult
        All pipeline artefacts (see the class docstring).
    """
    # ── Mode validation: manual injection XOR automatic discovery ─────────
    if manual_events is not None and event_discovery_config is not None:
        raise ValueError(
            "manual_events and event_discovery_config are mutually exclusive. "
            "Pass one or the other, not both."
        )

    # Whether the caller *chose* a bar size or inherited one.  Only the
    # declared-versus-measured check cares (F5, #179): a default nobody has
    # looked at is not a contradiction with the data, so it stays silent —
    # the same rule the other checks follow.
    timeframe_declared = is_set(timeframe)
    timeframe = coalesce(timeframe, default="1H")

    # ── Modulo 2 config drives the resolved ticker / metadata ─────────────
    cfg = alpha_config
    if cfg is None:
        resolved_ticker = ticker or asset
        # The AlphaConfig default horizon grid is calibrated on ~hourly bars;
        # on a daily-or-slower timeframe substitute the class-calibrated grid
        # so holding periods stay sane (48 bars would mean 48+ days).
        scaled_grid = default_horizon_grid(timeframe)
        if scaled_grid is not None:
            cfg = AlphaConfig(
                asset=resolved_ticker, timeframe=timeframe, horizon_grid=scaled_grid
            )
        else:
            cfg = AlphaConfig(asset=resolved_ticker, timeframe=timeframe)
    else:
        resolved_ticker = ticker or cfg.asset
        _warn_if_hourly_grid_on_slow_timeframe(cfg)

    report = _Reporter(progress, label=resolved_ticker)
    report.stage(f"start — {len(kpi_table)} bars")

    # ── Parameter resolution ──────────────────────────────────────────────
    # One context for the session, seeded from whatever the caller set
    # explicitly (see forgedge.resolver): every config field left at UNSET is
    # derived from it, and every field that was set is left alone and merely
    # checked.  Resolution never reads the data, only the timeframe, the schema
    # and the configs themselves.
    #
    # The configs of the stages that will actually run are materialised
    # *first*.  A ``None`` resolves to ``None``, so leaving them out meant the
    # module later built its own default and resolved it against a default
    # ``PipelineContext`` — i.e. against 1H, whatever the session declared.
    # That is invisible for a derive reading another config value, and wrong
    # for every derive reading the timeframe: on a 1D run `buy_delay_bar` came
    # back 6, the hourly value, for 100% of published rules (F5, #179).
    #
    # Gated on the stage flags, so a module that will not run keeps
    # contributing nothing — neither a derivation nor a check.  A config the
    # session is not going to execute has no business raising findings.
    _bundle = {
        "market_context": market_context_config,
        "event_discovery": event_discovery_config,
        "alpha": cfg,
        "rule_discovery": rule_discovery_config,
        "registry": registry_config,
    }
    if run_market_context:
        _bundle["market_context"] = market_context_config or MarketContextConfig()
    if manual_events is None:
        _bundle["event_discovery"] = event_discovery_config or DiscoveryConfig()
    if run_rule_discovery:
        _bundle["rule_discovery"] = rule_discovery_config or RuleDiscoveryConfig()
    if run_registry:
        _bundle["registry"] = registry_config or RegistryConfig()
    context = collect_context(
        _bundle,
        PipelineContext.from_frame(kpi_table, timeframe=timeframe,
                                   timeframe_declared=timeframe_declared),
    )
    coherence = config_report(
        _bundle["event_discovery"], _bundle["alpha"], _bundle["rule_discovery"],
        _bundle["registry"], _bundle["market_context"], ctx=context,
    )
    _bundle = coherence.configs
    market_context_config = _bundle["market_context"]
    event_discovery_config = _bundle["event_discovery"]
    cfg = _bundle["alpha"]
    rule_discovery_config = _bundle["rule_discovery"]
    registry_config = _bundle["registry"]
    resolution = coherence.trace
    report.stage(resolution.describe())
    if coherence.findings:
        logger.info("[%s] %s", resolved_ticker, coherence.one_line())
    if coherence.has_critical and strict:
        raise ValueError(
            "forge(): the configuration cannot produce a verdict — every "
            "candidate would be eliminated for configuration reasons, not for "
            "signal quality. Fix the values below, or pass strict=False to run "
            "anyway.\n" + coherence.one_line()
        )
    for _finding in coherence.findings:
        warnings.warn(f"{_finding.code}: {_finding.message}", UserWarning, stacklevel=2)

    # ── Modulo 0 — Market Context ─────────────────────────────────────────
    mc: Optional[MarketContext] = None
    already_enriched = REGIME_COL in kpi_table.columns
    if run_market_context and not already_enriched:
        report.stage("M0 Market Context — classifying regimes…")
        mc = MarketContext(kpi_table, config=market_context_config)
        enriched = mc.run()
    else:
        report.stage("M0 Market Context — skipped (regime already present)")
        enriched = kpi_table

    # ── Modulo 1 — Event Discovery (or manual injection) ──────────────────
    ed: Optional[EventDiscovery] = None
    if manual_events is not None:
        report.stage(f"M1 Event Discovery — injecting {len(manual_events)} manual event(s)…")
        candidates = _build_candidates_from_manual_events(manual_events, enriched)
        alpha_frame = enriched
    else:
        report.stage("M1 Event Discovery — mining event candidates…")
        ed = EventDiscovery(
            enriched, config=event_discovery_config, time_budget=time_budget
        )
        candidates = ed.run()
        alpha_frame = ed.df
    report.stage(f"M1 Event Discovery — {len(candidates)} candidate(s)")

    alpha_candidates = candidates
    if only_validated_events:
        validated = [
            c for c in candidates if c.validation is not None and c.validation.passed
        ]
        # Only narrow the set when validation actually *concluded*; otherwise
        # keep all candidates so the pipeline does not silently drop everything.
        #
        # `passed` is now tri-state: `None` means every fold was too short to
        # say anything at the candidate's own rate, which is a property of the
        # configuration rather than of the candidate (#177). `None` is falsy, so
        # the original `c.validation is not None` test would have found
        # validation "ran", narrowed to `validated`, and discarded the entire
        # candidate set — precisely what the comment above exists to prevent.
        if any(c.validation is not None and c.validation.passed is not None
               for c in candidates):
            alpha_candidates = validated
            report.stage(
                f"M1 Event Discovery — {len(validated)} walk-forward-validated candidate(s) kept"
            )
        elif any(c.validation is not None for c in candidates):
            report.stage(
                "M1 Event Discovery — walk-forward inconclusive on every candidate "
                "(folds too short at their own rate); keeping all candidates"
            )

    # ── Modulo 2 — Alpha Discovery ────────────────────────────────────────
    report.stage(f"M2 Alpha Discovery — evaluating {len(alpha_candidates)} candidate(s)…")
    ad = AlphaDiscovery(alpha_frame, alpha_candidates, cfg, time_budget=time_budget)
    contracts = ad.run()
    promoted = ad.promoted_contracts()
    report.stage(f"M2 Alpha Discovery — {len(promoted)}/{len(contracts)} promoted")
    effective_budget = ad._budget
    if effective_budget is not None:
        report.stage(effective_budget.describe())

    # ── Hypothesis ledger — the session's multiple-testing surface ────────
    ledger = HypothesisLedger(
        m1_candidates=len(alpha_candidates),
        m2_horizons=len(cfg.horizon_grid),
        m2_promoted=len(promoted),
        m2_return_tests=getattr(ad, "n_return_tests", 0),
    )
    report.stage(ledger.describe())

    # ── Search-level rotation null ─────────────────────────────────────────
    # The full calibrator (explicit K draws) supersedes the fast exact null.
    cal_report: Optional[CalibrationReport] = None
    if rotation_calibration is not None and promoted:
        report.stage(
            f"Rotation Calibrator — K={rotation_calibration.k} draws "
            f"(alpha={rotation_calibration.alpha})…"
        )
        cal = RotationCalibrator(
            alpha_frame, alpha_candidates, cfg, time_budget=time_budget
        )
        cal_report = cal.run(promoted, rotation_calibration)
        report.stage(
            f"Rotation Calibrator — Tippett p={cal_report.tippett_p:.4f}, "
            f"{len(cal_report.survivors)}/{len(promoted)} above null bar"
        )
    elif fast_null and promoted:
        report.stage("Fast rotation null — exact search-level null (all offsets)…")
        cal_report = FastRotationNull(
            alpha_frame, alpha_candidates, cfg, time_budget=time_budget
        ).run(promoted)
        report.stage(
            f"Fast rotation null — search p={cal_report.tippett_p:.4f}, "
            f"{len(cal_report.survivors)}/{len(promoted)} above null bar"
        )

    # ── Modulo 3 — Rule Discovery ─────────────────────────────────────────
    by_id = {c.event_id: c for c in candidates}
    rule_responses: List[Tuple[AlphaContract, RuleDiscoveryResponse]] = []
    to_backtest = _filter_by_grade(promoted, rule_discovery_grades)
    if run_rule_discovery and rule_discovery_grades is not None:
        skipped = len(promoted) - len(to_backtest)
        report.stage(
            f"M3 Rule Discovery — grade filter {_grades_label(rule_discovery_grades)} "
            f"kept {len(to_backtest)}/{len(promoted)} contract(s) ({skipped} skipped)"
        )
    if run_rule_discovery and to_backtest:
        report.stage(f"M3 Rule Discovery — backtesting {len(to_backtest)} contract(s)…")
    elif not run_rule_discovery:
        report.stage("M3 Rule Discovery — skipped")
    for contract in report.track(
        to_backtest if run_rule_discovery else [],
        desc="M3 Rule Discovery",
        total=len(to_backtest) if run_rule_discovery else 0,
    ):
        cand = by_id.get(contract.event_candidate_id)
        if cand is None:
            continue
        rd = RuleDiscovery(alpha_frame, contract, cand, config=rule_discovery_config)
        response = rd.run()
        rule_responses.append((contract, response))
        if not ledger.m3_grid_cells and response.grid_results:
            ledger.m3_grid_cells = len(response.grid_results)

    result = ForgeResult(
        enriched=enriched,
        candidates=candidates,
        contracts=contracts,
        promoted=promoted,
        rule_responses=rule_responses,
        ticker=resolved_ticker,
        event_frame=alpha_frame,
        market_context=mc,
        event_discovery=ed,
        alpha_discovery=ad,
        calibration=cal_report,
        ledger=ledger,
        time_budget=effective_budget,
        context=context,
        resolution=resolution,
        coherence=coherence,
    )
    if run_rule_discovery:
        report.stage(f"M3 Rule Discovery — {len(result.edges())} tradeable rule(s)")

    # ── Modulo 4 — Rule Registry ──────────────────────────────────────────
    if run_rule_discovery and run_registry:
        report.stage("M4 Rule Registry — cataloguing rules…")
        result.registry = RuleRegistry(
            result.submissions(),
            {resolved_ticker: alpha_frame},
            config=registry_config,
        ).run()
        report.stage(f"M4 Rule Registry — {len(result.registry.documents)} document(s)")

    report.stage("done")
    return result


def _warn_if_hourly_grid_on_slow_timeframe(cfg: AlphaConfig) -> None:
    """Warn when an explicit config keeps the hourly-calibrated default grid
    on a daily-or-slower timeframe.

    Only the untouched class default triggers the warning: a user who set any
    custom ``horizon_grid`` made a deliberate choice and is left alone.
    """
    default_grid = AlphaConfig.__dataclass_fields__["horizon_grid"].default
    if cfg.horizon_grid != default_grid:
        return
    scaled = default_horizon_grid(cfg.timeframe)
    if scaled is None:
        return
    warnings.warn(
        f"AlphaConfig.horizon_grid is the hourly-calibrated default "
        f"{default_grid} but timeframe={cfg.timeframe!r} means each bar is a "
        f"day or longer, so the grid scans holding periods of up to "
        f"{max(default_grid)} bars ({max(default_grid)} {cfg.timeframe} "
        f"periods). Pass an explicit horizon_grid (e.g. {scaled}) or use "
        f"forgedge.presets.forge_preset to scale it automatically.",
        UserWarning,
        stacklevel=3,
    )


def _contract_grade(contract: AlphaContract) -> Optional[str]:
    """Return the upper-case letter grade of a contract, or ``None`` if ungraded."""
    score = getattr(contract, "alpha_score", None)
    if score is None or score.grade is None:
        return None
    return str(score.grade).strip().upper()


def _grades_label(grades: Iterable[str]) -> str:
    """Render a grade set as a stable, readable label for status messages."""
    return "{" + ", ".join(sorted({g.strip().upper() for g in grades})) + "}"


def _filter_by_grade(
    contracts: List[AlphaContract], grades: Optional[Iterable[str]]
) -> List[AlphaContract]:
    """Keep only the contracts whose grade is in ``grades`` (case-insensitive).

    ``grades=None`` is the pass-through default (every contract is kept).  An
    ungraded contract is dropped whenever a filter is active, since it cannot be
    shown to meet the requested bar.
    """
    if grades is None:
        return contracts
    allowed = {g.strip().upper() for g in grades}
    return [c for c in contracts if _contract_grade(c) in allowed]


def _timestamps_from_frame(
    frame: pd.DataFrame, timestamp_col: str = "open_dt"
) -> pd.Series:
    """Extract a datetime Series (RangeIndex) from a frame for month indexing.

    Accepts a ``DatetimeIndex`` or a datetime/parseable ``timestamp_col``.
    """
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, dtype="datetime64[ns]")
    if timestamp_col in frame.columns:
        return pd.to_datetime(frame[timestamp_col]).reset_index(drop=True)
    raise ValueError(
        "Cannot evaluate the Consistency Gate for manual_events: the frame has "
        f"no DatetimeIndex and no '{timestamp_col}' column."
    )


def _build_candidates_from_manual_events(
    manual_events: List[CustomEvent],
    frame: pd.DataFrame,
) -> List[EventCandidate]:
    """Turn user ``CustomEvent`` formulas into gate-annotated EventCandidates.

    Each event is evaluated on ``frame``, crosses the Consistency Gate for
    diagnostics, and is returned regardless of the gate verdict (a failure is
    logged as a warning).  ``activation_stats`` are computed honestly from the
    activation series, independent of whether the gate passed.
    """
    from .event_discovery.consistency_gate import (
        ConsistencyGate,
        _build_month_index,
        _count_by_month,
    )

    gate_params = GateParams()
    gate = ConsistencyGate(gate_params)
    timestamps = _timestamps_from_frame(frame)
    month_index, n_total_months = _build_month_index(timestamps)

    candidates: List[EventCandidate] = []
    for ev in manual_events:
        cand = ev.to_event_candidate(frame, gate_params=gate_params)
        series = cand.event_series

        # Carry the same DatetimeIndex that M1 candidates expose, so Alpha/Rule
        # Discovery recognise the cached series and skip the apply() re-eval.
        if not isinstance(series.index, pd.DatetimeIndex):
            series = series.copy()
            series.index = pd.DatetimeIndex(timestamps.values, name="open_dt")
            cand.event_series = series

        gate_result = gate.evaluate_series(series, month_index, n_total_months)
        cand.consistency_gate = gate_result

        active = series.fillna(0).values.astype(bool)
        counts = _count_by_month(active, month_index, n_total_months)
        n_act = int(active.sum())
        n_active_months = int((counts > 0).sum())
        cand.activation_stats = ActivationStats(
            n_activations=n_act,
            n_active_months=n_active_months,
            zero_months=max(0, n_total_months - n_active_months),
            max_monthly_share=(float(counts.max()) / n_act) if n_act else float("nan"),
            mean_tpm=(n_act / n_total_months) if n_total_months else float("nan"),
        )

        if not gate_result.passed:
            logger.warning(
                "CustomEvent '%s' failed ConsistencyGate (%s) — proceeding anyway.",
                ev.name,
                gate_result.fail_reason,
            )
        candidates.append(cand)
    return candidates


def forge_multi(
    frames_by_ticker: Dict[str, pd.DataFrame],
    *,
    registry_config: Optional[RegistryConfig] = None,
    progress: bool = True,
    **forge_kwargs,
) -> Tuple[Dict[str, ForgeResult], RuleRegistry]:
    """Run :func:`forge` per ticker and pool the rules into one cross-ticker registry.

    This is the multi-asset entry point where the Rule Registry's cross-ticker
    backtest (Modulo 4, Step 4) becomes meaningful: every tradeable rule is
    replayed on every *other* ticker of the session with its thresholds
    recalibrated on the local distribution, and classified by genericity.

    Parameters
    ----------
    frames_by_ticker : dict[str, pd.DataFrame]
        ``ticker -> raw KPI table`` for every ticker of the session.
    registry_config : RegistryConfig, optional
        Modulo 4 configuration for the pooled registry.
    progress : bool, default True
        Print per-ticker status to ``stderr`` and forward the flag to every
        per-ticker :func:`forge` call.  Milestones are also logged at ``INFO``.
        Set ``False`` to silence all ``stderr`` output.
    **forge_kwargs
        Forwarded to every per-ticker :func:`forge` call (e.g. ``timeframe``,
        the per-module config objects).  Do **not** pass ``ticker`` / ``asset``
        (set automatically per ticker) or ``run_registry`` (the per-ticker
        registry is skipped in favour of the pooled one).

    Returns
    -------
    results : dict[str, ForgeResult]
        The per-ticker pipeline results (``registry`` is ``None`` on each — the
        pooled registry below supersedes them).
    registry : RuleRegistry
        The cross-ticker Rule Registry, already ``run()``.
    """
    forge_kwargs.pop("run_registry", None)
    forge_kwargs.pop("ticker", None)
    forge_kwargs.pop("asset", None)

    report = _Reporter(progress, label="multi")
    tickers = list(frames_by_ticker)
    report.stage(f"start — {len(tickers)} ticker(s): {', '.join(tickers)}")

    results: Dict[str, ForgeResult] = {}
    for i, tk in enumerate(tickers, start=1):
        report.stage(f"[{i}/{len(tickers)}] {tk}")
        results[tk] = forge(
            frames_by_ticker[tk],
            ticker=tk,
            run_registry=False,
            progress=progress,
            **forge_kwargs,
        )

    report.stage("pooling rules into cross-ticker registry…")
    registry = RuleRegistry.from_forge_results(results, config=registry_config).run()
    report.stage(f"done — {len(registry.documents)} pooled rule(s)")
    return results, registry
