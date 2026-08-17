"""Main RuleDiscovery orchestrator — FORGE pipeline Modulo 3.

Consumes an :class:`~forgedge.alpha_discovery.models.AlphaContract` plus its
originating :class:`~forgedge.event_discovery.models.EventCandidate`, and answers
the operational question the contract leaves open: *does the statistically
flagged pattern survive a realistic backtest — fees, finite fill rate, limit
orders, discrete target — and does it hold up out-of-sample?*

Flow (RuleDiscovery spec, Steps 1–5):

1. **Setup** — reconstruct the event's boolean activation series on the KPI table
   using the parameters stored on the candidate (Event Discovery's own replay,
   so activations match bit-for-bit), inject it as a signal column, and seed the
   operational grid from the contract's derived target.
2. **Backtest & scoring** — grid-screen the order mechanics in-sample, ranked by
   ``pf_score_tpm``.
3. **Selection & refinement** — pick the best operating point.
4. **Statistical validation + walk-forward OOS** — t-tests, Deflated Sharpe,
   temporal stability, and a walk-forward replay that re-selects parameters on
   each train window and scores them on the following untouched test window.
5. **Regime dependency** — per-regime performance and a concentration score.

The output is a :class:`RuleDiscoveryResponse` carrying the verdict
(``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE`` / ``INSUFFICIENT-DATA``) and the
validated rule.

Usage
-----
    from forgedge import (
        EventDiscovery, MarketContext, AlphaDiscovery, AlphaConfig,
        RuleDiscovery, RuleDiscoveryConfig,
    )

    enriched   = MarketContext(kpi).run()
    ed         = EventDiscovery(enriched)
    candidates = ed.run()
    ad         = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="ADAUSDC"))
    contracts  = ad.promoted_contracts()

    by_id = {c.event_id: c for c in candidates}
    for contract in contracts:
        cand = by_id[contract.event_candidate_id]
        rd   = RuleDiscovery(ed.df, contract, cand)
        response = rd.run()
        print(response.verdict)
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import replace
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

from ..alpha_discovery.models import AlphaContract
from ..event_discovery.discovery import _infer_timestamp_unit
from ..event_discovery.models import EventCandidate
from .analysis import excursion_stats, execution_envelope
from .backtest import _as_datetime64, _months_m, run_backtest
from .grid import grid_dataframe, run_grid, select_best
from .models import (
    BacktestParams,
    EntryOptimization,
    GridResult,
    GridSpec,
    RegimeBreakdown,
    RuleDiscoveryConfig,
    RuleDiscoveryResponse,
    ValidatedRule,
)
from ..resolver import resolve_config
from ..unset import UNSET, coalesce, is_set
from .validation import (
    _effective_sample,
    expectancy_mde,
    opportunity_sharpe,
    validate,
)
from .walkforward import _fmt as _fmt_ts, selection_windows, walk_forward

# RD-04 — the minimum executed-trade count is scaled to the in-sample length
# rather than a fixed absolute.  A hardcoded ``min_trades=30`` was architecturally
# incoherent with the rest of the pipeline: EventDiscovery promotes events with a
# period-relative ``min_tpm`` gate, so on a short IS (e.g. 12 months of 1D data)
# a genuinely strong rule could never reach 30 executed trades and was rejected by
# construction.  The gate is ``n_months * min_tpm`` — the same frequency criterion
# that admitted the event — floored at an absolute statistical minimum so a very
# short IS still demands enough trades to mean anything.
_MIN_TRADES_ABS = 10


def _dynamic_min_trades(n_months: float, min_tpm: float) -> int:
    """Minimum executed trades scaled to the IS period (spec RD-04)."""
    return max(_MIN_TRADES_ABS, int(n_months * min_tpm))


def _is_opportunity_sharpe(result, span_years: float) -> float:
    """``opportunity_sharpe`` of a grid cell, from its summary alone.

    ``BacktestSummary.sharpe_raw`` is already ``expectancy / std`` — the
    per-trade Sharpe — so the annualisation only needs the realised trade count,
    and the cell's trade ledger does not have to be materialised to rank it.
    """
    s = result.summary
    if not np.isfinite(s.sharpe_raw) or s.total_trades < 2 or span_years <= 0:
        return float("nan")
    return float(s.sharpe_raw) * math.sqrt(max(s.total_trades / span_years, 1e-12))


def _sortable(value: float) -> float:
    """``nan`` sorts last, so ``max()`` never picks an unmeasurable candidate."""
    return value if np.isfinite(value) else float("-inf")


def _round6(value: float) -> float:
    """Round for storage, leaving ``nan``/``inf`` alone."""
    return round(float(value), 6) if np.isfinite(value) else float(value)


def _pick_wf_params(splits, policy: str) -> BacktestParams:
    """Pick the published operating point from the walk-forward train selections.

    ``"last"`` — the most recent train window's winner (what you would trade
    next).  ``"consensus"`` — the most frequent parameter set across the
    splits, ties broken toward the most recent occurrence.
    """
    if policy == "consensus":
        def key(p: BacktestParams):
            return (p.buy_type, p.buy_drop_pct, p.buy_delay_bar, p.sell_pct, p.target_h)

        counts = Counter(key(s.params) for s in splits)
        top = max(counts.values())
        winners = {k for k, v in counts.items() if v == top}
        for s in reversed(splits):
            if key(s.params) in winners:
                return s.params
    return splits[-1].params


class RuleDiscovery:
    """FORGE Rule Discovery module (Modulo 3).

    Parameters
    ----------
    kpi_table : pd.DataFrame
        The KPI table that produced the candidate — Event Discovery's
        post-pipeline frame (``ed.df``) is ideal: it carries the native price
        columns, every derived feature, and the Market Context ``regime`` column.
        Must carry a datetime column (default ``open_dt``) or a DatetimeIndex.
    alpha_contract : AlphaContract
        The promoted contract from Alpha Discovery.  Its ``event_expression`` is
        used as-is; its ``derived_target`` seeds the operational grid.
    event_candidate : EventCandidate
        The candidate referenced by the contract.  Provides the deterministic
        event-reconstruction path (``apply``) and the SQL/formula expressions.
    config : RuleDiscoveryConfig, optional
        Grid, scoring, walk-forward and acceptance settings.
    """

    def __init__(
        self,
        kpi_table: pd.DataFrame,
        alpha_contract: AlphaContract,
        event_candidate: EventCandidate,
        config: Optional[RuleDiscoveryConfig] = None,
    ):
        self.config = resolve_config(config or RuleDiscoveryConfig(), "rule_discovery")
        self.contract = alpha_contract
        self.candidate = event_candidate

        if event_candidate.event_id != alpha_contract.event_candidate_id:
            raise ValueError(
                f"event_candidate.event_id ({event_candidate.event_id!r}) does not "
                f"match contract.event_candidate_id "
                f"({alpha_contract.event_candidate_id!r})."
            )

        self._frame = self._prepare_frame(kpi_table)
        self._mismatch_warned = False
        self.response: Optional[RuleDiscoveryResponse] = None
        # End of the selection span (walk-forward selection mode); the limit
        # entry optimiser restricts its sweep to this bound when set.
        self._selection_to: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> RuleDiscoveryResponse:
        """Execute the full pipeline and return the verdict response.

        The entry mechanism is selected by ``config.entry_mode`` (RD-130):

        * ``"limit"`` (default) — single-stage limit evaluation (unchanged).
        * ``"market"`` — single-stage baseline at next-open fill.
        * ``"auto"`` — two-stage: the **market** baseline decides the verdict, a
          **limit** optimiser then refines the operating point of EDGE/PARTIAL
          survivors at a comparable fill (see :meth:`_run_auto`).
        """
        mode = self._resolve_entry_mode()

        # ── Step 1 — setup ───────────────────────────────────────────────
        notes: List[str] = []
        base = self._seed_base_params(notes)

        if self.contract.derived_target.direction not in ("long", "short"):
            return self._reject(
                [f"contract direction {self.contract.derived_target.direction!r} "
                 "is not tradeable ('long' or 'short' required)"],
                base, notes,
            )

        self._inject_signal()

        if mode == "market":
            market_base = base.merged(buy_type="market")
            notes.append(
                "entry_mode=market — baseline entry (next-open fill, no entry optimiser)"
            )
            return self._run_stage(market_base, self._market_grid(market_base), notes)
        if mode == "auto":
            return self._run_auto(base, notes)
        return self._run_stage(base, self.config.grid, notes)

    def _run_stage(
        self, base: BacktestParams, grid_spec: GridSpec, notes: List[str]
    ) -> RuleDiscoveryResponse:
        """Single-entry-mode evaluation, dispatched on ``config.selection_mode``.

        ``"walk_forward"`` (default) confines the parameter selection and every
        verdict-feeding metric to the walk-forward's train windows (§3.4);
        ``"full_sample"`` is the legacy whole-table path.  The walk-forward mode
        falls back to the legacy path (with a note) when the data span is too
        short for a single split.
        """
        mode = str(getattr(self.config, "selection_mode", "full_sample")).lower().strip()
        if mode not in ("walk_forward", "full_sample"):
            raise ValueError(
                f"selection_mode must be 'walk_forward' or 'full_sample', got {mode!r}"
            )
        if mode == "walk_forward":
            resp = self._run_stage_wf(base, grid_spec, notes)
            if resp is not None:
                return resp
            notes.append(
                "selection_mode=walk_forward — data span too short for a single "
                "split; falling back to full-sample selection"
            )
        return self._run_stage_full(base, grid_spec, notes)

    def _run_stage_wf(
        self, base: BacktestParams, grid_spec: GridSpec, notes: List[str]
    ) -> Optional[RuleDiscoveryResponse]:
        """§3.4 — parameters are selected *only* inside walk-forward train windows.

        The published operating point comes from the per-split train selections
        (``config.wf_param_policy``: the last window's winner, or the consensus
        across windows).  The in-sample summary, the early-elimination screen,
        the statistical validation, the execution envelope, the MAE/MFE stats
        and the regime breakdown are all computed on the **selection span**
        ``[start, last train window end)`` — no metric that feeds the verdict
        or the ``ValidatedRule`` ever reads the final test window.  The
        early-elimination pre-screen runs on the *first* train window
        (selection-side data only, so hopeless rules stay cheap to reject).

        Returns ``None`` when the span cannot form a single walk-forward split
        (the caller falls back to full-sample selection).
        """
        cfg = self.config
        policy = str(getattr(cfg, "wf_param_policy", "last")).lower().strip()
        if policy not in ("last", "consensus"):
            raise ValueError(
                f"wf_param_policy must be 'last' or 'consensus', got {policy!r}"
            )

        windows = selection_windows(
            self._frame, grid_spec, base, cfg.walk_forward, cfg.timestamp_col
        )
        if not windows:
            return None
        first_from_s, first_to_s = _fmt_ts(windows[0][0]), _fmt_ts(windows[0][1])
        selection_to = _fmt_ts(windows[-1][1])

        # ── Step 2.3 — pre-screen on the first train window ──────────────
        if cfg.criteria.early_elimination:
            pre_grid = run_grid(
                self._frame, cfg.signal_col, base, grid_spec,
                scoring=cfg.scoring, timerange_from=first_from_s,
                timerange_to=first_to_s, timestamp_col=cfg.timestamp_col,
            )
            pre_best = select_best(pre_grid, cfg.criteria)
            if pre_best is None:
                return self._reject(
                    ["grid produced no evaluable configuration"], base, notes
                )
            elim = self._early_elimination(pre_best.summary)
            if elim:
                notes.append(
                    "selection_mode=walk_forward — early elimination on the first "
                    f"train window [{first_from_s}, {first_to_s})"
                )
                return self._reject(elim, pre_best.params, notes, pre_grid, pre_best.summary)

        # ── Step 4 — walk-forward: the only place parameters are selected ─
        wf = walk_forward(
            self._frame, cfg.signal_col, base, grid_spec, cfg.walk_forward,
            scoring=cfg.scoring, criteria=cfg.criteria, timestamp_col=cfg.timestamp_col,
            base_rate=float(self.contract.base_rate or 0.0),
        )
        if wf is None:
            return None
        best_params = _pick_wf_params(wf.splits, policy)
        self._selection_to = selection_to
        notes.append(
            "selection_mode=walk_forward — operating point from the "
            + ("consensus of the train windows" if policy == "consensus"
               else "last train window")
            + f"; IS metrics on [start, {selection_to})"
        )

        # Selection surface behind the published params (for the DSR trials).
        grid_results = list(wf.last_train_grid or [])
        if not grid_results:
            # reoptimise=False — no selection happened; a single evaluation of
            # the fixed params on the last train window stands in for the grid.
            fixed_summary = run_backtest(
                self._frame, cfg.signal_col, best_params,
                timerange_from=_fmt_ts(windows[-1][0]), timerange_to=selection_to,
                scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
            )
            grid_results = [GridResult(params=best_params, summary=fixed_summary)]
        n_trials = len(grid_results) * max(1, int(getattr(cfg, "n_trials_upstream", 1)))

        # ── IS summary of the published params on the selection span ─────
        is_summary, is_trades = run_backtest(
            self._frame, cfg.signal_col, best_params,
            timerange_to=selection_to,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col, return_trades=True,
        )
        elim = self._early_elimination(is_summary)

        # ── statistical validation + diagnostics, all on the selection span ─
        avg_hold = self._avg_holding_bars(is_trades)
        bpy = self._bars_per_year()
        stat_val = validate(
            is_trades, base_rate=float(self.contract.base_rate or 0.0),
            n_trials=n_trials, bars_per_year=bpy, avg_holding_bars=avg_hold,
        )
        envelope = execution_envelope(
            self._frame, cfg.signal_col, best_params,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
            timerange_to=selection_to,
        )
        excursion = excursion_stats(self._frame, is_trades)
        regime = self._regime_breakdown(is_trades)

        # ── Step 8 — verdict ─────────────────────────────────────────────
        verdict, reasons = self._decide(is_summary, stat_val, wf, regime)
        if elim:
            verdict = "NON-EDGE"
            reasons = elim + [r for r in reasons if r not in elim]
        validated = (
            ValidatedRule(
                expression=self.contract.event_expression,
                event_candidate_id=self.candidate.event_id,
                params=best_params,
            )
            if verdict != "NON-EDGE"
            else None
        )

        self.response = RuleDiscoveryResponse(
            date=self._discovery_date(),
            verdict=verdict,
            alpha_id=self.contract.alpha_id,
            asset=self.contract.asset,
            timeframe=self.contract.timeframe,
            validated_rule=validated,
            in_sample_summary=is_summary,
            walk_forward=wf,
            statistical_validation=stat_val,
            regime_analysis=regime,
            execution_envelope=envelope,
            excursion=excursion,
            grid_results=grid_results,
            rejection_reasons=reasons,
            notes=notes,
        )
        return self.response

    def _run_stage_full(
        self, base: BacktestParams, grid_spec: GridSpec, notes: List[str]
    ) -> RuleDiscoveryResponse:
        """Legacy full-sample evaluation: grid screen → verdict → response.

        Parameterised by ``base`` (carrying the entry ``buy_type``) and the grid
        ``grid_spec`` so the same core serves the limit, market and auto stages.
        With ``base``/``grid_spec`` left at their defaults this is bit-for-bit the
        pre-RD-130 ``run`` body.
        """
        cfg = self.config
        self._selection_to = None

        # ── Step 2/3 — in-sample grid screening and selection ────────────
        grid_results = run_grid(
            self._frame, cfg.signal_col, base, grid_spec,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
        )
        best = select_best(grid_results, cfg.criteria)
        if best is None:
            return self._reject(["grid produced no evaluable configuration"], base, notes)

        best_params = best.params
        is_summary = best.summary
        n_trials = len(grid_results) * max(1, int(getattr(cfg, "n_trials_upstream", 1)))

        # Re-run to obtain the trade ledger for the winning configuration.
        is_summary, is_trades = run_backtest(
            self._frame, cfg.signal_col, best_params,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col, return_trades=True,
        )

        # ── early elimination (Step 2.3) ─────────────────────────────────
        # A rule that fails the fast in-sample screen is definitively NON-EDGE.
        # By default we short-circuit here to skip the expensive walk-forward and
        # diagnostics; with criteria.early_elimination=False we run the full
        # pipeline anyway and fold these reasons into the final verdict.
        elim = self._early_elimination(is_summary)
        if elim and cfg.criteria.early_elimination:
            return self._reject(elim, best_params, notes, grid_results, is_summary)

        # ── Step 4 — statistical validation ──────────────────────────────
        avg_hold = self._avg_holding_bars(is_trades)
        bpy = self._bars_per_year()
        stat_val = validate(
            is_trades, base_rate=float(self.contract.base_rate or 0.0),
            n_trials=n_trials, bars_per_year=bpy, avg_holding_bars=avg_hold,
        )

        # ── Step 4 — walk-forward OOS ────────────────────────────────────
        wf = walk_forward(
            self._frame, cfg.signal_col, base, grid_spec, cfg.walk_forward,
            scoring=cfg.scoring, criteria=cfg.criteria, timestamp_col=cfg.timestamp_col,
            base_rate=float(self.contract.base_rate or 0.0),
        )
        if wf is None:
            notes.append("walk-forward skipped — data span too short for a split")

        # ── execution envelope + MAE/MFE (range of action, no target choice) ──
        envelope = execution_envelope(
            self._frame, cfg.signal_col, best_params,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
        )
        excursion = excursion_stats(self._frame, is_trades)

        # ── Step 5 — regime dependency ───────────────────────────────────
        regime = self._regime_breakdown(is_trades)

        # ── Step 8 — verdict ─────────────────────────────────────────────
        verdict, reasons = self._decide(is_summary, stat_val, wf, regime)
        # If the fast screen flagged the rule but early elimination is disabled,
        # the verdict is still NON-EDGE — now with the diagnostics computed above.
        if elim:
            verdict = "NON-EDGE"
            reasons = elim + [r for r in reasons if r not in elim]
        validated = (
            ValidatedRule(
                expression=self.contract.event_expression,
                event_candidate_id=self.candidate.event_id,
                params=best_params,
            )
            if verdict != "NON-EDGE"
            else None
        )

        self.response = RuleDiscoveryResponse(
            date=self._discovery_date(),
            verdict=verdict,
            alpha_id=self.contract.alpha_id,
            asset=self.contract.asset,
            timeframe=self.contract.timeframe,
            validated_rule=validated,
            in_sample_summary=is_summary,
            walk_forward=wf,
            statistical_validation=stat_val,
            regime_analysis=regime,
            execution_envelope=envelope,
            excursion=excursion,
            grid_results=grid_results,
            rejection_reasons=reasons,
            notes=notes,
        )
        return self.response

    # ------------------------------------------------------------------
    # RD-130 — entry mode (limit / market / auto)
    # ------------------------------------------------------------------

    def _resolve_entry_mode(self) -> str:
        mode = getattr(self.config, "entry_mode", "limit")
        if mode not in ("limit", "market", "auto"):
            raise ValueError(
                f"entry_mode must be 'limit', 'market' or 'auto', got {mode!r}"
            )
        return mode

    def _market_grid(self, market_base: BacktestParams) -> GridSpec:
        """Grid for a market stage.

        ``buy_drop_pct`` and ``buy_delay_bar`` are inert at next-open fill, so
        collapse them to a single value (avoiding redundant identical backtests)
        and only sweep the exit axes the user configured.
        """
        g = self.config.grid
        return GridSpec(
            buy_drop_pct=[market_base.buy_drop_pct],
            buy_delay_bar=[market_base.buy_delay_bar],
            sell_pct=g.sell_pct,
            target_h=g.target_h,
        )

    def _run_auto(self, base: BacktestParams, notes: List[str]) -> RuleDiscoveryResponse:
        """Two-stage evaluation: market decides the verdict, limit refines entry.

        Stage 1 runs the full pipeline in **market** mode — that verdict and all
        its evidence (walk-forward, statistics, regime, envelope) are
        authoritative.  Stage 2 runs a **limit** optimiser over ``buy_drop_pct``
        *only* on EDGE / PARTIAL survivors and adopts the improved operating point
        only when it still fills at ``>= criteria.min_fill_rate_opt``.  The
        optimiser can never change the verdict — it cannot fabricate an edge from a
        NON-EDGE-market rule.
        """
        market_base = base.merged(buy_type="market")
        market_grid = self._market_grid(market_base)
        notes.append("entry_mode=auto — Stage 1 market baseline (authoritative verdict)")
        resp = self._run_stage(market_base, market_grid, notes)

        if not resp.is_edge:
            resp.notes.append(
                "entry_mode=auto — market verdict NON-EDGE; limit optimiser skipped"
            )
            return resp

        opt = self._optimize_limit_entry(resp, base, market_grid)
        resp.entry_optimization = opt
        resp.notes.append(f"entry_mode=auto — {opt.reason}")
        if opt.adopted:
            # Swap only the entry mechanics of the operating point; the exit
            # (sell_pct / target_h) and the verdict are unchanged.
            resp.validated_rule = opt.limit_rule
        return resp

    def _limit_replay(self, limit_params: BacktestParams, market_grid: GridSpec):
        """Walk-forward **replay** of a fixed limit point — no re-optimisation.

        ``reoptimise=False`` evaluates the given parameters on every test window
        instead of re-selecting per window, which is exactly what "give this
        operating point an out-of-sample record" means.  It adds no selection,
        so it adds nothing to ``n_trials``.

        ``market_grid`` is passed as the spec even though no grid is run: the
        walk-forward's purge width is derived from the spec's widest trade span,
        so reusing the market stage's spec makes the two runs share their
        window geometry bar for bar.  Comparing two operating points scored on
        different windows would not be a comparison.
        """
        cfg = self.config
        replay_cfg = replace(cfg.walk_forward, reoptimise=False)
        return walk_forward(
            self._frame, cfg.signal_col, limit_params, market_grid, replay_cfg,
            scoring=cfg.scoring, criteria=cfg.criteria,
            timestamp_col=cfg.timestamp_col,
            base_rate=float(self.contract.base_rate or 0.0),
        )

    @staticmethod
    def _oos_span_years(wf) -> float:
        """Calendar years spanned by the concatenated test windows."""
        if wf is None or not wf.splits:
            return 0.0
        start = pd.Timestamp(wf.splits[0].test_from)
        end = pd.Timestamp(wf.splits[-1].test_to)
        return max((end - start).total_seconds() / (365.25 * 86400.0), 1e-9)

    def _optimize_limit_entry(
        self, resp: RuleDiscoveryResponse, base: BacktestParams,
        market_grid: GridSpec,
    ) -> EntryOptimization:
        """Evaluate a limit operating point against the market one, out-of-sample.

        Stage 2 in three moves: sweep ``buy_drop_pct`` in-sample (the exit stays
        the market winner's), replay the sweep's winner out-of-sample on the
        market point's own test windows, then adopt only if all three conditions
        hold **on that replay**.

        Why out-of-sample.  The previous criterion compared in-sample
        ``pf_score_tpm`` and adopted whichever was higher.  That published a
        point selected in-sample and never confirmed anywhere else, on strictly
        less evidence than the market point it replaced — which is the
        definition of selection.  Adopting the limit point *because it is
        better* obliges it to be better where selection did not reach.
        """
        cfg = self.config
        floor = cfg.criteria.min_fill_rate_opt
        retention = float(coalesce(cfg.criteria.min_net_gain_retention, default=0.5))
        market = resp.in_sample_summary
        winner = resp.validated_rule.params  # market operating point
        market_rule = resp.validated_rule

        def _kept(reason: str, failed: Optional[str] = None, **extra) -> EntryOptimization:
            """A decision that keeps the market point, with whatever was measured."""
            return EntryOptimization(
                selected_entry="market", adopted=False, min_fill_rate_opt=floor,
                market_profit_factor=market.profit_factor,
                market_fill_rate=market.fill_rate,
                market_pf_score_tpm=market.pf_score_tpm,
                limit_profit_factor=extra.pop("limit_profit_factor", None),
                limit_fill_rate=extra.pop("limit_fill_rate", None),
                limit_pf_score_tpm=extra.pop("limit_pf_score_tpm", None),
                limit_buy_drop_pct=extra.pop("limit_buy_drop_pct", None),
                reason=reason, failed_condition=failed,
                min_net_gain_retention=retention,
                market_rule=market_rule, **extra,
            )

        limit_base = winner.merged(buy_type="limit")
        limit_grid = GridSpec(
            buy_drop_pct=cfg.grid.buy_drop_pct,        # None → fanned around base
            sell_pct=[winner.sell_pct],                # exit held fixed
            target_h=[winner.target_h],
            buy_delay_bar=[base.buy_delay_bar],
        )
        # The sweep stays on the selection span — the final test window is
        # untouched by any selection, here as everywhere else.
        results = run_grid(
            self._frame, cfg.signal_col, limit_base, limit_grid,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
            timerange_to=self._selection_to,
        )
        tradeable = [
            r for r in results
            if np.isfinite(r.summary.fill_rate) and r.summary.fill_rate >= floor
        ]
        if not tradeable:
            return _kept(
                f"no limit entry reached fill ≥ {floor:.2f} in-sample; "
                "kept market operating point",
                failed="fill",
            )

        # Pick the in-sample candidate on the same quantity the adoption
        # decision uses, rather than on `pf_score_tpm`: a PF-based pick would
        # hand the OOS stage whichever point traded least, and the criterion
        # that follows exists precisely because that is not the same as best.
        is_span_years = max(market.n_months, 1) / 12.0
        best_limit = max(
            tradeable,
            key=lambda r: _sortable(_is_opportunity_sharpe(r, is_span_years)),
        )
        ls = best_limit.summary
        limit_params = best_limit.params.merged(buy_type="limit")
        seen = dict(
            limit_profit_factor=ls.profit_factor,
            limit_fill_rate=ls.fill_rate,
            limit_pf_score_tpm=ls.pf_score_tpm,
            limit_buy_drop_pct=best_limit.params.buy_drop_pct,
            # Computed here, not in the adopted branch: D9 asks for the DSR of
            # *each* point as an absolute metric, and a point that was measured
            # and rejected is exactly the case where a reader wants to see the
            # number that was measured.
            limit_validation=self._limit_validation(resp, limit_params, len(results)),
        )

        market_wf = resp.walk_forward
        if market_wf is None or market_wf.oos_trades is None:
            return _kept(
                "market point has no walk-forward record to compare against; "
                "kept market operating point (adoption requires OOS evidence)",
                **seen,
            )

        limit_wf = self._limit_replay(limit_params, market_grid)
        if limit_wf is None or limit_wf.oos_trades is None:
            return _kept(
                f"limit @ buy_drop={best_limit.params.buy_drop_pct} produced no "
                "out-of-sample trades on replay; kept market operating point",
                failed="fill", **seen,
            )

        span_years = self._oos_span_years(market_wf)
        m_sharpe = opportunity_sharpe(market_wf.oos_trades, span_years)
        l_sharpe = opportunity_sharpe(limit_wf.oos_trades, span_years)
        m_gain = float(market_wf.oos_summary.total_net_gain)
        l_gain = float(limit_wf.oos_summary.total_net_gain)
        l_fill = float(limit_wf.oos_summary.fill_rate)
        measured = dict(
            market_opportunity_sharpe=_round6(m_sharpe),
            limit_opportunity_sharpe=_round6(l_sharpe),
            market_oos_net_gain=_round6(m_gain),
            limit_oos_net_gain=_round6(l_gain),
            limit_oos_fill_rate=_round6(l_fill),
            limit_summary=limit_wf.oos_summary,
            market_summary=market_wf.oos_summary,
            limit_walk_forward=limit_wf,
            **seen,
        )

        # ── the three conditions, in the order that makes a failure readable ──
        if not (np.isfinite(l_fill) and l_fill >= floor):
            return _kept(
                f"limit @ buy_drop={best_limit.params.buy_drop_pct} filled "
                f"{l_fill:.2f} out-of-sample, below {floor:.2f}; kept market "
                "operating point",
                failed="fill", **measured,
            )
        if not (np.isfinite(l_sharpe) and np.isfinite(m_sharpe) and l_sharpe >= m_sharpe):
            return _kept(
                f"limit @ buy_drop={best_limit.params.buy_drop_pct} did not hold "
                f"its risk-adjusted return per unit of time out-of-sample "
                f"({l_sharpe:.3f} < market {m_sharpe:.3f}); kept market "
                "operating point",
                failed="sharpe", **measured,
            )
        gain_floor = retention * m_gain if m_gain > 0 else m_gain
        if not (np.isfinite(l_gain) and l_gain >= gain_floor):
            return _kept(
                f"limit @ buy_drop={best_limit.params.buy_drop_pct} kept only "
                f"{l_gain:.4f} of the market point's {m_gain:.4f} out-of-sample "
                f"net gain, below the {retention:.0%} floor; kept market "
                "operating point",
                failed="net_gain", **measured,
            )

        limit_rule = ValidatedRule(
            expression=self.contract.event_expression,
            event_candidate_id=self.candidate.event_id,
            params=limit_params,
        )
        return EntryOptimization(
            selected_entry="limit", adopted=True, min_fill_rate_opt=floor,
            market_profit_factor=market.profit_factor,
            market_fill_rate=market.fill_rate,
            market_pf_score_tpm=market.pf_score_tpm,
            reason=(
                f"limit @ buy_drop={best_limit.params.buy_drop_pct} held up "
                f"out-of-sample on all three conditions (fill {l_fill:.2f} ≥ "
                f"{floor:.2f}; annualised Sharpe {l_sharpe:.3f} ≥ market "
                f"{m_sharpe:.3f}; net gain {l_gain:.4f} ≥ {gain_floor:.4f}); "
                "operating point adopted"
            ),
            failed_condition=None,
            min_net_gain_retention=retention,
            market_rule=market_rule, limit_rule=limit_rule,
            **measured,
        )

    def _limit_validation(
        self, resp: RuleDiscoveryResponse, limit_params: BacktestParams,
        n_limit_cells: int,
    ):
        """In-sample statistics of the limit point, with **its own** trial count.

        The market point was chosen over Stage 1's cells; the limit point was
        chosen over those *and* Stage 2's, so its Deflated Sharpe pays a larger
        haircut.  Reported as an absolute metric per operating point (D9) — the
        ``min_dsr`` gate always reads the market point's, so the verdict never
        pays for Stage 2.
        """
        cfg = self.config
        market_trials = (resp.statistical_validation.n_trials_tested
                         if resp.statistical_validation is not None
                         else len(resp.grid_results or []) or 1)
        _summary, trades = run_backtest(
            self._frame, cfg.signal_col, limit_params,
            timerange_to=self._selection_to,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col, return_trades=True,
        )
        if trades is None or len(trades) < 2:
            return None
        return validate(
            trades, base_rate=float(self.contract.base_rate or 0.0),
            n_trials=int(market_trials) + max(int(n_limit_cells), 0),
            bars_per_year=self._bars_per_year(),
            avg_holding_bars=self._avg_holding_bars(trades),
        )

    def grid_summary(self) -> pd.DataFrame:
        """Flat DataFrame of the in-sample grid screening (call after ``run``)."""
        if self.response is None:
            raise RuntimeError("Call run() before grid_summary().")
        return grid_dataframe(self.response.grid_results)

    # ------------------------------------------------------------------
    # Step 1 — setup
    # ------------------------------------------------------------------

    def _seed_base_params(self, notes: List[str]) -> BacktestParams:
        """Seed operational defaults from the contract.

        The cost basis is seeded here too, and separately from the derived
        target: ``fee`` is not part of the target M2 derived, it is the
        assumption M2 *recorded*, and it is seeded even when
        ``use_contract_target`` is off — turning off the target does not mean
        agreeing to be charged a different fee than the contract documents
        (F7).  The seed only applies while the caller left ``base_params.fee``
        unresolved; an explicit fee on the config still wins.
        """
        base = self.config.base_params
        contract_fee = getattr(self.contract, "fee_per_side", UNSET)
        if not is_set(base.fee) and is_set(contract_fee) and contract_fee is not None:
            base = base.merged(fee=float(contract_fee))
            notes.append(f"fee seeded from contract cost basis: {float(contract_fee):g}/side")
        if not self.config.use_contract_target:
            return base.resolved()

        dt = self.contract.derived_target
        overrides = {}
        if dt.direction in ("long", "short"):
            overrides["direction"] = dt.direction
        # `is not None`, not truthiness: `h*=0` is a legal derived horizon since
        # #158 — a same-session round trip, exiting at the fill bar's own close
        # — and the old guard dropped it onto `base_params.target_h`, the hourly
        # default of 24.  A derived 0 became a silent 24 (F12).
        if dt.holding_period_h is not None and dt.holding_period_h >= 0:
            overrides["target_h"] = int(dt.holding_period_h)
        if dt.sell_pct and np.isfinite(dt.sell_pct) and dt.sell_pct > 0:
            # The derived sell_pct is a mean-advantage baseline; clamp to a
            # sane operational floor so the target is reachable intrabar.
            #
            # The floor is `criteria.min_sell_pct`, resolved from
            # `AlphaConfig.mfe_floor`.  It used to be a hardcoded `max(0.01, …)`
            # while M2's own floor was 0.005: the stricter of the two was the
            # one the user could not reach, and on intraday bars — where a
            # target on median MFE routinely sits under 1% — it replaced the
            # derived target with a constant, violating the pipeline's third
            # invariant (F11).
            floor = float(coalesce(self.config.criteria.min_sell_pct, default=0.005))
            overrides["sell_pct"] = round(max(floor, float(dt.sell_pct)), 4)
        if overrides:
            notes.append(
                f"seeded from contract target: "
                + ", ".join(f"{k}={v}" for k, v in overrides.items())
            )
        # `.resolved()` so that every params object descending from this one —
        # the whole grid, the walk-forward, whatever M4 later catalogues and
        # every report that renders it — carries values, never the sentinel.
        return base.merged(**overrides).resolved()

    def _inject_signal(self) -> None:
        """Reconstruct the event boolean series and add it as the signal column.

        An Event Candidate *is* an activation function.  Rule Discovery evaluates
        it on the candles it actually observes — the KPI frame — via the
        candidate's deterministic replay, :meth:`EventCandidate.apply`.  No
        threshold is re-fitted: this is feature *reconstruction*, exactly as the
        spec requires.

        The pre-computed ``EventCandidate.event_series`` is a **cache** of that
        same function evaluated on the training candles.  It is reused as a
        transparent fast path *only* when its index is identical to the observed
        frame's, where the cached evaluation equals ``apply()`` bit-for-bit.  When
        the candle sets differ the cache is silently inapplicable: reindexing it
        onto the frame would force every non-overlapping bar to "inactive" — at
        worst collapsing the signal to all-zeros and backtesting a rule that never
        fires — so the function is re-evaluated on the observed frame instead.
        This mirrors Alpha Discovery's ``_event_series`` no-recompute contract.
        """
        col = self.config.signal_col
        stored = self.candidate.event_series
        if stored is not None and stored.index.equals(self._frame.index):
            series = stored
        else:
            if stored is not None and not self._mismatch_warned:
                warnings.warn(
                    "Rule Discovery received candles whose index differs from the "
                    "event's stored activation series; the event is re-evaluated as "
                    "an activation function on the observed candles "
                    "(EventCandidate.apply). Windowed transforms reflect the history "
                    "available in the observed window.",
                    stacklevel=2,
                )
                self._mismatch_warned = True
            series = self.candidate.apply(self._frame)
        self._frame[col] = series.fillna(0).to_numpy()

    # ------------------------------------------------------------------
    # Step 2.3 — early elimination
    # ------------------------------------------------------------------

    def _early_elimination(self, s) -> List[str]:
        """Fast NON-EDGE screen (spec Section 2.3)."""
        cr = self.config.criteria
        reasons = []
        floor = _dynamic_min_trades(s.n_months, cr.min_tpm)
        if s.total_trades < floor:
            reasons.append(
                f"total_trades {s.total_trades} < {floor} "
                f"({s.n_months}mo × {cr.min_tpm} tpm, not significant)"
            )
        if np.isfinite(s.profit_factor) and s.profit_factor < 1.0:
            reasons.append(f"profit_factor {s.profit_factor:.2f} < 1.0 (losing in-sample)")
        if np.isfinite(s.fill_rate) and s.fill_rate < cr.min_fill_rate:
            reasons.append(
                f"fill_rate {s.fill_rate:.2f} < {cr.min_fill_rate} (buy_drop too deep)"
            )
        return reasons

    # ------------------------------------------------------------------
    # Step 5 — regime dependency
    # ------------------------------------------------------------------

    def _regime_breakdown(self, trades: pd.DataFrame) -> RegimeBreakdown:
        """Per-regime performance and a monthly-concentration dependency score."""
        # Monthly concentration score: 0 = uniform, 1 = all trades in one month.
        dep_score = self._dependency_score(trades)
        zero_months = self._zero_months(trades)

        per_regime: List[dict] = []
        avoid_in: List[str] = []
        regime_col = "regime"
        if regime_col in self._frame.columns and not trades.empty:
            regime_map = pd.Series(
                self._frame[regime_col].to_numpy(), index=self._frame.index
            )
            fill_dt = pd.Series(_as_datetime64(trades["fill_dt"]))
            regimes = fill_dt.map(regime_map).fillna("UNDEFINED").to_numpy()
            net = trades["net_pct_gain"].to_numpy()
            for label in pd.unique(regimes):
                mask = regimes == label
                sub = net[mask]
                if sub.size == 0:
                    continue
                pos = float(sub[sub > 0].sum())
                neg = float(-sub[sub < 0].sum())
                pf = (pos / neg) if neg > 0 else (9999.0 if pos > 0 else 0.0)
                wr = float((sub > 0).mean())
                exp = float(sub.mean())
                per_regime.append(
                    {
                        "regime": str(label),
                        "n_trades": int(sub.size),
                        "win_rate": round(wr, 4),
                        "profit_factor": round(pf, 4),
                        "expectancy": round(exp, 6),
                        "cum_gain": round(float(sub.sum()), 6),
                    }
                )
                if sub.size >= 5 and pf < 1.0:
                    avoid_in.append(str(label))

        return RegimeBreakdown(
            per_regime=per_regime,
            dependency_score=round(dep_score, 4),
            zero_months=zero_months,
            avoid_in=avoid_in,
        )

    @staticmethod
    def _dependency_score(trades: pd.DataFrame) -> float:
        """1 − normalised entropy of the monthly trade counts (spec 5.3)."""
        if trades.empty:
            return float("nan")
        months = _months_m(trades["fill_dt"])
        counts = months.value_counts().to_numpy(dtype=float)
        if counts.size <= 1:
            return 1.0
        p = counts / counts.sum()
        h = -float((p * np.log(p + 1e-12)).sum())
        h_max = math.log(counts.size)
        return 1.0 - (h / h_max) if h_max > 0 else 0.0

    def _zero_months(self, trades: pd.DataFrame) -> int:
        if trades.empty:
            return 0
        ts = (
            self._frame[self.config.timestamp_col]
            if self.config.timestamp_col in self._frame.columns
            else self._frame.index
        )
        dt = pd.DatetimeIndex(_as_datetime64(ts))
        start = dt.min().to_period("M")
        end = dt.max().to_period("M")
        n_months = max((end.year - start.year) * 12 + (end.month - start.month) + 1, 1)
        active = _months_m(trades["fill_dt"]).nunique()
        return max(n_months - active, 0)

    # ------------------------------------------------------------------
    # Step 8 — verdict
    # ------------------------------------------------------------------

    def _decide(self, s, stat_val, wf, regime):
        """Map the evidence to EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA.

        Hard gates → NON-EDGE; full-EDGE requirements → PARTIAL-EDGE; a
        would-be positive verdict whose pooled OOS evidence cannot support
        it → INSUFFICIENT-DATA (§3.2, ``criteria.power_gate``).
        """
        cr = self.config.criteria
        reasons: List[str] = []
        # Gates that failed because the *window* could not support the test, not
        # because the rule did badly.  These produce INSUFFICIENT-DATA, never
        # NON-EDGE (see the trade-floor block below).
        underpowered: List[str] = []

        # Hard NON-EDGE gates — the rule is simply not operable / not real.
        if not (np.isfinite(s.profit_factor) and s.profit_factor >= cr.partial_min_profit_factor):
            reasons.append(
                f"in-sample PF {s.profit_factor:.2f} < {cr.partial_min_profit_factor}"
            )
        # The trade floor is a *window* statement, not a verdict on the rule.
        #
        # When `n_months × min_tpm` cannot reach the floor, no candidate can
        # clear it — the selection span is simply too short to support the test,
        # and calling that NON-EDGE blames the rule for the configuration
        # (#173/#177).  M3 already has the honest answer for "the evidence
        # cannot support a verdict": INSUFFICIENT-DATA.  Below the floor with a
        # window that *could* have supplied it, the rule really did under-trade
        # and NON-EDGE stands.
        min_trades = _dynamic_min_trades(s.n_months, cr.min_tpm)
        window_too_short = s.n_months * cr.min_tpm < min_trades
        if s.total_trades < min_trades:
            detail = (f"total_trades {s.total_trades} < {min_trades} "
                      f"({s.n_months}mo × {cr.min_tpm} tpm)")
            if window_too_short:
                underpowered.append(
                    detail + f" — the {s.n_months}-month selection span cannot "
                    f"supply {min_trades} trades at criteria.min_tpm="
                    f"{cr.min_tpm:g} however good the rule is; this is a window "
                    f"length, not a verdict (#173)"
                )
            else:
                reasons.append(detail)
        if stat_val is not None and not (
            np.isfinite(stat_val.ttest_expectancy_p)
            and stat_val.ttest_expectancy_p < cr.max_ttest_p
        ):
            reasons.append(
                f"expectancy not significant (p={stat_val.ttest_expectancy_p})"
            )
        # Out-of-sample must hold up.
        if wf is not None and not (np.isfinite(wf.oos_summary.profit_factor)
                                   and wf.oos_summary.profit_factor >= 1.0):
            reasons.append(
                f"walk-forward OOS PF {wf.oos_summary.profit_factor:.2f} < 1.0"
            )

        if reasons:
            return "NON-EDGE", reasons
        if underpowered:
            return "INSUFFICIENT-DATA", underpowered

        # Full-EDGE requirements.
        edge_block: List[str] = []
        if not (np.isfinite(s.profit_factor) and s.profit_factor >= cr.min_profit_factor):
            edge_block.append(f"PF {s.profit_factor:.2f} < {cr.min_profit_factor}")
        if not (np.isfinite(s.win_rate_pct) and s.win_rate_pct >= cr.min_win_rate):
            edge_block.append(f"WR {s.win_rate_pct} < {cr.min_win_rate}")
        active_rate = (s.n_months - s.zero_months) / s.n_months if s.n_months > 0 else 0.0
        if active_rate < cr.min_active_month_rate:
            edge_block.append(
                f"active_months {s.n_months - s.zero_months}/{s.n_months}"
                f" = {active_rate:.0%} < {cr.min_active_month_rate:.0%}"
            )
        if stat_val is not None and np.isfinite(stat_val.deflated_sharpe) \
                and stat_val.deflated_sharpe < cr.min_dsr:
            edge_block.append(f"DSR {stat_val.deflated_sharpe:.2f} < {cr.min_dsr}")
        # A finite Sharpe whose deflation is undefined means the selection
        # haircut's radicand went negative — n_trials too large for n_obs, the
        # Sharpe is "not credible" per the spec.  Previously this silently
        # skipped the DSR gate; degrade honestly instead.
        if stat_val is not None and np.isfinite(stat_val.sharpe_ratio) \
                and not np.isfinite(stat_val.deflated_sharpe):
            edge_block.append(
                f"DSR undefined — selection bias too severe "
                f"(n_trials={stat_val.n_trials_tested} vs n_obs={s.total_trades})"
            )
        if stat_val is not None and stat_val.temporal_stability == "FAIL":
            edge_block.append("temporal stability FAIL")
        if regime is not None and np.isfinite(regime.dependency_score) \
                and regime.dependency_score > cr.max_regime_dependency:
            edge_block.append(
                f"regime dependency {regime.dependency_score:.2f} > {cr.max_regime_dependency}"
            )
        if wf is not None and wf.consistency < 0.5:
            edge_block.append(f"OOS consistency {wf.consistency:.2f} < 0.5")
        # Search-level rotation null (annotated on the contract by
        # FastRotationNull / RotationCalibrator): a full EDGE must beat the
        # multiple-testing surface of its own discovery session.  Inert when
        # the contract carries no annotation.
        rot_p = getattr(self.contract, "rotation_p", None)
        if rot_p is not None and np.isfinite(rot_p) and rot_p > cr.max_rotation_p:
            edge_block.append(
                f"search-level rotation null not cleared "
                f"(rotation_p={rot_p:.4f} > {cr.max_rotation_p})"
            )

        verdict = "PARTIAL-EDGE" if edge_block else "EDGE"

        # ── §3.2 — power-aware degradation of positive verdicts ──────────
        # A positive verdict must be *confirmable*: when the out-of-sample
        # evidence is too thin to support it, the honest verdict is
        # INSUFFICIENT-DATA, not a confident (PARTIAL-)EDGE.  The assessment
        # reads only the pooled OOS ledger — never per-window counts (the
        # walk-forward windows are short by design and are not gated
        # individually).  NON-EDGE verdicts are never rescued: underpowered
        # or not, the operational consequence is the same.
        if cr.power_gate:
            power_reasons = self._power_assessment(s, wf)
            if power_reasons:
                return "INSUFFICIENT-DATA", power_reasons + edge_block
        return verdict, edge_block

    def _power_assessment(self, s, wf) -> List[str]:
        """Reasons the pooled OOS evidence cannot support a positive verdict.

        Empty list = adequately powered.  Criteria (aggregate only, per §3.2):

        * no walk-forward at all — a positive verdict would rest on zero OOS;
        * pooled test-window trades below ``criteria.min_oos_trades``;
        * the pooled OOS sample's minimum detectable expectancy exceeds the
          claimed (in-sample) expectancy — the OOS could not confirm an
          effect of that size even if it were real.
        """
        cr = self.config.criteria
        if wf is None:
            return [
                "no walk-forward OOS available — a positive verdict cannot "
                "be confirmed on this span"
            ]
        oos = wf.oos_trades
        n_oos = 0 if oos is None else int(len(oos))
        if n_oos < cr.min_oos_trades:
            return [
                f"pooled OOS trades {n_oos} < {cr.min_oos_trades} — too few to "
                "confirm the verdict (counted across all test windows; "
                "individual windows are never gated)"
            ]
        net = oos["net_pct_gain"].to_numpy(dtype=float)
        # The bar an observed mean must clear is set by how much *independent*
        # evidence there is, not by how many rows the ledger has: overlapping
        # positions share a price path (F16).  The count gate above stays
        # nominal — `min_oos_trades` is about having a track record at all.
        mde = expectancy_mde(net, n_eff=_effective_sample(oos))
        claimed = s.expectancy
        if np.isfinite(mde) and np.isfinite(claimed) and mde > max(claimed, 0.0):
            return [
                f"OOS underpowered: minimum detectable expectancy {mde:.4f} > "
                f"claimed IS expectancy {claimed:.4f} — the pooled OOS sample "
                "cannot confirm an effect of the claimed size"
            ]
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reject(self, reasons, params, notes, grid_results=None, is_summary=None):
        from .backtest import _empty_summary
        if is_summary is None:
            is_summary = _empty_summary(1)
        self.response = RuleDiscoveryResponse(
            date=self._discovery_date(),
            verdict="NON-EDGE",
            alpha_id=self.contract.alpha_id,
            asset=self.contract.asset,
            timeframe=self.contract.timeframe,
            validated_rule=None,
            in_sample_summary=is_summary,
            walk_forward=None,
            statistical_validation=None,
            regime_analysis=None,
            grid_results=grid_results or [],
            rejection_reasons=reasons,
            notes=notes,
        )
        return self.response

    def _avg_holding_bars(self, trades: pd.DataFrame) -> Optional[float]:
        if trades.empty or "exit_rn" not in trades.columns:
            return None
        hold = (trades["exit_rn"] - trades["fill_rn"]).to_numpy(dtype=float)
        hold = hold[np.isfinite(hold) & (hold > 0)]
        return float(hold.mean()) if hold.size else None

    def _bars_per_year(self) -> float:
        idx = self._frame.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
            delta = pd.Series(idx).diff().dt.total_seconds().median()
            if delta and delta > 0:
                return 365.25 * 86400.0 / float(delta)
        return 24 * 365.25

    def _discovery_date(self) -> str:
        return self.config.discovery_date or date.today().isoformat()

    def _prepare_frame(self, kpi_table: pd.DataFrame) -> pd.DataFrame:
        """Chronologically-sorted copy with a DatetimeIndex and an ``open_dt`` column.

        Accepts a DatetimeIndex or a ``timestamp_col`` of datetimes, numeric Unix
        timestamps (unit auto-inferred), or ISO strings — the same contract as
        Event/Alpha Discovery.  Both the index *and* the timestamp column are kept
        populated: feature reconstruction aligns on the index, the backtest reads
        the column.
        """
        df = kpi_table.copy()
        ts_col = self.config.timestamp_col

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            if ts_col not in df.columns:
                df[ts_col] = df.index
            return df

        if ts_col not in df.columns:
            raise ValueError(
                f"Timestamp column {ts_col!r} not found and index is not a "
                "DatetimeIndex. Set RuleDiscoveryConfig(timestamp_col='...')."
            )

        raw = df[ts_col]
        if pd.api.types.is_datetime64_any_dtype(raw):
            parsed = pd.to_datetime(raw)
        elif pd.api.types.is_numeric_dtype(raw):
            parsed = pd.to_datetime(raw, unit=_infer_timestamp_unit(raw))
        else:
            parsed = pd.to_datetime(raw)

        df.index = pd.DatetimeIndex(parsed.to_numpy(), name=ts_col)
        df[ts_col] = df.index
        return df.sort_index()
