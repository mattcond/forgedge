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
(``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE``) and the validated rule.

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
    GridResult,
    RegimeBreakdown,
    RuleDiscoveryConfig,
    RuleDiscoveryResponse,
    ValidatedRule,
)
from .validation import validate
from .walkforward import walk_forward

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
        self.config = config or RuleDiscoveryConfig()
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> RuleDiscoveryResponse:
        """Execute the full pipeline and return the verdict response."""
        cfg = self.config

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

        # ── Step 2/3 — in-sample grid screening and selection ────────────
        grid_results = run_grid(
            self._frame, cfg.signal_col, base, cfg.grid,
            scoring=cfg.scoring, timestamp_col=cfg.timestamp_col,
        )
        best = select_best(grid_results, cfg.criteria)
        if best is None:
            return self._reject(["grid produced no evaluable configuration"], base, notes)

        best_params = best.params
        is_summary = best.summary
        n_trials = len(grid_results)

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
            self._frame, cfg.signal_col, base, cfg.grid, cfg.walk_forward,
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

    def grid_summary(self) -> pd.DataFrame:
        """Flat DataFrame of the in-sample grid screening (call after ``run``)."""
        if self.response is None:
            raise RuntimeError("Call run() before grid_summary().")
        return grid_dataframe(self.response.grid_results)

    # ------------------------------------------------------------------
    # Step 1 — setup
    # ------------------------------------------------------------------

    def _seed_base_params(self, notes: List[str]) -> BacktestParams:
        """Seed operational defaults from the contract's derived target."""
        base = self.config.base_params
        if not self.config.use_contract_target:
            return base

        dt = self.contract.derived_target
        overrides = {}
        if dt.direction in ("long", "short"):
            overrides["direction"] = dt.direction
        if dt.holding_period_h and dt.holding_period_h > 0:
            overrides["target_h"] = int(dt.holding_period_h)
        if dt.sell_pct and np.isfinite(dt.sell_pct) and dt.sell_pct > 0:
            # The derived sell_pct is a mean-advantage baseline; clamp to a
            # sane operational floor so the target is reachable intrabar.
            overrides["sell_pct"] = round(max(0.01, float(dt.sell_pct)), 4)
        if overrides:
            notes.append(
                f"seeded from contract target: "
                + ", ".join(f"{k}={v}" for k, v in overrides.items())
            )
        return base.merged(**overrides)

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
        """Map the evidence to EDGE / PARTIAL-EDGE / NON-EDGE (spec Section 8)."""
        cr = self.config.criteria
        reasons: List[str] = []

        # Hard NON-EDGE gates — the rule is simply not operable / not real.
        if not (np.isfinite(s.profit_factor) and s.profit_factor >= cr.partial_min_profit_factor):
            reasons.append(
                f"in-sample PF {s.profit_factor:.2f} < {cr.partial_min_profit_factor}"
            )
        min_trades = _dynamic_min_trades(s.n_months, cr.min_tpm)
        if s.total_trades < min_trades:
            reasons.append(
                f"total_trades {s.total_trades} < {min_trades} "
                f"({s.n_months}mo × {cr.min_tpm} tpm)"
            )
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
        if stat_val is not None and stat_val.temporal_stability == "FAIL":
            edge_block.append("temporal stability FAIL")
        if regime is not None and np.isfinite(regime.dependency_score) \
                and regime.dependency_score > cr.max_regime_dependency:
            edge_block.append(
                f"regime dependency {regime.dependency_score:.2f} > {cr.max_regime_dependency}"
            )
        if wf is not None and wf.consistency < 0.5:
            edge_block.append(f"OOS consistency {wf.consistency:.2f} < 0.5")

        if edge_block:
            return "PARTIAL-EDGE", edge_block
        return "EDGE", []

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
