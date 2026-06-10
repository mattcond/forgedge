"""Main AlphaDiscovery orchestrator — FORGE pipeline Modulo 2.

Consumes the ``EventCandidate`` list produced by Event Discovery, reconstructs
each event (and its underlying continuous feature) on the KPI table, measures
predictive power against an economic target, and compiles an ``AlphaContract``
per candidate.

The flow is strictly sequential and read-only: regimes come from Market
Context, events come from Event Discovery — Alpha Discovery recomputes
neither.  It reads each candidate's stored ``event_series``, reads the
``regime`` column from the enriched table, and reads the underlying feature
columns from Event Discovery's post-pipeline table.

Usage
-----
    from forgedge import (
        EventDiscovery, MarketContext,
        AlphaDiscovery, AlphaConfig, TargetDefinition,
    )

    enriched = MarketContext(kpi).run()          # Modulo 0 — adds 'regime'
    ed = EventDiscovery(enriched)
    candidates = ed.run()                        # Modulo 1 — Event Candidates

    ad = AlphaDiscovery(
        ed.df,                                   # regime + derived features
        candidates,
        AlphaConfig(target=TargetDefinition(holding_period_h=24, sell_pct=0.04)),
    )
    contracts = ad.run()                # all evaluated contracts
    promoted = ad.promoted_contracts()  # status == "HYPOTHESIS"
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

from ..event_discovery.discovery import _infer_timestamp_unit
from ..event_discovery.models import EventCandidate, build_feature_series
from . import stats
from .market_structure import analyse_market_structure
from .models import (
    AlphaConfig,
    AlphaContract,
    AlphaScore,
    EventStats,
    ICResult,
    MarketStructure,
    RegimeAnalysis,
    RegimeStat,
    TargetDefinition,
)
from .target import build_target


class AlphaDiscovery:
    """FORGE Alpha Discovery module (Modulo 2).

    Parameters
    ----------
    kpi_table : pd.DataFrame
        The KPI table that produced the candidates.  In the sequential FORGE
        flow this is Event Discovery's post-pipeline table (``ed.df``): it
        carries the Market Context ``regime`` column *and* every derived
        feature column, so Alpha Discovery recomputes nothing — it only reads.
        Any table with the same native columns also works (missing derived
        features are then replayed deterministically from the candidates'
        stored parameters).  Must carry a datetime column (default
        ``open_dt``) or a DatetimeIndex.
    event_candidates : list[EventCandidate]
        Output of ``EventDiscovery.run()``.  Each candidate's stored
        ``event_series`` is used as-is — events are never re-derived.
    config : AlphaConfig, optional
        Target definition and gates.  Defaults to a 24-bar / +4% long target
        with the documented promotion thresholds.
    """

    def __init__(
        self,
        kpi_table: pd.DataFrame,
        event_candidates: List[EventCandidate],
        config: Optional[AlphaConfig] = None,
    ):
        self.config = config or AlphaConfig()
        self.event_candidates = list(event_candidates)

        self._frame = self._prepare_frame(kpi_table)
        self._contracts: Optional[List[AlphaContract]] = None

        # Populated by run().
        self.base_rate: float = float("nan")
        self.market_structure: Optional[MarketStructure] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> List[AlphaContract]:
        """Evaluate every candidate and return one contract each.

        Returns the **full** list (promoted and rejected) so callers can audit
        why a candidate did not make it.  Use :meth:`promoted_contracts` for
        the ``HYPOTHESIS`` subset that is handed to Rule Discovery.
        """
        cfg = self.config
        tgt = cfg.target

        close = self._frame[cfg.close_col].astype(float)
        fwd_return, target_binary, base_rate = build_target(close, tgt)
        self.base_rate = base_rate
        self.market_structure = analyse_market_structure(close, fwd_return)

        regime = self._regime_series()
        ic_window = self._rolling_ic_window()

        # ── Per-candidate measurement ────────────────────────────────────
        measured = []
        for cand in self.event_candidates:
            event = self._event_series(cand)
            comp0 = cand.components[0]
            feature = self._feature_series(comp0)

            ic_res = self._measure_ic(feature, fwd_return, comp0.source_feature, ic_window)
            ev_stats = self._measure_event(event, target_binary, fwd_return, base_rate)
            regime_res = self._measure_regimes(event, feature, fwd_return, target_binary, regime)
            measured.append((cand, ic_res, ev_stats, regime_res))

        # ── Multiple-testing control across all candidates (Section 13) ──
        p_values = [m[2].p_value for m in measured]
        fdr_mask = stats.benjamini_hochberg(p_values, cfg.thresholds.fdr_q)

        # ── Compile contracts ────────────────────────────────────────────
        contracts: List[AlphaContract] = []
        for idx, ((cand, ic_res, ev_stats, regime_res), fdr_ok) in enumerate(
            zip(measured, fdr_mask)
        ):
            score = self._score(ic_res, ev_stats, regime_res)
            contract = self._build_contract(
                idx, cand, ic_res, ev_stats, regime_res, score, bool(fdr_ok)
            )
            contracts.append(contract)

        self._contracts = contracts
        return contracts

    def promoted_contracts(self) -> List[AlphaContract]:
        """Return only the contracts that cleared every gate (``HYPOTHESIS``)."""
        if self._contracts is None:
            raise RuntimeError("Call run() before promoted_contracts().")
        return [c for c in self._contracts if c.promoted]

    def summary(self) -> pd.DataFrame:
        """Flat, sortable summary of every evaluated candidate.

        Sorted by composite score (descending) so the strongest candidates are
        at the top.  Returns an empty, correctly-typed frame when there were no
        candidates.
        """
        if self._contracts is None:
            raise RuntimeError("Call run() before summary().")
        _cols = [
            "alpha_id", "status", "promoted", "event_candidate_id", "expression",
            "pattern_family", "feature", "ic", "ic_p_value", "ic_admitted",
            "rolling_ic_stable", "n_activations", "win_rate", "base_rate", "lift",
            "fwd_return_mean", "cohens_d", "t_stat", "p_value", "fdr_promoted",
            "regime_dependency", "regime_breadth", "composite_score", "grade",
            "rejection_reasons",
        ]
        if not self._contracts:
            return pd.DataFrame(columns=_cols)
        df = pd.DataFrame([c.to_dict() for c in self._contracts])
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 3 — IC measurement
    # ------------------------------------------------------------------

    def _measure_ic(
        self,
        feature: pd.Series,
        fwd_return: pd.Series,
        feature_name: str,
        ic_window: int,
    ) -> ICResult:
        """Spearman IC of the underlying feature vs the forward return."""
        f = feature.astype(float)
        ic, p = stats.spearmanr(f.to_numpy(), fwd_return.to_numpy())
        n = int((f.notna() & fwd_return.notna()).sum())

        stable, roll_mean, sign_consistency = self._rolling_ic_stability(
            f, fwd_return, ic, ic_window
        )

        th = self.config.thresholds
        weak_ic = (not np.isfinite(ic)) or abs(ic) < th.ic_min_abs
        weak_p = (not np.isfinite(p)) or p > th.ic_max_p
        admitted = not (weak_ic and weak_p)

        return ICResult(
            feature=feature_name,
            ic=float(ic) if np.isfinite(ic) else float("nan"),
            p_value=float(p) if np.isfinite(p) else float("nan"),
            n=n,
            rolling_ic_stable=stable,
            rolling_ic_mean=roll_mean,
            rolling_sign_consistency=sign_consistency,
            admitted=admitted,
        )

    def _rolling_ic_stability(
        self,
        feature: pd.Series,
        fwd_return: pd.Series,
        overall_ic: float,
        window: int,
    ):
        """Strided rolling IC; returns (stable, mean_ic, sign_consistency).

        A bounded number of evenly-spaced windows (≈20) is evaluated rather
        than every bar, keeping the cost flat regardless of dataset length.
        Stability means the rolling IC keeps the sign of the overall IC in at
        least 70% of windows.
        """
        n = len(feature)
        if not np.isfinite(overall_ic) or overall_ic == 0 or n <= window:
            return None, None, None

        stride = max(1, (n - window) // 20)
        ics = []
        f = feature.to_numpy()
        r = fwd_return.to_numpy()
        for start in range(0, n - window + 1, stride):
            wic, _ = stats.spearmanr(f[start : start + window], r[start : start + window])
            if np.isfinite(wic):
                ics.append(wic)
        if len(ics) < 3:
            return None, None, None

        ics_arr = np.asarray(ics)
        sign_consistency = float(np.mean(np.sign(ics_arr) == np.sign(overall_ic)))
        return bool(sign_consistency >= 0.7), float(ics_arr.mean()), sign_consistency

    # ------------------------------------------------------------------
    # Step 4 — win rate analysis
    # ------------------------------------------------------------------

    def _measure_event(
        self,
        event: pd.Series,
        target_binary: pd.Series,
        fwd_return: pd.Series,
        base_rate: float,
    ) -> EventStats:
        """Measure the binary event's predictive power against the target."""
        active = event.fillna(0).astype(bool)
        target_valid = target_binary.notna()

        active_valid = active & target_valid
        inactive_valid = (~active) & target_valid

        n_act = int(active_valid.sum())
        wr = float(target_binary[active_valid].mean()) if n_act else float("nan")
        lift = wr - base_rate if np.isfinite(wr) else float("nan")

        active_ret = fwd_return[active_valid].to_numpy()
        inactive_ret = fwd_return[inactive_valid].to_numpy()

        fwd_mean = float(np.nanmean(active_ret)) if active_ret.size else float("nan")
        d = stats.cohens_d(active_ret, inactive_ret)
        t, p = stats.ttest_ind(active_ret, inactive_ret, alternative="greater")

        return EventStats(
            n_activations=n_act,
            win_rate=wr,
            base_rate=base_rate,
            lift=lift,
            fwd_return_mean=fwd_mean,
            cohens_d=d,
            t_stat=t,
            p_value=p,
        )

    # ------------------------------------------------------------------
    # Step 5 — regime sensitivity
    # ------------------------------------------------------------------

    def _measure_regimes(
        self,
        event: pd.Series,
        feature: pd.Series,
        fwd_return: pd.Series,
        target_binary: pd.Series,
        regime: Optional[pd.Series],
    ) -> RegimeAnalysis:
        """Per-regime IC and win rate, plus the dependency classification."""
        if regime is None:
            return RegimeAnalysis([], "unknown", [], [], float("nan"))

        cfg = self.config
        active = event.fillna(0).astype(bool)
        target_valid = target_binary.notna()

        labels = list(regime.cat.categories) if hasattr(regime, "cat") else \
            list(pd.unique(regime.dropna()))

        per_regime: List[RegimeStat] = []
        evaluated = 0
        significant: List[str] = []
        weak: List[str] = []

        for label in labels:
            rmask = (regime == label)
            n_obs = int((rmask & feature.notna() & fwd_return.notna()).sum())
            if n_obs < cfg.min_regime_obs:
                per_regime.append(
                    RegimeStat(str(label), n_obs, float("nan"), float("nan"),
                               float("nan"), "insufficient")
                )
                continue

            ic_r, p_r = stats.spearmanr(
                feature[rmask].to_numpy(), fwd_return[rmask].to_numpy()
            )
            ev_in = active & rmask & target_valid
            wr_r = float(target_binary[ev_in].mean()) if int(ev_in.sum()) else float("nan")

            strength = self._regime_strength(ic_r, p_r)
            per_regime.append(
                RegimeStat(str(label), n_obs, float(ic_r), float(p_r), wr_r, strength)
            )
            evaluated += 1
            if strength in ("strong", "moderate"):
                significant.append(str(label))
            else:
                weak.append(str(label))

        breadth = (len(significant) / evaluated) if evaluated else float("nan")
        dependency = self._dependency_type(evaluated, len(significant))

        return RegimeAnalysis(per_regime, dependency, significant, weak, breadth)

    @staticmethod
    def _regime_strength(ic: float, p: float) -> str:
        """Classify a regime's IC into strong / moderate / negligible."""
        if not np.isfinite(ic) or not np.isfinite(p):
            return "negligible"
        if p < 0.05 and abs(ic) >= 0.05:
            return "strong"
        if p < 0.05:
            return "moderate"
        return "negligible"

    @staticmethod
    def _dependency_type(evaluated: int, n_significant: int) -> str:
        """Map (evaluated, significant) regime counts to a dependency label."""
        if evaluated == 0:
            return "unknown"
        if n_significant == 0:
            return "broken"
        if n_significant == evaluated and evaluated >= 2:
            return "agnostic"
        if n_significant == 1:
            return "specific"
        return "conditional"

    # ------------------------------------------------------------------
    # Step 6 — scoring
    # ------------------------------------------------------------------

    def _score(
        self, ic: ICResult, ev: EventStats, regime: RegimeAnalysis
    ) -> AlphaScore:
        """Composite alpha score and grade (doc Section 6.1/6.2)."""
        w_ic, w_lift, w_d, w_breadth = self.config.score_weights

        ic_norm = _clamp01(abs(ic.ic) / 0.10) if np.isfinite(ic.ic) else 0.0
        lift_norm = _clamp01(ev.lift / 0.30) if np.isfinite(ev.lift) else 0.0
        d_norm = _clamp01(ev.cohens_d / 0.80) if np.isfinite(ev.cohens_d) else 0.0
        breadth = regime.regime_breadth

        if np.isfinite(breadth):
            terms = [(w_ic, ic_norm), (w_lift, lift_norm), (w_d, d_norm),
                     (w_breadth, breadth)]
        else:
            # No regime information — drop the breadth term and renormalise.
            terms = [(w_ic, ic_norm), (w_lift, lift_norm), (w_d, d_norm)]

        total_w = sum(w for w, _ in terms)
        composite = sum(w * v for w, v in terms) / total_w if total_w else 0.0

        return AlphaScore(
            ic_magnitude=float(abs(ic.ic)) if np.isfinite(ic.ic) else float("nan"),
            lift=ev.lift,
            cohens_d=ev.cohens_d,
            regime_breadth=breadth,
            composite_score=round(float(composite), 4),
            grade=_grade(composite),
        )

    # ------------------------------------------------------------------
    # Step 7 — contract compilation
    # ------------------------------------------------------------------

    def _build_contract(
        self,
        idx: int,
        cand: EventCandidate,
        ic: ICResult,
        ev: EventStats,
        regime: RegimeAnalysis,
        score: AlphaScore,
        fdr_ok: bool,
    ) -> AlphaContract:
        """Assemble the AlphaContract and decide promotion."""
        cfg = self.config
        th = cfg.thresholds
        tgt = cfg.target

        reasons: List[str] = []
        if not ic.admitted:
            reasons.append(
                f"IC below admission (|IC|<{th.ic_min_abs} and p>{th.ic_max_p})"
            )
        if not (np.isfinite(ev.lift) and ev.lift >= th.min_lift):
            reasons.append(f"lift {ev.lift:.4f} < {th.min_lift}")
        if not (np.isfinite(ev.cohens_d) and ev.cohens_d >= th.min_cohens_d):
            reasons.append(f"cohens_d {ev.cohens_d:.4f} < {th.min_cohens_d}")
        if ev.n_activations < th.min_activations:
            reasons.append(f"n_activations {ev.n_activations} < {th.min_activations}")

        if th.use_fdr:
            if not fdr_ok:
                reasons.append(f"not significant under BH FDR (q={th.fdr_q})")
        else:
            if not (np.isfinite(ev.p_value) and ev.p_value < th.max_p_value):
                reasons.append(f"p_value {ev.p_value:.6f} >= {th.max_p_value}")

        promoted = not reasons
        ms = self.market_structure

        return AlphaContract(
            alpha_id=self._alpha_id(idx),
            version="1.0",
            discovery_date=self._discovery_date(),
            status="HYPOTHESIS" if promoted else "REJECTED",
            asset=cfg.asset,
            exchange=cfg.exchange,
            timeframe=cfg.timeframe,
            direction=tgt.direction,
            fee_per_side=cfg.fee_per_side,
            event_candidate_id=cand.event_id,
            event_expression=cand.expression,
            pattern_family=self._pattern_family(ms),
            target_definition=tgt,
            base_rate=self.base_rate,
            underlying_feature=ic,
            event_stats=ev,
            market_structure=ms,
            regime_analysis=regime,
            alpha_score=score,
            promoted=promoted,
            rejection_reasons=reasons,
            fdr_promoted=bool(fdr_ok),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_frame(self, kpi_table: pd.DataFrame) -> pd.DataFrame:
        """Return a chronologically-sorted copy with a DatetimeIndex.

        Accepts a DatetimeIndex or a ``timestamp_col`` holding datetimes,
        numeric Unix timestamps (unit auto-inferred), or ISO strings — the same
        contract as Event Discovery.  The timestamp column is dropped once the
        index is set so feature reconstruction sees only native data columns.
        """
        df = kpi_table.copy()
        ts_col = self.config.timestamp_col

        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()

        if ts_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{ts_col}' not found and index is not a "
                "DatetimeIndex. Set AlphaConfig(timestamp_col='...')."
            )

        raw = df[ts_col]
        if pd.api.types.is_datetime64_any_dtype(raw):
            parsed = pd.to_datetime(raw)
        elif pd.api.types.is_numeric_dtype(raw):
            parsed = pd.to_datetime(raw, unit=_infer_timestamp_unit(raw))
        else:
            parsed = pd.to_datetime(raw)

        df = df.drop(columns=[ts_col])
        df.index = pd.DatetimeIndex(parsed.to_numpy(), name=ts_col)
        return df.sort_index()

    def _event_series(self, cand: EventCandidate) -> pd.Series:
        """Return the candidate's boolean activation series, aligned to the frame.

        Alpha Discovery does **not** recompute events — the activation series
        produced by Event Discovery (``EventCandidate.event_series``, carrying
        its DatetimeIndex) is used as-is, reindexed onto the KPI frame.  Bars
        outside the candidate's discovery period align to NaN and are treated
        as inactive downstream.

        ``EventCandidate.apply()`` is used only as a fallback for candidates
        that were serialised without their series (``event_series is None``);
        even then it is a deterministic replay of the stored thresholds and
        windows, never a re-fit.
        """
        if cand.event_series is not None:
            return cand.event_series.reindex(self._frame.index)
        return cand.apply(self._frame)

    def _feature_series(self, comp) -> pd.Series:
        """Return the component's underlying continuous feature.

        Reads the column straight from the frame when it is already there —
        native features always are, and derived features (``ratio_…``,
        ``spread_…``) are too when the caller passes Event Discovery's
        post-pipeline table (``ed.df``).  Only when the column is absent is
        the feature replayed from the component's stored parameters
        (deterministic algebra on native columns — no re-fitting).
        """
        name = comp.source_feature
        if name in self._frame.columns:
            return self._frame[name]
        return build_feature_series(comp, self._frame)

    def _regime_series(self) -> Optional[pd.Series]:
        """Return the (optionally stability-filtered) regime series, or None."""
        cfg = self.config
        if cfg.regime_col not in self._frame.columns:
            return None
        regime = self._frame[cfg.regime_col]
        if cfg.use_stable_regime_only and cfg.regime_stable_col in self._frame.columns:
            stable = self._frame[cfg.regime_stable_col].astype(bool)
            regime = regime.where(stable)
        return regime

    def _rolling_ic_window(self) -> int:
        """Resolve the rolling-IC window in bars (default ≈ 60 days)."""
        cfg = self.config
        if cfg.rolling_ic_window is not None:
            return int(cfg.rolling_ic_window)
        bpd = cfg.bars_per_day or self._infer_bars_per_day()
        return max(2, int(round(60 * bpd)))

    def _infer_bars_per_day(self) -> float:
        """Infer bars-per-day from the median index spacing (fallback 24)."""
        idx = self._frame.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
            delta = pd.Series(idx).diff().dt.total_seconds().median()
            if delta and delta > 0:
                return 86400.0 / float(delta)
        return 24.0

    def _alpha_id(self, idx: int) -> str:
        cfg = self.config
        stamp = self._discovery_date().replace("-", "")[2:]
        return f"ALPHA-{cfg.asset}-{cfg.timeframe}-{stamp}-{idx:03d}"

    def _discovery_date(self) -> str:
        return self.config.discovery_date or date.today().isoformat()

    @staticmethod
    def _pattern_family(ms: Optional[MarketStructure]) -> str:
        if ms is None or ms.expected_family == "none":
            return "unspecified"
        return ms.expected_family


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    """Clamp a value to the closed unit interval [0, 1]."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _grade(score: float) -> str:
    """Map a composite score to a letter grade (doc Section 6.2)."""
    if score >= 0.75:
        return "A"
    if score >= 0.60:
        return "B+"
    if score >= 0.45:
        return "B"
    return "C"
