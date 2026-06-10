"""Main AlphaDiscovery orchestrator — FORGE pipeline Modulo 2.

Consumes the ``EventCandidate`` list produced by Event Discovery, reconstructs
each event (and its underlying continuous feature) on the KPI table, **derives
the economic target per event from the data**, validates it out-of-sample, and
compiles an ``AlphaContract`` per candidate.

Alpha Discovery takes no economic parameters as input.  For every candidate it
scans ``AlphaConfig.horizon_grid`` on the in-sample window (the first
``train_ratio`` of the table) and derives:

* ``holding_period_h`` — the horizon with the maximum |t-stat| separation
  between active-bar and inactive-bar forward returns;
* ``sell_pct``         — the mean forward return realised when the event is
  active at that horizon (its magnitude);
* ``direction``        — the sign of that mean advantage.

Every in-sample measure (IC, win rate, lift, Cohen's d, regime sensitivity)
is computed at the derived target.  The held-out temporal tail then replays
the derived target out-of-sample: promotion requires the OOS window to
confirm the advantage.

The flow is strictly sequential and read-only: regimes come from Market
Context, events come from Event Discovery — Alpha Discovery recomputes
neither.  It reads each candidate's stored ``event_series``, reads the
``regime`` column from the enriched table, and reads the underlying feature
columns from Event Discovery's post-pipeline table.

Usage
-----
    from forgedge import EventDiscovery, MarketContext, AlphaDiscovery, AlphaConfig

    enriched = MarketContext(kpi).run()          # Modulo 0 — adds 'regime'
    ed = EventDiscovery(enriched)
    candidates = ed.run()                        # Modulo 1 — Event Candidates

    ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="ADAUSDC"))
    contracts = ad.run()                # all evaluated contracts
    promoted = ad.promoted_contracts()  # status == "HYPOTHESIS"
"""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Tuple

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
    DerivedTarget,
    EventStats,
    ICResult,
    MarketStructure,
    OOSValidation,
    RegimeAnalysis,
    RegimeStat,
)
from .target import forward_returns


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
        Horizon grid, IS/OOS split and gates.  Defaults derive the target
        over horizons 1–48 bars with a 70/30 temporal split.
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
        self.market_structure: Optional[MarketStructure] = None
        self.split_idx: Optional[int] = None

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
        horizons = list(cfg.horizon_grid)

        close = self._frame[cfg.close_col].astype(float)
        n = len(close)
        split = int(round(n * cfg.train_ratio))
        split = min(max(split, 0), n)
        self.split_idx = split

        fwd = forward_returns(close, horizons)
        F = fwd.to_numpy()  # n × k, long convention

        # In-sample sufficient statistics for the vectorised horizon scan.
        F_is = F[:split]
        valid_is = np.isfinite(F_is)
        F0 = np.where(valid_is, F_is, 0.0)
        F0sq = F0 * F0
        cnt_t = valid_is.sum(axis=0).astype(float)
        sum_t = F0.sum(axis=0)
        sumsq_t = F0sq.sum(axis=0)

        # Step 2 — interpretive context, on the in-sample window only.
        med_h = horizons[len(horizons) // 2]
        self.market_structure = analyse_market_structure(
            close.iloc[:split], fwd[med_h].iloc[:split]
        )

        regime = self._regime_series()
        regime_is = regime.iloc[:split] if regime is not None else None
        ic_window = self._rolling_ic_window()

        # Per-run caches keyed on shared inputs: many candidates share the
        # same underlying feature and derived horizon.
        ic_cache: Dict[Tuple[str, int], ICResult] = {}
        regime_ic_cache: Dict[Tuple[str, int, str], Tuple[float, float, int]] = {}
        fwd_ext_cache: Dict[Tuple[int, str], pd.Series] = {}

        # ── Per-candidate derivation + measurement ───────────────────────
        measured = []
        for cand in self.event_candidates:
            event = self._event_series(cand)
            active = event.fillna(0).astype(bool).to_numpy()
            comp0 = cand.components[0]

            derived = self._derive_target(
                active[:split], valid_is, F0, F0sq, cnt_t, sum_t, sumsq_t, horizons
            )
            h_star = derived.holding_period_h
            j_star = horizons.index(h_star)

            feature = self._feature_series(comp0).astype(float)
            ic_res = self._measure_ic_cached(
                ic_cache, feature.iloc[:split], fwd[h_star].iloc[:split],
                comp0.source_feature, h_star, ic_window,
            )

            target_binary, base_rate_is = self._binary_target_cached(
                fwd_ext_cache, close, derived, split
            )

            ev_stats = self._measure_event(
                active[:split], F[:split, j_star], target_binary,
                base_rate_is, derived, is_window=True, split=split,
            )
            regime_res = self._measure_regimes(
                regime_ic_cache, active[:split],
                feature.iloc[:split], fwd[h_star].iloc[:split],
                target_binary.iloc[:split] if target_binary is not None else None,
                regime_is, comp0.source_feature, h_star,
            )
            oos_res = self._validate_oos(
                active, F[:, j_star], target_binary, derived, split, n
            )
            measured.append((cand, derived, ic_res, ev_stats, regime_res, oos_res))

        # ── Multiple-testing control across all candidates (Section 13) ──
        p_values = [m[3].p_value for m in measured]
        fdr_mask = stats.benjamini_hochberg(p_values, cfg.thresholds.fdr_q)

        # ── Compile contracts ────────────────────────────────────────────
        contracts: List[AlphaContract] = []
        for idx, ((cand, derived, ic_res, ev_stats, regime_res, oos_res), fdr_ok) in (
            enumerate(zip(measured, fdr_mask))
        ):
            score = self._score(ic_res, ev_stats, regime_res)
            contract = self._build_contract(
                idx, cand, derived, ic_res, ev_stats, regime_res, oos_res,
                score, bool(fdr_ok),
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
            "pattern_family", "holding_period_h", "sell_pct", "direction",
            "mean_advantage", "feature", "ic", "ic_p_value", "ic_admitted",
            "rolling_ic_stable", "n_activations", "win_rate", "base_rate", "lift",
            "fwd_return_mean", "cohens_d", "t_stat", "p_value", "fdr_promoted",
            "oos_passed", "oos_p_value", "oos_lift",
            "regime_dependency", "regime_breadth", "composite_score", "grade",
            "rejection_reasons",
        ]
        if not self._contracts:
            return pd.DataFrame(columns=_cols)
        df = pd.DataFrame([c.to_dict() for c in self._contracts])
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 1 — per-event target derivation (in-sample)
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_target(
        active_is: np.ndarray,
        valid_is: np.ndarray,
        F0: np.ndarray,
        F0sq: np.ndarray,
        cnt_t: np.ndarray,
        sum_t: np.ndarray,
        sumsq_t: np.ndarray,
        horizons: List[int],
    ) -> DerivedTarget:
        """Scan the horizon grid and derive ``(h*, sell_pct*, direction*)``.

        For every horizon the pooled two-sample t-stat of active vs inactive
        forward returns is computed from in-sample sufficient statistics
        (vectorised across the grid).  ``h*`` maximises |t|; the mean active
        return at ``h*`` provides the candidate ``sell_pct`` (its magnitude)
        and the direction (its sign).
        """
        m_f = active_is.astype(float)
        cnt_a = m_f @ valid_is
        sum_a = m_f @ F0
        sumsq_a = m_f @ F0sq

        cnt_b = cnt_t - cnt_a
        sum_b = sum_t - sum_a
        sumsq_b = sumsq_t - sumsq_a

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_a = np.where(cnt_a > 0, sum_a / np.maximum(cnt_a, 1), np.nan)
            mean_b = np.where(cnt_b > 0, sum_b / np.maximum(cnt_b, 1), np.nan)
            var_a = (sumsq_a - cnt_a * mean_a ** 2) / np.maximum(cnt_a - 1, 1)
            var_b = (sumsq_b - cnt_b * mean_b ** 2) / np.maximum(cnt_b - 1, 1)

            dof = cnt_a + cnt_b - 2
            sp2 = ((cnt_a - 1) * var_a + (cnt_b - 1) * var_b) / np.maximum(dof, 1)
            denom = np.sqrt(sp2 * (1.0 / np.maximum(cnt_a, 1) + 1.0 / np.maximum(cnt_b, 1)))
            t = (mean_a - mean_b) / denom

        usable = (cnt_a >= 2) & (cnt_b >= 2) & (dof > 0) & np.isfinite(t)
        t = np.where(usable, t, np.nan)

        advantage_by_h = {h: float(mean_a[j]) for j, h in enumerate(horizons)}
        t_stat_by_h = {h: float(t[j]) for j, h in enumerate(horizons)}

        if np.isfinite(t).any():
            j_star = int(np.nanargmax(np.abs(t)))
        else:
            j_star = 0

        adv = mean_a[j_star]
        if not np.isfinite(adv) or adv == 0.0:
            return DerivedTarget(
                holding_period_h=horizons[j_star],
                sell_pct=float("nan"),
                direction="undetermined",
                mean_advantage=float("nan"),
                advantage_by_h=advantage_by_h,
                t_stat_by_h=t_stat_by_h,
            )

        return DerivedTarget(
            holding_period_h=horizons[j_star],
            sell_pct=float(abs(adv)),
            direction="long" if adv > 0 else "short",
            mean_advantage=float(adv),
            advantage_by_h=advantage_by_h,
            t_stat_by_h=t_stat_by_h,
        )

    def _binary_target_cached(
        self,
        fwd_ext_cache: Dict[Tuple[int, str], pd.Series],
        close: pd.Series,
        derived: DerivedTarget,
        split: int,
    ) -> Tuple[Optional[pd.Series], float]:
        """Binary target at the derived parameters, with the rolling forward
        extreme cached per ``(h, direction)`` (``sell_pct`` varies per
        candidate but the extreme does not)."""
        if derived.direction not in ("long", "short"):
            return None, float("nan")

        h = derived.holding_period_h
        key = (h, derived.direction)
        if key not in fwd_ext_cache:
            if derived.direction == "long":
                ext = close.rolling(h, min_periods=h).max().shift(-h)
            else:
                ext = close.rolling(h, min_periods=h).min().shift(-h)
            fwd_ext_cache[key] = ext
        ext = fwd_ext_cache[key]

        if derived.direction == "long":
            hit = ext / close - 1.0 >= derived.sell_pct
        else:
            hit = ext / close - 1.0 <= -derived.sell_pct
        target = hit.astype(float).where(ext.notna(), np.nan)

        base_rate_is = float(target.iloc[:split].mean())
        return target, base_rate_is

    # ------------------------------------------------------------------
    # Step 3 — IC measurement (in-sample, at the derived horizon)
    # ------------------------------------------------------------------

    def _measure_ic_cached(
        self,
        cache: Dict[Tuple[str, int], ICResult],
        feature_is: pd.Series,
        fwd_is: pd.Series,
        feature_name: str,
        h: int,
        ic_window: int,
    ) -> ICResult:
        """Spearman IC of the underlying feature vs the forward return at the
        derived horizon, cached per ``(feature, horizon)`` — candidates that
        share both share the result."""
        key = (feature_name, h)
        if key in cache:
            return cache[key]

        ic, p = stats.spearmanr(feature_is.to_numpy(), fwd_is.to_numpy())
        n = int((feature_is.notna() & fwd_is.notna()).sum())

        stable, roll_mean, sign_consistency = self._rolling_ic_stability(
            feature_is, fwd_is, ic, ic_window
        )

        th = self.config.thresholds
        weak_ic = (not np.isfinite(ic)) or abs(ic) < th.ic_min_abs
        weak_p = (not np.isfinite(p)) or p > th.ic_max_p
        admitted = not (weak_ic and weak_p)

        result = ICResult(
            feature=feature_name,
            ic=float(ic) if np.isfinite(ic) else float("nan"),
            p_value=float(p) if np.isfinite(p) else float("nan"),
            n=n,
            rolling_ic_stable=stable,
            rolling_ic_mean=roll_mean,
            rolling_sign_consistency=sign_consistency,
            admitted=admitted,
        )
        cache[key] = result
        return result

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
    # Step 4 — win rate analysis (in-sample, at the derived target)
    # ------------------------------------------------------------------

    def _measure_event(
        self,
        active_is: np.ndarray,
        fwd_is: np.ndarray,
        target_binary: Optional[pd.Series],
        base_rate_is: float,
        derived: DerivedTarget,
        is_window: bool,
        split: int,
    ) -> EventStats:
        """Measure the binary event's predictive power at the derived target.

        Forward returns are *oriented* (sign-flipped for shorts) so that
        "favourable to the trade" is always positive — Cohen's d and the
        one-sided t-test then read the same way for both directions.
        """
        if derived.direction not in ("long", "short") or target_binary is None:
            n_act = int(active_is.sum())
            nan = float("nan")
            return EventStats(n_act, nan, base_rate_is, nan, nan, nan, nan, nan)

        orient = 1.0 if derived.direction == "long" else -1.0
        r = orient * fwd_is

        tgt_is = target_binary.iloc[:split].to_numpy()
        tgt_valid = np.isfinite(tgt_is)

        active_valid = active_is & tgt_valid
        inactive_valid = (~active_is) & tgt_valid

        n_act = int(active_valid.sum())
        wr = float(tgt_is[active_valid].mean()) if n_act else float("nan")
        lift = wr - base_rate_is if np.isfinite(wr) else float("nan")

        active_ret = r[active_is & np.isfinite(r)]
        inactive_ret = r[(~active_is) & np.isfinite(r)]

        fwd_mean = float(active_ret.mean()) if active_ret.size else float("nan")
        d = stats.cohens_d(active_ret, inactive_ret)
        t, p = stats.ttest_ind(active_ret, inactive_ret, alternative="greater")

        return EventStats(
            n_activations=n_act,
            win_rate=wr,
            base_rate=base_rate_is,
            lift=lift,
            fwd_return_mean=fwd_mean,
            cohens_d=d,
            t_stat=t,
            p_value=p,
        )

    # ------------------------------------------------------------------
    # Step 5 — regime sensitivity (in-sample)
    # ------------------------------------------------------------------

    def _measure_regimes(
        self,
        ic_cache: Dict[Tuple[str, int, str], Tuple[float, float, int]],
        active_is: np.ndarray,
        feature_is: pd.Series,
        fwd_is: pd.Series,
        target_is: Optional[pd.Series],
        regime_is: Optional[pd.Series],
        feature_name: str,
        h: int,
    ) -> RegimeAnalysis:
        """Per-regime IC and win rate, plus the dependency classification."""
        if regime_is is None:
            return RegimeAnalysis([], "unknown", [], [], float("nan"))

        cfg = self.config
        tgt = target_is.to_numpy() if target_is is not None else None
        tgt_valid = np.isfinite(tgt) if tgt is not None else None

        labels = list(regime_is.cat.categories) if hasattr(regime_is, "cat") else \
            list(pd.unique(regime_is.dropna()))

        per_regime: List[RegimeStat] = []
        evaluated = 0
        significant: List[str] = []
        weak: List[str] = []

        for label in labels:
            rmask = (regime_is == label).to_numpy()
            key = (feature_name, h, str(label))
            if key not in ic_cache:
                n_obs = int(
                    (rmask & feature_is.notna().to_numpy() & fwd_is.notna().to_numpy()).sum()
                )
                if n_obs < cfg.min_regime_obs:
                    ic_cache[key] = (float("nan"), float("nan"), n_obs)
                else:
                    ic_r, p_r = stats.spearmanr(
                        feature_is.to_numpy()[rmask], fwd_is.to_numpy()[rmask]
                    )
                    ic_cache[key] = (float(ic_r), float(p_r), n_obs)
            ic_r, p_r, n_obs = ic_cache[key]

            if n_obs < cfg.min_regime_obs:
                per_regime.append(
                    RegimeStat(str(label), n_obs, float("nan"), float("nan"),
                               float("nan"), "insufficient")
                )
                continue

            if tgt is not None:
                ev_in = active_is & rmask & tgt_valid
                wr_r = float(tgt[ev_in].mean()) if int(ev_in.sum()) else float("nan")
            else:
                wr_r = float("nan")

            strength = self._regime_strength(ic_r, p_r)
            per_regime.append(
                RegimeStat(str(label), n_obs, ic_r, p_r, wr_r, strength)
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
    # Out-of-sample confirmation of the derived target
    # ------------------------------------------------------------------

    def _validate_oos(
        self,
        active: np.ndarray,
        fwd_h: np.ndarray,
        target_binary: Optional[pd.Series],
        derived: DerivedTarget,
        split: int,
        n: int,
    ) -> Optional[OOSValidation]:
        """Replay the derived target on the held-out tail.

        The OOS window played no part in the derivation or in any in-sample
        measure; the derived ``(h*, sell_pct*, direction*)`` must confirm the
        oriented advantage here for the candidate to be promotable.
        """
        n_oos = n - split
        if n_oos <= 0:
            return None

        th = self.config.thresholds
        nan = float("nan")

        if derived.direction not in ("long", "short") or target_binary is None:
            return OOSValidation(n_oos, 0, nan, nan, nan, nan, nan, nan, False)

        orient = 1.0 if derived.direction == "long" else -1.0
        r = orient * fwd_h[split:]
        act = active[split:]

        tgt = target_binary.iloc[split:].to_numpy()
        tgt_valid = np.isfinite(tgt)
        n_act = int((act & tgt_valid).sum())

        base = float(tgt[tgt_valid].mean()) if tgt_valid.any() else nan
        wr = float(tgt[act & tgt_valid].mean()) if n_act else nan
        lift = wr - base if np.isfinite(wr) and np.isfinite(base) else nan

        act_ret = r[act & np.isfinite(r)]
        inact_ret = r[(~act) & np.isfinite(r)]
        mean_adv = float(act_ret.mean()) if act_ret.size else nan
        t, p = stats.ttest_ind(act_ret, inact_ret, alternative="greater")

        passed = (
            n_act >= th.min_oos_activations
            and np.isfinite(mean_adv) and mean_adv > 0
            and math.isfinite(p) and p < th.oos_max_p
        )

        return OOSValidation(
            n_bars=n_oos,
            n_activations=n_act,
            mean_advantage=mean_adv,
            t_stat=t,
            p_value=p,
            win_rate=wr,
            base_rate=base,
            lift=lift,
            passed=bool(passed),
        )

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
        derived: DerivedTarget,
        ic: ICResult,
        ev: EventStats,
        regime: RegimeAnalysis,
        oos: Optional[OOSValidation],
        score: AlphaScore,
        fdr_ok: bool,
    ) -> AlphaContract:
        """Assemble the AlphaContract and decide promotion."""
        cfg = self.config
        th = cfg.thresholds

        reasons: List[str] = []
        if derived.direction not in ("long", "short"):
            reasons.append("no derivable target (no finite advantage on the grid)")
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

        if oos is not None and not oos.passed:
            reasons.append(
                f"derived target not confirmed OOS "
                f"(p={oos.p_value:.4f} vs {th.oos_max_p}, "
                f"mean_adv={oos.mean_advantage:.5f}, n_act={oos.n_activations})"
            )

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
            direction=derived.direction,
            fee_per_side=cfg.fee_per_side,
            event_candidate_id=cand.event_id,
            event_expression=cand.expression,
            pattern_family=self._pattern_family(ms),
            derived_target=derived,
            base_rate=ev.base_rate,
            underlying_feature=ic,
            event_stats=ev,
            market_structure=ms,
            regime_analysis=regime,
            alpha_score=score,
            oos_validation=oos,
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
