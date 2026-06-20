"""Main AlphaDiscovery orchestrator — FORGE pipeline Modulo 2.

Consumes the ``EventCandidate`` list produced by Event Discovery, reconstructs
each event (and its underlying continuous feature) on the KPI table, **derives
the economic target per event from the data**, validates it out-of-sample, and
compiles an ``AlphaContract`` per candidate.

Alpha Discovery takes no economic parameters as input.  For every candidate it
scans ``AlphaConfig.horizon_grid`` on the in-sample window (the first
``train_ratio`` of the table) and derives:

* ``holding_period_h`` — the horizon with the maximum ``|z_h|``, the **excess
  log-return** ``Δ_h = μ_cond_h − μ_base_h`` standardised by a circular-rotation
  null (autocorrelation-robust, so clustered events don't pin ``h*`` to the
  grid edge);
* ``sell_pct``         — the ``q``-quantile of the Maximum Favorable Excursion
  (MFE) at ``h*`` across active IS bars;
* ``direction``        — the sign of the excess log-return ``Δ_h*``.  Working
  in log-space excess (conditional minus unconditional) strips the asset's
  drift, so a rule reads *short* in a bull run and *long* in a flat market
  from the data alone — never imposed a priori.

Every in-sample measure (IC, win rate, lift, Cohen's d, regime sensitivity)
is computed at the derived target.  The held-out temporal tail replays the
derived target out-of-sample, recording the OOS advantage.  All directed
contracts receive a grade (A–D) and are handed to Rule Discovery; statistical
measures inform the grade but do not gate promotion.

The flow is strictly sequential: regimes come from Market Context, events come
from Event Discovery — Alpha Discovery re-derives neither.  It evaluates each
candidate as an activation function on the candles it observes (a deterministic
replay of the event's frozen parameters, reusing the cached ``event_series``
verbatim when the candle set is identical), reads the ``regime`` column from the
enriched table, and reads the underlying feature columns from Event Discovery's
post-pipeline table.

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
import warnings
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
from .target import forward_log_returns, forward_returns


# Non-parametrizable floor: below this count the t-test is numerically
# unreliable regardless of the p-value.  Not a hard gate — triggers a
# diagnostic warning only.
_MIN_STATS_CASES: int = 10


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
        Output of ``EventDiscovery.run()``.  Each candidate is evaluated as an
        activation function on the observed candles — events are re-evaluated,
        never re-derived (the cached ``event_series`` is reused verbatim only
        when the candle set is identical).  See :meth:`_event_series`.
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
        self._mismatch_warned = False

        # Populated by run().
        self.market_structure: Optional[MarketStructure] = None
        self.split_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> List[AlphaContract]:
        """Evaluate every candidate and return one contract each.

        Returns the **full** list.  All directed contracts have status
        ``HYPOTHESIS`` and ``promoted=True``; undetermined-direction contracts
        are ``REJECTED``.  Use :meth:`promoted_contracts` for the directed
        subset handed to Rule Discovery, and :attr:`AlphaContract.alpha_score`
        for the A–D grade.
        """
        cfg = self.config
        horizons = list(cfg.horizon_grid)

        close = self._frame[cfg.close_col].astype(float)
        n = len(close)
        split = int(round(n * cfg.train_ratio))
        split = min(max(split, 0), n)
        self.split_idx = split

        fwd = forward_returns(close, horizons)
        F = fwd.to_numpy()  # n × k, simple, long convention (IC, MFE, binary)

        # Direction/horizon derivation runs in log-space: the excess log-return
        # Δ_h = μ_cond_h − μ_base_h is what carries the signal, robust to the
        # extreme outliers of crypto forward returns.
        logfwd = forward_log_returns(close, horizons)
        L = logfwd.to_numpy()  # n × k, log, long convention

        # In-sample arrays for the vectorised horizon scan, in log-space.
        # ``valid_is`` is the validity mask shared by every measure (a bar is
        # valid at horizon h when both the simple and log forward return are
        # finite — they share the same off-the-end / non-positive close gaps).
        F_is = F[:split]
        L_is = L[:split]
        valid_is = np.isfinite(F_is) & np.isfinite(L_is)
        L0 = np.where(valid_is, L_is, 0.0)
        cnt_t = valid_is.sum(axis=0).astype(float)
        sum_t = L0.sum(axis=0)

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
                active[:split], valid_is, L0, cnt_t, sum_t, horizons,
                close.iloc[:split].to_numpy(), cfg.mfe_quantile, cfg.mfe_floor,
                fdr_q=cfg.thresholds.fdr_q, min_direction_t=cfg.thresholds.min_direction_t,
                require_significant=cfg.thresholds.require_significant_direction,
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
                active, L[:, j_star], target_binary, derived, split, n
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
            score = self._score(ic_res, ev_stats, regime_res, derived, oos_res)
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
        L0: np.ndarray,
        cnt_t: np.ndarray,
        sum_t: np.ndarray,
        horizons: List[int],
        close_is: Optional[np.ndarray] = None,
        mfe_quantile: float = 0.5,
        mfe_floor: float = 0.005,
        fdr_q: float = 0.10,
        min_direction_t: float = 0.0,
        require_significant: bool = True,
    ) -> DerivedTarget:
        """Scan the horizon grid and derive ``(h*, sell_pct*, direction*)``.

        The signal lives in the **excess log-return**
        ``Δ_h = μ_cond_h − μ_base_h`` — the mean active-bar log-return minus the
        *unconditional* baseline over all valid bars.  Subtracting the baseline
        strips the asset's drift, so the direction reflects the event's own edge
        rather than the prevailing trend (``sign(Δ_h)`` — never imposed).

        ``h*`` maximises ``|z_h|``, the excess standardised by a **circular-
        rotation null** (see :meth:`_rotation_null`).  A naive
        ``T_h = Δ_h / (σ_cond / √n)`` treats the overlapping forward-return
        windows as independent: its denominator shrinks with the horizon, so for
        a *clustered* event (activations in runs — the common case) ``|T_h|`` is
        inflated on long horizons and ``h*`` pins to the long edge of the grid
        even with no real edge.  Standardising by a null that re-uses the
        event's own activation pattern (rotated against the returns) removes
        that bias.

        The rotation null yields ``p_value_by_h``; Benjamini-Hochberg at
        ``fdr_q`` yields ``h_sig`` and the ``statistically_weak`` flag (``h*``
        outside ``h_sig``).  ``h*`` is always chosen on ``|z_h|`` over the whole
        grid, but the **direction** is gated: when ``require_significant`` is
        set (default) and no horizon clears BH, the excess is indistinguishable
        from the null everywhere, so ``direction`` is ``"undetermined"`` rather
        than a coin-flip read off ``argmax|z_h|``.

        ``sell_pct`` is the ``mfe_quantile``-quantile of the Maximum Favorable
        Excursion at ``h*`` across active IS bars.  ``direction`` is
        ``"undetermined"`` when no horizon yields a finite ``Δ_h``, when
        ``|z_h*| < min_direction_t``, or (with ``require_significant``) when the
        target is ``statistically_weak``.
        """
        m_f = active_is.astype(float)
        cnt_a = m_f @ valid_is                 # active bars valid at each horizon
        sum_a = m_f @ L0                       # Σ log-return over active bars

        with np.errstate(divide="ignore", invalid="ignore"):
            mu_cond = np.where(cnt_a > 0, sum_a / np.maximum(cnt_a, 1), np.nan)
            mu_base = np.where(cnt_t > 0, sum_t / np.maximum(cnt_t, 1), np.nan)
            delta = mu_cond - mu_base          # Δ_h — excess log-return (signed)

        # Circular-rotation null: z_h = Δ_h / σ_null,h and the two-sided p-value.
        z, p_rot = AlphaDiscovery._rotation_null(
            active_is, valid_is, L0, mu_base, delta,
        )
        usable = (cnt_a >= 2) & np.isfinite(delta) & np.isfinite(z)
        z = np.where(usable, z, np.nan)
        p_rot = np.where(usable, p_rot, np.nan)
        score = np.abs(z)                       # h* = argmax |z_h|

        sig_mask = stats.benjamini_hochberg(p_rot, fdr_q)
        h_sig = tuple(int(horizons[j]) for j in range(len(horizons)) if sig_mask[j])

        advantage_by_h = {h: float(delta[j]) for j, h in enumerate(horizons)}
        t_stat_by_h = {h: float(z[j]) for j, h in enumerate(horizons)}
        score_by_h = {h: float(score[j]) for j, h in enumerate(horizons)}
        p_value_by_h = {h: float(p_rot[j]) for j, h in enumerate(horizons)}

        # h* on |z_h| over H_sig when non-empty, else over the whole grid (the
        # empty-gate case is surfaced via statistically_weak, not discarded).
        sel = score.copy()
        if sig_mask.any():
            sel = np.where(sig_mask, sel, np.nan)
        if np.isfinite(sel).any():
            j_star = int(np.nanargmax(np.where(np.isfinite(sel), sel, np.nan)))
        elif np.isfinite(score).any():
            j_star = int(np.nanargmax(np.where(np.isfinite(score), score, np.nan)))
        else:
            j_star = 0

        h_star = horizons[j_star]
        statistically_weak = h_star not in h_sig

        adv = delta[j_star]
        z_star = z[j_star]
        undetermined = (
            not np.isfinite(adv) or adv == 0.0
            or not np.isfinite(z_star) or abs(z_star) < min_direction_t
            # No BH-significant horizon: the excess is indistinguishable from the
            # rotation null everywhere, so argmax|z| would assign a direction off
            # a coin-flip (often the drift-driven long edge of the grid).
            or (require_significant and statistically_weak)
        )
        if undetermined:
            return DerivedTarget(
                holding_period_h=h_star,
                sell_pct=float("nan"),
                direction="undetermined",
                mean_advantage=float("nan"),
                advantage_by_h=advantage_by_h,
                t_stat_by_h=t_stat_by_h,
                score_by_h=score_by_h,
                p_value_by_h=p_value_by_h,
                h_sig=h_sig,
                statistically_weak=statistically_weak,
            )

        direction = "long" if adv > 0 else "short"
        sell_pct = (
            AlphaDiscovery._compute_mfe_quantile(
                close_is, active_is, h_star, direction, mfe_quantile, mfe_floor,
            )
            if close_is is not None
            else max(float(abs(adv)), mfe_floor)
        )

        return DerivedTarget(
            holding_period_h=h_star,
            sell_pct=sell_pct,
            direction=direction,
            mean_advantage=float(adv),
            advantage_by_h=advantage_by_h,
            t_stat_by_h=t_stat_by_h,
            score_by_h=score_by_h,
            p_value_by_h=p_value_by_h,
            h_sig=h_sig,
            statistically_weak=statistically_weak,
        )

    @staticmethod
    def _rotation_null(
        active_is: np.ndarray,
        valid_is: np.ndarray,
        L0: np.ndarray,
        mu_base: np.ndarray,
        delta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Circular-rotation null for the excess log-return, per horizon.

        Every non-trivial circular shift ``k`` of the event mask is a draw under
        the null "the event timing is unrelated to the returns": the rotated
        mask keeps the event's exact activation *clustering* (so overlapping
        forward windows are reproduced) while decoupling it from the returns.
        For each horizon the rotated excess is ``Δ_h^{(k)} = mean(L0 over the
        shifted active∩valid bars) − μ_base``; from the distribution over all
        shifts we read

        * ``σ_null,h`` — its standard deviation, the honest scale for ``z_h =
          Δ_h / σ_null,h`` (autocorrelation-robust: it widens on long horizons
          for clustered events exactly where a naive ``σ / √n`` would not);
        * ``p_h`` — ``(1 + #{|Δ^{(k)}| ≥ |Δ_h|}) / (1 + #shifts)``, a valid
          two-sided permutation p-value.

        All ``n−1`` shifts are evaluated at once via the circular cross-
        correlation (FFT), so the exact test costs ``O(n log n · k)`` — no Monte
        Carlo, no seed.  Returns ``(z, p)`` with ``nan`` where a horizon has no
        usable null.
        """
        n, k = L0.shape
        z = np.full(k, np.nan)
        p = np.full(k, np.nan)
        if n < 4:
            return z, p

        af = active_is.astype(float)
        vf = valid_is.astype(float)
        # All circular shifts via FFT: corr[k, j] = Σ_u af[u] · X[(u+k) mod n, j].
        fa_conj = np.conj(np.fft.rfft(af))
        num = np.fft.irfft(fa_conj[:, None] * np.fft.rfft(L0, axis=0), n=n, axis=0)
        den = np.fft.irfft(fa_conj[:, None] * np.fft.rfft(vf, axis=0), n=n, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            mu_shift = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
        boot_delta = mu_shift - mu_base[None, :]      # (n, k); row 0 = observed
        null = boot_delta[1:]                          # drop the identity shift

        n_valid = np.sum(np.isfinite(null), axis=0).astype(float)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # An all-NaN column (a horizon with no usable rotation) makes nanstd
            # warn about empty d.o.f.; it correctly yields nan, which we handle.
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.nanstd(null, axis=0)
            z = np.where((sd > 0) & np.isfinite(delta), delta / sd, np.nan)
            ge = np.nansum(np.abs(null) >= np.abs(delta)[None, :], axis=0)
            p = np.where(n_valid > 0, (1.0 + ge) / (1.0 + n_valid), np.nan)
        return z, p

    @staticmethod
    def _compute_mfe_quantile(
        close_is: np.ndarray,
        active_is: np.ndarray,
        h: int,
        direction: str,
        q: float,
        floor: float,
    ) -> float:
        """``q``-quantile of MFE over active IS bars at horizon ``h``.

        MFE at bar *i* is the best oriented return achievable anywhere in
        ``[i+1 .. i+h]``:
          long:  ``max(close_is[i+1..i+h]) / close_is[i] - 1``
          short: ``1 - min(close_is[i+1..i+h]) / close_is[i]``

        Returns ``nan`` when fewer than two active bars have a complete
        forward window, or when direction is undetermined.
        """
        n = len(close_is)
        max_bars = n - h  # bars with a complete h-bar forward window
        if max_bars <= 0 or direction not in ("long", "short"):
            return float("nan")

        # future[i, k] = close_is[i + 1 + k] for k in 0..h-1
        future = np.stack(
            [close_is[1 + k : max_bars + 1 + k] for k in range(h)], axis=1
        )
        c0 = close_is[:max_bars]
        valid_c0 = (c0 > 0) & np.isfinite(c0)

        if direction == "long":
            extreme = np.nanmax(future, axis=1)
            mfe = np.where(valid_c0, extreme / c0 - 1.0, np.nan)
        else:
            extreme = np.nanmin(future, axis=1)
            mfe = np.where(valid_c0, 1.0 - extreme / c0, np.nan)

        active_mask = active_is[:max_bars] & (mfe > 0) & np.isfinite(mfe)
        active_mfe = mfe[active_mask]

        if len(active_mfe) < 2:
            return float("nan")

        return max(float(np.quantile(active_mfe, q)), floor)

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
        if derived.direction not in ("long", "short") or not np.isfinite(derived.sell_pct):
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
        oriented advantage here for the candidate to be promotable.  ``fwd_h``
        is the forward **log**-return at ``h*`` (matching the log-space
        derivation); the active-vs-inactive t-test nets out the common drift,
        so the confirmation reads the event's edge, not the trend.
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

        # p-value alone determines passed — sample size is already encoded in p.
        passed = (
            np.isfinite(mean_adv) and mean_adv > 0
            and math.isfinite(p) and p < th.oos_max_p
        )

        mde = stats.min_detectable_effect(
            int(act_ret.size), int(inact_ret.size), th.oos_max_p
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
            min_detectable_effect=mde,
        )

    # ------------------------------------------------------------------
    # Step 6 — scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        ic: ICResult,
        ev: EventStats,
        regime: RegimeAnalysis,
        derived: DerivedTarget,
        oos: Optional[OOSValidation],
    ) -> AlphaScore:
        """Composite alpha score and grade (doc Section 6.1/6.2).

        The composite is a weighted average of the signal-quality terms, then
        adjusted for statistical robustness:

        * ``cohens_d`` is normalised **signed** in ``[-1, 1]`` — a negative
          Cohen's d (the conditioned group does *worse* than the background)
          now *penalises* the score instead of being silently clipped to zero;
        * ``|z*|`` — the rotation-null standardised excess at ``h*`` (the
          edge-to-noise ratio) — enters as its own term;
        * a ``statistically_weak`` target (selected horizon outside the
          BH-significant set) has its composite multiplied by
          ``statistically_weak_penalty``, so a horizon picked by the very
          selection bias the FDR control guards against cannot rank highly;
        * a passing out-of-sample confirmation adds ``oos_bonus``.

        The result is clamped to ``[0, 1]``.
        """
        cfg = self.config
        w_ic, w_lift, w_d, w_z, w_breadth = cfg.score_weights

        ic_norm = _clamp01(abs(ic.ic) / 0.10) if np.isfinite(ic.ic) else 0.0
        lift_norm = _clamp01(ev.lift / 0.30) if np.isfinite(ev.lift) else 0.0
        # Signed: an adverse effect (cohens_d < 0) drags the score down.
        d_norm = (
            float(np.clip(ev.cohens_d / 0.80, -1.0, 1.0))
            if np.isfinite(ev.cohens_d) else 0.0
        )
        z_star = derived.t_stat_by_h.get(derived.holding_period_h, float("nan"))
        z_norm = _clamp01(abs(z_star) / 3.0) if np.isfinite(z_star) else 0.0
        breadth = regime.regime_breadth

        terms = [(w_ic, ic_norm), (w_lift, lift_norm), (w_d, d_norm), (w_z, z_norm)]
        if np.isfinite(breadth):
            terms.append((w_breadth, breadth))
        # else: no regime information — drop the breadth term and renormalise.

        total_w = sum(w for w, _ in terms)
        composite = sum(w * v for w, v in terms) / total_w if total_w else 0.0

        # Robustness adjustments: weak-horizon penalty, OOS-confirmation bonus.
        if derived.statistically_weak:
            composite *= cfg.statistically_weak_penalty
        if oos is not None and oos.passed:
            composite += cfg.oos_bonus
        composite = _clamp01(composite)

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
        """Assemble the AlphaContract and grade it (A–D).

        Only contracts with an undetermined direction are rejected.  All other
        measures (IC, lift, Cohen's d, FDR, OOS) produce non-blocking
        diagnostics that feed the grade but do not gate promotion.  Rule
        Discovery is the sole economic judge.
        """
        cfg = self.config
        th = cfg.thresholds

        # Hard gate: no actionable direction means nothing to hand off.
        diagnostics: List[str] = []
        if derived.direction not in ("long", "short"):
            diagnostics.append(
                "no derivable target (no finite advantage on the grid)"
            )

        # Non-blocking diagnostics — inform the grade, do not gate promotion.
        if not ic.admitted:
            diagnostics.append(
                f"[diagnostic] IC weak (|IC|={abs(ic.ic):.4f} < {th.ic_min_abs} "
                f"or p={ic.p_value:.4f})"
            )
        if np.isfinite(ev.lift) and ev.lift < th.min_lift:
            diagnostics.append(
                f"[diagnostic] lift {ev.lift:.4f} < {th.min_lift}"
            )
        if np.isfinite(ev.cohens_d) and ev.cohens_d < th.min_cohens_d:
            diagnostics.append(
                f"[diagnostic] cohens_d {ev.cohens_d:.4f} < {th.min_cohens_d}"
            )
        if ev.n_activations < _MIN_STATS_CASES:
            diagnostics.append(
                f"[diagnostic] IS sample too small for reliable statistics "
                f"(n_activations={ev.n_activations} < {_MIN_STATS_CASES})"
            )

        if th.use_fdr:
            if not fdr_ok:
                diagnostics.append(
                    f"[diagnostic] not significant under BH FDR (q={th.fdr_q})"
                )
        else:
            if not (np.isfinite(ev.p_value) and ev.p_value < th.max_p_value):
                diagnostics.append(
                    f"[diagnostic] p_value {ev.p_value:.6f} >= {th.max_p_value}"
                )

        if oos is not None:
            if oos.n_activations < _MIN_STATS_CASES:
                diagnostics.append(
                    f"[diagnostic] OOS sample too small for reliable statistics "
                    f"(n_oos_activations={oos.n_activations} < {_MIN_STATS_CASES})"
                )
            if (math.isfinite(oos.min_detectable_effect)
                    and np.isfinite(ev.cohens_d)
                    and oos.min_detectable_effect > abs(ev.cohens_d)):
                diagnostics.append(
                    f"[diagnostic] OOS underpowered: MDE={oos.min_detectable_effect:.2f} "
                    f"> IS Cohen's d={ev.cohens_d:.2f}; OOS cannot confirm IS effect size"
                )
            if not oos.passed:
                diagnostics.append(
                    f"[diagnostic] OOS weak (p={oos.p_value:.4f} vs {th.oos_max_p}, "
                    f"mean_adv={oos.mean_advantage:.5f}, n_act={oos.n_activations})"
                )

        # All directed contracts are promoted — Rule Discovery is the economic judge.
        promoted = derived.direction in ("long", "short")
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
            rejection_reasons=diagnostics,
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
        """Return the candidate's activation series on the **observed** candles.

        An Event Candidate *is* an activation function.  Alpha Discovery
        evaluates it on the candles it actually observes (the KPI frame) via the
        candidate's deterministic replay, :meth:`EventCandidate.apply`.  That
        replay consumes the event's frozen parameters verbatim — thresholds,
        window sizes, ``diffnorm_std``, class labels — and never re-derives any
        of them, so the "no-recompute" contract is honoured: the event is
        *re-evaluated*, not *re-fitted*.

        The pre-computed ``EventCandidate.event_series`` is a **cache** of that
        same function evaluated on the *training* candles.  It is reused as a
        transparent fast path only when its index is identical to the observed
        frame's, where the cached evaluation equals ``apply()`` exactly.  When
        the candle sets differ the cache is silently inapplicable — reindexing
        it would force every non-overlapping bar to "inactive", measuring the
        event against returns on bars where it was never evaluated — so the
        function is evaluated on the observed frame instead.

        Rolling transforms (``rolling_pctrank``, ``rolling_zscore``, ``delta``,
        crossings) depend on the bars preceding each one; when the observed
        candles differ from the training ones, activations near the start of the
        observed window reflect the history actually available there.  This is
        the honest value of a windowed feature given the observed data, not an
        inconsistency.

        When the observed frame differs from the training frame, this method
        also checks whether the re-evaluated activation count is dramatically
        lower than the stored count and emits a second diagnostic warning.  A
        near-zero count on the observed frame (< 10 % of training activations
        and fewer than 2 activations) means the event's rolling-transform
        thresholds are rarely met in the new data context — the most common
        cause is that the observed frame contains a long pre-training history
        whose higher volatility inflates the rolling-window baselines
        (pctrank, z-score) so that the thresholds calibrated on the training
        window are no longer crossed.

        Correct workflow for evaluating persisted events on additional bars
        ----------------------------------------------------------------
        Pass only the **original training data concatenated with the new bars**,
        not a broader historical dataset.  This preserves the same rolling
        baselines for the training-period bars and appends the new period at the
        end::

            # Build evaluation frame: training + new bars only
            eval_df = pd.concat([train_df, new_bars_df]).drop_duplicates('open_dt')
            ad = AlphaDiscovery(eval_df, candidates, AlphaConfig(train_ratio=1))
        """
        stored = cand.event_series
        if stored is not None and stored.index.equals(self._frame.index):
            return stored
        if stored is not None and not self._mismatch_warned:
            warnings.warn(
                "Alpha Discovery received candles whose index differs from the "
                "event's stored activation series; the event is evaluated as an "
                "activation function on the observed candles "
                "(EventCandidate.apply). Windowed transforms reflect the history "
                "available in the observed window.",
                stacklevel=2,
            )
            self._mismatch_warned = True
        replayed = cand.apply(self._frame)
        if stored is not None:
            stored_count = int(stored.fillna(0).astype(bool).sum())
            new_count = int(replayed.fillna(0).astype(bool).sum())
            if stored_count > 0 and new_count < 2 and new_count < stored_count * 0.10:
                warnings.warn(
                    f"Event '{cand.event_id}' fired {stored_count} times on the "
                    f"training frame but only {new_count} time(s) on the observed "
                    f"frame.  The rolling-transform baselines (pctrank, z-score) "
                    f"are likely shifted by pre-training history in the observed "
                    f"data, making the training-calibrated thresholds very hard to "
                    f"cross.  This will cause AlphaDiscovery to return "
                    f"direction='undetermined' (no derivable target).  "
                    f"To preserve the training context, pass "
                    f"pd.concat([train_df, new_bars_df]) instead of the full "
                    f"historical dataset.  "
                    f"Event expression: {cand.expression}",
                    stacklevel=2,
                )
        return replayed

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
    """Map a composite score to a letter grade (doc Section 6.2).

    A  — strong evidence (≥ 0.75); B — solid (≥ 0.50);
    C  — moderate (≥ 0.25);        D — weak but still handed to Rule Discovery.
    """
    if score >= 0.75:
        return "A"
    if score >= 0.50:
        return "B"
    if score >= 0.25:
        return "C"
    return "D"
