"""Fast search-level rotation null via circular cross-correlation (FFT).

The :class:`~forgedge.calibration.calibrator.RotationCalibrator` re-runs the
whole of Alpha Discovery K times (once per sampled rotation) and keeps only a
handful of max-statistics per draw — ~K × the cost of Alpha Discovery for a
null of K points.  This module computes the same *kind* of null — the
distribution of the best standardised excess ``max_e max_h |z_h(e)|`` over the
full candidate set when the event timing is decoupled from the returns — but
**exactly, over every circular offset at once, with no sampling and no seed**,
at roughly the cost of a single Alpha Discovery pass.

How
---
Alpha Discovery already standardises each event's excess log-return by a
per-event circular-rotation null (``AlphaDiscovery._rotation_null``): for one
event it computes, via FFT cross-correlation, the mean active-bar log-return
under *every* shift ``k`` of the activation mask.  ``FastRotationNull`` runs
the identical computation per candidate, but instead of discarding the shifted
rows after extracting ``σ_null``, it keeps the full standardised matrix
``z[k, h] = (μ_shift[k, h] − μ_base[h]) / σ_null[h]`` and reduces it to a
per-offset search statistic

    S(k) = max over candidates of  max_h |z[k, h]|

Row ``k = 0`` is the identity shift — the real run — so ``S(0)`` is the real
search's best standardised excess and ``{S(k), k ≥ 1}`` is the exact null
distribution of that statistic over all non-trivial rotations.  Offsets are
exchangeable under the null "event timing is unrelated to returns", so the
empirical p ``(1 + #{S(k) ≥ S(0)}) / (1 + n_offsets)`` is a valid
permutation-style p-value for the *whole search*, correlation structure
included (every candidate is rotated jointly, so cross-candidate correlation
is preserved draw by draw).

Fidelity
--------
For each candidate, row 0 of the standardised matrix reproduces Alpha
Discovery's ``t_stat_by_h`` exactly (same FFT, same masking, same
``σ_null``), so the per-candidate real statistic equals
``max_h |t_stat_by_h[h]|`` of the contract.  The reduction uses the max over
the whole grid rather than the BH-restricted ``h*`` of the real pipeline —
the search-level statistic prices the *whole* grid the search looked at.

Yardstick scope
---------------
Only ``abs_z`` is computed: it is the pipeline's own selection statistic
(``h* = argmax |z_h|``) and the only yardstick that is exact under this
algebra.  ``composite`` / ``is_lift`` need per-draw MFE quantiles and scoring
that cannot be expressed as cross-correlations; for those, use the full
:class:`RotationCalibrator`.  With a single yardstick the Tippett stage
collapses to the statistic's own empirical p (no multi-statistic FWER cost to
pay).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..alpha_discovery.discovery import AlphaDiscovery
from ..alpha_discovery.models import AlphaConfig, AlphaContract
from ..alpha_discovery.target import forward_log_returns, forward_returns
from ..event_discovery.models import EventCandidate
from .models import CalibrationReport

logger = logging.getLogger(__name__)

import pandas as pd


def contract_abs_z(contract: AlphaContract) -> float:
    """Best standardised excess of a contract over its whole horizon grid.

    ``max_h |t_stat_by_h[h]|`` — the same per-candidate reduction used by the
    fast null, read from the contract's stored per-horizon profile.
    """
    dt = contract.derived_target
    vals = [abs(v) for v in dt.t_stat_by_h.values() if v is not None and np.isfinite(v)]
    return max(vals) if vals else float("nan")


class FastRotationNull:
    """Exact search-level rotation null over all circular offsets.

    Parameters
    ----------
    ed_df : pd.DataFrame
        Post-EventDiscovery table (``ForgeResult.event_frame``) — feature
        columns plus the close column.
    cands : list[EventCandidate]
        The candidate set Alpha Discovery evaluated
        (``ForgeResult.candidates``).  Activations are outcome-independent and
        are reused verbatim, exactly as in the slow calibrator.
    config : AlphaConfig
        The same config used in the real run (horizon grid, IS split, close
        column).
    """

    def __init__(
        self,
        ed_df: "pd.DataFrame",
        cands: List[EventCandidate],
        config: AlphaConfig,
    ) -> None:
        # Reuse AlphaDiscovery's frame preparation and event-series replay so
        # activations and the close series are bit-for-bit the real run's.
        self._ad = AlphaDiscovery(ed_df, cands, config)
        self._cands = list(cands)

    # ------------------------------------------------------------------

    def run(
        self,
        promoted: List[AlphaContract],
        alpha: float = 0.05,
    ) -> CalibrationReport:
        """Compute the exact null of the search statistic and annotate contracts.

        Parameters
        ----------
        promoted : list[AlphaContract]
            Promoted contracts of the real run.  Each is annotated in place
            with ``rotation_p`` (empirical p of its own ``abs_z`` against the
            null of the search max — FWER-style, like the slow calibrator) and
            ``rotation_threshold`` (the q(1−alpha) null bar).
        alpha : float
            Type-I error target for the null bar.  Default 0.05.

        Returns
        -------
        CalibrationReport
            ``k`` is the number of non-trivial offsets used (data length − 1,
            minus offsets with no usable statistic).
        """
        cfg = self._ad.config
        horizons = [int(h) for h in cfg.horizon_grid]
        close = self._ad._frame[cfg.close_col].astype(float)
        n = len(close)
        split = min(max(int(round(n * cfg.train_ratio)), 0), n)

        report_nan = CalibrationReport(
            k=0, alpha=alpha, real_stats={"abs_z": float("nan")},
            null_arrays={"abs_z": []}, per_stat_p={"abs_z": float("nan")},
            null_q={"abs_z": float("nan")}, tippett_p=float("nan"),
            tippett_min_p_real=float("nan"), tippett_best_stat="abs_z",
            survivors=[],
        )
        if split < 4:
            return report_nan

        # ── In-sample forward log-returns, exactly as AlphaDiscovery.run ──
        F_is = forward_returns(close, horizons).to_numpy()[:split]
        L_is = forward_log_returns(close, horizons).to_numpy()[:split]
        valid_is = np.isfinite(F_is) & np.isfinite(L_is)
        L0 = np.where(valid_is, L_is, 0.0)
        cnt_t = valid_is.sum(axis=0).astype(float)
        sum_t = L0.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_base = np.where(cnt_t > 0, sum_t / np.maximum(cnt_t, 1), np.nan)

        # Shared FFTs of the return/validity columns (one per horizon).
        FL = np.fft.rfft(L0, axis=0)
        FV = np.fft.rfft(valid_is.astype(float), axis=0)

        # Per-offset best statistic over all candidates; row 0 = the real run.
        best = np.full(split, -np.inf)
        n_used = 0
        for cand in self._cands:
            event = self._ad._event_series(cand)
            active = event.fillna(0).astype(bool).to_numpy()[:split]
            if not active.any():
                continue
            fa_conj = np.conj(np.fft.rfft(active.astype(float)))
            num = np.fft.irfft(fa_conj[:, None] * FL, n=split, axis=0)
            den = np.fft.irfft(fa_conj[:, None] * FV, n=split, axis=0)

            with np.errstate(divide="ignore", invalid="ignore"):
                mu_shift = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
            boot = mu_shift - mu_base[None, :]          # (split, H); row 0 real
            null_rows = boot[1:]
            n_valid = np.isfinite(null_rows).sum(axis=0)
            with np.errstate(invalid="ignore"):
                sd = np.where(n_valid > 1, np.nanstd(null_rows, axis=0), np.nan)
                z = np.where(sd > 0, boot / sd, np.nan)
            # Mirror _derive_target's usable gate: ≥ 2 active∩valid bars.
            z = np.where(den >= 1.5, z, np.nan)
            with np.errstate(invalid="ignore"):
                score = np.nanmax(
                    np.where(np.isfinite(z), np.abs(z), -np.inf), axis=1
                )
            best = np.fmax(best, score)
            n_used += 1

        if n_used == 0 or not np.isfinite(best[0]):
            return report_nan

        real = float(best[0])
        null = best[1:]
        null = null[np.isfinite(null)]
        if null.size == 0:
            return report_nan

        p_search = (1.0 + float(np.sum(null >= real))) / (1.0 + null.size)
        bar = float(np.quantile(null, 1.0 - alpha))

        # ── Per-contract annotation against the null of the search max ────
        survivors: List[AlphaContract] = []
        for c in promoted:
            val = contract_abs_z(c)
            if np.isfinite(val):
                c_p = (1.0 + float(np.sum(null >= val))) / (1.0 + null.size)
            else:
                c_p = float("nan")
            if hasattr(c, "rotation_p"):
                c.rotation_p = c_p
            if hasattr(c, "rotation_threshold"):
                c.rotation_threshold = bar
            if np.isfinite(val) and val > bar:
                survivors.append(c)

        logger.info(
            "FastRotationNull: %d candidates, %d offsets — real max|z|=%.3f, "
            "null q%d=%.3f, p=%.4f, %d/%d survivors",
            n_used, null.size, real, round((1 - alpha) * 100), bar, p_search,
            len(survivors), len(promoted),
        )
        return CalibrationReport(
            k=int(null.size),
            alpha=alpha,
            real_stats={"abs_z": real},
            null_arrays={"abs_z": null.tolist()},
            per_stat_p={"abs_z": p_search},
            null_q={"abs_z": bar},
            tippett_p=p_search,
            tippett_min_p_real=p_search,
            tippett_best_stat="abs_z",
            survivors=survivors,
        )
