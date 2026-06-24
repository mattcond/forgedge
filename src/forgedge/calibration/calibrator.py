"""Search-level rotation null calibrator (issue #116).

Contract-preserving alpha calibration: tells a real edge from one manufactured
by FORGE's multiple-testing surface without ever rebuilding the feature columns,
so it never breaks FORGE's contract (the user passes a pre-built feature table
and nothing else).

How it works
------------
1. Re-use the post-EventDiscovery table (``ed_df``) and the candidate set
   (``cands``) produced by a prior FORGE run — both are outcome-independent and
   already inside ``ForgeResult.event_frame`` / ``ForgeResult.candidates``.
2. K times, circularly rotate only the ``close`` column by a random offset and
   re-run *only* AlphaDiscovery, reusing the real candidate set.  Rotation
   preserves the exact return marginal and autocorrelation (one wrap seam) but
   decouples each event's cached activations from the forward outcome.
3. Record the maximum of each yardstick over all directed contracts per draw.
4. Apply a Tippett (min-p) combination over the in-sample yardsticks to
   adaptively select the best discriminant while paying the FWER cost of
   looking at multiple statistics.  OOS is excluded from the combination and
   stays a sealed independent confirmation gate (stage 2).

Usage — standalone (recommended)
---------------------------------
    result = forge(df, config)                    # run normally, no extra cost

    cal = RotationCalibrator(
        ed_df  = result.event_frame,
        cands  = result.candidates,
        config = alpha_config,
    )
    report = cal.run(result.promoted, RotationConfig(k=100))
    print(report.summary())

Usage — inline (via forge() knob)
----------------------------------
    result = forge(df, config, rotation_calibration=RotationConfig(k=100))
    # AlphaContract.rotation_p and .rotation_threshold are populated

Serialization of the candidate set (for cross-session replay)
--------------------------------------------------------------
    import pickle, pathlib
    pathlib.Path("cands.pkl").write_bytes(pickle.dumps(result.candidates))
    # later:
    cands = pickle.loads(pathlib.Path("cands.pkl").read_bytes())
    cal = RotationCalibrator(ed_df=pd.read_parquet("ed_df.parquet"), cands=cands, ...)
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..alpha_discovery.discovery import AlphaDiscovery
from ..alpha_discovery.models import AlphaConfig, AlphaContract
from ..event_discovery.models import EventCandidate
from .models import CalibrationReport, RotationConfig

logger = logging.getLogger(__name__)

_IN_SAMPLE_STATS = ("composite", "abs_z", "is_lift", "is_t")


def _rotate_close(frame: pd.DataFrame, k: int, close_col: str = "close") -> pd.DataFrame:
    """Return a copy of *frame* with only the close column circularly shifted by k."""
    out = frame.copy()
    out[close_col] = np.roll(frame[close_col].to_numpy(), k)
    return out


def _max_statistics(contracts: Sequence[AlphaContract]) -> Dict[str, float]:
    """Return per-yardstick maximum over all directed contracts."""
    directed = [c for c in contracts if c.direction in ("long", "short")]

    def _max(vals: list) -> float:
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        return float(max(finite)) if finite else float("nan")

    def _abs_z(c: AlphaContract) -> float:
        dt = c.derived_target
        return abs(dt.t_stat_by_h.get(dt.holding_period_h, float("nan")))

    return {
        "composite": _max([c.alpha_score.composite_score for c in directed]),
        "abs_z": _max([_abs_z(c) for c in directed]),
        "is_lift": _max([c.event_stats.lift for c in directed]),
        "is_t": _max([c.event_stats.t_stat for c in directed]),
        "oos_t": _max(
            [
                c.oos_validation.t_stat
                for c in directed
                if c.oos_validation is not None
            ]
        ),
    }


def _empirical_p(val: float, arr: np.ndarray) -> float:
    """One-sided empirical p: (1 + #{null >= val}) / (1 + K).  Inclusive, conservative."""
    return (1 + int(np.sum(arr >= val))) / (1 + len(arr))


def _tippett(
    real_stats: Dict[str, float],
    null_arrays: Dict[str, np.ndarray],
    in_sample_stats: Tuple[str, ...],
) -> Tuple[float, float, str, np.ndarray]:
    """Tippett (min-p) combination over in-sample statistics.

    Returns
    -------
    tippett_p : float
        Empirical p of the real min-p against the null min-p distribution.
    min_p_real : float
        The real run's minimum per-statistic p-value.
    best_stat : str
        The statistic that drives min_p_real.
    min_p_null : np.ndarray
        Null distribution of per-draw min-p values.
    """
    valid = {
        s: null_arrays[s]
        for s in in_sample_stats
        if s in null_arrays
        and null_arrays[s].size > 0
        and s in real_stats
        and np.isfinite(real_stats[s])
    }
    if not valid:
        return float("nan"), float("nan"), "", np.array([])

    # Per-statistic p-values for the real run
    p_real = {s: _empirical_p(real_stats[s], arr) for s, arr in valid.items()}
    min_p_real = min(p_real.values())
    best_stat = min(p_real, key=p_real.get)

    # Null distribution of per-draw min-p
    K_eff = min(len(arr) for arr in valid.values())
    min_p_null_list = []
    for j in range(K_eff):
        p_j = {
            s: _empirical_p(arr[j], arr)
            for s, arr in valid.items()
            if j < len(arr)
        }
        if p_j:
            min_p_null_list.append(min(p_j.values()))
    min_p_null = np.array(min_p_null_list)

    tippett_p = (
        (1 + int(np.sum(min_p_null <= min_p_real))) / (1 + len(min_p_null))
        if min_p_null.size
        else float("nan")
    )
    return tippett_p, min_p_real, best_stat, min_p_null


class RotationCalibrator:
    """Search-level rotation null calibrator for FORGE alpha contracts.

    Takes the post-EventDiscovery state (already inside ``ForgeResult``) and
    runs K circular rotations of the ``close`` column, re-running only
    AlphaDiscovery on each draw.  The resulting null distribution of max
    statistics is used to compute per-contract empirical p-values via the
    Tippett (min-p) combination.

    Parameters
    ----------
    ed_df : pd.DataFrame
        Post-EventDiscovery table — feature columns, regime, and the ``close``
        column.  Available as ``ForgeResult.event_frame``.
    cands : list[EventCandidate]
        Candidate set from EventDiscovery.  Available as
        ``ForgeResult.candidates``.  Reused as-is on every null draw (outcome-
        independent: EventDiscovery gates on feature statistics, not returns).
    config : AlphaConfig
        Same AlphaConfig used in the original run.
    """

    def __init__(
        self,
        ed_df: pd.DataFrame,
        cands: List[EventCandidate],
        config: AlphaConfig,
    ) -> None:
        self._ed_df = ed_df
        self._cands = list(cands)
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        promoted: List[AlphaContract],
        rotation_config: Optional[RotationConfig] = None,
    ) -> CalibrationReport:
        """Run K rotation draws and produce a :class:`CalibrationReport`.

        Also annotates each contract in *promoted* in-place with
        ``rotation_p`` and ``rotation_threshold`` if those fields exist.

        Parameters
        ----------
        promoted : list[AlphaContract]
            Promoted contracts from the real pipeline run (e.g.
            ``ForgeResult.promoted``).
        rotation_config : RotationConfig, optional
            Calibration hyper-parameters.  Defaults to ``RotationConfig()``.

        Returns
        -------
        CalibrationReport
        """
        rc = rotation_config or RotationConfig()
        close_col = self._config.close_col

        # ── 1. Real max statistics ──────────────────────────────────────────
        real_stats = _max_statistics(promoted)
        logger.info(
            "RotationCalibrator: %d promoted contracts, real stats: %s",
            len(promoted),
            {k: f"{v:.4f}" for k, v in real_stats.items()},
        )

        # ── 2. K rotation draws ─────────────────────────────────────────────
        n = len(self._ed_df)
        rng = np.random.default_rng(rc.seed)
        lo, hi = 10, n - 10
        null_lists: Dict[str, List[float]] = {k: [] for k in real_stats}

        for j in range(rc.k):
            k = int(rng.integers(lo, hi))
            rot = _rotate_close(self._ed_df, k, close_col=close_col)
            rot_cfg = copy.copy(self._config)
            rot_cfg.asset = f"ROT{j}"
            ad = AlphaDiscovery(rot, self._cands, rot_cfg)
            ad.run()
            st = _max_statistics(ad.promoted_contracts())
            for name, v in st.items():
                null_lists[name].append(v)
            if (j + 1) % 10 == 0 or j == rc.k - 1:
                logger.debug("RotationCalibrator: draw %d/%d done", j + 1, rc.k)

        # ── 3. Build null arrays and per-stat summaries ─────────────────────
        null_arrays: Dict[str, np.ndarray] = {}
        per_stat_p: Dict[str, float] = {}
        null_q: Dict[str, float] = {}
        for name in real_stats:
            arr = np.array([x for x in null_lists[name] if np.isfinite(x)])
            null_arrays[name] = arr
            if arr.size and np.isfinite(real_stats[name]):
                per_stat_p[name] = _empirical_p(real_stats[name], arr)
                null_q[name] = float(np.quantile(arr, 1 - rc.alpha))
            else:
                per_stat_p[name] = float("nan")
                null_q[name] = float("nan")

        # ── 4. Tippett combination (in-sample stats only) ───────────────────
        tippett_p, min_p_real, best_stat, _ = _tippett(
            real_stats, null_arrays, rc.in_sample_stats
        )

        # ── 5. Survivors above the null bar of the best in-sample stat ──────
        bar = null_q.get(best_stat, float("nan"))
        survivors: List[AlphaContract] = []
        if np.isfinite(bar) and best_stat:
            for c in promoted:
                val = _contract_stat(c, best_stat)
                if np.isfinite(val) and val > bar:
                    survivors.append(c)

        # ── 6. Annotate contracts in-place (optional fields) ────────────────
        for c in promoted:
            val = _contract_stat(c, best_stat) if best_stat else float("nan")
            stat_p = per_stat_p.get(best_stat, float("nan"))
            if np.isfinite(val) and np.isfinite(bar) and null_arrays.get(best_stat, np.array([])).size:
                c_p = _empirical_p(val, null_arrays[best_stat])
            else:
                c_p = float("nan")
            # Populate only if the fields exist (added in AlphaContract)
            if hasattr(c, "rotation_p"):
                c.rotation_p = c_p
            if hasattr(c, "rotation_threshold"):
                c.rotation_threshold = bar if np.isfinite(bar) else None

        return CalibrationReport(
            k=rc.k,
            alpha=rc.alpha,
            real_stats=real_stats,
            null_arrays={k: v.tolist() for k, v in null_arrays.items()},
            per_stat_p=per_stat_p,
            null_q=null_q,
            tippett_p=tippett_p,
            tippett_min_p_real=min_p_real,
            tippett_best_stat=best_stat,
            survivors=survivors,
        )


def _contract_stat(c: AlphaContract, name: str) -> float:
    """Extract the named yardstick from a single contract."""
    if name == "abs_z":
        dt = c.derived_target
        return abs(dt.t_stat_by_h.get(dt.holding_period_h, float("nan")))
    if name == "is_lift":
        return c.event_stats.lift
    if name == "is_t":
        return c.event_stats.t_stat
    if name == "oos_t":
        return c.oos_validation.t_stat if c.oos_validation else float("nan")
    return c.alpha_score.composite_score
