"""Models for the search-level rotation null calibration (issue #116)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..unset import UNSET, coalesce, is_set

if TYPE_CHECKING:
    from ..alpha_discovery.models import AlphaContract

# Statistics computed entirely from in-sample data.
# oos_t is deliberately excluded: using OOS to choose the discriminant would be
# model-selection on the holdout.  OOS stays a sealed independent gate (stage 2).
IN_SAMPLE_STATS: Tuple[str, ...] = ("composite", "abs_z", "is_lift", "is_t")


@dataclass
class RotationConfig:
    """Configuration for the :class:`RotationCalibrator`.

    Parameters
    ----------
    k : int
        Number of rotation draws.  K=100 is recommended for a stable q95;
        K=40 is usable for a quick sanity check.
    alpha : float
        Type-I error target used to set the null q(1-alpha) bar.  Default 0.05.

        One of the five per-hypothesis significance levels in the pipeline, and
        session-resolved from ``PipelineContext.alpha`` like the other four
        (F9, #182).  They were five independent 0.05s that agreed by
        coincidence; a caller who wants a different regime now changes one
        number instead of finding all five.
    seed : int
        RNG seed for reproducible rotation offsets.
    in_sample_stats : tuple of str
        Yardsticks included in the Tippett min-p combination.  Changing this
        only affects the Tippett stage; per-statistic verdict rows always cover
        all five statistics.
    """

    k: int = 100
    alpha: float = UNSET
    seed: int = 20260624
    in_sample_stats: Tuple[str, ...] = IN_SAMPLE_STATS

    def resolved(self, alpha: float = 0.05) -> "RotationConfig":
        """Return a copy with ``alpha`` filled in from the session level.

        ``RotationConfig`` is not part of the resolver's config bundle — it is
        an argument to :func:`forgedge.forge`, not a module config — so the
        session level reaches it here instead.  Standalone use with no session
        falls back to the documented 0.05, which is the value the field held
        before it became resolved.
        """
        if is_set(self.alpha):
            return self
        return replace(self, alpha=coalesce(self.alpha, default=alpha))


@dataclass
class CalibrationReport:
    """Output of :meth:`RotationCalibrator.run`.

    Attributes
    ----------
    k : int
        Number of rotation draws used.
    alpha : float
        Type-I error target used.
    real_stats : dict
        Maximum of each yardstick over the real promoted contracts.
    null_arrays : dict
        Per-yardstick list of K null-draw maxima.
    per_stat_p : dict
        Per-yardstick empirical p-value of real vs. null.
    null_q : dict
        q(1-alpha) of the null distribution for each yardstick.
    tippett_p : float
        Empirical p-value of the Tippett (min-p) combination over
        ``in_sample_stats``.  This is the primary calibration verdict.
    tippett_min_p_real : float
        The actual min-p value achieved by the real run.
    tippett_best_stat : str
        The in-sample statistic that drives the Tippett min-p.
    survivors : list[AlphaContract]
        Promoted contracts whose winning in-sample statistic exceeds
        the q(1-alpha) null bar.
    """

    k: int
    alpha: float
    real_stats: Dict[str, float]
    null_arrays: Dict[str, List[float]]
    per_stat_p: Dict[str, float]
    null_q: Dict[str, float]
    tippett_p: float
    tippett_min_p_real: float
    tippett_best_stat: str
    survivors: List["AlphaContract"] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable calibration summary."""
        lines = [
            f"Search-level rotation null  (K={self.k}, alpha={self.alpha})",
            "",
            f"{'statistic':10s} {'real':>9} {'null mean':>10} {'null q':>8} "
            f"{'emp p':>7}  verdict",
        ]
        import numpy as np
        for name, real_v in self.real_stats.items():
            arr = self.null_arrays.get(name, [])
            if not arr:
                continue
            a = [x for x in arr if x == x]  # drop NaN
            if not a:
                continue
            mean = sum(a) / len(a)
            q = self.null_q.get(name, float("nan"))
            p = self.per_stat_p.get(name, float("nan"))
            flag = (
                "SEPARATES" if p < self.alpha
                else "borderline" if p < 0.10
                else "weak"
            )
            lines.append(
                f"{name:10s} {real_v:9.4f} {mean:10.4f} {q:8.4f} {p:7.4f}  {flag}"
            )
        lines += [
            "",
            f"Tippett (min-p, in-sample only): {self.tippett_min_p_real:.4f}"
            f"  driven by '{self.tippett_best_stat}'",
            f"Tippett empirical p             : {self.tippett_p:.4f}"
            f"  ({'SEPARATES' if self.tippett_p < self.alpha else 'borderline' if self.tippett_p < 0.10 else 'weak'})",
            f"Survivors above null bar        : {len(self.survivors)}",
        ]
        return "\n".join(lines)
