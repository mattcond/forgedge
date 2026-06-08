"""Step 4 — Consistency Gate.

Filters events based solely on the temporal distribution of their activations.
No forward return is observed here.

Four criteria:
  1. Volume     — minimum total activations (MIN_ACT)
  2. Coverage   — minimum active months (MIN_MONTHS)
  3. Concentration — no single month dominates (MAX_CONC)
  4. Frequency  — minimum activations per month (MIN_TPM)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import GateParams, GateResult, RawEvent


class ConsistencyGate:
    def __init__(self, params: Optional[GateParams] = None):
        self.params = params or GateParams()

    def evaluate(
        self,
        active: np.ndarray,
        period_counts: np.ndarray,
        n_total_months: int,
    ) -> GateResult:
        """Evaluate the gate given pre-computed activation counts.

        Parameters
        ----------
        active:
            Boolean array (True = activated).
        period_counts:
            Array of activation counts per calendar month (length = n_total_months).
        n_total_months:
            Total number of months in the dataset.
        """
        p = self.params
        n_act = int(active.sum())

        if n_act < p.min_act:
            return GateResult(
                passed=False,
                n_activations=n_act,
                n_active_months=0,
                max_monthly_share=float("nan"),
                mean_tpm=float("nan"),
                fail_reason=f"volume: {n_act} < {p.min_act}",
            )

        n_active_months = int((period_counts > 0).sum())
        max_month_count = int(period_counts.max())
        max_conc = max_month_count / n_act
        mean_tpm = n_act / n_total_months

        if n_active_months < p.min_months:
            return GateResult(
                passed=False,
                n_activations=n_act,
                n_active_months=n_active_months,
                max_monthly_share=max_conc,
                mean_tpm=mean_tpm,
                fail_reason=f"coverage: {n_active_months} active months < {p.min_months}",
            )

        if max_conc > p.max_conc:
            return GateResult(
                passed=False,
                n_activations=n_act,
                n_active_months=n_active_months,
                max_monthly_share=max_conc,
                mean_tpm=mean_tpm,
                fail_reason=f"concentration: {max_conc:.2f} > {p.max_conc}",
            )

        if mean_tpm < p.min_tpm:
            return GateResult(
                passed=False,
                n_activations=n_act,
                n_active_months=n_active_months,
                max_monthly_share=max_conc,
                mean_tpm=mean_tpm,
                fail_reason=f"frequency: {mean_tpm:.2f} tpm < {p.min_tpm}",
            )

        return GateResult(
            passed=True,
            n_activations=n_act,
            n_active_months=n_active_months,
            max_monthly_share=max_conc,
            mean_tpm=mean_tpm,
        )

    def evaluate_series(
        self,
        event_series: pd.Series,
        month_index: np.ndarray,
        n_total_months: int,
    ) -> GateResult:
        """Evaluate from a boolean Series using pre-computed month index."""
        active = event_series.fillna(0).values.astype(bool)
        period_counts = _count_by_month(active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months)

    def filter(
        self,
        events: list[RawEvent],
        timestamps: pd.Series,
    ) -> list[RawEvent]:
        """Apply the gate to all events; return those that pass."""
        month_index, n_total_months = _build_month_index(timestamps)
        passing: list[RawEvent] = []
        for ev in events:
            result = self.evaluate_series(ev.series, month_index, n_total_months)
            ev.gate_result = result
            if result.passed:
                passing.append(ev)
        return passing

    def evaluate_and(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        month_index: np.ndarray,
        n_total_months: int,
    ) -> GateResult:
        """Fast evaluation of an AND composition from two boolean arrays."""
        active = s1 & s2
        period_counts = _count_by_month(active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_month_index(timestamps: pd.Series) -> tuple[np.ndarray, int]:
    """Map each row to a zero-based month index.

    Returns (month_index, n_total_months).
    """
    periods = timestamps.dt.to_period("M")
    unique_months = periods.sort_values().unique()
    month_to_idx = {m: i for i, m in enumerate(unique_months)}
    month_index = np.array([month_to_idx[p] for p in periods], dtype=np.int32)
    return month_index, len(unique_months)


def _count_by_month(
    active: np.ndarray,
    month_index: np.ndarray,
    n_months: int,
) -> np.ndarray:
    counts = np.zeros(n_months, dtype=np.int32)
    np.add.at(counts, month_index, active.astype(np.int32))
    return counts


# Keep this for external callers (e.g. discovery.py zero-month counter)
def _monthly_counts(active: pd.Series, timestamps: pd.Series) -> pd.Series:
    periods = timestamps.dt.to_period("M")
    all_months = periods.unique()
    counts = active.astype(int).groupby(periods).sum()
    full_index = pd.period_range(all_months.min(), all_months.max(), freq="M")
    return counts.reindex(full_index, fill_value=0)
