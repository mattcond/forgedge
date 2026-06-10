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
    """Applies the four-criterion temporal consistency filter to event series.

    The gate operates exclusively on the *when* of activations, never on
    *what follows* them.  This is essential to avoid look-ahead bias: the
    gate ensures events are structurally sound before any return-based
    analysis begins.

    Parameters
    ----------
    params : GateParams or None
        Configuration for all four thresholds.  Defaults to
        ``GateParams()`` (min_act=50, min_months=8, max_conc=0.40,
        min_tpm=2.0).
    """

    def __init__(self, params: Optional[GateParams] = None):
        self.params = params or GateParams()

    def evaluate(
        self,
        active: np.ndarray,
        period_counts: np.ndarray,
        n_total_months: int,
    ) -> GateResult:
        """Evaluate the four gate criteria given pre-computed counts.

        Criteria are checked in order from cheapest to most informative:

        1. **Volume** (``n_activations >= min_act``): rejects sparse events
           that lack statistical power.
        2. **Coverage** (``n_active_months >= min_months``): rejects bursts
           of activity confined to a short period.
        3. **Concentration** (``max_month_count / n_activations <= max_conc``):
           rejects events where one month dominates — even if multiple months
           are nominally active, the distribution may still be skewed.
        4. **Frequency** (``n_activations / n_total_months >= min_tpm``):
           rejects events whose average activation rate is too low relative
           to the full date range (complements coverage by checking density).

        Failing any criterion terminates evaluation immediately and returns
        a ``GateResult`` with ``passed=False`` and a ``fail_reason`` string.

        Parameters
        ----------
        active : np.ndarray
            Boolean array (True = activated) of length = n_rows in the dataset.
        period_counts : np.ndarray
            Per-month activation counts, length = n_total_months.  Computed
            via ``_count_by_month``.
        n_total_months : int
            Total number of calendar months spanned by the dataset.

        Returns
        -------
        GateResult
            ``passed=True`` if all criteria are satisfied, else
            ``passed=False`` with ``fail_reason`` describing the first
            failing criterion.
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
        """Evaluate the gate from a boolean Series using a pre-computed month index.

        Convenience wrapper around ``evaluate`` that converts the pandas
        Series to a numpy boolean array and accumulates per-month counts
        using the vectorised ``_count_by_month`` helper.  The month index
        is pre-computed once per call to ``filter`` and reused for all events
        to amortise the cost of the ``dt.to_period`` conversion.

        Parameters
        ----------
        event_series : pd.Series
            Boolean (0/1/NaN) event series.
        month_index : np.ndarray
            Integer array mapping each row to its zero-based month index,
            as produced by ``_build_month_index``.
        n_total_months : int
            Total number of calendar months, as returned by
            ``_build_month_index``.

        Returns
        -------
        GateResult
        """
        active = event_series.fillna(0).values.astype(bool)
        period_counts = _count_by_month(active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months)

    def filter(
        self,
        events: list[RawEvent],
        timestamps: pd.Series,
    ) -> list[RawEvent]:
        """Apply the gate to all events and return those that pass.

        Builds the month index once from ``timestamps``, then iterates over
        all events calling ``evaluate_series`` on each.  The ``gate_result``
        attribute is set on every event — including those that fail — so
        that downstream diagnostic tools can inspect rejection reasons.

        Parameters
        ----------
        events : list[RawEvent]
            All raw events produced by EventGenerator (Step 3).
        timestamps : pd.Series
            Datetime series aligned to the KPI table rows.

        Returns
        -------
        list[RawEvent]
            Subset of ``events`` for which ``gate_result.passed == True``.
        """
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
        """Evaluate the gate on the logical AND of two boolean numpy arrays.

        Used by ANDComposer for fast pair evaluation without constructing a
        pandas Series for the combined signal until the gate is confirmed to
        pass.  The AND is computed at the numpy level before casting to bool
        for ``evaluate``.

        Parameters
        ----------
        s1 : np.ndarray
            First boolean array (dtype bool or uint8).
        s2 : np.ndarray
            Second boolean array (same length as s1).
        month_index : np.ndarray
            Pre-computed month index from ``_build_month_index``.
        n_total_months : int
            Total number of months.

        Returns
        -------
        GateResult
        """
        active = s1 & s2
        period_counts = _count_by_month(active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_month_index(timestamps: pd.Series) -> tuple[np.ndarray, int]:
    """Map each row to a zero-based integer month index.

    Converts the timestamp series to pandas Period objects (monthly
    frequency) and then to a compact integer array.  The mapping is stable
    across calls for the same dataset, making it safe to pre-compute once
    and reuse for all events.

    Parameters
    ----------
    timestamps : pd.Series
        Datetime series aligned to the KPI table rows.

    Returns
    -------
    month_index : np.ndarray
        Integer array of shape (n_rows,) where each value is the zero-based
        index of the calendar month for that row.
    n_total_months : int
        Total number of distinct calendar months in the dataset.
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
    """Accumulate per-month activation counts using vectorised numpy scatter-add.

    Uses ``np.add.at`` for an unbuffered in-place addition, which correctly
    handles repeated indices (multiple rows in the same month).  This is
    significantly faster than a pandas ``groupby().sum()`` for large arrays
    because it avoids constructing intermediate Series and Period objects.

    Parameters
    ----------
    active : np.ndarray
        Boolean activation array (dtype bool or uint8).
    month_index : np.ndarray
        Zero-based month index for each row, as returned by
        ``_build_month_index``.
    n_months : int
        Length of the output array (number of distinct months).

    Returns
    -------
    np.ndarray
        Integer array of shape (n_months,) with per-month activation counts.
    """
    counts = np.zeros(n_months, dtype=np.int32)
    np.add.at(counts, month_index, active.astype(np.int32))
    return counts


# Keep this for external callers (e.g. discovery.py zero-month counter)
def _monthly_counts(active: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Compute per-month activation counts as a pandas Series (legacy helper).

    Returns a Series indexed by ``pd.Period`` covering every calendar month
    from the first to the last timestamp, with zero fill for inactive months.
    Used by ``EventDiscovery._count_zero_months`` which needs the full
    period range for zero-month counting.

    Parameters
    ----------
    active : pd.Series
        Boolean (0/1) activation series.
    timestamps : pd.Series
        Datetime series aligned to ``active``.

    Returns
    -------
    pd.Series
        Monthly activation counts indexed by ``pd.Period("M")``, with zeros
        for months where the event never fired.
    """
    periods = timestamps.dt.to_period("M")
    # Use positional groupby to avoid index-alignment surprises when `active`
    # carries a DatetimeIndex and `periods` carries a RangeIndex.
    counts = (
        pd.Series(active.values.astype(np.int32))
        .groupby(pd.Series(periods.values))
        .sum()
    )
    p_min, p_max = periods.min(), periods.max()
    if pd.isnull(p_min) or pd.isnull(p_max):
        return pd.Series(dtype=np.int32)
    full_index = pd.period_range(p_min, p_max, freq="M")
    return counts.reindex(full_index, fill_value=0)
