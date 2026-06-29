"""Step 4 — Consistency Gate.

Filters events based solely on the temporal distribution of their activations.
No forward return is observed here.

Three criteria (evaluated in order):
  1. Episodes  — minimum distinct activation episodes (min_episodes)
  2. Rate       — optional activation rate floor (min_tpm; skipped when 0)
  3. Dispersion — episode-level Index of Dispersion ≤ max_dispersion
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import GateParams, GateResult, RawEvent


class ConsistencyGate:
    """Applies the temporal consistency filter to event series.

    The gate operates exclusively on the *when* of activations, never on
    *what follows* them.  This is essential to avoid look-ahead bias: the
    gate ensures events are structurally sound before any return-based
    analysis begins.

    Dispersion is measured at the **episode** level, not the bar level.
    An episode is a maximal run of consecutive True bars, collapsed to its
    start bar.  This avoids inflating the Index of Dispersion for persistent
    states (e.g. ``RSI < 30`` that spans several consecutive days), while
    correctly flagging regime-concentrated events whose episodes cluster in
    a small subset of calendar months.

    Parameters
    ----------
    params : GateParams or None
        Configuration for all thresholds.  Defaults to
        ``GateParams()`` (min_tpm=0.0, max_dispersion=2.5, min_episodes=5).
    """

    def __init__(self, params: Optional[GateParams] = None):
        self.params = params or GateParams()

    def evaluate(
        self,
        active: np.ndarray,
        period_counts: np.ndarray,
        n_total_months: int,
        episode_counts: Optional[np.ndarray] = None,
    ) -> GateResult:
        """Evaluate the gate criteria given pre-computed counts.

        Criteria are checked in order:

        1. **Episodes** (``n_episodes >= min_episodes``): rejects events with
           too few distinct activation episodes for reliable statistics.
        2. **Rate** (``mean_tpm >= min_tpm`` when ``min_tpm > 0``): optional
           hard floor on activation rate; skipped when ``min_tpm == 0``.
        3. **Dispersion** (``episode_ID <= max_dispersion``): rejects events
           whose episodes are over-concentrated in a subset of months.
           ``episode_ID`` = Var / Mean of monthly episode-start counts.
           For a Poisson-like process, ``episode_ID ≈ 1``.  High values
           flag regime-concentration (burned alpha) or seasonal clustering.
           Skipped when ``episode_counts`` is not supplied (``episode_ID``
           reported as NaN).

        Parameters
        ----------
        active : np.ndarray
            Boolean array (True = activated) of length = n_rows.
        period_counts : np.ndarray
            Per-month bar-activation counts, length = n_total_months.
            Computed via ``_count_by_month``.
        n_total_months : int
            Total number of calendar months spanned by the dataset.
        episode_counts : np.ndarray or None
            Per-month episode-start counts, same length as ``period_counts``.
            Computed via ``_count_by_month(_to_episodes(active), ...)``.
            When None the dispersion criterion is skipped (episode_ID = NaN).

        Returns
        -------
        GateResult
            ``passed=True`` if all criteria are satisfied, else
            ``passed=False`` with ``fail_reason`` describing the first
            failing criterion.
        """
        p = self.params
        n_act = int(active.sum())

        # ── Diagnostic fields ────────────────────────────────────────────────
        n_active_months = int((period_counts > 0).sum()) if len(period_counts) > 0 else 0
        max_month_count = int(period_counts.max()) if len(period_counts) > 0 and n_act > 0 else 0
        max_conc = max_month_count / n_act if n_act > 0 else float("nan")
        mean_tpm = n_act / n_total_months if n_total_months > 0 else 0.0

        # Bar-level ID (kept for diagnostics; not the gate criterion)
        if n_total_months > 1 and n_act > 0:
            bar_mu = float(period_counts.mean())
            bar_var = float(period_counts.var(ddof=1))
            id_score = float(bar_var / bar_mu) if bar_mu > 0 else float("inf")
        else:
            id_score = 0.0

        # ── Episode-based calculations ───────────────────────────────────────
        episode_active = _to_episodes(active)
        n_episodes = int(episode_active.sum())

        # Episode-level ID (the gate criterion for dispersion)
        if episode_counts is not None and n_total_months > 1 and n_episodes > 0:
            ep_mu = float(episode_counts.mean())
            ep_var = float(episode_counts.var(ddof=1))
            episode_id = float(ep_var / ep_mu) if ep_mu > 0 else float("inf")
        else:
            episode_id = float("nan")

        # Effective sample size: deflates significance for clustered episodes
        if n_episodes > 0 and not np.isnan(episode_id):
            n_eff = float(n_episodes) / max(1.0, episode_id)
        else:
            n_eff = float(n_episodes)

        def _result(passed: bool, fail_reason: Optional[str] = None) -> GateResult:
            return GateResult(
                passed=passed,
                n_activations=n_act,
                n_active_months=n_active_months,
                max_monthly_share=max_conc,
                mean_tpm=mean_tpm,
                index_of_dispersion=id_score,
                n_episodes=n_episodes,
                episode_id=episode_id,
                n_eff=n_eff,
                fail_reason=fail_reason,
            )

        # Criterion 1: minimum distinct episodes for statistical power
        if n_episodes < p.min_episodes:
            return _result(False, f"episodes: {n_episodes} < {p.min_episodes}")

        # Criterion 2: rate (informative-only when min_tpm == 0; gates when > 0)
        if p.min_tpm > 0 and mean_tpm < p.min_tpm:
            return _result(False, f"rate: {mean_tpm:.2f} tpm < {p.min_tpm}")

        # Criterion 3: episode-level temporal dispersion
        if not np.isnan(episode_id) and episode_id > p.max_dispersion:
            return _result(False, f"dispersion: episode_ID={episode_id:.2f} > {p.max_dispersion}")

        return _result(True)

    def evaluate_series(
        self,
        event_series: pd.Series,
        month_index: np.ndarray,
        n_total_months: int,
    ) -> GateResult:
        """Evaluate the gate from a boolean Series using a pre-computed month index.

        Convenience wrapper around ``evaluate`` that converts the pandas
        Series to a numpy boolean array, accumulates per-month bar counts
        and per-month episode counts using the vectorised ``_count_by_month``
        helper.  The month index is pre-computed once per call to ``filter``
        and reused for all events to amortise the cost of the ``dt.to_period``
        conversion.

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
        episode_active = _to_episodes(active)
        episode_counts = _count_by_month(episode_active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months, episode_counts)

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
        episode_active = _to_episodes(active)
        episode_counts = _count_by_month(episode_active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months, episode_counts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_episodes(active: np.ndarray) -> np.ndarray:
    """Return a boolean array marking the first bar of each activation episode.

    An episode is a maximal run of consecutive True bars.  The episode-start
    is the first True bar following a False bar (or the series start).
    Collapsing runs to their start bar removes the inflation of the Index of
    Dispersion that occurs when a single regime event spans many consecutive
    bars (e.g. RSI < 30 persisting for a week on daily data).

    Parameters
    ----------
    active : np.ndarray
        Boolean activation array (dtype bool or uint8).

    Returns
    -------
    np.ndarray
        Boolean array of same length where True marks episode starts.
    """
    if len(active) == 0:
        return active.copy().astype(bool)
    shifted = np.empty(len(active), dtype=bool)
    shifted[0] = False
    shifted[1:] = active[:-1].astype(bool)
    return active.astype(bool) & ~shifted


def _build_month_index(timestamps: pd.Series) -> tuple[np.ndarray, int]:
    """Map each row to a zero-based integer month index.

    Converts the timestamp series to pandas Period objects (monthly
    frequency) and then to a compact integer array.  The mapping is stable
    across calls for the same dataset, making it safe to pre-compute once
    and reuse for all events.

    NaT timestamps are assigned the sentinel value ``-1`` so they are
    silently skipped by ``_count_by_month`` without raising a KeyError.

    Parameters
    ----------
    timestamps : pd.Series
        Datetime series aligned to the KPI table rows.

    Returns
    -------
    month_index : np.ndarray
        Integer array of shape (n_rows,) where each value is the zero-based
        index of the calendar month for that row, or ``-1`` for NaT rows.
    n_total_months : int
        Total number of distinct calendar months in the dataset.
    """
    periods = timestamps.dt.to_period("M")
    valid_mask = ~periods.isna()
    valid_periods = periods[valid_mask]
    if valid_periods.empty:
        return np.full(len(periods), -1, dtype=np.int32), 0
    unique_months = valid_periods.sort_values().unique()
    month_to_idx = {m: i for i, m in enumerate(unique_months)}
    month_index = np.full(len(periods), -1, dtype=np.int32)
    month_index[valid_mask.values] = [month_to_idx[p] for p in valid_periods]
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
    valid = month_index >= 0
    if valid.any():
        np.add.at(counts, month_index[valid], active[valid].astype(np.int32))
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
