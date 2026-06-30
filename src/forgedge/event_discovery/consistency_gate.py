"""Step 4 — Consistency Gate.

Filters events based solely on the temporal distribution of their activations.
No forward return is observed here.

Default mode — two criteria:
  1. Rate        — minimum activations per month (MIN_TPM)
  2. Dispersion  — per-bar Index of Dispersion ≤ MAX_DISPERSION

Rare mode (``GateParams.admit_rare``, issue #134) — for rare episodic events:
  1. Episode power      — at least MIN_EPISODES distinct activation episodes
  2. Episode dispersion — episode-level Index of Dispersion ≤ MAX_DISPERSION
  The MIN_TPM rate floor is dropped (informational only) and dispersion is
  measured on episodes (consecutive-activation runs) instead of per bar, so a
  persistent multi-bar state no longer inflates the monthly variance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import GateParams, GateResult, RawEvent


class ConsistencyGate:
    """Applies the two-criterion temporal consistency filter to event series.

    The gate operates exclusively on the *when* of activations, never on
    *what follows* them.  This is essential to avoid look-ahead bias: the
    gate ensures events are structurally sound before any return-based
    analysis begins.

    Parameters
    ----------
    params : GateParams or None
        Configuration for both thresholds.  Defaults to
        ``GateParams()`` (min_tpm=2.0, max_dispersion=2.5).
    """

    def __init__(self, params: Optional[GateParams] = None):
        self.params = params or GateParams()

    def evaluate(
        self,
        active: np.ndarray,
        period_counts: np.ndarray,
        n_total_months: int,
        month_index: Optional[np.ndarray] = None,
    ) -> GateResult:
        """Evaluate the gate criteria given pre-computed counts.

        Default mode (``admit_rare=False``) — two criteria, checked in order:

        1. **Rate** (``n_activations / n_total_months >= min_tpm``): rejects
           events whose average activation rate is too low.
        2. **Dispersion** (per-bar ``Var(monthly_counts) / Mean(monthly_counts)
           <= max_dispersion``): rejects over-dispersed events.

        Rare mode (``admit_rare=True``, issue #134) — the rate floor is
        dropped and dispersion is measured on **episodes** (maximal runs of
        consecutive active bars collapsed to their first bar):

        1. **Episode power** (``n_episodes >= min_episodes``): replaces the
           rate floor as the statistical-power guard.
        2. **Episode dispersion** (episode-level ID ``<= max_dispersion``):
           a persistent multi-bar state (e.g. a 3–5 bar ``RSI < 30`` stretch)
           no longer inflates the per-bar monthly variance, so genuinely
           well-distributed rare events pass.

        Episode metrics (``n_episodes``, ``episode_index_of_dispersion``,
        ``n_eff``) are reported as diagnostics whenever ``month_index`` is
        supplied, regardless of mode.

        Parameters
        ----------
        active : np.ndarray
            Boolean array (True = activated) of length = n_rows in the dataset.
        period_counts : np.ndarray
            Per-month activation counts, length = n_total_months.
        n_total_months : int
            Total number of calendar months spanned by the dataset.
        month_index : np.ndarray or None
            Per-row month index (as from ``_build_month_index``).  Required to
            compute episode-level metrics; when None the episode ID / n_eff are
            left NaN and ``admit_rare`` falls back to a per-bar episode count
            with no monthly dispersion (it still drops the rate floor).

        Returns
        -------
        GateResult
            ``passed=True`` if all criteria are satisfied, else
            ``passed=False`` with ``fail_reason`` describing the first
            failing criterion.
        """
        p = self.params
        n_act = int(active.sum())

        # Diagnostic fields — computed unconditionally for GateResult
        n_active_months = int((period_counts > 0).sum()) if len(period_counts) > 0 else 0
        max_month_count = int(period_counts.max()) if len(period_counts) > 0 and n_act > 0 else 0
        max_conc = max_month_count / n_act if n_act > 0 else float("nan")
        mean_tpm = n_act / n_total_months if n_total_months > 0 else 0.0

        # Per-bar Index of Dispersion: Var(monthly_counts) / Mean(monthly_counts)
        # When n_total_months <= 1 there is no temporal spread to measure;
        # set id_score = 0.0 so the dispersion criterion trivially passes.
        if n_total_months > 1 and n_act > 0:
            mu = float(period_counts.mean())
            var = float(period_counts.var(ddof=1))
            id_score = float(var / mu) if mu > 0 else float("inf")
        else:
            id_score = 0.0

        # Episode metrics (diagnostic; the gate acts on them only in rare mode)
        episodes = _episode_starts(active)
        n_episodes = int(episodes.sum())
        episode_id = float("nan")
        n_eff = float("nan")
        if month_index is not None and n_total_months > 1 and n_episodes > 0:
            epi_counts = _count_by_month(episodes, month_index, n_total_months)
            emu = float(epi_counts.mean())
            evar = float(epi_counts.var(ddof=1))
            episode_id = float(evar / emu) if emu > 0 else float("inf")
            # First-order design effect — deflates only, never inflates.
            n_eff = n_episodes / max(episode_id, 1.0)

        def _result(passed: bool, fail_reason: Optional[str] = None) -> GateResult:
            return GateResult(
                passed=passed,
                n_activations=n_act,
                n_active_months=n_active_months,
                max_monthly_share=max_conc,
                mean_tpm=mean_tpm,
                index_of_dispersion=id_score,
                n_episodes=n_episodes,
                episode_index_of_dispersion=episode_id,
                n_eff=n_eff,
                fail_reason=fail_reason,
            )

        if p.admit_rare:
            # Criterion 1: episode power (replaces the rate floor)
            if n_episodes < p.min_episodes:
                return _result(False, f"episodes: {n_episodes} < {p.min_episodes}")
            # Criterion 2: episode-level dispersion (skipped if not computable)
            if episode_id == episode_id and episode_id > p.max_dispersion:  # not NaN
                return _result(
                    False,
                    f"episode dispersion: ID={episode_id:.2f} > {p.max_dispersion}",
                )
            return _result(True)

        # Default mode — rate floor + per-bar dispersion
        if mean_tpm < p.min_tpm:
            return _result(False, f"rate: {mean_tpm:.2f} tpm < {p.min_tpm}")
        if id_score > p.max_dispersion:
            return _result(False, f"dispersion: ID={id_score:.2f} > {p.max_dispersion}")
        return _result(True)

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
        return self.evaluate(active, period_counts, n_total_months, month_index)

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
        active = (s1 & s2).astype(bool)
        period_counts = _count_by_month(active, month_index, n_total_months)
        return self.evaluate(active, period_counts, n_total_months, month_index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode_starts(active: np.ndarray) -> np.ndarray:
    """Mark the first bar of each maximal run of consecutive active bars.

    An *episode* is a stretch of contiguous ``True`` values; collapsing each
    run to its first bar lets dispersion be measured per-episode rather than
    per-bar, removing the inflation that persistent states (a multi-bar
    ``RSI < 30`` stretch) cause in per-bar monthly counts (issue #134).  A
    single-bar gap splits the run into two episodes (strict runs, matching the
    issue's definition).

    Parameters
    ----------
    active : np.ndarray
        Boolean activation array (dtype bool or uint8).

    Returns
    -------
    np.ndarray
        Boolean array, True only at the first bar of each consecutive run.
    """
    active = active.astype(bool)
    if active.size == 0:
        return active
    prev = np.empty_like(active)
    prev[0] = False
    prev[1:] = active[:-1]
    return active & ~prev

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
