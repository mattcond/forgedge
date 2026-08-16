"""Temporal structure of activity: episodes and concurrency.

Two questions that look alike and are not, kept together because the difference
between them is the point (issue #168):

* **Episodes** group activations by *signal*.  A persistent state — a multi-bar
  ``RSI < 30`` stretch — is one thing happening, not five, and counting bars
  inflates every per-bar rate and dispersion statistic (issue #134).
* **Concurrency** counts positions open on the *same price path*.  Two trades
  from two clearly separate episodes still overlap whenever the holding period
  outruns the gap between them.

They answer different questions and generally disagree.  On the EURJPY case in
issue #168: 120 signal bars, **76 episodes**, and an average of **3.71
concurrently open positions** — an effective sample nearer 32 than 118.

Which one you want depends on what you are about to do:

* sizing capital → concurrency (how many positions must be funded at once);
* measuring how often a signal *fires* → episodes;
* statistical inference → concurrency, because overlapping trades share price
  path and are not independent observations.

This module lives at the top level rather than inside Event Discovery because
both modules need it and neither owns it: M1 gates on episode counts, M3
measures both on the trade ledger the published rule actually produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["episode_starts", "episode_ids", "concurrency", "ConcurrencyStats"]


def episode_starts(active: np.ndarray, gap: int = 1) -> np.ndarray:
    """Mark the first bar of each activation *episode*.

    An *episode* is a stretch of activations in which interruptions of at most
    ``gap`` bars do not start a new episode.  Collapsing each episode to its
    first bar lets dispersion be measured per-episode rather than per-bar,
    removing the inflation that persistent states (a multi-bar ``RSI < 30``
    stretch) cause in per-bar monthly counts (issue #134).

    With ``gap=1`` (default) a single missing bar inside a run is bridged — on
    daily data a one-day interruption is not a new event.  ``gap=0`` gives
    strict consecutive runs.

    Parameters
    ----------
    active : np.ndarray
        Boolean activation array (dtype bool or uint8).
    gap : int
        Maximum interruption length (in bars) bridged within an episode.

    Returns
    -------
    np.ndarray
        Boolean array, True only at the first bar of each episode.  The marks
        are positioned on the original (un-bridged) activations, so a bridged
        gap bar is never itself marked.
    """
    active = np.asarray(active).astype(bool)
    if active.size == 0:
        return active

    # Bridge interruptions of <= gap bars between two active stretches.
    bridged = active.copy()
    if gap > 0:
        idx = np.flatnonzero(active)
        if idx.size > 1:
            ends = idx[:-1]
            starts = idx[1:]
            hole = starts - ends - 1  # inactive bars between consecutive activations
            fill_mask = (hole > 0) & (hole <= gap)
            for e, s in zip(ends[fill_mask], starts[fill_mask]):
                bridged[e + 1: s] = True

    prev = np.empty_like(bridged)
    prev[0] = False
    prev[1:] = bridged[:-1]
    starts_mask = bridged & ~prev
    # Position the marks back on real activations: a bridged gap bar that opens
    # a bridged run is not itself an activation and must not be marked.
    return starts_mask & active


def episode_ids(active: np.ndarray, gap: int = 1) -> np.ndarray:
    """Per-bar episode index, ``-1`` on bars that are not activations.

    Same episode semantics as :func:`episode_starts`; this returns the label
    rather than the boundary, so a trade opened on a given bar can be attributed
    to the episode that produced it.

    Bars bridged over a short interruption carry the surrounding episode's id
    only if they are themselves active — an inactive bridged bar stays ``-1``,
    since nothing happened there.
    """
    active = np.asarray(active).astype(bool)
    out = np.full(active.size, -1, dtype=np.int64)
    if active.size == 0:
        return out
    starts = episode_starts(active, gap)
    # Running episode number, incremented at each start and carried forward.
    labels = np.cumsum(starts) - 1
    out[active] = labels[active]
    return out


@dataclass(frozen=True)
class ConcurrencyStats:
    """How many positions a rule holds open at once.

    Attributes
    ----------
    mean : float
        Average number of open positions **over the bars where at least one is
        open**.  Idle stretches are excluded deliberately: including them would
        answer "how busy is this rule across the whole history", while the
        question capital sizing asks is "when this rule is working, how many
        positions am I funding".  ``nan`` when there are no trades.

        It is also the divisor that turns a nominal trade count into an
        effective one — ``total_trades / mean`` — which is what the inferential
        quantities need (F16, #177).
    peak : int
        Largest number simultaneously open.  The number that decides whether a
        rule is deployable at all with a given account.
    occupied_bars : int
        Bars with at least one position open.
    position_bars : int
        Total open-position bar-count (the area under the occupancy curve).
    n_trades : int
        Number of intervals measured — carried so ``effective_trades`` is
        answerable from this object alone.
    """

    mean: float
    peak: int
    occupied_bars: int
    position_bars: int
    n_trades: int

    @property
    def effective_trades(self) -> float:
        """``n_trades / mean`` — the sample size the overlap actually supports.

        Not used to restate the economics: profit factor, expectancy and net
        gain stay nominal, because they are reproducible in production given the
        capital to fund the concurrent positions (#168's own non-goal).  This is
        for the quantities that assume independent observations, and it is what
        F16 (#177) will consume.
        """
        if not np.isfinite(self.mean) or self.mean <= 0:
            return float("nan")
        return self.n_trades / self.mean


def concurrency(
    open_rn: np.ndarray,
    close_rn: np.ndarray,
    n_bars: Optional[int] = None,
) -> ConcurrencyStats:
    """Occupancy statistics for a set of ``[open, close]`` bar intervals.

    Both endpoints are **inclusive**: a trade filled and exited on the same bar
    occupies that one bar.  ``run_backtest`` produces exactly this — ``fill_rn``
    and ``exit_rn`` — and a same-session round trip (``target_h=0``) is a
    documented, legal case.

    Computed with a difference array rather than pairwise interval overlap: one
    pass over the bars instead of ``O(trades^2)``, which matters because this
    runs on every cell of the operational grid.
    """
    open_rn = np.asarray(open_rn, dtype=np.int64).ravel()
    close_rn = np.asarray(close_rn, dtype=np.int64).ravel()
    if open_rn.size == 0:
        return ConcurrencyStats(mean=float("nan"), peak=0, occupied_bars=0,
                                position_bars=0, n_trades=0)
    if open_rn.size != close_rn.size:
        raise ValueError(
            f"open_rn and close_rn must be the same length, got "
            f"{open_rn.size} and {close_rn.size}"
        )

    span = int(max(n_bars or 0, int(close_rn.max()) + 2))
    delta = np.zeros(span + 1, dtype=np.int64)
    np.add.at(delta, np.clip(open_rn, 0, span - 1), 1)
    np.add.at(delta, np.clip(close_rn + 1, 0, span), -1)
    occupancy = np.cumsum(delta)[:span]

    occupied = occupancy > 0
    occupied_bars = int(occupied.sum())
    position_bars = int(occupancy.sum())
    mean = position_bars / occupied_bars if occupied_bars else float("nan")
    # Deliberately unrounded: `effective_trades` divides by this, and rounding
    # here turns an exact 2.25 into 2.2500005625.  Rounding belongs at the point
    # of display, not at the point of measurement.
    return ConcurrencyStats(
        mean=float(mean),
        peak=int(occupancy.max()),
        occupied_bars=occupied_bars,
        position_bars=position_bars,
        n_trades=int(open_rn.size),
    )
