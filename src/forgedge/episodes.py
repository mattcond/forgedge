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

    Accepts a 2D batch (``(K, n_rows)``) as well as a single 1D series —
    the vectorized batch path (no per-row Python loop) is row-for-row
    identical to calling this function once per row, so callers that need
    to evaluate episode structure over many candidate series at once (e.g.
    ``ANDComposer``, #226) never have to re-derive this logic (#134's
    bridging semantics) themselves.

    Parameters
    ----------
    active : np.ndarray
        Boolean activation array (dtype bool or uint8), shape ``(n_rows,)``
        or ``(K, n_rows)``.
    gap : int
        Maximum interruption length (in bars) bridged within an episode.

    Returns
    -------
    np.ndarray
        Boolean array of the same shape as ``active``, True only at the
        first bar of each episode.  The marks are positioned on the
        original (un-bridged) activations, so a bridged gap bar is never
        itself marked.
    """
    active = np.asarray(active).astype(bool)
    if active.ndim == 2:
        return _episode_starts_batch(active, gap)
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


def _episode_starts_batch(active: np.ndarray, gap: int) -> np.ndarray:
    """Vectorized ``episode_starts`` across every row of a ``(K, n_rows)``
    matrix at once — no per-row Python loop over bridged holes.

    Bridging a gap of length <= ``gap`` between two active bars is a
    morphological closing: for each inactive position, find the nearest
    active bar before and after it (via a running max of "last seen active
    index", forward and reversed) and bridge when that interior hole is
    short enough.  Verified row-for-row identical to the 1D loop version
    above across randomized bursty fixtures at gap in {0, 1, 2, 3} (#226).

    Memory-conscious by construction (#228): the naive version of this
    computation keeps five or six ``(K, n_rows)`` ``int64`` temporaries alive
    at once, which peaked at ~4.8 GB on a realistic 5 000-pair x 23 352-row
    chunk (a ~41x amplification over the 117 MB input) — large enough to OOM
    a real ``ANDComposer.compose()`` call under permissive gate params, where
    most of a chunk survives the volume pre-filter and reaches this function
    at full size. Row indices fit comfortably in ``int32`` (no realistic bar
    count approaches 2**31), and ``np.maximum.accumulate(..., out=...)``
    reuses its input buffer instead of allocating a second one, so this
    version peaks at roughly half the per-array cost with fewer arrays alive
    at once (each intermediate is ``del``-eted the moment nothing downstream
    still needs it) — measured ~1.7x-2x lower peak on the same chunk shape.
    ``and_composer._pair_chunk_size`` (#228) is the complementary, more
    important fix: it bounds ``K`` itself against ``n_rows`` so peak memory
    stays under a fixed budget regardless of dataset length, rather than
    relying on a per-call constant-factor improvement alone.
    """
    K, n = active.shape
    if n == 0:
        return active.copy()

    bridged = active
    if gap > 0 and n > 1:
        idx = np.arange(n, dtype=np.int32)

        last_active = np.where(active, idx, np.int32(-1))
        np.maximum.accumulate(last_active, axis=1, out=last_active)

        next_active_rev = np.where(active[:, ::-1], idx, np.int32(-1))
        np.maximum.accumulate(next_active_rev, axis=1, out=next_active_rev)
        no_next = next_active_rev[:, ::-1] == -1
        next_active = np.where(no_next, np.int32(n), np.int32(n - 1) - next_active_rev[:, ::-1])
        del next_active_rev

        has_prev = last_active >= 0
        hole_len = next_active - last_active - 1
        del last_active, next_active

        bridge_mask = (~active) & has_prev & (~no_next) & (hole_len <= gap)
        del has_prev, no_next, hole_len
        bridged = active | bridge_mask
        del bridge_mask

    prev = np.concatenate([np.zeros((K, 1), dtype=bool), bridged[:, :-1]], axis=1)
    starts_mask = bridged & ~prev
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
