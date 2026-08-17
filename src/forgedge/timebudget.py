"""TimeBudget — one temporal axis, purged and embargoed, for the whole session.

FORGE's modules each cut the timeline on their own: Event Discovery's
``train_ratio``, Alpha Discovery's ``train_ratio``, Rule Discovery's
walk-forward.  The cuts are honest *as boundaries*, but forward-looking
quantities cross them: the forward return at bar ``t`` reads closes up to
``t + h``, so the last ``h`` in-sample bars are scored on out-of-sample
prices, and the OOS confirmation partially re-uses the very price path that
derived the target.  The same overlap exists in the walk-forward: a trade
entered near the end of a train window exits inside the test window, so the
parameter selection "sees" test outcomes.

``TimeBudget`` centralises the cut:

* ``split`` — the first bar that is *not* in-sample;
* **purge** — the last ``min(h, purge_bars)`` IS bars are excluded from every
  measure at horizon ``h``, because their forward window crosses the split;
* **embargo** — the first ``embargo_bars`` OOS bars are additionally skipped
  (serial-correlation quarantine; default 0 — the purge already removes the
  mechanical overlap).

Modules stay independently usable: each accepts an optional ``time_budget``
and, when none is given, builds one from its own config (purge on by
default — it is the removal of a mechanical look-ahead, not a preference).

One axis, three modules that use it differently (F6, #180)
----------------------------------------------------------
``forge()`` builds one budget and threads it through all three.  It used to
forward only what the caller had passed — ``None`` by default — so each module
cut its own timeline and ``ForgeResult.time_budget`` reported Alpha
Discovery's axis as if it were the session's.  Under ``forge_preset()`` that
meant reporting a 70 % split for a session in which Event Discovery had used
100 % of the span.

The three do not use the axis identically, and the budget now says so instead
of leaving it to be inferred:

* **Event Discovery** cuts at :attr:`TimeBudget.event_split`, which defaults
  to the whole span.  That is deliberate, not an oversight: by the pipeline's
  first invariant M1 never observes the forward return, so seeing every bar
  leaks no information *about returns*.  What crosses is distributional — the
  percentiles are computed over the whole span — which is the weaker form the
  seventh invariant asks for anyway (thresholds are distributional, per asset
  and period).
* **Alpha Discovery** cuts at :attr:`TimeBudget.split` and honours the purge:
  it is the module that reads forward returns, so it is the one the purge
  exists for.
* **Rule Discovery** keeps its own walk-forward geometry — the split is *not*
  its origin.  Forcing it to start at the session split leaves too little
  span to form folds: on the reference 28-month fixture it drops the
  ``balanced`` preset from four windows to none, which would remove the
  very gate the fifth invariant makes the tradeable verdict depend on.
  Instead each fold records whether its test window falls inside the
  in-sample region, via :meth:`TimeBudget.is_in_sample`, so the overlap
  between M3's OOS evidence and the data M2 fit the target on is measured
  and reported rather than invisible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TimeBudget:
    """A purged, embargoed IS/OOS cut of an ``n_bars``-long timeline.

    Attributes
    ----------
    n_bars : int
        Length of the timeline, in bars.
    split : int
        Index of the first out-of-sample bar (IS = ``[0, split)``).  Alpha
        Discovery's cut, and the boundary Rule Discovery measures its folds
        against.
    event_split : int or None
        Where Event Discovery cuts, when that differs from ``split``.  ``None``
        means "follow ``split``" — the behaviour of every budget built before
        this field existed, and what a caller who hands ``forge()`` an explicit
        budget still gets.  ``n_bars`` means the whole span, which is what
        :func:`forgedge.forge` writes for the preset's documented
        ``DiscoveryConfig.train_ratio=1.0``; see the module docstring for why
        that is a choice rather than a leak.
    horizon_bars : int
        The largest forward horizon (in bars) any measure will read.  Used as
        the default purge width.
    purge_bars : int
        Quarantine before the split: at horizon ``h`` the IS rows
        ``[split - min(h, purge_bars), split)`` are excluded because their
        forward window crosses into OOS.  ``0`` disables purging.
    embargo_bars : int
        Quarantine after the split: OOS measures start at
        ``split + embargo_bars``.  ``0`` (default) disables the embargo.
    """

    n_bars: int
    split: int
    horizon_bars: int = 0
    purge_bars: int = 0
    embargo_bars: int = 0
    event_split: Optional[int] = None

    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        n_bars: int,
        train_ratio: float = 0.7,
        horizon_bars: int = 0,
        purge_bars: Optional[int] = None,
        embargo_bars: int = 0,
        event_train_ratio: Optional[float] = None,
    ) -> "TimeBudget":
        """Build a budget from a train ratio, defaulting the purge to the horizon.

        ``purge_bars=None`` (default) sets the purge width to ``horizon_bars``
        — exactly the rows whose forward window crosses the split.  Pass
        ``purge_bars=0`` to disable purging (the pre-TimeBudget behaviour).

        ``event_train_ratio`` is Event Discovery's own ratio, recorded so the
        budget can state M1's axis rather than leave it to be inferred from
        ``split``, which M1 may not use.  ``None`` (default) leaves
        ``event_split`` unset, i.e. M1 follows ``split`` — the behaviour of
        every budget built before this parameter existed.
        """
        n_bars = int(n_bars)
        split = min(max(int(round(n_bars * float(train_ratio))), 0), n_bars)
        purge = int(horizon_bars) if purge_bars is None else int(purge_bars)
        event_split = None
        if event_train_ratio is not None:
            ratio = float(event_train_ratio)
            event_split = (
                n_bars if ratio >= 1.0
                else min(max(int(n_bars * ratio), 1), n_bars)
            )
        return cls(
            n_bars=n_bars,
            split=split,
            horizon_bars=int(horizon_bars),
            purge_bars=max(purge, 0),
            embargo_bars=max(int(embargo_bars), 0),
            event_split=event_split,
        )

    # ------------------------------------------------------------------

    @property
    def oos_start(self) -> int:
        """First bar of the embargoed OOS window."""
        return min(self.split + self.embargo_bars, self.n_bars)

    @property
    def has_oos(self) -> bool:
        return self.oos_start < self.n_bars

    def purge_slice(self, h: int) -> Tuple[int, int]:
        """IS row range ``[lo, hi)`` to exclude at horizon ``h``.

        Empty (``lo == hi``) when purging is disabled or there is no OOS tail
        (``split == n_bars`` — the crossing rows are then off the end of the
        data and already NaN).
        """
        if self.purge_bars <= 0 or self.split >= self.n_bars:
            return (self.split, self.split)
        lo = max(0, self.split - min(int(h), self.purge_bars))
        return (lo, self.split)

    @property
    def event_split_idx(self) -> int:
        """Event Discovery's cut as an index — ``split`` when unset."""
        return self.split if self.event_split is None else int(self.event_split)

    def is_in_sample(self, index: int) -> bool:
        """Whether bar ``index`` lies in the in-sample region.

        Rule Discovery's folds ask this of their test windows: a fold whose
        test window answers ``True`` is scoring the contract's target on data
        Alpha Discovery fit that target on, and the fold says so instead of
        the overlap being invisible (F6, #180).
        """
        return int(index) < self.split

    def describe(self) -> str:
        """One-line summary of the axis each module actually uses."""
        if self.event_split is None:
            m1 = f"M1 IS [0, {self.split}) (follows the split)"
        elif self.event_split >= self.n_bars:
            m1 = "M1 whole span, by choice (invariant #1: no forward return observed)"
        else:
            m1 = f"M1 IS [0, {self.event_split})"
        return (
            f"time budget: {self.n_bars} bars — {m1}; "
            f"M2 IS [0, {self.split}), purge {self.purge_bars} bar(s) before "
            f"the split, embargo {self.embargo_bars} bar(s), OOS "
            f"[{self.oos_start}, {self.n_bars}); "
            f"M3 keeps its own walk-forward geometry and reports which folds "
            f"test inside [0, {self.split})"
        )
