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
``forge()`` builds a single budget and threads it through, so the session
shares one axis.
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
        Index of the first out-of-sample bar (IS = ``[0, split)``).
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

    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        n_bars: int,
        train_ratio: float = 0.7,
        horizon_bars: int = 0,
        purge_bars: Optional[int] = None,
        embargo_bars: int = 0,
    ) -> "TimeBudget":
        """Build a budget from a train ratio, defaulting the purge to the horizon.

        ``purge_bars=None`` (default) sets the purge width to ``horizon_bars``
        — exactly the rows whose forward window crosses the split.  Pass
        ``purge_bars=0`` to disable purging (the pre-TimeBudget behaviour).
        """
        n_bars = int(n_bars)
        split = min(max(int(round(n_bars * float(train_ratio))), 0), n_bars)
        purge = int(horizon_bars) if purge_bars is None else int(purge_bars)
        return cls(
            n_bars=n_bars,
            split=split,
            horizon_bars=int(horizon_bars),
            purge_bars=max(purge, 0),
            embargo_bars=max(int(embargo_bars), 0),
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

    def describe(self) -> str:
        """One-line human-readable summary."""
        return (
            f"time budget: {self.n_bars} bars — IS [0, {self.split}), "
            f"purge {self.purge_bars} bar(s) before the split, "
            f"embargo {self.embargo_bars} bar(s), OOS [{self.oos_start}, "
            f"{self.n_bars})"
        )
