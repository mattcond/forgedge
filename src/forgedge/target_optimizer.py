"""TargetOptimizer — the target-first inversion of the FORGE pipeline (issue #100).

The standard FORGE pipeline is *event-centric*: Event Discovery mines events and
Alpha Discovery derives ``(h*, sell_pct*, direction*)`` independently per event.
``TargetOptimizer`` answers the complementary, *target-first* question:

    "Which events anticipate, by ``horizon`` bars, a ``min_return`` move on the
    ``side`` I care about?"

The user fixes the economic target up front (``TargetConfig(horizon, min_return,
side)``); the module then mines events and ranks them by how strongly they lift
the win rate of *that* target.  Fixing the target has two advantages over the
event-centric flow:

* **Statistical efficiency** — the multiple-comparisons cost is paid only over
  events, not over ``events × horizons × thresholds``, so FDR inflation drops.
* **Operational coherence** — every surviving event shares the same exit
  parameters ``(horizon, min_return, side)`` and is deployable as-is.

Pipeline
--------
::

    EventDiscovery (raw atoms, pre-gate)
        ↓
    1st pass — score atoms by lift on Y, prune lift < min_lift_atoms
        ↓  lift-positive atoms
    ANDComposer (guided — combines only lift-positive atoms, gate=None)
        ↓  AND compositions
    2nd pass — score atoms+compositions by lift, prune lift < min_lift_result
        ↓  lift-positive survivors
    ConsistencyGate (formal quality check)
        ↓  EventCandidates
    [optional] AlphaDiscovery (fixed-target mode) → AlphaContracts

Pruning lift-negative atoms before AND composition is lossless within the
pipeline: an atom with ``lift < 1`` on ``Y`` can only enter a positive-lift AND
by cherry-picking a degenerate low-support subset, which the ConsistencyGate
would reject anyway.  That guarantee holds **only at ``min_lift_atoms == 1.0``**:
a higher 1st-pass threshold removes moderately-positive atoms whose
*intersection* carries emergent lift, so use ``min_lift_result`` to shorten the
final list and leave ``min_lift_atoms`` at its default (issue #107).

This module is **standalone** — it does not touch ``forge()`` or
``ForgeResult``.  Its only public entry points are :meth:`TargetOptimizer.run`
and :meth:`TargetOptimizer.validate_oos`, plus :meth:`TargetOptimizer.discover_alpha`
for the optional Alpha Discovery hand-off.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .alpha_discovery.discovery import AlphaDiscovery
from .alpha_discovery.models import AlphaConfig, AlphaContract, TargetConfig
from .alpha_discovery.target import binary_target
from .event_discovery.and_composer import ANDComposer
from .event_discovery.consistency_gate import ConsistencyGate
from .event_discovery.discovery import DiscoveryConfig, EventDiscovery
from .event_discovery.models import EventCandidate, RawEvent

logger = logging.getLogger(__name__)

# Columns of the per-candidate scoring table returned by ``run`` /
# ``validate_oos``, in display order.
_RESULT_COLUMNS = [
    "event_id",
    "n_components",
    "expression",
    "n_activations",
    "win_rate_event",
    "win_rate_base",
    "lift",
    "z_score",
]


def _lift_score(
    event: pd.Series, target: pd.Series, min_activations: int
) -> Optional[dict]:
    """Two-proportion lift of an event's activations against a binary target.

    Parameters
    ----------
    event : pd.Series
        Boolean (0/1/NaN) activation series.
    target : pd.Series
        Binary target ``Y`` (1.0/0.0/NaN), aligned to ``event``'s index.
    min_activations : int
        Minimum number of active bars required to score; below this the
        conditional win rate is too noisy and ``None`` is returned.

    Returns
    -------
    dict or None
        ``None`` when the candidate cannot be scored (too few activations, no
        valid overlap, or a degenerate base rate).  Otherwise the raw scoring
        fields: ``n_activations``, ``win_rate_event`` (``p_event``),
        ``win_rate_base`` (``p_base``), ``lift`` and ``z_score``.
    """
    combined = pd.DataFrame({"event": event, "Y": target}).dropna()
    if combined.empty:
        return None

    active = combined["event"] == 1
    n_active = int(active.sum())
    if n_active < min_activations:
        return None

    p_base = float(combined["Y"].mean())
    # A degenerate base rate (no winners, or every bar a winner) carries no
    # information about the event's edge — lift is undefined or trivially 1.
    if p_base <= 0.0 or p_base >= 1.0:
        return None

    p_event = float(combined.loc[active, "Y"].mean())
    lift = p_event / p_base
    se = np.sqrt(p_base * (1.0 - p_base) / n_active)
    z = (p_event - p_base) / se if se > 0 else 0.0

    return {
        "n_activations": n_active,
        "win_rate_event": p_event,
        "win_rate_base": p_base,
        "lift": lift,
        "z_score": z,
    }


class TargetOptimizer:
    """Find the events that best predict a user-specified binary target.

    Parameters
    ----------
    train_df : pd.DataFrame
        In-sample OHLCV (+ feature) table.  Must carry a ``close`` column and a
        datetime column (default ``open_dt``) or a DatetimeIndex.  Event mining
        runs entirely on this frame; reserve a later slice for
        :meth:`validate_oos`.
    target_cfg : TargetConfig
        The fixed economic target ``(horizon, min_return, side)`` plus the
        scoring knobs ``min_activations``, ``min_lift_atoms`` and
        ``min_lift_result``.
    discovery_cfg : DiscoveryConfig, optional
        Event Discovery settings.  Defaults to a full-frame (``train_ratio=1.0``)
        discovery so mined atoms align row-for-row with the target; AND
        composition is re-run here under lift guidance, so ``max_and_components``
        is honoured but the gate is applied by this module, not by Event
        Discovery.

    Examples
    --------
    >>> opt = TargetOptimizer(train_df, TargetConfig(horizon=20, min_return=0.10, side="long"))
    >>> results = opt.run()          # ranked lift table, one row per survivor
    >>> oos = opt.validate_oos(full_df)
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        target_cfg: TargetConfig,
        discovery_cfg: Optional[DiscoveryConfig] = None,
    ):
        self.train_df = train_df
        self.target_cfg = target_cfg
        # Full-frame discovery: the whole train_df is in-sample, so raw atoms,
        # ed.df and the target share one index.  walk_forward stays off.
        self.discovery_cfg = discovery_cfg or DiscoveryConfig(train_ratio=1.0)

        # Populated by run().
        self._ed: Optional[EventDiscovery] = None
        self._candidates: Optional[List[EventCandidate]] = None
        self._results: Optional[pd.DataFrame] = None
        self._base_rate: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def base_rate(self) -> Optional[float]:
        """Unconditional win rate of the target on ``train_df`` (set by run())."""
        return self._base_rate

    @property
    def candidates(self) -> Optional[List[EventCandidate]]:
        """Gate-passing EventCandidates from the last run(), ranked by lift."""
        return self._candidates

    def run(self) -> pd.DataFrame:
        """Mine, lift-score and gate events; return the ranked survivor table.

        Returns
        -------
        pd.DataFrame
            One row per gate-passing candidate, sorted by ``lift`` descending,
            with columns ``event_id, n_components, expression, n_activations,
            win_rate_event, win_rate_base, lift, z_score``.  Empty (with the
            correct schema) when nothing survives.
        """
        cfg = self.target_cfg

        # ── Event Discovery — mine raw atoms (pre-gate) ───────────────────
        ed = EventDiscovery(self.train_df, config=self.discovery_cfg)
        ed.run()  # populates ed.raw_events and the post-normalisation ed.df
        self._ed = ed
        raw_events = ed.raw_events or []
        logger.info("TargetOptimizer: %d raw atoms from Event Discovery", len(raw_events))

        # ── Target Y on the post-normalisation close ──────────────────────
        close = ed.df["close"]
        target, base_rate = binary_target(
            close, cfg.horizon, cfg.min_return, cfg.side, cfg.target_mode,
            cfg.trend_sma_mult,
        )
        self._base_rate = base_rate
        logger.info(
            "TargetOptimizer: target +%.1f%% in %d bars (%s) — base rate %.1f%%",
            cfg.min_return * 100, cfg.horizon, cfg.side, base_rate * 100,
        )

        # The month index for the gate is taken from the discovery frame's index.
        timestamps = pd.Series(ed.df.index, index=ed.df.index)

        # ── 1st pass — lift-prune the raw atoms (controls AND composition) ─
        # ``min_lift_atoms`` removes only harmful atoms; keeping it at 1.0 lets
        # two moderately-positive atoms combine into an emergent-lift AND that a
        # high single threshold would have suppressed.
        lift_atoms = self._prune_by_lift(raw_events, target, cfg.min_lift_atoms)
        logger.info(
            "TargetOptimizer: %d atoms with lift >= %.2f", len(lift_atoms), cfg.min_lift_atoms
        )

        # ── Guided AND composition over lift-positive atoms ───────────────
        gate = ConsistencyGate(self.discovery_cfg.gate_params)
        composer = ANDComposer(gate)
        # gate=None: keep every volume-passing composition; the formal gate runs
        # once below over atoms and compositions together.
        compositions = composer.compose(
            lift_atoms,
            timestamps,
            max_components=self.discovery_cfg.max_and_components,
            gate=None,
        )
        logger.info("TargetOptimizer: %d AND compositions to score", len(compositions))

        # ── 2nd pass — lift-prune the final result set ────────────────────
        # ``min_lift_result`` filters the candidates actually returned (surviving
        # atoms *and* compositions), independently of how aggressively the 1st
        # pass pruned the composition input.
        survivors = self._prune_by_lift(
            lift_atoms + compositions, target, cfg.min_lift_result
        )
        logger.info(
            "TargetOptimizer: %d survivors with lift >= %.2f",
            len(survivors), cfg.min_lift_result,
        )

        # ── Formal ConsistencyGate over all lift-positive survivors ───────
        passing = gate.filter(survivors, timestamps)
        logger.info("TargetOptimizer: %d candidates passed the ConsistencyGate", len(passing))

        candidates = [ed._to_candidate(ev, idx, timestamps) for idx, ev in enumerate(passing)]

        # ── Score the final candidates and rank ───────────────────────────
        rows = []
        for cand in candidates:
            score = _lift_score(cand.event_series, target, cfg.min_activations)
            if score is None:
                continue
            rows.append(self._row(cand, score))

        results = self._frame(rows)
        # Keep candidates aligned (and ordered) with the result table for the
        # downstream hand-offs (validate_oos, discover_alpha).
        by_id = {c.event_id: c for c in candidates}
        self._candidates = [by_id[eid] for eid in results["event_id"]] if not results.empty else []
        self._results = results
        return results

    def validate_oos(self, full_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Re-score the top-``k`` in-sample events on a full (OOS) dataset.

        Each top event is re-evaluated as an activation function on ``full_df``
        and scored against the target recomputed on ``full_df``'s close.  A lift
        that holds (or improves) out of sample is evidence the edge is not an
        in-sample artefact.

        OOS activations below ``target_cfg.min_activations`` are **always
        scored** — the lift estimate is noisier but the information is still
        useful.  A ``logger.warning`` is emitted for each such event, and the
        result table carries a boolean ``low_sample_oos`` column so callers can
        filter or flag these rows programmatically.

        Parameters
        ----------
        full_df : pd.DataFrame
            The full dataset (train + later bars).  Must carry the same native
            columns as ``train_df``.
        top_k : int
            Number of top-lift in-sample events to re-score.  Default ``10``.

        Returns
        -------
        pd.DataFrame
            The in-sample rows for the top-``k`` events, left-joined with their
            out-of-sample counterparts (``*_oos`` columns) plus a boolean
            ``low_sample_oos`` flag (``True`` when ``n_activations_oos <
            target_cfg.min_activations``).

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called.
        """
        if self._results is None or self._candidates is None:
            raise RuntimeError("Call run() before validate_oos().")

        cfg = self.target_cfg
        is_top = self._results.head(top_k)
        if is_top.empty:
            return is_top.copy()

        close_full = full_df["close"]
        target_full, _ = binary_target(
            close_full, cfg.horizon, cfg.min_return, cfg.side, cfg.target_mode,
            cfg.trend_sma_mult,
        )

        cand_by_id = {c.event_id: c for c in self._candidates}
        rows = []
        for eid in is_top["event_id"]:
            cand = cand_by_id.get(eid)
            if cand is None:
                continue
            ev_full = cand.apply(full_df)
            ev_full.index = target_full.index
            # Count OOS activations independently of scoring so n_activations_oos
            # is always populated (even when the base rate is degenerate).
            combined = pd.DataFrame({"event": ev_full, "Y": target_full}).dropna()
            n_act_oos = int((combined["event"] == 1).sum())
            low_sample = n_act_oos < cfg.min_activations
            if low_sample and n_act_oos > 0:
                logger.warning(
                    "validate_oos: event '%s' has only %d OOS activations "
                    "(min_activations=%d) — lift estimate is noisy.",
                    eid, n_act_oos, cfg.min_activations,
                )
            # Always attempt scoring with min_activations=1 so OOS lift is never
            # silently suppressed by the sample-size gate.
            score = _lift_score(ev_full, target_full, min_activations=1)
            row = {"event_id": eid, "n_activations_oos": n_act_oos, "low_sample_oos": low_sample}
            if score is None:
                # No scorable activations (zero activations or degenerate base rate).
                row.update(win_rate_oos=np.nan, base_rate_oos=np.nan, lift_oos=np.nan, z_oos=np.nan)
            else:
                row.update(
                    win_rate_oos=round(score["win_rate_event"], 4),
                    base_rate_oos=round(score["win_rate_base"], 4),
                    lift_oos=round(score["lift"], 3),
                    z_oos=round(score["z_score"], 2),
                )
            rows.append(row)

        oos = pd.DataFrame(rows)
        return is_top.merge(oos, on="event_id", how="left")

    def discover_alpha(self, config: Optional[AlphaConfig] = None) -> List[AlphaContract]:
        """Hand the surviving candidates to Alpha Discovery in fixed-target mode.

        Runs the standard Alpha Discovery measurement chain (IC, regime, OOS,
        FDR, scoring) against the *fixed* target instead of deriving one per
        event, returning one contract per candidate.

        Parameters
        ----------
        config : AlphaConfig, optional
            Base configuration.  Its ``fixed_target`` is always set from this
            optimizer's :class:`TargetConfig` (overriding any provided value) so
            the measurement stays target-first.  When ``None``, a default
            ``AlphaConfig`` is used.

        Returns
        -------
        list[AlphaContract]
            One contract per surviving candidate (promoted and rejected).

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called, or produced no candidates.
        """
        if self._ed is None or not self._candidates:
            raise RuntimeError("Call run() first; no surviving candidates to evaluate.")

        cfg = config or AlphaConfig()
        cfg.fixed_target = self.target_cfg
        # Keep the binary-target definition consistent with the optimizer's
        # IS/OOS scoring (PROJ vs ABS, and the trend SMA window).
        cfg.target_mode = self.target_cfg.target_mode
        cfg.trend_sma_mult = self.target_cfg.trend_sma_mult
        ad = AlphaDiscovery(self._ed.df, self._candidates, cfg)
        return ad.run()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_by_lift(
        self, events: List[RawEvent], target: pd.Series, min_lift: float
    ) -> List[RawEvent]:
        """Keep only the events whose lift on ``target`` clears ``min_lift``.

        The threshold is passed explicitly because the two pruning passes use
        different ones: ``min_lift_atoms`` guards the AND-composition input, while
        ``min_lift_result`` filters the final result set (see issue #107).
        """
        cfg = self.target_cfg
        kept: List[RawEvent] = []
        for ev in events:
            score = _lift_score(ev.series, target, cfg.min_activations)
            if score is not None and score["lift"] >= min_lift:
                kept.append(ev)
        return kept

    @staticmethod
    def _row(cand: EventCandidate, score: dict) -> dict:
        """Assemble one display row from a candidate and its lift score."""
        return {
            "event_id": cand.event_id,
            "n_components": len(cand.components),
            "expression": cand.expression,
            "n_activations": score["n_activations"],
            "win_rate_event": round(score["win_rate_event"], 4),
            "win_rate_base": round(score["win_rate_base"], 4),
            "lift": round(score["lift"], 3),
            "z_score": round(score["z_score"], 2),
        }

    @staticmethod
    def _frame(rows: List[dict]) -> pd.DataFrame:
        """Build the ranked result frame (empty-safe, fixed schema)."""
        if not rows:
            return pd.DataFrame(columns=_RESULT_COLUMNS)
        return (
            pd.DataFrame(rows, columns=_RESULT_COLUMNS)
            .sort_values("lift", ascending=False)
            .reset_index(drop=True)
        )
