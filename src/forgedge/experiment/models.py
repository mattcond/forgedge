"""Config and result dataclasses for :mod:`forgedge.experiment`.

See ``step_wise_discovery.py``'s module docstring for the algorithm these
describe. Kept separate from the orchestration file per the repo's own
per-module convention (``models.py`` + an orchestration file — see the
``forgedge`` skill's *Contributing* section).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..event_discovery.models import EventCandidate, GateParams
from ..alpha_discovery.models import AlphaContract
from ..config_report import ConfigReport
from ..rule_discovery.models import RuleDiscoveryResponse
from ..timebudget import TimeBudget
from .redundancy import DEFAULT_MAX_ABS_CORR, DEFAULT_MAX_JACCARD

__all__ = [
    "StepWiseDiscoveryConfig",
    "SeedAttempt",
    "ChainResult",
    "StepWiseDiscoveryResult",
]


@dataclass
class StepWiseDiscoveryConfig:
    """Tunable parameters of the step-wise partition & compose search.

    None of these are asset- or timeframe-specific by design — anything
    that *is* (fee, mfe_floor, the KPI table's own indicator periods,
    which timeframe-native OU half-life window to use upstream when
    building the KPI table) belongs on the caller's
    ``event_discovery_config``/``alpha_config``/``rule_discovery_config``,
    resolved before construction, not here. This config only governs the
    search algorithm itself, so the same instance is meant to be reused
    across assets/timeframes.

    Attributes
    ----------
    n_seeds : int
        Number of distinct top-ranked, hold-out-confirmed seeds to search
        from. Each seed starts its own independent partition & compose
        chain.
    depth : int
        Maximum AND-composition depth per chain (``1`` = seed alone never
        composes; ``2`` = seed + at most one child; ...).
    train_ratio : float
        Fraction of the KPI table reserved for the OUTER hold-out split
        (never touched during search — see ``confirms_on_holdout`` in the
        orchestration file). This is independent of, and in addition to,
        any ``train_ratio``/walk-forward split the passed-in discovery/
        rule-discovery configs use internally.
    outer_horizon_bars : int or None
        Purge width for the outer hold-out split — the last
        ``outer_horizon_bars`` in-sample bars are excluded because their
        forward window crosses the split (same purge concept as
        ``TimeBudget`` itself, see ``forgedge.timebudget``). ``None``
        (default) derives it as ``max(alpha_config.horizon_grid)`` from the
        RESOLVED alpha config — the same quantity ``forge()`` itself uses
        to size its own internal budget — so a caller who doesn't have a
        more specific reason to pick a number gets a reasonable one for
        free. Pass an explicit value when you have one (e.g. an
        OU-half-life-derived scale from building the KPI table) — this
        class has no opinion on where that number should come from, only
        on how it's used once given.
    outer_embargo_bars : int or None
        Serial-correlation quarantine after the outer split. ``None``
        (default) derives it as the RESOLVED ``alpha_config.embargo_bars``,
        same rationale as ``outer_horizon_bars``.
    retain_ratio_floor : float
        Absolute floor on ``retain_ratio_min_k`` regardless of partition
        size (see ``StepWiseDiscovery._retain_ratio_min_for``).
    max_constituent_jaccard : float
        Boolean-activation-set redundancy threshold — see
        ``forgedge.experiment.redundancy``.
    max_constituent_abs_corr : float
        Continuous-series redundancy threshold — see
        ``forgedge.experiment.redundancy``.
    child_gate : GateParams or None
        Consistency Gate applied to child candidates evaluated on a
        partition. ``None`` (default) uses a gate deliberately looser than
        a typical session gate (``min_tpm=0.25, dispersion_margin=3.0,
        min_episodes=4``) — a sub-population search needs a permissive
        local gate, but not so permissive that a rare child intersected
        with an already-rare parent starves the composed rule of enough
        activations for Alpha Discovery to derive a target at all (the
        failure mode this default was raised to fix).
    min_composed_activations : int
        A parent-AND-child candidate below this many activations on the
        full search frame is rejected before even trying Alpha Discovery
        on it — fails fast with a clear reason instead of an opaque M2
        "no derivable target".
    min_local_partition_rows : int
        A rolling-transform child candidate whose local (partition-
        restricted) series has fewer than this many non-NaN rows is
        skipped — too little data to calibrate a threshold on.
    strict : bool
        Passed through to ``forge()``/``config_report()`` — raise on a
        ``FAIL``-level configuration incoherence rather than proceeding
        (see the ``forgedge`` invariant: "a configuration that cannot
        produce a verdict fails loudly, not silently").
    """

    n_seeds: int = 3
    depth: int = 2
    train_ratio: float = 0.80
    outer_horizon_bars: Optional[int] = None
    outer_embargo_bars: Optional[int] = None
    retain_ratio_floor: float = 0.2
    max_constituent_jaccard: float = DEFAULT_MAX_JACCARD
    max_constituent_abs_corr: float = DEFAULT_MAX_ABS_CORR
    child_gate: Optional[GateParams] = None
    min_composed_activations: int = 20
    min_local_partition_rows: int = 10
    strict: bool = True


@dataclass
class SeedAttempt:
    """One candidate examined during seed selection, accepted or not.

    Kept regardless of outcome so a caller can audit *why* a high-ranked
    candidate was skipped (hold-out disconfirmation vs. cross-seed
    redundancy) instead of only seeing the seeds that made it through.
    """

    family: str
    alpha_id: str
    verdict: str
    composite_score: float
    accepted: bool
    reason: str
    jaccard_vs_seeds: float = 0.0
    abs_corr_vs_seeds: float = 0.0


@dataclass
class ChainResult:
    """The outcome of one seed's partition & compose chain.

    ``candidate``/``contract``/``response`` describe the LAST CONFIRMED
    state of the chain — the seed alone if no composition ever confirmed
    on the hold-out, or the deepest composed rule that did. A composition
    attempt that fails hold-out confirmation is never written into these
    fields (see the orchestration file's reporting-order note) — it would
    silently discard a genuinely confirmed, shallower result.
    """

    chain_label: str
    seed_family: str
    candidate: EventCandidate
    contract: AlphaContract
    response: RuleDiscoveryResponse
    depth_reached: int
    confirm_reason: str
    stop_reason: str

    @property
    def expression(self) -> str:
        return self.candidate.expression

    @property
    def composite_score(self) -> float:
        return self.contract.alpha_score.composite_score

    @property
    def verdict(self) -> str:
        return self.response.verdict


@dataclass
class StepWiseDiscoveryResult:
    """Everything :meth:`StepWiseDiscovery.run` produces.

    Attributes
    ----------
    chains : list[ChainResult]
        One entry per seed, in seed-ranking order.
    seed_attempts : list[SeedAttempt]
        Every candidate examined during seed selection, in ranked order,
        whether or not it became a seed.
    feature_recurrence : dict[str, list[str]]
        Observation only, not a gate: which feature family recurred across
        which chains/depths during local child search — useful for
        spotting a dataset where one feature dominates every neighbourhood
        (see the ``forgedge.experiment`` design notes on this being an
        honest signal, not a bug to silence).
    search_df, holdout_df : pd.DataFrame
        The outer split actually used — ``holdout_df`` was never touched by
        any discovery/alpha/rule-discovery call in this run.
    time_budget : TimeBudget
        The outer split's budget (distinct from any internal walk-forward
        budget the passed-in ``rule_discovery_config`` uses).
    coherence : ConfigReport
        The resolved configuration this run actually executed with —
        same object shape as ``ForgeResult.coherence``.
    config : StepWiseDiscoveryConfig
        The algorithm parameters this run used.
    """

    chains: List[ChainResult]
    seed_attempts: List[SeedAttempt]
    feature_recurrence: Dict[str, List[str]]
    search_df: pd.DataFrame
    holdout_df: pd.DataFrame
    time_budget: TimeBudget
    coherence: ConfigReport
    config: StepWiseDiscoveryConfig

    def summary(self) -> pd.DataFrame:
        """One row per chain — the same shape as the scratch scripts' final table."""
        rows = [
            {
                "chain": c.chain_label,
                "seed_family": c.seed_family,
                "expression": c.expression,
                "depth_reached": c.depth_reached,
                "composite_score": c.composite_score,
                "verdict": c.verdict,
                "confirm_reason": c.confirm_reason,
                "stop_reason": c.stop_reason,
            }
            for c in self.chains
        ]
        return pd.DataFrame(rows)

    def edges(self) -> List[ChainResult]:
        """Chains whose final (last-confirmed) state is EDGE/PARTIAL-EDGE."""
        return [c for c in self.chains if c.response.is_edge]
