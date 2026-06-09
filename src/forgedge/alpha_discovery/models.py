"""Core data structures for the Alpha Discovery module (FORGE Modulo 2).

Alpha Discovery receives ``EventCandidate`` artifacts from Event Discovery and
measures their predictive power against an economic target.  The output is the
**Alpha Contract** — the formal interface consumed by Rule Discovery.

The dataclasses here mirror, field by field, the contract format documented in
``docs/modules/AlphaDiscovery.md`` (Section 2).  Nothing in this module
modifies an event's threshold, window, or expression: the contract references
the Event Candidate exactly as received and only adds statistical measures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TargetDefinition:
    """The economic target every candidate is measured against (Step 1).

    These parameters do not change within a discovery session — changing them
    is a new session.  Event Discovery never sees them; Alpha Discovery is the
    first module that knows what "a useful event" means.

    Attributes
    ----------
    holding_period_h : int
        Forward horizon, in bars, over which the target is evaluated
        (``target_h``).
    sell_pct : float
        Return threshold that defines the binary target (e.g. ``0.04`` → the
        forward maximum must reach at least +4% for a ``long``; for a
        ``short`` the forward minimum must reach at least -4%).
    direction : {'long', 'short'}
        Trade direction.  ``long`` looks for upside within the horizon,
        ``short`` for downside.
    fee_per_side : float
        Informational only — recorded in the contract so Rule Discovery knows
        the assumed cost basis.  Alpha Discovery does not net it out (that is
        Rule Discovery's job).
    asset, exchange, timeframe : str
        Scope metadata copied verbatim into the contract.
    """

    holding_period_h: int = 24
    sell_pct: float = 0.04
    direction: str = "long"
    fee_per_side: float = 0.002
    asset: str = "ASSET"
    exchange: str = ""
    timeframe: str = "1H"


@dataclass
class PromotionThresholds:
    """Admission and promotion gates (Steps 3, 4 and the FDR control).

    Attributes
    ----------
    ic_min_abs : float
        Minimum ``|IC|`` of the underlying feature for a candidate to survive
        the IC admission gate (Step 3.4).
    ic_max_p : float
        Maximum IC p-value accepted at the admission gate.  A candidate is
        admitted when ``|IC| >= ic_min_abs`` **or** ``p_ic < ic_max_p`` is not
        required on its own — see :meth:`AlphaDiscovery` for the exact rule
        (the doc discards only when *both* are weak).
    min_lift : float
        Minimum win-rate lift over the base rate (Step 4.3), e.g. ``0.08``.
    min_cohens_d : float
        Minimum Cohen's d effect size (Step 4.3).
    max_p_value : float
        Maximum t-test p-value when FDR control is disabled.
    min_activations : int
        Minimum number of event activations for a stable estimate.
    use_fdr : bool
        When ``True`` (default), promotion uses Benjamini-Hochberg across all
        evaluated candidates instead of the raw ``max_p_value`` threshold.
    fdr_q : float
        Target false-discovery rate for Benjamini-Hochberg (Section 13).
    """

    ic_min_abs: float = 0.02
    ic_max_p: float = 0.05
    min_lift: float = 0.08
    min_cohens_d: float = 0.15
    max_p_value: float = 0.05
    min_activations: int = 30
    use_fdr: bool = True
    fdr_q: float = 0.10


@dataclass
class AlphaConfig:
    """Top-level configuration for the Alpha Discovery pipeline.

    Attributes
    ----------
    target : TargetDefinition
        The economic target (required).
    thresholds : PromotionThresholds
        Admission / promotion gates.
    close_col : str
        Name of the close-price column used to build forward returns.
    timestamp_col : str
        Datetime column name (or DatetimeIndex name) on the KPI table.
    regime_col : str
        Column holding the market regime (produced by Market Context).  When
        absent from the table, regime sensitivity is skipped and the regime
        breadth term is dropped from the alpha score.
    regime_stable_col : str
        Optional ``regime_stable`` boolean column.  When present and
        ``use_stable_regime_only`` is set, only stable bars feed the per-regime
        IC to avoid transition-bar contamination.
    use_stable_regime_only : bool
        Restrict regime sensitivity to ``regime_stable == True`` bars.
    min_regime_obs : int
        Minimum bars (or activations) required to evaluate a regime; regimes
        below this are reported as ``insufficient`` and excluded from breadth.
    rolling_ic_window : int or None
        Window, in bars, for the rolling-IC stability check (Step 3.3).  When
        ``None`` it defaults to ``60`` days expressed in bars via
        ``bars_per_day``.
    bars_per_day : float or None
        Bars per calendar day, used only to size the default rolling-IC window.
        When ``None`` it is inferred from the timestamp spacing.
    score_weights : tuple of float
        Weights for ``(ic, lift, cohens_d, regime_breadth)`` in the composite
        alpha score (Step 6.1).
    discovery_date : str or None
        ISO date stamped onto every contract.  ``None`` → today's date.
    """

    target: TargetDefinition = field(default_factory=TargetDefinition)
    thresholds: PromotionThresholds = field(default_factory=PromotionThresholds)
    close_col: str = "close"
    timestamp_col: str = "open_dt"
    regime_col: str = "regime"
    regime_stable_col: str = "regime_stable"
    use_stable_regime_only: bool = False
    min_regime_obs: int = 10
    rolling_ic_window: Optional[int] = None
    bars_per_day: Optional[float] = None
    score_weights: Tuple[float, float, float, float] = (0.25, 0.30, 0.25, 0.20)
    discovery_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Measurement results
# ---------------------------------------------------------------------------

@dataclass
class MarketStructure:
    """Step 2 — interpretive context, computed once per session.

    Attributes
    ----------
    hurst : float
        Hurst exponent of the close price (``< 0.5`` mean-reverting,
        ``> 0.5`` trending).
    hurst_interpretation : str
        ``"mean_reverting"`` | ``"random_walk"`` | ``"trending"``.
    expected_family : str
        Alpha family the structure favours: ``"mean_reversion"`` |
        ``"momentum"`` | ``"none"``.
    autocorr : dict[int, float]
        ACF of the horizon return at the probed lags (in bars).
    """

    hurst: float
    hurst_interpretation: str
    expected_family: str
    autocorr: Dict[int, float] = field(default_factory=dict)


@dataclass
class ICResult:
    """Step 3 — Information Coefficient of the underlying continuous feature."""

    feature: str
    ic: float
    p_value: float
    n: int
    rolling_ic_stable: Optional[bool] = None
    rolling_ic_mean: Optional[float] = None
    rolling_sign_consistency: Optional[float] = None
    admitted: bool = False


@dataclass
class EventStats:
    """Step 4 — predictive power of the *binary* event vs the target."""

    n_activations: int
    win_rate: float
    base_rate: float
    lift: float
    fwd_return_mean: float
    cohens_d: float
    t_stat: float
    p_value: float


@dataclass
class RegimeStat:
    """Per-regime measurement (Step 5)."""

    regime: str
    n: int
    ic: float
    p_value: float
    win_rate: float
    strength: str  # "strong" | "moderate" | "negligible" | "insufficient"


@dataclass
class RegimeAnalysis:
    """Step 5 — aggregated regime sensitivity classification."""

    per_regime: List[RegimeStat]
    dependency_type: str  # "agnostic" | "conditional" | "specific" | "broken" | "unknown"
    active_regimes: List[str]
    weak_regimes: List[str]
    regime_breadth: float  # fraction of evaluated regimes that are significant


@dataclass
class AlphaScore:
    """Step 6 — composite alpha score and letter grade."""

    ic_magnitude: float
    lift: float
    cohens_d: float
    regime_breadth: float
    composite_score: float
    grade: str


@dataclass
class AlphaContract:
    """The Alpha Discovery output artifact (Step 7).

    One contract is produced per evaluated Event Candidate.  ``promoted``
    distinguishes the candidates that cleared every gate (``status``
    ``"HYPOTHESIS"``) from those rejected (``status`` ``"REJECTED"``).  The
    contract never alters the event expression — it only references and
    measures it.
    """

    alpha_id: str
    version: str
    discovery_date: str
    status: str  # "HYPOTHESIS" | "REJECTED"

    asset: str
    exchange: str
    timeframe: str
    direction: str

    event_candidate_id: str
    event_expression: str
    pattern_family: str

    target_definition: TargetDefinition
    base_rate: float

    underlying_feature: ICResult
    event_stats: EventStats
    market_structure: MarketStructure
    regime_analysis: RegimeAnalysis
    alpha_score: AlphaScore

    promoted: bool
    rejection_reasons: List[str] = field(default_factory=list)
    fdr_promoted: Optional[bool] = None

    handoff_status: str = "PENDING_RULE_DISCOVERY"
    rule_discovery_response: Optional[dict] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Flat dictionary suitable for a summary DataFrame row.

        Nested measurement objects are flattened with prefixes so the row can
        be sorted and filtered directly.  The full nested structure remains on
        the dataclass attributes for callers that need it.
        """
        ic = self.underlying_feature
        es = self.event_stats
        sc = self.alpha_score
        ra = self.regime_analysis
        return {
            "alpha_id": self.alpha_id,
            "status": self.status,
            "promoted": self.promoted,
            "event_candidate_id": self.event_candidate_id,
            "expression": self.event_expression,
            "pattern_family": self.pattern_family,
            "feature": ic.feature,
            "ic": ic.ic,
            "ic_p_value": ic.p_value,
            "ic_admitted": ic.admitted,
            "rolling_ic_stable": ic.rolling_ic_stable,
            "n_activations": es.n_activations,
            "win_rate": es.win_rate,
            "base_rate": es.base_rate,
            "lift": es.lift,
            "fwd_return_mean": es.fwd_return_mean,
            "cohens_d": es.cohens_d,
            "t_stat": es.t_stat,
            "p_value": es.p_value,
            "fdr_promoted": self.fdr_promoted,
            "regime_dependency": ra.dependency_type,
            "regime_breadth": ra.regime_breadth,
            "composite_score": sc.composite_score,
            "grade": sc.grade,
            "rejection_reasons": "; ".join(self.rejection_reasons),
        }

    def to_contract_dict(self) -> dict:
        """Full nested contract as a dictionary, ready to serialise to YAML/JSON.

        Mirrors the YAML layout in the documentation (Section 2).
        """
        return {
            "alpha_id": self.alpha_id,
            "version": self.version,
            "discovery_date": self.discovery_date,
            "status": self.status,
            "asset": self.asset,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "event_candidate_id": self.event_candidate_id,
            "event_expression": self.event_expression,
            "pattern_family": self.pattern_family,
            "target_definition": {
                **asdict(self.target_definition),
                "base_rate": self.base_rate,
            },
            "statistical_evidence": {
                "n_observations": self.underlying_feature.n,
                "underlying_feature": {
                    "name": self.underlying_feature.feature,
                    "ic": self.underlying_feature.ic,
                    "p_value": self.underlying_feature.p_value,
                    "ic_rolling_stable": self.underlying_feature.rolling_ic_stable,
                },
                "event_stats": {
                    "n_activations": self.event_stats.n_activations,
                    "win_rate": self.event_stats.win_rate,
                    "lift_vs_base": self.event_stats.lift,
                    "fwd_return_mean": self.event_stats.fwd_return_mean,
                    "cohens_d": self.event_stats.cohens_d,
                    "t_stat": self.event_stats.t_stat,
                    "p_value": self.event_stats.p_value,
                },
            },
            "market_structure": asdict(self.market_structure),
            "regime_analysis": [asdict(r) for r in self.regime_analysis.per_regime],
            "regime_dependency": {
                "type": self.regime_analysis.dependency_type,
                "active_regimes": self.regime_analysis.active_regimes,
                "weak_regimes": self.regime_analysis.weak_regimes,
                "regime_breadth": self.regime_analysis.regime_breadth,
            },
            "alpha_score": asdict(self.alpha_score),
            "promoted": self.promoted,
            "rejection_reasons": self.rejection_reasons,
            "fdr_promoted": self.fdr_promoted,
            "handoff_status": self.handoff_status,
            "rule_discovery_response": self.rule_discovery_response,
        }
