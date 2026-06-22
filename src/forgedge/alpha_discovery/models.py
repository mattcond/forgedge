"""Core data structures for the Alpha Discovery module (FORGE Modulo 2).

Alpha Discovery receives ``EventCandidate`` artifacts from Event Discovery,
**derives** each candidate's economic target (horizon, sell percentage,
direction) from the data, confirms it on a held-out temporal tail, and
measures the candidate's predictive power against it.  The output is the
**Alpha Contract** — the formal interface consumed by Rule Discovery.

The dataclasses here mirror, field by field, the contract format documented in
``docs/modules/AlphaDiscovery.md`` (Section 2).  Nothing in this module
modifies an event's threshold, window, or expression: the contract references
the Event Candidate exactly as received and only adds statistical measures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DerivedTarget:
    """The per-event economic target *derived* by Alpha Discovery (Step 1).

    Alpha Discovery receives no economic parameters: for every Event
    Candidate it scans a grid of candidate horizons and derives the target
    from the data — the **excess log-return** ``Δ_h = μ_cond_h − μ_base_h``
    (conditional mean minus the unconditional baseline, in log space) is the
    quantity that carries the signal, so direction and horizon are read from
    *it*, not from the raw conditional return that mixes the asset's drift in:

    * ``holding_period_h`` — the horizon that maximises ``|z_h|``, where
      ``z_h = Δ_h / σ_null,h`` is the excess log-return standardised by a
      **circular-rotation null** that preserves the event's activation
      clustering.  A naive t-statistic ``Δ_h / (σ_cond / √n)`` treats the
      overlapping forward-return windows as independent, so its denominator
      shrinks with the horizon and ``|T_h|`` is inflated on long horizons —
      pinning ``h*`` to the long edge of the grid for clustered events even
      with no real edge.  The rotation null re-derives the scale from the
      data's own autocorrelation, removing that bias;
    * ``mean_advantage``   — the signed excess log-return ``Δ_h*`` at ``h*``
      (long convention: positive ⇒ long edge, negative ⇒ short edge);
    * ``sell_pct``         — ``q``-quantile of the Maximum Favorable Excursion
      (MFE) at ``h*`` across active IS bars: the candidate take-profit baseline
      handed to Rule Discovery;
    * ``direction``        — the sign of the excess log-return ``Δ_h*``
      (``long`` when positive, ``short`` when negative).  It emerges from the
      data and is never imposed a priori: the same rule (e.g. ``RSI > 80``)
      yields *short* in a bull run and *long* in a flat market, because only
      the excess over the prevailing drift is read.

    Statistical significance (rotation-null p-values + Benjamini-Hochberg →
    ``h_sig``) is recorded as **non-blocking diagnostics**: with few activations
    the FDR gate can be empty even on a real edge, so ``h*`` is still chosen on
    ``|z_h|`` over the whole grid and the target is flagged
    ``statistically_weak`` instead of being discarded.

    These are **candidates**, not validated parameters: they are written into
    the Alpha Contract as the baseline that Rule Discovery refines and
    stress-tests with full order mechanics.

    Attributes
    ----------
    holding_period_h : int
        Selected horizon, in bars.
    sell_pct : float
        ``q``-quantile of the MFE at ``h*`` across active IS bars.
    direction : str
        ``"long"`` | ``"short"`` | ``"undetermined"`` (no finite excess
        log-return, or ``|z_h*|`` below ``min_direction_t``).
    mean_advantage : float
        Signed excess log-return ``Δ_h*`` at ``holding_period_h``.
    advantage_by_h : dict[int, float]
        Excess log-return ``Δ_h`` (signed, long convention) for every horizon
        scanned — the full profile, for transparency.
    t_stat_by_h : dict[int, float]
        Rotation-standardised excess statistic ``z_h`` (signed, long
        convention) per horizon — autocorrelation-robust, unlike a naive t-stat.
    score_by_h : dict[int, float]
        ``|z_h|`` selection score per horizon — the criterion used to choose
        ``h*``.
    p_value_by_h : dict[int, float]
        Circular-rotation-null p-value of ``|Δ_h|`` per horizon (two-sided);
        ``nan`` where it could not be computed. Diagnostic only.
    h_sig : tuple[int, ...]
        Horizons surviving the Benjamini-Hochberg control of ``p_value_by_h``
        at ``fdr_q``. Diagnostic only — never gates ``h*``.
    statistically_weak : bool
        ``True`` when ``h*`` is **not** in ``h_sig`` (no horizon cleared the FDR
        gate): the derived direction stands but the evidence is thin.
    fixed_target : bool
        ``True`` when the target was **user-specified** via
        ``AlphaConfig.fixed_target`` rather than derived from the data.  In that
        case ``holding_period_h``, ``sell_pct`` and ``direction`` come from the
        user; ``mean_advantage`` is ``nan`` and the per-horizon profile (when
        present) is the read-only data derivation kept for diagnostics.
    data_derived_horizon_h : int or None
        Fixed-target diagnostic: the horizon the data derivation *would* have
        selected (``None`` when the read-only derivation was skipped).  Lets a
        consumer check convergence — ``data_derived_horizon_h ≈ holding_period_h``
        means the data independently confirms the user's horizon.
    data_derived_sell_pct : float or None
        Fixed-target diagnostic: the ``sell_pct`` the data derivation would have
        produced at its own ``h*`` (``None``/``nan`` when unavailable).
    """

    holding_period_h: int
    sell_pct: float
    direction: str
    mean_advantage: float
    advantage_by_h: Dict[int, float] = field(default_factory=dict)
    t_stat_by_h: Dict[int, float] = field(default_factory=dict)
    score_by_h: Dict[int, float] = field(default_factory=dict)
    p_value_by_h: Dict[int, float] = field(default_factory=dict)
    h_sig: Tuple[int, ...] = ()
    statistically_weak: bool = False
    fixed_target: bool = False
    data_derived_horizon_h: Optional[int] = None
    data_derived_sell_pct: Optional[float] = None


@dataclass
class OOSValidation:
    """Out-of-sample confirmation of the derived target.

    The derivation of ``DerivedTarget`` is an in-sample optimisation over the
    horizon grid; to keep promotion honest, the derived target is replayed on
    a held-out temporal tail (controlled by ``AlphaConfig.train_ratio``) that
    played no part in the derivation or in any in-sample measure.

    Attributes
    ----------
    n_bars : int
        Bars in the OOS window.
    n_activations : int
        Event activations in the OOS window with a complete forward horizon.
    mean_advantage : float
        Mean *oriented* forward return of OOS active bars (positive =
        favourable to the derived direction).
    t_stat, p_value : float
        One-sided active-vs-inactive t-test on the oriented OOS returns.
    win_rate, base_rate, lift : float
        Binary-target measures on OOS at the derived ``(h, sell_pct,
        direction)``.
    passed : bool
        ``True`` when the OOS advantage keeps the derived sign and the t-test
        clears ``PromotionThresholds.oos_max_p``.
    """

    n_bars: int
    n_activations: int
    mean_advantage: float
    t_stat: float
    p_value: float
    win_rate: float
    base_rate: float
    lift: float
    passed: bool
    min_detectable_effect: float = float("nan")
    """Minimum Cohen's d detectable at ``oos_max_p`` given the OOS sample size.
    Compared to IS ``cohens_d`` to diagnose underpowered OOS windows."""


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
    use_fdr : bool
        When ``True`` (default), promotion uses Benjamini-Hochberg across all
        evaluated candidates instead of the raw ``max_p_value`` threshold.
    fdr_q : float
        Target false-discovery rate for Benjamini-Hochberg (Section 13).
    oos_max_p : float
        Maximum one-sided p-value for the out-of-sample confirmation of the
        derived target.  When the p-value clears this threshold with a positive
        mean advantage, ``oos_validation.passed`` is ``True``.  No minimum
        activation count is imposed — the p-value already encodes sample size.
        A non-parametrizable floor of 10 activations triggers a diagnostic
        warning about low statistical reliability (see ``AlphaDiscovery``).
    min_direction_t : float
        Minimum ``|z_h*|`` (rotation-standardised excess statistic at the
        selected horizon) for a direction to be assigned.  Below this floor the
        edge is not distinguishable from the rotation null and ``direction`` is
        set to ``"undetermined"``.  Default ``0.5``.
    require_significant_direction : bool
        When ``True`` (default), a direction is assigned only if the selected
        horizon clears the Benjamini-Hochberg control — i.e. ``h*`` is in
        ``DerivedTarget.h_sig`` (equivalently ``statistically_weak`` is
        ``False``).  When **no** horizon is BH-significant the excess is not
        statistically distinguishable from the rotation null at any horizon, so
        ``argmax|z_h|`` would assign a direction off a coin-flip (often the
        drift-driven long edge of the grid); this gate returns
        ``"undetermined"`` instead.  Set to ``False`` for the legacy
        non-blocking behaviour (always assign a direction subject only to
        ``min_direction_t``, flagging thin evidence via ``statistically_weak``).
    """

    ic_min_abs: float = 0.02
    ic_max_p: float = 0.05
    min_lift: float = 0.08
    min_cohens_d: float = 0.15
    max_p_value: float = 0.05
    use_fdr: bool = True
    fdr_q: float = 0.10
    oos_max_p: float = 0.10
    min_direction_t: float = 0.5
    require_significant_direction: bool = True


@dataclass
class TargetConfig:
    """A user-specified economic target for Alpha Discovery's fixed-target mode.

    In the TargetOptimizer workflow the holding horizon, take-profit threshold
    and side are chosen up front instead of being derived from the data.  Passed
    via :attr:`AlphaConfig.fixed_target`, it makes Alpha Discovery skip the
    per-event target *derivation* and measure every candidate against this
    target directly — every downstream measure (IC, win rate, lift, Cohen's d,
    regime sensitivity, OOS) runs unchanged.

    Attributes
    ----------
    horizon : int
        Holding period in bars (``> 0``).  Added to the forward-return grid if
        not already in ``AlphaConfig.horizon_grid``.
    min_return : float
        Take-profit threshold as a **fraction** (e.g. ``0.02`` = 2%), used as
        the derived target's ``sell_pct`` (``> 0``).
    side : str
        ``"long"`` or ``"short"`` — the trade direction, never overwritten by
        the data.
    min_activations : int
        Floor for valid lift scoring in the TargetOptimizer workflow: candidates
        firing on fewer than ``min_activations`` bars are skipped (the
        conditional win rate would be too noisy to trust).  Ignored by Alpha
        Discovery's fixed-target mode, which has its own promotion gates.
        Default ``10``.
    min_lift : float
        Prune threshold for the TargetOptimizer workflow: candidates whose
        conditional win rate gives ``lift < min_lift`` are discarded after each
        scoring pass.  Ignored by Alpha Discovery's fixed-target mode.
        Default ``1.0`` (keep only candidates that beat the base rate).
    target_mode : {"abs", "proj"}
        Binary-target definition (see :func:`binary_target`).  ``"abs"`` =
        absolute return from close; ``"proj"`` (default) = excess of the forward
        return over the local trend (PROJ_LOG), which strips the trend premium
        a long event would otherwise harvest in a bull market.  PROJ applies to
        **long** only — short reverts to ``"abs"`` (the bear trend *is* the alpha
        to capture, not noise to subtract).
    trend_sma_mult : float
        PROJ_LOG only.  Multiplier (× ``horizon``, in bars) for the trend SMA
        window.  Default ``2.0`` (SMA over ``2·h`` bars).  Everything is in
        bars, so the trend term auto-scales across timeframes; raise/lower this
        to smooth the trend more/less without touching ``horizon``.
    """

    horizon: int
    min_return: float
    side: str
    min_activations: int = 10
    min_lift: float = 1.0
    target_mode: Literal["abs", "proj"] = "proj"
    trend_sma_mult: float = 2.0

    def __post_init__(self):
        self.horizon = int(self.horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {self.horizon}.")
        self.min_return = float(self.min_return)
        if self.min_return <= 0:
            raise ValueError(f"min_return must be > 0 (fraction), got {self.min_return}.")
        self.side = str(self.side).lower().strip()
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side!r}.")
        self.min_activations = int(self.min_activations)
        if self.min_activations < 1:
            raise ValueError(
                f"min_activations must be >= 1, got {self.min_activations}."
            )
        self.target_mode = str(self.target_mode).lower().strip()
        if self.target_mode not in ("abs", "proj"):
            raise ValueError(
                f"target_mode must be 'abs' or 'proj', got {self.target_mode!r}."
            )
        self.min_lift = float(self.min_lift)
        if self.min_lift < 0:
            raise ValueError(f"min_lift must be >= 0, got {self.min_lift}.")
        self.trend_sma_mult = float(self.trend_sma_mult)
        if self.trend_sma_mult <= 0:
            raise ValueError(f"trend_sma_mult must be > 0, got {self.trend_sma_mult}.")


@dataclass
class AlphaConfig:
    """Top-level configuration for the Alpha Discovery pipeline.

    Alpha Discovery takes **no economic target as input**: the target
    (horizon, sell percentage, direction) is derived per event from the data
    over ``horizon_grid`` and confirmed on a held-out temporal tail
    (``train_ratio``).

    Attributes
    ----------
    horizon_grid : tuple of int
        Candidate holding horizons, in bars, scanned by the target
        derivation (Step 1).  For every Event Candidate the horizon that
        maximises ``|T_h|`` (the excess log-return t-statistic) is selected as
        ``h*``.
    mfe_quantile : float
        Quantile of the Maximum Favorable Excursion distribution (over active
        IS bars at ``h*``) used to set ``sell_pct``.  Default ``0.5``
        (median); increase toward 0.8 for a more aggressive TP seed.
    mfe_floor : float
        Minimum ``sell_pct`` (absolute), applied after the quantile
        computation.  Default ``0.005`` (50 bp).
    train_ratio : float
        Fraction of the (chronologically sorted) table used to derive the
        target and compute every in-sample measure; the remaining tail is
        reserved for the out-of-sample confirmation of the derived target.
        ``1.0`` disables the internal OOS split (not recommended — the
        derived horizon is then never validated out-of-sample).
    thresholds : PromotionThresholds
        Admission / promotion gates.
    asset, exchange, timeframe : str
        **Traceability metadata only** — copied verbatim into the contract's
        SCOPE section (and into the ``alpha_id``) so Rule Discovery and the
        Registry can attribute the alpha.  They have no effect whatsoever on
        any measurement.
    fee_per_side : float
        Informational only — recorded in the contract so Rule Discovery knows
        the assumed cost basis.  Alpha Discovery does not net it out (that is
        Rule Discovery's job).
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
        Weights for ``(ic, lift, cohens_d, z, regime_breadth)`` in the composite
        alpha score (Step 6.1) — ``z`` is ``|z_h*|``, the rotation-null
        standardised excess (the edge-to-noise ratio).  A legacy 4-tuple
        ``(ic, lift, cohens_d, breadth)`` is still accepted and upgraded with a
        default ``z`` weight.
    statistically_weak_penalty : float
        Multiplier applied to the composite when
        ``DerivedTarget.statistically_weak`` is ``True`` (the selected horizon
        did not clear the Benjamini-Hochberg gate).  Default ``0.6``; ``1.0``
        disables the penalty.
    oos_bonus : float
        Additive bonus applied to the composite when the out-of-sample
        confirmation passes (``OOSValidation.passed``), to separate confirmed
        edges from unconfirmed ones.  Default ``0.05``; ``0.0`` disables it.
    discovery_date : str or None
        ISO date stamped onto every contract.  ``None`` → today's date.
    fixed_target : TargetConfig or None
        When set, Alpha Discovery **skips target derivation** and measures every
        candidate against the user-specified ``(horizon, min_return, side)``.
        The horizon is added to the forward-return grid if absent.  ``None``
        (default) = derive the target from the data per event.
    fixed_target_diagnostic : bool
        Fixed-target mode only.  When ``True`` (default), Alpha Discovery still
        runs the target derivation in **read-only** mode to populate the
        per-horizon diagnostics and the ``data_derived_*`` convergence fields on
        the contract (the user's target is still what's measured).  ``False``
        skips it for a pure, slightly faster bypass (diagnostics left empty).
    target_mode : {"abs", "proj"}
        Binary-target definition used for win rate / lift / base rate (see
        :func:`binary_target`).  ``"proj"`` (default) measures the forward return
        in **excess of the local trend** (PROJ_LOG) so a long event that merely
        rides a bull trend is not credited with that trend's premium — markedly
        more stable IS→OOS.  ``"abs"`` is the legacy absolute-return target.
        PROJ applies to long only; short reverts to ``"abs"``.  Falls back to
        ``"abs"`` (with a warning) when history is shorter than the warmup.
    trend_sma_mult : float
        PROJ_LOG only.  Multiplier (× ``h``, in bars) for the trend SMA window
        ``w = round(trend_sma_mult · h)``.  Default ``2.0``.  Bar-relative, so it
        auto-scales across timeframes; the PROJ warmup is ``(trend_sma_mult+1)·h``.
    """

    horizon_grid: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48)
    mfe_quantile: float = 0.5
    mfe_floor: float = 0.005
    train_ratio: float = 0.7
    thresholds: PromotionThresholds = field(default_factory=PromotionThresholds)
    asset: str = "ASSET"
    exchange: str = ""
    timeframe: str = "1H"
    fee_per_side: float = 0.002
    close_col: str = "close"
    timestamp_col: str = "open_dt"
    regime_col: str = "regime"
    regime_stable_col: str = "regime_stable"
    use_stable_regime_only: bool = False
    min_regime_obs: int = 10
    rolling_ic_window: Optional[int] = None
    bars_per_day: Optional[float] = None
    score_weights: Tuple[float, ...] = (0.20, 0.25, 0.15, 0.25, 0.15)
    statistically_weak_penalty: float = 0.6
    oos_bonus: float = 0.05
    discovery_date: Optional[str] = None
    fixed_target: Optional[TargetConfig] = None
    fixed_target_diagnostic: bool = True
    target_mode: Literal["abs", "proj"] = "proj"
    trend_sma_mult: float = 2.0

    def __post_init__(self):
        if not self.horizon_grid or any(int(h) <= 0 for h in self.horizon_grid):
            raise ValueError("horizon_grid must contain positive horizons (bars).")
        self.horizon_grid = tuple(sorted({int(h) for h in self.horizon_grid}))
        if not (0.0 < self.train_ratio <= 1.0):
            raise ValueError(f"train_ratio must be in (0, 1], got {self.train_ratio}.")
        if not (0.0 < self.mfe_quantile <= 1.0):
            raise ValueError(f"mfe_quantile must be in (0, 1], got {self.mfe_quantile}.")
        if self.mfe_floor < 0:
            raise ValueError(f"mfe_floor must be >= 0, got {self.mfe_floor}.")
        self.target_mode = str(self.target_mode).lower().strip()
        if self.target_mode not in ("abs", "proj"):
            raise ValueError(
                f"target_mode must be 'abs' or 'proj', got {self.target_mode!r}."
            )
        self.trend_sma_mult = float(self.trend_sma_mult)
        if self.trend_sma_mult <= 0:
            raise ValueError(f"trend_sma_mult must be > 0, got {self.trend_sma_mult}.")
        # Back-compat: a legacy 4-tuple (ic, lift, cohens_d, breadth) is upgraded
        # to the 5-tuple (ic, lift, cohens_d, z, breadth) with a default z weight.
        w = tuple(float(x) for x in self.score_weights)
        if len(w) == 4:
            w = (w[0], w[1], w[2], 0.25, w[3])
        elif len(w) != 5:
            raise ValueError(
                "score_weights must have 5 elements (ic, lift, cohens_d, z, "
                f"breadth) or a legacy 4 (ic, lift, cohens_d, breadth); got {len(w)}."
            )
        self.score_weights = w


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

    # SCOPE — traceability metadata, no computational role
    asset: str
    exchange: str
    timeframe: str
    direction: str
    fee_per_side: float

    event_candidate_id: str
    event_expression: str
    pattern_family: str

    derived_target: DerivedTarget
    base_rate: float

    underlying_feature: ICResult
    event_stats: EventStats
    market_structure: MarketStructure
    regime_analysis: RegimeAnalysis
    alpha_score: AlphaScore
    oos_validation: Optional[OOSValidation]

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
        dt = self.derived_target
        oos = self.oos_validation
        return {
            "alpha_id": self.alpha_id,
            "status": self.status,
            "promoted": self.promoted,
            "event_candidate_id": self.event_candidate_id,
            "expression": self.event_expression,
            "pattern_family": self.pattern_family,
            "holding_period_h": dt.holding_period_h,
            "sell_pct": dt.sell_pct,
            "direction": dt.direction,
            "mean_advantage": dt.mean_advantage,
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
            "oos_passed": oos.passed if oos is not None else None,
            "oos_p_value": oos.p_value if oos is not None else float("nan"),
            "oos_lift": oos.lift if oos is not None else float("nan"),
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
            "fee_per_side": self.fee_per_side,
            "event_candidate_id": self.event_candidate_id,
            "event_expression": self.event_expression,
            "pattern_family": self.pattern_family,
            "derived_target": {
                **asdict(self.derived_target),
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
            "oos_validation": (
                asdict(self.oos_validation) if self.oos_validation is not None else None
            ),
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

    def persist(self, path) -> None:
        """Save this Alpha Contract to disk as a pickle file.

        Mirrors ``EventCandidate.persist``: the contract is the Alpha Discovery
        output artifact, pickled whole so Rule Discovery (or a later session)
        can reload it without re-running discovery.

        Reload with::

            import pickle
            contract = pickle.load(open(path, "rb"))
        """
        import pickle
        import pathlib
        pathlib.Path(path).write_bytes(pickle.dumps(self))
