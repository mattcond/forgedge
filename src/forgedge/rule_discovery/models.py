"""Core data structures for the Rule Discovery module (FORGE Modulo 3).

Rule Discovery receives an :class:`~forgedge.alpha_discovery.models.AlphaContract`
from Alpha Discovery and answers an operational question: *is the statistically
flagged pattern exploitable in a realistic backtest — with fees, a finite fill
rate, limit orders and a discrete target?*

The dataclasses here mirror, field by field, the structures documented in
``docs/modules/RuleDiscovery.md``.  Rule Discovery never alters the event
expression: it reconstructs the event with the parameters stored on the Event
Candidate, parametrises the order mechanics, backtests them, validates the
result out-of-sample with a walk-forward scheme, and emits a verdict
(``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestParams:
    """Operational parameters of a single backtest run.

    These are the only degrees of freedom Rule Discovery explores — the event
    expression itself is fixed.  Defaults follow the ``BASE_CONFIG`` of the
    Rule Discovery specification (Section 3.2).

    Attributes
    ----------
    buy_type : str
        ``"limit"`` — place a limit order at ``anchor * (1 - buy_drop_pct)``,
        filled if touched within ``buy_delay_bar`` bars; ``"market"`` — buy at
        the next bar's open.
    buy_drop_pct : float
        Discount applied to the anchor price for the limit order (e.g. ``0.01``
        = -1%).  Ignored when ``buy_type == "market"``.
    buy_delay_bar : int
        Number of bars the limit order stays live.  Ignored for market orders.
    buy_price_anchor : str
        Column used as the anchor for the limit price (default ``"close"``).
    sell_pct : float
        Take-profit target as a fraction of the fill price (e.g. ``0.04`` = +4%).
    target_h : int
        Maximum holding horizon in bars; the position is closed at the target
        bar's close if the take-profit was never reached.
    target_col : str
        Price column used for the close-at-horizon exit (default ``"close"``).
    target_hit_col : str
        Price column scanned to detect the take-profit during the exit window.
        ``"close"`` (default) reproduces the certified reference engine — the
        target is hit only when a bar *closes* at or above ``sell_price``
        (conservative).  ``"high"`` uses the intrabar touch described in the
        Rule Discovery spec (a limit-sell fill), which is more optimistic.
    fee : float
        Fee per side (e.g. ``0.002``).  The round-trip cost applied is ``fee * 2``.
    early_stopping : bool
        ``True`` — scan ``target_hit_col`` in the exit window and exit at
        ``sell_price`` on the first bar that reaches the take-profit.
        ``False`` — always exit at the target bar's close.
    """

    buy_type: str = "limit"
    buy_drop_pct: float = 0.010
    buy_delay_bar: int = 6
    buy_price_anchor: str = "close"
    sell_pct: float = 0.040
    target_h: int = 24
    target_col: str = "close"
    target_hit_col: str = "close"
    fee: float = 0.002
    early_stopping: bool = True

    def merged(self, **overrides) -> "BacktestParams":
        """Return a copy with ``overrides`` applied."""
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return BacktestParams(**data)


@dataclass
class ScoringParams:
    """Parameters of the composite scoring metrics (``pf_score_tpm`` & co).

    Mirrors the exposed knobs of ``backtest_scoring.md``.  The dynamic
    trade-count threshold is ``max(pf_min_trades, n_months * pf_min_tpm)``;
    ``pf_tpm_target`` is the target trades-per-month for the consistency term
    ``C_norm`` and **must be calibrated** against the realised ``tpm_mu``.

    Attributes
    ----------
    pf_min_trades : int
        Absolute floor of the dynamic trade-count threshold.
    pf_min_tpm : int
        Minimum trades/month feeding the dynamic threshold.
    pf_tpm_target : int
        Target trades/month for the monthly-consistency term ``C_norm``.
    """

    pf_min_trades: int = 15
    pf_min_tpm: int = 2
    pf_tpm_target: int = 3


@dataclass
class GridSpec:
    """Search grid over the operational parameters (Step 2.4).

    Each attribute is a list of candidate values; the screening evaluates the
    full cartesian product.  ``None`` means "do not vary — use the value from
    :class:`BacktestParams`".  When the grid is left empty Rule Discovery
    builds a sensible default centred on the Alpha Contract's derived target.
    """

    buy_drop_pct: Optional[List[float]] = None
    sell_pct: Optional[List[float]] = None
    target_h: Optional[List[int]] = None
    buy_delay_bar: Optional[List[int]] = None

    def is_empty(self) -> bool:
        return not any(
            getattr(self, f) for f in ("buy_drop_pct", "sell_pct", "target_h", "buy_delay_bar")
        )


@dataclass
class WalkForwardConfig:
    """Walk-forward out-of-sample validation parameters (Step 4).

    The timeline is divided into ``n_splits`` consecutive **test** windows.
    For every split the operational parameters are re-selected on the train
    window that precedes it (a grid screening optimising ``pf_score_tpm``), then
    evaluated *once* on the untouched test window.  Concatenating the per-split
    test trades gives the honest out-of-sample track record of the rule.

    Attributes
    ----------
    n_splits : int
        Number of out-of-sample test windows.
    train_span_months : int or None
        Length of each train window, in calendar months.  ``None`` → anchored
        walk-forward: the train window always starts at the beginning of the
        data and grows.  An integer → rolling walk-forward with a fixed-length
        train window.
    test_span_months : int or None
        Length of each test window, in months.  ``None`` → the post-train span
        is divided equally into ``n_splits`` windows.
    min_train_months : int
        Minimum train length required before the first test window.
    reoptimise : bool
        ``True`` (default) — re-run the grid screening on every train window
        (true walk-forward optimisation).  ``False`` — keep the in-sample best
        parameters fixed and only replay them out-of-sample.
    """

    n_splits: int = 4
    train_span_months: Optional[int] = None
    test_span_months: Optional[int] = None
    min_train_months: int = 6
    reoptimise: bool = True


@dataclass
class SelectionCriteria:
    """Acceptance thresholds for the in-sample selection and the verdict (Step 3/8).

    Attributes
    ----------
    min_profit_factor : float
        Minimum profit factor for ``EDGE``.
    min_win_rate : float
        Minimum win rate (0–1).
    min_trades : int
        Minimum number of executed trades.
    min_tpm : float
        Minimum trades/month.
    min_pf_score_tpm : float
        Minimum composite ``pf_score_tpm``.
    min_fill_rate : float
        Minimum fill rate — below it the limit discount is too deep.
    partial_min_profit_factor : float
        Lower PF bound for a ``PARTIAL-EDGE`` (between this and
        ``min_profit_factor``).
    max_zero_months_edge : int
        Maximum months with zero trades tolerated for a full ``EDGE``.
    max_zero_months_partial : int
        Maximum months with zero trades tolerated for a ``PARTIAL-EDGE``.
    max_regime_dependency : float
        Maximum regime-dependency score for a full ``EDGE``.
    min_dsr : float
        Minimum Deflated Sharpe Ratio for a full ``EDGE``.
    max_ttest_p : float
        Maximum p-value for the win-rate / expectancy significance tests.
    """

    min_profit_factor: float = 2.0
    min_win_rate: float = 0.55
    min_trades: int = 30
    min_tpm: float = 2.0
    min_pf_score_tpm: float = 0.30
    min_fill_rate: float = 0.40
    partial_min_profit_factor: float = 1.5
    max_zero_months_edge: int = 1
    max_zero_months_partial: int = 4
    max_regime_dependency: float = 0.30
    min_dsr: float = 1.0
    max_ttest_p: float = 0.05


@dataclass
class RuleDiscoveryConfig:
    """Top-level configuration for the Rule Discovery pipeline.

    Attributes
    ----------
    base_params : BacktestParams
        Fixed parameters and grid centre.  Rule Discovery overrides
        ``buy_type``/``sell_pct``/``target_h`` from the Alpha Contract when
        ``use_contract_target`` is set.
    scoring : ScoringParams
        Composite-scoring knobs.
    grid : GridSpec
        Operational grid; auto-built around the contract target when empty.
    walk_forward : WalkForwardConfig
        OOS walk-forward settings.
    criteria : SelectionCriteria
        Acceptance / verdict thresholds.
    use_contract_target : bool
        Seed ``sell_pct``/``target_h`` from the contract's derived target.
    timestamp_col : str
        Datetime column on the KPI table (default ``"open_dt"``).
    signal_col : str
        Name of the reconstructed boolean signal column injected into the
        candle table.
    discovery_date : str or None
        ISO date stamped onto the response; ``None`` → today.
    """

    base_params: BacktestParams = field(default_factory=BacktestParams)
    scoring: ScoringParams = field(default_factory=ScoringParams)
    grid: GridSpec = field(default_factory=GridSpec)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    criteria: SelectionCriteria = field(default_factory=SelectionCriteria)
    use_contract_target: bool = True
    timestamp_col: str = "open_dt"
    signal_col: str = "__rule_signal__"
    discovery_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BacktestSummary:
    """Aggregated metrics of a single backtest run.

    Holds the standard performance metrics plus the composite scores defined in
    ``backtest_scoring.md``.  All attributes carry ``nan`` when there are no
    executed trades.
    """

    total_signals: int
    total_trades: int
    fill_rate: float
    win_rate_pct: float
    winning_trades: int
    losing_trades: int
    total_net_gain: float
    expectancy: float
    std_net_gain: float
    profit_factor: float
    best_trade: float
    worst_trade: float
    target_hit_rate_pct: float
    # temporal distribution
    n_months: int
    active_months: int
    zero_months: int
    tpm_mu: float
    tpm_sigma: float
    c_norm: float
    # composite scores
    pf_score: float
    pf_score_tpm: float
    exp_score_tpm: float
    sharpe_raw: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionEnvelope:
    """Best/worst-case execution bracket for the selected configuration.

    The same rule is backtested under the two meaningful take-profit
    conventions, bracketing realistic execution without forcing a single
    choice:

    * ``conservative`` — ``target_hit_col="close"``: the target counts only when
      a bar *closes* at or above ``sell_price`` (understates — a real limit sell
      would fill intrabar).  Matches the certified reference engine.
    * ``optimistic`` — ``target_hit_col="high"``: the target fills on the first
      intrabar touch (overstates — assumes the limit sell always fills).

    The true performance of the rule lies between the two.
    """

    conservative: "BacktestSummary"
    optimistic: "BacktestSummary"

    def to_dict(self) -> dict:
        return {
            "conservative_close": self.conservative.to_dict(),
            "optimistic_high": self.optimistic.to_dict(),
        }


@dataclass
class ExcursionStats:
    """MAE / MFE — intra-trade adverse and favourable excursion.

    For every executed trade, over its realised holding window
    ``[fill+1 .. exit]``:

    * **MAE** (Maximum Adverse Excursion) — the deepest drawdown reached,
      ``(min low - buy_price) / buy_price`` (negative);
    * **MFE** (Maximum Favourable Excursion) — the highest run-up reached,
      ``(max high - buy_price) / buy_price`` (positive).

    These describe the rule's "range of action" — how far each trade swings
    against and in favour of the position before it closes — independently of
    which exit convention is chosen.  All values are fractions of the buy price.
    """

    n_trades: int
    mae_mean: float
    mae_median: float
    mae_worst: float
    mfe_mean: float
    mfe_median: float
    mfe_best: float
    mfe_reached_target_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GridResult:
    """A single grid combination and the metrics it produced."""

    params: BacktestParams
    summary: BacktestSummary

    def row(self) -> dict:
        """Flat dict combining the swept parameters and key metrics."""
        p = self.params
        s = self.summary
        return {
            "buy_drop_pct": p.buy_drop_pct,
            "sell_pct": p.sell_pct,
            "target_h": p.target_h,
            "buy_delay_bar": p.buy_delay_bar,
            "profit_factor": s.profit_factor,
            "win_rate_pct": s.win_rate_pct,
            "total_trades": s.total_trades,
            "expectancy": s.expectancy,
            "tpm_mu": s.tpm_mu,
            "fill_rate": s.fill_rate,
            "zero_months": s.zero_months,
            "pf_score_tpm": s.pf_score_tpm,
        }


@dataclass
class StatisticalValidation:
    """Statistical validation of the selected configuration (Step 4)."""

    ttest_winrate_t: float
    ttest_winrate_p: float
    ttest_expectancy_t: float
    ttest_expectancy_p: float
    deflated_sharpe: float
    sharpe_ratio: float
    n_trials_tested: int
    temporal_stability: str  # "PASS" | "WARN" | "FAIL"
    pf_first_half: float
    pf_second_half: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WalkForwardSplit:
    """One train→test split of the walk-forward validation."""

    split_idx: int
    train_from: str
    train_to: str
    test_from: str
    test_to: str
    params: BacktestParams
    train_summary: BacktestSummary
    test_summary: BacktestSummary


@dataclass
class WalkForwardResult:
    """Aggregated out-of-sample walk-forward result.

    Attributes
    ----------
    splits : list[WalkForwardSplit]
        Per-split detail, chronologically ordered.
    oos_summary : BacktestSummary
        Metrics computed on the concatenation of every test-window trade — the
        honest out-of-sample track record.
    n_profitable_splits : int
        Number of test windows with a positive net gain.
    consistency : float
        ``n_profitable_splits / n_splits`` — fraction of OOS windows in profit.
    oos_envelope : ExecutionEnvelope or None
        Execution bracket (close↔high) over the concatenated OOS trades — the
        out-of-sample twin of the in-sample envelope.
    oos_excursion : ExcursionStats or None
        MAE/MFE over the concatenated OOS trades.
    oos_validation : StatisticalValidation or None
        Win-rate / expectancy t-tests and Sharpe on the OOS trades.  The Sharpe
        is **not** deflated (``n_trials=1``): out-of-sample data played no part
        in the parameter selection, so no multiple-testing haircut applies.
    oos_trades : pd.DataFrame or None
        The concatenated per-trade ledger of every test window (close
        convention), excluded from ``repr`` to keep logs readable.
    """

    splits: List[WalkForwardSplit]
    oos_summary: BacktestSummary
    n_profitable_splits: int
    consistency: float
    oos_envelope: Optional["ExecutionEnvelope"] = None
    oos_excursion: Optional["ExcursionStats"] = None
    oos_validation: Optional["StatisticalValidation"] = None
    oos_trades: Optional["pd.DataFrame"] = field(default=None, repr=False)


@dataclass
class RegimeBreakdown:
    """Per-regime performance and the aggregate dependency score (Step 5)."""

    per_regime: List[dict]
    dependency_score: float
    zero_months: int
    avoid_in: List[str]


@dataclass
class ValidatedRule:
    """The operational rule emitted by Rule Discovery for a ``(PARTIAL-)EDGE``."""

    expression: str
    event_candidate_id: str
    params: BacktestParams

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "event_candidate_id": self.event_candidate_id,
            "entry_mode": self.params.buy_type,
            "buy_drop_pct": self.params.buy_drop_pct,
            "buy_delay_bar": self.params.buy_delay_bar,
            "sell_pct": self.params.sell_pct,
            "target_h": self.params.target_h,
            "fee": self.params.fee,
        }


@dataclass
class RuleDiscoveryResponse:
    """Verdict and full evidence produced by Rule Discovery (Section 8).

    This is the artifact handed to Rule Registry.  ``verdict`` is one of
    ``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE``; ``validated_rule`` is populated
    only for the first two.
    """

    date: str
    verdict: str  # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
    alpha_id: str
    asset: str
    timeframe: str
    validated_rule: Optional[ValidatedRule]
    in_sample_summary: BacktestSummary
    walk_forward: Optional[WalkForwardResult]
    statistical_validation: Optional[StatisticalValidation]
    regime_analysis: Optional[RegimeBreakdown]
    execution_envelope: Optional[ExecutionEnvelope] = None
    excursion: Optional[ExcursionStats] = None
    grid_results: List[GridResult] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def is_edge(self) -> bool:
        return self.verdict in ("EDGE", "PARTIAL-EDGE")

    def to_dict(self) -> dict:
        """Nested dictionary mirroring the YAML layout of the spec (Section 8)."""
        wf = self.walk_forward
        sv = self.statistical_validation
        ra = self.regime_analysis
        return {
            "date": self.date,
            "verdict": self.verdict,
            "alpha_id": self.alpha_id,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "validated_rule": self.validated_rule.to_dict() if self.validated_rule else None,
            "in_sample_results": self.in_sample_summary.to_dict(),
            "walk_forward": (
                {
                    "n_splits": len(wf.splits),
                    "consistency": wf.consistency,
                    "n_profitable_splits": wf.n_profitable_splits,
                    "oos_results": wf.oos_summary.to_dict(),
                    "oos_execution_envelope": (
                        wf.oos_envelope.to_dict() if wf.oos_envelope else None
                    ),
                    "oos_excursion": (
                        wf.oos_excursion.to_dict() if wf.oos_excursion else None
                    ),
                    "oos_statistical_validation": (
                        wf.oos_validation.to_dict() if wf.oos_validation else None
                    ),
                    "splits": [
                        {
                            "split_idx": s.split_idx,
                            "train": [s.train_from, s.train_to],
                            "test": [s.test_from, s.test_to],
                            "test_profit_factor": s.test_summary.profit_factor,
                            "test_win_rate": s.test_summary.win_rate_pct,
                            "test_trades": s.test_summary.total_trades,
                            "test_net_gain": s.test_summary.total_net_gain,
                        }
                        for s in wf.splits
                    ],
                }
                if wf is not None
                else None
            ),
            "statistical_validation": sv.to_dict() if sv else None,
            "execution_envelope": (
                self.execution_envelope.to_dict() if self.execution_envelope else None
            ),
            "excursion": self.excursion.to_dict() if self.excursion else None,
            "regime_analysis": (
                {
                    "dependency_score": ra.dependency_score,
                    "zero_months": ra.zero_months,
                    "avoid_in": ra.avoid_in,
                    "per_regime": ra.per_regime,
                }
                if ra is not None
                else None
            ),
            "rejection_reasons": self.rejection_reasons or None,
            "notes": self.notes,
        }
