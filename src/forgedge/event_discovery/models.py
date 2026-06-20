"""Core data structures for the Event Discovery module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class ColumnType(str, Enum):
    """Enumeration of the three column types recognised by the TypeClassifier.

    CONTINUOUS   — numeric series with more than two distinct values.
    BINARY       — numeric series with exactly two distinct values (0/1 flags).
    CATEGORICAL  — non-numeric or low-cardinality string series.
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass
class ColumnClassification:
    """Result of classifying a single DataFrame column (Step 0).

    Attributes
    ----------
    col_name : str
        Name of the original column.
    col_type : ColumnType
        Inferred type (CONTINUOUS / BINARY / CATEGORICAL).
    n_distinct : int
        Number of unique non-null values observed.
    is_scale_free : bool or None
        Result of the automatic scale-free heuristic.  None if not applicable
        (non-continuous columns are not tested).
    scale_free_override : bool or None
        User-supplied override.  When set, it takes precedence over
        ``is_scale_free`` in ``effective_scale_free``.
    """

    col_name: str
    col_type: ColumnType
    n_distinct: int
    is_scale_free: Optional[bool] = None
    scale_free_override: Optional[bool] = None

    @property
    def effective_scale_free(self) -> bool:
        """Return the scale-free flag that should be used by downstream steps.

        Resolution order:
        1. Non-continuous columns are never scale-free (False).
        2. If a user override exists, return it.
        3. Otherwise return the automatic detection result.
        """
        if self.col_type != ColumnType.CONTINUOUS:
            return False
        if self.scale_free_override is not None:
            return self.scale_free_override
        return bool(self.is_scale_free)

    @property
    def scale_free_overridden(self) -> bool:
        """True when a user override contradicts the automatic detection result.

        Useful for auditing which columns were forced into a different regime
        than the heuristic would have chosen.
        """
        return (
            self.scale_free_override is not None
            and self.scale_free_override != self.is_scale_free
        )


@dataclass
class GateParams:
    """Thresholds that govern the Consistency Gate (Step 4).

    Both parameters are rate/ratio invariant, making the same
    ``GateParams`` instance valid for both in-sample discovery and
    out-of-sample walk-forward validation without any scaling.

    Attributes
    ----------
    min_tpm : float
        Minimum average activations per month
        (total_activations / n_calendar_months).  Events below this rate
        lack statistical power and are discarded.
    max_dispersion : float
        Maximum allowed Index of Dispersion:
        ``Var(monthly_counts) / Mean(monthly_counts)``.
        For a pure Poisson process, ID = 1.  Values above the threshold
        indicate over-dispersed activations — bursty (concentrated in a
        few months) or periodic (clustered with regular gaps).
        Default 2.5 passes realistic financial signals while rejecting
        periodic bursts and single-month concentration.
    """
    min_tpm: float = 2.0
    max_dispersion: float = 2.5


@dataclass
class GateResult:
    """Outcome of one ConsistencyGate evaluation.

    Attributes
    ----------
    passed : bool
        True if all two gate criteria were satisfied.
    n_activations : int
        Total number of True bars in the event series.
    n_active_months : int
        Number of calendar months that contain at least one activation.
    max_monthly_share : float
        Fraction of total activations belonging to the most concentrated month
        (max_month_count / n_activations).
    mean_tpm : float
        Average activations per calendar month (n_activations / n_total_months).
    index_of_dispersion : float
        Index of Dispersion (Var/Mean of monthly counts).  NaN when
        insufficient months for sample variance (n_total_months <= 1).
    fail_reason : str or None
        Human-readable explanation of the first criterion that caused failure,
        or None when the gate passed.
    """

    passed: bool
    n_activations: int
    n_active_months: int
    max_monthly_share: float
    mean_tpm: float
    index_of_dispersion: float = float("nan")
    """Index of Dispersion (Var/Mean of monthly counts).  NaN when
    insufficient months for sample variance (n_total_months <= 1)."""
    fail_reason: Optional[str] = None


@dataclass
class WalkForwardConfig:
    """Parameters for walk-forward out-of-sample validation.

    After in-sample discovery, the OOS portion (controlled by
    ``DiscoveryConfig.train_ratio``) is divided into ``n_splits`` equal
    windows.  Each discovered event is replayed on every window via
    ``EventCandidate.apply()`` and evaluated against the Consistency Gate.
    An event is declared OOS-stable when it passes in at least
    ``min_pass_rate`` fraction of the windows.

    Attributes
    ----------
    n_splits : int
        Number of OOS windows to evaluate.  More splits → finer-grained
        stability signal but shorter individual windows.  The OOS period
        is divided equally; the last window absorbs any remainder.
    min_pass_rate : float
        Minimum fraction of OOS windows in which an event must pass the gate
        to be considered OOS-stable (0–1).  Default 0.6 means at least 3 out
        of 5 (or 2 out of 3) windows must pass.
    oos_gate_params : GateParams or None
        Gate thresholds to use for OOS evaluation.  When ``None`` (default),
        the IS ``gate_params`` are used unchanged — both ``min_tpm`` and
        ``max_dispersion`` are rate/ratio invariant and remain valid for
        any window length.
    """

    n_splits: int = 3
    min_pass_rate: float = 0.6
    oos_gate_params: Optional[GateParams] = None


@dataclass
class FoldResult:
    """Consistency gate outcome for one OOS validation fold.

    Attributes
    ----------
    fold_idx : int
        Zero-based index of this fold within the walk-forward sequence.
    n_rows : int
        Number of bars in this fold.
    gate_result : GateResult
        Full gate evaluation on the fold, including fail_reason when failed.
    """

    fold_idx: int
    n_rows: int
    gate_result: GateResult

    @property
    def passed(self) -> bool:
        """True when the gate passed on this fold."""
        return self.gate_result.passed


@dataclass
class ValidationResult:
    """Aggregated walk-forward OOS validation result for one event.

    Attributes
    ----------
    n_folds : int
        Total number of OOS folds evaluated.
    n_passed : int
        Number of folds in which the event passed the gate.
    pass_rate : float
        ``n_passed / n_folds``.
    passed : bool
        True when ``pass_rate >= WalkForwardConfig.min_pass_rate``.
    fold_results : list[FoldResult]
        Per-fold detail, ordered chronologically.
    """

    n_folds: int
    n_passed: int
    pass_rate: float
    passed: bool
    fold_results: list[FoldResult]


@dataclass
class EventComponent:
    """Metadata describing a single boolean condition within an event.

    For a simple event this is the only component.  For AND-composed events
    the parent ``EventCandidate`` holds a list of components, one per
    constituent condition.

    Attributes
    ----------
    source_feature : str
        Name of the original KPI column (e.g. ``close_rsi_25``).
    transform : str
        Transform applied: ``identity``, ``rolling_pctrank``,
        ``rolling_zscore``, ``delta``, ``binary_native``,
        ``categorical_onehot``, or ``and_composition``.
    transform_params : dict
        Parameters of the transform (e.g. ``{"window": 96}`` for pctrank).
    transformed_col : str
        Name of the column *after* the transform was applied.
    threshold : float
        Numerical threshold used to binarise the transformed series.
    threshold_type : str
        Label describing how the threshold was chosen (e.g.
        ``distributional_p05``, ``theoretical_z-2.0``).
    direction : str
        ``"below"`` (series < threshold) or ``"above"`` (series > threshold).
    event_type : str
        ``"threshold"`` for a persistent boolean, ``"crossing"`` for a
        one-bar transition signal.
    expression : str
        Human-readable string representation of the condition, e.g.
        ``"pr_close_rsi_25_96 > 0.85"``.
    """

    source_feature: str
    transform: str
    transform_params: dict
    transformed_col: str
    threshold: float
    threshold_type: str
    direction: str   # "below" | "above"
    event_type: str  # "threshold" | "crossing"
    expression: str
    source_cols: list = field(default_factory=list)
    """Original native column names used to build ``source_feature``.

    Empty for arity-1 (native) features.  For arity-2 features, contains
    ``[col_a, col_b]``; for arity-3, ``[value_col, lower_col, upper_col]``.
    Used by ``EventCandidate.apply()`` to reconstruct the feature series on
    new data without requiring the full FeatureGenerator to run.
    """
    event_formula: str = ""
    """Human-readable mathematical formula for this component.

    Shows the explicit computation chain from raw columns to boolean condition,
    using standard mathematical notation rather than SQL or encoded names.

    Examples::

        "close - rsi14 < 0.05"
        "pctrank(close - rsi14, w=168) < 0.10"
        "zscore(close / rsi14, w=48) > 1.5"
        "Δ(close, lag=1) crosses ↓ 0.0"
        "bb_pct_b(close, bb_lower, bb_upper) < 0.20"
    """
    sql_expression: str = ""
    """DuckDB-compatible SQL boolean expression that replicates this component.

    Can be used directly in a SELECT or WHERE clause against a table that
    contains the same source columns as the original KPI DataFrame (i.e.
    any table with the same schema as ``EventDiscovery.df``).

    For pctrank, an average-method rank is computed with ``list_filter``
    lambdas (DuckDB ≥ 0.8 required) so that tied values match pandas'
    ``rank(pct=True)`` exactly.  For zscore and delta, standard window
    functions (``AVG``, ``STDDEV_SAMP``, ``LAG``) are used.  Rolling
    transforms reproduce ``min_periods = max(2, window // 2)`` via a
    ``CASE WHEN`` guard.  The ``ORDER BY`` clause uses the timestamp column
    name from ``DiscoveryConfig.timestamp_col`` (defaults to ``"open_dt"``).

    Example usage in DuckDB::

        import duckdb
        rel = duckdb.from_df(ed.df.reset_index())
        rel.query("df", f"SELECT *, ({comp.sql_expression})::INT AS event_active FROM df")
    """
    components: list["EventComponent"] = field(default_factory=list)
    """Constituent sub-components for AND-composed events (``transform == "and_composition"``).

    Empty for all other event types.  Storing this as a declared dataclass
    field (rather than a dynamic attribute) ensures the list survives
    serialisation (pickle, deepcopy, JSON round-trips).
    """


@dataclass
class ActivationStats:
    """Aggregated statistics about an event's activation pattern.

    These are computed after the Consistency Gate to provide a richer
    picture than the gate metrics alone.

    Attributes
    ----------
    n_activations : int
        Total bar-level activations over the full dataset.
    n_active_months : int
        Number of calendar months with at least one activation.
    zero_months : int
        Number of calendar months with zero activations (complement of
        n_active_months relative to the full date range).
    max_monthly_share : float
        Fraction of activations concentrated in the busiest month.
    mean_tpm : float
        Average activations per calendar month.
    index_of_dispersion : float
        Index of Dispersion (Var/Mean of monthly counts).  NaN when
        insufficient months for sample variance (n_total_months <= 1).
    """

    n_activations: int
    n_active_months: int
    zero_months: int
    max_monthly_share: float
    mean_tpm: float
    index_of_dispersion: float = float("nan")


@dataclass
class RawEvent:
    """Intermediate event representation used inside the pipeline.

    Created by Step 3 (EventGenerator), annotated with a GateResult by
    Step 4 (ConsistencyGate), and consumed by Step 5 (ANDComposer) before
    being promoted to EventCandidate.

    Attributes
    ----------
    series : pd.Series
        Boolean (0/1/NaN) activation series aligned to the KPI table index.
    component : EventComponent
        Full metadata describing the condition that produced this event.
    gate_result : GateResult or None
        Populated by ConsistencyGate.filter(); None before gate evaluation.
    """

    series: pd.Series
    component: EventComponent
    gate_result: Optional[GateResult] = None

    @property
    def key(self) -> str:
        """Unique string key identifying this specific event (feature + transform + threshold).

        Used for deduplication and logging.  Format:
        ``<transformed_col>__<event_type>__<direction>__<threshold:.8f>``
        """
        c = self.component
        return f"{c.transformed_col}__{c.event_type}__{c.direction}__{c.threshold:.8f}"

    @property
    def transform_key(self) -> str:
        """String key identifying the transform slot, independent of threshold.

        Events sharing the same transform_key differ only in their threshold
        value — the AND composer uses this to avoid pairing redundant events
        (one threshold is a superset of another within the same slot).
        Format: ``<source_feature>__<transform>__<sorted_temporal_params>``.

        Only temporal parameters (``window``, ``lag``) are included; feature-level
        parameters such as ``diffnorm_std`` are excluded to keep the key stable
        and comparable across different assets.
        """
        c = self.component
        _temporal = {"window", "lag"}
        params_str = "_".join(
            f"{k}{v}" for k, v in sorted(c.transform_params.items())
            if k in _temporal
        )
        return f"{c.source_feature}__{c.transform}__{params_str}"


@dataclass
class EventCandidate:
    """Final output artifact produced by the Event Discovery pipeline.

    Consumed by the downstream Alpha Discovery module to evaluate whether
    the event's activations carry predictive power for future returns.

    Attributes
    ----------
    event_id : str
        Human-readable identifier encoding the feature, transform, and index
        (e.g. ``EVT-close_rsi_25-PR-0042``).
    status : str
        Pipeline status tag; always ``"CANDIDATE"`` at this stage.
    components : list[EventComponent]
        One component for simple events, two or three for AND compositions.
    expression : str
        Full human-readable boolean expression (ANDs joined with `` AND ``).
    activation_stats : ActivationStats
        Aggregated statistics about the event's temporal distribution.
    consistency_gate : GateResult
        Gate evaluation result; always ``passed=True`` for candidates
        returned from EventDiscovery.run().
    event_series : pd.Series or None
        The raw 0/1/NaN boolean series.  Stored for downstream use;
        excluded from repr to keep logs readable.
    """

    event_id: str
    status: str
    components: list[EventComponent]
    expression: str
    activation_stats: ActivationStats
    consistency_gate: GateResult
    event_series: Optional[pd.Series] = field(default=None, repr=False)
    validation: Optional[ValidationResult] = field(default=None, repr=False)
    """Walk-forward OOS validation result.

    ``None`` when ``DiscoveryConfig.walk_forward`` was not set.
    Populated by ``EventDiscovery.run()`` after IS discovery completes.
    """
    gate_params: Optional[GateParams] = field(default=None, repr=False)
    """GateParams used during IS discovery.

    Stored so that ``update_event()`` can re-evaluate ``consistency_gate.passed``
    on new data without requiring the original ``DiscoveryConfig``.
    """

    @property
    def sql_expression(self) -> str:
        """DuckDB-compatible SQL boolean expression for the full event.

        For single-component events, returns the component's ``sql_expression``
        directly.  For AND-composed events, joins each component's expression
        with ``AND``, wrapping each in parentheses for clarity.

        Returns an empty string if no component has a populated ``sql_expression``
        (e.g. legacy candidates created before this field was added).

        Example usage in DuckDB::

            import duckdb
            rel = duckdb.from_df(ed.df.reset_index())
            query = f"SELECT *, ({candidate.sql_expression})::INT AS active FROM df"
            rel.query("df", query)
        """
        parts = [c.sql_expression for c in self.components if c.sql_expression]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return " AND ".join(f"({p})" for p in parts)

    @property
    def event_formula(self) -> str:
        """Human-readable mathematical formula for the full event.

        For single-component events, returns the component's ``event_formula``
        directly.  For AND-composed events, joins each component's formula
        with ``AND``.

        Returns an empty string if no component has a populated ``event_formula``.

        Examples::

            "pctrank(close - rsi14, w=168) < 0.10"
            "(pctrank(close - rsi14, w=168) < 0.10) AND (zscore(close / open, w=48) > 1.5)"
        """
        parts = [c.event_formula for c in self.components if c.event_formula]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return " AND ".join(f"({p})" for p in parts)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Reconstruct the event boolean series on new (out-of-sample) data.

        Given a DataFrame that contains the same source columns used during
        training, this method replicates all pipeline steps — feature
        construction, temporal transform, threshold application — using the
        parameters stored in each ``EventComponent``.

        For ``diffnorm`` features, the in-sample standard deviation stored in
        ``transform_params["diffnorm_std"]`` is used as the normaliser, so the
        event preserves exactly the same scale as the training period.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the same column names as the original KPI table.
            Must contain all columns referenced in ``self.components``.

        Returns
        -------
        pd.Series
            Float series (0.0 / 1.0 / NaN) with the same index as ``df``.
            1.0 means the event is active on that bar.

        Raises
        ------
        KeyError
            If a required source column is missing from ``df``.
        """
        result: pd.Series | None = None
        for comp in self.components:
            part = _apply_component(comp, df)
            if result is None:
                result = part
            else:
                # Match pipeline AND behavior: NaN is treated as inactive (0),
                # consistent with the uint8 bitwise AND used in ANDComposer.
                result = (result.fillna(0).astype(bool) & part.fillna(0).astype(bool)).astype(float)
        if result is None:
            return pd.Series(float("nan"), index=df.index)
        return result

    def to_dict(self) -> dict:
        """Serialise the candidate to a flat dictionary for DataFrame construction.

        The ``components`` key contains a list of per-component dicts.
        All other keys are scalar values suitable for a summary DataFrame row.
        OOS validation columns (``oos_*``) are included only when
        ``self.validation`` is populated.
        """
        d: dict = {
            "event_id": self.event_id,
            "status": self.status,
            "expression": self.expression,
            "n_activations": self.activation_stats.n_activations,
            "n_active_months": self.activation_stats.n_active_months,
            "zero_months": self.activation_stats.zero_months,
            "max_monthly_share": self.activation_stats.max_monthly_share,
            "mean_tpm": self.activation_stats.mean_tpm,
            "index_of_dispersion": self.activation_stats.index_of_dispersion,
            "gate_passed": self.consistency_gate.passed,
        }
        if self.validation is not None:
            d["oos_pass_rate"] = self.validation.pass_rate
            d["oos_n_passed"] = self.validation.n_passed
            d["oos_n_folds"] = self.validation.n_folds
            d["oos_stable"] = self.validation.passed
        d["components"] = [
            {
                "source_feature": c.source_feature,
                "transform": c.transform,
                "transform_params": c.transform_params,
                "threshold": c.threshold,
                "threshold_type": c.threshold_type,
                "direction": c.direction,
                "event_type": c.event_type,
                "expression": c.expression,
                "event_formula": c.event_formula,
                "sql_expression": c.sql_expression,
            }
            for c in self.components
        ]
        return d

    def persist(self, path) -> None:
        """Save this candidate to disk as a pickle file.

        Reload with::

            import pickle
            cand = pickle.load(open(path, "rb"))
            cand.apply(df)
        """
        import pickle
        import pathlib
        pathlib.Path(path).write_bytes(pickle.dumps(self))

    def update_event(self, df: pd.DataFrame) -> None:
        """Re-evaluate the event on a new DataFrame, updating internal state in place.

        Replays the event on ``df`` using the fixed IS parameters (thresholds,
        windows, transform) and updates:

        - ``event_series`` — new boolean activation series on ``df``
        - ``activation_stats`` — recomputed from the new series
        - ``consistency_gate`` — gate metrics and ``passed`` recomputed on the
          new series using the original ``gate_params`` (if available; ``passed``
          is set to ``None``-equivalent ``False`` when ``gate_params`` is absent)

        Thresholds and transform parameters are **never** recalibrated — they
        remain fixed from the original IS discovery run.  This makes the method
        suitable for out-of-sample evaluation across different time horizons.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the same source columns used during discovery.
            Must contain a timestamp column or DatetimeIndex so that monthly
            statistics can be recomputed.

        Raises
        ------
        ValueError
            If ``df`` is empty or lacks the required source columns.
        """
        from .consistency_gate import ConsistencyGate, _build_month_index

        new_series = self.apply(df)
        self.event_series = new_series

        # Rebuild timestamp series from the DatetimeIndex set by apply()
        timestamps = pd.Series(new_series.index, dtype="datetime64[ns]")
        month_index, n_total_months = _build_month_index(timestamps)

        if self.gate_params is not None:
            self.consistency_gate = ConsistencyGate(self.gate_params).evaluate_series(
                new_series, month_index, n_total_months
            )
        else:
            # No gate_params stored: recompute metrics only, passed stays False
            gate = ConsistencyGate(GateParams()).evaluate_series(
                new_series, month_index, n_total_months
            )
            self.consistency_gate = GateResult(
                passed=False,
                n_activations=gate.n_activations,
                n_active_months=gate.n_active_months,
                max_monthly_share=gate.max_monthly_share,
                mean_tpm=gate.mean_tpm,
                index_of_dispersion=gate.index_of_dispersion,
                fail_reason="gate_params not stored on this candidate",
            )

        g = self.consistency_gate
        active = new_series.fillna(0).astype(bool)
        month_counts = {}
        for i, flag in enumerate(active):
            if flag:
                m = month_index[i]
                month_counts[m] = month_counts.get(m, 0) + 1
        zero_months = n_total_months - g.n_active_months
        self.activation_stats = ActivationStats(
            n_activations=g.n_activations,
            n_active_months=g.n_active_months,
            zero_months=zero_months,
            max_monthly_share=g.max_monthly_share,
            mean_tpm=g.mean_tpm,
            index_of_dispersion=g.index_of_dispersion,
        )


@dataclass
class CustomEvent:
    """User-defined event specified by a pandas-eval formula.

    Lets a user inject a hand-built event hypothesis straight into Alpha
    Discovery (M2) and Rule Discovery (M3) without running the automatic
    Event Discovery pipeline (M1).  The formula is evaluated with
    ``pd.DataFrame.eval`` so it can reference *any* column present in the
    DataFrame — including proprietary indicators that the FeatureGenerator
    would never produce.

    Parameters
    ----------
    formula : str
        Boolean expression compatible with ``pd.DataFrame.eval()``.
        Examples: ``"close_adj_v2 < 100"``, ``"rsi_14 > 70 and volume > 1e6"``.
    name : str, optional
        Human-readable label used in reports.  Defaults to the formula text.

    Examples
    --------
    >>> ev = CustomEvent("close_adj_v2 < 100", name="close_below_100")
    >>> series = ev.apply(df)            # boolean Series aligned to df.index
    >>> cand = ev.to_event_candidate(df)  # EventCandidate for the M2/M3 flow
    """

    formula: str
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.formula

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate the formula on ``df`` and return a boolean activation Series.

        NaN results (e.g. comparisons against missing values) are treated as
        inactive (``False``), matching the gate's treatment of NaN bars.

        Raises
        ------
        ValueError
            If the formula references unknown columns or is otherwise invalid.
        """
        return _eval_custom_formula(self.formula, df).astype(bool)

    def to_event_candidate(
        self,
        df: pd.DataFrame,
        gate_params: Optional[GateParams] = None,
    ) -> "EventCandidate":
        """Build an :class:`EventCandidate` that replays this formula on any frame.

        The returned candidate carries a single ``custom_formula`` component
        whose ``transform_params["formula"]`` holds the original expression, so
        ``EventCandidate.apply`` re-evaluates it deterministically on new data.
        The ``consistency_gate`` and ``activation_stats`` are placeholders here;
        :func:`forge` populates them after running the Consistency Gate.

        Parameters
        ----------
        df : pd.DataFrame
            Frame the formula is evaluated on to seed the cached
            ``event_series`` (aligned to ``df.index``).
        gate_params : GateParams, optional
            Stored on the candidate so ``update_event`` can re-evaluate the
            gate on new horizons without the original config.
        """
        bool_series = self.apply(df)
        series = bool_series.astype(float)
        n_act = int(bool_series.sum())

        component = EventComponent(
            source_feature=self.name,
            transform="custom_formula",
            transform_params={"formula": self.formula},
            transformed_col=self.name,
            threshold=float("nan"),
            threshold_type="custom",
            direction="above",
            event_type="threshold",
            expression=self.formula,
            sql_expression=self.formula,
            event_formula=self.formula,
        )
        placeholder_gate = GateResult(
            passed=False,
            n_activations=n_act,
            n_active_months=0,
            max_monthly_share=float("nan"),
            mean_tpm=float("nan"),
            fail_reason="gate not yet evaluated",
        )
        placeholder_stats = ActivationStats(
            n_activations=n_act,
            n_active_months=0,
            zero_months=0,
            max_monthly_share=float("nan"),
            mean_tpm=float("nan"),
            index_of_dispersion=float("nan"),
        )
        return EventCandidate(
            event_id=f"CUSTOM-{self.name}",
            status="CANDIDATE",
            components=[component],
            expression=self.formula,
            activation_stats=placeholder_stats,
            consistency_gate=placeholder_gate,
            event_series=series,
            gate_params=gate_params,
        )


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _eval_custom_formula(formula: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate a ``CustomEvent`` formula on ``df`` into a float 0/1 Series.

    NaN results are coerced to 0 (inactive).  A constant expression that
    ``df.eval`` collapses to a scalar is broadcast across ``df.index``.

    Raises
    ------
    ValueError
        If the formula cannot be evaluated (e.g. unknown column).
    """
    try:
        result = df.eval(formula)
    except Exception as exc:
        raise ValueError(
            f"CustomEvent formula '{formula}' failed to evaluate: {exc}"
        ) from exc
    if not isinstance(result, pd.Series):
        # Constant expression → broadcast scalar across the frame.
        result = pd.Series(result, index=df.index)
    return result.fillna(False).astype(float)


def build_feature_series(comp: "EventComponent", df: pd.DataFrame) -> pd.Series:
    """Reconstruct the *continuous underlying feature* of a component on ``df``.

    This is step 1 of the event replay — feature construction only, before
    any temporal transform or threshold is applied.  It returns the raw
    scale-free feature series (e.g. ``close_rsi_25``, a ``ratio_…``, a
    ``spread_…`` or a ternary position feature), which is exactly the
    "feature continua sottostante" that Alpha Discovery measures the
    Information Coefficient on.

    For ``binary_native`` and ``categorical_onehot`` components there is no
    continuous feature underneath; the raw source column is returned as-is so
    callers can still rank it if they choose to.

    Parameters
    ----------
    comp : EventComponent
        A fully populated component (``source_cols`` and ``transform_params``
        must contain all values stored during training).
    df : pd.DataFrame
        DataFrame with the native source columns referenced by ``comp``.

    Returns
    -------
    pd.Series
        The continuous feature series aligned to ``df.index`` (before any
        temporal transform).

    Raises
    ------
    KeyError
        If a required source column is missing, or if a ``diffnorm`` feature
        was created without its stored in-sample standard deviation.
    """
    sf = comp.source_feature

    if comp.transform == "custom_formula":
        # User-defined formula: the "continuous feature" is the 0/1 activation
        # itself (Alpha Discovery measures a point-biserial IC against it).
        return _eval_custom_formula(comp.transform_params["formula"], df)

    if comp.transform in ("binary_native", "categorical_onehot"):
        # No continuous feature underneath — return the raw column.
        return df[sf]

    sc = comp.source_cols
    if sf.startswith("ratio_") and len(sc) == 2:
        return (df[sc[0]] / df[sc[1]]).replace([float("inf"), float("-inf")], float("nan"))
    if sf.startswith("spread_") and len(sc) == 2:
        return ((df[sc[0]] - df[sc[1]]) / df[sc[1]]).replace([float("inf"), float("-inf")], float("nan"))
    if sf.startswith("diffnorm_") and len(sc) == 2:
        std = comp.transform_params.get("diffnorm_std")
        if std is None:
            raise KeyError(
                f"'diffnorm_std' missing from transform_params of component "
                f"'{comp.expression}'. Was this candidate created with an older "
                "version of EventDiscovery?"
            )
        if std == 0:
            return pd.Series(float("nan"), index=df.index, dtype=float)
        return (df[sc[0]] - df[sc[1]]) / std
    if sf.startswith("bb_pct_b_") and len(sc) == 3:
        val, lower, upper = sc
        denom = df[upper] - df[lower]
        return (df[val] - df[lower]) / denom.replace(0, float("nan"))
    if sf.startswith("pos_") and len(sc) == 3:
        val, mn, mx = sc
        denom = df[mx] - df[mn]
        return (df[val] - df[mn]) / denom.replace(0, float("nan"))
    # Arity-1 native feature — use source_feature name directly
    return df[sf]


def _apply_component(comp: "EventComponent", df: pd.DataFrame) -> pd.Series:
    """Reconstruct a single EventComponent's boolean series on ``df``.

    Steps
    -----
    1. Build the feature series (native, ratio, spread_pct, diffnorm, …).
    2. Apply the temporal transform (identity, pctrank, zscore, delta).
    3. Apply the threshold / crossing condition.

    Parameters
    ----------
    comp : EventComponent
        A fully populated component (source_cols and transform_params
        must contain all values stored during training).
    df : pd.DataFrame
        DataFrame with native source columns.

    Returns
    -------
    pd.Series
        Float 0/1/NaN series aligned to ``df.index``.
    """
    sf = comp.source_feature

    # ── 0. Custom user formula — evaluated verbatim, no transform/threshold ─
    if comp.transform == "custom_formula":
        return _eval_custom_formula(comp.transform_params["formula"], df)

    # ── 1. Feature series ────────────────────────────────────────────────
    if comp.transform in ("binary_native",):
        # Binary: active when column equals the stored threshold value
        raw = df[sf]
        return (raw == comp.threshold).astype(float).where(raw.notna(), float("nan"))

    if comp.transform == "categorical_onehot":
        cls = comp.transform_params.get("class")
        raw = df[sf]
        return (raw == cls).astype(float).where(raw.notna(), float("nan"))

    series = build_feature_series(comp, df)

    # ── 2. Temporal transform ────────────────────────────────────────────
    # Use the same min_periods heuristic as TransformLayer to match NaN positions.
    t = comp.transform
    if t == "rolling_pctrank":
        w = comp.transform_params["window"]
        min_p = max(2, w // 2)
        series = series.rolling(w, min_periods=min_p).rank(pct=True)
    elif t == "rolling_zscore":
        w = comp.transform_params["window"]
        min_p = max(2, w // 2)
        roll = series.rolling(w, min_periods=min_p)
        mu = roll.mean()
        sd = roll.std(ddof=1)  # explicit: match TransformLayer's sample std
        series = (series - mu) / sd.replace(0, float("nan"))
    elif t == "delta":
        series = series.diff(comp.transform_params["lag"])
    # identity: no transform

    # ── 3. Threshold / crossing ──────────────────────────────────────────
    thr = comp.threshold
    if comp.event_type == "crossing":
        if comp.direction == "below":
            current  = series < thr
            previous = series.shift(1) >= thr
        else:
            current  = series > thr
            previous = series.shift(1) <= thr
        result = (current & previous).astype(float)
        result[series.isna() | series.shift(1).isna()] = float("nan")
        return result
    else:
        if comp.direction == "below":
            return (series < thr).astype(float).where(series.notna(), float("nan"))
        return (series > thr).astype(float).where(series.notna(), float("nan"))
