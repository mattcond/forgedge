"""Main EventDiscovery orchestrator.

Usage
-----
    from forgedge import EventDiscovery

    df = pd.read_parquet("kpi_table.parquet")
    ed = EventDiscovery(df)
    candidates = ed.run()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .and_composer import ANDComposer
from .classifier import TypeClassifier
from .consistency_gate import ConsistencyGate, _monthly_counts
from .event_generator import EventGenerator
from .feature_generator import FeatureGenerator
from .models import (
    ActivationStats,
    ColumnType,
    EventCandidate,
    EventComponent,
    GateParams,
    GateResult,
    RawEvent,
)
from .transform_layer import TransformLayer


@dataclass
class DiscoveryConfig:
    """Configuration for the EventDiscovery pipeline.

    Attributes
    ----------
    gate_params : GateParams
        Thresholds for the Consistency Gate (Step 4).  Defaults to the
        conservative production settings (min_act=50, min_months=8,
        max_conc=0.40, min_tpm=2.0).
    max_categorical_classes : int
        Categorical columns with more distinct values than this limit are
        classified but excluded from the event generation pipeline.
    scale_free_overrides : dict[str, bool] or None
        Manual scale-free flags for specific columns, passed directly to
        ``TypeClassifier``.  Useful when the automatic heuristic
        produces a wrong result (e.g. short history for RSI).
    timestamp_col : str
        Name of the datetime column in the KPI table.  Also accepted as the
        index name when the DataFrame uses a DatetimeIndex.
    max_and_components : int
        Maximum number of single events to combine in one AND composition
        (2 or 3).  Values above 3 are technically accepted but strongly
        discouraged — they risk structural overfitting.
    """

    gate_params: GateParams = field(default_factory=GateParams)
    max_categorical_classes: int = 20
    scale_free_overrides: Optional[dict[str, bool]] = None
    timestamp_col: str = "open_dt"
    max_and_components: int = 2


class EventDiscovery:
    """FORGE Event Discovery module — Steps 0 through 5.

    Parameters
    ----------
    kpi_table:
        DataFrame with OHLCV columns and technical indicators.
        Must contain a datetime column (default ``open_dt``) or a
        DatetimeIndex.
    config:
        Optional configuration object.  Defaults to sensible values.
    """

    def __init__(
        self,
        kpi_table: pd.DataFrame,
        config: Optional[DiscoveryConfig] = None,
    ):
        self.df = kpi_table.copy()
        self.config = config or DiscoveryConfig()
        self._classifications: Optional[dict] = None
        self._candidates: Optional[list[EventCandidate]] = None
        self._timestamps: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> list[EventCandidate]:
        """Execute the full five-step pipeline and return Event Candidates.

        Pipeline steps
        --------------
        0. **TypeClassifier**: classifies every non-skipped column as
           CONTINUOUS, BINARY, or CATEGORICAL and detects scale-free
           behaviour for continuous columns.

        1. **FeatureGenerator**: builds the extended feature catalog —
           scale-free arity-1 pass-throughs, arity-2 ratio/spread_pct
           features, and arity-3 Bollinger/%B/rolling-range position features.

        2. **TransformLayer**: applies identity (scale-free only), rolling
           pctrank, rolling zscore, and delta transforms to every feature
           in the catalog, producing a flat list of TransformedSeries.

        3. **EventGenerator**: converts each TransformedSeries into threshold
           and crossing boolean events.  Binary and categorical columns are
           handled separately (no transform needed).

        4. **ConsistencyGate**: filters raw events by their temporal
           activation pattern (volume, coverage, concentration, frequency).

        5. **ANDComposer**: pairs (and optionally triples) the passing events
           with logical AND and re-applies the gate on the composed result.

        All passing single events and passing composed events are promoted
        to ``EventCandidate`` objects and stored in ``self._candidates``.

        Returns
        -------
        list[EventCandidate]
            All events (single and AND-composed) that passed the
            Consistency Gate.
        """
        cfg = self.config

        timestamps = self._prepare_timestamps()
        self._timestamps = timestamps

        # Step 0 — classify columns
        classifier = TypeClassifier(
            max_categorical_classes=cfg.max_categorical_classes,
            scale_free_overrides=cfg.scale_free_overrides or {},
        )
        classifications = classifier.fit(self.df)
        self._classifications = classifications

        continuous_cls = {
            col: cls for col, cls in classifications.items()
            if cls.col_type == ColumnType.CONTINUOUS
        }
        binary_cls = {
            col: cls for col, cls in classifications.items()
            if cls.col_type == ColumnType.BINARY
        }
        categorical_cls = {
            col: cls for col, cls in classifications.items()
            if cls.col_type == ColumnType.CATEGORICAL
            and cls.n_distinct <= cfg.max_categorical_classes
        }

        # Step 1 — generate derived features for continuous columns
        fg = FeatureGenerator()
        extended_df, derived_meta = fg.generate(self.df, classifications)
        # expose the full derived feature table (with DatetimeIndex) for debugging
        extended_df.index = self.df.index
        self.df = extended_df

        # Step 2 — apply temporal transforms
        transformer = TransformLayer()
        transformed_series = transformer.transform_all(extended_df, derived_meta)

        # Step 3 — generate boolean events
        ev_gen = EventGenerator()
        raw_events: list[RawEvent] = []

        for ts in transformed_series:
            raw_events.extend(ev_gen.generate_from_transformed(ts, ts_col=cfg.timestamp_col))

        for col in binary_cls:
            if col in self.df.columns:
                raw_events.extend(ev_gen.generate_from_binary(self.df[col], col))

        for col in categorical_cls:
            if col in self.df.columns:
                raw_events.extend(
                    ev_gen.generate_from_categorical(
                        self.df[col], col, cfg.max_categorical_classes
                    )
                )

        # Step 4 — Consistency Gate on single events
        gate = ConsistencyGate(cfg.gate_params)
        passing_single = gate.filter(raw_events, timestamps)

        # Step 5 — AND composition + gate on composed events
        composer = ANDComposer(gate)
        passing_composed = composer.compose(
            passing_single, timestamps, max_components=cfg.max_and_components
        )

        all_passing = passing_single + passing_composed

        # Build EventCandidate objects
        candidates = [
            self._to_candidate(ev, idx, timestamps)
            for idx, ev in enumerate(all_passing)
        ]
        self._candidates = candidates
        return candidates

    def get_classifications(self) -> Optional[dict]:
        """Return the column classifications from Step 0.

        Available only after ``run()`` has been called.  Returns the
        raw ``dict[str, ColumnClassification]`` produced by TypeClassifier,
        which is useful for debugging scale-free detection results and
        verifying that expected columns were not skipped.

        Returns
        -------
        dict[str, ColumnClassification] or None
            None if ``run()`` has not been called yet.
        """
        return self._classifications

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all Event Candidates.

        Flattens each candidate's ``to_dict()`` output (excluding the
        nested ``components`` list) into a single row.  When no candidates
        were found, returns an empty DataFrame with the correct column schema
        so that downstream code can rely on the column names without
        special-casing the empty case.

        Raises
        ------
        RuntimeError
            If called before ``run()``.

        Returns
        -------
        pd.DataFrame
            Columns: event_id, status, expression, n_activations,
            n_active_months, zero_months, max_monthly_share, mean_tpm,
            gate_passed.
        """
        if self._candidates is None:
            raise RuntimeError("Call run() before summary().")
        _summary_cols = [
            "event_id", "status", "expression",
            "n_activations", "n_active_months", "zero_months",
            "max_monthly_share", "mean_tpm", "gate_passed",
        ]
        rows = [c.to_dict() for c in self._candidates]
        if not rows:
            return pd.DataFrame(columns=_summary_cols)
        flat = [{k: v for k, v in row.items() if k != "components"} for row in rows]
        return pd.DataFrame(flat)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_timestamps(self) -> pd.Series:
        """Parse and normalise the timestamp column, setting a DatetimeIndex.

        Accepts the following input formats (checked in order):

        1. **DatetimeIndex** — used as-is; the index name is preserved.
        2. **Datetime-typed column** (``datetime64``, ``pd.Timestamp``) — parsed
           with ``pd.to_datetime`` and no unit conversion needed.
        3. **Numeric column** — the unit (``s`` / ``ms`` / ``us`` / ``ns``) is
           inferred automatically from the median value:

           ============  =================  ========================
           Unit          Median range       Approximate epoch range
           ============  =================  ========================
           seconds       1e9 – 1e10         2001 – 2286
           milliseconds  1e12 – 1e13        2001 – 2286
           microseconds  1e15 – 1e16        2001 – 2286
           nanoseconds   > 1e16             2001 – 2286
           ============  =================  ========================

        4. **String column** — passed directly to ``pd.to_datetime`` for
           ISO-8601 parsing.

        After parsing, ``self.df`` is updated so that:

        * The DatetimeIndex is set to the parsed timestamps.
        * The original timestamp column is **dropped** from ``self.df``
          (it is redundant once the index is set and would confuse the
          TypeClassifier).

        The returned Series has a plain ``RangeIndex`` so that pipeline
        steps that use positional alignment continue to work correctly.
        The DatetimeIndex is re-attached to ``event_series`` in
        ``_to_candidate`` so that callers can call ``resample`` directly.

        Raises
        ------
        ValueError
            If no timestamp source is found.

        Returns
        -------
        pd.Series
            Datetime series with RangeIndex, length ``n_rows``.
        """
        ts_col = self.config.timestamp_col

        # ── Case 1: already a DatetimeIndex ──────────────────────────────
        if isinstance(self.df.index, pd.DatetimeIndex):
            ts = self.df.index.to_series().reset_index(drop=True)
            # keep DatetimeIndex on self.df (drop=True avoids duplicate col)
            self.df = self.df.reset_index(drop=True)
            self.df.index = pd.DatetimeIndex(ts.values, name=ts_col)
            self.df = self.df.sort_index()
            return self.df.index.to_series().reset_index(drop=True)

        # ── Case 2 / 3 / 4: column-based ─────────────────────────────────
        if ts_col not in self.df.columns:
            raise ValueError(
                f"Timestamp column '{ts_col}' not found and index is not a "
                "DatetimeIndex.  Pass the correct column name via "
                "DiscoveryConfig(timestamp_col='...')."
            )

        raw = self.df[ts_col]

        if pd.api.types.is_datetime64_any_dtype(raw):
            # already datetime
            parsed = raw.reset_index(drop=True)
        elif pd.api.types.is_numeric_dtype(raw):
            unit = _infer_timestamp_unit(raw)
            parsed = pd.to_datetime(raw, unit=unit).reset_index(drop=True)
        else:
            # string / object — let pandas infer the format
            parsed = pd.to_datetime(raw).reset_index(drop=True)

        self.df = self.df.drop(columns=[ts_col]).reset_index(drop=True)
        self.df.index = pd.DatetimeIndex(parsed.values, name=ts_col)
        self.df = self.df.sort_index()
        return self.df.index.to_series().reset_index(drop=True)

    def _to_candidate(
        self,
        ev: RawEvent,
        idx: int,
        timestamps: pd.Series,
    ) -> EventCandidate:
        """Promote a passing RawEvent to an EventCandidate.

        Extracts the component list (handling the AND-composition case where
        components are stored as a ``_components`` dynamic attribute), computes
        the zero-months statistic, aggregates all stats into an
        ``ActivationStats`` object, and constructs the final ``EventCandidate``.

        Parameters
        ----------
        ev : RawEvent
            A passing event (gate_result.passed == True).
        idx : int
            Sequential index within the final ``all_passing`` list, used to
            produce a unique event ID suffix.
        timestamps : pd.Series
            Datetime series needed for zero-month counting.

        Returns
        -------
        EventCandidate
        """
        g = ev.gate_result  # always set after gate.filter / composer.compose

        # Retrieve actual components list
        comp = ev.component
        if comp.transform == "and_composition":
            components: list[EventComponent] = getattr(comp, "_components", [comp])
        else:
            components = [comp]

        zero_months = _count_zero_months(ev.series, timestamps)

        stats = ActivationStats(
            n_activations=g.n_activations if g else 0,
            n_active_months=g.n_active_months if g else 0,
            zero_months=zero_months,
            max_monthly_share=g.max_monthly_share if g else float("nan"),
            mean_tpm=g.mean_tpm if g else float("nan"),
        )

        # Attach DatetimeIndex so callers can call .resample() directly
        series_dt = ev.series.copy()
        series_dt.index = pd.DatetimeIndex(timestamps.values, name=self.config.timestamp_col)

        return EventCandidate(
            event_id=_make_event_id(components, idx),
            status="CANDIDATE",
            components=components,
            expression=comp.expression,
            activation_stats=stats,
            consistency_gate=g,
            event_series=series_dt,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_timestamp_unit(series: pd.Series) -> str:
    """Infer the Unix timestamp unit from the median value of a numeric series.

    Boundaries are chosen so that any timestamp between year 2001 and 2286
    maps to exactly one unit bucket.

    Parameters
    ----------
    series : pd.Series
        Numeric column whose values are Unix timestamps.

    Returns
    -------
    str
        One of ``"s"``, ``"ms"``, ``"us"``, ``"ns"``.
    """
    median_val = float(series.median())
    if median_val < 1e10:
        return "s"
    if median_val < 1e13:
        return "ms"
    if median_val < 1e16:
        return "us"
    return "ns"


def _count_zero_months(series: pd.Series, timestamps: pd.Series) -> int:
    """Count calendar months in which the event never fired.

    Fills the full date range (from first to last timestamp) with zeros so
    that months that are completely absent from the activation series are
    counted correctly.  A month is "zero" if its activation count is 0.

    Parameters
    ----------
    series : pd.Series
        Boolean (0/1/NaN) event series.
    timestamps : pd.Series
        Datetime series aligned to ``series``.

    Returns
    -------
    int
        Number of calendar months with zero activations.
    """
    # Strip both indices before groupby to avoid DatetimeIndex vs RangeIndex
    # alignment failures (event_series carries DatetimeIndex from the pipeline).
    if isinstance(series.index, pd.DatetimeIndex):
        periods = series.index.to_period("M")
    else:
        periods = pd.DatetimeIndex(timestamps.values).to_period("M")
    active_vals = series.fillna(0).values.astype(bool)
    counts = pd.Series(active_vals, index=periods).groupby(level=0).sum()
    all_months = pd.period_range(periods.min(), periods.max(), freq="M")
    counts = counts.reindex(all_months, fill_value=0)
    return int((counts == 0).sum())


def _make_event_id(components: list[EventComponent], idx: int) -> str:
    """Build a human-readable event ID string.

    Format: ``EVT-{source_feature}-{transform_abbrs}-{idx:04d}``

    The source feature is taken from the first component and truncated to
    20 characters.  Transform abbreviations are joined with ``x``
    (e.g. ``PRxZS`` for a pctrank × zscore AND composition).

    Parameters
    ----------
    components : list[EventComponent]
        Component list (length 1 for simple events, 2-3 for AND compositions).
    idx : int
        Sequential index suffix for uniqueness.

    Returns
    -------
    str
        Event ID string, e.g. ``"EVT-close_rsi_25-PR-0042"``.
    """
    if not components:
        return f"EVT-{idx:05d}"
    _abbr = {
        "identity": "ID",
        "rolling_pctrank": "PR",
        "rolling_zscore": "ZS",
        "delta": "DL",
        "binary_native": "BN",
        "categorical_onehot": "OH",
        "and_composition": "AND",
    }
    parts = [_abbr.get(c.transform, c.transform[:3].upper()) for c in components]
    feature = components[0].source_feature[:20].replace(" ", "_")
    transform_str = "x".join(parts)
    return f"EVT-{feature}-{transform_str}-{idx:04d}"
