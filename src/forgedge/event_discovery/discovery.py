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
        """Execute the full five-step pipeline and return Event Candidates."""
        cfg = self.config

        timestamps = self._extract_timestamps()
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

        # Step 2 — apply temporal transforms
        transformer = TransformLayer()
        transformed_series = transformer.transform_all(extended_df, derived_meta)

        # Step 3 — generate boolean events
        ev_gen = EventGenerator()
        raw_events: list[RawEvent] = []

        for ts in transformed_series:
            raw_events.extend(ev_gen.generate_from_transformed(ts))

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
        """Return column classifications from Step 0 (available after run())."""
        return self._classifications

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all Event Candidates."""
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

    def _extract_timestamps(self) -> pd.Series:
        ts_col = self.config.timestamp_col
        if ts_col in self.df.columns:
            ts = pd.to_datetime(self.df[ts_col])
            self.df = self.df.reset_index(drop=True)
            return ts.reset_index(drop=True)

        if isinstance(self.df.index, pd.DatetimeIndex):
            ts = self.df.index.to_series()
            self.df = self.df.reset_index(drop=False)
            return ts.reset_index(drop=True)

        raise ValueError(
            f"No datetime column '{ts_col}' found and index is not DatetimeIndex. "
            "Provide a DataFrame with an 'open_dt' column or a DatetimeIndex."
        )

    def _to_candidate(
        self,
        ev: RawEvent,
        idx: int,
        timestamps: pd.Series,
    ) -> EventCandidate:
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

        return EventCandidate(
            event_id=_make_event_id(components, idx),
            status="CANDIDATE",
            components=components,
            expression=comp.expression,
            activation_stats=stats,
            consistency_gate=g,
            event_series=ev.series,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_zero_months(series: pd.Series, timestamps: pd.Series) -> int:
    active = series.fillna(0).astype(bool)
    periods = timestamps.dt.to_period("M")
    counts = active.groupby(periods).sum()
    all_months = pd.period_range(periods.min(), periods.max(), freq="M")
    counts = counts.reindex(all_months, fill_value=0)
    return int((counts == 0).sum())


def _make_event_id(components: list[EventComponent], idx: int) -> str:
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
