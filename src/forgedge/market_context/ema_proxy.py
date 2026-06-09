"""v1.0 regime classifier — :class:`EMAProxyClassifier`.

Computes the ratio between a fast and a slow EMA as a proxy for the market
trend, and discretises it into a regime via configurable thresholds.

The EMA columns are looked up in the KPI Table by the FORGE naming
convention ``{col}_{type}_{period:02d}`` (e.g. ``close_ema_09``).  If they
exist (as in the QHF ``CandleKPI`` that precomputes them) they are used
directly; otherwise they are computed inline from ``source_col`` and the
intermediate value is discarded — it is never added to the KPI Table.

Only ``source_col`` (``close`` by default) needs to be present in the user's
data: the EMA indicators required to compute the regime are derived here.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .models import EMAProxyConfig, RegimeClassifier


class EMAProxyClassifier(RegimeClassifier):
    """Classify market regime from the fast/slow EMA ratio.

    Parameters
    ----------
    config : EMAProxyConfig
        EMA periods, source column and ratio thresholds.
    labels : list[str]
        Regime labels, ordered from most bearish to most bullish.  There must
        be exactly one more label than there are thresholds.

    Raises
    ------
    ValueError
        If the number of thresholds does not equal ``len(labels) - 1``, or if
        the thresholds are not strictly ascending.
    """

    def __init__(self, config: EMAProxyConfig, labels: List[str]):
        thresholds = list(config.thresholds)
        if len(thresholds) != len(labels) - 1:
            raise ValueError(
                f"EMAProxyClassifier needs exactly len(labels)-1 thresholds: "
                f"got {len(thresholds)} thresholds for {len(labels)} labels "
                f"({labels}). Expected {len(labels) - 1}."
            )
        if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError(
                f"thresholds must be strictly ascending, got {thresholds}."
            )
        self.config = config
        self.labels = list(labels)

    # ------------------------------------------------------------------
    # RegimeClassifier interface
    # ------------------------------------------------------------------

    def classify(self, kpi_table: pd.DataFrame) -> pd.Series:
        """Return an ordered categorical regime label for every bar.

        Steps
        -----
        1. Look up or compute the fast EMA (``ema_short``).
        2. Look up or compute the slow EMA (``ema_long``).
        3. Compute ``ratio = ema_short / ema_long``.
        4. Discretise the ratio into regime labels via the thresholds.

        Bars where the ratio is undefined (e.g. leading NaNs in a precomputed
        EMA column) are labelled ``NaN``.

        Raises
        ------
        KeyError
            If ``source_col`` is not present in the KPI Table.
        """
        cfg = self.config
        if cfg.source_col not in kpi_table.columns:
            raise KeyError(
                f"source_col '{cfg.source_col}' not found in KPI Table. "
                f"The Market Context module requires at least this column to "
                f"compute the EMA proxy."
            )

        ema_short = self._get_or_compute_ema(kpi_table, cfg.short_period)
        ema_long = self._get_or_compute_ema(kpi_table, cfg.long_period)

        ratio = ema_short / ema_long.replace(0, np.nan)

        # bins: [0, t0, t1, t2, t3, +inf]  →  len(labels) buckets
        bins = [0.0] + list(cfg.thresholds) + [np.inf]
        regime = pd.cut(ratio, bins=bins, labels=self.labels, ordered=True)

        return pd.Series(
            pd.Categorical(regime, categories=self.labels, ordered=True),
            index=kpi_table.index,
            name="regime",
        )

    def get_labels(self) -> List[str]:
        """Return the ordered label list (most bearish first)."""
        return list(self.labels)

    def get_config(self) -> dict:
        """Return the configuration used, for traceability in the report."""
        return {
            "classifier": "ema_proxy",
            "source_col": self.config.source_col,
            "short_period": self.config.short_period,
            "long_period": self.config.long_period,
            "thresholds": list(self.config.thresholds),
            "labels": list(self.labels),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_compute_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Return the EMA for ``period`` — from the table if present, else inline.

        Looks for the column ``{source_col}_ema_{period:02d}`` following the
        FORGE naming convention.  When absent, the EMA is computed inline with
        ``ewm(span=period, adjust=False)`` (the recursive form used by standard
        TA libraries, matching how precomputed columns are produced) and the
        result is *not* written back to the KPI Table.
        """
        col = f"{self.config.source_col}_ema_{period:02d}"
        if col in df.columns:
            return df[col]
        return (
            df[self.config.source_col]
            .ewm(span=period, adjust=False)
            .mean()
        )
