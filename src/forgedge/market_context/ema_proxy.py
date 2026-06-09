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

from typing import List, Optional

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

        mode = config.threshold_mode.lower().strip()
        if mode not in ("fixed", "balanced"):
            raise ValueError(
                f"threshold_mode must be 'fixed' or 'balanced', got "
                f"'{config.threshold_mode}'."
            )
        basis = config.threshold_basis.lower().strip()
        if basis not in ("global", "expanding"):
            raise ValueError(
                f"threshold_basis must be 'global' or 'expanding', got "
                f"'{config.threshold_basis}'."
            )
        if config.threshold_warmup < 1:
            raise ValueError(
                f"threshold_warmup must be >= 1, got {config.threshold_warmup}."
            )
        target = list(config.target_distribution)
        if mode == "balanced":
            if len(target) != len(labels):
                raise ValueError(
                    f"target_distribution needs one weight per label: got "
                    f"{len(target)} for {len(labels)} labels."
                )
            if any(p <= 0 for p in target):
                raise ValueError(
                    f"target_distribution weights must all be > 0, got {target}."
                )
            total = float(sum(target))
            target = [p / total for p in target]  # normalise relative weights

        self.config = config
        self.labels = list(labels)
        self.threshold_mode = mode
        self.threshold_basis = basis
        self.target_distribution = target
        # Populated by classify(): the cut points actually used (the last bar's,
        # for "expanding"), the mode that produced them ("fixed" / "balanced")
        # and the basis ("global" / "expanding"), for report traceability.
        self.resolved_thresholds: Optional[List[float]] = None
        self.resolved_threshold_mode: Optional[str] = None
        self.resolved_threshold_basis: Optional[str] = None

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

        if self.threshold_mode == "balanced" and self.threshold_basis == "expanding":
            codes = self._classify_balanced_expanding(ratio)
        else:
            thresholds, mode_used = self._resolve_thresholds(ratio)
            self.resolved_thresholds = thresholds
            self.resolved_threshold_mode = mode_used
            self.resolved_threshold_basis = (
                "global" if self.threshold_mode == "balanced" else None
            )
            codes = self._cut_codes(ratio.to_numpy(), np.asarray(thresholds))

        return pd.Series(
            pd.Categorical.from_codes(codes, categories=self.labels, ordered=True),
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
            "threshold_mode": self.threshold_mode,
            "threshold_basis": self.threshold_basis,
            "threshold_warmup": self.config.threshold_warmup,
            "thresholds": list(self.config.thresholds),
            "target_distribution": list(self.target_distribution),
            "resolved_thresholds": (
                list(self.resolved_thresholds)
                if self.resolved_thresholds is not None
                else None
            ),
            "resolved_threshold_mode": self.resolved_threshold_mode,
            "resolved_threshold_basis": self.resolved_threshold_basis,
            "labels": list(self.labels),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cut_codes(self, values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Map ratio values to integer label codes against a single threshold row.

        Reproduces ``pd.cut`` with ``right=True`` on bins
        ``[0, t0, …, t_{k-1}, +inf]``: a value's code is the count of thresholds
        it strictly exceeds.  ``NaN`` values get code ``-1`` (→ NaN category).
        """
        codes = (values[:, None] > thresholds[None, :]).sum(axis=1).astype(int)
        codes[np.isnan(values)] = -1
        return codes

    def _resolve_thresholds(self, ratio: pd.Series):
        """Return the global cut points and the mode that produced them.

        In ``"balanced"`` mode the thresholds are the quantiles of the EMA
        ratio at the cumulative boundaries of ``target_distribution``, computed
        **once over the whole series** — so the regime frequencies match that
        target exactly (but the labelling is not causal).  Falls back to the
        configured fixed ``thresholds`` when balancing is off or the ratio is
        degenerate (too few distinct values for strictly-increasing quantiles).
        """
        if self.threshold_mode == "balanced":
            cum = np.cumsum(self.target_distribution)[:-1]  # len = n_labels - 1
            valid = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) >= len(self.labels):
                q = [float(x) for x in valid.quantile(cum).to_numpy()]
                if all(b > a for a, b in zip(q, q[1:])):
                    return q, "balanced"
            # Degenerate ratio → fall back to the fixed thresholds.
        return list(self.config.thresholds), "fixed"

    def _classify_balanced_expanding(self, ratio: pd.Series) -> np.ndarray:
        """Causal ``"balanced"`` classification — quantiles over ``[0..t]`` only.

        For every bar the cut points are the expanding quantiles of the ratio
        up to and including that bar, so a label never depends on future data.
        The first ``threshold_warmup`` bars (insufficient history) fall back to
        the fixed ``thresholds``.  The target distribution then holds only
        approximately.
        """
        cum = np.cumsum(self.target_distribution)[:-1]
        warm = int(self.config.threshold_warmup)
        clean = ratio.replace([np.inf, -np.inf], np.nan)
        expanding = clean.expanding(min_periods=warm)

        thr_df = pd.concat([expanding.quantile(q) for q in cum], axis=1)
        thr_df.columns = range(len(cum))
        # Pre-warmup rows have NaN quantiles → use the fixed thresholds there.
        for j, fixed_val in enumerate(self.config.thresholds):
            thr_df[j] = thr_df[j].fillna(fixed_val)

        thr_arr = thr_df.to_numpy()
        values = ratio.to_numpy()
        codes = (values[:, None] > thr_arr).sum(axis=1).astype(int)
        codes[np.isnan(values)] = -1

        self.resolved_thresholds = [float(x) for x in thr_arr[-1]]
        self.resolved_threshold_mode = "balanced"
        self.resolved_threshold_basis = "expanding"
        return codes

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
