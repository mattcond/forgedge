"""Step 0 — Type classification and scale-free detection."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .models import ColumnClassification, ColumnType

_SKIP_COLS = {
    "open_dt", "close_time", "timestamp", "date", "datetime",
    "regime", "regime_stable", "open", "high", "low", "volume",
}


class TypeClassifier:
    """Classifies each DataFrame column as continuous, binary, or categorical.

    Continuous columns are further assessed for scale-free behaviour.
    The scale-free check is conservative: when in doubt, returns False.
    Users may override via ``scale_free_overrides``.

    Parameters
    ----------
    max_categorical_classes : int
        Columns classified as CATEGORICAL with more distinct values than this
        limit are still stored in the classification dict but are excluded from
        the event generation pipeline.
    scale_free_overrides : dict[str, bool] or None
        Manual overrides for specific columns.  These take precedence over the
        automatic heuristic and are the recommended path for short histories
        (under ~1000 bars), where the heuristic has too little data per block
        to be reliable — declare known bounded oscillators (RSI, %B, Stoch)
        as ``True`` explicitly.
    skip_cols : set[str] or None
        Additional column names to skip, merged with the built-in ``_SKIP_COLS``
        set (datetime, OHLCV raw columns, regime labels).
    support_overlap_threshold : float
        Minimum support overlap for the scale-free heuristic.  A column passes
        when every temporal block's robust support (``[q05, q95]``) overlaps
        the whole-series support by at least this fraction (intersection /
        union).  Higher values are stricter; default 0.5.
    max_scale_free_blocks : int
        Upper bound on the number of temporal blocks used by the scale-free
        heuristic.  The actual block count adapts to the series length so that
        each block keeps at least ``min_block_size`` bars; see
        ``_is_scale_free``.  Default 4.
    min_block_size : int
        Minimum bars per block for the adaptive block count.  Short series use
        fewer blocks (down to 2) so quantile estimates stay reliable.
        Default 250.
    scale_free_drift_threshold : float, optional
        **Deprecated** and ignored.  The scale-free heuristic no longer uses a
        windowed drift ratio (see issue #136).  Passing this argument emits a
        ``DeprecationWarning`` and has no effect.
    """

    def __init__(
        self,
        max_categorical_classes: int = 20,
        scale_free_overrides: Optional[dict[str, bool]] = None,
        skip_cols: Optional[set[str]] = None,
        support_overlap_threshold: float = 0.5,
        max_scale_free_blocks: int = 4,
        min_block_size: int = 250,
        scale_free_drift_threshold: Optional[float] = None,
    ):
        if scale_free_drift_threshold is not None:
            warnings.warn(
                "scale_free_drift_threshold is deprecated and ignored; the "
                "scale-free heuristic now uses a support-overlap test "
                "(support_overlap_threshold). See issue #136.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.max_categorical_classes = max_categorical_classes
        self.scale_free_overrides = scale_free_overrides or {}
        self.skip_cols = (skip_cols or set()) | _SKIP_COLS
        self.support_overlap_threshold = support_overlap_threshold
        self.max_scale_free_blocks = max_scale_free_blocks
        self.min_block_size = min_block_size

    def fit(self, df: pd.DataFrame) -> dict[str, ColumnClassification]:
        """Classify every column in ``df`` and return the results dict.

        Columns listed in ``skip_cols`` and all-NaN columns are silently
        ignored.  The returned dict maps column name → ColumnClassification
        and is passed directly to FeatureGenerator and EventGenerator.

        Parameters
        ----------
        df : pd.DataFrame
            The raw KPI table.  Must include a datetime column (skipped
            automatically via ``_SKIP_COLS``).

        Returns
        -------
        dict[str, ColumnClassification]
            One entry per non-skipped, non-empty column.
        """
        results: dict[str, ColumnClassification] = {}
        for col in df.columns:
            if col in self.skip_cols:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if series.nunique() <= 1:
                continue
            results[col] = self._classify(col, series)
        return results

    def _classify(self, col: str, series: pd.Series) -> ColumnClassification:
        """Determine the type of a single non-null series.

        Decision tree:
        1. Non-numeric dtype → CATEGORICAL (strings, objects).
        2. Exactly 2 distinct values → BINARY (boolean flags, 0/1 indicators).
        3. Otherwise → CONTINUOUS; run scale-free detection and apply overrides.

        Parameters
        ----------
        col : str
            Column name (used to look up any user override).
        series : pd.Series
            Already dropna'd values for the column.

        Returns
        -------
        ColumnClassification
        """
        n_distinct = int(series.nunique())

        if not pd.api.types.is_numeric_dtype(series):
            return ColumnClassification(
                col_name=col,
                col_type=ColumnType.CATEGORICAL,
                n_distinct=n_distinct,
            )

        if n_distinct == 2:
            return ColumnClassification(
                col_name=col,
                col_type=ColumnType.BINARY,
                n_distinct=n_distinct,
            )

        is_sf = self._is_scale_free(series)
        override = self.scale_free_overrides.get(col)
        return ColumnClassification(
            col_name=col,
            col_type=ColumnType.CONTINUOUS,
            n_distinct=n_distinct,
            is_scale_free=is_sf,
            scale_free_override=override,
        )

    def _is_scale_free(self, series: pd.Series) -> bool:
        """Return True only if the series support is stationary (period-invariant).

        The series is cut into ``k`` consecutive temporal blocks.  For each
        block the robust support (``[q05, q95]``) is compared against the
        whole-series support, and the *worst* block's overlap decides::

            g_lo, g_hi = q05(series), q95(series)          # global support
            for each block:
                b_lo, b_hi = q05(block), q95(block)
                overlap = intersection([b_lo,b_hi], [g_lo,g_hi])
                          / union([b_lo,b_hi], [g_lo,g_hi])
            scale_free = min(block overlaps) >= support_overlap_threshold

        A series is scale-free when *every* block revisits the full value
        range (``min overlap >= threshold``, default 0.5).

        Adaptive block count
        --------------------
        ``k`` adapts to the series length so each block keeps enough samples
        for a stable quantile estimate::

            k = max(2, min(max_scale_free_blocks, n // min_block_size))

        With the defaults (``max_scale_free_blocks=4``, ``min_block_size=250``)
        a 1-year daily series (~250–365 bars) uses ``k=2`` while a multi-year
        series uses ``k=4``.  More blocks give finer temporal resolution (they
        catch a series that drifts away and returns — a round-trip path that a
        single two-way split would miss) but need more data per block; the
        floor prevents fragmenting short series into noisy blocks.

        Why block-vs-global overlap
        ---------------------------
        A bounded, stationary oscillator (RSI, Stochastic, %B) keeps visiting
        the same support throughout the sample, so every block's support nearly
        equals the global one (overlap ≈ 0.8) — and, crucially, this holds for
        any ``k`` and any indicator period (RSI14 and RSI25 score the same).
        A trending or random-walk series (price, EMA) has each block covering
        only a slice of the global range, so at least one block overlaps it
        poorly (overlap ≈ 0).  Measuring against the *fixed* global support
        keeps the metric stable as ``k`` changes, so one threshold works
        across the whole adaptive range.

        This fixes three earlier defects:

        * The original mean-drift test produced a false negative on RSI/%B
          because their *mean* rises with an uptrend even though their domain
          is fixed (issue #133).
        * The four-window quantile-drift test (issue #133's fix) was not
          period-invariant: a slower oscillator (RSI25) drags the regime
          through its windows longer and drifted just over the threshold
          (issue #136).
        * The two-way split (issue #136's first fix) was blind to round-trip
          paths and degraded on long series where a single split is coarse.

        Short series
        ------------
        Below ~1000 bars the heuristic falls back to ``k=2`` and is inherently
        noisy (too few bars per block, and a weakly-trending 1-year price is
        hard to tell from bounded).  For such histories prefer explicit
        ``scale_free_overrides`` for known oscillators.

        Minimum data requirements
        -------------------------
        * ``n >= 48`` total non-null observations.

        Columns with a degenerate global support (``g_hi == g_lo``) always
        return False because they carry no level information.

        Parameters
        ----------
        series : pd.Series
            Numeric series (NaNs already dropped by the caller).

        Returns
        -------
        bool
            True if every block's support overlaps the global support by at
            least the threshold; False otherwise or when there is insufficient
            data.
        """
        series = series.dropna()
        n = len(series)
        if n < 48:
            return False

        k = max(2, min(self.max_scale_free_blocks, n // self.min_block_size))

        arr = series.to_numpy()
        g_lo, g_hi = np.quantile(arr, 0.05), np.quantile(arr, 0.95)
        if g_hi - g_lo == 0:
            return False

        block = n // k
        min_overlap = 1.0
        for i in range(k):
            seg = arr[i * block:] if i == k - 1 else arr[i * block: (i + 1) * block]
            b_lo, b_hi = np.quantile(seg, 0.05), np.quantile(seg, 0.95)
            intersection = max(0.0, min(b_hi, g_hi) - max(b_lo, g_lo))
            union = max(b_hi, g_hi) - min(b_lo, g_lo)
            overlap = intersection / union if union > 0 else 0.0
            if overlap < min_overlap:
                min_overlap = overlap

        return bool(min_overlap >= self.support_overlap_threshold)
