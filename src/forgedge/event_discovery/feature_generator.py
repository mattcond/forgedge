"""Step 1 — Feature Generation (arity 1, 2, 3).

Produces normalised derived features from native KPI columns, removing
the price-level problem before the Transform Layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .models import ColumnClassification, ColumnType


# ---------------------------------------------------------------------------
# Feature name parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedFeature:
    """Structured representation of a recognised KPI column name.

    Column names follow the convention ``<base>_<indicator>_<param>``
    (e.g. ``close_ema_25``, ``close_bb_lower_20``).  ``parse_feature``
    extracts the semantic parts so FeatureGenerator can group columns by
    family for arity-2 and arity-3 feature construction.

    Attributes
    ----------
    col : str
        Original column name as it appears in the DataFrame.
    base : str
        Underlying price/volume series (``close``, ``high``, ``low``,
        ``open``, ``volume``).
    indicator : str
        Indicator type string, resolved from the regex match groups
        (e.g. ``ema``, ``sma``, ``bb_lower``, ``raw``).
    params : list[int]
        Numeric parameters extracted from the column name (period lengths).
    family : str
        Semantic grouping key used to find pairable columns.  Examples:
        ``ema``, ``sma``, ``bollinger``, ``rolling_min``, ``volume_ma``.
    """

    col: str
    base: str       # "close", "volume", "high", "low"
    indicator: str  # "ema", "sma", "rsi", "bb_lower", etc.
    params: list[int]
    family: str     # semantic grouping key

    @property
    def full_key(self) -> str:
        """Composite key ``<base>_<indicator>`` for quick identity checks."""
        return f"{self.base}_{self.indicator}"


_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, indicator_template, family)
    # Bollinger bands — must come before generic {base}_{type}_{param}
    (r'^(close|high|low|open)_bb_(lower|upper|width|mid)_(\d+)$', "bb_{1}", "bollinger"),
    # Standard MA / oscillator
    (r'^(close|high|low|open|volume)_(ema|sma|rsi|dema|tema|wma|hma)_(\d+)$', "{1}", "{1}"),
    # Rolling min/max on a price column
    (r'^(close|high|low)_(min|max)_(\d+)$', "{1}", "rolling_{1}"),
    # Volume MA expressed as volume_sma_N or volume_ema_N
    (r'^volume_(sma|ema)_(\d+)$', "vol_{0}", "volume_ma"),
    # Volatility / return series
    (r'^(close|high|low)_vol_(\d+)$', "vol", "volatility"),
    (r'^(close|high|low)_ret_(\d+)$', "ret", "return"),
    # Raw OHLCV
    (r'^(close|high|low|open)$', "raw", "price"),
    (r'^volume$', "raw", "volume"),
]


def parse_feature(col: str) -> Optional[ParsedFeature]:
    """Try to parse ``col`` against the known naming patterns.

    Iterates through ``_PATTERNS`` in order.  The first matching pattern
    wins; earlier patterns take priority (Bollinger bands must be matched
    before the generic ``{base}_{indicator}_{param}`` pattern to avoid
    mis-classifying ``close_bb_lower_20`` as ``indicator=bb``,
    ``family=bb``).

    Parameters
    ----------
    col : str
        Column name to parse.

    Returns
    -------
    ParsedFeature or None
        Parsed structure if the column matches a known pattern, None if it
        does not (e.g. a custom or derived column that was already present
        in the DataFrame).
    """
    for pattern, ind_tmpl, fam_tmpl in _PATTERNS:
        m = re.match(pattern, col)
        if m is None:
            continue
        groups = m.groups()
        base = groups[0] if groups else col
        # Resolve template references like {0}, {1}
        indicator = _resolve(ind_tmpl, groups)
        family = _resolve(fam_tmpl, groups)
        params = [int(g) for g in groups if g and g.isdigit()]
        return ParsedFeature(col=col, base=base, indicator=indicator, params=params, family=family)
    return None


def _resolve(tmpl: str, groups: tuple) -> str:
    """Substitute positional placeholders ``{0}``, ``{1}`` … with regex groups.

    Used internally to build indicator and family strings from regex match
    groups without a full format-string approach (which would require named
    groups in every pattern).

    Parameters
    ----------
    tmpl : str
        Template string containing zero or more ``{N}`` placeholders.
    groups : tuple
        Regex match groups (may include None for optional groups).

    Returns
    -------
    str
        Template with placeholders replaced by the corresponding group value.
    """
    result = tmpl
    for i, g in enumerate(groups):
        result = result.replace(f"{{{i}}}", g or "")
    return result


# ---------------------------------------------------------------------------
# Derived feature metadata
# ---------------------------------------------------------------------------

@dataclass
class DerivedFeature:
    """Metadata record for one generated (or pass-through) feature.

    Created by FeatureGenerator and consumed by TransformLayer.  Each entry
    in the ``derived_meta`` dict corresponds to one column in
    ``extended_df``.

    Attributes
    ----------
    col : str
        Name of the column in ``extended_df``.
    series : pd.Series
        The actual numeric values (reference into ``extended_df``).
    is_scale_free : bool
        Whether this feature has a stable level over time.  All arity-2
        and arity-3 derived features are always scale-free by construction.
        Arity-1 features inherit the result of the classifier.
    arity : int
        Number of input columns used to produce this feature (1, 2, or 3).
    operation : str
        Construction method: ``"identity"``, ``"ratio"``, ``"spread_pct"``,
        or ``"position"``.
    source_cols : list[str]
        Names of the input columns (length == arity).
    """

    col: str
    series: pd.Series
    is_scale_free: bool
    arity: int                        # 1, 2, or 3
    operation: str                    # "identity", "ratio", "spread_pct", "diff_norm", "position"
    source_cols: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    """Operation-specific scalar parameters needed for out-of-sample replay.

    For ``diff_norm`` features this contains ``{"diffnorm_std": float}`` —
    the in-sample standard deviation used to normalise the difference series.
    Empty for all other operations (ratio, spread_pct, position are
    parameter-free formulas).
    """


# ---------------------------------------------------------------------------
# Feature Generator
# ---------------------------------------------------------------------------

class FeatureGenerator:
    """Generates normalised derived features from the native continuous columns.

    Arity 1  — scale-free native features passed through unchanged.
    Arity 2  — ratio / spread_pct between same-family pairs.
    Arity 3  — relative position for Bollinger/rolling-range triples.
    """

    def generate(
        self,
        df: pd.DataFrame,
        classifications: dict[str, ColumnClassification],
    ) -> tuple[pd.DataFrame, dict[str, DerivedFeature]]:
        """Build the extended feature catalog and return it alongside metadata.

        Processing order
        ----------------
        1. **Arity 1**: scale-free continuous columns are added to ``meta``
           as identity features (no new DataFrame column is needed — the
           original column is reused).
        2. **Arity 2**: ``_generate_arity2`` creates ratio and spread_pct
           features by pairing same-family columns (EMA pairs, price vs MA,
           volume vs volume-MA).
        3. **Arity 3**: ``_generate_arity3`` creates position features by
           combining a price column with its Bollinger or rolling-range bands.

        Parameters
        ----------
        df : pd.DataFrame
            Original KPI table (must not be modified in place).
        classifications : dict[str, ColumnClassification]
            Output of ``TypeClassifier.fit(df)``.

        Returns
        -------
        extended_df : pd.DataFrame
            Copy of ``df`` with additional derived columns appended.
        derived_meta : dict[str, DerivedFeature]
            One entry per feature that will be passed to the TransformLayer,
            including both arity-1 pass-throughs and newly computed columns.
        """
        continuous = {
            col: cls
            for col, cls in classifications.items()
            if cls.col_type == ColumnType.CONTINUOUS and col in df.columns
        }

        extended = df.copy()
        meta: dict[str, DerivedFeature] = {}

        # Arity 1 — pass scale-free natives through (no new column needed)
        for col, cls in continuous.items():
            if cls.effective_scale_free:
                meta[col] = DerivedFeature(
                    col=col,
                    series=df[col],
                    is_scale_free=True,
                    arity=1,
                    operation="identity",
                    source_cols=[col],
                )

        parsed = {col: parse_feature(col) for col in continuous}
        parsed = {col: pf for col, pf in parsed.items() if pf is not None}

        # Arity 2 — same-family pairs
        self._generate_arity2(df, parsed, extended, meta)

        # Arity 3 — Bollinger & rolling-range triples
        self._generate_arity3(df, parsed, extended, meta)

        return extended, meta

    # ------------------------------------------------------------------
    # Arity 2
    # ------------------------------------------------------------------

    def _generate_arity2(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> None:
        """Generate all arity-2 derived features and append them to ``extended``/``meta``.

        Three sub-cases are handled:

        **Same-indicator pairs** (e.g. EMA-9 / EMA-25, RSI-14 / RSI-25):
        Columns sharing the same ``(base, family)`` key are sorted by period
        parameter (smaller = faster).  For every (fast, slow) pair, a ratio
        ``fast / slow`` is computed.  This captures crossover dynamics in a
        scale-free form.  Column name: ``ratio_{base}_{ind}{p_fast}_{ind}{p_slow}``.

        **Price vs its own moving average** (e.g. close / close_ema_25):
        The raw close price divided by (or spread relative to) each of its
        MA columns gives a mean-reversion signal.  Computed as
        ``(close - MA) / MA``.  Column name: ``spread_{base}_{ind}{param}``.

        **Volume vs volume MA** (e.g. volume / volume_sma_25):
        Captures volume spikes in normalised form.
        Column name: ``ratio_volume_{ind}{param}``.

        All generated features are marked ``is_scale_free=True`` because the
        division/subtraction removes the price level by construction.

        Parameters
        ----------
        df : pd.DataFrame
            Original KPI table (read-only).
        parsed : dict[str, ParsedFeature]
            Pre-parsed feature metadata for continuous columns.
        extended : pd.DataFrame
            Target DataFrame — new columns are appended here.
        meta : dict[str, DerivedFeature]
            Target metadata dict — new entries are added here.
        """
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for col, pf in parsed.items():
            if pf.family in ("bollinger", "rolling_min", "rolling_max", "price", "volume"):
                continue  # handled by arity 3 or separately
            key = f"{pf.base}__{pf.family}"
            groups[key].append(col)

        for group_key, cols in groups.items():
            if len(cols) < 2:
                continue
            cols_sorted = sorted(cols, key=lambda c: parsed[c].params[0] if parsed[c].params else 0)
            # Generate ratio for all pairs within the group (fast / slow)
            for i, col_a in enumerate(cols_sorted):
                for col_b in cols_sorted[i + 1:]:
                    pf_a = parsed[col_a]
                    pf_b = parsed[col_b]
                    # col_a has smaller param → is "faster"
                    param_a = pf_a.params[0] if pf_a.params else 0
                    param_b = pf_b.params[0] if pf_b.params else 0
                    new_col = f"ratio_{pf_a.base}_{pf_a.indicator}{param_a:02d}_{pf_b.indicator}{param_b:02d}"
                    if new_col in extended.columns:
                        continue
                    series = _safe_ratio(df[col_a], df[col_b])
                    extended[new_col] = series
                    meta[new_col] = DerivedFeature(
                        col=new_col,
                        series=series,
                        is_scale_free=True,
                        arity=2,
                        operation="ratio",
                        source_cols=[col_a, col_b],
                    )
                    # diff_norm: (A - B) / std(A - B)
                    dn_col = f"diffnorm_{pf_a.base}_{pf_a.indicator}{param_a:02d}_{pf_b.indicator}{param_b:02d}"
                    if dn_col not in extended.columns:
                        dn_series, dn_std = _safe_diff_norm(df[col_a], df[col_b])
                        extended[dn_col] = dn_series
                        meta[dn_col] = DerivedFeature(
                            col=dn_col,
                            series=dn_series,
                            is_scale_free=True,
                            arity=2,
                            operation="diff_norm",
                            source_cols=[col_a, col_b],
                            params={"diffnorm_std": dn_std},
                        )

        # Price vs its own MA (close vs close_ema_N / close_sma_N)
        price_cols = [col for col, pf in parsed.items() if pf.family == "price"]
        ma_cols = [
            col for col, pf in parsed.items()
            if pf.family in ("ema", "sma", "wma", "hma") and pf.base == "close"
        ]
        for price_col in price_cols:
            pf_price = parsed[price_col]
            if pf_price.indicator != "raw":
                continue
            for ma_col in ma_cols:
                pf_ma = parsed[ma_col]
                param = pf_ma.params[0] if pf_ma.params else 0
                new_col = f"spread_{pf_ma.base}_{pf_ma.indicator}{param:02d}"
                if new_col in extended.columns:
                    continue
                series = _safe_spread_pct(df[price_col], df[ma_col])
                extended[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="spread_pct",
                    source_cols=[price_col, ma_col],
                )

        # Volume vs volume MA
        vol_raw = [col for col, pf in parsed.items() if pf.family == "volume"]
        vol_ma = [col for col, pf in parsed.items() if pf.family == "volume_ma"]
        for v_col in vol_raw:
            for vm_col in vol_ma:
                pf_vm = parsed[vm_col]
                param = pf_vm.params[0] if pf_vm.params else 0
                new_col = f"ratio_volume_{pf_vm.indicator}{param:02d}"
                if new_col in extended.columns:
                    continue
                series = _safe_ratio(df[v_col], df[vm_col])
                extended[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="ratio",
                    source_cols=[v_col, vm_col],
                )

    # ------------------------------------------------------------------
    # Arity 3
    # ------------------------------------------------------------------

    def _generate_arity3(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> None:
        """Generate all arity-3 derived features (relative position within a range).

        Two sub-cases:

        **Bollinger %B** (``bb_pct_b_{base}_{param}``):
        For each (base, period) pair that has both a ``bb_lower`` and
        ``bb_upper`` column, the close price position within the band is
        computed as ``(close - lower) / (upper - lower)``.  Values near 0
        indicate the price is at the lower band; near 1 at the upper band.
        A value outside [0, 1] means the price has broken out of the bands.

        **Rolling range position** (``pos_{base}_range{param}``):
        For each (base, period) pair that has both a ``close_min_N`` and
        ``close_max_N`` column, the price position within the rolling N-bar
        range is computed identically.  Values near 0/1 signal new lows/highs
        within the lookback window.

        Both features are scale-free by construction (division removes
        absolute price level).  Division by zero (when upper == lower,
        i.e. a flat market) is handled by ``_safe_position`` which replaces
        inf with pd.NA.

        Parameters
        ----------
        df : pd.DataFrame
            Original KPI table (read-only).
        parsed : dict[str, ParsedFeature]
            Pre-parsed feature metadata for continuous columns.
        extended : pd.DataFrame
            Target DataFrame — new columns are appended here.
        meta : dict[str, DerivedFeature]
            Target metadata dict — new entries are added here.
        """
        from collections import defaultdict

        # Bollinger bands: group by (base, param)
        bb_lower: dict[tuple, str] = {}
        bb_upper: dict[tuple, str] = {}
        for col, pf in parsed.items():
            if pf.family != "bollinger":
                continue
            param = pf.params[0] if pf.params else 0
            key = (pf.base, param)
            if "lower" in pf.indicator:
                bb_lower[key] = col
            elif "upper" in pf.indicator:
                bb_upper[key] = col

        close_cols = [col for col, pf in parsed.items() if pf.family == "price" and pf.base == "close"]
        close_col = close_cols[0] if close_cols else None

        for key in set(bb_lower) & set(bb_upper):
            base, param = key
            lower_col = bb_lower[key]
            upper_col = bb_upper[key]
            new_col = f"bb_pct_b_{base}_{param:02d}"
            if new_col not in extended.columns:
                base_cols = [col for col, pf in parsed.items()
                             if pf.family == "price" and pf.base == base]
                base_col = base_cols[0] if base_cols else (base if base in df.columns else None)
                if base_col is None:
                    continue
                series = _safe_position(df[base_col], df[lower_col], df[upper_col])
                extended[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=3,
                    operation="position",
                    source_cols=[base_col, lower_col, upper_col],
                )

        # Rolling range: group by (base, param)
        roll_min: dict[tuple, str] = {}
        roll_max: dict[tuple, str] = {}
        for col, pf in parsed.items():
            if pf.family not in ("rolling_min", "rolling_max"):
                continue
            param = pf.params[0] if pf.params else 0
            key = (pf.base, param)
            if pf.family == "rolling_min":
                roll_min[key] = col
            else:
                roll_max[key] = col

        for key in set(roll_min) & set(roll_max):
            base, param = key
            min_col = roll_min[key]
            max_col = roll_max[key]
            # Value = the base price column (usually "close")
            price_col = base if base in df.columns else close_col
            if price_col is None or price_col not in df.columns:
                continue
            new_col = f"pos_{base}_range{param:02d}"
            if new_col not in extended.columns:
                series = _safe_position(df[price_col], df[min_col], df[max_col])
                extended[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=3,
                    operation="position",
                    source_cols=[price_col, min_col, max_col],
                )


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------

def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """Compute ``a / b``, replacing ±inf with NaN.

    Division by zero yields ±inf in pandas; replacing those values with NaN
    (not pd.NA, which upcasts float64 to object) avoids propagating infinities
    into downstream rolling operations.
    """
    result = a / b
    return result.replace([float("inf"), float("-inf")], float("nan"))


def _safe_spread_pct(a: pd.Series, b: pd.Series) -> pd.Series:
    """Compute ``(a - b) / b``, replacing ±inf with NaN.

    Used for the price-vs-MA spread where ``b`` could momentarily be zero
    (unlikely in practice for a moving average of a positive price, but
    guarded defensively).
    """
    result = (a - b) / b
    return result.replace([float("inf"), float("-inf")], float("nan"))


def _safe_diff_norm(a: pd.Series, b: pd.Series) -> tuple[pd.Series, float]:
    """Compute ``(a - b) / std(a - b)`` over the full history.

    Normalises the difference series by its in-sample standard deviation,
    producing a z-score of the spread.  Returns a NaN series and std=0.0
    when the difference is constant (zero standard deviation).

    Returns
    -------
    tuple[pd.Series, float]
        ``(normalised_series, std_used)`` — the std is stored in
        ``DerivedFeature.params["diffnorm_std"]`` so that the event can be
        replicated on out-of-sample data without re-computing the normaliser.
    """
    diff = a - b
    std = float(diff.std())
    if std == 0 or pd.isna(std):
        return pd.Series(float("nan"), index=a.index, dtype=float), 0.0
    return diff / std, std


def _safe_position(value: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Compute the relative position of ``value`` within [``lower``, ``upper``].

    Formula: ``(value - lower) / (upper - lower)``.

    When ``upper == lower`` (flat market, zero-width band) the denominator is
    zero, producing ±inf.  These are replaced with NaN (not pd.NA, which
    upcasts float64 to object) so that downstream rolling operations are
    unaffected.
    """
    denom = upper - lower
    result = (value - lower) / denom
    return result.replace([float("inf"), float("-inf")], float("nan"))
