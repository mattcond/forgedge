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
    col: str
    base: str       # "close", "volume", "high", "low"
    indicator: str  # "ema", "sma", "rsi", "bb_lower", etc.
    params: list[int]
    family: str     # semantic grouping key

    @property
    def full_key(self) -> str:
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
    result = tmpl
    for i, g in enumerate(groups):
        result = result.replace(f"{{{i}}}", g or "")
    return result


# ---------------------------------------------------------------------------
# Derived feature metadata
# ---------------------------------------------------------------------------

@dataclass
class DerivedFeature:
    col: str
    series: pd.Series
    is_scale_free: bool
    arity: int                        # 1, 2, or 3
    operation: str                    # "identity", "ratio", "spread_pct", "diff_norm", "position"
    source_cols: list[str] = field(default_factory=list)


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
        """Return (extended_df, derived_meta) where extended_df contains all
        native continuous columns plus newly generated derived columns."""

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
        # Group by (base, indicator-family) to find same-family pairs
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
            if close_col is None:
                break
            base, param = key
            lower_col = bb_lower[key]
            upper_col = bb_upper[key]
            new_col = f"bb_pct_b_{base}_{param:02d}"
            if new_col not in extended.columns:
                series = _safe_position(df[close_col], df[lower_col], df[upper_col])
                extended[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=3,
                    operation="position",
                    source_cols=[close_col, lower_col, upper_col],
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
    result = a / b
    return result.replace([float("inf"), float("-inf")], pd.NA)


def _safe_spread_pct(a: pd.Series, b: pd.Series) -> pd.Series:
    result = (a - b) / b
    return result.replace([float("inf"), float("-inf")], pd.NA)


def _safe_position(value: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    denom = upper - lower
    result = (value - lower) / denom
    return result.replace([float("inf"), float("-inf")], pd.NA)
