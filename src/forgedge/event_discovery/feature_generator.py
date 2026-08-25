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
from .transform_layer import DELTA_LAGS

# Raw OHLC bases eligible for cross-column, cross-time ("lag-cross") pairing
# (issue #161) — e.g. "close[t] > low[t-1]".  Deliberately excludes volume
# (different units/semantics) and same-base cross-time comparisons (already
# covered by the native "return" family, ``close_ret_N`` = close[t] vs
# close[t-N]).  Restricted to columns actually present in the frame at
# generation time, and read straight from ``df`` rather than the classifier's
# ``parsed`` dict — like the "volume vs its own MA" block below, raw OHLC
# columns other than "close" never reach ``parsed`` (``TypeClassifier``
# skips them; see ``classifier._SKIP_COLS``), so a ``parsed``-only lookup
# would silently generate nothing for high/low/open.
_OHLC_BASES: tuple[str, ...] = ("open", "high", "low", "close")

# Lag-cross features are restricted to the identity transform (see
# DerivedFeature.transforms): they are already a fixed-lag comparison by
# construction, so TransformLayer stacking pctrank/zscore/delta on top would
# multiply feature_count x transform_count x threshold_count for combinations
# that mostly restate the same comparison in a more contrived form.  Identity
# alone still yields threshold *and* crossing events — enough to express
# "close[t] > low[t-1]", the shape this issue is about.
_LAG_CROSS_TRANSFORMS: frozenset[str] = frozenset({"identity"})

# Indicator × OHLC-base cross-time comparisons (issue #165) — e.g.
# "ma_12[t] > low[t-3]", completing #161 (OHLC×OHLC cross-time) and #162
# (same-time indicator pairing). Deliberately narrower than #161's own
# constant:
#
# * Default lags are (1, 3), not the global DELTA_LAGS = (1, 3, 6, 12) used
#   everywhere else in this module — a dedicated, smaller default so this
#   family's growth stays proportionate even though (unlike #161's lag-cross,
#   which is opt-out-only by being unconditional) it's on by default. See
#   DiscoveryConfig.indicator_lag_cross_lags for how a caller overrides it;
#   the same default is documented in kpi_builder's default_enricher.yaml as
#   a cross-referenced default value, not read from there at runtime (that
#   would require a new kpi_builder -> event_discovery coupling and a PyYAML
#   runtime dependency, neither of which exist anywhere else in the
#   pipeline — FeatureGenerator only ever sees the resulting DataFrame, never
#   the config that produced it).
# * Indicator side restricted to price-scale families (SMA/EMA/WMA/HMA) —
#   the units match a raw OHLC base, so a ratio against one is dimensionally
#   sound. RSI/vol/ret/mdd/BB are excluded (already bounded/normalised;
#   comparing e.g. RSI directly to a raw price is exactly the ATR-vs-NATR
#   mistake #162 avoided).
_INDICATOR_LAG_CROSS_DEFAULT_LAGS: tuple[int, ...] = (1, 3)
_PRICE_SCALE_FAMILIES: frozenset[str] = frozenset({"ema", "sma", "wma", "hma"})


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
    # Volume MA expressed as volume_sma_N or volume_ema_N — must precede the
    # generic base+indicator pattern below, whose base alternation already
    # includes "volume" and would otherwise match first and shadow this rule
    # (issue: dead pattern, never reached — base/indicator are captured here
    # explicitly so they resolve to "volume"/"sma" instead of the generic
    # rule's "volume"/"sma" reusing the same group positions incorrectly).
    (r'^(volume)_(sma|ema)_(\d+)$', "{1}", "volume_ma"),
    # MACD signal / histogram lines — must precede the plain MACD-line pattern
    # below, whose trailing "$" would otherwise never match these anyway
    # (more numeric groups), but ordering them first keeps the more specific
    # pattern visually adjacent to what it refines (issue #162).
    (r'^(close|high|low|open)_macd_(\d+)_(\d+)_signal_(\d+)$', "macd_signal", "macd_signal"),
    (r'^(close|high|low|open)_macd_(\d+)_(\d+)_hist_(\d+)$', "macd_hist", "macd_hist"),
    # MACD line itself (issue #162): two numeric params (fast, slow), distinct
    # from every other indicator here which has exactly one.
    (r'^(close|high|low|open)_macd_(\d+)_(\d+)$', "macd", "macd_line"),
    # Standard MA / oscillator (also scale-free ratio-style indicators that
    # benefit from same-family multi-period pairing: max_drawdown, ATR, NATR)
    (r'^(close|high|low|open|volume)_(ema|sma|rsi|dema|tema|wma|hma|mdd|atr|natr)_(\d+)$', "{1}", "{1}"),
    # Rolling min/max on a price column
    (r'^(close|high|low)_(min|max)_(\d+)$', "{1}", "rolling_{1}"),
    # Volatility / return series. "volume" is a valid base here (issue #162):
    # kpi_builder.returns()/multiple_returns() are generic over `on`, and a
    # volume-vs-price divergence combo needs a volume_ret_N to pair against
    # close_ret_N — not part of the default config (return is close-only by
    # default), so this only activates once a caller adds "volume" to the
    # return indicator's `columns`.
    (r'^(close|high|low)_vol_(\d+)$', "vol", "volatility"),
    (r'^(close|high|low|volume)_ret_(\d+)$', "ret", "return"),
    # candle_features() geometry columns (issue #162) — plain names, no
    # base/period suffix, so each needs its own literal alternative rather
    # than the generic {base}_{indicator}_{param} shape.  "color" (+1/-1/0)
    # is deliberately excluded: it's closer to categorical than a continuous
    # geometry measure, and the issue flags it for discussion rather than
    # prescribing inclusion.
    (r'^(body|upper_wick|lower_wick|close_pos|range_pct|gap)$', "{0}", "candle_geometry"),
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
    For ``ratio_lag``/``spread_pct_lag`` features this contains
    ``{"cross_lag": int}`` — the lag applied to the second operand (issue
    #161).  Empty for all other operations (ratio, spread_pct, position are
    parameter-free formulas).
    """
    transforms: Optional[frozenset[str]] = None
    """Restricts which TransformLayer transforms apply to this feature.

    ``None`` (default) means "all four" (identity/pctrank/zscore/delta), the
    behaviour every feature had before this field existed.  Set to a subset
    (e.g. ``frozenset({"identity"})``) to opt a feature out of further
    temporal transforms — used by the lag-cross family (issue #161): those
    features are already a fixed-lag comparison by construction, so stacking
    another rolling/delta transform on top multiplies TransformLayer's
    output combinatorially (feature_count × transform_count × threshold_count)
    for signal that mostly restates the same, already-lagged comparison in a
    more contrived form.  Restricting to identity keeps the direct threshold
    and crossing events — which is what expresses "close[t] > low[t-1]" —
    without that multiplication.
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
        indicator_lag_cross_lags: Optional[tuple[int, ...]] = None,
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
        indicator_lag_cross_lags : tuple[int, ...] or None
            Lag set for the indicator × OHLC-base cross-time family (issue
            #165, e.g. ``close_sma_12[t] > low[t-3]``).  ``None`` (default)
            uses ``_INDICATOR_LAG_CROSS_DEFAULT_LAGS`` (``(1, 3)``); pass an
            empty tuple to disable this family entirely.  See
            ``DiscoveryConfig.indicator_lag_cross_lags``.

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
        extended = self._generate_arity2(df, parsed, extended, meta)

        # Arity 2 — cross-column, cross-time OHLC pairs (issue #161)
        extended = self._generate_lag_cross(df, extended, meta)

        # Arity 2 — same-time pairings that never got wired up (issue #162)
        extended = self._generate_macd_pairs(df, parsed, extended, meta)
        extended = self._generate_price_volume_pairs(df, parsed, extended, meta)
        extended = self._generate_candle_geometry_pairs(df, parsed, extended, meta)

        # Arity 2 — indicator vs lagged OHLC-base cross-time pairs (issue #165)
        lags = (
            _INDICATOR_LAG_CROSS_DEFAULT_LAGS
            if indicator_lag_cross_lags is None
            else indicator_lag_cross_lags
        )
        extended = self._generate_indicator_lag_cross(df, parsed, extended, meta, lags)

        # Arity 3 — Bollinger & rolling-range triples
        extended = self._generate_arity3(df, parsed, extended, meta)

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
    ) -> pd.DataFrame:
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

        new_cols: dict[str, pd.Series] = {}

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
                    if new_col not in extended.columns and new_col not in new_cols:
                        series = _safe_ratio(df[col_a], df[col_b])
                        new_cols[new_col] = series
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
                    if dn_col not in extended.columns and dn_col not in new_cols:
                        dn_series, dn_std = _safe_diff_norm(df[col_a], df[col_b])
                        new_cols[dn_col] = dn_series
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
                if new_col in extended.columns or new_col in new_cols:
                    continue
                series = _safe_spread_pct(df[price_col], df[ma_col])
                new_cols[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="spread_pct",
                    source_cols=[price_col, ma_col],
                )

        # Volume vs volume MA. Reads raw "volume" straight from df rather than
        # from `parsed` — TypeClassifier never classifies raw OHLCV columns
        # (see classifier._SKIP_COLS), so "volume" never appears there and a
        # `parsed`-only lookup left this block permanently dead.
        vol_ma = [col for col, pf in parsed.items() if pf.family == "volume_ma"]
        if vol_ma and "volume" in df.columns:
            for vm_col in vol_ma:
                pf_vm = parsed[vm_col]
                param = pf_vm.params[0] if pf_vm.params else 0
                new_col = f"ratio_volume_{pf_vm.indicator}{param:02d}"
                if new_col in extended.columns or new_col in new_cols:
                    continue
                series = _safe_ratio(df["volume"], df[vm_col])
                new_cols[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="ratio",
                    source_cols=["volume", vm_col],
                )

        return _flush_new_columns(extended, new_cols)

    # ------------------------------------------------------------------
    # Arity 2 — cross-column, cross-time ("lag-cross") pairs
    # ------------------------------------------------------------------

    def _generate_lag_cross(
        self,
        df: pd.DataFrame,
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> pd.DataFrame:
        """Generate cross-column, cross-time OHLC comparisons (issue #161).

        Neither the same-family arity-2 pairing above (cross-column,
        same-time only) nor the Delta transform (same-column, cross-time
        only) can express a condition like *"today's close above yesterday's
        low"* — it is simultaneously cross-column and cross-time.  This
        method fills that specific gap by pairing every ordered combination
        of the raw OHLC bases actually present in ``df``
        (``open``/``high``/``low``/``close``) against a lagged copy of a
        different base, for each lag in ``DELTA_LAGS``.

        For bases ``a != b`` and each ``lag`` in ``DELTA_LAGS``, two scale-free
        combos are produced, mirroring the existing same-time ``ratio``/
        ``spread`` operations:

        * ``ratio_{a}_{b}_lag{lag}`` = ``a[t] / b[t-lag]``
        * ``spread_{a}_{b}_lag{lag}`` = ``(a[t] - b[t-lag]) / b[t-lag]``

        The lag is recorded as ``params={"cross_lag": lag}`` — a distinct key
        from the Delta transform's own ``transform_params["lag"]``, so the two
        don't collide when ``EventGenerator`` merges feature- and
        transform-level params onto ``EventComponent.transform_params``.

        Scope (deliberately bounded, see issue #161's "suggested scope")
        ------------------------------------------------------------------
        Restricted to the four raw OHLC bases — not indicator columns (e.g.
        moving averages) crossed against a lagged OHLC base.  A same-column
        cross-time ratio (``close[t]`` vs ``close[t-lag]``) is not generated
        either — that shape already exists natively as the ``return`` family
        (``close_ret_N``).  Widening beyond raw OHLC bases would multiply
        combinatorially with the number of indicator periods present (a
        default KPI config alone has ~24 MA-family columns), which is exactly
        the unbounded "feature surface" growth the issue explicitly warns
        against; it is left as a possible, separately-scoped follow-up.

        Parameters
        ----------
        df : pd.DataFrame
            Original KPI table (read-only) — raw OHLC columns are read
            directly from here (see ``_OHLC_BASES``), not from the
            classifier's ``parsed`` dict, which never contains them.
        extended : pd.DataFrame
            Target DataFrame — new columns are appended here.
        meta : dict[str, DerivedFeature]
            Target metadata dict — new entries are added here.
        """
        bases = [b for b in _OHLC_BASES if b in df.columns]
        if len(bases) < 2:
            return extended

        new_cols: dict[str, pd.Series] = {}

        for base_a in bases:
            series_a = df[base_a]
            for base_b in bases:
                if base_a == base_b:
                    continue
                for lag in DELTA_LAGS:
                    lagged_b = df[base_b].shift(lag)

                    ratio_col = f"ratio_{base_a}_{base_b}_lag{lag}"
                    if ratio_col not in extended.columns and ratio_col not in new_cols:
                        series = _safe_ratio(series_a, lagged_b)
                        new_cols[ratio_col] = series
                        meta[ratio_col] = DerivedFeature(
                            col=ratio_col,
                            series=series,
                            is_scale_free=True,
                            arity=2,
                            operation="ratio_lag",
                            source_cols=[base_a, base_b],
                            params={"cross_lag": lag},
                            transforms=_LAG_CROSS_TRANSFORMS,
                        )

                    spread_col = f"spread_{base_a}_{base_b}_lag{lag}"
                    if spread_col not in extended.columns and spread_col not in new_cols:
                        series = _safe_spread_pct(series_a, lagged_b)
                        new_cols[spread_col] = series
                        meta[spread_col] = DerivedFeature(
                            col=spread_col,
                            series=series,
                            is_scale_free=True,
                            arity=2,
                            operation="spread_pct_lag",
                            source_cols=[base_a, base_b],
                            params={"cross_lag": lag},
                            transforms=_LAG_CROSS_TRANSFORMS,
                        )

        return _flush_new_columns(extended, new_cols)

    # ------------------------------------------------------------------
    # Arity 2 — indicator vs lagged OHLC-base cross-time pairs (issue #165)
    # ------------------------------------------------------------------

    def _generate_indicator_lag_cross(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
        lags: tuple[int, ...],
    ) -> pd.DataFrame:
        """Pair a price-scale indicator against a lagged OHLC base (issue #165).

        Completes the investigation started in #161 (OHLC-base cross-time,
        e.g. ``close[t] > low[t-1]``) and #162 (same-time indicator pairing):
        neither reaches a condition like ``close_sma_12[t] > low[t-3]`` — an
        indicator compared against a *different* OHLC base *N bars earlier* —
        because both were deliberately scoped to avoid it (see each's own
        "restricted to raw OHLC bases" / "not indicator vs OHLC-base"
        docstring note) to keep their own growth bounded.

        Column name: ``ratio_{indicator_col}_{ohlc_base}_lag{lag}`` =
        ``indicator[t] / ohlc_base[t-lag]``. Reuses #161's exact
        ``cross_lag`` mechanism (``params={"cross_lag": lag}``, source_cols
        ``[indicator_col, ohlc_base]``) — the replay/SQL/formula dispatch in
        ``models.py``/``event_generator.py`` already handles this shape
        generically (a ``ratio_`` of two source columns with a ``cross_lag``
        param), so no changes were needed outside this module.

        Deliberately narrower than #161's own OHLC×OHLC lag-cross, matching
        the scope this issue was filed to pin down precisely (a generic
        "every indicator × every OHLC base × every lag" cross would run to
        ~1,000 new features before any transform fan-out — see the issue):

        * **Indicator side restricted to price-scale families**
          (``_PRICE_SCALE_FAMILIES`` — SMA/EMA/WMA/HMA): the only families
          whose units match a raw OHLC base, so a ratio against one is
          dimensionally sound. RSI/vol/ret/mdd/BB stay excluded — already
          bounded/normalised, so comparing them to a raw price the way #162
          avoided pairing raw ATR with anything but NATR for the same reason.
        * **``ratio_`` only** — no ``spread_``/``diffnorm_`` variant.
        * **Identity transform only** (``_LAG_CROSS_TRANSFORMS``) — mirrors
          #161's own restriction: the comparison is already fixed by
          construction, so stacking pctrank/zscore/delta on top mostly
          restates it at a much higher combinatorial cost.
        * **Lag set is a dedicated, smaller default** — see
          ``_INDICATOR_LAG_CROSS_DEFAULT_LAGS`` and
          ``DiscoveryConfig.indicator_lag_cross_lags`` for why this doesn't
          reuse the global ``DELTA_LAGS``.

        On by default (not opt-in): with the above restrictions applied, the
        growth is small enough that gating this behind a flag would quietly
        exclude the pattern class from exploratory search — a user would
        need to already suspect this shape of rule exists to go looking for
        it (this is the issue's own explicit reasoning for shipping it
        unconditionally, echoing why #161's lag-cross is unconditional too).

        Parameters
        ----------
        df : pd.DataFrame
            Original KPI table (read-only) — OHLC bases are read directly
            from here (see ``_OHLC_BASES``), same as ``_generate_lag_cross``.
        parsed : dict[str, ParsedFeature]
            Pre-parsed feature metadata for continuous columns.
        extended : pd.DataFrame
            Target DataFrame — new columns are appended here.
        meta : dict[str, DerivedFeature]
            Target metadata dict — new entries are added here.
        lags : tuple[int, ...]
            Lag set to pair against, e.g. ``(1, 3)``. Empty disables this
            family entirely.
        """
        if not lags:
            return extended

        indicator_cols = [
            col for col, pf in parsed.items() if pf.family in _PRICE_SCALE_FAMILIES
        ]
        if not indicator_cols:
            return extended
        ohlc_bases = [b for b in _OHLC_BASES if b in df.columns]
        if not ohlc_bases:
            return extended
        indicator_cols.sort()  # deterministic regardless of df.columns order

        new_cols: dict[str, pd.Series] = {}

        for ind_col in indicator_cols:
            series_ind = df[ind_col]
            for base in ohlc_bases:
                for lag in lags:
                    ratio_col = f"ratio_{ind_col}_{base}_lag{lag}"
                    if ratio_col in extended.columns or ratio_col in new_cols:
                        continue
                    lagged_base = df[base].shift(lag)
                    series = _safe_ratio(series_ind, lagged_base)
                    new_cols[ratio_col] = series
                    meta[ratio_col] = DerivedFeature(
                        col=ratio_col,
                        series=series,
                        is_scale_free=True,
                        arity=2,
                        operation="ratio_lag",
                        source_cols=[ind_col, base],
                        params={"cross_lag": lag},
                        transforms=_LAG_CROSS_TRANSFORMS,
                    )

        return _flush_new_columns(extended, new_cols)

    # ------------------------------------------------------------------
    # Arity 2 — same-timestamp pairings that never got wired up (issue #162)
    # ------------------------------------------------------------------

    def _generate_macd_pairs(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> pd.DataFrame:
        """Pair each MACD line against its own signal line (issue #162).

        MACD/signal is arguably the single most textbook MACD signal, but its
        column names (``close_macd_12_26``, ``close_macd_12_26_signal_09``)
        carry two or three numeric parameters, one more than every other
        pattern in ``_PATTERNS`` handles — before this method, ``parse_feature``
        returned None for both, so neither ever reached ``_generate_arity2``'s
        per-family grouping.

        Unlike the fast/slow same-family loop above (single-parameter,
        assumes any two columns in a family may be paired), a MACD line must
        be paired with *its own* signal line specifically — matched by the
        shared ``(base, fast, slow)`` key, not by sorting on one parameter.
        That mismatch is why this is a dedicated method rather than an
        extension of the generic grouping.

        MACD is the first *signed* same-family pairing in the generator (RSI,
        EMA, ATR, … are all non-negative), so the ratio can swing wildly or
        flip sign near a line's zero-crossing. ``_safe_ratio`` already
        degrades that to NaN (never inf/crash); ``diffnorm`` — which
        normalises by the std of the difference rather than dividing by an
        operand — is unaffected and is arguably the more robust of the two
        for this family, but both are emitted for consistency with every
        other same-family pairing.

        Column names: ``ratio_{base}_macd{fast:02d}_{slow:02d}_signal``,
        ``diffnorm_{base}_macd{fast:02d}_{slow:02d}_signal``.

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
        macd_line: dict[tuple, str] = {}
        macd_signal: dict[tuple, str] = {}
        for col, pf in parsed.items():
            if len(pf.params) < 2:
                continue
            key = (pf.base, pf.params[0], pf.params[1])
            if pf.family == "macd_line":
                macd_line[key] = col
            elif pf.family == "macd_signal":
                macd_signal[key] = col

        new_cols: dict[str, pd.Series] = {}

        for key in set(macd_line) & set(macd_signal):
            base, fast, slow = key
            line_col = macd_line[key]
            sig_col = macd_signal[key]
            tag = f"{base}_macd{fast:02d}_{slow:02d}_signal"

            ratio_col = f"ratio_{tag}"
            if ratio_col not in extended.columns and ratio_col not in new_cols:
                series = _safe_ratio(df[line_col], df[sig_col])
                new_cols[ratio_col] = series
                meta[ratio_col] = DerivedFeature(
                    col=ratio_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="ratio",
                    source_cols=[line_col, sig_col],
                )

            dn_col = f"diffnorm_{tag}"
            if dn_col not in extended.columns and dn_col not in new_cols:
                dn_series, dn_std = _safe_diff_norm(df[line_col], df[sig_col])
                new_cols[dn_col] = dn_series
                meta[dn_col] = DerivedFeature(
                    col=dn_col,
                    series=dn_series,
                    is_scale_free=True,
                    arity=2,
                    operation="diff_norm",
                    source_cols=[line_col, sig_col],
                    params={"diffnorm_std": dn_std},
                )

        return _flush_new_columns(extended, new_cols)

    def _generate_price_volume_pairs(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> pd.DataFrame:
        """Pair price % change against volume % change, same period (issue #162).

        Neither existing arity-2 branch pairs a price-family column against a
        volume-family one: "price vs its own MA" only pairs ``close`` against
        its own MA columns, and "volume vs volume MA" only pairs ``volume``
        against its own MA columns — there was no branch anywhere combining
        the two, so divergence signals (e.g. "new price high on falling
        volume") could never be constructed.

        A *raw* price-vs-volume combination isn't dimensionally sound as a
        ratio/spread: ``volume`` is an absolute, drifting level (not
        scale-free — same reason raw ``close`` isn't), and dividing it by a
        small, sign-crossing return fraction is numerically unstable near
        the return's zero-crossings. Pairing two already-comparable %-change
        quantities — ``close_ret_N`` against ``volume_ret_N`` at the *same*
        lookback ``N`` — is the dimensionally sound version of the same idea
        and is what this method builds.

        Requires the KPI table to carry a volume return column, which is
        **not** part of the default ``kpi_builder`` config (the ``return``
        indicator is close-only by default) — this method is a no-op until a
        caller adds ``"volume"`` to that indicator's ``columns``.

        Column names: ``ratio_close_retNN_volume_retNN``,
        ``diffnorm_close_retNN_volume_retNN``.

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
        price_ret: dict[int, str] = {}
        vol_ret: dict[int, str] = {}
        for col, pf in parsed.items():
            if pf.family != "return" or not pf.params:
                continue
            period = pf.params[0]
            if pf.base == "volume":
                vol_ret[period] = col
            elif pf.base == "close":
                price_ret[period] = col

        new_cols: dict[str, pd.Series] = {}

        for period in set(price_ret) & set(vol_ret):
            p_col = price_ret[period]
            v_col = vol_ret[period]
            tag = f"close_ret{period:02d}_volume_ret{period:02d}"

            ratio_col = f"ratio_{tag}"
            if ratio_col not in extended.columns and ratio_col not in new_cols:
                series = _safe_ratio(df[p_col], df[v_col])
                new_cols[ratio_col] = series
                meta[ratio_col] = DerivedFeature(
                    col=ratio_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="ratio",
                    source_cols=[p_col, v_col],
                )

            dn_col = f"diffnorm_{tag}"
            if dn_col not in extended.columns and dn_col not in new_cols:
                dn_series, dn_std = _safe_diff_norm(df[p_col], df[v_col])
                new_cols[dn_col] = dn_series
                meta[dn_col] = DerivedFeature(
                    col=dn_col,
                    series=dn_series,
                    is_scale_free=True,
                    arity=2,
                    operation="diff_norm",
                    source_cols=[p_col, v_col],
                    params={"diffnorm_std": dn_std},
                )

        return _flush_new_columns(extended, new_cols)

    def _generate_candle_geometry_pairs(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> pd.DataFrame:
        """Pair candle_features() geometry columns among themselves and vs NATR.

        ``kpi_builder.candle_features()`` produces ``body``/``upper_wick``/
        ``lower_wick``/``close_pos``/``range_pct``/``gap`` — all scale-free by
        construction, but bare-named (no ``{base}_{indicator}_{param}`` period
        suffix), so ``parse_feature`` never recognised them and they could
        only be used individually (arity-1 pass-through), never combined —
        e.g. "wick asymmetry vs body" or "gap size relative to the bar's own
        range" couldn't be constructed automatically.

        Every pair among the (up to 6) geometry columns present is generated
        — dedicated rather than routed through the generic same-family loop,
        because that loop's naming template assumes one shared, *numeric*
        ``base``/period across a group (``ratio_{base}_{ind}{param}_...``);
        these columns have neither, and forcing them through it would print a
        meaningless zero-padded period (``ratio_body_raw00_...``).

        Also paired against each ``close_natr_{N}`` present ("ATR-style
        volatility columns", per the issue) — **not** raw ``close_atr_{N}``,
        which is in absolute price units and would silently reintroduce a
        price-level dependency into an otherwise scale-free ratio; NATR
        (``atr / close``, a dimensionless fraction — see
        ``kpi_builder.indicators.multiple_atr``) is the dimensionally
        comparable volatility measure. ATR/NATR are disabled by default in
        ``kpi_builder``, so this half is a no-op unless a caller enables them.

        ``body`` and ``gap`` are signed (unlike ``upper_wick``/``lower_wick``/
        ``close_pos``/``range_pct``), so — as with the MACD pairing above —
        some ratios can be numerically unstable near a zero-crossing;
        ``_safe_ratio`` degrades that to NaN rather than inf/crash, same as
        every other pairing in this class.

        ``color`` (+1/-1/0) is deliberately excluded — the issue flags it as
        closer to categorical than a continuous geometry measure and asks for
        that to be a separate discussion, not decided here.

        Column names: ``ratio_{a}_{b}``, ``diffnorm_{a}_{b}`` (geometry pairs);
        ``ratio_{a}_close_natrNN``, ``diffnorm_{a}_close_natrNN`` (vs NATR).

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
        geometry = [
            col for col, pf in parsed.items() if pf.family == "candle_geometry"
        ]
        geometry.sort()  # deterministic pairing order regardless of df.columns order
        natr_cols = [col for col, pf in parsed.items() if pf.family == "natr"]

        new_cols: dict[str, pd.Series] = {}

        def _emit(col_a: str, col_b: str, tag: str) -> None:
            ratio_col = f"ratio_{tag}"
            if ratio_col not in extended.columns and ratio_col not in new_cols:
                series = _safe_ratio(df[col_a], df[col_b])
                new_cols[ratio_col] = series
                meta[ratio_col] = DerivedFeature(
                    col=ratio_col,
                    series=series,
                    is_scale_free=True,
                    arity=2,
                    operation="ratio",
                    source_cols=[col_a, col_b],
                )
            dn_col = f"diffnorm_{tag}"
            if dn_col not in extended.columns and dn_col not in new_cols:
                dn_series, dn_std = _safe_diff_norm(df[col_a], df[col_b])
                new_cols[dn_col] = dn_series
                meta[dn_col] = DerivedFeature(
                    col=dn_col,
                    series=dn_series,
                    is_scale_free=True,
                    arity=2,
                    operation="diff_norm",
                    source_cols=[col_a, col_b],
                    params={"diffnorm_std": dn_std},
                )

        for i, col_a in enumerate(geometry):
            for col_b in geometry[i + 1:]:
                _emit(col_a, col_b, f"{col_a}_{col_b}")
            for natr_col in natr_cols:
                _emit(col_a, natr_col, f"{col_a}_{natr_col}")

        return _flush_new_columns(extended, new_cols)

    # ------------------------------------------------------------------
    # Arity 3
    # ------------------------------------------------------------------

    def _generate_arity3(
        self,
        df: pd.DataFrame,
        parsed: dict[str, ParsedFeature],
        extended: pd.DataFrame,
        meta: dict[str, DerivedFeature],
    ) -> pd.DataFrame:
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

        new_cols: dict[str, pd.Series] = {}

        for key in set(bb_lower) & set(bb_upper):
            base, param = key
            lower_col = bb_lower[key]
            upper_col = bb_upper[key]
            new_col = f"bb_pct_b_{base}_{param:02d}"
            if new_col not in extended.columns and new_col not in new_cols:
                base_cols = [col for col, pf in parsed.items()
                             if pf.family == "price" and pf.base == base]
                base_col = base_cols[0] if base_cols else (base if base in df.columns else None)
                if base_col is None:
                    continue
                series = _safe_position(df[base_col], df[lower_col], df[upper_col])
                new_cols[new_col] = series
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
            if new_col not in extended.columns and new_col not in new_cols:
                series = _safe_position(df[price_col], df[min_col], df[max_col])
                new_cols[new_col] = series
                meta[new_col] = DerivedFeature(
                    col=new_col,
                    series=series,
                    is_scale_free=True,
                    arity=3,
                    operation="position",
                    source_cols=[price_col, min_col, max_col],
                )

        return _flush_new_columns(extended, new_cols)


# ---------------------------------------------------------------------------
# DataFrame accumulation (issue #219)
# ---------------------------------------------------------------------------

def _flush_new_columns(
    extended: pd.DataFrame, new_cols: dict[str, pd.Series]
) -> pd.DataFrame:
    """Merge ``new_cols`` into ``extended`` in one ``pd.concat``.

    Each ``_generate_*`` method below used to insert every new column with
    ``extended[col] = series`` inside nested loops — hundreds of single-column
    inserts on the same block manager, which pandas reallocates/fragments on
    every insert (see the ``PerformanceWarning`` this was raising, and issue
    #219). Callers now accumulate a method's new columns in a local dict and
    call this once at the end of the method instead.
    """
    if not new_cols:
        return extended
    return pd.concat([extended, pd.DataFrame(new_cols, index=extended.index)], axis=1)


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
