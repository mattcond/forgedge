"""forgedge.deployment.kpi_recipe — minimal KPI recipe per rule (#296).

``export_rules()`` persists every exported ``EventCandidate`` via
:mod:`pickle`. That reproduces the event exactly, but it is fragile across
library versions, opaque (it never says *which* KPI columns a monitoring job
must compute to keep that one rule alive) and unreadable outside Python.

A :class:`KpiRecipe` is the complementary, human-readable manifest: the
smallest ``kpi_builder`` configuration — ``build_features`` config,
``candle_features``/``pattern_features`` flags, ``lag_features`` requests —
that recomputes exactly the native KPI columns the event references, from
raw OHLC(V) candles alone. It is plain JSON (:meth:`KpiRecipe.to_dict` /
:meth:`KpiRecipe.from_dict`), and :func:`verify_kpi_recipe` checks that it
round-trips: the event replayed on the rebuilt table must activate on
exactly the same bars as on the reference table.

Why this lives here and not on ``EventCandidate``
-------------------------------------------------
``event_discovery`` deliberately does not depend on ``kpi_builder`` (see the
comments in ``event_discovery/feature_generator.py``), while resolving a
recipe necessarily calls ``kpi_builder.build_features()``. ``deployment`` is
the downstream orchestration layer that may legitimately depend on both.

Resolution is by construction, never by naming-convention regex: every
indicator/column/period combination in ``kpi_config`` is built once on a tiny
synthetic frame, and the produced column names are indexed back to the entry
that produced them (:func:`_column_index`, cached per config). Each candidate
then costs only dictionary lookups — O(1) ``build_features`` probes per
config, not per candidate.

Out of scope (reported in ``unresolved_columns``, never guessed)
----------------------------------------------------------------
- ``CustomEvent``/``manual_events`` formulas — a free ``DataFrame.eval``
  expression can reference arbitrary, even proprietary, columns.
- Columns no ``kpi_builder`` step produces (non-standard names, M0's
  ``regime`` columns, user-supplied indicators).

Usage::

    from forgedge.deployment import minimal_kpi_recipe, verify_kpi_recipe

    recipe = minimal_kpi_recipe(candidate)
    reduced = recipe.rebuild(candles)            # only the columns that matter
    check = verify_kpi_recipe(candidate, recipe, reference_kpi_table)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..kpi_builder import (
    DEFAULT_CONFIG,
    build_features,
    candle_features,
    lag_features,
    load_kpi_config,
    pattern_features,
)

__all__ = [
    "KpiRecipe",
    "KpiRecipeVerification",
    "minimal_kpi_recipe",
    "verify_kpi_recipe",
]

logger = logging.getLogger(__name__)

DEFAULT_BASE_COLS: Tuple[str, ...] = ("open_dt", "open", "high", "low", "close", "volume")

#: Indicators whose ``periods`` list is read in fixed-size *positional* groups
#: rather than as independent windows — MACD's ``(fast, slow, signal)``
#: triples (``indicators.multiple_macd``). Such groups must be kept intact and
#: in order: sorting/deduplicating ``[12, 26, 9]`` into ``[9, 12, 26]`` builds
#: a different MACD whose columns ``EventCandidate.apply()`` then can't find
#: (reproduced in the #295 prototype). Any future positional indicator must
#: be added here.
_POSITIONAL_PERIOD_GROUPS: Dict[str, int] = {"macd": 3}

#: Columns produced by ``candle_features()`` (``kpi_builder/candle.py``).
_CANDLE_FEATURE_COLS = frozenset({"body", "upper_wick", "lower_wick", "close_pos", "range_pct", "gap"})
#: Default column produced by ``pattern_features()``.
_PATTERN_COL = "candle_pattern"
#: Column produced by ``build_features(add_color=True)``.
_COLOR_COL = "color"

_LAG_RE = re.compile(r"^(?P<base>.+)_prev_(?P<lag>\d+)$")

_SYNTHETIC_ROWS = 8

# (indicator, source column, period group)
_IndexEntry = Tuple[str, str, Tuple[int, ...]]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KpiRecipe:
    """Minimal ``kpi_builder`` recipe recomputing the KPI columns of one event.

    Attributes
    ----------
    build_features_config : dict
        ``build_features()`` config restricted to the indicators the event
        needs: ``{indicator: {"enabled": True, "params": {"periods": [...],
        "columns": [...]}}}``. ``build_features`` takes the cross product of
        ``periods`` x ``columns`` per indicator, so when an event needs e.g.
        ``close_ema_12`` and ``low_ema_03`` the rebuilt table also carries
        ``close_ema_03``/``low_ema_12`` — a superset, never a missing column.
        MACD-like positional ``periods`` are kept as ordered triples.
    lag_features : list of dict
        ``[{"column": ..., "periods": [...]}]`` — ``lag_features()`` calls
        applied after every other step, in order (a lag of a lag comes after
        the lag it reads).
    candle_features : bool
        Whether ``candle_features()`` is needed (``body``, ``gap``, ...).
    pattern_features : bool
        Whether ``pattern_features()`` is needed (``candle_pattern``).
    color : bool
        Whether ``build_features``' built-in ``color`` column is needed.
    unresolved_columns : list of str
        Columns (or, for a ``CustomEvent``, the raw formula) no
        ``kpi_builder`` step is known to produce. Non-empty means the recipe
        is incomplete: :meth:`rebuild` cannot reproduce the event, and the
        pickled ``EventCandidate`` remains the only faithful export.
    base_columns : list of str
        Raw candle columns the recipe treats as given (never rebuilt);
        :meth:`rebuild` keeps those present in the candles and drops the rest.
    """

    build_features_config: dict = field(default_factory=dict)
    lag_features: List[dict] = field(default_factory=list)
    candle_features: bool = False
    pattern_features: bool = False
    color: bool = False
    unresolved_columns: List[str] = field(default_factory=list)
    base_columns: List[str] = field(default_factory=lambda: list(DEFAULT_BASE_COLS))

    @property
    def is_complete(self) -> bool:
        """``True`` when every referenced column was resolved."""
        return not self.unresolved_columns

    def to_dict(self) -> dict:
        """JSON-serialisable representation (plain lists/ints/bools)."""
        return {
            "build_features_config": {
                name: {
                    "enabled": True,
                    "params": {
                        "periods": [int(p) for p in conf["params"]["periods"]],
                        "columns": list(conf["params"]["columns"]),
                    },
                }
                for name, conf in self.build_features_config.items()
            },
            "lag_features": [
                {"column": req["column"], "periods": [int(p) for p in req["periods"]]}
                for req in self.lag_features
            ],
            "candle_features": bool(self.candle_features),
            "pattern_features": bool(self.pattern_features),
            "color": bool(self.color),
            "unresolved_columns": list(self.unresolved_columns),
            "base_columns": list(self.base_columns),
        }

    def to_json(self, **kwargs) -> str:
        """``json.dumps(self.to_dict(), **kwargs)``."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping) -> "KpiRecipe":
        """Inverse of :meth:`to_dict`; unknown keys (e.g. ``"verification"``
        in an ``export_rules`` file) are ignored."""
        return cls(
            build_features_config={
                name: {
                    "enabled": True,
                    "params": {
                        "periods": list(conf["params"]["periods"]),
                        "columns": list(conf["params"]["columns"]),
                    },
                }
                for name, conf in (data.get("build_features_config") or {}).items()
            },
            lag_features=[
                {"column": req["column"], "periods": list(req["periods"])}
                for req in (data.get("lag_features") or [])
            ],
            candle_features=bool(data.get("candle_features", False)),
            pattern_features=bool(data.get("pattern_features", False)),
            color=bool(data.get("color", False)),
            unresolved_columns=list(data.get("unresolved_columns") or []),
            base_columns=list(data.get("base_columns") or DEFAULT_BASE_COLS),
        )

    @classmethod
    def from_json(cls, text: str) -> "KpiRecipe":
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def rebuild(self, candles: pd.DataFrame, timestamp_col: str = "open_dt") -> pd.DataFrame:
        """Rebuild a reduced KPI Table from raw candles and this recipe.

        Only ``timestamp_col``, the recipe's ``base_columns`` and any raw
        column its indicators read are kept from ``candles`` — so passing a
        full KPI Table (e.g. ``ForgeResult.event_frame``) is safe: its
        pre-computed columns are discarded and recomputed, never reused.

        Parameters
        ----------
        candles : pd.DataFrame
            Raw OHLC(V) candles. ``timestamp_col`` must be a column, or the
            index must be a ``DatetimeIndex``.
        timestamp_col : str
            Timestamp column, also the output ordering column (default
            ``"open_dt"``).

        Returns
        -------
        pd.DataFrame
            Chronologically sorted, ``RangeIndex``-ed reduced KPI Table.

        Raises
        ------
        ValueError
            If the recipe has ``unresolved_columns`` — the result would
            silently lack columns the event needs.
        """
        if self.unresolved_columns:
            raise ValueError(
                "KpiRecipe.rebuild: recipe is incomplete, unresolved columns "
                f"{self.unresolved_columns} — fall back to the pickled EventCandidate."
            )
        frame = _with_timestamp_column(candles, timestamp_col)
        indicator_cols = {
            col
            for conf in self.build_features_config.values()
            for col in conf["params"]["columns"]
        }
        keep = [timestamp_col] + [
            c for c in frame.columns
            if c != timestamp_col and (c in self.base_columns or c in indicator_cols)
        ]
        reduced = build_features(
            frame[keep],
            config=self.build_features_config,
            timestamp_col=timestamp_col,
            output_timestamp_col=timestamp_col,
            add_color=self.color,
        )
        if self.candle_features:
            reduced = candle_features(reduced, order_on=timestamp_col)
        if self.pattern_features:
            reduced = pattern_features(reduced, order_on=timestamp_col)
        for req in self.lag_features:
            reduced = lag_features(reduced, req["column"], periods=req["periods"], order_on=timestamp_col)
        return reduced


@dataclass
class KpiRecipeVerification:
    """Outcome of :func:`verify_kpi_recipe`.

    Attributes
    ----------
    matches : bool
        ``True`` when the event activates on exactly the same bars on the
        rebuilt table as on the reference, over the evaluation window.
    n_compared_bars : int
        Bars inside the evaluation window.
    n_mismatched_bars : int
        Bars where the two activations differ.
    first_mismatch_at, last_mismatch_at : pd.Timestamp or None
        Timestamps of the first/last differing bar. Mismatches confined to
        the very start of the series (``last_mismatch_at`` early) are the
        signature of indicator warm-up (a recursive EMA started on a shorter
        history), not of a wrong recipe — see ``evaluation_mask``.
    error : str or None
        Why no comparison was possible (incomplete recipe, missing raw
        column, ...). ``matches`` is ``False`` whenever this is set.
    """

    matches: bool
    n_compared_bars: int = 0
    n_mismatched_bars: int = 0
    first_mismatch_at: Optional[pd.Timestamp] = None
    last_mismatch_at: Optional[pd.Timestamp] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-serialisable representation (timestamps as ISO strings)."""
        def _iso(ts):
            return None if ts is None else pd.Timestamp(ts).isoformat()

        return {
            "matches": bool(self.matches),
            "n_compared_bars": int(self.n_compared_bars),
            "n_mismatched_bars": int(self.n_mismatched_bars),
            "first_mismatch_at": _iso(self.first_mismatch_at),
            "last_mismatch_at": _iso(self.last_mismatch_at),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Column index (built once per kpi_config)
# ---------------------------------------------------------------------------

def _period_groups(name: str, periods: Sequence) -> List[Tuple[int, ...]]:
    size = _POSITIONAL_PERIOD_GROUPS.get(name)
    periods = [int(p) for p in periods]
    if size is None:
        return [(p,) for p in periods]
    if len(periods) % size != 0:
        raise ValueError(
            f"kpi_config['{name}']: 'periods' must be groups of {size} positional "
            f"values, got {len(periods)}: {periods}"
        )
    return [tuple(periods[i:i + size]) for i in range(0, len(periods), size)]


def _synthetic_candles(columns: Sequence[str]) -> pd.DataFrame:
    """Tiny, valid OHLCV frame: column *names* produced don't depend on length."""
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, _SYNTHETIC_ROWS))
    open_ = close + rng.normal(0.0, 0.5, _SYNTHETIC_ROWS)
    frame = pd.DataFrame(
        {
            "open_dt": pd.date_range("2000-01-01", periods=_SYNTHETIC_ROWS, freq="D"),
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": rng.uniform(1.0, 2.0, _SYNTHETIC_ROWS),
        }
    )
    for col in columns:
        if col not in frame.columns:
            frame[col] = close
    return frame


@lru_cache(maxsize=16)
def _column_index_cached(config_key: str) -> Dict[str, _IndexEntry]:
    config = json.loads(config_key)
    probe_cols = sorted(
        {c for conf in config.values() for c in (conf.get("params") or {}).get("columns", [])}
    )
    synthetic = _synthetic_candles(probe_cols)
    base = set(synthetic.columns)

    index: Dict[str, _IndexEntry] = {}
    for name, conf in config.items():
        params = conf.get("params") or {}
        for group in _period_groups(name, params.get("periods", [])):
            for col in params.get("columns", []):
                probe_cfg = {name: {"enabled": True, "params": {"periods": list(group), "columns": [col]}}}
                try:
                    produced = build_features(
                        synthetic, config=probe_cfg, timestamp_col="open_dt", add_color=False
                    )
                except Exception as exc:  # an indicator unusable on any frame
                    logger.warning("kpi_recipe: probe %s/%s/%s failed: %s", name, col, group, exc)
                    continue
                for out_col in produced.columns:
                    if out_col not in base:
                        index.setdefault(out_col, (name, col, group))
    return index


def _config_key(kpi_config: Mapping) -> str:
    """Canonical JSON of the indicator entries — ``enabled`` is ignored on
    purpose: a column an opt-in indicator produced must resolve even when the
    caller's config left it disabled by default."""
    from ..kpi_builder.builder import INDICATORS

    entries = {}
    for name, conf in kpi_config.items():
        if name not in INDICATORS or not isinstance(conf, Mapping):
            continue
        params = conf.get("params") or {}
        entries[name] = {
            "params": {
                "periods": [int(p) for p in params.get("periods", [])],
                "columns": [str(c) for c in params.get("columns", [])],
            }
        }
    return json.dumps(entries, sort_keys=True)


def _column_index(kpi_config: Mapping) -> Dict[str, _IndexEntry]:
    """``{produced column: (indicator, source column, period group)}``."""
    return _column_index_cached(_config_key(kpi_config))


def _resolve_kpi_config(kpi_config) -> Mapping:
    if kpi_config is None:
        return DEFAULT_CONFIG
    if isinstance(kpi_config, Mapping):
        return kpi_config
    if isinstance(kpi_config, (str, Path)):
        return load_kpi_config(kpi_config)
    raise TypeError(
        "kpi_config must be a dict, a YAML path or None; "
        f"got {type(kpi_config).__name__}"
    )


# ---------------------------------------------------------------------------
# Recipe extraction
# ---------------------------------------------------------------------------

def _referenced_columns(components) -> Tuple[List[str], List[str]]:
    """``(native columns, custom formulas)`` referenced by ``components``.

    Native columns: ``source_cols`` for arity-2+ features, ``source_feature``
    for arity-1/binary/categorical ones (exactly what ``EventCandidate.apply``
    reads). Nested ``and_composition`` components are walked recursively.
    """
    cols: List[str] = []
    formulas: List[str] = []
    for comp in components:
        nested = getattr(comp, "components", None) or []
        if nested:
            sub_cols, sub_formulas = _referenced_columns(nested)
            cols.extend(sub_cols)
            formulas.extend(sub_formulas)
            continue
        if comp.transform == "custom_formula":
            formulas.append(comp.transform_params.get("formula", comp.source_feature))
            continue
        cols.extend(comp.source_cols if comp.source_cols else [comp.source_feature])
    return cols, formulas


def minimal_kpi_recipe(
    candidate,
    kpi_config: Union[Mapping, str, Path, None] = None,
    base_cols: Sequence[str] = DEFAULT_BASE_COLS,
) -> KpiRecipe:
    """Minimal recipe recomputing only the KPI columns ``candidate`` references.

    Reads ``candidate.components[*].source_cols`` (``.source_feature`` for
    arity-1 components) and, for each native column not in ``base_cols``,
    looks up which ``kpi_builder`` step produces it: an entry of
    ``kpi_config`` (looked up in an index built once per config by probing
    ``build_features()`` itself), ``candle_features()``,
    ``pattern_features()``, the built-in ``color``, or a ``lag_features()``
    lag (``{col}_prev_{NN}``) of any of those — recursively.

    Parameters
    ----------
    candidate : EventCandidate
        Single or AND-composed event.
    kpi_config : dict, str, Path or None
        ``kpi_builder`` configuration the KPI Table was built with (``None``
        = ``DEFAULT_CONFIG``). Every entry is probed regardless of its
        ``enabled`` flag, so opt-in indicators (ATR, MACD, ...) resolve too.
    base_cols : sequence of str
        Raw columns taken as given (default timestamp + OHLCV).

    Returns
    -------
    KpiRecipe
        ``unresolved_columns`` lists what could not be traced back to a
        ``kpi_builder`` step (``CustomEvent`` formulas included verbatim).
    """
    config = _resolve_kpi_config(kpi_config)
    index = _column_index(config)
    base = set(base_cols)

    native, formulas = _referenced_columns(candidate.components)

    groups: Dict[str, Dict[str, list]] = {}       # indicator -> {"columns": [...], "groups": [...]}
    lags: Dict[str, set] = {}                     # column -> lag periods
    flags = {"candle": False, "pattern": False, "color": False}
    unresolved: List[str] = list(formulas)
    seen: set = set()

    def _resolve(col: str) -> bool:
        if col in base:
            return True
        if col in seen:
            return col not in unresolved
        seen.add(col)
        entry = index.get(col)
        if entry is not None:
            name, src, group = entry
            slot = groups.setdefault(name, {"columns": [], "groups": []})
            if src not in slot["columns"]:
                slot["columns"].append(src)
            if group not in slot["groups"]:
                slot["groups"].append(group)
            return True
        if col in _CANDLE_FEATURE_COLS:
            flags["candle"] = True
            return True
        if col == _PATTERN_COL:
            flags["pattern"] = True
            return True
        if col == _COLOR_COL:
            flags["color"] = True
            return True
        m = _LAG_RE.match(col)
        if m is not None:
            parent, lag = m.group("base"), int(m.group("lag"))
            # lag_features always writes `{col}_prev_{lag:02d}`; any other
            # spelling was not produced by it.
            if f"{parent}_prev_{lag:02d}" == col and _resolve(parent):
                lags.setdefault(parent, set()).add(lag)
                return True
        unresolved.append(col)
        return False

    for col in native:
        _resolve(col)

    build_cfg: Dict[str, dict] = {}
    for name, slot in groups.items():
        if name in _POSITIONAL_PERIOD_GROUPS:
            periods = [p for group in slot["groups"] for p in group]  # ordered, intact groups
        else:
            periods = sorted({p for group in slot["groups"] for p in group})
        build_cfg[name] = {
            "enabled": True,
            "params": {"periods": periods, "columns": sorted(slot["columns"])},
        }

    # A lag of a lag reads its parent lag column: apply shallower lags first.
    lag_reqs = sorted(
        ({"column": col, "periods": sorted(ps)} for col, ps in lags.items()),
        key=lambda req: (req["column"].count("_prev_"), req["column"]),
    )

    return KpiRecipe(
        build_features_config=build_cfg,
        lag_features=lag_reqs,
        candle_features=flags["candle"],
        pattern_features=flags["pattern"],
        color=flags["color"],
        unresolved_columns=list(dict.fromkeys(unresolved)),
        base_columns=list(base_cols),
    )


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def _with_timestamp_column(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col in frame.columns:
        return frame
    if isinstance(frame.index, pd.DatetimeIndex):
        out = frame.copy()
        out[timestamp_col] = frame.index
        return out.reset_index(drop=True)
    raise KeyError(
        f"timestamp column '{timestamp_col}' not found and the index is not a DatetimeIndex"
    )


def _timestamps(frame: pd.DataFrame, timestamp_col: str) -> pd.DatetimeIndex:
    if timestamp_col in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame[timestamp_col]))
    return pd.DatetimeIndex(frame.index)


def verify_kpi_recipe(
    candidate,
    recipe: KpiRecipe,
    candles: pd.DataFrame,
    evaluation_mask: Optional[pd.Series] = None,
    timestamp_col: str = "open_dt",
) -> KpiRecipeVerification:
    """Check that ``recipe`` reproduces ``candidate``'s activations exactly.

    ``candidate.apply(recipe.rebuild(candles))`` is compared bar by bar (by
    timestamp, NaN read as inactive — the pipeline's own AND semantics)
    against ``candidate.apply(candles)``.

    Parameters
    ----------
    candidate : EventCandidate
    recipe : KpiRecipe
        Usually ``minimal_kpi_recipe(candidate, ...)``.
    candles : pd.DataFrame
        The *reference* KPI Table — must already carry every column the
        event reads (e.g. ``ForgeResult.event_frame``) plus the raw OHLC(V)
        columns the recipe rebuilds from.
    evaluation_mask : pd.Series of bool, optional
        Bars to compare, aligned to ``candles.index``. Default: the whole
        series. Recursive indicators (EMA, ...) rebuilt from a shorter
        history than the reference's differ during a warm-up transient
        (measured: first 86 of 882 bars for EMA-12 on the ADA 1D fixture)
        and only then become bit-identical — when the reference was built
        on a longer history, pass a mask that excludes that start (e.g. the
        hold-out window). No default warm-up heuristic is applied (#296).
    timestamp_col : str
        Timestamp column (default ``"open_dt"``); a ``DatetimeIndex`` is
        accepted as a fallback.

    Returns
    -------
    KpiRecipeVerification
    """
    if recipe.unresolved_columns:
        return KpiRecipeVerification(
            matches=False, error=f"unresolved columns: {recipe.unresolved_columns}"
        )
    try:
        reference = candidate.apply(candles)
        rebuilt = recipe.rebuild(candles, timestamp_col=timestamp_col)
        replayed = candidate.apply(rebuilt)
    except (KeyError, ValueError) as exc:
        return KpiRecipeVerification(matches=False, error=f"{type(exc).__name__}: {exc}")

    ref = pd.Series(
        reference.fillna(0).astype(bool).astype(float).to_numpy(),
        index=_timestamps(candles, timestamp_col),
    )
    new = pd.Series(
        replayed.fillna(0).astype(bool).astype(float).to_numpy(),
        index=_timestamps(rebuilt, timestamp_col),
    )
    if not ref.index.is_unique or not new.index.is_unique:
        return KpiRecipeVerification(matches=False, error="duplicate timestamps")

    if evaluation_mask is not None:
        mask = pd.Series(evaluation_mask, index=candles.index).fillna(False).astype(bool).to_numpy()
        ref = ref[mask]

    # A bar missing from the rebuilt table reindexes to NaN, which never
    # compares equal — counted as a mismatch.
    diff = new.reindex(ref.index).to_numpy() != ref.to_numpy()
    mismatched = ref.index[diff]

    return KpiRecipeVerification(
        matches=len(mismatched) == 0,
        n_compared_bars=int(len(ref)),
        n_mismatched_bars=int(len(mismatched)),
        first_mismatch_at=mismatched[0] if len(mismatched) else None,
        last_mismatch_at=mismatched[-1] if len(mismatched) else None,
    )
