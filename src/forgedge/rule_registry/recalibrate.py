"""Threshold recalibration for the cross-ticker backtest (FORGE Modulo 4).

The cross-ticker backtest (spec Section 8) keeps a rule's logical structure
identical and only moves its *absolute* thresholds onto the target ticker's
distribution.  The thresholds were originally set by Event Discovery as
percentiles of the source distribution; to test the same rule elsewhere each
threshold is moved to the **same percentile** of the target distribution::

    rule on ADA:  close_rsi_25 < 30.5   where 30.5 = quantile(rsi25_ADA, 0.10)
    test on SOL:  close_rsi_25 < 27.8   where 27.8 = quantile(rsi25_SOL, 0.10)

The percentile is recovered empirically from the source data — ``q = P(x <=
threshold)`` on the transformed series — and re-applied to the target data.
This is fully general: relative transforms (rolling pctrank / zscore) are
self-normalising, so their recovered percentile maps back to almost the same
value on the target — the invariance the spec describes — while an ``identity``
threshold genuinely shifts.  Binary / categorical components carry category
values, not distributional thresholds, and are left untouched.

The result is a fresh :class:`EventCandidate` whose ``apply`` reconstructs the
recalibrated boolean signal on the target frame bit-for-bit through Event
Discovery's own replay path.
"""
from __future__ import annotations

import copy
from typing import Optional

import pandas as pd

from ..event_discovery.models import EventCandidate, EventComponent, build_feature_series


def _transformed_series(comp: EventComponent, df: pd.DataFrame) -> Optional[pd.Series]:
    """Rebuild a component's transformed series on ``df`` (feature + transform).

    This is steps 1–2 of Event Discovery's replay (feature construction and the
    temporal transform) *without* the threshold — the continuous series the
    threshold is a percentile of.  Returns ``None`` for components that have no
    distributional threshold (binary / categorical).
    """
    if comp.transform in ("binary_native", "categorical_onehot"):
        return None

    series = build_feature_series(comp, df)
    t = comp.transform
    if t == "rolling_pctrank":
        w = comp.transform_params["window"]
        series = series.rolling(w, min_periods=max(2, w // 2)).rank(pct=True)
    elif t == "rolling_zscore":
        w = comp.transform_params["window"]
        roll = series.rolling(w, min_periods=max(2, w // 2))
        mu = roll.mean()
        sd = roll.std(ddof=1)
        series = (series - mu) / sd.replace(0, float("nan"))
    elif t == "delta":
        series = series.diff(comp.transform_params["lag"])
    # identity: no temporal transform
    return series


def _recalibrated_threshold(
    comp: EventComponent,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Optional[float]:
    """Return the threshold moved to the target's matching percentile, or None.

    ``None`` means "keep the original threshold" — the component is not
    distributional, or one of the distributions is degenerate / empty.
    """
    src = _transformed_series(comp, source_df)
    if src is None:
        return None
    src = src.dropna()
    if src.empty:
        return None

    # Empirical CDF position of the original threshold on the source data.
    q = float((src <= comp.threshold).mean())
    q = min(max(q, 0.0), 1.0)

    tgt = _transformed_series(comp, target_df)
    if tgt is None:
        return None
    tgt = tgt.dropna()
    if tgt.empty:
        return None

    return float(tgt.quantile(q))


def _component_expression(comp: EventComponent) -> str:
    """Rebuild a single component's human-readable expression from its fields."""
    col = comp.transformed_col or comp.source_feature
    thr = comp.threshold
    thr_str = f"{thr:.6g}"
    if comp.event_type == "crossing":
        arrow = "↓" if comp.direction == "below" else "↑"
        return f"{col} crosses {arrow} {thr_str}"
    op = "<" if comp.direction == "below" else ">"
    return f"{col} {op} {thr_str}"


def recalibrate_candidate(
    candidate: EventCandidate,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> EventCandidate:
    """Return a copy of ``candidate`` with thresholds recalibrated on ``target_df``.

    Every distributional component's threshold is moved to the percentile of the
    target distribution that matches its percentile on the source distribution.
    The returned candidate's ``expression`` and per-component ``expression`` are
    rebuilt to show the adapted thresholds; its cached ``event_series`` is
    cleared so ``apply`` recomputes on the target frame.
    """
    adapted = copy.deepcopy(candidate)
    adapted.event_series = None

    for comp in adapted.components:
        new_thr = _recalibrated_threshold(comp, source_df, target_df)
        if new_thr is None:
            continue
        comp.threshold = new_thr
        comp.expression = _component_expression(comp)

    adapted.expression = " AND ".join(
        _component_expression(c) for c in adapted.components
    )
    return adapted
