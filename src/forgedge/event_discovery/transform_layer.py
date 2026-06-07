"""Step 2 — Transform Layer.

Applies four temporal transformations to every feature in the extended catalog:
  Identity    — raw value (only for scale-free features)
  Pctrank     — rolling percentile rank  (windows: 48, 96, 168)
  Z-score     — rolling standardisation  (windows: 48, 96, 168)
  Delta       — lag difference            (lags:    1, 3, 6, 12)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PCTRANK_WINDOWS: list[int] = [48, 96, 168]
ZSCORE_WINDOWS: list[int] = [48, 96, 168]
DELTA_LAGS: list[int] = [1, 3, 6, 12]


@dataclass
class TransformedSeries:
    col: str
    series: pd.Series
    transform: str          # "identity" | "rolling_pctrank" | "rolling_zscore" | "delta"
    transform_params: dict
    source_feature: str
    is_zscore: bool = False


class TransformLayer:
    """Applies all transforms to a set of features and returns the results."""

    def transform_all(
        self,
        df: pd.DataFrame,
        feature_meta: dict,   # col → DerivedFeature (from FeatureGenerator)
    ) -> list[TransformedSeries]:
        results: list[TransformedSeries] = []
        for col, meta in feature_meta.items():
            if col not in df.columns:
                continue
            series = df[col]
            results.extend(self.transform_one(series, col, meta.is_scale_free))
        return results

    def transform_one(
        self,
        series: pd.Series,
        col: str,
        is_scale_free: bool,
    ) -> list[TransformedSeries]:
        out: list[TransformedSeries] = []

        # Identity — only for scale-free series
        if is_scale_free:
            out.append(TransformedSeries(
                col=col,
                series=series,
                transform="identity",
                transform_params={},
                source_feature=col,
                is_zscore=False,
            ))

        # Pctrank
        for w in PCTRANK_WINDOWS:
            t_col = f"pr_{col}_{w}"
            t_series = _rolling_pctrank(series, w)
            out.append(TransformedSeries(
                col=t_col,
                series=t_series,
                transform="rolling_pctrank",
                transform_params={"window": w},
                source_feature=col,
                is_zscore=False,
            ))

        # Z-score
        for w in ZSCORE_WINDOWS:
            t_col = f"zs_{col}_{w}"
            t_series = _rolling_zscore(series, w)
            out.append(TransformedSeries(
                col=t_col,
                series=t_series,
                transform="rolling_zscore",
                transform_params={"window": w},
                source_feature=col,
                is_zscore=True,
            ))

        # Delta
        for lag in DELTA_LAGS:
            t_col = f"delta_{col}_{lag}"
            t_series = series.diff(lag)
            out.append(TransformedSeries(
                col=t_col,
                series=t_series,
                transform="delta",
                transform_params={"lag": lag},
                source_feature=col,
                is_zscore=False,
            ))

        return out


# ---------------------------------------------------------------------------
# Rolling computation helpers
# ---------------------------------------------------------------------------

def _rolling_pctrank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank in [0, 1].  Uses min_periods = window // 2."""
    min_p = max(2, window // 2)
    return series.rolling(window, min_periods=min_p).rank(pct=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score.  Uses min_periods = window // 2."""
    min_p = max(2, window // 2)
    roll = series.rolling(window, min_periods=min_p)
    mean = roll.mean()
    std = roll.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mean) / std
    return z.where(std > 0)
