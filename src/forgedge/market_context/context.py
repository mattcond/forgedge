"""Market Context Module — Modulo 0 of the FORGE pipeline.

Runs once at the start of a session, before Event Discovery.  Enriches the
KPI Table with two columns — ``regime`` and ``regime_stable`` — that are then
available, immutable, to every downstream module.

Usage
-----
    from forgedge import MarketContext

    df = pd.read_parquet("kpi_table.parquet")
    mc = MarketContext(df)
    enriched = mc.run()          # df + 'regime' + 'regime_stable'
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import pandas as pd

from ..resolver import measure_bar_hours
from ..unset import is_set
from .ema_proxy import EMAProxyClassifier
from .hurst import derive_ema_windows
from .models import (
    REGIME_COL,
    REGIME_STABLE_COL,
    MarketContextConfig,
    RegimeClassifier,
)


def build_classifier(config: MarketContextConfig) -> RegimeClassifier:
    """Instantiate the RegimeClassifier selected by ``config.classifier``.

    In v1.0 only ``"ema_proxy"`` is supported.  This is the single place that
    maps the configuration string to a concrete implementation, so adding an
    HMM / KMeans / custom classifier in v2.0 only touches this function.

    Raises
    ------
    ValueError
        If ``config.classifier`` is not a known implementation.
    """
    name = config.classifier.lower().strip()
    if name == "ema_proxy":
        return EMAProxyClassifier(config.ema_proxy, config.labels)
    raise ValueError(
        f"Unknown classifier '{config.classifier}'. "
        f"v1.0 supports only 'ema_proxy'."
    )


class MarketContext:
    """FORGE Market Context Module (Modulo 0).

    Classifies every bar of the KPI Table by market regime and appends the
    ``regime`` and ``regime_stable`` columns.  Runs once; its output is the
    only write any module makes to the KPI Table and is never modified
    afterwards.

    Parameters
    ----------
    kpi_table : pd.DataFrame
        Raw KPI Table — OHLCV plus any precomputed technical indicators.
        Only the classifier's ``source_col`` (``close`` by default) is
        strictly required.
    config : MarketContextConfig, optional
        Configuration object.  Defaults to the v1.0 EMA-proxy settings.
    classifier : RegimeClassifier, optional
        An explicit classifier instance.  When provided it overrides
        ``config.classifier`` — this is the hook for plugging a custom
        implementation without registering it in :func:`build_classifier`.
    """

    def __init__(
        self,
        kpi_table: pd.DataFrame,
        config: Optional[MarketContextConfig] = None,
        classifier: Optional[RegimeClassifier] = None,
    ):
        self.df = kpi_table
        # The one chokepoint where M0's session-resolved fields become values.
        self.config = (config or MarketContextConfig()).resolved()
        self.classifier = classifier or build_classifier(self.config)
        self._result: Optional[pd.DataFrame] = None
        # Populated by run() when the EMA windows are resolved from the data.
        self.window_resolution: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Classify every bar and return the enriched KPI Table.

        The input DataFrame is not mutated — a copy with the two new columns
        is returned.  Any intermediate indicators the classifier needed (e.g.
        inline EMAs) are *not* added to the table.

        Returns
        -------
        pd.DataFrame
            Copy of the KPI Table with ``regime`` (ordered categorical) and
            ``regime_stable`` (bool) appended.
        """
        self.df = _sort_by_time(self.df)
        self._resolve_ema_windows()
        regime = self.classifier.classify(self.df)
        # Align defensively to the table index in case the classifier reset it.
        regime = pd.Series(regime.values, index=self.df.index, name=REGIME_COL)
        regime = pd.Categorical(
            regime, categories=self.classifier.get_labels(), ordered=True
        )

        out = self.df.copy()
        out[REGIME_COL] = regime
        out[REGIME_STABLE_COL] = _rolling_stability(
            out[REGIME_COL], self.config.stable_window
        )
        self._result = out
        return out

    def get_config(self) -> dict:
        """Return the full configuration used, for traceability in the report."""
        return {
            "classifier": self.classifier.get_config(),
            "labels": list(self.config.labels),
            "stable_window": self.config.stable_window,
            "window_resolution": self.window_resolution,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_ema_windows(self) -> None:
        """Derive the EMA fast/slow spans from the data when auto-window is on.

        The Market Context Module *decides* the EMA spans per dataset from the
        Hurst / Ornstein-Uhlenbeck analysis (the local mean-reversion
        half-life).  The configured ``short_period`` / ``long_period`` are used
        only as a fallback when the half-life does not converge.

        The resolution outcome is recorded in ``self.window_resolution`` for
        report traceability with one of three sources:

        * ``"hurst_ou"``   — spans derived from the converged local half-life;
        * ``"fallback"``   — auto-window requested but did not converge;
        * ``"configured"`` — auto-window disabled, configured spans used as-is.

        Only applies to :class:`EMAProxyClassifier`; for any other classifier
        this is a no-op.
        """
        clf = self.classifier
        if not isinstance(clf, EMAProxyClassifier):
            return

        ema_cfg = clf.config
        if not ema_cfg.auto_window:
            self.window_resolution = {
                "source": "configured",
                "short_period": ema_cfg.short_period,
                "long_period": ema_cfg.long_period,
            }
            return

        src = ema_cfg.source_col
        if src not in self.df.columns:
            # classify() will raise the explanatory KeyError; nothing to derive.
            return

        est_bars, stride_bars, bar_hours = self._estimation_window_bars(ema_cfg)

        prices = self.df[src].astype(float)
        derived = derive_ema_windows(
            prices,
            estimation_window_bars=est_bars,
            stride_bars=stride_bars,
            fast_ratio=ema_cfg.fast_ratio,
            min_estimates=ema_cfg.min_window_estimates,
        )

        meta = {
            "unit": ema_cfg.window_unit,
            "estimation_window_bars": est_bars,
            "bar_hours": round(bar_hours, 4) if bar_hours is not None else None,
        }

        if derived is None:
            # Hurst/OU did not converge → keep the configured fallback spans.
            self.window_resolution = {
                "source": "fallback",
                "short_period": ema_cfg.short_period,
                "long_period": ema_cfg.long_period,
                "half_life_bars": None,
                **meta,
            }
            return

        # Rebuild the classifier with the data-derived spans so that both
        # classify() and get_config() reflect the values actually used.
        new_cfg = replace(
            ema_cfg,
            short_period=derived["short_period"],
            long_period=derived["long_period"],
        )
        self.classifier = EMAProxyClassifier(new_cfg, clf.get_labels())
        self.window_resolution = {
            "source": "hurst_ou",
            "short_period": derived["short_period"],
            "long_period": derived["long_period"],
            "half_life_bars": derived["half_life_bars"],
            "n_estimates": derived["n_estimates"],
            **meta,
        }

    def _estimation_window_bars(self, ema_cfg) -> tuple:
        """Resolve the estimation window / stride into bars for the chosen unit.

        Returns ``(estimation_bars, stride_bars, bar_hours)``.  In ``"bar"``
        mode ``bar_hours`` is ``None`` (not needed); in ``"day"`` mode the
        calendar-day window is converted to bars using the inferred (or
        configured) candle duration so the estimation horizon is the same
        wall-clock length on every timeframe.
        """
        unit = ema_cfg.window_unit.lower().strip()
        if unit == "bar":
            return (
                max(2, int(round(ema_cfg.window_estimation))),
                max(1, int(round(ema_cfg.window_stride))),
                None,
            )
        if unit == "day":
            bar_hours = self._infer_bar_hours(ema_cfg)
            est = max(2, int(round(ema_cfg.window_estimation * 24 / bar_hours)))
            stride = max(1, int(round(ema_cfg.window_stride * 24 / bar_hours)))
            return est, stride, bar_hours
        raise ValueError(
            f"window_unit must be 'bar' or 'day', got '{ema_cfg.window_unit}'."
        )

    def _infer_bar_hours(self, ema_cfg) -> float:
        """Infer the candle duration (hours) for day↔bar conversion.

        Uses ``ema_cfg.bar_hours`` when set, otherwise the median spacing of a
        DatetimeIndex or the first datetime column.

        Raises
        ------
        ValueError
            If the duration cannot be determined and was not provided.
        """
        # `is_set` rather than `is not None`: the field carries UNSET by
        # default now, and both it and an explicit None mean "measure it".
        if is_set(ema_cfg.bar_hours) and ema_cfg.bar_hours is not None:
            return float(ema_cfg.bar_hours)

        # One measurement, shared with the session context and Alpha
        # Discovery, rather than a third private copy (F5, #179).
        measured = measure_bar_hours(self.df)
        if measured is not None:
            return measured

        raise ValueError(
            "window_unit='day' needs the candle duration: provide a "
            "DatetimeIndex (or a datetime column) on the KPI table, or set "
            "ema_proxy.bar_hours explicitly."
        )

    def distribution(self) -> pd.DataFrame:
        """Return the per-regime bar count and share.

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.

        Returns
        -------
        pd.DataFrame
            Indexed by label (in order), columns ``n_bars`` and ``share``.
        """
        if self._result is None:
            raise RuntimeError("Call run() before distribution().")
        regime = self._result[REGIME_COL]
        counts = regime.value_counts().reindex(self.classifier.get_labels())
        total = int(counts.sum())
        return pd.DataFrame(
            {
                "n_bars": counts.astype("Int64"),
                "share": (counts / total).round(4) if total else counts,
            }
        )

    def regime_table(self, timestamp_col: Optional[str] = None) -> pd.DataFrame:
        """Return a compact ``[timestamp, regime, regime_stable]`` frame to join.

        Convenience accessor that extracts just the regime output plus its
        timestamp key, ready to be merged back onto the caller's original
        data::

            mc = MarketContext(kpi); mc.run()
            joined = original_df.merge(mc.regime_table(), on="open_dt", how="left")

        The timestamp is taken from (in order): the requested ``timestamp_col``,
        the table's DatetimeIndex, or the first datetime column found.  The
        returned frame has a plain RangeIndex and keeps ``regime`` as an ordered
        categorical.

        Parameters
        ----------
        timestamp_col : str, optional
            Name to use for the timestamp column (and, when it matches a column
            in the table, the source of the timestamps).  When ``None`` the name
            is inferred from the DatetimeIndex/datetime column, defaulting to
            ``"open_dt"``.

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.
        KeyError
            If ``timestamp_col`` is given but is neither a column nor the
            DatetimeIndex of the table.

        Returns
        -------
        pd.DataFrame
            Columns ``[<timestamp_col>, "regime", "regime_stable"]``.
        """
        if self._result is None:
            raise RuntimeError("Call run() before regime_table().")
        res = self._result

        name, ts_values = self._resolve_timestamp(res, timestamp_col)

        out = pd.DataFrame(
            {
                name: ts_values,
                REGIME_COL: pd.Categorical(
                    res[REGIME_COL].to_numpy(),
                    categories=self.classifier.get_labels(),
                    ordered=True,
                ),
                REGIME_STABLE_COL: res[REGIME_STABLE_COL].to_numpy(),
            }
        )
        return out

    @staticmethod
    def _resolve_timestamp(res: pd.DataFrame, timestamp_col: Optional[str]) -> tuple:
        """Resolve the timestamp key for :meth:`regime_table`.

        Returns ``(column_name, values)``.
        """
        if timestamp_col is not None:
            if timestamp_col in res.columns:
                return timestamp_col, res[timestamp_col].to_numpy()
            if (
                isinstance(res.index, pd.DatetimeIndex)
                and (res.index.name is None or res.index.name == timestamp_col)
            ):
                return timestamp_col, res.index.to_numpy()
            raise KeyError(
                f"timestamp_col '{timestamp_col}' is neither a column nor the "
                f"DatetimeIndex of the enriched table."
            )

        # Inference: DatetimeIndex first, then the first datetime column.
        if isinstance(res.index, pd.DatetimeIndex):
            return (res.index.name or "open_dt"), res.index.to_numpy()
        for col in res.columns:
            if pd.api.types.is_datetime64_any_dtype(res[col]):
                return col, res[col].to_numpy()
        # No datetime found — fall back to the existing index as the key.
        return (res.index.name or "index"), res.index.to_numpy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted in chronological order.

    Tries, in order: DatetimeIndex, first datetime-typed column.  Falls back
    to the original order when no time information is available (e.g. a pure
    numeric index with no datetime columns), so callers are always safe.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df.sort_values(col).reset_index(drop=True)
    return df


def _rolling_stability(regime: pd.Series, window: int) -> pd.Series:
    """Flag bars whose regime has been unchanged for the last ``window`` bars.

    A bar is stable when the current run of identical labels — counting the
    bar itself — has reached ``window`` consecutive bars.  The first
    ``window - 1`` bars of any run are therefore unstable (transition /
    warm-up), and bars with a ``NaN`` regime are never stable.

    Parameters
    ----------
    regime : pd.Series
        Categorical regime labels.
    window : int
        Number of consecutive identical bars required (``N``).

    Returns
    -------
    pd.Series
        Boolean series aligned to ``regime``.
    """
    labels = regime.astype("object")
    if window <= 1:
        # Every classified bar is trivially "stable" over a window of 1.
        return pd.Series(
            labels.notna().to_numpy(dtype=bool),
            index=regime.index,
            name="regime_stable",
        )

    # New run starts wherever the label differs from the previous bar.
    # NaN != NaN evaluates True here, so NaN bars always break the run.
    changed = labels.ne(labels.shift(1))
    run_id = changed.cumsum()
    run_len = labels.groupby(run_id).cumcount() + 1

    stable = (run_len >= window) & labels.notna()
    return pd.Series(stable.to_numpy(dtype=bool), index=regime.index,
                     name="regime_stable")
