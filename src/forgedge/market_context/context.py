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

from typing import Optional

import pandas as pd

from .ema_proxy import EMAProxyClassifier
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
        self.config = config or MarketContextConfig()
        self.classifier = classifier or build_classifier(self.config)
        self._result: Optional[pd.DataFrame] = None

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
        }

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
