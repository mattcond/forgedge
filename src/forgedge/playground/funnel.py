"""Playground use case cutting across all modules — the tracking issue's "Extra".

Not specific to one module, so it doesn't live in an ``mN.py`` file: the
end-to-end conversion funnel from Event Discovery candidates through to
tradeable rules, per asset.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from ..forge import ForgeResult

__all__ = ["conversion_funnel"]

_STAGES = ("candidates", "contracts", "promoted", "edges")


def conversion_funnel(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format end-to-end funnel count per asset, across every module.

    One row per ``(ticker, stage)`` with the population size at that stage:
    ``candidates`` (M1 gate survivors), ``contracts`` (every M2 evaluation,
    promoted and rejected), ``promoted`` (M2 hypotheses handed to M3), and
    ``edges`` (M3 ``EDGE``/``PARTIAL-EDGE`` verdicts — ``result.edges()``).

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``stage``, ``n``. Downstream, e.g.
        ``df.pivot(index="ticker", columns="stage", values="n")`` for a
        funnel table, or divide adjacent stages per ticker for per-step
        conversion rates.
    """
    rows: List[dict] = []

    for result in results:
        counts = {
            "candidates": len(result.candidates),
            "contracts": len(result.contracts),
            "promoted": len(result.promoted),
            "edges": len(result.edges()),
        }
        for stage in _STAGES:
            rows.append({"ticker": result.ticker, "stage": stage, "n": counts[stage]})

    columns = ["ticker", "stage", "n"]
    return pd.DataFrame(rows, columns=columns)
