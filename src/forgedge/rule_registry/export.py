"""Step 5 — flat-table export for the Rule Registry (FORGE Modulo 4).

One row per rule, every parameter flattened (spec Section 10).  The cross-ticker
block adds one ``pf_{TICKER}`` / ``wr_{TICKER}`` / ``trades_{TICKER}`` /
``verdict_{TICKER}`` column group per ticker tested in the session; a rule's own
source ticker is blank in its row.

FORGE's philosophy is *not to hide complexity* (Section 9): by default the table
keeps duplicates and non-generic rules with explicit flags, so the user can see
why a rule was not promoted.  ``RegistryConfig.export_duplicates`` /
``export_non_generic`` narrow it when desired.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .models import RegistryConfig, RuleDocument


def select_for_export(docs: List[RuleDocument], config: RegistryConfig) -> List[RuleDocument]:
    """Apply the export filters of Section 9 (duplicates / non-generic)."""
    out = []
    for d in docs:
        if not config.export_duplicates and d.is_duplicate:
            continue
        if not config.export_non_generic and d.is_generic is False:
            continue
        out.append(d)
    return out


def flat_table(docs: List[RuleDocument], config: Optional[RegistryConfig] = None) -> pd.DataFrame:
    """Build the flat registry table (spec Section 10).

    Parameters
    ----------
    docs : list[RuleDocument]
        Registry documents (already filtered for export, if desired).
    config : RegistryConfig, optional
        Only used to honour the export filters when passed; the column layout is
        fixed.  Pass ``None`` to lay out every document as-is.
    """
    if config is not None:
        docs = select_for_export(docs, config)

    # All tickers tested anywhere — gives a stable, uniform cross-ticker block.
    tickers: List[str] = []
    for d in docs:
        for t in d.cross_ticker:
            if t not in tickers:
                tickers.append(t)
    tickers.sort()

    rows = []
    for d in docs:
        p = d.params
        st = d.stats
        row = {
            # identification
            "rule_id": d.rule_id,
            "expression": d.expression,
            "source_ticker": d.source_ticker,
            "grade": d.grade,
            "source_alpha_id": d.source_alpha_id,
            "verdict": d.verdict,
            # source statistics
            "pf": st.get("pf"),
            "win_rate": st.get("win_rate"),
            "total_trades": st.get("total_trades"),
            "expectancy": st.get("expectancy"),
            "zero_months": st.get("zero_months"),
            "dsr": st.get("dsr"),
            # operational parameters
            "entry_mode": p.get("entry_mode"),
            "direction": p.get("direction"),
            "buy_drop_pct": p.get("buy_drop_pct"),
            "sell_pct": p.get("sell_pct"),
            "target_h": p.get("target_h"),
            "fee": p.get("fee"),
            # regime
            "regime_type": d.regime.get("type"),
            "regime_avoid": ",".join(d.regime.get("avoid_in", []) or []),
            # correlation (Step 2/3)
            "overlap_max": d.overlap_max,
            "gain_corr_max": d.gain_corr_max,
            "is_duplicate": d.is_duplicate,
            "duplicate_of": d.duplicate_of or "",
        }
        # cross-ticker block
        for t in tickers:
            res = d.cross_ticker.get(t)
            row[f"pf_{t}"] = res.pf if res else float("nan")
            row[f"wr_{t}"] = res.win_rate if res else float("nan")
            row[f"trades_{t}"] = res.total_trades if res else float("nan")
            row[f"verdict_{t}"] = res.verdict if res else ""
        row["cross_ticker_score"] = d.cross_ticker_score
        row["cross_ticker_total"] = d.cross_ticker_total
        row["is_generic"] = d.is_generic
        row["classification"] = d.classification
        rows.append(row)

    return pd.DataFrame(rows)


def export_table(
    docs: List[RuleDocument],
    path: str,
    config: Optional[RegistryConfig] = None,
) -> str:
    """Write the flat table to ``path`` as CSV or Excel and return the path.

    The format follows ``config.export_format`` (default ``"excel"``); when the
    optional ``openpyxl`` dependency is missing the writer falls back to CSV so
    the export never silently fails.
    """
    config = config or RegistryConfig()
    df = flat_table(docs, config)
    fmt = config.export_format

    if fmt == "excel":
        try:
            df.to_excel(path, index=False)
            return path
        except (ImportError, ModuleNotFoundError):
            # openpyxl not installed — degrade gracefully to CSV.
            if path.endswith(".xlsx") or path.endswith(".xls"):
                path = path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(path, index=False)
            return path

    df.to_csv(path, index=False)
    return path
