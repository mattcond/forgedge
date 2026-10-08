"""Do unsupervised chart setups help FORGE?  KPI vs KPI + ``setup`` column.

For each asset, ``forge()`` runs twice on the same data, same preset, same
fee and the same ``PYTHONHASHSEED``:

  A  baseline   KPI table (build_features + candle_features)
  B  +setup     same KPI table plus the categorical ``setup`` column from
                ``setup_labels.py`` (labels fitted on the first 50% only)

Event Discovery one-hot expands ``setup`` into ``is_setup_Sxx`` events; the
default two-pass composition then pairs them with indicator events
("event X AND setup Sxx") — i.e. the setup is used as *context* for an
event, judged by FORGE's own gates (M2 OOS tail, M3 walk-forward,
rotation null).  Nothing here looks at returns outside FORGE.

Must be launched with a pinned hash seed for run-to-run comparability:
    PYTHONHASHSEED=0 python experiments/chart_vision/forge_setups.py [assets...]
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from forgedge import forge, forge_preset

HERE = Path(__file__).resolve().parent
SETUPS = HERE / "results" / "setups_k12"
OUT = HERE / "results" / "forge_setups"

# per-side cost as a fraction of price (spread + commission, retail CFD/FX order of magnitude)
FEES = {"E_SandP-500": 0.0001, "E_DAAX": 0.0001, "EURUSD": 0.00009,
        "GBPUSD": 0.0001, "E_Brent": 0.0002, "BTCEUR": 0.001}
PRESET = "balanced"


def run(kpi, asset, fee):
    d, a, r = forge_preset(PRESET, timeframe="1H", asset=asset)
    a = dataclasses.replace(a, fee_per_side=fee)
    r = dataclasses.replace(r, base_params=dataclasses.replace(r.base_params, fee=fee))
    return forge(kpi, ticker=asset, timeframe="1H",
                 event_discovery_config=d, alpha_config=a, rule_discovery_config=r)


def rules_table(res, tag):
    cands = {c.event_id: c for c in (res.candidates or [])}
    rows = []
    for contract, resp in res.rule_responses:
        cand = cands.get(contract.event_candidate_id)
        expr = cand.expression if cand is not None else contract.event_candidate_id
        wf = resp.walk_forward
        oos = wf.oos_summary if wf is not None else None
        rows.append(dict(
            run=tag, alpha_id=contract.alpha_id, expression=expr,
            uses_setup="setup" in expr, direction=contract.direction,
            verdict=resp.verdict,
            is_trades=resp.in_sample_summary.total_trades,
            is_pf=resp.in_sample_summary.profit_factor,
            oos_trades=oos.total_trades if oos else None,
            oos_pf=oos.profit_factor if oos else None,
            oos_expectancy=oos.expectancy if oos else None,
            oos_win_pct=oos.win_rate_pct if oos else None,
            wf_profitable_splits=(f"{wf.n_profitable_splits}/{len(wf.splits)}" if wf else None),
            rejection="; ".join(resp.rejection_reasons)[:200],
        ))
    return pd.DataFrame(rows)


def counts(res, tbl):
    n_setup_cands = sum("setup" in c.expression for c in (res.candidates or []))
    v = tbl["verdict"].value_counts() if len(tbl) else pd.Series(dtype=int)
    vs = tbl[tbl.uses_setup]["verdict"].value_counts() if len(tbl) else pd.Series(dtype=int)
    return dict(candidates=len(res.candidates or []), setup_candidates=n_setup_cands,
                promoted=len(res.promoted or []), m3_evaluated=len(tbl),
                edge=int(v.get("EDGE", 0)), partial=int(v.get("PARTIAL-EDGE", 0)),
                edge_with_setup=int(vs.get("EDGE", 0)), partial_with_setup=int(vs.get("PARTIAL-EDGE", 0)))


def main(assets):
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for asset in assets:
        kpi_b = pd.read_parquet(SETUPS / f"{asset}_1H_setup.parquet")
        kpi_a = kpi_b.drop(columns=["setup"])
        for tag, kpi in (("A_baseline", kpi_a), ("B_with_setup", kpi_b)):
            t0 = time.time()
            res = run(kpi, asset, FEES[asset])
            tbl = rules_table(res, tag)
            tbl.insert(0, "asset", asset)
            tbl.to_csv(OUT / f"{asset}_{tag}_rules.csv", index=False)
            c = dict(asset=asset, run=tag, minutes=round((time.time() - t0) / 60, 1), **counts(res, tbl))
            summary.append(c)
            print(json.dumps(c), flush=True)
            pd.DataFrame(summary).to_csv(OUT / f"summary_{'_'.join(assets)}.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or list(FEES))
