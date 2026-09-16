"""Step 2 — run the FORGE pipeline per ticker on the DISCOVERY span only.

The hold-out (>= HO_START) is never loaded here: forge() only ever sees
kpi[open_dt < HO_START].  Results are pickled to work/runs/<TICKER>.pkl.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONHASHSEED") != "0":          # pitfall #24 — determinism
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import warnings, pickle, time, dataclasses
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import pandas as pd
from forgedge import forge, forge_preset

ROOT = Path("/home/user/forgedge")
RUNS = ROOT / "work" / "runs"; RUNS.mkdir(parents=True, exist_ok=True)
HO_START = "2026-01-01"
FEE_PER_SIDE = 0.001          # 0.10% per side -> 0.20% round trip

TICKERS = sys.argv[1:] or ["BTCEUR", "ETHEUR", "EURUSD", "GBPUSD", "COPPER", "BRENT", "DAX", "SP500"]
PRESET = os.environ.get("FORGE_PRESET", "balanced")

for tk in TICKERS:
    kpi = pd.read_parquet(ROOT / f"work/kpi/{tk}.parquet")
    disc = kpi[kpi["open_dt"] < HO_START].reset_index(drop=True)
    dc, ac, rc = forge_preset(PRESET, "1D", asset=tk)
    ac = dataclasses.replace(ac, fee_per_side=FEE_PER_SIDE)
    t0 = time.time()
    res = forge(disc, ticker=tk, timeframe="1D",
                event_discovery_config=dc, alpha_config=ac, rule_discovery_config=rc,
                run_registry=False, progress=False)
    dt = time.time() - t0
    edges = res.edges()
    split_dt = str(disc["open_dt"].iloc[res.time_budget.split].date()) if res.time_budget else None
    payload = dict(
        ticker=tk, preset=PRESET, fee_per_side=FEE_PER_SIDE, elapsed_s=dt,
        disc_start=str(disc["open_dt"].iloc[0].date()), disc_end=str(disc["open_dt"].iloc[-1].date()),
        is_oos_split_dt=split_dt, n_candidates=len(res.candidates),
        n_promoted=len(res.promoted), ledger=res.ledger.describe() if res.ledger else None,
        rules=[dict(
            ticker=tk, alpha_id=c.alpha_id, event_id=c.event_candidate_id,
            expression=cand_by_id[c.event_candidate_id].expression if (cand_by_id := {x.event_id: x for x in res.candidates}) else None,
            formula=cand_by_id[c.event_candidate_id].event_formula,
            grade=c.alpha_score.grade, composite=c.alpha_score.composite_score,
            rotation_p=c.rotation_p, direction=c.derived_target.direction,
            holding_h=c.derived_target.holding_period_h, verdict=r.verdict,
            params=r.validated_rule.params.resolved() if r.validated_rule else None,
            entry_mode_sel=(r.entry_optimization.selected_entry if r.entry_optimization else None),
            is_pf=r.in_sample_summary.profit_factor, is_trades=r.in_sample_summary.total_trades,
            oos_pf=(r.walk_forward.oos_summary.profit_factor if r.walk_forward and r.walk_forward.oos_summary else None),
            oos_trades=(r.walk_forward.oos_summary.total_trades if r.walk_forward and r.walk_forward.oos_summary else None),
            wf_consistency=(r.walk_forward.consistency if r.walk_forward else None),
            dsr=(r.statistical_validation.deflated_sharpe if r.statistical_validation else None),
            ttest_p=(r.statistical_validation.t_test_p_value if r.statistical_validation and hasattr(r.statistical_validation,"t_test_p_value") else None),
            candidate=cand_by_id[c.event_candidate_id],
        ) for c, r in edges],
    )
    with open(RUNS / f"{tk}.pkl", "wb") as fh:
        pickle.dump(payload, fh)
    verds = pd.Series([r.verdict for _, r in edges]).value_counts().to_dict() if edges else {}
    print(f"{tk:7s} {dt:7.1f}s  cand={len(res.candidates):5d} promoted={len(res.promoted):4d} "
          f"tradeable={len(edges):3d} {verds}  split={split_dt}", flush=True)
