"""Step 3 — build the candidate-rule pool and score it on IS/OOS only."""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "work"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from lib_eval import replay, portfolio_metrics, UNIT

ROOT = Path("/home/user/forgedge")
HO_START = "2026-01-01"

runs = {}
for p in sorted((ROOT / "work/runs").glob("*.pkl")):
    with open(p, "rb") as fh:
        runs[p.stem] = pickle.load(fh)

rows, pool = [], []
for tk, r in runs.items():
    split = r["is_oos_split_dt"]
    for rule in r["rules"]:
        rule["is_window"] = (r["disc_start"], split)
        rule["oos_window"] = (split, HO_START)
        s_is, t_is = replay(rule, r["disc_start"], split)
        s_oos, t_oos = replay(rule, split, HO_START)
        m_is = portfolio_metrics(t_is, s_is.total_signals, "IS", r["disc_start"], split)
        m_oos = portfolio_metrics(t_oos, s_oos.total_signals, "OOS", split, HO_START)
        rule["m_is"], rule["m_oos"] = m_is, m_oos
        pool.append(rule)
        rows.append(dict(
            ticker=tk, alpha_id=rule["alpha_id"], verdict=rule["verdict"], grade=rule["grade"],
            dir=rule["direction"], h=rule["params"].target_h, sell=rule["params"].sell_pct,
            entry=rule["params"].buy_type, drop=rule["params"].buy_drop_pct,
            rot_p=round(rule["rotation_p"], 4) if rule["rotation_p"] is not None else None,
            is_n=m_is.get("n_trades", 0), is_wr=m_is.get("win_rate"), is_pf=m_is.get("profit_factor"),
            is_net=m_is.get("net_gain_eur"),
            oos_n=m_oos.get("n_trades", 0), oos_wr=m_oos.get("win_rate"), oos_pf=m_oos.get("profit_factor"),
            oos_net=m_oos.get("net_gain_eur"), oos_dd=m_oos.get("max_dd_eur"),
            wf_cons=rule["wf_consistency"], dsr=rule["dsr"],
            formula=rule["formula"][:70],
        ))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "work/out/pool.csv", index=False)
with open(ROOT / "work/out/pool.pkl", "wb") as fh:
    pickle.dump(pool, fh)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(df.sort_values(["ticker", "oos_pf"], ascending=[True, False]).to_string(index=False))
print(f"\npool size = {len(df)}")
