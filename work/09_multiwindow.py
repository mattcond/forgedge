"""Score every rule on IS / OOS / HO1 / HO2 in one pass.

run_backtest's date filter only restricts which bars may OPEN a position
(on dt[signal_rn+1]); fill and exit scans always read the whole table.  So a
single full-range replay, bucketed by that entry bar, reproduces the
per-window ledgers exactly — at one replay per rule instead of five.
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "work"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from forgedge.rule_discovery.backtest import run_backtest
from lib_eval import kpi, portfolio_metrics, UNIT

ROOT = Path("/home/user/forgedge")
HO_START, HO_MID, HO_END = "2026-01-01", "2026-05-01", "2026-09-03"

runs = {p.stem: pickle.load(open(p, "rb")) for p in sorted((ROOT / "work/runs").glob("*.pkl"))}
SPLIT = {tk: r["is_oos_split_dt"] for tk, r in runs.items()}
START = {tk: r["disc_start"] for tk, r in runs.items()}

def windows(tk):
    return [("IS", START[tk], SPLIT[tk]), ("OOS", SPLIT[tk], HO_START),
            ("HO1", HO_START, HO_MID), ("HO2", HO_MID, None)]

records, ledgers = [], {}
for tk, r in runs.items():
    df = kpi(tk).copy()
    dt = pd.to_datetime(df["open_dt"]).to_numpy()
    for rule in r["rules"]:
        sig = rule["candidate"].apply(df).fillna(0.0)
        d = df.assign(__sig__=sig.astype(float))
        _, tr = run_backtest(d, "__sig__", rule["params"], timestamp_col="open_dt",
                             return_trades=True)
        active = sig.to_numpy().astype(bool)
        entry_rn = np.where(active)[0]; entry_rn = entry_rn[entry_rn + 1 < len(dt)]
        open_dt_all = dt[entry_rn + 1]

        if len(tr):
            tr = tr.copy()
            tr["open_dt"] = dt[tr["signal_rn"].to_numpy() + 1]
            tr["pnl_eur"] = tr["net_pct_gain"] * UNIT
            tr["fill_dt"] = pd.to_datetime(tr["fill_dt"]); tr["exit_dt"] = pd.to_datetime(tr["exit_dt"])
            tr["rule"] = rule["alpha_id"]; tr["ticker"] = tk
        rec = dict(ticker=tk, alpha_id=rule["alpha_id"], direction=rule["direction"],
                   target_h=rule["params"].target_h, buy_type=rule["params"].buy_type,
                   grade=rule["grade"], rotation_p=rule["rotation_p"],
                   wf_consistency=rule["wf_consistency"], dsr=rule["dsr"],
                   formula=rule["formula"])
        per_win_trades = {}
        for lab, a, b in windows(tk):
            lo = np.datetime64(pd.Timestamp(a)); hi = np.datetime64(pd.Timestamp(b)) if b else None
            m = (open_dt_all >= lo) & ((open_dt_all < hi) if hi is not None else True)
            nsig = int(m.sum())
            if len(tr):
                sel = (tr["open_dt"] >= lo) & ((tr["open_dt"] < hi) if hi is not None else True)
                t = tr[sel]
            else:
                t = tr
            per_win_trades[lab] = t
            mm = portfolio_metrics(t, nsig, lab, a, b or HO_END)
            rec[f"{lab}_n"] = mm.get("n_trades", 0)
            rec[f"{lab}_pf"] = mm.get("profit_factor", 0.0)
            rec[f"{lab}_net"] = mm.get("net_gain_eur", 0.0)
            rec[f"{lab}_wr"] = mm.get("win_rate")
            rec[f"{lab}_dd"] = mm.get("max_dd_eur")
        # HO = HO1 + HO2 combined
        tho = pd.concat([per_win_trades["HO1"], per_win_trades["HO2"]]) if len(tr) else tr
        nho = rec["HO1_n"] + rec["HO2_n"]
        mm = portfolio_metrics(tho, rec.get("HO1_n", 0) + rec.get("HO2_n", 0) or 1, "HO", HO_START, HO_END)
        rec.update(HO_n=mm.get("n_trades", 0), HO_pf=mm.get("profit_factor", 0.0),
                   HO_net=mm.get("net_gain_eur", 0.0), HO_wr=mm.get("win_rate"),
                   HO_dd=mm.get("max_dd_eur"))
        records.append(rec)
        ledgers[rule["alpha_id"]] = {k: v for k, v in per_win_trades.items()}
    print(f"  {tk}: {len(r['rules'])} rules", flush=True)

mw = pd.DataFrame(records)
mw.to_csv(ROOT / "work/out/multiwindow.csv", index=False)
with open(ROOT / "work/out/ledgers.pkl", "wb") as fh:
    pickle.dump(ledgers, fh)
print("\nrules scored:", len(mw))

# sanity: IS/OOS must reproduce the earlier per-window run
old = pd.read_csv(ROOT / "work/out/pool.csv")[["alpha_id", "is_n", "is_pf", "oos_n", "oos_pf"]]
chk = mw.merge(old, on="alpha_id")
print("IS n match :", bool((chk.IS_n == chk.is_n).all()),
      "| OOS n match:", bool((chk.OOS_n == chk.oos_n).all()),
      "| IS pf max abs diff:", float((chk.IS_pf - chk.is_pf).abs().max()))
