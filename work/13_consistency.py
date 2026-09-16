"""Portfolios of rules that win in IS, in OOS AND in the hold-out.

The hold-out never entered discovery — forge() only ever saw data before
2026-01-01.  It enters here, at selection, as a third confirmation window:
a rule is kept only if it is profitable in all three.

Ranking then goes beyond the three windows: among the rules that qualify,
prefer the ones that win in the most independent CALENDAR QUARTERS. Winning
three windows can be three lucky streaks; winning 14 of 19 quarters cannot.
"""
from __future__ import annotations
import sys, pickle, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "work"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from lib_eval import portfolio_metrics

ROOT = Path("/home/user/forgedge")
mw  = pd.read_csv(ROOT / "work/out/multiwindow.csv")
led = pickle.load(open(ROOT / "work/out/ledgers.pkl", "rb"))
HO_START, HO_END = "2026-01-01", "2026-09-03"
MIN_N, MIN_PF, MAX_H = 4, 1.10, 20
CLS = {"BTCEUR":"crypto","ETHEUR":"crypto","DAX":"index","SP500":"index",
       "COPPER":"commodity","BRENT":"commodity","EURUSD":"fx","GBPUSD":"fx"}

def all_trades(aid):
    parts = [led[aid][w] for w in ("IS","OOS","HO1","HO2") if len(led[aid][w])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ── quarterly consistency, computed once per rule over the whole history ──
qstat = {}
for aid in mw.alpha_id:
    t = all_trades(aid)
    if not len(t):
        qstat[aid] = (0, 0, 0.0, 0.0); continue
    q = t.assign(q=pd.to_datetime(t["open_dt"]).dt.to_period("Q")).groupby("q")["pnl_eur"].sum()
    n_q = len(q); n_pos = int((q > 0).sum())
    worst = float(q.min())
    qstat[aid] = (n_q, n_pos, n_pos / n_q if n_q else 0.0, worst)
mw["q_n"]    = mw.alpha_id.map(lambda a: qstat[a][0])
mw["q_pos"]  = mw.alpha_id.map(lambda a: qstat[a][1])
mw["q_rate"] = mw.alpha_id.map(lambda a: qstat[a][2])
mw["q_worst"]= mw.alpha_id.map(lambda a: qstat[a][3])

# ── the user's gate: winner in IS, in OOS and in the hold-out ──
def gate(d):
    m = (d.target_h >= 0) & (d.target_h <= MAX_H)
    for w in ("IS", "OOS", "HO"):
        m &= (d[f"{w}_n"] >= MIN_N) & (d[f"{w}_pf"] >= MIN_PF) & (d[f"{w}_net"] > 0)
    return m

sub = mw[gate(mw)].copy()
pf_min = np.minimum.reduce([np.minimum(sub[f"{w}_pf"], 4.0) for w in ("IS","OOS","HO")])
sub["pf_min"] = pf_min
sub["score"]  = 0.50*sub.q_rate + 0.30*((pf_min-1)/3.0).clip(0,1) + 0.20*sub.wf_consistency.fillna(0)
sub = sub.sort_values("score", ascending=False)
print(f"Regole vincenti in IS + OOS + HO (h<={MAX_H}): {len(sub)}")
print(f"  trimestri vinti  mediana {sub.q_pos.median():.0f}/{sub.q_n.median():.0f} "
      f"({sub.q_rate.median()*100:.0f}%)   max {sub.q_rate.max()*100:.0f}%")
print(f"  ticker: {dict(sub.ticker.value_counts())}\n")

def jacc(a, b):
    A = set(pd.to_datetime(led[a]["IS"]["open_dt"])) if len(led[a]["IS"]) else set()
    B = set(pd.to_datetime(led[b]["IS"]["open_dt"])) if len(led[b]["IS"]) else set()
    return len(A & B)/len(A | B) if (A and B) else 0.0

NAMES = ["K1 — Consistenza", "K2 — Consistenza", "K3 — Consistenza"]
folios = {n: [] for n in NAMES}; used = set()
for pname in NAMES * 5:
    f = folios[pname]
    if len(f) >= 5: continue
    held = {r["ticker"] for r in f}; kl = [CLS[r["ticker"]] for r in f]; pick = None
    for _, r in sub.iterrows():
        if r.alpha_id in used or r.ticker in held: continue
        if kl.count(CLS[r.ticker]) >= 2: continue
        if any(jacc(r.alpha_id, x["alpha_id"]) > 0.60 for x in f if x["ticker"] == r.ticker): continue
        pick = r; break
    if pick is None:
        print(f"  !! {pname} incompleto a {len(f)}"); continue
    used.add(pick.alpha_id); f.append(pick.to_dict())

SPAN = {"IS": ("2021-01-03","2024-07-04"), "OOS": ("2024-07-04", HO_START), "HO": (HO_START, HO_END)}
def win_trades(f, w):
    keys = ("HO1","HO2") if w == "HO" else (w,)
    parts = [led[r["alpha_id"]][k] for r in f for k in keys if len(led[r["alpha_id"]][k])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

out = {}
COLS = ["window","n_signals","n_trades","win_rate","profit_factor","net_gain_eur","max_dd_eur",
        "max_dd_pct","max_concurrent","max_exposure_eur","mean_concurrent",
        "return_on_max_exposure_pct","expectancy_eur","avg_hold_days","trades_per_month",
        "sharpe_ann","pct_months_positive"]
pd.set_option("display.width", 250)
for n, f in folios.items():
    if len(f) < 5: continue
    wins = []
    for w in ("IS","OOS","HO"):
        nsig = int(sum(r[f"{w}_n"] for r in f))
        wins.append(portfolio_metrics(win_trades(f, w), max(nsig,1), w, *SPAN[w]))
    tr = pd.concat([win_trades(f, w) for w in ("IS","OOS","HO")], ignore_index=True)
    q = tr.assign(q=pd.to_datetime(tr["open_dt"]).dt.to_period("Q")).groupby("q")["pnl_eur"].sum()
    out[n] = dict(rules=f, windows=wins, quarters={str(k): round(float(v),1) for k,v in q.items()})
    print(f"\n{'='*130}\n{n}")
    print(pd.DataFrame([dict(ticker=r["ticker"], dir=r["direction"], h=int(r["target_h"]),
          grade=r["grade"], q=f"{int(r['q_pos'])}/{int(r['q_n'])}",
          IS_pf=round(r["IS_pf"],2), OOS_pf=round(r["OOS_pf"],2), HO_pf=round(r["HO_pf"],2),
          formula=r["formula"][:52]) for r in f]).to_string(index=False))
    print(pd.DataFrame(wins)[COLS].to_string(index=False))
    print(f"  trimestri positivi: {(q>0).sum()}/{len(q)}   peggiore {q.min():+.0f} €   migliore {q.max():+.0f} €")

pickle.dump(out, open(ROOT/"work/out/setK.pkl","wb"))
