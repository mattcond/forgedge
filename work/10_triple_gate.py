"""Portfolios selected by verifying metrics across three windows.

SET A — gate on IS + OOS + HO (the literal request).  263 rules qualify.
        Nothing is left unseen, so its numbers are descriptive, not predictive.
SET B — gate on IS + OOS + HO1 only, keeping HO2 (2026-05-01 ->) sealed.
        Same three-window logic, but one clean window survives to measure it.
"""
from __future__ import annotations
import sys, pickle, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "work"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from lib_eval import portfolio_metrics

ROOT = Path("/home/user/forgedge")
mw = pd.read_csv(ROOT / "work/out/multiwindow.csv")
led = pickle.load(open(ROOT / "work/out/ledgers.pkl", "rb"))
runs = {p.stem: pickle.load(open(p, "rb")) for p in sorted((ROOT / "work/runs").glob("*.pkl"))}
SPLIT = {tk: r["is_oos_split_dt"] for tk, r in runs.items()}
START = {tk: r["disc_start"] for tk, r in runs.items()}
HO_START, HO_MID, HO_END = "2026-01-01", "2026-05-01", "2026-09-03"
MIN_N, MIN_PF = 4, 1.10
CLASS = {"BTCEUR":"crypto","ETHEUR":"crypto","DAX":"index","SP500":"index",
         "COPPER":"commodity","BRENT":"commodity","EURUSD":"fx","GBPUSD":"fx"}

def gate(d, wins):
    m = np.ones(len(d), bool)
    for w in wins:
        m &= (d[f"{w}_n"] >= MIN_N) & (d[f"{w}_pf"] >= MIN_PF) & (d[f"{w}_net"] > 0)
    return m

def rank(d, wins):
    """Rank on the WEAKEST window — a rule must earn its place in all of them."""
    pf = np.minimum.reduce([np.minimum(d[f"{w}_pf"].to_numpy(), 4.0) for w in wins])
    nmin = np.minimum.reduce([d[f"{w}_n"].to_numpy() for w in wins]) / 20.0
    cons = d["wf_consistency"].fillna(0).to_numpy()
    return 0.55*(pf-1) + 0.20*np.minimum(nmin,1.0) + 0.25*cons

def jacc(a, b):
    A = set(pd.to_datetime(led[a]["IS"]["open_dt"])) if len(led[a]["IS"]) else set()
    B = set(pd.to_datetime(led[b]["IS"]["open_dt"])) if len(led[b]["IS"]) else set()
    return len(A & B)/len(A | B) if (A and B) else 0.0

def draft(cands, names, max_class=2):
    folios = {n: [] for n in names}; used=set()
    for pname in names*5:
        f = folios[pname]
        if len(f) >= 5: continue
        held = {r["ticker"] for r in f}
        kl = [CLASS[r["ticker"]] for r in f]
        pick=None
        for _, r in cands.iterrows():
            if r.alpha_id in used or r.ticker in held: continue
            if kl.count(CLASS[r.ticker]) >= max_class: continue
            if any(jacc(r.alpha_id, x["alpha_id"]) > 0.60 for x in f if x["ticker"]==r.ticker): continue
            pick = r; break
        if pick is None:
            print(f"  !! {pname} incompleto a {len(f)} regole"); continue
        used.add(pick.alpha_id); f.append(pick.to_dict())
    return folios

def evaluate(folio, wins, tk_windows):
    out=[]
    for w in wins:
        tr = pd.concat([led[r["alpha_id"]][w] for r in folio if len(led[r["alpha_id"]][w])],
                       ignore_index=True) if any(len(led[r["alpha_id"]][w]) for r in folio) else pd.DataFrame()
        nsig = int(sum(r[f"{w}_n"] for r in folio))
        a, b = tk_windows[w]
        out.append(portfolio_metrics(tr, max(nsig,1), w, a, b))
    return out

WIN_SPAN = {"IS":("2021-01-03", "2024-07-04"), "OOS":("2024-07-04", HO_START),
            "HO1":(HO_START, HO_MID), "HO2":(HO_MID, HO_END), "HO":(HO_START, HO_END)}

results={}
for tag, gwins, ewins, names in (
    ("A", ("IS","OOS","HO"),  ("IS","OOS","HO1","HO2"), ["A1","A2","A3"]),
    ("B", ("IS","OOS","HO1"), ("IS","OOS","HO1","HO2"), ["B1","B2","B3"]),
):
    m = gate(mw, gwins); sub = mw[m].copy()
    sub["score"] = rank(sub, gwins)
    sub = sub.sort_values("score", ascending=False)
    print(f"\nSET {tag} — gate {'+'.join(gwins)}: {len(sub)} regole ammissibili "
          f"su {sub.ticker.nunique()} ticker  {sorted(sub.ticker.unique())}")
    folios = draft(sub, names)
    books={}
    for n, f in folios.items():
        if len(f) < 5: continue
        books[n] = dict(rules=f, windows=evaluate(f, ewins, WIN_SPAN))
    results[tag]=books

COLS=["window","n_trades","win_rate","profit_factor","net_gain_eur","max_dd_eur","max_dd_pct",
      "max_concurrent","max_exposure_eur","expectancy_eur","sharpe_ann","pct_months_positive"]
pd.set_option("display.width",240)
for tag, books in results.items():
    for n, b in books.items():
        print(f"\n{'='*128}\n{n}   ({'SET '+tag})")
        print(pd.DataFrame([dict(ticker=r["ticker"], dir=r["direction"], h=r["target_h"],
              grade=r["grade"], formula=r["formula"][:62]) for r in b["rules"]]).to_string(index=False))
        print(pd.DataFrame(b["windows"])[COLS].to_string(index=False))

with open(ROOT/"work/out/triple.pkl","wb") as fh:
    pickle.dump(results, fh)
