import sys, pickle, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'work'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path("/home/user/forgedge")
K=pickle.load(open(ROOT/"work/out/setK.pkl","rb"))
mw=pd.read_csv(ROOT/"work/out/multiwindow.csv")
led=pickle.load(open(ROOT/"work/out/ledgers.pkl","rb"))
MIN_N,MIN_PF,MAX_H=4,1.10,20
def gate(d):
    m=(d.target_h>=0)&(d.target_h<=MAX_H)
    for w in ("IS","OOS","HO"): m&=(d[f"{w}_n"]>=MIN_N)&(d[f"{w}_pf"]>=MIN_PF)&(d[f"{w}_net"]>0)
    return m
sub=mw[gate(mw)]
out=dict(n_pool=int(len(mw)), n_qualify=int(len(sub)),
         by_ticker={k:int(v) for k,v in sub.ticker.value_counts().items()},
         books={})
for n,b in K.items():
    rules=[]
    for r in b["rules"]:
        aid=r["alpha_id"]
        per={}
        for w,keys in (("IS",("IS",)),("OOS",("OOS",)),("HO",("HO1","HO2"))):
            t=pd.concat([led[aid][k] for k in keys if len(led[aid][k])],ignore_index=True) \
              if any(len(led[aid][k]) for k in keys) else pd.DataFrame()
            per[w]=dict(n=int(len(t)), net=round(float(t["pnl_eur"].sum()),1) if len(t) else 0.0,
                        pf=round(float(r[f"{w}_pf"]),2),
                        wr=round(float((t["pnl_eur"]>0).mean()*100),1) if len(t) else None)
        rules.append(dict(ticker=r["ticker"],direction=r["direction"],target_h=int(r["target_h"]),
                          grade=r["grade"],formula=r["formula"],sell_pct=round(float(r.get("sell_pct",0) or 0),4),
                          q_pos=int(r["q_pos"]),q_n=int(r["q_n"]),per=per))
    out["books"][n]=dict(rules=rules,windows=b["windows"],quarters=b["quarters"])
json.dump(out,open(ROOT/"work/out/payload3.json","w"),default=str)
print("qualify:",out["n_qualify"],"| books:",list(out["books"]),"|",len(json.dumps(out,default=str)),"bytes")
