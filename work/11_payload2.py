import sys, pickle, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'work'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path("/home/user/forgedge")
mw=pd.read_csv(ROOT/"work/out/multiwindow.csv")
MIN_N,MIN_PF=4,1.10
CLS={"BTCEUR":"crypto","ETHEUR":"crypto","DAX":"index","SP500":"index",
     "COPPER":"commodity","BRENT":"commodity","EURUSD":"fx","GBPUSD":"fx"}
def gate(d,wins):
    m=np.ones(len(d),bool)
    for w in wins: m&=(d[f"{w}_n"]>=MIN_N)&(d[f"{w}_pf"]>=MIN_PF)&(d[f"{w}_net"]>0)
    return m

out={}
out["counts"]={"+".join(w) if w else "none": int(gate(mw,w).sum()) for w in
               [("IS",),("IS","OOS"),("IS","OOS","HO"),("IS","OOS","HO1"),("IS","OOS","HO1","HO2")]}
out["n_total"]=int(len(mw))

def experiment(d, lab):
    rows=[]
    for name,wins in [("no gate",()),("IS + OOS",("IS","OOS")),("IS + OOS + HO1",("IS","OOS","HO1"))]:
        m=gate(d,wins) if wins else np.ones(len(d),bool); s=d[m]
        rows.append(dict(gate=name,n=int(len(s)),pct=round(float((s.HO2_net>0).mean()*100),1),
                         med_pf=round(float(s.HO2_pf.median()),3),med_net=round(float(s.HO2_net.median()),2)))
    return dict(label=lab, universe=int(len(d)), rows=rows)
out["exp_all"]=experiment(mw[mw.HO2_n>=3],"all horizons")
cap=mw[(mw.target_h>=1)&(mw.target_h<=20)&(mw.HO2_n>=3)]
out["exp_cap"]=experiment(cap,"horizon <= 20 bars")

byc={}
for name,wins in [("IS + OOS",("IS","OOS")),("IS + OOS + HO1",("IS","OOS","HO1"))]:
    s=cap[gate(cap,wins)].assign(cls=lambda x:x.ticker.map(CLS))
    byc[name]={k:dict(n=int(len(v)),pct=round(float((v.HO2_net>0).mean()*100),1),
                      med_pf=round(float(v.HO2_pf.median()),2)) for k,v in s.groupby("cls")}
out["by_class"]=byc

books={}
for tag,path in (("A","triple.pkl"),("B","triple.pkl"),("C","setC.pkl")):
    obj=pickle.load(open(ROOT/f"work/out/{path}","rb"))
    src = obj[tag] if tag in obj else obj
    for n,b in src.items():
        books[n]=dict(set=tag,
            rules=[dict(ticker=r["ticker"],direction=r["direction"],target_h=int(r["target_h"]),
                        grade=r["grade"],formula=r["formula"],
                        ho2_n=int(r["HO2_n"]),ho2_net=round(float(r["HO2_net"]),1)) for r in b["rules"]],
            windows=b["windows"])
out["books"]=books
json.dump(out,open(ROOT/"work/out/payload2.json","w"),default=str)
print("books:",list(books)); print("bytes:",len(json.dumps(out,default=str)))
