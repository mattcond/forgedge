import sys, pickle, json; sys.path.insert(0,'work')
import pandas as pd, numpy as np

def eq(trades):
    df=pd.DataFrame(trades)
    if df.empty: return {"x":[],"y":[]}
    df["exit_dt"]=pd.to_datetime(df["exit_dt"])
    s=df.sort_values("exit_dt").set_index("exit_dt")["pnl_eur"].groupby(level=0).sum().cumsum()
    return {"x":[d.strftime("%Y-%m-%d") for d in s.index],"y":[round(float(v),2) for v in s.values]}

def monthly(trades):
    df=pd.DataFrame(trades)
    if df.empty: return {}
    df["exit_dt"]=pd.to_datetime(df["exit_dt"])
    m=df.groupby(df.exit_dt.dt.to_period("M"))["pnl_eur"].sum().round(1)
    return {str(k):float(v) for k,v in m.items()}

out={}
for tag,path in (("v1","work/out/report.pkl"),("v2","work/out/report_v2.pkl")):
    rep=pickle.load(open(path,"rb")); books={}
    for pname,d in rep.items():
        per_rule={}
        for w in ("IS","OOS","HO"):
            t=pd.DataFrame(d["trades"][w])
            per_rule[w]={r["alpha_id"]: dict(
                n=int((t.rule==r["alpha_id"]).sum()),
                net=round(float(t.loc[t.rule==r["alpha_id"],"pnl_eur"].sum()),2),
                wr=round(float((t.loc[t.rule==r["alpha_id"],"pnl_eur"]>0).mean()),3) if (t.rule==r["alpha_id"]).any() else None
            ) for r in d["rules"]}
        books[pname]=dict(rules=d["rules"], windows=d["windows"],
                          equity={w:eq(d["trades"][w]) for w in ("IS","OOS","HO")},
                          monthly={w:monthly(d["trades"][w]) for w in ("IS","OOS","HO")},
                          per_rule=per_rule)
    out[tag]=books

ph=pd.read_csv("work/out/pool_ho.csv"); e=ph[ph.eligible & (ph.ho_n>0)]
out["pool"]=dict(
    n_all=int(len(ph)), n_elig=int(len(e)),
    all_pos=round(float((ph[ph.ho_n>0].ho_net>0).mean()),4),
    elig_pos=round(float((e.ho_net>0).mean()),4),
    elig_med_net=round(float(e.ho_net.median()),2), elig_med_pf=round(float(e.ho_pf.median()),3),
    by_ticker={k: dict(n=int(v.shape[0]), pct_pos=round(float((v.ho_net>0).mean()),3),
                       med_net=round(float(v.ho_net.median()),2), med_pf=round(float(v.ho_pf.median()),3))
               for k,v in e.groupby("ticker")},
    by_dir={k: dict(n=int(v.shape[0]), pct_pos=round(float((v.ho_net>0).mean()),3),
                    med_net=round(float(v.ho_net.median()),2)) for k,v in e.groupby("direction")},
)
out["bh"]={"BRENT":51.9,"BTCEUR":-11.6,"COPPER":14.7,"DAX":5.5,"ETHEUR":-19.2,"SP500":11.9,"EURUSD":-1.4,"GBPUSD":0.1}
out["bh_dd"]={"BRENT":-37.5,"BTCEUR":-38.4,"COPPER":-16.5,"DAX":-13.6,"ETHEUR":-52.8,"SP500":-9.5,"EURUSD":-5.6,"GBPUSD":-4.8}
out["runs"]={"BTCEUR":[13882,1388,271],"ETHEUR":[14291,1630,156],"EURUSD":[13563,1792,2],"GBPUSD":[13744,2755,7],
             "COPPER":[13457,1603,92],"BRENT":[9263,2831,738],"DAX":[12636,2012,130],"SP500":[12701,1146,114]}
out["splits"]={"BTCEUR":"2025-03-10","ETHEUR":"2025-03-10","EURUSD":"2024-07-03","GBPUSD":"2024-07-03",
               "COPPER":"2024-07-05","BRENT":"2024-06-20","DAX":"2024-07-07","SP500":"2024-07-04"}
json.dump(out, open("work/out/payload.json","w"), default=str)
print("payload:", len(json.dumps(out, default=str)), "bytes")
