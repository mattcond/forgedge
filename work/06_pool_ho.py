"""Replay EVERY eligible rule on the hold-out — is the failure selection-specific
or pool-wide?  This is a diagnostic run AFTER the hold-out was already spent."""
import sys, pickle, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'work'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from lib_eval import replay, portfolio_metrics

pool = pickle.load(open('work/out/pool.pkl','rb'))
MIN_OOS_TRADES,MIN_OOS_PF,MIN_IS_PF,MIN_IS_TRADES,MAXH = 4,1.10,1.20,10,20
def elig(r):
    i,o=r["m_is"],r["m_oos"]
    return (1<=r["params"].target_h<=MAXH and i.get("n_trades",0)>=MIN_IS_TRADES
            and o.get("n_trades",0)>=MIN_OOS_TRADES and i.get("profit_factor",0)>=MIN_IS_PF
            and o.get("profit_factor",0)>=MIN_OOS_PF and o.get("net_gain_eur",0)>0 and i.get("net_gain_eur",0)>0)

rows=[]
for r in pool:
    s,t = replay(r,"2026-01-01",None)
    m = portfolio_metrics(t, s.total_signals, "HO", "2026-01-01", "2026-09-03")
    rows.append(dict(ticker=r["ticker"], alpha_id=r["alpha_id"], direction=r["direction"],
                     eligible=elig(r), target_h=r["params"].target_h,
                     is_pf=r["m_is"].get("profit_factor"), oos_pf=r["m_oos"].get("profit_factor"),
                     ho_n=m.get("n_trades",0), ho_pf=m.get("profit_factor"),
                     ho_net=m.get("net_gain_eur",0.0), ho_wr=m.get("win_rate")))
df=pd.DataFrame(rows); df.to_csv('work/out/pool_ho.csv', index=False)

for lab, sub in (("ALL 1510 tradeable", df), ("815 ELIGIBLE (IS+OOS gates)", df[df.eligible])):
    s = sub[sub.ho_n > 0]
    print(f"\n--- {lab} | {len(s)} rules with >=1 hold-out trade ---")
    print(f"  share with HO net gain > 0 : {(s.ho_net>0).mean():6.1%}")
    print(f"  share with HO PF > 1       : {(s.ho_pf>1).mean():6.1%}")
    print(f"  median HO net (EUR/rule)   : {s.ho_net.median():7.2f}")
    print(f"  mean   HO net (EUR/rule)   : {s.ho_net.mean():7.2f}")
    print(f"  median HO PF               : {s.ho_pf.median():6.2f}")
print("\n--- eligible, by direction ---")
e=df[df.eligible & (df.ho_n>0)]
print(e.groupby("direction").agg(n=("ho_net","size"), pct_pos=("ho_net", lambda x:(x>0).mean()),
                                 med_net=("ho_net","median"), med_pf=("ho_pf","median")).round(3).to_string())
print("\n--- eligible, by ticker ---")
print(e.groupby("ticker").agg(n=("ho_net","size"), pct_pos=("ho_net", lambda x:(x>0).mean()),
                              med_net=("ho_net","median"), med_pf=("ho_pf","median")).round(3).to_string())
