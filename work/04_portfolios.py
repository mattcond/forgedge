"""Step 4 — build 3 disjoint 5-rule portfolios from IS/OOS evidence, then
evaluate each on IS / OOS / HOLD-OUT.  The hold-out is read for the first
time here, and plays no part in selection."""
from __future__ import annotations
import sys, pickle, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "work"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from lib_eval import replay, portfolio_metrics, exposure_curve, UNIT

ROOT = Path("/home/user/forgedge"); OUT = ROOT / "work/out"
HO_START, HO_END = "2026-01-01", None

pool = pickle.load(open(OUT / "pool.pkl", "rb"))

# ── selection gates (IS + OOS only) ───────────────────────────────────────
MIN_OOS_TRADES, MIN_OOS_PF, MIN_IS_PF, MIN_IS_TRADES = 4, 1.10, 1.20, 10
MAX_TARGET_H = 20      # <= 20 daily bars, so the 8-month hold-out can close trades
MAX_OVERLAP  = 0.60    # Jaccard on IS entry dates, between two rules on the same ticker

def eligible(r):
    i, o = r["m_is"], r["m_oos"]
    return (1 <= r["params"].target_h <= MAX_TARGET_H
            and i.get("n_trades", 0) >= MIN_IS_TRADES and o.get("n_trades", 0) >= MIN_OOS_TRADES
            and i.get("profit_factor", 0) >= MIN_IS_PF and o.get("profit_factor", 0) >= MIN_OOS_PF
            and o.get("net_gain_eur", 0) > 0 and i.get("net_gain_eur", 0) > 0)

def score(r):
    """Rank on out-of-sample robustness; every term is IS/OOS-only."""
    i, o = r["m_is"], r["m_oos"]
    pf_o = min(o["profit_factor"], 4.0); pf_i = min(i["profit_factor"], 4.0)
    stab = 1.0 - min(abs(pf_i - pf_o) / max(pf_i, pf_o), 1.0)      # IS/OOS PF agreement
    cons = r["wf_consistency"] if r["wf_consistency"] is not None else 0.0
    dsr = r["dsr"] if r["dsr"] is not None and np.isfinite(r["dsr"]) else 0.0
    n_o = min(o["n_trades"] / 20.0, 1.0)
    edge = 1.0 if r["verdict"] == "EDGE" else 0.6
    return (0.30 * (pf_o - 1) + 0.15 * (pf_i - 1) + 0.20 * stab + 0.15 * cons
            + 0.10 * min(dsr, 2.0) / 2.0 + 0.10 * n_o) * edge

cands = sorted([r for r in pool if eligible(r)], key=score, reverse=True)
print(f"pool={len(pool)}  eligible={len(cands)}  tickers={sorted({r['ticker'] for r in cands})}")

# ── snake draft: 3 portfolios x 5 rules, distinct tickers inside a portfolio,
#    no rule shared across portfolios ──────────────────────────────────────
NAMES = ["P1 — Core", "P2 — Diversified", "P3 — Satellite"]
folios = {n: [] for n in NAMES}
used, taken_by_ticker = set(), {}

def entry_days(r):
    """IS entry dates, for the redundancy check."""
    if "_days" not in r:
        from lib_eval import replay
        _, t = replay(r, r["is_window"][0], r["is_window"][1])
        r["_days"] = set(pd.to_datetime(t["fill_dt"]).dt.normalize()) if len(t) else set()
    return r["_days"]

def too_similar(r):
    """Reject a rule that fires on nearly the same days as one already drafted."""
    a = entry_days(r)
    if not a:
        return True
    for other in taken_by_ticker.get(r["ticker"], []):
        b = entry_days(other)
        if b and len(a & b) / len(a | b) > MAX_OVERLAP:
            return True
    return False

for pname in NAMES * 5:
    held = {r["ticker"] for r in folios[pname]}
    pick = next((r for r in cands if r["alpha_id"] not in used
                 and r["ticker"] not in held and not too_similar(r)), None)
    if pick is None:   # relax the distinct-ticker rule if the pool is exhausted
        pick = next((r for r in cands if r["alpha_id"] not in used and not too_similar(r)), None)
    if pick is None:
        print(f"!! not enough eligible rules to fill {pname}"); break
    used.add(pick["alpha_id"]); folios[pname].append(pick)
    taken_by_ticker.setdefault(pick["ticker"], []).append(pick)

WINDOWS = lambda r: [("IS", r["is_window"][0], r["is_window"][1]),
                     ("OOS", r["oos_window"][0], HO_START),
                     ("HO", HO_START, HO_END)]

report = {}
for pname, rules in folios.items():
    if len(rules) < 5:
        continue
    blocks, per_rule, trade_store = [], [], {}
    for wlabel, frm, to in WINDOWS(rules[0]):
        alltr, nsig = [], 0
        for r in rules:
            s, t = replay(r, frm, to)
            nsig += s.total_signals
            alltr.append(t)
            per_rule.append(dict(window=wlabel, ticker=r["ticker"], alpha_id=r["alpha_id"],
                                 **{k: v for k, v in portfolio_metrics(t, s.total_signals, wlabel, frm, to).items()
                                    if k not in ("window", "start", "end")}))
        tr = pd.concat(alltr, ignore_index=True)
        trade_store[wlabel] = tr
        blocks.append(portfolio_metrics(tr, nsig, wlabel, frm, to))
    report[pname] = dict(
        rules=[dict(ticker=r["ticker"], alpha_id=r["alpha_id"], verdict=r["verdict"], grade=r["grade"],
                    direction=r["direction"], formula=r["formula"], expression=r["expression"],
                    target_h=r["params"].target_h, sell_pct=r["params"].sell_pct,
                    buy_type=r["params"].buy_type, buy_drop_pct=r["params"].buy_drop_pct,
                    buy_delay_bar=r["params"].buy_delay_bar, fee=r["params"].fee,
                    rotation_p=r["rotation_p"], wf_consistency=r["wf_consistency"], dsr=r["dsr"],
                    score=round(score(r), 4)) for r in rules],
        windows=blocks, per_rule=per_rule,
        trades={k: v.to_dict("records") for k, v in trade_store.items()},
    )

with open(OUT / "report.pkl", "wb") as fh:
    pickle.dump(report, fh)

pd.set_option("display.width", 260, "display.max_columns", 60)
COLS = ["window", "n_signals", "n_trades", "fill_rate", "win_rate", "profit_factor", "net_gain_eur",
        "max_dd_eur", "max_dd_pct", "max_concurrent", "max_exposure_eur", "mean_concurrent",
        "return_on_max_exposure_pct", "expectancy_eur", "avg_hold_days", "trades_per_month",
        "sharpe_ann", "pct_months_positive"]
for pname, d in report.items():
    print("\n" + "=" * 130); print(pname)
    print(pd.DataFrame([dict(ticker=r["ticker"], alpha=r["alpha_id"][:20], verdict=r["verdict"],
                             grade=r["grade"], dir=r["direction"], h=r["target_h"],
                             sell=r["sell_pct"], entry=r["buy_type"], drop=r["buy_drop_pct"],
                             rot_p=round(r["rotation_p"], 4) if r["rotation_p"] else None,
                             formula=r["formula"][:60]) for r in d["rules"]]).to_string(index=False))
    print(pd.DataFrame(d["windows"])[COLS].to_string(index=False))
