"""Setup as *context*: does conditioning a validated FORGE event on the chart
setup improve it — more than conditioning on a fake setup does?

Per asset:

1. Baseline ``forge()`` (preset ``balanced`` 1H, asset fee) on the plain KPI
   table.  The top ``N_PARENTS`` promoted contracts by M2 composite score are
   the *parent* events X (already through M1 + M2).
2. Each parent's activation series is materialised as a 0/1 column ``ev_i``
   (its thresholds are frozen by M1 — invariant #2) and a fake setup column
   ``fsetup`` is built by circularly shifting ``setup`` by half the series
   (same label frequencies and run lengths, no link to the chart at that bar).
3. A second ``forge()`` evaluates, through M2 + M3 with its own null and
   walk-forward, the manual hypotheses
       P_i                         (the parent alone, same session)
       P_i & setup == Sk           for every real setup k
       P_i & fsetup == Sk          for every fake setup k   (control)
4. Question: do real-setup children beat their parent (OOS expectancy,
   verdict) more often / by more than fake-setup children do?

Labels come from ``setup_labels.py`` (fitted on the first 50% of each series,
no returns).  Launch with a pinned hash seed:
    PYTHONHASHSEED=0 python experiments/chart_vision/forge_conditioning.py --k 4 E_SandP-500
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forgedge import CustomEvent, forge, forge_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "forge_conditioning"
FEES = {"E_SandP-500": 0.0001, "E_DAAX": 0.0001, "EURUSD": 0.00009,
        "GBPUSD": 0.0001, "E_Brent": 0.0002}
N_PARENTS = 20
W_SETUP = 20  # candles per setup chart (setup_labels.W)


def configs(asset):
    d, a, r = forge_preset("balanced", timeframe="1H", asset=asset)
    fee = FEES[asset]
    a = dataclasses.replace(a, fee_per_side=fee)
    r = dataclasses.replace(r, base_params=dataclasses.replace(r.base_params, fee=fee))
    return d, a, r


def stats(resp):
    wf = resp.walk_forward
    oos = wf.oos_summary if wf is not None else None
    return dict(verdict=resp.verdict,
                is_trades=resp.in_sample_summary.total_trades,
                is_expectancy=resp.in_sample_summary.expectancy,
                oos_trades=oos.total_trades if oos else np.nan,
                oos_expectancy=oos.expectancy if oos else np.nan,
                oos_pf=oos.profit_factor if oos else np.nan,
                oos_win_pct=oos.win_rate_pct if oos else np.nan,
                wf_profitable=(wf.n_profitable_splits / len(wf.splits)) if wf and wf.splits else np.nan)


def main(asset, k):
    OUT.mkdir(parents=True, exist_ok=True)
    kpi = pd.read_parquet(HERE / "results" / f"setups_k{k}" / f"{asset}_1H_setup.parquet")
    d, a, r = configs(asset)
    t0 = time.time()

    # 1. baseline run -> parents (cached: the baseline is the expensive part)
    cache = OUT / f"{asset}_k{k}_parents.pkl"
    if cache.exists():
        parents_df = pd.read_pickle(cache)
    else:
        base = forge(kpi.drop(columns=["setup"]), ticker=asset, timeframe="1H",
                     event_discovery_config=d, alpha_config=a, rule_discovery_config=r)
        summ = base.summary()
        score = dict(zip(summ["alpha_id"], summ["composite_score"]))
        cands = {c.event_id: c for c in base.candidates}
        promoted = sorted(base.promoted, key=lambda c: -score.get(c.alpha_id, -np.inf))
        parents, seen = [], set()
        for c in promoted:
            cand = cands[c.event_candidate_id]
            if cand.expression in seen:
                continue
            seen.add(cand.expression)
            parents.append(cand)
            if len(parents) == N_PARENTS:
                break
        base_verdicts = {cands[c.event_candidate_id].expression: resp.verdict
                         for c, resp in base.rule_responses}
        print(f"[{asset}] baseline done in {(time.time()-t0)/60:.1f} min, "
              f"{len(base.promoted)} promoted, parents={len(parents)}", flush=True)
        parents_df = pd.DataFrame(dict(expression=[c.expression for c in parents],
                                       series=[c.event_series for c in parents],
                                       baseline_verdict=[base_verdicts.get(c.expression) for c in parents]))
        parents_df.to_pickle(cache)

    # 2. materialise parents + fake setup
    kc = kpi.copy()
    ts = pd.DatetimeIndex(pd.to_datetime(kc["open_dt"]))
    for i, ser in enumerate(parents_df["series"]):
        s = ser.reindex(ts)
        kc[f"ev_{i:02d}"] = s.fillna(0).astype(int).to_numpy()
    kc["fsetup"] = np.roll(kc["setup"].to_numpy(), len(kc) // 2)
    # numeric control: "20-bar return in its lowest quantile", same frequency as
    # the decline setup and threshold fitted on the same first 50% -> a setup
    # expressible with ONE number, no image
    ret20 = kc["close"] / kc["close"].shift(W_SETUP - 1) - 1
    fit = np.arange(len(kc)) < len(kc) // 2
    decline = (kc.loc[fit].groupby("setup")["close_ret_24"].median().idxmin()
               if "close_ret_24" in kc else None)
    freq = (kc.loc[fit, "setup"] == decline).mean()
    thr = ret20[fit].quantile(freq)
    kc["nsetup"] = np.where(ret20 <= thr, decline, "none")
    kc.loc[ret20.isna(), "nsetup"] = None

    labels = sorted(kc["setup"].dropna().unique())
    manual, meta = [], []
    for i in range(len(parents_df)):
        manual.append(CustomEvent(f"ev_{i:02d} == 1", name=f"P{i:02d}"))
        meta.append(dict(parent=i, kind="parent", setup=None))
        for col, kind, labs in (("setup", "real", labels), ("fsetup", "fake", labels),
                                ("nsetup", "numeric", [decline])):
            for lab in labs:
                manual.append(CustomEvent(f"ev_{i:02d} == 1 and {col} == '{lab}'",
                                          name=f"P{i:02d}&{kind}{lab}"))
                meta.append(dict(parent=i, kind=kind, setup=lab))

    # 3. evaluate every hypothesis in one session (same null, same walk-forward)
    res = forge(kc, ticker=asset, timeframe="1H", manual_events=manual,
                alpha_config=a, rule_discovery_config=r, two_pass_composition=False)
    m2 = res.summary().drop_duplicates("expression").set_index("expression")
    M2_COLS = ["direction", "holding_period_h", "lift", "mean_advantage", "win_rate", "base_rate",
               "oos_passed", "oos_lift", "oos_p_value", "grade", "composite_score"]
    by_name = {}
    cand_by_id = {c.event_id: c for c in res.candidates}
    for c in res.contracts:
        by_name.setdefault(cand_by_id[c.event_candidate_id].expression, {})["contract"] = c
    for c, resp in res.rule_responses:
        by_name.setdefault(cand_by_id[c.event_candidate_id].expression, {})["resp"] = resp

    rows = []
    for ev, m in zip(manual, meta):
        cand_expr = next((cd.expression for cd in res.candidates if cd.expression in (ev.formula, ev.name)), ev.formula)
        info = by_name.get(cand_expr, {})
        c, resp = info.get("contract"), info.get("resp")
        row = dict(asset=asset, name=ev.name, formula=ev.formula, **m,
                   parent_expr=parents_df["expression"].iloc[m["parent"]],
                   parent_baseline_verdict=parents_df["baseline_verdict"].iloc[m["parent"]],
                   n_active=int(ev.apply(kc).sum()),
                   m2_promoted=bool(c is not None and c in (res.promoted or [])),
                   **{f"m2_{col}": (m2.at[cand_expr, col] if cand_expr in m2.index else None)
                      for col in M2_COLS})
        row.update(stats(resp) if resp is not None else dict(verdict="not-in-M3"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{asset}_k{k}_hypotheses.csv", index=False)
    print(f"[{asset}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(df.groupby("kind")["verdict"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("assets", nargs="+")
    args = ap.parse_args()
    for a_ in args.assets:
        main(a_, args.k)
