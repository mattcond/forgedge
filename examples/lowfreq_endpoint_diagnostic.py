"""Economic-endpoint null test: do EDGE/PARTIAL verdicts separate ADA from noise?

Runs the full chain (MarketContext->EventDiscovery->AlphaDiscovery->RuleDiscovery)
on real ADA 2024 and on phase-randomized surrogates, counting how many of the
top-N promoted alphas earn a tradeable verdict. If noise yields as many EDGEs as
real data, the pipeline's final output is not trustworthy at 1D frequency.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd

from forgedge import (AlphaConfig, AlphaDiscovery, DiscoveryConfig, EventDiscovery,
                      MarketContext, RuleDiscovery, RuleDiscoveryConfig)
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.event_discovery.models import GateParams
from forgedge.rule_discovery.models import (BacktestParams, GridSpec, ScoringParams,
                                            SelectionCriteria, WalkForwardConfig)

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowfreq_null_diagnostic import build_features, make_surrogate_ohlc, RAW

TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 12

RULE_CFG = RuleDiscoveryConfig(
    base_params=BacktestParams(direction="long", buy_type="limit", buy_drop_pct=0.010,
                               buy_delay_bar=3, sell_pct=0.040, target_h=5,
                               target_hit_col="high", fee=0.001, early_stopping=True),
    grid=GridSpec(buy_drop_pct=[0.005, 0.010, 0.015], sell_pct=[0.025, 0.04, 0.06],
                  target_h=[3, 5, 7]),
    scoring=ScoringParams(pf_min_trades=8, pf_min_tpm=1, pf_tpm_target=2),
    walk_forward=WalkForwardConfig(n_splits=3, min_train_months=4),
    criteria=SelectionCriteria(min_profit_factor=1.5, min_win_rate=0.50, min_tpm=1.0,
                               min_fill_rate=0.30, partial_min_profit_factor=1.2,
                               min_active_month_rate=0.60, early_elimination=False),
)


def run_chain(feat_df: pd.DataFrame, label: str) -> dict:
    df_is = feat_df[feat_df["open_dt"] < "2025-01-01"].copy().reset_index(drop=True)
    enriched = MarketContext(df_is.copy()).run()
    ed = EventDiscovery(enriched.copy(), DiscoveryConfig(
        gate_params=GateParams(min_tpm=1.5, max_dispersion=3.0),
        timestamp_col="open_dt", max_and_components=2, train_ratio=1.0))
    cands = ed.run()
    ad = AlphaDiscovery(ed.df, cands, AlphaConfig(
        horizon_grid=(1, 2, 3, 5, 7, 10), train_ratio=0.70, asset=label, timeframe="1D",
        thresholds=PromotionThresholds(min_lift=0.06, min_cohens_d=0.10,
                                       use_fdr=True, fdr_q=0.15, oos_max_p=0.15)))
    ad.run()
    promoted = sorted(ad.promoted_contracts(),
                      key=lambda c: c.alpha_score.composite_score, reverse=True)[:TOPN]
    by_id = {c.event_id: c for c in cands}
    verdicts = {"EDGE": 0, "PARTIAL-EDGE": 0, "NON-EDGE": 0}
    for ct in promoted:
        cand = by_id.get(ct.event_candidate_id)
        if cand is None:
            continue
        resp = RuleDiscovery(ed.df, ct, cand, RULE_CFG).run()
        verdicts[resp.verdict] += 1
    verdicts["tradeable"] = verdicts["EDGE"] + verdicts["PARTIAL-EDGE"]
    verdicts["n_promoted_tested"] = len(promoted)
    return verdicts


if __name__ == "__main__":
    n_sur = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    real = run_chain(build_features(RAW), "REAL")
    print(f"REAL    : {real}   ({time.time()-t0:.0f}s)")
    trades = []
    for s in range(n_sur):
        t0 = time.time()
        rng = np.random.default_rng(2000 + s)
        st = run_chain(build_features(make_surrogate_ohlc(rng)), f"SURR{s}")
        trades.append(st["tradeable"])
        print(f"SURR[{s}]: {st}   ({time.time()-t0:.0f}s)")
    tr = np.array(trades)
    print("\n" + "=" * 60)
    print(f"TRADEABLE (EDGE+PARTIAL) of top-{TOPN} promoted:")
    print(f"  REAL              : {real['tradeable']}")
    print(f"  SURROGATE mean±sd : {tr.mean():.1f} ± {tr.std():.1f}  (range {tr.min()}–{tr.max()})")
