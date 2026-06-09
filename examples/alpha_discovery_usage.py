"""
ForgeEdge — Alpha Discovery (Modulo 2) end-to-end
==================================================
Dataset: ADAUSDC 1h

Catena completa della pipeline:

    Market Context  →  Event Discovery  →  Alpha Discovery

Market Context etichetta ogni barra per regime; Event Discovery genera gli
Event Candidate (senza guardare il forward return); Alpha Discovery misura il
potere predittivo di ciascun candidato rispetto a un target economico e
promuove quelli con evidenza statistica sufficiente in Alpha Contract.

Prerequisiti
------------
    pip install forgedge openpyxl

Esecuzione
----------
    python examples/alpha_discovery_usage.py path/to/test1h.xlsx
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "src")
from forgedge import (
    AlphaConfig,
    AlphaDiscovery,
    DiscoveryConfig,
    EventDiscovery,
    MarketContext,
    TargetDefinition,
)
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.event_discovery.models import GateParams


# ---------------------------------------------------------------------------
# 1. Caricamento e preprocessing
# ---------------------------------------------------------------------------

EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "test1h.xlsx"
SYMBOL = "ADAUSDC"

df = pd.read_excel(EXCEL_PATH)
df = df[df["symbol"] == SYMBOL].copy().sort_values("open_time")
df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms")
df = df.drop(columns=["symbol", "timeframe", "open_time"])

print(f"Rows : {len(df)}")
print(f"Range: {df['open_dt'].iloc[0].date()} → {df['open_dt'].iloc[-1].date()}")


# ---------------------------------------------------------------------------
# 2. Modulo 0 — Market Context (aggiunge la colonna 'regime')
# ---------------------------------------------------------------------------

mc = MarketContext(df.copy())
enriched = mc.run()
print("\n─── Distribuzione dei regimi ───")
print(mc.distribution())


# ---------------------------------------------------------------------------
# 3. Modulo 1 — Event Discovery (Event Candidates)
# ---------------------------------------------------------------------------

ed = EventDiscovery(
    enriched.copy(),
    DiscoveryConfig(
        gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
        timestamp_col="open_dt",
    ),
)
candidates = ed.run()
print(f"\nEvent Candidates: {len(candidates)}")


# ---------------------------------------------------------------------------
# 4. Modulo 2 — Alpha Discovery
# ---------------------------------------------------------------------------

config = AlphaConfig(
    target=TargetDefinition(
        holding_period_h=24,    # orizzonte di 24 barre
        sell_pct=0.04,          # target binario +4%
        direction="long",
        asset=SYMBOL,
        exchange="binance_spot",
        timeframe="1H",
    ),
    thresholds=PromotionThresholds(
        min_lift=0.08,          # almeno +8pp sopra il base rate
        min_cohens_d=0.15,      # effect size minimo
        min_activations=30,
        use_fdr=True,           # controllo del False Discovery Rate (Benjamini-Hochberg)
        fdr_q=0.10,
    ),
)

ad = AlphaDiscovery(enriched.copy(), candidates, config)
contracts = ad.run()

ms = ad.market_structure
print("\n─── Struttura di mercato (Step 2) ───")
print(f"Hurst       : {ms.hurst:.3f}  ({ms.hurst_interpretation})")
print(f"Famiglia    : {ms.expected_family}")
print(f"ACF return  : " + "  ".join(f"lag{l}={v:+.3f}" for l, v in ms.autocorr.items()))
print(f"\nBase rate   : {ad.base_rate:.1%}")

promoted = ad.promoted_contracts()
print(f"\nContratti valutati : {len(contracts)}")
print(f"Promossi (HYPOTHESIS): {len(promoted)}")


# ---------------------------------------------------------------------------
# 5. Top alpha promossi
# ---------------------------------------------------------------------------

summary = ad.summary()
cols = [
    "expression", "feature", "ic", "n_activations", "win_rate", "lift",
    "cohens_d", "p_value", "regime_dependency", "composite_score", "grade",
]
print("\n─── Top 10 Alpha promossi (per composite score) ───")
print(summary[summary["promoted"]].head(10)[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Ispezione del miglior Alpha Contract
# ---------------------------------------------------------------------------

if promoted:
    best = max(promoted, key=lambda c: c.alpha_score.composite_score)
    print("\n─── Miglior Alpha Contract ───")
    print(f"alpha_id          : {best.alpha_id}")
    print(f"event_candidate_id: {best.event_candidate_id}")
    print(f"expression        : {best.event_expression}")
    print(f"IC                : {best.underlying_feature.ic:+.4f} "
          f"(p={best.underlying_feature.p_value:.2e}, "
          f"stable={best.underlying_feature.rolling_ic_stable})")
    es = best.event_stats
    print(f"win_rate / base   : {es.win_rate:.1%} / {es.base_rate:.1%}  "
          f"(lift {es.lift:+.1%})")
    print(f"cohens_d / t / p  : {es.cohens_d:.3f} / {es.t_stat:.2f} / {es.p_value:.2e}")
    print(f"regime dependency : {best.regime_analysis.dependency_type}  "
          f"(breadth {best.regime_analysis.regime_breadth:.2f})")
    print(f"grade             : {best.alpha_score.grade} "
          f"({best.alpha_score.composite_score:.2f})")

    print("\n─── Regime sensitivity ───")
    for r in best.regime_analysis.per_regime:
        print(f"  {r.regime:<12} n={r.n:>5}  IC={r.ic:+.4f}  "
              f"p={r.p_value:.4f}  WR={r.win_rate:.1%}  [{r.strength}]")
