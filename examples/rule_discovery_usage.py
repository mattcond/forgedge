"""
ForgeEdge — Rule Discovery (Modulo 3) end-to-end
================================================
Dataset: ADAUSDC 1h

Catena completa della pipeline — strettamente sequenziale:

    Market Context → Event Discovery → Alpha Discovery → Rule Discovery

Alpha Discovery produce gli Alpha Contract (pattern con evidenza statistica e
target derivato). Rule Discovery riceve ogni contratto **e il suo Event
Candidate**, ricostruisce la feature/segnale dell'evento con i parametri
salvati sul candidato (stessa attivazione bit-per-bit di Event/Alpha Discovery),
parametrizza la meccanica di un ordine limite realistico (fee, fill rate, target
discreto) e risponde alla domanda operativa: il pattern è sfruttabile?

La validazione è onesta: oltre allo screening in-sample, la strategia viene
testata **out-of-sample con un walk-forward** — i parametri operativi vengono
ri-selezionati su ogni finestra di train e valutati una sola volta sulla
finestra di test successiva, mai vista. Il verdetto finale
(EDGE / PARTIAL-EDGE / NON-EDGE) tiene conto della tenuta OOS, della
significatività statistica (t-test, Deflated Sharpe), della stabilità temporale
e della dipendenza dal regime.

Prerequisiti
------------
    pip install forgedge openpyxl

Esecuzione
----------
    python examples/rule_discovery_usage.py path/to/test1h.xlsx
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
    RuleDiscovery,
    RuleDiscoveryConfig,
)
from forgedge.rule_discovery import (
    BacktestParams,
    GridSpec,
    ScoringParams,
    SelectionCriteria,
    RuleWalkForwardConfig,
    text_report,
)


# ---------------------------------------------------------------------------
# 1. Caricamento e preprocessing
# ---------------------------------------------------------------------------

EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "test1h.xlsx"
SYMBOL = "ADAUSDC"

df = pd.read_excel(EXCEL_PATH)
df = df[df["symbol"] == SYMBOL].copy().sort_values("open_time")
df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms")
df = df.drop(columns=["symbol", "timeframe", "open_time"])
df = df.dropna().reset_index(drop=True)

print(f"Rows : {len(df)}")
print(f"Range: {df['open_dt'].iloc[0].date()} → {df['open_dt'].iloc[-1].date()}")


# ---------------------------------------------------------------------------
# 2-4. Moduli 0-2 — Market Context, Event Discovery, Alpha Discovery
# ---------------------------------------------------------------------------

enriched = MarketContext(df.copy()).run()
ed = EventDiscovery(enriched.copy(), DiscoveryConfig(timestamp_col="open_dt"))
candidates = ed.run()
print(f"Event Candidates : {len(candidates)}")

ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset=SYMBOL, timeframe="1H"))
ad.run()
promoted = ad.promoted_contracts()
print(f"Alpha promossi   : {len(promoted)}")

by_id = {c.event_id: c for c in candidates}


# ---------------------------------------------------------------------------
# 5. Modulo 3 — Rule Discovery
# ---------------------------------------------------------------------------

# La configurazione esplora solo i parametri operativi (la regola è fissa).
# Il target del contratto (sell_pct, target_h) viene usato come centro del grid.
config = RuleDiscoveryConfig(
    base_params=BacktestParams(
        buy_type="limit",
        buy_drop_pct=0.010,
        buy_delay_bar=6,
        fee=0.002,
        early_stopping=True,
    ),
    grid=GridSpec(
        buy_drop_pct=[0.006, 0.010, 0.014],
        sell_pct=[0.030, 0.040, 0.050],
        target_h=[12, 24, 48],
    ),
    scoring=ScoringParams(pf_min_trades=15, pf_min_tpm=2),
    walk_forward=RuleWalkForwardConfig(n_splits=4, min_train_months=6),
    criteria=SelectionCriteria(min_profit_factor=2.0, min_win_rate=0.55),
)

verdicts = {"EDGE": 0, "PARTIAL-EDGE": 0, "NON-EDGE": 0}
edges = []

for contract in promoted:
    rd = RuleDiscovery(ed.df, contract, by_id[contract.event_candidate_id], config)
    resp = rd.run()
    verdicts[resp.verdict] += 1
    if resp.is_edge:
        edges.append(resp)

print("\n─── Verdetti Rule Discovery ───")
for v, n in verdicts.items():
    print(f"  {v:<13}: {n}")


# ---------------------------------------------------------------------------
# 6. Dettaglio delle regole validate
# ---------------------------------------------------------------------------

edges.sort(key=lambda r: r.walk_forward.oos_summary.profit_factor
           if r.walk_forward else 0.0, reverse=True)

for resp in edges[:5]:
    print("\n" + text_report(resp))

# Il report mostra anche, per ogni regola validata:
#   · EXECUTION ENVELOPE — bracket PF/WR/expectancy tra convenzione conservativa
#     (close, riproduce il backtest certificato) e ottimistica (high, touch
#     intrabar). La performance reale sta tra le due, senza scegliere il target.
#   · MAE / MFE — escursione avversa/favorevole intra-trade: il "range d'azione"
#     della regola, indipendente dalla regola di uscita.
