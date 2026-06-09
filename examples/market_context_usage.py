"""
ForgeEdge — Market Context Module (Modulo 0)
============================================
Dataset: ADAUSDC / DOGEUSDC 1h

Mostra come arricchire la KPI Table con la colonna di regime prima di
Event Discovery, e come usare la tooling Hurst/OU per scegliere le finestre
EMA fast/slow.

Prerequisiti
------------
    pip install forgedge openpyxl

Esecuzione
----------
    python examples/market_context_usage.py path/to/test1h.xlsx [SYMBOL]
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "src")
from forgedge import MarketContext
from forgedge.market_context.hurst import suggest_ema_windows


EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "test1h.xlsx"
SYMBOL = sys.argv[2] if len(sys.argv) > 2 else "ADAUSDC"


# ---------------------------------------------------------------------------
# 1. Caricamento — la KPI Table dell'utente contiene solo OHLCV (+ alcuni KPI).
#    Nessuna colonna EMA: il modulo le calcola inline a partire dal close.
# ---------------------------------------------------------------------------

raw = pd.read_excel(EXCEL_PATH)
d = raw[raw["symbol"] == SYMBOL].sort_values("open_time").copy()
d["open_dt"] = pd.to_datetime(d["open_time"], unit="ms")
d = d[d["close"] > 0].reset_index(drop=True)

kpi = d[["open_dt", "open", "high", "low", "close", "volume"]].copy()
print(f"Symbol : {SYMBOL}")
print(f"Rows   : {len(kpi)}")
print(f"Range  : {kpi['open_dt'].iloc[0].date()} → {kpi['open_dt'].iloc[-1].date()}")


# ---------------------------------------------------------------------------
# 2. Anteprima (facoltativa) della scelta finestre via Hurst tooling.
#    MarketContext effettua questa stessa derivazione automaticamente in run();
#    qui la mostriamo solo per ispezione.
# ---------------------------------------------------------------------------

sug = suggest_ema_windows(kpi.set_index("open_dt")["close"], timeframe="1h")
print("\n─── EMA window analysis (local OU half-life) ───")
print(f"  Hurst median        : {sug['hurst_median']}  (<0.5 → mean reverting)")
print(f"  Half-life           : {sug['half_life_hours']}h")
print(f"  Suggested short/long: {sug['suggested_short_period']} / {sug['suggested_long_period']}")


# ---------------------------------------------------------------------------
# 3. Esecuzione del Market Context Module.
#    auto_window=True (default): le finestre EMA vengono decise dai dati;
#    i 9/25 restano solo come fallback se l'half-life OU non converge.
# ---------------------------------------------------------------------------

mc = MarketContext(kpi)  # config di default → auto_window attivo
enriched = mc.run()

print("\n─── Window resolution (deciso da MarketContext) ───")
for k, v in mc.window_resolution.items():
    print(f"  {k}: {v}")

new_cols = set(enriched.columns) - set(kpi.columns)
print(f"\nColonne aggiunte: {sorted(new_cols)}")
print(f"Nessuna colonna EMA intermedia trapelata: "
      f"{'close_ema_09' not in enriched.columns}")


# ---------------------------------------------------------------------------
# 4. Distribuzione del regime
# ---------------------------------------------------------------------------

print("\n─── Regime distribution ───")
print(mc.distribution().to_string())
print(f"\nBarre stabili (regime invariato ≥ {mc.config.stable_window} barre): "
      f"{enriched['regime_stable'].mean():.1%}")


# ---------------------------------------------------------------------------
# 5. Tracciabilità della configurazione (va nel report di sessione)
# ---------------------------------------------------------------------------

print("\n─── Config (traceability) ───")
for k, v in mc.get_config().items():
    print(f"  {k}: {v}")
