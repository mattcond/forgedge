"""
ForgeEdge — KPI Builder (Modulo -1): da candele OHLCV alla KPI Table per forge()
================================================================================
Dataset: sintetico (random walk) se non passi un file.

Mostra il flusso consigliato, ESTERNO a forge: a partire dalle sole candele
l'utente costruisce le feature, le geometrie della candela e i lag, poi passa
la tabella a forge() (oppure passa la propria tabella — forge non cambia).

    build_features(candles, timestamp_col=...)   # indicatori base (+ open_dt)
        -> candle_features(df)                    # geometria candela (continua)
        -> lag_features(df, *cols)                # lag di colonne reali
        -> forge(df, ...)

In coda, opzionale e NON consigliato, pattern_features() per i pattern
candlestick nominali come singola colonna categorica.

Prerequisiti
------------
    pip install forgedge            # + openpyxl/pyarrow se carichi xlsx/parquet

Esecuzione
----------
    python examples/kpi_builder_usage.py                       # dati sintetici
    python examples/kpi_builder_usage.py candles.parquet ts    # tuoi dati + col. timestamp
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from forgedge import (
    build_features,
    candle_features,
    lag_features,
    pattern_features,
    summary_report,
    forge,
    forge_preset,
)


# ---------------------------------------------------------------------------
# 0. Candele grezze: OHLC(V) + una colonna timestamp.
#    Passa un file (xlsx/parquet/csv) come 1° argomento e il nome della colonna
#    timestamp come 2°; altrimenti si genera un random walk sintetico (1D).
# ---------------------------------------------------------------------------

def load_candles(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def synthetic_candles(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    openp = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, 0.01, n)) * close
    high = np.maximum(openp, close) + span
    low = np.minimum(openp, close) - span
    return pd.DataFrame({
        "open_time": 1_600_000_000_000 + np.arange(n) * 86_400_000,  # 1D in ms
        "open": openp, "high": high, "low": low, "close": close,
        "volume": rng.uniform(10, 100, n),
    })


PATH = sys.argv[1] if len(sys.argv) > 1 else None
TS_COL = sys.argv[2] if len(sys.argv) > 2 else "open_time"
candles = load_candles(PATH) if PATH else synthetic_candles()
print(f"Candele      : {candles.shape}  colonne={list(candles.columns)}")
print(f"Timestamp col: {TS_COL}")


# ---------------------------------------------------------------------------
# 1. (opzionale) Diagnostica qualità dei dati PRIMA di costruire i KPI.
#    Puramente consultiva: non blocca e non solleva.
# ---------------------------------------------------------------------------
print("\n--- summary_report (diagnostica, opzionale) ---")
summary_report(candles, timeframe="1D")


# ---------------------------------------------------------------------------
# 2. Pipeline KPI Builder consigliata (esterna a forge).
#    Una config compatta per un esempio veloce; config=None usa il default completo.
# ---------------------------------------------------------------------------
DEMO_CONFIG = {
    "ema":             {"enabled": True, "params": {"periods": [9, 25, 50], "columns": ["close"]}},
    "rsi":             {"enabled": True, "params": {"periods": [14], "columns": ["close"]}},
    "bollinger_bands": {"enabled": True, "params": {"periods": [20], "columns": ["close"]}},
    "min":             {"enabled": True, "params": {"periods": [24], "columns": ["close"]}},
    "max":             {"enabled": True, "params": {"periods": [24], "columns": ["close"]}},
    "return":          {"enabled": True, "params": {"periods": [1, 6, 24], "columns": ["close"]}},
}

print("\n--- KPI Builder ---")
kpi = build_features(candles, DEMO_CONFIG, timestamp_col=TS_COL)
print(f"build_features  : {kpi.shape}  (indicatori base + open_dt + color)")

kpi = candle_features(kpi)
print(f"candle_features : {kpi.shape}  (+ body, upper_wick, lower_wick, close_pos, range_pct, gap)")

# Lag su colonne REALI: per nome esplicito e/o pattern (niente nomi a memoria).
kpi = lag_features(kpi, "close", "color", like="_ema_", periods=[1, 2, 3])
n_lag = sum("_prev_" in c for c in kpi.columns)
print(f"lag_features    : {kpi.shape}  (+ {n_lag} colonne *_prev_NN)")


# ---------------------------------------------------------------------------
# 3. La KPI Table (build -> candle -> lag) è pronta per forge(). L'utente può
#    passare questa o la propria (forge resta invariato). Fermiamo ad Alpha
#    Discovery per un demo veloce (su dati casuali è normale trovare pochi eventi).
# ---------------------------------------------------------------------------
print(f"\nKPI Table finale: {kpi.shape}  | open_dt dtype = {kpi['open_dt'].dtype}")

disc, alpha, rd = forge_preset("balanced", timeframe="1D", asset="DEMO")
result = forge(
    kpi,
    ticker="DEMO",
    timeframe="1D",
    event_discovery_config=disc,
    alpha_config=alpha,
    run_rule_discovery=False,   # togli per la pipeline completa (M3 + M4)
    progress=False,
)
print(f"forge(kpi)      : M1 candidati = {len(result.candidates)}  "
      f"M2 promossi = {len(result.promoted)}")


# ---------------------------------------------------------------------------
# 4. (opzionale, NON consigliato) pattern candlestick nominali → categorica.
#    pattern_features aggiunge una sola colonna 'candle_pattern'
#    ("HAMMER"/"DOJI"/.../None) che FORGE classifica categorica.
#
#    'candle_pattern' è scorabile end-to-end da forge(): Alpha Discovery misura
#    un IC point-biserial sull'one-hot binarizzato. Resta opt-in per QUALITÀ,
#    non per limiti tecnici: i pattern nominali usano soglie fisse, quindi per
#    la discovery automatica si preferisce candle_features() (continua).
# ---------------------------------------------------------------------------
kpi_pat = pattern_features(kpi)
print("\npattern_features (opt-in) — colonna categorica 'candle_pattern':")
print(kpi_pat["candle_pattern"].value_counts(dropna=False).to_string())
print("\nOK — da candele grezze a KPI Table consumata da forge().")
