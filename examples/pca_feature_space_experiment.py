"""
ForgeEdge — Esperimento: trasformare lo spazio delle feature di M1 con PCA
===========================================================================
Domanda: ricodificare le feature che Event Discovery (M1) usa per generare
eventi — sostituendo gli indicatori originali con le loro componenti
principali — porta un vantaggio sui risultati finali della pipeline?

Confronta due iterazioni sullo STESSO dataset 1D e con gli STESSI parametri
di forge() (preset "balanced", stesso DiscoveryConfig/AlphaConfig/
RuleDiscoveryConfig in entrambe le run):

    Iterazione 1 (baseline):  KPI -> forge()
    Iterazione 2 (PCA):       KPI -> PCA(feature) -> forge()

Nell'iterazione 2 tutte le colonne feature della KPI Table (tutto tranne
OHLCV/open_dt/color, cioè indicatori, geometria candela e lag) vengono
sostituite da N componenti principali (SVD su dati standardizzati). Le
colonne OHLCV e il timestamp restano invariate: servono a Market Context,
al backtest e a build_features stesso, non sono "feature di M1" in questo
senso.

Nota metodologica: la PCA qui è fittata sull'intera storia (fit globale, non
per-finestra/rolling), quindi introduce una lieve fuga di informazione
rispetto agli split IS/OOS interni della pipeline — un vantaggio *a favore*
dell'iterazione PCA, non contro. Il confronto è quindi conservativo nella
direzione "la PCA aiuta".

Dataset: examples/data/AMZN_1D.csv (quotazioni giornaliere AMZN, formato
investing.com). Sostituibile con qualunque altro CSV OHLCV 1D in
examples/data/.

Esecuzione:
    python examples/pca_feature_space_experiment.py [csv_path] [n_components]
"""
from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from forgedge import (
    build_features,
    candle_features,
    lag_features,
    summary_report,
    forge,
    forge_preset,
)

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "examples/data/AMZN_1D.csv"
N_COMPONENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TICKER = "AMZN"

# Colonne "identità" che NON sono feature di M1: restano invariate nella
# versione PCA (servono a Market Context, backtest, build_features).
BASE_COLS = {"open_time", "open_dt", "open", "high", "low", "close", "volume", "color"}


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Carica un CSV OHLCV in formato investing.com (Date, Price, Open, High,
    Low, Vol. con suffisso K/M/B, Change %) e lo normalizza a
    open_time/open/high/low/close/volume, ordinato per data crescente."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"price": "close", "vol.": "volume"})

    def parse_vol(v):
        v = str(v).strip().upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(v[-1:], 1.0)
        num = v[:-1] if v[-1:] in ("K", "M", "B") else v
        try:
            return float(num.replace(",", "")) * mult
        except ValueError:
            return np.nan

    df["volume"] = df["volume"].apply(parse_vol)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).astype(float)
    df["open_time"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def pca_transform(kpi: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Sostituisce le colonne feature della KPI Table con le loro componenti
    principali (SVD su dati standardizzati, fit sulle righe complete).

    Le righe con almeno un NaN in una feature (inizio serie, per via delle
    finestre rolling più lunghe) restano NaN nelle componenti — stesso
    comportamento degli indicatori originali, non un artefatto della PCA.
    """
    feature_cols = [c for c in kpi.columns if c not in BASE_COLS and pd.api.types.is_numeric_dtype(kpi[c])]

    X = kpi[feature_cols].to_numpy(dtype=float)
    complete = ~np.isnan(X).any(axis=1)
    print(f"  Feature columns: {len(feature_cols)}  |  righe complete: {int(complete.sum())}/{len(kpi)}")

    mean = X[complete].mean(axis=0)
    std = X[complete].std(axis=0)
    std[std == 0] = 1.0
    Xz = (X[complete] - mean) / std

    _, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    n_components = min(n_components, Vt.shape[0])
    components = Vt[:n_components]

    var_ratio = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(var_ratio)[:n_components]
    print(f"  Varianza spiegata da {n_components} componenti: {cum_var[-1]:.1%}  "
          f"(prime 3: {var_ratio[0]:.1%}, {var_ratio[1]:.1%}, {var_ratio[2]:.1%})")

    scores = np.full((len(kpi), n_components), np.nan)
    scores[complete] = Xz @ components.T

    out = kpi[[c for c in kpi.columns if c in BASE_COLS]].copy()
    for i in range(n_components):
        out[f"pc_{i + 1:02d}"] = scores[:, i]
    return out, var_ratio[:n_components]


def run_iteration(label: str, kpi: pd.DataFrame, disc_cfg, alpha_cfg, rd_cfg, ticker: str):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    t0 = time.time()
    try:
        result = forge(
            kpi,
            ticker=ticker,
            timeframe="1D",
            event_discovery_config=disc_cfg,
            alpha_config=alpha_cfg,
            rule_discovery_config=rd_cfg,
            progress=False,
        )
    except ValueError as exc:
        print(f"  forge() ha sollevato ValueError (config incoerente): {exc}")
        return None
    dt = time.time() - t0

    summ = result.summary()
    verdicts = summ["rule_verdict"].value_counts(dropna=False).to_dict() if "rule_verdict" in summ.columns else {}
    n_edges = len(result.edges())

    print(f"  tempo            : {dt:.1f}s")
    print(f"  M1 candidati     : {len(result.candidates)}")
    print(f"  M2 contratti     : {len(result.contracts)}  (promossi: {len(result.promoted)})")
    print(f"  M3 verdicts      : {verdicts}")
    print(f"  EDGE/PARTIAL-EDGE (result.edges()): {n_edges}")
    if result.event_discovery.event_distribution_report:
        print("  --- event_distribution_report (M1) ---")
        print("  " + result.event_discovery.event_distribution_report.replace("\n", "\n  "))

    return {
        "label": label, "n_cand": len(result.candidates), "n_contracts": len(result.contracts),
        "n_promoted": len(result.promoted), "verdicts": verdicts, "n_edges": n_edges, "time_s": dt,
    }


def main():
    candles = load_ohlcv_csv(DATA_PATH)
    print(f"Candele {TICKER} 1D: {candles.shape}  "
          f"{candles['open_time'].iloc[0].date()} -> {candles['open_time'].iloc[-1].date()}")

    kpi_baseline = build_features(candles, timestamp_col="open_time")
    kpi_baseline = candle_features(kpi_baseline)
    kpi_baseline = lag_features(kpi_baseline, "close", like="_ema_", periods=[1, 2, 3])
    print(f"KPI table (baseline): {kpi_baseline.shape}")

    rep = summary_report(kpi_baseline, timeframe="1D", return_report=True, verbose=False)
    print(f"summary_report: critical={rep.has_critical} warnings={rep.has_warnings}")

    print("\nCostruzione KPI table PCA...")
    kpi_pca, _ = pca_transform(kpi_baseline, N_COMPONENTS)
    print(f"KPI table (PCA): {kpi_pca.shape}  colonne={list(kpi_pca.columns)}")

    # STESSI parametri in entrambe le iterazioni: un solo forge_preset(),
    # riusato tal quale per la run baseline e per la run PCA.
    disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset=TICKER)

    res1 = run_iteration("ITERAZIONE 1 — baseline: KPI -> forge()", kpi_baseline,
                          disc_cfg, alpha_cfg, rd_cfg, ticker=TICKER)
    res2 = run_iteration("ITERAZIONE 2 — KPI -> PCA -> forge() (stessi parametri)", kpi_pca,
                          disc_cfg, alpha_cfg, rd_cfg, ticker=TICKER)

    print(f"\n{'=' * 70}\nCONFRONTO\n{'=' * 70}")
    for res in (res1, res2):
        if res is None:
            continue
        print(f"{res['label']:<55} cand={res['n_cand']:>5}  contratti={res['n_contracts']:>4}  "
              f"promossi={res['n_promoted']:>3}  edges={res['n_edges']:>3}  verdicts={res['verdicts']}")


if __name__ == "__main__":
    main()
