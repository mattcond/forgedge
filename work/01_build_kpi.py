"""Step 1 — build daily KPI Tables from examples/data/*_1DAY.csv (+ AMZN_1D.csv).

One parquet per ticker in work/kpi/.  Daily-calibrated indicator periods
(the packaged default_enricher.yaml is hourly-calibrated: 96/168 bars means
4h/7d on 1H but 4.5/8 months on 1D).
"""
from __future__ import annotations
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from forgedge import build_features, candle_features, lag_features, summary_report

ROOT = Path("/home/user/forgedge")
OUT = ROOT / "work" / "kpi"; OUT.mkdir(parents=True, exist_ok=True)

# Daily calibration: 3d..200d instead of the packaged 3..168 *hourly* grid.
CFG = {
    "moving_average": {"enabled": True, "params": {"periods": [3, 5, 10, 20, 50, 100, 200],
                                                   "columns": ["close", "high", "low", "volume"]}},
    "ema":            {"enabled": True, "params": {"periods": [3, 5, 10, 20, 50, 100, 200],
                                                   "columns": ["close", "high", "low"]}},
    "volatility":     {"enabled": True, "params": {"periods": [5, 10, 20, 60], "columns": ["close"]}},
    "min":            {"enabled": True, "params": {"periods": [3, 5, 10, 20, 50, 100], "columns": ["close", "high", "low"]}},
    "max":            {"enabled": True, "params": {"periods": [3, 5, 10, 20, 50, 100], "columns": ["close", "high", "low"]}},
    "return":         {"enabled": True, "params": {"periods": [1, 2, 3, 5, 10, 20, 60], "columns": ["close"]}},
    "rsi":            {"enabled": True, "params": {"periods": [14, 25], "columns": ["close"]}},
    "bollinger_bands":{"enabled": True, "params": {"periods": [20], "columns": ["close"]}},
    "max_drawdown":   {"enabled": True, "params": {"periods": [10, 20, 60, 120], "columns": ["close"]}},
    "atr":            {"enabled": True, "params": {"periods": [14, 28], "columns": ["close"]}},
}


def load_semicolon(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")


def load_amzn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').lstrip("﻿") for c in df.columns]
    df = df.rename(columns={"Date": "timestamp", "Price": "close", "Open": "open",
                            "High": "high", "Low": "low", "Vol.": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    def _vol(v):
        s = str(v).strip().replace(",", "")
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1:], 1.0)
        try:
            return float(s[:-1]) * mult if mult != 1.0 else float(s)
        except ValueError:
            return np.nan
    df["volume"] = df["volume"].map(_vol)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")


SOURCES = {
    "BTCEUR":     (ROOT / "examples/data/BTCEUR_1DAY.csv", load_semicolon),
    "ETHEUR":     (ROOT / "examples/data/ETHEUR_1DAY.csv", load_semicolon),
    "EURUSD":     (ROOT / "examples/data/EURUSD_1DAY.csv", load_semicolon),
    "GBPUSD":     (ROOT / "examples/data/GBPUSD_1DAY.csv", load_semicolon),
    "COPPER":     (ROOT / "examples/data/COPPER.CMDUSD_1DAY.csv", load_semicolon),
    "BRENT":      (ROOT / "examples/data/E_Brent_1DAY.csv", load_semicolon),
    "DAX":        (ROOT / "examples/data/E_DAAX_1DAY.csv", load_semicolon),
    "SP500":      (ROOT / "examples/data/E_SandP-500_1DAY.csv", load_semicolon),
    "AMZN":       (ROOT / "examples/data/AMZN_1D.csv", load_amzn),
}

rows = []
for tk, (path, loader) in SOURCES.items():
    raw = loader(path).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpi = build_features(raw, config=CFG, timestamp_col="timestamp", timestamp_unit=None)
        kpi = candle_features(kpi)
        kpi = lag_features(kpi, "close", "high", "low", "volume", periods=[1, 2, 3, 5])
    kpi = kpi.sort_values("open_dt").reset_index(drop=True)
    kpi.to_parquet(OUT / f"{tk}.parquet")
    rep = summary_report(kpi, timestamp_col="open_dt", timeframe="1D", verbose=False, return_report=True)
    rows.append(dict(ticker=tk, bars=len(kpi), cols=kpi.shape[1],
                     start=str(kpi["open_dt"].iloc[0].date()), end=str(kpi["open_dt"].iloc[-1].date()),
                     dq=rep.worst, dq_note=rep.one_line()[:90]))

print(pd.DataFrame(rows).to_string(index=False))
