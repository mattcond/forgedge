"""
ForgeEdge — Rule Registry (Modulo 4) end-to-end
===============================================
Dataset: multi-ticker 1h (e.g. ADAUSDC, SOLUSDC, BTCUSDC accodati in un CSV)

Quarto e ultimo modulo della pipeline FORGE. Riceve le regole validate da Rule
Discovery — un pool per ogni ticker presente nel CSV — e le raccoglie in un
registro in-memory per la sessione corrente:

    Step 1  Ingestion              un documento per regola EDGE / PARTIAL-EDGE
    Step 2  Matrici correlazione   Jaccard (attivazioni) + Spearman (gain)
    Step 3  Deduplicazione         marca (non elimina) i duplicati
    Step 4  Cross-ticker backtest  ogni regola testata su tutti gli altri
                                    ticker, con le soglie ricalcolate sulla
                                    distribuzione locale
    Step 5  Export                 tabella piatta (CSV/Excel) + report HTML

Il registro è stateless: viene costruito da zero a ogni sessione. La tabella
piatta esportata è l'unico artefatto di persistenza.

Prerequisiti
------------
    pip install forgedge openpyxl     # openpyxl solo per l'export Excel

Esecuzione
----------
    python examples/rule_registry_usage.py path/to/multi_ticker_1h.xlsx
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "src")
from forgedge import forge
from forgedge.event_discovery import DiscoveryConfig
from forgedge.rule_registry import RegistryConfig, RuleRegistry


# ---------------------------------------------------------------------------
# 1. Caricamento — un CSV/Excel multi-ticker
# ---------------------------------------------------------------------------

EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "test1h.xlsx"

raw = pd.read_excel(EXCEL_PATH)
raw["open_dt"] = pd.to_datetime(raw["open_time"], unit="ms")
tickers = sorted(raw["symbol"].unique())
print(f"Ticker trovati: {tickers}")


# ---------------------------------------------------------------------------
# 2. Pipeline completa, separatamente per ogni ticker
# ---------------------------------------------------------------------------
# La generalizzabilità cross-ticker è una proprietà MISURATA a posteriori dal
# Rule Registry — non costruita a priori mescolando i dati (spec Sezione 2).

ed_cfg = DiscoveryConfig(timestamp_col="open_dt")
results = {}
for ticker in tickers:
    df = raw[raw["symbol"] == ticker].copy().sort_values("open_dt")
    df = df.drop(columns=["symbol", "timeframe", "open_time"]).dropna()
    df = df.reset_index(drop=True)
    results[ticker] = forge(
        df, asset=ticker, timeframe="1H", event_discovery_config=ed_cfg
    )
    print(f"  {ticker:<10}: {len(results[ticker].edges())} regole validate")


# ---------------------------------------------------------------------------
# 3. Modulo 4 — Rule Registry
# ---------------------------------------------------------------------------

registry = RuleRegistry.from_forge_results(
    results,
    RegistryConfig(
        overlap_threshold=0.70,      # Jaccard per la deduplicazione
        # Il PASS cross-ticker ha due metà: un floor assoluto (è tradeable là)
        # e una ritenzione sul PF di casa (trasferisce davvero). Entrambe
        # lasciate a UNSET si risolvono a 1.5 e 0.8 — il floor dall'asticella
        # con cui M3 ha ammesso la regola.
        generic_ratio_threshold=2 / 3,
        export_format="excel",
    ),
).run()

print("\n─── Registro ───")
print(registry.summary().to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Export — tabella piatta + report HTML autocontenuto
# ---------------------------------------------------------------------------

flat_path = registry.export("forge_flat_table.xlsx")
print(f"\nTabella piatta : {flat_path}")

html = registry.html_report(timeframe="1H")
with open("forge_report.html", "w", encoding="utf-8") as fh:
    fh.write(html)
print("Report HTML    : forge_report.html")

# Il report HTML è autocontenuto (SVG inline, nessuna CDN): equity curve per
# regola, heatmap delle correlazioni (Jaccard/Spearman), cross-ticker summary
# con badge GENERIC/PARTIAL/SPECIFIC/ISOLATED, banner sui duplicati e trade log.
