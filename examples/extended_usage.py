"""
ForgeEdge — Esempio esteso di utilizzo
=======================================
Dataset: ADAUSDC 1h

Prerequisiti
------------
    pip install forgedge openpyxl

Esecuzione
----------
    python examples/extended_usage.py path/to/test1h.xlsx
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from forgedge import DiscoveryConfig, EventDiscovery
from forgedge.event_discovery.models import GateParams


# ---------------------------------------------------------------------------
# 1. Caricamento e preprocessing
# ---------------------------------------------------------------------------

EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "test1h.xlsx"
SYMBOL = "ADAUSDC"

df = pd.read_excel(EXCEL_PATH)

# Filtra il simbolo desiderato
df = df[df["symbol"] == SYMBOL].copy()

# I timestamp sono Unix milliseconds (int64): usare unit='ms' è obbligatorio.
# Senza unit='ms' pandas li interpreta come nanosecondi → 1970 → gate fallisce.
df.index = pd.to_datetime(df["open_time"], unit="ms")
df.index.name = "open_dt"
df = df.drop(columns=["open_time"])
df = df.sort_index()

print(f"Rows : {len(df)}")
print(f"Range: {df.index[0].date()} → {df.index[-1].date()}")


# ---------------------------------------------------------------------------
# 2. Configurazione ed esecuzione del pipeline
# ---------------------------------------------------------------------------

config = DiscoveryConfig(
    gate_params=GateParams(
        min_act=50,      # almeno 50 attivazioni totali
        min_months=8,    # attivo in almeno 8 mesi distinti
        max_conc=0.40,   # nessun mese assorbe più del 40% delle attivazioni
        min_tpm=2.0,     # almeno 2 attivazioni/mese in media
    ),
)

ed = EventDiscovery(df, config)
candidates = ed.run()

print(f"\nTotal candidates: {len(candidates)}")


# ---------------------------------------------------------------------------
# 3. Summary DataFrame ordinabile
# ---------------------------------------------------------------------------

# Il pipeline usa un RangeIndex internamente; per resample è necessario
# riallineare l'event_series con i timestamp estratti dal pipeline.
timestamps = ed._timestamps


def monthly_activations(event_series: pd.Series, ts: pd.Series) -> pd.Series:
    """Attivazioni mensili usando i timestamps del pipeline."""
    s = event_series.copy()
    s.index = pd.to_datetime(ts.values)
    return s.resample("ME").sum()


records = []
for c in candidates:
    g = c.consistency_gate   # GateResult: passed, n_activations, n_active_months,
                             #             max_monthly_share, mean_tpm, fail_reason
    s = c.activation_stats   # ActivationStats: + zero_months
    comp = c.components[0]   # primo EventComponent
    records.append({
        "expression":  c.expression,
        "arity":       len(c.components),
        "n_act":       g.n_activations,
        "n_months":    g.n_active_months,
        "zero_months": s.zero_months,
        "max_conc":    round(g.max_monthly_share, 4),
        "tpm":         round(g.mean_tpm, 2),
        "source":      comp.source_feature,
        "transform":   comp.transform,
        "event_type":  comp.event_type,
        "_candidate":  c,
    })

summary = pd.DataFrame(records)
singles = summary[summary["arity"] == 1].sort_values("max_conc")
ands = summary[summary["arity"] > 1].sort_values(
    ["max_conc", "n_act"], ascending=[True, False]
)

print(f"\nSingles: {len(singles)} | ANDs: {len(ands)}")


# ---------------------------------------------------------------------------
# 4. Top 10 single events (minima concentrazione mensile)
# ---------------------------------------------------------------------------

print("\n─── Top 10 single events (lowest max_conc) ───")
cols = ["expression", "n_act", "n_months", "zero_months", "max_conc", "tpm"]
print(singles.head(10)[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Top 10 AND compositions
# ---------------------------------------------------------------------------

print("\n─── Top 10 AND compositions (lowest max_conc) ───")
cols_and = ["expression", "arity", "n_act", "n_months", "max_conc", "tpm"]
print(ands.head(10)[cols_and].to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Ispezione migliore AND candidate
# ---------------------------------------------------------------------------

best_and_row = ands.iloc[0]
best_and = best_and_row["_candidate"]

print(f"\n─── Migliore AND candidate ───")
print(f"Expression : {best_and.expression}")
print(f"Activations: {best_and.consistency_gate.n_activations}")
print(f"Months     : {best_and.consistency_gate.n_active_months}")
print(f"Max conc   : {best_and.consistency_gate.max_monthly_share:.4f}")
print(f"Avg/month  : {best_and.consistency_gate.mean_tpm:.2f}")
print(f"Zero months: {best_and.activation_stats.zero_months}")
print(f"Components :")
for comp in best_and.components:
    print(f"  [{comp.transform}] {comp.expression}")


# ---------------------------------------------------------------------------
# 7. Breakdown mensile del migliore AND
# ---------------------------------------------------------------------------

monthly = monthly_activations(best_and.event_series, timestamps)
monthly.index = monthly.index.strftime("%Y-%m")
print("\nAttivazioni mensili (mesi non-zero):")
print(monthly[monthly > 0].to_string())


# ---------------------------------------------------------------------------
# 8. Classificazioni delle colonne
# ---------------------------------------------------------------------------

classifications = ed.get_classifications()
print("\n─── Column classifications ───")
for col, clf in sorted(classifications.items()):
    sf = (
        f"  scale_free={clf.effective_scale_free}"
        if clf.col_type.name == "CONTINUOUS"
        else ""
    )
    print(f"  {col:<35} {clf.col_type.name}{sf}")


# ---------------------------------------------------------------------------
# 9. Breakdown per transform type
# ---------------------------------------------------------------------------

print("\n─── Singles per transform ───")
print(
    summary[summary["arity"] == 1]
    .groupby("transform")["n_act"]
    .agg(["count", "mean", "median"])
    .round(1)
)


# ---------------------------------------------------------------------------
# 10. Grafico mensile (opzionale, richiede matplotlib)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    best_single_row = singles.iloc[0]
    best_single = best_single_row["_candidate"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Pannello A — migliore single event
    m_single = monthly_activations(best_single.event_series, timestamps)
    m_single.index = m_single.index.strftime("%Y-%m")
    ax = axes[0]
    m_single.plot(kind="bar", ax=ax, color="steelblue", width=0.8)
    ax.set_title(
        f"Single: {best_single_row['expression']}\n"
        f"N={best_single_row['n_act']}  months={best_single_row['n_months']}  "
        f"max_conc={best_single_row['max_conc']}  tpm={best_single_row['tpm']}"
    )
    ax.set_ylabel("Attivazioni")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.axhline(best_single_row["tpm"], color="red", linestyle="--", label="avg/month")
    ax.legend()

    # Pannello B — migliore AND composition
    g = best_and.consistency_gate
    ax2 = axes[1]
    monthly.plot(kind="bar", ax=ax2, color="darkorange", width=0.8)
    ax2.set_title(
        f"AND: {best_and.expression}\n"
        f"N={g.n_activations}  months={g.n_active_months}  "
        f"max_conc={g.max_monthly_share:.4f}  tpm={g.mean_tpm:.2f}"
    )
    ax2.set_xlabel("Mese")
    ax2.set_ylabel("Attivazioni")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2.axhline(g.mean_tpm, color="red", linestyle="--", label="avg/month")
    ax2.legend()

    plt.tight_layout()
    out = "forgedge_best_candidates.png"
    plt.savefig(out, dpi=120)
    print(f"\nGrafico salvato → {out}")
except ImportError:
    print("\n(matplotlib non installato — grafico saltato)")
