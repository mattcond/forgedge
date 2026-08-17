"""
ForgeEdge — KPI Table Extraction for 1D Daily Data
===================================================
Fonte dati: ADA_1D_FULL.parquet  (2024-01-01 → 2026-06-19)

Pipeline:
  FORGE     — MarketContext → EventDiscovery → AlphaDiscovery → RuleDiscovery (IS 2024)
  TARGETOPT — TargetOptimizer con orizzonte 5 giorni (IS 2024)

Per ogni pipeline si estraggono i 5 migliori eventi (per composite_score / lift).
Per ciascun evento si calcola il backtest su tre finestre temporali:
  IS  2024: 2024-01-01 → 2025-01-01
  OOS 2025: 2025-01-01 → 2026-01-01
  OOS 2026: 2026-01-01 → (fine dati)

KPI estratti per ogni finestra:
  WR   = win rate (%)
  PF   = profit factor
  N ATTIVAZIONI (LIMIT)  = eseguiti con ordine limite
  N ATTIVAZIONI (MARKET) = eseguiti con ordine market
  GAIN = net gain totale (%)

Esecuzione:
  python examples/kpi_table_1d.py /path/to/ADA_1D_FULL.parquet
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.insert(0, "src")

from forgedge import (
    AlphaConfig,
    AlphaDiscovery,
    DiscoveryConfig,
    EventDiscovery,
    MarketContext,
)
from forgedge.alpha_discovery.models import PromotionThresholds, TargetConfig
from forgedge.event_discovery.models import GateParams
from forgedge.rule_discovery.backtest import run_backtest
from forgedge.rule_discovery.models import (
    BacktestParams,
    ScoringParams,
    GridSpec,
    SelectionCriteria,
    RuleWalkForwardConfig,
    RuleDiscoveryConfig,
)
from forgedge.target_optimizer import TargetOptimizer

# ─────────────────────────────────────────────────────────────────────────────
# 0. Caricamento dati
# ─────────────────────────────────────────────────────────────────────────────

PARQUET_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/root/.claude/uploads/97a39385-fb76-5589-adb0-bb331ce97505/"
    "0bdebcc8-ADA_1D_FULL.parquet"
)

df_full = pd.read_parquet(PARQUET_PATH)
df_full = df_full.sort_values("open_dt").reset_index(drop=True)
df_full["open_dt"] = pd.to_datetime(df_full["open_dt"])

print(f"Dati totali : {len(df_full)} barre")
print(f"Range       : {df_full['open_dt'].iloc[0].date()} → {df_full['open_dt'].iloc[-1].date()}")
print(f"Distribuzione anni:")
print(df_full.groupby(df_full["open_dt"].dt.year).size().to_string())
print()

# IS training = solo 2024
df_is = df_full[df_full["open_dt"] < "2025-01-01"].copy().reset_index(drop=True)
print(f"IS (2024) : {len(df_is)} barre  "
      f"({df_is['open_dt'].iloc[0].date()} → {df_is['open_dt'].iloc[-1].date()})")

# ─────────────────────────────────────────────────────────────────────────────
# Finestre temporali per il backtest
# ─────────────────────────────────────────────────────────────────────────────
PERIODS = [
    ("IS 2024",  "2024-01-01", "2025-01-01"),
    ("OOS 2025", "2025-01-01", "2026-01-01"),
    ("OOS 2026", "2026-01-01", None),
]

# ─────────────────────────────────────────────────────────────────────────────
# Parametri di backtest — dati 1D, orizzonte 5 giorni
# ─────────────────────────────────────────────────────────────────────────────
HORIZON_DAYS = 5

BASE_LIMIT_PARAMS = BacktestParams(
    direction="long",
    buy_type="limit",
    buy_drop_pct=0.010,
    buy_delay_bar=3,
    sell_pct=0.040,
    target_h=HORIZON_DAYS,
    target_hit_col="high",
    fee=0.001,
    early_stopping=True,
)
BASE_MARKET_PARAMS = BASE_LIMIT_PARAMS.merged(buy_type="market")

SCORING = ScoringParams(pf_min_trades=5, pf_min_tpm=1)
SIGNAL_COL = "__kpi_signal__"


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computazione full-dataset (MarketContext + EventDiscovery una sola volta)
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparazione full-dataset (2024-2026) per il backtest…")
mc_full = MarketContext(df_full.copy())
enriched_full = mc_full.run()
ed_full = EventDiscovery(
    enriched_full.copy(),
    DiscoveryConfig(timestamp_col="open_dt", train_ratio=1.0),
)
ed_full.run()

# ed_full.df usa open_dt come DatetimeIndex; il backtest richiede anche la colonna.
full_frame_base = ed_full.df.copy()
if isinstance(full_frame_base.index, pd.DatetimeIndex) and "open_dt" not in full_frame_base.columns:
    full_frame_base["open_dt"] = full_frame_base.index
print(f"Full frame pronto: {len(full_frame_base)} barre, colonne {list(full_frame_base.columns[:5])}…")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_signal(candidate) -> pd.DataFrame:
    """Applica il segnale booleano del candidato sull'intero dataset."""
    df = full_frame_base.copy()
    stored = candidate.event_series
    if stored is not None and stored.index.equals(df.index):
        series = stored
    else:
        series = candidate.apply(df)
    df[SIGNAL_COL] = series.fillna(0).to_numpy()
    return df


def _backtest_period(frame: pd.DataFrame, params: BacktestParams,
                     period_from: str | None, period_to: str | None) -> dict:
    """Esegue il backtest su una finestra temporale e restituisce i KPI."""
    try:
        summary, _ = run_backtest(
            frame, SIGNAL_COL, params,
            timerange_from=period_from,
            timerange_to=period_to,
            scoring=SCORING,
            timestamp_col="open_dt",
            return_trades=True,
        )
        pf = summary.profit_factor
        if not np.isfinite(pf):
            pf = 9999.0 if summary.winning_trades > 0 else 0.0
        return {
            "wr":      round(summary.win_rate_pct, 1),
            "pf":      round(pf, 2),
            "n_trades": summary.total_trades,
            "gain":    round(summary.total_net_gain * 100, 2),
        }
    except Exception:
        return {"wr": float("nan"), "pf": float("nan"), "n_trades": 0, "gain": float("nan")}


def _kpi_row(event_id: str, expression: str, candidate) -> dict:
    """Calcola i KPI per i tre periodi (limit e market) e restituisce una riga."""
    frame = _inject_signal(candidate)
    row = {"EVENT_ID": event_id, "EXPRESSION": expression}

    for period_label, p_from, p_to in PERIODS:
        lim = _backtest_period(frame, BASE_LIMIT_PARAMS, p_from, p_to)
        mkt = _backtest_period(frame, BASE_MARKET_PARAMS, p_from, p_to)

        row[f"{period_label}_WR"]       = lim["wr"]
        row[f"{period_label}_PF"]       = lim["pf"]
        row[f"{period_label}_N_LIMIT"]  = lim["n_trades"]
        row[f"{period_label}_N_MARKET"] = mkt["n_trades"]
        row[f"{period_label}_GAIN"]     = lim["gain"]

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORGE pipeline su IS 2024
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PIPELINE: FORGE  (IS 2024)")
print("=" * 60)

mc_is = MarketContext(df_is.copy())
enriched_is = mc_is.run()
print(f"Regimi IS: {mc_is.distribution()['n_bars'].to_dict()}")

ed_is = EventDiscovery(
    enriched_is.copy(),
    DiscoveryConfig(
        gate_params=GateParams(min_tpm=1.5, max_dispersion=3.0),
        timestamp_col="open_dt",
        max_and_components=2,
        train_ratio=1.0,
    ),
)
forge_candidates = ed_is.run()
print(f"Event Candidates FORGE: {len(forge_candidates)}")

alpha_cfg = AlphaConfig(
    horizon_grid=(1, 2, 3, 5, 7, 10),
    train_ratio=0.70,
    asset="ADAUSDC",
    exchange="binance_spot",
    timeframe="1D",
    thresholds=PromotionThresholds(
        min_lift=0.06,
        min_cohens_d=0.10,
        use_fdr=True,
        fdr_q=0.15,
        oos_max_p=0.15,
    ),
)

ad = AlphaDiscovery(ed_is.df, forge_candidates, alpha_cfg)
forge_contracts = ad.run()
forge_promoted = ad.promoted_contracts()
print(f"Alpha promossi FORGE: {len(forge_promoted)}/{len(forge_contracts)}")

forge_promoted_sorted = sorted(
    forge_promoted, key=lambda c: c.alpha_score.composite_score, reverse=True
)
forge_top5 = forge_promoted_sorted[:5]

# Se meno di 5 promossi, completa con i migliori non promossi
if len(forge_top5) < 5:
    non_prom = [c for c in forge_contracts if c not in forge_promoted]
    non_prom_sorted = sorted(
        non_prom,
        key=lambda c: c.alpha_score.composite_score if c.alpha_score else -999,
        reverse=True,
    )
    forge_top5 = forge_top5 + non_prom_sorted[:5 - len(forge_top5)]

by_cand_id = {c.event_id: c for c in forge_candidates}

forge_rows = []
for i, contract in enumerate(forge_top5, start=1):
    cand = by_cand_id.get(contract.event_candidate_id)
    if cand is None:
        print(f"  [{i}] SKIP — candidato non trovato")
        continue
    score = contract.alpha_score.composite_score
    print(f"  [{i}] score={score:.3f}  {contract.event_expression[:75]}")
    forge_rows.append(_kpi_row(f"FORGE-{i}", contract.event_expression, cand))

# ─────────────────────────────────────────────────────────────────────────────
# 2. TARGETOPT pipeline su IS 2024
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PIPELINE: TARGETOPT  (IS 2024, orizzonte 5gg)")
print("=" * 60)

target_cfg = TargetConfig(
    horizon=HORIZON_DAYS,
    min_return=0.030,
    side="long",
    min_activations=10,
    min_lift_atoms=1.0,
    min_lift_result=1.0,
    target_mode="abs",
)

opt = TargetOptimizer(
    enriched_is.copy(),
    target_cfg,
    DiscoveryConfig(
        gate_params=GateParams(min_tpm=1.5, max_dispersion=3.0),
        timestamp_col="open_dt",
        max_and_components=2,
        train_ratio=1.0,
    ),
)
to_results = opt.run()
to_candidates = opt.candidates or []
print(f"TargetOptimizer: {len(to_results)} candidati, base rate {opt.base_rate:.1%}")

to_cand_by_id = {c.event_id: c for c in to_candidates}

# Top-5 per lift (tabella già ordinata discendente)
to_top5: list[tuple] = []
for _, row in to_results.iterrows():
    cand = to_cand_by_id.get(row["event_id"])
    if cand is not None:
        to_top5.append((row, cand))
    if len(to_top5) >= 5:
        break

to_rows = []
for i, (res_row, cand) in enumerate(to_top5, start=1):
    expr = getattr(cand, "expression", None) or getattr(cand, "sql_expression", None) or str(cand.event_id)
    lift_val = res_row.get("lift", float("nan"))
    print(f"  [{i}] lift={lift_val:.3f}  {str(expr)[:75]}")
    to_rows.append(_kpi_row(f"TARGETOPT-{i}", str(expr), cand))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compila e stampa tabella KPI
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TABELLA KPI")
print("=" * 60)

all_rows = []
for i, row in enumerate(forge_rows, start=1):
    all_rows.append({"ID": i, "Pipeline": "FORGE", **row})
for i, row in enumerate(to_rows, start=1):
    all_rows.append({"ID": i, "Pipeline": "TARGETOPT", **row})

kpi_df = pd.DataFrame(all_rows)


def _fmt(v, is_pf=False):
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    if is_pf and isinstance(v, float) and v >= 9999:
        return "∞"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


W = 148
print()
print("-" * W)
header = (
    f"{'ID':>2} {'Pipeline':<10} {'EVENT_ID':<14} "
    f"{'IS24_WR':>8} {'IS24_PF':>8} {'IS24_NL':>8} {'IS24_NM':>8} {'IS24_G%':>9} | "
    f"{'OS25_WR':>8} {'OS25_PF':>8} {'OS25_NL':>8} {'OS25_NM':>8} {'OS25_G%':>9} | "
    f"{'OS26_WR':>8} {'OS26_PF':>8} {'OS26_NL':>8} {'OS26_NM':>8} {'OS26_G%':>9}"
)
print(header)
print("-" * W)

for _, r in kpi_df.iterrows():
    def _g(col): return r.get(col, float("nan"))
    print(
        f"{int(r['ID']):>2} {r['Pipeline']:<10} {r['EVENT_ID']:<14} "
        f"{_fmt(_g('IS 2024_WR')):>8} "
        f"{_fmt(_g('IS 2024_PF'), True):>8} "
        f"{int(_g('IS 2024_N_LIMIT') or 0):>8} "
        f"{int(_g('IS 2024_N_MARKET') or 0):>8} "
        f"{_fmt(_g('IS 2024_GAIN')):>9} | "
        f"{_fmt(_g('OOS 2025_WR')):>8} "
        f"{_fmt(_g('OOS 2025_PF'), True):>8} "
        f"{int(_g('OOS 2025_N_LIMIT') or 0):>8} "
        f"{int(_g('OOS 2025_N_MARKET') or 0):>8} "
        f"{_fmt(_g('OOS 2025_GAIN')):>9} | "
        f"{_fmt(_g('OOS 2026_WR')):>8} "
        f"{_fmt(_g('OOS 2026_PF'), True):>8} "
        f"{int(_g('OOS 2026_N_LIMIT') or 0):>8} "
        f"{int(_g('OOS 2026_N_MARKET') or 0):>8} "
        f"{_fmt(_g('OOS 2026_GAIN')):>9}"
    )

print("-" * W)
print()
print("Legenda colonne:")
print("  WR   = Win Rate (%)")
print("  PF   = Profit Factor (∞ = nessun trade perdente)")
print("  NL   = N Attivazioni (ordine LIMIT eseguito)")
print("  NM   = N Attivazioni (ordine MARKET eseguito)")
print("  G%   = Gain netto cumulato (%)")
print()
print("Parametri backtest comune:")
print(f"  Orizzonte   : {HORIZON_DAYS} giorni")
print(f"  Fee/lato    : {BASE_LIMIT_PARAMS.fee*100:.1f}%")
print(f"  Limit drop  : {BASE_LIMIT_PARAMS.buy_drop_pct*100:.1f}%  "
      f"(delay ≤{BASE_LIMIT_PARAMS.buy_delay_bar} barre)")
print(f"  Take-profit : {BASE_LIMIT_PARAMS.sell_pct*100:.1f}%  "
      f"(rilevato su {BASE_LIMIT_PARAMS.target_hit_col})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Dettaglio espressioni evento
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ESPRESSIONI EVENTO")
print("=" * 60)
for _, r in kpi_df.iterrows():
    tag = f"[{r['Pipeline']}-{int(r['ID'])}]"
    print(f"{tag:<16} {r.get('EXPRESSION', '')}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Salva CSV
# ─────────────────────────────────────────────────────────────────────────────
out_path = "kpi_table_1d.csv"
kpi_df.to_csv(out_path, index=False)
print(f"\nTabella salvata in: {out_path}")
