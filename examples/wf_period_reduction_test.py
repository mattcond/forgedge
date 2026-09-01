"""
ForgeEdge — Walk-Forward Reproducibility Test (period reduction)
==================================================================
Dataset: 1D OHLCV CSV (default: examples/data/AMZN_1D.csv)

Domanda che questo script risponde: se accorcio lo storico di N mesi e
rieseguo *l'intera* pipeline FORGE da zero, (1) viene riscoperta la stessa
regola e (2) il profit factor che M3 riporta in walk-forward per quella
regola è quello che si ottiene rigiocandola sui mesi esclusi?

Due iterazioni:

  Iterazione 1 — pipeline completa (Market Context → Event Discovery →
      Alpha Discovery → Rule Discovery) sull'intero storico. Tra i verdetti
      PARTIAL-EDGE/EDGE con profit factor > 1 in walk-forward, si sceglie
      una regola a condizione singola (niente AND) i cui trade OOS coprono
      abbastanza gli ultimi N mesi da rendere il confronto significativo.

  Iterazione 2 — stesso dataset, storico troncato di N mesi (i.e. gli ultimi
      N mesi non entrano né in Event Discovery né in Alpha/Rule Discovery),
      pipeline rieseguita da zero. Si cerca tra i risultati una regola sulla
      stessa feature con lo stesso segno di soglia: le soglie di FORGE sono
      distribuzionali (percentili), quindi un piccolo spostamento tra le due
      iterazioni è atteso — non un fallimento del test.

Verifiche
---------
  1) La regola dell'iterazione 2 esiste ed è ancora PARTIAL-EDGE/EDGE sulla
     stessa feature/segno — "la stessa regola" a meno del piccolo shift di
     soglia dovuto al campione di training più corto.
  2) Si rigioca (run_backtest, stessi BacktestParams) sia la regola esatta
     dell'iterazione 1 sia quella riscoperta dall'iterazione 2 SOLO sui mesi
     isolati, e si confrontano i due profit factor con quello che il ledger
     OOS walk-forward di forge aveva già riportato in iterazione 1 per la
     stessa finestra — tre stime indipendenti dello stesso numero.

Esecuzione
----------
    python examples/wf_period_reduction_test.py [path/to/data_1D.csv] \\
        [--months 6] [--preset balanced] [--ticker AMZN] [--timeframe 1D]
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from forgedge import build_features, candle_features, forge, forge_preset
from forgedge.rule_discovery.backtest import run_backtest
from forgedge.rule_discovery.models import ScoringParams

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "examples" / "data" / "AMZN_1D.csv"

_EXPR_RE = re.compile(r"^([A-Za-z0-9_]+)\s*(<|>)\s*(-?[\d.]+)$")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Caricamento dati — CSV "Date,Price,Open,High,Low,Vol.,Change %" → OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    """Legge un CSV stile investing.com e restituisce OHLCV ascendente."""
    df = pd.read_csv(csv_path)
    df["volume"] = (
        df["Vol."]
        .str.replace("M", "e6", regex=False)
        .str.replace("K", "e3", regex=False)
        .astype(float)
    )
    df["open_time"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns={"Price": "close", "Open": "open", "High": "high", "Low": "low"})
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("open_time").reset_index(drop=True)
    # Alcuni provider hanno artefatti di arrotondamento (es. low > open/close
    # di pochi centesimi su una barra): si riallinea invece di far fallire
    # la validazione OHLC di summary_report().
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    return df


def build_kpi(csv_path: Path) -> pd.DataFrame:
    candles = load_ohlcv(csv_path)
    kpi = build_features(candles, timestamp_col="open_time")
    kpi = candle_features(kpi)
    return kpi


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pf_from_net(net: pd.Series) -> float:
    """Profit factor da una serie di net_pct_gain — stessa formula di
    ``forgedge.rule_discovery.backtest._summarise_arrays``."""
    arr = np.asarray(net)
    pos = arr[arr > 0].sum()
    neg = -arr[arr < 0].sum()
    if neg == 0:
        return 9999.0 if pos > 0 else 0.0
    return pos / neg


def parse_single_condition(expression: str):
    """Ritorna (feature, operatore, soglia) se l'espressione è una singola
    condizione (niente AND), altrimenti None."""
    m = _EXPR_RE.match(expression.strip())
    if not m:
        return None
    feature, op, thr = m.groups()
    return feature, op, float(thr)


def run_pipeline(kpi: pd.DataFrame, ticker: str, timeframe: str, preset: str):
    disc_cfg, alpha_cfg, rd_cfg = forge_preset(preset, timeframe=timeframe, asset=ticker)
    return forge(
        kpi, ticker=ticker, timeframe=timeframe,
        event_discovery_config=disc_cfg,
        alpha_config=alpha_cfg,
        rule_discovery_config=rd_cfg,
        strict=False,
    )


def replay_rule_on_window(event_frame: pd.DataFrame, candidate, params, window_from: pd.Timestamp):
    """Rigioca un candidato con i suoi BacktestParams solo dopo ``window_from``,
    riusando un frame già arricchito (indicatori calcolati sull'intero storico
    così l'evento è valutabile correttamente anche nella finestra isolata)."""
    frame = event_frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex) and "open_dt" not in frame.columns:
        frame["open_dt"] = frame.index
    signal_col = "__signal__"
    frame[signal_col] = candidate.apply(frame).fillna(0).to_numpy()
    summary, trades = run_backtest(
        frame, signal_col, params,
        timerange_from=str(window_from.date()), timerange_to=None,
        scoring=ScoringParams(), timestamp_col="open_dt", return_trades=True,
    )
    return summary, trades


# ─────────────────────────────────────────────────────────────────────────────
# Iterazione 1 — pipeline completa + scelta della regola target
# ─────────────────────────────────────────────────────────────────────────────

def rank_target_rules(result, cutoff: pd.Timestamp, min_isolated_trades: int):
    """Tra i verdetti PARTIAL-EDGE/EDGE con pf>1 in walk-forward e condizione
    singola, ritorna tutti quelli con abbastanza trade OOS dopo ``cutoff`` da
    rendere il confronto sui mesi isolati significativo, ordinati per
    consistency/pf decrescenti. Più di uno perché non tutte le regole
    dell'iterazione 1 sono garantite sopravvivere (stessa feature/segno,
    verdetto ancora edge) nell'iterazione 2 rieseguita su meno dati — le
    soglie sono percentili distribuzionali e uno spostamento può, per una
    singola regola, farla scendere sotto il gate; si prova quindi la
    successiva in classifica invece di fallire al primo tentativo."""
    candidates = []
    for contract, resp in result.rule_responses:
        if resp.verdict not in ("PARTIAL-EDGE", "EDGE"):
            continue
        wf = resp.walk_forward
        if wf is None or wf.oos_summary.profit_factor <= 1.0:
            continue
        parsed = parse_single_condition(contract.event_expression)
        if parsed is None:
            continue
        ot = wf.oos_trades
        if ot is None or ot.empty:
            continue
        iso = ot[pd.to_datetime(ot["fill_dt"]) > cutoff]
        if len(iso) < min_isolated_trades:
            continue
        candidates.append((contract, resp, parsed, iso))

    candidates.sort(key=lambda t: (t[1].walk_forward.consistency, t[1].walk_forward.oos_summary.profit_factor), reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Iterazione 2 — stessa feature/segno tra le regole riscoperte
# ─────────────────────────────────────────────────────────────────────────────

def find_matching_rule(result2, feature: str, op: str, threshold: float):
    """Cerca tra i rule_responses dell'iterazione 2 una regola a condizione
    singola sulla stessa feature/segno, e ritorna la più vicina in soglia."""
    matches = []
    for contract, resp in result2.rule_responses:
        if resp.verdict not in ("PARTIAL-EDGE", "EDGE"):
            continue
        parsed = parse_single_condition(contract.event_expression)
        if parsed is None:
            continue
        f2, op2, thr2 = parsed
        if f2 == feature and op2 == op:
            matches.append((contract, resp, thr2, abs(thr2 - threshold)))
    if not matches:
        return None
    matches.sort(key=lambda t: t[3])
    return matches[0][:3]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV), help="CSV OHLCV 1D (default: examples/data/AMZN_1D.csv)")
    ap.add_argument("--months", type=int, default=6, help="mesi da escludere dallo storico nell'iterazione 2 (default 6)")
    ap.add_argument("--preset", default="balanced", choices=["sniper", "balanced", "sweep", "burst"])
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--timeframe", default="1D")
    ap.add_argument("--min-isolated-trades", type=int, default=5, dest="min_isolated_trades")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Dataset: {csv_path}")
    kpi_full = build_kpi(csv_path)
    end_full = kpi_full["open_dt"].max()
    cutoff = end_full - pd.DateOffset(months=args.months)
    print(f"Storico completo : {kpi_full['open_dt'].min().date()} -> {end_full.date()}  ({len(kpi_full)} barre)")
    print(f"Cutoff (-{args.months} mesi) : {cutoff.date()}")
    print()

    # ── Iterazione 1 ────────────────────────────────────────────────────────
    print("=" * 78)
    print("ITERAZIONE 1 — pipeline completa sull'intero storico")
    print("=" * 78)
    result1 = run_pipeline(kpi_full, args.ticker, args.timeframe, args.preset)
    n_partial_edge = sum(1 for _, r in result1.rule_responses if r.verdict in ("PARTIAL-EDGE", "EDGE"))
    print(f"\nContratti valutati: {len(result1.contracts)}  promossi: {len(result1.promoted)}  "
          f"PARTIAL-EDGE/EDGE: {n_partial_edge}")

    ranked = rank_target_rules(result1, cutoff, args.min_isolated_trades)
    if not ranked:
        print("\nNessuna regola PARTIAL-EDGE/EDGE a condizione singola con pf>1 WF e "
              f">= {args.min_isolated_trades} trade negli ultimi {args.months} mesi. Interrompo.")
        return
    print(f"\n{len(ranked)} regola/e candidata/e (condizione singola, pf>1 WF, "
          f">= {args.min_isolated_trades} trade isolati) — si proverà in ordine di consistency/pf finché "
          "una non sopravvive anche nell'iterazione 2.")

    # ── Iterazione 2 ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"ITERAZIONE 2 — stesso dataset, storico troncato di {args.months} mesi")
    print("=" * 78)
    kpi_short = kpi_full[kpi_full["open_dt"] <= cutoff].copy().reset_index(drop=True)
    print(f"Storico troncato: {kpi_short['open_dt'].min().date()} -> {kpi_short['open_dt'].max().date()}  "
          f"({len(kpi_short)} barre)")

    result2 = run_pipeline(kpi_short, args.ticker, args.timeframe, args.preset)
    n_partial_edge2 = sum(1 for _, r in result2.rule_responses if r.verdict in ("PARTIAL-EDGE", "EDGE"))
    print(f"\nContratti valutati: {len(result2.contracts)}  promossi: {len(result2.promoted)}  "
          f"PARTIAL-EDGE/EDGE: {n_partial_edge2}")

    match = None
    for attempt, (contract1, resp1, (feature, op, thr1), iso1) in enumerate(ranked, start=1):
        m = find_matching_rule(result2, feature, op, thr1)
        if m is not None:
            match = m
            break
        print(f"  tentativo {attempt}: '{contract1.event_expression}' non riscoperta nell'iterazione 2, provo la successiva…")

    print(f"\nRegola scelta (iter. 1, tentativo {attempt}/{len(ranked)}): {contract1.event_expression}")
    wf1 = resp1.walk_forward
    cand1 = {c.event_id: c for c in result1.candidates}[resp1.validated_rule.event_candidate_id]
    print(f"  verdetto={resp1.verdict}  IS pf={resp1.in_sample_summary.profit_factor:.4f}  "
          f"WF pf={wf1.oos_summary.profit_factor:.4f}  consistency={wf1.consistency}")
    print(f"  params: {resp1.validated_rule.params}")

    forge_isolated_pf = pf_from_net(iso1["net_pct_gain"])
    print(f"\n  PF di forge sul ledger OOS ristretto agli ultimi {args.months} mesi "
          f"(n={len(iso1)}): {forge_isolated_pf:.4f}")

    replay1_summary, _ = replay_rule_on_window(result1.event_frame, cand1, resp1.validated_rule.params, cutoff)
    print(f"  PF rigiocando la regola esatta (run_backtest, stessa finestra): "
          f"{replay1_summary.profit_factor:.4f}  (n={replay1_summary.total_trades})")

    print()
    print("=" * 78)
    print("VERIFICHE")
    print("=" * 78)
    if match is None:
        print(f"[1] FALLITA — nessuna delle {len(ranked)} regole candidate dell'iterazione 1 "
              "è stata riscoperta (stessa feature/segno, ancora PARTIAL-EDGE/EDGE) nell'iterazione 2.")
        return

    contract2, resp2, thr2 = match
    shift_pct = abs(thr2 - thr1) / abs(thr1) * 100 if thr1 else float("nan")
    print(f"[1] OK — stessa feature/segno riscoperta: {contract2.event_expression}")
    print(f"    soglia iter.1={thr1:.6g}  soglia iter.2={thr2:.6g}  shift={shift_pct:.2f}%")
    print(f"    verdetto iter.2={resp2.verdict}  WF pf iter.2={resp2.walk_forward.oos_summary.profit_factor:.4f}")

    cand2 = {c.event_id: c for c in result2.candidates}[resp2.validated_rule.event_candidate_id]
    replay2_summary, _ = replay_rule_on_window(result1.event_frame, cand2, resp2.validated_rule.params, cutoff)
    diff_pct = abs(replay2_summary.profit_factor - forge_isolated_pf) / forge_isolated_pf * 100 if forge_isolated_pf else float("nan")

    print()
    print(f"[2] PF sugli ultimi {args.months} mesi isolati, tre stime indipendenti:")
    print(f"      forge, ledger OOS walk-forward (iter. 1)........ {forge_isolated_pf:.4f}  (n={len(iso1)})")
    print(f"      replay della regola esatta di iter. 1........... {replay1_summary.profit_factor:.4f}  (n={replay1_summary.total_trades})")
    print(f"      replay della regola riscoperta in iter. 2....... {replay2_summary.profit_factor:.4f}  (n={replay2_summary.total_trades})")
    print(f"    scarto tra iter.1 (forge) e iter.2 (replay): {diff_pct:.2f}%")


if __name__ == "__main__":
    main()
