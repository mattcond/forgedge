"""
ForgeEdge — Holdout validation of newly integrated indicators on 1D tickers
============================================================================
Ultime PR hanno integrato nuovi indicatori nel KPI Builder (CCI, WILLR,
Stochastic %K/%D, WMA, TRIMA, ADX, Aroon, A/D — vedi
``src/forgedge/kpi_builder/config.py``).  Questo script:

  1. Carica ogni ticker 1DAY presente in ``examples/data``.
  2. Costruisce la KPI Table con i nuovi indicatori abilitati.
  3. CENSURA la serie storica: l'ultimo periodo (default 6 mesi) è tenuto da
     parte come holdout e non viene mai passato a forge() — Event/Alpha/Rule
     Discovery vedono solo il periodo di training.
  4. Esegue la pipeline end-to-end (forge()) sul solo training per estrarre
     2-5 regole per ticker (EDGE / PARTIAL-EDGE su M3).
  5. Congela ciascuna regola selezionata (espressione dell'evento + parametri
     operativi di BacktestParams, invariato per definizione — vedi invariante
     #2 della pipeline) e la ri-applica SOLO sulla finestra di holdout, mai
     vista durante la selezione, per verificare se l'edge sopravvive fuori
     campione.

Uso
---
    python examples/holdout_1d_new_indicators.py [--holdout-months 6] [--out report.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgedge import build_features, candle_features, forge, forge_preset
from forgedge.kpi_builder.config import DEFAULT_CONFIG
from forgedge.rule_discovery.backtest import run_backtest

DATA_DIR = Path(__file__).resolve().parent / "data"

# Indicatori aggiunti nelle ultime PR (commit dba2d7d / 9c299dd) — disabilitati
# di default in DEFAULT_CONFIG per non cambiare l'output per chi non li chiede.
NEW_INDICATORS = ["cci", "willr", "stochastic", "wma", "trima", "adx", "aroon", "ad"]

# Ticker 1DAY presenti in examples/data (esclusi 1HOUR/5MIN).
TICKERS_1D = [
    "AMZN_1D.csv",
    "BTCEUR_1DAY.csv",
    "COPPER.CMDUSD_1DAY.csv",
    "ETHEUR_1DAY.csv",
    "EURUSD_1DAY.csv",
    "GBPUSD_1DAY.csv",
    "E_Brent_1DAY.csv",
    "E_DAAX_1DAY.csv",
    "E_SandP-500_1DAY.csv",
]

PRESET_FALLBACK_ORDER = ["balanced", "burst", "sweep"]

MIN_HOLDOUT_TRADES = 3


# ---------------------------------------------------------------------------
# 1. Caricamento e normalizzazione candele
# ---------------------------------------------------------------------------

def _load_candles(path: Path) -> pd.DataFrame:
    """Load one ticker's raw candles into a common open/high/low/close/volume
    + open_dt schema, sorted ascending."""
    if path.name == "AMZN_1D.csv":
        # investing.com-style export: comma+quoted, most-recent-first,
        # volume as "31.50M", Price==close.
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]

        def _parse_vol(v):
            if isinstance(v, str):
                v = v.strip()
                mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(v[-1], None)
                if mult is not None:
                    return float(v[:-1]) * mult
                try:
                    return float(v)
                except ValueError:
                    return np.nan
            return v

        out = pd.DataFrame({
            "open_dt": pd.to_datetime(df["date"], format="%m/%d/%Y"),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["price"].astype(float),
            "volume": df["vol."].map(_parse_vol),
        })
        out = out.sort_values("open_dt").reset_index(drop=True)
        return out

    # Semicolon-separated intrabroker export: timestamp;open;high;low;close;volume;ticker
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    df["open_dt"] = pd.to_datetime(df["timestamp"])
    df = df[["open_dt", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("open_dt").reset_index(drop=True)
    return df


def _build_kpi_table(candles: pd.DataFrame) -> pd.DataFrame:
    """build_features() with the new PR indicators enabled + candle geometry."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for name in NEW_INDICATORS:
        cfg[name]["enabled"] = True
    kpi = build_features(candles, cfg, timestamp_col="open_dt")
    kpi = candle_features(kpi)
    return kpi


# ---------------------------------------------------------------------------
# 2. forge() sul solo training (holdout censurato)
# ---------------------------------------------------------------------------

def _run_forge_with_fallback(kpi_train: pd.DataFrame, ticker: str, two_pass: bool = False):
    """Try presets in order, from the most selective to the most permissive.

    A preset can fail two different ways: it can raise (config_report finds
    the resolved configuration structurally incoherent for this history —
    pitfall #8, a low daily rate deriving an M3 training window too long for
    the pooled OOS span this history has), or it can run cleanly but promote
    zero edges. Either way we keep trying more permissive presets; the first
    preset that yields at least one edge wins, and if none do we fall back to
    the last preset that at least ran without raising (an honest "0 edges"
    result beats no result).

    ``two_pass`` (default ``False``) toggles ``forge()``'s
    ``two_pass_composition`` — M2's grade-guided AND-composition (issue #254
    Phase 8). It defaults off because it is O(n^2) in the 1D candidate pool
    and OOM-killed (>13.9 GB RSS) on this container when run against a KPI
    table this wide across the *whole* 9-ticker batch; opt in per-ticker
    (``--two-pass --tickers ...``) for a single, monitored run."""
    last_err = None
    last_clean: tuple | None = None  # (result, preset) of the last non-raising run
    for preset in PRESET_FALLBACK_ORDER:
        disc, alpha, rd = forge_preset(preset, timeframe="1D", asset=ticker)
        try:
            result = forge(
                kpi_train,
                ticker=ticker,
                timeframe="1D",
                event_discovery_config=disc,
                alpha_config=alpha,
                rule_discovery_config=rd,
                two_pass_composition=two_pass,
                progress=False,
            )
        except ValueError as exc:
            last_err = exc
            continue
        last_clean = (result, preset)
        if result.edges():
            return result, preset
    if last_clean is not None:
        return last_clean
    raise RuntimeError(f"[{ticker}] every preset raised on config_report: {last_err}")


# ---------------------------------------------------------------------------
# 3. Verifica in holdout — regola congelata (espressione + parametri fissi)
# ---------------------------------------------------------------------------

def _verify_on_holdout(kpi_full: pd.DataFrame, cand, params, holdout_from: str) -> dict:
    """Reconstruct the frozen event on the FULL (train+holdout) history and
    backtest with the frozen operational params, counting only entries opened
    inside the holdout window (fills/exits may reach past it, as usual)."""
    signal = cand.apply(kpi_full)
    df = kpi_full.copy()
    df["__signal__"] = signal.fillna(0)
    summary = run_backtest(
        df, "__signal__", params,
        timerange_from=holdout_from, timerange_to=None,
        timestamp_col="open_dt",
    )
    return {
        "total_trades": summary.total_trades,
        "win_rate_pct": summary.win_rate_pct,
        "profit_factor": summary.profit_factor,
        "total_net_gain": summary.total_net_gain,
        "expectancy": summary.expectancy,
    }


def _survives(holdout_stats: dict) -> str:
    n = holdout_stats["total_trades"]
    if n < MIN_HOLDOUT_TRADES:
        return "INCONCLUSIVE (too few holdout trades)"
    pf = holdout_stats["profit_factor"]
    gain = holdout_stats["total_net_gain"]
    if pf is not None and not (isinstance(pf, float) and np.isnan(pf)) and pf > 1.0 and gain > 0:
        return "SURVIVES"
    return "FAILS"


# ---------------------------------------------------------------------------
# 4. Selezione 2-5 regole per ticker
# ---------------------------------------------------------------------------

def _rank_key(pair):
    _, resp = pair
    gain = resp.in_sample_summary.total_net_gain
    pf = resp.in_sample_summary.profit_factor
    if resp.walk_forward is not None and resp.walk_forward.oos_summary is not None:
        gain = resp.walk_forward.oos_summary.total_net_gain
        pf = resp.walk_forward.oos_summary.profit_factor
    gain = gain if gain is not None and not (isinstance(gain, float) and np.isnan(gain)) else -np.inf
    pf = pf if pf is not None and not (isinstance(pf, float) and np.isnan(pf)) else -np.inf
    return (gain, pf)


def process_ticker(path: Path, holdout_months: int, two_pass: bool = False) -> dict:
    ticker = path.stem
    candles = _load_candles(path)
    kpi_full = _build_kpi_table(candles)

    last_date = kpi_full["open_dt"].max()
    cutoff = last_date - pd.DateOffset(months=holdout_months)
    kpi_train = kpi_full[kpi_full["open_dt"] < cutoff].copy().reset_index(drop=True)
    holdout_bars = int((kpi_full["open_dt"] >= cutoff).sum())

    info = {
        "ticker": ticker,
        "bars_total": len(kpi_full),
        "bars_train": len(kpi_train),
        "bars_holdout": holdout_bars,
        "range_total": [str(kpi_full["open_dt"].min().date()), str(last_date.date())],
        "cutoff": str(cutoff.date()),
        "two_pass_composition": two_pass,
        "rules": [],
    }

    try:
        result, preset_used = _run_forge_with_fallback(kpi_train, ticker, two_pass=two_pass)
    except RuntimeError as exc:
        info["error"] = str(exc)
        return info
    info["preset_used"] = preset_used
    if two_pass:
        info["n_grading_candidates_1d"] = len(result.grading_candidates or [])
        info["n_pooled_candidates_pass2"] = len(result.candidates)
        info["n_composed_candidates"] = sum(
            1 for c in result.candidates if len(c.components) > 1
        )

    edges = result.edges()
    info["n_edges_pipeline"] = len(edges)
    if not edges:
        info["note"] = "0 edges promoted by the pipeline on the training window"
        return info

    edges_sorted = sorted(edges, key=_rank_key, reverse=True)
    selected = edges_sorted[:5]

    candidates_by_id = {c.event_id: c for c in result.candidates}

    for contract, resp in selected:
        cand = candidates_by_id.get(contract.event_candidate_id)
        if cand is None or resp.validated_rule is None:
            continue
        holdout_stats = _verify_on_holdout(
            kpi_full, cand, resp.validated_rule.params, str(cutoff.date())
        )
        rule_info = {
            "alpha_id": contract.alpha_id,
            "event_expression": cand.expression,
            "n_components": len(cand.components),
            "direction": resp.validated_rule.params.direction,
            "pipeline_verdict": resp.verdict,
            "pipeline_in_sample": {
                "total_trades": resp.in_sample_summary.total_trades,
                "win_rate_pct": resp.in_sample_summary.win_rate_pct,
                "profit_factor": resp.in_sample_summary.profit_factor,
                "total_net_gain": resp.in_sample_summary.total_net_gain,
            },
            "holdout": holdout_stats,
            "holdout_verdict": _survives(holdout_stats),
        }
        info["rules"].append(rule_info)

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-months", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tickers", nargs="*", default=None,
                     help="Subset of filenames from examples/data to run (default: all 1D)")
    ap.add_argument("--two-pass", action="store_true",
                     help="Enable forge()'s two_pass_composition (M2 grade-guided AND "
                          "composition). Off by default: O(n^2) in the 1D candidate pool, "
                          "OOM-prone on a wide KPI table across the whole batch — use with "
                          "--tickers to try it on one ticker at a time.")
    args = ap.parse_args()

    files = args.tickers or TICKERS_1D
    report = []
    for fname in files:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"!! missing file: {path}")
            continue
        print(f"\n=== {fname} ===")
        try:
            info = process_ticker(path, args.holdout_months, two_pass=args.two_pass)
        except Exception as exc:  # keep going across tickers
            info = {"ticker": path.stem, "error": f"{type(exc).__name__}: {exc}"}
        report.append(info)
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue
        print(f"  bars total/train/holdout = {info['bars_total']}/{info['bars_train']}/{info['bars_holdout']}")
        print(f"  preset={info.get('preset_used')}  pipeline edges={info.get('n_edges_pipeline')}")
        if args.two_pass and "n_pooled_candidates_pass2" in info:
            print(f"  1D pool={info['n_grading_candidates_1d']}  "
                  f"pass2 pool={info['n_pooled_candidates_pass2']}  "
                  f"composed={info['n_composed_candidates']}")
        for r in info["rules"]:
            tag = f"AND x{r['n_components']}" if r["n_components"] > 1 else "single"
            print(f"  - {r['alpha_id']:>28s} [{tag:>8s}] {r['pipeline_verdict']:>13s} -> "
                  f"holdout: {r['holdout_verdict']} "
                  f"(trades={r['holdout']['total_trades']}, "
                  f"pf={r['holdout']['profit_factor']}, "
                  f"gain={r['holdout']['total_net_gain']})")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nSaved report -> {args.out}")


if __name__ == "__main__":
    main()
