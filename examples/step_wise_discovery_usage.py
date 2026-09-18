"""
ForgeEdge — StepWiseDiscovery (forgedge.experiment) usage
============================================================
Dataset: 1D OHLCV CSV (default: examples/data/EURUSD_1DAY.csv)

``StepWiseDiscovery`` answers a narrower, more exploratory question than a
plain ``forge()`` call: "starting from the strongest hold-out-confirmed
single-condition rules, can we grow each one, one AND-condition at a time,
by searching only the sub-population it already selects for a second
dimension?" It is a per-seed *local* search on top of ``forge()``'s own
single global pass — see ``forgedge/experiment/step_wise_discovery.py``'s
module docstring for the full algorithm and its accumulated fixes
(rolling-transform child recovery without look-ahead, redundancy checks by
name/Jaccard/continuous-correlation, retry across confirmed children, a
reporting-order fix that never lets a failed composition attempt overwrite
a genuinely confirmed shallower result).

Perché passare event_discovery_config/alpha_config/rule_discovery_config
esplicitamente
-------------------------------------------------------------------------
``StepWiseDiscovery`` non ha idea di quale asset o timeframe stia guardando
oltre le stringhe ``ticker``/``timeframe``: qualunque calibrazione
economica resta compito del chiamante. Il caso più diretto è la fee:
``AlphaConfig.fee_per_side`` di default (0.002, 20 bps/lato) è calibrata
per crypto — su un cambio FX come EURUSD è ~20x il costo reale (un pip
vale circa 0.00009 sul mid). Usata senza modifiche azzera sistematicamente
ogni edge indipendentemente dalla qualità del segnale (profit_factor >= 2.0
irraggiungibile). Questo script mostra la diagnostica minima da fare prima
di fidarsi di un risultato: confrontare il costo di round-trip configurato
con la mossa mediana dell'asset, non limitarsi ad accettare il default.

Prerequisiti
------------
    pip install forgedge

Esecuzione
----------
    python examples/step_wise_discovery_usage.py
    python examples/step_wise_discovery_usage.py path/to/data_1D.csv \\
        --ticker COPPER --timeframe 1D --fee-per-side 0.00004 --n-seeds 3 --depth 2

Su dati giornalieri, un run tipico richiede ~1-3 minuti (iterazione 1 più
fino a ``--n-seeds`` catene indipendenti di ``--depth`` profondità); su dati
orari, molto di più — vedi la nota "timeframe" nel modulo stesso.

Un risultato senza alcuna catena EDGE/PARTIAL-EDGE è un esito valido, non un
errore: l'invariante di forgedge resta "una configurazione onesta non forza
un verdetto positivo" — vedi la skill ``forgedge`` per il razionale.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import pandas as pd

from forgedge import build_features, candle_features, forge_preset
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "examples" / "data" / "EURUSD_1DAY.csv"
# Verified pip-based EURUSD round-trip cost — see module docstring. NOT a
# generic default: pass --fee-per-side explicitly for any other asset.
DEFAULT_EURUSD_FEE_PER_SIDE = 0.00009


# ---------------------------------------------------------------------------
# 1. Data loading — this repo's own semicolon-separated OHLCV CSVs
# ---------------------------------------------------------------------------

def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_kpi(csv_path: Path) -> pd.DataFrame:
    """A minimal KPI table — EMA/RSI/ATR/MACD/returns at two periods.

    StepWiseDiscovery has no opinion on how the KPI table was built: this
    is one reasonable, generic choice, not a requirement. Pick indicator
    periods that make sense for your own data (e.g. derived from an
    OU-half-life estimate, as the module docstring's "timeframe" note
    discusses) if you have a better one.
    """
    candles = load_ohlcv(csv_path)
    kpi = build_features(candles, timestamp_col="timestamp")
    kpi = candle_features(kpi)
    return kpi


# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV),
                         help="OHLCV CSV path (semicolon-separated, this repo's own format)")
    parser.add_argument("--ticker", default="EURUSD")
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--preset", default="balanced",
                         choices=["sniper", "balanced", "sweep", "burst"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--fee-per-side", type=float, default=None,
        help="Override AlphaConfig.fee_per_side / RuleDiscoveryConfig.base_params.fee. "
             "Defaults to the verified EURUSD pip-based value ONLY when --ticker=EURUSD "
             "(the bundled default dataset); required for any other asset — see the "
             "printed diagnostic if you skip it.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    csv_path = Path(args.csv)

    kpi = build_kpi(csv_path)
    print(f"KPI table: {kpi.shape}, range {kpi['open_dt'].min()} -> {kpi['open_dt'].max()}")

    disc_cfg, alpha_cfg, rd_cfg = forge_preset(args.preset, timeframe=args.timeframe, asset=args.ticker)

    # --- Fee diagnostic — see module docstring. Do this for every new asset,
    # not just the ones this repo already knows the answer for. ---
    median_move = kpi["close"].pct_change().abs().median()
    if args.fee_per_side is not None:
        fee = args.fee_per_side
    elif args.ticker.upper() == "EURUSD":
        fee = DEFAULT_EURUSD_FEE_PER_SIDE
    else:
        fee = alpha_cfg.fee_per_side
    round_trip = 2 * fee
    print(f"\nDiagnostica fee: fee_per_side={fee:.6f}, round-trip={round_trip:.6f} "
          f"= {round_trip / median_move:.1%} della mossa mediana di {args.ticker} "
          f"({median_move:.5f}).")
    if round_trip / median_move > 0.5:
        print("  ATTENZIONE: il costo di round-trip supera il 50% della mossa mediana — "
              "probabile azzeramento sistematico di ogni edge. Passare --fee-per-side "
              "con un valore realistico per questo asset prima di fidarsi dei verdetti.")

    alpha_cfg = dataclasses.replace(alpha_cfg, fee_per_side=fee)
    rd_cfg = dataclasses.replace(
        rd_cfg, base_params=dataclasses.replace(rd_cfg.base_params, fee=fee),
    )

    engine = StepWiseDiscovery(
        kpi, ticker=args.ticker, timeframe=args.timeframe,
        event_discovery_config=disc_cfg, alpha_config=alpha_cfg, rule_discovery_config=rd_cfg,
        config=StepWiseDiscoveryConfig(n_seeds=args.n_seeds, depth=args.depth),
    )
    result = engine.run()

    print()
    print("\n".join(engine.log_lines))

    print("\n=== Risultato finale ===")
    with pd.option_context("display.max_colwidth", 100, "display.width", 200):
        print(result.summary().to_string(index=False))

    edges = result.edges()
    if edges:
        print(f"\n{len(edges)} catena/e con verdetto finale EDGE/PARTIAL-EDGE:")
        for c in edges:
            print(f"  {c.chain_label}: {c.expression} "
                  f"(profondita={c.depth_reached}, verdetto={c.verdict})")
    else:
        print("\nNessuna catena ha raggiunto un verdetto EDGE/PARTIAL-EDGE finale — "
              "esito onesto, non un errore: forgedge non forza un verdetto positivo "
              "su una configurazione che non lo sostiene.")


if __name__ == "__main__":
    main()
