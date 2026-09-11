# Hold-out study — three 5-rule portfolios on 1D data

Reproduces the study end to end from `examples/data/*_1DAY.csv`. Every script is
re-runnable; only the scripts and the small CSV summaries are tracked (see
`.gitignore` — the KPI tables, `forge()` results and trade ledgers are derived).

| Step | Script | Produces |
|---|---|---|
| 1 | `01_build_kpi.py` | `kpi/<TICKER>.parquet` — daily-calibrated KPI tables for 8 instruments |
| 2 | `02_discover.py [TICKERS…]` | `runs/<TICKER>.pkl` — one `forge()` run per instrument, **discovery span only** |
| 3 | `03_pool.py` | `out/pool.csv` — every tradeable rule replayed on IS and OOS |
| 4 | `04_portfolios.py` | 3 portfolios, snake-drafted, evaluated on IS/OOS/**HO** |
| 4b | `04b_portfolios_v2.py` | diversification-constrained variant (built *after* the hold-out was observed) |
| 5 | `06_pool_ho.py` | `out/pool_ho.csv` — every rule replayed on the hold-out (diagnostic) |
| 6 | `07_payload.py` + `08_build_report.py` | `out/payload.json`, then the standalone HTML report |

`lib_eval.py` holds the shared replay and portfolio-metric helpers.

## Protocol

* **Hold-out**: `2026-01-01 → 2026-09-02` (8.1 months) is cut in `02_discover.py`
  before `forge()` sees anything. Steps 1–4 never read it.
* **Discovery**: `2021-01-03 → 2025-12-31` (crypto from `2023-03-14`). The IS/OOS
  boundary is the session `TimeBudget` split that `forge()` derives itself.
* **Sizing**: every opened position is 100 EUR of notional. Positions overlap
  freely, so peak concurrency sets the capital the book needs.
* **Costs**: 0.10% per side (0.20% round trip), set on `AlphaConfig.fee_per_side`
  so the resolver propagates it to `BacktestParams.fee`.
* **Determinism**: `02_discover.py` re-execs with `PYTHONHASHSEED=0` (SKILL.md
  pitfall #24) so repeated runs promote the same contracts.

## Result

All three portfolios clear IS and OOS (PF 3.8–6.2) and lose money on the hold-out
(PF 0.30–0.84). Replaying the whole pool on the hold-out shows why: 31.5% of the
815 gated rules were profitable there, against 34.8% of all 1,510 tradeable rules
— passing the IS+OOS gates made a rule *less* likely to survive. No rule ever
cleared the rotation null (`rotation_p = 1.00` on all but one leg).

## Parte 2 — il gate a tre finestre

`09_multiwindow.py` ri-segna tutte le 1.510 regole su **IS / OOS / HO1 / HO2**
in un solo replay per regola: il filtro di `run_backtest` agisce solo sulla barra
di apertura (`dt[signal_rn+1]`), quindi un replay full-range bucketizzato su
quella barra riproduce esattamente le finestre separate — verificato contro
`pool.csv` (conteggi identici, differenza massima di PF 0.0).

| Step | Script | Produce |
|---|---|---|
| 7 | `09_multiwindow.py` | `out/multiwindow.csv`, `out/ledgers.pkl` |
| 8 | `10_triple_gate.py` | Set A (gate IS+OOS+HO) e Set B (gate IS+OOS+HO1, HO2 sigillata) |
| 9 | `11_payload2.py` + `12_report_part2.py` | sezione 2 del report HTML |

**Esito.** HO1 = `2026-01-01 → 2026-05-01` entra nel gate, HO2 = `2026-05-01 →
2026-09-02` resta sigillata. Sopravvivenza su HO2: nessun gate 29,6% · IS+OOS
30,5% · IS+OOS+HO1 **24,5%**. Con cap orizzonte ≤20 barre (così ogni regola può
chiudere dentro HO2): 29,6% · 28,5% · **21,8%**. Aggiungere una finestra al gate
peggiora la finestra successiva.

L'unico segnale che regge è la **classe di asset**: le regole su indici
sopravvivono al 67,8% (gate IS+OOS) e 67,4% (gate IS+OOS+HO1), praticamente
invariate; commodity 16,9% → 10,1%, crypto 22,6% → 15,0%.
