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
