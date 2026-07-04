# FORGE — Feature-Oriented Rule Generation Engine

FORGE is a quantitative research system for the **systematic discovery of algorithmic trading rules** from historical market data. Starting from a KPI Table (OHLCV + technical indicators), FORGE identifies boolean events with stable temporal structure, measures their predictive power against an economic target derived from the data, and produces formal contracts ready for operational validation.

[🇮🇹 Versione italiana](README_it.md)

---

## Why FORGE

Systematic edge research suffers from three recurring problems:

- **Look-ahead bias** — event thresholds calibrated while observing returns already "know" the future before discovery
- **In-sample optimisation** — thresholds and horizons tuned on the same window used for evaluation produce circular backtests
- **Missing operational separation** — statistical evidence of predictive power is not the same as profitability under real fees and order mechanics

FORGE addresses all three with a **strictly separated pipeline**: each module answers exactly one question and passes only a formal artefact to the next. No module can access the next module's data; no threshold can be recalibrated after discovery.

---

## Pipeline

```
KPI Table (OHLCV + technical indicators)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 0 — Market Context                                       │
│  Classifies every bar by market regime (5 levels).               │
│  Output: KPI Table + 'regime' and 'regime_stable' columns        │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 1 — Event Discovery                                      │
│  Discovers boolean events from the temporal structure of         │
│  indicators. Never sees the forward return.                      │
│  Output: list[EventCandidate]                                    │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 2 — Alpha Discovery                                      │
│  Derives target per event, measures IS predictive power,         │
│  confirms on the OOS tail. First exposure to forward return.     │
│  Output: list[AlphaContract]                                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 3 — Rule Discovery                                       │
│  Realistic backtest with order mechanics (limit orders, fees).   │
│  Output: EDGE / PARTIAL-EDGE / NON-EDGE + operational parameters │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 4 — Rule Registry                                        │
│  Deduplication, cross-ticker backtest, genericity classification.│
│  Output: flat table + self-contained HTML report                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key invariants

| Invariant | What it prevents |
|---|---|
| Module 1 never sees the forward return | Look-ahead bias in event selection |
| Thresholds are immutable after discovery | Threshold optimisation on the evaluation sample |
| Target horizon, direction, and take-profit are derived from data per event | Economic assumptions pre-baking the result |
| The tradeable verdict is gated by walk-forward OOS (Module 3); Alpha's OOS confirmation is recorded on every contract | Post-hoc confirmation of a foregone conclusion |

---

## Installation

FORGE depends solely on `numpy` and `pandas`. No `scipy`, `statsmodels`, or ML library required: all statistical primitives (Spearman, t-test, OU regression, Benjamini-Hochberg FDR, incomplete beta) are implemented in pure numpy.

```bash
pip install forgedge
```

---

## Quick start

```python
import pandas as pd
from forgedge import forge

# KPI Table with OHLCV + technical indicators ('close' column required)
kpi = pd.read_parquet("kpi_table.parquet")

# Full pipeline: from KPI Table to validated rules in a single call
result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.summary())                         # one row per candidate + rule_verdict
for contract, response in result.edges():       # EDGE / PARTIAL-EDGE only
    print(contract.alpha_id, response.verdict)
print(result.registry.summary())                # Module 4 — catalogued rules
```

On a daily-or-slower `timeframe` the default holding-period grid is
automatically replaced by a daily-calibrated one (the stock grid is calibrated
on ~hourly bars); passing your own `AlphaConfig` keeps full control.  For
frequency-consistent per-module configuration use
`forgedge.presets.forge_preset("balanced", timeframe="1D", asset=...)`.

Multi-ticker sessions with `forge_multi`:

```python
from forgedge import forge_multi

frames = {"BTCUSDC": btc_kpi, "ETHUSDC": eth_kpi, "ADAUSDC": ada_kpi}
results, registry = forge_multi(frames, timeframe="1H")

# GENERIC: rule generalises to ≥ 2/3 of tested tickers
df = registry.flat_table()
print(df[["rule_id", "classification", "pf", "cross_ticker_score"]])

# Self-contained HTML report (inline SVG, no CDN)
html = registry.html_report(timeframe="1H")
with open("report.html", "w") as f:
    f.write(html)
```

---

## The three concepts

FORGE structures the discovery process around three formal concepts that each answer a distinct question and produce a distinct artefact.

### Event — observing the market without bias

An **event** is a boolean condition on historical bars discovered from the temporal structure of indicators — without ever computing a forward return. Thresholds are distributional (asset-specific percentiles) and immutable once fixed.

```python
c = candidates[0]
print(c.expression)            # "rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118"
signal = c.apply(new_kpi)      # pd.Series bool — deterministic, no look-ahead
```

The ConsistencyGate filters out events with unstable temporal structure (too few activations, seasonal clustering, low monthly frequency) before any return is computed.

### Alpha — measuring predictive power

An **alpha** is the empirical answer to: *given that the event activated, what happens statistically in the next h bars?* Horizon, direction, and take-profit level (`sell_pct`) are all **derived from data** — never assumed — by scanning `|mean_advantage|/√h` across a horizon grid and taking the MFE quantile of active bars.

```python
c = promoted[0]
dt = c.derived_target
print(f"{dt.direction} at h={dt.holding_period_h}h  sell_pct={dt.sell_pct:.4f}")
print(f"Grade {c.alpha_score.grade}  |  OOS lift: {c.oos_validation.lift:.4f}")
```

The only hard rejection gate is an undetermined direction (no finite advantage across any horizon). All other statistical metrics (IC, Cohen's d, lift, FDR) contribute to the A–D grade without blocking promotion.

### Rule — trading realistically

A **rule** is the operational verdict on an alpha contract. Rule Discovery runs a realistic backtest with limit order entry, take-profit exit, horizon stop, and per-side fees — then validates the best parameter configuration on a rolling walk-forward OOS.

```python
resp = RuleDiscovery(ed.df, contract, cand).run()
print(resp.verdict)                               # "EDGE", "PARTIAL-EDGE", "NON-EDGE"
if resp.is_edge:
    p = resp.validated_rule.params
    print(f"Entry: limit -{p.buy_drop_pct:.2%}  TP: +{p.sell_pct:.2%}  h={p.target_h}")
    print(f"IS PF: {resp.in_sample_summary.profit_factor:.2f}"
          f"  WF consistency: {resp.walk_forward.consistency:.0%}")
```

---

## Module overview

| Module | Question answered | Key output |
|---|---|---|
| 0 — Market Context | Which regime is this bar in? | `regime` column (5 levels), `regime_stable` |
| 1 — Event Discovery | Is this indicator configuration stable and repeatable? | `EventCandidate` — immutable thresholds, `apply()` |
| 2 — Alpha Discovery | Does the event predict an oriented return? | `AlphaContract` — derived target, A–D grade |
| 3 — Rule Discovery | Is this alpha profitable under real order mechanics? | `RuleDiscoveryResponse` — EDGE verdict, `ValidatedRule` |
| 4 — Rule Registry | Does this rule generalise across tickers? | Flat table, HTML report — GENERIC / PARTIAL / SPECIFIC |

---

## Implementation status

| Module | Status |
|---|---|
| 0 — Market Context | ✅ Implemented |
| 1 — Event Discovery | ✅ Implemented |
| 2 — Alpha Discovery | ✅ Implemented |
| 3 — Rule Discovery | ✅ Implemented |
| 4 — Rule Registry | 🚧 WIP |

---

## Documentation

| File | Contents |
|---|---|
| [`concepts_en.md`](src/forgedge/docs/specs/concepts_en.md) | Conceptual guide: event, alpha, and rule — from market to signal |
| [`how_to_use_en.md`](src/forgedge/docs/specs/how_to_use_en.md) | End-to-end production pipeline guide with full configuration |
| [`modulo_0_en.md`](src/forgedge/docs/specs/modulo_0_en.md) | Market Context: regime classification, EMAProxy, configuration |
| [`modulo_1_en.md`](src/forgedge/docs/specs/modulo_1_en.md) | Event Discovery: 5-step pipeline, ConsistencyGate, EventCandidate |
| [`modulo_2_en.md`](src/forgedge/docs/specs/modulo_2_en.md) | Alpha Discovery: derived target, IC, OOS, AlphaContract |
| [`modulo_3_en.md`](src/forgedge/docs/specs/modulo_3_en.md) | Rule Discovery: backtest, EDGE verdict, walk-forward, reports |
| [`modulo_4_en.md`](src/forgedge/docs/specs/modulo_4_en.md) | Rule Registry: deduplication, cross-ticker, genericity, export |

---

## License

MIT
