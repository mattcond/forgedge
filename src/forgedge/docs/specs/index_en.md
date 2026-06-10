# FORGE — Library Technical Manual (v0.1.0)

FORGE (**Feature-Oriented Rule Generation Engine**) is a quantitative research
system that discovers, validates, and formalises algorithmic trading rules from
historical market data.

The system is organised as a modular 5-stage pipeline. Modules run in sequence:
each stage consumes the output of the previous one and produces a formal artefact
that becomes the input for the next.

---

## Pipeline and modules

```
KPI Table (OHLCV + technical indicators)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 0 — Market Context                                      │
│  Classifies every bar by market regime.                         │
│  Output: KPI Table + 'regime' and 'regime_stable' columns       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 1 — Event Discovery                                     │
│  Discovers boolean events from the temporal structure of        │
│  indicators. Never sees the forward return.                     │
│  Output: list[EventCandidate]                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 2 — Alpha Discovery                                     │
│  Measures predictive power of events against an economic        │
│  target. First exposure to the forward return.                  │
│  Output: list[AlphaContract]                                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 3 — Rule Discovery            [not implemented v0.1.0]  │
│  Realistic backtest with order mechanics (limit orders, fees).  │
│  Output: Edge/Non-Edge verdict + operational parameters         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 4 — Rule Registry             [not implemented v0.1.0]  │
│  Deduplication, cross-asset backtest, CSV/HTML report export.   │
│  Output: Validated rules ready for the execution system         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation and dependencies

FORGE depends solely on `numpy` and `pandas`.
`scipy` and `statsmodels` are not required: all statistical primitives
(Spearman correlation, t-test, OU regression, Benjamini-Hochberg FDR) are
implemented in pure numpy.

```python
pip install numpy pandas
```

---

## Minimal end-to-end example

```python
import pandas as pd
from forgedge import (
    MarketContext,
    EventDiscovery,
    AlphaDiscovery, AlphaConfig, TargetDefinition,
)

# KPI Table with OHLCV + technical indicators
kpi = pd.read_parquet("kpi_table.parquet")   # must contain a 'close' column

# Module 0 — market regime
enriched = MarketContext(kpi).run()            # adds 'regime' and 'regime_stable'

# Module 1 — event discovery
ed = EventDiscovery(enriched)
candidates = ed.run()                          # list[EventCandidate]
print(f"{len(candidates)} candidates found")
print(ed.summary().head())

# Module 2 — alpha measurement
ad = AlphaDiscovery(
    ed.df,
    candidates,
    AlphaConfig(target=TargetDefinition(holding_period_h=24, sell_pct=0.04)),
)
contracts = ad.run()
promoted = ad.promoted_contracts()             # list[AlphaContract] with status "HYPOTHESIS"
print(f"{len(promoted)} promoted out of {len(contracts)}")
print(ad.summary().head())
```

---

## Documentation structure

| File | Module | Topics covered |
|---|---|---|
| `modulo_0_en.md` | Market Context (0) | Regime classification, EMAProxy, configuration, output methods |
| `modulo_1_en.md` | Event Discovery (1) | 5-step pipeline, data structures, walk-forward, SQL/formula export |
| `modulo_2_en.md` | Alpha Discovery (2) | IC measurement, win rate, regime analysis, scoring, Alpha Contract |

Italian versions are in `modulo_0_it.md`, `modulo_1_it.md`, `modulo_2_it.md`.

---

## Core architectural principles

**Separation of domains.** Each module answers exactly one question:
Module 1 asks only "does this event have stable temporal structure?"
without ever seeing the forward return. Module 2 asks "does this event predict
the target?" without optimising thresholds. Module 3 asks "is this pattern
operationally tradeable?" without re-running statistical discovery.

**Immutable thresholds.** Event thresholds (e.g. `RSI < 30.5`) are fixed by
Module 1 and propagated unchanged through the entire pipeline. Neither Module 2
nor Module 3 can modify them. A new threshold requires a new Event Discovery session.

**Distributional thresholds.** Thresholds are not fixed values but percentiles of the
transformed series distribution on the specific asset. The same structural pattern
produces different thresholds on different assets (e.g. RSI p10 = 30.5 on ADA,
27.8 on BTC) while preserving the same statistical meaning.

**No look-ahead bias.** The forward return never enters Module 1.
Scale-free classification is deliberately conservative (false negatives cost less
than false positives). Distributional thresholds are computed on the in-sample series.
