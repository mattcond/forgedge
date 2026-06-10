# FORGE — Feature-Oriented Rule Generation Engine

FORGE is a quantitative research system for the **systematic discovery of
algorithmic trading rules** from historical market data. Starting from a KPI
Table (OHLCV + technical indicators), FORGE identifies boolean events with stable
temporal structure, measures their predictive power against an economic target
derived from the data, and produces formal contracts ready for operational
validation.

---

## What FORGE is

Systematic trading edge research suffers from three recurring problems:
look-ahead bias in signal selection, threshold optimisation on the same sample
used to evaluate the signal, and lack of separation between the statistical and
operational phases.

FORGE addresses these problems with a strictly separated pipeline architecture:

- **Module 1 never sees the forward return.** Events are discovered solely from
  the temporal structure of indicators — distribution, frequency, stability —
  without any exposure to future returns.
- **Thresholds are immutable.** Once fixed by Event Discovery, event thresholds
  (`RSI < 30.5`, `spread_ema < -0.012`) pass through the pipeline unchanged.
  No downstream module can recalibrate them.
- **The target is derived from data per event.** Alpha Discovery takes no
  economic parameters as input: for each event it scans a horizon grid and
  selects the one that maximises the statistical separation between active and
  inactive bars. The target is a measurement, not an assumption.
- **Out-of-sample confirmation is a gate, not a check.** OOS validation on the
  last 30% of the dataset is a formal promotion requirement — not an optional
  post-hoc review.

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
│  and confirms on the OOS tail. First exposure to forward return. │
│  Output: list[AlphaContract]                                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 3 — Rule Discovery                [not implemented]      │
│  Realistic backtest with order mechanics (limit orders, fees).   │
│  Output: Edge/Non-Edge verdict + operational parameters          │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 4 — Rule Registry                 [not implemented]      │
│  Deduplication, cross-asset backtest, report export.             │
│  Output: Validated rules ready for the execution system          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation status

| Module | Status |
|---|---|
| 0 — Market Context | ✅ Implemented |
| 1 — Event Discovery | ✅ Implemented |
| 2 — Alpha Discovery | ✅ Implemented |
| 3 — Rule Discovery | 🔲 Not implemented |
| 4 — Rule Registry | 🔲 Not implemented |

---

## Installation

FORGE depends solely on `numpy` and `pandas`. No `scipy`, `statsmodels`, or ML
library is required: all statistical primitives (Spearman, t-test, OU regression,
Benjamini-Hochberg FDR, incomplete beta function) are implemented in pure numpy.

```bash
pip install numpy pandas
```

---

## Quick start

```python
import pandas as pd
from forgedge import MarketContext, EventDiscovery, AlphaDiscovery, AlphaConfig

# KPI Table with OHLCV + technical indicators ('close' column required)
kpi = pd.read_parquet("kpi_table.parquet")

# Module 0 — market regime
enriched = MarketContext(kpi).run()

# Module 1 — event discovery
ed = EventDiscovery(enriched)
candidates = ed.run()
print(f"{len(candidates)} event candidates found")

# Module 2 — alpha measurement with data-derived target
ad = AlphaDiscovery(
    ed.df,
    candidates,
    AlphaConfig(asset="BTC", timeframe="1H"),
)
contracts = ad.run()
promoted = ad.promoted_contracts()
print(f"{len(promoted)} hypotheses promoted out of {len(contracts)} evaluated")
print(ad.summary().head())
```

---

## Design principles

**Separation of domains.** Each module answers exactly one question. Module 1:
"does this event have stable temporal structure?" Module 2: "does this event
predict the target?" Module 3: "is this pattern operationally tradeable?" No
module answers the next one's question; none accesses the previous module's data
beyond the formal artefact it produced.

**Immutable thresholds.** Event thresholds are fixed by Event Discovery based on
the in-sample distribution of the asset and are never modified by downstream
modules. Recalibrating them requires a new discovery session.

**Distributional thresholds.** Thresholds are not absolute values but percentiles
of the feature distribution on the specific asset. The same structural pattern
produces different thresholds on different assets (`RSI p10 = 30.5` on ADA,
`27.8` on BTC) while preserving the same statistical meaning.

**No look-ahead bias.** The forward return never enters Event Discovery. The
economic target is derived by Alpha Discovery **per event** on an IS window and
confirmed on an OOS tail that has taken no part in any prior computation.

---

## Documentation

| File | Contents |
|---|---|
| `how_to_use_en.md` | Practical guide to the end-to-end production pipeline |
| `modulo_0_en.md` | Market Context: regime, EMAProxy, configuration |
| `modulo_1_en.md` | Event Discovery: 5-step pipeline, EventCandidate, walk-forward |
| `modulo_2_en.md` | Alpha Discovery: derived target, OOS, AlphaContract |

Italian versions are in the corresponding `*_it.md` files.
