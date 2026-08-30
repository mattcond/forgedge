# FORGE — Feature-Oriented Rule Generation Engine

FORGE is a quantitative research system for the **systematic discovery of
algorithmic trading rules** from historical market data. Starting from a KPI
Table (OHLCV + technical indicators), FORGE identifies boolean events with stable
temporal structure, measures their predictive power against an economic target
derived from the data, and produces formal contracts ready for operational
validation.

---

> **Looking for a single, complete reference?** The
> [forgedge manual](../../../../docs/manual-en.md) is a comprehensive,
> example-verified guide covering installation through production
> architecture, troubleshooting, an FAQ, and a glossary. It is the recommended
> starting point; this index and the module specs below go deeper on the
> internals of each pipeline stage.

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
│  Module 3 — Rule Discovery                                       │
│  Realistic backtest with order mechanics (limit orders, fees).   │
│  Output: Edge/Non-Edge verdict + operational parameters          │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Module 4 — Rule Registry                                        │
│  Deduplication, cross-ticker backtest, report export.            │
│  Output: Flat table + HTML report with genericity badges          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation status

| Module | Status |
|---|---|
| 0 — Market Context | ✅ Implemented |
| 1 — Event Discovery | ✅ Implemented |
| 2 — Alpha Discovery | ✅ Implemented |
| 3 — Rule Discovery | ✅ Implemented |
| 4 — Rule Registry | ✅ Implemented |

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

The orchestrator accepts each module's configuration as an optional keyword
argument. For the step-by-step pipeline with full configuration options, see
`how_to_use_en.md`.

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

**Centralised, validated configuration.** Every `forge()` run resolves its
per-module configuration through a central `PipelineContext` (`resolver.py`),
which fills in unset fields and then validates the whole bundle with
`config_report()`. By default (`strict=True`) a structurally incoherent
configuration raises `ValueError` **before** the pipeline runs, instead of
executing to completion and quietly producing a wall of rejected candidates.
Pass `strict=False` to downgrade these findings to warnings and run anyway.

---

## Documentation

| File | Contents |
|---|---|
| `concepts_en.md` | Conceptual guide: event, alpha, and rule — from market to signal |
| `how_to_use_en.md` | Practical guide to the end-to-end production pipeline |
| `modulo_0_en.md` | Market Context: regime, EMAProxy, configuration |
| `modulo_1_en.md` | Event Discovery: 5-step pipeline, EventCandidate, walk-forward |
| `modulo_2_en.md` | Alpha Discovery: derived target, OOS, AlphaContract |
| `modulo_3_en.md` | Rule Discovery: backtest, EDGE verdict, walk-forward OOS, reports |
| `modulo_4_en.md` | Rule Registry: deduplication, cross-ticker, genericity, export |
| `configuration_en.md` | Complete configuration reference: every dataclass field, type, default, and description |
| `playground_en.md` | Playground: usage guide for the analysis helpers over pooled `ForgeResult` output (all 11 use cases, issue #237) |
| `deployment_en.md` | Deployment: usage guide for the promotion-gate/export/monitoring functions that put discovered rules into production |

`forgedge.playground` — a read-only analysis layer over pooled `ForgeResult` output — is covered in the main manual (`docs/manual-en.md`, §9). Its tracking checklist (issue #237) is complete (10 functions + the cross-cutting `conversion_funnel`), but it remains a diagnostic layer, not a stable core API. [`playground_en.md`](playground_en.md) is its detailed usage reference (signatures, parameters, verified examples); [`modules/Playground.md`](../modules/Playground.md) covers the design rationale and internal algorithms instead.

`forgedge.deployment` — its sibling module for putting promoted rules into production (quality-gating, disk export, a monitoring manifest) — was split out of `forgedge.playground` by PR #247 (issue #245) because those functions have real effects a read-only name no longer described honestly. [`deployment_en.md`](deployment_en.md) is its usage reference; [`modules/Deployment.md`](../modules/Deployment.md) covers the design rationale.

Italian versions are in the corresponding `*_it.md` files.

Working on this codebase with Claude Code? The `forgedge` skill in
`.claude/skills/forgedge/` covers the library's API, verdicts, and pipeline
invariants in a form Claude can use directly.
