# The forgedge Manual

*A complete, practical guide to the FORGE (Feature-Oriented Rule Generation Engine) Python library.*

Version covered: `forgedge==0.1.3`. Every code example in this manual was executed against the actual library in this repository; every number quoted from a run is a real, verified value, not an invented illustration. Where the manual states something the authors themselves documented (as opposed to something inferred from reading the code), it says so explicitly.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Why FORGE Exists](#2-why-forge-exists)
3. [When to Use It — and When Not To](#3-when-to-use-it--and-when-not-to)
4. [Core Concepts](#4-core-concepts)
5. [Installation](#5-installation)
6. [Your First Working Example](#6-your-first-working-example)
7. [Quick Start: A Full Pipeline Run](#7-quick-start-a-full-pipeline-run)
8. [Anatomy of the Workflow](#8-anatomy-of-the-workflow)
9. [Main API and Components](#9-main-api-and-components)
10. [Configuration](#10-configuration)
11. [Error Handling](#11-error-handling)
12. [Progressive Use Cases](#12-progressive-use-cases)
13. [Working with the Data in This Repository](#13-working-with-the-data-in-this-repository)
14. [Design Choices](#14-design-choices)
15. [Opt-in Behaviors](#15-opt-in-behaviors)
16. [Trade-offs](#16-trade-offs)
17. [Performance and Scalability](#17-performance-and-scalability)
18. [Testing](#18-testing)
19. [Integrating forgedge into a Real Application](#19-integrating-forgedge-into-a-real-application)
20. [A Production-Ready Architecture](#20-a-production-ready-architecture)
21. [Troubleshooting](#21-troubleshooting)
22. [Best Practices](#22-best-practices)
23. [Anti-patterns](#23-anti-patterns)
24. [FAQ](#24-faq)
25. [Glossary](#25-glossary)
26. [API Reference (Quick Lookup)](#26-api-reference-quick-lookup)

---

## 1. Introduction

`forgedge` is a Python library that takes a table of historical price data plus technical indicators (a **KPI Table**) and systematically discovers **boolean trading rules** — expressions like `rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118` — that have measurable, out-of-sample-confirmed predictive power over future price movement.

It does this through a five-stage pipeline, each stage answering exactly one question and handing a formal, inspectable object to the next stage:

```
KPI Table  →  [Market Context]  →  [Event Discovery]  →  [Alpha Discovery]  →  [Rule Discovery]  →  [Rule Registry]
 (input)         regime tag         boolean events        economic target       EDGE / NON-EDGE      catalog + report
```

The output of a `forgedge` session is not a prediction and not a trained model. It is a set of **formal rule specifications** — a boolean condition, a direction (long/short), a holding period, a take-profit percentage, and a verdict backed by walk-forward out-of-sample statistics. What you do with that specification (build a signal generator, wire it into an execution engine, review it manually) is entirely up to you; `forgedge` does not place orders, does not talk to an exchange, and holds no position state.

This manual assumes you are a competent Python developer who has never seen this library before. It does not assume you know quantitative finance jargon in advance — every term (KPI Table, event, alpha, walk-forward, ConsistencyGate, rotation null...) is introduced before it is used. By the end, you should be able to install the library, run a full discovery session on real data, understand every major configuration knob and what it costs you to change, and have a concrete plan for wiring `forgedge` into a larger application.

---

## 2. Why FORGE Exists

The library's own documentation ([`README.md`](../README.md)) states the problem directly:

> "Systematic edge research suffers from three recurring problems: **look-ahead bias** — event thresholds calibrated while observing returns already 'know' the future before discovery; **in-sample optimisation** — thresholds and horizons tuned on the same window used for evaluation produce circular backtests; **missing operational separation** — statistical evidence of predictive power is not the same as profitability under real fees and order mechanics."

The architecture is the answer to those three problems, not a side effect of it. The deep architectural guide shipped in the repository (`src/forgedge/docs/README.md`, Italian) states this as an explicit, non-negotiable constraint rather than a coding convention:

> *(translated)* "FORGE maintains a strict separation between three operational domains. **Every boundary is an architectural constraint, not a convention.**" The three domains are: **temporal structure** (Event Discovery observes only the timing pattern of indicators), **statistical predictiveness** (Alpha Discovery is the first stage that reads a forward return), and **operability** (Rule Discovery and Rule Registry are the only stages that know about fees, fills, and drawdown).

Concretely, this means:

- **Event Discovery (Module 1) never computes a forward return.** The boolean expressions it mines are chosen purely from how an indicator's own value has behaved historically — never from what happened afterward. This is stated in the source as an architectural fact, not a tuning choice: *"Event Discovery lavora completamente cieco rispetto al target economico... Questo non è un dettaglio implementativo — è un vincolo architetturale che elimina una categoria intera di look-ahead bias"* ("this is not an implementation detail — it is an architectural constraint that eliminates an entire category of look-ahead bias," `src/forgedge/docs/README.md`).
- **Thresholds are fixed once discovered and never re-tuned against results.** Wanting a "better" threshold means running Event Discovery again on a different in-sample window — never mutating a discovered candidate.
- **The economic target (how long to hold, which direction, what take-profit) is derived from the data by Alpha Discovery, never assumed by the caller.** This keeps a user's economic hunch (e.g., "surely a 2% move in 10 bars") from silently biasing which events look good.
- **Rule Discovery is the only economic judge.** A boolean expression can look statistically excellent in Module 2 and still be rejected in Module 3 because it isn't profitable once realistic order mechanics (limit fills, fees, delays) are simulated.

### The name and metaphor

The repository's architecture guide is explicit that the name is a deliberate metaphor, not an acronym pun: *(translated)* "Like a metallurgical forge turns raw ore into a worked tool through successive stages, FORGE turns a table of technical indicators into operational boolean rules through four sequential modules — without ever making assumptions about the execution system that will use them." (`src/forgedge/docs/README.md`)

---

## 3. When to Use It — and When Not To

### Use `forgedge` when...

- You have (or can build) a table of historical OHLCV bars plus technical indicators for one or more tickers, and you want to **systematically search** for boolean conditions that historically preceded a statistically significant price move — rather than hand-picking indicator thresholds from intuition.
- You want that search to carry **honest statistical guardrails against overfitting** by construction: out-of-sample confirmation, walk-forward validation, a search-level multiple-testing correction (the rotation null, §14–15), and a Deflated Sharpe Ratio check — not bolted on after the fact.
- You want **auditable output**: every candidate carries its own rejection reasons, every contract records why it was or wasn't promoted, and the pipeline's own intermediate artefacts (`ForgeResult.candidates`, `.contracts`, `.event_frame`) remain inspectable after the run.
- Your dependency budget is tight. `forgedge` depends on **only `numpy` and `pandas`** — no `scipy`, no `statsmodels`, no ML framework. All statistical primitives (Spearman correlation, t-tests, the incomplete beta function, Benjamini-Hochberg FDR control, OU-process half-life regression) are reimplemented in pure numpy.

### Do NOT use `forgedge` for...

- **Order execution or position management.** It has no exchange connectivity, no order placement, no portfolio state. This is stated explicitly and repeatedly in the repo's own documentation: *"FORGE non esegue ordini, non gestisce posizioni, non si connette a exchange... È un sistema di ricerca, non di esecuzione"* — "FORGE does not execute orders, manage positions, or connect to exchanges... it is a research system, not an execution system" (`src/forgedge/docs/README.md`). If you need that, `forgedge` produces the *specification* (a `ValidatedRule`, an entry/exit parameter set) that a separate execution system implements.
- **Machine-learning-based signal generation.** The design docs (`docs/analysis/forge2_functional_analysis.md`) explicitly list this as a rejected direction, with a stated reason: *(translated)* "No ML/feature learning in discovery. The differentiating value is that every rule is a readable, auditable boolean expression; a trained model would break the contract far more than any bug would." If you're looking for a library that fits a classifier or a neural net to price data, this is not it, by design.
- **A single indicator/threshold you already know you want to test.** `forgedge`'s value is in the systematic search plus statistical guardrails; if you already have one specific hypothesis (e.g., "test whether RSI < 30 predicts a bounce"), the library's `CustomEvent` mechanism (§9, §12 Use Case 5) lets you inject it directly, but you'd be using a small fraction of what the library does.
- **Sub-second/tick-level microstructure research.** Every worked example, every statistical calibration, and the library's own low-frequency robustness analysis (`docs/analysis/lowfreq_robustness.md`) are built and tested around 1-hour-to-1-day bar data. Nothing in the code technically forbids other frequencies, but see §16 (Trade-offs) and §21 (Troubleshooting) for the very real statistical-power problems that appear on short in-sample windows.
- **Anything requiring a persistent, cross-session catalog of discovered rules out of the box.** Module 4 (Rule Registry) is explicitly stateless and rebuilt from scratch every session — persistence is the *host application's* responsibility (§19–20), not the library's.

---

## 4. Core Concepts

Before touching any API, build this mental model. `forgedge` is organized around three formal concepts, each the deliverable of one stage of the pipeline, plus one input concept and one supporting concept:

| Concept | Produced by | Answers |
|---|---|---|
| **KPI Table** | you (or `forgedge.kpi_builder`) | "What data am I feeding the pipeline?" |
| **Regime** | Module 0 — Market Context | "What kind of market condition is bar *t* in?" |
| **Event** (`EventCandidate`) | Module 1 — Event Discovery | "Is this indicator condition stable and repeatable over time?" |
| **Alpha** (`AlphaContract`) | Module 2 — Alpha Discovery | "Given the event fired, what happens statistically in the following bars?" |
| **Rule** (`RuleDiscoveryResponse`) | Module 3 — Rule Discovery | "Is this alpha actually profitable under realistic order mechanics?" |

### KPI Table

A `pandas.DataFrame` with:

- a **`close`** column (float, required by every module),
- a **datetime source** — either a column (default name `open_dt`) or a `DatetimeIndex`,
- any number of **feature columns**: RSI, EMA, Bollinger Bands, volatility, spreads, candle geometry — anything you want `forgedge` to consider.

`forgedge` classifies every non-timestamp column automatically (continuous numeric, discrete/binary, categorical) and decides how to transform it — you don't tell it which columns are "features to use."

### Event

An **event** is a boolean condition on historical bars — e.g. `rsi_14 < 31.2` or the AND-composition `rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118` — discovered purely from the temporal structure of one or more indicator columns, **without ever looking at a forward return**. Its thresholds are *distributional* (percentiles of the indicator's own history on this specific asset) rather than hardcoded constants, which is what makes the same discovery *procedure* transferable across assets even though the resulting *numeric* threshold differs per asset. An event's thresholds are immutable the moment Event Discovery has fixed them.

### Alpha

An **alpha** is the empirical answer to: *given that this event just fired, what happens, on average, over the next h bars?* Alpha Discovery derives — never assumes — three things per event: the best holding period `h*`, a direction (long or short), and a take-profit level `sell_pct`. It measures predictive power (Information Coefficient, lift over base rate, Cohen's d) and confirms it on an out-of-sample tail. The only thing that can hard-reject an alpha is an **undetermined direction** — no horizon in the scanned grid produces a finite, sign-determined advantage. Every other statistical weakness (low IC, low lift, failed OOS confirmation) becomes a non-blocking diagnostic that lowers the contract's letter grade (A–D) instead of rejecting it outright.

### Rule

A **rule** is the operational verdict on an alpha contract, produced by simulating a realistic backtest — limit-order entry, take-profit exit, a horizon-based stop, per-side fees — and then validating the selected parameter set on a rolling **walk-forward** out-of-sample split. The verdict is one of four strings: `"EDGE"`, `"PARTIAL-EDGE"`, `"NON-EDGE"`, or `"INSUFFICIENT-DATA"` (§9, §15). Rule Discovery is explicitly the pipeline's *only* economic judge — an alpha with a perfect statistical profile can still come out `NON-EDGE`.

### Flow, inputs, and invariants at a glance

```
   KPI Table (you provide, or build with kpi_builder)
        │
        ▼
   Module 0 — classify each bar's market regime
        │  output: + 'regime' (5-level ordered categorical), + 'regime_stable'
        ▼
   Module 1 — mine boolean events from indicator history ONLY
        │  output: list[EventCandidate]   (thresholds now frozen)
        ▼
   Module 2 — derive an economic target per event, measure predictive power
        │  output: list[AlphaContract]    (h*, direction, sell_pct — all derived)
        ▼
   Module 3 — realistic backtest + walk-forward OOS validation
        │  output: EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA
        ▼
   Module 4 — dedup, cross-ticker generalisation test, catalog + HTML report
```

Data only ever flows forward. No later module reaches back into an earlier module's internals, and — this is the invariant worth internalizing before writing any code against this library — **no module can access information that, chronologically, its predecessor could not have had**. That single rule is what the whole architecture exists to enforce.

---

## 5. Installation

`forgedge` requires **Python ≥ 3.9** and depends only on `pandas>=1.5` and `numpy>=1.23` (from `pyproject.toml`). It has no optional dependency needed for the core pipeline.

```bash
pip install forgedge
```

For running the test suite or contributing, install the `dev` extra:

```bash
pip install "forgedge[dev]"     # adds pytest>=7.0
```

If you're working from a clone of this repository (as this manual's examples were verified):

```bash
git clone https://github.com/mattcond/forgedge
cd forgedge
pip install -e ".[dev]"
```

A few things `forgedge` does **not** need that you might expect: no `scipy`, no `statsmodels`, no database, no configuration file, no environment variables, no network access, no GPU. It's a stateless, pure-computation library — everything it needs is what you hand it in memory.

If you plan to read the parquet fixture used throughout this manual (`tests/fixtures/ADA_1D_TRAIN.parquet`), you also need a parquet engine — `forgedge` itself doesn't require one, but `pandas.read_parquet` does:

```bash
pip install pyarrow
```

**Verifying the install:**

```python
import forgedge
print(forgedge.__version__)   # "0.1.3" at the time of writing
```

---

## 6. Your First Working Example

This is the smallest possible piece of code that demonstrates the library's central idea: discovering a *boolean event from an indicator's own history*, with no forward return involved at all. It uses Module 1 (Event Discovery) in isolation, on a tiny synthetic table, so you can see exactly what an `EventCandidate` looks like before anything about profitability enters the picture.

```python
import numpy as np
import pandas as pd
from forgedge import EventDiscovery

# A minimal KPI Table: 500 hourly bars with one RSI-like oscillator.
rng = np.random.default_rng(42)
n = 500
kpi = pd.DataFrame({
    "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
    "close": 100 * np.cumprod(1 + rng.normal(0, 0.004, n)),
    "close_rsi_14": np.clip(50 + rng.normal(0, 15, n).cumsum() * 0.05, 1, 99),
})

ed = EventDiscovery(kpi)
candidates = ed.run()

print(f"{len(candidates)} event candidates passed the Consistency Gate")

c = candidates[0]
print("expression:", c.expression)
print("activations:", c.activation_stats.n_activations)
print("mean trades/month:", round(c.activation_stats.mean_tpm, 3))
```

**Verified output** (this manual's author ran this exact code against this exact repository — your own numbers may differ slightly with a different numpy version, since the RNG stream can shift across versions, but the *shape* is representative):

```
309 event candidates passed the Consistency Gate
expression: close crosses_below 95.45
activations: 10
mean trades/month: 10.0
```

`candidates[0]` happens to be a "crossing" event built directly on the raw `close` column, not, as you might guess, one built on `close_rsi_14`. Checking `ed.get_classifications()` on this exact run reveals why, and it's a genuinely useful surprise:

```python
print(ed.get_classifications())
# {'close': ColumnClassification(..., is_scale_free=True, ...),
#  'close_rsi_14': ColumnClassification(..., is_scale_free=False, ...)}
```

That's the **opposite** of the intuitive guess (surely a bounded oscillator is "scale-free" and a raw price series isn't?). It's a real, verified result, and it's worth using to correct that intuition rather than papering over it: the scale-free test (`TypeClassifier._is_scale_free`, `event_discovery/classifier.py`) does not look at whether a series happens to be bounded in `[0,100]`. It splits the series into blocks and checks whether each block's *value range* overlaps enough with the others' (`support_overlap_threshold`, default 0.5) — i.e. "does this series occupy roughly the same range throughout the sample, or does it drift to new levels over time?" This toy example's synthetic `close_rsi_14` was built as `50 + rng.normal(0,15,n).cumsum()*0.05` — a slow **cumulative** wobble that drifts across its clipped range in a blocky, non-stationary way — so it failed that test, while the mildly-drifting geometric random walk in `close` happened to pass it on this particular seed. **The lesson generalizes beyond this toy example:** scale-free classification is a genuine statistical test on your actual data, not a lookup based on the column's name or its nominal bounds — a real RSI computed correctly would very likely classify differently, but you shouldn't assume any column's classification without checking `ed.get_classifications()`.

### What just happened, internally

1. `EventDiscovery(kpi)` classified both columns as `CONTINUOUS`, then ran the scale-free test on each independently — see the box above for what that test actually measures. The scale-free flag governs whether a column is eligible to build *ratio/spread* features against other same-family columns (Step 2 below); it does not gate whether a column can have its own direct threshold/crossing events, which is why `close` — despite not being what you'd casually call "scale-free" — still produced the `close crosses_below 95.45` candidate that topped this run's list.
2. It generated **transformed versions** of each column: rolling percentile rank, rolling z-score, and simple deltas, over several window lengths.
3. For each (base or transformed) series, it tried a catalog of **distributional thresholds** (e.g. the 10th percentile of that series' own history) and **theoretical thresholds** (fixed z-score levels like −2.0), each producing a candidate boolean expression — a "crossing" event on a raw column's own value (as with `close` here), or something shaped like `close_rsi_14 < P10 [P10=...]` for a transformed series (present among this run's 309 candidates — filter with `[c for c in candidates if "rsi" in c.expression]` to find one).
4. Every candidate was passed through the **Consistency Gate**: does it fire often enough (`min_tpm`), consistently enough across months (not bursty — `max_dispersion`), and with enough total observations to be statistically meaningful? Candidates that fail are silently discarded — they never become `EventCandidate` objects.
5. **At no point did any of this look at `close`'s future value.** The gate only ever inspects *when* the event fired, never *what happened afterward*.

### Implicit configuration you should notice

You called `EventDiscovery(kpi)` with no `config=` argument. That means `DiscoveryConfig()` defaults were used silently:

- `train_ratio=1.0` — the *entire* table was used for discovery, no OOS split was reserved (there's no walk-forward validation happening here; see §10 for how to enable it).
- `gate_params=GateParams()` — the default Consistency Gate thresholds: `min_tpm=0.5` (at least 0.5 qualifying "episodes" per month), `max_dispersion=1.5` (activations shouldn't be too bursty), `event_counting="episode"` (§15).
- `max_and_components=2` — Event Discovery also tried composing pairs of single-column events with AND, subject to the same gate.

None of these choices involved the forward return of `close` — that concept doesn't exist yet at this stage of the pipeline.

---

## 7. Quick Start: A Full Pipeline Run

Section 6 showed one module in isolation. This section runs the **entire pipeline** with the single orchestrator function, `forge()`, on real data included in this repository: `tests/fixtures/ADA_1D_TRAIN.parquet` — 882 daily bars of ADA (Cardano) OHLCV plus 22 precomputed technical-indicator columns, spanning 2024-01-01 to 2026-05-31. This is the same fixture the library's own golden regression tests pin their expected values against.

```python
import pandas as pd
from forgedge import forge

kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
print(kpi.shape)                      # (882, 26)
print(list(kpi.columns)[:6])          # ['open_dt', 'high', 'low', 'close', 'open', 'close_ret_03']

result = forge(kpi, ticker="ADAUSDC", timeframe="1D", progress=False)

print("M1 candidates:", len(result.candidates))
print("M2 promoted:  ", len(result.promoted))
print("M3 responses: ", len(result.rule_responses))
print("tradeable (edges()):", len(result.edges()))
```

**Verified output** (this manual's author ran this exact code against this exact repository):

```
(882, 26)
['open_dt', 'high', 'low', 'close', 'open', 'close_ret_03']
M1 candidates: 5241
M2 promoted:   370
M3 responses:  370
tradeable (edges()): 54
```

### Interpreting the output

- **5241 event candidates** survived Event Discovery's Consistency Gate — remember, none of these have been checked against a forward return yet.
- **370 of them were promoted** by Alpha Discovery to `AlphaContract` "HYPOTHESIS" status — meaning each has a determined direction (long or short) and a derived holding period / take-profit.
- **370 rule responses** — every promoted contract was run through Rule Discovery's realistic backtest and walk-forward validation (this fixture has `run_rule_discovery=True` by default).
- **54 are tradeable** (`result.edges()` — verdict `EDGE` or `PARTIAL-EDGE`). On this specific dataset with default settings, digging one level deeper shows *all 54* are `PARTIAL-EDGE`, not full `EDGE`:

```python
from collections import Counter
print(Counter(r.verdict for _, r in result.rule_responses))
# Counter({'NON-EDGE': 314, 'PARTIAL-EDGE': 54, 'INSUFFICIENT-DATA': 2})
```

Zero full `EDGE` verdicts is not a bug and not a sign the library "isn't working" — it is the default rotation-null gate (§14–15) doing exactly what it's designed to do. Look at the single best `PARTIAL-EDGE` candidate on this data:

```python
partial = [(c, r) for c, r in result.rule_responses if r.verdict == "PARTIAL-EDGE"]
partial.sort(key=lambda x: x[1].in_sample_summary.profit_factor, reverse=True)
c, r = partial[0]
print(c.event_expression, "|", c.direction)
print(r.in_sample_summary.profit_factor, r.in_sample_summary.total_trades)
print(r.walk_forward.consistency, r.walk_forward.oos_summary.profit_factor)
print(r.rejection_reasons)
```

Verified output:

```
delta_diffnorm_close_vol12_vol24_6 < -0.899244 | short
16.882 46
1.0 9.721
['active_months 11/20 = 55% < 80%', 'search-level rotation null not cleared (rotation_p=1.0000 > 0.05)']
```

This rule has an outstanding in-sample profit factor (16.9), a walk-forward that is *positive in 100% of test windows*, and an OOS profit factor of 9.7 — and it is still capped at `PARTIAL-EDGE`. The `rejection_reasons` tell you exactly why: it's only active in 55% of the months in its window (below the 80% coverage bar), and — the more important reason — its search-level rotation-null p-value is 1.0, meaning FORGE's own randomized-rotation null test (§14) found that a purely rotated, outcome-decoupled version of the search does just as well or better. This is the library being honest about the size of its own search space, not a false negative.

### What `forge()` did, that you didn't ask it to do explicitly

This is important, and it's the first thing that surprises new users. `forge(kpi, ticker="ADAUSDC", timeframe="1D")` with no further config silently did all of the following:

1. Ran Module 0 (Market Context) since you didn't pass `run_market_context=False` and the table had no `regime` column already.
2. Substituted a **daily-calibrated** `horizon_grid` for Alpha Discovery, because `timeframe="1D"` is daily-or-slower and you passed no explicit `AlphaConfig` — the class's own default grid `(1,2,3,4,6,8,12,16,24,36,48)` is calibrated for roughly-hourly bars, and using it verbatim on daily data would scan holding periods up to 48 *days*.
3. **Enriched** each event's horizon grid with additional points around 0.5×/1×/2× that event's own dominant indicator window (`AlphaConfig.horizon_enrichment`, on by default) — a union with the base grid, never a restriction.
4. Ran the **fast search-level rotation null** (`fast_null=True` by default) and annotated `rotation_p`/`rotation_threshold` on every promoted contract — this is exactly what produced the `PARTIAL-EDGE` cap you just saw.
5. Built a shared, **purged** `TimeBudget` for Event/Alpha Discovery (§15) even though you passed no `time_budget=` argument.
6. Recorded a `HypothesisLedger` on `result.ledger`, tallying how large the session's search surface actually was.
7. Ran Rule Discovery on all 370 promoted contracts with `selection_mode="walk_forward"` (the default) — meaning the published operating parameters came from inside walk-forward train windows only, never from a peek at the final test window.
8. Skipped Module 4 (Rule Registry) — not because you disabled it, but because `RuleRegistry.from_forge_results` needs multiple tickers to say anything about cross-ticker generalisation; with a single-ticker `forge()` call it still runs and produces a registry, but every rule is classified `ISOLATED` (§9).

None of items 2–7 are things you configured. They are all defaults chosen by the library's authors specifically so that the "quick start" path and the "hand-tuned" path don't silently diverge in honesty. §14–15 explain each of these in depth, including which ones you can turn off and what it costs you to do so.

---

## 8. Anatomy of the Workflow

This section walks through the five modules end to end, in enough mechanical detail that you can predict what changing an input will do — not just what each module's public method is called.

### Module 0 — Market Context

**Input:** the KPI Table. **Output:** the same table plus `regime` (an ordered categorical: `STRONG_BEAR < BEAR < NEUTRAL < BULL < STRONG_BULL`) and `regime_stable` (bool).

Internally, the default classifier (`EMAProxyClassifier`) computes `ratio = ema_short / ema_long` and buckets it against four thresholds (fixed-mode defaults `[0.975, 0.990, 1.010, 1.025]`). The interesting part is *where the EMA windows come from*: by default (`auto_window=True`), the module estimates the local half-life of an Ornstein-Uhlenbeck mean-reverting process from the price series itself (via a Hurst/OU regression: `dP_t = const + kappa·P_{t-1} + ε`, `half_life = -log(2)/log(1+kappa)`), then derives `long_period = round(half_life)` and `short_period = round(half_life × 0.435)`. Only if that estimate fails to converge does it fall back to the fixed defaults of 9/25. `mc.window_resolution["source"]` tells you which happened: `"hurst_ou"` (converged), `"fallback"`, or `"configured"` (you set `auto_window=False` and gave explicit windows).

`regime_stable` is `True` only once the current regime has held for at least `stable_window` (default 12) consecutive bars — the first 11 bars of a fresh regime transition are `regime_stable=False`.

**Important:** Event Discovery (Module 1) does not read `regime` at all — it's present in the table but ignored during Steps 0–5. Only Alpha Discovery's regime-sensitivity analysis (Step 5 of its own pipeline) reads it.

### Module 1 — Event Discovery

This is a five-step internal pipeline (`EventDiscovery.run()`), and it's worth knowing the steps by name because error messages and configuration knobs reference them:

1. **Column classification** (`TypeClassifier`) — every non-timestamp column is labeled `CONTINUOUS`, `BINARY` (exactly 2 distinct values), or `CATEGORICAL` (non-numeric, or numeric with ≤ `max_categorical_classes` distinct values, default 20). Continuous columns get a further "scale-free" flag: is this series bounded/intrinsic (RSI, a percentage) or price-level-dependent (a raw EMA)? The heuristic is deliberately conservative — it prefers to miss a scale-free series (false negative) over wrongly calling a price-dependent one scale-free (false positive), because the latter would generate thresholds contaminated by absolute price level.
2. **Feature generation** (`FeatureGenerator`) — derives new columns of arity 2–3: ratios (`a/b`), percentage spreads (`(a-b)/b`), normalized differences (`(a-b)/σ(a-b)`), Bollinger %B, and position-in-range. The bulk of arity-2 combinations pair columns in the same semantic family (two EMAs, not an EMA and an RSI), but the generator also has several **dedicated, narrowly-scoped arity-2 pairings** beyond that general rule, each added to close a specific gap the family rule couldn't reach:
   - **cross-column, cross-time OHLC pairs** — e.g. "today's close above yesterday's low" (every ordered pair of raw OHLC bases present, against a lagged copy, at the same lags used elsewhere for delta transforms); always on.
   - **a MACD line against its own signal line** — matched by shared `(base, fast, slow)`, not the generic same-family grouping, because a line must pair with *its own* signal, never an unrelated MACD configuration's; only fires if you enabled `"macd"` in `build_features()` (disabled by default, §9).
   - **price % change against volume % change at the same lookback** — a price-vs-volume divergence signal (e.g. "new price high on falling volume"); only fires if your KPI Table carries a volume-based return column, which is not part of the default `kpi_builder` config.
   - **`candle_features()`'s six geometry columns against each other, and against `close_natr_N`** — these bare-named columns (`body`, `gap`, …) don't match the `{base}_{indicator}_{period}` convention at all, so without this dedicated pairing they could only ever be used standalone; paired against the *normalized* ATR (`natr`), never the raw one, to avoid reintroducing a price-level dependency.
   - **a price-scale indicator (SMA/EMA/WMA/HMA only) against a lagged raw OHLC base** — e.g. `close_sma_12[t] > low[t-3]`; deliberately restricted to price-scale indicator families (a ratio against RSI/volatility/return/drawdown/Bollinger columns wouldn't be dimensionally sound) and to the `ratio_` operation only. On by default, with its own lag set — `DiscoveryConfig.indicator_lag_cross_lags`, default `(1, 3)` (§10) — distinct from the general delta-transform lag set.

   All of these except the first are genuinely recent additions (landed after most of this manual's Module 1 walkthrough was drafted, and corrected here after re-reading the current source directly, per this manual's own verification discipline, §21). The ones gated on a non-default `kpi_builder` config (MACD, volume returns, ATR/NATR) cost nothing on a default KPI Table; the indicator-vs-lagged-OHLC-base pairing is unconditional and does add measurable cost — quantified in §17.
3. **Temporal transforms** (`TransformLayer`) — every feature from steps 1–2 additionally gets rolling-percentile-rank, rolling-z-score, and simple-delta versions over several window lengths (48/96/168 bars for pctrank/zscore; 1/3/6/12 bars for delta).
4. **Event generation** (`EventGenerator`) — for every (feature, transform) pair, tries a catalog of distributional thresholds (percentiles p3…p97 of that series' own history) and — for z-scored series only — fixed theoretical thresholds (±1.0, ±1.5, ±2.0). Each threshold produces both a "persistent" event (active on every bar the condition holds) and a "crossing" event (active only on the bar where the condition first becomes true).
5. **Consistency Gate** (§10, §15) — every raw event candidate must pass a rate/dispersion filter before it becomes an `EventCandidate`. Gate-passing single events are then combined pairwise (or in triples, if `max_and_components=3`) with AND, and each composition is re-submitted to the same gate — only compositions that themselves pass become AND-composed candidates.

**Nothing in these five steps reads a forward return.** That is the single most important fact about this module.

### Module 2 — Alpha Discovery

Given the candidate list from Module 1, and *for the first time in the pipeline*, this module reads the forward price path. Per event candidate:

1. **Horizon and direction derivation.** For every horizon `h` in the (possibly enriched) grid, it computes `mean_advantage[h]` (the oriented average forward-return advantage of active bars vs. all bars) and a horizon-selection score `|mean_advantage[h]| / sqrt(h)` — a "Sharpe-like" deflation that avoids favoring short horizons just because their t-statistic denominator is smaller. `h* = argmax_h score[h]`. Direction is `"long"` if `mean_advantage[h*] > 0`, `"short"` if `< 0`, and `"undetermined"` — the pipeline's one hard rejection gate — if no horizon in the grid produces a finite, sign-determined advantage at all.
2. **Take-profit derivation.** `sell_pct = max(quantile(MFE, mfe_quantile=0.5), mfe_floor=0.005)`, where MFE is the maximum favorable excursion within `h*` bars of every active bar. This is a *derived* number, not a config input.
3. **Predictive-power measurement (IS):** Information Coefficient (Spearman correlation between the raw feature and the forward return, computed once and cached per `(feature, horizon)`), win rate / lift over base rate, Cohen's d, a one-sided t-test.
4. **OOS confirmation** (when `train_ratio < 1.0`, default is 0.7): the same derived target is replayed on the held-out tail, and passes if it has enough activations, a positive oriented advantage, and a low enough p-value. Failing this is a **non-blocking diagnostic**, not a rejection.
5. **Regime sensitivity** — per-regime IC and win rate, with a `dependency_type` classification (`agnostic`/`conditional`/`specific`/`broken`/`unknown`).
6. **Composite scoring** — a weighted combination of the above metrics into a 0–1 `composite_score`, mapped to a letter grade A (≥0.75) through D (<0.25).
7. **Contract compilation.** All candidates with a determined direction become `AlphaContract` objects with `status="HYPOTHESIS"`; every other metric above only ever appends a string to `diagnostics` — it never blocks promotion, and `rejection_reasons` stays empty on a promoted contract. This is stated as a deliberate design principle: statistical weaknesses "feed the grade, they don't gate/reject — Rule Discovery is the sole economic judge" (`src/forgedge/docs/README.md`, translated).

### Module 3 — Rule Discovery

Given a promoted `AlphaContract` and its originating `EventCandidate`, Rule Discovery simulates a realistic order-execution backtest and validates it out of sample. It does **not** re-optimize the event's thresholds or override the derived target — it uses `derived_target.holding_period_h`/`sell_pct` only as the *center* of an operational parameter grid to search.

The execution mechanics, exactly:

1. On a signal bar, a limit order is placed at `anchor × (1 − buy_drop_pct)` (long) or `anchor × (1 + buy_drop_pct)` (short).
2. If price touches that limit within `buy_delay_bar` bars, it fills; otherwise the order is cancelled (this is where fill rate comes from).
3. After a fill, the position closes at whichever comes first: the first bar closing past the take-profit level, or the close of bar `target_h` after the fill (the "horizon stop").

A subtlety worth internalizing precisely: **`target_h` counts bars *after* the fill bar**, and the signal→fill gap is always exactly 1 bar (you cannot act on a bar's own close before it has happened) — so the total signal-to-exit span is `1 + target_h` bars. `target_h=0` is legal and means "exit at the fill bar's own close," not "no horizon."

The grid of `(buy_drop_pct, sell_pct, target_h, buy_delay_bar)` combinations is screened in-sample, the best-scoring combination becomes the operating point, and that fixed operating point is then re-validated on a rolling walk-forward split (train windows expand or roll, `n_splits` test windows are concatenated into one "honest OOS track record"). Statistical validation on top of this includes a Deflated Sharpe Ratio, temporal-stability check (first-half vs. second-half PF), and regime-dependency breakdown.

The verdict logic (exact gates, from `SelectionCriteria` defaults) is covered in §9 and §15 — the short version: `NON-EDGE` is any hard failure (too few trades, IS PF below 1.5, OOS PF below 1.0, expectancy not significant); `EDGE` requires clearing every one of a stricter set of gates including the rotation-null check; anything that clears the `NON-EDGE` floor but not every `EDGE` gate is `PARTIAL-EDGE`; and `INSUFFICIENT-DATA` (§15) is a fourth verdict that demotes a would-be `EDGE`/`PARTIAL-EDGE` when the pooled OOS evidence is statistically too thin to support the claim, regardless of how good the point estimate looks.

### Module 4 — Rule Registry

Ingests every `EDGE`/`PARTIAL-EDGE` rule submission (NON-EDGE is silently skipped, not an error) into a `RuleDocument`. It computes two correlation matrices across all documents (Jaccard overlap of activation dates; Spearman correlation of gains on a shared, zero-padded date axis), flags — but never deletes — the weaker of any pair of near-duplicate rules (Jaccard ≥ `overlap_threshold`, default 0.70), and replays every rule on every *other* ticker in the session with its absolute thresholds re-percentiled onto that ticker's own distribution. A rule is `GENERIC` if it clears the cross-ticker profit-factor bar on at least `generic_ratio_threshold` (default exactly `2/3`) of the other tickers tested; otherwise `PARTIAL`, `SPECIFIC`, or (if it was flagged a duplicate) `ISOLATED`. The registry is entirely stateless across sessions — the exported flat table or HTML report is the only persistence artefact it produces.

---

## 9. Main API and Components

This section covers the primary, user-facing classes and functions — not an exhaustive listing (that's §26), but enough to write real code against every module. Every signature below is quoted from the current source, not paraphrased.

### The orchestrator

```python
forge(
    kpi_table: pd.DataFrame, *,
    ticker: str | None = None, asset: str = "ASSET", timeframe: str = "1H",
    market_context_config: MarketContextConfig | None = None,
    event_discovery_config: DiscoveryConfig | None = None,
    alpha_config: AlphaConfig | None = None,
    rotation_calibration: RotationConfig | None = None,
    fast_null: bool = True,
    time_budget: TimeBudget | None = None,
    rule_discovery_config: RuleDiscoveryConfig | None = None,
    registry_config: RegistryConfig | None = None,
    manual_events: list[CustomEvent] | None = None,
    run_market_context: bool = True,
    run_rule_discovery: bool = True,
    run_registry: bool = True,
    only_validated_events: bool = False,
    rule_discovery_grades: Iterable[str] | None = None,
    progress: bool = True,
) -> ForgeResult
```

`manual_events` and `event_discovery_config` are mutually exclusive — passing both raises `ValueError`. `ticker` falls back to `alpha_config.asset`, then to `asset`, when not given explicitly.

`ForgeResult` — the return value — carries every intermediate artefact:

| Field | Type | Meaning |
|---|---|---|
| `enriched` | `pd.DataFrame` | KPI Table after Market Context |
| `event_frame` | `pd.DataFrame` | Event Discovery's post-pipeline frame — **pass this, not `enriched`**, to anything that needs derived features |
| `candidates` | `list[EventCandidate]` | every Module 1 output |
| `contracts` | `list[AlphaContract]` | every Module 2 output, promoted *and* rejected |
| `promoted` | `list[AlphaContract]` | the subset with `status="HYPOTHESIS"` |
| `rule_responses` | `list[tuple[AlphaContract, RuleDiscoveryResponse]]` | one pair per promoted contract, if M3 ran |
| `registry` | `RuleRegistry \| None` | Module 4 output, if it ran |
| `calibration` | `CalibrationReport \| None` | the rotation-null report |
| `ledger` | `HypothesisLedger \| None` | search-surface bookkeeping |
| `time_budget` | `TimeBudget \| None` | the effective IS/OOS split used |
| `market_context`, `event_discovery`, `alpha_discovery` | module instances | live objects for drill-down (`.distribution()`, `.summary()`, …) |

Methods: `.edges()` → `(contract, response)` pairs where `response.is_edge` is true; `.validated_rules()`; `.submissions()`; `.summary()` (a `pd.DataFrame`, one row per candidate, augmented with `rule_verdict`).

`forge_multi(frames_by_ticker: dict[str, pd.DataFrame], *, registry_config=None, progress=True, **forge_kwargs) -> tuple[dict[str, ForgeResult], RuleRegistry]` runs `forge()` once per ticker and pools every tradeable rule into one cross-ticker registry — this is the natural way to get genuinely `GENERIC`-classified rules out of Module 4.

### Presets

```python
forge_preset(preset: str, timeframe: str, asset: str = "ASSET",
             train_ratio: float = 0.70, **overrides
             ) -> tuple[DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig]
```

`preset` is one of `PRESETS = ["sniper", "balanced", "sweep", "burst"]`. This returns a *triple* — one calibrated config object per module (M1/M2/M3) — pre-tuned for a search *profile*, and scaled for the timeframe you pass. `preset_info(preset=None)` prints the resolved numeric parameters for one or all presets.

### Module 0 — `MarketContext`

```python
MarketContext(kpi_table: pd.DataFrame, config: MarketContextConfig | None = None)
mc.run() -> pd.DataFrame
mc.distribution()          # bar-share per regime, for diagnostics
mc.window_resolution       # {"source": "hurst_ou" | "fallback" | "configured", ...}
```

### Module 1 — `EventDiscovery`, `EventCandidate`, `CustomEvent`

```python
EventDiscovery(kpi_table: pd.DataFrame, config: DiscoveryConfig | None = None,
                time_budget: TimeBudget | None = None)
ed.run() -> list[EventCandidate]
ed.df           # post-pipeline frame — pass this to AlphaDiscovery, not kpi_table
ed.summary()    # pd.DataFrame, one row per candidate
```

An `EventCandidate`'s important attributes: `event_id`, `expression` (the boolean condition, as a string), `event_formula` (a human-readable rendering), `sql_expression` (a DuckDB-compatible SQL translation of the same condition, useful if you want to evaluate the event outside Python), `components`, `activation_stats` (`n_activations`, `n_active_months`, `zero_months`, `max_monthly_share`, `mean_tpm`), `consistency_gate` (a `GateResult`), `validation` (a `ValidationResult`, only if walk-forward was configured). Its method `.apply(df) -> pd.Series[bool]` deterministically re-evaluates the *stored* thresholds on any new frame — no recalibration, no look-ahead — and `.persist(path)` gives a full pickle round-trip (the only method the docs describe as fully invertible; `.to_dict()`'s JSON form is not).

```python
CustomEvent(formula: str, name: str = "")
```

For manually injecting your own hypothesis (e.g. `CustomEvent("close_rsi_14 < 30")`) instead of running automatic discovery. Formulas are evaluated with `pandas.DataFrame.eval()`. A `CustomEvent` still crosses the Consistency Gate, but a failure only logs a warning — it is never dropped. AND composition is not performed on manually injected events. Use via `forge(..., manual_events=[...])` — mutually exclusive with `event_discovery_config`.

### Module 2 — `AlphaDiscovery`, `AlphaContract`

```python
AlphaDiscovery(kpi_table_or_ed_df: pd.DataFrame, candidates: list[EventCandidate],
                config: AlphaConfig, time_budget: TimeBudget | None = None)
ad.run() -> list[AlphaContract]
ad.promoted_contracts(min_lift: float | None = None) -> list[AlphaContract]
ad.summary()   # pd.DataFrame, sorted by composite score
```

**Critical:** pass `ed.df` (Event Discovery's post-pipeline frame), not the original KPI Table — it already carries the derived ratio/spread/transform columns the event expressions reference.

An `AlphaContract`'s important attributes: `alpha_id`, `status` (`"HYPOTHESIS"`/`"REJECTED"`), `event_candidate_id` (links back to the originating `EventCandidate`), `derived_target` (`holding_period_h`, `sell_pct`, `direction`, `base_rate`, `mean_advantage`), `oos_validation`, `event_stats` (`win_rate`, `lift`, `cohens_d`, `p_value`), `regime_analysis`, `alpha_score` (`composite_score`, `grade`), `rejection_reasons` (blocking causes only — empty on a promoted contract), `diagnostics` (non-blocking observations that feed the grade; routinely non-empty on a promoted contract), `rotation_p`/`rotation_threshold` (set by the search-level rotation null).

### Module 3 — `RuleDiscovery`, `RuleDiscoveryResponse`

```python
RuleDiscovery(event_frame: pd.DataFrame, contract: AlphaContract, candidate: EventCandidate,
               config: RuleDiscoveryConfig | None = None)
resp = rd.run() -> RuleDiscoveryResponse
```

The `event_candidate` you pass must be the one `contract.event_candidate_id` actually points to, or the constructor raises `ValueError`.

#### Overlap — how much capital these numbers take

`run_backtest` opens a position on **every** active bar, with no flat-state
check. That is deliberate and stays that way: it is a legitimate,
capital-permitting policy, and the reported economics are reproducible live
*given enough capital to fund the concurrent positions*. What was missing was
any way to find out how much capital that is (issue #168) — the reports carried
a fixed sentence about "overlapping positions" with no number attached, so a
rule needing 1× the capital of a single position and one needing 12× read
identically.

`BacktestSummary` now measures it on the ledger the published parameters
actually produce:

| field | question it answers |
|---|---|
| `n_episodes` | how often does this signal *fire*? |
| `mean_concurrent_positions` | when it is working, how many positions am I funding? |
| `max_concurrent_positions` | can I deploy this at all on my account? |

`trades` (from `return_trades=True`) carries an `episode_id` per row, so
`trades.groupby("episode_id").size()` is available without reimplementing the
grouping.

**Episodes and concurrency are different measures and generally disagree.**
Episodes group by *signal* — a five-bar `RSI < 30` stretch is one thing
happening, not five. Concurrency groups by *price path* — and trades from
clearly separate episodes still overlap whenever the holding period outruns the
gap between them. On the case in #168: 120 signal bars, 76 episodes, and a mean
of 3.71 concurrent positions.

Which one you want depends on the question: capital sizing → concurrency; how
often a signal fires → episodes; statistical inference → concurrency, because
overlapping trades share a price path and are not independent observations.
`total_trades / mean_concurrent_positions` is the sample size the overlap
actually supports (118 nominal → ≈32 effective in that case). The inferential
consequences are #177's business; this is the measurement it needs.

`forgedge.episodes` exposes the primitives — `episode_starts`, `episode_ids`,
`concurrency` — for callers who want them directly.


#### Entry mode — what the verdict measures

`entry_mode` defaults to **`"auto"`** (it was `"limit"` before #185), and the
change is worth understanding because it moves verdicts.

In `"limit"` mode the grid varies `buy_drop_pct`, so the limit entry does two
jobs at once: order mechanic *and* entry-price optimiser. A deeper discount
fills less often and — this is the part that matters — **only on the paths that
came back down to it**. The profit factor rises on a subset of trades that is
not the tradeable population. That is the *fill confound*, and under `"limit"`
the verdict partly measures the entry price rather than the signal.

`"auto"` splits the two readings apart:

- **Stage 1** evaluates the rule at a market entry (next-open fill, ≈100%). This
  verdict is authoritative. Stage 2 can never turn a `NON-EDGE` into an edge.
- **Stage 2** sweeps `buy_drop_pct` on the survivors, **replays** the winner
  out-of-sample on Stage 1's own test windows, and publishes it only if it
  clears all three adoption conditions.

The adoption conditions, all measured on that replay:

| # | condition | what it stops |
|---|---|---|
| 1 | `fill_rate >= min_fill_rate_opt` | a PF inflated by rare fills |
| 2 | `opportunity_sharpe >= market's` | a point that trades less for a slightly better edge |
| 3 | `net_gain >= min_net_gain_retention × market's` | a tiny µ with a tiny σ |

Condition 2 uses a **different Sharpe from the one on
`StatisticalValidation`**, and the difference is the whole criterion.
`validate()` annualises by *capacity* — `bars_per_year / avg_holding_bars`, how
many non-overlapping holding periods fit in a year. That is the right
denominator for "how good is this rule", because it does not reward a rule for
the accident of firing often. But both operating points sit on the *same* rule
and hold for the same length, so capacity is **identical for the two** and the
`sqrt` factor cancels: the comparison collapses onto the per-trade Sharpe,
which is exactly the opportunity-blind metric the criterion exists to escape.

`opportunity_sharpe` counts realised trades instead — `(µ/σ) × sqrt(trades per
year)` — so halving the trades costs `sqrt(2) ≈ 1.41×` that the per-trade edge
has to beat. On the issue's own worked example:

| | market | limit | ratio |
|---|---|---|---|
| Sharpe per trade | 0.267 | 0.375 | 1.41× |
| annualised by capacity | 5.095 | 7.164 | 1.41× — unchanged |
| `opportunity_sharpe` | 1.461 | 1.299 | **0.89×** |
| total return | 48% | 36% | **0.75×** |

The capacity reading adopts a point that earns a quarter less.

Both points are reported in full on `RuleDiscoveryResponse.entry_optimization`
— each with its own rule, out-of-sample summary and statistics, plus
`failed_condition` naming which condition stopped an adoption. The limit point's
walk-forward is a *replay* (`reoptimise=False`), so it adds no selection and no
`n_trials`; its Deflated Sharpe carries its own larger trial count (Stage 1 +
Stage 2 cells) as an absolute metric, while the `min_dsr` gate always reads the
market point's — the verdict never pays for Stage 2.

`"limit"` remains fully supported and is the right choice when the limit order
*is* the strategy rather than an execution refinement.


A `RuleDiscoveryResponse`'s important attributes: `verdict` (`"EDGE"|"PARTIAL-EDGE"|"NON-EDGE"|"INSUFFICIENT-DATA"`), `is_edge` (true for the first two), `rejection_reasons`, `validated_rule` (carries `.params`, a `BacktestParams`), `in_sample_summary` (`total_trades`, `profit_factor`, `win_rate_pct`, `expectancy`, `tpm_mu`), `execution_envelope` (`.conservative`/`.optimistic` — see §17), `walk_forward` (`.oos_summary`, `.consistency`), `statistical_validation` (`.temporal_stability`, `.deflated_sharpe`), `regime_analysis`, `excursion` (MAE/MFE), `entry_optimization` (only populated when `entry_mode="auto"`, §15).

`from forgedge.rule_discovery import text_report, html_report` build human-readable/HTML reports from a response; `resp.to_dict()` gives a JSON-serializable form.

### Module 4 — `RuleRegistry`

```python
RuleRegistry(submissions: list[RuleSubmission], frames: dict[str, pd.DataFrame],
              config: RegistryConfig | None = None)
RuleRegistry.from_forge_results(results: dict[str, ForgeResult], config=None)   # preferred entry point
reg = registry.run()
reg.summary(); reg.flat_table(); reg.documents; reg.matrices   # .jaccard, .spearman
reg.export("rules.xlsx")             # or .csv, per RegistryConfig.export_format
reg.html_report(timeframe="1H")      # self-contained HTML, inline SVG, no CDN
```

`frames` must be the *post-Event-Discovery* frames (i.e. `ForgeResult.event_frame` per ticker), not raw KPI Tables — the cross-ticker replay needs the derived feature columns the rule expressions reference.

### KPI Builder — building a KPI Table from raw candles

```python
from forgedge import build_features, candle_features, lag_features, pattern_features

kpi = build_features(candles, config=None, *, timestamp_col, output_timestamp_col="open_dt",
                      timestamp_unit="ms", add_color=True, sort_output=True) -> pd.DataFrame
kpi = candle_features(kpi, *, order_on="open_dt", add_gap=True, round_to=5) -> pd.DataFrame
kpi = lag_features(kpi, *cols, periods=(1,2,3), like=None, order_on="open_dt") -> pd.DataFrame
```

`build_features` computes a configurable set of base indicators (SMA, EMA, RSI, Bollinger Bands, ATR, MACD, rolling min/max, returns, volatility, max-drawdown) from raw OHLCV, and derives the `open_dt` timestamp column `forge()` expects. `config` accepts a `dict`, a YAML file path, or `None` for the packaged default. Indicators referencing columns absent from `candles` (e.g. `volume`) are silently skipped with a `logger.warning`, not an error — OHLC-only input is safe. `"atr"` and `"macd"` ship **disabled by default** in that packaged config.

`candle_features` adds six scale-free candlestick-geometry columns (`body`, `upper_wick`, `lower_wick`, `close_pos`, `range_pct`, `gap`), all in `[-1,1]`/`[0,1]` regardless of price level. `lag_features` appends `{col}_prev_{NN}` shifted copies of named or pattern-matched columns.

`pattern_features(df, *, patterns=None, order_on="open_dt", col="candle_pattern")` is a fourth, deliberately **opt-in** function — see §15.

**Column-naming convention that matters:** for a column to be recognised as part of a *same-family ratio pair* by Event Discovery's feature generator, its name must match `{base}_{indicator}_{period}` with `base ∈ {close, high, low, open, volume}` and `indicator ∈ {ema, sma, rsi, dema, tema, wma, hma, mdd, atr, natr}` (or the dedicated Bollinger/volatility/return/MACD naming patterns). A column that doesn't match this convention still works as a standalone feature and can still be reached by one of the *other*, more narrowly-scoped arity-2 pairings described in §8 (cross-time OHLC pairs, MACD-vs-signal, price-vs-volume return, `candle_features()` geometry pairs, indicator-vs-lagged-OHLC-base) — it's specifically the generic same-family grouping it opts out of. If a custom indicator you added never shows up composed with anything in Event Discovery's candidates, check its name against this convention first.

### Configuration coherence — `config_report`

```python
config_report(event_discovery=None, alpha=None, rule_discovery=None,
              registry=None, market_context=None, *, ctx=None, kpi=None,
              timeframe="1H") -> ConfigReport
```

`summary_report` validates the **data**; `config_report` validates the
**configuration**, in the same `Finding` vocabulary. It answers two questions in
one output, because they only make sense together: *with what configuration am I
about to run*, and *is that configuration internally satisfiable*.

Both it and `forge()` go through the **same resolver**, so what the report shows
is by construction what the pipeline will execute — `rep.configs` are the actual
objects, not a reconstruction. It never raises, never warns and never mutates
what it was handed.

```python
rep = config_report(disc, alpha, rd, kpi=kpi, timeframe="1D")
print(rep.to_text())          # the resolution trace, then the diagnostics
if rep.has_critical:
    raise ValueError(rep.one_line())
```

Thirteen constraints, each relating materialisations of one latent parameter.
Three are `FAIL` — reserved for a configuration that makes a stage
**structurally incapable** of producing a verdict: `wf_bucket_too_short`
(issue #173), `m1_oos_fold_too_short`, `oos_span_too_short`. The other ten are
`WARN`. Every message carries the value to set, not just the failure.

> **`forge(strict=True)` is the default, and this is a behaviour change.** A
> `FAIL` now raises `ValueError` instead of running. Such a run cannot tell you
> anything: every candidate is eliminated for configuration reasons and the
> resulting wall of rejections is indistinguishable from "the signal is bad",
> which is what you were trying to measure. Pass `strict=False` to downgrade
> everything to `UserWarning` and run anyway. Non-critical incoherences are
> always warnings, never errors. **No verdict changes** — what changes is that
> some runs no longer start.

> **Known: `forge_preset("balanced", "1D")` is currently flagged.** At stock
> preset values `min_train_months=6 × criteria.min_tpm=0.80 = 4.8` against a
> floor of 10 — the audit's F2, and the reason daily-data users see mass
> early-elimination. It is fixed by deriving `min_train_months` from the rate
> (issue #177); until then, daily preset runs need `strict=False`.

#### What the resolver fills in

A field left alone is not a value: it is a question the session answers. These
are the fields whose default now comes from the session rather than from a class
body — set any one of them and every module that reads the same quantity
follows.

| latent parameter | fields it materialised as | resolved default |
|---|---|---|
| the timestamp column | `timestamp_col` on M1 / M2 / M3 / M4 | `"open_dt"` |
| the price series | `AlphaConfig.close_col`, `BacktestParams.{target_col, buy_price_anchor}` | `"close"` |
| the regime columns | `AlphaConfig.{regime_col, regime_stable_col}` | `"regime"` / `"regime_stable"` |
| the cost basis | `AlphaConfig.fee_per_side`, `BacktestParams.fee` | `0.002` |
| the genericity bar | `RegistryConfig.{cross_pf_threshold, min_cross_pf_retention}` | `1.5` / `0.8` |

Two of these were live bugs rather than tidiness. `AlphaConfig.fee_per_side`
stamped the contract while `BacktestParams.fee` charged the backtest and nothing
connected them, so `AlphaConfig(fee_per_side=0.0005)` produced contracts
documenting 5 bp and a backtest charging 20 — silently, since the two agreed
only by sharing a default. And `forge_preset(timestamp_col="ts")` configured M1
alone, so M2 failed later asking for a value you believed you had already given.

Propagation is not symmetric with seeding, in one deliberate place, and the
reason is worth stating precisely. `buy_price_anchor` is **not a schema field**:
it names the *reference level* the limit offset is applied to —
`buy_price = anchor × (1 ∓ buy_drop_pct)` — and any numeric column on the candle
table is legal there, including a derived indicator. `buy_price_anchor=
"close_sma_3"` with `buy_drop_pct=0.10` is how you say *"place a limit at 90% of
the 3-bar SMA"*; the engine has no other way to express it.

So the anchor is *filled in* from the price column — its default reference level
is the close, and renaming the column has to carry that along — but it never
*seeds* the context, and it is not checked for equality against `close_col`.
Seeding from it would push `"close_sma_3"` back out into `AlphaConfig.close_col`
and have M2 measure forward returns on a moving average. `target_col` is
different — the horizon exit must be priced on the series M2 measured returns on
— so a disagreement there is reported.

> **Genericity is now a transfer test, not a quality test.**
> `cross_pf_threshold` used to default to `2.0` independently of M3, while
> `partial_min_profit_factor` admits rules at `1.5` — so a `PARTIAL-EDGE` rule
> had to do *better* away from home than at home to be called generic, and the
> entire class was excluded from genericity by construction. The verdict is now
> `PASS ⟺ pf ≥ floor AND pf ≥ retention × pf_home`: the absolute half asks *is
> it tradeable there*, the relative half asks *does it transfer*. Quality stays
> on the M3 verdict and the grade, where the registry already records it.
> `CrossTickerResult.bar` reports the number each verdict was measured against.

### Data quality — `summary_report`

```python
summary_report(df: pd.DataFrame, *, timestamp_col="open_dt",
                price_cols=("open","high","low","close"), timeframe=None,
                return_high_move=0.5, top_n=5, verbose=True,
                return_report=False) -> DataQualityReport | None
```

A cheap, **purely advisory** diagnostic pass over price columns — it never raises and never blocks the pipeline. It checks schema/NaNs/infinities, price-scale consistency (mixed magnitudes — a common symptom of a data-feed bug), OHLC internal consistency, return outliers (a MAD-based robust z-score plus an absolute-move threshold), and time continuity (gaps, duplicate or out-of-order timestamps). Every finding is a `Finding(level, code, message)` with `level ∈ {"OK","WARN","FAIL"}`; `DataQualityReport` exposes `.worst`, `.has_critical`, `.has_warnings`, `.one_line()`, `.to_text()`, and the full `.findings` list.

### Time budget, hypothesis ledger, calibration

```python
TimeBudget.build(n_bars: int, train_ratio: float = 0.7, horizon_bars: int = 0,
                  purge_bars: int | None = None, embargo_bars: int = 0) -> TimeBudget
```

`HypothesisLedger` (`result.ledger`) — plain bookkeeping (`m1_candidates`, `m2_horizons`, `m2_promoted`, `m3_grid_cells`, `m2_surface`, `total_surface`), not a correction mechanism.

```python
FastRotationNull(event_frame, candidates, alpha_config, time_budget=None).run(promoted) -> CalibrationReport
RotationCalibrator(event_frame, candidates, alpha_config, time_budget=None).run(promoted, RotationConfig(...)) -> CalibrationReport
```

Both are covered in depth in §14–15.

---

## 10. Configuration

Every module accepts a dataclass carrying its knobs. This section covers the ones you're most likely to actually tune, with defaults quoted exactly from source.

### `DiscoveryConfig` (Module 1)

| Field | Default | Meaning |
|---|---|---|
| `gate_params` | `GateParams()` | Consistency Gate thresholds — see below |
| `max_categorical_classes` | `20` | above this many distinct values, a non-numeric column is dropped, not one-hot-encoded |
| `timestamp_col` | `"open_dt"` | |
| `max_and_components` | `2` | `1`=singles only, `2`=+pairs, `3`=+pairs+triples |
| `train_ratio` | `1.0` | `<1.0` reserves a tail for Module 1's own (optional) walk-forward validation |
| `walk_forward` | `None` | set a `EventWalkForwardConfig` to enable event-level OOS validation (§15 — opt-in) |
| `diversity_gate_enabled` | `False` | opt-in near-duplicate suppression (§15) |
| `diversity_threshold` | `0.85` | Jaccard bar for the diversity gate, when enabled |
| `indicator_lag_cross_lags` | `(1, 3)` | lag set for the price-scale-indicator-vs-lagged-OHLC-base feature pairing (§8); pass `()` to disable that pairing entirely |

`GateParams` (the Consistency Gate):

| Field | Default | Meaning |
|---|---|---|
| `min_tpm` | `0.5` | minimum average triggers per month (unit depends on `event_counting`) |
| `max_dispersion` | `1.5` | maximum allowed Index of Dispersion (Var/Mean of monthly counts) |
| `event_counting` | `"episode"` | `"episode"` counts maximal runs of consecutive activations; `"bar"` counts every individual bar (§15) |
| `min_episodes` | `10` | absolute floor on episode count, `"episode"` mode only |
| `episode_gap` | `1` | max bar-gap that still belongs to the same episode |

### `AlphaConfig` (Module 2)

| Field | Default | Meaning |
|---|---|---|
| `horizon_grid` | `(1,2,3,4,6,8,12,16,24,36,48)` | **hourly-calibrated** — see the daily-data pitfall in §21 |
| `train_ratio` | `0.7` | IS/OOS split for Alpha Discovery's own confirmation |
| `embargo_bars` | `0` | opt-in extra OOS buffer, §15 |
| `horizon_enrichment` | `(0.5, 1.0, 2.0)` | on by default; adds horizons around each event's own dominant window, §15 |
| `thresholds` | `PromotionThresholds()` | statistical thresholds that drive the grade, not a hard gate (except direction) |
| `fee_per_side` | `0.002` | recorded for Rule Discovery **and charged by it** — not applied here (M2 does not net fees out), but it is the same value, no longer an independent copy |
| `target_mode` | `"proj"` | excess-over-trend scoring by default for long events, §15 |
| `trend_sma_mult` | `2.0` | trend SMA window multiplier for `target_mode="proj"` |
| `use_stable_regime_only` | `False` | opt-in, restricts regime analysis to `regime_stable=True` bars |

`PromotionThresholds`: `ic_min_abs=0.02`, `ic_max_p=0.05`, `min_lift=0.08`, `min_cohens_d=0.15`, `use_fdr=True`, `fdr_q=0.10`, `oos_max_p=0.10`, `min_direction_t=0.5`, `require_significant_direction=True`. **Only `direction=="undetermined"` blocks promotion** — every other threshold here informs the A–D grade.

### `RuleDiscoveryConfig` (Module 3)

| Field | Default | Meaning |
|---|---|---|
| `base_params` | `BacktestParams()` | seed operating point |
| `grid` | auto-built | search grid around the contract's derived target when left empty |
| `walk_forward` | `RuleWalkForwardConfig()` | `n_splits=4`, `min_train_months=6`, `purge_bars=None` (→ horizon under test) |
| `criteria` | `SelectionCriteria()` | verdict gates — see §15 |
| `entry_mode` | `"auto"` | `"auto"`/`"market"`/`"limit"`, §15 |
| `selection_mode` | `"walk_forward"` | operating point chosen inside train windows only; `"full_sample"` is the legacy behavior |

`BacktestParams` defaults: `direction="long"`, `buy_type="limit"`, `buy_drop_pct=0.010`, `buy_delay_bar=6`, `sell_pct=0.040`, `target_h=24`, `fee=0.002`, `early_stopping=True`.

`SelectionCriteria` defaults you're most likely to touch: `min_profit_factor=2.0`, `min_win_rate=0.55`, `min_tpm=2.0` (the sole frequency gate — the executed-trade floor is `max(10, n_months × min_tpm)`, not a fixed count), `min_fill_rate=0.40`, `min_dsr=1.0`, `max_rotation_p=0.05`, `power_gate=True` (§15), `early_elimination=True` (set `False` to force full diagnostics even on a fast-screened `NON-EDGE`).

### `RegistryConfig` (Module 4)

`overlap_threshold=0.70` (Jaccard dedup bar), `cross_pf_threshold=1.5` + `min_cross_pf_retention=0.8` (the two halves of the cross-ticker `PASS` — absolute floor, and fraction of the home PF retained), `generic_ratio_threshold=2/3` (**pass this as `2/3`, not `0.67`** — a rule passing exactly 2-of-3 tickers has ratio `0.6666...`, which clears `>= 2/3` but not `>= 0.67`; the docs flag this precision issue explicitly), `export_format="excel"`.

### Configuring via presets instead

Rather than assembling `DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig` by hand, `forge_preset(preset, timeframe, asset, train_ratio=0.70, **overrides)` returns all three pre-tuned for a named search *profile*:

```python
from forgedge import forge, forge_preset

disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="ADAUSDC")
result = forge(kpi, ticker="ADAUSDC", timeframe="1D",
               event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
               rule_discovery_config=rd_cfg)
```

| Preset | Profile |
|---|---|
| `"sniper"` | Rare, regular, high-precision events, simple rules. Needs a long IS window (≥2 years on 1D). Do **not** pair with the rotation calibrator (too few events to calibrate against). |
| `"balanced"` | Moderate frequency, sensible default for most assets/timeframes. |
| `"sweep"` | Wide, permissive search — designed to pair with `rotation_calibration=RotationConfig(k>=100)` and a `min_lift` filter downstream. |
| `"burst"` | Time-concentrated events (regime-change, momentum). High dispersion explicitly tolerated. |

`overrides` accepted by name: M1 side — `min_tpm`, `max_dispersion`, `max_and_components`, `timestamp_col`, `event_counting`; M2 side — `min_lift`, `min_cohens_d`, `fdr_q`, `oos_max_p`, `horizon_grid`, `bars_per_day`; M3 side — `rd_min_tpm`. An unrecognized override key raises `TypeError`.

---

## 11. Error Handling

`forgedge` has **no custom exception hierarchy** — every raised error is a plain Python built-in (`ValueError`, `KeyError`, `RuntimeError`, `TypeError`, `ImportError`, `FileNotFoundError`). There is nothing `forgedge`-specific to catch selectively; catch the built-ins.

### Validation philosophy

There is **no single up-front schema validation** of the KPI Table. `forge()` itself does not call `summary_report` or check columns before starting — validation is lazy and distributed: each module validates the specific columns/timestamp source it needs, exactly when it needs them, and fails fast with `ValueError`/`KeyError` if they're missing. The one deliberate exception is `kpi_builder.build_features()`, which silently *skips* (with a `logger.warning`, not an exception) any indicator whose required input columns aren't present — so OHLC-only candles are always safe to pass in, even against a config that also asks for volume-based indicators.

`summary_report()` (§9) is the library's answer to "I want to validate before I commit to a run" — but it is entirely **opt-in**: it never raises, never blocks, and is never called automatically. If you want a hard stop on bad data, you write that check yourself:

```python
rep = summary_report(kpi, return_report=True, verbose=False)
if rep.has_critical:
    raise ValueError(f"Fix data issues first: {rep.one_line()}")
```

### Common exceptions, verified

These are not paraphrased — each was actually triggered and its message captured.

```python
from forgedge import forge, CustomEvent, DiscoveryConfig

forge(kpi, manual_events=[CustomEvent("close < 50")], event_discovery_config=DiscoveryConfig())
# ValueError: manual_events and event_discovery_config are mutually exclusive.
# Pass one or the other, not both.
```

```python
from forgedge import AlphaDiscovery, AlphaConfig

ad = AlphaDiscovery(kpi, [], AlphaConfig())
ad.promoted_contracts()
# RuntimeError: Call run() before promoted_contracts().
```

This "call `.run()` first" `RuntimeError` pattern is consistent across `MarketContext.distribution()`, `EventDiscovery.summary()`, `AlphaDiscovery.summary()`/`.promoted_contracts()`, `RuleDiscovery.grid_summary()`, and `TargetOptimizer.validate_oos()`/`.discover_alpha()` — if you see it, you called an accessor before the corresponding `.run()`.

```python
from forgedge.event_discovery.models import GateParams
GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)
# TypeError: GateParams.__init__() got an unexpected keyword argument 'min_act'
```

This one is worth calling out specifically: `min_act`/`min_months`/`max_conc` were fields of an **older** `GateParams` API. Several of the scripts under `examples/` in this repository (`alpha_discovery_usage.py`, `extended_usage.py`, `kpi_table_1d.py`, `search_rotation_calibration.py`, `lowfreq_null_diagnostic.py`, `lowfreq_endpoint_diagnostic.py`) still construct `GateParams` this old way and **will raise this exact `TypeError` if you run them against the current library version, as installed in this repository**. This manual flags it explicitly rather than silently working around it: if you copy code from those example scripts, translate `GateParams(min_act=..., min_months=..., max_conc=..., min_tpm=...)` to the current fields (`min_tpm`, `max_dispersion`, `event_counting`, `min_episodes`, `episode_gap` — §10). `examples/kpi_builder_usage.py` does **not** have this problem — it was verified to run correctly end to end against the current API (§13).

### Summary table of what raises what

| Exception | Typical trigger |
|---|---|
| `ValueError` | invalid enum-like string (`direction`, `target_mode`, `buy_type`, `entry_mode`, `selection_mode`, `threshold_mode`, `timeframe`, `preset`, …), out-of-range numeric config field, mutually-exclusive `forge()` arguments, mismatched contract/candidate pair passed to `RuleDiscovery` |
| `KeyError` | a required column is missing — OHLC columns, `timestamp_col`, `source_col`, an unknown candlestick pattern name |
| `RuntimeError` | an accessor called before `.run()` |
| `TypeError` | wrong input type to `build_features`/`lag_features`, or an unrecognized `forge_preset(**overrides)` key |
| `ImportError` | `load_kpi_config()` called with a YAML path but PyYAML isn't installed |
| `FileNotFoundError` | `load_kpi_config()` given a path that doesn't exist |

### Warnings you should not ignore

`forgedge` uses `warnings.warn` (not exceptions) for situations that are valid but probably not what you meant:

- **`UserWarning` — stale hourly `horizon_grid` on daily-or-slower data.** Fired by `forge()` when you pass an explicit `AlphaConfig` that still carries the untouched hourly-calibrated default grid on a `timeframe` of a day or longer.
- **`UserWarning` — observed-candle index mismatch.** Fired by `AlphaDiscovery`, `RuleDiscovery`, and Rule Registry ingestion when the frame you hand them has a different index than the event's cached training activation series — meaning the event gets re-evaluated via `.apply()` instead of reusing the cache. A second, more serious variant of this same warning fires when the re-evaluated activation count collapses to under 10% of the training count — this is a strong signal you're about to get `direction="undetermined"` because rolling-transform baselines (pctrank, z-score) have shifted (§21).
- **`DeprecationWarning`** — the legacy `TargetConfig.min_lift` field (superseded by `min_lift_atoms`/`min_lift_result`) and a legacy `TypeClassifier` constructor argument (`scale_free_drift_threshold`).

### Degraded-but-non-fatal behavior worth knowing

Some conditions are neither errors nor warnings — they change behavior silently, logged only at INFO/DEBUG or recorded as a diagnostic string:

- A `CustomEvent` that fails the Consistency Gate is **kept**, not dropped — only `logger.warning`'d.
- `binary_target(..., target_mode="proj")` reverts to `"abs"` when there isn't enough history for the trend-SMA warmup, logged at `WARNING`.
- `RuleDiscoveryConfig(selection_mode="walk_forward")` silently falls back to full-sample selection when the data span is too short for even one walk-forward split — recorded as a note in the response, not raised.

---

## 12. Progressive Use Cases

### Use Case 1 — Hello World: discovering one event

Already covered in full in §6. The paradigm to take away: **Event Discovery finds structure without ever looking at what happens next.** Everything else in the library builds on top of that one event.

### Use Case 2 — A realistic single-asset workflow, from raw candles

A more complete scenario: you have raw OHLCV candles (not yet a KPI Table), you want indicators computed for you, and you want to run the full statistical pipeline through Alpha Discovery. This is `examples/kpi_builder_usage.py` from this repository, **verified to run correctly, unmodified, against the current library**:

```python
from forgedge import (
    build_features, candle_features, lag_features,
    summary_report, forge, forge_preset,
)

# candles: a DataFrame with open_time (epoch ms), open, high, low, close, volume
summary_report(candles, timeframe="1D")   # opt-in pre-flight check, never blocks

DEMO_CONFIG = {
    "ema":             {"enabled": True, "params": {"periods": [9, 25, 50], "columns": ["close"]}},
    "rsi":             {"enabled": True, "params": {"periods": [14], "columns": ["close"]}},
    "bollinger_bands": {"enabled": True, "params": {"periods": [20], "columns": ["close"]}},
    "min":             {"enabled": True, "params": {"periods": [24], "columns": ["close"]}},
    "max":             {"enabled": True, "params": {"periods": [24], "columns": ["close"]}},
    "return":          {"enabled": True, "params": {"periods": [1, 6, 24], "columns": ["close"]}},
}

kpi = build_features(candles, DEMO_CONFIG, timestamp_col="open_time")
kpi = candle_features(kpi)
kpi = lag_features(kpi, "close", "color", like="_ema_", periods=[1, 2, 3])

disc, alpha, rd = forge_preset("balanced", timeframe="1D", asset="DEMO")
result = forge(kpi, ticker="DEMO", timeframe="1D",
               event_discovery_config=disc, alpha_config=alpha,
               run_rule_discovery=False, progress=False)

print(f"M1 candidates = {len(result.candidates)}  M2 promoted = {len(result.promoted)}")
```

**Verified output** (running this script as-is, on 2000 bars of synthetic candle data generated by the script itself):

```
build_features  : (2000, 21)  (base indicators + open_dt + color)
candle_features : (2000, 27)  (+ body, upper_wick, lower_wick, close_pos, range_pct, gap)
lag_features    : (2000, 42)  (+ 15 *_prev_NN columns)
forge(kpi)      : M1 candidates = 5015  M2 promoted = 1091
```

Interpretation: `build_features` turned 6 raw columns into 21 (the requested indicators plus `open_dt` and `color`). `candle_features` added 6 scale-free geometry columns. `lag_features` added 15 shifted copies. The resulting 42-column KPI Table produced over 5000 event candidates and just over a thousand promoted contracts — on **synthetic random-walk data**, which is itself an instructive result (§16, §21): a random walk *should* produce a large number of statistically "significant-looking" but economically meaningless candidates, which is exactly why Rule Discovery and the rotation null exist downstream.

### Use Case 3 — Real repository data: ADA and AMZN

Covered fully in §7 (ADA quick start) and §13 (both datasets in depth, including the AMZN raw-CSV cleaning workflow).

### Use Case 4 — Advanced configuration

Four configuration axes you're likely to actually need to change, each shown as a real diff from the default:

**a) Loosen the Consistency Gate for a shorter or lower-frequency dataset.**

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import GateParams

# Default GateParams(min_tpm=0.5, max_dispersion=1.5) is already fairly
# permissive; raising min_tpm trades away rare/marginal events for
# statistical power per event (see §16's frequency-vs-selectivity trade-off).
config = DiscoveryConfig(gate_params=GateParams(min_tpm=1.5, max_dispersion=2.0))
ed = EventDiscovery(kpi, config=config)
```

**b) Switch Rule Discovery's entry model from limit to market orders.**

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig

# "limit" (default) can suffer a "fill confound": a deep, rarely-filled limit
# can show an inflated PF on a small, non-representative subset of trades.
# "market" isolates the signal's edge from the entry mechanism entirely.
config = RuleDiscoveryConfig(entry_mode="market")
resp = RuleDiscovery(event_frame, contract, candidate, config=config).run()
```

**c) Run the full, sampled rotation calibrator instead of the default fast one — with the `"sweep"` preset, as the docs recommend pairing them.**

```python
from forgedge import forge, RotationConfig

disc, alpha, rd = forge_preset("sweep", timeframe="1D", asset="ADAUSDC")
result = forge(kpi, ticker="ADAUSDC", timeframe="1D",
               event_discovery_config=disc, alpha_config=alpha, rule_discovery_config=rd,
               rotation_calibration=RotationConfig(k=100))   # supersedes fast_null
promoted = result.alpha_discovery.promoted_contracts(min_lift=0.05)
```

**d) Switch Alpha Discovery's binary-target scoring from trend-excess (`"proj"`, the default) to absolute return (`"abs"`).**

```python
from forgedge import AlphaConfig
config = AlphaConfig(asset="ADAUSDC", timeframe="1D", target_mode="abs")
```

For each: what changes, why it exists, and the cost of changing it are covered per-item in §15 (opt-in behaviors) and §16 (trade-offs) — this section is the "how," those are the "why."

### Use Case 5 — Errors and problematic data

**Invalid data: an empty table.**

```python
import pandas as pd
from forgedge import summary_report

rep = summary_report(pd.DataFrame(columns=["open", "high", "low", "close"]),
                      verbose=False, return_report=True)
print(rep.worst, [f.code for f in rep.findings])
# FAIL ['empty']
```

`summary_report` never raises — it reports. If your application needs a hard stop, you write it: `if rep.has_critical: raise ValueError(rep.one_line())`.

**A genuinely stale, incompatible configuration (a real bug you can reproduce today).**

```python
from forgedge.event_discovery.models import GateParams
GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)
# TypeError: GateParams.__init__() got an unexpected keyword argument 'min_act'
```

This is the old `GateParams` API, which several of this repository's own `examples/*.py` scripts still use. See §11 for the full explanation and which scripts are affected.

**Mutually exclusive configuration.**

```python
from forgedge import forge, CustomEvent, DiscoveryConfig
forge(kpi, manual_events=[CustomEvent("close < 50")], event_discovery_config=DiscoveryConfig())
# ValueError: manual_events and event_discovery_config are mutually exclusive.
```

**The most common silent "no results" trap: `direction="undetermined"`.**

This isn't an exception — it's a contract with `status="REJECTED"` and one specific rejection reason. It's the single hard rejection gate in Alpha Discovery (§8, §9), and by far the most common reason a promising-looking event never becomes a usable rule. It happens when no horizon in the grid produces a finite, sign-determined average forward advantage — commonly because:

- the horizon grid genuinely doesn't cover the timescale at which the event has an effect (mitigated, but not eliminated, by `AlphaConfig.horizon_enrichment`, on by default — §15);
- you fed Alpha Discovery a frame whose observed index differs from the event's training activation series (§11's second `UserWarning`), so the re-evaluated activation count collapsed and there's no statistical power left to determine a sign at all — a scenario covered concretely in §21.

```python
rejected = [c for c in result.contracts if not c.promoted]
for c in rejected[:3]:
    print(c.event_candidate_id, c.rejection_reasons)
# every rejected contract's rejection_reasons contains "no derivable target"
# — the only hard-gate rejection reason this module produces
```

**A non-conforming feature-column name.**

```python
kpi["my_custom_signal"] = ...   # doesn't match {base}_{indicator}_{period}
```

This does *not* raise or warn. `my_custom_signal` still works as a standalone Event Discovery feature — it simply never gets paired into a ratio/spread feature with another column (§9). If you expected to see `ratio_my_custom_signal_something` in the candidate list and it's not there, this naming convention is the first thing to check, not a bug report.

### Use Case 6 — A realistic application: a monitoring service around a discovery session

This sketches how a session's output becomes something a larger system consumes, using only real, verified `forgedge` calls (the "wrapping" code — `MonitoringService`, the storage class — is illustrative application code, not part of the library; §19 is explicit about that boundary).

```python
import pandas as pd
from forgedge import forge, RuleSpec, rule_performance_report

class MonitoringService:
    """Wraps one forge() discovery session and exposes it for ongoing monitoring."""

    def __init__(self, kpi_table: pd.DataFrame, ticker: str, timeframe: str):
        self.result = forge(kpi_table, ticker=ticker, timeframe=timeframe, progress=False)
        # RuleSpec.from_forge_result(): one spec per tradeable (EDGE/PARTIAL-EDGE) rule,
        # each carrying the params + candidate needed to replay it later.
        self.specs = RuleSpec.from_forge_result(self.result)

    def published_rules(self) -> list[str]:
        return [s.name for s in self.specs]

    def health_report_html(self, fresh_candles: pd.DataFrame) -> str:
        # Replays every published rule deterministically on fresh_candles via the
        # same EventCandidate.apply() path Rule Discovery itself uses — fresh_candles
        # need not be the discovery table.
        return rule_performance_report(self.specs, fresh_candles,
                                        title=f"{self.result.ticker} monitoring")


svc = MonitoringService(kpi, ticker="ADAUSDC", timeframe="1D")
print(f"{len(svc.published_rules())} published rules: {svc.published_rules()[:3]} ...")
html = svc.health_report_html(kpi)   # in production: discovery table + genuinely new bars
```

**Verified output** on the ADA fixture: `54 published rules: ['RULE_ADA_01', 'RULE_ADA_02', 'RULE_ADA_03'] ...`, and a generated HTML report of ~5 MB (inline SVG charts, no external resources — safe to store or email as a single file). `RuleSpec` names follow the same `RULE_{TICKER}_{NN}` convention Module 4 uses internally, even when you're not going through `RuleRegistry` at all.

---

## 13. Working with the Data in This Repository

Two real datasets ship with this repository, plus one derived artefact. Neither is a toy — both are genuine market data.

### `tests/fixtures/ADA_1D_TRAIN.parquet`

882 daily OHLCV bars of ADAUSDC (Cardano), 2024-01-01 to 2026-05-31, with 22 additional precomputed indicator columns already attached: `close_ret_{03,12,96}` (returns), Bollinger Bands (`close_bb_{mid,upper,lower,width}_20`), a rolling max-drawdown (`close_mdd_48`), EMAs at two windows on both `close` and `low` (with three lagged copies each, `_prev_01..03`), and rolling volatility (`close_vol_{05,12,24}`). This is the exact fixture the library's own `tests/test_golden.py` regression suite pins its expected values against — meaning if you run `forge()` on it with the arguments shown in §7, you are reproducing part of the library's own test suite by hand.

```python
import pandas as pd
kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
```

No preprocessing needed — this table is already a valid KPI Table (has `close`, has `open_dt` as `datetime64[ns]`, chronologically sorted). This is the dataset used throughout §7 and §12's Use Cases 4–6.

### `examples/data/AMZN_1D.csv`

1378 daily bars of AMZN (Amazon), an export from a financial-data provider in a common but **not directly `forge()`-compatible** raw format:

```python
import pandas as pd
raw = pd.read_csv("examples/data/AMZN_1D.csv")
print(raw.columns.tolist())
# ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
print(raw.head(2))
#          Date   Price    Open    High     Low     Vol. Change %
# 0  06/30/2026  238.71  237.50  241.53  237.57   31.50M   -0.60%
# 1  06/29/2026  240.14  234.22  249.71  233.80   77.62M    3.20%
```

Note what's wrong with this table, from `forgedge`'s point of view: the close price is called `"Price"`, not `"close"`; volume is a **string** with a unit suffix (`"31.50M"`); dates are **descending** (most recent first) and stored as `MM/DD/YYYY` strings, not a `datetime64` column or index. This is realistic, and it is exactly the kind of table `summary_report` is meant to catch problems in before you waste a `forge()` run on it. The repository's own worked walkthrough (`examples/forge_amzn_walkthrough.ipynb`), which this manual verified cell-by-cell, shows the necessary cleaning:

```python
raw.columns = [c.strip().lower().replace(".", "").replace(" ", "_") for c in raw.columns]
raw = raw.rename(columns={"price": "close", "vol": "volume", "change_%": "chg_pct"})

raw["open_dt"] = pd.to_datetime(raw["date"], format="%m/%d/%Y")
raw = raw.sort_values("open_dt").reset_index(drop=True)   # ascending order — required

for col in ["open", "high", "low", "close"]:
    raw[col] = pd.to_numeric(raw[col].astype(str).str.replace(",", ""), errors="coerce")

raw["volume"] = (raw["volume"].astype(str).str.replace("M", "e6").str.replace("B", "e9"))
raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")

candles = raw[["open_dt", "open", "high", "low", "close", "volume"]]
```

After cleaning, the notebook runs `summary_report(candles, timestamp_col="open_dt", timeframe="1D", return_report=True)` and — this is worth quoting directly because it is a genuine, real finding, not a manufactured teaching example — it surfaces a real anomaly: *(translated from the notebook's own markdown)* "the **last bar** of the dataset (2026-06-30, 'today's bar' at the moment of extraction) has `open` slightly below the recorded `low` — a typical symptom of a still-**incomplete** bar (the market session was still open when the CSV was downloaded)." This is a genuinely useful, real-world illustration of why you run `summary_report` before, not after, a discovery session: an incomplete trailing bar is not a reason to distrust your whole dataset, but it is a reason to consider dropping that one row before running Event Discovery, since a partial bar's OHLC relationship violates the assumption every other bar in the table satisfies.

The notebook then proceeds through `build_features()` (with `atr`/`macd` explicitly enabled, unlike the demo config in Use Case 2), `forge_preset("sweep", ...)`, and a full `forge()` call, landing — on real AMZN data — at exactly **2 `PARTIAL-EDGE`** signals and **zero** full `EDGE`, out of 3035 M1 candidates and 508 M2 promoted contracts. The notebook's own interpretation of this (§14 covers the "why" in depth) is explicit and worth quoting: *(translated)* "The honest result: almost no edge, and that's the right outcome."

### `kpi_table_1d.csv` (repository root)

A pre-generated **output artefact**, not an input dataset — it's what `examples/kpi_table_1d.py` produces after comparing FORGE's own event-discovery path against the standalone `TargetOptimizer` (§9) on ADA daily data across three time windows (2024 in-sample, 2025 and 2026 out-of-sample). Its 19 columns (`ID, Pipeline, EVENT_ID, EXPRESSION, IS 2024_WR, IS 2024_PF, ...`) are a worked example of comparing two discovery strategies' out-of-sample decay side by side — useful to read as a *reference output shape*, not something you need to regenerate to use this manual.

### Notebooks

`notebooks/01_event_discovery.ipynb` through `06_rule_registry.ipynb` walk each module individually and in depth; `notebooks/hurst.ipynb` is a dedicated deep-dive on the Hurst/OU half-life estimation Market Context uses for automatic EMA window selection (§8). `examples/forge_amzn_walkthrough.ipynb` (used throughout this section) is the only one built entirely around a dataset shipped in this repository and is the closest thing to a canonical, fully-reproducible, real-data walkthrough.

---

## 14. Design Choices

This section distinguishes three categories throughout, marked explicitly: **(documented)** — the authors' own stated reasoning, quoted or closely paraphrased from repository documentation; **(measured)** — a real, reported result from a run on real data, not an invented benchmark; **(inferred)** — this manual's own reading of the code, clearly flagged as such, never presented as a fact the authors stated.

### The three-domain separation is architectural, not conventional (documented)

Already introduced in §2. The repository's architecture guide is explicit that this is a constraint, not a style choice, and gives the specific reason: keeping the forward return out of Event Discovery *"elimina una categoria intera di look-ahead bias"* — eliminates an entire category of look-ahead bias, by construction, rather than by discipline.

### "Rigidity upstream buys freedom downstream" (documented)

A trade-off stated explicitly, in the authors' own words, in `src/forgedge/docs/README.md`: *(translated)* "Upstream rigidity is the guarantee that what reaches Alpha Discovery is a genuine measurement — not an artefact of optimization against the target. **It buys the freedom to explore freely downstream**, because the underlying thresholds were never contaminated by any economic choice." This is the rationale for why Event Discovery's thresholds are immutable: the cost (you can't tune a threshold after the fact, even if you're sure a slightly different one would work better) is what makes everything downstream of it trustworthy.

### Statistical measures inform the grade; only direction gates promotion (documented)

Also explicit in the same source: statistical weaknesses in Alpha Discovery "feed the grade, they don't gate/reject — Rule Discovery is the sole economic judge." This is a genuinely non-obvious design choice worth dwelling on: a naive implementation would reject a candidate for weak IC or low lift. `forgedge` instead lets everything with a determined direction through to Module 3, and treats the statistical weaknesses purely as information that (a) lowers the letter grade and (b) shows up verbatim in `rejection_reasons` — even on contracts that *were* promoted. The result, verified in §7: the ADA quick-start's best `PARTIAL-EDGE` candidate carries a grade of `A` and a list of non-blocking diagnostics, alongside the one gate (rotation null) that actually capped its verdict.

### Anti-goals — what the authors explicitly decided not to build (documented)

The functional-analysis document (`docs/analysis/forge2_functional_analysis.md`) states four rejected directions with reasons, verbatim (translated):

- **"No ML/feature learning in discovery."** Reason given: *"The differentiating value is that every rule is a readable, auditable boolean expression; a trained model would break the contract far more than any bug would."*
- **"No persistent registry/database."** Reason given: *"In-memory + export (flat table, HTML) is the right level of ambition; persistence is the host's problem."* — this is why §19–20 of this manual treat persistence as an *application* responsibility, not a library feature.
- **"No probabilistic verdicts instead of the triad."** Reason given: *"The triad (+ INSUFFICIENT-DATA) IS the contract; confidence goes alongside the verdict, not in its place."*
- **"No external statistical dependencies."** Reason given: *"The pure-numpy primitives are an auditability asset, not a debt."*

### The rotation null exists because the library's own audit found its promotion counts were unreliable at low frequency (documented + measured)

This is the most important design decision to understand if you're going to trust `forgedge`'s output, and it is backed by real numbers the authors report from their own audit (`docs/analysis/lowfreq_robustness.md`, on `ADA_1D_FULL.parquet`, 901 daily bars):

> "**REAL ADA** promoted **58** alphas from 2542 candidates on a 2024 in-sample window; **phase-randomized noise** (5 runs, same recipe, same statistics, autocorrelation preserved but any real predictability destroyed) promoted **146 ± 45** (range 101–207). *Noise is promoted more often than the real asset.*"

And, one stage further downstream, at the level of tradeable `EDGE`/`PARTIAL-EDGE` verdicts:

> "Pure noise still earns ~2–3 EDGEs out of the top 12 tested (**~20% per-rule false-positive floor**)."

This is the authors' own diagnosis of a real weakness in a version of the pipeline **before** the rotation null existed as a default. The fix — `FastRotationNull`, on by default since — is described in the same document with its own measured cost and effect: *"computes the exact search-level rotation null over every circular offset (FFT cross-correlation, ~1 s on this dataset — no K, no seed)... On this ADA dataset the search p is ≈ 0.70: every former EDGE is honestly capped at PARTIAL-EDGE."* That last sentence is a genuinely striking piece of self-critical, verified reporting: the authors ran their own fix against their own best-looking prior results and reported that it downgraded every one of them.

### The design was chosen to keep the user-facing contract identical while fixing internal honesty (documented)

The functional-analysis document frames this explicitly as the guiding constraint of the whole redesign: *(translated)* "starting from scratch, what should be kept vs. redesigned — **without changing the contract with the user**?" And the closing synthesis states the diagnosis plainly: *(translated)* "the system counts its own evidence very well inside each module, but nobody counts the evidence of the entire chain — and on slow data this turns multiple testing into EDGE verdicts that pure noise can replicate one time in five." The practical upshot for you as a user: every default described in §15 below that sounds like it's making the pipeline *more conservative* (the rotation null, the power gate, purged time splits) exists specifically to close that gap, and every one of them was added without changing `forge()`'s call signature in a breaking way.

### Why the horizon grid needed to become timeframe-aware (documented + measured)

`lowfreq_robustness.md` identifies this as one specific, code-referenced weak point: *"`horizon_grid` is not frequency-scaled... `forge(..., timeframe="1D")` uses it verbatim → holding periods up to 48 days. Unlike MarketContext / Hurst / rolling-IC / bars-per-year, which do auto-scale, the horizon grid is a silent footgun."* This is why §7 of this manual spends a paragraph on the fact that `forge()` substitutes a daily grid for you — it's a fix for a documented, previously-real bug, not a cosmetic feature.

### Why deduplication only flags, never deletes (documented)

`src/forgedge/docs/README.md` gives a specific, non-obvious reason two structurally overlapping rules might both be worth keeping: *(translated)* "A pair flagged `INDEPENDENT_CONFIRMATION` is not a duplicate to discard — it's independent confirmation of the same edge via two different mechanisms." (Note: the two-level classification this passage describes — `DUPLICATE_STRUCTURAL`/`DUPLICATE_BEHAVIORAL`/`INDEPENDENT_CONFIRMATION` — is listed in the repository's own roadmap as **not yet implemented**; the current, shipped behavior is the simpler binary `is_duplicate` flag described in §8/§9. This manual flags that gap explicitly rather than describing a planned feature as current.)

---

## 15. Opt-in Behaviors

This is the section the task description asked to treat with particular care. "Opt-in" here means specifically: a behavior that does **not** happen unless you explicitly set a non-default value. Several defaults below (the rotation null, purging, the power gate, horizon enrichment) are themselves *on* by default and must be explicitly turned *off* — these are flagged as "default-on, opt-out" rather than miscategorized as opt-in.

| Feature | Default | How to enable | Benefit | Cost / trade-off | When to use it |
|---|---|---|---|---|---|
| **Event-level walk-forward OOS** (`DiscoveryConfig.walk_forward`) | `None` (off) | `DiscoveryConfig(train_ratio=0.8, walk_forward=EventWalkForwardConfig(n_splits=3, min_pass_rate=0.6))` | Confirms an event's *temporal structure* — not yet its predictive power — is stable across multiple OOS windows before you even reach Alpha Discovery | Fewer bars available for Event Discovery's own IS mining; adds Module 1 runtime | You suspect your indicator catalog might be overfitting to one regime and want a first-pass filter before the (much more expensive) Alpha/Rule Discovery stages |
| **Diversity Gate** (`DiscoveryConfig.diversity_gate_enabled`) | `False` | `DiscoveryConfig(diversity_gate_enabled=True, diversity_threshold=0.85)` | Drops near-duplicate single events (Jaccard ≥ threshold on activation dates) before AND composition, so composed candidates aren't wasted on redundant pairs | Can discard a genuinely distinct event that happens to correlate structurally with another on this specific window; adds an O(n²)-ish comparison pass over gate-passing singles | Large indicator catalogs where you've observed many near-identical candidates surviving the gate |
| **`event_counting="bar"`** (`GateParams`) | `"episode"` | `GateParams(event_counting="bar")` | Reproduces the pre-#134 (episode-counting) gate behavior exactly, bar for bar | Persistent multi-bar states (e.g. a 3–5 bar RSI<30 stretch) inflate monthly-count variance and can get wrongly rejected — this is the documented reason `"episode"` became the default | You need byte-for-byte reproducibility with a pre-episode-counting session, or you have a specific reason to weight by raw bar count |
| **`target_mode="abs"`** (`AlphaConfig`, `TargetConfig`) | `"proj"` | `AlphaConfig(target_mode="abs")` | Scores the binary target on raw forward return, not excess-over-trend — simpler, matches "textbook" backtest conventions | `"proj"` exists specifically so a long event riding a bull-market drift isn't credited with the market's own trend as if it were the event's edge; `"abs"` reintroduces that risk | Comparing against a pre-`"proj"` baseline, or working with an asset/period where trend-following is explicitly the strategy you want measured, not filtered out |
| **`AlphaConfig.embargo_bars` / `RuleWalkForwardConfig.embargo_bars`** | `0` | `AlphaConfig(embargo_bars=5)` | Adds a serial-correlation quarantine buffer at the start of the OOS window, beyond what purging alone removes | Shrinks the OOS sample further, which — combined with an already-thin daily-data OOS tail — can push a contract toward `INSUFFICIENT-DATA` | Assets with known strong short-lag autocorrelation beyond what the purge width (= max horizon) already accounts for |
| **`RotationCalibrator` (explicit, via `rotation_calibration=`)** | `None` (superseded by the default `FastRotationNull`) | `forge(..., rotation_calibration=RotationConfig(k=100))` | Multi-yardstick (composite, lift, t-stats) calibration via a Tippett min-p combination — catches discriminating statistics `FastRotationNull`'s single `abs_z` yardstick can miss | `~K ×` the cost of one Alpha Discovery pass (roughly seconds-to-minutes per draw on real data, per the design doc's own measurement of "~4 s/draw") vs. `FastRotationNull`'s ~1 second total | Paired explicitly with the `"sweep"` preset; whenever you want the full report object, not just the pass/fail annotation `fast_null` leaves on each contract |
| **`fast_null=False`** | `fast_null=True` (default-on, opt-out) | `forge(..., fast_null=False)` | Skips the default rotation-null check entirely — faster, and a would-be-capped contract can reach a full `EDGE` verdict | Reintroduces exactly the "noise gets promoted as often as signal" problem §14 documents with real numbers — this is not a performance knob to flip casually | Debugging/prototyping only, or when you're going to run the full `RotationCalibrator` separately anyway and don't want the fast pass's annotation |
| **`time_budget=` (explicit `TimeBudget`)** | `None` — but purging is still on by default even without it | `forge(..., time_budget=TimeBudget.build(n_bars=len(kpi), horizon_bars=48, embargo_bars=5))` | One shared, explicit IS/OOS axis across Event and Alpha Discovery, with a controllable embargo | More to configure correctly; get the `horizon_bars` argument wrong and the purge width is wrong too | Multi-module pipelines built by hand (not via `forge()`) where you need Event and Alpha Discovery's splits to agree exactly |
| **`purge_bars=0`** (opting *out* of the default-on purge) | purge width = `max(horizon_grid)` (default-on) | `TimeBudget.build(..., purge_bars=0)`, or `RuleWalkForwardConfig(purge_bars=0)` for M3 | Reproduces pre-purging numeric results exactly (useful for comparing against old runs, or the library's own older golden-test values) | Reintroduces a real, if usually small, look-ahead: IS bars whose forward window crosses into OOS are no longer excluded | Only when you specifically need historical reproducibility, not for normal use |
| **`power_gate=False`** (`SelectionCriteria`) | `True` (default-on, opt-out) | `RuleDiscoveryConfig(criteria=SelectionCriteria(power_gate=False))` | A verdict that would otherwise be demoted to `INSUFFICIENT-DATA` for lacking OOS statistical power is allowed through as `EDGE`/`PARTIAL-EDGE` | You lose the pipeline's own signal that the OOS evidence is too thin to trust the point estimate — a contract can look tradeable purely because nobody checked whether the sample size could support the claim | Essentially never for a live decision; only for inspecting what the *un-gated* verdict would have been |
| **`entry_mode="limit"`** (opting *out* of the default two-stage evaluation) | `"auto"` (default since #185) | `RuleDiscoveryConfig(entry_mode="limit")` | The grid optimises `buy_drop_pct` as part of the verdict, so a strategy whose *edge is the limit order itself* is measured as one thing rather than split into signal + execution | Reintroduces the fill confound into the verdict: a deeper discount fills only on the paths that came back to it, so the PF rises on a subset that is not the tradeable population | When the limit order genuinely *is* the strategy. For a signal you are trying to measure, `"auto"` separates the two readings and still publishes the limit point when it earns it |
| **`selection_mode="full_sample"`** | `"walk_forward"` | `RuleDiscoveryConfig(selection_mode="full_sample")` | Restores the legacy behavior: the operating point is chosen by screening the *entire* table, not just walk-forward train windows | The IS profit factor and the published operating parameters can be influenced by data that later becomes part of the OOS test window — a documented, real leak the walk-forward mode exists to close | Legacy compatibility only; the documented reason to avoid it in new work is explicit in the source |
| **`AlphaConfig.horizon_enrichment=None`** (opting out) | `(0.5, 1.0, 2.0)` (default-on) | `AlphaConfig(horizon_enrichment=None)` | Restricts Alpha Discovery strictly to the base `horizon_grid`, with no per-event additions | On the authors' own measurement, **34 of 247** promoted alphas on the ADA dataset found their best horizon *only* because of this enrichment — turning it off would silently lose those | Reproducing a pre-enrichment baseline, or when you have a specific reason the base grid must not be extended |
| **`pattern_features()`** (a separate function call, not a config flag) | not called | `from forgedge import pattern_features; kpi = pattern_features(kpi)` | Adds a single categorical `candle_pattern` column (ten named formations: HAMMER, DOJI, engulfing patterns, …) that flows end-to-end through `forge()` as one-hot events | Named patterns encode fixed, human-chosen thresholds; `candle_features()`'s continuous geometry is preferred for automatic discovery specifically because FORGE derives its own asset-adaptive thresholds instead | Manual/exploratory work where you specifically want to test named-pattern hypotheses, not for the default automatic-discovery path |
| **`use_stable_regime_only=True`** (`AlphaConfig`) | `False` | `AlphaConfig(use_stable_regime_only=True)` | Regime-sensitivity analysis (Step 5 of Alpha Discovery) only considers bars where the regime has held for ≥ `stable_window` bars — cleaner per-regime statistics | Fewer observations per regime, which can push `min_regime_obs` (default 10) out of reach for less common regimes | You've observed regime-transition bars contaminating the per-regime breakdown with noise from the *previous* regime |
| **`early_elimination=False`** (`SelectionCriteria`) | `True` | `RuleDiscoveryConfig(criteria=SelectionCriteria(early_elimination=False))` | Forces the full walk-forward and diagnostics pipeline to run even on a rule that fails the fast in-sample screen — useful for uniform reporting across NON-EDGE rules too | Meaningfully more compute per rule (the whole point of early elimination is to skip that work on rules that are already going to fail) | Auditing/reporting scenarios where you want every rule's OOS behavior populated, not just the survivors' |

---

## 16. Trade-offs

These are the trade-offs actually visible in the code and the design documentation — not a generic list.

**Automation vs. control.** `forge()` makes a large number of decisions for you by default (horizon grid scaling, the rotation null, purging, horizon enrichment) specifically so a "quick start" run and a "carefully tuned" run don't silently diverge in statistical honesty (§14). The cost is that a first-time reader genuinely cannot predict `forge()`'s full behavior from its call signature alone — §7's "what `forge()` did that you didn't ask for" list exists because that gap is real, not because this manual is being unusually thorough.

**Frequency vs. selectivity, in the Consistency Gate.** The library's own calibration analysis (`docs/analysis/search_rotation_calibration.md`) reports this explicitly, with real numbers, as a genuine trade-off rather than a bug to fix: raising `min_tpm` from 1.5 to 3.0 dropped candidates mined from 2621 to 584 and dropped the best real alpha's lift from 0.707 to 0.287 — but the *surviving* candidate went from 9 in-sample activations (which failed OOS: win rate 0.78 IS → 0.40 OOS) to 29 activations (which passed OOS, p=0.000). Their own words: *(translated)* "higher `min_tpm` lowers lift... you trade 'extreme but fragile' for 'modest but confirmable'." There is no default value of `min_tpm` that is simply "correct" — it's a dial between "rare events with dramatic-looking but statistically fragile edges" and "frequent events with modest but well-confirmed edges," and which end of it you want depends on how much in-sample data you actually have.

**Statistical rigor vs. wall-clock cost.** The default `FastRotationNull` was specifically engineered to make this trade-off nearly free (≈1 second, per the design doc's own measurement, vs. the `RotationCalibrator`'s ≈4 seconds *per draw* at `K=100`) — but it pays for that speed by covering only one yardstick (`abs_z`), where the full calibrator covers several combined via Tippett's method. If `abs_z` isn't the statistic that would have discriminated your particular data (the calibration doc shows this switches depending on `min_tpm`, i.e. depending on your own configuration), the fast null's single-yardstick coverage is a real, not hypothetical, gap.

**Eager vs. lazy validation.** There is no single schema-validation pass over your KPI Table (§11). This buys you the ability to pass an arbitrary, evolving table without pre-registering its schema anywhere — but it means a malformed column surfaces as a `KeyError`/`ValueError` potentially several modules deep into a `forge()` call, rather than at the door. `summary_report()` exists specifically to let you opt into eager validation when you want it, without forcing it on every call.

**Reproducibility vs. performance in Rule Discovery.** `SelectionCriteria.early_elimination=True` (default) discards a rule from further processing the moment it fails a cheap in-sample screen — this is a real performance win (the whole point is to avoid running walk-forward and full statistical validation on rules that are already going to fail), but it means `NON-EDGE` rules by default carry incomplete diagnostics (no populated `walk_forward`). §15's `early_elimination=False` entry is exactly the escape hatch for when uniform reporting matters more than speed.

**Purity of dependencies vs. reimplementation risk.** The explicit anti-goal of "no external statistical dependencies" (§14) means Spearman correlation, the t-test, the incomplete beta function, and Benjamini-Hochberg FDR control are all hand-rolled in numpy rather than delegated to `scipy.stats`/`statsmodels`. The stated benefit is auditability — you can read exactly what `forgedge` computes without stepping into a much larger external library's source. The (this manual's own, clearly inferred, not stated) corresponding risk is that these primitives don't automatically inherit `scipy`'s decades of edge-case hardening; the extensive test suite (§18) is presumably the mitigation, though the design docs don't frame it that way explicitly.

---

## 17. Performance and Scalability

Nothing in this section is an invented benchmark. Every number is either measured and reported in the repository's own documentation, or a qualitative complexity observation drawn directly from the code.

**Reported measurements (from `docs/analysis/` and this manual's own verified runs):**

- `FastRotationNull` on real ADA daily data: **~1 second**, computing the exact rotation null over every circular offset via FFT — reported by the design document, and consistent with the near-instant `result.calibration.summary()` output this manual captured in §7.
- The (heavier) `RotationCalibrator`, sampled: **~4 seconds per draw** on the same data, per `docs/analysis/search_rotation_calibration.md` — meaning `K=100` is on the order of several minutes, not seconds. This is the direct cost of the statistical-rigor-vs-speed trade-off in §16.
- This manual's own verified `forge()` run on the 882-bar ADA fixture, single-threaded, produced 5241 candidates → 370 promoted → 370 rule-discovery responses. The library's design docs separately flag Module 3 as the compute-heavy stage at scale: *(translated, `docs/analysis/forge2_functional_analysis.md`)* "M3 sequential (**255 contracts × ~0.4 s** on small data)" — i.e. Rule Discovery's per-contract walk-forward backtest is the module whose cost scales most directly with how many contracts Alpha Discovery promoted.
- The same document reports the test suite itself takes **~8.5 minutes**, "dominated by repeated full pipelines" — a fact more relevant to contributing to the library (§18) than to using it, but indicative of how much compute a full `forge()` call represents.

**Complexity observations from the code (this manual's own reading, not a stated benchmark):**

- Event Discovery's Step 1–3 (feature generation × temporal transforms × threshold catalog) is combinatorial in the number of native feature columns: each continuous column can produce several transform variants, each transform variant is tested against roughly a dozen thresholds, and arity-2 feature combinations are attempted both between same-family columns and via the several dedicated pairings described in §8. A KPI Table with many similarly-named indicator columns (e.g. ten different RSI periods) will generate a correspondingly larger candidate pool before the Consistency Gate prunes it — this is precisely why `max_categorical_classes`, `max_and_components`, `indicator_lag_cross_lags`, and the gate's own rate/dispersion thresholds exist as the practical levers to control candidate-pool size (§10).
- **A real, measured cost figure for one of §8's pairings** (from the commit that introduced `indicator_lag_cross_lags`, not an invented benchmark): on a 36-column EMA/SMA KPI Table (3 OHLC bases × 6 periods × 2 indicator families), enabling the indicator-vs-lagged-OHLC-base pairing (on by default) measured **+24% `EventDiscovery.run()` time and +21% candidate count** versus the same table with that pairing disabled (73.8 s / 23179 candidates vs. 59.6 s / 19205 candidates on that fixture). The commit message is explicit that this exceeded the original back-of-envelope estimate because "family" turned out to mean more than one representative column once period multiplicity was accounted for — worth knowing if you're tuning `indicator_lag_cross_lags` or `max_and_components` on a KPI Table with many indicator periods.
- Rule Discovery's grid screening is, per contract, a small grid search (`buy_drop_pct × sell_pct × target_h × buy_delay_bar`) run once in-sample and then re-run once per walk-forward split — so its cost scales as `(grid size) × (n_splits + 1) × (backtest cost per configuration)`, independently for every promoted contract. The authors' own audit explicitly names this as "embarrassingly parallel" across contracts (§14) — but the library does not parallelize it itself; that would be an application-level responsibility (§19).
- `pandas`-level cost: several internal feature-generation steps insert columns one at a time into a growing DataFrame rather than concatenating all at once, which this manual's own verified test run surfaced directly as a `pandas.errors.PerformanceWarning` ("DataFrame is highly fragmented... Consider joining all columns at once using pd.concat(axis=1)") during the golden test suite. This is an internal implementation detail, not something you can configure around, but it's worth knowing the warning is expected and not a sign your own code did something wrong.

**No GPU, no distributed computing, no async I/O anywhere in the library** — every module is synchronous, single-process, single-threaded numpy/pandas code. If you need to run discovery across many tickers, `forge_multi()` still runs them sequentially, one ticker at a time (§9); parallelizing across tickers or across Rule Discovery's per-contract backtests is squarely an application-level concern (§19–20), not something the library does for you.

---

## 18. Testing

The repository's own test suite is the best template for testing code that uses `forgedge` — it's substantial (**586 test functions across 15 files**, `testpaths = ["tests"]` in `pyproject.toml`) and its conventions are consistent enough to be worth adopting directly.

### Running it

```bash
pip install -e ".[dev]"      # pytest>=7.0
pytest                        # whole suite
pytest tests/test_rule_discovery.py                          # one module
pytest tests/test_forge.py::TestForgeManualEvents             # one class
pytest tests/test_forge.py::TestForgeManualEvents::test_mutual_exclusion_raises   # one test
pytest tests/test_golden.py                                    # just the regression pins
pytest -k golden                                                # by keyword
```

### The house style, as observed directly in the suite

- **One test file per source module**, named `test_<module>.py` — `test_event_discovery.py` ↔ `event_discovery/`, `test_rule_discovery.py` ↔ `rule_discovery/`, and so on, plus two cross-cutting files: `test_forge.py` (orchestrator wiring) and `test_golden.py` (end-to-end regression pinning).
- **No mocking anywhere**, except two uses of pytest's `monkeypatch` in `test_alpha_discovery.py`, both used as a "must-not-be-called" tripwire rather than a stub:

  ```python
  def test_events_come_from_stored_series_not_apply(self, monkeypatch):
      """Fast path: cached activation series must be reused, apply() must not run."""
      def _boom(self, frame):
          raise AssertionError("Alpha Discovery recomputed an event via apply()")
      monkeypatch.setattr(EventCandidate, "apply", _boom)
      ...
  ```

  Every other test builds real, seeded synthetic price series with `np.random.default_rng(seed)` and runs the actual pipeline code against them — there is no fake I/O to mock, because the library has none.
- **Deterministic, seeded, purpose-documented synthetic data** is the dominant fixture idiom — nearly every test file defines its own local `_make_kpi_table()`/`_ohlc_kpi_table()`-style helper, each with a docstring explaining *why* that specific signal shape was chosen (e.g. "low `feat` predicts a positive next-bar return, so events on `feat` should lift the win rate"). `tests/conftest.py` provides exactly two shared, module-scoped fixtures (`kpi_4380`, `kpi_8760` — synthetic hourly tables of ~6 and ~12 months) for tests that don't need a bespoke signal shape.
- **One real fixture file** for regression testing: `tests/fixtures/ADA_1D_TRAIN.parquet` (§13), used by `test_golden.py`'s session-scoped `forge_result` fixture, which runs `forge()` exactly once and derives dozens of individual field-level assertions from that single run — isolating which pipeline stage broke without re-running the pipeline per assertion.
- **`pytest.approx(..., rel=...)`** is the standard idiom for pinning floats — never bare equality on a float. **`pytest.raises(..., match=...)`** and **`pytest.warns(...)`** are used consistently (~53 uses across 8 files) to pin exact error/warning messages, not just exception types.
- **`@pytest.mark.parametrize` is used exactly once** in the whole suite (`test_target_optimizer.py`) — the codebase's own convention strongly favors one explicit, named test method per behavior over parametrized tables, even for structurally similar cases.

### A worked golden-test example, illustrating the pattern

```python
# tests/test_golden.py (structure, condensed)
@pytest.fixture(scope="session")
def forge_result():
    kpi = pd.read_parquet(FIXTURE_PATH)   # tests/fixtures/ADA_1D_TRAIN.parquet
    return forge(kpi, ticker="ADAUSDC", timeframe="1D",
                 run_rule_discovery=True, run_registry=False, progress=False)

class TestGoldenEventDiscovery:
    def test_n_activations(self, golden_candidate):
        assert golden_candidate.activation_stats.n_activations == 27  # pytest.approx not needed, int
    def test_mean_tpm(self, golden_candidate):
        assert golden_candidate.activation_stats.mean_tpm == pytest.approx(0.931034, rel=1e-4)
```

The file's own inline comments document the golden values' *history*: they've been re-pinned multiple times as legitimate pipeline changes landed (the episode-counting default, the timeframe-scaled horizon grid, the walk-forward selection change, the power gate). This is the intended workflow, and it's worth adopting for your own tests built on top of `forgedge`: **a golden test breaking is expected on a legitimate behavior change** — the correct response is to re-pin the value with a comment explaining why, not to assume the test (or the change) is wrong.

### What to test in your own code that calls `forgedge`

- **Config wiring**, mirroring `test_forge.py`: does your application correctly assemble `DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig` from whatever configuration source you use (env vars, a settings file, a preset), and does it fail the way you expect on invalid combinations?
- **Edge cases with synthetic data you fully control**, mirroring the rest of the suite: an event that never fires, a table with a single regime throughout, an empty `promoted` list reaching Rule Discovery, a rotation-null verdict that caps everything to `PARTIAL-EDGE`. These are cheap to construct with `numpy`'s seeded RNG and don't require real market data.
- **Regression-pin your own downstream logic** the way `test_golden.py` pins the library's: if your application derives a decision (e.g. "only act on grade-A EDGE rules") from `ForgeResult`, write a test that runs a fixed input through `forge()` once and asserts on the derived decision, not just on `forgedge`'s raw output — that's the layer most likely to silently drift as you tune your own configuration.
- **Do not mock `forge()` itself** in tests that are meant to validate your integration logic — the library has no I/O to fake, and the whole value of the pipeline is in its actual statistical behavior; mocking it away tests nothing real. Use a small, fast synthetic KPI Table (see `conftest.py`'s pattern) instead of a real multi-year dataset to keep your own test suite fast.

---

## 19. Integrating forgedge into a Real Application

`forgedge` is a **research library**, and this section is explicit, throughout, about the boundary between what the library gives you and what your application must build on top of it. Nothing below is a `forgedge` API — it is this manual's guidance on architecture, clearly separated from §9's API reference.

### What forgedge gives you

- A pure function of its inputs: `forge(kpi_table, ...) -> ForgeResult`. No hidden state, no required setup/teardown, no background threads.
- Every intermediate artefact of a run, inspectable after the fact (`ForgeResult.candidates`, `.contracts`, `.event_frame`, `.calibration`, `.ledger`).
- Deterministic replay of a discovered rule against new data (`EventCandidate.apply()`, `RuleDiscovery` on fresh candles, `rule_performance_report()`) — this is the mechanism §12's Use Case 6 and this section's "monitoring" component both build on.
- JSON-serializable exports at each stage (`AlphaContract.to_contract_dict()`, `RuleDiscoveryResponse.to_dict()`) and a full pickle round-trip for `EventCandidate` (`.persist(path)`).

### What your application must build

- **Persistence.** `forgedge` (specifically Module 4) is explicitly stateless across sessions (§14's anti-goals). If you need a durable catalog of discovered rules across many discovery runs over time, that catalog lives in *your* database, populated from `RuleRegistry.flat_table()` / `.export()` output or from your own serialization of `ForgeResult` fields.
- **Scheduling.** Nothing in `forgedge` decides *when* to re-run discovery, or when to re-check a published rule against fresh data. That's an external scheduler (cron, an Airflow/Prefect DAG, a queue worker) calling into your application code, which calls `forgedge`.
- **Secrets and data acquisition.** `forgedge` never talks to an exchange or a data vendor — getting a KPI Table into memory (API keys, rate limits, retries) is entirely your application's job, upstream of the KPI Builder step.
- **Execution.** The single most important boundary. A `ValidatedRule`'s `BacktestParams` (`buy_drop_pct`, `sell_pct`, `target_h`, `buy_delay_bar`, `fee`) is a *specification* for how an order should behave under the pipeline's own backtest simulation — not a live order. Turning that specification into an actual order against a real exchange, with its own latency, slippage, and partial-fill behavior, is execution-system work that sits entirely outside this library, by explicit design (§2, §3).
- **Monitoring and alerting.** `rule_performance_report()` (§9, §12) produces a static HTML snapshot — it does not push a notification, does not poll on a schedule, and does not know about your alerting system. Wiring "signal is now active" (a real flag the report computes: `EventCandidate.apply(latest_bars).iloc[-1]`) into a Slack/email/webhook notification is your application layer.

### A minimal service sketch

This composes only real, verified `forgedge` calls, wrapped in illustrative (not library) application code:

```python
import pandas as pd
from dataclasses import dataclass
from forgedge import forge, ForgeResult, RuleSpec, RuleDiscovery

@dataclass
class DiscoverySession:
    """Application-level wrapper: one forge() run plus what it takes to monitor it later."""
    ticker: str
    timeframe: str
    result: ForgeResult
    specs: list[RuleSpec]

    @classmethod
    def run(cls, kpi_table: pd.DataFrame, ticker: str, timeframe: str) -> "DiscoverySession":
        result = forge(kpi_table, ticker=ticker, timeframe=timeframe, progress=False)
        specs = RuleSpec.from_forge_result(result)
        return cls(ticker=ticker, timeframe=timeframe, result=result, specs=specs)

    def persist(self, store) -> None:
        # 'store' is YOUR persistence layer — forgedge has none.
        # AlphaContract.to_contract_dict() / RuleDiscoveryResponse.to_dict()
        # are the JSON-serializable building blocks.
        rows = [
            {"ticker": self.ticker, "rule_id": spec.name,
             "contract": next(c for c in self.result.promoted
                               if c.event_candidate_id == spec.candidate.event_id).to_contract_dict()}
            for spec in self.specs
        ]
        store.save_rules(self.ticker, rows)

    def check_signals(self, fresh_bars: pd.DataFrame) -> list[str]:
        """Which published rules are firing on the latest bars, right now."""
        active = []
        for spec in self.specs:
            fires = spec.candidate.apply(fresh_bars).fillna(0).astype(bool)
            if len(fires) and fires.iloc[-1]:
                active.append(spec.name)
        return active

    def revalidate(self, eval_df: pd.DataFrame) -> dict[str, str]:
        """Re-check whether each published rule still holds, on discovery-plus-new data.
        Uses RuleDiscovery (not AlphaDiscovery) — see §21 for why that distinction matters."""
        by_id = {c.event_id: c for c in self.result.candidates}
        verdicts = {}
        for spec in self.specs:
            contract = next(c for c in self.result.promoted
                             if c.event_candidate_id == spec.candidate.event_id)
            cand = by_id[contract.event_candidate_id]
            resp = RuleDiscovery(eval_df, contract, cand).run()
            verdicts[spec.name] = resp.verdict
        return verdicts
```

This is genuinely runnable against the ADA fixture:

```python
kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
session = DiscoverySession.run(kpi, ticker="ADAUSDC", timeframe="1D")
print(f"{len(session.specs)} published rules")     # 54
print(session.check_signals(kpi))                    # rules firing on the fixture's own last bar
```

---

## 20. A Production-Ready Architecture

Adapting the general pattern to what this codebase actually supports — a stateless research/discovery step, feeding a persistence and monitoring layer the library does not provide:

```
                     ┌───────────────────────────┐
                     │   Data ingestion service   │   ← application responsibility:
                     │  (exchange/vendor client)  │     API keys, retries, rate limits
                     └─────────────┬─────────────┘
                                   │  raw OHLCV candles
                                   ▼
                     ┌───────────────────────────┐
                     │  forgedge.kpi_builder      │   build_features / candle_features /
                     │  (feature engineering)     │   lag_features  →  KPI Table
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  forgedge.summary_report   │   opt-in pre-flight data-quality gate
                     │  (data quality gate)       │   (application decides: block or warn)
                     └─────────────┬─────────────┘
                                   │  validated KPI Table
                                   ▼
              ┌────────────────────────────────────────────┐
              │         forgedge.forge()  (discovery)       │   scheduled job — NOT request-path;
              │  M0 Market Context → M1 Event Discovery →   │   one run per ticker per re-discovery
              │  M2 Alpha Discovery → M3 Rule Discovery →   │   cycle (weekly/monthly, not per-request)
              │  M4 Rule Registry (forge_multi for many)    │
              └─────────────────────┬────────────────────────┘
                                   │  ForgeResult (candidates, contracts,
                                   │  rule_responses, calibration, ledger)
                                   ▼
                     ┌───────────────────────────┐
                     │   Application persistence  │   ← YOUR database. forgedge has none.
                     │  (rule catalog, versioned) │     Store RuleSpec/AlphaContract dicts,
                     └─────────────┬─────────────┘     candidate .persist() pickles if needed.
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
        ┌─────────────────────┐      ┌─────────────────────────┐
        │  Monitoring worker    │      │  Review / reporting UI   │
        │  (scheduled, polls    │      │  rule_performance_report │
        │  fresh candles,       │      │  → HTML, served or       │
        │  EventCandidate       │      │    emailed to a human    │
        │  .apply() +           │      └─────────────────────────┘
        │  RuleDiscovery replay)│
        └──────────┬────────────┘
                    │  "signal active" / verdict-changed events
                    ▼
        ┌─────────────────────┐
        │  Alerting / execution │   ← entirely external to forgedge, by
        │  system (human review,│     explicit design (§2, §19)
        │  or a separate order- │
        │  management system)   │
        └─────────────────────┘
```

### End-to-end flow, and what's library vs. application at each step

1. **Ingestion** (application) → raw candles.
2. **Feature engineering** (`forgedge.kpi_builder` — library) → KPI Table.
3. **Data-quality gate** (`forgedge.summary_report` — library function, but the *decision* to block on `has_critical` is application logic; the library never blocks on its own).
4. **Discovery** (`forge()`/`forge_multi()` — library) — this is the expensive, stateful-feeling-but-actually-pure step. It should run as a **scheduled background job**, not inline in a request path: §17 established there's no async I/O and no parallelism inside the library, so a `forge()` call blocks its calling thread for as long as Rule Discovery's per-contract walk-forward backtests take.
5. **Persistence** (application) — the `ForgeResult`'s promoted contracts, rule responses, and the ledger/calibration report are your durable record. `AlphaContract.to_contract_dict()` and `RuleDiscoveryResponse.to_dict()` give you JSON-ready dicts to store; `EventCandidate.persist(path)` gives a full pickle round-trip if you need to reconstruct the live object later (its `.apply()` method) rather than just its numbers.
6. **Monitoring** (application, built on library primitives) — a separate scheduled process that, on each new batch of candles, calls `EventCandidate.apply()` for the "is it firing right now" flag and `RuleDiscovery(...).run()` (not `AlphaDiscovery` — §21) for "does the verdict still hold." `rule_performance_report()` is the library's one built-in artefact for a human-readable version of this.
7. **Alerting/execution** (entirely application/external) — outside the library's scope by explicit design.

### Observability, versioning, and resource management — library support vs. application responsibility

| Concern | Library support | Application responsibility |
|---|---|---|
| Logging | `forge()` logs every pipeline stage at `INFO` via the standard `logging` module regardless of the `progress` flag — `logging.basicConfig(level=logging.INFO)` surfaces it | Routing those logs to your log aggregator; correlating a run's logs with its persisted `ForgeResult` |
| Progress reporting | `progress=True` (default) prints a stage tracker and a `tqdm` progress bar (or a dependency-free fallback) to `stderr` | Surfacing progress in a UI, if any |
| Errors | Plain built-in exceptions, no custom hierarchy (§11) | Catching them at the boundary of your discovery job; deciding retry vs. fail policy |
| Metrics | None built in — `HypothesisLedger`/`CalibrationReport` are structured data you can turn into metrics yourself | Exporting `ledger.m2_surface`, `calibration.tippett_p`, the verdict-count distribution, etc. to your metrics system |
| Timeouts | None — a `forge()` call runs to completion or raises | Wrapping the call in your own timeout/cancellation if a job scheduler requires one |
| Retries | None — the library is deterministic given the same input, so a naive retry is safe but pointless unless the input itself was the problem | Retry logic belongs at the data-ingestion layer, where transient I/O failures actually occur |
| Caching | None — every `forge()` call recomputes everything from the KPI Table you pass | Caching intermediate `ForgeResult`s (or at least `event_frame`) keyed by input hash, if you re-run discovery on largely-unchanged data |
| Secrets | None — the library takes no credentials of any kind | API keys for your data source live entirely in your ingestion layer, never near `forgedge` |
| Versioning | `forgedge.__version__`; the library's own golden tests demonstrate that behavior *can* change between versions even at fixed config (§14, §18) | Pin your `forgedge` version; re-run your own regression tests (§18) on upgrade, expecting some pinned values to legitimately need re-pinning |
| Rollback | None — stateless, nothing to roll back inside the library | Rolling back means reverting to a previously-persisted rule catalog in *your* database |

---

## 21. Troubleshooting

Each entry: symptom → likely cause → how to confirm → fix → how to prevent it next time.

### "Zero promoted contracts, or every contract is `direction='undetermined'`"

- **Cause A — genuinely no predictive events in this data/config combination.** Not everything a KPI Table produces will have predictive power; this can be a correct, honest result.
- **Cause B — the observed frame's index doesn't match the event's training activation series.** Confirm: did `AlphaDiscovery`/`RuleDiscovery` emit the `UserWarning` about "candles whose index differs from the event's stored activation series," possibly followed by the stronger variant about the activation count collapsing to under 10% of the training count (§11)? If so, this is the cause. **Fix:** if you intended to extend the training window with new bars, pass `pd.concat([train_df, new_bars_df])`, not just `new_bars_df` alone — rolling-transform baselines (pctrank, z-score) need the preceding history to mean the same thing they meant during discovery. **Prevent:** treat that specific `UserWarning` as an error in CI for any code path that's supposed to be extending, not replacing, the discovery window.
- **Cause C — horizon grid genuinely doesn't cover the event's timescale.** Confirm via `c.derived_target.score_by_h` / `.advantage_by_h` — is the score curve still rising at the grid's edge? `AlphaConfig.horizon_enrichment` (on by default) mitigates this but doesn't eliminate it for genuinely long-timescale events. **Fix:** widen `horizon_grid` explicitly.

### "I ran an `examples/*.py` script and got `TypeError: GateParams.__init__() got an unexpected keyword argument 'min_act'`"

- **Cause:** confirmed, real, reproducible (§11) — several example scripts in this repository predate a `GateParams` API change and still construct it with the old field names (`min_act`, `min_months`, `max_conc`). **Fix:** translate to the current fields — `GateParams(min_tpm=2.0, max_dispersion=2.5, event_counting="bar")` reproduces the old script's intent most closely (`event_counting="bar"` restores the pre-refactor counting semantics the old `min_act`/`min_months`/`max_conc` fields implied). **Confirm which scripts are affected:** `alpha_discovery_usage.py`, `extended_usage.py`, `kpi_table_1d.py`, `search_rotation_calibration.py`, `lowfreq_null_diagnostic.py`, `lowfreq_endpoint_diagnostic.py`. `kpi_builder_usage.py` is unaffected (verified, §13).

### "All my verdicts are `PARTIAL-EDGE`, never full `EDGE`"

- **Cause:** almost certainly the default rotation-null gate (`rotation_p > SelectionCriteria.max_rotation_p`, default 0.05), not a bug. Confirm: check `rejection_reasons` for `"search-level rotation null not cleared"`. This is expected, common, and — per the library's own audit — the *intended* behavior on datasets where the search surface is large relative to the sample size (§14, and the real ADA/AMZN examples in §7/§13, where it happened on both).
- **Fix, if you've confirmed you actually want to see through the gate for inspection purposes:** `forge(..., fast_null=False)` or raise `SelectionCriteria.max_rotation_p` — but read §15's cost/risk entry for both before doing this for anything other than debugging.

### "`RuleRegistry` output has empty `classification`/`is_generic` columns"

- **Cause:** a single-ticker session. Cross-ticker classification is mathematically undefined with nothing to cross-test against — every document becomes `ISOLATED` for lack of test tickers, not because the rule failed a test. Confirm: `reg.documents[i].cross_ticker_total == 0`.
- **Fix:** use `forge_multi()` across at least two tickers if you want genuine `GENERIC`/`PARTIAL`/`SPECIFIC` classification.

### "A custom feature column never shows up combined with anything"

- **Cause:** the column name doesn't match Event Discovery's `{base}_{indicator}_{period}` family-recognition convention (§9). It still works as a standalone feature — it's just never paired into a ratio/spread. Confirm: rename it to match the convention and re-run; if candidates involving it as a *pair* now appear, this was the cause.

### "`summary_report` flags `scale_mixed` on data I'm confident is fine"

- **Cause:** `summary_report` flags a magnitude gap of ≥ 2 orders of 10 between bars as a likely data-feed bug (e.g. some bars in cents, others in dollars). A genuine multi-order-of-magnitude price history (e.g. an asset that went from $0.01 to $50) can trigger this legitimately.
- **Confirm:** inspect `[f for f in rep.findings if f.code == "scale_mixed"].message` — it reports the actual min/median/max, so you can judge whether the spread is a real historical price range or a unit-conversion bug.
- **Fix:** if it's a real price history, this finding is a false positive you can safely note and move past — `summary_report` never blocks anything on its own (§11); there's nothing to "fix" in your data.

### "`RuleDiscovery.__init__` raises `ValueError` about a mismatched candidate/contract"

- **Cause:** you passed an `EventCandidate` whose `event_id` doesn't match `contract.event_candidate_id` — a common mistake when building the `by_id = {c.event_id: c for c in candidates}` lookup dict incorrectly, or reusing a candidate from a different `forge()` run.
- **Fix:** always derive the candidate from the *same* `ForgeResult`/`EventDiscovery.run()` output as the contract: `by_id[contract.event_candidate_id]`, as shown throughout §9/§12.

### "Re-running Alpha Discovery on new data to 'check if the edge still holds' gives worse or nonsensical results"

- **Cause:** this is a **methodology error**, not a bug — `AlphaDiscovery` *re-derives* direction and horizon from whatever data you give it. On history spanning incompatible market regimes, the same boolean condition can have preceded opposite-signed returns in different regimes, and the re-derivation can flip direction or collapse to `undetermined`, even though the *original* discovered edge is intact.
- **Fix:** the correct tool for "does a published rule still hold" is **`RuleDiscovery`**, not `AlphaDiscovery` — it replays the *fixed*, previously-derived target rather than re-deriving it. See §9's Module 3 API, §12's Use Case 6, and §19's `revalidate()` sketch.

---

## 22. Best Practices

- **Run `summary_report()` before every discovery session, and decide explicitly what to do with `has_critical`/`has_warnings`.** It costs almost nothing and the library will never do this for you (§11).
- **Pass `ed.df` (or `ForgeResult.event_frame`), not the original KPI Table, to anything downstream of Event Discovery when building the pipeline by hand.** This is the single most common source of a `KeyError` on a derived feature column (§9).
- **Treat a `PARTIAL-EDGE`-only result as informative, not as a failure to fix.** Given the library's own documented history of a previously-too-permissive pipeline (§14), a run that produces mostly `PARTIAL-EDGE` and few or zero `EDGE` is very often the pipeline being honest about a genuinely difficult dataset — check `rejection_reasons` before assuming something is misconfigured.
- **Use `forge_preset()` rather than hand-assembling three config objects, unless you have a specific reason not to.** The presets exist specifically to keep M1/M2/M3 settings mutually consistent (§10) — a common way to get a subtly-wrong result is tuning one module's config without correspondingly adjusting another's.
- **Pin your `forgedge` version and treat a `pytest`-style golden test on your own downstream logic as a first-class part of your test suite (§18).** The library's own behavior has legitimately changed between versions (§14's whole narrative is a series of such changes); code depending on `forgedge` should notice when that happens, on your own schedule, not silently.
- **Use `RuleDiscovery`, never `AlphaDiscovery`, to check whether a previously discovered edge still holds on new data.** This is the single most consequential correctness rule in this manual (§21).
- **Log `result.ledger.describe()` and `result.calibration.summary()` (or `.tippett_p`) alongside every discovery run you persist.** They're cheap, already computed, and are exactly the numbers you'll want later if you ever need to explain *why* a given verdict was or wasn't reached.

---

## 23. Anti-patterns

- **Re-deriving Alpha Discovery's target on fresh data to "monitor" a published rule.** Covered in depth in §21 — the correct tool is `RuleDiscovery`. Doing this wrong doesn't crash; it silently produces a worse-looking (or `undetermined`) result that reads as "the edge decayed" when the actual cause is methodological.
- **Turning off the rotation null (`fast_null=False`) to "get more EDGE verdicts."** This does not make your rules better — it removes the check that was specifically added, with measured justification (§14), because the pipeline's earlier default behavior promoted noise nearly as often as signal on low-frequency data. If you find yourself doing this to make a result look better, that is exactly the situation the gate exists to catch.
- **Copying `GateParams(min_act=..., min_months=..., max_conc=..., min_tpm=...)` from this repository's own `examples/*.py` scripts without checking they're current.** They aren't (§11, §21) — this specific anti-pattern is not hypothetical, it's reproducible today against this exact codebase.
- **Treating a non-`EDGE` verdict as "no signal" without reading `rejection_reasons`.** A `PARTIAL-EDGE` or even `NON-EDGE` response often carries specific, actionable diagnostic strings (a rotation-null p-value, an active-months ratio, a fill-rate number) that tell you exactly what's marginal — discarding the response object and only checking `verdict` throws that information away.
- **Building `AlphaDiscovery` directly (bypassing `forge()`) on daily-or-slower data without setting `horizon_grid` explicitly.** `forge()`'s automatic daily-grid substitution (§7) only happens inside `forge()` itself — constructing `AlphaDiscovery` by hand does not get it, and you will silently scan holding periods of up to 48 days by default (§9, §21).
- **Assuming a promoted `AlphaContract` (`status="HYPOTHESIS"`) is itself a trading signal.** It's the output of Module 2, not Module 3 — it hasn't been backtested with realistic order mechanics or validated out of sample by Rule Discovery yet. Treat `AlphaContract.status == "HYPOTHESIS"` as "worth backtesting," not "worth trading."
- **Mocking `forge()` in your own application's tests.** As covered in §18 — there's no I/O to fake, and the entire value of the call is in its real statistical output; a mock tells you nothing about whether your integration code is correct.

---

## 24. FAQ

**Does `forgedge` place orders or connect to an exchange?**
No. It has no execution capability whatsoever, by explicit design (§2, §3, §19). It produces rule specifications; an execution system you build separately implements them.

**Why did my `forge()` run promote a lot of contracts but end up with very few (or zero) full `EDGE` verdicts?**
This is very likely the default rotation-null check doing exactly what it's meant to do (§14, §21) — check `rejection_reasons` for `"search-level rotation null not cleared"` before assuming something is wrong.

**Can I use `forgedge` on 1-minute or tick data?**
Nothing technically prevents it, but every worked example, calibration analysis, and the library's own low-frequency robustness study are built and tested on hourly-to-daily bars. Very short in-sample windows at very high frequency introduce the same statistical-power problems §16 and §21 describe for daily data, likely more acutely. This manual cannot verify behavior at that frequency because nothing in the repository tests it directly.

**Why does `AlphaConfig`'s class default `horizon_grid` look wrong for my daily data?**
Because it's hourly-calibrated by design, and `forge()` (not `AlphaConfig` itself) substitutes a daily-calibrated grid for you automatically when `timeframe` is a day or slower (§7, §9, §14). Constructing `AlphaDiscovery` directly bypasses that substitution.

**Is the output of `forge()` deterministic?**
Given the same input DataFrame and the same configuration, yes — there's no randomness in Event/Alpha/Rule Discovery's own logic. `RotationCalibrator` (not the default `FastRotationNull`, which is exact/exhaustive) uses a seeded RNG (`RotationConfig.seed`, default `20260624`) for its sampled draws, so it too is reproducible given the same seed.

**Does `forgedge` persist anything between runs?**
No — explicitly and by design (§14's anti-goals, §19). Every `forge()` call is a pure function of its input; persistence is entirely your application's responsibility.

**What's the difference between an `EventCandidate`, an `AlphaContract`, and a `RuleDiscoveryResponse`?**
They're the outputs of Modules 1, 2, and 3 respectively (§4, §8) — a boolean condition with no economic meaning yet; the same condition plus a derived economic target and a statistical grade; and finally a backtested, walk-forward-validated verdict. A `RuleSpec` (§9, §12) is a fourth, lighter-weight object that bundles what you need to *replay* a tradeable rule later.

**Why is `GateParams`'s default `min_tpm` so low (0.5)?**
It's deliberately permissive so the Consistency Gate doesn't discard genuinely rare-but-real events before they even reach Alpha Discovery's much more powerful statistical checks (§10, §16). If you find Event Discovery producing too many low-quality candidates, raising `min_tpm` is the intended lever (§16's frequency-vs-selectivity trade-off) — but understand the trade-off before doing it.

---

## 25. Glossary

- **AND composition** — combining two (or three) single-column events with a boolean AND to form a more specific compound event, itself re-checked against the Consistency Gate.
- **AlphaContract** — Module 2's output object: an event candidate plus a derived economic target (direction, holding period, take-profit) and a statistical grade.
- **base rate** — the unconditional win rate of the derived binary target, measured across *all* bars, not just active ones — the baseline `lift` is measured against.
- **Consistency Gate** — Module 1's filter on raw event candidates: minimum activation rate, maximum dispersion (burstiness), and (in `"episode"` mode) a minimum episode count.
- **DerivedTarget** — the `(holding_period_h, sell_pct, direction)` triple Alpha Discovery computes for one event candidate, from the data, never assumed.
- **dispersion (Index of Dispersion)** — Variance/Mean of an event's monthly activation counts; 1.0 for a Poisson (memoryless) process, higher for bursty/clustered activations.
- **EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA** — Rule Discovery's four possible verdicts (§8, §9, §15); `is_edge` is true for the first two.
- **embargo** — an optional additional buffer of bars at the start of an OOS window, beyond what purging removes, to further quarantine serial correlation. Opt-in, default 0 (§15).
- **EventCandidate** — Module 1's output object: an immutable boolean expression plus its activation statistics and gate result.
- **fill rate** — the fraction of signals where the simulated limit order actually filled within `buy_delay_bar` bars.
- **HypothesisLedger** — a `ForgeResult.ledger` object recording how large a session's search surface was (candidates × horizons × grid cells) — bookkeeping, not a correction.
- **IC (Information Coefficient)** — the Spearman correlation between a feature's raw value and the forward return at the derived horizon, computed in-sample.
- **KPI Table** — the input `pandas.DataFrame`: `close` + a datetime source + any number of feature columns.
- **lift** — win rate on active bars minus the base rate.
- **MAE / MFE** — Maximum Adverse / Favorable Excursion — the worst/best unrealized return a trade experienced between fill and exit.
- **regime** — Module 0's 5-level ordered classification of market condition (`STRONG_BEAR`…`STRONG_BULL`).
- **rotation null** — a search-level statistical calibration that circularly rotates the `close` column (decoupling event timing from outcome) to build an empirical null distribution for the pipeline's own best statistic, correcting for the multiple-testing exposure of the whole search (§14, §15).
- **RuleDiscoveryResponse** — Module 3's output object: the verdict plus every supporting statistic (IS summary, walk-forward result, statistical validation, execution envelope).
- **RuleSpec** — a lightweight bundle (name, candidate, params, verdict) for replaying a tradeable rule against new data, independent of the Rule Registry.
- **TimeBudget** — the shared, purged (and optionally embargoed) IS/OOS bar-index split used by Event and Alpha Discovery.
- **walk-forward** — validating a fixed operating point on a sequence of rolling out-of-sample test windows, each preceded by its own train window, concatenated into one OOS track record.

---

## 26. API Reference (Quick Lookup)

This is a compact index, not a replacement for §9–10's fuller treatment. Every entry names the module it lives in.

| Symbol | Module | One-line purpose |
|---|---|---|
| `forge()` | `forgedge` | run the full pipeline end to end |
| `forge_multi()` | `forgedge` | `forge()` per ticker + one pooled cross-ticker registry |
| `ForgeResult` | `forgedge` | the return value of `forge()` — every intermediate artefact |
| `forge_preset()` / `preset_info()` / `PRESETS` | `forgedge` | pre-tuned `(DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig)` triples |
| `MarketContext` / `MarketContextConfig` / `EMAProxyConfig` | `forgedge` | Module 0 — regime classification |
| `EventDiscovery` / `DiscoveryConfig` / `EventCandidate` / `CustomEvent` | `forgedge` | Module 1 — event mining |
| `AlphaDiscovery` / `AlphaConfig` / `AlphaContract` / `PromotionThresholds` | `forgedge` | Module 2 — target derivation and predictive-power measurement |
| `RuleDiscovery` / `RuleDiscoveryConfig` / `RuleDiscoveryResponse` / `BacktestParams` / `SelectionCriteria` | `forgedge` | Module 3 — realistic backtest and walk-forward verdict |
| `RuleRegistry` / `RegistryConfig` / `RuleSubmission` / `RuleDocument` | `forgedge` | Module 4 — dedup, cross-ticker, catalog |
| `build_features()` / `candle_features()` / `lag_features()` / `pattern_features()` | `forgedge` | raw candles → KPI Table |
| `summary_report()` / `DataQualityReport` / `Finding` | `forgedge` | opt-in data-quality diagnostics |
| `TimeBudget` | `forgedge` | shared purged/embargoed IS/OOS split |
| `HypothesisLedger` | `forgedge` | search-surface bookkeeping |
| `FastRotationNull` / `RotationCalibrator` / `RotationConfig` / `CalibrationReport` | `forgedge` | search-level multiple-testing calibration |
| `TargetOptimizer` / `TargetConfig` | `forgedge` | standalone target-first alternative workflow |
| `RuleSpec` / `rule_performance_report()` | `forgedge` | replay published rules on new candles, HTML report |
| `text_report()` / `html_report()` | `forgedge.rule_discovery` | human/HTML reports from a `RuleDiscoveryResponse` |
| `run_backtest()` | `forgedge.rule_discovery` | the underlying single-configuration backtest engine |

For every dataclass's full field list and default, see §10 (the ones you're likely to tune) or the source directly — every default quoted anywhere in this manual was verified against the installed package at the time of writing.

---

*This manual was written by reading the `forgedge` source code, its test suite, its own documentation and design-analysis notes, and by executing real code against the version of the library installed from this repository. Where a claim reflects the authors' own stated reasoning rather than this manual's reading of the code, it is quoted and attributed as such throughout. The companion Italian edition, `docs/manuale-it.md`, covers identical content.*




