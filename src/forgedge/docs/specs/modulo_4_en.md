# Module 4 — Rule Registry

Rule Registry is the fifth and final module in the FORGE pipeline. It receives
the rules validated by Rule Discovery — one pool per ticker present in the
session — and consolidates them into a single in-memory registry. It then
computes correlation matrices, flags duplicate rules, runs a cross-ticker
backtest on every rule, and produces the final session artefacts: a flat table
and a self-contained HTML report.

Rule Registry does not re-evaluate the quality of any individual rule on its
source ticker. Rule Discovery already did that. What Rule Registry measures are
the *relations* between rules (activation overlap, gain correlation,
deduplication) and their *generalisability* across the other tickers of the
session.

**Stateless registry.** The registry is rebuilt from scratch on every FORGE
session. There is no persistent catalog. The exported flat table is the only
persistence artefact; the user manages it as they see fit.

**Promotion principle.** Only `EDGE` and `PARTIAL-EDGE` submissions enter the
registry. `NON-EDGE` submissions are silently skipped by the ingestion step —
no error is raised and no document is created.

---

## Basic usage

There are two ways to construct a `RuleRegistry`.

### Path 1 — manual construction from submissions

```python
from forgedge import RuleRegistry, RuleSubmission, RegistryConfig

submissions = [
    RuleSubmission(ticker="ADAUSDC", response=ada_resp_1, candidate=ada_cand_1),
    RuleSubmission(ticker="ADAUSDC", response=ada_resp_2, candidate=ada_cand_2),
    RuleSubmission(ticker="SOLUSDC", response=sol_resp_1, candidate=sol_cand_1),
]
frames = {
    "ADAUSDC": ada_kpi_df,
    "SOLUSDC": sol_kpi_df,
    "BTCUSDC": btc_kpi_df,
}

registry = RuleRegistry(submissions, frames).run()

df   = registry.flat_table()
html = registry.html_report()
```

`frames` must contain a KPI table for every ticker referenced in any
submission. It may also contain additional tickers that have no submissions —
they will participate in the cross-ticker backtest as target tickers only.

### Path 2 — from per-ticker ForgeResult runs

```python
from forgedge import forge, RuleRegistry, RegistryConfig

results = {}
for ticker, df in ticker_frames.items():
    results[ticker] = forge(df, asset=ticker, timeframe="1H")

registry = RuleRegistry.from_forge_results(
    results,
    config=RegistryConfig(
        overlap_threshold=0.70,
        cross_pf_threshold=1.5,          # absolute floor (derived from M3)
        min_cross_pf_retention=0.8,     # and the home-PF fraction to retain
        generic_ratio_threshold=2 / 3,
        export_format="excel",
    ),
).run()

flat_path = registry.export("forge_flat_table.xlsx")

html = registry.html_report(timeframe="1H")
with open("forge_report.html", "w", encoding="utf-8") as fh:
    fh.write(html)
```

`from_forge_results` extracts every tradeable response (`is_edge=True`) from
each `ForgeResult`, constructs the corresponding `RuleSubmission` objects, and
registers the enriched KPI frame of each run for the cross-ticker backtest.

---

## Position in the pipeline

```
list[RuleSubmission]  (from Rule Discovery, Module 3)
dict[ticker, kpi_df]  (post-pipeline Event Discovery frames)
        │
        ▼
  RuleRegistry.run()
  ├─ Step 1  ingest()              → list[RuleDocument]
  ├─ Step 2  compute_correlations() → CorrelationMatrices
  ├─ Step 3  deduplicate()          → None (flags in-place)
  └─ Step 4  cross_ticker()         → None (annotates in-place)
        │
        ▼
  Step 5 — Output (on demand)
  ├─ flat_table()  → pd.DataFrame
  ├─ export(path)  → CSV or Excel file
  ├─ html_report() → self-contained HTML string
  └─ summary()     → compact pd.DataFrame
```

Rule Registry is the only module that compares rules to one another and measures
their behaviour across tickers. Every upstream module (Market Context, Event
Discovery, Alpha Discovery, Rule Discovery) runs independently per ticker;
cross-ticker analysis is a property measured *a posteriori*, not built in by
mixing data at inference time.

---

## Five-step pipeline

### Step 1 — Ingestion

For each submission in the list, `ingest()` checks whether the response is
eligible for the registry (`EDGE` or `PARTIAL-EDGE` verdict). Submissions with
`NON-EDGE` verdicts are silently skipped.

For each eligible submission, one `RuleDocument` is created. The document
carries all the information produced by Rule Discovery plus three parallel
activation arrays reconstructed by replaying the backtest on the source ticker
frame: `activation_idx`, `activation_dates`, and `gains`.

**Rule ID format.** Each document receives a unique rule identifier of the form:

```
RULE_{SHORT_TICKER}_{NN:02d}
```

where `SHORT_TICKER` is the base asset with the quote currency suffix stripped,
and `NN` is a two-digit counter scoped per short ticker within the session. For
example:

| Source ticker | SHORT_TICKER | First rule ID | Second rule ID |
|---|---|---|---|
| `ADAUSDC` | `ADA` | `RULE_ADA_01` | `RULE_ADA_02` |
| `SOLUSDC` | `SOL` | `RULE_SOL_01` | — |
| `BTCUSDC` | `BTC` | `RULE_BTC_01` | — |

The quote suffix is stripped by matching against known quote currencies
(`USDC`, `USDT`, `BUSD`, `USD`, `EUR`, `BTC`, `ETH`). If no match is found,
the full upper-cased ticker is used as the short label.

**Activation arrays.** The three parallel arrays are the key data structure for
Step 2 and capture every *executed trade* on the source ticker:

- `activation_idx` — integer bar indices in the source frame where each trade
  was filled
- `activation_dates` — ISO date strings (`YYYY-MM-DD`) corresponding to each
  fill bar
- `gains` — net return of each executed trade (fees included, matching the Rule
  Discovery engine)

All three arrays have the same length (number of executed trades). Bars with no
trade are not represented in these arrays; the correlation step re-aligns them
on a shared date axis by contributing a gain of `0.0`.

---

### Step 2 — Correlation matrices

`compute_correlations()` builds two square symmetric matrices over the full set
of documents and annotates each document with its per-rule maxima.

#### Matrix A — Jaccard (temporal overlap)

The Jaccard similarity between two rules A and B is computed on their
`activation_dates` sets:

```
jaccard(A, B) = |dates_A ∩ dates_B| / |dates_A ∪ dates_B|
```

A value of `1.0` means the two rules activated on exactly the same calendar
dates. A value of `0.0` means no date in common. The matrix is symmetric and
its diagonal is `1.0`.

#### Matrix B — Spearman (gain correlation)

The Spearman correlation is computed on the date-aligned gain series. The union
of all activation dates across all rules defines a shared date axis; for any
rule, dates on which it had no trade contribute a gain of `0.0`.

The Spearman correlation is computed without scipy: the raw gains are ranked,
ties are broken by mean rank, and the Pearson correlation is computed on the
ranked series.

**Minimum active threshold.** If the number of dates on which *both* rules were
active simultaneously is below `cross_min_active` (default `10`), the Spearman
correlation for that pair is reported as `0.0`. This prevents noisy estimates
from very sparse overlap.

**Per-rule maxima.** After the matrices are built, each document is annotated
with:

- `overlap_max` — the highest Jaccard value this rule has with any other rule
  (excluding the diagonal self-comparison)
- `gain_corr_max` — the highest Spearman value this rule has with any other
  rule

These fields are `None` when the registry contains only one document (no pairs
to compare).

---

### Step 3 — Deduplication

`deduplicate()` flags — but never deletes — the weaker rule of every
overlapping pair. The registry retains all documents so the user can inspect
the full picture.

**Overlap condition.** Two rules are considered overlapping when their Jaccard
similarity meets or exceeds `overlap_threshold` (default `0.70`).

**Weaker rule.** Within an overlapping pair, the rule with the lower source-
ticker profit factor (`pf`) is the weaker one. In the event of an exact tie,
the rule with the higher index in the document list is flagged.

**Chain-awareness.** Deduplication is chain-aware. If rule A dominates rule B
(B is flagged `duplicate_of` A), and rule B also overlaps rule C, then C is
flagged `duplicate_of` B — not of A. The `duplicate_of` field always points to
the immediate dominant, forming a tree rather than collapsing all descendants
onto a single root.

**Fields populated:**

- `is_duplicate` — `True` if this rule was flagged; `False` otherwise
- `duplicate_of` — the `rule_id` of the immediate dominant, or `None` if not
  a duplicate

---

### Step 4 — Cross-ticker backtest

`cross_ticker()` replays each rule on every *other* ticker in the session and
assigns a genericity classification to each document.

#### Threshold recalibration

Each rule's boolean expression contains two kinds of thresholds:

- **Absolute thresholds** — numeric literals that are meaningful only on the
  distribution of the source ticker (e.g. `RSI < 31.2`, `volume > 850000`).
  For the target ticker, the registry maps the threshold to its in-sample
  percentile on the source ticker's distribution, then looks that same
  percentile up on the target ticker's distribution. The adapted threshold
  represents the same relative position in the target ticker's distribution as
  the original did in the source ticker's.
- **Relative thresholds** — features based on `pctrank` or `zscore` are
  already expressed relative to the local distribution and are approximately
  invariant across tickers. They are carried over unchanged.

The adapted expression is stored in `CrossTickerResult.expression_adapted`.

#### Per-(rule, target\_ticker) metrics

For each (rule, target ticker) pair, the registry runs the backtest with the
recalibrated expression and the same `BacktestParams` as the source ticker:

| Field | Description |
|---|---|
| `pf` | Profit factor on the target ticker |
| `win_rate` | Win rate on the target ticker (0–1) |
| `total_trades` | Number of executed trades |
| `zero_months` | Months with no trades |
| `verdict` | `"PASS"` when the rule clears **both** halves of the transfer criterion (below) |
| `bar` | The profit factor it had to reach here — the higher of the two halves |

##### The transfer criterion

```
PASS  ⟺  pf >= cross_pf_threshold           (it is tradeable there)
         AND  pf >= min_cross_pf_retention · pf_home    (it actually transfers)
```

A single absolute bar answers *"is it good elsewhere?"*, while every label in
the `GENERIC`/`PARTIAL`/`SPECIFIC`/`ISOLATED` vocabulary asks *"does it
transfer?"*. Those come apart in both directions:

| rule | PF home | PF away | single `2.0` bar |
|---|---|---|---|
| transfers perfectly | 1.6 | 1.6 | FAIL — and on every ticker, so `ISOLATED` |
| degraded by a third | 3.0 | 2.05 | PASS — while a third of the edge is gone |

The first row was not an edge case: `partial_min_profit_factor` admits rules at
`1.5`, so **the whole `PARTIAL-EDGE` class was structurally excluded from
genericity**. Lowering the single bar to the home PF fixes that row and makes
the second worse still — the *weakest* rules would then get the *easiest*
genericity test. Two halves keep the floor a floor and let the ratio measure
transfer; quality stays where the registry already records it, on the M3
verdict and the grade.

#### Genericity classification

After all target tickers have been evaluated, each document receives:

- `cross_ticker_score` — number of `PASS` verdicts across target tickers
- `cross_ticker_total` — total number of other tickers evaluated
- `is_generic` — `True` when `cross_ticker_score / cross_ticker_total >=
  generic_ratio_threshold` (default `2/3`)

The `classification` field is then set to one of four labels:

| Label | Condition |
|---|---|
| `GENERIC` | `is_generic=True` and `is_duplicate=False` |
| `PARTIAL` | `is_generic=False`, `cross_ticker_score > 0`, `is_duplicate=False` |
| `SPECIFIC` | `is_generic=False`, `cross_ticker_score = 0`, `is_duplicate=False` |
| `ISOLATED` | `is_duplicate=True` (regardless of cross-ticker score) |

**Single-ticker sessions.** When the session contains only one ticker, Step 4
produces no `CrossTickerResult` entries for any document (`cross_ticker_total =
0`). All documents receive `classification = "ISOLATED"` because the
genericity ratio is undefined.

---

### Step 5 — Export

Step 5 is on-demand: the methods below are available after `run()` completes
but are not called automatically.

See [Output methods](#output-methods) for the full API description.

---

## Data structures

### `RuleDocument`

The central document in the registry. One document per ingested EDGE /
PARTIAL-EDGE rule.

**Identification fields:**

| Field | Type | Description |
|---|---|---|
| `rule_id` | `str` | Unique rule identifier, e.g. `RULE_ADA_01` |
| `expression` | `str` | Boolean rule expression as a string |
| `source_ticker` | `str` | Full ticker the rule was extracted on, e.g. `ADAUSDC` |
| `source_alpha_id` | `str` | Alpha Discovery contract ID of the originating signal |
| `verdict` | `str` | Rule Discovery verdict: `"EDGE"` or `"PARTIAL-EDGE"` |
| `grade` | `str` | Letter grade from Alpha Discovery, or derived from Rule Discovery evidence |

**Activation arrays** (parallel, length = number of executed trades):

| Field | Type | Description |
|---|---|---|
| `activation_idx` | `list[int]` | Bar indices in the source frame where trades were filled |
| `activation_dates` | `list[str]` | ISO date strings (`YYYY-MM-DD`) for each filled trade |
| `gains` | `list[float]` | Net return of each executed trade |

**Operational parameters:**

| Field | Type | Description |
|---|---|---|
| `params` | `dict` | `BacktestParams` fields as a plain dictionary |

**Source-ticker statistics:**

| Field | Type | Description |
|---|---|---|
| `stats` | `dict` | Performance metrics from Rule Discovery: `pf`, `win_rate`, `total_trades`, `tpm_mu`, `zero_months`, `expectancy`, etc. |

**Regime metadata:**

| Field | Type | Description |
|---|---|---|
| `regime` | `dict` | Regime analysis from Rule Discovery (`dependency_score`, `avoid_in`, `per_regime`) |

**Fields populated by Step 2/3:**

| Field | Type | Description |
|---|---|---|
| `overlap_max` | `float \| None` | Highest Jaccard similarity with any other rule |
| `gain_corr_max` | `float \| None` | Highest Spearman gain correlation with any other rule |
| `is_duplicate` | `bool \| None` | `True` if this rule was flagged as a duplicate |
| `duplicate_of` | `str \| None` | `rule_id` of the immediate dominant rule, or `None` |

**Fields populated by Step 4:**

| Field | Type | Description |
|---|---|---|
| `cross_ticker` | `dict[str, CrossTickerResult]` | Keyed by target ticker; one entry per other ticker |
| `cross_ticker_score` | `int \| None` | Count of PASS verdicts across target tickers |
| `cross_ticker_total` | `int \| None` | Total number of target tickers evaluated |
| `is_generic` | `bool \| None` | `True` when score / total >= `generic_ratio_threshold` |
| `classification` | `str \| None` | `"GENERIC"`, `"PARTIAL"`, `"SPECIFIC"`, or `"ISOLATED"` |

The convenience property `doc.pf` returns `float(doc.stats.get("pf", nan))`
and is used internally by the deduplication step to rank overlapping pairs.

`doc.to_dict()` returns a fully nested plain dictionary suitable for JSON or
YAML serialisation. Internal fields (`_candidate`, `_bt_params`, `_trades`)
are excluded.

---

### `CrossTickerResult`

Stores the outcome of replaying one rule on one alternative ticker.

| Field | Type | Description |
|---|---|---|
| `ticker` | `str` | The target ticker |
| `expression_adapted` | `str` | Rule expression with recalibrated absolute thresholds |
| `pf` | `float` | Profit factor on the target ticker |
| `win_rate` | `float` | Win rate (0–1) |
| `total_trades` | `int` | Number of executed trades |
| `zero_months` | `int` | Months with no trades |
| `verdict` | `str` | `"PASS"` when `pf >= bar`, else `"FAIL"` |
| `bar` | `float` | The profit factor required here: `max(cross_pf_threshold, min_cross_pf_retention × pf_home)` |

`result.to_dict()` returns a flat dictionary of all fields.

---

### `CorrelationMatrices`

Returned by `compute_correlations()` and stored as `registry.matrices`.

| Field | Type | Description |
|---|---|---|
| `rule_ids` | `list[str]` | Row and column labels, in registry document order |
| `jaccard` | `pd.DataFrame` | Symmetric Jaccard matrix; shape `(N, N)`, index and columns are `rule_ids` |
| `spearman` | `pd.DataFrame` | Symmetric Spearman matrix; shape `(N, N)`, index and columns are `rule_ids` |

Both matrices are symmetric and have `1.0` on their diagonal (for Jaccard) or
`1.0` on their diagonal for Spearman — although Spearman diagonal is not used
by any downstream step.

---

## Output methods

### `registry.run() → RuleRegistry`

Executes Steps 1–4 in sequence and returns `self`. This is the entry point for
the standard pipeline. Call `run()` before any output method.

```python
registry = RuleRegistry(submissions, frames, config=cfg).run()
```

### `registry.ingest() → list[RuleDocument]`

Executes Step 1 only. Builds one `RuleDocument` per eligible submission and
stores the result in `registry.documents`. Returns the list of documents.
Called automatically by `run()`.

### `registry.compute_correlations() → CorrelationMatrices`

Executes Step 2 only. Computes the Jaccard and Spearman matrices, annotates
each document with `overlap_max` and `gain_corr_max`, and stores the matrices
in `registry.matrices`. Returns the `CorrelationMatrices` object.
Called automatically by `run()`.

### `registry.deduplicate() → None`

Executes Step 3 only. Flags duplicate documents in-place by setting
`is_duplicate` and `duplicate_of` on each `RuleDocument`. If the matrices have
not been computed yet, this method calls `compute_correlations()` first.
Returns `None`. Called automatically by `run()`.

### `registry.cross_ticker() → None`

Executes Step 4 only. Populates the `cross_ticker` dict, `cross_ticker_score`,
`cross_ticker_total`, `is_generic`, and `classification` on each document.
Returns `None`. Called automatically by `run()`.

### `registry.flat_table(apply_filters=False) → pd.DataFrame`

Returns the flat registry table — one row per document, all steps' fields
included as columns.

When `apply_filters=False` (the default), every document is included regardless
of `export_duplicates` and `export_non_generic` settings in the config. FORGE
surfaces complexity rather than hiding it; the default keeps the full picture.

When `apply_filters=True`, the config's filter flags are applied:

- `export_duplicates=False` — rows where `is_duplicate=True` are excluded
- `export_non_generic=False` — rows where `is_generic=False` are excluded

```python
# Full table — all rules
df = registry.flat_table()

# Filtered table — only non-duplicate rules
df_filtered = registry.flat_table(apply_filters=True)
```

### `registry.export(path) → str`

Writes the flat table to `path` using the format specified by
`RegistryConfig.export_format` (`"excel"` or `"csv"`). The filter flags in the
config are applied to the written file. Returns the path string that was
written.

```python
path = registry.export("forge_flat_table.xlsx")
print(f"Written to: {path}")
```

### `registry.html_report(**kwargs) → str`

Renders a self-contained HTML report and returns it as a string. The HTML
contains no external assets: SVG charts (equity curves, correlation heatmaps)
are embedded inline, and all CSS is inlined. The report can be saved to a file
and opened in any browser without an internet connection.

The report includes:

- Session header with date and ticker list
- Per-rule sections: verdict, grade, expression, IS metrics, execution
  parameters, regime summary
- Correlation heatmaps (Jaccard and Spearman) when `html_charts=True`
- Cross-ticker summary table with GENERIC / PARTIAL / SPECIFIC / ISOLATED
  badges
- Duplicate banners identifying which rule each duplicate is dominated by
- Trade log when `html_include_tradelog=True`

If the matrices have not been computed yet when `html_report()` is called, they
are computed on the fly.

```python
html = registry.html_report(timeframe="1H")
with open("forge_report.html", "w", encoding="utf-8") as fh:
    fh.write(html)
```

### `registry.summary() → pd.DataFrame`

Returns a compact one-row-per-rule overview suitable for quick inspection in a
terminal or notebook. Columns:

| Column | Description |
|---|---|
| `rule_id` | Document identifier |
| `source_ticker` | Ticker the rule was extracted on |
| `grade` | Letter grade |
| `pf` | Source-ticker profit factor |
| `win_rate` | Source-ticker win rate (0–1) |
| `total_trades` | Source-ticker executed trade count |
| `overlap_max` | Highest Jaccard similarity with any other rule |
| `is_duplicate` | Whether this rule is flagged as a duplicate |
| `duplicate_of` | rule_id of the dominant rule, or `None` |
| `cross_ticker_score` | Number of PASS verdicts |
| `cross_ticker_total` | Total target tickers evaluated |
| `is_generic` | Whether the rule meets the generic ratio threshold |
| `classification` | `GENERIC`, `PARTIAL`, `SPECIFIC`, or `ISOLATED` |

```python
print(registry.summary().to_string(index=False))
```

---

## `RuleSubmission`

The input unit handed to the registry. One submission per validated rule.

| Field | Type | Description |
|---|---|---|
| `ticker` | `str` | Source ticker, e.g. `"ADAUSDC"` |
| `response` | `RuleDiscoveryResponse` | The Rule Discovery verdict and validated rule |
| `candidate` | `EventCandidate` | The Event Candidate referenced by the validated rule; provides the deterministic replay path used by the cross-ticker backtest |
| `grade` | `str \| None` | Optional letter grade from Alpha Discovery; when `None`, the registry derives a grade from the Rule Discovery evidence |

Only submissions where `response.verdict` is `"EDGE"` or `"PARTIAL-EDGE"` are
ingested. Passing a `NON-EDGE` submission does not raise an error; it is simply
skipped.

---

## Full configuration reference — `RegistryConfig`

```python
from forgedge import RegistryConfig

config = RegistryConfig(
    overlap_threshold=0.70,
    gain_corr_threshold=0.70,
    cross_pf_threshold=1.5,
    min_cross_pf_retention=0.8,
    generic_ratio_threshold=2 / 3,
    cross_min_active=10,
    export_format="excel",
    export_duplicates=True,
    export_non_generic=True,
    html_include_tradelog=True,
    html_charts=True,
    timestamp_col="open_dt",
    session_date=None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `overlap_threshold` | `0.70` | Jaccard similarity at or above which two rules are considered overlapping (Step 3). The weaker rule of each overlapping pair is flagged as a duplicate. |
| `gain_corr_threshold` | `0.70` | Spearman correlation threshold used for reporting ("same regime exposure" reading). Does not drive deduplication; the deduplication step uses only Jaccard. |
| `cross_pf_threshold` | `1.5` | Absolute profit-factor floor for a `PASS` verdict (Step 4) — half the criterion. Session-resolved from `SelectionCriteria.partial_min_profit_factor`: the bar that admitted the rule at home. |
| `min_cross_pf_retention` | `0.8` | The other half: fraction of the rule's **home** profit factor it must retain on the target ticker. |
| `generic_ratio_threshold` | `2/3` | Minimum fraction of `PASS` verdicts for a rule to be considered generic (`is_generic=True`). The default is exactly `2/3` — not `0.67` — so that a rule passing 2 of 3 target tickers is correctly classified as `PARTIAL` rather than `GENERIC`. |
| `cross_min_active` | `10` | Minimum number of dates on which both rules were simultaneously active before a Spearman correlation is computed. Below this count, the Spearman value is reported as `0.0`. |
| `export_format` | `"excel"` | Format written by `export()`. One of `"excel"` or `"csv"`. |
| `export_duplicates` | `True` | When `False` and `apply_filters=True`, rows where `is_duplicate=True` are excluded from the flat table and the exported file. |
| `export_non_generic` | `True` | When `False` and `apply_filters=True`, rows where `is_generic=False` are excluded from the flat table and the exported file. |
| `html_include_tradelog` | `True` | Append the cross-session trade log to the HTML report. |
| `html_charts` | `True` | Embed inline SVG charts (equity curves, Jaccard and Spearman heatmaps) in the HTML report. |
| `timestamp_col` | `"open_dt"` | Name of the datetime column in every KPI frame. If the frame has a `DatetimeIndex`, this column is created automatically if absent. |
| `session_date` | `None` | ISO date string (`YYYY-MM-DD`) stamped onto the HTML report header. When `None`, today's date is used. |

---

## Advanced usage patterns

### 1. Manual construction with explicit submissions

```python
from forgedge import (
    RuleRegistry, RuleSubmission, RegistryConfig,
    RuleDiscovery, RuleDiscoveryConfig,
)

# Assume `ed`, `contracts`, `candidates` are already built for each ticker
by_id = {c.event_id: c for c in candidates}
submissions = []
for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    resp = RuleDiscovery(ed.df, contract, cand).run()
    if resp.is_edge:
        submissions.append(
            RuleSubmission(
                ticker="ADAUSDC",
                response=resp,
                candidate=cand,
                grade=contract.alpha_score.grade,
            )
        )

frames = {"ADAUSDC": ada_df, "SOLUSDC": sol_df}
registry = RuleRegistry(submissions, frames).run()
```

### 2. Filtering only non-duplicate and generic rules from the flat table

```python
config = RegistryConfig(
    export_duplicates=False,
    export_non_generic=False,
)
registry = RuleRegistry(submissions, frames, config=config).run()

# All rules visible in the session
full_df = registry.flat_table(apply_filters=False)

# Only non-duplicate generic rules
filtered_df = registry.flat_table(apply_filters=True)
print(f"Full: {len(full_df)} rules  |  Filtered: {len(filtered_df)} rules")
```

### 3. Inspecting the correlation matrices

```python
registry.run()

matrices = registry.matrices   # CorrelationMatrices

print("Rule IDs:", matrices.rule_ids)

print("\nJaccard matrix:")
print(matrices.jaccard.to_string())

print("\nSpearman matrix:")
print(matrices.spearman.round(3).to_string())

# Pairs with high Jaccard overlap
import numpy as np
jac = matrices.jaccard.copy()
np.fill_diagonal(jac.values, 0.0)
high_overlap = jac[jac >= 0.70].stack()
print("\nOverlapping pairs (Jaccard >= 0.70):")
print(high_overlap)
```

### 4. Accessing cross-ticker results for a specific rule

```python
# After registry.run()
doc = registry.documents[0]
print(f"Rule: {doc.rule_id}  Classification: {doc.classification}")
print(f"Cross-ticker score: {doc.cross_ticker_score}/{doc.cross_ticker_total}")

for ticker, result in doc.cross_ticker.items():
    print(
        f"  {ticker:<12}: {result.verdict:<4}  "
        f"PF={result.pf:.2f}  trades={result.total_trades}"
    )
    print(f"    adapted: {result.expression_adapted}")
```

### 5. Selective export — non-duplicate only, generic only

```python
# Export only non-duplicate rules to CSV
config = RegistryConfig(
    export_format="csv",
    export_duplicates=False,
    export_non_generic=True,
)
registry = RuleRegistry(submissions, frames, config=config).run()
registry.export("forge_nodups.csv")

# Export only generic rules to Excel
config_generic = RegistryConfig(
    export_format="excel",
    export_duplicates=False,
    export_non_generic=False,
)
registry_g = RuleRegistry(submissions, frames, config=config_generic).run()
registry_g.export("forge_generic.xlsx")
```

---

## Operational notes

**Stateless registry.** There is no persistent rule catalog. Every FORGE session
builds the registry from scratch. If you want to accumulate rules across
sessions, append the exported flat tables and manage the merged file externally.
The flat table is the designed persistence artefact.

**KPI frames.** The `frames` dict must contain the post-pipeline DataFrames from
Event Discovery, not the original raw KPI tables. These frames carry the feature
columns (KPIs, regime labels, etc.) that the rule expressions reference. Passing
a frame that is missing a required column will cause the cross-ticker backtest
to raise an error when replaying that rule.

**NON-EDGE submissions.** Passing a `RuleSubmission` whose `response.verdict`
is `"NON-EDGE"` is not an error. The submission is checked by `is_ingestable()`
at the start of the ingestion step and silently skipped. No document is
created for it.

**Single-ticker session.** When the session contains only one ticker, all rules
will have `cross_ticker_total = 0` and `classification = "ISOLATED"`. The cross-
ticker step runs without error — it simply has no target tickers to evaluate.
The flat table and HTML report are still produced normally.

**Threshold recalibration uses the IS distribution.** In Step 4, absolute
thresholds are mapped to percentiles using the full in-sample distribution of
the *source* ticker, and those percentiles are looked up on the full in-sample
distribution of the *target* ticker. The live (real-time) distribution is not
used for recalibration.

**`generic_ratio_threshold` precision.** The default value is `2.0 / 3.0`
(Python floating-point division), not the literal `0.67`. This distinction
matters: a rule that passes exactly 2 of 3 tickers has a ratio of
`0.6666...`, which meets `>= 2/3` but does not meet `>= 0.67`. Always pass
`2 / 3` rather than `0.67` when you intend the two-thirds boundary.
