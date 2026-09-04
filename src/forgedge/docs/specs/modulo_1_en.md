# Module 1 — Event Discovery

Module 1 discovers boolean events from the temporal structure of indicators in
the KPI Table. An **event** is a boolean condition (e.g. `RSI < 30.5 AND pctrank_96 < 0.10`)
that fires on a subset of bars. The module works **without ever seeing the forward
return**: it evaluates only whether an event has a stable and statistically
plausible temporal activation pattern.

The output is a list of `EventCandidate`, one per event that clears the
ConsistencyGate. These candidates are then passed to Module 2 to measure their
predictive power.

---

## Basic usage

```python
from forgedge import EventDiscovery
from forgedge.event_discovery.discovery import DiscoveryConfig
from forgedge.event_discovery.models import GateParams

ed = EventDiscovery(enriched_kpi)       # kpi already enriched with regime from Module 0
candidates = ed.run()

print(f"Found {len(candidates)} candidates")
print(ed.summary().sort_values("mean_tpm", ascending=False).head(10))
```

The default configuration uses production parameters (`min_tpm=0.5`,
`event_counting="episode"`, `min_episodes=10`, `dispersion_margin=1.3`,
etc. — see Step 4 below). To explore with more permissive thresholds:

```python
config = DiscoveryConfig(
    gate_params=GateParams(min_tpm=0.3, min_episodes=5, dispersion_margin=1.6),
    max_and_components=2,
)
ed = EventDiscovery(enriched_kpi, config=config)
candidates = ed.run()
```

---

## The five-step pipeline

### Step 0 — Column classification (`TypeClassifier`)

Every column of the DataFrame (except the timestamp) is classified into one
of three types:

| Type | Criterion | Downstream treatment |
|---|---|---|
| `CONTINUOUS` | Numeric, > 2 distinct values | Full pipeline (Steps 1–3) |
| `BINARY` | Exactly 2 distinct values | Skips Steps 1–2, goes directly to Step 3 |
| `CATEGORICAL` | Non-numeric, or ≤ `max_categorical_classes` values | One-hot in Step 3; if > limit, excluded |

For CONTINUOUS columns, the **scale-free** property is also detected:
a series is scale-free if its values are intrinsically bounded (e.g. RSI in
[0,100], percentages) and do not depend on the asset's price level. This
property determines whether the `identity` transform is included in Step 2.

Detection is an asymmetric, conservative heuristic: it prefers false negatives
(classifying as not-scale-free a series that is) over false positives
(classifying as scale-free a series that is not, generating events with thresholds
dependent on the absolute price level).

**Manual override:** if the heuristic produces a wrong result for a specific
column, it can be corrected:

```python
config = DiscoveryConfig(
    scale_free_overrides={"close_rsi_14": True, "volume": False}
)
```

After `run()`, classifications are inspectable via `ed.get_classifications()`:

```python
cls = ed.get_classifications()
for col, c in cls.items():
    print(f"{col}: {c.col_type.value}, scale_free={c.effective_scale_free}")
```

---

### Step 1 — Feature generation (`FeatureGenerator`)

From the native KPI Table features, derived features of arity 1, 2 and 3 are generated:

| Arity | Operation | Formula | Example |
|---|---|---|---|
| 1 | Pass-through | `f` | `close_rsi_25` (scale-free only) |
| 2 | Ratio | `a / b` | `ratio_close_ema_09_ema_25` |
| 2 | Spread percentage | `(a - b) / b` | `spread_close_bb_upper_lower` |
| 2 | Diffnorm | `(a - b) / σ(a-b)` | `diffnorm_close_sma_09_sma_25` |
| 3 | %B Bollinger | `(val - lo) / (hi - lo)` | `bb_pct_b_close_bb_lower_upper` |
| 3 | Position in range | `(val - min) / (max - min)` | `pos_close_min_24_max_48` |

Arity-2 features are generated only between columns of the same family
(e.g. two EMAs on the same source, not an EMA and RSI). Zero denominators produce
`NaN` (not `±inf` or `pd.NA`), preserving the `float64` dtype.

For the `diffnorm` feature, the standard deviation `σ(a-b)` is computed on the
in-sample period and stored in `transform_params["diffnorm_std"]`.
This value is reused for OOS replay to preserve the same scale as the in-sample:
the OOS feature is normalised with the IS standard deviation, not recomputed.

---

### Step 2 — Temporal transforms (`TransformLayer`)

Every feature in the catalogue receives the following transforms:

| Transform | Code | Windows | Applies to |
|---|---|---|---|
| Identity | `identity` | — | Scale-free only |
| Rolling percentile rank | `rolling_pctrank` | 48, 96, 168 bars | All continuous |
| Rolling z-score | `rolling_zscore` | 48, 96, 168 bars | All continuous |
| Delta (difference) | `delta` | 1, 3, 6, 12 bars | All continuous |

`min_periods` for rolling windows: `max(2, window // 2)`.
This means the 96-bar window starts producing values after just 48 bars,
even though the estimate is less stable.

**Why these transforms?**
- `identity`: for already scale-free, stationary features (RSI, %B), the raw
  value is directly comparable across the full dataset.
- `rolling_pctrank`: converts any series into a [0,1] range relative to recent
  history. Robust to outliers and requires no stationarity.
- `rolling_zscore`: sensitive to local distribution. Useful for detecting
  statistical deviations from the recent mean.
- `delta`: captures short-term changes (momentum or reversal at specific lags).

---

### Step 3 — Event generation (`EventGenerator`)

Each transformed series is converted into boolean events by applying thresholds.
The Threshold Catalog distinguishes two families:

**Distributional thresholds** (based on percentiles of the transformed series):
```
p3, p5, p8, p10, p15     (lower tails — extreme conditions on the downside)
p85, p90, p92, p95, p97  (upper tails — extreme conditions on the upside)
```

**Theoretical thresholds** (for the zscore transform):
```
-2.0, -1.5, -1.0, +1.0, +1.5, +2.0
```

For each threshold, a `threshold` event is always generated. A `crossing`
event is generated too, but **only for the `identity` transform, and only
on `direction="below"` thresholds** — absolute thresholds are not
meaningful for rolling pctrank/zscore (re-anchored on every bar) or for
delta (which oscillates around zero), so crossing events are restricted to
`identity`. Within `identity`, only the downward crossing formula
(`series_t < threshold AND series_{t-1} >= threshold`) is defined: there is
no symmetric high-direction (`above`) crossing in the generated pool at
all — the use case this targets is oversold/extreme-low detection, and it
was never given an upside counterpart.

| Type | Description | When active | Generated for |
|---|---|---|---|
| `threshold` | Persistent state | Every bar where the condition is true | All transforms, both directions |
| `crossing` | Instantaneous transition | Only the bar where the series crosses the threshold | `identity` transform, `direction="below"` only |

`crossing` events signal "the signal just entered the zone", useful for entry
logic. `threshold` events capture "the signal has been in zone for an arbitrary
number of bars", more appropriate for regime filters.

**BINARY columns:** generates one `binary_native` event per value (0 and 1) —
no transforms required.

**CATEGORICAL columns:** generates one `categorical_onehot` event per class
where `n_distinct ≤ max_categorical_classes`. Classes with too many distinct
values are excluded because they resemble identifiers rather than signals.

---

### Step 4 — ConsistencyGate (`ConsistencyGate`)

The gate filters events based on their temporal activation distribution.
The rationale: an event with an unstable temporal structure (e.g. all triggers
concentrated in a single month, or statistically indistinguishable from a
random process) is not a reliable candidate for alpha discovery.

The gate runs one of two modes, selected by `GateParams.event_counting`.

**`"episode"` mode (default, issue #134)** — counts *episodes* (maximal runs
of consecutive activations, bridging gaps up to `episode_gap` bars) instead
of individual bars. This removes a counting artifact whereby a persistent
multi-bar state (e.g. a 3–5 bar `RSI < 30` stretch) inflates the monthly
variance and gets wrongly rejected. An event **passes** if and only if it
satisfies **all 3** criteria:

| Criterion | Parameter | Default | Rationale |
|---|---|---|---|
| Rate | `min_tpm` | 0.5 | Episodes per month on average, at least this |
| Statistical power | `min_episodes` | 10 | Absolute floor on the number of episodes, so downstream significance tests (Modules 2/3) have enough sample |
| Dispersion | `dispersion_margin` | 1.3 | Episode-level Index of Dispersion must not exceed `poisson_floor(n_months) x dispersion_margin` |

The dispersion criterion (issue #205) compares against a *Poisson χ² floor*
— the largest Index of Dispersion still consistent with a purely random
process at the event's own rate — rather than a fixed absolute value. This
guarantees the gate never rejects an event that a random process could
plausibly have produced, while still letting a preset's own tolerance for
burstiness (`dispersion_margin`) actually bind: measured across all four
presets and four timeframes, a fixed `max_dispersion` threshold never bound
in 12 of 16 combinations because the floor alone was almost always the
stricter constraint. `max_dispersion` plays no role in `"episode"` mode.

**`"bar"` mode** — the historical, 100% backward-compatible behaviour:
counts individual activated bars rather than episodes. An event **passes**
if and only if it satisfies **both** criteria:

| Criterion | Parameter | Default | Rationale |
|---|---|---|---|
| Rate | `min_tpm` | 0.5 | Activated bars per month on average, at least this |
| Dispersion | `max_dispersion` | 1.5 | Per-bar Index of Dispersion (Var/Mean of monthly counts) must not exceed this absolute value — no Poisson floor involved |

`min_episodes` and `dispersion_margin` play no role in `"bar"` mode.

`GateResult` includes a `fail_reason` field with the first failing criterion
(useful for debugging and parameter tuning), plus diagnostic fields that are
always computed regardless of mode: `n_episodes` (episode count),
`episode_index_of_dispersion` (episode-level Index of Dispersion, computed
whenever a month index is available) and `n_eff` — an effective sample size
(`n_episodes / max(episode_ID, 1.0)`) intended for downstream significance
deflation in Modules 2/3.

**Preset-specific `min_episodes` (issue #206):** `forge_preset()` does not
apply one flat default — `sniper`, `balanced` and `burst` keep 10 (the
statistical-power floor is the point for those profiles), while `sweep`
lowers it to 5, permissive by design, since it delegates statistical rigor
to the downstream `RotationCalibrator`. Because `min_episodes` is an
absolute count, the in-sample discovery window needed to clear it with 95%
confidence depends on both `min_episodes` and `min_tpm` — the naive
`min_episodes / min_tpm` window satisfies the floor only in expectation.
`config_report()`'s `m1_is_window_too_short` (WARN) flags the gap between
the configured combination and the discovery span actually available
(`span_months x train_ratio`).

---

### Step 4b — Diversity Gate (opt-in)

After the ConsistencyGate and before AND composition, an optional deduplication
step removes near-duplicate single events using **Jaccard similarity on activation
dates**. Two events whose activation-date sets have Jaccard similarity ≥
`diversity_threshold` are considered near-duplicates; the weaker one (fewer IS
activations) is dropped.

**Why it matters:** Without the Diversity Gate, the AND Composer can produce a
large number of structurally redundant composed events (e.g. `RSI < 31 AND RSI < 30`
where the two components share >85% of their activation dates). Deduplicating the
single-event pool before AND composition reduces both the search space and the risk
of redundant candidates reaching Module 2.

| Setting | Value | Notes |
|---|---|---|
| Enabled by default | No (`diversity_gate_enabled=False`) | Opt-in only — no breaking change |
| Threshold default | `diversity_threshold=0.85` | Empirical: at p99 of the inter-event Jaccard distribution on 12 months of 1H data, Jaccard=0.47 — values above 0.70 are genuine near-duplicates |

**Activation:** set `diversity_gate_enabled=True` in `DiscoveryConfig`.

```python
from forgedge import EventDiscovery, DiscoveryConfig

config = DiscoveryConfig(
    diversity_gate_enabled=True,    # opt-in Jaccard deduplication
    diversity_threshold=0.85,       # remove near-duplicates with Jaccard ≥ 0.85
)
ed = EventDiscovery(enriched, config=config)
candidates = ed.run()
```

---

### Step 5 — AND Composition (`ANDComposer`)

**Off by default since issue #254 Phase 8.** `DiscoveryConfig()`'s class
default is now `max_and_components=1`, which skips this step entirely —
`forge()`'s own default path (`two_pass_composition=True`) composes
downstream instead, after Module 2's first grading pass, using the letter
grade as the pairing criterion rather than this step's purely structural
one (see Module 2's spec, and `forgedge.composition.grade_guided_compose`).
Raise `max_and_components` above `1` to get this step back — either
standalone (`EventDiscovery` used directly, as in the *Basic usage* example
above) or via `forge(two_pass_composition=False)`, which reproduces the
pre-#254 single-pass pipeline exactly. Everything below in this section
describes that step as it behaves whenever it does run — nothing about its
own mechanics changed in Phase 8.

The composer combines pairs (and optionally triples) of gate-passing events with
logical AND, searching for combinations that maintain temporal consistency.

**Admissibility rules for AND:**
- ✅ Same feature, different transforms (e.g. `identity` AND `pctrank_96` on RSI)
- ✅ Semantically distinct features (e.g. RSI AND volume)
- ❌ Same transform + different thresholds on the same feature (one is a superset of the other)

The composed event `A AND B` is re-submitted to the ConsistencyGate. Only
compositions that also pass the compound gate are promoted to candidates.

`max_and_components` (default `1` since Phase 8, was `2`) limits the number
of components — `1` disables this step. Values > 3 are accepted but
strongly discouraged due to structural overfitting risk.

**Example of a valid AND composition:**
```
RSI25 < 30.5                          (identity threshold, p10)
AND
pctrank(RSI25, w=96) < 0.10           (rolling pctrank, p10)

→ "RSI is in the oversold zone AND it has rarely been so in the last 96 bars"
→ Semantically rich, non-redundant combination
```

---

### Event-distribution diagnostic (`event_distribution_report`)

After `run()`, `ed.event_distribution_report` is **always** populated (issue
#215) — a public `str` attribute, independent of `config_report()`. The two
are complementary: `config_report()` resolves config-vs-config coherence
before any data is seen, so it can never detect that a specific asset's raw
candidates are pathologically failing the gate; `event_distribution_report`
aggregates the actual `GateResult` scalars collected for every raw
(pre-AND-composition) candidate produced by Step 3, whether or not
`cfg.retain_raw_events` keeps the full candidate population around
afterward.

It reports:
- the total raw candidate count and how many survived the ConsistencyGate;
- the observed median rate (tpm) and dispersion against the configured
  thresholds, and what fraction of candidates land on the wrong side of
  each;
- when gate survival is **below a 15% threshold**, a concrete parameter
  suggestion at the observed median.

```python
print(ed.event_distribution_report)
```

The text itself is emitted in Italian regardless of the calling context
(it is also the M1 stage log line), for example:

```
M1 Event Discovery — 8213 candidati generati, 512 superano il
Consistency Gate (6.2%).
tpm osservato: mediana=0.31 (soglia min_tpm=0.5, 61.4% sotto soglia).
dispersione osservata: mediana=2.10 (soglia effettiva=1.69, 38.0% sopra soglia).
Meno del 15% dei candidati generati supera il Consistency Gate (512/8213 = 6.2%).
Prova questi parametri (mediana osservata su tpm e dispersione): min_tpm<=0.31, dispersion_margin>=1.62.
```

---

## Data structures

### `EventCandidate`

Final artefact produced by `run()`. Each candidate represents a boolean event
that cleared the ConsistencyGate.

```python
cand = candidates[0]
print(cand.event_id)       # "EVT-close_rsi_25-PR-0042"
print(cand.expression)     # "pr_close_rsi_25_96 < 0.10"
print(cand.event_formula)  # "pctrank(close_rsi_25, w=96) < 0.10"
print(cand.activation_stats.n_activations)    # 329
print(cand.activation_stats.mean_tpm)         # 4.7
print(cand.consistency_gate.max_monthly_share) # 0.18
```

| Field | Type | Description |
|---|---|---|
| `event_id` | `str` | Unique identifier in the format `EVT-{feature}-{transform_abbr}-{idx:04d}` |
| `status` | `str` | Always `"CANDIDATE"` at this stage |
| `components` | `list[EventComponent]` | 1 for simple events, 2-3 for AND composition |
| `expression` | `str` | Human-readable boolean expression (transformed column names) |
| `event_formula` | `str` | Formula in standard mathematical notation |
| `activation_stats` | `ActivationStats` | Temporal distribution statistics |
| `consistency_gate` | `GateResult` | Always `passed=True` for candidates from `run()` |
| `event_series` | `pd.Series` | Boolean 0/1/NaN series with DatetimeIndex |
| `validation` | `ValidationResult \| None` | Walk-forward OOS result; None if not configured |
| `sql_expression` | `str` (property) | DuckDB-compatible expression |

#### `event_id` format

```
EVT-{source_feature[:20]}-{transform_abbrs}-{idx:04d}

Abbreviations:
  ID  = identity
  PR  = rolling_pctrank
  ZS  = rolling_zscore
  DL  = delta
  BN  = binary_native
  OH  = categorical_onehot
  AND = and_composition

Examples:
  EVT-close_rsi_25-PR-0042        (pctrank on RSI)
  EVT-close_rsi_25-IDxPR-0117     (AND: identity × pctrank on RSI)
```

### `EventComponent`

Each `EventCandidate` contains one or more components, one for each boolean
condition in the AND expression.

```python
comp = cand.components[0]
print(comp.source_feature)    # "close_rsi_25"
print(comp.transform)         # "rolling_pctrank"
print(comp.transform_params)  # {"window": 96}
print(comp.threshold)         # 0.1
print(comp.threshold_type)    # "distributional_p10"
print(comp.direction)         # "below"
print(comp.event_type)        # "threshold"
print(comp.expression)        # "pr_close_rsi_25_96 < 0.10"
print(comp.event_formula)     # "pctrank(close_rsi_25, w=96) < 0.10"
print(comp.source_cols)       # [] for arity-1, ["col_a", "col_b"] for arity-2
```

| Field | Description |
|---|---|
| `source_feature` | Source feature name (e.g. `close_rsi_25`, `ratio_close_ema_09_ema_25`) |
| `transform` | Transform type: `identity`, `rolling_pctrank`, `rolling_zscore`, `delta`, `binary_native`, `categorical_onehot` |
| `transform_params` | Transform parameters: `{"window": 96}` for pctrank/zscore, `{"lag": 3}` for delta, `{"diffnorm_std": 0.023}` for diffnorm |
| `transformed_col` | Name of the transformed column (e.g. `pr_close_rsi_25_96`) |
| `threshold` | Numeric threshold value (e.g. `0.10`, `30.5`) |
| `threshold_type` | Origin: `"distributional_p10"`, `"theoretical_z-1.5"`, etc. |
| `direction` | `"below"` or `"above"` |
| `event_type` | `"threshold"` (persistent) or `"crossing"` (transition) |
| `expression` | Human-readable string of the single condition |
| `event_formula` | Formula in standard notation (e.g. `pctrank(close_rsi_25, w=96) < 0.10`) |
| `source_cols` | Original native columns (empty for arity-1; `[col_a, col_b]` for arity-2; `[val, lo, hi]` for arity-3) |
| `sql_expression` | DuckDB-compatible SQL expression |

### `ActivationStats`

```python
stats = cand.activation_stats
print(stats.n_activations)               # total activations
print(stats.n_active_months)             # months with at least one activation
print(stats.zero_months)                 # months with no activations
print(stats.max_monthly_share)           # share of the most concentrated month
print(stats.mean_tpm)                    # average activations per month
print(stats.index_of_dispersion)         # per-bar Index of Dispersion (Var/Mean of monthly counts)
print(stats.n_episodes)                  # number of distinct activation episodes
print(stats.episode_index_of_dispersion) # Index of Dispersion of monthly episode counts
print(stats.n_eff)                       # effective sample size: n_episodes / max(episode_ID, 1.0)
```

`index_of_dispersion`, `n_episodes`, `episode_index_of_dispersion` and
`n_eff` are the same diagnostic fields `GateResult` carries (see Step 4) —
always computed, not only when the active mode's criteria actually read
them.

---

## Mathematical formula (`event_formula`)

The `event_formula` field on `EventComponent` and the `event_formula` property
on `EventCandidate` provide a standard mathematical notation representation,
more readable than transformed column names.

The formula is built in three phases by `_build_event_formula`:

| Phase | Example outputs |
|---|---|
| Feature (`_formula_feature`) | `close_rsi_25`, `close / open`, `(sma_09 - sma_25) / std(sma_09 - sma_25)  [std=0.0032]`, `bb_pct_b(close, bb_lower, bb_upper)` |
| Transform (`_formula_transform`) | `pctrank(close_rsi_25, w=96)`, `zscore(close / open, w=48)`, `Δ(close_rsi_25, lag=1)` |
| Condition (`_formula_condition`) | `... < P10 [P10=0.10]`, `... > -1.5`, `... crosses ↓ P5 [P5=-0.03]` |

**`FORMULA [VALUE]` convention:** distributional thresholds (derived from
in-sample percentiles of the transformed series) are displayed as `P10 [P10=0.10]`,
where `P10` is the percentile label and `0.10` is the actual value on that dataset.
Theoretical z-score thresholds (`-2.0`, `-1.5`, etc.) are fixed constants and are
shown without annotation. The same convention applies to the `diffnorm` feature
denominator: `std(x-y)  [std=0.0032]` indicates the IS standard deviation used
for normalisation is `0.0032`.

Complete examples for an AND candidate:

```python
cand.event_formula
# "(pctrank(close_rsi_25, w=48) < P10 [P10=0.10]) AND (zscore(close_rsi_25, w=96) > -1.5)"
```

---

## SQL export (`sql_expression`)

Each `EventComponent` contains a DuckDB-compatible SQL expression that replicates
the condition identically to the pandas pipeline, using the same rolling windows.

```python
import duckdb
rel = duckdb.from_df(ed.df.reset_index())

# Check which bars have the event active
query = f"SELECT open_dt, ({cand.sql_expression})::INT AS active FROM df"
result = rel.query("df", query)
```

Features of the SQL expression:
- `pctrank`: uses `list_filter` lambdas (DuckDB ≥ 0.8) with average-method rank
  to exactly reproduce `pandas.rank(pct=True)`.
- `zscore` and `delta`: use standard window functions (`AVG`, `STDDEV_SAMP`, `LAG`).
- `min_periods = max(2, window // 2)` replicated via `CASE WHEN`.
- Order guaranteed via `ORDER BY {timestamp_col}`.

---

## Replay on new data: `EventCandidate.apply(df)`

The `apply(df)` method reconstructs the boolean series on an OOS DataFrame
using only the parameters stored in the components, without re-running the pipeline.

```python
oos_kpi = pd.read_parquet("kpi_oos.parquet")
oos_series = cand.apply(oos_kpi)
print(oos_series.value_counts())
```

The native columns (`source_cols`) must be present in the DataFrame for correct
reconstruction. The `build_feature_series(comp, df)` function reconstructs
the underlying continuous feature (without the transform), useful for passing
it to Alpha Discovery:

```python
from forgedge.event_discovery.models import build_feature_series

feature_series = build_feature_series(cand.components[0], oos_kpi)
```

### Persisting to disk: `EventCandidate.persist(path)`

`persist(path)` serialises the full candidate to disk as a pickle file,
preserving the complete structure (components, thresholds, activation stats,
walk-forward validation). To reload, use the standard `pickle` module:

```python
# Save
cand.persist("contracts/ev_btc_rsi25.pkl")

# Reload — returns a ready-to-use EventCandidate
import pickle
cand_loaded = pickle.load(open("contracts/ev_btc_rsi25.pkl", "rb"))
cand_loaded.apply(new_df)   # works immediately
```

`persist()` is the recommended way to archive individual candidates across
sessions or share them between processes. For multi-candidate archives, the
JSON/CSV form of `to_dict()` is more readable but not invertible (it does
not reconstruct the full `EventCandidate` object); `persist()` is the only
method that guarantees a complete round-trip.

### OOS re-evaluation: `EventCandidate.update_event(df)`

`update_event(df)` re-evaluates the candidate on a new DataFrame **in place**,
using only the IS parameters fixed at discovery time (thresholds, transform
windows) — without recalibrating them. It updates:

- `event_series` — new boolean series on the passed DataFrame
- `activation_stats` — recomputed from the new series
- `consistency_gate` — metrics and `passed` recomputed with the original
  `gate_params` (if stored; without `gate_params` the gate is marked `False`)

```python
# Evaluate a candidate discovered on historical data against more recent data
new_kpi = pd.read_parquet("btc_1h_new.parquet")
cand.update_event(new_kpi)

print(f"Activations on the new period: {cand.activation_stats.n_activations}")
print(f"Gate passed: {cand.consistency_gate.passed}")
print(f"Active months: {cand.activation_stats.n_active_months}")

# The signal is now updated on the new data
new_signal = cand.event_series
```

`update_event()` is the alternative to `apply()` when you also want the
activation stats and gate updated (not just the boolean series): after the
call, `cand.event_series` and `cand.activation_stats` reflect the new period.
Thresholds never change — they remain fixed from the IS discovery run.

---

### `CustomEvent` — manual event injection

`CustomEvent` lets a user define a hypothesis formula and inject it directly into
Alpha Discovery (Module 2) and Rule Discovery (Module 3), bypassing the automatic
Event Discovery pipeline entirely. The formula is evaluated with
`pd.DataFrame.eval()` so it can reference any column in the DataFrame — including
proprietary indicators that the automatic `FeatureGenerator` would never produce.

Import: `from forgedge import CustomEvent`

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `formula` | `str` | — | Boolean expression for `pd.DataFrame.eval()`. Examples: `"rsi_14 > 70"`, `"close < bb_lower and volume > 1e6"`. |
| `name` | `str` | `""` | Human-readable label used in reports. Defaults to the formula text. |

**Methods:**

- `apply(df)` → `pd.Series[bool]`: Evaluates the formula on `df`. NaN results are treated as inactive (False).
- `to_event_candidate(df, gate_params=None)` → `EventCandidate`: Builds a full `EventCandidate` from the formula. The consistency gate and activation stats are populated on `df`; thresholds are not distributional (there are none — the formula is user-defined).

**Important behaviour:**

- When used via `forge(manual_events=[...])`, each `CustomEvent` still crosses the ConsistencyGate. A failure emits a `logger.warning` but does not drop the event — the user's hypothesis is always forwarded.
- AND composition is not performed in manual injection mode.
- The resulting `EventCandidate` has `event_id = "CUSTOM-{name}"`.

**Code example — standalone usage:**

```python
from forgedge import CustomEvent, AlphaDiscovery, AlphaConfig

# Define a custom hypothesis
ev = CustomEvent("rsi_14 > 70 and volume > 1e6", name="rsi_overbought_volume")

# Apply to any frame
signal = ev.apply(df)                            # pd.Series bool

# Build a full EventCandidate for M2/M3
cand = ev.to_event_candidate(df)
print(cand.event_id)                             # "CUSTOM-rsi_overbought_volume"
print(cand.activation_stats.n_activations)

# Run Alpha Discovery directly
ad = AlphaDiscovery(df, [cand], AlphaConfig(asset="BTC", timeframe="1H"))
contracts = ad.run()
```

**Code example — via `forge()` with manual injection:**

```python
from forgedge import forge, CustomEvent

events = [
    CustomEvent("close_adj_v2 < 100", name="close_below_100"),
    CustomEvent("rsi_14 < 25 and spread_ema < -0.02", name="rsi_extreme_spread"),
]

# manual_events bypasses Module 1 (Event Discovery); Module 2 and 3 run normally
result = forge(
    kpi,
    ticker="BTCUSDC",
    timeframe="1H",
    manual_events=events,
)
for contract, resp in result.rule_responses:
    print(contract.alpha_id, resp.verdict)
```

Note: `manual_events` and `event_discovery_config` are mutually exclusive —
passing both raises `ValueError`.

---

## Walk-forward OOS validation

Walk-forward validation is optional and verifies that an in-sample event
maintains the same activation structure on unseen data. It is a measure of
stability, not predictive power (that is Module 2's job).

```python
from forgedge.event_discovery.models import EventWalkForwardConfig

config = DiscoveryConfig(
    train_ratio=0.80,           # 80% IS, 20% OOS
    walk_forward=EventWalkForwardConfig(
        n_splits=3,             # divide OOS into 3 windows
        min_pass_rate=0.60,     # must pass the gate in at least 2/3 windows
    ),
)
ed = EventDiscovery(enriched_kpi, config=config)
candidates = ed.run()

# Only OOS-stable candidates
stable = ed.validated_candidates()
print(f"{len(stable)} stable candidates out of {len(candidates)}")
```

### How it works

1. The dataset is split: first `train_ratio` rows = IS, the rest = OOS.
2. The full pipeline (Steps 0–5) runs **on IS data only**.
3. For each candidate, the OOS period is divided into `n_splits` equal
   windows (folds).
4. On each fold, `apply()` reconstructs the boolean series. The last 168 IS
   bars are prepended as rolling context to avoid NaN warmup at the start of
   the fold.
5. The ConsistencyGate is applied to each fold with `min_episodes` **forced
   to 0** rather than scaled (issue #177): being an absolute count,
   `min_episodes` does not survive being applied to a shorter window — on a
   2-month fold the class default of 10 demands 5 episodes/month against an
   in-sample requirement that can be as low as 1/month, ~5x stricter by
   construction. `min_tpm`, `max_dispersion` and `dispersion_margin` are
   rate/ratio invariant and carry over **unchanged** from the IS
   `gate_params` (or from `wf.oos_gate_params` when explicitly set — still
   with `min_episodes` forced to 0 either way).
6. Each fold is separately checked for **testability**: the candidate's own
   in-sample rate implies an expected episode count `lam = is_rate x
   n_months` for that fold's length. When `lam < MIN_FOLD_LAMBDA` (3.0), an
   empty fold has ≥5% probability of occurring even for a candidate that
   kept its in-sample rate perfectly — such a fold is marked
   `indeterminate` and **excluded from the `pass_rate` denominator**, never
   counted as a failure.
7. A candidate is OOS-stable (`passed=True`) when it passes the gate in at
   least `min_pass_rate` of the **testable** folds (`n_testable`, not
   `n_folds`). When no fold is testable, `passed` is `None` — inconclusive,
   not failed; `forge()`'s `only_validated_events` filter treats `None`
   distinctly from `False` so a walk-forward that could not run does not
   silently drop every candidate.

### `ValidationResult` output

`ValidationResult` fields: `n_folds` (total folds run), `n_passed`, `n_testable`
(folds long enough to say anything — the `pass_rate` denominator), `pass_rate`
(`n_passed / n_testable`, `nan` when nothing was testable), `passed`
(`bool | None`), `fold_results` (per-fold detail, including the indeterminate
folds). Each `FoldResult` additionally carries `indeterminate` (bool) and
`lam` (the fold's expected episode count at the candidate's in-sample rate).

```python
for cand in candidates:
    if cand.validation:
        v = cand.validation
        print(f"{cand.event_id}: {v.n_passed}/{v.n_testable} testable folds "
              f"({v.n_folds} total), pass_rate={v.pass_rate:.2f}, stable={v.passed}")
        # v.passed is None when no fold was testable — inconclusive, not failed
        for fold in v.fold_results:
            if fold.indeterminate:
                status = f"INDETERMINATE (lam={fold.lam:.2f} < 3.0)"
            else:
                status = "OK" if fold.passed else fold.gate_result.fail_reason
            print(f"  fold {fold.fold_idx}: {fold.n_rows} bars, {status}")
```

---

## Full configuration reference

### `DiscoveryConfig`

| Parameter | Default | Description |
|---|---|---|
| `gate_params` | `GateParams()` | ConsistencyGate thresholds |
| `max_categorical_classes` | 20 | Categorical columns with more classes are excluded from the pipeline |
| `scale_free_overrides` | `None` | Manual override: `{"col": True/False}` |
| `timestamp_col` | `"open_dt"` | Datetime column name (or DatetimeIndex name) |
| `max_and_components` | 1 (was 2 pre-#254 Phase 8) | Maximum components per AND composition; `1` disables Step 5 (this module's own composition), the precondition for `forge(two_pass_composition=True)` — the default. Raise for the legacy single-pass path or a caller with its own composition stage |
| `retain_raw_events` | `False` (was `True` pre-#254 Phase 8) | Whether `EventDiscovery.raw_events` stays populated after `run()`; needed only by `TargetOptimizer`, which sets `True` explicitly on its own fallback config |
| `train_ratio` | 1.0 | IS fraction (1.0 = no OOS split, walk-forward disabled) |
| `walk_forward` | `None` | Walk-forward config; `None` = no OOS validation |

### `GateParams`

| Parameter | Default | Description |
|---|---|---|
| `min_tpm` | 0.5 | Minimum average triggers per month, in the unit chosen by `event_counting` (episodes/month in `"episode"` mode, bars/month in `"bar"` mode) |
| `max_dispersion` | 1.5 | Maximum per-bar Index of Dispersion — **`"bar"` mode only**; not read in `"episode"` mode |
| `dispersion_margin` | 1.3 | **`"episode"` mode only** — multiplier over the Poisson χ² floor: `eff_max_dispersion = poisson_floor(n_months) x dispersion_margin` |
| `event_counting` | `"episode"` | Counting unit for the rate/dispersion criteria: `"episode"` or `"bar"` |
| `min_episodes` | 10 | Absolute floor on episode count (statistical-power guard, `"episode"` mode only); applied **in-sample only**, forced to 0 on walk-forward OOS folds |
| `episode_gap` | 1 | Maximum gap (bars) that still belongs to the same episode; `0` = strict consecutive runs |

### `EventWalkForwardConfig`

| Parameter | Default | Description |
|---|---|---|
| `n_splits` | 3 | Number of OOS windows |
| `min_pass_rate` | 0.60 | Minimum fraction of **testable** windows that must pass the gate |
| `oos_gate_params` | `None` | Explicit OOS gate; if `None`, the IS `gate_params` are used, with `min_episodes` always forced to 0 regardless |

---

## Timestamp parsing

The timestamp column accepts all common formats:

| Format | Behaviour |
|---|---|
| `DatetimeIndex` on the DataFrame | Used directly |
| `datetime64` column | Parsed with `pd.to_datetime` |
| Numeric column (Unix epoch) | Unit inferred from median value: `<1e10`=s, `<1e13`=ms, `<1e16`=µs, else ns |
| String column | Parsed as ISO 8601 |

The timestamp column is **removed** from `df` after parsing so it does not
enter the TypeClassifier. `ed.df` always has a `DatetimeIndex`.

---

## Reading and filtering results

```python
# Tabular summary of all candidates
summary = ed.summary()
print(summary.columns.tolist())
# ['event_id', 'status', 'expression', 'n_activations', 'n_active_months',
#  'zero_months', 'max_monthly_share', 'mean_tpm', 'index_of_dispersion',
#  'gate_passed', 'oos_pass_rate', 'oos_n_passed', 'oos_n_folds', 'oos_stable']  # OOS cols only if walk-forward configured

# Candidates with arity 2 (AND composition)
and_candidates = [c for c in candidates if len(c.components) == 2]

# Candidates using pctrank
pctrank_cands = [c for c in candidates if any(comp.transform == "rolling_pctrank"
                                               for comp in c.components)]

# Boolean series of a candidate
series = cand.event_series   # pd.Series with DatetimeIndex
print(series.resample("ME").sum())  # activations per month
```
