# Module 1 — Event Discovery (Codebase Spec)

> **Code reference:** `src/forgedge/event_discovery/`
> **Functional analysis:** `docs/modules/EventDiscovery.md`
> **Status:** ✅ Implemented. Core logic aligned with the functional analysis;
> additional features are present in the code (SQL export, OOS apply,
> detailed walk-forward) that are not documented.

---

## 1. Position in the pipeline

```
KPI Table + regime (from Module 0)
        │
        ▼
  EventDiscovery.run()
        │
        ▼
   list[EventCandidate]  ──► Alpha Discovery (Module 2)
```

Module 1 never sees the forward return. It operates solely on the temporal
structure of indicators in the KPI Table.

---

## 2. Public interface

### `EventDiscovery` (`discovery.py`)

```python
EventDiscovery(kpi_table, config=None)
```

| Method / Property | Description |
|---|---|
| `run() → list[EventCandidate]` | Runs the full pipeline; returns all candidates that pass the gate |
| `summary() → pd.DataFrame` | Tabular summary of all candidates (after `run()`) |
| `validated_candidates() → list[EventCandidate]` | Only candidates that pass walk-forward OOS validation |
| `get_classifications() → dict` | Column classifications from Step 0 (for debugging) |
| `is_period → tuple or None` | (start, end) of the in-sample period |
| `oos_period → tuple or None` | (start, end) of the OOS period, or None if no split |
| `df` | Post-pipeline KPI Table (with derived columns added by Step 1) |

---

## 3. Five-step pipeline

### Step 0 — TypeClassifier (`classifier.py`)

Classifies every non-timestamp column of the DataFrame:

| Type | Criterion |
|---|---|
| `BINARY` | Exactly 2 distinct non-null values |
| `CATEGORICAL` | Non-numeric, or numeric with distinct values ≤ `max_categorical_classes` |
| `CONTINUOUS` | Numeric with more than 2 distinct values |

For CONTINUOUS columns, **scale-free** detection is also run
(asymmetric conservative heuristic — false negative costs less than false positive).
The result may be overridden via `scale_free_overrides`.

Output per column: `ColumnClassification` with:
- `col_type: ColumnType`
- `n_distinct: int`
- `is_scale_free: bool | None`
- `scale_free_override: bool | None`
- `effective_scale_free: bool` (property: override > automatic > False)
- `scale_free_overridden: bool` (property: True if override contradicts automatic)

CATEGORICAL columns with `n_distinct > max_categorical_classes` (default: 20)
are classified but excluded from event generation.

---

### Step 1 — FeatureGenerator (`feature_generator.py`)

Builds derived features from the native feature catalogue.

| Arity | Operation | Example | Condition |
|---|---|---|---|
| 1 (pass-through) | Identity | `close_rsi_25` | Scale-free only |
| 2 (ratio) | `a / b` | `ratio_close_ema_09_ema_25` | Same family, same source |
| 2 (spread%) | `(a - b) / b` | `spread_close_bb_upper_lower` | Same family |
| 2 (diffnorm) | `(a - b) / std(a-b)` | `diffnorm_close_sma_09_sma_25` | Same family, same source |
| 3 (bb_pct_b) | `(val - lo) / (hi - lo)` | `bb_pct_b_close_bb_lower_upper` | Bollinger Bands |
| 3 (pos) | `(val - min) / (max - min)` | `pos_close_min_24_max_48` | Rolling min/max |

For `diffnorm` features the standard deviation is computed on the in-sample set
and stored in `transform_params["diffnorm_std"]` for OOS replay.

---

### Step 2 — TransformLayer (`transform_layer.py`)

Applies 4 temporal transforms to every feature in the catalogue:

| Transform | Code | Windows | Applies to |
|---|---|---|---|
| Identity | `identity` | — | Scale-free only |
| Rolling pctrank | `rolling_pctrank` | 48, 96, 168 bars | All continuous |
| Rolling zscore | `rolling_zscore` | 48, 96, 168 bars | All continuous |
| Delta (diff) | `delta` | 1, 3, 6, 12 bars | All continuous |

`min_periods` for rolling windows: `max(2, window // 2)`.

---

### Step 3 — EventGenerator (`event_generator.py`)

Converts each transformed series into boolean events:

| Event type | Description |
|---|---|
| `threshold` | Persistent: series < threshold or series > threshold |
| `crossing` | Transition: bar t crosses the threshold relative to bar t-1 |

**Thresholds from the Threshold Catalog (distributional):**
- Percentiles of the transformed series: p3, p5, p10, p20, p25, p75, p80, p90, p95, p97
- Theoretical z-score thresholds: -2.0, -1.5, -1.0, 0, +1.0, +1.5, +2.0 (for zscore transform)

**BINARY columns:** one event per value (0 or 1) — transform: `binary_native`.
**CATEGORICAL columns:** one event per class (one-hot) — transform: `categorical_onehot`.

---

### Step 4 — ConsistencyGate (`consistency_gate.py`)

Filters events by their temporal activation distribution.
An event **passes** if and only if it satisfies all 4 criteria:

| Criterion | Parameter | Default | Description |
|---|---|---|---|
| Volume | `min_act` | 50 | Total activations across the full dataset |
| Coverage | `min_months` | 8 | Calendar months with at least one activation |
| Concentration | `max_conc` | 0.40 | Maximum share of activations in a single month |
| Frequency | `min_tpm` | 2.0 | Average activations per month (total / n_months) |

`fail_reason` in `GateResult` reports the first failing criterion.

---

### Step 5 — ANDComposer (`and_composer.py`)

Combines gate-passing events with logical AND.
Composition is allowed between:
- **Same feature, different transforms** (e.g. identity + pctrank96 on the same RSI)
- **Semantically distinct features**

Composition is **forbidden** between:
- Same transform + different thresholds on the same feature (one is a superset of the other)

Composed events are re-submitted to the ConsistencyGate.
`max_and_components` (default: 2) caps the number of components.

---

## 4. Core data structures

### `EventCandidate` (`models.py`)

Output artefact of Module 1.

| Field | Type | Description |
|---|---|---|
| `event_id` | `str` | ID in format `EVT-{feature}-{transform_abbr}-{idx:04d}` |
| `status` | `str` | Always `"CANDIDATE"` at this stage |
| `components` | `list[EventComponent]` | 1 for simple events, 2-3 for AND composition |
| `expression` | `str` | Human-readable boolean expression (components joined with ` AND `) |
| `activation_stats` | `ActivationStats` | Temporal distribution statistics |
| `consistency_gate` | `GateResult` | Always `passed=True` for candidates returned by `run()` |
| `event_series` | `pd.Series` | Boolean 0/1/NaN series with DatetimeIndex |
| `validation` | `ValidationResult | None` | Walk-forward OOS result; None if not configured |
| `sql_expression` | `str` (property) | DuckDB-compatible boolean expression |

**`apply(df)` method:** reconstructs the boolean series on new OOS data,
replicating feature construction + temporal transform + threshold using the
parameters stored in the components.
For `diffnorm` features, uses the in-sample `transform_params["diffnorm_std"]`.

### `EventComponent` (`models.py`)

| Field | Type | Description |
|---|---|---|
| `source_feature` | `str` | Source feature name (e.g. `close_rsi_25`) |
| `transform` | `str` | Transform type applied |
| `transform_params` | `dict` | Transform parameters (e.g. `{"window": 96}`) |
| `transformed_col` | `str` | Column name after the transform |
| `threshold` | `float` | Binarisation threshold |
| `threshold_type` | `str` | Threshold origin (e.g. `"distributional_p10"`) |
| `direction` | `str` | `"below"` or `"above"` |
| `event_type` | `str` | `"threshold"` or `"crossing"` |
| `expression` | `str` | Human-readable condition string |
| `source_cols` | `list` | Original native columns (for arity-2/3 features) |
| `sql_expression` | `str` | DuckDB-compatible SQL expression |

### `GateResult` (`models.py`)

| Field | Description |
|---|---|
| `passed: bool` | True if all 4 criteria are satisfied |
| `n_activations: int` | Total activations |
| `n_active_months: int` | Months with at least one activation |
| `max_monthly_share: float` | Share of the most concentrated month |
| `mean_tpm: float` | Average activations per month |
| `fail_reason: str | None` | First failing criterion, or None if passed |

### `ActivationStats` (`models.py`)

Same fields as `GateResult` plus `zero_months` (months with no activations).

---

## 5. `event_id` format

The code generates:
```
EVT-{source_feature[:20]}-{transform_abbrs}-{idx:04d}
```

Transform abbreviations:
```python
"identity"           → "ID"
"rolling_pctrank"    → "PR"
"rolling_zscore"     → "ZS"
"delta"              → "DL"
"binary_native"      → "BN"
"categorical_onehot" → "OH"
"and_composition"    → "AND"
```

Examples:
```
EVT-close_rsi_25-PR-0042       # pctrank on RSI
EVT-close_rsi_25-PRxZS-0117    # AND: pctrank × zscore on RSI
```

---

## 6. Walk-forward OOS validation

Optional, configured via `DiscoveryConfig(train_ratio=..., walk_forward=WalkForwardConfig(...))`.

### Flow

1. The dataset is split temporally: first `train_ratio` rows = IS, the rest = OOS.
2. The full pipeline (Steps 0–5) runs on IS data only.
3. For each candidate, the OOS period is divided into `n_splits` equal windows.
4. On each window, `EventCandidate.apply()` reconstructs the boolean series.
   The last `_MAX_CONTEXT_BARS = 168` IS bars are prepended as rolling context
   to avoid NaN warmup at the start of each fold.
5. The ConsistencyGate is applied with gate parameters scaled proportionally
   to the OOS window size:
   - `min_act` and `min_months` scaled proportionally (floor: 5 and 1)
   - `max_conc` and `min_tpm` unchanged (they are rates, not counts)
6. A candidate is OOS-stable if it passes in at least `min_pass_rate` (default 0.6) of the windows.

### `WalkForwardConfig`

| Parameter | Default | Description |
|---|---|---|
| `n_splits` | 3 | Number of OOS windows |
| `min_pass_rate` | 0.6 | Minimum fraction of windows that must pass |
| `oos_gate_params` | None | Explicit OOS gate; if None, parameters are scaled automatically |

### Output in `EventCandidate.validation`

```python
ValidationResult:
    n_folds: int
    n_passed: int
    pass_rate: float
    passed: bool
    fold_results: list[FoldResult]   # per-fold detail
```

---

## 7. SQL export (`sql_expression`)

Each `EventComponent` carries a `sql_expression` field:
a DuckDB-compatible boolean expression that replicates the condition
applying the same rolling windows, pctrank, zscore, and delta.

Features:
- Uses `list_filter` lambdas for pctrank (DuckDB ≥ 0.8)
- Uses standard window functions (`AVG`, `STDDEV_SAMP`, `LAG`) for zscore and delta
- `min_periods = max(2, window // 2)` replicated via `CASE WHEN`
- `ORDER BY` uses `DiscoveryConfig.timestamp_col` (default `"open_dt"`)

Example usage:
```python
import duckdb
rel = duckdb.from_df(ed.df.reset_index())
rel.query("df", f"SELECT *, ({candidate.sql_expression})::INT AS active FROM df")
```

---

## 8. Configuration

### `DiscoveryConfig` (`discovery.py`)

| Parameter | Default | Description |
|---|---|---|
| `gate_params` | `GateParams()` | ConsistencyGate thresholds (Step 4) |
| `max_categorical_classes` | 20 | Categorical column threshold (> N: excluded from pipeline) |
| `scale_free_overrides` | `None` | Manual scale-free flags per column |
| `timestamp_col` | `"open_dt"` | Datetime column name (or DatetimeIndex name) |
| `max_and_components` | 2 | Maximum components per AND composition |
| `train_ratio` | 1.0 | IS fraction (1.0 = no OOS split) |
| `walk_forward` | `None` | Walk-forward config; None = no OOS validation |

### `GateParams` (`models.py`)

| Parameter | Default | Description |
|---|---|---|
| `min_act` | 50 | Minimum total activations |
| `min_months` | 8 | Minimum active months |
| `max_conc` | 0.40 | Maximum monthly concentration |
| `min_tpm` | 2.0 | Minimum average activations per month |

---

## 9. Timestamp parsing

`DiscoveryConfig.timestamp_col` can come from:
1. **DatetimeIndex** — used directly
2. **datetime64 column** — parsed with `pd.to_datetime`
3. **Numeric column** — unit inferred automatically from the median value (s/ms/us/ns)
4. **String column** — parsed with `pd.to_datetime` (ISO 8601)

The timestamp column is **removed** from `self.df` after parsing
(it is redundant once the index is set and must not enter the TypeClassifier).

---

## 10. Alignment with the functional analysis

### ✅ Aligned

- 5 steps + type classification (Steps 0–5)
- `GateParams` with 4 criteria (min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)
- `ColumnType`: CONTINUOUS, BINARY, CATEGORICAL
- `ColumnClassification` with `effective_scale_free` and `scale_free_override`
- `max_categorical_classes = 20`
- Feature generation arity 1/2/3
- 4 transforms (identity/pctrank/zscore/delta) with documented windows
- Threshold and crossing events with distributional and theoretical thresholds
- AND composition with admissibility rules
- Gate re-applied on composed events
- Walk-forward OOS validation (`train_ratio`, `WalkForwardConfig`)
- `EventCandidate` with all documented fields

### ➕ Added in code (not in the functional analysis)

- **`sql_expression`** on `EventComponent` and `EventCandidate` — DuckDB export
- **`EventCandidate.apply(df)`** — deterministic replay of the boolean series on new data
- **`build_feature_series(comp, df)`** — module function for the underlying continuous feature
- **`_MAX_CONTEXT_BARS = 168`** — IS tail used as rolling context in OOS validation
- **`_scale_gate_params()`** — proportional scaling of IS parameters for OOS windows
- **`validated_candidates()`** — convenience method for OOS-stable candidates only
- **`is_period` and `oos_period`** — properties for IS/OOS period timestamps
- **`source_cols`** on `EventComponent` — original native columns for arity-2/3 features
- **`diffnorm_std`** stored in `transform_params` — IS normaliser preserved for OOS

### ⚠️ Divergences from the functional analysis

- **`event_id` format:** The code generates `EVT-close_rsi_25-PR-0042` (sequential index).
  The functional analysis shows a more detailed format encoding the window and threshold
  in the ID (e.g. `EVT-close_rsi_25-ID×PR-P105-W096-P010`).
  The code uses a sequential index rather than encoding parameters in the ID.

- **AND composition separator:** The code uses lowercase `x` (`PRxZS`);
  the functional analysis uses `×` (Unicode multiplication, `ID×PR`).
