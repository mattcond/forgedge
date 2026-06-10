# Module 2 — Alpha Discovery (Codebase Spec)

> **Code reference:** `src/forgedge/alpha_discovery/`
> **Functional analysis:** `docs/modules/AlphaDiscovery.md`
> **Status:** ✅ Implemented. Core logic aligned with the functional analysis;
> additional normalisation caps and fields are present in the code.
> The documented `rule_discovery_hints` field is absent from the model.

---

## 1. Position in the pipeline

```
KPI Table + regime  (Module 0)
list[EventCandidate] (Module 1)
        │
        ▼
  AlphaDiscovery.run()
        │
        ▼
  list[AlphaContract]   (all: HYPOTHESIS + REJECTED)
  promoted_contracts()  (HYPOTHESIS only)  ──► Rule Discovery (not implemented)
```

Alpha Discovery is the **first module that sees the forward return**.
It neither recomputes nor modifies event thresholds.

---

## 2. Public interface

### `AlphaDiscovery` (`discovery.py`)

```python
AlphaDiscovery(kpi_table, event_candidates, config=None)
```

| Method / Property | Description |
|---|---|
| `run() → list[AlphaContract]` | Evaluates all candidates; returns all contracts (promoted + rejected) |
| `promoted_contracts() → list[AlphaContract]` | Only contracts with `status == "HYPOTHESIS"` |
| `summary() → pd.DataFrame` | Tabular summary, sorted by `composite_score` descending |
| `base_rate` | Win rate without filter (populated by `run()`) |
| `market_structure` | Market structure (Hurst + ACF), computed once per session |

`AlphaDiscovery` consumes `ed.df` (Event Discovery's post-pipeline table) which
already carries the `regime` column (from Market Context) and derived feature columns.
When a derived feature is absent from the table, it is reconstructed deterministically
from the parameters stored in the component.

---

## 3. Eight-step pipeline

### Step 1 — Target definition (`target.py`)

```python
build_target(close, tgt) → (fwd_return, target_binary, base_rate)
```

- `fwd_return`: maximum forward return over `holding_period_h` bars
  (for direction `long`: max(close[t+1..t+h]) / close[t] - 1)
- `target_binary`: 1 if `fwd_return >= sell_pct`, 0 otherwise
- `base_rate`: mean of `target_binary` over all valid bars (unconditional win rate)

`sell_pct` and `holding_period_h` come from `TargetDefinition`.
`fee_per_side` is informational only (not netted from the target — that is Rule Discovery's job).

---

### Step 2 — Market structure analysis (`market_structure.py`)

```python
analyse_market_structure(close, fwd_return) → MarketStructure
```

| Field | Description |
|---|---|
| `hurst` | Hurst exponent of the close price (DFA) |
| `hurst_interpretation` | `"mean_reverting"` / `"random_walk"` / `"trending"` |
| `expected_family` | `"mean_reversion"` / `"momentum"` / `"none"` |
| `autocorr` | ACF of the forward return at selected lags |

Computed **once per session**, not per candidate.

---

### Step 3 — IC measurement (`discovery.py: _measure_ic`)

**Spearman rank correlation** between the underlying continuous feature and `fwd_return`.

**Admission gate:** a candidate is admitted unless BOTH are true:
- `|IC| < ic_min_abs` (default: 0.02)
- `p_value > ic_max_p` (default: 0.05)

That is: a candidate is **rejected** at the IC gate only when IC is both weak AND non-significant.
If it is significant (low p) even with a small IC, it passes the IC gate.

**Rolling IC stability:** evaluated on ≈20 evenly-spaced windows of width `rolling_ic_window`
(default: 60 days in bars, inferred from `bars_per_day`).
A candidate is stable if the same IC sign appears in at least 70% of the windows.

Output: `ICResult` with fields:
- `ic`, `p_value`, `n`: global statistics
- `rolling_ic_stable: bool | None`: True if sign consistent in ≥70% of windows
- `rolling_ic_mean: float | None`: mean rolling IC
- `rolling_sign_consistency: float | None`: fraction of windows with the same sign as global IC
- `admitted: bool`: True if passes the IC gate

---

### Step 4 — Win rate analysis (`discovery.py: _measure_event`)

Measures the predictive power of the **binary event** on the target:

| Metric | Description |
|---|---|
| `win_rate` | `target_binary.mean()` on bars where the event is active |
| `lift` | `win_rate - base_rate` |
| `fwd_return_mean` | Mean forward return on active bars |
| `cohens_d` | Effect size: standardised difference between active and inactive returns |
| `t_stat`, `p_value` | Independent t-test (one-sided: `alternative="greater"`) |

Output: `EventStats`.

---

### Step 5 — Regime sensitivity (`discovery.py: _measure_regimes`)

For each regime with at least `min_regime_obs` (default: 10) observations:
- Spearman IC of the continuous feature vs `fwd_return`
- Conditional win rate of the event

**Regime strength classification:**

| Strength | Condition |
|---|---|
| `"strong"` | `p < 0.05` and `|IC| >= 0.05` |
| `"moderate"` | `p < 0.05` and `|IC| < 0.05` |
| `"negligible"` | `p >= 0.05` (not significant) |
| `"insufficient"` | Fewer than `min_regime_obs` observations |

**Regime dependency classification:**

| Dependency type | Condition |
|---|---|
| `"agnostic"` | All evaluated regimes are significant (strong or moderate) and ≥2 |
| `"conditional"` | More than 1 significant regime, but not all |
| `"specific"` | Exactly 1 significant regime |
| `"broken"` | 0 significant regimes |
| `"unknown"` | No regime column available |

If `use_stable_regime_only = True` and `regime_stable` is present,
only bars with `regime_stable = True` are used (no transition bars).

Output: `RegimeAnalysis` with `per_regime`, `dependency_type`, `active_regimes`,
`weak_regimes`, `regime_breadth` (fraction of significant over evaluated regimes).

---

### Step 6 — Alpha scoring (`discovery.py: _score`)

**Composite score** (0–1):

```
composite = Σ(w_i * norm_i) / Σ(w_i)
```

| Term | Default weight | Normalisation |
|---|---|---|
| IC magnitude | 0.25 | `min(|IC| / 0.10, 1.0)` — saturates at IC=10% |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` — saturates at lift=30% |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` — saturates at d=0.80 |
| Regime breadth | 0.20 | `regime_breadth` (already 0–1) |

When regime information is unavailable, the `regime_breadth` term is dropped
and the remaining weights are **renormalised** (not replaced with 0).

**Grade from composite score:**

| Grade | Score |
|---|---|
| `A` | ≥ 0.75 |
| `B+` | ≥ 0.60 |
| `B` | ≥ 0.45 |
| `C` | < 0.45 |

---

### Step 7 — Alpha Contract compilation (`discovery.py: _build_contract`)

**Promotion gate** — a candidate is promoted (`status = "HYPOTHESIS"`)
only when it satisfies **all** of the following criteria:

| Criterion | Parameter | Default |
|---|---|---|
| IC admitted | `ic_min_abs`, `ic_max_p` | 0.02, 0.05 |
| Lift ≥ threshold | `min_lift` | 0.08 (+8pp) |
| Cohen's d ≥ threshold | `min_cohens_d` | 0.15 |
| Activations ≥ threshold | `min_activations` | 30 |
| FDR Benjamini-Hochberg | `fdr_q` | 0.10 (when `use_fdr=True`) |
| p_value < threshold | `max_p_value` | 0.05 (when `use_fdr=False`) |

`rejection_reasons` lists all failing criteria (not just the first one).

---

### Step 8 — FDR control (`stats.py: benjamini_hochberg`)

FDR control is applied **across all candidates at once** before compiling contracts.
Algorithm: Benjamini-Hochberg (BH).
Target: q = `fdr_q` (default 0.10) → at most 10% false positives among promoted candidates.

When `use_fdr = True`, BH replaces the `max_p_value` threshold for promotion.
The `fdr_promoted` field on the contract indicates whether the candidate clears BH,
independent of the final promotion outcome (useful for auditing).

---

## 4. Core data structures

### `AlphaContract` (`models.py`)

| Field | Type | Description |
|---|---|---|
| `alpha_id` | `str` | `ALPHA-{asset}-{timeframe}-{stamp}-{idx:03d}` |
| `version` | `str` | `"1.0"` |
| `discovery_date` | `str` | ISO date (today or `AlphaConfig.discovery_date`) |
| `status` | `str` | `"HYPOTHESIS"` or `"REJECTED"` |
| `asset`, `exchange`, `timeframe`, `direction` | `str` | Target metadata |
| `event_candidate_id` | `str` | Link to the source EventCandidate |
| `event_expression` | `str` | Boolean event expression |
| `pattern_family` | `str` | `"mean_reversion"` / `"momentum"` / `"unspecified"` |
| `target_definition` | `TargetDefinition` | Target parameters |
| `base_rate` | `float` | Unconditional win rate |
| `underlying_feature` | `ICResult` | Step 3: IC of the continuous feature |
| `event_stats` | `EventStats` | Step 4: binary predictive metrics |
| `market_structure` | `MarketStructure` | Step 2: Hurst + ACF |
| `regime_analysis` | `RegimeAnalysis` | Step 5: regime sensitivity |
| `alpha_score` | `AlphaScore` | Step 6: composite score and grade |
| `promoted` | `bool` | True if all gates passed |
| `rejection_reasons` | `list[str]` | Failed gates (empty if promoted) |
| `fdr_promoted` | `bool | None` | True if passes BH FDR |
| `handoff_status` | `str` | `"PENDING_RULE_DISCOVERY"` (default) |
| `rule_discovery_response` | `dict | None` | Response from Rule Discovery (not implemented) |

**Serialisation methods:**
- `to_dict()` — flat dict for DataFrame (includes `pattern_family`)
- `to_contract_dict()` — nested dict for YAML/JSON export

### `ICResult` (`models.py`)

| Field | Description |
|---|---|
| `ic: float` | Global Spearman correlation |
| `p_value: float` | IC p-value |
| `n: int` | Valid observations |
| `rolling_ic_stable: bool | None` | True if sign consistent in ≥70% of windows |
| `rolling_ic_mean: float | None` | Mean rolling IC |
| `rolling_sign_consistency: float | None` | Fraction of windows with same sign |
| `admitted: bool` | True if passes the IC gate |

### `EventStats` (`models.py`)

| Field | Description |
|---|---|
| `n_activations: int` | Bars with event active |
| `win_rate: float` | Conditional win rate |
| `base_rate: float` | Unconditional win rate |
| `lift: float` | `win_rate - base_rate` |
| `fwd_return_mean: float` | Mean forward return on active bars |
| `cohens_d: float` | Effect size (active vs inactive) |
| `t_stat: float` | t-statistic |
| `p_value: float` | p-value (one-sided t-test) |

### `RegimeStat` (`models.py`)

| Field | Description |
|---|---|
| `regime: str` | Regime name |
| `n: int` | Observations in regime |
| `ic: float` | Feature IC in regime |
| `p_value: float` | IC p-value in regime |
| `win_rate: float` | Conditional win rate in regime |
| `strength: str` | `"strong"` / `"moderate"` / `"negligible"` / `"insufficient"` |

### `AlphaScore` (`models.py`)

| Field | Description |
|---|---|
| `ic_magnitude: float` | \|IC\| |
| `lift: float` | Lift |
| `cohens_d: float` | Cohen's d |
| `regime_breadth: float` | Fraction of significant regimes |
| `composite_score: float` | Composite score 0–1 |
| `grade: str` | A / B+ / B / C |

---

## 5. Statistical primitives (`stats.py`)

The module implements all statistics in **pure numpy** (no scipy dependency):

| Function | Algorithm |
|---|---|
| `spearmanr(x, y)` | Spearman correlation via numpy rank |
| `cohens_d(group1, group2)` | Mean difference / pooled standard deviation |
| `ttest_ind(x, y, alternative)` | Independent t-test, p-value via incomplete beta function |
| `benjamini_hochberg(p_values, q)` | Benjamini-Hochberg FDR control |
| `betai(a, b, x)` | Regularised incomplete beta function (Lentz continued-fraction) |

Student-t tail probabilities are obtained from the regularised incomplete beta function
implemented with Lentz's continued-fraction algorithm (*Numerical Recipes*).
Precision is ≈1e-6 relative to scipy values.

---

## 6. Configuration

### `TargetDefinition` (`models.py`)

| Parameter | Default | Description |
|---|---|---|
| `holding_period_h` | 24 | Forward horizon in bars |
| `sell_pct` | 0.04 | Return threshold for binary target (e.g. 0.04 = +4%) |
| `direction` | `"long"` | `"long"` or `"short"` |
| `fee_per_side` | 0.002 | Informational only (not netted from target) |
| `asset` | `"ASSET"` | |
| `exchange` | `""` | |
| `timeframe` | `"1H"` | |

### `PromotionThresholds` (`models.py`)

| Parameter | Default | Description |
|---|---|---|
| `ic_min_abs` | 0.02 | Minimum \|IC\| for IC gate |
| `ic_max_p` | 0.05 | Maximum p-value for IC gate |
| `min_lift` | 0.08 | Minimum lift (+8pp) |
| `min_cohens_d` | 0.15 | Minimum Cohen's d |
| `max_p_value` | 0.05 | Maximum p-value (when `use_fdr=False`) |
| `min_activations` | 30 | Minimum activations |
| `use_fdr` | `True` | Use BH instead of `max_p_value` |
| `fdr_q` | 0.10 | Target false-discovery rate for BH |

### `AlphaConfig` (`models.py`)

| Parameter | Default | Description |
|---|---|---|
| `target` | `TargetDefinition()` | Economic target |
| `thresholds` | `PromotionThresholds()` | Admission and promotion gates |
| `close_col` | `"close"` | Close price column |
| `timestamp_col` | `"open_dt"` | Datetime column name |
| `regime_col` | `"regime"` | Regime column (from Market Context) |
| `regime_stable_col` | `"regime_stable"` | Regime stable column |
| `use_stable_regime_only` | `False` | Use only stable bars for regime analysis |
| `min_regime_obs` | 10 | Minimum observations to evaluate a regime |
| `rolling_ic_window` | `None` | Rolling IC window (None → 60 days in bars) |
| `bars_per_day` | `None` | Bars per day (None → inferred) |
| `score_weights` | `(0.25, 0.30, 0.25, 0.20)` | Weights: (IC, lift, cohens_d, breadth) |
| `discovery_date` | `None` | ISO date for contracts (None → today) |

---

## 7. Alignment with the functional analysis

### ✅ Aligned

- 8-step pipeline with correct sequence
- `PromotionThresholds` with all documented values
- `AlphaContract` with all primary fields
- `ICResult`, `EventStats`, `RegimeStat`, `RegimeAnalysis`, `AlphaScore`
- `MarketStructure` (Hurst + ACF)
- Benjamini-Hochberg FDR with `fdr_q = 0.10`
- Regime strength classification (strong/moderate/negligible/insufficient)
- Dependency type classification (agnostic/conditional/specific/broken)
- Grade thresholds (A≥0.75, B+≥0.60, B≥0.45, C<0.45)
- Score weights: (0.25, 0.30, 0.25, 0.20)
- `use_stable_regime_only` + `min_regime_obs`
- `handoff_status = "PENDING_RULE_DISCOVERY"` and `rule_discovery_response`

### ➕ Added in code (not in the functional analysis)

- **Score normalisation caps:**
  - |IC| normalised as `min(|IC|/0.10, 1.0)` — saturates at IC=10%
  - Lift normalised as `min(lift/0.30, 1.0)` — saturates at lift=30%
  - Cohen's d normalised as `min(d/0.80, 1.0)` — saturates at d=0.80
- **Weight renormalisation** when regime is unavailable (not zero-substitution)
- **`rolling_sign_consistency`** on `ICResult` — fraction of windows with same IC sign
- **`rolling_ic_mean`** on `ICResult` — mean rolling IC
- **`pattern_family`** on `AlphaContract` — derived from `MarketStructure.expected_family`
- **`to_contract_dict()`** — full nested YAML/JSON contract serialisation
- **`bars_per_day` inference** from median DatetimeIndex spacing
- **Rolling IC** computed on ≈20 evenly-spaced windows (stride = max(1, (n-w)//20)),
  not on every bar — flat cost independent of dataset length
- **`fdr_promoted`** — flag separate from `promoted` for FDR auditing
- **Pure numpy statistics** without scipy (Lentz continued-fraction for p-values)

### ⚠️ Divergences from the functional analysis

- **IC gate:** The functional analysis documents the IC and p-value gates as independent.
  The code uses `not (weak_ic AND weak_p)`: a candidate is **admitted** when IC is strong
  **OR** the p-value is significant. Rejection at the IC gate only happens when both are weak.

### ❌ Missing in code (documented in the functional analysis)

- **`rule_discovery_hints`** in the contract (Section 7 of AlphaDiscovery.md):
  fields such as `entry_mode`, `buy_drop_pct_range`, `sell_pct_range`, `min_pf_target`,
  `exclusion_conditions`. The code has only `rule_discovery_response: dict | None`
  (generic, unstructured), which remains `null` because Rule Discovery is not yet implemented.
