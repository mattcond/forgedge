# Module 2 — Alpha Discovery

Alpha Discovery is the third module in the FORGE pipeline and the **first that
sees the forward return**. It receives the `EventCandidate` list produced by
Event Discovery and measures the predictive power of each event against a
user-defined economic target. The output is a list of `AlphaContract` — one per
candidate — that records all statistical measurements and determines whether the
candidate has been promoted as an actionable hypothesis.

---

## Basic usage

```python
from forgedge import (
    MarketContext,
    EventDiscovery,
    AlphaDiscovery, AlphaConfig, TargetDefinition,
)
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")

# Modules 0 and 1: regime + event discovery
enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)
candidates = ed.run()

# Module 2: predictive measurement
config = AlphaConfig(
    target=TargetDefinition(
        holding_period_h=24,
        sell_pct=0.04,
        direction="long",
        asset="BTC",
        timeframe="1H",
    )
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

promoted = ad.promoted_contracts()
print(f"{len(promoted)} promoted out of {len(contracts)} evaluated")
print(ad.summary().head())
```

`run()` returns the **full** list (promoted and rejected), so callers can audit
why a candidate did not make it. `promoted_contracts()` filters to only the
contracts with `status == "HYPOTHESIS"`.

---

## Position in the pipeline

```
KPI Table + regime  (Module 0)
list[EventCandidate] (Module 1)
        │
        ▼
  AlphaDiscovery.run()
        │
        ▼
  list[AlphaContract]   (all: HYPOTHESIS + REJECTED)
  promoted_contracts()  (HYPOTHESIS only) ──► Rule Discovery (not implemented)
```

Alpha Discovery **does not recompute** event thresholds, windows, or derived
features. It reads the `regime` column from Event Discovery's post-pipeline
DataFrame (`ed.df`) and uses the `event_series` stored in each candidate
without re-deriving it.

---

## 8-step pipeline

### Step 1 — Target definition

The economic target is built from the `close` column via `build_target()`:

- **`fwd_return`**: maximum forward return over `holding_period_h` bars.
  For `direction="long"`: `max(close[t+1..t+h]) / close[t] - 1`.
  For `direction="short"`: `1 - min(close[t+1..t+h]) / close[t]`.
- **`target_binary`**: 1 if `fwd_return >= sell_pct`, 0 otherwise.
- **`base_rate`**: frequency of `target_binary == 1` over the entire valid
  sample (unconditional win rate). Exposed as `ad.base_rate` after `run()`.

`fee_per_side` is recorded in the contract for informational purposes — Alpha
Discovery does not net it out. That is Rule Discovery's responsibility.

---

### Step 2 — Market structure analysis

Computed **once per session** (not per candidate) on `close` and `fwd_return`:

```python
ad.market_structure.hurst                 # float: Hurst exponent (DFA)
ad.market_structure.hurst_interpretation  # "mean_reverting" | "random_walk" | "trending"
ad.market_structure.expected_family       # "mean_reversion" | "momentum" | "none"
ad.market_structure.autocorr              # dict[int, float]: ACF at selected lags
```

`expected_family` is copied into each contract as `pattern_family`
(`"unspecified"` when `"none"`), providing interpretive context about the
category of alpha the event is expected to belong to.

The Hurst threshold is `0.5`: below → mean-reverting, above → trending.

---

### Step 3 — IC (Information Coefficient) measurement

For each candidate, the continuous feature underlying its first component
(e.g. the raw RSI value before thresholding) is correlated with `fwd_return`
using Spearman correlation:

```
IC = ρ(feature, fwd_return)   [Spearman]
```

All statistical computations are implemented in **pure numpy** without any
scipy dependency.

#### IC admission gate

A candidate passes the IC gate if **not** both of the following are true:

- `|IC| < ic_min_abs` (default: 0.02) — IC is weak
- `p_value > ic_max_p` (default: 0.05) — not statistically significant

In other words: a candidate is **rejected at the IC gate only when the IC is
weak AND the p-value is not significant**. If the IC is small but statistically
significant (low p-value), the candidate still passes the gate.

#### Rolling IC stability

Temporal stability of the IC is assessed over ≈20 evenly-spaced windows of
width `rolling_ic_window` (default: 60 days expressed in bars, inferred from
the median DatetimeIndex spacing):

```
stride = max(1, (n - window) // 20)
```

A candidate is stable (`rolling_ic_stable = True`) if the rolling IC sign
matches the overall IC sign in at least 70% of windows
(`rolling_sign_consistency >= 0.70`). The fixed ≈20-window design keeps the
computational cost flat regardless of dataset length.

Output — `ICResult`:
```python
ic.ic                       # float: global Spearman IC
ic.p_value                  # float: IC p-value
ic.n                        # int: valid observations
ic.admitted                 # bool: passes the IC gate
ic.rolling_ic_stable        # bool | None: sign consistent in ≥70% of windows
ic.rolling_ic_mean          # float | None: mean rolling IC
ic.rolling_sign_consistency # float | None: fraction of windows with matching sign
```

---

### Step 4 — Win rate analysis

Measures the **binary event's** predictive power against the target:

| Metric | Formula |
|---|---|
| `n_activations` | Bars with event active and valid target |
| `win_rate` | `mean(target_binary)` on active bars |
| `lift` | `win_rate - base_rate` |
| `fwd_return_mean` | Mean forward return on active bars |
| `cohens_d` | `(mean_active - mean_inactive) / std_pooled` |
| `t_stat`, `p_value` | One-sided independent t-test (`alternative="greater"`) |

The p-value uses a two-sample t-test with `alternative="greater"`: "the mean
return on event-active bars is greater than on non-active bars". The incomplete
beta function (Lentz continued-fraction algorithm, *Numerical Recipes*) produces
p-values with ≈1e-6 precision vs scipy.

Output — `EventStats`:
```python
ev.n_activations   # int
ev.win_rate        # float: e.g. 0.38 → 38% of activations hit the target
ev.base_rate       # float: unconditional win rate (copy of ad.base_rate)
ev.lift            # float: e.g. 0.06 → +6pp over base rate
ev.fwd_return_mean # float: mean return on active bars
ev.cohens_d        # float: effect size
ev.t_stat          # float
ev.p_value         # float
```

---

### Step 5 — Regime sensitivity

For each regime defined in the `regime` column (from Module 0), with at least
`min_regime_obs` observations (default: 10):

1. Computes the Spearman IC of the continuous feature vs `fwd_return` in that regime
2. Computes the conditional win rate of the event in that regime

If `use_stable_regime_only = True` and `regime_stable` is present in the
DataFrame, only bars with `regime_stable = True` are used for per-regime IC
(excludes transition bars from the computation).

#### Regime strength classification

| Strength | Condition |
|---|---|
| `"strong"` | `p < 0.05` and `\|IC\| ≥ 0.05` |
| `"moderate"` | `p < 0.05` and `\|IC\| < 0.05` |
| `"negligible"` | `p ≥ 0.05` (not significant) |
| `"insufficient"` | Fewer than `min_regime_obs` observations |

#### Regime dependency classification

| `dependency_type` | Condition |
|---|---|
| `"agnostic"` | All evaluated regimes are significant (strong or moderate) and count ≥ 2 |
| `"conditional"` | More than 1 significant regime, but not all |
| `"specific"` | Exactly 1 significant regime |
| `"broken"` | 0 significant regimes |
| `"unknown"` | No regime column available |

Output — `RegimeAnalysis`:
```python
ra.per_regime       # list[RegimeStat]: measurements per regime
ra.dependency_type  # str: dependency classification
ra.active_regimes   # list[str]: significant regimes (strong or moderate)
ra.weak_regimes     # list[str]: negligible regimes
ra.regime_breadth   # float: len(active) / len(evaluated)
```

Each `RegimeStat`:
```python
rs.regime    # str: regime name
rs.n         # int: observations in regime
rs.ic        # float: IC in regime
rs.p_value   # float: IC p-value in regime
rs.win_rate  # float: conditional win rate in regime
rs.strength  # str: "strong" | "moderate" | "negligible" | "insufficient"
```

---

### Step 6 — Alpha scoring

The **composite score** (0–1) is a weighted average of four normalised
components:

| Component | Default weight | Normalisation |
|---|---|---|
| IC magnitude | 0.25 | `min(\|IC\| / 0.10, 1.0)` — saturates at IC=10% |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` — saturates at lift=30% |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` — saturates at d=0.80 |
| Regime breadth | 0.20 | `regime_breadth` (already 0–1) |

When the regime column is not available, the `regime_breadth` term is removed
and the remaining weights are **renormalised** (not replaced with 0):

```
composite = Σ(w_i * norm_i) / Σ(w_i)   [over available terms only]
```

#### Grade from composite score

| Grade | Score |
|---|---|
| `A` | ≥ 0.75 |
| `B+` | ≥ 0.60 |
| `B` | ≥ 0.45 |
| `C` | < 0.45 |

Output — `AlphaScore`:
```python
sc.ic_magnitude    # float: raw |IC|
sc.lift            # float: raw lift
sc.cohens_d        # float: raw Cohen's d
sc.regime_breadth  # float: fraction of significant regimes
sc.composite_score # float: 0–1, rounded to 4 decimal places
sc.grade           # str: "A" | "B+" | "B" | "C"
```

---

### Step 7 — Contract compilation

A candidate is promoted (`status = "HYPOTHESIS"`) only when it passes **all**
of the following gates:

| Gate | Parameter | Default | Note |
|---|---|---|---|
| IC admitted | `ic_min_abs`, `ic_max_p` | 0.02, 0.05 | Logic: `not (weak_ic AND weak_p)` |
| Lift ≥ threshold | `min_lift` | 0.08 | +8 percentage points above base rate |
| Cohen's d ≥ threshold | `min_cohens_d` | 0.15 | Minimum effect size |
| Activations ≥ threshold | `min_activations` | 30 | Stable win rate estimate |
| Statistical significance | `use_fdr`/`fdr_q` or `max_p_value` | BH q=0.10 | See Step 8 |

`rejection_reasons` lists **all** failed gates (not just the first), providing
a complete picture of why a candidate was rejected:

```python
for c in contracts:
    if not c.promoted:
        print(c.event_candidate_id, c.rejection_reasons)
```

---

### Step 8 — FDR control (Benjamini-Hochberg)

When `use_fdr = True` (default), Benjamini-Hochberg (BH) multiple-testing
correction is applied across **all candidates simultaneously** before compiling
contracts, using the t-test p-values:

```
BH at q = fdr_q (default 0.10) → at most 10% false positives among promoted candidates
```

BH replaces the `max_p_value` threshold as the significance criterion.

The `fdr_promoted` field records whether a candidate passes BH independently of
the final promotion outcome — useful for auditing:

```python
# Candidates that passed BH but were rejected for other gates
bh_ok_but_rejected = [c for c in contracts if c.fdr_promoted and not c.promoted]
```

When `use_fdr = False`, the raw `max_p_value` threshold is applied directly.

---

## Data structure: `AlphaContract`

```python
c = promoted[0]

# Identifiers
c.alpha_id            # str: "ALPHA-BTC-1H-260610-000"
c.version             # str: "1.0"
c.discovery_date      # str: ISO date (today or AlphaConfig.discovery_date)
c.status              # str: "HYPOTHESIS" | "REJECTED"
c.pattern_family      # str: "mean_reversion" | "momentum" | "unspecified"

# Origin
c.asset, c.exchange, c.timeframe, c.direction  # str
c.event_candidate_id  # str: link to the source EventCandidate
c.event_expression    # str: e.g. "rsi_14 < 30.5 AND spread_ema_9_25 < -0.012"

# Target
c.target_definition   # TargetDefinition
c.base_rate           # float: unconditional win rate

# Statistical measurements
c.market_structure    # MarketStructure: Hurst + ACF (Step 2)
c.underlying_feature  # ICResult: IC of the continuous feature (Step 3)
c.event_stats         # EventStats: binary event metrics (Step 4)
c.regime_analysis     # RegimeAnalysis: regime sensitivity (Step 5)
c.alpha_score         # AlphaScore: composite score and grade (Step 6)

# Promotion outcome
c.promoted            # bool
c.rejection_reasons   # list[str]: failed gates (empty if promoted)
c.fdr_promoted        # bool | None

# Handoff to Rule Discovery
c.handoff_status      # str: "PENDING_RULE_DISCOVERY"
c.rule_discovery_response  # dict | None (Rule Discovery not yet implemented)
```

#### `alpha_id` format

```
ALPHA-{asset}-{timeframe}-{stamp}-{idx:03d}
```

Where `stamp` is the discovery date as `YYMMDD` (e.g. `260610` for 2026-06-10)
and `idx` is the candidate's sequential index (000, 001, ...).

---

## Output methods

### `ad.run() → list[AlphaContract]`

Evaluates all candidates and returns the complete list (promoted + rejected).
Must be called before any other method.

### `ad.promoted_contracts() → list[AlphaContract]`

Returns only contracts with `status == "HYPOTHESIS"`.
Requires `run()` to have been called.

### `ad.summary() → pd.DataFrame`

Flat, sortable summary of every evaluated candidate, sorted by `composite_score`
descending. Each row is a contract with all key metrics as columns:

```python
df = ad.summary()
df.columns
# alpha_id, status, promoted, event_candidate_id, expression, pattern_family,
# feature, ic, ic_p_value, ic_admitted, rolling_ic_stable, n_activations,
# win_rate, base_rate, lift, fwd_return_mean, cohens_d, t_stat, p_value,
# fdr_promoted, regime_dependency, regime_breadth, composite_score, grade,
# rejection_reasons
```

### `c.to_dict() → dict`

Flat dictionary equivalent to one row in `summary()`. Useful for building
custom DataFrames from a subset of contracts.

### `c.to_contract_dict() → dict`

Full nested contract as a dictionary, ready for YAML/JSON serialisation:

```python
import json
for c in promoted:
    print(json.dumps(c.to_contract_dict(), indent=2))
```

---

## Full configuration reference

### `TargetDefinition`

| Parameter | Default | Description |
|---|---|---|
| `holding_period_h` | `24` | Forward horizon in bars |
| `sell_pct` | `0.04` | Return threshold for binary target (e.g. 0.04 = +4%) |
| `direction` | `"long"` | `"long"` or `"short"` |
| `fee_per_side` | `0.002` | Informational only; not deducted from target |
| `asset` | `"ASSET"` | Asset name |
| `exchange` | `""` | Exchange name |
| `timeframe` | `"1H"` | Bar timeframe |

### `PromotionThresholds`

| Parameter | Default | Description |
|---|---|---|
| `ic_min_abs` | `0.02` | Minimum \|IC\| for IC gate |
| `ic_max_p` | `0.05` | Maximum p-value for IC gate |
| `min_lift` | `0.08` | Minimum lift (+8pp) |
| `min_cohens_d` | `0.15` | Minimum Cohen's d |
| `max_p_value` | `0.05` | Maximum p-value (when `use_fdr=False`) |
| `min_activations` | `30` | Minimum activations for a stable estimate |
| `use_fdr` | `True` | Use BH instead of `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate for BH |

### `AlphaConfig`

| Parameter | Default | Description |
|---|---|---|
| `target` | `TargetDefinition()` | Economic target |
| `thresholds` | `PromotionThresholds()` | Admission and promotion gates |
| `close_col` | `"close"` | Close price column |
| `timestamp_col` | `"open_dt"` | Datetime column name (or DatetimeIndex name) |
| `regime_col` | `"regime"` | Regime column (from Market Context) |
| `regime_stable_col` | `"regime_stable"` | Regime stability column |
| `use_stable_regime_only` | `False` | Restrict regime analysis to stable bars |
| `min_regime_obs` | `10` | Minimum observations to evaluate a regime |
| `rolling_ic_window` | `None` | Rolling IC window (None → 60 days in bars) |
| `bars_per_day` | `None` | Bars per day (None → inferred from spacing) |
| `score_weights` | `(0.25, 0.30, 0.25, 0.20)` | Weights: (IC, lift, cohens_d, breadth) |
| `discovery_date` | `None` | ISO date for contracts (None → today) |

---

## Advanced usage patterns

### Inspecting rejected candidates

```python
contracts = ad.run()
for c in contracts:
    if not c.promoted:
        print(f"{c.event_candidate_id}: {c.rejection_reasons}")
```

Example output:
```
EV-BTC-1H-001: ['lift 0.0312 < 0.08']
EV-BTC-1H-004: ['IC below admission (|IC|<0.02 and p>0.05)', 'n_activations 18 < 30']
```

### Filtering by grade

```python
df = ad.summary()
grade_a = df[df["grade"] == "A"]
grade_b_plus = df[df["grade"].isin(["A", "B+"])]
```

### Inspecting regime sensitivity

```python
for c in promoted:
    print(f"Event: {c.event_candidate_id}")
    print(f"  Regime dependency: {c.regime_analysis.dependency_type}")
    for rs in c.regime_analysis.per_regime:
        print(f"  {rs.regime}: IC={rs.ic:.3f}, win_rate={rs.win_rate:.3f}, strength={rs.strength}")
```

### YAML export of promoted contracts

```python
import yaml
for c in promoted:
    with open(f"{c.alpha_id}.yaml", "w") as f:
        yaml.dump(c.to_contract_dict(), f)
```

### Running without Market Context

If the DataFrame does not contain a `regime` column, regime sensitivity is
skipped and the `regime_breadth` term is dropped from the score (weights
renormalised):

```python
# Works without regime — regime_analysis.dependency_type will be "unknown"
ad = AlphaDiscovery(kpi_without_regime, candidates, config)
contracts = ad.run()
# alpha_score.regime_breadth = NaN, composite_score still calculated
# on (IC, lift, cohens_d) with weights renormalised to sum to 1.0
```

### Tightening promotion gates

```python
from forgedge.alpha_discovery.models import PromotionThresholds

strict = AlphaConfig(
    target=TargetDefinition(holding_period_h=48, sell_pct=0.06),
    thresholds=PromotionThresholds(
        min_lift=0.12,
        min_cohens_d=0.20,
        min_activations=50,
        fdr_q=0.05,
    )
)
ad = AlphaDiscovery(ed.df, candidates, strict)
```

---

## Statistical primitives (`stats.py`)

All statistical primitives are implemented in **pure numpy** with no scipy or
statsmodels dependency:

| Function | Algorithm |
|---|---|
| `spearmanr(x, y)` | Spearman correlation via numpy ranking |
| `cohens_d(group1, group2)` | `(mean1 - mean2) / std_pooled` |
| `ttest_ind(x, y, alternative)` | Independent t-test, p-value via incomplete beta |
| `benjamini_hochberg(p_values, q)` | Benjamini-Hochberg FDR control |
| `betai(a, b, x)` | Regularised incomplete beta (Lentz continued-fraction) |

Student-t probabilities are computed via the regularised incomplete beta
function implemented with the Lentz continued-fraction algorithm (*Numerical
Recipes*). Precision is ≈1e-6 relative to scipy values.

---

## Downstream usage

Alpha Discovery produces contracts with `handoff_status = "PENDING_RULE_DISCOVERY"`.
Rule Discovery (Module 3, not yet implemented) will consume the promoted
contracts to run a realistic backtest with order mechanics, fees, and
operational edge validation.

The `rule_discovery_response` field is reserved for Rule Discovery's response
and remains `None` until the module is implemented.
