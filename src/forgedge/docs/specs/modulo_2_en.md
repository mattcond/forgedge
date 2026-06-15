# Module 2 — Alpha Discovery

Alpha Discovery is the third module in the FORGE pipeline and the **first that
sees the forward return**. It receives the `EventCandidate` list produced by
Event Discovery and, for each candidate, **derives the economic target directly
from the data** — selecting the horizon with the best signal-to-horizon ratio,
deriving the take-profit from the distribution of favourable excursions, and
measuring predictive power against the derived target. The output is a list of
`AlphaContract` — one per candidate — recording the derived target, all
in-sample statistical measurements, the out-of-sample confirmation, and the
A–D grade.

**Promotion principle:** all contracts with a determined direction (`"long"` or
`"short"`) are promoted to `HYPOTHESIS` and forwarded to Rule Discovery.
Statistical measurements (IC, lift, Cohen's d, FDR, OOS) produce non-blocking
diagnostics that inform the grade but do not prevent promotion. Rule Discovery
is the final economic judge.

---

## Basic usage

```python
from forgedge import (
    MarketContext, EventDiscovery,
    AlphaDiscovery, AlphaConfig,
)
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")

enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)
candidates = ed.run()

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    # horizon, sell_pct and direction are derived automatically
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

# All contracts with a determined direction are promoted
promoted = ad.promoted_contracts()
print(f"{len(promoted)} promoted (HYPOTHESIS) out of {len(contracts)} evaluated")
print(ad.summary()[["expression", "direction", "holding_period_h",
                    "sell_pct", "grade", "oos_passed"]].head())
```

`AlphaConfig` **does not require a `TargetDefinition`**: horizon, sell_pct, and
direction are derived per event from the data itself. `asset` and `timeframe`
are traceability metadata and have no effect on any measurement.

---

## Position in the pipeline

```
KPI Table + regime  (Module 0)
list[EventCandidate] (Module 1)
        │
        ▼
  AlphaDiscovery.run()
  ├─ Step 1: per-event target derivation (IS)
  ├─ Step 2: market structure (IS)
  ├─ Step 3: IC and rolling stability (IS)
  ├─ Step 4: win rate and Cohen's d (IS)
  ├─ Step 5: regime sensitivity (IS)
  ├─ OOS: confirmation of derived target (OOS tail)
  ├─ Step 6: alpha scoring (A–D)
  └─ Step 7: contract compilation
        │
        ▼
  list[AlphaContract]   (all: HYPOTHESIS + REJECTED)
  promoted_contracts()  (HYPOTHESIS: determined direction) ──► Rule Discovery
```

The DataFrame is split chronologically into **in-sample (IS)** and
**out-of-sample (OOS)** via `train_ratio` (default: 70/30). All statistical
measurements (IC, win rate, regime) are computed on the IS window; the derived
target is then confirmed on the OOS tail.

---

## 8-step pipeline

### Step 1 — Per-event target derivation

Alpha Discovery **takes no economic parameters as input**. For each candidate,
it scans a horizon grid (`horizon_grid`, default: `(1, 2, 3, 4, 6, 8, 12, 16,
24, 36, 48)` bars) on the IS window and selects:

**Horizon selection `h*`:**
```python
score[h] = |mean_advantage[h]| / sqrt(h)
h*       = argmax_h score[h]
```

The `|mean_advantage| / √h` criterion is a Sharpe-like deflation that balances
the size of the advantage against the growing variance of longer horizons. It
avoids the systematic bias of max-t-stat toward short horizons, where the
t-test denominator is structurally small.

**`sell_pct` derivation:**
```python
MFE_i    = max favourable excursion in the h* bars following active bar i
sell_pct = max(quantile(MFE, mfe_quantile), mfe_floor)
```

`sell_pct` is the `mfe_quantile` quantile (default: 0.5 = median) of the
Maximum Favorable Excursion distribution of IS active bars at `h*`. This
anchors the baseline take-profit to the actual excursion distribution rather
than a mean that mixes winning and losing bars. The floor `mfe_floor` (default:
50 bp) ensures an operationally meaningful take-profit.

**Output — `DerivedTarget`:**
```python
dt.holding_period_h  # int: selected horizon
dt.sell_pct          # float: MFE quantile at h*
dt.direction         # str: "long" | "short" | "undetermined"
dt.mean_advantage    # float: mean oriented return at h*
dt.advantage_by_h    # dict[int, float]: mean active return per horizon
dt.t_stat_by_h       # dict[int, float]: t-stat per horizon
dt.score_by_h        # dict[int, float]: |mean_advantage| / sqrt(h) per horizon
```

If no horizon produces a finite advantage, `direction = "undetermined"` and the
contract is rejected. All subsequent measurements (IC, win rate, regime) are
computed **at the derived target** (`h*`, `sell_pct*`, `direction*`).

---

### Step 2 — Market structure analysis

Computed **once per session** on the IS window, at the median horizon of the
grid:

```python
ad.market_structure.hurst                 # float: Hurst exponent (DFA)
ad.market_structure.hurst_interpretation  # "mean_reverting" | "random_walk" | "trending"
ad.market_structure.expected_family       # "mean_reversion" | "momentum" | "none"
ad.market_structure.autocorr              # dict[int, float]: ACF at selected lags
```

`expected_family` is copied into each contract as `pattern_family`
(`"unspecified"` when `"none"`). Hurst threshold: below 0.5 → mean-reverting,
above → trending.

---

### Step 3 — IC (Information Coefficient) measurement

**On the IS window**, the continuous feature underlying the candidate's first
component is correlated with the forward return at the derived `h*` using
Spearman:

```
IC = ρ(feature, fwd_return_h*)   [Spearman, IS only]
```

The result is **cached per `(feature, horizon)`**: candidates sharing the same
feature and `h*` reuse the cached `ICResult` without recomputation.

#### IC admission gate (non-blocking)

The IC is classified as weak if **both** conditions hold:
- `|IC| < ic_min_abs` (default: 0.02)
- `p_value > ic_max_p` (default: 0.05)

If the IC is small but statistically significant, the candidate is still
admitted (`ic.admitted = True`). A weak IC produces a
`"[diagnostic] IC weak …"` entry in `rejection_reasons` but does not block
promotion.

#### Rolling IC stability (IS)

Stability assessed over ≈20 evenly-spaced windows of width `rolling_ic_window`
(default: 60 days in bars):

```
stride = max(1, (n_is - window) // 20)
```

`rolling_ic_stable = True` when the rolling IC sign matches the overall sign in
at least 70% of windows.

---

### Step 4 — Win rate analysis (IS, at the derived target)

Forward returns are **oriented** before measurement: for `direction="short"`,
returns are multiplied by -1 so "favourable to the trade" is always positive.

| Metric | Description |
|---|---|
| `n_activations` | IS bars with active event and valid target |
| `win_rate` | `mean(target_binary_is)` on active bars |
| `lift` | `win_rate - base_rate_is` |
| `fwd_return_mean` | Mean oriented return on active bars (IS) |
| `cohens_d` | `(mean_active - mean_inactive) / std_pooled` |
| `t_stat`, `p_value` | One-sided t-test (`alternative="greater"`) |

`base_rate_is` is computed **on the IS** at the derived target — each candidate
has its own base rate. Lift, cohens_d, and activations produce non-blocking
diagnostics if below threshold.

---

### Step 5 — Regime sensitivity (IS)

For each regime, with at least `min_regime_obs` IS observations:

1. Spearman IC of feature vs `fwd_return_h*` in the regime (IS)
2. Conditional win rate in the regime (IS)

Results per regime are **cached per `(feature, horizon, regime)`**, analogously
to the global IC. If `use_stable_regime_only = True`, only IS bars with
`regime_stable = True` are used.

Strength classification: `strong` / `moderate` / `negligible` / `insufficient`.
Dependency type: `agnostic` / `conditional` / `specific` / `broken` / `unknown`.

---

### OOS validation (non-blocking)

After all IS measurements, the derived target `(h*, sell_pct*, direction*)` is
**replayed on the OOS tail** (the last `1 - train_ratio` of the dataset):

- OOS forward returns are oriented to the derived direction
- Win rate, lift, mean_advantage, and the t-test are measured on the OOS window
- `oos.passed = True` when all three criteria hold:
  1. `n_oos_activations >= min_oos_activations` (default: 10)
  2. `mean_advantage > 0` (oriented advantage stays positive on OOS)
  3. `p_value < oos_max_p` (default: 0.10)

**Failing OOS confirmation produces a non-blocking diagnostic** —
`"[diagnostic] OOS weak …"` in `rejection_reasons` — but does not prevent
promotion. The OOS statistical signal contributes to the grade.

Output — `OOSValidation`:
```python
oos.n_bars          # int: bars in the OOS window
oos.n_activations   # int: activations with complete forward horizon in OOS
oos.mean_advantage  # float: mean oriented return on OOS (> 0 = confirmed)
oos.t_stat          # float
oos.p_value         # float: one-sided t-test
oos.win_rate        # float: OOS win rate at the derived target
oos.base_rate       # float: OOS base rate at the derived target
oos.lift            # float: OOS lift
oos.passed          # bool
```

When `train_ratio = 1.0`, IS covers the full dataset and `oos_validation` is
`None` for every contract (OOS split disabled — not recommended in production).

---

### Step 6 — Alpha scoring

Composite score (0–1):

| Component | Default weight | Normalisation |
|---|---|---|
| IC magnitude | 0.25 | `min(|IC| / 0.10, 1.0)` |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` |
| Regime breadth | 0.20 | `regime_breadth` (0–1) |

When regime is unavailable, the breadth term is removed and remaining weights
are renormalised.

**Grade:** A ≥ 0.75 | B ≥ 0.50 | C ≥ 0.25 | D < 0.25

All four grades are forwarded to Rule Discovery — the grade signals the
strength of statistical evidence, not exclusion.

---

### Step 7 — Contract compilation

The **only rejection gate** is the absence of a determined direction:

| Gate | Condition |
|---|---|
| **Hard (blocks)** | `direction == "undetermined"` — no finite advantage across the entire grid |
| **Diagnostics (non-blocking)** | Weak IC, lift below threshold, cohens_d below threshold, insufficient activations, non-significant FDR/p-value, weak OOS |

Diagnostics are annotated with the `[diagnostic]` prefix in
`rejection_reasons`. A promoted contract may have a non-empty `rejection_reasons`
list — it documents the statistical weaknesses detected.

```python
for c in contracts:
    print(f"{c.event_candidate_id}: promoted={c.promoted}, grade={c.alpha_score.grade}")
    for r in c.rejection_reasons:
        print(f"  {r}")
# Example:
# EVT-rsi_25-PR-0042: promoted=True, grade=C
#   [diagnostic] lift 0.0520 < 0.08
#   [diagnostic] OOS weak (p=0.143 vs 0.10, mean_adv=0.00210, n_act=7)
```

BH FDR is applied to IS p-values of all candidates simultaneously;
`fdr_promoted` records the outcome but does not block promotion.

---

## Data structure: `AlphaContract`

```python
c = promoted[0]

# Identifiers
c.alpha_id            # str: "ALPHA-BTC-1H-260610-000"
c.version             # str: "1.0"
c.discovery_date      # str: ISO date
c.status              # str: "HYPOTHESIS" | "REJECTED"
c.pattern_family      # str: "mean_reversion" | "momentum" | "unspecified"

# Scope metadata (traceability only)
c.asset               # str
c.exchange            # str
c.timeframe           # str
c.fee_per_side        # float: informational, not deducted

# Origin and derived target
c.event_candidate_id  # str: link to the source EventCandidate
c.event_expression    # str: boolean event expression
c.direction           # str: "long" | "short" | "undetermined"
c.derived_target      # DerivedTarget: target derived from data (Step 1)
c.base_rate           # float: IS base rate at the derived target

# IS statistical measurements
c.market_structure    # MarketStructure: Hurst + ACF (Step 2)
c.underlying_feature  # ICResult: IS IC (Step 3)
c.event_stats         # EventStats: IS win rate (Step 4)
c.regime_analysis     # RegimeAnalysis: IS regime sensitivity (Step 5)

# OOS confirmation
c.oos_validation      # OOSValidation | None

# Score and promotion
c.alpha_score         # AlphaScore: composite score and grade (Step 6)
c.promoted            # bool: True if direction is "long" or "short"
c.rejection_reasons   # list[str]: diagnostics ([diagnostic] prefix = non-blocking)
c.fdr_promoted        # bool | None: BH FDR outcome (non-blocking)

# Handoff to Rule Discovery
c.handoff_status      # str: "PENDING_RULE_DISCOVERY"
c.rule_discovery_response  # dict | None
```

---

## Output methods

### `ad.run() → list[AlphaContract]`

Derives the target per event, runs all IS + OOS measurements, and returns the
complete list. Must be called first.

Properties populated by `run()`:
- `ad.market_structure` — IS market structure
- `ad.split_idx` — row index of the IS/OOS boundary

### `ad.promoted_contracts() → list[AlphaContract]`

Only contracts with `status == "HYPOTHESIS"` (determined direction).

### `contract.persist(path)`

Saves the `AlphaContract` to disk as a pickle file. The contract can be reloaded
in a later session and passed directly to `RuleDiscovery` without re-running
Alpha Discovery.

```python
import pickle, pathlib

pathlib.Path("contracts").mkdir(exist_ok=True)
for c in promoted:
    c.persist(f"contracts/{c.alpha_id}.pkl")

# Reload in a later session
contract = pickle.load(open("contracts/ALPHA-BTC-1H-000.pkl", "rb"))
```

### `ad.summary() → pd.DataFrame`

Flat summary sorted by `composite_score` descending:

```
alpha_id, status, promoted, event_candidate_id, expression, pattern_family,
holding_period_h, sell_pct, direction, mean_advantage,
feature, ic, ic_p_value, ic_admitted, rolling_ic_stable,
n_activations, win_rate, base_rate, lift, fwd_return_mean, cohens_d, t_stat, p_value,
fdr_promoted, oos_passed, oos_p_value, oos_lift,
regime_dependency, regime_breadth, composite_score, grade, rejection_reasons
```

---

## Full configuration reference

### `AlphaConfig`

| Parameter | Default | Description |
|---|---|---|
| `horizon_grid` | `(1,2,3,4,6,8,12,16,24,36,48)` | Candidate holding horizons in bars |
| `mfe_quantile` | `0.5` | MFE quantile for deriving sell_pct (0.5 = median) |
| `mfe_floor` | `0.005` | sell_pct floor (50 bp) after the quantile |
| `train_ratio` | `0.7` | IS fraction of the dataset (0 < x ≤ 1.0) |
| `thresholds` | `PromotionThresholds()` | Diagnostic thresholds (not promotion gates) |
| `asset` | `"ASSET"` | Traceability metadata (copied to contract and alpha_id) |
| `exchange` | `""` | Traceability metadata |
| `timeframe` | `"1H"` | Traceability metadata |
| `fee_per_side` | `0.002` | Informational only; not deducted from target |
| `close_col` | `"close"` | Close price column |
| `timestamp_col` | `"open_dt"` | Datetime column name (or DatetimeIndex name) |
| `regime_col` | `"regime"` | Regime column (from Market Context) |
| `regime_stable_col` | `"regime_stable"` | Regime stability column |
| `use_stable_regime_only` | `False` | Restrict regime analysis to stable bars |
| `min_regime_obs` | `10` | Minimum IS observations to evaluate a regime |
| `rolling_ic_window` | `None` | Rolling IC window (None → 60 days in bars) |
| `bars_per_day` | `None` | Bars per day (None → inferred from spacing) |
| `score_weights` | `(0.25, 0.30, 0.25, 0.20)` | Weights: (IC, lift, cohens_d, breadth) |
| `discovery_date` | `None` | ISO date for contracts (None → today) |

### `PromotionThresholds`

These parameters control **diagnostics** — not promotion gates.

| Parameter | Default | Description |
|---|---|---|
| `ic_min_abs` | `0.02` | \|IC\| threshold for classifying IC as weak |
| `ic_max_p` | `0.05` | Maximum p-value for classifying IC as weak |
| `min_lift` | `0.08` | Minimum lift (diagnostic) |
| `min_cohens_d` | `0.15` | Minimum Cohen's d (diagnostic) |
| `max_p_value` | `0.05` | Maximum p-value (when `use_fdr=False`) (diagnostic) |
| `min_activations` | `30` | Minimum IS activations (diagnostic) |
| `use_fdr` | `True` | Use BH instead of `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate for BH |
| `oos_max_p` | `0.10` | Maximum p-value for OOS confirmation (diagnostic) |
| `min_oos_activations` | `10` | Minimum OOS activations for confirmation (diagnostic) |

---

## Advanced usage patterns

### Configuring the grid and IS/OOS split

```python
config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72),
    mfe_quantile=0.6,        # more aggressive take-profit (60th percentile MFE)
    mfe_floor=0.010,         # floor at 100 bp
    train_ratio=0.75,
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()
print(f"IS bars: {ad.split_idx}, OOS bars: {len(ad._frame) - ad.split_idx}")
```

### Inspecting the derived target

```python
for c in promoted:
    dt = c.derived_target
    print(f"{c.event_candidate_id}: h={dt.holding_period_h}, "
          f"direction={dt.direction}, sell_pct={dt.sell_pct:.4f}")
    # Score per horizon (|mean_advantage| / sqrt(h))
    best_h = max(dt.score_by_h, key=dt.score_by_h.get)
    print(f"  h* score={dt.score_by_h[best_h]:.5f}  "
          f"mean_adv={dt.advantage_by_h[best_h]:.4f}")
```

### Reading diagnostics on a promoted contract

```python
for c in promoted:
    if c.rejection_reasons:
        print(f"{c.event_candidate_id} (grade={c.alpha_score.grade}):")
        for r in c.rejection_reasons:
            print(f"  {r}")
```

### Filtering by OOS robustness

```python
# Promoted contracts with OOS confirmation
oos_robust = [
    c for c in promoted
    if c.oos_validation is not None and c.oos_validation.passed
]

# Promoted contracts with weak OOS (use with caution in Rule Discovery)
oos_weak = [
    c for c in promoted
    if c.oos_validation is None or not c.oos_validation.passed
]
```

### Disabling the OOS split (not recommended)

```python
config = AlphaConfig(train_ratio=1.0, asset="BTC")
# oos_validation will be None for every contract
```

### YAML export of promoted contracts

```python
import yaml
for c in promoted:
    with open(f"{c.alpha_id}.yaml", "w") as f:
        yaml.dump(c.to_contract_dict(), f)
```

---

## Statistical primitives (`stats.py`)

All implemented in **pure numpy** with no scipy or statsmodels dependency:

| Function | Algorithm |
|---|---|
| `spearmanr(x, y)` | Spearman correlation via numpy ranking |
| `cohens_d(group1, group2)` | `(mean1 - mean2) / std_pooled` |
| `ttest_ind(x, y, alternative)` | Independent t-test, p-value via incomplete beta |
| `benjamini_hochberg(p_values, q)` | Benjamini-Hochberg FDR control |
| `betai(a, b, x)` | Regularised incomplete beta (Lentz continued-fraction) |

---

## Downstream usage

Alpha Discovery produces contracts with `handoff_status = "PENDING_RULE_DISCOVERY"`.
Rule Discovery (Module 3) consumes the promoted contracts to run a realistic
backtest with order mechanics, fees, and operational edge validation.

The `derived_target` is the starting point for Rule Discovery: `holding_period_h`
and `sell_pct` are candidates, not validated parameters — Rule Discovery uses
them as the centre of its operational grid. The precise sizing is Rule
Discovery's responsibility.
