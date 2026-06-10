# Module 2 — Alpha Discovery

Alpha Discovery is the third module in the FORGE pipeline and the **first that
sees the forward return**. It receives the `EventCandidate` list produced by
Event Discovery and, for each candidate, **derives the economic target directly
from the data** — selecting the horizon that maximises the statistical separation
between bars where the event is active and bars where it is not. The output is a
list of `AlphaContract` — one per candidate — recording the derived target, all
in-sample statistical measurements, the out-of-sample confirmation, and the
promotion outcome.

---

## Basic usage

```python
from forgedge import (
    MarketContext,
    EventDiscovery,
    AlphaDiscovery, AlphaConfig,
)
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")

# Modules 0 and 1: regime + event discovery
enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)
candidates = ed.run()

# Module 2: target derivation + predictive measurement
config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    # horizon is derived automatically from the grid — no target to specify
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

promoted = ad.promoted_contracts()
print(f"{len(promoted)} promoted out of {len(contracts)} evaluated")
print(ad.summary().head())
```

Unlike previous versions, `AlphaConfig` **does not require a `TargetDefinition`**:
horizon, sell_pct, and direction are derived per event from the data itself.
`asset` and `timeframe` are traceability metadata and have no effect on any
measurement.

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
  ├─ Step 6: alpha scoring
  └─ Step 7: contract compilation + FDR
        │
        ▼
  list[AlphaContract]   (all: HYPOTHESIS + REJECTED)
  promoted_contracts()  (HYPOTHESIS only) ──► Rule Discovery (not implemented)
```

The DataFrame is split chronologically into **in-sample (IS)** and
**out-of-sample (OOS)** via `train_ratio` (default: 70/30). All statistical
measurements (IC, win rate, regime) are computed on IS; the derived target is
then confirmed on the OOS tail.

---

## 8-step pipeline

### Step 1 — Per-event target derivation

Alpha Discovery **takes no economic parameters as input**. For each candidate
it scans a horizon grid (`horizon_grid`, default: `(1, 2, 3, 4, 6, 8, 12, 16,
24, 36, 48)` bars) on the IS window and selects:

- **`holding_period_h`** — the horizon that maximises `|t-stat|` of the
  difference between active-bar and inactive-bar forward returns;
- **`mean_advantage`** — the mean oriented forward return of active bars at
  that horizon;
- **`sell_pct`** — `abs(mean_advantage)`: the take-profit baseline;
- **`direction`** — `"long"` when `mean_advantage > 0`, `"short"` when < 0,
  `"undetermined"` when non-finite.

The computation is **vectorised** over the grid: a set of sufficient statistics
(count, sum, sum-of-squares) is precomputed once and reused across all
candidates, keeping the cost linear in the number of candidates.

Output — `DerivedTarget`:
```python
dt.holding_period_h  # int: selected horizon
dt.sell_pct          # float: |mean_advantage| — take-profit baseline
dt.direction         # str: "long" | "short" | "undetermined"
dt.mean_advantage    # float: mean oriented return at h*
dt.advantage_by_h    # dict[int, float]: mean active return per horizon
dt.t_stat_by_h       # dict[int, float]: t-stat per horizon
```

All subsequent measurements (IC, win rate, regime) are computed **at the
derived target** (`h*`, `sell_pct*`, `direction*`).

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

#### IC admission gate

A candidate passes the IC gate if **not** both of the following are true:
- `|IC| < ic_min_abs` (default: 0.02) — IC is weak
- `p_value > ic_max_p` (default: 0.05) — not statistically significant

If the IC is small but statistically significant (low p-value), the candidate
still passes the gate.

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
Cohen's d and the one-sided t-test then have the same interpretation for both
long and short events.

| Metric | Description |
|---|---|
| `n_activations` | IS bars with active event and valid target |
| `win_rate` | `mean(target_binary_is)` on active bars |
| `lift` | `win_rate - base_rate_is` |
| `fwd_return_mean` | Mean oriented return on active bars (IS) |
| `cohens_d` | `(mean_active - mean_inactive) / std_pooled` |
| `t_stat`, `p_value` | One-sided t-test (`alternative="greater"`) |

`base_rate_is` is computed **on the IS** at the derived target — each candidate
has its own base rate (different horizon and sell_pct yield a different base
rate).

---

### Step 5 — Regime sensitivity (IS)

For each regime, with at least `min_regime_obs` IS observations:

1. Spearman IC of feature vs `fwd_return_h*` in the regime (IS)
2. Conditional win rate in the regime (IS)

Results per regime are **cached per `(feature, horizon, regime)`**, analogously
to the global IC.

If `use_stable_regime_only = True`, only IS bars with `regime_stable = True` are
used.

Strength classification and dependency type are unchanged: strong/moderate/
negligible/insufficient; agnostic/conditional/specific/broken/unknown.

---

### OOS validation

After all IS measurements, the derived target `(h*, sell_pct*, direction*)` is
**replayed on the OOS tail** (the last `1 - train_ratio` of the dataset):

- OOS forward returns are oriented to the derived direction
- Win rate, lift, mean_advantage, and the t-test are measured on the OOS window
- The candidate **passes OOS confirmation** (`oos.passed = True`) when all three:
  1. `n_oos_activations >= min_oos_activations` (default: 10)
  2. `mean_advantage > 0` (oriented advantage stays positive on OOS)
  3. `p_value < oos_max_p` (default: 0.10)

Failing OOS confirmation is a rejection gate: a candidate is promoted only when
all IS gates **and** OOS confirmation are cleared.

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
oos.passed          # bool: True when all three criteria are met
```

When `train_ratio = 1.0`, IS covers the full dataset and `oos_validation` is
`None` (OOS split disabled — not recommended in production).

---

### Step 6 — Alpha scoring

Composite score (0–1) unchanged from the previous version:

| Component | Default weight | Normalisation |
|---|---|---|
| IC magnitude | 0.25 | `min(\|IC\| / 0.10, 1.0)` |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` |
| Regime breadth | 0.20 | `regime_breadth` (0–1) |

When regime is unavailable, the breadth term is removed and remaining weights
are renormalised. Grade: A ≥ 0.75, B+ ≥ 0.60, B ≥ 0.45, C < 0.45.

---

### Step 7 — Contract compilation

A candidate is promoted (`status = "HYPOTHESIS"`) only when it passes **all**
of the following gates:

| Gate | Parameter | Default |
|---|---|---|
| Target derivable | — | direction ≠ "undetermined" |
| IC admitted | `ic_min_abs`, `ic_max_p` | 0.02, 0.05 |
| Lift ≥ threshold | `min_lift` | 0.08 |
| Cohen's d ≥ threshold | `min_cohens_d` | 0.15 |
| Activations ≥ threshold | `min_activations` | 30 |
| Statistical significance | `use_fdr`/`fdr_q` or `max_p_value` | BH q=0.10 |
| OOS confirmation | `oos_max_p`, `min_oos_activations` | 0.10, 10 |

`rejection_reasons` lists all failed gates:
```python
for c in contracts:
    if not c.promoted:
        print(c.event_candidate_id, c.rejection_reasons)
```

BH FDR (Step 8) is applied to IS t-test p-values of all candidates
simultaneously before compiling contracts.

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
c.promoted            # bool
c.rejection_reasons   # list[str]: failed gates (empty if promoted)
c.fdr_promoted        # bool | None

# Handoff to Rule Discovery
c.handoff_status      # str: "PENDING_RULE_DISCOVERY"
c.rule_discovery_response  # dict | None
```

---

## Output methods

### `ad.run() → list[AlphaContract]`

Derives the target per event, runs all IS + OOS measurements, and returns the
complete list (promoted + rejected). Must be called first.

Properties populated by `run()`:
- `ad.market_structure` — IS market structure
- `ad.split_idx` — row index of the IS/OOS boundary

### `ad.promoted_contracts() → list[AlphaContract]`

Only contracts with `status == "HYPOTHESIS"`.

### `ad.summary() → pd.DataFrame`

Flat summary sorted by `composite_score` descending. Includes OOS columns
compared to the previous version:

```
alpha_id, status, promoted, event_candidate_id, expression, pattern_family,
holding_period_h, sell_pct, direction, mean_advantage,
feature, ic, ic_p_value, ic_admitted, rolling_ic_stable,
n_activations, win_rate, base_rate, lift, fwd_return_mean, cohens_d, t_stat, p_value,
fdr_promoted, oos_passed, oos_p_value, oos_lift,
regime_dependency, regime_breadth, composite_score, grade, rejection_reasons
```

### `c.to_dict() → dict`

Flat dictionary (one row of `summary()`).

### `c.to_contract_dict() → dict`

Full nested contract as a dictionary, ready for YAML/JSON serialisation.

---

## Full configuration reference

### `AlphaConfig`

| Parameter | Default | Description |
|---|---|---|
| `horizon_grid` | `(1,2,3,4,6,8,12,16,24,36,48)` | Candidate holding horizons in bars |
| `train_ratio` | `0.7` | IS fraction of the dataset (0 < x ≤ 1.0) |
| `thresholds` | `PromotionThresholds()` | Admission and promotion gates |
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

| Parameter | Default | Description |
|---|---|---|
| `ic_min_abs` | `0.02` | Minimum \|IC\| for IC gate |
| `ic_max_p` | `0.05` | Maximum p-value for IC gate |
| `min_lift` | `0.08` | Minimum lift (+8pp) |
| `min_cohens_d` | `0.15` | Minimum Cohen's d |
| `max_p_value` | `0.05` | Maximum p-value (when `use_fdr=False`) |
| `min_activations` | `30` | Minimum IS activations |
| `use_fdr` | `True` | Use BH instead of `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate for BH |
| `oos_max_p` | `0.10` | Maximum p-value for OOS confirmation |
| `min_oos_activations` | `10` | Minimum OOS activations for confirmation |

---

## Advanced usage patterns

### Configuring the grid and IS/OOS split

```python
config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72),   # custom grid
    train_ratio=0.75,                        # 75% IS / 25% OOS
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
    # Full horizon profile
    for h, adv in dt.advantage_by_h.items():
        print(f"  h={h}: advantage={adv:.4f}, t={dt.t_stat_by_h[h]:.2f}")
```

### Inspecting OOS confirmation

```python
for c in contracts:
    oos = c.oos_validation
    if oos is not None:
        print(f"{c.event_candidate_id}: OOS passed={oos.passed}, "
              f"lift={oos.lift:.4f}, p={oos.p_value:.4f}")
```

### Candidates rejected only by OOS

```python
# Promising on IS but not confirmed OOS
is_ok_oos_fail = [
    c for c in contracts
    if not c.promoted
    and c.oos_validation is not None
    and not c.oos_validation.passed
    and all("OOS" not in r for r in c.rejection_reasons[:5])
]
```

### Disabling the OOS split (not recommended)

```python
# train_ratio=1.0 disables the split; oos_validation will be None for every contract
config = AlphaConfig(train_ratio=1.0, asset="BTC")
```

### Tightening promotion gates

```python
from forgedge.alpha_discovery.models import PromotionThresholds

strict = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    horizon_grid=(6, 12, 24, 48),
    train_ratio=0.75,
    thresholds=PromotionThresholds(
        min_lift=0.12,
        min_cohens_d=0.20,
        min_activations=50,
        fdr_q=0.05,
        oos_max_p=0.05,
        min_oos_activations=15,
    )
)
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
Rule Discovery (Module 3, not yet implemented) will consume the promoted
contracts to run a realistic backtest with order mechanics, fees, and
operational edge validation.

The `derived_target` in promoted contracts is the starting point for Rule
Discovery's calibration of operational parameters: `holding_period_h` and
`sell_pct` are candidates, not validated parameters — they are the product of
an IS grid optimisation, and the OOS confirmation shows the signal persists
out-of-sample, but precise sizing is Rule Discovery's responsibility.
