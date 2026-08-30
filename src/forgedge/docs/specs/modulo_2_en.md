# Module 2 — Alpha Discovery

Alpha Discovery is the third module in the FORGE pipeline and the **first that
sees the forward return**. It receives the `EventCandidate` list produced by
Event Discovery and, for each candidate, **derives the economic target directly
from the data** — selecting the horizon with the strongest rotation-null-
standardised excess, deriving the take-profit from the distribution of
favourable excursions, and
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
it scans a horizon grid (`horizon_grid`, session-resolved per timeframe class —
see the Full Configuration Reference) on the IS window and derives the
**excess log-return** at every horizon:

```python
Δ_h = μ_cond_h − μ_base_h   # conditional mean minus the unconditional baseline
```

`μ_cond_h` is the mean active-bar log-return at horizon `h`; `μ_base_h` is the
*unconditional* mean log-return over all valid bars at the same horizon.
Subtracting the baseline strips the asset's own drift out of the signal, so
`Δ_h` reflects the event's edge alone — never the prevailing trend. This is
why the same rule (e.g. `RSI > 80`) can derive *short* in a bull run and
*long* in a flat market: only the excess over the prevailing drift is read,
never a raw return that mixes the two together.

**Horizon selection `h*`:**
```python
z_h = Δ_h / σ_null,h          # excess standardised by a circular-rotation null
h*  = argmax_h |z_h|
```

`σ_null,h` is **not** a `1/√h` deflation. It comes from a **circular-rotation
null**: every non-trivial circular shift of the event's activation mask is
correlated against the same forward-return series (evaluated all at once via
an FFT-based cross-correlation), and the standard deviation of that empirical
null distribution is `σ_null,h`.

This replaces an earlier, simpler criterion — `score[h] = |Δ_h| / √h`, a
"Sharpe-like" deflation — which was dropped because it fails on **clustered
events**. A naive t-statistic `T_h = Δ_h / (σ_cond / √n)` treats the
overlapping forward-return windows as independent observations, so its
denominator shrinks with `√n` (correlated with `√h`); for an event whose
activations arrive in runs — the common case — this inflates `|T_h|` at long
horizons and systematically pins `h*` to the long edge of the grid even with
no real edge. The circular-rotation null re-derives the standardisation from
the data's own autocorrelation structure instead of assuming independence,
removing that bias.

The rotation null also yields a two-sided p-value per horizon
(`p_value_by_h`); Benjamini-Hochberg control at `fdr_q` over that set of
p-values produces `h_sig`, the horizons statistically distinguishable from
the null. `h*` is still always chosen as `argmax|z_h|` over the *whole*
grid — `h_sig` never restricts the search — but when `h*` falls outside
`h_sig` the target is flagged `statistically_weak` (used in Step 6 scoring
and the direction gate below) rather than discarded.

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
dt.holding_period_h        # int: selected horizon h*
dt.sell_pct                # float: MFE quantile at h*
dt.direction               # str: "long" | "short" | "undetermined"
dt.mean_advantage          # float: signed excess log-return Δ_h* at h*
dt.advantage_by_h          # dict[int, float]: excess log-return Δ_h per horizon
dt.t_stat_by_h             # dict[int, float]: rotation-standardised z_h per horizon
dt.score_by_h              # dict[int, float]: |z_h| selection score per horizon
dt.p_value_by_h            # dict[int, float]: circular-rotation-null p-value per horizon (diagnostic only)
dt.h_sig                   # tuple[int, ...]: horizons surviving BH control at fdr_q (diagnostic only)
dt.statistically_weak      # bool: True when h* is not in h_sig
dt.fixed_target            # bool: True when the target was user-specified (fixed-target mode) rather than derived
dt.data_derived_horizon_h  # int | None: fixed-target mode only — the horizon the data derivation would have picked
dt.data_derived_sell_pct   # float | None: fixed-target mode only — the sell_pct the data derivation would have produced
```

`direction = "undetermined"` (the contract is rejected) when **any** of:
- no horizon produces a finite excess log-return `Δ_h`;
- `|z_h*| < min_direction_t` (default `0.5`) — the excess at the selected
  horizon is not distinguishable from the rotation null;
- `require_significant_direction = True` (the default, on
  `PromotionThresholds`) **and** `h*` is not in `h_sig` — no horizon cleared
  the Benjamini-Hochberg gate, so `argmax|z_h|` would otherwise assign a
  direction off what is effectively a coin-flip (often the drift-driven long
  edge of the grid). Set `require_significant_direction = False` for the
  legacy non-blocking behaviour — a direction is always assigned subject only
  to `min_direction_t`, with thin evidence surfaced via `statistically_weak`
  instead of gated.

All subsequent measurements (IC, win rate, regime) are computed **at the
derived target** (`h*`, `sell_pct*`, `direction*`).

#### Horizon-grid enrichment

`AlphaConfig.horizon_enrichment` (default `(0.5, 1.0, 2.0)`) adds, **per
event**, horizons around the event's own structural timescale to the base
`horizon_grid` scanned above — a union, never a restriction. For every
candidate, `EventCandidate.dominant_window()` returns `w`, the slowest
indicator/transform window embedded in the event's own conditioning info
(e.g. an `ema_9`-based event has `w = 9`); for each multiplier `m` in
`horizon_enrichment`, `round(m · w)` is added to the horizons scanned for
that candidate. So an `ema_9` event also scans `h ≈ 5, 9, 18` even when the
base grid skips them — the default multipliers cover the empirically
supported band (reaction-type events tend to resolve in about half the
window; cycle-type events in one to two windows). Enriched horizons are
capped at `split // horizon_enrichment_min_obs` (default `20`) so a slow
conditioning window cannot demand a holding period the in-sample data can't
statistically support; the cap never restricts the base `horizon_grid`
itself. Every added horizon is counted by the session ledger and priced by
the search-level rotation null like any other. Set `horizon_enrichment=None`
(or `()`) to disable it and scan only the base grid.

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
`"IC weak …"` entry in `diagnostics` but does not block
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
**replayed on the OOS tail** (the last `1 - train_ratio` of the dataset,
offset by `embargo_bars` if set — see the Full Configuration Reference):

- OOS forward returns are oriented to the derived direction
- Win rate, lift, mean_advantage, and the t-test are measured on the OOS window
- `oos.passed = True` when both criteria hold:
  1. `mean_advantage > 0` (oriented advantage stays positive on OOS)
  2. `p_value < oos_max_p` (default: 0.10, one-sided)

  No minimum activation count is imposed as a separate gate — the p-value
  already encodes sample size. A non-parametrizable floor of 10 activations
  instead triggers a non-blocking diagnostic about low statistical
  reliability; there is no `min_oos_activations` field to tune.

**Failing OOS confirmation produces a non-blocking diagnostic** —
`"OOS weak …"` in `diagnostics` — but does not prevent
promotion. The OOS statistical signal contributes to the grade.

Output — `OOSValidation`:
```python
oos.n_bars                # int: bars in the OOS window
oos.n_activations         # int: activations with complete forward horizon in OOS
oos.mean_advantage        # float: mean oriented return on OOS (> 0 = confirmed)
oos.t_stat                # float
oos.p_value               # float: one-sided t-test
oos.win_rate              # float: OOS win rate at the derived target
oos.base_rate             # float: OOS base rate at the derived target
oos.lift                  # float: OOS lift
oos.passed                # bool
oos.min_detectable_effect # float: minimum Cohen's d detectable at oos_max_p given the OOS sample size — compare to IS cohens_d to diagnose an underpowered OOS window
```

When `train_ratio = 1.0`, IS covers the full dataset and `oos_validation` is
`None` for every contract (OOS split disabled — not recommended in production).

---

### Step 6 — Alpha scoring

Composite score (0–1), a weighted average of five signal-quality terms:

| Component | Default weight | Normalisation |
|---|---|---|
| IC magnitude | 0.20 | `min(|IC| / 0.10, 1.0)` |
| Lift | 0.25 | `min(lift / 0.30, 1.0)` |
| Cohen's d | 0.15 | `clip(d / 0.80, -1.0, 1.0)` — **signed** |
| `z` (rotation-null excess) | 0.25 | `min(|z_h*| / 3.0, 1.0)` |
| Regime breadth | 0.15 | `regime_breadth` (0–1) |

`z` is `|z_h*|`, the rotation-null-standardised excess statistic at the
selected horizon (Step 1) — the edge-to-noise ratio. The Cohen's d term is
normalised **signed** rather than clipped to zero on the low end: a negative
Cohen's d (the conditioned group performs *worse* than the background)
actively penalises the composite instead of contributing nothing. When
regime is unavailable, the breadth term is dropped and the remaining weights
are renormalised. `score_weights` also still accepts a legacy 4-tuple
`(ic, lift, cohens_d, breadth)`, upgraded with a default `z` weight.

Two further adjustments apply after the weighted sum:
- if the derived target is `statistically_weak` (`h*` outside `h_sig`,
  Step 1), the composite is **multiplied** by `statistically_weak_penalty`
  (default `0.6`) — a horizon selected by exactly the selection bias the FDR
  control exists to catch cannot rank highly;
- if the OOS confirmation passed (`oos_validation.passed`), `oos_bonus`
  (default `0.05`) is **added**, separating confirmed edges from unconfirmed
  ones.

The result is clamped to `[0, 1]`.

**Grade:** A ≥ 0.75 | B ≥ 0.50 | C ≥ 0.25 | D < 0.25

All four grades are forwarded to Rule Discovery — the grade signals the
strength of statistical evidence, not exclusion.

---

### Step 7 — Contract compilation

The **only rejection gate** is the absence of a determined direction:

| Gate | Condition |
|---|---|
| **Hard (blocks)** | `direction == "undetermined"` — no finite excess log-return across the grid, `\|z_h*\| < min_direction_t`, or (with `require_significant_direction=True`, the default) `h*` not in `h_sig` (see Step 1) |
| **Diagnostics (non-blocking)** | Weak IC, lift below threshold, cohens_d below threshold, insufficient activations, non-significant FDR/p-value, weak OOS |

Diagnostics live in their own field, `diagnostics`. `rejection_reasons` holds
only what actually blocked promotion, so it is **empty on a promoted contract**;
`diagnostics` documents the statistical weaknesses detected and is routinely
non-empty on promoted contracts.

```python
for c in contracts:
    print(f"{c.event_candidate_id}: promoted={c.promoted}, grade={c.alpha_score.grade}")
    for d in c.diagnostics:
        print(f"  {d}")
# Example:
# EVT-rsi_25-PR-0042: promoted=True, grade=C
#   lift 0.0520 < 0.08
#   OOS weak (p=0.143 vs 0.10, mean_adv=0.00210, n_act=7)
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
c.fee_per_side        # float: not deducted here — but this is the cost
                      # Rule Discovery charges (session-resolved)

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
c.rejection_reasons   # list[str]: blocking causes only — empty when promoted
c.diagnostics         # list[str]: non-blocking observations that feed the grade
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
regime_dependency, regime_breadth, composite_score, grade, rejection_reasons,
diagnostics
```

---

## Full configuration reference

### `AlphaConfig`

| Parameter | Default | Description |
|---|---|---|
| `horizon_grid` | `UNSET` → session-resolved by timeframe class: `(1,2,4,8,12,24)` on 1H/4H, `(1,2,3,5,7,10)` on 1D and up, `(1,2,5,10,20,50)` sub-hourly (standalone fallback outside `forge()`: `(1,2,4,8,12,24)`) | Candidate holding horizons in bars, scanned in Step 1 |
| `mfe_quantile` | `0.5` | MFE quantile for deriving sell_pct (0.5 = median) |
| `mfe_floor` | `0.005` | sell_pct floor (50 bp) after the quantile |
| `train_ratio` | `0.7` | IS fraction of the dataset (0 < x ≤ 1.0) |
| `embargo_bars` | `0` | Extra quarantine bars after the IS/OOS split before OOS confirmation starts — guards against serial correlation beyond the mechanical forward-window purge |
| `horizon_enrichment` | `(0.5, 1.0, 2.0)` | Per-event multipliers of `EventCandidate.dominant_window()` added to `horizon_grid` (union, never a restriction). `None`/`()` disables enrichment |
| `horizon_enrichment_min_obs` | `20` | Statistical cap for enriched horizons: `h <= split // horizon_enrichment_min_obs` |
| `thresholds` | `PromotionThresholds()` | Admission / promotion gates — mostly diagnostic (see below) |
| `asset` | `"ASSET"` | Traceability metadata (copied to contract and alpha_id) |
| `exchange` | `""` | Traceability metadata |
| `timeframe` | `"1H"` | Not metadata-only: drives session-resolution of `horizon_grid` and other bar-counting fields |
| `fee_per_side` | `0.002` *(session-resolved)* | Not deducted from the target here; it is the cost basis M3 charges, propagated into `BacktestParams.fee` |
| `close_col` | `"close"` *(session-resolved)* | Close price column; propagates to `BacktestParams.{target_col, buy_price_anchor}` |
| `timestamp_col` | `"open_dt"` | Datetime column name (or DatetimeIndex name) |
| `regime_col` | `"regime"` | Regime column (from Market Context) |
| `regime_stable_col` | `"regime_stable"` | Regime stability column |
| `use_stable_regime_only` | `False` | Restrict regime analysis to stable bars |
| `min_regime_obs` | `10` | Minimum IS observations to evaluate a regime |
| `rolling_ic_window` | `None` | Rolling IC window (None → 60 days in bars) |
| `bars_per_day` | `None` | Bars per day (None → inferred from spacing) |
| `score_weights` | `(0.20, 0.25, 0.15, 0.25, 0.15)` | Weights: (IC, lift, cohens_d, z, breadth). A legacy 4-tuple (IC, lift, cohens_d, breadth) is also accepted. |
| `statistically_weak_penalty` | `0.6` | Composite multiplier when `statistically_weak=True`. |
| `oos_bonus` | `0.05` | Additive composite bonus when the OOS confirmation passes. |
| `discovery_date` | `None` | ISO date for contracts (None → today) |
| `fixed_target` | `None` | `TargetConfig` — when set, skips target *derivation* and measures every candidate against this user-specified target (see Fixed-target mode below) |
| `fixed_target_diagnostic` | `True` | Fixed-target mode only: still runs the data derivation read-only to populate `data_derived_*` convergence diagnostics |
| `target_mode` | `"proj"` | Binary-target definition: `"proj"` measures excess of the local trend (PROJ_LOG); `"abs"` is the legacy absolute-return target. PROJ applies to long only |
| `trend_sma_mult` | `2.0` | PROJ_LOG only: trend SMA window = `round(trend_sma_mult · h)` bars |

### `PromotionThresholds`

Most of these fields control **diagnostics**, not promotion gates — the only
blocking gate anywhere in Alpha Discovery is direction determination (Step
1/7), and `min_direction_t` / `require_significant_direction` are the two
fields that actually participate in it.

| Parameter | Default | Description |
|---|---|---|
| `ic_min_abs` | `0.02` | \|IC\| threshold for classifying IC as weak (diagnostic) |
| `ic_max_p` | `0.05` | Maximum p-value for classifying IC as weak (diagnostic) |
| `min_lift` | `0.08` | Minimum lift (diagnostic) |
| `min_cohens_d` | `0.15` | Minimum Cohen's d (diagnostic) |
| `max_p_value` | `0.05` | Maximum p-value, reachable only when `use_fdr=False` (diagnostic; inert under every preset, which all set `use_fdr=True`) |
| `use_fdr` | `True` | Use BH instead of `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate for BH (drives Step 1's `h_sig`) |
| `oos_max_p` | `0.10` | Maximum one-sided p-value for OOS confirmation |
| `min_direction_t` | `0.5` | Minimum \|z_h*\| for a direction to be assigned — below this, `direction = "undetermined"` (**gates promotion**) |
| `require_significant_direction` | `True` | A direction is assigned only if `h*` is in `h_sig` (BH-significant); otherwise `direction = "undetermined"` (**gates promotion**). `False` restores the legacy non-blocking behaviour |

`PromotionThresholds` has no `min_activations` or `min_oos_activations`
fields — there is no minimum-activation-count promotion gate anywhere in
Alpha Discovery; sample size is absorbed into the p-values themselves (Step
1's rotation-null p-values, the OOS t-test).

### `TargetConfig` (fixed-target mode)

Passed via `AlphaConfig.fixed_target` to bypass per-event target derivation —
see Fixed-target mode below.

| Parameter | Default | Description |
|---|---|---|
| `horizon` | *(required)* | Holding period in bars (`> 0`); added to `horizon_grid` if absent |
| `min_return` | *(required)* | Take-profit threshold as a fraction (e.g. `0.02` = 2%), used as `sell_pct` |
| `side` | *(required)* | `"long"` or `"short"` — never overwritten by the data |
| `min_activations` | `10` | TargetOptimizer-workflow floor for valid lift scoring; ignored by Alpha Discovery's fixed-target mode |
| `min_lift_atoms` | `1.0` | TargetOptimizer 1st-pass (atomic events) prune threshold; ignored by fixed-target mode |
| `min_lift_result` | `1.0` | TargetOptimizer 2nd-pass (final result set) prune threshold; ignored by fixed-target mode |
| `target_mode` | `"proj"` | `"abs"` or `"proj"` (PROJ_LOG) — see `AlphaConfig.target_mode` |
| `trend_sma_mult` | `2.0` | PROJ_LOG trend SMA multiplier — see `AlphaConfig.trend_sma_mult` |

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
    # Score per horizon (|z_h|, the rotation-null-standardised excess)
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

## Fixed-target mode

The default flow *derives* `(h*, sell_pct*, direction*)` per event (Step 1).
`AlphaConfig.fixed_target` is the one documented exception: set it to a
`TargetConfig` and Alpha Discovery **skips derivation entirely**, measuring
every candidate against a single user-specified `(horizon, min_return, side)`
instead. Every downstream measurement — IC, win rate, lift, Cohen's d, regime
sensitivity, OOS confirmation, Step 6 scoring — runs unchanged against that
fixed target. This is the mechanism `TargetOptimizer` uses internally to
evaluate many event candidates against one common economic target rather than
letting each pick its own.

```python
from forgedge import AlphaConfig
from forgedge.alpha_discovery.models import TargetConfig

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    fixed_target=TargetConfig(horizon=12, min_return=0.02, side="long"),
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

c = contracts[0]
dt = c.derived_target
dt.fixed_target            # True
dt.holding_period_h        # 12 (the user's horizon, added to horizon_grid if absent)
dt.sell_pct                # 0.02 (from min_return)
dt.direction               # "long" (from side — never overwritten by the data)
dt.mean_advantage          # nan — not measured in fixed-target mode
dt.data_derived_horizon_h  # the horizon the data derivation *would* have picked
dt.data_derived_sell_pct   # the sell_pct the data derivation *would* have produced
```

With `fixed_target_diagnostic=True` (the default), the ordinary data
derivation still runs **read-only** alongside the fixed target, populating
`data_derived_horizon_h`/`data_derived_sell_pct` and the per-horizon
diagnostics on the contract — a consumer can then check
`data_derived_horizon_h ≈ holding_period_h` as a convergence signal that the
data independently confirms the user's chosen horizon. Set it `False` for a
pure, slightly faster bypass with those diagnostics left empty.

The binary target itself (used for win rate / lift / base rate) is governed
by `target_mode` (also on `TargetConfig`, mirrored on `AlphaConfig` for the
normal derived-target flow):

- `"abs"` — the legacy absolute-return target: the raw forward return against
  `min_return`.
- `"proj"` (default) — **PROJ_LOG**: the forward return in excess of the
  local trend, computed against an SMA of window `round(trend_sma_mult · h)`
  bars (default multiplier `2.0`). This strips the trend premium a long event
  would otherwise be credited with in a bull market — the same drift-removal
  idea as the derived target's excess log-return (Step 1), applied here to
  the user-specified target instead. PROJ applies to **long** only; a
  `"short"` target reverts to `"abs"` (the bear trend *is* the alpha to
  capture, not noise to subtract). Falls back to `"abs"` with a warning when
  history is shorter than the PROJ warmup (`(trend_sma_mult + 1) · h` bars).

`TargetOptimizer` sits outside the parameter-coherence resolver (its
`discover_alpha()` builds an internal `AlphaConfig` without going through
`forge()`'s session resolver), so on daily-or-slower data the same
hourly-grid fallback that affects standalone `AlphaConfig` use also affects
it — pass `horizon_grid` explicitly on the config handed to
`discover_alpha()` for daily-or-slower use.

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
