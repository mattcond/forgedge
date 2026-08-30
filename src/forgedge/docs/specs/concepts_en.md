# FORGE — From Market to Signal: Event, Alpha, and Rule

This document explains how FORGE produces trading signals from historical data,
progressively building the three fundamental concepts of the system: the
**event**, the **alpha**, and the **rule**. Each concept answers a precise
question and produces a formal artefact that the next one receives as input —
without ever passing data backwards, reopening closed optimisations, or mixing
domains.

---

## The problem FORGE addresses

Systematic trading edge research is haunted by a paradox: it is always possible
to find patterns that "would have worked" in the past. You just need to
optimise thresholds long enough, pick the right periods, look at the return
while building the signal. The result is an attractive backtest and a strategy
that does not work in production.

The three main sources of this distortion are:

1. **Look-ahead bias in signal selection.** If the event threshold is chosen
   because it produces positive returns, the threshold already "knows" the
   future. Any subsequent backtest is circular.

2. **Optimisation on the same sample as evaluation.** If you search for the
   best horizon, take-profit, or regime filter on the same data where you
   measure Profit Factor, the PF reflects sample noise, not structural edge.

3. **No separation between the statistical and operational phases.** Knowing
   that an event statistically predicts a positive return is not the same as
   knowing it is profitable under realistic fees, slippage, and order mechanics.

FORGE answers these three problems with a three-module pipeline separated by
formal boundaries: no module can access the next module's data, no threshold
can be recalibrated after discovery.

---

## 1. The Event: Observing the Market without Bias

### Definition

An **event** is a boolean condition on historical bars: for every bar in the
KPI Table, the event is either `True` (active) or `False` (inactive). It is
not a trading signal — it is a structured market observation.

```
Event: RSI_14 < 31.2
        │
        ├─ Bar 4712 (RSI=29.8): True  ← event active
        ├─ Bar 4713 (RSI=33.1): False
        ├─ Bar 4714 (RSI=30.0): True  ← event active
        └─ ...
```

An event can be composed of multiple conditions in AND:

```
Event: RSI_14 < 31.2  AND  spread_ema_9_25 < -0.0118
```

The formal name in the codebase is `EventCandidate`. The boolean expression is
stored in `c.expression` and `c.event_formula`.

### What makes an event useful

Not all events are useful. An event that activates once, or always in the same
month, or at a frequency of 0.1% of bars, cannot support reliable measurements.
The selection criteria are purely **structural**. The default counting unit is
the **episode** (a run of consecutive activations), not the raw activated bar
— a signal that stays active for several bars in a row counts once, not once
per bar:

| Criterion | Meaning |
|---|---|
| Rate | Episodes per month ≥ `min_tpm` (default 0.5 episodes/month) |
| Power | At least `min_episodes` episodes in the sample (default 10) |
| Dispersion | Episode-level Index of Dispersion (Var/Mean of monthly episode counts) ≤ a Poisson-χ² floor scaled by `dispersion_margin` (default 1.3) — `eff_max_dispersion = poisson_floor(n_months) × dispersion_margin` |

The gate never rejects an event that is statistically consistent with a
random process at its own observed rate, while still letting a preset's own
tolerance for burstiness actually bind. There is no longer a fixed "no month
above 40% of activations" rule — `max_monthly_share` is still reported, but
only as a diagnostic; the real dispersion check is the Poisson-χ² test above.
(A legacy `"bar"` counting mode, closer to the old semantics, remains
available for backward compatibility.)

The ConsistencyGate in Module 1 filters out all events that do not meet these
criteria. What passes the gate is an event with **stable temporal structure**
— regardless of whether it produces positive or negative returns.

### Distributional thresholds, not absolute ones

The threshold `RSI_14 < 31.2` is not chosen because 31.2 is a magic number.
It is the **10th percentile of the IS distribution of RSI on that specific
asset**. This is called a "distributional threshold": it represents an extreme
state of the indicator *for that asset*, not a universal absolute value.

The result is that the same structural pattern ("RSI in low zone") produces
different thresholds on different assets:

| Asset | RSI p10 | Meaning |
|---|---|---|
| BTC | 27.8 | BTC enters oversold at 27.8 |
| ADA | 31.2 | ADA enters oversold at 31.2 |
| ETH | 29.1 | ETH enters oversold at 29.1 |

The structure is the same (RSI at 10th percentile); the threshold is adaptive.

### Isolation from the forward return

The fundamental principle of Module 1 is that it **never sees the future
return**. Features are classified, thresholds are fixed from the IS
distribution, temporal stability is measured — all without computing a single
forward return.

This guarantees that event thresholds are not contaminated by knowledge of
what happens afterwards. There is no look-ahead bias in signal selection.

### Thresholds are immutable

Once Event Discovery fixes the thresholds — `RSI_14 < 31.2`,
`spread_ema_9_25 < -0.0118` — they never change, even if downstream modules
found that a different threshold would produce a higher Profit Factor. Changing
the threshold would require a new Event Discovery session on a new IS sample.

This prevents the most subtle form of look-ahead bias: optimising thresholds
after seeing returns.

### Artefact: `EventCandidate`

```python
c = candidates[0]
c.expression      # "rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118"
c.event_formula   # human-readable version with percentile notation
c.activation_stats.n_activations   # 87 IS activations
c.activation_stats.n_active_months # active in 18 distinct months

# Apply the event to any DataFrame with the same native columns
signal = c.apply(new_kpi_table)    # pd.Series bool
```

The `EventCandidate` is portable: it can be applied to future data without
requiring the discovery session. Thresholds are stored in the components.

---

## 2. The Alpha: Measuring Predictive Power

### The transition to the forward return

An event describes a market configuration. But it says nothing about what
happens next. It is normal for RSI to be at its 10th percentile — it can be
a bounce signal or the beginning of a further drop.

The **alpha** is the empirical answer to the question: *given that the event
activated, what happens statistically in the next h bars?*

This is the **first exposure to the future return** in the entire pipeline.
The forward return has never been computed before this point.

### Deriving the target without assumptions

Alpha Discovery does not receive a time horizon, trading direction, or
take-profit level from the user. It **derives them from the data** for each
event.

**Horizon selection `h*`:**

The horizon grid itself is not a fixed constant — it is session-resolved per
timeframe class by `PipelineContext`: `(1, 2, 3, 5, 7, 10)` bars on daily
timeframes, `(1, 2, 4, 8, 12, 24)` on intraday, `(1, 2, 5, 10, 20, 50)` on
HFT. (Building an `AlphaConfig` standalone, outside `forge()`, falls back to
the hourly/intraday grid `(1, 2, 4, 8, 12, 24)` regardless of the declared
timeframe.)

For each event, Alpha Discovery computes the excess log-return at every
horizon in the grid, `Δ_h = mean_advantage[h]` — the mean active-bar
log-return minus the *unconditional* baseline over all valid bars at that
horizon — then standardises it against a **circular-rotation null**: the
event's own activation pattern is rotated against the actual return series
many times to build a null distribution of what `Δ_h` looks like under no
real relationship, giving a standardised score `z_h = Δ_h / σ_null,h` and a
p-value for each horizon. The selected horizon is:

```
h* = argmax |z_h|
```

This replaced an earlier, naive `Δ_h / (σ_cond / √n)` deflation (dividing by
the naive standard error, roughly a `1/√h`-shaped correction) because that
approach treats the overlapping forward-return windows of a clustered or
episodic event as independent samples: its denominator shrinks with the
horizon regardless of whether the event actually clusters in runs, which
inflates the score at long horizons and pins `h*` to the far edge of the
grid even with no real edge. Standardising against a null built from the
event's own activation pattern removes that bias.

A Benjamini-Hochberg FDR gate is then applied to the p-values across the
whole grid, producing the set of horizons `h_sig` that clear significance.
`h*` is always chosen as the `argmax |z_h|` over the *whole* grid — but if
`h*` falls outside `h_sig`, the target is flagged `statistically_weak` (this
penalises, rather than discards, the resulting alpha score — see the
composite score below).

**Direction derivation:**

`direction` is `"undetermined"` — and the event is rejected — in any of
three cases:

- no horizon yields a finite excess `Δ_h` (non-finite across the whole grid);
- the standardised excess at the selected horizon is too small,
  `|z_h*| < min_direction_t` (default 0.5);
- (default behaviour, `require_significant_direction=True`) `h*` itself is
  not BH-significant (`statistically_weak`) — the excess is statistically
  indistinguishable from the rotation null everywhere, so reading a direction
  off `argmax|z_h|` would amount to a coin-flip, often biased toward the
  asset's own drift at the long edge of the grid.

Otherwise the direction is `"long"` when `Δ_h*` (`mean_advantage[h*]`) is
positive, `"short"` when negative.

**`sell_pct` derivation:**

The baseline take-profit is not the mean return (which includes losing bars),
but the **quantile of the Maximum Favorable Excursion (MFE) distribution** of
active bars at horizon `h*`:

```
sell_pct = max(quantile(MFE_active_bars, 0.5), 0.005)
```

The MFE of an active bar is the maximum favourable excursion reached in the
`h*` bars following that bar. The 50th percentile of this distribution is the
conservative take-profit estimate: half of IS activations reached or exceeded
this excursion.

### Measuring statistical evidence

With the derived target `(h*, direction*, sell_pct*)`, Alpha Discovery measures
statistical evidence on IS:

| Measure | Meaning |
|---|---|
| **IC** (Information Coefficient) | Spearman correlation between the continuous feature and the forward return at `h*`. Measures the strength of the continuous signal before the threshold. |
| **Win rate and lift** | Frequency of active bars with positive oriented return; lift = win_rate − base_rate. Measures how often the event is "right". |
| **Cohen's d** | Separation between the return distribution on active vs inactive bars. Measures effect size. |
| **Rolling IC** | Sign stability of IC across ≈20 rolling windows. Measures whether the relationship is stable over time. |

All IS measurements use statistical primitives implemented in **pure numpy** —
Spearman, t-test, Benjamini-Hochberg FDR, incomplete beta.

### Out-of-sample confirmation

After the IS measurements, the derived target `(h*, direction*, sell_pct*)` is
**replayed on the OOS tail** (the last 30% of the dataset, never touched by IS):

- OOS returns are oriented to the derived direction
- Win rate, lift, mean_advantage, and t-test are measured on the OOS window
- OOS is considered confirmed when: mean_advantage > 0 and p-value < 0.10
  (`oos_max_p`). There is no separate activation-count floor — the source
  comment is explicit that "p-value alone determines passed: sample size is
  already encoded in p" (a small OOS sample simply makes the p-value harder
  to clear).

OOS confirmation is a **non-blocking diagnostic**: a contract with weak OOS is
still promoted if it has a determined direction, but its grade reflects the
lack of confirmation. This avoids rejecting rare but structurally solid events
where the OOS window has too few activations.

### The A–D grade

The composite score (0–1) integrates the IS measurements. The default
weights (`AlphaConfig.score_weights`) are `(0.20, 0.25, 0.15, 0.25, 0.15)`
for `(ic, lift, cohens_d, z, breadth)` — a **five**-term formula:

```
score = 0.20 × IC_norm + 0.25 × lift_norm + 0.15 × d_norm
      + 0.25 × z_norm  + 0.15 × regime_breadth
```

`z_norm` is the normalised rotation-null standardised excess at `h*`
(`|z_h*|`, the edge-to-noise ratio computed above) — a term the naive
formula omits entirely. `d_norm` is **signed** in `[-1, 1]`: a negative
Cohen's d (the conditioned group performs worse than the background) actively
*penalises* the score rather than being clipped to zero.

Two further adjustments are applied after the weighted sum:

- if the selected horizon is `statistically_weak` (outside the BH-significant
  set), the composite is multiplied by `statistically_weak_penalty` (default
  `0.6`) — a horizon picked by the very selection bias the FDR control guards
  against cannot rank highly;
- if the OOS confirmation passes, `oos_bonus` (default `0.05`) is added.

The result is clamped to `[0, 1]`. Each raw component is normalised on a
0–1 scale before weighting. The grade:

| Grade | Score | Meaning |
|---|---|---|
| A | ≥ 0.75 | Strong statistical evidence |
| B | ≥ 0.50 | Solid evidence |
| C | ≥ 0.25 | Moderate evidence |
| D | < 0.25 | Weak evidence — potentially noise |

All four grades are forwarded to Rule Discovery. The grade is not a promotion
filter: it indicates the strength of the statistical evidence.

### Artefact: `AlphaContract`

```python
c = promoted[0]

# The target derived from the data
dt = c.derived_target
print(f"Horizon: {dt.holding_period_h}h")
print(f"Direction: {dt.direction}")        # "long" or "short"
print(f"sell_pct:  {dt.sell_pct:.4f}")    # e.g. 0.0312 = 3.12%

# IS statistical evidence
print(f"IC: {c.underlying_feature.ic:.4f}, p={c.underlying_feature.p_value:.4f}")
print(f"Lift: {c.event_stats.lift:.4f}")
print(f"Grade: {c.alpha_score.grade}")    # "A", "B", "C" or "D"

# OOS confirmation
oos = c.oos_validation
print(f"OOS passed: {oos.passed}, OOS lift: {oos.lift:.4f}")

# Status
print(f"Status: {c.status}")              # "HYPOTHESIS" or "REJECTED"
```

An `AlphaContract` with `status="HYPOTHESIS"` is the formal hypothesis that
the event has predictive power in the indicated direction at the indicated
horizon. It is not a certainty — it is a hypothesis to be tested operationally.

---

## 3. The Rule: Trading Realistically

### Why the alpha is not enough

An event with grade A, IC=0.08, lift=0.12, confirmed OOS — that is a
noteworthy result. But it does not answer the question that matters in
production: *can I make money trading this signal, with real fees, with limit
orders that might not get filled, on a horizon I must close even if the market
hasn't moved enough?*

Statistical evidence measures the separation of distributions. It does not
measure operational profitability. That is Rule Discovery's responsibility.

### The entry mechanics: a two-stage auto evaluation

Rule Discovery translates the `AlphaContract` into a backtest with realistic
order mechanics. Since issue #185, `RuleDiscoveryConfig.entry_mode` defaults
to `"auto"`, which runs the evaluation in **two stages** rather than
assuming a limit entry from the start:

**Stage 1 — market entry (authoritative for the verdict).** The rule is
backtested entering at the next bar's open (fill ≈ 100%). This isolates the
*signal's* edge from any entry-price optimisation, and its verdict is final:
Stage 2 can refine which parameters get published, but it can never turn a
`NON-EDGE` from Stage 1 into an edge.

**Stage 2 — optional limit-price sweep.** On a Stage-1 survivor, Rule
Discovery optionally sweeps `buy_drop_pct` (a limit order at
`fill_price = close × (1 − buy_drop_pct)`, valid for `buy_delay_bar` bars,
filled when price drops to that level in the subsequent bars using close as
the conservative approximation) and replays the winning candidate
out-of-sample. The limit price is **adopted** — published in place of the
market entry — only if it clears all three OOS conditions:

1. `fill_rate >= min_fill_rate_opt` — no PF inflated by rare fills.
2. `opportunity_sharpe >= market's` — a *per-trade-frequency* Sharpe, so a
   point that trades less often must earn more per trade to compensate.
3. `net_gain >= min_net_gain_retention × market's` — a backstop for the case
   the Sharpe cannot see (a tiny mu with a tiny sigma).

`RuleDiscoveryResponse.entry_optimization.failed_condition` names which of
the three stopped adoption (`"fill"` / `"sharpe"` / `"net_gain"`), or `None`
when adopted. This exists because the old limit-only default let a deep,
rarely-filled limit inflate profit factor on a non-representative subset of
trades (the "fill confound") — the entry doubled as both order mechanic and
entry-price optimiser, so the verdict ended up measuring the entry price
rather than the signal.

**Exit (either entry):**
- **Take-profit:** exit when the price rises to
  `take_profit = fill_price × (1 + sell_pct)`
- **Horizon stop:** if take-profit is not reached within `target_h` bars from
  fill, close at that bar's close
- **Short:** symmetric mirror — entry above anchor, take-profit below fill

**Fee:** deducted on both entry and exit on every trade (`fee_per_side`).

`entry_mode="limit"` — the pre-#185 default — is still fully supported: the
grid varies `buy_drop_pct` directly and the limit entry doubles as an
entry-price optimiser with no market-entry baseline. It remains the right
choice when the limit order *is* the strategy, not merely an execution
refinement. A third mode, `entry_mode="market"`, runs Stage 1 alone with no
entry optimiser at all.

### The parameter grid

Rule Discovery does not assume the optimal values of `buy_drop_pct`, `sell_pct`,
and `target_h`. Rather than a fixed menu of candidate values, `build_grid()`
constructs a small **symmetric fan** arithmetically around each contract-derived
base value:

```
buy_drop_pct = [d − 0.005, d − 0.002, d, d + 0.002, d + 0.005]   (floored at 0.001)
sell_pct     = [s − 0.02,  s − 0.01,  s, s + 0.01,  s + 0.02]    (floored at 0.005)
target_h     = {round(h × 0.5), round(h × 1.0), round(h × 2.0)}
buy_delay_bar = [base value]   — a single value, not swept, unless the caller
                                 sets GridSpec.buy_delay_bar explicitly
```

where `d`, `s`, and `h` are `buy_drop_pct`, `sell_pct`, and `target_h` taken
from the `AlphaContract`'s derived target (via `base.resolved()`). Any axis
the caller sets explicitly on `GridSpec` overrides this auto-fan.

For each configuration the composite score `pf_score_tpm` is computed,
balancing Profit Factor, trading frequency, and monthly consistency.
Configurations are screened against a dynamic trade-count floor,
`max(pf_min_trades, n_months × pf_min_tpm)` — `pf_min_trades` defaults to 15
and `pf_min_tpm` to 2 trades/month, so the floor grows with the length of the
selection span rather than staying fixed at a flat count. Configurations
below that floor, or with PF < 1, are discarded immediately.

The best configuration among those that pass the selection thresholds is
forwarded to statistical validation and walk-forward.

### Walk-forward validation

The IS backtest alone would still be optimistic: the best parameter set was
chosen on IS, so the IS result is upward-biased.

Walk-forward OOS splits the history into rolling windows (train + test):

```
  [────────────── train 1 ──────────────] [test 1]
        [────────────── train 2 ──────────────] [test 2]
              [────────────── train 3 ──────────────] [test 3]
```

On each train window the parameters are re-optimised. On the subsequent test
window the PF is measured with those parameters. `wf.consistency` is the
fraction of test windows with PF > 1 — the most direct measure of OOS
robustness.

### The EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA verdict

The final verdict integrates hard and soft gates:

**NON-EDGE (hard gate):**
- PF < 1.5 on IS (`partial_min_profit_factor`)
- All months with negative or zero result (too irregular)
- No configuration with adequate fill rate

**EDGE** requires all of:
- PF ≥ 2.0 on IS (`min_profit_factor`)
- Win rate ≥ 55%
- Deflated Sharpe ≥ 1.0 (penalised for the number of configurations tested)
- Regime dependency < 30% (edge is not concentrated in a single regime)
- Temporal stability: PF first half ≈ PF second half
- Walk-forward `consistency ≥ 0.5` (at least half the OOS test windows have
  PF > 1)
- The search-level rotation null is cleared: `rotation_p ≤ max_rotation_p`.
  This is the `fast_null`/`FastRotationNull` machinery `forge()` runs by
  default, pricing the multiple-testing surface of the whole discovery
  search — a contract that only wins that lottery is capped below full EDGE
  even if every gate above passes.

**PARTIAL-EDGE:** passes the hard (NON-EDGE) gates but not all EDGE gates.

**INSUFFICIENT-DATA:** a would-be `EDGE`/`PARTIAL-EDGE` verdict is downgraded
to `INSUFFICIENT-DATA` when the pooled out-of-sample evidence is too thin to
support a confident positive call (gated by `SelectionCriteria.power_gate`,
default `True`, via `_power_assessment()`). A `NON-EDGE` is never rescued
this way — underpowered or not, the operational consequence is the same. The
verdict is therefore one of **four** values:
`"EDGE" | "PARTIAL-EDGE" | "NON-EDGE" | "INSUFFICIENT-DATA"`
(`RuleDiscoveryResponse.verdict`).

### Artefact: `ValidatedRule` (inside `RuleDiscoveryResponse`)

```python
resp = rd.run()

print(f"Verdict: {resp.verdict}")    # "EDGE", "PARTIAL-EDGE", "NON-EDGE" or "INSUFFICIENT-DATA"
print(f"Is edge: {resp.is_edge}")    # True for EDGE and PARTIAL-EDGE

if resp.is_edge:
    vr = resp.validated_rule
    params = vr.params
    print(f"Entry: limit at -{params.buy_drop_pct:.2%} from close")
    print(f"Take-profit: +{params.sell_pct:.2%} from fill")
    print(f"Max horizon: {params.target_h} bars")
    print(f"Direction: {params.direction}")

# IS metrics
s = resp.in_sample_summary
print(f"Trades: {s.total_trades}, PF: {s.profit_factor:.2f}, WR: {s.win_rate_pct:.2%}")

# Walk-forward
wf = resp.walk_forward
print(f"OOS consistency: {wf.consistency:.0%}")  # e.g. 75% of windows with PF>1
```

A `ValidatedRule` contains precise operational parameters — not statistical
estimates, but values ready for an execution system.

---

## 4. From Contract to Trading Signal

### When the event reactivates on new data

Once the system has produced an `AlphaContract` (HYPOTHESIS) and a
`ValidatedRule` (EDGE or PARTIAL-EDGE), the operational logic is simple:

> When the event activates on new data → generate a trading signal.

```python
# New data arriving (live or batch)
new_kpi = fetch_latest_bars(asset="BTC", timeframe="1H")

# The signal is the event applied to the new data
signal = event_candidate.apply(new_kpi)    # pd.Series bool

# Bars with True signal are entry candidates
signal_bars = new_kpi[signal]
```

### The order parameters

The `ValidatedRule` translates the signal into precise order instructions:

```python
params = validated_rule.params

for ts, row in signal_bars.iterrows():
    # Compute the limit price (long)
    limit_price = row["close"] * (1 - params.buy_drop_pct)
    take_profit = limit_price * (1 + params.sell_pct)
    max_bars    = params.target_h   # close within this number of bars

    # Instruction to the execution system
    place_limit_order(
        direction   = params.direction,    # "long" or "short"
        limit_price = limit_price,
        take_profit = take_profit,
        max_bars    = max_bars,
        fee         = params.fee,
    )
```

Each parameter is the result of a structured empirical process:
- `direction` — derived by Alpha Discovery (sign of mean_advantage)
- `buy_drop_pct` — optimised by Rule Discovery on IS, validated on OOS
- `sell_pct` — rooted in the IS MFE distribution, refined by Rule Discovery
- `target_h` — rooted in the Alpha horizon, refined by Rule Discovery

### Filtering by regime

A statistically valid edge may be more robust in certain regimes.
`RuleDiscoveryResponse` exposes the per-regime analysis:

```python
ra = resp.regime_analysis
print(f"Regimes to avoid: {ra.avoid_in}")
print(f"Regime dependency score: {ra.dependency_score:.2f}")

# Filter the signal by regime
enriched = MarketContext(new_kpi).run()
regime_now = enriched["regime"].iloc[-1]

if regime_now not in ra.avoid_in:
    # Edge is present in this regime → proceed with the order
    pass
```

### The full flow from data to signal

```
Historical data (KPI Table)
       │
       ▼
[Module 0] Classify every bar by market regime
       │
       ▼
[Module 1] Discover events with stable temporal structure
           → EventCandidate  (distributional thresholds, immutable)
       │
       ▼
[Module 2] Measure predictive power (no prior assumptions)
           → AlphaContract   (horizon, direction, sell_pct derived from data)
       │
       ▼
[Module 3] Validate operationally with realistic order mechanics
           → ValidatedRule   (precise parameters for the execution system)
       │
       ▼
New data (live)
       │
       ▼
EventCandidate.apply()  →  boolean signal per bar
       │ True
       ▼
ValidatedRule.params    →  limit_price, take_profit, max_bars, direction
       │
       ▼
Execution system        →  order
```

---

## 5. Why This Separation Is Necessary

### The three formal boundaries

The FORGE pipeline enforces three boundaries that cannot be crossed in either
direction:

**Boundary 1: Module 1 does not see the forward return.**
If the event were discovered by looking at returns, the thresholds would be
calibrated to maximise PF — not to capture a real market structure. Immutable
thresholds are the direct consequence of this boundary.

**Boundary 2: Alpha Discovery takes no economic parameters as input.**
Horizon and direction are not user assumptions: they are empirical measurements.
If the user could specify "I believe this event is a long signal at 24 bars",
the measurement would confirm the hypothesis even when the data partially
contradicts it.

**Boundary 3: Rule Discovery evaluates on a rolling OOS, not on IS.**
The optimal operational parameter configuration found on IS would be
overfitted. Walk-forward separates optimisation from measurement on every
window.

### Immutable thresholds as a guarantee

The principle that event thresholds never change — even if a different threshold
would produce a higher PF — is the protection against the most subtle form of
look-ahead bias: optimising event thresholds *after* seeing returns.

Recalibrating them would require a new Event Discovery session, on a new IS
sample, with a new historical series.

### The derived target as measurement, not assumption

The fact that Alpha Discovery derives the horizon, direction, and sell_pct from
the data — instead of receiving them as parameters — means every `AlphaContract`
is an empirical answer, not the verification of a preconceived hypothesis.

A user might intuit that "low RSI is a bounce signal at 24 hours". FORGE
measures whether the event has a stronger advantage at 6, 12, 24, or 36 hours,
and selects the horizon with the best signal-to-horizon ratio — which may differ
from the intuition.

### OOS as a formal gate, not a post-hoc check

The OOS window in Alpha Discovery and the walk-forward in Rule Discovery are not
optional analyses run after promotion: they are formal conditions of the
pipeline. OOS data participates in no IS computation — it is an independent
observer that has never "seen" the thresholds, derived target, or optimal
operational configuration.

### Two layers that keep the boundaries honest across a session

Two mechanisms, both invisible in the individual module descriptions above,
sit underneath every `forge()` run and enforce the separation in practice:

**A central parameter resolver.** `forge()` builds a `PipelineContext` (the
session's single source of truth for timeframe, schema column names, fee,
and statistical policy — `forgedge.resolver`) and resolves every config
field a caller left unset against it, before Module 0 ever executes. The
resolved bundle is then checked for internal contradictions by
`config_report()` (`ConfigReport`): `forge(strict=True)` — the default —
raises a `ValueError` rather than running a session whose configuration
cannot structurally produce a verdict (e.g. an M3 selection window too short
for the arrival rate it was told to demand). A wall of silent rejections is
indistinguishable from "the signal is bad"; refusing to start is the honest
response.

**Rotation-null calibration of the search itself.** Every individual
`AlphaContract` is already standardised against its own circular-rotation
null (§2 above). By default `forge()` additionally runs a *search-level*
rotation null (`fast_null=True`, `calibration.fast_null.FastRotationNull`)
that prices the multiple-testing surface of the whole discovery session —
how many candidates were tried, not just how strong one candidate looks in
isolation — and annotates the result on `AlphaContract.rotation_p`. Rule
Discovery's EDGE gate reads this value directly (§3 above): a contract that
only wins the multiple-testing lottery is capped at `PARTIAL-EDGE` even if
every other gate passes.

---

## Summary of the three concepts

| Concept | Question | Module | Artefact | Gate |
|---|---|---|---|---|
| **Event** | "Is this market configuration stable and repeatable?" | Module 1 | `EventCandidate` | ConsistencyGate (structural, no forward return) |
| **Alpha** | "Does this event statistically predict an oriented return?" | Module 2 | `AlphaContract` | direction ≠ "undetermined" (single hard gate) |
| **Rule** | "Is this alpha profitable under realistic order mechanics?" | Module 3 | `ValidatedRule` | EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA |

A trading signal emerges from the intersection of these three verifications: a
structurally stable market configuration (`Event`), with empirical evidence of
predictive power (`Alpha`), that holds under realistic operational conditions
(`Rule`).

---

## References

| File | Contents |
|---|---|
| `index_en.md` | System overview and quick start |
| `how_to_use_en.md` | Practical end-to-end guide with full configuration |
| `configuration_en.md` | Global configuration reference — every config dataclass and field |
| `modulo_0_en.md` | Market Context: regime classification |
| `modulo_1_en.md` | Event Discovery: pipeline, ConsistencyGate, EventCandidate |
| `modulo_2_en.md` | Alpha Discovery: target derivation, IC, OOS, AlphaContract |
| `modulo_3_en.md` | Rule Discovery: backtest, EDGE verdict, walk-forward, reports |
| `modulo_4_en.md` | Rule Registry: dedup, cross-ticker replay, genericity |
