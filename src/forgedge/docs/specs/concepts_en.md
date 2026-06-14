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
The selection criteria are purely **structural**:

| Criterion | Meaning |
|---|---|
| Minimum frequency | The event must fire often enough to allow robust statistical measurements (≥ 30–50 IS activations) |
| Temporal distribution | Activations must be spread over time — not concentrated in a single month or year |
| Maximum concentration | No single month may contain more than 40% of total activations |
| Monthly frequency | At least 2 activations per month on average |

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

For each event, Alpha Discovery scans a grid of horizons
(1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48 bars) and computes:

```
score[h] = |mean_advantage[h]| / sqrt(h)
```

where `mean_advantage[h]` is the difference between the mean return of active
bars and the mean return of inactive bars at horizon h.

The `|mean_advantage| / √h` criterion balances the size of the advantage
against the growing variance of longer horizons — a 0.5% advantage at 4 bars
is more meaningful than the same advantage at 36 bars, where the return
distribution is more dispersed. The selected horizon `h*` is the one with the
highest score.

**Direction derivation:**

If `mean_advantage[h*] > 0`, the direction is `"long"` — the event precedes
positive returns. If `< 0`, it is `"short"`. If non-finite across all horizons,
`"undetermined"` — the event is rejected.

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
- OOS is considered confirmed when: n_activations ≥ 10, mean_advantage > 0,
  p-value < 0.10

OOS confirmation is a **non-blocking diagnostic**: a contract with weak OOS is
still promoted if it has a determined direction, but its grade reflects the
lack of confirmation. This avoids rejecting rare but structurally solid events
where the OOS window has too few activations.

### The A–D grade

The composite score (0–1) integrates the IS measurements:

```
score = 0.25 × IC_norm + 0.30 × lift_norm + 0.25 × d_norm + 0.20 × regime_breadth
```

where each component is normalised on a 0–1 scale. The grade:

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

### The limit order mechanics

Rule Discovery translates the `AlphaContract` into a backtest with realistic
order mechanics:

**Entry (long):**
- When the event activates, a limit order is placed at
  `fill_price = close × (1 − buy_drop_pct)`
- The order is valid for `buy_delay_bar` bars
- A fill occurs when the price drops to the limit level in the subsequent bars
  (using close as the conservative approximation)

**Exit:**
- **Take-profit:** exit when the price rises to
  `take_profit = fill_price × (1 + sell_pct)`
- **Horizon stop:** if take-profit is not reached within `target_h` bars from
  fill, close at that bar's close
- **Short:** symmetric mirror — entry above anchor, take-profit below fill

**Fee:** deducted on both entry and exit on every trade (`fee_per_side`).

### The parameter grid

Rule Discovery does not assume the optimal values of `buy_drop_pct`, `sell_pct`,
and `target_h`. It explores a grid centred on the values derived from the
`AlphaContract`:

```
Grid: buy_drop_pct × sell_pct × target_h × buy_delay_bar
      ─────────────────────────────────────────────────────
      [0.5%, 1.0%, 1.5%, 2.0%] × [0.03, 0.04, 0.05, 0.06] × [12, 24, 36, 48] × [3, 6]
```

For each configuration the composite score `pf_score_tpm` is computed,
balancing Profit Factor, trading frequency, and monthly consistency.
Configurations with fewer than 20 trades or PF < 1 are discarded immediately.

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

### The EDGE / PARTIAL-EDGE / NON-EDGE verdict

The final verdict integrates hard and soft gates:

**NON-EDGE (hard gate):**
- PF < 1.5 on IS (`partial_min_profit_factor`)
- All months with negative or zero result (too irregular)
- No configuration with adequate fill rate

**EDGE:**
- PF ≥ 2.0 on IS (`min_profit_factor`)
- Win rate ≥ 55%
- Deflated Sharpe ≥ 1.0 (penalised for the number of configurations tested)
- Regime dependency < 30% (edge is not concentrated in a single regime)
- Temporal stability: PF first half ≈ PF second half

**PARTIAL-EDGE:** passes the hard gates but not all EDGE gates.

### Artefact: `ValidatedRule` (inside `RuleDiscoveryResponse`)

```python
resp = rd.run()

print(f"Verdict: {resp.verdict}")    # "EDGE", "PARTIAL-EDGE" or "NON-EDGE"
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
print(f"Strong regimes: {ra.strong_in}")

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

---

## Summary of the three concepts

| Concept | Question | Module | Artefact | Gate |
|---|---|---|---|---|
| **Event** | "Is this market configuration stable and repeatable?" | Module 1 | `EventCandidate` | ConsistencyGate (structural, no forward return) |
| **Alpha** | "Does this event statistically predict an oriented return?" | Module 2 | `AlphaContract` | direction ≠ "undetermined" (single hard gate) |
| **Rule** | "Is this alpha profitable under realistic order mechanics?" | Module 3 | `ValidatedRule` | EDGE / PARTIAL-EDGE / NON-EDGE |

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
| `modulo_1_en.md` | Event Discovery: pipeline, ConsistencyGate, EventCandidate |
| `modulo_2_en.md` | Alpha Discovery: target derivation, IC, OOS, AlphaContract |
| `modulo_3_en.md` | Rule Discovery: backtest, EDGE verdict, walk-forward, reports |
