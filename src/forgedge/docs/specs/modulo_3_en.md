# Module 3 — Rule Discovery

Rule Discovery is the fourth module in the FORGE pipeline. It receives an
`AlphaContract` promoted by Alpha Discovery and answers the operational question
the contract leaves open: **does the statistically identified pattern survive a
realistic backtest — with fees, finite fill rate, limit orders and a discrete
target — and hold up out of sample?**

The output is a `RuleDiscoveryResponse` with a `EDGE` / `PARTIAL-EDGE` /
`NON-EDGE` verdict and, in the first two cases, a `ValidatedRule` with the
validated operational parameters.

---

## Basic usage

```python
from forgedge import (
    MarketContext, EventDiscovery, AlphaDiscovery, AlphaConfig,
    RuleDiscovery, RuleDiscoveryConfig,
)

enriched   = MarketContext(kpi).run()
ed         = EventDiscovery(enriched)
candidates = ed.run()

ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="BTC", timeframe="1H"))
contracts  = ad.run()
promoted   = ad.promoted_contracts()

# Index candidates by ID (required by RuleDiscovery)
by_id = {c.event_id: c for c in candidates}

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        vr = resp.validated_rule
        print(f"  entry: {vr.params.buy_type}  drop={vr.params.buy_drop_pct}"
              f"  sell={vr.params.sell_pct}  h={vr.params.target_h}")
```

---

## Position in the pipeline

```
list[AlphaContract] promoted (Module 2)
EventCandidate counterparts (Module 1)
        │
        ▼
  RuleDiscovery.run()  [per contract]
        │
        ▼
  RuleDiscoveryResponse
  ├─ verdict: EDGE | PARTIAL-EDGE | NON-EDGE
  ├─ validated_rule   (if EDGE or PARTIAL-EDGE)
  ├─ in_sample_summary
  ├─ execution_envelope + excursion (MAE/MFE)
  ├─ walk_forward OOS
  ├─ statistical_validation
  └─ regime_analysis
        │
        ▼
  Rule Registry (not implemented)
```

Rule Discovery is **the only module that uses prices for an execution
simulation**. It does not re-optimise event thresholds or modify the derived
target parameters: it uses the event expression and
`derived_target.sell_pct` / `derived_target.holding_period_h` as the seed
for the operational grid.

---

## Five-step pipeline

### Step 1 — Setup

**Signal reconstruction.** The boolean event is reconstructed deterministically
on the DataFrame via `EventCandidate.apply()` (or from the stored `event_series`)
and injected as the `__rule_signal__` column.

**Seed from contract parameters.** With `use_contract_target=True` (default),
the starting backtest parameters are initialised from:
- `target_h = derived_target.holding_period_h`
- `sell_pct = max(0.01, derived_target.sell_pct)` (clamped to an operational floor)

These become the centre of the screening grid, not fixed parameters.

**Direction propagation.** The engine supports both `"long"` and `"short"`.
The direction is read from `derived_target.direction` and set on
`BacktestParams.direction`. The short side is the exact symmetric mirror of
long: the limit entry sits *above* the anchor at `anchor × (1 + buy_drop_pct)`,
fills when `high` reaches it, and the take-profit is *below* the fill price at
`fill × (1 − sell_pct)`, reached when the price falls to it. Net gain is
`(fill − exit) / fill`. Only values other than `"long"` or `"short"` produce
an immediate `NON-EDGE`.

---

### Step 2 — In-sample grid screening

The operational grid explores the Cartesian product of:

| Dimension | Controlled by |
|---|---|
| `buy_drop_pct` | `GridSpec.buy_drop_pct` |
| `sell_pct` | `GridSpec.sell_pct` |
| `target_h` | `GridSpec.target_h` |
| `buy_delay_bar` | `GridSpec.buy_delay_bar` |

When `GridSpec` is empty (default), FORGE automatically builds a sensible grid
centred on the values derived from the contract.

**Execution mechanics for each configuration:**

1. On signal, a limit order is placed at `anchor * (1 - buy_drop_pct)`
2. If the price touches the limit within `buy_delay_bar` bars, the order is
   filled. Otherwise it is cancelled.
3. After fill, the position is closed:
   - at the first bar that closes at ≥ `sell_price = fill_price * (1 + sell_pct)`, or
   - at the close of bar `target_h` (horizon stop)

> **`target_h` counts bars *after* the fill bar, not the total signal→exit span.**
> The signal→fill gap is always 1 bar (point-in-time correctness — you cannot
> act on a bar's own close before it happens), so the total span from signal to
> exit is `1 + target_h`. "Hold for N bars from entry" therefore maps to
> `target_h = N - 1`. `target_h = 0` is a legal, meaningful value — it exits at
> the fill bar's own close (a same-session round-trip) — not a placeholder for
> "no horizon".

Each configuration is evaluated via the composite score `pf_score_tpm`, which
balances Profit Factor, trading frequency and monthly consistency.

**Early elimination (Step 2.3):** configurations with < 20 trades, PF < 1.0,
or `fill_rate < min_fill_rate` are discarded before the expensive validation
(walk-forward + diagnostics). With `SelectionCriteria(early_elimination=False)`
the full pipeline runs even for these configurations: the verdict is still
`NON-EDGE`, but walk-forward and all diagnostics are populated — useful for
uniform reporting or to observe the OOS behaviour of weak rules.

---

### Step 3 — Selection and refinement

The best configuration is the one with the maximum `pf_score_tpm` among those
that meet the `SelectionCriteria` thresholds. If no configuration is selectable,
the verdict is `NON-EDGE`.

---

### Step 4 — Statistical validation

On the selected configuration, on the IS period:

| Metric | Description |
|---|---|
| `ttest_winrate_t/p` | t-test win rate vs contract base rate |
| `ttest_expectancy_t/p` | t-test expectancy vs zero |
| `sharpe_ratio` | Annualised Sharpe |
| `deflated_sharpe` | Sharpe deflated by n_trials (penalises data snooping) |
| `temporal_stability` | `"PASS"` / `"WARN"` / `"FAIL"`: PF first half vs second |
| `n_trials_tested` | Number of configurations tested in the grid |

#### Walk-forward OOS

The timeline is divided into `n_splits` consecutive test windows. For each split:
1. The grid is re-screened on the preceding train window (with `reoptimise=True`)
2. The best parameters are applied once on the test window
3. Trades from the test windows are concatenated → honest OOS track record

`WalkForwardResult.oos_summary` contains the aggregate metrics on OOS trades.
`WalkForwardResult.consistency` = fraction of windows with positive net gain.

---

### Step 5 — Regime dependency

For each regime present in the DataFrame, performance metrics are computed on
IS trades: PF, win rate, expectancy, cumulative net gain.

`dependency_score`: normalised entropy of the monthly trade distribution
(0 = trades distributed evenly across months; 1 = all trades concentrated
in a single month).

`avoid_in`: regimes with ≥ 5 trades and PF < 1.0 — regimes to avoid in production.

---

## Verdict: EDGE / PARTIAL-EDGE / NON-EDGE

### `NON-EDGE` gates (hard — immediate rejection)

| Condition | Parameter |
|---|---|
| IS PF < `partial_min_profit_factor` (1.5) | — |
| IS trades < `max(10, n_months × min_tpm)` (dynamic floor) | — |
| t-test expectancy p ≥ `max_ttest_p` (0.05) | — |
| Walk-forward OOS PF < 1.0 | — |

If any one is violated: `NON-EDGE`.

### `EDGE` gates (all required for a full verdict)

| Condition | Parameter |
|---|---|
| IS PF ≥ `min_profit_factor` (2.0) | — |
| IS win rate ≥ `min_win_rate` (0.55) | — |
| `zero_months` ≤ `max_zero_months_edge` (1) | — |
| DSR ≥ `min_dsr` (1.0) | — |
| `temporal_stability` ≠ `"FAIL"` | — |
| `dependency_score` ≤ `max_regime_dependency` (0.30) | — |
| OOS consistency ≥ 0.50 | — |

If all satisfied: `EDGE`. If only the NON-EDGE gates are satisfied but not
all EDGE gates: `PARTIAL-EDGE`.

```python
resp.verdict         # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
resp.is_edge         # True if EDGE or PARTIAL-EDGE
resp.rejection_reasons  # failed gates (empty list if EDGE)
```

---

## Execution Envelope and MAE/MFE

### `ExecutionEnvelope`

The same configuration is backtested with two different exit conventions,
defining a realistic execution range:

| Variant | `target_hit_col` | Description |
|---|---|---|
| `conservative` | `"close"` | Target counts only when a bar **closes** past sell_price. Understates (real limit sell could fill intrabar). Matches the certified reference engine for both directions. |
| `optimistic` | `"high"` (long) / `"low"` (short) | Target counts on first intrabar touch. Overstates (assumes limit sell always fills). |

Real performance lies between the two. Use `optimistic_hit_col` to resolve the
correct column for a given direction:

```python
from forgedge.rule_discovery import optimistic_hit_col

col = optimistic_hit_col("short")  # → "low"
col = optimistic_hit_col("long")   # → "high"

env = resp.execution_envelope
print(f"conservative PF: {env.conservative.profit_factor:.2f}")
print(f"optimistic   PF: {env.optimistic.profit_factor:.2f}")
```

### `ExcursionStats` (MAE/MFE)

For every executed trade, over the window `[fill+1 .. exit]`:
- **MAE** (Maximum Adverse Excursion): maximum drawdown reached,
  `(min_low - buy_price) / buy_price` (negative)
- **MFE** (Maximum Favourable Excursion): maximum run-up reached,
  `(max_high - buy_price) / buy_price` (positive)

```python
ex = resp.excursion
print(f"Mean MAE: {ex.mae_mean:.4f}, worst: {ex.mae_worst:.4f}")
print(f"Mean MFE: {ex.mfe_mean:.4f}, reached target: {ex.mfe_reached_target_pct:.1f}%")
```

---

## Data structure: `RuleDiscoveryResponse`

```python
resp = rd.run()

resp.verdict             # str: "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
resp.is_edge             # bool: True if EDGE or PARTIAL-EDGE
resp.alpha_id            # str: source contract ID
resp.asset, resp.timeframe  # str

# Validated rule (None if NON-EDGE)
resp.validated_rule.expression       # boolean expression
resp.validated_rule.params           # BacktestParams with optimal configuration
resp.validated_rule.event_candidate_id

# IS metrics
resp.in_sample_summary.total_trades
resp.in_sample_summary.profit_factor
resp.in_sample_summary.win_rate_pct
resp.in_sample_summary.expectancy
resp.in_sample_summary.tpm_mu        # average trades/month
resp.in_sample_summary.zero_months   # months with no trades

# Execution envelope + MAE/MFE
resp.execution_envelope   # ExecutionEnvelope | None
resp.excursion            # ExcursionStats | None

# Walk-forward OOS
resp.walk_forward.oos_summary.profit_factor
resp.walk_forward.consistency          # fraction of windows with net gain > 0
resp.walk_forward.n_profitable_splits
resp.walk_forward.oos_envelope         # ExecutionEnvelope OOS
resp.walk_forward.oos_excursion        # ExcursionStats OOS

# Statistical validation
resp.statistical_validation.deflated_sharpe
resp.statistical_validation.ttest_expectancy_p
resp.statistical_validation.temporal_stability  # "PASS"/"WARN"/"FAIL"

# Regime
resp.regime_analysis.dependency_score
resp.regime_analysis.avoid_in          # list[str] regimes to avoid
resp.regime_analysis.per_regime        # list[dict]

# Audit
resp.rejection_reasons   # list[str]
resp.notes               # list[str]
resp.grid_results        # list[GridResult]
```

### `BacktestSummary` — full field reference

| Field | Description |
|---|---|
| `total_signals` | Total signals in the dataset |
| `total_trades` | Executed trades (signal filled) |
| `fill_rate` | `total_trades / total_signals` |
| `win_rate_pct` | Win rate (0–1) |
| `winning_trades`, `losing_trades` | Counts |
| `total_net_gain` | Sum of net returns (fees included) |
| `expectancy` | `total_net_gain / total_trades` |
| `std_net_gain` | Standard deviation of net returns |
| `profit_factor` | `sum(wins) / sum(losses)` |
| `best_trade`, `worst_trade` | Best/worst individual return |
| `target_hit_rate_pct` | % of trades that reached the target |
| `n_months` | Total months in the dataset |
| `active_months` | Months with at least one trade |
| `zero_months` | `n_months - active_months` |
| `tpm_mu`, `tpm_sigma` | Mean and std dev of trades/month |
| `c_norm` | Monthly regularity: `min(1, 1 / max(index_of_dispersion, 1))`. Scale-free — a Poisson process scores 1 at *any* rate, and only variance in excess of Poisson is penalised (#178). |
| `pf_score`, `pf_score_tpm` | Composite score: `pf × sigmoid(trade count)` and `pf × c_norm` |
| `exp_score_tpm` | Same regularity penalty applied to expectancy: `expectancy × c_norm` |
| `sharpe_raw` | Raw Sharpe (not annualised) |

---

## Output methods

### `rd.run() → RuleDiscoveryResponse`

Runs the full pipeline. Must be called before any other method.

### `rd.grid_summary() → pd.DataFrame`

Flat DataFrame of all configurations tested in the IS grid, sortable by
`pf_score_tpm`:

```python
grid_df = rd.grid_summary()
print(grid_df.sort_values("pf_score_tpm", ascending=False).head())
# Columns: buy_drop_pct, sell_pct, target_h, buy_delay_bar,
#          profit_factor, win_rate_pct, total_trades, expectancy,
#          tpm_mu, fill_rate, zero_months, pf_score_tpm
```

### `text_report(resp) → str` and `html_report(resp) → str`

```python
from forgedge.rule_discovery import text_report, html_report

# Compact text report
print(text_report(resp))

# Self-contained HTML report (no CDN, works offline)
with open(f"{resp.alpha_id}.html", "w") as f:
    f.write(html_report(resp))
```

The HTML includes: verdict section, validated rule parameters, IS metrics,
execution envelope, MAE/MFE, walk-forward per split, statistical validation,
regime breakdown. Fully implemented with inline CSS and no external dependencies.

### `resp.to_dict() → dict`

Fully nested dictionary for YAML/JSON serialisation:

```python
import json
with open(f"{resp.alpha_id}_rule_discovery.json", "w") as f:
    json.dump(resp.to_dict(), f, indent=2)
```

### `resp.persist(path)`

Saves the full `RuleDiscoveryResponse` to disk as a pickle file. Useful for
archiving the complete rule — verdict, metrics, trade log, walk-forward results —
without re-running the backtest.

```python
import pickle, pathlib

pathlib.Path("rules").mkdir(exist_ok=True)
for contract, resp in rule_responses:
    if resp.is_edge:
        resp.persist(f"rules/{resp.alpha_id}.pkl")

# Reload in a later session
resp = pickle.load(open("rules/ALPHA-BTC-1H-000.pkl", "rb"))
print(resp.verdict, resp.in_sample_summary.profit_factor)
```

---

## Full configuration reference

### `BacktestParams`

| Parameter | Default | Description |
|---|---|---|
| `direction` | `"long"` | `"long"` or `"short"`. Short is the exact mirror of long: entry above anchor, take-profit below fill |
| `buy_type` | `"limit"` | `"limit"` or `"market"` |
| `buy_drop_pct` | `0.010` | Magnitude of the limit offset from anchor (e.g. 0.01 = 1%): a discount for long, a premium for short |
| `buy_delay_bar` | `6` | Lifetime of the limit order in bars |
| `buy_price_anchor` | `"close"` *(session-resolved)* | Column the limit offset is applied to — any numeric column, including a derived indicator (`"close_sma_3"` with `buy_drop_pct=0.10` = a limit at 90% of the 3-bar SMA). Filled in from `close_col` so a renamed price column carries the default anchor along; an explicit anchor is a level of its own |
| `sell_pct` | `0.040` | Take-profit as a fraction of fill price |
| `target_h` | `24` | Bars held *after* the fill bar before the close-at-horizon exit (signal→exit span is always `1 + target_h`). `0` = same-session round-trip (fill bar's own close) |
| `target_col` | `"close"` *(session-resolved)* | Column used for the horizon exit. Must name the same series as `close_col` |
| `target_hit_col` | `"close"` | Column used to detect the take-profit. Conservative = `"close"` for both directions; optimistic = `"high"` for long, `"low"` for short (use `optimistic_hit_col(direction)`) |
| `fee` | `0.002` *(session-resolved)* | Fee per side, derived from `AlphaConfig.fee_per_side` — the contract's cost basis is the cost charged |
| `early_stopping` | `True` | Exit at take-profit; if False, always exit at horizon |

### `GridSpec`

| Parameter | Default | Description |
|---|---|---|
| `buy_drop_pct` | `None` | List of discounts to test (None = auto) |
| `sell_pct` | `None` | List of targets to test (None = auto) |
| `target_h` | `None` | List of horizons to test (None = auto) |
| `buy_delay_bar` | `None` | List of delays to test (None = auto) |

### `RuleWalkForwardConfig` (Rule Discovery)

| Parameter | Default | Description |
|---|---|---|
| `n_splits` | `4` | Number of OOS test windows |
| `train_span_months` | `None` | Train months (None = anchored, grows) |
| `test_span_months` | `None` | Test months (None = divided equally) |
| `min_train_months` | `6` | Minimum train before the first test window |
| `reoptimise` | `True` | Re-optimise grid on each train window |

### `SelectionCriteria`

| Parameter | Default | Description |
|---|---|---|
| `min_profit_factor` | `2.0` | Minimum PF for EDGE |
| `min_win_rate` | `0.55` | Minimum win rate (0–1) |
| `min_tpm` | `2.0` | Minimum trades/month — also sets the dynamic trade floor `max(10, n_months × min_tpm)` |
| `min_pf_score_tpm` | `0.30` | Minimum composite score |
| `min_fill_rate` | `0.40` | Minimum fill rate |
| `partial_min_profit_factor` | `1.5` | Minimum PF for PARTIAL-EDGE |
| `max_zero_months_edge` | `1` | Maximum zero months for EDGE |
| `max_zero_months_partial` | `4` | Maximum zero months for PARTIAL-EDGE |
| `max_regime_dependency` | `0.30` | Maximum regime dependency score for EDGE |
| `min_dsr` | `1.0` | Minimum Deflated Sharpe for EDGE |
| `max_ttest_p` | `0.05` | Maximum p-value for expectancy t-test |
| `early_elimination` | `True` | When `True`, rules that fail the fast in-sample screen (Step 2.3) are rejected before walk-forward and diagnostics; when `False`, the full pipeline runs — verdict is still `NON-EDGE` but with all diagnostics populated |

### `RuleDiscoveryConfig`

| Parameter | Default | Description |
|---|---|---|
| `base_params` | `BacktestParams()` | Fixed parameters and grid centre |
| `scoring` | `ScoringParams()` | Composite score parameters |
| `grid` | `GridSpec()` | Operational grid (empty = auto) |
| `walk_forward` | `RuleWalkForwardConfig()` | Walk-forward OOS settings |
| `criteria` | `SelectionCriteria()` | Selection and verdict thresholds |
| `use_contract_target` | `True` | Seed sell_pct/target_h from the contract |
| `timestamp_col` | `"open_dt"` | Datetime column name |
| `signal_col` | `"__rule_signal__"` | Name of the injected signal column |
| `discovery_date` | `None` | ISO date for the response (None = today) |

---

## Advanced usage patterns

### Custom grid

```python
from forgedge.rule_discovery import GridSpec

config = RuleDiscoveryConfig(
    grid=GridSpec(
        buy_drop_pct=[0.005, 0.010, 0.015, 0.020],
        sell_pct=[0.03, 0.04, 0.05, 0.06],
        target_h=[12, 24, 36, 48],
        buy_delay_bar=[3, 6, 12],
    )
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
```

### Full grid analysis

```python
# All tested configurations, sorted by score
grid_df = rd.grid_summary()
top10 = grid_df.nlargest(10, "pf_score_tpm")
print(top10[["sell_pct", "buy_drop_pct", "target_h", "profit_factor",
             "win_rate_pct", "tpm_mu", "pf_score_tpm"]])
```

### Export reports for all PARTIAL-EDGE+

```python
from forgedge.rule_discovery import html_report, text_report

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    if resp.is_edge:
        # HTML for human review
        with open(f"reports/{resp.alpha_id}.html", "w") as f:
            f.write(html_report(resp))
        # JSON for integration
        import json
        with open(f"reports/{resp.alpha_id}.json", "w") as f:
            json.dump(resp.to_dict(), f, indent=2)
    else:
        print(f"NON-EDGE {resp.alpha_id}: {resp.rejection_reasons}")
```

### Linking the response to the contract

`AlphaContract.rule_discovery_response` is the field reserved for Rule Discovery:

```python
import json
contract.rule_discovery_response = resp.to_dict()
# The contract now carries the operational response for the Rule Registry
```

---

## Operational notes

- **`target_hit_col`:** the default `"close"` reproduces the certified reference
  engine (conservative). For the optimistic intrabar convention, use
  `optimistic_hit_col(direction)` — `"high"` for long, `"low"` for short.
- **Rule Registry:** Module 4 not yet implemented. The `ValidatedRule` produced
  by Rule Discovery is already in the form ready for ingestion into the registry.
