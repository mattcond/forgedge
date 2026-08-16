# FORGE — End-to-End Production Pipeline Guide

This guide walks through building and configuring a complete FORGE pipeline,
from a raw KPI Table to alpha contracts ready for Rule Discovery. It covers
data preparation, per-module production configuration, result interpretation,
and recurring patterns for multi-asset workflows.

---

## 1. Input data requirements

### KPI Table format

FORGE takes a `pd.DataFrame` with:

- **`close` column** — closing price, required by all three modules
- **Datetime column** — default `open_dt`; accepted as a column or as the
  DatetimeIndex name. Must be chronologically sortable.
- **Feature columns** — any technical indicator (RSI, EMA, volume, spread,
  ratio, oscillators). FORGE automatically classifies each column by type
  (continuous numeric, discrete numeric, categorical).

```python
# Minimum accepted schema
# Index: DatetimeIndex  or  'open_dt' column with datetime64 dtype
# Required columns: 'close'
# Optional columns: any technical feature

import pandas as pd

kpi = pd.DataFrame({
    "open_dt": pd.date_range("2022-01-01", periods=5000, freq="1h"),
    "close":   [...],          # float, required
    "rsi_14":  [...],          # example feature
    "volume":  [...],          # example feature
    "macd":    [...],          # example feature
})
```

### Building a KPI Table from raw candles: `kpi_builder`

If you only have raw OHLCV candles (no indicators yet), `forgedge.kpi_builder`
turns them into a FORGE-ready KPI Table in three steps:

```python
from forgedge import build_features, candle_features, lag_features, forge_preset, forge

# 1. Base indicators (SMA, EMA, RSI, Bollinger, rolling min/max, returns, vol, MDD)
#    from a dict or YAML config; also derives 'open_dt' and sorts chronologically
kpi = build_features(
    candles,                  # raw OHLCV, any timestamp column
    timestamp_col="open_time",  # required keyword arg
)

# 2. Candlestick geometry — continuous, scale-free (body, wicks, close_pos, range_pct, gap)
kpi = candle_features(kpi)

# 3. Lagged copies of selected columns (or all columns matching a substring)
kpi = lag_features(kpi, "close", "color", like="_ema_", periods=[1, 2, 3])

disc, alpha, rd = forge_preset("balanced", timeframe="1D", asset="DEMO")
result = forge(kpi, event_discovery_config=disc, alpha_config=alpha, rule_discovery_config=rd)
```

`build_features(candles, config=None, *, timestamp_col, output_timestamp_col="open_dt", timestamp_unit="ms", add_color=True, sort_output=True)`
accepts `config` as a `dict`, a path to a YAML file, or `None` for the packaged
default (`DEFAULT_CONFIG` / `default_enricher.yaml` — copy and edit it to
change periods/columns/enabled indicators). Indicators referencing columns
absent from `candles` are skipped with a warning, so OHLC-only input is safe.
Two indicators ship **disabled by default** (enable them explicitly in
`config`): `"atr"` (`periods=[14, 28]` → `close_atr_14`/`close_atr_28` plus a
normalised `close_natr_14`/`close_natr_28` companion, needs `high`/`low`) and
`"macd"` (`periods=[12, 26, 9]`, a flat `(fast, slow, signal)` triple → e.g.
`close_macd_12_26`, `close_macd_12_26_signal_09`, `close_macd_12_26_hist_09`).

**Column naming convention.** For a custom column (yours or a built one) to be
recognised as part of a same-family ratio pair by Event Discovery, its name
must match `{base}_{indicator}_{period}` with `base` in `{close, high, low,
open, volume}` and `indicator` in `{ema, sma, rsi, dema, tema, wma, hma, mdd,
atr, natr}` (or the dedicated `{base}_bb_{lower|upper|width|mid}_{period}` /
`{base}_{vol|ret}_{period}` patterns). Columns that don't match this pattern
still work as standalone features but are not paired into ratios; `mdd`/`atr`
support was added after an early bug silently dropped custom columns using
those suffixes — if a feature you added doesn't show up in `EventDiscovery`
candidates, check its name against this convention first.

`candle_features(df, *, order_on="open_dt", add_gap=True, round_to=5)` adds six
scale-free geometry columns (`body`, `upper_wick`, `lower_wick`, `close_pos`,
`range_pct`, `gap`) in `[-1, 1]`/`[0, 1]`, so FORGE derives its own
asset-adaptive thresholds instead of relying on fixed pattern definitions.

`lag_features(df, *cols, periods=(1, 2, 3), like=None, order_on="open_dt")`
appends `{col}_prev_{w:02d}` columns; pass explicit column names and/or
`like="_ema_"` to lag every column containing that substring.

All three functions sort by `order_on` (default `open_dt`) internally, so
input does not need to be pre-sorted — but a `pattern_features()` call (below)
on data lacking both `open_dt` and a `DatetimeIndex` raises `KeyError` if any
multi-bar pattern is requested.

**Optional, opt-in:** `pattern_features(df, *, patterns=None, order_on="open_dt", col="candle_pattern")`
adds a single categorical `candle_pattern` column (e.g. `"HAMMER"`, `"DOJI"`,
or `None`) from ten named candlestick formations. It flows end-to-end through
`forge()` — Event Discovery one-hots it into `== "HAMMER"` events and Alpha
Discovery scores them via point-biserial IC — but it stays opt-in **on quality
grounds, not technical ones**: named patterns encode fixed human thresholds,
whereas `candle_features()`'s continuous geometry lets FORGE derive its own
asset-adaptive thresholds, which is preferred for automatic discovery.

### Validating data quality: `summary_report`

Before feeding a table to `forge()`, run a cheap diagnostic pass over the
price columns — it never raises or blocks the pipeline, it only reports:

```python
from forgedge import summary_report

summary_report(kpi)   # prints a multi-section report to stdout

rep = summary_report(kpi, return_report=True, verbose=False)
if rep.has_critical:
    raise ValueError(f"Fix data issues first: {rep.one_line()}")
elif rep.has_warnings:
    print(rep.one_line())
```

`summary_report(df, *, timestamp_col="open_dt", price_cols=("open", "high", "low", "close"), timeframe=None, return_high_move=0.5, top_n=5, verbose=True, return_report=False)`
checks schema/NaNs/infinities, price-scale consistency (mixed magnitudes),
OHLC internal consistency (`high >= low`, `close` within `[low, high]`, …),
return outliers (MAD z-score and an absolute `return_high_move` threshold) and
time continuity (gaps, duplicate timestamps, out-of-order bars). Each
`Finding` carries a `level` (`"OK"` / `"WARN"` / `"FAIL"`), a stable `code`
(e.g. `"scale_mixed"`, `"ohlc_hl"`, `"gaps"`) and a human-readable `message`.
The returned `DataQualityReport` exposes `.worst`, `.has_critical`,
`.has_warnings`, `.one_line()` and `.to_text()`; `.findings` is the full list
for programmatic filtering.

### EMA naming convention

Market Context looks for EMA columns in the table using the pattern
`{source_col}_ema_{period:02d}` (e.g. `close_ema_09`, `close_ema_25`). If the
columns exist they are used directly; otherwise they are computed inline. If
your table already produces these columns (e.g. via a framework like
`CandleKPI`), Module 0 is zero-copy for the EMAs.

### Recommended data volume

| Timeframe | Recommended minimum | Optimal |
|---|---|---|
| 1H | 6 months (≈4,000 bars) | 2 years (≈17,000 bars) |
| 4H | 1 year (≈2,200 bars) | 3 years (≈6,500 bars) |
| 1D | 2 years (≈730 bars) | 5 years (≈1,800 bars) |

More data improves both the OU half-life estimate (Module 0) and the
ConsistencyGate stability (Module 1). Below 2,000 bars the Module 1 gates may
be too strict for thinly traded assets.

---

## 2. Module 0 — Market Context

### Minimal configuration (production)

```python
from forgedge import MarketContext

enriched = MarketContext(kpi).run()
# adds 'regime' (ordered Categorical) and 'regime_stable' (bool)
```

The default configuration (`auto_window=True`, `threshold_mode="fixed"`,
`stable_window=12`) is calibrated for crypto 1H and is a good starting point.

### Configuration for non-standard timeframes

```python
from forgedge import MarketContext, MarketContextConfig, EMAProxyConfig

config = MarketContextConfig(
    ema_proxy=EMAProxyConfig(
        auto_window=True,
        window_unit="day",          # recommended: same history amount on any timeframe
        window_estimation=168.0,    # 168-day OU estimation window
        bar_hours=4.0,              # explicit if no DatetimeIndex (e.g. 4H bars)
        stable_window=12,           # consecutive bars for regime_stable=True
    )
)
mc = MarketContext(kpi, config=config)
enriched = mc.run()
```

### Verifying regime quality

```python
print(mc.distribution())
# Balanced distribution across 5 regimes is a good sign.
# If one regime exceeds 50% of bars, consider threshold_mode="balanced".

print(mc.window_resolution)
# source="hurst_ou"    → EMAs derived from data  (optimal)
# source="fallback"   → OU did not converge; default spans (9/25) used
# source="configured" → auto_window=False; manually configured spans used
```

### When to use `threshold_mode="balanced"`

If the regime distribution is heavily skewed (e.g. a strongly trending asset),
consider the balanced mode:

```python
config = MarketContextConfig(
    ema_proxy=EMAProxyConfig(
        threshold_mode="balanced",
        threshold_basis="expanding",  # causal: no look-ahead
        target_distribution=[0.15, 0.20, 0.30, 0.20, 0.15],
    )
)
```

---

## 3. Module 1 — Event Discovery

### Minimal configuration (production)

```python
from forgedge import EventDiscovery

ed = EventDiscovery(enriched)
candidates = ed.run()
print(f"{len(candidates)} event candidates")
print(ed.summary().head(20))
```

### Configuration with walk-forward OOS

For production, enabling walk-forward validation is strongly recommended:

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams

config = DiscoveryConfig(
    train_ratio=0.80,           # 80% IS for discovery, 20% reserved for OOS
    walk_forward=EventWalkForwardConfig(
        n_splits=4,             # split OOS into 4 windows
        min_pass_rate=0.75,     # event must pass the gate in ≥75% of windows
    ),
    gate_params=GateParams(     # ConsistencyGate thresholds
        min_tpm=0.5,             # ≥0.5 episodes/month on average (default)
        max_dispersion=1.5,      # Index of Dispersion ≤ 1.5 (Var/Mean of monthly counts, default)
    ),
    max_and_components=2,       # 1=singles only, 2=+pairs, 3=+pairs+triples (conservative default)
)
ed = EventDiscovery(enriched, config=config)
candidates = ed.run()

# Filter to events that passed walk-forward
wf_stable = [c for c in candidates if c.validation and c.validation.passed]
print(f"{len(wf_stable)} OOS-stable events out of {len(candidates)} total")
```

**Episode vs bar counting (`GateParams.event_counting`).** The rate
(`min_tpm`) and dispersion (`max_dispersion`) criteria count either **bars**
or **episodes** (maximal runs of consecutive activations bridged by gaps of
up to `episode_gap` bars, default `1`). `"episode"` is the default: it
removes the per-bar counting artifact whereby a persistent multi-bar state
(e.g. a 3–5 bar `RSI < 30` stretch) inflates monthly variance and gets
wrongly rejected — a single episode counts once, regardless of how many bars
it spans. For impulse events (crossovers, one-bar candlestick patterns) the
two modes are identical. `event_counting="bar"` reproduces the historical
(pre-episode) gate behaviour exactly:

```python
GateParams(min_tpm=2.0, max_dispersion=2.5, event_counting="bar")
```

`min_episodes` (default `10`) is an additional statistical-power floor on the
absolute episode count, applied only in `"episode"` mode. In `"episode"` mode
the effective dispersion threshold is also automatically raised to a Poisson
χ² floor whenever the user's `max_dispersion` would reject an event that is
statistically indistinguishable from a random (Poisson) process at the
observed rate — so the gate never rejects pure-noise-consistent timing.

With `train_ratio < 1.0` and `walk_forward` active, each candidate exposes
`c.validation` (a `ValidationResult`) with:
- `c.validation.passed` — True if the event passes in ≥ `min_pass_rate` windows
- `c.validation.pass_rate` — fraction of windows passed
- `c.validation.fold_results` — per-window breakdown

### Reading an EventCandidate

```python
c = candidates[0]
print(c.event_id)         # "EV-BTC-1H-260610-000"
print(c.expression)       # "rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118"
print(c.event_formula)    # human-readable formula
print(c.sql_expression)   # DuckDB/SQL query to filter active bars

# Activation statistics
s = c.activation_stats
print(f"Activations: {s.n_activations}, Active months: {s.n_active_months}")
print(f"Max monthly share: {s.max_monthly_share:.2%}")

# Apply event to new data (production, no look-ahead)
new_bars_active = c.apply(new_kpi_table)   # pd.Series bool
```

### Checking ConsistencyGate quality

```python
for c in candidates:
    gate = c.consistency_gate
    print(f"{c.event_id}: acts={gate.n_activations}, "
          f"months={gate.n_active_months}, passed={gate.passed}")
```

### Manual event injection: `CustomEvent`

When you want to test a user-defined hypothesis without running automatic
discovery, bypass Module 1 entirely using `CustomEvent` and the
`manual_events` argument of `forge()`:

```python
from forgedge import forge, CustomEvent

events = [
    CustomEvent("rsi_14 < 25 and spread_ema < -0.02", name="rsi_extreme_spread"),
    CustomEvent("volume > volume_ema_20 * 2", name="volume_spike"),
]

# manual_events bypasses Module 1; ConsistencyGate still runs
# (failure emits a warning but does not drop the event)
result = forge(
    kpi,
    ticker="BTCUSDC",
    timeframe="1H",
    manual_events=events,
)

for contract, resp in result.rule_responses:
    print(contract.alpha_id, resp.verdict)
```

Key notes:
- `manual_events` and `event_discovery_config` are mutually exclusive — passing both raises `ValueError`.
- AND composition is not performed in manual injection mode.
- Each resulting candidate has `event_id = "CUSTOM-{name}"`.
- For standalone use (without `forge()`), use `CustomEvent.apply(df)` or `CustomEvent.to_event_candidate(df)`.

---

## 4. Module 2 — Alpha Discovery

### Minimal configuration (production)

```python
from forgedge import AlphaDiscovery, AlphaConfig

ad = AlphaDiscovery(
    ed.df,                          # post-pipeline DataFrame from Event Discovery
    candidates,                     # list[EventCandidate]
    AlphaConfig(
        asset="BTC",
        timeframe="1H",
        train_ratio=0.70,           # 70% IS / 30% OOS (default)
    ),
)
contracts = ad.run()
promoted = ad.promoted_contracts()
```

`promoted_contracts()` accepts an optional `min_lift` filter that further
restricts the promoted set to contracts whose IS lift (`event_stats.lift`) is
`>= min_lift` — useful to trim the long tail of marginal-lift hypotheses before
the expensive Rule Discovery backtest:

```python
promoted = ad.promoted_contracts(min_lift=0.05)   # keep only lift >= 0.05
```

`None` (the default) applies no lift filter; `0.0` keeps strictly-positive lift.

**Critical note:** pass `ed.df`, not the original KPI Table. `ed.df` already
contains the derived features (ratio, spread) computed during Event Discovery;
passing the original table works but is slightly slower as features are
recomputed deterministically from the stored component parameters.

**Default `horizon_grid` is timeframe-scaled.** When `AlphaConfig.horizon_grid`
is left unset, `forge()` (not the `AlphaConfig` class default itself) resolves
it from `timeframe`: daily-or-slower timeframes (`"1D"`, `"3D"`, `"1W"`, …) get
`(1, 2, 3, 5, 7, 10)` bars; intraday timeframes keep the hourly-calibrated
`(1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48)`. Building `AlphaDiscovery` directly
(bypassing `forge()`) still gets the intraday-calibrated class default
regardless of `timeframe` — pass `horizon_grid` explicitly for daily-or-slower
data in that case. If you pass an explicit `AlphaConfig` that still carries the
untouched hourly default on a daily-or-slower `timeframe`, `forge()` emits a
`UserWarning` (any custom grid you set is respected silently).

### Advanced configuration with custom grid

```python
from forgedge import AlphaConfig
from forgedge.alpha_discovery.models import PromotionThresholds

config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72, 96),  # horizons in 4H bars
    train_ratio=0.75,
    target_mode="proj",             # "proj" (default) | "abs" — see below
    trend_sma_mult=2.0,             # PROJ trend SMA window = round(2.0 * h) bars
    thresholds=PromotionThresholds(
        ic_min_abs=0.02,
        ic_max_p=0.05,
        min_lift=0.08,
        min_cohens_d=0.15,
        use_fdr=True,
        fdr_q=0.10,
        oos_max_p=0.10,
    ),
    fee_per_side=0.001,             # recorded in the contract for Rule Discovery
)
ad = AlphaDiscovery(ed.df, candidates, config)
```

**Target mode — `target_mode="proj"` (default) vs `"abs"`.** The binary target
that drives win rate, lift and the derived take-profit can be scored two ways:

- `"abs"` — absolute forward return: `log(fwd_max / close)`.
- `"proj"` — *excess return over the local trend*:
  `log(fwd_max / close) − log(SMA_w[t] / SMA_w[t−h])`, where the trend SMA window
  is `w = round(trend_sma_mult * h)` bars. A long event that merely rides a bull
  trend is therefore not credited with that trend premium — only the edge *above*
  the drift counts. `"proj"` is the default and applies to **long** events only;
  for shorts it reverts to `"abs"` (the bear drift *is* the alpha to capture).
  When the IS history is shorter than `≈ (trend_sma_mult + 1) * h` bars, scoring
  also reverts to `"abs"` with a warning.


### Reading the results

```python
# Summary sorted by composite score
df = ad.summary()
print(df[["expression", "direction", "holding_period_h", "sell_pct",
          "lift", "cohens_d", "oos_passed", "grade"]].head(10))

# Detail on a promoted contract
c = promoted[0]

# Derived target
dt = c.derived_target
print(f"Horizon: {dt.holding_period_h}h, Direction: {dt.direction}")
print(f"sell_pct: {dt.sell_pct:.4f}, mean_advantage: {dt.mean_advantage:.4f}")

# OOS confirmation
oos = c.oos_validation
print(f"OOS: passed={oos.passed}, lift={oos.lift:.4f}, p={oos.p_value:.4f}")

# Regime sensitivity
for rs in c.regime_analysis.per_regime:
    print(f"  {rs.regime}: IC={rs.ic:.3f}, win_rate={rs.win_rate:.3f}, {rs.strength}")
print(f"Dependency: {c.regime_analysis.dependency_type}")
```

### Diagnosing rejected and promoted contracts

```python
# The only hard rejection gate is undetermined direction
rejected = [c for c in contracts if not c.promoted]
for c in rejected:
    print(f"REJECTED {c.event_candidate_id}: {c.rejection_reasons}")
# Only cause: "no derivable target" → no horizon produces a finite advantage

# Promoted contracts carry non-blocking diagnostics in their own field.
# `rejection_reasons` is empty on a promoted contract — it holds blocking
# causes only.
for c in promoted:
    if c.diagnostics:
        print(f"PROMOTED {c.event_candidate_id} (grade={c.alpha_score.grade}):")
        for d in c.diagnostics:
            print(f"  {d}")
# Examples of diagnostics on promoted contracts:
# "IC weak (|IC|=0.012 < 0.02, p=0.083)"
# "lift 0.052 < 0.08"
# "OOS weak (p=0.143 vs 0.10, mean_adv=0.0021, n_act=7)"
# "not significant under BH FDR"
# These statistical weaknesses inform the grade (A–D) but do not block promotion.
```

---

## 5. Module 3 — Rule Discovery

### Minimal configuration (production)

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig

# by_id built after ed.run() and ad.run()
by_id = {c.event_id: c for c in candidates}

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        vr = resp.validated_rule
        print(f"  drop={vr.params.buy_drop_pct}  sell={vr.params.sell_pct}"
              f"  h={vr.params.target_h}  PF={resp.in_sample_summary.profit_factor:.2f}")
```

### Reading the response

```python
resp = rd.run()

# Main verdict
print(resp.verdict)           # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
print(resp.is_edge)           # True for EDGE and PARTIAL-EDGE
print(resp.rejection_reasons) # list of failed gates (empty if EDGE)

# IS metrics
s = resp.in_sample_summary
print(f"Trades: {s.total_trades}, PF: {s.profit_factor:.2f}, WR: {s.win_rate_pct:.2%}")
print(f"Expectancy: {s.expectancy:.4f}, tpm: {s.tpm_mu:.1f}")

# Execution envelope (conservative vs optimistic)
env = resp.execution_envelope
print(f"PF conservative: {env.conservative.profit_factor:.2f}")
print(f"PF optimistic:   {env.optimistic.profit_factor:.2f}")

# Walk-forward OOS
wf = resp.walk_forward
print(f"OOS PF: {wf.oos_summary.profit_factor:.2f}, consistency: {wf.consistency:.0%}")

# Compact text report
from forgedge.rule_discovery import text_report
print(text_report(resp))
```

### Selecting an entry mode (`entry_mode`)

`RuleDiscoveryConfig.entry_mode` (default `"limit"`) controls how the entry
order is evaluated during backtesting:

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig

config = RuleDiscoveryConfig(entry_mode="auto")
rd = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
```

- `"limit"` (default) — the original behaviour: the entry is a limit order at
  `anchor * (1 ∓ buy_drop_pct)`, and the grid optimises `buy_drop_pct` as an
  entry-price lever. Can suffer from the "fill confound" — a deep, low-fill
  limit can show a high profit factor on a rare, non-representative subset of
  trades.
- `"market"` — pure baseline at next-open fill (≈100% fill rate), no entry
  optimiser. Isolates the signal's edge from the entry mechanism.
- `"auto"` — two-stage: Stage 1 evaluates in market mode and that verdict
  (plus walk-forward, regime and envelope evidence) is authoritative. Stage 2
  runs the limit optimiser only on EDGE/PARTIAL-EDGE survivors, adopting the
  improved entry only if it still fills at `>= criteria.min_fill_rate_opt`
  (default `0.80`). The optimiser can refine but never fabricate an edge from
  a market-mode NON-EDGE.

When `entry_mode="auto"`, `resp.entry_optimization` (an `EntryOptimization`)
records the decision:

```python
eo = resp.entry_optimization
print(eo.selected_entry)   # "market" | "limit" — the adopted mode
print(eo.adopted)          # True if the limit optimiser improved on market
print(eo.reason)           # human-readable explanation
print(f"market PF={eo.market_profit_factor:.2f} fill={eo.market_fill_rate:.0%}")
```

### Generating HTML reports for review

```python
from forgedge.rule_discovery import html_report
import json

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    if resp.is_edge:
        with open(f"reports/{resp.alpha_id}.html", "w") as f:
            f.write(html_report(resp))
        with open(f"reports/{resp.alpha_id}.json", "w") as f:
            json.dump(resp.to_dict(), f, indent=2)
    else:
        print(f"NON-EDGE: {resp.alpha_id} — {resp.rejection_reasons}")
```

### Full diagnostics on NON-EDGE rules (`early_elimination=False`)

By default, a rule that fails the fast IS screen (too few trades, PF < 1, or
insufficient fill rate) is rejected immediately without running the walk-forward
and diagnostics — saving compute. The trade-count floor is **not** a fixed
absolute: it scales with the in-sample length as
`max(10, n_months * min_tpm)` (spec RD-04), so the requirement is neither too
strict on short IS periods nor too lax on long ones. `min_tpm` (default `2.0`)
on `SelectionCriteria` is the sole frequency gate that drives it. With
`early_elimination=False` the full
pipeline runs regardless: the verdict is still `NON-EDGE`, but walk-forward,
regime analysis, and MAE/MFE are all populated — useful for uniform reporting
or to inspect the OOS behaviour of weak rules:

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig, SelectionCriteria

config = RuleDiscoveryConfig(
    criteria=SelectionCriteria(early_elimination=False),
)
rd   = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
# resp.walk_forward is now populated even when resp.verdict == "NON-EDGE"
print(f"OOS PF: {resp.walk_forward.oos_summary.profit_factor:.2f}")
```

---

## 6. Module 4 — Rule Registry

### Minimal configuration (production)

The Rule Registry receives validated rules from all tickers in the session and
produces the flat table and HTML report. The fastest path is `from_forge_results`,
which builds submissions and frames automatically from each ticker's `ForgeResult`:

```python
from forgedge import RuleRegistry, RegistryConfig

# Run forge() independently for each ticker
results = {}
for ticker, kpi in kpi_tables.items():
    results[ticker] = forge(kpi, asset=ticker, timeframe="1H")

# Module 4: one row per EDGE / PARTIAL-EDGE
registry = RuleRegistry.from_forge_results(results).run()

print(registry.summary().to_string(index=False))
```

### Configuration with custom parameters

```python
config = RegistryConfig(
    overlap_threshold=0.70,        # Jaccard >= 0.70 → duplicate
    cross_pf_threshold=1.5,        # absolute PF floor for a cross-ticker PASS
    min_cross_pf_retention=0.8,    # and the fraction of home PF it must keep
    generic_ratio_threshold=2/3,   # >= 2/3 of tickers PASS → GENERIC
    export_format="excel",         # "csv" or "excel"
    html_charts=True,              # inline SVG in the report
)
registry = RuleRegistry.from_forge_results(results, config=config).run()
```

### Reading the results

```python
# Full flat table (one row per document)
df = registry.flat_table()
print(df[["rule_id", "source_ticker", "pf", "is_duplicate",
          "classification", "cross_ticker_score"]].to_string())

# Non-duplicate and generic only
df_clean = df[~df["is_duplicate"] & df["classification"].isin(["GENERIC", "PARTIAL"])]

# Access the cross-ticker results for a single rule
doc = registry.documents[0]
for ticker, ct in doc.cross_ticker.items():
    print(f"  {ticker}: PF={ct.pf:.2f}, {ct.verdict}")
print(f"Classification: {doc.classification}")  # GENERIC / PARTIAL / SPECIFIC / ISOLATED

# Correlation matrices (Jaccard and Spearman)
m = registry.matrices
print(m.jaccard.round(2))
print(m.spearman.round(2))
```

### Exporting the flat table and HTML report

```python
# Flat table as CSV or Excel
flat_path = registry.export("forge_flat_table.xlsx")

# Self-contained HTML report (inline SVG, no CDN)
html = registry.html_report(timeframe="1H")
with open("forge_report.html", "w", encoding="utf-8") as f:
    f.write(html)
```

### Manual construction with RuleSubmission

When not using `forge()` as the orchestrator, submissions can be built explicitly:

```python
from forgedge import RuleDiscovery, RuleRegistry, RuleSubmission

submissions = []
for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    resp = RuleDiscovery(ed.df, contract, cand).run()
    if resp.is_edge:
        submissions.append(RuleSubmission(
            ticker="BTC",
            response=resp,
            candidate=cand,
            grade=contract.alpha_score.grade,
        ))

frames = {"BTC": ed.df}
registry = RuleRegistry(submissions, frames).run()
```

---

## 7. Full production pipeline

### One-call orchestrator: `forge`

The five modules are wired together by the built-in `forge` orchestrator, so the
whole session is a single call. It accepts the KPI Table plus the configuration
object of each module and returns a `ForgeResult` carrying every artefact —
including the Rule Registry (Module 4) built from the run's tradeable rules:

```python
from forgedge import forge

result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.summary())                      # one row per candidate + rule_verdict
for contract, response in result.edges():    # EDGE / PARTIAL-EDGE only
    print(contract.alpha_id, response.verdict)

print(result.registry.summary())             # Module 4 — catalogued rules
```

### Quick start with presets: `forge_preset`

Rather than hand-tuning every gate, `forge_preset` returns a ready-made
`(DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig)` **triple** — one
calibrated config per module (M1/M2/M3) — for a named search profile and your
timeframe. The four presets are:

| Preset | Profile |
|---|---|
| `"sniper"` | Rare, regular, high-precision events with simple rules. Needs a long IS (≥ 2 years on 1D). Do **not** pair with the rotation calibrator. |
| `"balanced"` | Moderate frequency, good IS/OOS balance. Sensible default for most assets/timeframes. |
| `"sweep"` | Wide sweep, many candidates, permissive thresholds. Designed to run with the rotation calibrator — pair with `rotation_calibration=RotationConfig(k>=100)` and `promoted_contracts(min_lift=0.05)`. |
| `"burst"` | Time-concentrated events (regime-change, momentum, volume spikes). High dispersion explicitly allowed. |

```python
from forgedge import forge, forge_preset

disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="BTC")
result = forge(
    kpi,
    event_discovery_config=disc_cfg,
    alpha_config=alpha_cfg,
    rule_discovery_config=rd_cfg,
)
```

`forge_preset(preset, timeframe, asset="ASSET", train_ratio=0.70, **overrides)`
accepts keyword overrides for any computed parameter — `min_tpm`,
`max_dispersion`, `max_and_components`, `timestamp_col`, `event_counting`
(Discovery side), `min_lift`, `min_cohens_d`, `fdr_q`, `oos_max_p`,
`horizon_grid`, `bars_per_day` (Alpha side), and `rd_min_tpm` (Rule Discovery
side). Each preset calibrates `min_tpm` from a per-mode daily rate —
`event_counting="episode"` (the default) uses an episodes/month target,
`"bar"` uses a bars/month target — scaled to your timeframe. Call
`preset_info()` (all presets) or `preset_info("sweep")` (one) to print the
resolved parameters, and `PRESETS` for the list of names.

Per-module configuration is passed through dedicated keyword arguments:

```python
from forgedge import (
    forge, MarketContextConfig, EMAProxyConfig,
    DiscoveryConfig, AlphaConfig,
)
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams
from forgedge.alpha_discovery.models import PromotionThresholds

result = forge(
    kpi,
    asset="BTC",
    timeframe="1H",
    market_context_config=MarketContextConfig(
        ema_proxy=EMAProxyConfig(auto_window=True, window_unit="day", bar_hours=1.0),
    ),
    event_discovery_config=DiscoveryConfig(
        train_ratio=0.80,
        walk_forward=EventWalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        gate_params=GateParams(min_tpm=0.5, max_dispersion=1.5),  # defaults shown explicitly
    ),
    alpha_config=AlphaConfig(
        train_ratio=0.70,
        thresholds=PromotionThresholds(min_lift=0.08, min_cohens_d=0.15),
    ),
)
```

Useful switches:

- `ticker="BTCUSDC"` — the label used for the Rule Registry pool and the Alpha Contract
  metadata (falls back to `alpha_config.asset` or `asset`).
- `run_market_context=False` — feed a table that already carries `regime` (Module 0 is
  also skipped automatically when the `regime` column is present).
- `run_rule_discovery=False` — stop after Alpha Discovery to get the promoted hypotheses
  (`result.promoted`) without backtesting them (Module 4 is then skipped too).
- `run_registry=False` — stop after Rule Discovery, without building the Module 4 registry.
- `only_validated_events=True` — hand Alpha Discovery only the walk-forward-validated
  candidates (when Event Discovery ran with `walk_forward`).
- `rule_discovery_grades=("A", "B")` — restrict the (expensive) Rule Discovery backtest to
  promoted Alpha Contracts whose letter grade (`A` / `B` / `C` / `D`) is in this set.
  Contracts filtered out still appear in `result.contracts` / `result.promoted` for audit
  but receive no rule response and never reach the Rule Registry.  Comparison is
  case-insensitive.  When omitted, every promoted contract is backtested.
- `rotation_calibration=RotationConfig(k=100)` — run the (slower, sampled) search-level
  rotation null inline after Alpha Discovery instead of the default fast exact null (see
  *Search-level calibration* below). Each promoted contract then carries `rotation_p` /
  `rotation_threshold`. Supersedes `fast_null` when set.
- `fast_null=False` — skip the default fast, exact search-level rotation null (`fast_null=True`
  by default; near-zero extra cost, see *Search-level calibration* below).
- `time_budget=TimeBudget.build(n_bars=len(kpi), horizon_bars=48, embargo_bars=5)` — share one
  purged/embargoed IS/OOS time axis across Event Discovery and Alpha Discovery instead of each
  module cutting the timeline independently (see *Purging and embargo* below). Default `None`
  builds one automatically from `train_ratio` and the resolved horizon grid.
- `progress=True` — print per-stage status and a Rule Discovery progress bar to `stderr`
  (useful for long runs). Independently of the flag, every milestone is logged at `INFO`
  on the `forgedge.forge` logger, so `logging.basicConfig(level=logging.INFO)` surfaces the
  same information. The bar uses `tqdm` when installed, else a dependency-free fallback.

`ForgeResult` exposes the live module instances (`result.market_context`,
`result.event_discovery`, `result.alpha_discovery`) for drill-down, plus
`result.candidates`, `result.contracts`, `result.promoted`,
`result.rule_responses`, `result.event_frame` (the post-pipeline frame) and
`result.registry` (the Module 4 `RuleRegistry`).

### Multi-ticker sessions: `forge_multi`

The Rule Registry's cross-ticker backtest only has other tickers to replay
against when the session covers several. `forge_multi` runs `forge` per ticker
and pools every tradeable rule into one cross-ticker registry:

```python
from forgedge import forge_multi

frames = {
    "BTCUSDC": btc_kpi,
    "ETHUSDC": eth_kpi,
    "ADAUSDC": ada_kpi,
}
results, registry = forge_multi(frames, timeframe="1H")

print(registry.summary())            # rule_id, source_ticker, cross-ticker score, class
registry.export("rules.xlsx")        # flat table (Module 4 persistence artefact)
html = registry.html_report()        # self-contained HTML

# Per-ticker drill-down is still available
for ticker, res in results.items():
    print(ticker, len(res.edges()), "tradeable rules")
```

Pass the per-module config objects (`event_discovery_config`, `alpha_config`, …)
as keyword arguments — they are forwarded to every per-ticker run. Do **not**
pass `ticker` / `asset` (set automatically per ticker) or `run_registry` (the
pooled registry supersedes the per-ticker ones).

### Search-level calibration: `FastRotationNull` and `RotationCalibrator`

A promoted contract can still be an artefact of FORGE's multiple-testing
surface — with enough candidates and horizons, *some* edge appears by chance.
**`forge()` runs a search-level rotation null by default** (`fast_null=True`):
`FastRotationNull` computes, via one FFT pass per candidate, the *exact* null
distribution of the search's best standardised excess over every circular
offset of the `close` column at once — no sampling, no seed, ~1–2 s even on
thousands of candidates. It annotates each promoted contract with
`rotation_p` / `rotation_threshold` and stores the report on
`ForgeResult.calibration`:

```python
result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.calibration.tippett_p)          # search-level p-value
for c in result.promoted:
    print(c.alpha_id, c.rotation_p, c.rotation_threshold)
```

Rule Discovery then requires `rotation_p <= criteria.max_rotation_p` (default
`0.05`) for a full `EDGE` verdict — a contract that only won the
multiple-testing lottery of its own discovery session is capped at
`PARTIAL-EDGE` (still tradeable via `resp.is_edge`). `FastRotationNull` only
computes the `abs_z` yardstick (the one statistic that reduces exactly to a
cross-correlation); pass `fast_null=False` to `forge()` to skip it entirely.

For the full multi-yardstick calibration (`composite`, `is_lift`, …, via a
Tippett min-p combination) or a large K sanity check, use the standalone,
sampled `RotationCalibrator` instead — heavier (~K × the cost of one Alpha
Discovery pass) but not limited to `abs_z`:

```python
from forgedge import forge, RotationCalibrator, RotationConfig

result = forge(kpi, ticker="BTCUSDC", timeframe="1H", run_rule_discovery=False)

cal = RotationCalibrator(
    result.event_frame,     # post-EventDiscovery table
    result.candidates,      # the real candidate set (reused on every draw)
    alpha_cfg,              # the same AlphaConfig used in the run
)
report = cal.run(result.promoted, RotationConfig(k=100))

print(report.summary())             # human-readable verdict
print(report.tippett_p)             # primary calibration p-value
print(len(report.survivors), "contracts above the null bar")
```

`RotationConfig(k=100, alpha=0.05, seed=..., in_sample_stats=...)` controls the
number of draws (K≥100 for a stable q95; K=40 for a quick sanity check), the
Type-I target and the yardsticks fed to the Tippett combination. The returned
`CalibrationReport` exposes `tippett_p`, `tippett_best_stat`, `per_stat_p`,
`null_q`, `real_stats`, `null_arrays` and `survivors` (the promoted contracts
whose winning statistic clears the null bar). The same calibration can run
**inline** during `forge()` via `rotation_calibration=RotationConfig(...)` —
that path supersedes the default `fast_null` pass and writes the same
`rotation_p` / `rotation_threshold` fields; use the standalone calibrator above
when you want the full report object without slowing down the main run.

### Auditing the search surface: `HypothesisLedger`

Every `forge()` run also returns `result.ledger` (a `HypothesisLedger`) — plain
bookkeeping of how many hypotheses the session actually consumed, so the
surface behind a published verdict is auditable even though it is *priced* by
the rotation null above, not by this count directly (event-level hypotheses
are heavily correlated, so plugging their raw count into an analytic haircut
would overstate the selection bias):

```python
print(result.ledger.describe())
# "hypothesis surface: 640 candidates × 6 horizons = 3840 return-tests;
#  12 promoted; ~180 grid cells/rule (total ≲ 691200)"

result.ledger.m1_candidates   # candidates handed to Alpha Discovery
result.ledger.m2_horizons     # horizons scanned per candidate
result.ledger.m2_promoted     # contracts promoted
result.ledger.m3_grid_cells   # Rule Discovery operational-grid size (0 until backtested)
result.ledger.m2_surface      # m1_candidates * m2_horizons
result.ledger.total_surface   # upper bound: m2_surface * max(m3_grid_cells, 1)
```

`RuleDiscoveryConfig.n_trials_upstream` remains available for callers who want
the analytic Deflated-Sharpe haircut to include an explicit upstream factor
instead of relying on the rotation null.

### Purging and embargo: `TimeBudget`

Each module used to cut the IS/OOS timeline independently. The split boundary
itself was honest, but forward-looking quantities crossed it: the forward
return at IS bar `t` reads closes up to `t + h`, so the last `h` IS bars are
partially scored on OOS prices, and a walk-forward trade entered near the end
of a train window can exit inside the test window. `TimeBudget` is the single
shared axis that fixes this — **purging** removes the IS rows whose
forward-looking window overlaps the OOS side; an optional **embargo** adds a
buffer of bars at the *start* of the OOS window for serial-correlation
quarantine (`0` by default — purging alone removes the mechanical overlap).

```python
from forgedge import forge, TimeBudget

budget = TimeBudget.build(
    n_bars=len(kpi),
    train_ratio=0.70,
    horizon_bars=48,     # widest horizon scanned — sets the default purge width
    embargo_bars=5,      # optional extra buffer at the start of OOS
)
result = forge(kpi, time_budget=budget)
print(result.time_budget.describe())
```

`TimeBudget.build(n_bars, train_ratio=0.7, horizon_bars=0, purge_bars=None, embargo_bars=0)`
defaults `purge_bars` to `horizon_bars` when omitted. Passed to `forge()` it is
threaded into both `EventDiscovery` and `AlphaDiscovery` so they share one
split; `ForgeResult.time_budget` exposes the effective budget. **Purging is on
by default** for Alpha Discovery (purge width = `max(horizon_grid)`) and for
Rule Discovery's walk-forward (via `RuleWalkForwardConfig.purge_bars` /
`embargo_bars`, `None`/`0` by default — `None` also defaults to the horizon
being tested) — this is a real, if usually small, numeric change from
pre-`TimeBudget` results (boundary rows that used to leak OOS information are
now excluded). To reproduce old, unpurged numbers exactly, pass an explicit
`TimeBudget.build(n_bars=..., purge_bars=0)` and, for Rule Discovery,
`RuleWalkForwardConfig(purge_bars=0)`. `AlphaConfig.embargo_bars` (default `0`) and
`RuleWalkForwardConfig.embargo_bars` (default `0`) change nothing unless you opt
in — only purging is on by default.

### Building it by hand

If you need finer control, the same flow can be written explicitly — this is
exactly what `forge` runs internally:

```python
import pandas as pd
from forgedge import (
    MarketContext, MarketContextConfig, EMAProxyConfig,
    EventDiscovery, DiscoveryConfig,
    AlphaDiscovery, AlphaConfig,
    RuleDiscovery,
)
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.rule_discovery import html_report


def run_forge_pipeline(
    kpi: pd.DataFrame,
    asset: str,
    timeframe: str,
    bar_hours: float,
) -> tuple:
    """Run the full FORGE pipeline and return (rule_responses, ad, ed).

    Parameters
    ----------
    kpi : pd.DataFrame
        KPI Table with 'close' column and 'open_dt' column (or DatetimeIndex).
    asset, timeframe : str
        Traceability metadata for the AlphaContracts.
    bar_hours : float
        Bar duration in hours (e.g. 1.0 for 1H, 4.0 for 4H).

    Returns
    -------
    rule_responses : list[tuple[AlphaContract, RuleDiscoveryResponse]]
        Pairs of (contract, response) for every promoted contract.
    ad : AlphaDiscovery
        Instance with access to summary(), market_structure, split_idx.
    ed : EventDiscovery
        Instance with access to df (enriched DataFrame) and candidates.
    """

    # ── Module 0: regime ────────────────────────────────────────────────
    mc_config = MarketContextConfig(
        ema_proxy=EMAProxyConfig(
            auto_window=True,
            window_unit="day",
            bar_hours=bar_hours,
            stable_window=12,
        )
    )
    enriched = MarketContext(kpi, config=mc_config).run()

    # ── Module 1: events ────────────────────────────────────────────────
    ed_config = DiscoveryConfig(
        train_ratio=0.80,
        walk_forward=EventWalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        gate_params=GateParams(min_tpm=0.5, max_dispersion=1.5),
        max_and_components=2,
    )
    ed = EventDiscovery(enriched, config=ed_config)
    candidates = ed.run()

    # ── Module 2: alpha ─────────────────────────────────────────────────
    alpha_config = AlphaConfig(
        asset=asset,
        timeframe=timeframe,
        horizon_grid=(1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48),
        train_ratio=0.70,
        thresholds=PromotionThresholds(
            min_lift=0.08,
            min_cohens_d=0.15,
            use_fdr=True,
            fdr_q=0.10,
            oos_max_p=0.10,
        ),
    )
    ad = AlphaDiscovery(ed.df, candidates, alpha_config)
    ad.run()
    promoted = ad.promoted_contracts()

    # ── Module 3: rule discovery ─────────────────────────────────────────
    by_id = {c.event_id: c for c in candidates}
    rule_responses = []
    for contract in promoted:
        cand = by_id[contract.event_candidate_id]
        rd   = RuleDiscovery(ed.df, contract, cand)
        resp = rd.run()
        rule_responses.append((contract, resp))

    return rule_responses, ad, ed


# Usage
kpi = pd.read_parquet("btc_1h.parquet")
rule_responses, ad, ed = run_forge_pipeline(kpi, asset="BTC", timeframe="1H", bar_hours=1.0)

for contract, resp in rule_responses:
    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        print(f"  PF={resp.in_sample_summary.profit_factor:.2f}"
              f"  OOS={resp.walk_forward.oos_summary.profit_factor:.2f}")
```

---

## 8. Target-first workflow: `TargetOptimizer`

The standard pipeline is *event-first*: discover structurally-consistent events,
then derive the best target for each. `TargetOptimizer` inverts this — you
**fix** the economic target up front (a horizon, an excess-return threshold and a
side) and it finds the events that best predict it. It reuses Event Discovery,
AND composition and the Consistency Gate internally, scoring every candidate by
two-proportion **lift** against the fixed binary target.

It is a fully standalone module — it does **not** touch `forge()` or
`ForgeResult`. The only entry point is the constructor:

```python
from forgedge import TargetOptimizer, TargetConfig

opt = TargetOptimizer(
    train_df,                       # IS OHLCV (+ features); needs 'close' + datetime
    TargetConfig(horizon=20, min_return=0.10, side="long"),
)
results = opt.run()                 # pd.DataFrame, one row per surviving candidate,
                                    # sorted by lift descending
```

`results` columns: `event_id`, `n_components`, `expression`, `n_activations`,
`win_rate_event` (conditional rate), `win_rate_base` (unconditional rate),
`lift` (= `win_rate_event / win_rate_base`) and `z_score`.

### `TargetConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `horizon` | `int` | *required* | Holding period in bars (> 0). |
| `min_return` | `float` | *required* | Take-profit threshold as a fraction (e.g. `0.10` = 10%). |
| `side` | `str` | *required* | `"long"` or `"short"`. |
| `min_activations` | `int` | `10` | Candidates firing on fewer bars are skipped (lift unstable). |
| `min_lift_atoms` | `float` | `1.0` | 1st-pass prune: keep only raw atoms with lift ≥ this *before* AND composition. Lossless at the default `1.0`. |
| `min_lift_result` | `float` | `1.0` | 2nd-pass prune: filter the final result set (atoms + compositions). Raise to keep only the strongest signals. |
| `target_mode` | `"abs"` \| `"proj"` | `"proj"` | Binary-target definition — same `abs`/`proj` semantics as Alpha Discovery (§4). |
| `trend_sma_mult` | `float` | `2.0` | PROJ trend SMA window = `round(trend_sma_mult * horizon)` bars. |

> The two thresholds were split from a single `min_lift` (still accepted but
> deprecated, applied to both passes with a `DeprecationWarning`). `min_lift_atoms`
> gates discovery; `min_lift_result` only trims the output.

`TargetOptimizer(train_df, target_cfg, discovery_cfg=None)` accepts an optional
`DiscoveryConfig`; when omitted it defaults to `DiscoveryConfig(train_ratio=1.0)`
(full-frame discovery).

### Validating and handing off

```python
# Candidate objects aligned with the results table
cands = opt.candidates              # list[EventCandidate]
print(opt.base_rate)                # unconditional win rate of the target (set after run())

# Out-of-sample lift on the full dataset for the top-K survivors
oos = opt.validate_oos(full_df, top_k=10)

# Promote the survivors into Alpha Contracts in fixed-target mode
#   (standard IC / regime / OOS / FDR / scoring against the *fixed* target)
contracts = opt.discover_alpha()    # list[AlphaContract]
```

`discover_alpha()` keeps the binary-target definition (`target_mode`,
`trend_sma_mult`) consistent with the optimizer's own scoring, so the lift you
saw in `run()` matches what Alpha Discovery measures.

---

## 9. Multi-asset workflow

Run an independent session per asset — sessions share no state. Each asset
gets its own distributional thresholds, OU half-lives, and contracts.

```python
assets = {
    "BTC": ("btc_1h.parquet",  1.0),
    "ADA": ("ada_1h.parquet",  1.0),
    "ETH": ("eth_4h.parquet",  4.0),
}

all_promoted = {}
for asset, (path, bar_hours) in assets.items():
    kpi = pd.read_parquet(path)
    promoted, ad = run_forge_pipeline(kpi, asset=asset, timeframe="1H", bar_hours=bar_hours)
    all_promoted[asset] = promoted
    print(f"{asset}: {len(promoted)} promoted hypotheses")

# Cross-asset summary
rows = []
for asset, contracts in all_promoted.items():
    for c in contracts:
        rows.append({
            "asset": asset,
            "alpha_id": c.alpha_id,
            "expression": c.event_expression,
            "grade": c.alpha_score.grade,
            "direction": c.direction,
            "holding_period_h": c.derived_target.holding_period_h,
        })
cross = pd.DataFrame(rows).sort_values("grade")
print(cross)
```

---

## 10. OOS replay: applying discovered events to new data

`EventCandidate.apply(df)` deterministically replays the event on any DataFrame
with the same native columns, using the thresholds fixed at discovery time. This
is the mechanism for forward testing and production use:

```python
# New data (arrived after the discovery session)
new_data = pd.read_parquet("btc_1h_new.parquet")

for c in promoted:
    active_mask = c.apply(new_data)   # pd.Series bool
    n_fires = active_mask.sum()
    print(f"{c.event_id}: {n_fires} activations on new data")
    if active_mask.any():
        print(new_data[active_mask][["open_dt", "close"]].tail(3))
```

`apply()` uses the expression and thresholds stored in the `EventCandidate` —
it requires no access to the original session and re-optimises nothing.

### Monitoring a discovered edge on new data

When new market data becomes available after a discovery session, the correct
tool to verify the edge still holds is **Rule Discovery**, not Alpha Discovery.

Alpha Discovery re-derives direction and optimal horizon from whatever data you
pass it. Running it on a dataset that predates the training window mixes
activations from incompatible market regimes — for example, a vol-spike event
discovered as LONG in 2024–2026 may have also fired frequently during a 2022
crash, where the same condition was followed by strongly *negative* returns. The
mean advantage from those two opposing populations averages toward zero, and
Alpha Discovery returns `direction="undetermined"` ("no derivable target") — not
because the edge is gone, but because the question posed is wrong.

Rule Discovery does not re-derive anything. It takes the `AlphaContract` with
its fixed `direction`, `holding_period_h`, and `sell_pct`, and measures whether
the trading rule still produces positive expected value on the new bars:

```python
import pandas as pd
from forgedge import RuleDiscovery

# Append only the genuinely new bars — do not extend into pre-training history
new_bars = full_df[full_df["open_dt"] > train_df["open_dt"].max()]
eval_df  = pd.concat([train_df, new_bars]).drop_duplicates("open_dt")

for contract, cand in discovered_rules:
    resp = RuleDiscovery(eval_df, contract, cand).run()
    print(f"{contract.alpha_id}: {resp.verdict}")
    print(f"  PF={resp.in_sample_summary.profit_factor:.2f}"
          f"  WR={resp.in_sample_summary.win_rate_pct:.0%}"
          f"  OOS-consistency={resp.walk_forward.consistency:.0%}")
```

**What to expect on new data**

A small drop in profit factor (5–15%) is normal as the IS period expands. The
`walk_forward.consistency` metric is more informative for monitoring: a drop of
25 percentage points or more signals a weakening edge. A verdict change from
PARTIAL-EDGE to NON-EDGE warrants investigating whether the market regime has
shifted.

**Why AlphaDiscovery on full historical data produces "no derivable target"**

Running `AlphaDiscovery(full_df, events, AlphaConfig(train_ratio=1))` on data
that predates the training period asks the wrong question: "does this event have
a measurable edge across *all* of recorded history?" The answer is
regime-dependent. Activations in the discovery window (e.g. 2024–2026) and
activations in an earlier crash cycle (e.g. 2022) belong to two populations with
opposite forward-return characteristics. Their combined mean advantage can cancel
to near zero at every horizon, causing `_derive_target` to return
`direction="undetermined"` even when activation counts are well above the minimum
threshold.

`AlphaDiscovery._event_series()` emits a `UserWarning` in the specific case
where the activation count on the observed frame drops to near zero (fewer than 2
activations and less than 10% of the stored training count). The more common
"no derivable target" outcome, however, arises at a higher level — from the
cancellation of opposing signals across incompatible market regimes — and is
resolved by switching to Rule Discovery rather than re-running Alpha Discovery.

---

## 11. Persisting artefacts

### Saving promoted contracts

```python
import json, yaml

# JSON (all contracts)
with open("alpha_contracts.json", "w") as f:
    json.dump([c.to_contract_dict() for c in promoted], f, indent=2)

# YAML (single contract — human-readable)
for c in promoted:
    with open(f"contracts/{c.alpha_id}.yaml", "w") as f:
        yaml.dump(c.to_contract_dict(), f, allow_unicode=True)
```

### Saving the summary as CSV

```python
ad.summary().to_csv("alpha_summary.csv", index=False)
```

### Saving candidates (full pre-promotion list)

The recommended way to archive full candidates is `persist()`, which performs
a complete pickle round-trip (components, thresholds, activation stats, and
walk-forward validation all included):

```python
import pathlib

pathlib.Path("candidates").mkdir(exist_ok=True)
for c in candidates:
    c.persist(f"candidates/{c.event_id}.pkl")

# Reload in a later session
import pickle
cand = pickle.load(open("candidates/EV-BTC-1H-260610-000.pkl", "rb"))
cand.apply(new_kpi)   # ready to use immediately
```

For human-readable tabular archives (less complete, not invertible):

```python
candidates_data = [
    {
        "event_id": c.event_id,
        "expression": c.expression,
        "event_formula": c.event_formula,
        "sql_expression": c.sql_expression,
    }
    for c in candidates
]
pd.DataFrame(candidates_data).to_csv("event_candidates.csv", index=False)
```

---

## 12. Pre-production checklist

Before promoting a `ValidatedRule` to the Rule Registry, verify:

**Input data**
- [ ] `summary_report(kpi, return_report=True, verbose=False).has_critical == False` — no FAIL-level data-quality findings

**Module 0**
- [ ] `mc.window_resolution["source"] == "hurst_ou"` — adaptive EMAs derived from data
- [ ] `mc.distribution()` shows at least 3 regimes with share > 5% — non-degenerate distribution

**Module 1**
- [ ] All candidates passed to AlphaDiscovery have `c.validation.passed == True` (walk-forward)

**Module 2**
- [ ] `len(promoted) >= 3` — enough hypotheses for diversification; if 0, features produce no finite advantage — add features or widen the horizon grid
- [ ] No promoted contract with `oos.n_activations < 15` — unstable OOS estimate
- [ ] At least one promoted contract with grade ≥ B (`composite_score >= 0.50`)
- [ ] Check `regime_analysis.dependency_type` — avoid `"broken"` for cross-regime robustness

**Module 3**
- [ ] At least one contract with `resp.verdict == "EDGE"` — if only PARTIAL-EDGE, investigate the failed gates
- [ ] `resp.walk_forward.consistency >= 0.50` — half the OOS windows have positive net gain
- [ ] `resp.statistical_validation.temporal_stability != "FAIL"` — edge is not concentrated in time
- [ ] `resp.regime_analysis.avoid_in` is a short list — edge works across multiple regimes
- [ ] Check execution envelope: `env.conservative.profit_factor > 1.5` — edge holds in the conservative case

**Module 4**
- [ ] At least one rule with `classification in ("GENERIC", "PARTIAL")` — the edge is not specific to the discovery ticker alone
- [ ] Check `df[df["is_duplicate"]]["duplicate_of"]` — rules flagged as duplicates must not be promoted to production
- [ ] `doc.cross_ticker_score > 0` for GENERIC rules — generalisation is confirmed on at least one external ticker
- [ ] Review the Jaccard matrix (`registry.matrices.jaccard`) — pairs with overlap ≥ 0.85 indicate structural redundancy between signals
