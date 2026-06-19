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
from forgedge.event_discovery.models import WalkForwardConfig, GateParams

config = DiscoveryConfig(
    train_ratio=0.80,           # 80% IS for discovery, 20% reserved for OOS
    walk_forward=WalkForwardConfig(
        n_splits=4,             # split OOS into 4 windows
        min_pass_rate=0.75,     # event must pass the gate in ≥75% of windows
    ),
    gate_params=GateParams(     # ConsistencyGate thresholds
        min_act=50,             # ≥50 IS activations
        min_months=8,           # active in ≥8 distinct calendar months
        max_conc=0.40,          # ≤40% of activations in any single month
        min_tpm=2.0,            # ≥2.0 activations/month on average
    ),
    max_and_components=2,       # max 2 components in AND (conservative default)
)
ed = EventDiscovery(enriched, config=config)
candidates = ed.run()

# Filter to events that passed walk-forward
wf_stable = [c for c in candidates if c.validation and c.validation.passed]
print(f"{len(wf_stable)} OOS-stable events out of {len(candidates)} total")
```

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
print(f"Max monthly concentration: {s.max_monthly_concentration:.2%}")

# Apply event to new data (production, no look-ahead)
new_bars_active = c.apply(new_kpi_table)   # pd.Series bool
```

### Checking ConsistencyGate quality

```python
for c in candidates:
    gate = c.gate_result
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

**Critical note:** pass `ed.df`, not the original KPI Table. `ed.df` already
contains the derived features (ratio, spread) computed during Event Discovery;
passing the original table works but is slightly slower as features are
recomputed deterministically from the stored component parameters.

### Advanced configuration with custom grid

```python
from forgedge import AlphaConfig
from forgedge.alpha_discovery.models import PromotionThresholds

config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72, 96),  # horizons in 4H bars
    train_ratio=0.75,
    thresholds=PromotionThresholds(
        ic_min_abs=0.02,
        ic_max_p=0.05,
        min_lift=0.08,
        min_cohens_d=0.15,
        min_activations=30,
        use_fdr=True,
        fdr_q=0.10,
        oos_max_p=0.10,
        min_oos_activations=10,
    ),
    fee_per_side=0.001,             # recorded in the contract for Rule Discovery
)
ad = AlphaDiscovery(ed.df, candidates, config)
```

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

# Promoted contracts may carry non-blocking diagnostics
for c in promoted:
    if c.rejection_reasons:
        print(f"PROMOTED {c.event_candidate_id} (grade={c.alpha_score.grade}):")
        for r in c.rejection_reasons:
            print(f"  {r}")
# Examples of diagnostics on promoted contracts ([diagnostic] prefix):
# "[diagnostic] IC weak (|IC|=0.012 < 0.02, p=0.083)"
# "[diagnostic] lift 0.052 < 0.08"
# "[diagnostic] OOS weak (p=0.143 vs 0.10, mean_adv=0.0021, n_act=7)"
# "[diagnostic] not significant under BH FDR"
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

By default, a rule that fails the fast IS screen (< 20 trades, PF < 1, or
insufficient fill rate) is rejected immediately without running the walk-forward
and diagnostics — saving compute. With `early_elimination=False` the full
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
    cross_pf_threshold=2.0,        # minimum PF for cross-ticker PASS
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

Per-module configuration is passed through dedicated keyword arguments:

```python
from forgedge import (
    forge, MarketContextConfig, EMAProxyConfig,
    DiscoveryConfig, AlphaConfig,
)
from forgedge.event_discovery.models import WalkForwardConfig, GateParams
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
        walk_forward=WalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
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
from forgedge.event_discovery.models import WalkForwardConfig, GateParams
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
        walk_forward=WalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
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
            min_activations=30,
            use_fdr=True,
            fdr_q=0.10,
            oos_max_p=0.10,
            min_oos_activations=10,
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

## 8. Multi-asset workflow

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

## 9. OOS replay: applying discovered events to new data

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

### Re-evaluating Alpha Discovery with persisted events

When persisted events are reloaded and passed to `AlphaDiscovery` on a larger
historical frame, all rolling transforms (`pctrank`, `zscore`) are recomputed on
that frame from scratch. If the observed frame contains pre-training history, the
rolling baselines shift: thresholds calibrated on the training distribution may
rarely or never fire, and `AlphaDiscovery` returns `direction="undetermined"` for
every event ("no derivable target; no finite advantage on the grid").

**The rule:** pass only `train_df + new_bars_df`, not a full historical dataset.

```python
import pickle
import pandas as pd
from forgedge import AlphaDiscovery, AlphaConfig

events = [pickle.load(open(f"events/{e}.pkl", "rb")) for e in event_ids]

# ✗ WRONG — pre-training history shifts the rolling baselines
ad = AlphaDiscovery(full_historical_df, events, AlphaConfig(train_ratio=1))

# ✓ CORRECT — training period + new bars only
eval_df = pd.concat([train_df, new_bars_df]).drop_duplicates("open_dt")
ad = AlphaDiscovery(eval_df, events, AlphaConfig(train_ratio=1))
contracts = ad.run()
```

The concat approach preserves the exact rolling context of the training period.
Every `pctrank` and `zscore` value on the training bars is identical to what was
observed during discovery; new bars extend the rolling window from the same tail.

Rolling-transform events (those with `pctrank` or `zscore` components) are most
sensitive to this. An expression such as `pr_close_vol_05_48 > 0.854` means "5-day
vol is in the 85th percentile of the past 48 bars." On the training data the
percentile is computed against the training-period distribution. On a full historical
frame that begins in an earlier, different-regime period, the same absolute vol value
can rank at the 50th percentile — the threshold never fires. As a result,
`cand.apply()` returns all-False, every horizon has `cnt_a = 0`, and there is no
finite mean advantage to report.

**Diagnosing "no derivable target" after reload**

If `AlphaDiscovery` returns "no derivable target" after reloading persisted events,
count activations on the observed frame and compare against the stored training count:

```python
import pickle
import numpy as np
from forgedge.alpha_discovery.discovery import AlphaDiscovery
from forgedge.alpha_discovery.models import AlphaConfig
from forgedge.alpha_discovery.target import forward_returns

for event_id in event_ids:
    cand = pickle.load(open(f"events/{event_id}.pkl", "rb"))
    ad   = AlphaDiscovery(eval_df, [cand], AlphaConfig(train_ratio=1))
    ff   = ad._frame
    active   = cand.apply(ff).fillna(0).astype(bool)
    n_new    = int(active.sum())
    n_stored = int(cand.event_series.fillna(0).astype(bool).sum()) if cand.event_series is not None else None
    fwd      = forward_returns(ff["close"].astype(float), [1])
    cnt_a_h1 = float(active.astype(float).to_numpy() @ np.isfinite(fwd.to_numpy())[:, 0])
    print(f"{event_id}: stored={n_stored}, on_frame={n_new}, cnt_a[h=1]={cnt_a_h1:.0f}")
```

If `on_frame` is less than 10% of `stored`, the rolling baselines have shifted due to
pre-training history in `eval_df`. Rebuild `eval_df` as
`pd.concat([train_df, new_bars_df])`.

`AlphaDiscovery._event_series()` emits a `UserWarning` automatically when the observed
activation count is less than 2 and less than 10% of the stored training count, with
the event expression and the recommended fix included in the message.

---

## 10. Persisting artefacts

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

## 11. Pre-production checklist

Before promoting a `ValidatedRule` to the Rule Registry, verify:

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
