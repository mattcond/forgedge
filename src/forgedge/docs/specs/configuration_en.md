# FORGE — Complete Configuration Reference

All configuration objects in FORGE are Python dataclasses. They ship with
sensible defaults calibrated for crypto 1H data and can be instantiated with no
arguments for typical use cases. Fields are mutable after construction: you only
need to set the fields you want to change from their defaults. Every
configuration object is passed explicitly to its corresponding pipeline
component; there are no global singletons or environment-level settings.

---

## Module 0 — Market Context

### EMAProxyConfig

Configures the EMA-proxy classifier: data source, automatic window derivation,
regime thresholds, and adaptive calibration.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_col` | str | `"close"` | OHLCV column on which the EMAs are computed. The EMAs themselves do not need to be present in the KPI Table. |
| `auto_window` | bool | `True` | When True, derives EMA spans from the data via Hurst/OU analysis. When False, uses `short_period`/`long_period` directly. |
| `short_period` | int | `9` | Fast EMA span — used as fallback when `auto_window=False` or OU does not converge. |
| `long_period` | int | `25` | Slow EMA span — same fallback conditions as above. |
| `thresholds` | list[float] | `[0.975, 0.990, 1.010, 1.025]` | 4 cut points on the `ema_short/ema_long` ratio in `"fixed"` mode. |
| `threshold_mode` | str | `"fixed"` | `"fixed"`: absolute threshold cuts. `"balanced"`: thresholds are recomputed as quantiles of the EMA ratio to match `target_distribution`. |
| `target_distribution` | list[float] | `[0.10, 0.20, 0.40, 0.20, 0.10]` | Target regime frequencies for `"balanced"` mode, one per label. |
| `threshold_basis` | str | `"global"` | `"global"`: quantiles computed over the whole series (not causal). `"expanding"`: causal quantiles — look-ahead free but approximate. |
| `threshold_warmup` | int | `200` | Leading bars that use fixed thresholds before `"expanding"` mode is stable. |
| `window_unit` | str | `"day"` | Unit for `window_estimation`/`window_stride`: `"day"` (calendar days) or `"bar"` (candles). |
| `window_estimation` | float | `168` | Width of the EMA derivation window in the selected unit. 168 days ≈ 24 weeks. |
| `window_stride` | float | `1` | Step between successive estimates, in the same unit. |
| `bar_hours` | float \| None | `None` | Explicit candle duration in hours (e.g. `4.0` for 4H bars). Inferred from DatetimeIndex when None. |
| `fast_ratio` | float | `1/2.3 ≈ 0.435` | Fast-to-slow span ratio for automatic derivation. |
| `min_window_estimates` | int | `10` | Minimum number of converging OU estimates to trust the automatic derivation. |

```python
from forgedge import MarketContext, MarketContextConfig, EMAProxyConfig

enriched = MarketContext(
    kpi,
    config=MarketContextConfig(
        stable_window=12,
        ema_proxy=EMAProxyConfig(
            auto_window=True,
            window_unit="day",
            bar_hours=4.0,              # 4H bars
            threshold_mode="balanced",
            target_distribution=[0.10, 0.20, 0.40, 0.20, 0.10],
            threshold_basis="expanding",  # causal, no look-ahead
        ),
    ),
).run()
```

---

### MarketContextConfig

Top-level configuration container for Module 0.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `classifier` | str | `"ema_proxy"` | Classifier implementation. In v1.0 only `"ema_proxy"` is available. |
| `ema_proxy` | EMAProxyConfig | `EMAProxyConfig()` | EMA-proxy classifier configuration. |
| `labels` | list[str] | `["STRONG_BEAR","BEAR","NEUTRAL","BULL","STRONG_BULL"]` | Regime labels ordered from most bearish to most bullish. |
| `stable_window` | int | `12` | Number of consecutive identical bars required for `regime_stable=True`. |

```python
from forgedge import MarketContext, MarketContextConfig, EMAProxyConfig

enriched = MarketContext(
    kpi,
    config=MarketContextConfig(
        stable_window=6,          # more reactive to regime changes
        ema_proxy=EMAProxyConfig(
            auto_window=True,
            window_unit="day",
            bar_hours=1.0,
        ),
    ),
).run()
```

---

## Module 1 — Event Discovery

### GateParams

Thresholds for the ConsistencyGate — the structural filter that verifies whether
an event has stable temporal structure.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_act` | int | `50` | Minimum number of IS activations. |
| `min_months` | int | `8` | Minimum number of distinct calendar months with at least one activation. |
| `max_conc` | float | `0.40` | Maximum allowed concentration in a single month (fraction of total activations). |
| `min_tpm` | float | `2.0` | Minimum average activation frequency (trades per month). |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import GateParams

config = DiscoveryConfig(
    gate_params=GateParams(
        min_act=30,        # less strict for shorter datasets
        min_months=6,
        max_conc=0.50,
        min_tpm=1.5,
    )
)
```

---

### WalkForwardConfig (Module 1)

Configures Module 1 walk-forward OOS validation. This is the Module 1
`WalkForwardConfig` (`event_discovery.models`); Module 3 has a distinct
`WalkForwardConfig` importable directly from `forgedge`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_splits` | int | `3` | Number of equal OOS windows on which each event is replayed. |
| `min_pass_rate` | float | `0.6` | Minimum fraction of OOS windows in which an event must pass the gate to be marked OOS-stable. |
| `oos_gate_params` | GateParams \| None | `None` | Gate thresholds for OOS evaluation. When None, thresholds are scaled automatically proportional to the OOS window length relative to IS. |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import WalkForwardConfig, GateParams

config = DiscoveryConfig(
    train_ratio=0.80,
    walk_forward=WalkForwardConfig(
        n_splits=4,
        min_pass_rate=0.75,   # stricter: 3 of 4 windows must pass
    ),
)
```

---

### DiscoveryConfig

Main configuration for Module 1. Controls gate thresholds, AND composition,
IS/OOS split, and walk-forward.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gate_params` | GateParams | `GateParams()` | ConsistencyGate thresholds. |
| `max_categorical_classes` | int | `20` | Categorical columns with more distinct values than this limit are classified but excluded from event generation. |
| `scale_free_overrides` | dict[str,bool] \| None | `None` | Manual scale-free flag overrides for specific columns (e.g. `{"rsi_14": True}`). Useful when the auto-heuristic fails on short history. |
| `timestamp_col` | str | `"open_dt"` | Datetime column name in the KPI Table (or DatetimeIndex name). |
| `max_and_components` | int | `2` | Maximum number of single events to combine in one AND composition. Values > 3 are accepted but strongly discouraged (structural overfitting risk). |
| `train_ratio` | float | `1.0` | IS fraction (0 < train_ratio ≤ 1.0). Default 1.0 = all IS, no split. |
| `walk_forward` | WalkForwardConfig \| None | `None` | Walk-forward OOS configuration. Active only when `train_ratio < 1.0`. |
| `diversity_gate_enabled` | bool | `False` | When True, applies Jaccard-based deduplication of single events after the ConsistencyGate and before AND composition. Opt-in — no breaking change. |
| `diversity_threshold` | float | `0.85` | Maximum tolerated Jaccard similarity between any two kept events. Only used when `diversity_gate_enabled=True`. At p99 of the inter-event Jaccard distribution (12 months of 1H data), Jaccard=0.47 — values above 0.70 are genuine near-duplicates. |

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import GateParams, WalkForwardConfig

ed = EventDiscovery(
    enriched,
    config=DiscoveryConfig(
        train_ratio=0.80,
        max_and_components=2,
        gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
        walk_forward=WalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        scale_free_overrides={"rsi_14": True},  # force scale-free on RSI
        diversity_gate_enabled=True,            # opt-in Jaccard deduplication
        diversity_threshold=0.85,
    ),
)
candidates = ed.run()
```

---

## Module 2 — Alpha Discovery

### PromotionThresholds

IS statistical thresholds that contribute to the A–D grade. They do not block
promotion (except in extreme cases) — they inform the grade.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ic_min_abs` | float | `0.02` | Minimum absolute IC (Spearman). Below this threshold the IC metric is recorded as a `[diagnostic]`. |
| `ic_max_p` | float | `0.05` | Maximum p-value for IC significance. |
| `min_lift` | float | `0.08` | Minimum lift (win_rate − base_rate). |
| `min_cohens_d` | float | `0.15` | Minimum Cohen's d (distribution separation between active and inactive bars). |
| `max_p_value` | float | `0.05` | Maximum t-test p-value on the mean advantage. |
| `min_activations` | int | `30` | Minimum IS activations for a contract to be promoted. |
| `use_fdr` | bool | `True` | Apply Benjamini-Hochberg FDR correction across the test family. |
| `fdr_q` | float | `0.10` | Target FDR level (q). |
| `oos_max_p` | float | `0.10` | Maximum OOS confirmation p-value. |
| `min_oos_activations` | int | `10` | Minimum OOS activations to treat OOS confirmation as reliable. |
| `min_direction_t` | float | `0.5` | Minimum `\|z_h*\|` (rotation-standardised excess) to assign a direction; below it → `undetermined`. |
| `require_significant_direction` | bool | `True` | When True, a direction is assigned only if `h*` clears Benjamini-Hochberg (not `statistically_weak`); otherwise → `undetermined`. False = legacy non-blocking behaviour. |

```python
from forgedge import AlphaConfig, PromotionThresholds

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    thresholds=PromotionThresholds(
        ic_min_abs=0.03,
        min_lift=0.10,
        min_cohens_d=0.20,
        min_activations=40,
        oos_max_p=0.05,
    ),
)
```

---

### AlphaConfig

Main configuration for Module 2. Controls the horizon grid, target derivation,
IS/OOS split, and traceability metadata.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `horizon_grid` | tuple[int,...] | `(1,2,3,4,6,8,12,16,24,36,48)` | Horizon grid (in bars) scanned to derive `h*`. |
| `mfe_quantile` | float | `0.5` | Quantile of the MFE distribution of active bars used as baseline `sell_pct`. |
| `mfe_floor` | float | `0.005` | Floor for `sell_pct`: take-profit cannot be < 0.5% regardless of MFE. |
| `train_ratio` | float | `0.7` | IS fraction for statistical measurement. The remaining `1 - train_ratio` is the OOS tail. |
| `thresholds` | PromotionThresholds | `PromotionThresholds()` | IS thresholds for statistical metrics. |
| `asset` | str | `"ASSET"` | Asset name (traceability in AlphaContracts). |
| `exchange` | str | `""` | Exchange/market (optional, traceability). |
| `timeframe` | str | `"1H"` | Timeframe (traceability). |
| `fee_per_side` | float | `0.002` | Fee per side (0.2%), recorded in the contract for Rule Discovery. |
| `close_col` | str | `"close"` | Close price column. |
| `timestamp_col` | str | `"open_dt"` | Datetime column. |
| `regime_col` | str | `"regime"` | Regime column (from Module 0). |
| `regime_stable_col` | str | `"regime_stable"` | Regime-stable column (from Module 0). |
| `use_stable_regime_only` | bool | `False` | When True, excludes bars with `regime_stable=False` from per-regime analysis. |
| `min_regime_obs` | int | `10` | Minimum observations per regime to compute reliable per-regime metrics. |
| `rolling_ic_window` | int \| None | `None` | Rolling IC window size. When None, computed automatically (≈ n/20). |
| `bars_per_day` | float \| None | `None` | Bars per day for Deflated Sharpe computation. When None, derived from the timestamp. |
| `score_weights` | tuple[float,...] | `(0.20, 0.25, 0.15, 0.25, 0.15)` | Composite score weights (IC, lift, Cohen's d, z, regime breadth). A legacy 4-tuple (IC, lift, Cohen's d, breadth) is also accepted. |
| `statistically_weak_penalty` | float | `0.6` | Composite-score multiplier when the target is `statistically_weak`. `1.0` disables it. |
| `oos_bonus` | float | `0.05` | Additive composite-score bonus when the OOS confirmation passes. `0.0` disables it. |
| `discovery_date` | str \| None | `None` | Discovery date (ISO, e.g. `"2026-01-15"`). When None, uses today's date. |
| `fixed_target` | TargetConfig \| None | `None` | When set, **skip target derivation** and measure every candidate against the user's `(horizon, min_return, side)`. The horizon is added to the forward-return grid if absent. |
| `fixed_target_diagnostic` | bool | `True` | Fixed-target mode only: also run the derivation read-only to fill the per-horizon diagnostics and `data_derived_*` convergence fields. `False` = pure bypass. |
| `target_mode` | `"abs"` \| `"proj"` | `"proj"` | Binary-target definition for win rate / lift / base rate. `"proj"` (PROJ_LOG) scores the long forward return in **excess of the local trend** (`log(fwd_max/close) − log(SMA_w/SMA_w[−h]) ≥ log(1+sell_pct)`), so a trend-riding long is not credited with the trend premium — markedly more stable IS→OOS. `"abs"` = legacy absolute return. PROJ applies to long only (short → abs); reverts to abs when history `< (trend_sma_mult+1)·h`. |
| `trend_sma_mult` | float | `2.0` | PROJ_LOG only: trend SMA window = `round(trend_sma_mult·h)` bars. Bar-relative (auto-scales across timeframes); lower tracks the horizon more tightly, higher smooths the trend. |

#### TargetConfig

User-specified economic target for `AlphaConfig.fixed_target` and the TargetOptimizer workflow.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `horizon` | int | — | Holding period in bars (`> 0`). |
| `min_return` | float | — | Take-profit threshold as a fraction (e.g. `0.02` = 2%), used as `sell_pct` (`> 0`). |
| `side` | str | — | `"long"` or `"short"` — never overwritten by the data. |
| `min_activations` | int | `10` | TargetOptimizer: min activations for valid lift scoring. |
| `min_lift` | float | `1.0` | TargetOptimizer: prune threshold on conditional lift. |
| `target_mode` | `"abs"` \| `"proj"` | `"proj"` | Binary-target definition (see `AlphaConfig.target_mode`). |
| `trend_sma_mult` | float | `2.0` | PROJ_LOG trend SMA window multiplier (see `AlphaConfig.trend_sma_mult`). |

```python
from forgedge import AlphaDiscovery, AlphaConfig, PromotionThresholds

ad = AlphaDiscovery(
    ed.df,
    candidates,
    AlphaConfig(
        asset="ADAUSDC",
        timeframe="4H",
        horizon_grid=(4, 8, 12, 24, 48, 72, 96),   # horizons in 4H bars
        train_ratio=0.75,
        fee_per_side=0.001,
        thresholds=PromotionThresholds(
            min_lift=0.08,
            min_cohens_d=0.15,
            min_activations=30,
            oos_max_p=0.10,
        ),
    ),
)
contracts = ad.run()
```

---

## Module 3 — Rule Discovery

### BacktestParams

Execution parameters for a single backtest run: direction, order type,
entry/exit levels, and fee.

> `BacktestParams` is rarely used directly. The grid is defined via `GridSpec`;
> `RuleDiscovery` derives `direction`, `sell_pct`, and `target_h` from the
> AlphaContract.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `direction` | str | `"long"` | Trade direction: `"long"` or `"short"`. Normally derived from the AlphaContract. |
| `buy_type` | str | `"limit"` | Entry order type. In v1.0 only `"limit"` is supported. |
| `buy_drop_pct` | float | `0.010` | Percentage drop below close at which the limit order is placed (1%). |
| `buy_delay_bar` | int | `6` | Maximum number of bars after the event signal in which the limit can be filled. |
| `buy_price_anchor` | str | `"close"` | Column used as anchor for the entry price. |
| `sell_pct` | float | `0.040` | Take-profit as percentage from fill price (4%). |
| `target_h` | int | `24` | Horizon stop in bars: if take-profit is not reached within this number of bars, close at that bar's close. |
| `target_col` | str | `"close"` | Column used to check horizon stop. |
| `target_hit_col` | str | `"close"` | Column used to check take-profit hit. |
| `fee` | float | `0.002` | Per-side fee (0.2%). |
| `early_stopping` | bool | `True` | When True, the grid search stops early when the top-K ranking is stable (compute optimisation). |

---

### ScoringParams

Weights and thresholds used by the grid scoring function (`pf_score_tpm`) to
balance Profit Factor and trading frequency.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pf_min_trades` | int | `15` | Minimum number of trades for a configuration to be included in the ranking. |
| `pf_min_tpm` | int | `2` | Minimum trading frequency (trades/month) for frequency to contribute positively to the score. |
| `pf_tpm_target` | int | `3` | Target trading frequency (trades/month): at this level the frequency score is maximal. |

---

### GridSpec

Defines the parameter search space for the Rule Discovery grid search.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `buy_drop_pct` | list[float] \| None | `None` | List of `buy_drop_pct` values to explore. When None, the default grid is used. |
| `sell_pct` | list[float] \| None | `None` | List of `sell_pct` values to explore. |
| `target_h` | list[int] \| None | `None` | List of horizon values (in bars) to explore. |
| `buy_delay_bar` | list[int] \| None | `None` | List of delay-bar values to explore. |

```python
from forgedge.rule_discovery.models import GridSpec

grid = GridSpec(
    buy_drop_pct=[0.005, 0.010, 0.015],
    sell_pct=[0.030, 0.040, 0.050, 0.060],
    target_h=[12, 24, 36],
    buy_delay_bar=[3, 6],
)
```

---

### WalkForwardConfig (Module 3)

Configures Module 3 walk-forward OOS validation. Distinct from Module 1's
`WalkForwardConfig`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_splits` | int | `4` | Number of rolling train+test windows. |
| `train_span_months` | int \| None | `None` | Training window width in months. When None, computed automatically. |
| `test_span_months` | int \| None | `None` | Test window width in months. When None, computed automatically. |
| `min_train_months` | int | `6` | Minimum months required for the training window. |
| `reoptimise` | bool | `True` | When True, re-optimises parameters on each training window. When False, uses the fixed IS configuration. |

---

### SelectionCriteria

Module 3 promotion gates. Defines the conditions for EDGE, PARTIAL-EDGE, and
NON-EDGE verdicts.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_profit_factor` | float | `2.0` | Minimum IS PF for EDGE. |
| `min_win_rate` | float | `0.55` | Minimum IS win rate for EDGE (55%). |
| `min_trades` | int | `30` | Minimum IS trade count for EDGE. |
| `min_tpm` | float | `2.0` | Minimum average trading frequency (trades/month) for EDGE. |
| `min_pf_score_tpm` | float | `0.30` | Minimum composite PF×TPM score to include a configuration in the selection. |
| `min_fill_rate` | float | `0.40` | Minimum limit order fill rate: at least 40% of events must result in a trade. |
| `partial_min_profit_factor` | float | `1.5` | Minimum IS PF for PARTIAL-EDGE (does not reach EDGE but not NON-EDGE). |
| `max_zero_months_edge` | int | `1` | Maximum zero-or-negative months allowed for EDGE. |
| `max_zero_months_partial` | int | `4` | Maximum zero-or-negative months allowed for PARTIAL-EDGE. |
| `max_regime_dependency` | float | `0.30` | Maximum regime dependency: if > 30% of trades are concentrated in a single regime, this triggers as a soft gate. |
| `min_dsr` | float | `1.0` | Minimum Deflated Sharpe Ratio (corrected for the number of configurations tested). |
| `max_ttest_p` | float | `0.05` | Maximum t-test p-value on mean net gain. |
| `early_elimination` | bool | `True` | When True (default), immediately rejects configurations that fail the fast IS screen (< 20 trades, PF < 1, insufficient fill rate) without running the walk-forward — saves compute. When False, the full pipeline always runs — useful for uniform diagnostics on NON-EDGE rules. |

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig, SelectionCriteria

config = RuleDiscoveryConfig(
    criteria=SelectionCriteria(
        min_profit_factor=1.8,   # less strict for high-volatility assets
        min_win_rate=0.50,
        early_elimination=False, # full diagnostics even on NON-EDGE
    ),
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
```

---

### RuleDiscoveryConfig

Main configuration for Module 3. Aggregates all sub-configurators.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_params` | BacktestParams | `BacktestParams()` | Base backtest parameters (starting point for the grid search). |
| `scoring` | ScoringParams | `ScoringParams()` | Grid scoring weights. |
| `grid` | GridSpec | `GridSpec()` | Grid search space. When all fields are None, the default grid is used. |
| `walk_forward` | WalkForwardConfig | `WalkForwardConfig()` | Walk-forward OOS configuration. |
| `criteria` | SelectionCriteria | `SelectionCriteria()` | EDGE/PARTIAL-EDGE/NON-EDGE verdict criteria. |
| `use_contract_target` | bool | `True` | When True, uses `direction`, `sell_pct`, and `target_h` from the AlphaContract as the grid starting point. |
| `timestamp_col` | str | `"open_dt"` | Datetime column. |
| `signal_col` | str | `"__rule_signal__"` | Internal temporary column for the event signal. |
| `discovery_date` | str \| None | `None` | Discovery date (ISO). |

```python
from forgedge import (
    RuleDiscovery, RuleDiscoveryConfig, BacktestParams,
    SelectionCriteria, WalkForwardConfig,
)
from forgedge.rule_discovery.models import GridSpec, ScoringParams

config = RuleDiscoveryConfig(
    base_params=BacktestParams(fee=0.001),
    grid=GridSpec(
        buy_drop_pct=[0.005, 0.010, 0.015, 0.020],
        sell_pct=[0.030, 0.040, 0.050, 0.060],
        target_h=[12, 24, 36, 48],
        buy_delay_bar=[3, 6],
    ),
    walk_forward=WalkForwardConfig(n_splits=5, min_train_months=8),
    criteria=SelectionCriteria(min_profit_factor=2.0, min_win_rate=0.55),
    scoring=ScoringParams(pf_tpm_target=4),
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
```

---

## Module 4 — Rule Registry

### RegistryConfig

Module 4 configuration. Controls deduplication, genericity classification, and
export.

> `generic_ratio_threshold = 2/3` — the value is exactly 2/3 (not 0.67). On 3
> external tickers: 2 PASS → GENERIC, 1 PASS → PARTIAL.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `overlap_threshold` | float | `0.70` | Jaccard threshold above which two rules are considered duplicates (≥ 70% overlap in activation dates). |
| `gain_corr_threshold` | float | `0.70` | Spearman threshold above which two rules have correlated gains. Used as a secondary metric in the correlation matrix. |
| `cross_pf_threshold` | float | `2.0` | Minimum PF on an external ticker to count as PASS in the cross-ticker backtest. |
| `generic_ratio_threshold` | float | `2/3 ≈ 0.667` | Minimum fraction of external tickers with PASS to classify a rule as GENERIC. PARTIAL if ≥ 1 but < 2/3. |
| `cross_min_active` | int | `10` | Minimum activations on an external ticker to include it in the cross-ticker count. |
| `export_format` | str | `"excel"` | Flat table export format: `"excel"` or `"csv"`. |
| `export_duplicates` | bool | `True` | When True, includes duplicate rules in the exported table (flagged with `is_duplicate=True`). |
| `export_non_generic` | bool | `True` | When True, includes SPECIFIC and ISOLATED rules in the exported table. |
| `html_include_tradelog` | bool | `True` | When True, includes a per-rule trade log in the HTML report. |
| `html_charts` | bool | `True` | When True, generates inline SVG charts (equity curve, heatmap) in the HTML report. |
| `timestamp_col` | str | `"open_dt"` | Datetime column in the provided frames. |
| `session_date` | str \| None | `None` | Session date (ISO). When None, uses today's date. |

```python
from forgedge import RuleRegistry, RegistryConfig

config = RegistryConfig(
    overlap_threshold=0.65,         # more aggressive deduplication
    cross_pf_threshold=1.8,         # less strict for illiquid assets
    generic_ratio_threshold=0.5,    # GENERIC if ≥ 50% of tickers PASS
    export_format="csv",
    html_charts=True,
    export_duplicates=False,         # exclude duplicates from export
)
registry = RuleRegistry.from_forge_results(results, config=config).run()
```

---

## Import summary

> `WalkForwardConfig` imported directly from `forgedge` is the Module 3 (Rule
> Discovery) version. The Module 1 version must be imported from
> `forgedge.event_discovery.models`.

| Class | Import | Module |
|---|---|---|
| `EMAProxyConfig` | `from forgedge import EMAProxyConfig` | 0 — Market Context |
| `MarketContextConfig` | `from forgedge import MarketContextConfig` | 0 — Market Context |
| `GateParams` | `from forgedge.event_discovery.models import GateParams` | 1 — Event Discovery |
| `WalkForwardConfig` (M1) | `from forgedge.event_discovery.models import WalkForwardConfig` | 1 — Event Discovery |
| `DiscoveryConfig` | `from forgedge import DiscoveryConfig` | 1 — Event Discovery |
| `PromotionThresholds` | `from forgedge import PromotionThresholds` | 2 — Alpha Discovery |
| `AlphaConfig` | `from forgedge import AlphaConfig` | 2 — Alpha Discovery |
| `BacktestParams` | `from forgedge import BacktestParams` | 3 — Rule Discovery |
| `ScoringParams` | `from forgedge.rule_discovery.models import ScoringParams` | 3 — Rule Discovery |
| `GridSpec` | `from forgedge.rule_discovery.models import GridSpec` | 3 — Rule Discovery |
| `WalkForwardConfig` (M3) | `from forgedge import WalkForwardConfig` | 3 — Rule Discovery |
| `SelectionCriteria` | `from forgedge import SelectionCriteria` | 3 — Rule Discovery |
| `RuleDiscoveryConfig` | `from forgedge import RuleDiscoveryConfig` | 3 — Rule Discovery |
| `RegistryConfig` | `from forgedge import RegistryConfig` | 4 — Rule Registry |
