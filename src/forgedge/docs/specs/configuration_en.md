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
| `bar_hours` | float \| None | *(resolved: from `timeframe`)* | Explicit candle duration in hours (e.g. `4.0` for 4H bars). Session-resolved; left unset without a session it is inferred from the DatetimeIndex. |
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
| `stable_window` | int | *(resolved: 12h of unchanged regime, min 2 bars)* | Number of consecutive identical bars required for `regime_stable=True`. 12 on 1H, 3 on 4H, 2 on 1D, 48 on 15m — a flat 12 asked for twelve *days* on daily candles. |

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

Since issue #134, the gate operates in one of two counting modes, selected by
`event_counting`: **`"episode"`** (the default) counts maximal runs of
consecutive activations rather than raw bars, so a persistent multi-bar state
(a 3–5 bar `RSI < 30` stretch) is not wrongly penalised as if it were several
independent triggers; **`"bar"`** reproduces the pre-#134 counting exactly,
bar by bar. The two modes are identical for impulse events (crossovers,
candlestick patterns — one bar per episode).

The dual mode changes which dispersion field the gate actually reads
(issue #205): `max_dispersion` is the raw Index-of-Dispersion threshold, but
it is read **only in `"bar"` mode**. In `"episode"` mode the gate instead
compares against `poisson_floor(n_months) × dispersion_margin` — a margin
over the statistically-defensible Poisson floor, not an absolute ID — because
the floor almost always dominated a fixed `max_dispersion` in practice
(measured: 12 of 16 preset×timeframe combinations never had `max_dispersion`
bind). So on the default `event_counting="episode"`, tune `dispersion_margin`,
not `max_dispersion`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_tpm` | float | `0.5` | Minimum average *triggers* per month, in the unit chosen by `event_counting` (episodes/month in `"episode"` mode, bars/month in `"bar"` mode). Default 0.5 ≈ "at least one episode every two months". |
| `max_dispersion` | float | `1.5` | Maximum allowed Index of Dispersion (`Var/Mean` of monthly counts). **`"bar"` mode only** — in `"episode"` mode (the default) this field is not read by the gate at all; see `dispersion_margin`. |
| `dispersion_margin` | float | `1.3` | **`"episode"` mode's dispersion tolerance** — a margin over the Poisson χ² floor (`eff_max_dispersion = poisson_floor(n_months) × dispersion_margin`), not an absolute Index of Dispersion. `1.05` stays close to what a Poisson process itself would produce; `3.0` tolerates Poisson-implausible clustering on purpose. Unread in `"bar"` mode. |
| `event_counting` | `"episode"` \| `"bar"` | `"episode"` | Counting unit for the rate/dispersion criteria — see above. |
| `min_episodes` | int | `10` | Absolute floor on the number of episodes required to pass in `"episode"` mode (statistical-power guard). Ignored in `"bar"` mode, and applied in-sample only. `forge_preset()` lowers this to `5` on `"sweep"` (permissive by design); other presets keep `10`. |
| `episode_gap` | int | `1` | Maximum gap, in bars, that still belongs to the same episode. With the default `1`, a single missing bar inside a run does not start a new episode. `0` gives strict consecutive runs. |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import GateParams

config = DiscoveryConfig(
    gate_params=GateParams(
        min_tpm=0.3,             # less strict for shorter datasets
        dispersion_margin=1.6,   # more slack above the Poisson floor
        min_episodes=5,
        event_counting="episode",  # default; "bar" reproduces pre-#134 behaviour
    )
)
```

> The fields above replaced an older `GateParams(min_act, min_months,
> max_conc, min_tpm)` schema; none of `min_act`/`min_months`/`max_conc` exist
> any more, and constructing `GateParams` with them raises `TypeError`
> today. Several of this repo's own `examples/*.py` scripts still use the old
> schema — see the `forgedge` skill's pitfall list before copying from them.

---

### EventWalkForwardConfig (Module 1)

Configures Module 1 walk-forward OOS validation. Formerly named
`WalkForwardConfig`, which collided with Module 3's distinct class of the
same name; the old name still works as an alias in
`event_discovery.models`, but prefer the explicit one.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_splits` | int | `3` | Number of equal OOS windows on which each event is replayed. |
| `min_pass_rate` | float | `0.6` | Minimum fraction of OOS windows in which an event must pass the gate to be marked OOS-stable. |
| `oos_gate_params` | GateParams \| None | `None` | Gate thresholds for OOS evaluation. When None, thresholds are scaled automatically proportional to the OOS window length relative to IS. |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams

config = DiscoveryConfig(
    train_ratio=0.80,
    walk_forward=EventWalkForwardConfig(
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
| `timestamp_col` | str | `"open_dt"` *(session-resolved)* | Datetime column name in the KPI Table (or DatetimeIndex name). |
| `max_and_components` | int | `2` | Maximum number of single events to combine in one AND composition. Values > 3 are accepted but strongly discouraged (structural overfitting risk). |
| `train_ratio` | float | `1.0` | IS fraction (0 < train_ratio ≤ 1.0). Default 1.0 = all IS, no split. |
| `walk_forward` | EventWalkForwardConfig \| None | `None` | Walk-forward OOS configuration. Active only when `train_ratio < 1.0`. |
| `diversity_gate_enabled` | bool | `False` | When True, applies Jaccard-based deduplication of single events after the ConsistencyGate and before AND composition. Opt-in — no breaking change. |
| `diversity_threshold` | float | `0.85` | Maximum tolerated Jaccard similarity between any two kept events. Only used when `diversity_gate_enabled=True`. At p99 of the inter-event Jaccard distribution (12 months of 1H data), Jaccard=0.47 — values above 0.70 are genuine near-duplicates. |
| `indicator_lag_cross_lags` | tuple[int,...] | `(1, 3)` | Lag set for the indicator × OHLC-base cross-time feature family (issue #165, e.g. `close_sma_12[t] > low[t-3]`), restricted to price-scale indicators (SMA/EMA/WMA/HMA) vs a raw OHLC base. Pass `()` to disable this always-on family entirely. |
| `retain_raw_events` | bool | `True` | Whether `EventDiscovery.raw_events` (the full pre-gate candidate population, every one carrying its own full-length activation series) stays resident after `.run()` (issue #232). Keep `True` for `TargetOptimizer`, which reads `.raw_events` directly; a `forge()`-only config (which never reads it) can set `False` for a measured 4.2x reduction in retained memory. |

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import GateParams, EventWalkForwardConfig

ed = EventDiscovery(
    enriched,
    config=DiscoveryConfig(
        train_ratio=0.80,
        max_and_components=2,
        gate_params=GateParams(min_tpm=0.5, dispersion_margin=1.3, min_episodes=10),
        walk_forward=EventWalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        scale_free_overrides={"rsi_14": True},  # force scale-free on RSI
        diversity_gate_enabled=True,            # opt-in Jaccard deduplication
        diversity_threshold=0.85,
        retain_raw_events=False,                # forge()-only config: skip the memory cost
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
| `ic_min_abs` | float | `0.02` | Minimum absolute IC (Spearman). Below this threshold the IC metric is recorded in `AlphaContract.diagnostics` (non-blocking). |
| `ic_max_p` | float | *(resolved: `ctx.alpha` = 0.05)* | Maximum p-value for IC significance. One of the five per-hypothesis alphas, session-resolved (#182). Feeds a non-blocking diagnostic that only weighs on the grade. |
| `min_lift` | float | `0.08` | Minimum lift (win_rate − base_rate). |
| `min_cohens_d` | float | `0.15` | Minimum Cohen's d (distribution separation between active and inactive bars). |
| `max_p_value` | float | *(resolved: `ctx.alpha` = 0.05)* | Maximum t-test p-value on the mean advantage. Session-resolved (#182), but **reachable only with `use_fdr=False`** — every preset and the class default set `use_fdr=True`, so under any preset this field is inert. |
| `use_fdr` | bool | `True` | Apply Benjamini-Hochberg FDR correction across the test family. |
| `fdr_q` | float | `0.10` | Target FDR level (q). **Not** tied to `ctx.alpha`: a `q` is a false-discovery rate over a family, an alpha is a per-test error rate. Chosen by the preset, because the right `q` depends on how wide the search is (#182). |
| `oos_max_p` | float | `0.10` | Maximum OOS confirmation p-value. **Not** tied to `ctx.alpha`, and legitimately looser: a confirmation level for an already-selected hypothesis — one pre-specified test, no multiplicity, on a small sample by construction (#182). |
| `min_direction_t` | float | `0.5` | Minimum `\|z_h*\|` (rotation-standardised excess) to assign a direction; below it → `undetermined`. |
| `require_significant_direction` | bool | `True` | When True, a direction is assigned only if `h*` clears Benjamini-Hochberg (not `statistically_weak`); otherwise → `undetermined`. False = legacy non-blocking behaviour. |

> `min_activations`/`min_oos_activations` do **not** exist on this dataclass
> any more — the IS/OOS sample-size check is now a hardcoded module constant,
> `_MIN_STATS_CASES = 10`, in `alpha_discovery/discovery.py` (a non-blocking
> diagnostic, not a configurable field). Constructing `PromotionThresholds`
> with either name raises `TypeError` today.

```python
from forgedge import AlphaConfig, PromotionThresholds

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    thresholds=PromotionThresholds(
        ic_min_abs=0.03,
        min_lift=0.10,
        min_cohens_d=0.20,
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
| `horizon_grid` | tuple[int,...] | *(resolved: the session's horizon class)* | Horizon grid (in bars) scanned to derive `h*`. `(1,2,4,8,12,24)` on hourly and 4H, `(1,2,3,5,7,10)` on daily and slower, `(1,2,5,10,20,50)` on sub-hourly — class-calibrated like `BacktestParams.target_h`, not wall-clock converted (#196). It used to be substituted only when no `AlphaConfig` was passed at all, so an explicit config on daily candles scanned up to 48 *days*. |
| `mfe_quantile` | float | `0.5` | Quantile of the MFE distribution of active bars used as baseline `sell_pct`. |
| `mfe_floor` | float | `0.005` | Floor for `sell_pct`: take-profit cannot be < 0.5% regardless of MFE. |
| `train_ratio` | float | `0.7` | IS fraction for statistical measurement. The remaining `1 - train_ratio` is the OOS tail. |
| `embargo_bars` | int | `0` | Extra quarantine after the IS/OOS split: OOS confirmation starts `embargo_bars` bars after the split. Default `0` — the purge already removes the mechanical forward-window overlap; this additionally guards against serial correlation. |
| `horizon_enrichment` | tuple[float,...] \| None | `(0.5, 1.0, 2.0)` | Per-event horizon-grid enrichment from the event's own structural timescale: for every candidate, `round(m · w)` for each multiplier `m` (where `w` is `EventCandidate.dominant_window()`) is **added** to the base `horizon_grid` (union, never a restriction), capped by `horizon_enrichment_min_obs`. `None`/`()` disables enrichment. |
| `horizon_enrichment_min_obs` | int | `20` | Statistical cap for enriched horizons: an added `h` must leave at least this many non-overlapping forward windows in the IS span (`h <= split // min_obs`). Never restricts the base `horizon_grid`. |
| `thresholds` | PromotionThresholds | `PromotionThresholds()` | IS thresholds for statistical metrics. |
| `asset` | str | `"ASSET"` | Asset name (traceability in AlphaContracts). |
| `exchange` | str | `""` | Exchange/market (optional, traceability). |
| `timeframe` | str | `"1H"` | Timeframe (traceability). |
| `fee_per_side` | float | `0.002` *(session-resolved)* | Fee per side (0.2%), recorded in the contract **and charged by the backtest** — it now propagates into `BacktestParams.fee` instead of being an independent copy. |
| `close_col` | str | `"close"` *(session-resolved)* | Close price column. Propagates to `BacktestParams.{target_col, buy_price_anchor}`. |
| `timestamp_col` | str | `"open_dt"` *(session-resolved)* | Datetime column. |
| `regime_col` | str | `"regime"` *(session-resolved)* | Regime column (from Module 0). |
| `regime_stable_col` | str | `"regime_stable"` *(session-resolved)* | Regime-stable column (from Module 0). |
| `use_stable_regime_only` | bool | `False` | When True, excludes bars with `regime_stable=False` from per-regime analysis. |
| `min_regime_obs` | int | `10` | Minimum observations per regime to compute reliable per-regime metrics. |
| `rolling_ic_window` | int \| None | `None` | Rolling IC window size. When None, computed automatically (≈ n/20). |
| `bars_per_day` | float \| None | *(resolved: from `timeframe`)* | Bars per day, sizing the default rolling-IC window. Session-resolved; left unset without a session it is inferred from the timestamps. |
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
| `min_activations` | int | `10` | TargetOptimizer: min activations for valid lift scoring. Candidates firing on fewer bars are skipped (conditional win rate too noisy). Ignored by Alpha Discovery's fixed-target mode. |
| `min_lift_atoms` | float | `1.0` | TargetOptimizer's **1st pass** (atomic events, pre-AND): prune threshold on conditional lift. The "lossless" pruning property holds only at the default `1.0` — values above it actively suppress AND compositions with emergent lift. |
| `min_lift_result` | float | `1.0` | TargetOptimizer's **2nd pass**: prune threshold on the final result set (surviving atoms *and* compositions). Raise it to shorten the result list without touching AND discovery. |
| `min_lift` | float \| None | `None` | **Deprecated** — use `min_lift_atoms`/`min_lift_result` instead. When set, applies to both passes (the legacy single-threshold behaviour) and raises `DeprecationWarning`; a value above `1.0` then also suppresses AND discovery, since the legacy field drove the 1st pass too. |
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
| `buy_delay_bar` | int | *(resolved: 6h of live order)* | Bars the limit order stays live. 6 on 1H, 2 on 4H, 1 on 1D, 24 on 15m — a flat 6 left the order resting six *days* on daily candles. |
| `buy_price_anchor` | str | `"close"` *(session-resolved)* | Column the limit offset is applied to: `buy_price = anchor × (1 ∓ buy_drop_pct)`. **Any numeric column is legal**, including a derived indicator — `buy_price_anchor="close_sma_3", buy_drop_pct=0.10` means "a limit at 90% of the 3-bar SMA". Filled in from `close_col` so that renaming the price column carries the *default* anchor along; an explicit anchor is a reference level of its own and does **not** redefine the session's price column. |
| `sell_pct` | float | `0.040` | Take-profit as percentage from fill price (4%). |
| `target_h` | int | *(resolved: top of the session's horizon class)* | Horizon stop in bars: if take-profit is not reached within this number of bars, close at that bar's close. 24 on hourly, 10 on daily, 50 on sub-hourly — class-calibrated like `horizon_grid`, not wall-clock converted. Normally seeded from the contract's `holding_period_h` before this applies. |
| `target_col` | str | `"close"` *(session-resolved)* | Column used to check horizon stop. Must name the same series `close_col` does; a disagreement is reported. |
| `target_hit_col` | str | `"close"` | Column used to check take-profit hit. |
| `fee` | float | `0.002` *(session-resolved)* | Per-side fee (0.2%), derived from `AlphaConfig.fee_per_side`. |
| `early_stopping` | bool | `True` | When True, the grid search stops early when the top-K ranking is stable (compute optimisation). |

---

### ScoringParams

Thresholds used by the grid scoring function to combine Profit Factor with the
regularity of the trade arrivals. `pf_score_tpm = profit_factor × c_norm`, where
`c_norm` is the inverse index of dispersion (`μ/σ²`, capped at 1) — scale-free,
so a rule is not penalised for trading more often, only for trading in bursts.
"Enough trades" is a separate question, enforced by `criteria.min_tpm` and the
dynamic trade floor below.

Both fields are **session-resolved**: leave them unset and `resolve()` fills
them in, `config_report()` shows the value that will run.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pf_min_trades` | int | *(resolved: `15`)* | Absolute floor of the dynamic trade count `max(pf_min_trades, n_months × pf_min_tpm)` that feeds `pf_score`. |
| `pf_min_tpm` | float | *(resolved: `criteria.min_tpm`)* | Rate floor of that dynamic count. Tracks the gate's own rate — 0.8/month on 1D, 76.8 on 15m — instead of a fixed 2 that agreed with the gate on no timeframe. |

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

### RuleWalkForwardConfig (Module 3)

Configures Module 3 walk-forward OOS validation. Formerly named
`WalkForwardConfig`; the old name still works as an alias, both in
`rule_discovery.models` and at the top level (`forgedge.WalkForwardConfig`
has always resolved to this one).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_splits` | int | `4` | Number of rolling train+test windows. |
| `train_span_months` | int \| None | `None` | Training window width in months. When None, computed automatically. |
| `test_span_months` | int \| None | `None` | Test window width in months. When None, computed automatically. |
| `min_train_months` | int | *derived* *(session-resolved)* | Minimum months required for the training window — the span the early-elimination screen runs on. **Derived from `criteria.min_tpm`** with a 95 % Poisson margin, so the window can actually supply the trade floor it is about to demand (#173): 20 months at `min_tpm=0.80`, not the old fixed 6. The naive `floor / rate` gives 12.5 and comes up short about 44 % of the time. |
| `reoptimise` | bool | `True` | When True, re-optimises parameters on each training window. When False, uses the fixed IS configuration. |
| `purge_bars` | int \| None | `None` | Purge width, in bars, at the end of every **train** window — entries opened in the last `purge_bars` bars have fill/exit windows reaching into the adjacent test window, so parameter selection would otherwise be scored on test prices. `None` (default) sizes the purge automatically from the resolved grid (largest `target_h` plus the fill delay); `0` disables purging (pre-`TimeBudget` behaviour). Deliberately **not** unified with `TimeBudget.purge_bars` (F6, #180) — that one is the forward-return horizon, this one is the worst-case trade span; different crossings of different boundaries. |
| `embargo_bars` | int | `UNSET` *(session-resolved)* | Extra quarantine at the start of every **test** window, in bars. Same policy as `AlphaConfig.embargo_bars` ("how many bars of serial correlation to quarantine after a boundary"), so it is session-resolved from it — an explicit value here still wins. |

---

### SelectionCriteria

Module 3 promotion gates. Defines the conditions for EDGE, PARTIAL-EDGE, and
NON-EDGE verdicts.

Four of these fields are **preset-parametrized** (#207) rather than flat
universal defaults — `forge_preset()` sets them per profile because the
presets' own descriptions explicitly diverge on precision-vs-volume:

| field | class default | `sniper` | `balanced` | `sweep` | `burst` |
|---|---|---|---|---|---|
| `min_profit_factor` | `2.0` | `2.5` | `2.0` | `1.8` | `2.0` |
| `min_win_rate` | `0.55` | `0.60` | `0.55` | `0.50` | `0.55` |
| `min_pf_score_tpm` | `0.30` | `0.40` | `0.30` | `0.25` | `0.30` |
| `min_fill_rate_opt` | `0.80` | `0.80` | `0.80` | `0.70` | `0.80` |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_profit_factor` | float | `2.0` | Minimum IS PF for EDGE. Preset-parametrized — see table above. |
| `min_win_rate` | float | `0.55` | Minimum IS win rate for EDGE (55%). Preset-parametrized — see table above. |
| `min_tpm` | float | `2.0` *(session-resolved)* | Minimum average trading frequency (trades/month) for EDGE. Also the sole trade-count gate: the minimum executed-trade count is dynamic, `max(10, n_months × min_tpm)`, scaling with the IS length (spec RD-04) instead of a fixed absolute threshold. Resolved from `PipelineContext.target_rate_tpm × rate_retention` when M1's rate was declared; otherwise the documented `2.0` stands. |
| `min_pf_score_tpm` | float | `0.30` | Minimum composite PF×TPM score to include a configuration in the selection. Preset-parametrized — see table above. |
| `min_fill_rate` | float | `0.40` | Minimum limit order fill rate: at least 40% of events must result in a trade. **Inert under the default `entry_mode="auto"`** (Stage 1 is a market entry, fill ≈ 100%) — meaningful only under `entry_mode="limit"`; the floor that actually bites under `"auto"` is `min_fill_rate_opt`. |
| `min_sell_pct` | float | `0.005` *(session-resolved)* | Operational floor on the take-profit seeded from the contract's derived target. Resolved from `AlphaConfig.mfe_floor`; it used to be a hardcoded `max(0.01, …)` inside `_seed_base_params`, so the binding constraint was the one the caller could not configure (F11). |
| `min_fill_rate_opt` | float | `0.80` | First adoption condition under `entry_mode="auto"`: the limit point may be published only if it still fills at ≥ this rate **out-of-sample**, avoiding the fill-collapse confound. Preset-parametrized — see table above. |
| `min_net_gain_retention` | float | `0.5` *(session-resolved)* | Third adoption condition: the fraction of the market point's OOS net gain the limit point must retain. Deliberately loose — a backstop against a tiny mu with a tiny sigma, which the Sharpe cannot see because it is scale-free in mu. |
| `partial_min_profit_factor` | float | `1.5` | Minimum IS PF for PARTIAL-EDGE (does not reach EDGE but not NON-EDGE). |
| `min_active_month_rate` | float | `0.80` | Minimum fraction of IS months that must contain at least one trade for a full EDGE: `active_months / n_months >= min_active_month_rate`. Rate-based (replacing the older absolute `max_zero_months_edge`/`max_zero_months_partial`, which no longer exist) so it is timeframe-agnostic: on 1H data the rate is naturally near 1.0, on 1D data a Poisson process with dispersion up to `max_dispersion` yields 0.75–0.95, which the default accommodates. |
| `max_regime_dependency` | float | `0.30` | Maximum regime dependency: if > 30% of trades are concentrated in a single regime, this triggers as a soft gate. |
| `min_dsr` | float | `1.0` | Minimum Deflated Sharpe Ratio (corrected for the number of configurations tested). An undefined DSR (selection haircut's radicand went negative — selection bias too severe to be credible) also blocks a full EDGE. |
| `max_ttest_p` | float | *(resolved: `ctx.alpha` = 0.05)* | Maximum t-test p-value on mean net gain. Session-resolved (#182). The pipeline's **only hard per-hypothesis gate** — it produces `NON-EDGE` in `_decide`. No preset has ever touched it. |
| `max_rotation_p` | float | *(resolved: `ctx.alpha` = 0.05)* | Maximum search-level rotation-null p-value (`AlphaContract.rotation_p`) for a full EDGE — prices the whole discovery surface, so a rule that only won the multiple-testing lottery is capped at PARTIAL-EDGE. Session-resolved (#182); inert when the contract carries no rotation-null annotation. A strict value under `"sweep"` is intentional: `"sweep"`'s upstream permissiveness (`fdr_q=0.25`) is predicated on this gate filtering downstream, paired with `RotationConfig(k>=100)`. |
| `power_gate` | bool | `True` | §3.2 power-aware verdicts: when True, an EDGE/PARTIAL-EDGE is degraded to `INSUFFICIENT-DATA` when the OOS evidence can't support it — no walk-forward was possible, pooled OOS trades below `min_oos_trades`, or the pooled OOS sample's minimum detectable expectancy exceeds the IS expectancy being claimed. `NON-EDGE` verdicts are never rescued. |
| `min_oos_trades` | int | `10` | Minimum pooled walk-forward test trades (across all test windows) for a confident positive verdict under `power_gate`. Below it → `INSUFFICIENT-DATA`. Never applied per window. |
| `early_elimination` | bool | `True` | When True (default), immediately rejects configurations that fail the fast IS screen (< 20 trades, PF < 1, insufficient fill rate) without running the walk-forward — saves compute. When False, the full pipeline always runs — useful for uniform diagnostics on NON-EDGE rules. |

> `max_zero_months_edge`/`max_zero_months_partial` do **not** exist on this
> dataclass any more — replaced by the rate-based `min_active_month_rate`
> above. Constructing `SelectionCriteria` with either name raises
> `TypeError` today.

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
| `walk_forward` | RuleWalkForwardConfig | `RuleWalkForwardConfig()` | Walk-forward OOS configuration. |
| `criteria` | SelectionCriteria | `SelectionCriteria()` | EDGE/PARTIAL-EDGE/NON-EDGE verdict criteria. |
| `entry_mode` | str | `"auto"` | Entry evaluation mode: `"auto"` (default — Stage 1 market entry decides the verdict, Stage 2 sweeps `buy_drop_pct`, replays the winner out-of-sample and publishes it only if it clears all three adoption conditions), `"market"` (next-open baseline alone, fill ≈ 100%, no optimiser) or `"limit"` (the pre-#185 default: the grid optimises `buy_drop_pct`, so the entry doubles as an entry-price optimiser). |
| `use_contract_target` | bool | `True` | When True, uses `direction`, `sell_pct`, and `target_h` from the AlphaContract as the grid starting point. |
| `timestamp_col` | str | `"open_dt"` | Datetime column. |
| `signal_col` | str | `"__rule_signal__"` | Internal temporary column for the event signal. |
| `discovery_date` | str \| None | `None` | Discovery date (ISO). |
| `selection_mode` | `"walk_forward"` \| `"full_sample"` | `"walk_forward"` | Where the published operating point is *selected* (§3.4). `"walk_forward"` (default): the operating point comes from the walk-forward's train windows only (per `wf_param_policy`); no metric feeding the verdict or the published `ValidatedRule` ever reads the final test window. `"full_sample"` falls back to the pre-#217 selection over the whole IS span. |
| `wf_param_policy` | `"last"` \| `"consensus"` | `"last"` | How `selection_mode="walk_forward"` picks the published operating point from the per-split train selections: `"last"` (default) — the most recent train window's winner (what you would trade next); `"consensus"` — the most frequent parameter set across splits, ties broken toward the most recent. |
| `n_trials_upstream` | int | `1` | Multiplier folded into the Deflated-Sharpe `n_trials` on top of the grid-cell count, for an explicit upstream search factor (e.g. sibling contracts receiving a verdict). Default `1` = grid cells only (historical behaviour). |

```python
from forgedge import (
    RuleDiscovery, RuleDiscoveryConfig, BacktestParams,
    SelectionCriteria, RuleWalkForwardConfig,
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
    walk_forward=RuleWalkForwardConfig(n_splits=5, min_train_months=8),
    criteria=SelectionCriteria(min_profit_factor=2.0, min_win_rate=0.55),
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
| `cross_pf_threshold` | float | `1.5` *(session-resolved)* | Absolute PF floor on an external ticker — half of the PASS criterion. Derived from `SelectionCriteria.partial_min_profit_factor`: the bar that admitted the rule at home. Was an independent `2.0`, which excluded every PARTIAL-EDGE rule from genericity by construction. |
| `min_cross_pf_retention` | float | `0.8` *(session-resolved)* | The other half: the fraction of its **home** PF the rule must retain on the external ticker. `PASS ⟺ pf ≥ cross_pf_threshold AND pf ≥ retention × pf_home`. |
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
    cross_pf_threshold=1.8,         # raise the absolute floor for illiquid assets
    min_cross_pf_retention=0.7,     # and tolerate a little more decay away from home
    generic_ratio_threshold=0.5,    # GENERIC if ≥ 50% of tickers PASS
    export_format="csv",
    html_charts=True,
    export_duplicates=False,         # exclude duplicates from export
)
registry = RuleRegistry.from_forge_results(results, config=config).run()
```

---

## Presets — `forge_preset()`

The defaults documented above (`GateParams`, `SelectionCriteria`, and several
`AlphaConfig`/`PromotionThresholds` fields) are routinely overridden as a
group by four named presets, chosen by **search profile**, not by asset:

| Preset | Profile |
|---|---|
| `"sniper"` | Rare/regular events, high precision. Requires a long IS window. Do **not** pair with the rotation calibrator. |
| `"balanced"` | Sensible default — moderate frequency, good IS/OOS balance. |
| `"sweep"` | Wide/permissive search, many candidates. Pair with `rotation_calibration=RotationConfig(k>=100)` and a `min_lift` filter on `promoted_contracts()`. |
| `"burst"` | Time-concentrated events (momentum, regime-change) — high dispersion tolerated on purpose. |

```python
from forgedge import forge, forge_preset

disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="BTC")
result = forge(kpi, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
                rule_discovery_config=rd_cfg)
```

`forge_preset(preset, timeframe, asset="ASSET", train_ratio=0.70, **overrides)`
sets M1/M2/M3's frequency criteria (and several dependent fields) consistently
for the chosen timeframe, and returns the ready-to-use `(DiscoveryConfig,
AlphaConfig, RuleDiscoveryConfig)` triple. Any of the parameters it computes
can be overridden by name via `**overrides`:

| Module | Accepted override keys |
|---|---|
| M1 (Event Discovery) | `min_tpm`, `max_dispersion`, `dispersion_margin`, `min_episodes`, `max_and_components`, `timestamp_col`, `event_counting` |
| M2 (Alpha Discovery) | `min_lift`, `min_cohens_d`, `fdr_q`, `oos_max_p`, `horizon_grid`, `bars_per_day` |
| M3 (Rule Discovery) | `rd_min_tpm`, `min_profit_factor`, `min_win_rate`, `min_pf_score_tpm`, `min_fill_rate_opt` |

`forgedge.presets.preset_info()` prints the resolved numeric parameters for
any (or all) presets, useful for checking what a preset actually resolves to
on a given timeframe before running it. See the `forgedge` skill's pitfall
list for two real, timeframe-specific gotchas: `"1D"` presets can raise
`config_report()`'s `oos_span_too_short` on a modest history (the checker
doing its job, not a broken preset), and `"sniper"` should not be combined
with the rotation calibrator.

---

## Parameter resolution — `UNSET` and the resolver

Configuration fields fall into three states rather than two:

| state | meaning | who writes it |
|---|---|---|
| explicit | the caller chose this value | never overwritten; it is an *input* to derivation |
| `UNSET` | "the resolver decides" | derived from the constraints that relate it to what was set |
| no constraint | neither the preset nor the session has an opinion | the documented class default, exactly as before |

`UNSET` exists because a plain default cannot answer the question a resolver has
to ask: `AlphaConfig.timestamp_col == "open_dt"` looks identical whether the
caller wrote it or inherited it.

```python
from forgedge import UNSET, PipelineContext, collect_context, resolve

bundle = {"event_discovery": disc, "alpha": alpha, "rule_discovery": rd}
ctx = collect_context(bundle, PipelineContext.from_frame(kpi, timeframe="1D"))
resolved, trace, violations = resolve(bundle, ctx)
print(trace.to_text())
```

`resolve()` returns **copies** — inspecting a configuration is never a side
effect on it — and is idempotent. `forge()` does this once at start-up and
exposes the result on `ForgeResult.context` / `.resolution` / `.coherence`.

Derivation reads the timeframe, the schema and the configs' own values; it never
reads the data (`n_bars`, `span_months`), which are visible to the *check* half
only. That keeps `resolve()` total without the frame — which is what lets an
inspection show exactly what will run — and it removes the temptation to cap a
requirement to fit the available history instead of reporting that it does not
fit.

Precedence, strongest first: an explicit `PipelineContext` → `forge()`'s own
arguments → fields set in any config → the class default. Two configs that
disagree on the same value are **reported**, never silently reconciled.

## Import summary

> The two walk-forward configs are now explicitly named:
> `EventWalkForwardConfig` (Module 1) and `RuleWalkForwardConfig` (Module 3),
> both importable from `forgedge`. The legacy `WalkForwardConfig` alias still
> resolves — to the Module 3 class at the top level, and to each module's own
> class inside `event_discovery.models` / `rule_discovery.models`.

| Class | Import | Module |
|---|---|---|
| `EMAProxyConfig` | `from forgedge import EMAProxyConfig` | 0 — Market Context |
| `MarketContextConfig` | `from forgedge import MarketContextConfig` | 0 — Market Context |
| `GateParams` | `from forgedge.event_discovery.models import GateParams` | 1 — Event Discovery |
| `EventWalkForwardConfig` | `from forgedge import EventWalkForwardConfig` | 1 — Event Discovery |
| `DiscoveryConfig` | `from forgedge import DiscoveryConfig` | 1 — Event Discovery |
| `PromotionThresholds` | `from forgedge import PromotionThresholds` | 2 — Alpha Discovery |
| `AlphaConfig` | `from forgedge import AlphaConfig` | 2 — Alpha Discovery |
| `BacktestParams` | `from forgedge import BacktestParams` | 3 — Rule Discovery |
| `ScoringParams` | `from forgedge.rule_discovery.models import ScoringParams` | 3 — Rule Discovery |
| `GridSpec` | `from forgedge.rule_discovery.models import GridSpec` | 3 — Rule Discovery |
| `RuleWalkForwardConfig` | `from forgedge import RuleWalkForwardConfig` | 3 — Rule Discovery |
| `SelectionCriteria` | `from forgedge import SelectionCriteria` | 3 — Rule Discovery |
| `RuleDiscoveryConfig` | `from forgedge import RuleDiscoveryConfig` | 3 — Rule Discovery |
| `RegistryConfig` | `from forgedge import RegistryConfig` | 4 — Rule Registry |
