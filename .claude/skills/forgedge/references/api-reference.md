# forgedge — API reference

Full public API surface (everything exported from `src/forgedge/__init__.py`),
grouped by module, plus the field defaults of every configuration dataclass
that a caller is likely to tune. Read `SKILL.md` first for the pipeline
overview and invariants — this file is the lookup table for exact names,
signatures and defaults when writing code.

All snippets assume `from forgedge import ...` unless a submodule path is
shown explicitly.

For narrative treatment — design rationale, worked examples with verified
output, error handling, troubleshooting, best practices/anti-patterns, an
FAQ and a glossary — see `docs/manual-en.md` (`docs/manuale-it.md` for
Italian). This file only replicates the exact lookup surface: signatures,
dataclass fields and their defaults.

## Table of contents

- [Orchestrator](#orchestrator)
- [Presets](#presets)
- [M0 — Market Context](#m0--market-context)
- [M1 — Event Discovery](#m1--event-discovery)
- [M2 — Alpha Discovery](#m2--alpha-discovery)
- [Search-level calibration](#search-level-calibration)
- [M3 — Rule Discovery](#m3--rule-discovery)
- [M4 — Rule Registry](#m4--rule-registry)
- [KPI Builder](#kpi-builder)
- [Data quality](#data-quality)
- [Time budget (purging / embargo)](#time-budget-purging--embargo)
- [Hypothesis ledger](#hypothesis-ledger)
- [Target-first workflow: TargetOptimizer](#target-first-workflow-targetoptimizer)
- [Rule monitoring / performance reports](#rule-monitoring--performance-reports)
- [Errors and warnings](#errors-and-warnings)

---

## Orchestrator

```
forge(
    kpi_table: pd.DataFrame, *,
    ticker: str | None = None, asset: str = "ASSET", timeframe: str = "1H",
    market_context_config: MarketContextConfig | None = None,
    event_discovery_config: DiscoveryConfig | None = None,
    alpha_config: AlphaConfig | None = None,
    rotation_calibration: RotationConfig | None = None,
    fast_null: bool = True,
    time_budget: TimeBudget | None = None,
    rule_discovery_config: RuleDiscoveryConfig | None = None,
    registry_config: RegistryConfig | None = None,
    manual_events: list[CustomEvent] | None = None,
    run_market_context: bool = True,
    run_rule_discovery: bool = True,
    run_registry: bool = True,
    only_validated_events: bool = False,
    rule_discovery_grades: Iterable[str] | None = None,
    progress: bool = True,
) -> ForgeResult
```

`manual_events` and `event_discovery_config` are mutually exclusive
(`ValueError` if both are set). `ticker` falls back to `alpha_config.asset`
then `asset`. `rotation_calibration`, when set, supersedes the default
`fast_null` pass.

`ForgeResult` fields: `enriched`, `candidates: list[EventCandidate]`,
`contracts: list[AlphaContract]` (promoted *and* rejected — inspect
`contract.rejection_reasons`), `promoted: list[AlphaContract]`,
`rule_responses: list[tuple[AlphaContract, RuleDiscoveryResponse]]`,
`ticker`, `event_frame` (the frame M2/M3/M4 actually read — pass this, not
`enriched`, when building things by hand from a `ForgeResult`), `registry:
RuleRegistry | None`, `market_context`, `event_discovery`, `alpha_discovery`
(live module instances for drill-down), `calibration: CalibrationReport |
None`, `ledger: HypothesisLedger | None`, `time_budget: TimeBudget | None`.

`ForgeResult` methods: `.edges()` → `(contract, response)` pairs with
`response.is_edge` (EDGE or PARTIAL-EDGE); `.validated_rules()` → responses
carrying a non-null `validated_rule`; `.submissions()` → `RuleSubmission`
list ready for the Rule Registry; `.summary()` → `alpha_discovery.summary()`
augmented with a `rule_verdict` column.

```
forge_multi(
    frames_by_ticker: dict[str, pd.DataFrame], *,
    registry_config: RegistryConfig | None = None,
    progress: bool = True,
    **forge_kwargs,
) -> tuple[dict[str, ForgeResult], RuleRegistry]
```

Runs `forge()` once per ticker (`run_registry=False` internally) and pools
every tradeable rule into one cross-ticker `RuleRegistry`. Do not pass
`ticker`/`asset`/`run_registry` in `forge_kwargs` — they're set/overridden
automatically.

## Presets

```
forge_preset(
    preset: str, timeframe: str, asset: str = "ASSET", train_ratio: float = 0.70,
    **overrides,
) -> tuple[DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig]
```

`preset` ∈ `PRESETS` = `["sniper", "balanced", "sweep", "burst"]`. Overrides
accepted by name: M1 — `min_tpm`, `max_dispersion`, `max_and_components`,
`timestamp_col`, `event_counting`; M2 — `min_lift`, `min_cohens_d`, `fdr_q`,
`oos_max_p`, `horizon_grid`, `bars_per_day`; M3 — `rd_min_tpm`. Unknown
override keys raise `TypeError`.

`preset_info(preset: str | None = None) -> None` prints the resolved
parameters for one preset or all of them.

`default_horizon_grid(timeframe: str) -> tuple[int, ...] | None` — the
daily-calibrated grid `forge()` substitutes on daily-or-slower timeframes;
`None` on intraday timeframes (keep `AlphaConfig`'s hourly-calibrated
default) or unparseable strings.

## M0 — Market Context

```
MarketContext(kpi_table: pd.DataFrame, config: MarketContextConfig | None = None)
mc.run() -> pd.DataFrame            # + 'regime' (ordered Categorical), 'regime_stable' (bool)
mc.distribution()                   # bar share per regime, for diagnostics
mc.window_resolution                # {"source": "hurst_ou" | "fallback" | "configured", ...}
```

`EMAProxyConfig` key fields: `auto_window: bool = True`, `window_unit:
"day"|"bar" = "day"`, `window_estimation: float` (OU estimation window),
`bar_hours: float` (explicit when no DatetimeIndex), `stable_window: int =
12`, `threshold_mode: "fixed"|"balanced" = "fixed"`, `threshold_basis:
"fixed"|"expanding"`, `target_distribution: list[float]` (only for
`"balanced"`). Regimes run `STRONG_BEAR → BEAR → NEUTRAL → BULL →
STRONG_BULL`; fallback EMA spans are `9`/`25` when the OU half-life estimate
does not converge.

`RegimeClassifier` — pluggable interface; `EMAProxyClassifier` is the only
shipped implementation (v1.0). Looks for `{source_col}_ema_{period:02d}`
columns and computes them inline if absent.

## M1 — Event Discovery

```
EventDiscovery(kpi_table: pd.DataFrame, config: DiscoveryConfig | None = None,
                time_budget: TimeBudget | None = None)
ed.run() -> list[EventCandidate]
ed.df           # post-pipeline frame with derived features — pass this to AlphaDiscovery, not kpi_table
ed.summary()    # pd.DataFrame, one row per candidate
```

`DiscoveryConfig` fields and defaults (`src/forgedge/event_discovery/discovery.py`):
`gate_params: GateParams = GateParams()`, `max_categorical_classes: int = 20`,
`scale_free_overrides: dict[str, bool] | None = None`, `timestamp_col: str =
"open_dt"`, `max_and_components: int = 2`, `train_ratio: float = 1.0`,
`walk_forward: EventWalkForwardConfig | None = None`, `diversity_gate_enabled:
bool = False`, `diversity_threshold: float = 0.85`, `indicator_lag_cross_lags:
tuple[int, ...] = (1, 3)` (lag set for the price-scale-indicator-vs-lagged-
OHLC-base feature family below; pass `()` to disable that family entirely).

`GateParams` (Consistency Gate, Step 4) — `min_tpm: float = 0.5`,
`max_dispersion: float = 1.5`, `event_counting: "episode"|"bar" = "episode"`,
`min_episodes: int = 10`, `episode_gap: int = 1`. `"episode"` counts maximal
runs of consecutive activations (bridged by gaps ≤ `episode_gap`) rather than
raw bars — the default because a persistent multi-bar state otherwise
inflates monthly-count variance and gets wrongly rejected. `"bar"` reproduces
the pre-#134 behaviour exactly.

**Arity-2 feature pairings beyond same-family ratios.** `FeatureGenerator`
pairs same-family columns (two EMAs, two RSIs, …) by default, plus five
dedicated, narrowly-scoped pairings added to close specific gaps that rule
couldn't reach: (1) cross-column, cross-time OHLC pairs (e.g. "close above
yesterday's low") — always on; (2) a MACD line against its own signal line,
matched by shared `(base, fast, slow)` — only fires when `"macd"` is enabled
in `build_features()` (disabled by default there); (3) price-%-change vs
volume-%-change at the same lookback — only fires if the KPI Table carries a
volume-return column (not part of the default `kpi_builder` config); (4)
`candle_features()`'s six geometry columns against each other and against
`close_natr_N` (never the raw `atr`) — these bare-named columns don't match
the `{base}_{indicator}_{period}` convention, so without this pairing they'd
only ever be standalone; (5) a price-scale indicator (SMA/EMA/WMA/HMA only)
against a lagged raw OHLC base (e.g. `close_sma_12[t] > low[t-3]`) — always
on, governed by `DiscoveryConfig.indicator_lag_cross_lags` above, and the one
with a measured, non-trivial runtime cost (+24% `EventDiscovery.run()` time /
+21% candidate count on a 36-indicator-column fixture). A column that opts
out of the generic same-family grouping (naming-convention mismatch) can
still be reached by one of these five. See `docs/manual-en.md` §8/§17 for the
full narrative and the measured cost breakdown.

`EventWalkForwardConfig` (M1; legacy alias `WalkForwardConfig` in
`event_discovery.models`) — `n_splits:
int = 3`, `min_pass_rate: float = 0.6`, `oos_gate_params: GateParams | None =
None` (defaults to the IS `gate_params`). With this set and `train_ratio <
1.0`, each candidate exposes `.validation: ValidationResult` with `.passed`,
`.pass_rate`, `.fold_results`.

`EventCandidate` — key attributes: `event_id`, `expression` (boolean string),
`event_formula` (human-readable), `sql_expression` (DuckDB/SQL), `components:
list[EventComponent]`, `activation_stats: ActivationStats`
(`n_activations`, `n_active_months`, `zero_months`, `max_monthly_share`,
`mean_tpm`), `consistency_gate: GateResult`, `validation: ValidationResult |
None`. Method `.apply(df) -> pd.Series[bool]` — deterministic, no look-ahead,
re-evaluates the stored thresholds on any new frame; this is the path
`forgedge` itself always uses and is correct for every candidate, including
the arity-2 pairings above. `.persist(path)` — full pickle round-trip
(components, thresholds, activation stats, validation). Caveat:
`sql_expression` for a lag-cross feature (pairings 1/5 above) combined with a
rolling pctrank/zscore transform emits a best-effort SQL translation
containing a nested window function some engines (including DuckDB) reject —
treat it as a convenience export, not a guaranteed-portable one, for that
specific combination; use `.apply()` if you need certainty.

`CustomEvent(formula: str, name: str = "")` — manual event injection, either
standalone (`.apply(df)`, `.to_event_candidate(df, gate_params=...)`) or via
`forge(..., manual_events=[...])`. Still crosses the Consistency Gate (a
failure logs a warning but does not drop the event); AND composition is
skipped in this mode.

## M2 — Alpha Discovery

```
AlphaDiscovery(kpi_table_or_ed_df: pd.DataFrame, candidates: list[EventCandidate],
                config: AlphaConfig, time_budget: TimeBudget | None = None)
ad.run() -> list[AlphaContract]
ad.promoted_contracts(min_lift: float | None = None) -> list[AlphaContract]
ad.summary()   # pd.DataFrame sorted by composite score
```

`AlphaConfig` fields and defaults (`src/forgedge/alpha_discovery/models.py`):
`horizon_grid: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48)`
(hourly-calibrated — see SKILL.md pitfall #2 for daily-or-slower data),
`mfe_quantile: float = 0.5`, `mfe_floor: float = 0.005`, `train_ratio: float
= 0.7`, `embargo_bars: int = 0`, `horizon_enrichment: tuple[float, ...] |
None = (0.5, 1.0, 2.0)` (adds horizons at 0.5×/1×/2× each event's dominant
indicator window — union, statistically capped, never a restriction),
`horizon_enrichment_min_obs: int = 20`, `thresholds: PromotionThresholds =
PromotionThresholds()`, `asset: str = "ASSET"`, `timeframe: str = "1H"`,
`fee_per_side: float = UNSET` (0.002), `close_col: str = UNSET` (`"close"`),
`timestamp_col: str = UNSET` (`"open_dt"`), `regime_col: str = UNSET`
(`"regime"`), `regime_stable_col: str = UNSET` (`"regime_stable"`) — all five
session-resolved, resolved default in brackets, `use_stable_regime_only: bool = False`, `min_regime_obs:
int = 10`, `bars_per_day: float | None = None`, `fixed_target: TargetConfig
| None = None` (fixed-target / `TargetOptimizer` mode), `target_mode:
"abs"|"proj" = "proj"`, `trend_sma_mult: float = 2.0`.

`PromotionThresholds` — `ic_min_abs: float = 0.02`, `ic_max_p: float = 0.05`,
`min_lift: float = 0.08`, `min_cohens_d: float = 0.15`, `max_p_value: float =
0.05` (used only when `use_fdr=False`), `use_fdr: bool = True`, `fdr_q: float
= 0.10`, `oos_max_p: float = 0.10`, `min_direction_t: float = 0.5`,
`require_significant_direction: bool = True`. **The only hard rejection gate
is undetermined direction** ("no derivable target" — no horizon produces a
finite advantage); every other metric here feeds the A–D grade, not a
pass/fail gate.

`target_mode="proj"` (default, long events only) scores the binary target as
excess return over a local trend SMA (`window = round(trend_sma_mult * h)`)
rather than the raw forward return (`"abs"`), so trend drift isn't credited
as edge; shorts always use `"abs"`.

`AlphaContract` — key attributes: `alpha_id`, `status: "HYPOTHESIS"|
"REJECTED"`, `event_candidate_id`, `event_expression`, `derived_target:
DerivedTarget` (`holding_period_h`, `sell_pct`, `direction`, `base_rate`,
`mean_advantage`), `oos_validation: OOSValidation` (`passed`, `lift`,
`p_value`, `n_bars`), `event_stats: EventStats` (`win_rate`, `lift`,
`cohens_d`, `p_value`, …), `regime_analysis: RegimeAnalysis`
(`dependency_type: "agnostic"|"conditional"|"specific"|"broken"|"unknown"`),
`alpha_score: AlphaScore` (`composite_score`, `grade: "A"|"B"|"C"|"D"`),
`rejection_reasons: list[str]` (blocking causes only — empty on a promoted
contract; in practice just "no derivable target"), `diagnostics: list[str]`
(non-blocking observations that feed the grade — weak IC/lift/Cohen's d, FDR,
thin IS/OOS samples; routinely non-empty on promoted contracts), `rotation_p` /
`rotation_threshold`
(set by the search-level rotation null, see below). `.to_contract_dict()`
for JSON/YAML export.

## Search-level calibration

```
FastRotationNull(event_frame, candidates, alpha_config, time_budget=None).run(promoted) -> CalibrationReport
RotationCalibrator(event_frame, candidates, alpha_config, time_budget=None).run(promoted, RotationConfig(...)) -> CalibrationReport
```

`forge()` runs `FastRotationNull` by default (`fast_null=True`) — exact null
distribution of the best standardised excess over every circular offset, via
FFT, `abs_z` yardstick only, ~seconds even on thousands of candidates.
`RotationCalibrator` is the heavier, sampled (`RotationConfig(k=...)`)
multi-yardstick alternative (`composite`, `is_lift`, … combined via Tippett
min-p) — pass `rotation_calibration=RotationConfig(k=100)` to `forge()` to
use it inline instead (supersedes `fast_null`), or run it standalone against
`ForgeResult.event_frame` / `.candidates` for the full report without
slowing the main run.

`RotationConfig` — `k: int = 100`, `alpha: float = 0.05`, `seed: int =
20260624`, `in_sample_stats: tuple[str, ...]`.

`CalibrationReport` — `tippett_p`, `tippett_best_stat`, `per_stat_p`,
`null_q`, `real_stats`, `null_arrays`, `survivors` (promoted contracts whose
statistic clears the null bar), `.summary()`.

Rule Discovery requires `rotation_p <= criteria.max_rotation_p` (default
`0.05`) for a full `EDGE` verdict; otherwise the verdict is capped at
`PARTIAL-EDGE` (still tradeable — `resp.is_edge`).

## M3 — Rule Discovery

```
RuleDiscovery(event_frame: pd.DataFrame, contract: AlphaContract, candidate: EventCandidate,
               config: RuleDiscoveryConfig | None = None)
resp = rd.run() -> RuleDiscoveryResponse
```

`RuleDiscoveryConfig` fields and defaults (`src/forgedge/rule_discovery/models.py`):
`base_params: BacktestParams`, `scoring: ScoringParams`, `grid: GridSpec`
(auto-built around the contract target when empty), `walk_forward:
RuleWalkForwardConfig`, `criteria: SelectionCriteria`, `entry_mode: "limit"|
"market"|"auto" = "limit"`, `use_contract_target: bool = True`,
`timestamp_col: str = "open_dt"`, `selection_mode: "walk_forward"|
"full_sample" = "walk_forward"` (operating point selected inside WF train
windows only — the final test window is never read by any selection),
`wf_param_policy: str = "last"`, `n_trials_upstream: int = 1`.

`BacktestParams` — `direction: str = "long"`, `buy_type: str = "limit"`,
`buy_drop_pct: float = 0.010`, `buy_delay_bar: int = 6`, `buy_price_anchor:
str = UNSET` (`"close"`), `sell_pct: float = 0.040`, `target_h: int = 24`,
`target_col: str = UNSET` (`"close"`), `target_hit_col: str = "close"`, `fee:
float = UNSET` (0.002), `early_stopping: bool = True`.  The three `UNSET`
fields are session-resolved — `fee` from `AlphaConfig.fee_per_side`, the two
columns from `close_col`; `target_hit_col` is not, because it names an exit
*convention* rather than a schema fact.  `BacktestParams.resolved()` applies
the documented defaults, and `run_backtest` calls it, so a hand-built params
object handed straight in behaves exactly as before.

`RuleWalkForwardConfig` (M3; legacy alias `WalkForwardConfig`, which is also
what top-level `forgedge.WalkForwardConfig` resolves to) — `n_splits: int
= 4`, `train_span_months: int | None = None` (`None` = anchored/expanding
train window), `test_span_months: int | None = None`, `min_train_months: int
= 6`, `reoptimise: bool = True`, `purge_bars: int | None = None` (`None`
defaults to the horizon under test), `embargo_bars: int = 0`.

`SelectionCriteria` — `min_profit_factor: float = 2.0`, `min_win_rate: float
= 0.55`, `min_tpm: float = 2.0` (sole frequency gate — the executed-trade
floor is `max(10, n_months * min_tpm)`, not a fixed constant), `min_fill_rate:
float = 0.40`, `min_fill_rate_opt: float = 0.80` (floor for the
`entry_mode="auto"` limit-optimisation stage), `min_dsr: float = 1.0`,
`max_ttest_p: float = 0.05`, `max_rotation_p: float = 0.05`, `power_gate:
bool = True` (demotes to `INSUFFICIENT-DATA` when pooled OOS evidence can't
support the verdict), `min_oos_trades: int = 10`, `early_elimination: bool =
True` (set `False` to force the full walk-forward/diagnostics pipeline even
on a fast-screened `NON-EDGE`).

`entry_mode`: `"limit"` (default, grid optimises `buy_drop_pct`, can suffer
the "fill confound"), `"market"` (baseline at next-open, ≈100% fill,
isolates the signal's edge), `"auto"` (two-stage: market-mode verdict is
authoritative, limit optimiser only refines EDGE/PARTIAL-EDGE survivors that
still clear `min_fill_rate_opt`) — records `resp.entry_optimization`.

`RuleDiscoveryResponse` — `verdict: "EDGE"|"PARTIAL-EDGE"|"NON-EDGE"|
"INSUFFICIENT-DATA"`, `is_edge: bool` (True for EDGE/PARTIAL-EDGE),
`rejection_reasons: list[str]`, `validated_rule: ValidatedRule | None`
(`.params: BacktestParams`), `in_sample_summary: BacktestSummary`
(`total_trades`, `profit_factor`, `win_rate_pct`, `expectancy`, `tpm_mu`),
`execution_envelope: ExecutionEnvelope | None` (`.conservative` /
`.optimistic`), `walk_forward: WalkForwardResult | None`
(`.oos_summary`, `.consistency`), `statistical_validation:
StatisticalValidation | None` (`.temporal_stability: "PASS"|"WARN"|"FAIL"`,
`.deflated_sharpe`), `regime_analysis: RegimeBreakdown | None`
(`.avoid_in`), `excursion: ExcursionStats | None` (MAE/MFE), `grid_results:
list[GridResult]`.

`from forgedge.rule_discovery import text_report, html_report` — human /
HTML report builders from a `RuleDiscoveryResponse`. `resp.to_dict()` for
JSON export.

## M4 — Rule Registry

```
RuleRegistry(submissions: list[RuleSubmission], frames: dict[str, pd.DataFrame],
              config: RegistryConfig | None = None)
RuleRegistry.from_forge_results(results: dict[str, ForgeResult], config=None)  # preferred entry point
reg = registry.run()
reg.summary(); reg.flat_table(); reg.documents; reg.matrices  # .jaccard, .spearman
reg.export("rules.xlsx")            # or .csv, per RegistryConfig.export_format
reg.html_report(timeframe="1H")     # self-contained HTML, inline SVG
```

`RegistryConfig` fields and defaults (`src/forgedge/rule_registry/models.py`):
`overlap_threshold: float = 0.70` (Jaccard ≥ this → duplicate, weaker PF
flagged), `gain_corr_threshold: float = 0.70` (reporting only),
`cross_pf_threshold: float = UNSET` (1.5 — absolute PF floor for a
cross-ticker `PASS`, session-resolved from
`SelectionCriteria.partial_min_profit_factor`),
`min_cross_pf_retention: float = UNSET` (0.8 — fraction of the rule's *home*
PF it must retain), `generic_ratio_threshold: float = 2/3`, `cross_min_active: int = 10`,
`export_format: "excel"|"csv" = "excel"`, `export_duplicates: bool = True`,
`export_non_generic: bool = True`, `html_include_tradelog: bool = True`,
`html_charts: bool = True`, `timestamp_col: str = "open_dt"`.

`RuleDocument` — `rule_id`, `expression`, `source_ticker`,
`source_alpha_id`, `verdict`, `grade`, `is_duplicate`, `duplicate_of`,
`cross_ticker: dict[str, CrossTickerResult]`, `cross_ticker_score`,
`is_generic`, `classification: "GENERIC"|"PARTIAL"|"SPECIFIC"|"ISOLATED"`.

`CrossTickerResult` — `ticker`, `expression_adapted`, `pf`, `win_rate`,
`total_trades`, `zero_months`, `verdict`, `bar` (the PF this rule had to reach
on this ticker).

Classification: a target ticker is a `PASS` when the rule clears **both**
halves of the transfer criterion —
`pf >= cross_pf_threshold AND pf >= min_cross_pf_retention * pf_home` — and the
rule is `GENERIC` when it PASSes on ≥ `generic_ratio_threshold` of the other
tickers it was replayed on (with thresholds recalibrated on each ticker's local
distribution — the rule's logical structure stays fixed); otherwise
`PARTIAL`/`SPECIFIC`/`ISOLATED`.  The absolute half asks *is it tradeable
there*, the relative half asks *does it transfer*: a single bar at 2.0 failed
every `PARTIAL-EDGE` rule (admitted at 1.5) on every ticker, and passed a rule
that had lost a third of its edge.

`RuleSubmission(ticker, response, candidate, grade=None)` — manual
construction path when not orchestrating via `forge()`/`forge_multi()`.

## KPI Builder

```
from forgedge import build_features, candle_features, lag_features, pattern_features

build_features(candles, config=None, *, timestamp_col, output_timestamp_col="open_dt",
                timestamp_unit="ms", add_color=True, sort_output=True) -> pd.DataFrame
candle_features(df, *, order_on="open_dt", add_gap=True, round_to=5) -> pd.DataFrame
lag_features(df, *cols, periods=(1, 2, 3), like=None, order_on="open_dt") -> pd.DataFrame
pattern_features(df, *, patterns=None, order_on="open_dt", col="candle_pattern") -> pd.DataFrame
```

`build_features` — `config` is a dict, a YAML path, or `None` for the
packaged default (`src/forgedge/kpi_builder/default_enricher.yaml`; copy and
edit to change periods/columns/enabled indicators). Indicators referencing
columns absent from `candles` are skipped with a warning. `"atr"` and
`"macd"` ship **disabled by default** — enable explicitly in `config`.

Column naming: for family/ratio recognition in Event Discovery, a column
must match `{base}_{indicator}_{period}` with `base ∈ {close, high, low,
open, volume}` and `indicator ∈ {ema, sma, rsi, dema, tema, wma, hma, mdd,
atr, natr}`, or the `{base}_bb_{lower|upper|width|mid}_{period}` /
`{base}_{vol|ret}_{period}` patterns. Non-conforming names still work as
standalone features.

`candle_features` — adds `body`, `upper_wick`, `lower_wick`, `close_pos`,
`range_pct`, `gap` (scale-free geometry in `[-1,1]`/`[0,1]`) — preferred over
`pattern_features` for automatic discovery since FORGE derives its own
asset-adaptive thresholds on continuous geometry instead of fixed pattern
definitions.

`pattern_features` — opt-in, adds one categorical `candle_pattern` column
(ten named formations, e.g. `"HAMMER"`, `"DOJI"`); flows through
`forge()` as one-hot events scored via point-biserial IC.

All three sorting functions sort by `order_on` internally — input need not
be pre-sorted, except a `pattern_features()` call on data lacking both
`open_dt` and a `DatetimeIndex` raises `KeyError`.

## Data quality

```
summary_report(df, *, timestamp_col="open_dt", price_cols=("open","high","low","close"),
                timeframe=None, return_high_move=0.5, top_n=5, verbose=True,
                return_report=False) -> DataQualityReport | None
```

Never raises or blocks the pipeline — checks schema/NaNs/infinities,
price-scale consistency, OHLC internal consistency, return outliers (MAD
z-score), time continuity (gaps, duplicate/out-of-order timestamps). Each
`Finding` has `.level: "OK"|"WARN"|"FAIL"`, a stable `.code`
(`"scale_mixed"`, `"ohlc_hl"`, `"gaps"`, …), `.message`. `DataQualityReport`
exposes `.worst`, `.has_critical`, `.has_warnings`, `.one_line()`,
`.to_text()`, `.findings` (full list).

## Parameter resolution (`UNSET`, `PipelineContext`, `resolve`)

`forgedge.resolver` is the single place where the pipeline's *latent parameters*
— quantities with one meaning for the whole pipeline, materialised as several
independent config fields — are named, propagated and checked. It is a
constraint propagator, not a defaulting mechanism.

`UNSET` (`from forgedge import UNSET`) separates "the caller chose this" from
"class default". A field left at `UNSET` means *the resolver decides*; any other
value, including one equal to the historical default, is a choice and is
**never** overwritten. `UNSET` is falsy, is not arithmetic (`UNSET * 2` raises,
so an unresolved value reaching a computation fails loudly), and survives
`copy`/`deepcopy`/`pickle` as the same object.

`PipelineContext` — session facts no module owns individually: `timeframe`,
`timestamp_col`, `close_col`, `regime_col`, `regime_stable_col`, `fee_per_side`,
`alpha`, `min_sample`, `target_rate_tpm`, `cross_pf_retention`, plus the data facts `n_bars` /
`span_months`. Bar arithmetic via `bars_per_day` / `bars_per_month` /
`bar_hours` / `months_of()` / `bars_of()`. Build with
`PipelineContext.from_frame(kpi, timeframe="1D")`.

`resolve(bundle, ctx, active_stages=(PROPAGATION,)) -> (bundle, trace,
violations)`. The bundle is `{"market_context", "event_discovery", "alpha",
"rule_discovery", "registry"}`; missing entries are skipped. **Returns copies —
the caller's configs are never mutated.** Idempotent. Two modes off one table:
a field at `UNSET` is *derived*; a field that is set is left alone and the
relation is *checked* instead, producing a `Violation` that
`config_report()` (#176) renders.

**Derivation never reads the data.** `n_bars` / `span_months` are visible to
check mode only — otherwise a resolver could cap a requirement to fit the
available span, silently weakening the gate the caller asked for.

`collect_context(bundle, base, **overrides)` seeds the context from fields the
caller set explicitly (precedence: overrides → explicit context → config
fields → class default). This is what makes a `timestamp_col` set on one module
reach the other three.

`ResolutionTrace` / `Derivation` — ordered record of every field derived, with
`default → resolved`, the rule that fired, the inputs it read, and `superseded`
when a more conservative derivation won. `trace.describe()` for one line,
`trace.to_text()` for the table. Carried on `ForgeResult.resolution`;
`ForgeResult.context` and `ForgeResult.coherence` alongside it.

`resolve_config(cfg, kind, ctx=None)` — single-config path; every module's
constructor calls it, so "what you inspect is what runs" holds for hand-built
pipelines too.

`poisson_min_window(floor, rate, confidence=0.95)` — smallest window reaching
`floor` with the given probability under Poisson arrivals, ≈ `1.65 × floor /
rate`. The naive `floor / rate` satisfies the constraint *in expectation* only:
at floor=10, rate=0.8 it gives λ=10.4, under which fewer than 10 events occur
~44 % of the time.

Currently only the `PROPAGATION` stage derives (schema columns). `STATISTICAL`
constraints are registered and checked; each has its `derive` switched on by the
issue that owns the fix (#177, #178, #181, #182).

## Time budget (purging / embargo)

```
TimeBudget.build(n_bars: int, train_ratio: float = 0.7, horizon_bars: int = 0,
                  purge_bars: int | None = None, embargo_bars: int = 0) -> TimeBudget
```

`purge_bars` defaults to `horizon_bars` when omitted. Passed to `forge()` via
`time_budget=...`, threaded into both `EventDiscovery` and `AlphaDiscovery`
so they share one IS/OOS split; purging removes IS rows whose forward
window crosses into OOS (on by default for Alpha Discovery — purge width =
`max(horizon_grid)` — and for Rule Discovery's walk-forward). To reproduce
pre-purging numbers exactly, pass `purge_bars=0` explicitly (and
`RuleWalkForwardConfig(purge_bars=0)` for M3). Embargo is `0`/opt-in everywhere.
`result.time_budget.describe()` for a human-readable summary.

## Hypothesis ledger

`HypothesisLedger` (`result.ledger`) — plain bookkeeping, not a correction
mechanism (that's the rotation null): `m1_candidates`, `m2_horizons`,
`m2_promoted`, `m2_return_tests`, `m3_grid_cells`, `m2_surface` (=
`m1_candidates * m2_horizons`), `total_surface` (upper bound). `.describe()`
for a one-line human summary.

## Target-first workflow: TargetOptimizer

Standalone module — does not touch `forge()`/`ForgeResult`. Inverts the
standard event-first flow: fixes the economic target up front and searches
for events that best predict it.

```python
from forgedge import TargetOptimizer, TargetConfig

opt = TargetOptimizer(train_df, TargetConfig(horizon=20, min_return=0.10, side="long"))
results = opt.run()          # pd.DataFrame, sorted by lift desc
cands = opt.candidates       # list[EventCandidate], aligned with results
oos = opt.validate_oos(full_df, top_k=10)
contracts = opt.discover_alpha()   # list[AlphaContract], fixed-target mode
```

`TargetConfig` — `horizon: int` (required), `min_return: float` (required,
fraction), `side: "long"|"short"` (required), `min_activations: int = 10`,
`min_lift_atoms: float = 1.0` (1st-pass prune, before AND composition),
`min_lift_result: float = 1.0` (2nd-pass prune, on the final result set),
`target_mode: "abs"|"proj" = "proj"`, `trend_sma_mult: float = 2.0`.
`TargetOptimizer(train_df, target_cfg, discovery_cfg=None)` — `discovery_cfg`
defaults to `DiscoveryConfig(train_ratio=1.0)`.

## Rule monitoring / performance reports

```python
from forgedge import RuleSpec, rule_performance_report

specs = RuleSpec.from_forge_result(result)          # one spec per tradeable rule
html = rule_performance_report(result, fresh_candles)   # or rule_performance_report(specs, fresh_candles)
```

Replays each rule deterministically via the same `EventCandidate.apply()`
path Rule Discovery uses, so `fresh_candles` need not be the discovery
table. The HTML report includes: equity vs buy & hold (dual, independent
axes), monthly activation trend split IS/OOS, gain/loss distribution, return
KDEs (low/close/high forward returns at `target_h`), base-vs-event MAE→net
scatter, rolling expectancy (edge-decay detector), per-regime performance,
most recent trades, and a "signal active now" badge.

## Errors and warnings

No custom exception hierarchy — every raised error is a plain Python
built-in. There is no up-front schema validation of the KPI Table either:
each module validates only the columns/timestamp source it needs, exactly
when it needs them (the one exception is `build_features()`, which silently
*skips*, with `logger.warning`, any indicator whose required input columns
are absent — OHLC-only candles are always safe to pass). `summary_report()`
is the opt-in way to validate eagerly; it never raises on its own.

| Exception | Typical trigger |
|---|---|
| `ValueError` | invalid enum-like string (`direction`, `target_mode`, `buy_type`, `entry_mode`, `selection_mode`, `threshold_mode`, `timeframe`, `preset`, …), out-of-range numeric config field, mutually-exclusive `forge()` arguments (`manual_events` + `event_discovery_config`), a candidate/contract pair passed to `RuleDiscovery` that don't reference each other |
| `KeyError` | a required column is missing — OHLC columns, `timestamp_col`, `source_col`, an unknown candlestick pattern name |
| `RuntimeError` | an accessor called before `.run()` — consistently, across `MarketContext.distribution()`, `EventDiscovery.summary()`, `AlphaDiscovery.summary()`/`.promoted_contracts()`, `RuleDiscovery.grid_summary()`, `TargetOptimizer.validate_oos()`/`.discover_alpha()` |
| `TypeError` | wrong input type to `build_features`/`lag_features`, an unrecognized `forge_preset(**overrides)` key, or a `GateParams` call using the old field names (`min_act`/`min_months`/`max_conc`, still present in several stale `examples/*.py` scripts) |
| `ImportError` | `load_kpi_config()` given a YAML path but PyYAML isn't installed |
| `FileNotFoundError` | `load_kpi_config()` given a path that doesn't exist |

Warnings raised via `warnings.warn` (not exceptions), worth not silencing:

- **`UserWarning` — stale hourly `horizon_grid` on daily-or-slower data** —
  `forge()` fires this when an explicit `AlphaConfig` still carries the
  untouched hourly default grid on a `timeframe` of a day or longer.
- **`UserWarning` — observed-candle index mismatch** — `AlphaDiscovery`,
  `RuleDiscovery`, and Rule Registry ingestion fire this when the frame they
  receive has a different index than an event's cached training activation
  series (the event falls back to `.apply()` re-evaluation). A stronger
  variant fires when the re-evaluated activation count collapses under ~10%
  of the training count — a strong signal of an imminent spurious
  `direction="undetermined"` (extend training data with
  `pd.concat([train_df, new_bars])`, not `new_bars` alone).
- **`DeprecationWarning`** — the legacy `TargetConfig.min_lift` field
  (superseded by `min_lift_atoms`/`min_lift_result`) and a legacy
  `TypeClassifier` constructor argument (`scale_free_drift_threshold`).

Degraded-but-non-fatal behaviour (neither raises nor warns, only logs at
INFO/DEBUG or shows up as a diagnostic string): a `CustomEvent` that fails
the Consistency Gate is kept, not dropped; `target_mode="proj"` reverts to
`"abs"` when there's not enough history for the trend-SMA warmup;
`RuleDiscoveryConfig(selection_mode="walk_forward")` silently falls back to
full-sample selection when the data span is too short for even one
walk-forward split.
