# forgedge — API reference

Full public API surface (everything exported from `src/forgedge/__init__.py`),
grouped by module, plus the field defaults of every configuration dataclass
that a caller is likely to tune. Read `SKILL.md` first for the pipeline
overview and invariants — this file is the lookup table for exact names,
signatures and defaults when writing code.

All snippets assume `from forgedge import ...` unless a submodule path is
shown explicitly. Every default below was read directly from
`src/forgedge/**/models.py` (not transcribed from the manual) — where a field
now defaults to the `UNSET` sentinel, that is stated explicitly, along with
the **fallback** value used when the field is resolved with no session
context (standalone module use, no `PipelineContext`).

For narrative treatment — design rationale, worked examples with verified
output, error handling, troubleshooting, best practices/anti-patterns, an
FAQ and a glossary — see `docs/manual-en.md` (`docs/manuale-it.md` for
Italian) and `docs/analysis/pipeline_parameter_coherence.md` (the resolver's
design doc). This file only replicates the exact lookup surface.

## Table of contents

- [Orchestrator](#orchestrator)
- [Presets](#presets)
- [Configuration resolution and coherence](#configuration-resolution-and-coherence)
- [M0 — Market Context](#m0--market-context)
- [M1 — Event Discovery](#m1--event-discovery)
- [M2 — Alpha Discovery](#m2--alpha-discovery)
- [Search-level calibration](#search-level-calibration)
- [M3 — Rule Discovery](#m3--rule-discovery)
- [Episodes and concurrency](#episodes-and-concurrency)
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
    ticker: str | None = None, asset: str = "ASSET", timeframe: str = UNSET,
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
    strict: bool = True,
    rule_discovery_grades: Iterable[str] | None = None,
    progress: bool = True,
) -> ForgeResult
```

`manual_events` and `event_discovery_config` are mutually exclusive
(`ValueError` if both are set). `ticker` falls back to `alpha_config.asset`
then `asset`. `rotation_calibration`, when set, supersedes the default
`fast_null` pass.

`timeframe` defaults to the `UNSET` sentinel, not the literal `"1H"` — an
**inherited** default is deliberately distinct from a **declared** one:
`config_report()`'s `timeframe_mismatch` check only fires against a timeframe
you actually wrote, so an omitted one never gets flagged as disagreeing with
the data's real bar spacing (pass it explicitly to get that check). It still
*behaves* as `"1H"` when nothing else is available.

`strict` (default `True`): if the resolved configuration bundle carries a
`FAIL`-level `config_report()` finding — a stage judged structurally
incapable of producing a verdict — `forge()` raises `ValueError` **before**
running Module 0. `strict=False` downgrades every finding (`FAIL` and `WARN`
alike) to a `UserWarning` and runs anyway. Non-critical incoherences
(`WARN`) are always warnings regardless of `strict`.

`ForgeResult` fields: `enriched`, `candidates: list[EventCandidate]`,
`contracts: list[AlphaContract]` (promoted *and* rejected — inspect
`contract.rejection_reasons`/`.diagnostics`), `promoted: list[AlphaContract]`,
`rule_responses: list[tuple[AlphaContract, RuleDiscoveryResponse]]`,
`ticker`, `event_frame` (the frame M2/M3/M4 actually read — pass this, not
`enriched`, when building things by hand from a `ForgeResult`), `registry:
RuleRegistry | None`, `market_context`, `event_discovery`, `alpha_discovery`
(live module instances for drill-down), `calibration: CalibrationReport |
None`, `ledger: HypothesisLedger | None`, `time_budget: TimeBudget | None`,
`context: PipelineContext | None` (the session's resolved facts),
`resolution: ResolutionTrace | None` (every field the resolver derived, and
why), `coherence: ConfigReport | None` (the resolved configs plus every
constraint violation found — produced by the *same* resolver call the
pipeline actually ran with, so `coherence.configs` is literally what
executed).

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
accepted by name: M1 — `min_tpm`, `max_dispersion`, `dispersion_margin`,
`min_episodes`, `max_and_components`, `timestamp_col`, `event_counting`; M2 —
`min_lift`, `min_cohens_d`, `fdr_q`, `oos_max_p`, `horizon_grid`,
`bars_per_day`; M3 — `rd_min_tpm`, `min_profit_factor`, `min_win_rate`,
`min_pf_score_tpm`, `min_fill_rate_opt`. Unknown override keys raise
`TypeError`.

M1's `min_tpm`/`max_dispersion` are scaled from the preset's daily-calibrated
spec to `timeframe` via `_TFClass` (daily / intraday / hft bucket).
`dispersion_margin` is **not** scaled — it is a multiplier over the Poisson
floor, already scale-free, and it is what governs dispersion under
`event_counting="episode"` (the default); `max_dispersion` governs only
`"bar"` mode (#205 — comparing an absolute `max_dispersion` against
`max(max_dispersion, poisson_floor)` left it dead code on 12 of 16 measured
preset×timeframe combinations, `"sniper"` on all of them). Per-preset
`dispersion_margin`: 1.05 `sniper`, 1.30 `balanced`, 1.60 `sweep`, 3.00
`burst`. `min_episodes` is likewise **not** timeframe-scaled (an absolute
episode count; the timeframe's effect is already carried by `min_tpm`'s own
scaling) but *is* per-preset (#206): 10 on `sniper`/`balanced`/`burst`, 5 on
`sweep` — `sniper` and `sweep` share the same low stock rate (0.3
episodes/month) and used to share `min_episodes=10` too, needing 53.3 months
(4.44 years) at 95 % Poisson confidence regardless of preset philosophy;
`sniper` keeps 10 (statistical rigor is the point, so the fix was correcting
its description, not weakening the floor), `sweep` lowers it to 5 since it is
permissive by design and already defers rigor to the `RotationCalibrator`.
M3's rate is set at the preset's own ratio to M1's (1.00 on
`sniper`/`sweep`, 0.80 on `balanced`/`burst` — a deliberate per-profile fill
margin, not one flat
number) rather than left to a class default, so `RuleWalkForwardConfig
.min_train_months` — left `UNSET` by the preset — is derived correctly from
a rate the preset actually chose (see *Configuration resolution* below).
`horizon_grid`/`bars_per_day` on the returned `AlphaConfig` are also left
`UNSET`; the resolver fills them from the same `timeframe`. `timestamp_col`
on `DiscoveryConfig` is left `UNSET` unless overridden — a schema fact the
session propagates to every module, not a per-preset choice.

M3's *economic* quality bar is per-preset too (#207): `min_profit_factor`,
`min_win_rate`, `min_pf_score_tpm`, `min_fill_rate_opt` used to be one shared
class default on all four presets despite `"sniper"`/`"sweep"`'s
descriptions explicitly diverging on precision-vs-volume — the
`RotationCalibrator` `"sweep"` relies on prices the *search's* statistical
noise, not a single candidate's own economics, so it was never a substitute.
Stock values: `(2.5, 0.60, 0.40, 0.80)` `"sniper"`, `(2.0, 0.55, 0.30, 0.80)`
`"balanced"`/`"burst"` (class defaults, untouched), `(1.8, 0.50, 0.25, 0.70)`
`"sweep"`. None are timeframe-scaled — ratios/rates already.

`preset_info(preset: str | None = None) -> None` prints the resolved
parameters — including the M3 scoring knobs and every significance
threshold, each pulled through the resolver against a `PipelineContext(timeframe="1D")`
— for one preset or all of them.

`default_horizon_grid(timeframe: str) -> tuple[int, ...] | None` — kept as a
public helper but no longer the mechanism that substitutes `AlphaConfig`'s
grid (the resolver does that on every path since #196, not only when
`forge()` built the config itself). Returns the daily-class grid on a
daily-or-slower timeframe, `None` otherwise — useful for a caller asking
"is this timeframe in the slow class at all", not for scaling by hand.

## Configuration resolution and coherence

Seventeen configuration dataclasses across the pipeline materialise the same
handful of *session facts* — bar duration, schema column names, fee,
arrival rate, significance level — as ~50 independently-defaulted fields.
`docs/analysis/pipeline_parameter_coherence.md` is the full audit (16 named
findings, all closed) that motivated centralising them. Three modules do
this job:

### `forgedge.unset` — the "chosen vs. inherited" sentinel

```python
from forgedge import UNSET
from forgedge.unset import is_set, coalesce   # also importable from forgedge.unset directly
```

`UNSET` is a falsy singleton (`bool(UNSET) is False`), copy/pickle-stable,
and defines no arithmetic or comparison operators — `UNSET < 3` raises
`TypeError` rather than silently producing a number. A dataclass field left
at `UNSET` means *"the resolver decides"*; any other value, including one
equal to the historical class default, means *"this was chosen"* and the
resolver never overwrites it. `is_set(value) -> bool` and
`coalesce(*values, default=None) -> Any` (first non-`UNSET` argument, else
`default`) are the two helpers built on it.

**Consequence for hand-built configs**: a dataclass field that now defaults
to `UNSET` (see each module's field table below) holds the literal sentinel
object, not a number, until something resolves it. Every module constructor
(`EventDiscovery`, `AlphaDiscovery`, `RuleDiscovery`, `RuleRegistry`,
`RotationCalibrator`) calls `forgedge.resolve_config()` on its own config
internally, so this is usually invisible — but a bare
`dataclasses.asdict(BacktestParams())` or arithmetic on a field before any
resolution step will surface `UNSET` directly.

### `forgedge.resolver` — `PipelineContext`, the resolver, and the trace

```
PipelineContext(
    timeframe: str = "1H", timeframe_declared: bool = True,
    timestamp_col: str = "open_dt", close_col: str = "close",
    regime_col: str = "regime", regime_stable_col: str = "regime_stable",
    fee_per_side: float = 0.002,
    alpha: float = 0.05, min_sample: int = 10,
    target_rate_tpm: float | None = None, rate_retention: float = 1.0,
    bars_per_episode: float = 1.76,
    cross_pf_retention: float = 0.8, net_gain_retention: float = 0.5,
    n_bars: int = 0, span_months: float = 0.0,
    inferred_bar_hours: float | None = None,
)
PipelineContext.from_frame(frame, *, timeframe="1H", **overrides) -> PipelineContext
```

The session's single source of truth for facts no module owns individually.
Properties: `.bar_minutes`, `.bar_hours`, `.bars_per_day`, `.bars_per_month`;
methods `.months_of(n_bars)`, `.bars_of(months)`. `target_rate_tpm` is
**collected from**, not fed into, the session: it is `None` unless Event
Discovery's `GateParams.min_tpm` was actually set away from its class
default (0.5) — a class default is not a declaration, so an inherited one
never silently propagates as if chosen (this is why setting only M1's rate
and expecting M3 to follow it requires the rate to be genuinely non-default;
`forge_preset()` always sets an explicit M3 rate itself, so this rarely
matters when using presets). `rate_retention` (default `1.0`) is
deliberately *not* a shrinkage margin — see the module's own docstring for
the measured reason a naive `<1.0` value costs history twice over
(#200: it lengthens `min_train_months` *and* shrinks the pooled OOS trade
count at once). `bars_per_episode` (default `1.76`, measured median on
`ADA_1D_TRAIN`) converts a *declared* `event_counting="episode"` rate to the
bar rate M3 actually counts — it opens a trade on every active bar, no
episode concept — before `rate_retention` applies; unread in `"bar"` mode,
where M1 and M3 already share a unit (#204).

```
Derivation(order: int, field: str, default: Any, resolved: Any, rule: str,
           reason: str, inputs: tuple[str, ...] = (), superseded: bool = False)
ResolutionTrace(derivations: list[Derivation] = [], untouched: int = 0)
```

`ResolutionTrace.effective` — derivations that actually took effect (a
conflict on the same field keeps the more conservative value; the loser
stays in the trace with `superseded=True`, not discarded). `.describe()` —
one-line summary in the idiom of `HypothesisLedger.describe()`.
`.to_text(verbose=False)` — the full `default → resolved` table with the
rule that fired.

```
Violation(code: str, level: "FAIL" | "WARN", message: str, fields: tuple[str, ...] = ())
Constraint(code, level, stage, free, derived=None, derive=None, check=None)
```

A `Constraint` relates materialisations of one latent parameter: `free` are
the dotted paths it reads, `derived` is the path it may *write* (`None` for
a check-only constraint), `derive`/`check` are the callables. `stage` is one
of `PROPAGATION` (moves an identical value between configs — this is the
group whose `derive` actually runs by default), `STATISTICAL` (a check whose
matching derive constraint, registered separately under `PROPAGATION`,
already performs the fix — e.g. `wf_bucket_too_short`'s Poisson-margin sizing
of `min_train_months` derives unconditionally; the `STATISTICAL`-tagged
entry with the same code supplies the *check* half), or `STRUCTURAL`
(check-only, always). **Every check runs regardless of stage** — only
`derive` is stage-gated, and today every field with a `derive` function does
run it by default (`resolve()`'s default `active_stages=(PROPAGATION,)`
covers all of them).

```
resolve(bundle: dict, ctx: PipelineContext | None = None, *,
        active_stages: Sequence[str] = (PROPAGATION,)
        ) -> tuple[dict, ResolutionTrace, list[Violation]]
collect_context(bundle: dict, base: PipelineContext | None = None, **overrides) -> PipelineContext
resolve_config(cfg: Any, kind: str, ctx: PipelineContext | None = None, **kwargs) -> Any
```

`bundle` keys: `"market_context"`, `"event_discovery"`, `"alpha"`,
`"rule_discovery"`, `"registry"` — missing/`None` entries are skipped, so a
partial bundle resolves fine. `resolve()` never mutates the caller's
configs (works on a copy) and is idempotent. `collect_context()` seeds a
`PipelineContext` from whatever the caller set explicitly in any config
(precedence: explicit `overrides` > an explicit `base` context > a field set
in some config > the class default) — this is the mechanism that makes
`forge_preset(timestamp_col="ts")` reach every module instead of only the
one it was set on. `resolve_config(cfg, kind, ctx=None)` resolves a single
config standalone — every module's own constructor opens with this, with
**no context passed** (so a bare `PipelineContext()` — `timeframe="1H"` — is
what standalone module construction resolves against; only `forge()` builds
one from the real data and declared timeframe). See `SKILL.md` pitfall #2.

### `forgedge.config_report` — check without running

```
config_report(event_discovery=None, alpha=None, rule_discovery=None,
              registry=None, market_context=None, *, ctx=None, kpi=None,
              timeframe="1H", verbose=False) -> ConfigReport
```

`summary_report()`'s sibling: that validates the **data**, this validates the
**configuration**, in the same `Finding(level, code, message)` vocabulary.
`config_report()` and `forge()` call the *same* resolver, so
`rep.configs` is literally what a `forge()` call with these configs would
execute. `kpi`, when given, only feeds the **checks** (span-dependent ones
have nothing to compare against otherwise) — resolution itself never reads
the data. Never raises, warns, or mutates its inputs.

`ConfigReport` — `.trace: ResolutionTrace`, `.findings: list[Finding]`,
`.configs: dict` (the resolved configs), `.context: PipelineContext`,
`.worst` (`"OK"`/`"WARN"`/`"FAIL"`), `.has_critical`, `.has_warnings`,
`.one_line()`, `.to_text()` (full resolution trace + diagnostics report).

Three `FAIL`-level constraints (a stage judged structurally incapable of
producing a verdict): `wf_bucket_too_short` (M3's selection window vs. its
own declared rate — the fix for issue #173), `m1_oos_fold_too_short` (an M1
walk-forward fold too short to test its own rate), `oos_span_too_short` (the
pooled OOS span can't reach `min_oos_trades` at the declared rate). Ten
`WARN`-level constraints: `m3_stricter_than_m1`, `scoring_uncalibrated`,
`timeframe_mismatch`, `split_disagreement`, `registry_stricter_than_m3`,
`alpha_level_drift`, `tp_floor_conflict`, `entry_mode_inert_gate`,
`schema_mismatch`, `fee_mismatch`. Every message carries the value to set,
not just the failure.

**What the resolver fills in by default** (session facts, propagated to
every config that materialises them): the timestamp column (all four
`timestamp_col` fields → `"open_dt"`), the price series
(`AlphaConfig.close_col`, `BacktestParams.target_col` → `"close"`; note
`BacktestParams.buy_price_anchor` is filled from the same source but never
fed back — it is a *reference level* for the limit offset, not necessarily
the schema's close column, so `buy_price_anchor="close_sma_3"` is legal and
is never checked against `close_col`), the regime columns
(`AlphaConfig.regime_col`/`regime_stable_col`), the cost basis
(`AlphaConfig.fee_per_side`, `BacktestParams.fee` → `0.002`, one value now
instead of two independent copies that only agreed by sharing a default),
the genericity bar (`RegistryConfig.cross_pf_threshold`/
`min_cross_pf_retention`), every bar-counting field
(`AlphaConfig.horizon_grid`/`bars_per_day`, `BacktestParams.target_h`/
`buy_delay_bar`, `MarketContextConfig.stable_window`,
`EMAProxyConfig.bar_hours`), the M3 arrival rate
(`SelectionCriteria.min_tpm`, `RuleWalkForwardConfig.min_train_months`,
`ScoringParams.pf_min_tpm`/`pf_min_trades`), and the five per-hypothesis
significance thresholds (`PromotionThresholds.max_p_value`/`ic_max_p`,
`SelectionCriteria.max_ttest_p`/`max_rotation_p`, all from
`PipelineContext.alpha`, default `0.05` — `fdr_q` and `oos_max_p` are
deliberately *not* included, since one is a false-discovery rate over a
family and the other is a confirmation threshold, not per-hypothesis alphas).

## M0 — Market Context

```
MarketContext(kpi_table: pd.DataFrame, config: MarketContextConfig | None = None)
mc.run() -> pd.DataFrame            # + 'regime' (ordered Categorical), 'regime_stable' (bool)
mc.distribution()                   # bar share per regime, for diagnostics
mc.window_resolution                # {"source": "hurst_ou" | "fallback" | "configured", ...}
```

`EMAProxyConfig` key fields: `source_col: str = "close"`, `auto_window: bool
= True`, `short_period: int = 9`, `long_period: int = 25`, `threshold_mode:
"fixed"|"balanced" = "fixed"`, `threshold_basis: str = "global"`,
`window_unit: "day"|"bar" = "day"`, `window_estimation: float = 168`,
`window_stride: float = 1`, `bar_hours: float | None = UNSET` (`UNSET`
means "measure it from the timestamps" — a real, deliberate behaviour, not
merely an unresolved default; explicit only when there's no DatetimeIndex to
measure from), `fast_ratio: float = 1/2.3`, `min_window_estimates: int = 10`.
Regimes run `STRONG_BEAR → BEAR → NEUTRAL → BULL → STRONG_BULL`; fallback
EMA spans are `9`/`25` when the OU half-life estimate does not converge.

`MarketContextConfig` — `classifier: str = "ema_proxy"`, `ema_proxy:
EMAProxyConfig`, `labels: list[str]` (5 regime names), `stable_window: int =
UNSET` (session-resolved; fallback `12` with no context — the historical
default, unchanged).

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
ed.event_distribution_report   # str | None — populated by run(), see below
```

`event_distribution_report` (#215) — an always-computed, plain-text diagnostic
populated inside `run()` itself (`None` before). `config_report()` is blind to
data by construction (it must resolve without a frame), so it can only catch
config-vs-config incoherence — never a preset that is internally coherent yet
still rejects every candidate a specific asset actually produces, which used
to surface only as a bare `"M1 Event Discovery — 0 candidate(s)"` log line
with no diagnosis. This aggregates the `GateResult` already attached to every
raw (pre-AND-composition) event by the Consistency Gate — no new data pass —
into the observed tpm/dispersion median and the share failing each criterion,
against the configured thresholds (`eff_max_dispersion =
poisson_floor(n_months) x dispersion_margin` in `"episode"` mode, raw
`max_dispersion` in `"bar"` mode). Below a 15% gate-survival rate it also
appends a concrete parameter suggestion at the observed median on both
measures (`min_tpm<=...`, plus `dispersion_margin>=...` or `max_dispersion>=
...` depending on `event_counting`). `forge()`'s M1 stage line uses this text
in place of the old bare count whenever Event Discovery actually ran (manual
event injection via `manual_events=` skips `EventDiscovery.run()` entirely, so
that path keeps the bare-count line); `result.event_discovery
.event_distribution_report` exposes it for code-level inspection at no extra
cost, since the `EventDiscovery` instance is already retained on `ForgeResult`.

`DiscoveryConfig` fields and defaults (`src/forgedge/event_discovery/discovery.py`):
`gate_params: GateParams = GateParams()`, `max_categorical_classes: int = 20`,
`scale_free_overrides: dict[str, bool] | None = None`, `timestamp_col: str =
UNSET` (session-resolved, `"open_dt"` fallback), `max_and_components: int =
2`, `train_ratio: float = 1.0`, `walk_forward: EventWalkForwardConfig | None
= None`, `diversity_gate_enabled: bool = False`, `diversity_threshold: float
= 0.85`, `indicator_lag_cross_lags: tuple[int, ...] = (1, 3)` (lag set for
the price-scale-indicator-vs-lagged-OHLC-base feature family below; pass
`()` to disable that family entirely).

`GateParams` (Consistency Gate, Step 4) — `min_tpm: float = 0.5`,
`max_dispersion: float = 1.5`, `dispersion_margin: float = 1.3`,
`event_counting: "episode"|"bar" = "episode"`, `min_episodes: int = 10`,
`episode_gap: int = 1`. `"episode"` counts maximal runs of consecutive
activations (bridged by gaps ≤ `episode_gap`) rather than raw bars — the
default because a persistent multi-bar state otherwise inflates
monthly-count variance and gets wrongly rejected. `"bar"` reproduces the
pre-#134 behaviour exactly. `max_dispersion` and `dispersion_margin` are
mode-exclusive, not redundant (#205): in `"bar"` mode the dispersion
criterion is the raw `ID <= max_dispersion`, no floor; in `"episode"` mode
it is `episode_ID <= dispersion_margin x poisson_floor(n_months)` and
`max_dispersion` is not read at all — comparing the absolute value against
`max(max_dispersion, poisson_floor)` used to leave a preset's own tolerance
dead code whenever the floor (a function of calendar months only) exceeded
it, which measurement showed was most of the time. `min_episodes` is
absolute and, being unrelated to `min_tpm`, implies an in-sample *discovery*
window that depends on both: `config_report()`'s `m1_is_window_too_short`
(WARN) says when the configured span can't reach it at 95 % Poisson
confidence (#206) — `poisson_min_window(min_episodes, min_tpm)`, not the
naive `min_episodes / min_tpm`. **Note:** an older `GateParams` API
(`min_act`, `min_months`, `max_conc`) appears in several `examples/*.py`
scripts and now raises `TypeError` — see *Errors and warnings*.
AND-composed pairs/triples (`ANDComposer`, below) are judged by this same
mode-aware criterion as single events — `"episode"` mode gates on episode
rate/count/dispersion, `"bar"` mode on raw `max_dispersion`, with no
special-casing for arity (#226; before the fix, composed events were
always judged bar-mode-only regardless of `event_counting`). The episode-mode
computation this added is memory-bounded regardless of dataset length (#228):
`ANDComposer` internally chunks its vectorized pair/triple loops at a size
that shrinks for long histories under a fixed budget, rather than a size
fixed at 5000 regardless of `n_rows` — relevant only if you're reading
`and_composer.py` internals or profiling `compose()`, not part of its public
behaviour.

`ANDComposer.compose()`'s pair/triple enumeration order is permuted with a
fixed, deterministic seed (#230) — under a permissive gate (a low `min_tpm`,
including `GateParams()`'s own default), the pre-fix row-major order let
`_MAX_PAIRS=2000` fill entirely with pairs sharing one single component;
the permutation samples the whole candidate pool instead, with no change to
reproducibility (same input always yields the same composed events).
`compose()` also accepts an opt-in keyword-only `max_constituent_jaccard:
float | None = None` — when set, rejects a pair whose two constituents'
Jaccard similarity on their activation series exceeds the threshold, the
same formula `DiversityGate` already applies to single events, at no extra
cost (reuses the volume pre-filter's own intersection count). Disabled by
default; only meaningfully reduces redundancy once the enumeration order is
representative (i.e. after the #230 shuffle — negligible effect on its own).

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

`EventWalkForwardConfig` (event-level; **renamed** from the ambiguous
`WalkForwardConfig` — the top-level `forgedge.WalkForwardConfig` alias now
resolves to the *rule*-level class instead, see M3 below) — `n_splits: int =
3`, `min_pass_rate: float = 0.6`, `oos_gate_params: GateParams | None = None`
(defaults to the IS `gate_params`).

With this set and `train_ratio < 1.0`, each candidate exposes `.validation:
ValidationResult` with `.passed: bool | None` — **tri-state**: `True`/`False`
as before, or `None` when every walk-forward fold was too short to say
anything at the candidate's own declared rate (a fold is *testable* only
once its expected episode count reaches 3; below that it is marked
`FoldResult.indeterminate=True` and excluded from the pass/fail denominator
rather than counted as a failure). `None` means *inconclusive*, not
*failed* — code that filters on `c.validation is not None` to mean
"validation concluded" is now wrong (it also matches the indeterminate
case); filter on `c.validation.passed is not None` instead. `.pass_rate`,
`.n_passed`, `.n_testable`, `.fold_results` are the other `ValidationResult`
fields.

`EventCandidate` — key attributes: `event_id`, `expression` (boolean string),
`event_formula` (human-readable), `sql_expression` (DuckDB/SQL — see the
caveat below), `components: list[EventComponent]`, `activation_stats:
ActivationStats` (`n_activations`, `n_active_months`, `zero_months`,
`max_monthly_share`, `mean_tpm`, `index_of_dispersion`, `n_episodes`,
`episode_index_of_dispersion`, `n_eff`), `consistency_gate: GateResult`,
`validation: ValidationResult | None`. Method `.apply(df) ->
pd.Series[bool]` — deterministic, no look-ahead, re-evaluates the stored
thresholds on any new frame; this is the path `forgedge` itself always uses
and is correct for every candidate, including the arity-2 pairings above.
`.persist(path)` — full pickle round-trip (components, thresholds,
activation stats, validation). Caveat: `sql_expression` for a lag-cross
feature (pairings 1/5 above) combined with a rolling pctrank/zscore
transform emits a best-effort SQL translation containing a nested window
function some engines (including DuckDB) reject — treat it as a convenience
export, not a guaranteed-portable one, for that specific combination; use
`.apply()` if you need certainty.

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

**Critical:** pass `ed.df` (Event Discovery's post-pipeline frame), not the
original KPI Table — it already carries the derived ratio/spread/transform
columns the event expressions reference.

`AlphaConfig` fields and defaults (`src/forgedge/alpha_discovery/models.py`):
`horizon_grid: tuple[int, ...] = UNSET` (session-resolved; **hourly**
fallback `(1, 2, 4, 8, 12, 24)` with no context — see `SKILL.md` pitfall #2,
this is the one place a hand-built config on daily-or-slower data still
needs an explicit value if not going through `forge()`), `mfe_quantile:
float = 0.5`, `mfe_floor: float = 0.005`, `train_ratio: float = 0.7`,
`embargo_bars: int = 0`, `horizon_enrichment: tuple[float, ...] | None =
(0.5, 1.0, 2.0)` (adds horizons at 0.5×/1×/2× each event's dominant
indicator window — union, statistically capped, never a restriction; on the
reference fixture 34 of 247 promoted alphas found their best horizon *only*
because of this), `horizon_enrichment_min_obs: int = 20`, `thresholds:
PromotionThresholds = PromotionThresholds()`, `asset: str = "ASSET"`,
`timeframe: str = "1H"` (a plain literal, unlike the session's own
`timeframe` — this field is traceability metadata copied onto contracts,
not itself session-resolved), `fee_per_side: float = UNSET` (→ `0.002`),
`close_col: str = UNSET` (→ `"close"`), `timestamp_col: str = UNSET` (→
`"open_dt"`), `regime_col: str = UNSET` (→ `"regime"`), `regime_stable_col:
str = UNSET` (→ `"regime_stable"`), `use_stable_regime_only: bool = False`,
`min_regime_obs: int = 10`, `bars_per_day: float | None = UNSET`,
`fixed_target: TargetConfig | None = None` (fixed-target /
`TargetOptimizer` mode), `target_mode: "abs"|"proj" = "proj"`,
`trend_sma_mult: float = 2.0`.

`PromotionThresholds` — `ic_min_abs: float = 0.02`, `ic_max_p: float =
UNSET` (per-hypothesis alpha, → `0.05`), `min_lift: float = 0.08`,
`min_cohens_d: float = 0.15`, `max_p_value: float = UNSET` (→ `0.05`; used
only when `use_fdr=False`, and every preset sets `use_fdr=True` — inert
under every preset), `use_fdr: bool = True`, `fdr_q: float = 0.10` (a
false-discovery rate over the horizon family, deliberately *not*
alpha-derived — a different kind of quantity than a per-hypothesis
significance level), `oos_max_p: float = 0.10` (a confirmation threshold,
also not alpha-derived), `min_direction_t: float = 0.5`,
`require_significant_direction: bool = True`. **The only hard rejection gate
is undetermined direction** ("no derivable target" — no horizon produces a
finite advantage); every other metric here feeds the A–D grade via
`diagnostics`, not a pass/fail gate.

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
`rotation_p` / `rotation_threshold` (set by the search-level rotation null).
Two separate lists, **not one merged field**: `rejection_reasons: list[str]`
— causes that actually *blocked* promotion, empty on every promoted contract
(in practice, at most "no derivable target"); `diagnostics: list[str]` —
non-blocking observations that inform the grade (weak IC/lift/Cohen's d,
failed BH-FDR, thin IS/OOS sample, weak OOS confirmation) and routinely
non-empty even on a promoted, grade-A contract. Before this split both lived
in `rejection_reasons` with a `"[diagnostic] "` prefix on the non-blocking
ones — code checking `if contract.rejection_reasons:` to mean "something is
wrong" now needs `diagnostics` too, or it will read as clean when it isn't.
`.to_contract_dict()` for JSON/YAML export (includes both lists).

## Search-level calibration

```
FastRotationNull(event_frame, candidates, alpha_config, time_budget=None).run(promoted, alpha=0.05) -> CalibrationReport
RotationCalibrator(event_frame, candidates, alpha_config, time_budget=None).run(promoted, RotationConfig(...)) -> CalibrationReport
```

`forge()` runs `FastRotationNull` by default (`fast_null=True`) — exact null
distribution of the best standardised excess over every circular offset, via
FFT, `abs_z` yardstick only, ~seconds even on thousands of candidates. Its
`.run()` now takes an `alpha` keyword (default `0.05`) — `forge()` passes the
session's resolved `PipelineContext.alpha`, so a session-wide significance
level reaches the null bar too (#182).
`RotationCalibrator` is the heavier, sampled (`RotationConfig(k=...)`)
multi-yardstick alternative (`composite`, `is_lift`, … combined via Tippett
min-p) — pass `rotation_calibration=RotationConfig(k=100)` to `forge()` to
use it inline instead (supersedes `fast_null`; `forge()` resolves its
`alpha` against the session context too), or run it standalone against
`ForgeResult.event_frame` / `.candidates` for the full report without
slowing the main run.

`RotationConfig` — `k: int = 100`, `alpha: float = 0.05`, `seed: int =
20260624`, `in_sample_stats: tuple[str, ...]`.

`CalibrationReport` — `tippett_p`, `tippett_best_stat`, `per_stat_p`,
`null_q`, `real_stats`, `null_arrays`, `survivors` (promoted contracts whose
statistic clears the null bar), `.summary()`.

Rule Discovery requires `rotation_p <= criteria.max_rotation_p` (session-
resolved from `PipelineContext.alpha`, fallback `0.05`) for a full `EDGE`
verdict; otherwise the verdict is capped at `PARTIAL-EDGE` (still tradeable
— `resp.is_edge`).

## M3 — Rule Discovery

```
RuleDiscovery(event_frame: pd.DataFrame, contract: AlphaContract, candidate: EventCandidate,
               config: RuleDiscoveryConfig | None = None,
               time_budget: TimeBudget | None = None)
resp = rd.run() -> RuleDiscoveryResponse
```

The `event_candidate` you pass must be the one `contract.event_candidate_id`
actually points to, or the constructor raises `ValueError`. `time_budget`
(new) does not change the walk-forward's own origin — M3 keeps its own
geometry — it lets the response report which folds' *test* windows land
inside the span Alpha Discovery fit the target on
(`WalkForwardSplit.tests_in_sample`), informational rather than a gate.

`RuleDiscoveryConfig` fields and defaults (`src/forgedge/rule_discovery/models.py`):
`base_params: BacktestParams`, `scoring: ScoringParams`, `grid: GridSpec`
(auto-built around the contract target when empty — its `target_h` fan now
reaches down to `0`, not floored at `1`), `walk_forward:
RuleWalkForwardConfig`, `criteria: SelectionCriteria`, `entry_mode: "limit"|
"market"|"auto" = "auto"` (**changed from `"limit"`, issue #185** — see
*Entry mode* below), `use_contract_target: bool = True`, `timestamp_col: str
= UNSET` (→ `"open_dt"`), `signal_col: str = "__rule_signal__"`,
`discovery_date: Optional[str] = None`, `selection_mode: "walk_forward"|
"full_sample" = "walk_forward"` (operating point selected inside WF train
windows only — the final test window is never read by any selection),
`wf_param_policy: str = "last"`, `n_trials_upstream: int = 1`.

`BacktestParams` — `direction: str = "long"`, `buy_type: str = "limit"`,
`buy_drop_pct: float = 0.010`, `sell_pct: float = 0.040`,
`target_hit_col: str = "close"` (deliberately a literal, not schema-resolved
— it picks an exit *convention*, conservative `"close"` vs. optimistic
`"high"`/`"low"`, not a schema fact), `early_stopping: bool = True`. Five
fields are `UNSET`/session-resolved, each with a **fallback** used by
`.resolved()` when there is no session context (a hand-built
`BacktestParams` passed straight to `run_backtest`): `buy_delay_bar: int =
UNSET` (→ `6`; session-resolved from a **6-hour** wall-clock duration
converted to bars — 6 on 1H, 2 on 4H, 1 on 1D, 24 on 15m — not a flat bar
count), `buy_price_anchor: str = UNSET` (→ `"close"`; any numeric candle
column is legal here, e.g. `"close_sma_3"` for "limit at 90% of the 3-bar
SMA" with `buy_drop_pct=0.10` — this field is filled from the price column
but never checked against it, since it names a reference *level*, not a
schema fact), `target_h: int = UNSET` (→ `24`; session-resolved as the *top
of the session's horizon class* — 24 hourly, 10 daily, 50 sub-hourly — not a
wall-clock conversion, matching `AlphaConfig.horizon_grid`'s own
calibration; in practice seeded from the contract's `holding_period_h`
before this default is even consulted), `target_col: str = UNSET` (→
`"close"`; session-resolved from `close_col`), `fee: float = UNSET` (→
`0.002`; session-resolved from `AlphaConfig.fee_per_side` — one cost basis,
not two copies that merely shared a default). `target_h=0` is legal and
means "exit at the fill bar's own close" (`fill_rn == exit_rn`), not "no
horizon" — total signal→exit span is always `1 (signal→fill) + target_h`.

`ScoringParams` — `pf_min_trades: int = UNSET` (→ `15`), `pf_min_tpm: float
= UNSET` (→ `2`; session-resolved from `criteria.min_tpm` — it used to be a
fixed `2` while the gate it's meant to agree with ranges from 0.8 on daily
bars to 76.8 on 15-minute ones). `pf_tpm_target` **no longer exists** —
`c_norm` (the dispersion term of `pf_score_tpm = profit_factor * c_norm`) is
now `min(1, 1 / max(index_of_dispersion, 1))`, scale-free, so only
burstiness in excess of a Poisson process is penalised at any rate; the old
formula penalised a rule for the variance its own arrival rate necessarily
produces.

`RuleWalkForwardConfig` (**renamed** from the ambiguous `WalkForwardConfig`
— see M1 above) — `n_splits: int = 4`, `train_span_months: int | None =
None` (`None` = anchored/expanding train window), `test_span_months: int |
None = None`, `min_train_months: int = UNSET` (**no standalone fallback** —
sized with a 95% Poisson margin from `criteria.min_tpm`, e.g. 11 months at
the `"balanced"` preset's daily rate, not the historical flat `6`; the naive
`floor / rate` inversion under-supplies about 44% of the time, which is why
it's a margin and not a division), `reoptimise: bool = True`, `purge_bars:
int | None = None` (`None` defaults to the resolved grid's largest `target_h`
plus fill delay; deliberately **not** unified with `TimeBudget.purge_bars` —
different boundaries, a fold's worst-case trade span vs. the session's
forward-return horizon), `embargo_bars: int = UNSET` (session-resolved from
`AlphaConfig.embargo_bars` — same policy, different boundary; an explicit
value here still wins). The top-level `forgedge.WalkForwardConfig` alias
resolves to this class, not the M1 one.

`SelectionCriteria` — `min_profit_factor: float = 2.0` (preset-parametrized,
#207 — 2.5 `"sniper"`, 1.8 `"sweep"`, class default on `"balanced"`/
`"burst"`), `min_win_rate: float
= 0.55` (likewise, #207: 0.60/0.50), `min_tpm: float = UNSET` (**the root of a chain** — resolved from
`PipelineContext.target_rate_tpm × bars_per_episode × rate_retention` when
M1's rate was *declared* and counted in episodes (`bars_per_episode` is 1 in
`"bar"` mode — M1 and M3 already share a unit there, #204), else the
documented default `2.0` stands; `min_train_months` and
`scoring.pf_min_tpm` both derive from this, so pinning it manually
disconnects them, see `SKILL.md` pitfall #8), `min_pf_score_tpm: float =
0.30` (preset-parametrized, #207: 0.40 `"sniper"`, 0.25 `"sweep"` — the grid's
own objective should track the same precision-vs-volume intent as the PF/
win-rate gates beside it), `min_fill_rate: float = 0.40` (inert under the default
`entry_mode="auto"`, where Stage 1 is a market entry filling ≈100% — kept
for `entry_mode="limit"`; `config_report()`'s `entry_mode_inert_gate` warns
if this was moved off its default under a mode that makes it inert),
`min_fill_rate_opt: float = 0.80` (floor for the `entry_mode="auto"` Stage-2
adoption; preset-parametrized, #207 — the fill floor that actually binds
under the default mode, unlike its inert sibling above: 0.70 `"sweep"`, class
default elsewhere), `min_net_gain_retention: float = UNSET` (→ `0.5`; Stage-2
adoption condition 3, see *Entry mode* below), `min_sell_pct: float = UNSET`
(→ `0.005`; session-resolved from `AlphaConfig.mfe_floor` — it used to be a
hardcoded `max(0.01, sell_pct)` inside `_seed_base_params`, twice M2's own
floor, silently overriding a derived take-profit target on intraday data),
`partial_min_profit_factor: float = 1.5`, `min_active_month_rate: float =
0.80`, `max_regime_dependency: float = 0.30`, `min_dsr: float = 1.0`,
`max_ttest_p: float = UNSET` (→ `0.05`; per-hypothesis alpha, session-
resolved — **the pipeline's only hard per-hypothesis gate**, producing
`NON-EDGE` in `_decide`; no preset has ever touched it), `max_rotation_p:
float = UNSET` (→ `0.05`; per-hypothesis alpha against a *different* null —
the whole discovery surface's — session-resolved; a strict value under
`"sweep"` is intended, not drift, since `"sweep"`'s upstream permissiveness
is predicated on this gate filtering downstream), `power_gate: bool = True`
(demotes to `INSUFFICIENT-DATA` when pooled OOS evidence can't support the
verdict — no walk-forward was possible, pooled OOS trades below
`min_oos_trades`, or the pooled sample's minimum detectable expectancy
exceeds the claimed one; reads only the concatenated test-window ledger,
never per-window counts; never rescues a `NON-EDGE`), `min_oos_trades: int =
10`, `early_elimination: bool = True` (set `False` to force the full
walk-forward/diagnostics pipeline even on a fast-screened `NON-EDGE`).

### Entry mode — what the verdict measures

`entry_mode="auto"` (default since #185) splits the verdict from the entry
optimisation:

- **Stage 1** backtests at a market entry (next-open fill, ≈100%). This
  verdict is **authoritative** — Stage 2 can never turn a `NON-EDGE` into an
  edge, only choose which parameters get published.
- **Stage 2** sweeps `buy_drop_pct` on a Stage-1 survivor, **replays** the
  winner out-of-sample on Stage 1's own test windows (`reoptimise=False`, so
  it adds no selection and no extra `n_trials` to the walk-forward proper —
  its Deflated Sharpe carries its own larger trial count as an absolute
  metric, but the `min_dsr` gate always reads the market point's), and
  adopts it only if **all three**, measured out-of-sample, hold: (1)
  `fill_rate >= min_fill_rate_opt`; (2) `opportunity_sharpe >= market's` — a
  *per-trade-frequency* Sharpe (`(µ/σ) × sqrt(trades per year)`), deliberately
  not `StatisticalValidation.sharpe_ratio`, which annualises by *capacity*
  (`bars_per_year / avg_holding_bars`) and is identical for both points since
  they hold for the same length — using it would collapse the comparison
  onto the per-trade Sharpe alone, rewarding a rule for trading less; (3)
  `net_gain >= min_net_gain_retention × market's` — a backstop against a tiny
  mu/tiny sigma combination the Sharpe alone can't see.

`entry_mode="limit"` (grid varies `buy_drop_pct`, doubles as entry-price
optimiser — can suffer the "fill confound": a deep, rarely-filled limit
inflates PF on a non-representative subset of trades) and `"market"` (pure
baseline, no optimiser) remain fully supported; `"limit"` is the right
choice when the limit order *is* the strategy, not merely an execution
refinement.

`EntryOptimization` (populated only under `entry_mode="auto"`) — `
selected_entry: "market"|"limit"` (the published point), `authoritative:
str = "market"` (where the verdict came from — always `"market"`),
`adopted: bool`, `failed_condition: "fill"|"sharpe"|"net_gain"|None` (which
condition stopped Stage-2 adoption, or `None` when adopted or when there was
no limit candidate at all — this is what tells "the limit point was better
but didn't survive OOS" apart from "the limit point was never better"),
`min_fill_rate_opt`, `min_net_gain_retention: float` (thresholds actually
applied), `market_rule`/`limit_rule: ValidatedRule | None` (both operating
points as publishable rules), `market_summary`/`limit_summary:
BacktestSummary | None` (their **out-of-sample** summaries over identical
test windows), `limit_walk_forward: WalkForwardResult | None` (the limit
point's replay), `limit_validation: StatisticalValidation | None`,
`market_opportunity_sharpe`/`limit_opportunity_sharpe`,
`market_oos_net_gain`/`limit_oos_net_gain`, `limit_oos_fill_rate`,
`limit_buy_drop_pct`, `market_profit_factor`/`market_fill_rate`/
`market_pf_score_tpm` (Stage-1 in-sample metrics), `limit_profit_factor`/
`limit_fill_rate`/`limit_pf_score_tpm` (best in-sample limit candidate,
retained for continuity — `pf_score_tpm` is no longer the adoption
criterion), `reason: str` (human-readable). `.to_dict()` for JSON export.

### Nominal economics, effective inference

`run_backtest` opens a position on every active bar with no flat-state check
— deliberate and unchanged, and the reported economics are reproducible live
given the capital to fund concurrent positions. `BacktestSummary` gained
`n_episodes: int = 0` (how often the signal fires — episode-grouped, not
per-bar), `mean_concurrent_positions: float = nan` (average open positions
**over bars where at least one is open** — idle stretches excluded on
purpose, since capital sizing asks "when working, how many positions am I
funding", not "how busy across all history" — also the divisor that turns
nominal trades into effective ones), `max_concurrent_positions: int = 0`
(the number that decides deployability at all). The trade ledger
(`run_backtest(..., return_trades=True)`) carries an `episode_id` per row.

`StatisticalValidation` gained `n_effective: float = nan` —
`total_trades / mean_concurrent_positions`, the sample size the overlap
actually supports. Consumed by the t-test's standard error/degrees of
freedom, `deflated_sharpe`'s `n_obs`, and `expectancy_mde`/the power gate —
**not** by `total_trades`, profit factor, expectancy or net gain, which stay
nominal (the economics are reproducible; only quantities assuming
independent observations needed the correction). On issue #168's reference
case (118 nominal, ≈32 effective): t overstated `sqrt(118/32) ≈ 1.93×` (a
t=2.6, p≈0.005 was really t=1.35, p≈0.09); DSR's correction moved from 0.820
to 0.741. `n_effective` is `nan` (not a guess) when the ledger carries no
bar geometry to measure overlap from.

`WalkForwardSplit` gained `tests_in_sample: bool | None = None` — whether
this split's *test* window falls inside the session's own IS span (only
meaningful when `RuleDiscovery` was given a `time_budget`); `WalkForwardResult`
gained `n_splits_in_sample: int | None = None`, the count of such splits.

`RuleDiscoveryResponse` — `verdict: "EDGE"|"PARTIAL-EDGE"|"NON-EDGE"|
"INSUFFICIENT-DATA"`, `is_edge: bool` (True for EDGE/PARTIAL-EDGE),
`rejection_reasons: list[str]`, `notes: list[str]` (unchanged shape — M3 did
*not* get the `AlphaContract`-style `diagnostics` split), `validated_rule:
ValidatedRule | None` (`.params: BacktestParams`), `in_sample_summary:
BacktestSummary` (`total_trades`, `profit_factor`, `win_rate_pct`,
`expectancy`, `tpm_mu`, `n_episodes`, `mean_concurrent_positions`,
`max_concurrent_positions`), `execution_envelope: ExecutionEnvelope | None`
(`.conservative` / `.optimistic`), `walk_forward: WalkForwardResult | None`
(`.oos_summary`, `.consistency`, `.n_splits_in_sample`),
`statistical_validation: StatisticalValidation | None`
(`.temporal_stability`, `.deflated_sharpe`, `.n_effective`),
`regime_analysis: RegimeBreakdown | None` (`.avoid_in`), `excursion:
ExcursionStats | None` (MAE/MFE), `entry_optimization: EntryOptimization |
None` (only under `entry_mode="auto"`), `grid_results: list[GridResult]`.

`text_report(resp, *, contract=None)` / `html_report(resp, *, contract=None)`
(also importable from top-level `forgedge`, not just `forgedge.rule_discovery`)
— human/HTML report builders from a `RuleDiscoveryResponse`: every parameter
of the validated rule, IS/OOS stats, walk-forward per-split breakdown,
statistical validation, entry-mode optimisation, regime breakdown, rejection
reasons. `contract` is optional — an `AlphaContract` (Modulo 2) is a
different object than `resp` (Modulo 3) and carries `rotation_p`/
`rotation_threshold`, which `RuleDiscoveryResponse` has no reason to
duplicate; passing it folds a "rotation calibration" section into the same
report. `rule_summary_report(resp, *, contract=None, fmt="text"|"html",
save=False, path=None)` is a thin, discoverable wrapper over the two —
`save=True` also writes the rendered report to `path` (default
`f"{resp.alpha_id}.{txt|html}"` in the cwd) while still returning the
string, so `print(rule_summary_report(resp, save=True))` works in one call.
`resp.to_dict()` for JSON export. Don't confuse any of these with
`rule_performance_report()` (pattern 5) — that one *replays* `RuleSpec`s on
fresh candles to build monitoring charts (equity curve, rolling expectancy,
MAE/MFE distributions); these three render the verdict M3 already computed,
no replay.

## Episodes and concurrency

```
from forgedge import episode_starts, episode_ids, concurrency, ConcurrencyStats

starts = episode_starts(active: np.ndarray, gap: int = 1) -> np.ndarray   # bool, True at each episode's first bar
ids    = episode_ids(active: np.ndarray, gap: int = 1) -> np.ndarray      # int, -1 on inactive bars
stats  = concurrency(open_rn: np.ndarray, close_rn: np.ndarray, n_bars: int | None = None) -> ConcurrencyStats
```

Lives at the top level (`forgedge.episodes`) since both M1 and M3 need it and
neither owns it. **Episodes** group activations by *signal* — a persistent
multi-bar state is one thing happening, not several (removes the per-bar
counting artefact `event_counting="episode"` also targets). **Concurrency**
groups by *price path* — trades from clearly separate episodes still
overlap whenever the holding period outruns the gap between them; this is
what statistical inference needs, since overlapping trades are not
independent observations. `gap` (default `1`) is the maximum bar-interruption
still bridged into the same episode.

`ConcurrencyStats` — `mean: float` (over occupied bars only), `peak: int`,
`occupied_bars: int`, `position_bars: int`, `n_trades: int`, and the
property `.effective_trades` = `n_trades / mean`. Both endpoints of an
`[open, close]` interval are inclusive — a same-bar round trip
(`target_h=0`) occupies exactly one bar.

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

`frames` must be the *post-Event-Discovery* frames (`ForgeResult.event_frame`
per ticker), not raw KPI Tables — the cross-ticker replay needs the derived
feature columns the rule expressions reference.

`RegistryConfig` fields and defaults (`src/forgedge/rule_registry/models.py`):
`overlap_threshold: float = 0.70` (Jaccard ≥ this → duplicate, weaker PF
flagged), `gain_corr_threshold: float = 0.70` (reporting only),
`cross_pf_threshold: float = UNSET` (session-resolved from M3's
`partial_min_profit_factor` — **no longer an independent `2.0`**, see
*Genericity* below), `min_cross_pf_retention: float = UNSET` (→ `0.8`, from
`PipelineContext.cross_pf_retention`), `generic_ratio_threshold: float =
2/3` (pass as a fraction, not `0.67` — a rule passing exactly 2-of-3 tickers
has ratio `0.6666...`, clearing `>= 2/3` but not `>= 0.67`), `cross_min_active:
int = 10`, `export_format: "excel"|"csv" = "excel"`, `export_duplicates: bool
= True`, `export_non_generic: bool = True`, `html_include_tradelog: bool =
True`, `html_charts: bool = True`, `timestamp_col: str = UNSET` (→
`"open_dt"`).

### Genericity is a transfer test, not a quality test

`cross_pf_threshold` used to default to `2.0` independently of M3, while
`partial_min_profit_factor` admits rules at `1.5` — so a `PARTIAL-EDGE` rule
had to do *better* away from home than at home to be called generic, and
that whole class was excluded from genericity by construction. The verdict
is now `PASS ⟺ pf >= floor AND pf >= retention × pf_home`: the absolute half
asks "is it tradeable there", the relative half asks "does it transfer".
Quality stays on the M3 verdict and grade, where the registry already
records it. `CrossTickerResult.bar: float = nan` reports the number each
verdict was actually measured against (the max of the two conditions'
effective thresholds).

`RuleDocument` — `rule_id`, `expression`, `source_ticker`,
`source_alpha_id`, `verdict`, `grade`, `is_duplicate`, `duplicate_of`,
`cross_ticker: dict[str, CrossTickerResult]`, `cross_ticker_score`,
`is_generic`, `classification: "GENERIC"|"PARTIAL"|"SPECIFIC"|"ISOLATED"`.

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
`.to_text()`, `.findings` (full list). `Finding` is reused verbatim by
`config_report()` (above) for coherence findings — same vocabulary, two
different subjects (data vs. configuration).

## Time budget (purging / embargo)

```
TimeBudget.build(n_bars: int, train_ratio: float = 0.7, horizon_bars: int = 0,
                  purge_bars: int | None = None, embargo_bars: int = 0,
                  event_train_ratio: float | None = None) -> TimeBudget
```

`purge_bars` defaults to `horizon_bars` when omitted. `event_train_ratio`
(new, F6/#180) records Event Discovery's own split ratio when it differs
from `train_ratio`, so the budget can state M1's actual axis
(`TimeBudget.event_split`/`.event_split_idx`) instead of it being inferred
from `split`, which M1 may not use at all — `forge_preset()`'s M1
`train_ratio=1.0` (the whole table, invariant #1: M1 never observes a
forward return, so an OOS reservation protects nothing there) now shows up
correctly as "M1 whole span, by choice" in `budget.describe()` rather than
implying a 70/30 split M1 never made.

Passed to `forge()` via `time_budget=...`, threaded into `EventDiscovery`,
`AlphaDiscovery` **and now `RuleDiscovery`** (F6 closed the gap where M3 cut
its own timeline with no awareness of the session axis — it still uses its
own walk-forward geometry, but can now report which folds' test windows
overlap the span M2 fit the target on, via `.is_in_sample(index)` /
`WalkForwardSplit.tests_in_sample`). Purging removes IS rows whose forward
window crosses into OOS (on by default for Alpha Discovery — purge width =
`max(horizon_grid)` — and for Rule Discovery's own walk-forward). To
reproduce pre-purging numbers exactly, pass `purge_bars=0` explicitly (and
`RuleWalkForwardConfig(purge_bars=0)` for M3). Embargo is `0`/opt-in
everywhere. `budget.describe()` for a human-readable summary of every
module's actual axis.

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
defaults to `DiscoveryConfig(train_ratio=1.0)`. `opt.validate_oos()` /
`.discover_alpha()` raise `RuntimeError` if called before `.run()`, same
"call run() first" pattern as every other module.

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
validates data eagerly, opt-in; `config_report()` validates configuration
eagerly, opt-in; `forge(strict=True)` (the default) is what actually turns a
structural configuration `FAIL` into a stopped run.

| Exception | Typical trigger |
|---|---|
| `ValueError` | invalid enum-like string (`direction`, `target_mode`, `buy_type`, `entry_mode`, `selection_mode`, `threshold_mode`, `timeframe`, `preset`, …), out-of-range numeric config field, mutually-exclusive `forge()` arguments (`manual_events` + `event_discovery_config`), a candidate/contract pair passed to `RuleDiscovery` that don't reference each other, **`forge(strict=True)` when `config_report()` finds a `FAIL`-level configuration** (see below) |
| `KeyError` | a required column is missing — OHLC columns, `timestamp_col`, `source_col`, an unknown candlestick pattern name |
| `RuntimeError` | an accessor called before `.run()` — consistently, across `MarketContext.distribution()`, `EventDiscovery.summary()`, `AlphaDiscovery.summary()`/`.promoted_contracts()`, `RuleDiscovery.grid_summary()`, `TargetOptimizer.validate_oos()`/`.discover_alpha()` |
| `TypeError` | wrong input type to `build_features`/`lag_features`, an unrecognized `forge_preset(**overrides)` key, a `GateParams` call using the old field names (`min_act`/`min_months`/`max_conc`, still present in several stale `examples/*.py` scripts), or arithmetic/comparison on a field still holding the `UNSET` sentinel (a hand-built config read before any `resolve_config()` pass) |
| `ImportError` | `load_kpi_config()` given a YAML path but PyYAML isn't installed |
| `FileNotFoundError` | `load_kpi_config()` given a path that doesn't exist |

`forge(strict=True)` (default) raises `ValueError` with a message ending in
`coherence.one_line()` when the resolved configuration bundle carries a
`FAIL`-level finding (`wf_bucket_too_short`, `m1_oos_fold_too_short`, or
`oos_span_too_short` — see *Configuration resolution and coherence* above).
Every other coherence finding, and every finding at all under
`strict=False`, is emitted as a `UserWarning` instead — never blocks.

Warnings raised via `warnings.warn` (not exceptions), worth not silencing:

- **`UserWarning` — one per `config_report()` finding**, when the finding
  is not (or is downgraded from) a `FAIL`. Message is `f"{code}: {message}"`.
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
walk-forward split; an M1 walk-forward fold too short to test its own
declared rate is marked `FoldResult.indeterminate=True` and excluded from
`ValidationResult`'s pass/fail count rather than counted as a failure
(`ValidationResult.passed` is then `None`, not `False`).
