---
name: forgedge
description: Use whenever working with the forgedge / FORGE (Feature-Oriented Rule Generation Engine) codebase or Python library — writing or debugging code that imports `forgedge`, calling `forge()` / `forge_multi()` / `forge_preset()` / `config_report()`, building or validating a KPI Table, working with Event Discovery, Alpha Discovery, Rule Discovery or the Rule Registry, interpreting EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA verdicts, replaying an `EventCandidate` or monitoring a published rule on fresh candles, or fixing/extending forgedge's own source and tests. Trigger this any time forgedge, FORGE, "KPI table", `EventCandidate`, `AlphaContract`, `RuleDiscoveryResponse`, `PipelineContext`, walk-forward OOS, or look-ahead bias comes up in the context of this repository — even if the user does not name the skill explicitly or only pastes an error/traceback from the library.
---

# forgedge / FORGE

FORGE is a quantitative-research pipeline that discovers algorithmic trading
rules from a **KPI Table** (OHLCV + technical indicators) and turns them into
statistically validated, backtested rule specifications. It is a pure
`numpy` + `pandas` library (no ML dependency) — a **research** system, not an
**execution** system: it never places orders or talks to an exchange, it only
produces boolean expressions plus operational parameters that some other
system can implement.

Read this file fully before writing code against `forgedge` or editing its
source — the pipeline's value comes entirely from architectural boundaries
that are easy to violate by accident (see *Invariants* below). For anything
beyond what fits here, jump to `references/api-reference.md` (full public API,
config dataclass fields and defaults) or `docs/manual-en.md`
(`docs/manuale-it.md` for Italian) — a comprehensive, example-verified manual
covering installation through production architecture, troubleshooting, best
practices/anti-patterns, an FAQ and a glossary. This file is a condensed
operating summary distilled from that manual and the current source, not a
replacement for either — where the two disagree, this file follows the
source (the manual is large and occasionally lags a fast-moving fix; every
claim below was checked against `src/forgedge/` directly).

## The pipeline: five modules, one direction, resolved once

```
KPI Table (OHLCV + indicators)
  │
  ▼ resolve   build the session's PipelineContext, resolve every UNSET config
  │           field, run config_report() — a FAIL raises under strict=True
  ▼ M0  Market Context      classify each bar's regime            → + 'regime' / 'regime_stable'
  ▼ M1  Event Discovery     mine boolean events, NO forward return → list[EventCandidate]
  ▼ M2  Alpha Discovery     derive target, measure predictive power → list[AlphaContract]
  ▼ M3  Rule Discovery      realistic backtest, walk-forward OOS    → EDGE/PARTIAL-EDGE/NON-EDGE/INSUFFICIENT-DATA
  ▼ M4  Rule Registry       dedup, cross-ticker, genericity          → flat table + HTML report
```

| # | Module | Answers | Key classes | Status |
|---|---|---|---|---|
| 0 | Market Context | Which regime is this bar in? | `MarketContext`, `MarketContextConfig`, `EMAProxyConfig` | done |
| 1 | Event Discovery | Is this indicator pattern stable and repeatable? | `EventDiscovery`, `DiscoveryConfig`, `EventCandidate`, `CustomEvent`, `EventWalkForwardConfig` | done |
| 2 | Alpha Discovery | Does the event predict an oriented return? | `AlphaDiscovery`, `AlphaConfig`, `AlphaContract` | done |
| 3 | Rule Discovery | Is it profitable under real order mechanics? | `RuleDiscovery`, `RuleDiscoveryConfig`, `RuleDiscoveryResponse`, `RuleWalkForwardConfig` | done |
| 4 | Rule Registry | Does it generalise across tickers? | `RuleRegistry`, `RegistryConfig`, `RuleDocument` | WIP |

Data only ever flows forward. Each module consumes exactly the formal
artefact the previous one produced and passes a new one on — it never reaches
back into an earlier module's internals.

**Every run now starts with parameter resolution.** `forge()` builds a
`PipelineContext` (the session's single source of truth for timeframe, schema
column names, fee, and statistical policy — see `forgedge.resolver`), resolves
every config field a caller left unset against it, and runs
`config_report()` to check the *resolved* bundle for internal contradictions
(§*Configuration coherence* below) before Module 0 ever executes. This closed
a real, measured class of bugs where two individually reasonable settings in
different modules were jointly impossible and surfaced as a wall of silent
rejections indistinguishable from "the signal is bad" — see
`docs/analysis/pipeline_parameter_coherence.md` for the full audit
(17 config classes, 158 fields, 16 findings) that motivated this layer.

## Invariants — do not break these

These are the whole reason the pipeline exists; a code change or a piece of
generated code that violates one of them silently reintroduces the exact bias
FORGE was built to eliminate. If a task seems to require breaking one, stop
and flag it rather than working around it quietly.

1. **Event Discovery (M1) never observes the forward return.** Thresholds and
   AND-compositions are chosen purely from the temporal structure of
   indicators. The moment a return/target leaks into M1, every downstream
   statistic is look-ahead-biased.
2. **Event thresholds are immutable once Event Discovery has fixed them.**
   Wanting a different threshold means minting a new `EventCandidate` by
   re-running M1 with a different config — never mutating a discovered
   candidate's threshold in place.
3. **The economic target is derived per event, never assumed up front.**
   Alpha Discovery (M2) picks `holding_period_h`, `direction` and `sell_pct`
   from the data (`h* = argmax|z_h|` etc.) — `TargetOptimizer` is the one
   documented exception (see below), and even there the fixed target is
   explicit and separate from the standard flow.
4. **Three domains stay separate: temporal structure (M1) → statistical
   predictivity (M2) → operability/fees/fills (M3-M4).** Don't let fee or
   fill-rate logic creep into M1/M2, and don't let M3/M4 re-derive statistical
   promotion gates that M2 already owns.
5. **The tradeable verdict is gated by walk-forward OOS in M3.** M2's own OOS
   confirmation is recorded on the contract, but M3 is the only economic
   judge — a contract that looks great in M2 can still be `NON-EDGE`.
6. **FORGE is a research tool, not an execution engine.** It has no order
   placement, no exchange connectivity, no position management. Don't add any.
7. **Thresholds are distributional (asset/period-specific percentiles), not
   hardcoded constants.** This is what makes an `EventCandidate` transferable
   across assets and time windows.
8. **FORGE is stateless across sessions.** Nothing persists between runs
   except what the caller explicitly exports (contracts, the registry's flat
   table). Don't design features that assume hidden cross-session state.
9. **A configuration that cannot produce a verdict fails loudly, not
   silently.** `forge(strict=True)` (the default) raises `ValueError` before
   running anything when `config_report()` finds a `FAIL`-level
   contradiction (e.g. an M3 selection window structurally too short for the
   arrival rate it was told to demand). A wall of rejections is
   indistinguishable from "the signal is bad" — refusing to start is the
   honest response, not a bug to route around with `strict=False` by
   default.

## Quick-start patterns

Full parameter docs for every function/class below live in
`references/api-reference.md`.

### 1. One-call pipeline

```python
import pandas as pd
from forgedge import forge

kpi = pd.read_parquet("kpi_table.parquet")   # needs 'close' + 'open_dt' (or DatetimeIndex)
result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.summary())                # one row per candidate + rule_verdict
for contract, response in result.edges():   # EDGE / PARTIAL-EDGE only
    print(contract.alpha_id, response.verdict)
print(result.registry.summary())        # M4 — catalogued rules
```

`forge()` may raise `ValueError` before touching M0 if the resolved
configuration is structurally incoherent (invariant #9) — see pattern 6.
Declaring `timeframe` explicitly matters even though `"1H"` is the fallback:
an *inherited* default is not compared against the data, but a *declared* one
is (`config_report`'s `timeframe_mismatch` check). On a daily-or-slower
timeframe every bar-counting field (`AlphaConfig.horizon_grid`,
`BacktestParams.buy_delay_bar`/`target_h`, `MarketContextConfig.stable_window`,
…) is now session-resolved to match — see pitfall #2 for the one place this
substitution still does not reach. `forge()` also runs the default fast
rotation null (`fast_null=True`) that prices the search's multiple-testing
surface; a contract that only wins that lottery is capped at `PARTIAL-EDGE`
even if every other gate passes.

`ForgeResult` also carries `.context` (the resolved `PipelineContext`),
`.resolution` (a `ResolutionTrace` — every field the resolver derived, the
rule that fired, and the inputs it read), and `.coherence` (the
`ConfigReport` the run executed with). Log `result.resolution.describe()`
next to `result.ledger.describe()` — together they say what was searched and
with what configuration.

**M2 composes by default now (issue #254 Phase 8).** `forge(two_pass_composition=True)`
is the default: Event Discovery stays 1D-only, `AlphaDiscovery` grades every
1D candidate A-D, `forgedge.composition.grade_guided_compose()` pairs them
by grade (a much stronger signal than M1's own structural tpm/dispersion
pairing — see `docs/analysis/issue_254_two_pass_composition_plan.md`), and
a second `AlphaDiscovery` pass evaluates the composed pool from scratch.
`ForgeResult.grading_candidates`/`.grading_contracts` expose pass 1's
pre-composition artefacts; `.candidates`/`.contracts`/`.promoted` still mean
"what M3 actually operated on" (pass 2's pooled output). Pass
`two_pass_composition=False` to get the pre-#254 single-pass behaviour, in
which case `event_discovery_config.max_and_components` (default `1`, i.e.
composition off) is the lever to raise instead. See
`references/api-reference.md`'s M1/M2 sections for the full parameter list.

### 2. Presets instead of hand-tuned gates

```python
from forgedge import forge, forge_preset

disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="BTC")
result = forge(kpi, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
                rule_discovery_config=rd_cfg)
```

Four presets, chosen by search profile, not by asset:
`"sniper"` (rare/regular, high precision — do **not** pair with the rotation
calibrator), `"balanced"` (sensible default), `"sweep"` (wide/permissive —
pair with `rotation_calibration=RotationConfig(k>=100)` and
`promoted_contracts(min_lift=0.05)`), `"burst"` (time-concentrated,
momentum/regime-change events). `forge_preset()` sets M3's rate at the
preset's own ratio to M1's and leaves `walk_forward.min_train_months` unset
so the resolver sizes it from that rate with a 95% Poisson margin, instead
of the historical flat `6` — but **on `"1D"` this can still raise under
`strict=True`** on a realistic-length history: a low daily arrival rate
derives a long training window (e.g. ~20 months at `"balanced"`'s rate),
which can leave too little pooled OOS span for `min_oos_trades` — verified
on this repo's own 29-month reference fixture, `"sniper"`/`"balanced"`/
`"sweep"` at `"1D"` all raise `oos_span_too_short` there, `"burst"` (a
higher rate) does not. This is not a broken preset — it's the coherence
checker correctly refusing to run a session that can't support the test it's
configured to demand — see pattern 6 and pitfall #8. `forgedge.presets.preset_info()`
prints the resolved numeric parameters for any/all presets.

### 3. Manual step-by-step (for drill-down or contributing)

```python
from forgedge import (
    MarketContext, MarketContextConfig, EMAProxyConfig,
    EventDiscovery, DiscoveryConfig, EventWalkForwardConfig,
    AlphaDiscovery, AlphaConfig,
    RuleDiscovery,
)

enriched = MarketContext(kpi, config=MarketContextConfig()).run()

ed = EventDiscovery(enriched, config=DiscoveryConfig(train_ratio=0.80))
candidates = ed.run()

ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="BTC", timeframe="1H"))
contracts = ad.run()
promoted = ad.promoted_contracts()

by_id = {c.event_id: c for c in candidates}
for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    resp = RuleDiscovery(ed.df, contract, cand).run()
    print(contract.alpha_id, resp.verdict)
```

`forge()` runs exactly this sequence internally — use this form when you need
to inspect an intermediate artefact (`ed.summary()`, `mc.distribution()`,
`ad.summary()`) or when contributing a change to one module in isolation.
Every module's constructor still calls `forgedge.resolve_config()` on its own
config even when built this way, so schema/fee/UNSET-field propagation
partially applies standalone too — but only `forge()` builds a
`PipelineContext` from the actual `timeframe` and KPI table, so a
timeframe-scaled field (`horizon_grid`, `target_h`, `buy_delay_bar`, …) left
unset on a hand-built config resolves to its **hourly** fallback regardless of
what `timeframe=` you wrote on the config object (pitfall #2).

Note the import: the walk-forward config classes were renamed and are no
longer interchangeable — `EventWalkForwardConfig` (M1, event-level OOS
validation) vs `RuleWalkForwardConfig` (M3, walk-forward backtest). The
top-level `forgedge.WalkForwardConfig` is a legacy alias for
`RuleWalkForwardConfig` only — prefer the explicit names (pitfall #7).

### 4. Building a KPI Table from raw OHLCV

```python
from forgedge import build_features, candle_features, lag_features

kpi = build_features(candles, timestamp_col="open_time")   # base indicators + open_dt
kpi = candle_features(kpi)                                 # scale-free candle geometry
kpi = lag_features(kpi, "close", like="_ema_", periods=[1, 2, 3])
```

Custom/indicator columns must follow `{base}_{indicator}_{period}` (e.g.
`close_rsi_14`, `close_ema_09`) to be recognised as a semantic family and
paired for ratio features in Event Discovery — see pitfall #4. Validate the
result before feeding it to `forge()`:

```python
from forgedge import summary_report
rep = summary_report(kpi, return_report=True, verbose=False)
if rep.has_critical:
    raise ValueError(rep.one_line())
```

Event Discovery also pairs columns beyond that generic same-family grouping,
via five dedicated, narrower arity-2 feature families: cross-column/cross-time
OHLC pairs and price-scale-indicator-vs-lagged-OHLC-base (both always on —
the latter tunable/disableable via `DiscoveryConfig.indicator_lag_cross_lags`,
default `(1, 3)`) fire on any KPI Table; MACD-vs-signal, price-vs-volume
return, and `candle_features()` geometry-vs-`natr` only fire when their
prerequisite columns exist (e.g. `"macd"` enabled in `build_features()`). A
column that opts out of the generic naming convention can still be reached by
one of these — see `docs/manual-en.md` §8 for exactly which columns trigger
which, and §17 for the measured cost of the always-on ones.

### 5. Monitoring a discovered edge on fresh data

Use **Rule Discovery**, not Alpha Discovery, to check whether a published
edge still holds — see pitfall #3 for why re-running Alpha Discovery on new
data is the wrong tool.

```python
from forgedge import RuleDiscovery, RuleSpec, rule_performance_report

new_bars = full_df[full_df["open_dt"] > train_df["open_dt"].max()]
eval_df = pd.concat([train_df, new_bars]).drop_duplicates("open_dt")
resp = RuleDiscovery(eval_df, contract, cand, time_budget=result.time_budget).run()
print(resp.verdict, resp.walk_forward.consistency)

# Or, for every tradeable rule of a forge() run at once, with an HTML report:
specs = RuleSpec.from_forge_result(result)
html = rule_performance_report(result, fresh_candles)
```

`rule_performance_report()` *replays* rules on fresh candles for monitoring
charts (equity curve, rolling expectancy, MAE/MFE) — a different job from
just reading what a single verdict already says. For that,
`rule_summary_report(resp, contract=contract, fmt="text"|"html", save=True)`
(also `text_report`/`html_report` directly) renders every parameter/IS/OOS/
walk-forward statistic the response already carries, with `contract=`
optionally folding in `rotation_p`/`rotation_threshold` from the originating
`AlphaContract` (Modulo 2, a different object than `resp`). `save=True`
writes it to disk (default `f"{resp.alpha_id}.{txt|html}"`) while still
returning the string.

Passing the original run's `time_budget` is optional but lets Rule Discovery
report which walk-forward folds land inside the span Alpha Discovery actually
fit the target on (`WalkForwardSplit.tests_in_sample`) — informative, not a
gate; it does not change the verdict.

### 6. Checking configuration coherence before a run

```python
from forgedge import config_report

rep = config_report(disc_cfg, alpha_cfg, rd_cfg, kpi=kpi, timeframe="1D")
print(rep.to_text())          # resolution trace, then every finding
if rep.has_critical:
    raise ValueError(rep.one_line())
```

The sibling of `summary_report()`: that one validates the **data**, this one
validates the **configuration**, in the same `Finding(level, code, message)`
vocabulary. `config_report()` and `forge()` run the *same* resolver, so
`rep.configs` is literally what a `forge()` call with these configs would
execute — not a reconstruction. It never raises, warns, or mutates its
inputs; `forge(strict=True)` (the default) is what turns a `FAIL` finding
into a stopped run. Useful whenever you assemble configs by hand and want to
know what will actually happen before spending a full pipeline run finding
out.

`config_report()` is blind to data by construction (it must resolve without a
frame), so it cannot catch the case where a preset is internally coherent yet
still rejects every candidate a *specific asset* actually produces — that
used to surface only as a bare `"M1 Event Discovery — 0 candidate(s)"` log
line with no diagnosis, easily mistaken for "the pipeline is broken" rather
than "these thresholds and this asset's event statistics disagree" (#215).
`EventDiscovery.event_distribution_report` (`str | None`, populated by
`.run()`, `None` before) closes that gap: an always-computed diagnostic of
the tpm/dispersion distribution actually observed across every raw
candidate the gate evaluated, against the configured thresholds — below a
15% gate-survival rate it also suggests concrete parameters at the observed
median. `forge()`'s M1 stage line carries this text; `result.event_discovery
.event_distribution_report` exposes it for code-level inspection.

## Common pitfalls

1. **Passing the wrong frame to `AlphaDiscovery`.** When building the pipeline
   by hand, pass `ed.df` (Event Discovery's post-pipeline frame, with derived
   features already attached), not the original KPI Table.
2. **Silent hourly-fallback bar-counting fields when *not* using `forge()`.**
   `AlphaConfig.horizon_grid`, `BacktestParams.{target_h, buy_delay_bar}` and
   several other fields default to the `UNSET` sentinel and are meant to be
   filled by the session's `PipelineContext`. `forge()` builds that context
   from the real `timeframe` and KPI table, so every path through it resolves
   correctly. Constructing `AlphaDiscovery`/`RuleDiscovery` directly still
   calls the resolver internally, but with a **default** `PipelineContext`
   (`timeframe="1H"`) — it does *not* read the `timeframe=` you set on the
   config object itself. Left unset outside `forge()`, these fields fall back
   to their hourly calibration (`horizon_grid=(1,2,4,8,12,24)`,
   `target_h=24`, `buy_delay_bar=6`) regardless of your data's real bar size —
   on daily candles that scans holding periods of up to 24-48 days by
   accident. Pass `horizon_grid`/`target_h`/`buy_delay_bar` explicitly for
   standalone daily-or-slower use, or just use `forge_preset()`
   (`forgedge.presets.default_horizon_grid()` covers only the `AlphaConfig`
   grid, not the M3 fields).
3. **Re-running Alpha Discovery on history that predates training to
   "monitor" an edge.** Mixing pre-training regimes into M2 commonly yields a
   spurious `direction="undetermined"` ("no derivable target") — not because
   the edge is gone, but because opposing-regime activations cancel each
   other's mean advantage. Use `RuleDiscovery` on the new bars instead
   (pattern 5 above).
4. **Non-conforming column names silently opt out of pairing.** A feature
   that doesn't match `{base}_{indicator}_{period}` (with `base` in
   `close/high/low/open/volume`) still works standalone but is never paired
   into a ratio/spread feature by Event Discovery — if a custom indicator
   never shows up in candidates, check its name first.
5. **`manual_events` and `event_discovery_config` are mutually exclusive** in
   `forge()` — passing both raises `ValueError`. Manual injection also skips
   AND composition entirely.
6. **Preset/calibrator mismatch.** `"sniper"` assumes a long, clean IS window
   and should not be combined with the rotation calibrator (too few events to
   calibrate against); `"sweep"` is permissive by design and is meant to be
   paired with `rotation_calibration=RotationConfig(k>=100)` plus a
   `min_lift` filter on the promoted contracts.
7. **Treating the two `WalkForwardConfig`s as one class.** `EventWalkForwardConfig`
   (M1, `n_splits` = OOS *validation* windows) and `RuleWalkForwardConfig` (M3,
   `n_splits` = walk-forward *test* windows) share a field name with different
   semantics and used to share a class name too. The bare top-level
   `forgedge.WalkForwardConfig` now resolves to `RuleWalkForwardConfig` only —
   importing it and handing it to `DiscoveryConfig(walk_forward=...)` is a
   `TypeError` waiting to happen. Import the explicit name for the module
   you're configuring.
8. **`forge(strict=True)` raising `oos_span_too_short` (or
   `wf_bucket_too_short`) on `forge_preset(..., "1D")` with a modest history.**
   `min_train_months` is correctly sized from `criteria.min_tpm` with a
   Poisson margin instead of a flat `6` — but a *low* daily rate (the point of
   `"sniper"`, and `"balanced"`/`"sweep"` at their stock values) derives a
   *long* training window, which can leave too little pooled OOS span for
   `min_oos_trades` on a history in the 2-3-year range. Verified empirically
   on this repo's own 29-month reference fixture: `"sniper"`/`"balanced"`/
   `"sweep"` at `"1D"` all raise; `"burst"` (higher rate) doesn't. This is the
   checker doing its job, not a broken preset — the raised message names the
   exact fix (`min_tpm` to raise to, or `min_oos_trades` to lower); longer
   history, a higher-rate preset, or `strict=False` (to see the
   `INSUFFICIENT-DATA`-heavy result anyway) are the other options. Separately,
   hand-pinning `RuleWalkForwardConfig.min_train_months` or
   `SelectionCriteria.min_tpm` to fixed numbers instead of leaving them
   `UNSET` opts back into the disconnected-values bug class this whole layer
   exists to catch (`m3_stricter_than_m1`, `wf_bucket_too_short`).
9. **Copying `GateParams(...)` from this repo's own `examples/*.py` scripts.**
   Several scripts (`alpha_discovery_usage.py`, `extended_usage.py`,
   `kpi_table_1d.py`, `search_rotation_calibration.py`,
   `lowfreq_null_diagnostic.py`, `lowfreq_endpoint_diagnostic.py`) predate a
   `GateParams` API change and still pass `min_act`/`min_months`/`max_conc`,
   which now raise `TypeError`. Translate to the current fields (`min_tpm`,
   `max_dispersion`, `event_counting`, `min_episodes`, `episode_gap`) —
   `event_counting="bar"` reproduces the old counting semantics most closely.
   `examples/kpi_builder_usage.py` is unaffected.
10. **Calling an accessor before `.run()`.** `MarketContext.distribution()`,
    `EventDiscovery.summary()`, `AlphaDiscovery.summary()`/
    `.promoted_contracts()`, `RuleDiscovery.grid_summary()`, and
    `TargetOptimizer.validate_oos()`/`.discover_alpha()` all raise
    `RuntimeError: Call run() before ...` until the corresponding `.run()` has
    executed — a deliberate guard, not a bug to route around.
11. **Extending a discovery window with only the new bars.** Re-evaluating an
    `EventCandidate` on a frame whose index differs from its cached training
    activation series triggers a `UserWarning` and a fallback to `.apply()`;
    if the re-evaluated activation count collapses under ~10% of the training
    count, that's usually why direction reads `"undetermined"` — rolling
    baselines (pctrank, z-score) lost the history they need, not because the
    edge disappeared. Fix: `pd.concat([train_df, new_bars])`, never `new_bars`
    alone.
12. **Reading `AlphaContract.rejection_reasons` expecting the old
    `[diagnostic]`-prefixed non-blocking entries.** Those now live on a
    separate `AlphaContract.diagnostics` list. `rejection_reasons` holds only
    causes that actually blocked promotion (in practice, just "no derivable
    target") and is empty on every promoted contract; a promoted contract
    routinely carries several `diagnostics` entries. Code written against the
    old, merged field will silently see an empty `rejection_reasons` on
    contracts it used to find diagnostics in.
13. **`RuleDiscoveryConfig.entry_mode` now defaults to `"auto"`, not
    `"limit"`.** This changes verdicts, not just execution — see the
    *Entry mode* note below before assuming a verdict means what it used to.
14. **`TargetOptimizer` sits outside the parameter-coherence resolver.**
    `target_optimizer.py` imports nothing from `forgedge.resolver`/
    `forgedge.unset` — it predates, or was deliberately kept outside, the
    #173–#185 audit. `TargetConfig` itself has no `UNSET` fields, so it's
    unaffected either way, but `discover_alpha()`'s internal `AlphaConfig()`
    is built without ever going through `forge()`'s session resolver (the
    module docstring says as much: "does not touch `forge()` or
    `ForgeResult`"). On daily-or-slower data this reaches the same
    hourly-grid footgun as pitfall #2, through `TargetOptimizer` instead of
    `AlphaDiscovery` directly — pass `horizon_grid` explicitly on the
    `AlphaConfig` you hand to `discover_alpha()`. Also note `discover_alpha()`
    silently overwrites `config.fixed_target`/`.target_mode`/
    `.trend_sma_mult` on whatever `AlphaConfig` you pass it, even if you set
    `fixed_target` yourself.
15. **`GridSpec`'s auto-fan covers `buy_drop_pct`/`sell_pct`/`target_h`, not
    `buy_delay_bar`.** `build_grid()` only ever emits the single
    `base_params.buy_delay_bar` value in every cell unless you set
    `GridSpec.buy_delay_bar` explicitly — easy to miss since the other three
    axes do fan out automatically.
16. **`RuleRegistry.flat_table()`'s default does not filter.**
    `apply_filters=False` by default — duplicates and non-generic rules stay
    in a plain `reg.flat_table()` call regardless of `RegistryConfig
    .export_duplicates`/`.export_non_generic`; those flags only apply via
    `flat_table(apply_filters=True)` or `reg.export(...)`/`.html_report(...)`.
    Cross-ticker replay doesn't reuse a rule's literal threshold either — it
    recalibrates each threshold-bearing component onto the target ticker's
    own empirical distribution at the same percentile the threshold occupied
    on the source ticker (`recalibrate_candidate`), so `GENERIC`/`PARTIAL`/
    `SPECIFIC`/`ISOLATED` measures transfer of a *relative* pattern, not
    reuse of an absolute number.
17. **A high-`min_tpm` contract wrongly early-eliminated with `"total_trades
    0 < 35 (1mo × ... tpm, not significant)"` under `selection_mode
    ="walk_forward"`.** Fixed in #217 — `_early_elimination`'s pre-screen on
    the walk-forward's *first train window* used to re-derive its trade
    floor as `n_months × min_tpm`, the formula that's correct over a full
    selection span but wrong on that window, which `min_train_months` already
    sizes to reach exactly 10 trades at `min_tpm` (95% Poisson margin) — the
    two agree only while `min_tpm` is low enough that the window doesn't
    round down to its 1-month floor. If you're on an older `forgedge` and see
    plausible contracts eliminated this way at a high `min_tpm` (e.g. the
    bar-equivalent rate of an hourly preset), that's this bug, not a real
    frequency shortfall — upgrade rather than loosening `min_tpm`.
18. **`ANDComposer`-composed events ignoring `event_counting="episode"`
    entirely, always gating on bar-level `max_dispersion`.** Fixed in #226 —
    `and_composer.py`'s vectorized Pass 2 gate never adopted episode-mode
    semantics (rate + `min_episodes` + `episode_index_of_dispersion` vs a
    Poisson-floor-derived `eff_max_dispersion`; `max_dispersion` itself is
    inert in this mode), so composed pairs/triples were silently
    suppressed or wrongly accepted relative to what `ConsistencyGate
    .evaluate()` would have decided for the same series under the
    (default) episode mode. The fix extracted the gate decision itself
    (`consistency_gate._gate_pass`/`_eff_max_dispersion`) and the batch
    episode-detection primitive (`episodes.episode_starts` now dispatches
    on a 2D input) as the single shared implementation both the scalar
    single-event path and the composer's batched path call — not a
    re-synchronized duplicate. If you're on an older `forgedge`, AND-composed
    candidates under episode mode (the default) cannot be trusted to reflect
    the same pass/fail criteria as single events; upgrade rather than
    reasoning about the discrepancy.
19. **`ANDComposer.compose()` risking OOM under permissive `GateParams`
    (low `min_tpm`) on a realistic multi-year dataset.** Fixed in #228 — a
    side effect of #226's own fix: the episode-mode computation it added
    (`episodes._episode_starts_batch`) built several `(K, n_rows)` `int64`
    temporaries that all stayed alive at once, ~41x the input's own size
    (measured ~4.8 GB peak on a 5 000-pair x 23 352-row chunk vs 117 MB of
    input). `and_composer._CHUNK_SIZE=5000`'s own comment assumed the volume
    pre-filter always thins a chunk down to "≪ 5000" before that computation
    runs — true only for strict gate params; a low `min_tpm` (including the
    library's own `GateParams()` default) lets most of a chunk pass, so the
    worst case (full-size chunk) is the *normal* case under permissive
    params, not a rare edge case. The fix: `_episode_starts_batch` now uses
    `int32` and frees intermediates as soon as nothing downstream needs them
    (~2.2x lower peak on its own), and `and_composer._pair_chunk_size(n_rows)`
    bounds the chunk size itself against a fixed ~1 GB memory budget instead
    of assuming `n_rows` stays small — `_CHUNK_SIZE` is now only its upper
    bound, unchanged on the short histories it was calibrated against.
    Chunking is purely an implementation detail (verified: forcing a tiny
    chunk size produces byte-identical composed events to the default one).
    If you're on an older `forgedge` running AND composition with a
    permissive M1 gate (default `GateParams()`, `"sweep"`-style presets) on
    more than roughly a year of hourly-or-finer data, this is a real OOM
    risk, not a data-size fluke — upgrade rather than working around it by
    tightening `min_tpm`/`max_and_components`.
20. **AND-composed pairs/triples all sharing the same single component
    under a permissive M1 gate.** Fixed in #230. `ANDComposer.compose()`
    enumerates candidate pairs via `np.where(np.triu(valid_mask, k=1))` —
    row-major, exhausting every pair involving the *first* pool index before
    ever trying a second one — combined with "stop once `_MAX_PAIRS`/
    `_MAX_TRIPLES` is reached." Under permissive gate params (a low
    `min_tpm`, including the library's own `GateParams()` default) where
    most examined pairs pass, that early exit fires before the loop ever
    advances past the first index: measured on a realistic pool (~7000-10000
    events post-gate), *every one* of 2000 kept pairs shared the same single
    component, paired with 2000 different partners — not 2000 independent
    combinations. Permuting the pool's own order doesn't fix this (only
    changes *which* event dominates); the enumeration order of the pairs
    themselves has to be permuted. The fix: `_shuffle_order()` permutes pair
    enumeration (and each triple seed's own third-candidate search) with a
    fixed, deterministic seed — reproducibility unaffected (same input +
    config always yields the same composed events) — measured 3-4x more
    distinct source features touched and single-event reuse collapsing from
    2000 to single digits on the same fixture. `ANDComposer.compose()` also
    gained an opt-in `max_constituent_jaccard` parameter (default `None`,
    disabled) rejecting a pair whose two constituents' Jaccard similarity
    exceeds the threshold, at no extra cost — a real but secondary effect
    that only matters once enumeration order is representative; on its own
    (without the shuffle) it has almost no effect, since a non-representative
    sample rarely contains near-duplicates to begin with. Raising
    `_MAX_PAIRS`/`_MAX_TRIPLES` was evaluated as an alternative and rejected:
    real, non-trivial time/memory cost (an OOM was reproduced at
    `_MAX_PAIRS=10000` under a permissive preset in the same evaluation
    session) without fixing the underlying skew — a higher cap still let one
    component dominate, just more diluted. If you're on an older `forgedge`,
    AND-composed candidates from a large pool under permissive gate params
    cannot be assumed diverse — upgrade rather than raising the caps as a
    workaround.
21. **`EventDiscovery.run()` risking OOM on a rich KPI Table before M1 even
    reports how many candidates survived the gate.** Fixed in #232 — the
    generate-everything-then-filter pattern kept every raw candidate's full
    activation series resident for the whole run, gate-failing ones
    included. Measured on a 186-column KPI Table (SUIUSDC, ~2.7y hourly):
    50874 raw candidates cost **9.5 GB** just for their series, before the
    gate ever ran; at a permissive `min_tpm`/`dispersion_margin` where ~24%
    passed, that stacked with the resulting larger AND-composition pool to
    exceed a 15 GB container. Freeing a candidate's series as soon as it's
    gated instead of after the whole run measured **4.2x less** memory
    retained by the population. Fix: generation and gating are now
    interleaved per batch (one `TransformedSeries`, or one binary/
    categorical column, at a time — the granularity `EventGenerator` already
    produced them at) instead of collecting the full raw population first;
    `event_distribution_report`'s statistics are collected incrementally
    from each `GateResult` as it's produced, so nothing about the diagnostic
    changes. New `DiscoveryConfig.retain_raw_events: bool = True` — default
    **flipped to `False` in issue #254 Phase 8** (the common case is the
    standard `forge()` pipeline, which never reads `.raw_events`; measured
    ~2x lower peak RSS end-to-end at realistic scale on top of the 4.2x
    figure above). `TargetOptimizer`, which reads `EventDiscovery.raw_events`
    directly, pre-gate, sets `True` explicitly on its own fallback config
    rather than relying on this class default — do **not** set `retain_raw_events
    =False` on a `DiscoveryConfig` you hand to `TargetOptimizer` yourself,
    it would silently lose the atom pool it needs. If you're on an older
    `forgedge` running M1 on a wide KPI Table with permissive gate params,
    this is a real OOM risk that tightening `min_tpm` only works around, not
    fixes — upgrade instead.
22. **`forge(only_validated_events=True)` silently filtering nothing on a
    preset.** Fixed in #250. `only_validated_events` drops any `EventCandidate`
    whose `validation.passed` is not `True` — but that field is only ever set
    when M1's *own* walk-forward (`EventDiscovery._run_walk_forward`, distinct
    from M3's) actually ran, which needs **both**
    `DiscoveryConfig.walk_forward` configured **and** `train_ratio<1.0`.
    `forge_preset()` fixes `train_ratio=1.0` for M1 deliberately (its
    thresholds are purely distributional and never see the forward return —
    invariant #1 already rules out look-ahead bias unconditionally, so an
    M1-side OOS tail buys no extra safety, only less visibility into rare
    events) and does not accept `walk_forward` as an override key at all
    (`TypeError: Unexpected override keys`). So `only_validated_events=True`
    paired with an as-shipped preset leaves every candidate's
    `validation.passed=None` — the filter drops nothing, silently, and the
    caller has no signal the validation they asked for never ran.
    `config_report()`/`forge()` now surface this as the `WARN`-level
    `only_validated_events_inert` finding whenever `PipelineContext
    .only_validated_events` is `True` and the resolved `event_discovery`
    config can't run its own walk-forward. To actually use
    `only_validated_events`, override the preset's `DiscoveryConfig`
    directly: `dataclasses.replace(disc_cfg, train_ratio=0.7,
    walk_forward=EventWalkForwardConfig(n_splits=...))`.
23. **Code written against pre-#254-Phase-8 `forge()`/`DiscoveryConfig`
    defaults getting a different composition path (or a `ValueError`)
    without any code change.** As of Phase 8, `forge(two_pass_composition
    =True)` and `DiscoveryConfig(max_and_components=1, retain_raw_events
    =False)` are the defaults — previously `two_pass_composition=False`
    (M1's own structural `ANDComposer`) and `max_and_components=2`/
    `retain_raw_events=True`. A hand-built `DiscoveryConfig(max_and_components
    =2, ...)` passed to `forge()` with the (now-default) `two_pass_composition
    =True` raises `ValueError` (composition must happen in one place, not
    both) where it silently ran M1's own AND-composer before; pass
    `two_pass_composition=False` alongside it to keep the old behaviour
    exactly. Code that read `EventDiscovery.raw_events` off a bare
    `DiscoveryConfig()` now gets `None` instead of the pre-gate population —
    pass `retain_raw_events=True` explicitly. `TargetOptimizer` is
    unaffected either way — its own fallback `DiscoveryConfig` pins both
    fields explicitly, decoupled from this class-default change (see
    pitfall #21 and `references/api-reference.md`'s TargetOptimizer
    section).

### Entry mode and what a verdict now measures

`entry_mode="auto"` (default since #185, was `"limit"`) runs Rule Discovery
in two stages: Stage 1 backtests at a market entry (next-open fill, ≈100%)
and that verdict is **authoritative** — Stage 2 can never turn a `NON-EDGE`
into an edge. Stage 2 sweeps `buy_drop_pct` on a survivor, replays the
winner out-of-sample on Stage 1's own test windows, and adopts it only if all
three hold: `fill_rate >= min_fill_rate_opt`, `opportunity_sharpe >=`
market's (a *per-trade-frequency* Sharpe, not `StatisticalValidation`'s
capacity-annualised one — a point that trades less has to earn more per
trade to win this comparison), and `net_gain >= min_net_gain_retention ×`
market's. `RuleDiscoveryResponse.entry_optimization.failed_condition` names
which of the three stopped adoption (`"fill"` / `"sharpe"` / `"net_gain"`),
or `None` when adopted. This exists because the old `"limit"`-only mode let a
deep, rarely-filled limit inflate profit factor on a non-representative
subset of trades (the "fill confound") — `"limit"` is still fully supported
and is the right choice when the limit order *is* the strategy, not merely an
execution refinement.

### Nominal vs. effective sample size

`run_backtest` opens a position on every active bar with no flat-state check
— correct and unchanged, but positions can overlap (issue #168: 120 signal
bars, 76 episodes, 3.71 concurrently open positions on average in the
reference case). Overlapping trades share a price path and are not
independent observations, so `BacktestSummary` now reports
`n_episodes`/`mean_concurrent_positions`/`max_concurrent_positions`, the
trade ledger (`return_trades=True`) carries an `episode_id` per row, and
`StatisticalValidation.n_effective` (`total_trades / mean_concurrent_positions`)
is what the t-test, Deflated Sharpe and the power gate now consume instead of
the nominal trade count — a correction that can matter: 118 nominal trades
against ≈32 effective overstates significance by `sqrt(118/32) ≈ 1.93×`. The
economics (profit factor, expectancy, net gain) stay **nominal** — this is
reproducible capital-permitting reality, not something to "fix". Primitives
live in `forgedge.episodes` (`episode_starts`, `episode_ids`, `concurrency`)
for anyone measuring this directly. `StatisticalValidation.deflated_sharpe`
is a multiplicative haircut (`sharpe × sqrt(1 - γ·ln(n_trials)/ln(n_obs))`,
`γ` = Euler-Mascheroni, `n_obs` = the *effective* count above), not the
probabilistic Bailey/López de Prado DSR — it's a no-op when `n_trials<=1`
(every OOS validation, since OOS data played no part in selection) and
`nan` when the radicand goes negative. `WalkForwardSplit.tests_in_sample`
flags a fold whose *test* window starts before the session's own IS/OOS
boundary — OOS with respect to M3's own parameter selection but not with
respect to the target Alpha Discovery derived; `WalkForwardResult
.n_splits_in_sample` counts them. Purely a quality annotation, not a gate —
3 of 4 folds carry this flag on this repo's own reference fixture under
`forge()` defaults.

## Best practices

- Run `summary_report()` before every discovery session and decide
  explicitly what to do with `has_critical`/`has_warnings` — the library
  never validates or blocks data on its own (see pattern 4). Run
  `config_report()` the same way over hand-built configs before a long run
  (pattern 6) — `forge()` already does this internally and will raise on a
  `FAIL` by default, but checking first is cheaper feedback when iterating on
  configuration.
- Prefer `forge_preset()` over hand-assembling `DiscoveryConfig`/
  `AlphaConfig`/`RuleDiscoveryConfig` — the presets keep M1/M2/M3 frequency
  criteria mutually consistent, which is easy to get subtly wrong by hand.
  Its output can still trip `config_report()` on a modest history at `"1D"`
  (pitfall #8) — that's the checker doing its job, not a reason to
  hand-tune instead.
- Treat a run that comes back mostly or entirely `PARTIAL-EDGE` as
  informative, not broken. Check `rejection_reasons` (often
  `"search-level rotation null not cleared"`) and `diagnostics` before
  assuming misconfiguration — an earlier, more permissive version of this
  pipeline promoted noise almost as often as real signal on low-frequency
  data (`docs/analysis/lowfreq_robustness.md`), which is exactly what the
  default rotation null (see pattern 1) now guards against.
- Log `result.ledger.describe()`, `result.resolution.describe()` and
  `result.calibration.summary()` (or `.tippett_p`) alongside every run you
  persist — all three are cheap, already computed, and are exactly what
  you'll want later to explain why a verdict landed where it did.
- Use `RuleDiscovery`, never `AlphaDiscovery`, to check whether a previously
  discovered edge still holds on new data (pitfall #3).

## Contributing to forgedge itself

Source lives under `src/forgedge/`, one directory per module plus a few
top-level orchestration files:

```
src/forgedge/
├── forge.py              orchestrator: forge(), forge_multi(), ForgeResult
├── presets.py             forge_preset(), preset_info(), default_horizon_grid()
├── resolver.py             PipelineContext, ResolutionTrace, Constraint, resolve(), resolve_config()
├── config_report.py        config_report(), ConfigReport — coherence checking, mirrors summary_report()
├── unset.py                 UNSET sentinel, is_set(), coalesce()
├── episodes.py              episode_starts/episode_ids/concurrency — shared by M1 and M3
├── market_context/        M0
├── event_discovery/       M1
├── alpha_discovery/       M2
├── rule_discovery/        M3
├── rule_registry/         M4
├── calibration/           FastRotationNull, RotationCalibrator
├── kpi_builder/            build_features / candle_features / lag_features / pattern_features
├── ledger.py               HypothesisLedger
├── timebudget.py           TimeBudget (purged/embargoed IS/OOS split, shared to M3 too — F6)
├── target_optimizer.py    TargetOptimizer (target-first alternative workflow)
├── rule_report.py          RuleSpec, rule_performance_report
├── summary_report.py       data-quality diagnostics
└── docs/                   packaged module + spec docs (see below)
```

Each module directory typically has `models.py` (dataclasses/config),
`discovery.py`/equivalent orchestration file, and focused helper modules —
follow the pattern already in the module you're touching rather than
introducing a new one. A config field that materialises a session-wide fact
(schema column name, bar duration, arrival rate, significance level, fee)
should default to `UNSET` and be picked up by a resolver `Constraint` in
`resolver.py`, not hardcoded — read `docs/analysis/pipeline_parameter_coherence.md`
first if you're adding one; it is the design rationale for the whole
resolver/`config_report` layer and states the vocabulary (`Constraint`,
`Derivation`, `Violation`, the `PROPAGATION`/`STATISTICAL`/`STRUCTURAL`
stages) precisely.

Every public name lives in `src/forgedge/__init__.py`'s `__all__` — add new
public API there, and keep dataclass field docstrings in numpydoc style
(`Attributes` sections), matching the existing modules.

**Tests**: `tests/` — 800+ test functions across 18 files, run with
`pytest` (`testpaths = ["tests"]` in `pyproject.toml`, dev extra
`pytest>=7.0`). One `test_<module>.py` per module — including
`test_resolver.py`, `test_config_report.py` and `test_episodes.py` for the
newer top-level modules — plus `tests/test_golden.py` (end-to-end
regression, pinned via a session-scoped `forge()` fixture over
`tests/fixtures/ADA_1D_TRAIN.parquet`) and `tests/test_forge.py`
(orchestrator wiring). House style worth matching: no mocking except a
handful of `monkeypatch` "must-not-be-called" tripwires — everything else
runs the real pipeline against seeded synthetic data
(`np.random.default_rng(seed)`) built by local `_make_kpi_table()`-style
helpers, each documented with *why* that signal shape was chosen;
`pytest.approx(..., rel=...)` for every float assertion, never bare
equality; `pytest.raises(..., match=...)` / `pytest.warns(...)` to pin exact
messages, not just exception types. A golden value breaking is expected when
a legitimate pipeline change lands — re-pin it with a comment explaining why,
don't assume the test (or the change) is wrong. Run the whole suite with
`pytest` from the repo root, or a single module with
`pytest tests/test_event_discovery.py`.

The library depends on `numpy>=1.23` and `pandas>=1.5` only — don't add a new
runtime dependency without a strong reason; `scipy`/`statsmodels`-equivalent
primitives (Spearman, t-test, OU regression, Benjamini-Hochberg FDR,
incomplete beta) are deliberately reimplemented in pure numpy.

## Where to go deeper

Prefer these over re-deriving explanations from source — they're kept
current with the code, with the caveat above that source wins on conflict:

| Topic | File |
|---|---|
| Comprehensive practical manual — installation through production architecture, troubleshooting, best practices/anti-patterns, FAQ, glossary; every example verified against this repo | `docs/manual-en.md` (`docs/manuale-it.md` for Italian) |
| The parameter-coherence audit — design rationale for the resolver/`config_report`/`PipelineContext` layer, all 16 findings with measured numbers | `docs/analysis/pipeline_parameter_coherence.md` |
| Project overview, pipeline diagram, quick start | `README.md` (`README_it.md` for Italian) |
| Deep architectural guide — artefact YAML formats, principles, roadmap | `src/forgedge/docs/README.md` (Italian) |
| Concepts — event / alpha / rule, from first principles | `src/forgedge/docs/specs/concepts_en.md` (`_it.md`) |
| End-to-end production guide — config per module, checklists | `src/forgedge/docs/specs/how_to_use_en.md` (`_it.md`) |
| Global configuration reference | `src/forgedge/docs/specs/configuration_en.md` (`_it.md`) |
| Per-module spec | `src/forgedge/docs/specs/modulo_{0..4}_en.md` (`_it.md`) |
| Technical analyses (low-freq robustness, rotation-null calibration) | `docs/analysis/*.md` |
| Runnable examples per module, incl. the coherence audit and entry-mode impact | `examples/*.py` (several predate a `GateParams` API change — see pitfall #9 before copying `GateParams(...)` from one) |
| Interactive walkthroughs | `notebooks/0{1..6}_*.ipynb`, `notebooks/hurst.ipynb` |
| Full public API + config dataclass fields/defaults | `references/api-reference.md` (in this skill) |
