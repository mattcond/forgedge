# How not to use forge — silent misuse patterns

**Question:** which ways of calling `forgedge` produce no exception, no
`ValueError`, no `RuntimeError` — the pipeline runs to completion and hands
back a well-formed `ForgeResult`/`AlphaContract`/`RuleDocument` — but the
numbers inside that object are silently wrong, misleading, or not what the
caller intended?

**Method:** for each pattern below, the misuse is actually constructed
against this repository's own code and reference fixture
(`tests/fixtures/ADA_1D_TRAIN.parquet`, 882 daily ADA bars), run side by side
with the corresponding correct usage on the *same* inputs, and the
difference is measured, not asserted. Every number quoted here came out of
`python examples/how_not_to_use_repro.py` against the version of the
package in this repository at the time of writing — reproduce any of them
with `python examples/how_not_to_use_repro.py <section>` (section names are
given in each heading below).

This document complements, rather than duplicates, `docs/manual-en.md`
§21–23 (Troubleshooting / Best Practices / Anti-patterns) — those sections
narrate the *consequences* a user hits downstream (a confusing verdict, a
strange log line); this document narrates the *usage decision* that causes
them, with an executable proof, and exists specifically so an AI coding
agent working against `forgedge` can check "am I about to do one of these?"
before writing code, not just debug it afterwards. It is referenced from
`.claude/skills/forgedge/SKILL.md` for exactly that reason.

None of the six patterns below is a bug in the sense of "this should raise
and doesn't." Each is either a deliberate, documented default, or a
structural consequence of how session-scoped parameter resolution works
(`forgedge.resolver`, invariant #8/#9 in the skill file). The problem they
share is *discoverability*: nothing about calling the function wrong looks
different from calling it right, so the mistake only surfaces later, as a
result that looks like "the signal is weak" or "the edge decayed" instead of
"the call was wrong."

---

## 1. Bypassing `forge()` on non-hourly data (`hourly_fallback`)

**What looks fine:** constructing `AlphaConfig(asset="ADA", timeframe="1D")`
or `RuleDiscoveryConfig()` by hand and building `AlphaDiscovery`/
`RuleDiscovery` directly — for drill-down, for contributing a change to one
module, or simply because the one-call `forge()` API feels like more than is
needed. `timeframe="1D"` is right there on the config object.

**What actually happens:** setting `timeframe` on the config object does
nothing to the bar-counting fields that default to `UNSET`
(`AlphaConfig.horizon_grid`, `RuleDiscoveryConfig.base_params.target_h`/
`.buy_delay_bar`, …). Every module's constructor *does* call
`forgedge.resolve_config()` on its own config — but with no real
`PipelineContext` to resolve against, it falls back silently to the class
default calibration, which is **hourly**. `timeframe` on the config object is
read by application code (e.g. `AlphaConfig.asset`/`.timeframe` show up in
reports), not by the resolver's own `UNSET`-field derivation for these
specific fields, unless a `PipelineContext` built from the real
`timeframe=`/KPI table (which only `forge()` constructs) reaches it.

**Measured, this repository, `examples/how_not_to_use_repro.py hourly_fallback`:**

| Field | Resolved with **no** `PipelineContext` (standalone construction) | Resolved with the **real daily** `PipelineContext` (`forge()`) |
|---|---|---|
| `AlphaConfig.horizon_grid` | `(1, 2, 4, 8, 12, 24)` | `(1, 2, 3, 5, 7, 10)` |
| `RuleDiscoveryConfig.base_params.target_h` | `24` | `10` |
| `RuleDiscoveryConfig.base_params.buy_delay_bar` | `6` | `1` |

Same config text (`timeframe="1D"` written on the object both times).
`horizon_grid` silently scans holding periods of up to **24 daily bars**
(almost 5 weeks) instead of up to 10; the limit order is left resting for
**6 daily bars** instead of 1. No exception, no warning, no log line — the
run just proceeds and produces a `RuleDiscoveryResponse` whose operating
point was tuned for a hypothetical 24-hour-a-day-1-hour-bar session applied
to daily candles.

**Fix:** use `forge()` (or `forge_preset()`), which builds a real
`PipelineContext` from the actual `timeframe=` and KPI table before
resolving anything. If you must build modules by hand (drill-down,
contributing), pass `horizon_grid`/`target_h`/`buy_delay_bar` explicitly, or
construct a `PipelineContext.from_frame(kpi, timeframe=...)` and resolve
against it yourself (`resolve_config(cfg, "alpha", ctx)`) before handing the
config to the module.

**Skill reference:** pitfall #2 / #14 (this is the same footgun the
`TargetOptimizer` note describes, generalised to every hand-built config).

---

## 2. Assuming `GridSpec`'s auto-fan covers every axis (`grid_delay`)

**What looks fine:** leaving `RuleDiscoveryConfig(grid=GridSpec())` at its
defaults and trusting "the grid search" to explore the operating point —
`build_grid()`'s own docstring says it "fills the unset grid axes with a
sensible default around `base`."

**What actually happens:** `build_grid()` auto-fans `buy_drop_pct`,
`sell_pct` and `target_h` into a small symmetric range around the seed
`BacktestParams` — but **not** `buy_delay_bar`, which stays pinned to the
single base value in *every* grid cell unless the caller sets
`GridSpec.buy_delay_bar` to a sequence explicitly.

**Measured, `examples/how_not_to_use_repro.py grid_delay`:**

```
buy_drop_pct fan: [0.005, 0.008, 0.01, 0.012, 0.015]   (5 values)
sell_pct fan:     [0.01, 0.02, 0.03, 0.04, 0.05]        (5 values)
target_h fan:     [6, 12, 24]                           (3 values)
buy_delay_bar:    [3]                                   (1 value, in all 75 cells)
```

75 grid cells searched, and every single one uses the exact same fill-delay
window. A caller who wanted to know "is a 1-bar or a 6-bar fill delay
better for this rule" gets a fully-populated, plausible-looking grid result
that never actually asked that question.

**Fix:** pass `RuleDiscoveryConfig(grid=GridSpec(buy_delay_bar=(1, 3, 6)))`
explicitly whenever the fill-delay window is one of the things you want
searched, not assumed.

**Skill reference:** pitfall #15; also in `docs/manual-en.md` §21
("Troubleshooting").

---

## 3. Handing `TargetOptimizer.discover_alpha()` a config you expect it to keep (`target_optimizer`)

**What looks fine:** `TargetOptimizer.discover_alpha(config=my_alpha_cfg)`
where `my_alpha_cfg` already has `fixed_target`, `target_mode` and
`trend_sma_mult` set to values you chose — the method's own signature takes
an optional `AlphaConfig`, which reads like "use this, filling in only what's
missing."

**What actually happens:** the docstring is explicit that this is
intentional ("Its `fixed_target` is always set from this optimizer's
`TargetConfig` \[…\] overriding any provided value"), but the effect is
easy to trigger by accident: three fields on **the exact object you passed
in** are silently overwritten in place, no matter what you set them to,
before the config is ever used.

**Measured, `examples/how_not_to_use_repro.py target_optimizer`:**

```
caller_cfg.fixed_target BEFORE: 48h / short   (my own TargetConfig)
caller_cfg.fixed_target AFTER:  12h / long    (the optimizer's own TargetConfig)
caller_cfg.fixed_target is optimizer_target (not my_own_target): True
```

`target_mode` and `trend_sma_mult` are overwritten the same way. If you
built one `AlphaConfig` you intend to reuse across both a `TargetOptimizer`
run and a plain `forge()`/`AlphaDiscovery` run, the *same object* now
silently carries the optimizer's target after the first call.

**Fix:** never pass an `AlphaConfig` you intend to reuse elsewhere into
`discover_alpha()` — pass a fresh one (or `None`, the default), and treat
`fixed_target`/`target_mode`/`trend_sma_mult` on any config you hand to
`TargetOptimizer` as write-only from the caller's perspective.

**Skill reference:** pitfall #14 (this document adds the reproduction; the
skill entry states the fact).

---

## 4. Naming a custom indicator column outside the recognised convention (`column_naming`)

**What looks fine:** adding any custom feature to the KPI Table under a
descriptive name of your choosing (`my_custom_signal`, `overbought_flag`,
…) — it's a plain `float`/`bool` column, `build_features()`/Event Discovery
don't require registering columns anywhere.

**What actually happens:** Event Discovery's `FeatureGenerator` recognises
same-family pairing candidates (EMA-vs-EMA crossovers, price-vs-MA spreads,
…) only for columns whose name matches `{base}_{indicator}_{period}` with
`base` in `close/high/low/open/volume` and `period` a plain integer. A
column outside that convention is still used as a **standalone (arity-1)**
feature — it activates events on its own — but is invisible to every
**arity-2** (ratio/spread) construction, silently, with no log line naming
the skipped column.

**Measured, `examples/how_not_to_use_repro.py column_naming`** (same EMA-9
values under two names):

```
parse_feature('my_custom_signal') -> None
parse_feature('close_ema_9')      -> ParsedFeature(base='close', indicator='ema', params=[9], family='ema')

arity>=2 features built from 'my_custom_signal': 0
arity>=2 features built from 'close_ema_9' (identical values): 13
```

Identical numbers, identical information content — 13 derived ratio/spread
candidates from the compliant name, zero from the non-compliant one.

**Fix:** name custom indicator columns `{base}_{indicator}_{period}`
(`base` ∈ `close/high/low/open/volume`, `period` a bare integer) whenever
you want them eligible for pairing. If a custom feature never shows up
combined with anything downstream, check its name first, before suspecting
the feature itself lacks predictive value.

**Skill reference:** pitfall #4; `docs/manual-en.md` §21's "custom feature
column never shows up combined with anything" entry.

---

## 5. Monitoring a published edge by feeding Alpha Discovery only the new bars (`monitoring`)

**What looks fine:** you published a rule from an `EventCandidate` trained
on `train_df`; new candles have arrived (`new_bars_df`); to "check if the
edge still holds," you re-run `AlphaDiscovery(new_bars_df, candidates,
config)` on just the new window — it's the smallest, most direct-looking
change to the code that produced the contract in the first place.

**What actually happens — two compounding problems, both silent:**

1. **This is the wrong module.** `AlphaDiscovery` *re-derives* direction,
   horizon and take-profit from whatever data it's given; it does not
   replay a fixed target. `RuleDiscovery` is the tool for "does a published
   verdict still hold" (it replays the fixed target). This is the single
   most heavily documented anti-pattern in this codebase (skill pitfall #3,
   manual §21/§23) — restated here only because the next point compounds it.
2. **Even setting that aside, the frame itself is wrong.** Rolling-transform
   baselines (pctrank, z-score) inside an event's activation logic need the
   preceding history to mean what it meant during training. `new_bars_df`
   alone has none of that context — its own rolling windows reset from
   scratch — so events under-fire relative to their trained calibration.
   `AlphaDiscovery._event_series()` (`alpha_discovery/discovery.py:1611`)
   detects the index mismatch and emits a `UserWarning`, and a second, louder
   `UserWarning` when a candidate's activation count collapses to under 10%
   of its training count — but a `UserWarning` is easy to miss in production
   logging, and nothing stops the run.

**Measured, `examples/how_not_to_use_repro.py monitoring`** (ADA fixture,
75/25 train/monitor split, 3 797 event candidates re-evaluated both ways):

| | `AlphaDiscovery(new_bars_df` **alone**`)` | `AlphaDiscovery(pd.concat([train_df, new_bars_df]))` |
|---|---|---|
| `UserWarning`s raised | **357** | 1 |
| contracts with `direction='undetermined'` | **3 400 / 3 797 (89.5%)** | 3 240 / 3 797 (85.3%) |

357 index-mismatch/activation-collapse warnings on the wrong frame versus 1
on the concatenated frame — the mechanism is not hypothetical, it fires on
this repository's own fixture. 160 additional contracts (4.2 percentage
points) flip to `undetermined` purely from the frame-construction mistake,
*on top of* the more fundamental error of using `AlphaDiscovery` instead of
`RuleDiscovery` at all — the two errors are usually made together, and a
`"the edge decayed"` read of the result is doubly wrong.

**Fix:** use `RuleDiscovery`, not `AlphaDiscovery`, to check whether a
published rule still holds. If you ever do need to feed `AlphaDiscovery` an
extended window, pass `pd.concat([train_df, new_bars_df]).drop_duplicates(...)`,
never `new_bars_df` alone — and treat the `UserWarning` about "candles whose
index differs from the event's stored activation series" as an error in CI
for any code path meant to be extending, not replacing, the discovery
window.

**Skill reference:** pitfalls #3 and #11; `docs/manual-en.md` §21/§23 (the
manual's single most emphasized correctness rule).

---

## 6. Trusting `RuleRegistry.flat_table()`'s default output as "the clean list" (`registry_filter`)

**What looks fine:** after a `forge_multi()` cross-ticker session,
`reg.flat_table()` (or the module-level `flat_table(docs)`) reads like "the
registry's rules, as a table" — the natural call for "give me the rules I
can act on."

**What actually happens:** this is deliberate design, stated in
`rule_registry/export.py`'s own module docstring ("FORGE's philosophy is
*not to hide complexity*[…]by default the table keeps duplicates and
non-generic rules with explicit flags") — but it fails on **two**
independent axes at once if a caller doesn't know to look:
`RuleRegistry.flat_table()`'s own `apply_filters` keyword argument defaults
to `False` (so it doesn't even pass a `RegistryConfig` through to the export
step), **and** `RegistryConfig.export_duplicates`/`.export_non_generic`
themselves default to `True` — so even explicitly passing a bare
`RegistryConfig()` doesn't filter anything either. Nothing raises; the
table is simply larger and includes rows a downstream consumer likely
didn't want.

**Measured, `examples/how_not_to_use_repro.py registry_filter`** (a
registry of 3 documents: 1 clean/generic rule, 1 duplicate of it, 1
ticker-specific/non-generic rule):

```
flat_table(docs)                                        -> 3 rows
flat_table(docs, config=RegistryConfig())                -> 3 rows  (both defaults keep everything)
flat_table(docs, config=RegistryConfig(export_duplicates=False,
                                        export_non_generic=False))  -> 1 row
```

A caller who calls `reg.flat_table()` (or passes a bare `RegistryConfig()`)
expecting a deduped, generalises-across-tickers-only table silently gets 3x
too many rows here — in a real registry with thousands of candidate rules
and a nontrivial duplicate rate, this is a material double-count, not a
cosmetic one.

**Fix:** call `RuleRegistry.flat_table(apply_filters=True)`, or build a
`RegistryConfig(export_duplicates=False, export_non_generic=False)` and pass
it explicitly, or use `RuleRegistry.export(...)`/`.html_report(...)` (which
apply the filters for you) whenever you want "the rules that survived
deduplication and demonstrated cross-ticker generality," not "every
document the registry holds."

**Skill reference:** pitfall #16.

---

## A seventh pattern, already measured elsewhere in this repository

**Two identical `forge()` runs, same code, same data, promote a different
number of contracts — silently, no exception.** This isn't reproduced fresh
in `how_not_to_use_repro.py` (it requires comparing two separate *process*
invocations, not two calls inside one script, and a full-size search
surface to make the effect visible) but it is the same category of
misuse-without-exception as the six above, and it's already been measured
on this exact codebase: `feature_generator.py` iterates `set()`
intersections whose order depends on Python's per-process string hash
randomization (`PYTHONHASHSEED` unset by default), which propagates to
which candidate survives a pool-truncating cap — verified empirically at
850/851/855 promoted contracts across repeated runs on the same full-history
dataset (`docs/manual-en.md` §21). Treating an unpinned `forge()` result as
bit-for-bit reproducible across process restarts (for a CI regression test
or an audit trail) is the misuse; the fix is `PYTHONHASHSEED` pinned before
the interpreter starts (see the manual entry for the exact re-exec pattern).

---

## Summary table

| # | Pattern | Symptom if you don't know | Measured effect |
|---|---|---|---|
| 1 | Bypassing `forge()` on daily-or-slower data | Verdict looks tuned for the wrong session length | 24-bar vs 10-bar horizon grid top; 6-bar vs 1-bar fill delay |
| 2 | Trusting `GridSpec`'s auto-fan for `buy_delay_bar` | "Optimal" fill delay is whatever the seed happened to be | 1 distinct value across all 75 grid cells |
| 3 | Reusing an `AlphaConfig` across a `TargetOptimizer` call | Your own `fixed_target`/`target_mode` vanish | 3/3 fields silently overwritten in place |
| 4 | Non-conforming custom column names | Feature "never shows up combined with anything" | 0 vs 13 arity-2 features from identical values |
| 5 | Monitoring with `AlphaDiscovery(new_bars_df)` instead of `RuleDiscovery` | "The edge decayed" | 357 vs 1 warnings; +4.2pp `undetermined` |
| 6 | `flat_table()`'s default output | Registry looks bigger/messier than expected | 3 rows vs 1 row on a 3-document toy registry |
| 7 | Treating unpinned `forge()` as bit-for-bit reproducible | Two "identical" runs disagree | 850/851/855 promoted (measured previously, cited here) |

*This document was written by constructing each misuse against
`forgedge`'s actual source and this repository's reference fixture, running
it side by side with the corresponding correct usage, and recording the
measured difference — not by inference from reading the code alone. Every
number above is reproducible with
`python examples/how_not_to_use_repro.py <section>`.*
