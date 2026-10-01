# FORGE — Deployment: Putting Discovered Rules into Production

`forgedge.deployment` is the sibling of `forgedge.playground` that picks up
where the analysis toolkit leaves off: given the tradeable
(`EDGE`/`PARTIAL-EDGE`) contracts a `forge()` session produced, it decides
which ones are solid enough to go live, writes them to disk in a replayable
format, and indexes what was exported for a periodic monitoring job. Unlike
`forgedge.playground`, this module has **real effects** — `promotion_gate()`
makes a go/no-go decision, `export_rules()` writes files.

This is a usage guide: signatures, parameters, return columns, and verified
examples. For the design rationale (why the module was split out of
`forgedge.playground`, why the three functions run in a fixed sequence, why
only `export_rules` touches the filesystem) see
`src/forgedge/docs/modules/Deployment.md`.

**Naming history:** these three functions originally lived inside
`forgedge.playground` (issue #245). They were moved to their own top-level
module by PR #247 because they have real effects that a read-only
"playground" name no longer described honestly — see
`src/forgedge/docs/modules/Playground.md` §10 for the full story. No
behavior changed, only the import path:

```python
# before (issue #245, now stale)
from forgedge.playground import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest

# after (PR #247, current)
from forgedge.deployment import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest
```

---

## Basic usage

```python
from forgedge import forge
from forgedge.deployment import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest

result_ada = forge(kpi_ada, ticker="ADAUSDC", timeframe="1D")
result_btc = forge(kpi_btc, ticker="BTCUSDC", timeframe="1D")
results = [result_ada, result_btc]

# registries are optional — pass them to also gate on is_duplicate/is_isolated
gate = promotion_gate(results, registries=[result_ada.registry, result_btc.registry])
exported = export_rules(results, "exported_rules/", registries=[result_ada.registry, result_btc.registry])
manifest = monitoring_manifest(results)
```

The intended sequence is `forge() -> promotion_gate() [filter] -> export_rules()
[write, on the promotable rules] -> monitoring_manifest() [index the
export]` — `export_rules()` re-runs the same gate computation internally
(see the design doc), so it never disagrees with `promotion_gate()` about
what is promotable.

---

## `PromotionGateConfig`

Dataclass holding the promotion policy shared by `promotion_gate()` and
`export_rules()`. Every flag below is always computed and reported on every
row regardless of these settings — the `block_*`/`require_consistency`
fields only decide which flags feed into the final `promotable` column, so
turning a check off never loses visibility into what it would have flagged.

| Field | Default | Effect |
|---|---|---|
| `min_consistency` | `0.5` | Floor on `RuleDiscoveryResponse.walk_forward.consistency` (fraction of profitable OOS walk-forward folds) — the same floor the pipeline itself uses internally for a positive verdict. |
| `require_consistency` | `True` | Whether `min_consistency` participates in `promotable` at all. |
| `block_rotation_only` | `False` | Block a `PARTIAL-EDGE` whose only obstacle to a full `EDGE` was the search-level rotation null. Default `False` — a rotation-only miss is usually an acceptable trade-off, not a red flag. |
| `block_duplicate` | `True` | Block a rule the Rule Registry marked `is_duplicate=True`. |
| `block_isolated` | `True` | Block a rule classified `"ISOLATED"` on cross-ticker replay. No effect (`is_isolated` stays `None`) when no `registries` were supplied. |
| `min_fold_stability_score` | `None` | Floor on the fold-variance-penalized stability score (#253): `mean(fold_pf) - std(fold_pf)` over `RuleDiscoveryResponse.walk_forward.splits`' per-fold `test_summary.profit_factor` (each capped at `fold_pf_cap` first). Catches a rule whose pooled walk-forward PF looks strong only because one high-variance fold (often the `9999.0` "zero losing trades" sentinel) dominates the aggregate. `None` disables the gate; a rule with fewer than two walk-forward splits always passes it (sample std is undefined for one fold). |
| `fold_pf_cap` | `10.0` | Cap applied to each fold's `test_summary.profit_factor` before computing `fold_stability_score`, so a single sentinel-value fold can't dominate the mean/std. |

```python
config = PromotionGateConfig(block_rotation_only=True, min_consistency=0.6)
config = PromotionGateConfig(min_fold_stability_score=1.0)   # #253
```

---

## `promotion_gate(results, registries=None, config=PromotionGateConfig()) -> pd.DataFrame`

Long-format quality gate over every tradeable (`EDGE`/`PARTIAL-EDGE`)
contract.

Computes, per contract, the same flags the M3/M4 playground functions
expose individually (`lottery_only_winners`'s `rotation_only`,
`duplicate_clusters`'s `is_duplicate`, `classification_by_grade`'s
`"ISOLATED"` classification, walk-forward `consistency`, and the
fold-variance-penalized `fold_stability_score`, #253), then combines them
into `promotable` per `config`. Pure — no filesystem I/O.

**Parameters:**
- `results: Iterable[ForgeResult]` — R, one or more `forge()`/`forge_multi()` outputs.
- `registries: Iterable[RuleRegistry], optional` — sourced for `is_duplicate`/`classification` (see `modules/Deployment.md` for why this is separate from `results`, same reasoning as `forgedge.playground`'s M4 functions). `None` skips those two checks (columns stay `None`) rather than failing.
- `config: PromotionGateConfig` — which checks block promotion, and at what threshold.

**Returns columns:** `ticker`, `alpha_id`, `grade`, `verdict`,
`rotation_only`, `is_duplicate`, `is_isolated`, `consistency`,
`fold_stability_score`, `promotable`.

```python
gate = promotion_gate(results, registries=[result_ada.registry, result_btc.registry])
gate[gate["promotable"]].groupby("ticker").size()   # how many rules clear the gate, per ticker
```

**Verified**, pooling `forge_multi()` over ADAUSDC (the repository's
reference fixture) and a second, synthetic series labelled BTCUSDC — the
same pool used by every "Verified" example in `playground_en.md`'s M1/M3/M4
sections and below:

```
pg.shape == (96, 10)
pg["promotable"].value_counts()
# False    91
# True      5
pg.groupby("ticker")["promotable"].sum()
# ADAUSDC    0
# BTCUSDC    5
```

`min_fold_stability_score` is `None` by default, so it contributes nothing
to `promotable` here — the counts above are unchanged from before #253; the
new `fold_stability_score` column is populated for audit regardless.

Every `ADAUSDC` contract is blocked on this fixture — `duplicate_clusters`
(`forgedge.playground`) already showed 51% of pooled contracts are
duplicates and `classification_by_grade` showed most rules classify
`"ISOLATED"`, and the two default-`True` blocks (`block_duplicate`,
`block_isolated`) compound on a ticker where both are common. This is the
gate doing its conservative-by-default job, not a bug.

---

## `export_rules(results, output_dir, *, registries=None, config=PromotionGateConfig(), promotable_only=True, include_kpi_recipe=True, kpi_config=None, kpi_recipe_warmup_bars=0) -> pd.DataFrame`

Writes one `.pkl` (event) + one `.yaml` (rule parameters) per exported
contract. The only function in this module — and in the library as a whole
outside of explicit report-writing helpers — whose purpose is a filesystem
side effect.

Runs the same computation as `promotion_gate()` internally (so the two never
disagree on what is promotable) and, for every selected contract, writes:

- **`{output_dir}/{alpha_id}.pkl`** — the `EventCandidate` via `pickle`, carrying its deterministic activation function (`EventCandidate.apply`) — no manual reconstruction needed to replay the event later.
- **`{output_dir}/{alpha_id}.yaml`** — `ValidatedRule.to_dict()` (the published operating point: direction, entry mode, buy/sell parameters, horizon, fee) plus `ticker`/`alpha_id`/`verdict` for context, written with a small dependency-free YAML writer (every value is a flat scalar, so no YAML library is needed).
- **`{output_dir}/{alpha_id}.kpi_recipe.json`** (`include_kpi_recipe=True`, issue #296) — the minimal `kpi_builder` recipe recomputing only the KPI columns the event reads, plus a `"verification"` block (the round-trip check on `result.event_frame`). See [*Minimal KPI recipe*](#minimal-kpi-recipe-issue-296) below. The `.pkl` is always written too: it stays the faithful fallback for events the recipe can't rebuild.

**Parameters:**
- `results: Iterable[ForgeResult]` — R.
- `output_dir: str | Path` — directory to write into; created if missing.
- `registries: Iterable[RuleRegistry], optional` — forwarded to the underlying gate computation.
- `config: PromotionGateConfig` — forwarded to the underlying gate computation.
- `promotable_only: bool, default True` — export only contracts the gate marks `promotable`. Set `False` to export every tradeable contract regardless of the gate (the gate columns are still reported for audit).
- `include_kpi_recipe: bool, default True` — also write `{alpha_id}.kpi_recipe.json`.
- `kpi_config: dict | str | Path | None` — the `kpi_builder` config the KPI Table was built with (`None` = `DEFAULT_CONFIG`), forwarded to `minimal_kpi_recipe()`.
- `kpi_recipe_warmup_bars: int, default 0` — bars excluded from the start of each `result.event_frame` when verifying the recipe. Leave `0` when the KPI Table was built with `kpi_builder` from the very candles given to `forge()`; set it when the table was built on a longer history and truncated (see *Warm-up* below). No heuristic default.

A row whose candidate doesn't resolve or whose `response.validated_rule` is
`None` is silently skipped — no file written, no exception.

**Returns columns:** `ticker`, `alpha_id`, `event_candidate_id`, `verdict`,
`promotable`, `pkl_path`, `yaml_path`, `kpi_recipe_path`,
`kpi_recipe_verified` — one row per contract actually exported.
`kpi_recipe_verified` is `None` when no recipe was written or when the
result carries no `event_frame` to verify against.

```python
exported = export_rules(results, "exported_rules/", registries=[result_ada.registry, result_btc.registry])
len(exported)   # how many contracts were actually written to disk
```

**Verified**, same pool, default config (`promotable_only=True`) — measured
before #296 added `kpi_recipe_path`/`kpi_recipe_verified` (and one
`.kpi_recipe.json` per contract), so today the frame is `(5, 9)` and 15
files are written:

```
exp.shape == (5, 7)
# 10 files written (5 .pkl + 5 .yaml)
exp.iloc[0][["ticker", "alpha_id", "verdict", "promotable"]]
# ticker       BTCUSDC
# alpha_id     ALPHA-BTCUSDC-1D-260830-1196
# verdict      PARTIAL-EDGE
# promotable       True
```

All 5 exported contracts are `BTCUSDC` — consistent with `promotion_gate`
above finding zero promotable `ADAUSDC` contracts on this fixture.

---

## `monitoring_manifest(results: Iterable[ForgeResult], exported: pd.DataFrame | None = None) -> pd.DataFrame`

Long-format index of every tradeable rule, for a periodic re-check job.

Applies `RuleSpec.from_forge_result` (already provided by
`forgedge.rule_report` for a single run — see the manual's §9, pattern 5)
across all of R, so a monitoring job has one file listing every rule to
replay on fresh candles via `RuleDiscovery` — never `AlphaDiscovery` — instead
of reconstructing the reference to each rule by hand.

**Returns columns:** `ticker`, `rule_name`, `event_candidate_id`, `is_end`,
`verdict`, `oos_expectancy`, `kpi_recipe_path`, `kpi_recipe_verified`. Join
on `event_candidate_id` against `export_rules`'s output to restrict to rules
that were actually exported.

Pass `exported=` (the frame `export_rules` returned) to fill
`kpi_recipe_path`/`kpi_recipe_verified` (joined on `(ticker,
event_candidate_id)`): the re-check job then knows which rules it can keep
alive from the minimal recipe alone. Without it the two columns are `None`;
rows are never filtered by it.

```python
manifest = monitoring_manifest(results, exported=exported)
manifest.merge(exported[["event_candidate_id"]], on="event_candidate_id")   # restrict to exported rules only
```

**Verified**, same pool:

```
mm.shape == (96, 6)   # measured before #296; now (96, 8)
mm["verdict"].value_counts()
# PARTIAL-EDGE    96
```

Every tradeable rule on this fixture is `PARTIAL-EDGE` — the same fact
`lottery_only_winners` in `forgedge.playground` observes from the analysis
side (`src/forgedge/docs/specs/playground_en.md`).

---

## Minimal KPI recipe (issue #296)

```python
from forgedge.deployment import KpiRecipe, minimal_kpi_recipe, verify_kpi_recipe

recipe = minimal_kpi_recipe(candidate)                 # kpi_config=None -> DEFAULT_CONFIG
recipe.to_dict()
# {'build_features_config': {'volatility': {'enabled': True,
#                                           'params': {'periods': [12, 24], 'columns': ['close']}}},
#  'lag_features': [], 'candle_features': False, 'pattern_features': False,
#  'color': False, 'unresolved_columns': [], 'base_columns': [...]}

reduced = recipe.rebuild(candles)                      # only the columns this rule needs
check = verify_kpi_recipe(candidate, recipe, result.event_frame)
check.matches, check.n_mismatched_bars, check.last_mismatch_at

KpiRecipe.from_json(open("exported_rules/ALPHA-....kpi_recipe.json").read())  # back from disk
```

The pickled `EventCandidate` reproduces an event exactly but is fragile
across library versions, opaque (it never says which KPI columns a
monitoring job must compute) and unreadable outside Python. A `KpiRecipe`
is the human-readable complement: the smallest `kpi_builder` recipe —
`build_features` config, `candle_features`/`pattern_features`/`color`
flags, ordered `lag_features` requests — recomputing exactly the native
columns the event's components read (`source_cols`, or `source_feature` for
arity-1 components), from raw OHLC(V) candles.

- **Why here, not on `EventCandidate`.** `event_discovery` deliberately does
  not depend on `kpi_builder`; resolving a recipe has to call
  `build_features()`. `deployment` is the downstream layer that may depend
  on both.
- **Resolution by construction.** Every indicator/column/period of
  `kpi_config` is built once on a tiny synthetic frame and the produced
  column names are indexed back to their entry — cached per config, so a
  whole export costs a fixed number of probes (~130 for `DEFAULT_CONFIG`,
  ~0.2 s) and each candidate only dictionary lookups. Every entry is probed
  regardless of `enabled`, so opt-in indicators (ATR, MACD, Aroon, …)
  resolve too; Aroon's prefix-less `aroon_up_NN` needs no special case.
- **Positional periods.** MACD's `periods` is a list of `(fast, slow,
  signal)` triples: the recipe keeps each needed triple intact and in order
  (sorting `[12, 26, 9]` into `[9, 12, 26]` builds a different MACD — a bug
  reproduced in the #295 prototype). Any future positional indicator must
  be added to `kpi_recipe._POSITIONAL_PERIOD_GROUPS`.
- **Superset per indicator.** `build_features` takes `periods × columns` per
  indicator, so an event needing `close_ema_12` and `low_ema_03` also gets
  `close_ema_03`/`low_ema_12`. Never a missing column.
- **Out of scope → `unresolved_columns`.** `CustomEvent` formulas (listed
  verbatim), M0's `regime` columns, non-standard or user-supplied columns,
  and lag spellings `lag_features()` never writes (`close_prev_2`).
  `rebuild()` refuses an incomplete recipe (`ValueError`);
  `verify_kpi_recipe()` reports it (`matches=False`, `error=...`).

**Warm-up.** `verify_kpi_recipe()` compares the event's activations bar by
bar (by timestamp, NaN read as inactive) on the reference table vs on the
rebuilt one. When the reference KPI Table was built by `kpi_builder` from
the same candles, the round-trip is exact: **23 488 / 23 488** candidates
of an Event Discovery run on the ADA 1D fixture's OHLC rebuilt with
`DEFAULT_CONFIG` + MACD/ATR/Stochastic/Aroon + `candle_features` + lags and
a lag-of-lag. When the reference was built on a longer history and then
truncated — like `tests/fixtures/ADA_1D_TRAIN.parquet` itself — recursive
and rolling indicators rebuilt from the shorter span differ during a
transient (EMA-12: first 86 of 882 bars; propagated through a 168-bar
pctrank window, up to ~260 bars), and the un-rounded `close_bb_mid_*`/
`close_bb_width_*` columns can differ by floating-point noise anywhere
(rolling sums are path-dependent), which very rarely flips a comparison
sitting exactly on a threshold. Restrict the comparison with
`evaluation_mask=` (or `export_rules(kpi_recipe_warmup_bars=...)`) and read
`n_mismatched_bars`/`first_mismatch_at`/`last_mismatch_at` — no heuristic
default window is applied in this first version.

Still open (issue #296, phase 3): whether `monitoring_manifest()` should
aggregate the recipes into one file indexed by `alpha_id` instead of one
JSON per rule once real export sizes are measured.

---

## What's next

This module's three use cases (issue #245) are all implemented — there is no
open checklist here the way `forgedge.playground` has (or had) one. Future
production-deployment use cases, if any, would be tracked as new GitHub
issues against this module rather than reopening #245.
