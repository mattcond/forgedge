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

## `export_rules(results, output_dir, *, registries=None, config=PromotionGateConfig(), promotable_only=True) -> pd.DataFrame`

Writes one `.pkl` (event) + one `.yaml` (rule parameters) per exported
contract. The only function in this module — and in the library as a whole
outside of explicit report-writing helpers — whose purpose is a filesystem
side effect.

Runs the same computation as `promotion_gate()` internally (so the two never
disagree on what is promotable) and, for every selected contract, writes:

- **`{output_dir}/{alpha_id}.pkl`** — the `EventCandidate` via `pickle`, carrying its deterministic activation function (`EventCandidate.apply`) — no manual reconstruction needed to replay the event later.
- **`{output_dir}/{alpha_id}.yaml`** — `ValidatedRule.to_dict()` (the published operating point: direction, entry mode, buy/sell parameters, horizon, fee) plus `ticker`/`alpha_id`/`verdict` for context, written with a small dependency-free YAML writer (every value is a flat scalar, so no YAML library is needed).

**Parameters:**
- `results: Iterable[ForgeResult]` — R.
- `output_dir: str | Path` — directory to write into; created if missing.
- `registries: Iterable[RuleRegistry], optional` — forwarded to the underlying gate computation.
- `config: PromotionGateConfig` — forwarded to the underlying gate computation.
- `promotable_only: bool, default True` — export only contracts the gate marks `promotable`. Set `False` to export every tradeable contract regardless of the gate (the gate columns are still reported for audit).

A row whose candidate doesn't resolve or whose `response.validated_rule` is
`None` is silently skipped — no file written, no exception.

**Returns columns:** `ticker`, `alpha_id`, `event_candidate_id`, `verdict`,
`promotable`, `pkl_path`, `yaml_path` — one row per contract actually
exported.

```python
exported = export_rules(results, "exported_rules/", registries=[result_ada.registry, result_btc.registry])
len(exported)   # how many contracts were actually written to disk
```

**Verified**, same pool, default config (`promotable_only=True`):

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

## `monitoring_manifest(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format index of every tradeable rule, for a periodic re-check job.

Applies `RuleSpec.from_forge_result` (already provided by
`forgedge.rule_report` for a single run — see the manual's §9, pattern 5)
across all of R, so a monitoring job has one file listing every rule to
replay on fresh candles via `RuleDiscovery` — never `AlphaDiscovery` — instead
of reconstructing the reference to each rule by hand.

**Returns columns:** `ticker`, `rule_name`, `event_candidate_id`, `is_end`,
`verdict`, `oos_expectancy`. Join on `event_candidate_id` against
`export_rules`'s output to restrict to rules that were actually exported.

```python
manifest = monitoring_manifest(results)
manifest.merge(exported[["event_candidate_id"]], on="event_candidate_id")   # restrict to exported rules only
```

**Verified**, same pool:

```
mm.shape == (96, 6)
mm["verdict"].value_counts()
# PARTIAL-EDGE    96
```

Every tradeable rule on this fixture is `PARTIAL-EDGE` — the same fact
`lottery_only_winners` in `forgedge.playground` observes from the analysis
side (`src/forgedge/docs/specs/playground_en.md`).

---

## What's next

This module's three use cases (issue #245) are all implemented — there is no
open checklist here the way `forgedge.playground` has (or had) one. Future
production-deployment use cases, if any, would be tracked as new GitHub
issues against this module rather than reopening #245.
