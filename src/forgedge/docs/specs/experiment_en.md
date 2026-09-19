# FORGE — Experiment: Pipelines of Experiments Built on `forge()`

`forgedge.experiment` is a third sibling alongside `forgedge.playground`
(read-only analysis) and `forgedge.deployment` (production) — it **runs its
own multi-stage pipeline on top of `forge()`**, instead of only reading or
acting on a `ForgeResult` that already exists. `StepWiseDiscovery`, its
first class, grows hold-out-confirmed single-condition rules one
AND-condition at a time by searching each rule's own active sub-population
for a second dimension — a per-seed local search, distinct from `forge()`'s
own single global, grade-guided two-pass composition.

This is a usage guide: signature, parameters, return fields, and verified
examples. For the design rationale (why config resolution happens exactly
once, why redundancy is checked both within a chain and across seeds, the
research history that motivated every one of its fixes) see
`src/forgedge/docs/modules/Experiment.md`.

```python
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig
```

---

## Basic usage

```python
import pandas as pd
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig

kpi = pd.read_parquet("kpi_table.parquet")   # 'close' + a timestamp column, like forge()

engine = StepWiseDiscovery(
    kpi, ticker="BTCUSDC", timeframe="1D",
    config=StepWiseDiscoveryConfig(n_seeds=3, depth=2),
)
result = engine.run()

print(result.summary())          # one row per chain: expression, depth_reached, verdict, ...
for chain in result.edges():     # chains whose final (last-confirmed) state is EDGE/PARTIAL-EDGE
    print(chain.chain_label, chain.expression, chain.verdict)
```

Any asset-specific calibration (`AlphaConfig.fee_per_side`, `mfe_floor`,
gate thresholds, ...) is the caller's responsibility, passed already
resolved via `event_discovery_config`/`alpha_config`/`rule_discovery_config`
— `StepWiseDiscovery` treats them as given and never second-guesses them.
Leaving all three `None` (the default) resolves the same way an unconfigured
`forge()` call would.

---

## `StepWiseDiscovery(kpi, *, ticker=None, asset="ASSET", timeframe, event_discovery_config=None, alpha_config=None, rule_discovery_config=None, config=None, run_market_context=True)`

**Parameters:**
- `kpi: pd.DataFrame` — the full KPI Table. Never mutated.
- `ticker: str, optional` — forwarded to `forge()` and to every direct `AlphaDiscovery`/`RuleDiscovery` call this class makes, so IDs stay consistent across iteration 1 and every chain.
- `asset: str` — traceability fallback, same semantics as `forge()`'s own `asset` parameter (used only when neither `ticker` nor an explicit `alpha_config` sets one).
- `timeframe: str` — bar size, e.g. `"1D"`, `"1H"`. Required keyword.
- `event_discovery_config: DiscoveryConfig, optional` — `max_and_components` is forced to `1` regardless of what's passed; composition is this class's own job.
- `alpha_config: AlphaConfig, optional`
- `rule_discovery_config: RuleDiscoveryConfig, optional`
- `config: StepWiseDiscoveryConfig, optional` — search-algorithm parameters (see field table below). Defaults to `StepWiseDiscoveryConfig()`.
- `run_market_context: bool, default True` — forwarded to the internal `forge()` call for iteration 1.

**`.run() -> StepWiseDiscoveryResult`** — executes the full algorithm (module docstring / `modules/Experiment.md` §3) and returns the result. Raises `ValueError` before running anything if `config.strict` (default `True`) and the resolved configuration has a `FAIL`-level coherence finding — the same fail-fast contract `forge(strict=True)` has.

**`.log_lines: list[str]`** — every diagnostic line logged during `.run()` (TimeBudget summary, seed acceptance/rejection with reasons, per-depth candidate counts, composed-rule attempts), in order. Useful to print alongside `result.summary()` the same way the manual recommends logging `result.ledger.describe()`/`result.resolution.describe()` next to a plain `forge()` run.

---

## `StepWiseDiscoveryConfig`

None of these fields are asset- or timeframe-specific — that calibration belongs on the configs passed to the constructor instead.

| Field | Default | Effect |
|---|---|---|
| `n_seeds` | `3` | Number of distinct seeds to start an independent chain from. |
| `depth` | `2` | Maximum AND-composition depth per chain. |
| `train_ratio` | `0.80` | Fraction of the KPI table reserved for the OUTER hold-out split — independent of, and in addition to, any internal split the passed-in configs use. |
| `outer_horizon_bars` / `outer_embargo_bars` | `None` / `None` | Purge/embargo width for the outer split. `None` derives them from the RESOLVED `alpha_config.horizon_grid`/`embargo_bars` — the same quantity `forge()` itself would use; an explicit value always takes priority (e.g. an OU-half-life-derived scale, if you have one). |
| `retain_ratio_floor` | `0.2` | Absolute floor on `retain_ratio_min_k = max(floor, min_trades_M3 / n_rows)`, recomputed at every depth from the CURRENT partition's actual size — an adaptive floor tied to M3's own statistical requirement (`rule_discovery_config.criteria.min_oos_trades`), not a fixed row count. |
| `max_constituent_jaccard` | `0.85` | Same default as `forgedge.event_discovery.diversity_gate`/`ANDComposer(max_constituent_jaccard=...)` — see `forgedge.experiment.redundancy`. |
| `max_constituent_abs_corr` | `0.95` | Deliberately stricter than the Jaccard threshold — must only catch near-duplicate information, not merely-correlated-but-distinct signals. |
| `child_gate` | `None` | Consistency Gate for children evaluated on a partition. `None` uses `GateParams(min_tpm=0.25, dispersion_margin=3.0, min_episodes=4, event_counting="episode", episode_gap=1)` — looser than a typical session gate (a sub-population search needs it), but not so loose that a rare child intersected with an already-rare parent starves the composed rule of activations before Alpha Discovery can derive a target. |
| `min_composed_activations` | `20` | A parent-AND-child candidate below this many activations on the full search frame is rejected before even trying Alpha Discovery — fails fast with a clear reason instead of an opaque M2 "no derivable target". |
| `min_local_partition_rows` | `10` | A rolling-transform child whose local (partition-restricted) series has fewer non-NaN rows than this is skipped. |
| `strict` | `True` | Forwarded to `forge()`/`config_report()` — raise on a `FAIL`-level incoherence rather than proceeding. |

```python
config = StepWiseDiscoveryConfig(n_seeds=5, depth=3, max_constituent_abs_corr=0.90)
```

---

## `StepWiseDiscoveryResult`

- **`chains: list[ChainResult]`** — one per seed, in seed-ranking order.
- **`seed_attempts: list[SeedAttempt]`** — every candidate examined during seed selection, in ranked order, whether or not it became a seed (`family`, `alpha_id`, `verdict`, `composite_score`, `accepted`, `reason`, `jaccard_vs_seeds`, `abs_corr_vs_seeds`) — lets a caller audit *why* a high-ranked candidate was skipped (hold-out disconfirmation vs. cross-seed redundancy).
- **`feature_recurrence: dict[str, list[str]]`** — observation only, not a gate: which feature family recurred across which chains/depths during local child search.
- **`search_df`, `holdout_df`: pd.DataFrame** — the outer split actually used. `holdout_df` was never touched by any discovery/alpha/rule-discovery call in this run.
- **`time_budget: TimeBudget`** — the outer split's budget.
- **`coherence: ConfigReport`** — the resolved configuration this run executed with, same shape as `ForgeResult.coherence`.
- **`config: StepWiseDiscoveryConfig`** — the algorithm parameters this run used.
- **`.summary() -> pd.DataFrame`** — one row per chain: `chain`, `seed_family`, `expression`, `depth_reached`, `composite_score`, `verdict`, `confirm_reason`, `stop_reason`.
- **`.edges() -> list[ChainResult]`** — chains whose final (last-confirmed) state is `EDGE`/`PARTIAL-EDGE`.

## `ChainResult`

`chain_label`, `seed_family`, `candidate: EventCandidate`, `contract: AlphaContract`, `response: RuleDiscoveryResponse`, `depth_reached: int`, `confirm_reason: str`, `stop_reason: str`, plus convenience properties `.expression`, `.composite_score`, `.verdict`. `candidate`/`contract`/`response` describe the LAST CONFIRMED state of the chain — the seed alone if no composition ever confirmed on the hold-out, or the deepest composed rule that did; a composition attempt that fails hold-out confirmation is never written here (see `modules/Experiment.md` §2, principle 5).

---

## `forgedge.experiment.redundancy` — the diversity checks

Pure functions, no pipeline dependency — usable and testable standalone:

- **`family_key(candidate) -> str`** — groups a candidate by feature family, stripping numeric periods from `source_feature` (name-based, coarser than the two checks below).
- **`max_jaccard(bool_series, used_series_list) -> float`** — `J(A, B) = |A∩B| / |A∪B|` against every series in the list; same definition as `forgedge.event_discovery.diversity_gate`.
- **`max_abs_corr(components, used_cols, series_by_col) -> float`** — max `|Pearson correlation|` of each component's CONTINUOUS pre-threshold series (looked up in `series_by_col`, keyed by `transformed_col`) against every column in `used_cols`. Closes a gap Jaccard alone leaves: two components whose continuous signal is near-identical (an ATR-ratio and its NATR analogue) can still have divergent boolean activations purely from where each threshold landed.
- **`is_redundant(*, bool_series, used_bool_series, components, used_cols, series_by_col, max_jaccard_threshold=0.85, max_abs_corr_threshold=0.95) -> (bool, float, float)`** — combines both checks; returns the two scores even when neither trips the threshold, for logging.

---

## Verified example

On this repo's own `tests/fixtures/ADA_1D_TRAIN.parquet` (882 bars), default
`DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig`,
`StepWiseDiscoveryConfig(n_seeds=2, depth=2)`:

```python
import pandas as pd
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig

kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
engine = StepWiseDiscovery(kpi, ticker="ADAUSDC", timeframe="1D",
                            config=StepWiseDiscoveryConfig(n_seeds=2, depth=2))
result = engine.run()
print(result.summary()[["chain", "expression", "depth_reached", "verdict"]].to_string(index=False))
```

```
                                        chain                                                            expression  depth_reached  verdict
seed1:diffnorm_close_vol_vol|rolling_pctrank                          pr_diffnorm_close_vol12_vol24_168 < 0.172619              0  PARTIAL-EDGE
          seed2:ratio_close_ema_ema|identity  (ratio_close_ema03_ema12 < 0.952059) AND (zs_close_ema_03_96 < -1.5)              1  NON-EDGE
```

`seed2` reaches depth 1 — a genuinely confirmed two-condition composed rule
(search-level verdict `NON-EDGE`, but a positive mean expectancy across
post-holdout walk-forward folds on the concatenated search+holdout
evaluation). `seed1` stays single-condition: the one composition attempted
at depth 0 failed hold-out confirmation, so the chain correctly reports the
last confirmed state (the seed) rather than the failed attempt. Neither
chain reaches depth 2 on this fixture.

A second, runnable example against this repo's own bundled OHLCV data
(EURUSD 1D, with the fee-per-side diagnostic a non-crypto asset needs before
trusting a verdict): `examples/step_wise_discovery_usage.py`.

---

## What's next

`StepWiseDiscovery` is the first class in this module — future experiment
pipelines built on `forge()` (e.g. a cross-ticker or cross-timeframe search
orchestrator) would be added as new classes here, not by growing this one's
scope.
