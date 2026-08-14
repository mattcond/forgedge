---
name: forgedge
description: Use whenever working with the forgedge / FORGE (Feature-Oriented Rule Generation Engine) codebase or Python library — writing or debugging code that imports `forgedge`, calling `forge()` / `forge_multi()` / `forge_preset()`, building or validating a KPI Table, working with Event Discovery, Alpha Discovery, Rule Discovery or the Rule Registry, interpreting EDGE / PARTIAL-EDGE / NON-EDGE verdicts, replaying an `EventCandidate` or monitoring a published rule on fresh candles, or fixing/extending forgedge's own source and tests. Trigger this any time forgedge, FORGE, "KPI table", `EventCandidate`, `AlphaContract`, `RuleDiscoveryResponse`, walk-forward OOS, or look-ahead bias comes up in the context of this repository — even if the user does not name the skill explicitly or only pastes an error/traceback from the library.
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
operating summary distilled from that manual, not a replacement for it.

## The pipeline: five modules, one direction

```
KPI Table (OHLCV + indicators)
  │
  ▼ M0  Market Context      classify each bar's regime            → + 'regime' / 'regime_stable'
  ▼ M1  Event Discovery     mine boolean events, NO forward return → list[EventCandidate]
  ▼ M2  Alpha Discovery     derive target, measure predictive power → list[AlphaContract]
  ▼ M3  Rule Discovery      realistic backtest, walk-forward OOS    → EDGE/PARTIAL-EDGE/NON-EDGE
  ▼ M4  Rule Registry       dedup, cross-ticker, genericity          → flat table + HTML report
```

| # | Module | Answers | Key classes | Status |
|---|---|---|---|---|
| 0 | Market Context | Which regime is this bar in? | `MarketContext`, `MarketContextConfig`, `EMAProxyConfig` | done |
| 1 | Event Discovery | Is this indicator pattern stable and repeatable? | `EventDiscovery`, `DiscoveryConfig`, `EventCandidate`, `CustomEvent` | done |
| 2 | Alpha Discovery | Does the event predict an oriented return? | `AlphaDiscovery`, `AlphaConfig`, `AlphaContract` | done |
| 3 | Rule Discovery | Is it profitable under real order mechanics? | `RuleDiscovery`, `RuleDiscoveryConfig`, `RuleDiscoveryResponse` | done |
| 4 | Rule Registry | Does it generalise across tickers? | `RuleRegistry`, `RegistryConfig`, `RuleDocument` | WIP |

Data only ever flows forward. Each module consumes exactly the formal
artefact the previous one produced and passes a new one on — it never reaches
back into an earlier module's internals.

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

On a daily-or-slower `timeframe`, `forge()` automatically substitutes a
daily-calibrated `horizon_grid` — see pitfall #2 before building `AlphaConfig`
by hand. `forge()` also runs the default fast rotation null (`fast_null=True`)
that prices the search's multiple-testing surface; a contract that only wins
that lottery is capped at `PARTIAL-EDGE` even if every other gate passes.

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
momentum/regime-change events). `forgedge.presets.preset_info()` prints the
resolved numeric parameters for any/all presets.

### 3. Manual step-by-step (for drill-down or contributing)

```python
from forgedge import (
    MarketContext, MarketContextConfig, EMAProxyConfig,
    EventDiscovery, DiscoveryConfig,
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
resp = RuleDiscovery(eval_df, contract, cand).run()
print(resp.verdict, resp.walk_forward.consistency)

# Or, for every tradeable rule of a forge() run at once, with an HTML report:
specs = RuleSpec.from_forge_result(result)
html = rule_performance_report(result, fresh_candles)
```

## Common pitfalls

1. **Passing the wrong frame to `AlphaDiscovery`.** When building the pipeline
   by hand, pass `ed.df` (Event Discovery's post-pipeline frame, with derived
   features already attached), not the original KPI Table.
2. **Silent hourly `horizon_grid` on daily-or-slower data.** `AlphaConfig`'s
   class default is calibrated on ~hourly bars. `forge()` substitutes a
   daily-calibrated grid automatically and warns if an explicit `AlphaConfig`
   keeps the untouched hourly default on a slow `timeframe`; building
   `AlphaDiscovery` directly bypasses that substitution, so pass
   `horizon_grid` explicitly (see `forgedge.presets.default_horizon_grid` or
   just use `forge_preset`) or you'll scan 48+ *day* holding periods by
   accident.
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
7. **Copying `GateParams(...)` from this repo's own `examples/*.py` scripts.**
   Several scripts (`alpha_discovery_usage.py`, `extended_usage.py`,
   `kpi_table_1d.py`, `search_rotation_calibration.py`,
   `lowfreq_null_diagnostic.py`, `lowfreq_endpoint_diagnostic.py`) predate a
   `GateParams` API change and still pass `min_act`/`min_months`/`max_conc`,
   which now raise `TypeError`. Translate to the current fields (`min_tpm`,
   `max_dispersion`, `event_counting`, `min_episodes`, `episode_gap`) —
   `event_counting="bar"` reproduces the old counting semantics most closely.
   `examples/kpi_builder_usage.py` is unaffected.
8. **Calling an accessor before `.run()`.** `MarketContext.distribution()`,
   `EventDiscovery.summary()`, `AlphaDiscovery.summary()`/
   `.promoted_contracts()`, `RuleDiscovery.grid_summary()`, and
   `TargetOptimizer.validate_oos()`/`.discover_alpha()` all raise
   `RuntimeError: Call run() before ...` until the corresponding `.run()` has
   executed — a deliberate guard, not a bug to route around.
9. **Extending a discovery window with only the new bars.** Re-evaluating an
   `EventCandidate` on a frame whose index differs from its cached training
   activation series triggers a `UserWarning` and a fallback to `.apply()`;
   if the re-evaluated activation count collapses under ~10% of the training
   count, that's usually why direction reads `"undetermined"` — rolling
   baselines (pctrank, z-score) lost the history they need, not because the
   edge disappeared. Fix: `pd.concat([train_df, new_bars])`, never `new_bars`
   alone.

## Best practices

- Run `summary_report()` before every discovery session and decide
  explicitly what to do with `has_critical`/`has_warnings` — the library
  never validates or blocks on its own (see pattern 4).
- Prefer `forge_preset()` over hand-assembling `DiscoveryConfig`/
  `AlphaConfig`/`RuleDiscoveryConfig` — the presets keep M1/M2/M3 frequency
  criteria mutually consistent, which is easy to get subtly wrong by hand.
- Treat a run that comes back mostly or entirely `PARTIAL-EDGE` as
  informative, not broken. Check `rejection_reasons` (often
  `"search-level rotation null not cleared"`) before assuming
  misconfiguration — an earlier, more permissive version of this pipeline
  promoted noise almost as often as real signal on low-frequency data
  (`docs/analysis/lowfreq_robustness.md`), which is exactly what the default
  rotation null (see pattern 1) now guards against.
- Log `result.ledger.describe()` and `result.calibration.summary()` (or
  `.tippett_p`) alongside every run you persist — cheap, already computed,
  and exactly what you'll want later to explain why a verdict landed where
  it did.

## Contributing to forgedge itself

Source lives under `src/forgedge/`, one directory per module plus a few
top-level orchestration files:

```
src/forgedge/
├── forge.py              orchestrator: forge(), forge_multi(), ForgeResult
├── presets.py             forge_preset(), preset_info(), default_horizon_grid()
├── market_context/        M0
├── event_discovery/       M1
├── alpha_discovery/       M2
├── rule_discovery/        M3
├── rule_registry/         M4
├── calibration/           FastRotationNull, RotationCalibrator
├── kpi_builder/            build_features / candle_features / lag_features / pattern_features
├── ledger.py               HypothesisLedger
├── timebudget.py           TimeBudget (purged/embargoed IS/OOS split)
├── target_optimizer.py    TargetOptimizer (target-first alternative workflow)
├── rule_report.py          RuleSpec, rule_performance_report
├── summary_report.py       data-quality diagnostics
└── docs/                   packaged module + spec docs (see below)
```

Each module directory typically has `models.py` (dataclasses/config),
`discovery.py`/equivalent orchestration file, and focused helper modules —
follow the pattern already in the module you're touching rather than
introducing a new one.

Every public name lives in `src/forgedge/__init__.py`'s `__all__` — add new
public API there, and keep dataclass field docstrings in numpydoc style
(`Attributes` sections), matching the existing modules.

**Tests**: `tests/` — 586 test functions across 15 files, run with `pytest`
(`testpaths = ["tests"]` in `pyproject.toml`, dev extra `pytest>=7.0`). One
`test_<module>.py` per module plus `tests/test_golden.py` (end-to-end
regression, pinned via a session-scoped `forge()` fixture over
`tests/fixtures/ADA_1D_TRAIN.parquet`) and `tests/test_forge.py`
(orchestrator wiring). House style worth matching: no mocking except two
`monkeypatch` "must-not-be-called" tripwires in `test_alpha_discovery.py` —
everything else runs the real pipeline against seeded synthetic data
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
current with the code:

| Topic | File |
|---|---|
| Comprehensive practical manual — installation through production architecture, troubleshooting, best practices/anti-patterns, FAQ, glossary; every example verified against this repo | `docs/manual-en.md` (`docs/manuale-it.md` for Italian) |
| Project overview, pipeline diagram, quick start | `README.md` (`README_it.md` for Italian) |
| Deep architectural guide — artefact YAML formats, principles, roadmap | `src/forgedge/docs/README.md` (Italian) |
| Concepts — event / alpha / rule, from first principles | `src/forgedge/docs/specs/concepts_en.md` (`_it.md`) |
| End-to-end production guide — config per module, checklists | `src/forgedge/docs/specs/how_to_use_en.md` (`_it.md`) |
| Global configuration reference | `src/forgedge/docs/specs/configuration_en.md` (`_it.md`) |
| Per-module spec | `src/forgedge/docs/specs/modulo_{0..4}_en.md` (`_it.md`) |
| Technical analyses (low-freq robustness, rotation-null calibration) | `docs/analysis/*.md` |
| Runnable examples per module (several predate a `GateParams` API change — see pitfall #7 before copying `GateParams(...)` from one) | `examples/*.py` |
| Interactive walkthroughs | `notebooks/0{1..6}_*.ipynb`, `notebooks/hurst.ipynb` |
| Full public API + config dataclass fields/defaults | `references/api-reference.md` (in this skill) |
