# FORGE — Playground: Analysis Helpers over ForgeResult

`forgedge.playground` is a small, read-only layer of functions that take
**R** — one or more `ForgeResult` objects, the output of `forge()`/`forge_multi()`
pooled from a whole research session — and turn them into long-format
`pandas.DataFrame`s answering cross-cutting questions about the *pipeline's
own behaviour*: how nervous a regime boundary is, why Rule Discovery discards
grade-A contracts, which feature families Alpha Discovery can never orient.
No function here ever calls `forge()`, `RuleDiscovery`, or any other M0–M4
component — every function only reads attributes already sitting on the
`ForgeResult` objects you pass it.

This is a usage guide: signatures, parameters, return columns, and verified
examples. For the design rationale (why long-format, why `Iterable[ForgeResult]`,
the internal algorithm behind each function) see `src/forgedge/docs/modules/Playground.md`.

**This module is explicitly a diagnostic layer, not a stable core API on the
level of `forge()` or `RuleDiscovery`.** Its tracking checklist (GitHub issue
#237) of 11 planned use cases is now complete — all 10 functions below plus
the cross-cutting `conversion_funnel` are implemented — but signatures can
still refine as new real-world use cases surface (see the family-bucketing
bug history under `undetermined_direction_by_family`). Check the installed
version's docstrings if a signature below looks out of date.

---

## Basic usage

```python
from forgedge import forge
from forgedge.playground import *   # the intended import — __all__ is explicit

result_ada = forge(kpi_ada, ticker="ADAUSDC", timeframe="1D")
result_btc = forge(kpi_btc, ticker="BTCUSDC", timeframe="1D")

results = [result_ada, result_btc]   # pool as many ForgeResult as you have — "R"

regime_transitions(results)
regime_time_share(results)
discard_reasons_by_grade(results, grade="A")
undetermined_direction_by_family(results)
```

Every function takes the same `results` list — build it once per session
(across tickers, across re-runs over time) and hand it to whichever function
answers the question you have.

---

## `regime_transitions(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format log of **every** regime flip observed, with the run length that
preceded it.

Reads `result.enriched` (the KPI Table after Market Context); silently
**skips** any result whose `enriched` has no `regime` column (Market Context
disabled) instead of raising.

**Returns columns:** `ticker`, `bar_index` (integer row position of the flip),
`timestamp`, `from_regime`, `to_regime`, `run_length_before` (consecutive bars
in the source regime, including the last bar before the flip).

**Timestamp resolution**, in order of preference: the frame's `DatetimeIndex`
if present, else its `open_dt` column, else the bar's integer position itself
— never an error, so this also works on hand-built frames in a test or a
notebook.

A constant-regime series returns an empty `DataFrame` (with the correct
columns, never `None`). A `NaN` regime right after the start of the series
never counts as a flip *from* `NaN`.

```python
df = regime_transitions(results)
df[df["run_length_before"] <= 2].groupby("ticker").size()   # rank tickers by boundary nervousness
```

**Verified**, pooling a real `forge()` run on `ADAUSDC` (the repository's
reference fixture) with a second, synthetic series labelled `BTCUSDC`:

```
df.shape == (236, 6)
df[df["run_length_before"] <= 2].groupby("ticker").size()
# ADAUSDC    45
# BTCUSDC    41
```

---

## `regime_time_share(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format share of bars each ticker spends in each regime — over the
*classified* (non-`NaN`) bars of that result only.

Same skip rule as `regime_transitions`: any result without a `regime` column
is silently skipped.

**Returns columns:** `ticker`, `regime`, `n_bars` (absolute count), `share`
(in `[0, 1]`).

```python
df = regime_time_share(results)
df.sort_values("share", ascending=False).groupby("ticker").head(1)   # dominant regime per ticker
```

**Verified**, same two-ticker pool:

```
top = share.sort_values("share", ascending=False).groupby("ticker").head(1)
top[["ticker", "regime", "share"]]
#  ticker      regime    share
# BTCUSDC STRONG_BEAR 0.438776
# ADAUSDC STRONG_BEAR 0.407029
```

Both tickers in this example spend over 40% of their history in `STRONG_BEAR`
— exactly the kind of signal this function exists to surface: any rule
discovered on either of these two assets deserves an explicit check for how
much it was conditioned on a single regime.

---

## `discard_reasons_by_grade(results: Iterable[ForgeResult], grade: str = "A") -> pd.DataFrame`

Long-format breakdown of *why* Rule Discovery verdicts `NON-EDGE` on alpha
contracts of a given letter grade.

Reads `result.rule_responses` (every promoted contract paired with its Rule
Discovery verdict — **not** `result.edges()`, which by construction never
contains `NON-EDGE`). Keeps contracts whose grade matches (case-insensitive)
and whose `response.verdict == "NON-EDGE"`, then explodes `rejection_reasons`
one row per reason. A contract with an empty `rejection_reasons` list still
emits one row, with `reason=None` — it is never silently dropped. A contract
with no `alpha_score` (never graded) never matches any grade filter.

**Returns columns:** `ticker`, `alpha_id`, `event_candidate_id`, `reason`,
`failed_condition` (from `response.entry_optimization.failed_condition` when
that object exists, else `None`).

```python
df = discard_reasons_by_grade(results, grade="A")
df["reason"].value_counts()                        # which reasons dominate
pd.crosstab(df["failed_condition"], df["reason"])   # cross-tab against entry_mode="auto"'s outcome
```

**Verified**, same two-ticker pool, `grade="B"`:

```
df.shape == (561, 5)
df["reason"].value_counts().head(3)
# total_trades 4 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    34
# total_trades 9 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    20
# total_trades 6 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    19
```

The walk-forward first-window trade floor (issue #217 — see the manual's
§9, "An unreachable floor is a window, not a verdict") dominates the discard
reasons for grade-B contracts on this fixture — a fact no single
`RuleDiscoveryResponse.rejection_reasons` list makes this visible on its own.

---

## `undetermined_direction_by_family(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format link between a source-feature's semantic family and the
resulting contract's derived `direction` (including `"undetermined"`).

Reads `result.contracts` (every evaluated contract, promoted and rejected
alike — **not** `result.promoted`). Resolves each contract's originating
`EventCandidate` via `event_candidate_id`; a contract whose id doesn't
resolve against `result.candidates` is silently skipped (no row, no
exception). Emits **one row per component** of the candidate's expression,
so a family that only ever appears inside a composed AND-event is still
counted, not hidden behind the composite's own `event_id`.

**Family classification**, exact and order-sensitive: a component with
`len(source_cols) == 2` is `"cross_pair"`; `== 3` is `"cross_triple"`; any
other length (including native, arity-1 components) falls through to a
`{base}_{indicator}_{period}` name match (e.g. `close_rsi_25` → `"rsi"`),
with `"other"` as the final fallback only when that regex doesn't match.

**Returns columns:** `ticker`, `alpha_id`, `event_candidate_id`, `family`,
`direction`.

```python
df = undetermined_direction_by_family(results)
df.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())
```

**Verified**, on `ADAUSDC` alone:

```
fam = undetermined_direction_by_family([result_ada])
fam.shape == (7356, 5)
rate = fam.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())
rate.sort_values(ascending=False)
# family
# cross_triple    0.945455
# ret             0.915367
# cross_pair      0.915001
# vol             0.903101
# other           0.892857
# mdd             0.850000
```

The `undetermined` rate on this fixture sits in a narrow 85–95% band across
every family reached — no family stands out as reliably orientable on this
data. (An earlier version of this document reported only `cross_pair`/
`cross_triple`/`other` here, with no native family ever appearing — that was
the symptom of a real classification bug, since fixed; see
`src/forgedge/docs/modules/Playground.md` §4 for the full story. The numbers
above are from the corrected function, re-verified against a live run.)

---

## `dead_event_candidates(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format classification of every gate-surviving candidate's fate in M2.

Reads `result.contracts` (indexed by `event_candidate_id`) and
`result.candidates` (every `EventCandidate` that already passed the
Consistency Gate), then labels each candidate `"dead"` (zero contracts
derived from it), `"undetermined_only"` (contracts exist, but every one has
`direction == "undetermined"`), or `"actionable"` (at least one contract with
a derived direction).

**Returns columns:** `ticker`, `event_candidate_id`, `expression`,
`n_contracts`, `n_undetermined`, `status`.

```python
df = dead_event_candidates(results)
df[df["status"] != "actionable"].groupby("ticker").size()   # M1->M2 waste per asset
```

**Verified**, on a fresh two-ticker pool — `ADAUSDC` (the repository's
reference fixture) plus a synthetic `BTCUSDC` series, built via
`forge_multi()` this time (not a plain `forge()` per ticker like the M0/M2
examples above) so a genuine pooled cross-ticker `RuleRegistry` is available
for the M4 examples below. Every "Verified" block from here through the end
of this document uses this same `forge_multi()` pool:

```
dead.shape == (11550, 6)
dead.groupby(["ticker", "status"]).size()
# ticker    status
# ADAUSDC   actionable             468
#           undetermined_only    4888
# BTCUSDC   actionable            3238
#           undetermined_only    2956
```

No `"dead"` row appears for either ticker on this fixture: `len(result.contracts) == len(result.candidates)` holds exactly for both (Alpha Discovery evaluates every gate-surviving candidate exactly once), so the only real split observed here is whether that one contract ever got a derived direction.

---

## `gate_survival_observed(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format Consistency Gate outcome for every raw candidate evaluated —
pass and fail alike.

Reads `result.event_discovery.raw_events` (the full pre-gate population,
present when `DiscoveryConfig.retain_raw_events=True`, the default), each
annotated with its `GateResult`, alongside the `GateParams` that decided
pass/fail. Silently **skips** any result where Event Discovery wasn't run,
or ran with `retain_raw_events=False`.

**Returns columns:** `ticker`, `mean_tpm`, `index_of_dispersion`,
`episode_index_of_dispersion`, `n_episodes`, `passed`, `fail_reason`,
`min_tpm`, `max_dispersion`, `dispersion_margin`, `event_counting` — the last
four repeat the configured thresholds on every row for direct per-row
observed-vs-threshold comparison.

```python
df = gate_survival_observed(results)
df.groupby("ticker")["passed"].mean()                                          # observed survival rate per asset
df.groupby("ticker").apply(lambda g: (g["mean_tpm"] < g["min_tpm"]).mean())     # how much rejection is tpm-driven
```

**Verified**, same pool:

```
gs.shape == (10535, 11)
gs.groupby("ticker")["passed"].mean()
# ADAUSDC    0.746607
# BTCUSDC    0.694371
gs.groupby("ticker").apply(lambda g: (g["mean_tpm"] < g["min_tpm"]).mean())
# ADAUSDC    0.044271
# BTCUSDC    0.057450
```

---

## `diagnostics_vs_verdict(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format link between M2's non-blocking diagnostics and the M3 verdict.

Explodes `AlphaContract.diagnostics` — observations that inform the alpha
grade but gate nothing in M2 — against the `RuleDiscoveryResponse.verdict`
M3 later assigned the same contract. A contract with no diagnostics still
emits one row, with `diagnostic=None`.

**Returns columns:** `ticker`, `alpha_id`, `grade`, `diagnostic`, `verdict`.

```python
df = diagnostics_vs_verdict(results)
pd.crosstab(df["diagnostic"], df["verdict"], normalize="index")   # which diagnostics skew NON-EDGE
```

**Verified**, same pool:

```
dv.shape == (5145, 5)
dv["diagnostic"].value_counts(dropna=False).head(3)
# NaN                                                                     1883
# OOS sample too small for reliable statistics (n_oos_activations=7 < 10)  143
# OOS sample too small for reliable statistics (n_oos_activations=8 < 10)  142
```

Every non-`NaN` diagnostic on this fixture is a variant of the same
OOS-sample-size warning — a concrete illustration of the kind of pattern
this function exists to surface: this exact wording, at this frequency,
would be a strong candidate to promote into an actual M2 gate.

---

## `lottery_only_winners(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format flag for `PARTIAL-EDGE` contracts blocked only by the
search-level rotation null.

Filtered to `verdict == "PARTIAL-EDGE"` contracts only. `rotation_only` is
true when `rejection_reasons` has exactly one entry and it starts with
`"search-level rotation null not cleared"` — a contract that cleared every
economic/statistical gate and only lost the multiple-testing lottery, as
opposed to one still failing on PF, DSR, OOS consistency, etc.

**Returns columns:** `ticker`, `alpha_id`, `grade`, `rotation_p`,
`rotation_threshold`, `n_reasons`, `rotation_only`.

```python
df = lottery_only_winners(results)
df.groupby("grade")["rotation_only"].mean()
```

**Verified**, same pool:

```
lw.shape == (96, 7)
lw["rotation_only"].value_counts()
# False    89
# True      7
```

`lw.shape[0] == 96` matches the total edge count across both tickers
(`44 + 52`, see `conversion_funnel` below) — on this fixture, with default
(non-preset) configuration, **every** tradeable contract lands at
`PARTIAL-EDGE`, never a full `EDGE` (see `monitoring_manifest` in
`deployment_en.md` for the same fact from the deployment side).

---

## `classification_by_grade(registries: Iterable[RuleRegistry]) -> pd.DataFrame`

Long-format link between a rule's originating alpha grade and its
cross-ticker classification.

Unlike every other function in this document, this one (and
`duplicate_clusters` below) takes `RuleRegistry` objects directly, not
`ForgeResult` — the cross-ticker classification lives on the **pooled**
registry `forge_multi()` returns separately (each per-ticker `ForgeResult`
has `.registry = None` on that path). Pass `[result.registry]` for a
single-ticker `forge()` run, or `[registry]` for a `forge_multi()` pooled
registry. One row per `RuleDocument` with a non-`None` `classification`
(`None` when Step 4 never ran).

**Returns columns:** `rule_id`, `source_ticker`, `grade`, `classification`.

```python
df = classification_by_grade(registries)
pd.crosstab(df["grade"], df["classification"], normalize="index")   # do grade-A rules skew GENERIC?
```

**Verified**, on the pooled `forge_multi()` registry over ADAUSDC + BTCUSDC:

```
cbg.shape == (96, 4)
pd.crosstab(cbg["grade"], cbg["classification"])
# classification  GENERIC  ISOLATED
# grade
# A                     8        46
# B                    13        23
# C                     2         4
```

Only `GENERIC`/`ISOLATED` appear on this fixture — no rule lands in the
middle (`PARTIAL`/`SPECIFIC`) between "holds on every other ticker" and
"holds on none." Grade A does *not* skew more `GENERIC` here (8/54 ≈ 15%)
than grade B (13/36 ≈ 36%) — if anything the reverse, on this two-ticker
pool — a fact only visible by asking the question this function exists to
ask, not something the alpha grade itself would predict.

---

## `duplicate_clusters(registries: Iterable[RuleRegistry]) -> pd.DataFrame`

Long-format dedup outcome for every rule in the registry.

Same `Iterable[RuleRegistry]` input as `classification_by_grade` above (see
that section for why). One row per `RuleDocument`, unfiltered, flagging
whether it was marked a duplicate and, if so, which surviving `rule_id` it
was folded into.

**Returns columns:** `rule_id`, `source_ticker`, `grade`, `is_duplicate`,
`duplicate_of`.

```python
df = duplicate_clusters(registries)
df["is_duplicate"].mean()                                                              # overall dedup rate
df[df["is_duplicate"]].groupby("duplicate_of").size().sort_values(ascending=False)     # largest absorption clusters
```

**Verified**, same pooled registry:

```
dc.shape == (96, 5)
dc["is_duplicate"].mean()
# 0.5104166666666666
dc[dc["is_duplicate"]].groupby("duplicate_of").size().sort_values(ascending=False).head(5)
# duplicate_of
# RULE_ADA_23    3
# RULE_ADA_16    2
# RULE_BTC_28    2
# RULE_BTC_29    2
# RULE_ADA_41    2
```

Just over half of every tradeable rule pooled across both tickers (`51%`) is
marked a duplicate on this fixture — consistent with `promotion_gate`'s
default `block_duplicate=True` in `forgedge.deployment` blocking the
majority of contracts by this one flag alone (see
`src/forgedge/docs/specs/deployment_en.md`).

---

## `conversion_funnel(results: Iterable[ForgeResult]) -> pd.DataFrame`

Long-format end-to-end funnel count per asset, across every module — the one
use case not anchored to a single module.

One row per `(ticker, stage)` with the population size at that stage:
`candidates` (M1 gate survivors), `contracts` (every M2 evaluation, promoted
and rejected), `promoted` (M2 hypotheses handed to M3), `edges` (M3
`EDGE`/`PARTIAL-EDGE` verdicts, `result.edges()`).

**Returns columns:** `ticker`, `stage`, `n`.

```python
df = conversion_funnel(results)
df.pivot(index="ticker", columns="stage", values="n")   # the funnel table, one row per ticker
```

**Verified**, same pool:

```
cf.pivot(index="ticker", columns="stage", values="n")
# stage    candidates  contracts  edges  promoted
# ADAUSDC        5356       5356     44      468
# BTCUSDC        6194       6194     52     3238
```

`candidates == contracts` exactly for both tickers — the same fact
`dead_event_candidates` above relies on (every gate-surviving candidate is
evaluated by Alpha Discovery exactly once). The conversion rate from
`promoted` to `edges` is far lower on `BTCUSDC` (3238 → 52, ~1.6%) than on
`ADAUSDC` (468 → 44, ~9.4%) despite `BTCUSDC` promoting almost seven times
more contracts — a visible example of why a raw `promoted` count alone
overstates how much of a session's search actually pays off.

---

## What's next

The issue #237 tracking checklist is now complete — all 11 use cases across
M0-M4 plus the cross-cutting funnel above are implemented. What comes next
is a different concern: putting a discovered rule into **production**
(quality-gating it, exporting it to disk, indexing it for monitoring) lives
in the sibling module `forgedge.deployment`, not here — see
`src/forgedge/docs/specs/deployment_en.md` for its usage guide and
`src/forgedge/docs/modules/Playground.md` §10 / `modules/Deployment.md` for
why it was split out (issue #245, PR #247).
