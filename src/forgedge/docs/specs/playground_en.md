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

**This module is explicitly still evolving, not a stable core API on the
level of `forge()` or `RuleDiscovery`.** It follows an open checklist of 11
planned use cases (GitHub issue #237); four are implemented today. Check the
installed version's docstrings if a signature below looks out of date.

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

## What's next

Seven more use cases are open on the same tracking checklist (issue #237) —
two more for M1, two for M3, two for M4, and one end-to-end conversion-rate
case that cuts across every module. See
`src/forgedge/docs/modules/Playground.md` §5 for the full roadmap table and
the design principles a new use case is expected to follow.
