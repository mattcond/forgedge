# Pipeline-wide parameter coherence — audit and proposal

**Starting point:** issue [#173](https://github.com/mattcond/forgedge/issues/173) —
`GateParams.min_tpm` (M1) is never checked against `WalkForwardConfig.min_train_months`
(M3), so lowering the event rate in M1 silently eliminates every candidate at M3's
first fold.

**Question this audit answers:** #173 is not a bug in one gate. It is one instance of a
*shape*. How many other instances does the pipeline contain, and what single mechanism
would remove the whole class?

**Method:** full sweep of every configuration dataclass in `src/forgedge/`
(17 classes, 158 fields) plus every module-level numeric constant that acts as a
threshold; then empirical confirmation of each candidate finding on this repo's own
golden fixture (`tests/fixtures/ADA_1D_TRAIN.parquet` — 882 daily bars,
2024-01-01 → 2026-05-31 ≈ 29 months).

Reproduce every number quoted below with:

```bash
python examples/pipeline_coherence_audit.py rates m1 m2 m3
```

---

## 1. The shape, stated generally

FORGE has a set of **latent parameters** — quantities that have exactly *one* meaning
for the whole pipeline:

> "how often do we expect this thing to fire", "how many observations make a statistic
> credible", "how long is one bar", "where does in-sample end", "what does a round trip
> cost", "what counts as significant".

None of them is represented anywhere. Each is *materialised* as several independent
config fields — different names, different units, different owners, each with its own
class default — scattered across M0…M4. Three consequences follow mechanically:

1. **No single source of truth.** Setting the "same" parameter in one module leaves the
   other copies at their defaults.
2. **No cross-check.** Nothing verifies that the copies are mutually satisfiable. Two
   individually reasonable values can be jointly impossible.
3. **Silent failure.** The pipeline has no notion of "internally inconsistent
   configuration", so an impossible combination surfaces as *rejections* — indistinguishable
   from "the signal is bad", which is exactly what the user was trying to measure.

Issue #173 is case (2)+(3) for the pair *(arrival rate, evaluation-window length)*.
The audit found **sixteen** instances across seven latent parameters, four of them with
the same severity as #173.

> **Revision note.** F16 was added, and F9 revised down from MEDIUM to LOW, during the
> design review that turned this audit into a work plan
> ([#184](https://github.com/mattcond/forgedge/issues/184)). Both changes are marked in
> place in §3.

`forge_preset()` is the only thing in the codebase that acts as a centraliser today, and
its module docstring states the intent explicitly:

> *"Il `min_tpm` usato da EventDiscovery (M1) e RuleDiscovery (M3) devono essere
> consistenti … forge_preset() imposta lo stesso floor di frequenza per tutti e tre i
> moduli, scalato al timeframe."*

It keeps that promise for exactly **two** fields (`GateParams.min_tpm`,
`SelectionCriteria.min_tpm`) out of the ~12 fields that encode the same latent parameter.
Everything else it constructs is left at a class default that its own scaling logic
contradicts.

---

## 2. Inventory — seven latent parameters, ~50 materialisations

| Latent parameter | Materialised as | Owned/scaled by `forge_preset`? |
|---|---|---|
| **Arrival rate** (events or trades per month) | `GateParams.min_tpm` (M1) · `SelectionCriteria.min_tpm` (M3) · `ScoringParams.pf_min_tpm` (M3) · `ScoringParams.pf_tpm_target` (M3) | first two only |
| **Minimum credible sample** (absolute count) | `GateParams.min_episodes` · `_MIN_STATS_CASES` (M2, module constant) · `_MIN_TRADES_ABS` (M3, module constant) · `ScoringParams.pf_min_trades` · `SelectionCriteria.min_oos_trades` · `AlphaConfig.horizon_enrichment_min_obs` · `AlphaConfig.min_regime_obs` · `RegistryConfig.cross_min_active` — all **nominal**; only M1 has an *effective* count (`ActivationStats.n_eff`), see F16 | **none** |
| **Evaluation window** the floors apply to | `DiscoveryConfig.train_ratio` · M1 `WalkForwardConfig.n_splits` · `AlphaConfig.train_ratio` · `AlphaConfig.embargo_bars` · M3 `WalkForwardConfig.{min_train_months, n_splits, train_span_months, test_span_months, purge_bars, embargo_bars}` · `TimeBudget.split` | `AlphaConfig.train_ratio` only |
| **Bar duration / timeframe** | `forge(timeframe=)` · `AlphaConfig.timeframe` (declared *metadata-only*) · `AlphaConfig.bars_per_day` · `EMAProxyConfig.bar_hours` (else inferred) · `MarketContextConfig.stable_window` · `BacktestParams.{target_h, buy_delay_bar}` · M3 `WalkForwardConfig.*_months` | M2 fields only |
| **Economic constants** | `AlphaConfig.fee_per_side` vs `BacktestParams.fee` · `AlphaConfig.mfe_floor` vs hardcoded `0.01` in `_seed_base_params` · `SelectionCriteria.{min_profit_factor, partial_min_profit_factor}` vs `RegistryConfig.cross_pf_threshold` | none |
| **Significance level** | `PromotionThresholds.{max_p_value, ic_max_p, oos_max_p, fdr_q}` · `SelectionCriteria.{max_ttest_p, max_rotation_p}` · `RotationConfig.alpha` · M1 `WalkForwardConfig.min_pass_rate` | `fdr_q`, `oos_max_p` only |
| **Schema / column names** | `timestamp_col` on `DiscoveryConfig`, `AlphaConfig`, `RuleDiscoveryConfig`, `RegistryConfig` (+ 6 function defaults) · `AlphaConfig.close_col` vs `BacktestParams.{target_col, target_hit_col, buy_price_anchor}` · `AlphaConfig.{regime_col, regime_stable_col}` | `DiscoveryConfig` only |

---

## 3. Findings

Ranked by measured impact. **CRITICAL** = systematically eliminates candidates for
configuration reasons, with no diagnostic; **HIGH** = silently changes what the pipeline
selects or reports; **MEDIUM/LOW** = latent or cosmetic.

### F1 — M1: `GateParams.min_episodes` is absolute, the OOS fold length is not · CRITICAL

`GateParams`' docstring asserts:

> *"All parameters are rate/ratio invariant, making the same `GateParams` instance valid
> for both in-sample discovery and out-of-sample walk-forward validation without any
> scaling."*

`min_episodes: int = 10` is an absolute count, so the claim is false. The walk-forward
path reuses the IS instance verbatim (`discovery.py:630` —
`oos_params = wf.oos_gate_params or cfg.gate_params`) against folds whose length is set by
two *unrelated* parameters, `DiscoveryConfig.train_ratio` and
`WalkForwardConfig.n_splits`.

Measured (`min_tpm=1.0`, `train_ratio=0.80`, `n_splits=3` — the exact configuration
`forge.py`'s own module docstring recommends for production):

| | |
|---|---|
| OOS fold length | 59 bars ≈ **2.0 months** |
| IS gate requires | 1.0 episodes/month |
| Same gate requires on a fold | 10 episodes ⇒ **5.0 episodes/month** |
| ⇒ OOS gate is | **5.1× stricter than the IS gate, by construction** |
| Fold evaluations passing | **37 / 5112 = 0.7 %** |
| — failing on `episodes: N < 10` | 2759 (54.0 %) |
| — failing on `rate` | 2315 (45.3 %) |
| Candidates declared OOS-stable | **9 / 1704 = 0.5 %** |

With `only_validated_events=True` this discards 99.5 % of the search for a reason that
has nothing to do with out-of-sample stability. The `rate` failures are not clean either:
a true 1.0 ep/month Poisson process measured over a 2-month window falls below 1.0 about
40 % of the time from sampling noise alone, so the fold-level rate test is also mostly
measuring window length.

Related: `_scale_gate_params()` (`event_discovery/discovery.py:755`) exists to solve
exactly this, returns its input unchanged with a comment repeating the false invariance
claim, and **is never called**.

### F2 — M3: `min_train_months` × `min_tpm` < `_MIN_TRADES_ABS` (= issue #173) · CRITICAL

Reproduced independently on the ADA fixture, confirming #173's AMZN report:
`forge_preset("balanced", "1D", min_tpm=0.30, rd_min_tpm=0.25)`, everything else default.

```
M1 3941 candidates | M2 599/3941 promoted | M3 343 backtested
verdicts: NON-EDGE 250 · PARTIAL-EDGE 92 · INSUFFICIENT-DATA 1 · EDGE 0

M3 rejection reasons
   214  (62.4%)  total_trades N < N (Nmo × N tpm, not significant)   ← the early-elimination
    93  (27.1%)  search-level rotation null not cleared
    79  (23.0%)  active_months N/N = N% < N%
```

The root cause is visible in one line of the run's own statistics:

```
selection-span length: median 6 months  (= walk_forward.min_train_months,
                                          NOT the in-sample span of 29 months)
```

Under `selection_mode="walk_forward"` (the default), `n_months` in
`_dynamic_min_trades(n_months, min_tpm)` is the **first train window**, so the floor is
`max(10, 6 × 0.25) = 10` — an implied **1.67 trades/month against a configured 0.25**,
6.7× stricter. The realised median rate was 1.33 tr/month, just under the bar: 62 % of
verdicts eliminated before any economics were evaluated.

The `rates` stage shows this is not confined to permissive overrides. Even at stock
preset values the floor is inconsistent with the declared rate whenever the first train
window is short:

| preset "balanced" @ 1D | value |
|---|---|
| M1 `min_tpm` | 1.00 ep/month |
| M3 `criteria.min_tpm` (deliberately ~0.8× M1) | 0.80 tr/month |
| early-elim floor over `min_train_months=6` | `max(10, 4.8) = 10` ⇒ **1.67 tr/month implied** |

The floor is 2.1× the configured M3 rate and 1.67× the M1 rate — the intended ordering
(M3 slightly *more* permissive than M1) is inverted by the absolute term.

### F3 — M3: `ScoringParams` belongs to no one, and it is the selection objective · CRITICAL

`ScoringParams(pf_min_trades=15, pf_min_tpm=2, pf_tpm_target=3)` is **never touched by
`forge_preset()`** — it keeps identical values on 1D and on 15m, where the preset's own
frequency gate differs by 96×. These are not diagnostics: `pf_score_tpm` is the objective
the grid screening *maximises* (`grid.py:99,128`) and a gate in `_passes`
(`grid.py:139`), so it decides which operating point gets published.

`pf_score_tpm = PF × c_norm`, with `c_norm` a band-pass centred on `pf_tpm_target`:

| realised tpm_mu | c_norm | PF needed to clear `min_pf_score_tpm=0.30` |
|---|---|---|
| 0.5 | 0.098 | **3.07** |
| 1.0 | 0.167 | 1.80 |
| 3.0 | **0.366** | 0.82 |
| 10.0 | 0.240 | 1.25 |
| 30.0 | 0.154 | **1.94** |

Two distinct defects follow:

* **Low-frequency (1D):** measured on the ADA run, `c_norm` median **0.143**, with
  **97.1 %** of rules below 0.30. `min_pf_score_tpm=0.30` therefore acts as a *hidden,
  rate-dependent PF bar* of 1.8–3.1 — silently overriding the explicit
  `min_profit_factor=2.0` / `partial_min_profit_factor=1.5`.
* **High-frequency (15m, 1H):** the preset's frequency gate demands 19–77 trades/month
  while the objective function *penalises* everything above 3/month. The gate and the
  score point in opposite directions; grid selection will prefer the cell that trades
  least, against an explicit requirement to trade more.

### F4 — M3: `min_oos_trades` is absolute, the pooled test span is derived · HIGH

Same shape as F2, one gate downstream. `criteria.min_oos_trades=10` is compared against
the pooled walk-forward test trades, whose count is `post-train span × min_tpm` — a
function of `min_train_months`, `n_splits`, `test_span_months` and the data length, none
of which is checked against 10. On the ADA run only 1 `INSUFFICIENT-DATA` appeared
because F2 had already converted the candidates to `NON-EDGE` (never rescued). This is
precisely #173's observation that fixing `min_train_months` makes
`INSUFFICIENT-DATA ("pooled OOS trades 7 < 10")` appear: F2 masks F4.

### F5 — Bar duration has three independent sources of truth · HIGH

`forge(timeframe=…)` is the declared truth and drives the M2 horizon grid.
`AlphaConfig.timeframe` is documented as *"traceability metadata only — no effect
whatsoever on any measurement"*, yet `forge()` and `forge_preset()` both read a timeframe
to scale real parameters. `MarketContext` neither receives it nor is told about it: it
independently *infers* bar duration from timestamp spacing (`context.py:_infer_bar_hours`),
falling back to `EMAProxyConfig.bar_hours` — a third copy. Nothing reconciles the three.

M3 has no concept of a timeframe at all, which is why its defaults are quietly hourly:
`BacktestParams.target_h=24` and `buy_delay_bar=6` mean *24 days* and *6 days* on 1D
data — the same footgun already fixed for `AlphaConfig.horizon_grid`
(`_warn_if_hourly_grid_on_slow_timeframe`), left unfixed one module later.
`forge_preset()` does not scale either field. `MarketContextConfig.stable_window=12`
bars is a fourth unscaled bar-count.

### F6 — The IS/OOS split is cut three times, and `TimeBudget` is not threaded by default · HIGH

`timebudget.py`'s module docstring states:

> *"`forge()` builds a single budget and threads it through, so the session shares one axis."*

It does not. `forge()` forwards whatever the caller passed (default `None`); M1 then cuts
at `DiscoveryConfig.train_ratio`, M2 independently at `AlphaConfig.train_ratio`, and M3
never receives a budget at all — it cuts its own timeline with `WalkForwardConfig`.
Under `forge_preset()` the two ratios are **1.0 and 0.70**: M1 uses the whole table while
M2 holds out 30 %. `ForgeResult.time_budget` reports only M2's axis, so the run's
recorded "temporal axis" is not the axis M1 or M3 used. `forge_preset()` accepts a
`train_ratio` argument that reaches M2 only.

### F7 — Fee is specified twice and reconciled never · MEDIUM

`AlphaConfig.fee_per_side=0.002` is stamped onto every `AlphaContract`; `BacktestParams.fee=0.002`
is what the backtest actually charges (`backtest.py:246`). `_seed_base_params` seeds
`direction`, `target_h` and `sell_pct` from the contract but **not** `fee` — grep confirms
no code path reads `contract.fee_per_side`. They agree only because both defaults are
0.002: a user who sets `AlphaConfig(fee_per_side=0.0005)` gets contracts documenting 5 bp
and backtests charging 20 bp, with no warning.

### F8 — M4 requires a stricter PF than M3 ever demanded · MEDIUM

`RegistryConfig.cross_pf_threshold=2.0` is an independent copy of
`SelectionCriteria.min_profit_factor=2.0`. But `PARTIAL-EDGE` rules — admitted at
`partial_min_profit_factor=1.5` — *do* reach the registry (`is_edge` is true). Any rule
promoted between PF 1.5 and 2.0 is structurally incapable of scoring a cross-ticker
`PASS`, so it can never be classified `GENERIC` however well it transfers. Lowering M3's
PF bar silently tightens M4's relative bar. `RegistryConfig.cross_min_active=10` is a
further un-scaled absolute count (F1's shape, in M4).

### F9 — Significance is spelled seven ways with no stated relationship · LOW

> **Revised down from MEDIUM.** The first draft of this finding claimed that "a preset's
> permissiveness dilutes as it travels downstream". Checking what each threshold actually
> gates, that claim holds for **one** of the seven, and the headline example was wrong.
> The corrected analysis is below.

Seven numbers, and they are not the same quantity:

| threshold | module | what it gates | kind |
|---|---|---|---|
| `max_p_value` 0.05 | M2 | p of the excess-return t-test | per-hypothesis α |
| `ic_max_p` 0.05 | M2 | p of the feature's IC (Spearman) | per-hypothesis α |
| `max_ttest_p` 0.05 | M3 | p of expectancy on the trade ledger | per-hypothesis α |
| `max_rotation_p` 0.05 | M3 | p of the whole **search surface** | per-hypothesis α, different null |
| `RotationConfig.alpha` 0.05 | calibration | survivor bar | per-hypothesis α |
| `fdr_q` 0.10 | M2 | **false-discovery rate** over the horizon family | **q — not an α** |
| `min_pass_rate` 0.6 | M1 | fraction of folds that must pass | **a vote, not a probability** |

Tying `fdr_q` to α would be a category error: `q=0.10` means "10 % of my promotions may be
false", `α=0.05` means "this single test has a 5 % false-positive rate". `min_pass_rate`
is not a probability at all.

**Where the original claim fails.** The example given was `"sweep"` (`fdr_q=0.25`,
deliberately permissive) still facing `max_rotation_p=0.05`. That is the **intended
design**, not drift: `presets.py` documents that `"sweep"` must be paired with
`RotationConfig(k>=100)` — its upstream permissiveness is *predicated* on the rotation
null filtering downstream. A strict `max_rotation_p` is the filter `"sweep"` relies on.

Checking the rest:

* `max_p_value` is reachable only when `use_fdr=False`, and every preset (and the class
  default) sets `use_fdr=True` — **it is inert under every preset**, a public field that
  does nothing;
* `ic_max_p` feeds a non-blocking diagnostic that only weights the grade;
* `oos_max_p` and `fdr_q` are already scaled per preset.

**What remains** is one real observation: `max_ttest_p` is the pipeline's only *hard*
per-hypothesis gate — it produces `NON-EDGE` in `_decide` — and no preset ever touches it.
Plus the general hazard of seven numbers with no declared relationship, which is a trap
for whoever edits them next rather than a measured defect.

### F10 — `timestamp_col` exists in four configs; `forge_preset` sets one · MEDIUM

`forge_preset(..., timestamp_col="ts")` configures M1 only. M2, M3 and M4 keep
`"open_dt"` and fail later with `Set AlphaConfig(timestamp_col='...')` — an error the user
already believed they had answered. Same for `AlphaConfig.close_col` vs
`BacktestParams.target_col` / `target_hit_col` / `buy_price_anchor`: four column names for
"the price series", independently defaulted.

### F11 — M3 silently overrides M2's minimum take-profit · LOW (latent)

`_seed_base_params` clamps the derived target with a hardcoded, non-configurable
`max(0.01, sell_pct)`, while M2's own floor is `AlphaConfig.mfe_floor=0.005`. The two
disagree by 2×, and the stricter one is the one the user cannot reach. On ADA 1D it never
binds (minimum derived `sell_pct` = 0.025), but on intraday timeframes, where a
median-MFE target is routinely sub-1 %, it would replace the data-derived target with a
constant — violating invariant #3 ("the economic target is derived per event") without
any note beyond `seeded from contract target`.

### F12 — `target_h = 0` is legal on the contract but cannot survive seeding · LOW

`BacktestParams.target_h`'s docstring (post-#158) states `0` is *"a legal, meaningful
value … not a 'no horizon' placeholder"*. `_seed_base_params` guards with
`if dt.holding_period_h and dt.holding_period_h > 0`, so a derived `h*=0` falls through to
`base_params.target_h = 24` — the hourly default of F5. Two sources of truth, and the
fallback is the wrong one.

### F13 — Two different classes named `WalkForwardConfig` · LOW (hygiene)

`event_discovery.models.WalkForwardConfig(n_splits=3, min_pass_rate, oos_gate_params)` and
`rule_discovery.models.WalkForwardConfig(n_splits=4, min_train_months, …)` share a name,
share a field name with different semantics (`n_splits`), and are imported from different
paths in the same `forge()` call site. Any cross-module coherence rule has to disambiguate
them first.

### F14 — M2's sample floor is the top "rejection reason" and rejects nothing · LOW

`_MIN_STATS_CASES=10` on the OOS tail produces
`[diagnostic] OOS sample too small for reliable statistics` on **89.7 %** of contracts at
`train_ratio=0.70` and **96.8 %** at `0.85` — the single most frequent entry in
`rejection_reasons`. It is genuinely non-blocking (promotion needs only a derivable
direction, and `OOSValidation.passed` is correctly p-value-based, so sample size is
already encoded in `p`). The defect is therefore *reporting*: the field named
`rejection_reasons` is dominated by a non-rejection whose frequency is a function of
`train_ratio`, which trains users to ignore it. Note the inversion — tightening
`train_ratio` from 0.70 to 0.85 *raises* the count from 1493 to 1611.

### F15 — Dead code asserting a false invariant · LOW

`_scale_gate_params()` (F1) — never called, and its docstring is the canonical statement
of the invariant F1 disproves.

### F16 — The floors count nominal trades; the statistics need effective ones · HIGH

Same latent parameter as F1/F2/F4 (*minimum credible sample*), different defect: not
floor-versus-window, but **nominal count versus effective count**.

`run_backtest` is fully vectorised — `entry_rn = np.where(active)[0]` — with no
position-state machine, so every active bar opens a trade and **positions overlap
freely**. Issue [#168](https://github.com/mattcond/forgedge/issues/168) measured it: on
EURJPY 1D with `target_h=36`, 118 trades from 120 signal bars / 76 episodes, **3.71
concurrently open positions on average, 12 at peak**.

That entry policy is *correct* and must not change: given enough capital, those 118 trades
are exactly reproducible in production. #168 says so explicitly in its non-goals, and it
also explicitly defers "the separate statistical-independence concern … a distinct, deeper
problem noted separately in discussion". **F16 is that deferred problem.**

The defect is that the inference machinery consumes the nominal count as if the
observations were independent:

| quantity | reads | should read |
|---|---|---|
| `total_trades`, PF, expectancy, net gain | nominal | **nominal — this is the economics, leave it** |
| t-test df (`max_ttest_p` gate) | nominal | effective |
| `n_obs` of `deflated_sharpe` | nominal | effective |
| `expectancy_mde` / power gate | nominal | effective |

With #168's own numbers, nominal 118 against an effective ≈ 32 (118 / 3.71):

```
t-test:  t scales as sqrt(n) → overstated by sqrt(118/32) = 1.93×
         a t of 2.6 (p≈0.005) is really t=1.35 (p≈0.09) — significance evaporates
DSR:     n_obs 118 → 32 at n_trials=15: correction 0.820 → 0.741, ~11 % overstated
```

The t-test channel is the serious one: `max_ttest_p` is one of the three hard `NON-EDGE`
gates, so the pipeline is currently *admitting* rules on overstated significance.

**M1 already has the concept.** `ConsistencyGate` computes
`n_eff = n_episodes / episode_index_of_dispersion` and carries it on `ActivationStats`.
M2 and M3 have no equivalent — the canonical audit shape: a notion that exists in one
module and is missing in the two downstream modules that need it.

Note that episode grouping and concurrency are **different** measures: episodes capture
"same signal cluster", concurrency captures "same price path", and trades from *different*
episodes still overlap when `target_h` exceeds the inter-episode gap. Inference needs the
second. #168's proposed `episode_id` on the trade ledger is the prerequisite machinery.

---

## 4. Proposal — three layers

The layers are independently shippable and ordered by cost. **Layer 2 alone closes
#173 and F1–F4**, which is the recommended first step.

> **This section states the proposal as originally drafted.** The design review in
> [#184](https://github.com/mattcond/forgedge/issues/184) settled sixteen open decisions
> and refined it — Layer 1 became a *directed-constraint resolver* with a resolution
> trace rather than a defaulting mechanism, and the constraint table below is shared by
> the resolver (derive mode) and `config_report()` (check mode). The issues are the
> authority on the final design; §5 carries the resulting order of work.

### Layer 1 — `PipelineContext`: name the latent parameters

A small frozen dataclass holding the session facts that no module owns individually:

```python
@dataclass(frozen=True)
class PipelineContext:
    """The session's ground truth for facts shared by every module."""
    timeframe: str = "1H"
    n_bars: int = 0
    span_months: float = 0.0
    # economics
    fee_per_side: float = 0.002
    # schema
    timestamp_col: str = "open_dt"
    close_col: str = "close"
    regime_col: str = "regime"
    # statistical policy
    target_rate_tpm: Optional[float] = None    # arrival rate, in this timeframe's units
    min_sample: int = 10                       # minimum credible observation count
    alpha: float = 0.05                        # significance level

    @property
    def bars_per_month(self) -> float: ...
    @property
    def bars_per_day(self) -> float: ...
    @property
    def bar_hours(self) -> float: ...
    def months_of(self, n_bars: int) -> float: ...
    def bars_of(self, months: float) -> int: ...
```

`forge()` builds it once from its own arguments plus the data, and each module config
gains `resolve(ctx) -> Self` filling only the fields the user left unset. This requires
one enabling change: **"unset" must be distinguishable from "class default"**. Today
`_warn_if_hourly_grid_on_slow_timeframe` works around its absence by comparing against
`AlphaConfig.__dataclass_fields__["horizon_grid"].default` — a technique that does not
generalise. Introduce a module-level `UNSET` sentinel for the fields a context can fill,
keeping the current value as the documented fallback when no context is supplied, so
standalone module use is unchanged.

This kills F5, F6, F7, F10, F12 by construction, and gives F1–F4 and F8, F9 a place to
compute against.

### Layer 2 — `config_report()`: check coherence, mirroring `summary_report()`

The repo already has the right pattern for advisory diagnostics: `summary_report()`
returns a `DataQualityReport` of `Finding(level, code, message)` with
`has_critical` / `has_warnings` / `one_line()` / `to_text()`, and never blocks. The
proposal is its exact sibling — `summary_report()` validates the **data**,
`config_report()` validates the **configuration**:

```python
from forgedge import config_report

rep = config_report(disc_cfg, alpha_cfg, rd_cfg, registry_cfg,
                    ctx=PipelineContext(timeframe="1D", n_bars=len(kpi)))
print(rep.to_text())
if rep.has_critical:
    raise ValueError(rep.one_line())
```

`forge()` runs it at start, logs `rep.one_line()` next to `ledger.describe()` (which it
already does), emits one `UserWarning` per critical finding, and stores the report on
`ForgeResult.coherence` so a run can explain itself afterwards. A `strict=True` flag
raises instead — opt-in, so nothing existing breaks.

Reusing `Finding` verbatim means no new pattern and no new public vocabulary.

**Initial rule table.** Every rule states an inequality between materialisations of one
latent parameter, and every message carries the *suggested value*, not just the failure:

| Code | Rule | Covers |
|---|---|---|
| `wf_bucket_too_short` | `wf.min_train_months × criteria.min_tpm >= _MIN_TRADES_ABS` | **#173**, F2 |
| `m1_oos_fold_too_short` | `fold_months × gate.min_tpm >= gate.min_episodes`, where `fold_months = months((1-train_ratio)·n_bars / n_splits)` | F1 |
| `m3_stricter_than_m1` | `criteria.min_tpm <= gate.min_tpm` | F2 |
| `scoring_uncalibrated` | `pf_tpm_target` within ~[0.5×, 2×] of `criteria.min_tpm`; `pf_min_tpm <= criteria.min_tpm`; `pf_min_trades <= _dynamic_min_trades(span, min_tpm)` | F3 |
| `oos_span_too_short` | `pooled_test_months × criteria.min_tpm >= criteria.min_oos_trades` | F4 |
| `timeframe_mismatch` | `AlphaConfig.timeframe`, inferred bar spacing and `EMAProxyConfig.bar_hours` agree; `max(horizon_grid)`, `target_h`, `buy_delay_bar` are sane in bars | F5 (absorbs the existing horizon-grid warning) |
| `split_disagreement` | M1 and M2 `train_ratio` agree, or a `TimeBudget` was supplied | F6 |
| `fee_mismatch` | `alpha.fee_per_side == rd.base_params.fee` | F7 |
| `registry_stricter_than_m3` | `registry.cross_pf_threshold <= criteria.partial_min_profit_factor` | F8 |
| `alpha_level_drift` | the p-thresholds ordered consistently with the preset's intent | F9 |
| `schema_mismatch` | one `timestamp_col` / price column across all four configs | F10 |
| `tp_floor_conflict` | M3's TP floor `<= alpha.mfe_floor` | F11 |

Each rule is a pure function of the configs plus the context — trivially unit-testable in
the house style (seeded synthetic frames, `pytest.warns(..., match=...)` pinning the exact
message), with no pipeline run required.

### Layer 3 — make the floors window-aware, and let `forge_preset` own what it implies

Layer 2 makes the inconsistencies *visible*; Layer 3 stops manufacturing them.

1. **Absolute floors become "insufficient window", not "rejected candidate".** When
   `window_months × rate < abs_floor`, the honest statement is *this window cannot
   support this test*, not *this rule has too few trades*. Emit the distinct outcome
   (`INSUFFICIENT-DATA` already exists in M3 for exactly this) plus a coherence finding —
   never a `NON-EDGE`. Apply to `min_episodes` (F1), `_MIN_TRADES_ABS` (F2),
   `min_oos_trades` (F4).
2. **Rate-relative gates on short windows.** For fold-level tests, compare against the
   Poisson lower bound at the configured rate rather than the point estimate — the
   `ConsistencyGate` already does exactly this for dispersion (`_chi2_ppf_095` floor);
   extend the same reasoning to the rate and episode-count criteria so a 2-month fold is
   not asked to prove a 12-month rate.
3. **Extend `forge_preset`'s ownership** to every field its own scaling logic implies:
   `ScoringParams` (all three), `walk_forward.min_train_months`, `criteria.min_oos_trades`,
   `BacktestParams.{target_h, buy_delay_bar}`, `timestamp_col` on all four configs,
   `fee`, and the M1 `train_ratio` (currently pinned to 1.0 while M2 gets 0.70).
   Everything it constructs should be a value it chose, not a default it forgot.
4. **Housekeeping:** delete `_scale_gate_params` (F15); correct the `GateParams`
   docstring's rate-invariance claim (F1) and `timebudget.py`'s "threads it through"
   claim (F6); rename one of the two `WalkForwardConfig` classes (F13) —
   `EventWalkForwardConfig` / `RuleWalkForwardConfig`, with aliases for compatibility;
   re-tag `[diagnostic]` entries out of `rejection_reasons` into a separate `diagnostics`
   list (F14).

### Compatibility

Layers 1–2 are purely additive: new module, new optional argument, new `ForgeResult`
field, warnings only. Layer 3 changes verdicts — `INSUFFICIENT-DATA` where a `NON-EDGE`
used to be, and preset-scaled `ScoringParams` — so it will move golden values in
`tests/test_golden.py`. Per the repo's convention that is expected for a legitimate
pipeline change and should be re-pinned with a comment, not worked around.

---

## 5. Order of work

Settled in [#184](https://github.com/mattcond/forgedge/issues/184), which is the live
tracking issue. The classification in §3 is by severity; this is by execution order, and
the two deliberately differ — the cheap, enabling, zero-golden-delta work comes first, and
the most invasive change comes last so its diff stays isolated.

| step | issue | scope | closes |
|---|---|---|---|
| 1 | [#183](https://github.com/mattcond/forgedge/issues/183) *(part)* | rename the two `WalkForwardConfig`; split `diagnostics` out of `rejection_reasons` | F13, F14 |
| 2 | [#175](https://github.com/mattcond/forgedge/issues/175) | directed-constraint resolver: `UNSET`, `PipelineContext`, resolution trace | enabling |
| 3 | [#176](https://github.com/mattcond/forgedge/issues/176) | `config_report()` — resolved config + coherence findings, `strict=True` | **#173** *(as diagnostic)* |
| 4 | [#181](https://github.com/mattcond/forgedge/issues/181) | duplicated constants: fee, columns, M3↔M4 PF bar | F7, F8, F10 |
| 5 | [#185](https://github.com/mattcond/forgedge/issues/185) | `entry_mode` default → `"auto"`, both operating points published | — |
| 6 | [#168](https://github.com/mattcond/forgedge/issues/168) | expose episode / overlap info on the trade ledger | prerequisite of F16 |
| 7 | [#177](https://github.com/mattcond/forgedge/issues/177) | window-aware floors, Poisson-bound fold gates, effective sample size | F1, **F2 = #173**, F4, F11, F12, F15, F16 |
| 8 | [#178](https://github.com/mattcond/forgedge/issues/178) | `ScoringParams` owned and rebuilt on a dispersion-based `c_norm` | F3 |
| 9 | [#179](https://github.com/mattcond/forgedge/issues/179) | one source of truth for bar duration | F5 |
| 10 | [#180](https://github.com/mattcond/forgedge/issues/180) | one IS/OOS axis, `TimeBudget` threaded to M3 | F6 |
| 11 | [#182](https://github.com/mattcond/forgedge/issues/182) | significance thresholds derived from one `alpha` | F9 |

Steps 1–3 close #173 *diagnostically* — a warning at start-up carrying the value to set —
without changing a single verdict. Step 7 removes the cause. Steps 4–11 prevent the same
configuration from being built again.

**Governing principle:** one PR per issue, and every golden-value re-pin attributable to a
single cause. Steps 5 and 7–11 each move `tests/test_golden.py`; batching them would make
it impossible to write *why* in the re-pin comment, which is the repo's convention.

### Tails found in use, after the eleven steps closed

Two latent parameters survived the plan and surfaced only once the resolved pipeline was
actually run. Both are the same shape as the eleven — one meaning, several fields — and
both were found by asking a question of a *working* session rather than by reading code,
which is why the plan could not have contained them.

| tail | issue | what was still independent |
|---|---|---|
| of F5 | [#196](https://github.com/mattcond/forgedge/issues/196) | `AlphaConfig.horizon_grid` followed the session only when `forge()` built the config itself; an explicit config on daily candles still scanned up to 48 *days* |
| of F2/F4 | [#200](https://github.com/mattcond/forgedge/issues/200) | `SelectionCriteria.min_tpm` did not follow the rate Event Discovery was told to demand, so `min_train_months` (#177) and `pf_min_tpm` (#178) stayed sized for a rate the session no longer had |
| of #200 | [#204](https://github.com/mattcond/forgedge/issues/204) | the fix above applied M1→M3's fill ratio straight to a *declared episode* rate, but M3 counts bars, not episodes — a unit gap one factor upstream of #200's own |
| of F1 | [#205](https://github.com/mattcond/forgedge/issues/205) | a preset's own dispersion tolerance (`daily_max_dispersion`) never bound in `"episode"` mode on most preset×timeframe combinations — silently overridden by a Poisson floor that does not scale with timeframe while the preset value does |
| of F1 | [#206](https://github.com/mattcond/forgedge/issues/206) | `min_episodes` was an absolute floor fixed across every preset regardless of rate — `sniper`'s description promised "≥2 anni" but its own stock rate needed 4.44 years at 95% Poisson confidence, and no resolver check said so |

\#200 is the more instructive of the two, because the fix is *smaller* than it first looks.
The obvious reading — M1 counts episodes, M3 counts filled trades, so M3 should ask for
somewhat less — is wrong at the session level, and measurement is what says so. Lowering
M3's floor costs history twice: `min_train_months` is sized from the floor with a Poisson
margin, so a lower floor *lengthens* the training window; and the pooled out-of-sample
trade count is `test_months × min_tpm`, so a lower floor *shrinks* it. Both move the wrong
way at once:

| `criteria.min_tpm` | `min_train_months` | minimum span for a verdict |
|---|---|---|
| 2.0 | 8 | 13.0 months |
| 1.6 | 10 | 16.2 months |

A 25 % cut in the floor demands 25 % more data. A 14-month session that could produce a
verdict at 2.0 cannot at 1.6 — which is precisely the failure the whole plan exists to
remove. So the resolver propagates the declared rate **unchanged** (`rate_retention=1.0`),
and the fill margin stays where such judgements belong: in `forge_preset`'s specs, which
disagree about it on purpose (1.00 on `sniper` and `sweep`, 0.80 on `balanced` and
`burst`).

The second lesson is one #179 had already taught and #200 had to re-learn: **an inherited
class default is not a declaration.** `GateParams.min_tpm` defaults to 0.5 and
`SelectionCriteria.min_tpm` to 2.0 — two numbers that disagree by 4x and were never
designed to relate. Propagating the inherited one would not have reconciled them, it would
have propagated the wrong one: M3's floor to 0.4, `min_train_months` from 8 to 40, more
than the reference history can supply, and the walk-forward gone entirely. The resolver
therefore stays silent unless somebody actually chose a rate — the same distinction
`timeframe_declared` draws.

#204 found the gap #200 left behind, and it hid in plain sight inside #200's own fix: the
fill ratio (M3 asks slightly less than M1, because not every episode fills) is legitimate,
but it was being applied to the *episode* rate directly. M3 has no notion of episodes — it
"opens a trade on every active bar, with no flat-state check" — so it counts bars, and a
declared episode rate has to be converted to a bar rate *before* the fill ratio means
anything. Skipping that conversion understated M3's floor by `bars_per_episode` (~1.76,
measured median on `ADA_1D_TRAIN`): `forge_preset("balanced", "1D", min_tpm=2)`'s M3 rate
was 1.6, corrected to 2.93, and the stock preset's own `min_train_months` moved from 20
months to 11 — the exact shape of #200's finding, one conversion earlier in the chain.

The fix generalises #200's own machinery rather than adding a parallel one:
`PipelineContext.bars_per_episode` sits beside `rate_retention` as another *policy* default
(the resolver never reads data to derive it, by the same invariant), and `_derive_m3_rate`
applies it only when `event_counting="episode"` — in `"bar"` mode M1 and M3 already share a
unit, so the factor is 1 and #200's original derivation is untouched. The one thing #204
could not leave alone was `m3_stricter_than_m1`'s own *check*: comparing M1's declared rate
to M3's derived one without the same conversion would have turned every declared episode
rate into a false "M3 is stricter" warning, which is precisely the coherence layer
mis-firing on the fix meant to close it.

#205 is a different shape from #200/#204 — not a missing conversion, but a comparison that
silently favoured whichever side happened to be larger. `ConsistencyGate` has always
compared an event's episode-level dispersion against `max(max_dispersion, poisson_floor)`,
a floor whose purpose is legitimate: never reject an event that is statistically
consistent with a random process at its own rate. The bug was in how the two sides scaled.
The floor is a function of calendar months only (≈1.3–2.2 over any realistic 6–60-month
window); `max_dispersion` was *scaled down* for faster timeframes on top of that. Measured
across all four presets and four timeframes, 12 of 16 combinations had `max_dispersion`
never binding, and `sniper` — the preset built for "regular" events — never bound on any of
them. A preset's own tolerance for burstiness was discarded more often the faster the
timeframe got, the opposite of what the scaling was for.

The fix does not touch the floor's purpose, only how a preset expresses its tolerance over
it: `dispersion_margin` is a multiplier (`eff_max_dispersion = poisson_floor x
dispersion_margin`), not an absolute Index of Dispersion, so the floor can no longer
swallow it — a tight margin (`sniper=1.05`) stays close to what a Poisson process itself
would produce, a loose one (`burst=3.00`) tolerates clustering on purpose. `max_dispersion`
keeps its old, absolute meaning in `"bar"` mode, where there is no floor to be swallowed by
in the first place — the two fields are mode-exclusive rather than one superseding the
other, the same shape #179 used for `event_counting`'s two units.

#206 sits one level above #205 in the same criterion: F1 already established that M1's
*out-of-sample fold* length has to fit `min_tpm` (`m1_oos_fold_too_short`, #177). Nobody had
asked the same question of M1's own *in-sample discovery* window against `min_episodes` — a
different, absolute-count criterion (Criterion 2 of the Consistency Gate), which the naive
`min_episodes / rate` division satisfies only in expectation, #173's mistake recurring at
the one place the whole plan had not yet reached. At `sniper`'s stock rate (0.3
episodes/month) the naive estimate is 33.3 months; the window that actually clears the
floor 95% of the time is 53.3 (4.44 years) — more than double what the preset's own
description claimed ("Richiede IS lungo, ≥2 anni").

The fix has two parts, deliberately independent. A resolver check (`m1_is_window_too_short`,
mirroring `m1_oos_fold_too_short`'s shape but WARN rather than FAIL — a candidate above the
configured rate can still clear the floor on less data, so this is not a structural
impossibility the way an empty OOS fold is) names the gap in `config_report()` before a long
run comes back nearly empty. And `min_episodes` becomes preset-parametrized rather than a
flat class default, resolving the fork the issue explicitly left open: `sniper` keeps 10 —
statistical rigor is the entire point of the preset, so the fix is to correct the
description to the measured number, not to weaken the floor — while `sweep`, sharing
`sniper`'s low rate but none of its precision claim, lowers it to 5, consistent with a
preset that is permissive by design and already defers rigor to the `RotationCalibrator`
downstream. `balanced` and `burst` were already coherent at 10 (16 and 10.7 months) and are
untouched.
