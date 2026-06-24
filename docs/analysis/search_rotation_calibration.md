# Search-level rotation null — contract-preserving alpha calibration

**Goal.** Tell a real edge from one manufactured by FORGE's multiple-testing
surface (thousands of mined candidates + horizon scan + AND composition)
**without rebuilding features** — so it never breaks FORGE's contract (the user
passes a pre-built feature table and nothing else).

**Prototype:** `examples/search_rotation_calibration.py`
**Dataset:** `ADA_1D_FULL.parquet`, IS window = 2024 (366 daily bars).

## Method

1. Run the real pipeline once (MarketContext → EventDiscovery → AlphaDiscovery);
   keep the candidate set and the real promoted contracts.
2. K times, circularly **rotate only the `close` column** of the post-ED table
   by a random offset and re-run **only AlphaDiscovery**, reusing the real
   candidate set. Rotation preserves the exact return marginal and
   autocorrelation (one wrap seam) but decouples each event's (real, cached)
   activations from the forward outcome.
3. Record the **maximum** of a yardstick statistic over all directed contracts
   per run → null distribution of "the best alpha the pipeline manufactures when
   event and outcome are decoupled". Empirical p = `(1 + #{null ≥ real}) / (1+K)`.

**Why it respects the contract:** only `close` (already in the table) and the
feature columns (already in the table) are used; features are never regenerated.
**Why it is cheap:** in FORGE mode EventDiscovery mines events from features and
gates on tpm/dispersion — outcome-independent — so the candidate set is identical
under real and null and only AlphaDiscovery is re-run (~4 s/draw here).

## Result (K = 100, min_tpm = 1.5)

| statistic | real | null mean | null q95 | emp p | verdict |
|---|---|---|---|---|---|
| composite_score | 1.000 | 0.898 | 0.979 | 0.0495 | **saturates** (4/100 noise draws also hit ~1.0) |
| abs_z (rotation z) | 3.67 | 3.64 | 4.61 | 0.45 | does **not** separate |
| is_t (IS t-test) | 6.36 | 5.39 | 7.96 | 0.17 | does **not** separate |
| oos_t (OOS t-test) | 3.64 | 2.99 | 5.51 | 0.23 | does **not** separate |
| **is_lift** | **0.707** | 0.467 | 0.644 | **0.0099** | **separates** (+2.5 sd) |

The 2 alphas that clear the lift bar are *not* the top-by-composite events but two
*long* `delta_ratio_close_ema` events — and on inspection both have only **9 IS
activations** and **fail OOS** (WR 0.78 IS → 0.40 OOS). High lift on 9 samples is
WR=7/9: impressive but fragile. This motivates the upstream `min_tpm` fix below.

## Findings

1. **The method works and is contract-safe / cheap.** Rotating only the outcome
   and reusing the real candidate set gives a valid search-level (Westfall–Young
   style) max-statistic null without any feature recipe.

2. **The pipeline's internal significance stats are consumed by the search.**
   Picking the best of 2621 candidates yields |z|≈3.7, IS t≈6.4 and even
   **OOS t≈3.6 with a fully decoupled outcome**. Noise routed through the same
   search produces equally extreme z/t/OOS-t (p = 0.17–0.41). This is exactly the
   multiple-testing the internal BH-FDR cannot see, and confirms an *external*
   search-level null is needed.

3. **Only `lift` survives.** The best real alpha's IS lift (0.707) sits +2.4 sd
   above what rotation manufactures (q95 = 0.643), p = 0.024. Win-rate lift is far
   harder to fabricate by decoupling than a t-stat (which the search inflates via
   low-variance flukes).

4. **The calibrating statistic changes which alphas are selected.** The 2 alphas
   that clear the lift bar are *not* the top-by-composite (short-vol) events but
   two *long* `delta_ratio_close_ema` events. Calibrating on the saturated
   composite would select the wrong candidates.

## Design implications

- Calibrate on **`lift`** (an unbounded economic magnitude), **not** on
  `composite_score` nor on the internal z/t/OOS-t — those are already exhausted
  by the search.
- On ADA 1D the pipeline yields **2 credible alphas out of 111** promoted. The
  filter is strict but is the first discriminant that actually works at p < 0.05
  (the earlier count-based surrogate test *inverted*; see
  `lowfreq_robustness.md`).
- This is a candidate for the v2 calibrator (issue #115) that does **not** need
  the feature-construction module: it is implementable today.

## Upstream fix: raise `min_tpm` instead of filtering downstream (K=100)

The 2 survivors above had only **9 IS activations** and **failed OOS** — elegant
false positives. The right lever is to raise Event Discovery's `min_tpm` so events
must fire more often. `mean_tpm = n_act / n_total_months`, so `min_tpm=3` over 12 IS
months floors activations at ~36.

Full K=100 results with min_tpm=3.0:

| statistic | real | null mean | null sd | null q95 | emp p | verdict |
|---|---|---|---|---|---|---|
| **composite** | **1.000** | 0.819 | 0.088 | 0.941 | **0.0297** | **separates** (+2.1 sd) |
| abs_z | 2.963 | 3.216 | 0.458 | 3.959 | 0.693 | does not separate |
| is_lift | 0.287 | 0.312 | 0.080 | 0.440 | 0.624 | **saturates** (real < null mean) |
| is_t | 6.357 | 4.767 | 1.055 | 6.455 | 0.069 | borderline |
| oos_t | 3.638 | 2.458 | 1.340 | 4.968 | 0.149 | does not separate |

Comparison across both `min_tpm` regimes (K=100 for 1.5, K=100 for 3.0):

| | min_tpm=1.5 (K=100) | min_tpm=3.0 (K=100) |
|---|---|---|
| candidates mined | 2621 | **584** |
| directed contracts | 111 | **16** |
| best real lift | 0.707 | 0.287 |
| separating statistic | is_lift (p=0.0099) | **composite** (p=0.030) |
| survivors above null | 2 | **1** |
| survivor IS activations | 9, 9 | **29** |
| survivor OOS | **failed** (WR 0.78→0.40) | **passed** (p=0.000, OOS WR 0.50) |

Key points:

1. **`min_tpm` filters during mining, not after.** 584 candidates vs 2621 means a
   smaller "best-of-N" search, so the rotation null bar itself drops (null is_t
   mean 5.39 → 4.77; null composite q95 0.979 → 0.941). A *downstream*
   `min_is_activations` gate cannot do this — the multiple-testing damage is
   already done by the time 2621 candidates exist. The upstream lever is
   structurally better.

2. **The single survivor confirms OOS.** `pr_diffnorm_close_vol12_vol24_48 < 0.167`
   (short, h=10, 29 IS activations, IS WR=0.48) is the only alpha in this study that
   is *both* above the rotation null *and* OOS-confirmed (n=18, WR=0.50, p=0.000).
   The min_tpm=1.5 survivors were not.

3. **The discriminating statistic switches.** With min_tpm=3.0, `is_lift` *saturates*
   in the other direction: the real lift (0.287) falls *below* the null mean (0.312).
   Frequent events fire too often to produce high lift, so noise's lift distribution
   overlaps or exceeds the real one. `composite` takes over as discriminator because
   it aggregates IC, stability, and regime breadth — dimensions that noise cannot
   simultaneously inflate even with 584 candidates. This switch is expected: the
   effective search size dropped 4.5× and each surviving event has 3× the IS sample.

4. **Trade-off:** higher `min_tpm` lowers lift (frequent events are less selective),
   so lift stops being the discriminator — you trade "extreme but fragile" for
   "modest but confirmable". On 1D with little data this is the right trade; on 1H
   you lose less. The pipeline core is not weak — the **default `min_tpm` is simply
   not scaled for low-frequency data** (ties into the frequency-aware defaults of
   Fix #1). Because `min_tpm` is trades-per-*month*, a fixed value yields a roughly
   constant *absolute* activation count across timeframes — exactly what sample-size
   robustness needs.

Reproduce: `FORGEDGE_MIN_TPM=3.0 python examples/search_rotation_calibration.py 100`

## Caveats

- The wrap seam corrupts ~h forward returns per draw; with a random offset this
  adds noise to the null max and, if anything, *inflates* it (conservative).
- Single asset, single IS window. A close-derived event whose `source_feature`
  is the raw `close` column would partially leak in the IC term only (a minor,
  weight-0.20 component); lift/z/t/OOS are driven by the cached real activations
  vs the rotated outcome and are properly nulled.
- With min_tpm=3.0 the composite separates at p=0.030; this is credible at K=100
  but borderline. The survivor's OOS confirmation (p=0.000) is the stronger signal.

## Tippett (min-p) combination — honest adaptive statistic selection

The natural next question is: can we let the data pick the best discriminating
statistic without cherry-picking? Yes, via Tippett's combination:

1. For each of the 5 yardsticks, compute the empirical p-value of the real
   maximum against its own null rotation distribution: `p_s`.
2. `min_p_real = min_s p_s` — the Tippett statistic.
3. For each null draw j, compute `p_j[s]` (empirical rank of `null[s][j]`
   within its null array) and take `min_p_j = min_s p_j[s]`.
4. Empirical Tippett p = `(1 + #{min_p_null ≤ min_p_real}) / (1+K)`.

This pays the price of having looked at several yardsticks: if statistics are
correlated (they are — all driven by the same underlying edge), the correction
is mild; if they were independent it would be Bonferroni-like. No threshold on
n_act, no prior choice of discriminant — the data drives.

### Two-stage architecture: OOS held out of selection

Selection and confirmation must run on **disjoint data partitions**:

| stage | data | role | mechanism |
|---|---|---|---|
| 1 — Tippett | in-sample | **selection** | data picks among `composite, abs_z, is_lift, is_t` vs rotation null |
| 2 — OOS | out-of-sample | **confirmation** | fixed gate, applied **once** to the stage-1 survivor |

`oos_t` is therefore **excluded** from the Tippett combination. Using OOS to
choose the discriminant would be model-selection on the holdout — and a holdout
you have selected on is no longer a holdout. Stage 1 picks *which* in-sample
yardstick discriminates; stage 2 is an independent gate, not a choice. The
overall false-positive rate multiplies (≈ p_Tippett × p_OOS).

**K=100, min_tpm=3.0 results (Tippett over in-sample yardsticks only):**

| | value |
|---|---|
| Per-statistic p-values | composite: 0.030, is_t: 0.069, is_lift: 0.624, abs_z: 0.693 |
| Tippett min-p (real) | 0.0297 (driven by composite) |
| Null min-p mean | 0.322   null q05 = 0.039 |
| **Tippett empirical p** | **0.059** (5/100 null draws produced min-p ≤ 0.030) |

The Tippett correction moves the composite p from 0.030 to 0.059 — the cost of
looking across the in-sample yardsticks. Borderline at 0.05, but this is only
the search-level test. The survivor also passes the independent OOS (p=0.000,
n=18, WR=0.50) — a third evidence source now genuinely orthogonal to the
rotation null. The combined picture (borderline search-level test + OOS
confirmation + 29 IS activations) is defensible.

**Including vs excluding `oos_t` is numerically neutral here, architecturally not.**
Running the combination with all 5 statistics (oos_t mixed into selection) gives
the *same* Tippett p = 0.059. The null min-p *mean* drops (0.322 → 0.263, since
oos_t adds another chance at a small p), but the *tail* crossing the 0.030
threshold is unchanged — the null draws with a small oos_t p are the same draws
already crossing via composite/is_t (correlated: one spurious edge measured from
several angles). So oos_t brings no new competitors into the decisive tail.
Excluding it costs nothing on the p-value and buys a genuinely sealed OOS gate.

**Design recommendation:** use the two-stage Tippett-over-in-sample + sealed-OOS
design in the v2 calibrator instead of hardcoding which statistic to use. It is
implementable on top of the same K rotation draws at zero extra cost.
