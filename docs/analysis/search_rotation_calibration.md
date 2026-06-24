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

## Upstream fix: raise `min_tpm` instead of filtering downstream (K=40)

The 2 survivors above had only **9 IS activations** and **failed OOS** — elegant
false positives. The right lever is to raise Event Discovery's `min_tpm` so events
must fire more often. `mean_tpm = n_act / n_total_months`, so `min_tpm=3` over 12 IS
months floors activations at ~36.

| | min_tpm=1.5 | min_tpm=3.0 |
|---|---|---|
| candidates mined | 2621 | **584** |
| directed contracts | 111 | **16** |
| best real lift | 0.707 | 0.287 |
| separating statistic | lift (p=0.0099) | composite (p=0.049) |
| survivors above null | 2 | 1 |
| survivor IS activations | 9, 9 | **29** |
| survivor OOS | **failed** (WR 0.78→0.40) | **passed** (p=0.000, OOS WR 0.50) |

Key points:

1. **`min_tpm` filters during mining, not after.** 584 candidates vs 2621 means a
   smaller "best-of-N" search, so the rotation null bar itself drops (null is_t
   mean 5.39 → 4.82; null composite q95 0.979 → 0.964). A *downstream*
   `min_is_activations` gate cannot do this — the multiple-testing damage is
   already done by the time 2621 candidates exist. The upstream lever is
   structurally better.

2. **The single survivor confirms OOS.** `pr_diffnorm_close_vol12_vol24_48 < 0.167`
   (short, h=10, 29 activations) is the only alpha in this study that is *both*
   above the rotation null *and* OOS-confirmed (p=0.000). The min_tpm=1.5 survivors
   were not.

3. **Trade-off:** higher `min_tpm` lowers lift (frequent events are less
   selective), so lift stops being the discriminator — you trade "extreme but
   fragile" for "modest but confirmable". On 1D with little data this is the right
   trade; on 1H you lose less. The pipeline core is not weak — the **default
   `min_tpm` is simply not scaled for low-frequency data** (ties into the
   frequency-aware defaults of Fix #1). Because `min_tpm` is trades-per-*month*, a
   fixed value yields a roughly constant *absolute* activation count across
   timeframes — exactly what sample-size robustness needs.

Reproduce: `FORGEDGE_MIN_TPM=3.0 python examples/search_rotation_calibration.py 40`

## Caveats

- p = 0.024 at K = 40 is encouraging, not overwhelming; K ≈ 100 would tighten the
  quantile. The 2 survivors deserve individual scrutiny.
- The wrap seam corrupts ~h forward returns per draw; with a random offset this
  adds noise to the null max and, if anything, *inflates* it (conservative).
- Single asset, single IS window. A close-derived event whose `source_feature`
  is the raw `close` column would partially leak in the IC term only (a minor,
  weight-0.20 component); lift/z/t/OOS are driven by the cached real activations
  vs the rotated outcome and are properly nulled.
