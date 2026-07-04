# FORGE on low-frequency (1D) data — robustness analysis

**Dataset:** `ADA_1D_FULL.parquet` — 901 daily bars, 2024-01-01 → 2026-06-19.
**Question:** is the pipeline reliable on daily data? Where are the functional weak points?
**Method:** code audit + a *surrogate-data null test*. Features are rebuilt from OHLC
with one recipe applied identically to the real close and to spectrum/return-matched
surrogates, then the pipeline is run on both. Reproduce with:

```bash
FORGEDGE_PARQUET=/path/ADA_1D_FULL.parquet python examples/lowfreq_null_diagnostic.py 8
FORGEDGE_PARQUET=/path/ADA_1D_FULL.parquet python examples/lowfreq_endpoint_diagnostic.py 3 12
```

## Headline empirical results

Training/discovery on the 2024 in-sample window (366 bars; Alpha's internal split
leaves ~256 train / ~110 confirm).

**1. Promoted-alpha count does not separate ADA from noise (it inverts).**

| | candidates | promoted alphas |
|---|---|---|
| REAL ADA | 2542 | **58** |
| Phase-randomized noise (n=5) | ~2755 | **146 ± 45** (101–207) |

Noise is promoted *more* often than the real asset. The promoted-alpha count is not
a usable signal at 1D.

**2. EDGE/PARTIAL verdicts (the tradeable output) separate, but weakly and with a high noise floor.**

| top-12 promoted → tradeable verdicts | result |
|---|---|
| REAL ADA | **12** (5 EDGE + 7 PARTIAL) |
| Phase-randomized noise — keeps autocorrelation (n=3) | 5.7 ± 3.7 (1–10) |
| IID-shuffled noise — destroys autocorrelation (n=3) | 2.3 ± 0.9 (1–3) |

Pure noise still earns ~2–3 EDGEs out of 12 (**~20 % per-rule false-positive floor**).
The phase-vs-IID gap (5.7 vs 2.3) shows much of what passes is **linear momentum already
present in the return spectrum**, not a discovered nonlinear edge.

**Diagnosis:** the statistical core (rotation null, BH-FDR, t-tests) is sound, but on
~1 year of daily bars every out-of-sample check rests on a tiny sample, so the
validation cannot reliably tell signal from noise. The weakness is *sample-size /
multiple-testing exposure*, not arithmetic.

## Root-cause weak points (code-level)

1. **`horizon_grid` is not frequency-scaled.** Default
   `(1,2,3,4,6,8,12,16,24,36,48)` (`alpha_discovery/models.py:455`) is hours. `forge(...,
   timeframe="1D")` (`forge.py:396`) uses it verbatim → holding periods up to **48 days**.
   Unlike MarketContext / Hurst / rolling-IC / bars-per-year, which *do* auto-scale, the
   horizon grid is a silent footgun.
   **[FIXED]** `forge()` with a default `alpha_config` now substitutes the
   daily-calibrated grid (`presets.default_horizon_grid`) on daily-or-slower
   timeframes, and warns when an explicit config keeps the hourly default grid
   on such a timeframe.

2. **No sample-size / multiple-testing guard.** Event Discovery mines ~2500 candidates
   from ~256 in-sample bars with no warning. BH-FDR controls expected false discoveries
   *given the p-values*, but nothing caps `n_candidates` relative to `n_bars` or calibrates
   promotion against a surrogate null.

3. **Alpha's internal IS/OOS confirmation split is regime-confounded at short length.**
   `train_ratio=0.70` on 12 months → ~3.5-month confirmation window. In a trending year
   (2024: 35 % STRONG_BEAR, 23 % STRONG_BULL) edges flip between the two adjacent windows,
   so real events fail confirmation while a stationary noise process passes — the likely
   mechanism behind result (1). No `min_confirm_months` / embargo guard exists.

4. **Walk-forward windows are too thin on 1D.** `_build_splits` (`rule_discovery/walkforward.py:50`)
   on 12 months with `min_train=4, n_splits=3` → ~2-month test windows ≈ a handful of
   trades each. The OOS PF/WR that gate the verdict are estimated on near-empty samples,
   which is why the surrogate EDGE count has such high variance (1–10).

## Recommended functional improvements (priority order)

1. **Auto-scale `horizon_grid` from the timeframe** (or warn when grid max × bar-hours is
   absurd, e.g. > ~30 days). Cheap, removes a silent footgun. *(low risk)*

2. **Surrogate-calibrated promotion / empirical FDR.** Run K phase-randomized surrogates
   through Alpha Discovery and promote only the excess over the surrogate count
   distribution. This auto-adapts to frequency and directly fixes result (1). The harness
   in `examples/lowfreq_null_diagnostic.py` is a working prototype. *(medium)*
   **[IMPLEMENTED]** `calibration.FastRotationNull` computes the exact
   search-level rotation null over every circular offset (FFT
   cross-correlation, ~1 s on this dataset — no K, no seed), runs by default
   inside `forge()`, and Rule Discovery now requires clearing it
   (`SelectionCriteria.max_rotation_p`) for a full EDGE verdict.  On this ADA
   dataset the search p is ≈ 0.70: every former EDGE is honestly capped at
   PARTIAL-EDGE.  The session's multiple-testing surface is recorded on
   `ForgeResult.ledger` (`forgedge.ledger.HypothesisLedger`).

3. **Minimum-sample guards with honest degradation.** Refuse/flag when IS bars per
   candidate, Alpha confirmation months, or trades-per-walk-forward-window fall below a
   floor — surface it in the verdict instead of emitting a confident EDGE. *(medium)*

4. **Purged + embargoed splits** for Alpha confirmation and the walk-forward, so adjacent
   windows don't leak through overlapping forward-return horizons. *(higher effort)*

## Caveats

- Feature rebuild reproduces 14/21 derived columns to ~1e-6; the EMA family differs by
  1–8 % (parquet EMA seeding) and `close_mdd_48` uses a different lookback. This shifts the
  *absolute* real baseline slightly (rebuilt real promotes 58 vs 111 on the exact parquet)
  but not the comparison: real and surrogate use the identical recipe.
- Phase randomization is a *conservative* null — it preserves linear autocorrelation, so
  it credits the pipeline with any genuine momentum structure. The IID shuffle brackets the
  strict overfitting floor.
