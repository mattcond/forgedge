# Does transforming M1's feature space with PCA help? — experiment

**Dataset:** `examples/data/AMZN_1D.csv` — 1378 daily bars, 2021-01-04 → 2026-06-30.
**Question:** does re-encoding the columns Event Discovery (M1) mines events from —
replacing the original technical indicators with their principal components —
improve the pipeline's downstream results?
**Method:** run the identical `forge()` configuration twice on the same asset/timeframe,
changing only the feature space M1 sees. Reproduce with:

```bash
python examples/pca_feature_space_experiment.py examples/data/AMZN_1D.csv 12
```

## Setup

1. Build the KPI Table with the standard recipe (`build_features` default config →
   `candle_features` → `lag_features`), 179 columns, 171 of them numeric
   feature columns (SMA/EMA/RSI/Bollinger/returns/drawdowns/candle geometry/lags).
2. **Iteration 1 (baseline):** `KPI -> forge()`.
3. **Iteration 2 (PCA):** all 171 feature columns are replaced by their first 12
   principal components (SVD on z-scored data, fit on the 900/1378 complete-case
   rows — the rest are NaN at the start of the series due to rolling-window
   lookback, same as the raw indicators). 12 components explain 97.6% of variance
   (PC1 alone 81.2% — expected, since most of these indicators are highly
   collinear price-scale transforms of `close`). OHLCV/`open_dt`/`color` are left
   untouched — those aren't "M1 features" in this sense, they feed Market
   Context and the backtest directly.
4. Both iterations use **the exact same** `forge_preset("balanced", timeframe="1D",
   asset="AMZN")` triple (`DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig`) —
   nothing about the search criteria changes between runs, only the columns M1
   is allowed to build events from.

Methodological note: the PCA fit here is global (fit once on the full history),
not walk-forward/rolling — a mild look-ahead relative to the pipeline's own
internal IS/OOS splits. That leakage, if anything, biases the comparison *in
favour of* the PCA iteration, not against it.

## Result

| | M1 raw candidates | M1 post-gate | M2 contracts | M2 promoted | M3 EDGE/PARTIAL-EDGE |
|---|---:|---:|---:|---:|---:|
| **Baseline** (raw indicator space) | 46 960 | 6 021 (12.8%) | 6 721 | ~1 065¹ | **94** (all PARTIAL-EDGE) |
| **PCA** (12 components, 97.6% var.) | 2 676 | 481 (18.0%) | 981 | 70 | **0** |

Baseline M3 verdicts: `{PARTIAL-EDGE: 94, NON-EDGE: ~960, INSUFFICIENT-DATA: 15}`.
PCA M3 verdicts: `{NON-EDGE: 70}` — every one of the 70 promoted PCA contracts
failed Rule Discovery's realistic backtest; none reached even `PARTIAL-EDGE`.

Verified reproducible across two independent runs of
`examples/pca_feature_space_experiment.py`: raw/post-gate candidate counts,
M2 contract counts, and the **94 vs. 0** edge outcome are bit-identical run
to run. ¹Only the baseline's *promoted* count (1072 vs. 1063 across the two
runs, ~1%) varies — expected per the forgedge skill's pitfall #24
(`PYTHONHASHSEED`-dependent set-iteration order affects which candidate
survives a pool-truncating cap during AND composition); it does not change
which verdicts are reached or the 94/0 headline.

## Interpretation

PCA-transforming the feature space is not neutral — on this asset/config it is
sharply counterproductive along two separate axes:

1. **M1's raw candidate pool collapses ~17.5x (46 960 → 2 676).** Event
   Discovery's five dedicated arity-2 feature families (cross-column/cross-time
   OHLC pairs, indicator-vs-lagged-OHLC, MACD-vs-signal, price-vs-volume,
   candle-vs-`natr`) and the generic same-family `{base}_{indicator}_{period}`
   ratio pairing all key off indicator *names* and *semantics* (see pitfall #4
   in the forgedge skill file). A `pc_07` column carries none of that — it opts
   out of every family-based pairing the KPI Builder naming convention exists
   to enable, so M1 falls back to close to single-column threshold events on
   12 abstract axes instead of 171 named, economically legible ones.
2. **Every PCA-space contract that reaches M3 fails the realistic backtest.**
   The 70 promoted PCA contracts pass Alpha Discovery's statistical filter
   (predictive of the derived target in-sample/confirm) but none survive
   Rule Discovery's order-mechanics/walk-forward gate — a textbook sign of
   contracts that look predictive on a statistical target but don't correspond
   to a stable, exploitable market condition. A principal component is a
   linear combination across indicators of very different character (bounded
   oscillators like RSI, unbounded price-scale MAs, return/drawdown ratios,
   candle geometry) — its distributional percentile threshold has no
   equivalent to "RSI crossed 70" or "close broke above its 25-period EMA": it
   is a direction in an abstract rotated space, not a named technical
   condition, and evidently not one this pipeline's realistic-backtest gate
   finds durable here.

Both effects point the same way: FORGE's design leans on *interpretable,
named* indicator columns — both for the cross-column feature families that
give M1 most of its raw candidate volume, and (more fundamentally) for
producing conditions that hold up under M3's out-of-sample backtest. PCA
strips exactly the structure the pipeline is built to exploit.

## Caveats

- Single asset (AMZN), single history window, single preset (`"balanced"`),
  single component count (12, chosen for ~97% variance). Not a sweep — a
  different `n_components` or a rolling/walk-forward PCA fit (removing the
  look-ahead noted above, which if anything should only make the PCA
  iteration look *worse*, not better) could shift the exact counts but is
  unlikely to reverse the qualitative result given how it fails on two
  independent mechanisms (pairing eligibility *and* backtest survivability).
- This does not test partial/hybrid feature spaces (e.g. PCA only on one
  indicator family, keeping the rest named) — a plausible follow-up if the
  goal is dimensionality reduction without losing named-column pairing
  entirely.

## Conclusion

On this dataset and configuration, transforming M1's feature space via PCA
before `forge()` is a clear regression, not an improvement: it collapses the
raw candidate pool ~17x and turns 94 `PARTIAL-EDGE` results into 0. The
experiment does not support pursuing a PCA-transformed feature space for M1;
the pipeline's value comes substantially from working in the original,
named indicator space rather than an abstract rotated one.
