# Holdout validation of newly integrated indicators on 1D tickers

**Trigger:** commits `dba2d7d`/`9c299dd` added CCI, WILLR, Stochastic %K/%D, WMA,
TRIMA, ADX, Aroon and A/D to the KPI Builder and `FeatureGenerator`.
**Question:** with those indicators enabled, does the end-to-end pipeline extract
rules on real 1D tickers that hold up on data the pipeline never saw?
**Method:** for every 1DAY ticker under `examples/data/`, censor the last 6 months
as holdout, run `forge()` only on the training window, select 2-5 promoted
rules, freeze each rule's event expression and operational parameters, then
replay it purely on the untouched holdout window. Reproduce with:

```bash
python examples/holdout_1d_new_indicators.py --out report.json
```

## Method detail

1. **Loader.** 9 tickers: `AMZN_1D`, `BTCEUR_1DAY`, `COPPER.CMDUSD_1DAY`,
   `ETHEUR_1DAY`, `EURUSD_1DAY`, `GBPUSD_1DAY`, `E_Brent_1DAY`, `E_DAAX_1DAY`,
   `E_SandP-500_1DAY` — normalized to `open_dt`/`open`/`high`/`low`/`close`/`volume`,
   ascending.
2. **KPI Table.** `build_features()` with the default config plus the 8 new
   indicators explicitly enabled, then `candle_features()`.
3. **Censoring.** `cutoff = last_bar_date - 6 months`; `kpi_train = kpi[open_dt <
   cutoff]` is the only data `forge()` ever sees. The holdout bars from `cutoff`
   onward are held out of every discovery/selection step (Event Discovery,
   Alpha Discovery, Rule Discovery, and the preset's own internal walk-forward
   all operate only on `kpi_train`).
4. **Pipeline run.** `forge(kpi_train, timeframe="1D", two_pass_composition=False,
   ...)` with a preset fallback `balanced → burst → sweep`, advancing whenever a
   preset either raises on `config_report` (structurally incoherent for this
   history — see the forgedge skill's pitfall #8) or promotes zero edges.
   `two_pass_composition=False`: the default grade-guided second pass is O(n²) in
   the 1D candidate pool and OOM-killed (>13.9 GB RSS) on a KPI table this wide;
   every preset already pins `max_and_components=1`, so this only forgoes
   composed-pair events, not single-condition ones.
5. **Selection.** Up to 5 `(contract, response)` pairs from `result.edges()`
   (`PARTIAL-EDGE`/`EDGE`), ranked by OOS net gain (falling back to in-sample net
   gain when a response carries no walk-forward).
6. **Frozen holdout replay.** For each selected rule: reconstruct its event on
   the **full** (train+holdout) KPI table via `EventCandidate.apply()` — which
   uses only the stored, immutable thresholds from training (invariant: event
   thresholds are fixed once Event Discovery selects them) — then
   `run_backtest()` with the rule's frozen `BacktestParams`, restricted to
   `timerange_from=cutoff` so only entries opened in the holdout window count
   (fills/exits may still reach past it, as usual for a backtest).
7. **Verdict.** `SURVIVES` if holdout `total_trades >= 3` and
   `profit_factor > 1.0` and `total_net_gain > 0`; `FAILS` if it has enough
   trades but doesn't clear that bar; `INCONCLUSIVE` if it fired fewer than 3
   times in the 6-month holdout (rare events on daily bars often do).

## Results

45 rules extracted (5 per ticker × 9 tickers, except where noted). All verdicts
from the pipeline itself were `PARTIAL-EDGE` — expected on 1D data per the
forgedge skill (`fast_null=True`'s rotation null caps a search-lottery winner at
`PARTIAL-EDGE`), not a sign of misconfiguration.

| Ticker | Train/Holdout bars | Preset used | Pipeline edges | Survives | Fails | Inconclusive |
|---|---|---|---|---|---|---|
| AMZN_1D | 1253 / 125 | balanced | 54 | 1 | 4 | 0 |
| BTCEUR_1DAY | 1046 / 185 | balanced | 130 | 2 | 3 | 0 |
| COPPER.CMDUSD_1DAY | 1605 / 158 | balanced | 20 | 4 | 0 | 1 |
| ETHEUR_1DAY | 1047 / 185 | balanced | 25 | 1 | 4 | 0 |
| EURUSD_1DAY | 1614 / 159 | sweep | 13 | 2 | 0 | 3 |
| GBPUSD_1DAY | 1613 / 159 | sweep | 62 | 5 | 0 | 0 |
| E_Brent_1DAY | 1316 / 132 | balanced | 260 | 0 | 5 | 0 |
| E_DAAX_1DAY | 1601 / 158 | balanced | 80 | 5 | 0 | 0 |
| E_SandP-500_1DAY | 1607 / 158 | balanced | 42 | 5 | 0 | 0 |
| **Total** | | | | **25** | **16** | **4** |

Two tickers needed the `sweep` fallback (`balanced`/`burst` promoted zero edges
on their training window) — EURUSD and GBPUSD, both FX pairs whose daily returns
are the hardest to find structural, repeatable events in among this set.

### Per-rule detail

Every selected rule, with its frozen event expression and both in-sample (IS —
the training window the pipeline searched) and holdout performance. `PF`
marked `9999*` is the zero-losing-trades sentinel, not a real ratio — see
Caveats. `net gain` is the **sum of per-trade returns**, not a compounded
equity curve (see Caveats) — read it as "cumulative edge over N trades", not
as a realistic portfolio return.

#### AMZN_1D

Bars total/train/holdout: 1378/1253/125 &nbsp;·&nbsp; range 2021-01-04 → 2026-06-30 &nbsp;·&nbsp; holdout from 2025-12-30 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 54

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `delta_ratio_aroon14_updown_3 < -0.3` | long | 93 | 82.8% | 2.9893 | 359.5% | 9 | 88.9% | 3.6153 | 36.8% | **SURVIVES** |
| 2 | `pr_ratio_close_ret03_ret06_48 < 0.145833` | long | 118 | 66.9% | 2.0339 | 239.5% | 12 | 50.0% | 0.4979 | -35.6% | **FAILS** |
| 3 | `pr_ratio_close_ret03_ret06_96 < 0.15625` | long | 129 | 70.5% | 1.9245 | 237.4% | 11 | 45.5% | 0.3732 | -44.5% | **FAILS** |
| 4 | `pr_ratio_close_ret03_ret06_48 < 0.104167` | long | 80 | 72.5% | 2.5448 | 184.4% | 9 | 44.4% | 0.3767 | -36.3% | **FAILS** |
| 5 | `delta_ratio_close_ret03_ret06_1 < -3.04372` | long | 83 | 74.7% | 2.3088 | 193.3% | 6 | 33.3% | 0.2365 | -37.3% | **FAILS** |

#### BTCEUR_1DAY

Bars total/train/holdout: 1231/1046/185 &nbsp;·&nbsp; range 2023-03-14 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 130

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `delta_ratio_close_mdd12_mdd24_3 > 0.118342` | long | 114 | 68.4% | 4.1695 | 466.3% | 18 | 27.8% | 0.3155 | -18.5% | **FAILS** |
| 2 | `zs_close_ret_03_48 > 1` | long | 117 | 64.1% | 3.3603 | 598.7% | 18 | 38.9% | 0.5559 | -14.4% | **FAILS** |
| 3 | `delta_diffnorm_close_mdd12_mdd24_1 > 0.0366718` | long | 102 | 63.7% | 4.5425 | 721.7% | 33 | 51.5% | 1.5895 | 47.7% | **SURVIVES** |
| 4 | `ratio_close_ret03_ret168 > 0.213578` | long | 96 | 77.1% | 7.5573 | 680.8% | 17 | 35.3% | 0.7766 | -20.0% | **FAILS** |
| 5 | `delta_close_rsi_25_3 > 4.63439` | long | 121 | 62.8% | 3.3387 | 646.8% | 14 | 50.0% | 1.4523 | 9.0% | **SURVIVES** |

#### COPPER.CMDUSD_1DAY

Bars total/train/holdout: 1763/1605/158 &nbsp;·&nbsp; range 2021-01-03 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 20

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `zs_diffnorm_volume_sma03_sma25_96 > 1` | long | 191 | 71.2% | 4.0851 | 917.2% | 7 | 100.0% | 9999* | 68.0% | **SURVIVES** |
| 2 | `zs_diffnorm_volume_sma03_sma12_96 > 1` | long | 204 | 68.1% | 3.3819 | 855.3% | 6 | 100.0% | 9999* | 60.2% | **SURVIVES** |
| 3 | `pr_volume_sma_03_48 > 0.895833` | long | 170 | 71.2% | 3.2311 | 699.4% | 2 | 100.0% | 9999* | 18.1% | **INCONCLUSIVE (too few holdout trades)** |
| 4 | `diffnorm_volume_sma03_sma25 > 0.912647` | long | 191 | 70.2% | 3.1586 | 571.5% | 32 | 78.1% | 8.5881 | 128.3% | **SURVIVES** |
| 5 | `delta_ratio_close_ret06_ret12_1 > 1.1031` | long | 177 | 70.1% | 2.2501 | 346.0% | 14 | 85.7% | 15.6553 | 47.5% | **SURVIVES** |

#### ETHEUR_1DAY

Bars total/train/holdout: 1232/1047/185 &nbsp;·&nbsp; range 2023-03-14 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 25

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `delta_ratio_aroon25_updown_6 < -0.736833` | long | 82 | 63.4% | 5.3723 | 491.5% | 18 | 72.2% | 3.4265 | 94.6% | **SURVIVES** |
| 2 | `pr_ratio_close_rsi14_rsi25_96 < 0.0786398` | long | 77 | 77.9% | 4.5090 | 313.0% | 18 | 5.6% | 0.0370 | -209.3% | **FAILS** |
| 3 | `pr_close_rsi_14_48 < 0.0625` | long | 61 | 78.7% | 4.9507 | 265.3% | 12 | 8.3% | 0.0628 | -112.3% | **FAILS** |
| 4 | `pr_ratio_close_rsi14_rsi25_96 < 0.114583` | long | 114 | 66.7% | 2.2279 | 276.0% | 22 | 4.5% | 0.0268 | -271.8% | **FAILS** |
| 5 | `zs_diffnorm_close_sma03_sma12_48 < -1` | long | 116 | 69.8% | 3.4991 | 342.1% | 27 | 37.0% | 0.2705 | -118.1% | **FAILS** |

#### EURUSD_1DAY

Bars total/train/holdout: 1773/1614/159 &nbsp;·&nbsp; range 2021-01-03 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `sweep` &nbsp;·&nbsp; pipeline edges: 13

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `pr_diffnorm_close_vol05_vol96_48 < 0.0416667` | short | 33 | 75.8% | 4.4620 | 16.9% | 2 | 0.0% | 0.0000 | -2.2% | **INCONCLUSIVE (too few holdout trades)** |
| 2 | `pr_ratio_close_vol05_vol96_48 < 0.0416667` | short | 31 | 71.0% | 3.9870 | 13.9% | 1 | 0.0% | 0.0000 | -0.0% | **INCONCLUSIVE (too few holdout trades)** |
| 3 | `delta_diffnorm_close_ret03_ret96_6 > 0.64239` | long | 31 | 100.0% | 9999* | 10.2% | 0 | — | 0.0000 | 0.0% | **INCONCLUSIVE (too few holdout trades)** |
| 4 | `pr_upper_wick_96 > 0.96875` | short | 41 | 70.7% | 2.5874 | 27.6% | 6 | 66.7% | 1.1418 | 0.2% | **SURVIVES** |
| 5 | `pr_diffnorm_gap_upper_wick_96 < 0.0414352` | short | 42 | 71.4% | 2.6919 | 29.4% | 6 | 66.7% | 1.1418 | 0.2% | **SURVIVES** |

#### GBPUSD_1DAY

Bars total/train/holdout: 1772/1613/159 &nbsp;·&nbsp; range 2021-01-03 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `sweep` &nbsp;·&nbsp; pipeline edges: 62

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `delta_diffnorm_volume_sma12_sma96_6 > 0.573894` | long | 113 | 77.9% | 6.0402 | 272.8% | 17 | 100.0% | 9999* | 22.6% | **SURVIVES** |
| 2 | `delta_diffnorm_volume_sma12_sma96_3 > 0.339094` | long | 114 | 71.1% | 4.1461 | 222.3% | 17 | 100.0% | 9999* | 20.5% | **SURVIVES** |
| 3 | `delta_diffnorm_volume_sma03_sma96_12 > 0.986424` | long | 114 | 69.3% | 3.8677 | 218.8% | 16 | 100.0% | 9999* | 18.2% | **SURVIVES** |
| 4 | `delta_diffnorm_volume_sma12_sma96_3 > 0.468172` | long | 71 | 80.3% | 7.4291 | 200.7% | 13 | 100.0% | 9999* | 16.2% | **SURVIVES** |
| 5 | `delta_diffnorm_volume_sma03_sma96_12 > 1.38408` | long | 72 | 81.9% | 8.4348 | 211.2% | 12 | 100.0% | 9999* | 15.5% | **SURVIVES** |

#### E_Brent_1DAY

Bars total/train/holdout: 1448/1316/132 &nbsp;·&nbsp; range 2021-01-04 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 260

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ratio_low_open_lag3 < 0.949797` | long | 172 | 64.0% | 2.4221 | 314.9% | 40 | 55.0% | 0.8079 | -24.5% | **FAILS** |
| 2 | `spread_low_open_lag3 < -0.0502034` | long | 172 | 64.0% | 2.4221 | 314.9% | 40 | 55.0% | 0.8079 | -24.5% | **FAILS** |
| 3 | `delta_ratio_low_ema03_ema96_1 < -0.012564` | long | 116 | 68.1% | 2.7787 | 287.5% | 38 | 52.6% | 0.6252 | -82.6% | **FAILS** |
| 4 | `delta_diffnorm_low_ema03_ema168_1 < -0.0998195` | long | 172 | 62.2% | 1.8140 | 251.0% | 39 | 53.8% | 0.5186 | -116.3% | **FAILS** |
| 5 | `ratio_low_ema_03_open_lag3 < 0.957456` | long | 173 | 61.3% | 1.8901 | 248.0% | 42 | 50.0% | 0.6173 | -62.5% | **FAILS** |

Rules 1 and 2 above are the same underlying pattern (`ratio_low_open_lag3` and
`spread_low_open_lag3` are algebraically equivalent transforms of the same
`low` vs. lagged-`open` comparison), which is why their stats are identical.

#### E_DAAX_1DAY

Bars total/train/holdout: 1759/1601/158 &nbsp;·&nbsp; range 2021-01-03 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 80

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `zs_spread_close_ema25_48 < -1` | long | 279 | 81.0% | 5.9281 | 817.1% | 25 | 92.0% | 31.0942 | 108.8% | **SURVIVES** |
| 2 | `zs_close_rsi_14_48 < -1` | long | 270 | 82.6% | 6.5798 | 782.6% | 25 | 92.0% | 30.0625 | 105.0% | **SURVIVES** |
| 3 | `zs_diffnorm_volume_sma03_sma25_48 > 1` | long | 199 | 86.4% | 4.8094 | 593.5% | 15 | 100.0% | 9999* | 70.7% | **SURVIVES** |
| 4 | `pr_spread_close_ema12_48 < 0.125` | long | 166 | 80.7% | 5.7051 | 470.6% | 17 | 88.2% | 20.7369 | 71.3% | **SURVIVES** |
| 5 | `pr_diffnorm_volume_sma09_sma12_48 > 0.875` | long | 155 | 85.2% | 3.3712 | 357.3% | 14 | 100.0% | 9999* | 56.4% | **SURVIVES** |

#### E_SandP-500_1DAY

Bars total/train/holdout: 1765/1607/158 &nbsp;·&nbsp; range 2021-01-03 → 2026-09-02 &nbsp;·&nbsp; holdout from 2026-03-02 &nbsp;·&nbsp; preset `balanced` &nbsp;·&nbsp; pipeline edges: 42

| # | Event expression | Dir | IS trades | IS WR | IS PF | IS net gain | Holdout trades | Holdout WR | Holdout PF | Holdout net gain | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `pr_close_ret_03_96 < 0.144048` | long | 169 | 78.1% | 4.3409 | 713.6% | 11 | 100.0% | 9999* | 83.0% | **SURVIVES** |
| 2 | `pr_ratio_close_pos_upper_wick_48 < 0.166667` | long | 165 | 80.6% | 4.0455 | 638.1% | 11 | 100.0% | 9999* | 79.8% | **SURVIVES** |
| 3 | `zs_close_ret_03_96 < -1` | long | 183 | 79.2% | 3.9875 | 666.0% | 12 | 100.0% | 9999* | 77.5% | **SURVIVES** |
| 4 | `delta_close_ad_14_1 < -0.0413935` | long | 182 | 77.5% | 3.5795 | 485.1% | 19 | 100.0% | 9999* | 94.4% | **SURVIVES** |
| 5 | `ratio_close_ret01_ret12 > 0.589114` | long | 182 | 76.9% | 3.8296 | 495.0% | 11 | 90.9% | 15.4587 | 36.6% | **SURVIVES** |

### Rules that use the newly integrated indicators directly

- `AMZN_1D`: `delta_ratio_aroon14_updown_3 < -0.3` (Aroon up/down spread) —
  **SURVIVES** holdout (9 trades, PF 3.62, net gain +36.8%).
- `ETHEUR_1DAY`: `delta_ratio_aroon25_updown_6 < -0.736833` (Aroon) —
  **SURVIVES** holdout (18 trades, PF 3.43, net gain +94.6%).
- `E_SandP-500_1DAY`: `delta_close_ad_14_1 < -0.0413935` (Accumulation/
  Distribution) — **SURVIVES** holdout (19 trades, PF ≫1, net gain +94.4%).

The rest of the surviving rules lean on pre-existing families (RSI, EMA/SMA
ratios, z-scores/percentile-ranks of returns and volatility, volume SMA
ratios, candle geometry) — the new indicators contributed real, holdout-
confirmed rules on 3 of 9 tickers, without dominating the pool (expected: they
compete against a much larger set of pre-existing families for the same
promotion gates).

### Rules that failed holdout (selection worked as a filter, not a rubber stamp)

`E_Brent_1DAY` is the clean negative control: all 5 selected rules had strong
in-sample profit factors (1.8–2.8) but every one reversed sign in holdout
(PF 0.52–0.81, net gain −25% to −116%) — the training-window pattern did not
carry forward. `ETHEUR_1DAY` and `AMZN_1D` show the same mix: a handful of
rules survive, most don't, which is the expected shape of a real out-of-sample
filter rather than a pipeline that always says yes.

## Caveats

- **`profit_factor = 9999.0` is a sentinel, not a real ratio.** `run_backtest()`
  caps profit factor at 9999.0 when a rule has zero losing trades in the
  window being measured (`backtest.py:491`) to avoid division by zero. Several
  holdout windows here are short enough (11-25 trades) that a genuinely decent
  rule can go a full 6 months without a loser by chance — treat every `9999.0`
  as "no losers observed", not as an extreme edge. `total_net_gain` and
  `win_rate_pct` are the sturdier numbers in those rows.
- **`net gain` is a plain sum of per-trade returns, not a compounded equity
  curve** (`BacktestSummary.total_net_gain = net.sum()`, `backtest.py:569`).
  `run_backtest()` opens a position on every active bar with no flat-state
  check, so the IS figures here (up to +917%) reflect 100+ overlapping,
  independently-sized trades added together, not a single account compounding
  — they measure the strength/consistency of the edge per trade, not an
  achievable portfolio return. The holdout figures (max ~130%, over far fewer
  trades) are directionally the same but should be read the same way.
- **Holdout sample sizes are small.** 6 months of daily bars is ~125-185 rows;
  a rule that fires a few times a month produces single-digit-to-low-double-
  digit holdout trades. 4 of 45 rules (COPPER ×1, EURUSD ×3) fired fewer than
  3 times and are marked `INCONCLUSIVE` rather than forced into a verdict —
  a longer holdout (or a higher-frequency timeframe) would sharpen this.
- **`two_pass_composition=False`** means every extracted rule here is a single
  boolean condition, not an AND-composition. This was a memory-budget
  necessity on this container (see Method step 4), not evidence composed
  rules wouldn't hold up — that's a natural follow-up once run on a host with
  more headroom (or a narrower KPI config).
- Ranking by OOS/in-sample net gain (Method step 5) tends to surface several
  near-duplicate rules on the same underlying signal when a ticker's pool is
  dominated by one family (e.g. GBPUSD's five picks are all `volume_sma`
  ratio variants at different lags) — a `max_constituent_jaccard`-style
  de-duplication pass ahead of ranking would diversify the picks.

## Reproduce / extend

`examples/holdout_1d_new_indicators.py --tickers <file.csv> --holdout-months 6
--out report.json` reruns a subset; the full JSON per-ticker/per-rule detail
(pipeline in-sample stats + holdout stats for every selected rule) is produced
the same way for the whole set.
