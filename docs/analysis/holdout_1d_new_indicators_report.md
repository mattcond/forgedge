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
