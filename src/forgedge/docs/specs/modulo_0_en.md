# Module 0 — Market Context

Market Context is the first module in the FORGE pipeline. Its job is to
classify every bar of the KPI Table into a market regime, adding two columns —
`regime` and `regime_stable` — that remain immutable for the entire session.
All downstream modules read these columns but never modify them.

---

## Basic usage

```python
from forgedge import MarketContext
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")   # must contain a 'close' column
mc = MarketContext(kpi)
enriched = mc.run()

# The enriched table has the same columns as kpi, plus 'regime' and 'regime_stable'
print(enriched[["close", "regime", "regime_stable"]].tail(10))
print(mc.distribution())
```

`run()` does not mutate the original DataFrame — it always returns a copy.

---

## Output: added columns

### `regime` — ordered categorical

Every bar receives a regime label chosen from five values ordered from most
bearish to most bullish:

```
STRONG_BEAR  <  BEAR  <  NEUTRAL  <  BULL  <  STRONG_BULL
```

The pandas dtype is `Categorical` with `ordered=True`, so ordinal comparisons
(`regime > "BEAR"`) and groupby operations work directly.

### `regime_stable` — boolean

`True` when the bar's regime has remained unchanged for at least `stable_window`
consecutive bars (counting the current bar).
Used to exclude transition bars from regime sensitivity analyses.

Example: with `stable_window=12`, the first 11 bars of each new regime have
`regime_stable=False`. Only from the 12th consecutive bar onward is the bar
considered "stable".

Bars with a `NaN` regime (leading NaN from EMA warmup) always have
`regime_stable=False`.

---

## How classification works (EMAProxyClassifier)

The default classifier computes the ratio between a fast EMA and a slow EMA:

```
ratio = ema_short / ema_long
```

and discretises it into five regimes using four thresholds. The ratio is a proxy
for the price's distance from its local mean-reversion level: a high ratio means
price is above the short-term mean (uptrend), a low ratio means below (downtrend).

### Looking up EMA columns in the KPI Table

The classifier first looks for EMA columns in the KPI Table using the FORGE
naming convention:

```
{source_col}_ema_{period:02d}   →   e.g. "close_ema_09", "close_ema_25"
```

If the column is present (as in the QHF framework's `CandleKPI` which
pre-computes them), it is used directly. If absent, the EMA is computed inline
with `ewm(span=period, adjust=False)` and the result is **not** written to the table.

### Thresholds and mapping in `"fixed"` mode (default)

| Regime | Condition on ratio |
|---|---|
| `STRONG_BEAR` | ratio < 0.975 |
| `BEAR` | 0.975 ≤ ratio < 0.990 |
| `NEUTRAL` | 0.990 ≤ ratio < 1.010 |
| `BULL` | 1.010 ≤ ratio < 1.025 |
| `STRONG_BULL` | ratio ≥ 1.025 |

The values 0.975/0.990/1.010/1.025 mean that the EMAs must diverge by at least
±1% (BEAR/BULL) or ±2.5% (STRONG) to move out of NEUTRAL.

---

## Automatic EMA window derivation

With `auto_window=True` (default), FORGE derives EMA windows from the data
rather than using fixed values, via a Hurst/Ornstein-Uhlenbeck analysis.

### Why derive windows from data?

The slow EMA should have a span ≈ the local mean-reversion half-life of the
price, so the `ema_short / ema_long` ratio genuinely captures the distance from
the equilibrium level. Using fixed spans (9/25) on all assets is an approximation:
crypto with fast mean-reversion needs shorter spans; trending assets need longer ones.

### Algorithm

1. On the price series (`source_col`, default `close`), local half-lives are
   estimated via discrete OU regression on rolling windows:
   ```
   dP_t = const + kappa * P_{t-1} + ε    [numpy.linalg.lstsq]
   half-life = -log(2) / log(1 + kappa)  [valid only when kappa < 0]
   ```
2. The median of converging half-lives is computed.
3. `long_period = round(hl)`, `short_period = round(hl * fast_ratio)` (default `1/2.3`).
4. If at least `min_window_estimates` (default 10) estimates converge, the result
   is valid; otherwise it falls back to the configured periods.

### Estimation window size (`window_unit`)

The estimation window can be expressed in **days** (`window_unit="day"`, default)
or **bars** (`window_unit="bar"`).

- **`"day"` mode** (recommended): a 168-day window corresponds to the same
  amount of data on any timeframe. On 1H = 168 × 24 = 4032 bars, on 4H =
  168 × 6 = 1008 bars. Requires a DatetimeIndex or a datetime column; if neither
  is available, set `bar_hours` explicitly.
- **`"bar"` mode**: always 168 bars, regardless of timeframe. Convenient when
  the DataFrame has no time information.

### Resolution source

After `run()`, `mc.window_resolution` reports how spans were chosen:

| `source` | Meaning |
|---|---|
| `"hurst_ou"` | Converging half-life; `short_period` and `long_period` derived from data |
| `"fallback"` | `auto_window=True` but OU did not converge; configured `short_period`/`long_period` used |
| `"configured"` | `auto_window=False`; configured spans used directly |

```python
mc.run()
print(mc.window_resolution)
# {'source': 'hurst_ou', 'short_period': 9, 'long_period': 22,
#  'half_life_bars': 21.4, 'n_estimates': 47, 'unit': 'day', ...}
```

---

## Threshold modes

### `threshold_mode="fixed"` (default)

Absolute thresholds on the ratio. Each bar's regime is determined by the
point value of the ratio at that bar. The default values `[0.975, 0.990, 1.010, 1.025]`
are empirically calibrated on crypto 1H data.

When to use: standard production, when a regime should have absolute meaning
(e.g. "fast EMA is at least 2.5% above the slow EMA").

### `threshold_mode="balanced"`

Thresholds are computed as quantiles of the ratio to match the regime
distribution to `target_distribution` (default: bell with 10% tails —
`[0.10, 0.20, 0.40, 0.20, 0.10]`).

When to use: when a pre-determined regime frequency is needed (e.g. to ensure
balanced samples in Module 2's regime sensitivity analysis).

**Note:** in balanced mode, thresholds lose their absolute meaning (STRONG_BULL
no longer guarantees a "+2.5% divergence").

#### `threshold_basis` in balanced mode

- **`"global"`** (default): quantiles computed once over the entire sample.
  Achieves the target distribution exactly but is not causal (past labels depend
  on future data). Appropriate for in-sample analysis.
- **`"expanding"`**: quantiles computed over `[0..t]` for each bar `t`.
  Causal (no look-ahead), but the target distribution is only approximate.
  The first `threshold_warmup` bars (default 200) use fixed thresholds as a
  fallback while sufficient history accumulates.

```python
from forgedge.market_context.models import EMAProxyConfig, MarketContextConfig
from forgedge import MarketContext

config = MarketContextConfig(
    ema_proxy=EMAProxyConfig(
        threshold_mode="balanced",
        threshold_basis="expanding",
        target_distribution=[0.15, 0.20, 0.30, 0.20, 0.15],
    )
)
mc = MarketContext(kpi, config=config)
enriched = mc.run()
print(mc.distribution())          # frequencies close to the target
```

---

## Output methods

### `mc.distribution() → pd.DataFrame`

Returns the bar count and share per regime:

```python
print(mc.distribution())
#              n_bars  share
# STRONG_BEAR     450   0.21
# BEAR            386   0.18
# NEUTRAL         536   0.25
# BULL            365   0.17
# STRONG_BULL     386   0.18
```

### `mc.regime_table(timestamp_col=None) → pd.DataFrame`

Returns a compact `[timestamp, regime, regime_stable]` frame useful for
joining regime information onto external DataFrames without bringing all columns.

```python
regime_df = mc.regime_table()
# Join regime onto original table
merged = original_df.merge(regime_df, on="open_dt", how="left")
```

### `mc.get_config() → dict`

Returns the full configuration used, including the effective EMA window values
and thresholds (useful for traceability and reproducibility):

```python
cfg = mc.get_config()
print(cfg["window_resolution"])
print(cfg["classifier"]["resolved_thresholds"])
```

---

## Full configuration reference

### `EMAProxyConfig`

| Parameter | Default | Description |
|---|---|---|
| `source_col` | `"close"` | OHLCV column used to compute the EMAs |
| `auto_window` | `True` | Derive EMA windows from Hurst/OU analysis |
| `short_period` | `9` | Fast EMA span (used as fallback if auto does not converge, or if `auto_window=False`) |
| `long_period` | `25` | Slow EMA span (same) |
| `thresholds` | `[0.975, 0.990, 1.010, 1.025]` | Fixed cut points for the ratio (must be strictly ascending, length = n_labels - 1) |
| `threshold_mode` | `"fixed"` | `"fixed"` or `"balanced"` |
| `target_distribution` | `[0.10, 0.20, 0.40, 0.20, 0.10]` | Target distribution for balanced mode (relative weights, normalised internally) |
| `threshold_basis` | `"global"` | `"global"` or `"expanding"` (balanced mode only) |
| `threshold_warmup` | `200` | Leading bars that use fixed thresholds in expanding mode |
| `window_unit` | `"day"` | Unit for OU estimation window/stride (`"day"` or `"bar"`) |
| `window_estimation` | `168.0` | Width of the OU estimation window |
| `window_stride` | `1.0` | Step between successive estimates |
| `bar_hours` | `None` | Candle duration in hours (explicit override; if None, inferred from DatetimeIndex) |
| `fast_ratio` | `1/2.3` | Fast/slow span ratio for auto-derivation |
| `min_window_estimates` | `10` | Minimum converging OU estimates required to trust the derivation |

### `MarketContextConfig`

| Parameter | Default | Description |
|---|---|---|
| `classifier` | `"ema_proxy"` | Classifier implementation (only `"ema_proxy"` available in v0.1.0) |
| `ema_proxy` | `EMAProxyConfig()` | Parameters for `EMAProxyClassifier` |
| `labels` | `DEFAULT_LABELS` | Ordered regime labels (most bearish to most bullish) |
| `stable_window` | `12` | Consecutive bars required for `regime_stable=True` |

---

## Extending the classifier

The classifier is built on an ABC interface (`RegimeClassifier`) that allows
replacing `EMAProxyClassifier` with any alternative implementation (HMM, KMeans,
custom) without touching downstream modules.

```python
from forgedge.market_context.models import RegimeClassifier
import pandas as pd

class MyClassifier(RegimeClassifier):
    def __init__(self, labels):
        self.labels = labels

    def classify(self, kpi_table: pd.DataFrame) -> pd.Series:
        # ... custom logic ...
        return pd.Series(...)   # ordered categorical labels

    def get_labels(self):
        return self.labels

    def get_config(self):
        return {"classifier": "my_classifier"}

# Pass the classifier directly (bypasses config.classifier)
mc = MarketContext(kpi, classifier=MyClassifier(["BEAR", "NEUTRAL", "BULL"]))
enriched = mc.run()
```

The `build_classifier(config)` function in `context.py` is the single point that
maps the string name to a concrete implementation. To register a new classifier
permanently, it is sufficient to add a case to that function.

---

## Offline EMA window analysis

The `suggest_ema_windows` function in `hurst.py` allows offline analysis of
optimal windows for an asset before configuring the module:

```python
from forgedge.market_context.hurst import suggest_ema_windows

result = suggest_ema_windows(kpi["close"], timeframe="1h")
print(result)
# {
#   "half_life_candles": 21.4,
#   "half_life_hours": 21.4,
#   "suggested_short_period": 9,
#   "suggested_long_period": 21,
#   "n_estimates": 47,
#   "hurst_median": 0.389
# }
```

A `hurst_median < 0.5` confirms the series is mean-reverting over the estimation
window: the EMA ratio is a sensible regime proxy.

---

## Downstream usage

**Event Discovery (Module 1):** ignores the regime during discovery.
The `regime` column is available in the DataFrame but is not read during Steps 0–5.

**Alpha Discovery (Module 2):** reads `regime` and `regime_stable` to stratify
IC and win rate by regime (Step 5 — Regime Sensitivity Analysis).
If `use_stable_regime_only=True`, only bars with `regime_stable=True` are included
in the stratification.

The canonical way to pass the enriched table to downstream modules is:

```python
enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)      # regime is already in the table
candidates = ed.run()

ad = AlphaDiscovery(ed.df, candidates, config)   # ed.df has regime + derived features
```
