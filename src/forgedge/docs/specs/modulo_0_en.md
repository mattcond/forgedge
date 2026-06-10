# Module 0 — Market Context (Codebase Spec)

> **Code reference:** `src/forgedge/market_context/`
> **Functional analysis:** `docs/modules/MarketContext.md`
> **Status:** ✅ Implemented and aligned with the functional analysis.
> Some configuration options are richer than documented.

---

## 1. Position in the pipeline

Market Context is the first module to run.
It enriches the KPI Table with two columns — `regime` and `regime_stable` —
that remain immutable for the entire session.
No downstream module may modify them.

```
KPI Table (OHLCV + indicators)
        │
        ▼
  MarketContext.run()
        │
        ▼
KPI Table + 'regime' + 'regime_stable'   ──► Event Discovery
                                          ──► Alpha Discovery
                                          ──► (Rule Discovery — not implemented)
```

---

## 2. Public interface

### `MarketContext` (`context.py`)

```python
MarketContext(kpi_table, config=None, classifier=None)
```

| Method / Property | Description |
|---|---|
| `run() → pd.DataFrame` | Classifies every bar; returns a copy with `regime` + `regime_stable` |
| `get_config() → dict` | Full configuration used, including EMA window resolution details |
| `distribution() → pd.DataFrame` | Per-regime bar count and share (n_bars, share) |
| `regime_table(timestamp_col) → pd.DataFrame` | Compact `[timestamp, regime, regime_stable]` frame for external joins |
| `window_resolution` | Dict with source tag ("hurst_ou" / "fallback" / "configured") and values used |

`run()` does **not mutate** the input DataFrame — it returns a copy.
Any intermediate EMA indicators computed inline are **not** added to the table.

---

## 3. `RegimeClassifier` interface (`models.py`)

Every classifier implementation must satisfy this ABC:

```python
class RegimeClassifier(ABC):
    def classify(kpi_table: pd.DataFrame) → pd.Series   # ordered categorical labels
    def get_labels() → list[str]                          # most bearish to most bullish
    def get_config() → dict                               # for traceability in report
```

Default labels (ordered):
```
STRONG_BEAR | BEAR | NEUTRAL | BULL | STRONG_BULL
```

Output column name constants:
```python
REGIME_COL        = "regime"          # ordered categorical
REGIME_STABLE_COL = "regime_stable"   # bool
```

---

## 4. v1.0 implementation: `EMAProxyClassifier` (`ema_proxy.py`)

### 4.1 Classification logic

1. Retrieve or compute `ema_short` and `ema_long` from `source_col`.
2. Compute `ratio = ema_short / ema_long`.
3. Discretise the ratio into regime labels via the configured thresholds.

**EMA lookup by naming convention:**
```
{source_col}_ema_{period:02d}   →  e.g. "close_ema_09", "close_ema_25"
```
If the column is absent from the KPI Table, the EMA is computed inline
with `ewm(span=period, adjust=False)` and is **not** written back to the table.

### 4.2 Threshold mode (`threshold_mode`)

| Mode | Behaviour |
|---|---|
| `"fixed"` (default) | Absolute thresholds applied to the EMA ratio |
| `"balanced"` | Thresholds computed as quantiles of the ratio to match `target_distribution` |

In `"balanced"` mode, `threshold_basis` controls causality:

| Basis | Behaviour | Causal? |
|---|---|---|
| `"global"` (default) | Quantiles computed once over the entire sample | No (look-ahead) |
| `"expanding"` | Quantiles at bar t computed over history `[0..t]` | Yes |

In `"expanding"` mode, the first `threshold_warmup` bars use the fixed thresholds as
a fallback before sufficient history has accumulated.
If `"balanced"` mode produces non-strictly-increasing thresholds (degenerate ratio),
it automatically falls back to `"fixed"`.

### 4.3 Default thresholds and target distribution

```python
thresholds            = [0.975, 0.990, 1.010, 1.025]
target_distribution   = [0.10, 0.20, 0.40, 0.20, 0.10]   # bell with 10% tails
```

Threshold → regime mapping in fixed mode:

| Regime | Condition |
|---|---|
| STRONG_BEAR | ratio < 0.975 |
| BEAR | 0.975 ≤ ratio < 0.990 |
| NEUTRAL | 0.990 ≤ ratio < 1.010 |
| BULL | 1.010 ≤ ratio < 1.025 |
| STRONG_BULL | ratio ≥ 1.025 |

### 4.4 `regime_stable` column (`context.py: _rolling_stability`)

A bar has `regime_stable = True` when its regime has been unchanged for at least
`stable_window` consecutive bars (counting the bar itself).
Bars with a `NaN` regime are always `False`.
Default: `stable_window = 12`.

---

## 5. EMA window auto-derivation (`hurst.py`, `context.py`)

When `auto_window = True` (default), the EMA spans are derived automatically
from Hurst / Ornstein-Uhlenbeck analysis on the price series.

### 5.1 Derivation flow

```
prices (source_col)
    │
    ▼
rolling_halflife(prices, window_bars, stride_bars)
    │  Local OU fit: dP_t = const + kappa * P_(t-1) + eps  [numpy.linalg.lstsq]
    │  Half-life = -log(2) / log(1 + kappa)   [valid only when kappa < 0]
    ▼
median(half_life_series)   →  long_period  = round(hl)
                           →  short_period = round(hl * fast_ratio)   [default: 1/2.3]
```

The result is considered **converged** only when at least `min_window_estimates`
(default: 10) local estimates yield a mean-reverting fit and `short_period < long_period`.

### 5.2 Estimation window unit (`window_unit`)

| Unit | Behaviour |
|---|---|
| `"day"` (default) | Window and stride in calendar days, converted to bars via `bar_hours` |
| `"bar"` | Window and stride in bar count (timeframe-agnostic) |

In `"day"` mode, the candle duration (`bar_hours`) is inferred from the
DatetimeIndex or the first datetime column. If unavailable and `bar_hours`
is not set explicitly, a `ValueError` is raised.

### 5.3 Resolution sources recorded in `window_resolution`

| Source | Meaning |
|---|---|
| `"hurst_ou"` | Spans derived from a converged OU half-life |
| `"fallback"` | Auto-window requested but OU did not converge; configured periods used |
| `"configured"` | `auto_window = False`; configured periods used as-is |

### 5.4 Default values (calibrated on crypto 1H)

```python
short_period         = 9       # ≈ half-life / 2.3
long_period          = 25      # ≈ local half-life (~20 h on ADA/DOGE 1H)
fast_ratio           = 1/2.3
min_window_estimates = 10
window_estimation    = 168     # days (≈ 6 months on 1H)
window_stride        = 1       # day
```

---

## 6. Full configuration

### `EMAProxyConfig` (`models.py`)

| Parameter | Default | Description |
|---|---|---|
| `source_col` | `"close"` | OHLCV column used to compute the EMAs |
| `auto_window` | `True` | Derive EMA spans from Hurst/OU analysis |
| `short_period` | `9` | Fast EMA span (fallback if auto does not converge) |
| `long_period` | `25` | Slow EMA span (fallback) |
| `thresholds` | `[0.975, 0.990, 1.010, 1.025]` | Fixed cut points for the ratio |
| `threshold_mode` | `"fixed"` | `"fixed"` or `"balanced"` |
| `target_distribution` | `[0.10, 0.20, 0.40, 0.20, 0.10]` | Target distribution for balanced mode |
| `threshold_basis` | `"global"` | `"global"` or `"expanding"` |
| `threshold_warmup` | `200` | Leading bars that use fixed thresholds in expanding mode |
| `window_unit` | `"day"` | Unit for estimation window/stride (`"day"` or `"bar"`) |
| `window_estimation` | `168.0` | Width of the OU estimation window |
| `window_stride` | `1.0` | Step between successive estimates |
| `bar_hours` | `None` | Explicit candle duration in hours (inferred when None) |
| `fast_ratio` | `1/2.3` | Fast span as a fraction of the slow span |
| `min_window_estimates` | `10` | Minimum converging estimates required |

### `MarketContextConfig` (`models.py`)

| Parameter | Default | Description |
|---|---|---|
| `classifier` | `"ema_proxy"` | Classifier implementation to use |
| `ema_proxy` | `EMAProxyConfig()` | Parameters for `EMAProxyClassifier` |
| `labels` | `DEFAULT_LABELS` | Ordered regime labels |
| `stable_window` | `12` | Consecutive bars required for `regime_stable = True` |

---

## 7. Statistical primitives used

| Function | File | Algorithm |
|---|---|---|
| `hurst_dfa(series)` | `hurst.py` | Detrended Fluctuation Analysis (DFA) |
| `ou_halflife(series)` | `hurst.py` | Discrete OU regression via `numpy.linalg.lstsq` |
| `rolling_halflife(prices, window, stride)` | `hurst.py` | Rolling-window OU half-life |
| `derive_ema_windows(prices, ...)` | `hurst.py` | EMA spans from median local half-life |
| `variance_ratio_profile(series, lags)` | `hurst.py` | Variance Ratio per lag (VR < 1 = mean-reversion) |
| `suggest_ema_windows(prices, timeframe, ...)` | `hurst.py` | User-facing helper (offline analysis) |

---

## 8. Alignment with the functional analysis

### ✅ Aligned

- `RegimeClassifier` interface (ABC with `classify`, `get_labels`, `get_config`)
- 5 ordered regime labels
- `EMAProxyClassifier` with EMA ratio logic
- Fixed and balanced threshold modes
- Global and expanding basis
- Auto-window derivation from Hurst/OU
- Fallback to configured periods if OU does not converge
- EMA lookup by naming convention `{col}_ema_{period:02d}`
- Inline computation when column is absent (not written to table)
- `regime_stable` with configurable window (default 12)
- `build_classifier()` as the single instantiation point

### ➕ Added in code (not in the functional analysis)

- **`window_unit`** — `"day"` vs `"bar"` mode for cross-timeframe consistency
- **`bar_hours`** — explicit candle-duration override
- **`fast_ratio`** and **`min_window_estimates`** as explicit configuration fields
- **`threshold_warmup`** — leading bars that use fixed thresholds in expanding mode
- **Three resolution source tags** recorded: `"hurst_ou"`, `"fallback"`, `"configured"`
- **`distribution()`** — public method for per-regime bar counts
- **`regime_table()`** — compact frame for external joins
- **`resolved_thresholds`** in `get_config()` — tracks the thresholds actually used vs configured

### ❌ Divergences

No divergences from the functional analysis.
