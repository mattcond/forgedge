# Modulo 0 — Market Context (Spec da Codebase)

> **Riferimento codice:** `src/forgedge/market_context/`
> **Analisi funzionale:** `docs/modules/MarketContext.md`
> **Stato:** ✅ Implementato e allineato con l'analisi funzionale.
> Alcune opzioni di configurazione sono più ricche di quanto documentato.

---

## 1. Posizione nella pipeline

Il Market Context è il primo modulo ad essere eseguito.
Arricchisce la KPI Table con due colonne — `regime` e `regime_stable` —
che rimangono immutabili per tutta la sessione.
Nessun modulo downstream può modificarle.

```
KPI Table (OHLCV + indicatori)
        │
        ▼
  MarketContext.run()
        │
        ▼
KPI Table + 'regime' + 'regime_stable'   ──► Event Discovery
                                          ──► Alpha Discovery
                                          ──► (Rule Discovery — non implementato)
```

---

## 2. Interfaccia pubblica

### `MarketContext` (`context.py`)

```python
MarketContext(kpi_table, config=None, classifier=None)
```

| Metodo / Proprietà | Descrizione |
|---|---|
| `run() → pd.DataFrame` | Classifica ogni barra; restituisce copia con `regime` + `regime_stable` |
| `get_config() → dict` | Configurazione completa usata, inclusa risoluzione finestre EMA |
| `distribution() → pd.DataFrame` | Conteggio e quota per regime (n_bars, share) |
| `regime_table(timestamp_col) → pd.DataFrame` | Frame compatto `[timestamp, regime, regime_stable]` per join esterni |
| `window_resolution` | Dict con sorgente ("hurst_ou" / "fallback" / "configured") e valori usati |

Il metodo `run()` **non muta** il DataFrame in input — restituisce una copia.
Gli indicatori EMA intermedi calcolati inline non vengono aggiunti alla tabella.

---

## 3. Interfaccia `RegimeClassifier` (`models.py`)

Tutte le implementazioni del classificatore devono rispettare questa ABC:

```python
class RegimeClassifier(ABC):
    def classify(kpi_table: pd.DataFrame) → pd.Series   # etichette categoriali ordinate
    def get_labels() → list[str]                          # da più bearish a più bullish
    def get_config() → dict                               # per tracciabilità nel report
```

Etichette predefinite (ordinate):
```
STRONG_BEAR | BEAR | NEUTRAL | BULL | STRONG_BULL
```

Nomi delle colonne di output (costanti):
```python
REGIME_COL       = "regime"          # categoriale ordinata
REGIME_STABLE_COL = "regime_stable"  # bool
```

---

## 4. Implementazione v1.0: `EMAProxyClassifier` (`ema_proxy.py`)

### 4.1 Logica di classificazione

1. Recupera o calcola `ema_short` e `ema_long` dalla colonna `source_col`.
2. Calcola `ratio = ema_short / ema_long`.
3. Discretizza il ratio in etichette di regime tramite soglie configurate.

**Lookup EMA per convenzione di naming:**
```
{source_col}_ema_{period:02d}   →  es. "close_ema_09", "close_ema_25"
```
Se la colonna non è presente nella KPI Table, l'EMA viene calcolata
inline con `ewm(span=period, adjust=False)` e **non** scritta nella tabella.

### 4.2 Modalità di soglia (`threshold_mode`)

| Modalità | Comportamento |
|---|---|
| `"fixed"` (default) | Soglie assolute applicate al ratio EMA |
| `"balanced"` | Soglie calcolate come quantili del ratio per far combaciare `target_distribution` |

In modalità `"balanced"`, `threshold_basis` controlla la causalità:

| Basis | Comportamento | Causalità |
|---|---|---|
| `"global"` (default) | Quantili calcolati sull'intero campione una sola volta | No (look-ahead) |
| `"expanding"` | Quantili calcolati su `[0..t]` per ogni barra | Sì (causale) |

In modalità `"expanding"`, le prime `threshold_warmup` barre usano le soglie fixed come fallback.
Se la modalità `"balanced"` produce soglie non strettamente crescenti (ratio degenere),
ricade automaticamente su `"fixed"`.

### 4.3 Soglie e distribuzione target predefinite

```python
thresholds            = [0.975, 0.990, 1.010, 1.025]
target_distribution   = [0.10, 0.20, 0.40, 0.20, 0.10]  # campana con code 10%
```

Mapping soglie → regime in modalità fixed:

| Regime | Condizione |
|---|---|
| STRONG_BEAR | ratio < 0.975 |
| BEAR | 0.975 ≤ ratio < 0.990 |
| NEUTRAL | 0.990 ≤ ratio < 1.010 |
| BULL | 1.010 ≤ ratio < 1.025 |
| STRONG_BULL | ratio ≥ 1.025 |

### 4.4 Colonna `regime_stable` (`context.py: _rolling_stability`)

Una barra è `regime_stable = True` quando il suo regime è rimasto
invariato per almeno `stable_window` barre consecutive (contando la barra stessa).
Le barre con regime `NaN` sono sempre `False`.
Default: `stable_window = 12`.

---

## 5. Auto-derivazione finestre EMA (`hurst.py`, `context.py`)

Quando `auto_window = True` (default), le finestre EMA vengono derivate automaticamente
dall'analisi Hurst/Ornstein-Uhlenbeck sui prezzi.

### 5.1 Flusso di derivazione

```
prezzi (source_col)
    │
    ▼
rolling_halflife(prices, window_bars, stride_bars)
    │  OU fit locale: dP_t = const + kappa * P_(t-1) + eps  [numpy.linalg.lstsq]
    │  Half-life = -log(2) / log(1 + kappa)   [valida solo se kappa < 0]
    ▼
median(half_life_series)   →  long_period = round(hl)
                           →  short_period = round(hl * fast_ratio)   [default: 1/2.3]
```

Il risultato è considerato **convergente** solo se almeno `min_window_estimates`
(default: 10) stime locali producono un fit mean-reverting
e `short_period < long_period`.

### 5.2 Unità della finestra di stima (`window_unit`)

| Unità | Comportamento |
|---|---|
| `"day"` (default) | Finestra e stride in giorni di calendario, convertiti in barre via `bar_hours` |
| `"bar"` | Finestra e stride in numero di barre (timeframe-agnostico) |

In modalità `"day"`, la durata della candela (`bar_hours`) viene inferita dal
DatetimeIndex o dalla prima colonna datetime. Se non disponibile e `bar_hours`
non è configurato esplicitamente, viene sollevato un `ValueError`.

### 5.3 Fonti di risoluzione registrate in `window_resolution`

| Source | Significato |
|---|---|
| `"hurst_ou"` | Finestre derivate da half-life OU convergente |
| `"fallback"` | Auto-window richiesto ma OU non convergeva; usati i periodi configurati |
| `"configured"` | `auto_window = False`; usati i periodi configurati as-is |

### 5.4 Valori predefiniti (calibrati su crypto 1H)

```python
short_period      = 9      # ≈ half-life / 2.3
long_period       = 25     # ≈ half-life locale (~20h su ADA/DOGE 1H)
fast_ratio        = 1/2.3
min_window_estimates = 10
window_estimation = 168    # giorni (≈ 6 mesi su 1H)
window_stride     = 1      # giorno
```

---

## 6. Configurazione completa

### `EMAProxyConfig` (`models.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `source_col` | `"close"` | Colonna OHLCV su cui calcolare le EMA |
| `auto_window` | `True` | Deriva le finestre EMA da Hurst/OU |
| `short_period` | `9` | Span EMA veloce (fallback se auto non converge) |
| `long_period` | `25` | Span EMA lento (fallback) |
| `thresholds` | `[0.975, 0.990, 1.010, 1.025]` | Soglie fixed per il ratio |
| `threshold_mode` | `"fixed"` | `"fixed"` o `"balanced"` |
| `target_distribution` | `[0.10, 0.20, 0.40, 0.20, 0.10]` | Distribuzione target per modalità balanced |
| `threshold_basis` | `"global"` | `"global"` o `"expanding"` |
| `threshold_warmup` | `200` | Barre iniziali che usano soglie fixed in expanding mode |
| `window_unit` | `"day"` | Unità per finestra/stride di stima (`"day"` o `"bar"`) |
| `window_estimation` | `168.0` | Ampiezza finestra di stima OU |
| `window_stride` | `1.0` | Passo tra stime successive |
| `bar_hours` | `None` | Durata candela esplicita in ore (se None, inferita) |
| `fast_ratio` | `1/2.3` | Span veloce come frazione dello span lento |
| `min_window_estimates` | `10` | Stime minime convergenti richieste |

### `MarketContextConfig` (`models.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `classifier` | `"ema_proxy"` | Implementazione del classificatore da usare |
| `ema_proxy` | `EMAProxyConfig()` | Parametri per EMAProxyClassifier |
| `labels` | `DEFAULT_LABELS` | Etichette di regime ordinate |
| `stable_window` | `12` | Barre consecutive per `regime_stable = True` |

---

## 7. Primitive statistiche usate

| Funzione | File | Algoritmo |
|---|---|---|
| `hurst_dfa(series)` | `hurst.py` | Detrended Fluctuation Analysis (DFA) |
| `ou_halflife(series)` | `hurst.py` | Regressione OU discreta via `numpy.linalg.lstsq` |
| `rolling_halflife(prices, window, stride)` | `hurst.py` | Half-life OU su finestra rolling |
| `derive_ema_windows(prices, ...)` | `hurst.py` | Span EMA da mediana half-life locale |
| `variance_ratio_profile(series, lags)` | `hurst.py` | Variance Ratio per lag (VR < 1 = mean-reversion) |
| `suggest_ema_windows(prices, timeframe, ...)` | `hurst.py` | Helper user-facing (analisi offline) |

---

## 8. Allineamento con l'analisi funzionale

### ✅ Allineato

- Interfaccia `RegimeClassifier` (ABC con `classify`, `get_labels`, `get_config`)
- 5 etichette di regime ordinate
- Implementazione `EMAProxyClassifier` con ratio EMA
- Modalità fixed e balanced
- Basis global ed expanding
- Derivazione auto-window da Hurst/OU
- Fallback a periodi configurati se OU non converge
- Lookup EMA per naming convention `{col}_ema_{period:02d}`
- Calcolo inline se colonna non presente (non scritta nella tabella)
- `regime_stable` con finestra configurabile (default 12)
- `build_classifier()` come unico punto di istanza

### ➕ Aggiunto nel codice (non nella documentazione funzionale)

- **`window_unit`** — modalità `"day"` vs `"bar"` per uniformità cross-timeframe
- **`bar_hours`** — override esplicito della durata candela
- **`fast_ratio`** e **`min_window_estimates`** come campi di configurazione espliciti
- **`threshold_warmup`** — numero di barre iniziali in expanded mode che usano soglie fixed
- **Tre sorgenti di risoluzione** registrate: `"hurst_ou"`, `"fallback"`, `"configured"`
- **`distribution()`** — metodo pubblico per distribuzione per regime
- **`regime_table()`** — frame compatto per join esterni
- **`resolved_thresholds`** in `get_config()` — traccia le soglie effettivamente usate vs configurate

### ❌ Divergenze

Nessuna divergenza rispetto all'analisi funzionale.
