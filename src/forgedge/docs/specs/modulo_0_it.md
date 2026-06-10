# Modulo 0 — Market Context

Il Market Context è il primo modulo della pipeline FORGE. Il suo compito è
classificare ogni barra della KPI Table in un regime di mercato, aggiungendo
due colonne — `regime` e `regime_stable` — che rimangono immutabili per tutta
la sessione. Tutti i moduli downstream leggono queste colonne ma non le modificano.

---

## Utilizzo di base

```python
from forgedge import MarketContext
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")   # deve contenere la colonna 'close'
mc = MarketContext(kpi)
enriched = mc.run()

# La tabella arricchita contiene le stesse colonne di kpi, più 'regime' e 'regime_stable'
print(enriched[["close", "regime", "regime_stable"]].tail(10))
print(mc.distribution())
```

`run()` non muta il DataFrame originale — restituisce sempre una copia.

---

## Output: colonne aggiunte

### `regime` — categoriale ordinata

Ogni barra riceve un'etichetta di regime scelta tra cinque valori ordinati
da più bearish a più bullish:

```
STRONG_BEAR  <  BEAR  <  NEUTRAL  <  BULL  <  STRONG_BULL
```

Il tipo pandas è `Categorical` con `ordered=True`, quindi i confronti
ordinali (`regime > "BEAR"`) e i groupby funzionano direttamente.

### `regime_stable` — booleano

`True` quando il regime della barra è rimasto invariato per almeno
`stable_window` barre consecutive (contando la barra corrente).
Serve a escludere le barre di transizione dalle analisi di regime sensitivity.

Esempio: con `stable_window=12`, le prime 11 barre di ogni nuovo regime
hanno `regime_stable=False`. Solo dalla 12ª barra consecutiva in avanti
la barra è considerata "stabile".

Le barre con regime `NaN` (leading NaN da EMA warmup) hanno sempre
`regime_stable=False`.

---

## Come funziona la classificazione (EMAProxyClassifier)

Il classificatore di default calcola il rapporto tra una EMA veloce e una EMA lenta:

```
ratio = ema_short / ema_long
```

e lo discretizza in cinque regimi tramite quattro soglie. Il ratio è un proxy
della distanza del prezzo dal suo livello di mean-reversion locale: un ratio
alto indica che il prezzo è sopra la media di breve termine (trend rialzista),
uno basso che è sotto (trend ribassista).

### Ricerca delle colonne EMA nella KPI Table

Il classificatore cerca prima le colonne EMA nella KPI Table usando
la convenzione di naming FORGE:

```
{source_col}_ema_{period:02d}   →   es. "close_ema_09", "close_ema_25"
```

Se la colonna è presente (come nella `CandleKPI` del framework QHF che le
precalcola), viene usata direttamente. Se è assente, l'EMA viene calcolata
inline con `ewm(span=period, adjust=False)` e il risultato **non** viene
scritto nella tabella.

### Soglie e mapping in modalità `"fixed"` (default)

| Regime | Condizione sul ratio |
|---|---|
| `STRONG_BEAR` | ratio < 0.975 |
| `BEAR` | 0.975 ≤ ratio < 0.990 |
| `NEUTRAL` | 0.990 ≤ ratio < 1.010 |
| `BULL` | 1.010 ≤ ratio < 1.025 |
| `STRONG_BULL` | ratio ≥ 1.025 |

I valori 0.975/0.990/1.010/1.025 significano che le EMA devono divergere
di almeno il ±1% (BEAR/BULL) o il ±2.5% (STRONG) per uscire da NEUTRAL.

---

## Derivazione automatica delle finestre EMA

Con `auto_window=True` (default), FORGE deriva le finestre EMA dai dati
anziché usare valori fissi, tramite un'analisi Hurst/Ornstein-Uhlenbeck.

### Perché derivare le finestre dai dati?

L'EMA lenta deve avere span ≈ half-life della mean-reversion locale del prezzo,
così il ratio `ema_short / ema_long` cattura effettivamente la distanza dal
livello di equilibrio. Usare span fissi (9/25) su tutti gli asset è un'approssimazione:
crypto con mean-reversion rapida necessitano span più corti; asset trending più lunghi.

### Algoritmo

1. Sul prezzo (`source_col`, default `close`), si stimano half-life locali
   tramite regressione OU discreta su finestre rolling:
   ```
   dP_t = const + kappa * P_{t-1} + ε    [numpy.linalg.lstsq]
   half-life = -log(2) / log(1 + kappa)  [valida solo se kappa < 0]
   ```
2. Si calcola la mediana delle half-life convergenti.
3. `long_period = round(hl)`, `short_period = round(hl * fast_ratio)` (default `1/2.3`).
4. Se convergono almeno `min_window_estimates` (default 10) stime,
   il risultato è valido; altrimenti si ricade sui periodi configurati.

### Ampiezza della finestra di stima (`window_unit`)

La finestra di stima può essere espressa in **giorni** (`window_unit="day"`,
default) o in **barre** (`window_unit="bar"`).

- **Modalità `"day"`** (raccomandata): la finestra di 168 giorni corrisponde
  alla stessa quantità di dati su qualsiasi timeframe. Su 1H = 168 × 24 = 4032
  barre, su 4H = 168 × 6 = 1008 barre. Richiede un DatetimeIndex o una colonna
  datetime; in assenza di entrambi, impostare `bar_hours` esplicitamente.
- **Modalità `"bar"`**: 168 barre sempre, indipendentemente dal timeframe.
  Conveniente se il DataFrame non ha informazioni temporali.

### Sorgente di risoluzione

Dopo `run()`, `mc.window_resolution` riporta come sono stati scelti gli span:

| `source` | Significato |
|---|---|
| `"hurst_ou"` | Half-life convergente; `short_period` e `long_period` derivati dai dati |
| `"fallback"` | `auto_window=True` ma OU non convergeva; usati `short_period`/`long_period` configurati |
| `"configured"` | `auto_window=False`; span configurati usati direttamente |

```python
mc.run()
print(mc.window_resolution)
# {'source': 'hurst_ou', 'short_period': 9, 'long_period': 22,
#  'half_life_bars': 21.4, 'n_estimates': 47, 'unit': 'day', ...}
```

---

## Modalità di soglia

### `threshold_mode="fixed"` (default)

Soglie assolute sul ratio. Ogni barra ottiene il regime in base al valore
puntuale del ratio in quella barra. I valori default `[0.975, 0.990, 1.010, 1.025]`
sono calibrati empiricamente su dati crypto 1H.

Quando usarla: produzione standard, quando si vuole che il regime abbia
un significato assoluto (es. "EMA veloce è almeno 2.5% sopra l'EMA lenta").

### `threshold_mode="balanced"`

Le soglie vengono calcolate come quantili del ratio per far corrispondere
la distribuzione dei regimi a `target_distribution` (default: campana con
code 10% — `[0.10, 0.20, 0.40, 0.20, 0.10]`).

Quando usarla: quando si vuole che i cinque regimi abbiano frequenze
predeterminate (es. per garantire campioni bilanciati nell'analisi di regime
sensitivity del Modulo 2).

**Nota:** in modalità balanced, le soglie perdono il significato assoluto
(non è più garantito che STRONG_BULL significhi "+2.5% di divergenza").

#### `threshold_basis` in modalità balanced

- **`"global"`** (default): quantili calcolati una sola volta sull'intero
  campione. Raggiunge esattamente la distribuzione target ma non è causale
  (le etichette passate dipendono da dati futuri). Appropriato per analisi
  in-sample.
- **`"expanding"`**: quantili calcolati su `[0..t]` per ogni barra `t`.
  Causale (nessun look-ahead), ma la distribuzione target è approssimata.
  Le prime `threshold_warmup` barre (default 200) usano le soglie fixed come
  fallback mentre si accumula storia sufficiente.

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
print(mc.distribution())          # frequenze vicine al target
```

---

## Metodi di output

### `mc.distribution() → pd.DataFrame`

Ritorna il conteggio e la quota di barre per regime:

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

Ritorna un frame compatto `[timestamp, regime, regime_stable]` utile per
joinare il regime su DataFrame esterni senza portarsi dietro tutte le colonne.

```python
regime_df = mc.regime_table()
# Unisce il regime sulla tabella originale
merged = original_df.merge(regime_df, on="open_dt", how="left")
```

### `mc.get_config() → dict`

Ritorna la configurazione completa usata, inclusi i valori effettivi delle
finestre EMA e delle soglie (utile per tracciabilità e riproducibilità):

```python
cfg = mc.get_config()
print(cfg["window_resolution"])
print(cfg["classifier"]["resolved_thresholds"])
```

---

## Configurazione completa

### `EMAProxyConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `source_col` | `"close"` | Colonna OHLCV su cui calcolare le EMA |
| `auto_window` | `True` | Deriva le finestre EMA da analisi Hurst/OU |
| `short_period` | `9` | Span EMA veloce (usato come fallback se auto non converge, o se `auto_window=False`) |
| `long_period` | `25` | Span EMA lenta (idem) |
| `thresholds` | `[0.975, 0.990, 1.010, 1.025]` | Soglie fixed per il ratio (devono essere strettamente crescenti, lunghezza = n_labels - 1) |
| `threshold_mode` | `"fixed"` | `"fixed"` o `"balanced"` |
| `target_distribution` | `[0.10, 0.20, 0.40, 0.20, 0.10]` | Distribuzione target per modalità balanced (pesi relativi, normalizzati internamente) |
| `threshold_basis` | `"global"` | `"global"` o `"expanding"` (solo per balanced) |
| `threshold_warmup` | `200` | Barre iniziali che usano soglie fixed in expanding mode |
| `window_unit` | `"day"` | Unità per finestra/stride di stima OU (`"day"` o `"bar"`) |
| `window_estimation` | `168.0` | Ampiezza della finestra di stima OU |
| `window_stride` | `1.0` | Passo tra stime successive |
| `bar_hours` | `None` | Durata candela in ore (override esplicito; se None, inferita da DatetimeIndex) |
| `fast_ratio` | `1/2.3` | Rapporto span veloce/lento per derivazione auto |
| `min_window_estimates` | `10` | Stime OU convergenti minime richieste per considerare valida la derivazione |

### `MarketContextConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `classifier` | `"ema_proxy"` | Implementazione del classificatore (solo `"ema_proxy"` disponibile in v0.1.0) |
| `ema_proxy` | `EMAProxyConfig()` | Parametri per `EMAProxyClassifier` |
| `labels` | `DEFAULT_LABELS` | Etichette di regime ordinate (da più bearish a più bullish) |
| `stable_window` | `12` | Barre consecutive richieste per `regime_stable=True` |

---

## Estendere il classificatore

Il classificatore è basato su un'interfaccia ABC (`RegimeClassifier`) che
permette di sostituire `EMAProxyClassifier` con qualsiasi implementazione
alternativa (HMM, KMeans, custom) senza toccare i moduli downstream.

```python
from forgedge.market_context.models import RegimeClassifier
import pandas as pd

class MioClassificatore(RegimeClassifier):
    def __init__(self, labels):
        self.labels = labels

    def classify(self, kpi_table: pd.DataFrame) -> pd.Series:
        # ... logica personalizzata ...
        return pd.Series(...)   # etichette categoriali ordinate

    def get_labels(self):
        return self.labels

    def get_config(self):
        return {"classifier": "mio_classificatore"}

# Passaggio diretto del classificatore (bypassa config.classifier)
mc = MarketContext(kpi, classifier=MioClassificatore(["BEAR", "NEUTRAL", "BULL"]))
enriched = mc.run()
```

Il metodo `build_classifier(config)` in `context.py` è il singolo punto
che mappa il nome stringa a un'implementazione concreta. Per registrare
un nuovo classificatore in modo permanente è sufficiente aggiungere
un caso a quella funzione.

---

## Analisi offline delle finestre EMA

La funzione `suggest_ema_windows` in `hurst.py` permette di analizzare
off-line le finestre ottimali per un asset prima di configurare il modulo:

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

Un `hurst_median < 0.5` conferma che la serie è mean-reverting sulla finestra
di stima: il ratio EMA è un proxy sensato del regime.

---

## Utilizzo downstream

**Event Discovery (Modulo 1):** ignora il regime durante la discovery.
La colonna `regime` è disponibile nel DataFrame ma non viene letta
nella pipeline di Step 0–5.

**Alpha Discovery (Modulo 2):** legge `regime` e `regime_stable` per
stratificare IC e win rate per regime (Step 5 — Regime Sensitivity Analysis).
Se `use_stable_regime_only=True`, include nella stratificazione solo le barre
con `regime_stable=True`.

Il modo canonico di passare la tabella arricchita ai moduli downstream è:

```python
enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)      # il regime è già nella tabella
candidates = ed.run()

ad = AlphaDiscovery(ed.df, candidates, config)   # ed.df ha regime + feature derivate
```
