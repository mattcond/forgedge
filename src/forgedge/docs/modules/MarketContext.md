# FORGE — Market Context Module
> Modulo 0 della pipeline FORGE.
> Gira una volta sola all'inizio della sessione, prima di Event Discovery.
> Arricchisce la KPI Table con una colonna `regime` che classifica ogni barra
> per contesto di mercato. Questa colonna è disponibile a tutti i moduli
> successivi — non viene mai modificata durante la sessione.

---

## Indice

1. [Posizionamento nella Pipeline](#1-posizionamento-nella-pipeline)
2. [Responsabilità](#2-responsabilità)
3. [Interfaccia RegimeClassifier](#3-interfaccia-regimeclassifier)
4. [v1.0 — EMAProxyClassifier](#4-v10--emaproxyclassifier)
5. [Configurazione](#5-configurazione)
6. [Output sulla KPI Table](#6-output-sulla-kpi-table)
7. [Come viene usato dai moduli successivi](#7-come-viene-usato-dai-moduli-successivi)
8. [Estensibilità — v2.0 e oltre](#8-estensibilità--v20-e-oltre)
9. [Esempio: ADAUSDC 1H 2025](#9-esempio-adausdc-1h-2025)

---

## 1. Posizionamento nella Pipeline

```
KPI Table (input grezzo — OHLCV + indicatori tecnici)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  MODULO 0 — MARKET CONTEXT MODULE                                  │
│                                                                   │
│  Classifica ogni barra per regime di mercato                      │
│  Aggiunge colonna 'regime' alla KPI Table                         │
│  Gira una volta — output immutabile per tutta la sessione         │
└───────────────────────────────────┬───────────────────────────────┘
                                    │  KPI Table arricchita
                                    │  (+ colonna 'regime')
              ┌─────────────────────┼───────────────────────┐
              │                     │                       │
              ▼                     ▼                       ▼
     Event Discovery       Alpha Discovery         Rule Discovery
     (ignora 'regime')     (Regime Sensitivity)    (Regime Dependency)
```

Il Market Context Module è l'unico modulo che scrive sulla KPI Table.
Tutti gli altri moduli la leggono soltanto.

Event Discovery ignora la colonna `regime` per design — lavora senza
osservare né il forward return né il contesto di mercato. La colonna
è disponibile ma non viene consumata.

---

## 2. Responsabilità

**Domanda:** *"in quale contesto di mercato si trova ogni barra?"*

**Input:** KPI Table grezza — le colonne OHLCV e gli indicatori tecnici
già presenti, incluse eventuali EMA precomputate.

**Output:** KPI Table con colonna `regime` aggiunta.

**Vincoli:**
- Gira **una sola volta** per sessione, all'avvio
- L'output è **immutabile** — nessun modulo successivo modifica `regime`
- Non osserva il forward return
- Non genera feature per il discovery — produce solo contesto

---

## 3. Interfaccia RegimeClassifier

Il Market Context Module non implementa direttamente la logica di
classificazione — delega a un oggetto che implementa l'interfaccia
`RegimeClassifier`. Questo è il punto di estensibilità che permette
di sostituire il classificatore in versioni future senza toccare
nessun altro modulo.

```
INTERFACE RegimeClassifier:

    FUNCTION classify(kpi_table) → pd.Series
        """
        Input:  KPI Table completa (DataFrame pandas)
        Output: Serie di label categoriche, indice allineato alla KPI Table
                es. "STRONG_BEAR" | "BEAR" | "NEUTRAL" | "BULL" | "STRONG_BULL"
        """

    FUNCTION get_labels() → list[str]
        """
        Ritorna la lista ordinata di label possibili
        (dal più ribassista al più rialzista)
        """

    FUNCTION get_config() → dict
        """
        Ritorna la configurazione usata — per tracciabilità nel report
        """
```

Qualsiasi implementazione che rispetti questa interfaccia può essere
pluggata nel Market Context Module senza modifiche a valle.

---

## 4. v1.0 — EMAProxyClassifier

In v1.0, il classificatore implementato è `EMAProxyClassifier`.
Calcola il ratio tra una EMA veloce e una EMA lenta come proxy
del trend di mercato, e lo discretizza in regime tramite soglie
configurabili.

> `short_period` e `long_period` non sono fissi: vengono risolti a monte dal
> Market Context Module a partire dall'analisi Hurst/OU dei dati (vedi
> [Scelta delle finestre EMA](#scelta-delle-finestre-ema-fastslow--automatica)).
> Quando `classify` viene invocato, le due finestre sono già state decise.

### Logica

```
FUNCTION classify(kpi_table):

    // Step 1: recupera o calcola la EMA veloce
    short_col = f"{source_col}_ema_{short_period:02d}"
    SE short_col IN kpi_table.columns:
        ema_short = kpi_table[short_col]
    ALTRIMENTI:
        ema_short = kpi_table[source_col].ewm(span=short_period).mean()
        // calcolato inline — NON aggiunto alla KPI Table

    // Step 2: recupera o calcola la EMA lenta
    long_col = f"{source_col}_ema_{long_period:02d}"
    SE long_col IN kpi_table.columns:
        ema_long = kpi_table[long_col]
    ALTRIMENTI:
        ema_long = kpi_table[source_col].ewm(span=long_period).mean()

    // Step 3: calcola il ratio
    ratio = ema_short / ema_long

    // Step 4: classifica per soglie
    regime = pd.cut(ratio, bins=[0] + thresholds + [+inf], labels=labels)

    RETURN regime
```

### Naming convention per il lookup

Il modulo costruisce il nome della colonna seguendo la stessa convenzione
usata da FORGE per tutti gli indicatori: `{col}_{tipo}_{periodo:02d}`.

```
source_col = "close"
short_period = 9   →  cerca "close_ema_09"
long_period  = 25  →  cerca "close_ema_25"
```

Se queste colonne esistono nella KPI Table (come nel caso del CandleKPI
di QHF, che le precomputa), vengono usate direttamente.
Se non esistono, vengono calcolate inline senza lasciare traccia
nella KPI Table — il valore intermedio viene scartato dopo la classificazione.

### Semantica dei regime label

| Label | Ratio EMA | Interpretazione |
|---|---|---|
| `STRONG_BEAR` | < 0.975 | EMA veloce oltre 2.5% sotto la lenta — trend ribassista forte |
| `BEAR` | 0.975 – 0.990 | EMA veloce tra 1% e 2.5% sotto la lenta — trend ribassista moderato |
| `NEUTRAL` | 0.990 – 1.010 | EMA veloce e lenta allineate — mercato laterale |
| `BULL` | 1.010 – 1.025 | EMA veloce tra 1% e 2.5% sopra la lenta — trend rialzista moderato |
| `STRONG_BULL` | > 1.025 | EMA veloce oltre 2.5% sopra la lenta — trend rialzista forte |

Le soglie `[0.975, 0.990, 1.010, 1.025]` sono calibrate empiricamente
su crypto 1H. Sono configurabili.

---

## 5. Configurazione

```yaml
# forge_config.yaml

market_context:
  classifier: "ema_proxy"    # implementazione da usare
                              # in v2.0: "hmm" | "kmeans" | "custom"

  ema_proxy:
    source_col:    "close"   # colonna OHLCV su cui calcolare le EMA
    auto_window:   true      # decide short/long dai dati (Hurst/OU)
    window_unit:   "day"     # "day" (default, coerente tra timeframe) | "bar"
    window_estimation: 168   # W: 168 giorni se unit="day", 168 barre se unit="bar"
    window_stride:  1        # passo tra le stime, stessa unità di W
    short_period:  9         # fallback EMA veloce (se l'analisi non converge)
    long_period:   25        # fallback EMA lenta  (se l'analisi non converge)
    thresholds:    [0.975, 0.990, 1.010, 1.025]

  labels:
    - "STRONG_BEAR"
    - "BEAR"
    - "NEUTRAL"
    - "BULL"
    - "STRONG_BULL"
```

### Parametri e default

| Parametro | Default | Note |
|---|---|---|
| `classifier` | `"ema_proxy"` | Implementazione del RegimeClassifier |
| `ema_proxy.source_col` | `"close"` | Colonna su cui calcolare le EMA |
| `ema_proxy.auto_window` | `true` | Decide `short`/`long` dall'analisi Hurst/OU dei dati |
| `ema_proxy.window_unit` | `"day"` | Unità di `window_estimation`/`window_stride`: `"day"` (coerente tra TF) o `"bar"` (per-TF) |
| `ema_proxy.window_estimation` | `168` | Ampiezza finestra di stima — giorni se `"day"`, barre se `"bar"` |
| `ema_proxy.window_stride` | `1` | Passo tra le stime, stessa unità di `window_estimation` |
| `ema_proxy.bar_hours` | `null` | Durata candela (h) per `"day"`; inferita dall'indice se assente |
| `ema_proxy.short_period` | `9` | EMA veloce — fallback se l'analisi non converge |
| `ema_proxy.long_period` | `25` | EMA lenta — fallback se l'analisi non converge |
| `ema_proxy.thresholds` | `[0.975, 0.990, 1.010, 1.025]` | Soglie di discretizzazione del ratio |
| `labels` | `["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]` | Label regime (ordinate dal più ribassista) |

---

## 6. Output sulla KPI Table

Il modulo aggiunge due colonne alla KPI Table:

```
regime          → label categorica ordinata per ogni barra
                  dtype: pd.Categorical con ordering
                  valori: "STRONG_BEAR" | "BEAR" | "NEUTRAL" | "BULL" | "STRONG_BULL"

regime_stable   → True se il regime è invariato nelle ultime N barre consecutive
                  dtype: bool
                  default N: 12 (parametro configurabile)
                  uso: escludere barre di transizione dall'analisi di regime
```

Entrambe le colonne fanno parte del contratto dell'interfaccia:
ogni implementazione futura di `RegimeClassifier` deve produrle.

### Schema della KPI Table dopo Market Context

```
KPI Table (prima)
  open_dt | open | high | low | close | volume | close_rsi_25 | close_ema_09 | ...

KPI Table (dopo Market Context)
  open_dt | open | high | low | close | volume | close_rsi_25 | close_ema_09 | ...
  | regime | regime_stable
```

Le colonne EMA intermedie usate per calcolare il ratio **non vengono
aggiunte** alla KPI Table se non erano già presenti — solo `regime`
e `regime_stable` sono nuove.

### Accesso al solo regime — `regime_table()`

Oltre alla KPI Table arricchita restituita da `run()`, il modulo espone
`regime_table()`, che ritorna un DataFrame compatto
`[timestamp, regime, regime_stable]` pronto per il join con il dato di origine:

```python
mc = MarketContext(kpi)
mc.run()

rt = mc.regime_table()          # colonne: open_dt | regime | regime_stable
joined = source_df.merge(rt, on="open_dt", how="left")
```

Il nome della colonna timestamp è inferito (DatetimeIndex o prima colonna
datetime, default `open_dt`) ed è sovrascrivibile con
`regime_table(timestamp_col="...")`. `regime` resta un categorical ordinato.

---

## 7. Come viene usato dai moduli successivi

### Event Discovery
Non usa la colonna `regime`. La riceve nella KPI Table ma la ignora
per design — opera senza contesto di mercato.

### Alpha Discovery — Step 5 (Regime Sensitivity)
Legge `kpi_table['regime']` per stratificare le misure di IC e win rate:

```python
# Alpha Discovery Step 5 — legge il regime dalla KPI Table
for regime_label in kpi_table['regime'].cat.categories:
    regime_mask  = kpi_table['regime'] == regime_label
    ec_in_regime = event_mask & regime_mask
    if ec_in_regime.sum() < 10: continue

    ic_r, p_r = spearmanr(
        kpi_table.loc[regime_mask, source_feature],
        kpi_table.loc[regime_mask, 'fwd_return_24']
    )
    wr_r = kpi_table.loc[ec_in_regime, 'target'].mean()
```

### Rule Discovery — Step 5 (Regime Dependency)
Legge `kpi_table['regime']` per analizzare la distribuzione dei trade
per contesto di mercato:

```python
# Rule Discovery Step 5 — legge il regime dalla KPI Table
trades_with_regime = perf_trades.merge(
    kpi_table[['open_dt', 'regime']],
    left_on='fill_open_dt', right_on='open_dt'
)
regime_perf = trades_with_regime.groupby('regime').agg(...)
```

### Rule Registry
Usa `regime` nel trade log del report HTML e nella tabella piatta
per arricchire il contesto di ogni trade.

---

## 8. Estensibilità — v2.0 e oltre

In v2.0, l'implementazione `EMAProxyClassifier` può essere sostituita
senza modificare nessun modulo a valle. Il campo `classifier` nella
configurazione seleziona l'implementazione.

### Esempi di classificatori futuri

**HMM (Hidden Markov Model):**
```yaml
market_context:
  classifier: "hmm"
  hmm:
    n_regimes:    4
    features:     ["close_ret_12", "close_vol_24"]
    n_iter:       100
```

**KMeans:**
```yaml
market_context:
  classifier: "kmeans"
  kmeans:
    n_clusters:   5
    features:     ["close_ret_24", "close_vol_24", "close_rsi_14"]
    random_state: 42
```

**Classificatore custom (utente):**
```yaml
market_context:
  classifier: "custom"
  custom:
    module: "my_classifiers.MyRegimeClassifier"
    params: { ... }
```

Ogni classificatore produce le stesse due colonne (`regime`, `regime_stable`)
con gli stessi label configurati — i moduli a valle non devono sapere
quale classificatore è stato usato.

> **Nota:** i label rimangono `["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL",
> "STRONG_BULL"]` anche con classificatori diversi da EMAProxy. In un HMM
> con 5 stati, lo stato con il return medio più basso viene mappato a
> `STRONG_BEAR`, quello con il return più alto a `STRONG_BULL`. Il mapping
> è automatico per ordinamento del return medio per stato.

---

## 9. Esempio: ADAUSDC 1H 2025

### Configurazione usata

```yaml
market_context:
  classifier:  "ema_proxy"
  ema_proxy:
    source_col:   "close"
    short_period: 9
    long_period:  25
    thresholds:   [0.975, 0.990, 1.010, 1.025]
  labels: ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
```

### Lookup

```
Cerca "close_ema_09" nella KPI Table → TROVATA ✅  (precomputa da enricher QHF)
Cerca "close_ema_25" nella KPI Table → TROVATA ✅
Calcola ratio = close_ema_09 / close_ema_25  (inline, non salvato)
```

### Distribuzione regime su 8.760 barre

| Regime | N barre | % | Periodo prevalente |
|---|---:|---:|---|
| `STRONG_BEAR` | 1.842 | 21% | Q1 e Q4 2025 |
| `BEAR` | 1.576 | 18% | Transizioni |
| `NEUTRAL` | 2.190 | 25% | Distribuito |
| `BULL` | 1.533 | 17% | Q3 2025 parziale |
| `STRONG_BULL` | 1.619 | 18% | Q3 2025 (uptrend) |

### Regime e RI_01

Alpha Discovery usa questa distribuzione per stratificare la Win Rate
Analysis dell'evento `close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10`:

```
STRONG_BEAR:  24 trade  WR=87.5%  → forte
BEAR:         38 trade  WR=84.2%  → forte
NEUTRAL:      29 trade  WR=79.3%  → buono
BULL:          8 trade  WR=50.0%  → debole
STRONG_BULL:   3 trade  WR=33.3%  → evitare

Classificazione: regime-conditional
→ documentato nel campo regime_dependency dell'Alpha Contract
```

---

### Scelta delle finestre EMA (fast/slow) — automatica

Le finestre EMA **non sono valori fissi**: il Market Context Module le *decide
per ogni dataset caricato* a partire dall'analisi del coefficiente di Hurst e
dell'half-life del processo Ornstein-Uhlenbeck (vedi
`forgedge.market_context.hurst` e `notebooks/hurst.ipynb`).

```
FUNCTION _resolve_ema_windows(prices):
    // half-life OU locale, stimata su finestre rolling (robusta al drift)
    hl_bars = median( rolling_halflife(prices, estimation_window_bars) )

    SE l'half-life converge (abbastanza finestre mean-reverting):
        long_period  = round(hl_bars)            // EMA lenta ≈ half-life
        short_period = round(hl_bars * 1/2.3)    // EMA veloce ≈ half-life / 2.3
        source = "hurst_ou"
    ALTRIMENTI:
        long_period  = 25   // fallback
        short_period = 9    // fallback
        source = "fallback"
```

Il rapporto `ema_short / ema_long` è quindi un proxy di *quanto il prezzo è
sopra/sotto il proprio livello di mean-reversion* — esattamente il segnale di
trend che il classificatore discretizza — con le finestre calibrate sulla
dinamica effettiva dell'asset.

I default `short_period=9` / `long_period=25` restano nella configurazione
**solo come fallback**, usati quando il coefficiente non converge (serie
puramente trending o storia troppo corta). Per forzare finestre fisse si
imposta `ema_proxy.auto_window: false`.

> **Nota:** con `auto_window: true` le finestre derivate possono non
> coincidere con eventuali EMA precomputate nella KPI Table (es. `close_ema_25`):
> in tal caso la EMA della finestra derivata viene calcolata inline dal `close`.
> Le colonne precomputate restano comunque inalterate nella tabella. Per usare
> esattamente le EMA precomputate si imposta `auto_window: false` con i
> `short_period` / `long_period` corrispondenti.

La risoluzione effettiva (`source`, spans usati, half-life stimata) è esposta
in `MarketContext.window_resolution` e in `get_config()` per la tracciabilità
nel report.

#### Esempio sui dati forniti

Con il default (`window_unit="day"`, `W=168` giorni) le finestre derivate sono
coerenti tra 1H e 1D — riflettono la mean-reversion alla scala dell'orizzonte
di stima (168 giorni):

| Asset | 1H slow | 1D slow | coerenza |
|---|---:|---:|---|
| ADAUSDC | 545h | 552h | ✅ |
| DOGEUSDC | 643h | 648h | ✅ |

Restringendo l'orizzonte a `window_unit="bar"`, `W=168` (1 settimana su 1H) si
recupera invece la scala intraday: ADAUSDC → short=9 / long=21 (~21h),
DOGEUSDC → short=9 / long=20 (~20h). Entrambi mean-reverting (Hurst ≪ 0.5),
half-life intraday ~20-21h.

#### Coerenza tra timeframe — `window_unit`

La finestra di stima è **un unico valore** (`ema_proxy.window_estimation`, 168)
la cui *unità* dipende da `ema_proxy.window_unit`:

- **`"day"` (default)** — `W = 168` significa 168 **giorni** su *ogni* timeframe
  (anche 1H), convertiti in barre tramite la durata candela (inferita
  dall'indice datetime o impostata con `bar_hours`). L'orizzonte di stima è
  identico in wall-clock su ogni timeframe, quindi l'half-life — e le EMA
  derivate — risultano coerenti in tempo reale. `window_stride` segue la stessa
  unità di `W` (default 1 giorno). Richiede informazione temporale (indice o
  colonna datetime, oppure `bar_hours`) e storia sufficiente (> `W` giorni):
  in mancanza si ricade sui fallback `9` / `25`.
- **`"bar"`** — `W = 168` significa 168 **candele**, indipendentemente dal
  timeframe: 1 settimana su 1H, ma 168 giorni su 1D. È *timeframe-agnostica* e
  non richiede informazione temporale, ma le finestre derivate **non sono
  confrontabili in tempo reale** tra timeframe diversi. Lo stesso valore di
  `window_estimation` viene semplicemente reinterpretato da giorni a barre.

Aggregando ADAUSDC 1H → 1D, con `W = 168`:

| | `unit="bar"` (168 barre) | `unit="day"` (168 giorni) |
|---|---|---|
| **1H** | stima su 168b (1 settimana) → slow **21h** | stima su 4032b (168g) → slow **545h** |
| **1D** | stima su 168b (168 giorni) → slow **576h** ❌ | stima su 168b (168g) → slow **552h** ✅ |

In modalità `"bar"` l'orizzonte di stima cambia con il timeframe e l'half-life
salta da 21h (1H) a 576h (1D). In modalità `"day"` l'orizzonte è 168 giorni su
entrambi e l'half-life resta coerente (~545h, ovvero la mean-reversion a scala
multi-settimanale catturata da quell'orizzonte).

> **Limite fisico:** una dinamica sotto-timeframe (es. la mean-reversion
> intraday ~21h) non è risolvibile su candele 1D. Con un `W` ampio (168 giorni)
> entrambi i timeframe convergono sulla scala lenta e coincidono; con un `W`
> piccolo (es. 7 giorni) su 1H si recupera la scala ~21h, ma su 1D degenera a
> una EMA di 1-2 barre.

---

*Market Context Module — FORGE (Feature-Oriented Rule Generation Engine)*
*Versione 1.0 · Maggio 2026 · Parte di FORGE v1.0*
*Status: Implementato — `forgedge.market_context`*
