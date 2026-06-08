# FORGE — Event Discovery Module
> Primo modulo della pipeline di ricerca quantitativa FORGE.
> Dato un Variable Catalog di feature, genera sistematicamente tutti gli eventi
> booleani candidati attraverso cinque step distinti — due di costruzione
> e tre di assemblaggio — preceduti da una fase di classificazione del tipo
> delle feature, senza mai osservare il forward return.
> L'output sono **Event Candidate** pronti per Alpha Discovery.

---

## Indice

1. [Posizionamento nel sistema](#1-posizionamento-nel-sistema)
2. [Architettura in cinque step](#2-architettura-in-cinque-step)
3. [Step 0 — Classificazione del tipo e ammissibilità](#3-step-0--classificazione-del-tipo-e-ammissibilità)
4. [Step 1 — Feature Generation](#4-step-1--feature-generation)
5. [Step 2 — Transform Layer](#5-step-2--transform-layer)
6. [Step 3 — Event Generation](#6-step-3--event-generation)
7. [Step 4 — Consistency Gate](#7-step-4--consistency-gate)
8. [Step 5 — AND Composition](#8-step-5--and-composition)
9. [Output: Event Candidate](#9-output-event-candidate)
10. [Decodifica dei nomi degli eventi](#10-decodifica-dei-nomi-degli-eventi)
11. [Esempio end-to-end: da `close_rsi_25` a `rsi25_pr96 < 0.10`](#11-esempio-end-to-end-da-close_rsi_25-a-rsi25_pr96--010)

---

## 1. Posizionamento nel sistema

```
┌──────────────────────────────────────────────────────────────────┐
│                        PIPELINE FORGE                            │
│                                                                  │
│  ┌────────────────────┐                                          │
│  │  EVENT DISCOVERY   │  ← questo modulo                        │
│  │  Input:  Variable  │    Costruisce e filtra eventi            │
│  │          Catalog   │    senza guardare il forward return      │
│  │  Output: Event     │                                          │
│  │          Candidates│                                          │
│  └─────────┬──────────┘                                          │
│            │ Event Candidates                                    │
│            ▼                                                     │
│  ┌────────────────────┐                                          │
│  │  ALPHA DISCOVERY   │  Misura il potere predittivo             │
│  │                    │  IC, quantile analysis, regime           │
│  └─────────┬──────────┘  ← qui entrano sell_pct e target_h      │
│            │ Alpha Contract                                      │
│            ▼                                                     │
│  ┌────────────────────┐                                          │
│  │  RULE DISCOVERY    │  Traduce in regola operativa             │
│  └─────────┬──────────┘                                          │
│            │ Regola validata + parametri                         │
│            ▼                                                     │
│  ┌────────────────────┐                                          │
│  │  RULE REGISTRY     │  Deduplica, cross-ticker backtest,      │
│  │                    │  matrici correlazione, export           │
│  └─────────┬──────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                          │
│  │  STRATEGIST (QHF)  │  Deploy in produzione                   │
│  └────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Principio fondamentale

Event Discovery **non conosce il target**. Non sa cosa sia `sell_pct`, `target_h` o il forward return. Opera esclusivamente sulla struttura temporale delle serie — costruisce feature, applica trasformazioni, genera booleani, filtra per qualità distribuzionale. La domanda a cui risponde è:

> **"Quali eventi booleani si possono costruire dal catalogo, e quali hanno proprietà temporali stabili e non banali?"**

La risposta a "questi eventi predicono qualcosa di utile?" spetta ad Alpha Discovery.

---

## 2. Architettura in cinque step

```
Variable Catalog (feature native CandleKPI)
         │
         │  CLASSIFICAZIONE PRELIMINARE
         ▼
┌────────────────────────────────────────┐
│  STEP 0 — Classificazione del tipo     │
│  Per ogni colonna: continuous |        │
│  binary | categorical                  │
│  • continuous → pipeline completa      │
│  • binary     → è già un evento        │
│  • categorical→ one-hot in N eventi    │
│    (scartata se > 20 classi)           │
│  Solo le continue: domanda scale-free  │
└───────────────┬────────────────────────┘
                │  continue → Step 1; binary/categorical → Step 3
                │  COSTRUZIONE
                ▼
┌────────────────────────────────────────┐
│  STEP 1 — Feature Generation           │
│  Produce feature derivate normalizzate │
│  da una, due o tre feature native      │
│  Arietà: unaria | binaria | ternaria   │
└───────────────┬────────────────────────┘
                │  Feature Catalog esteso
                ▼
┌────────────────────────────────────────┐
│  STEP 2 — Transform Layer              │
│  Applica trasformazioni temporali      │
│  a ogni feature del catalogo esteso    │
│  Identità | Pctrank | Zscore | Delta   │
└───────────────┬────────────────────────┘
                │  Serie trasformate
                │
                │  ASSEMBLAGGIO E TEST
                ▼
┌────────────────────────────────────────┐
│  STEP 3 — Event Generation             │
│  Applica soglie alle serie trasformate │
│  + eventi booleani da binary/categorical│
│  Produce eventi booleani candidati     │
│  Threshold | Crossing                  │
└───────────────┬────────────────────────┘
                │  ~2.400 eventi grezzi
                ▼
┌────────────────────────────────────────┐
│  STEP 4 — Consistency Gate             │
│  Filtra per struttura temporale        │
│  Nessun forward return osservato       │
│  Criteri: volume, copertura,           │
│           concentrazione, frequenza    │
└───────────────┬────────────────────────┘
                │  Eventi che passano
                ▼
┌────────────────────────────────────────┐
│  STEP 5 — AND Composition              │
│  Combina eventi singoli in condizioni  │
│  composte. Riapplica il gate.          │
└───────────────┬────────────────────────┘
                │
                ▼
         Event Candidates
         → Alpha Discovery
```

> **Step 1 e 2** costruiscono serie continue — lavorano in domini algebrico e statistico.
> **Step 3, 4 e 5** assemblano booleani e li filtrano — lavorano in dominio temporale.
> Nessuno dei cinque step osserva il forward return.

---

## 3. Step 0 — Classificazione del tipo e ammissibilità

Prima di costruire qualsiasi feature, il modulo classifica ogni colonna del Variable Catalog. La ragione è che la pipeline di trasformazione (Step 1-2) assume serie **continue numeriche**: applica algebra, percentili, medie e deviazioni standard. Una feature non numerica o categorica non è materiale da trasformare — o è già un evento, o non ha senso come serie ordinabile.

Questa fase è **automatica e senza configurazione** (out-of-the-box): l'utente carica il proprio file con decine di colonne e il modulo decide da solo come trattarle. L'utente non deve annotare nulla a mano.

### Rilevamento del tipo

La classificazione combina il dtype della colonna e la sua cardinalità (numero di valori distinti):

| Condizione | Tipo | Percorso |
|---|---|---|
| dtype non numerico (stringhe) | `categorical` | one-hot → N eventi booleani; salta Step 1-2 |
| dtype numerico, esattamente 2 valori distinti | `binary` | è già un evento booleano; salta Step 1-2 |
| dtype numerico, > 2 valori distinti | `continuous` | pipeline completa Step 1-5 |

### Trattamento per tipo

**Continuous** — segue la pipeline standard (Step 1-5). Per queste, e solo per queste, si pone la domanda scale-free (vedi sotto).

**Binary** — una colonna a due valori (es. colore candela Red/Green) è concettualmente già un evento. Non viene trasformata: entra direttamente allo Step 3 come serie booleana già pronta, e da lì prosegue verso gate e composizione AND.

**Categorical** — una colonna multi-classe (es. forma candela: doji, hammer, spinning top…) viene espansa in **one-hot**: una serie booleana per ogni classe (`is_doji`, `is_hammer`, …). Ogni serie è un evento che salta Step 1-2. Una composizione come `is_doji AND close_rsi_25 < p10` è perfettamente legittima a valle.

### Cap sulle categoriche ad alta cardinalità

Una categorica con più di **20 classi distinte** viene **scartata**. Il one-hot produrrebbe oltre 20 eventi quasi sempre troppo sparsi per superare il Consistency Gate, e una cardinalità così alta segnala tipicamente un identificatore o testo libero, non un segnale di mercato. La soglia è il parametro `max_categorical_classes` (default 20).

### Classificazione scale-free (solo continue)

L'identità (Step 2) applica una soglia assoluta calcolata sull'intera storia della serie. Questo è affidabile solo su serie **scale-free** — già normalizzate o stazionarie. Su una serie con shock di livello, la soglia statica verrebbe trascinata dal regime estremo e diventerebbe inefficace; per questo l'identità si applica **solo alle serie scale-free**, mentre pctrank e z-score (robusti agli shock per costruzione) coprono tutte le continue.

La classificazione scale-free è **automatica e conservativa**:

- **v1.0 — euristico** sui valori della serie (es. stabilità di scala su finestre successive). Tarato in modo asimmetrico: **nel dubbio, non scale-free**. Un falso positivo corrompe un evento (soglia trascinata); un falso negativo costa solo qualche evento in meno. L'asimmetria dei costi impone la prudenza.
- **Futuro** — test statistico di stazionarietà in media (ADF/KPSS) e in varianza, che renderebbe la classificazione asset-adattiva.

### Override e trasparenza

La classificazione scale-free automatica può sbagliare, quindi è prevista una correzione manuale: l'utente può ispezionare la decisione del modulo per ogni colonna e, se necessario, **forzarla** (override sovrano). L'override è rispettato senza discussione, ma la divergenza dall'euristico resta tracciata, così la decisione non è mai silenziosa. Le regole certe (tipo, cap classi) non hanno override perché non sono stime: non c'è incertezza da correggere.

### Limite noto

Una categorica multi-classe **codificata come interi** (es. forma passata come `0,1,2,3,4` invece che come stringhe) è indistinguibile da una continua guardando solo i dati, e verrà classificata `continuous` — con il rischio di un ordinamento finto nelle soglie. La mitigazione è passare quella colonna come stringhe; la classificazione risultante è comunque visibile nel report di trasparenza, dove l'utente esperto può correggerla.

---

## 4. Step 1 — Feature Generation

Il Feature Generator prende le feature native del CandleKPI e produce feature derivate **normalizzate**, rimuovendo il problema del price level. Le feature di prezzo assoluto (`close`, `ema_09`, `sma_25`, ecc.) non entrano direttamente nelle trasformazioni — vanno prima combinate tra loro per produrre serie scale-free.

### Il problema del price level

Con 126 feature native, 94 collassano in un unico cluster di correlazione perché misurano tutte la stessa grandezza: il livello del prezzo. L'IC apparentemente alto di queste feature è spurio — cattura il trend strutturale dell'asset, non un edge operativo. Servono operazioni che rimuovano il livello e lascino solo la struttura.

### Arietà 1 — Unaria

Trasformazione su una singola feature. Si applica quando la feature è già normalizzata o quando una funzione monotona migliora la distribuzione.

```
Input:      f(A)
Quando:     feature già in scala relativa, o distribuzione molto skewed
```

| Operazione | Formula | Quando usarla |
|---|---|---|
| Identità | `A` | Feature già normalizzate: RSI, BB%b, volatilità |
| Logaritmo | `log(A)` | Feature positive con distribuzione right-skewed |
| Inverso | `1 / A` | Quando il reciproco ha interpretazione diretta |
| Valore assoluto | `\|A\|` | Quando interessa la magnitudine, non la direzione |

**Esempio — `close_rsi_25`:**

```
Feature:     close_rsi_25    (scala [0–100], già normalizzata)
Operazione:  Identità
Output:      close_rsi_25    (invariato, entra direttamente allo Step 2)

Statistiche: min=10.0  p05=25.8  p10=30.5  mean=47.5  p90=64.2  max=88.0
```

`close_rsi_25` non richiede trasformazione unaria perché è già scale-free.
Passando all'arietà 2 sarebbe possibile costruire `rsi14/rsi25` per misurare la divergenza
tra i due oscillatori — ma questo appartiene all'arietà 2, non all'unaria.

---

### Arietà 2 — Binaria

Operazione su due feature della stessa **famiglia semantica** (stessa unità di misura). Produce una feature relazionale che misura quanto due serie si discostano tra loro.

```
Input:      f(A, B)
Vincolo:    A e B devono appartenere alla stessa famiglia semantica
            oppure A deve essere un prezzo e B la sua media mobile
```

**Famiglie semantiche nel CandleKPI:**

| Famiglia | Feature incluse | Operazione tipica |
|---|---|---|
| EMA | `ema_03`, `ema_09`, `ema_12`, `ema_25`, `ema_96` | Ratio breve/lungo |
| SMA | `sma_03`, `sma_09`, `sma_12`, `sma_25` | Ratio o spread |
| Prezzo vs MA | `close` + qualsiasi MA | Spread percentuale |
| Volume | `volume`, `volume_sma_12`, `volume_sma_25` | Ratio |
| Rolling min/max | `close_min_24`, `close_max_24` | Spread |

**Catalogo operazioni binarie:**

| Operazione | Formula | Output | Semantica |
|---|---|---|---|
| Ratio | `A / B` | Scala intorno a 1 | "Quanto A è sopra/sotto B?" |
| Spread percentuale | `(A - B) / B` | Centrato su 0 | "Distanza percentuale di A da B" |
| Differenza normalizzata | `(A - B) / std(A-B)` | Z-score della differenza | "Quanto è anomala la distanza?" |

**Esempio — `ratio_ema09_25`:**

```
Feature A:   close_ema_09   (prezzo, scala assoluta)
Feature B:   close_ema_25   (prezzo, scala assoluta)
Operazione:  A / B
Output:      ratio_ema09_25 = close_ema_09 / close_ema_25

Interpretazione: quanto EMA9 è sopra o sotto EMA25?
  > 1.0  → EMA9 sopra EMA25 (trend rialzista)
  < 1.0  → EMA9 sotto EMA25 (trend ribassista)
  < 0.975 → EMA9 oltre 2.5% sotto EMA25 (trend ribassista forte)

Statistiche: min=0.9992  p05=0.9994  p10=0.9994
             mean=0.9996  p90=0.9998  max=1.0000
```

**Esempio — `pct_from_sma25`:**

```
Feature A:   close           (prezzo)
Feature B:   close_sma_25    (media mobile del prezzo)
Operazione:  (A - B) / B
Output:      pct_from_sma25 = (close - close_sma_25) / close_sma_25

Interpretazione: distanza percentuale del prezzo dalla sua media 25h
  > 0   → prezzo sopra la media
  < -0.03 → prezzo oltre 3% sotto la media (potenziale oversold)
```

---

### Arietà 3 — Ternaria

Operazione su tre feature con struttura `(lower, value, upper)` o `(min, value, max)`. Produce una feature posizionale che misura dove si trova `value` nell'intervallo definito da `lower` e `upper`.

```
Input:      f(A, B, C)
Vincolo:    struttura (min/lower, value, max/upper)
Operazione: posizione relativa (A - B) / (C - B)
Output:     valore in [0, 1]
```

**Esempio — `bb_pct_b`:**

```
Feature A (value):  close
Feature B (lower):  close_bb_lower_20
Feature C (upper):  close_bb_upper_20
Operazione:         (A - B) / (C - B)
Output:             bb_pct_b

Interpretazione: dove si trova il prezzo nelle bande di Bollinger?
  0.0  → prezzo alla banda inferiore
  0.5  → prezzo alla banda media (SMA20)
  1.0  → prezzo alla banda superiore
  < 0  → prezzo sotto la banda inferiore (evento raro ~3%)

Statistiche: min=-0.356  p05=-0.063  p10=-0.017
             mean=0.145  p90=0.317   max=1.128
```

**Esempio — `price_position_in_range`:**

```
Feature A (value):  close
Feature B (min):    close_min_24
Feature C (max):    close_max_24
Operazione:         (A - B) / (C - B)
Output:             price_position_24h

Interpretazione: dove si trova il prezzo nel range delle ultime 24h?
  0.0 → prezzo al minimo delle 24h
  1.0 → prezzo al massimo delle 24h
```

---

### Selezione delle coppie/triplette

Il Feature Generator non esplora tutte le combinazioni possibili — con 30 feature sarebbero 4.495 coppie e 4.060 triplette. Si usa una regola di selezione a due livelli:

1. **Stesso cluster semantico** — solo feature della stessa famiglia (EMA con EMA, SMA con SMA, ecc.)
2. **IC minimo su almeno un input** — almeno una delle feature della coppia deve avere `|IC univariato| > threshold` dal catalogo

Questo riduce lo spazio a poche decine di coppie significative.

---

## 5. Step 2 — Transform Layer

Il Transform Layer prende ogni feature del catalogo esteso (native + derivate da Step 1) e produce versioni trasformate nel tempo. Ogni trasformazione risponde a una domanda diversa sulla stessa serie.

### Le quattro trasformazioni

#### Identità (nessuna trasformazione)

```
Domanda:    "Il valore corrente è rilevante direttamente?"
Formula:    output_t = feature_t
Quando:     feature già normalizzate (RSI, BB%b, ratio EMA)
Parametri:  nessuno
Vincolo:    applicata SOLO a serie scale-free (vedi Step 0)
```

L'identità si applica **esclusivamente alle serie classificate scale-free** allo Step 0. La sua soglia è calcolata in-sample sull'intera serie (valore singolo congelato, poi propagato invariato a valle). Questa soglia statica è affidabile solo se la scala della serie è stabile: su una serie con shock di livello verrebbe trascinata dal regime estremo. Pctrank e z-score, robusti agli shock per costruzione, coprono invece tutte le serie continue, incluse quelle non scale-free.

#### Pctrank rolling

```
Domanda:    "Il valore corrente è raro rispetto alla storia recente?"
Formula:    output_t = rank(feature_t, finestra_W) / W
Output:     serie in [0, 1], distribuzione uniforme per costruzione
Parametri:  finestra W ∈ {48, 96, 168}
```

Proprietà chiave: il pctrank è **regime-agnostico**. In qualsiasi periodo, circa il 10% delle barre avrà pctrank < 0.10. Non dipende dal livello assoluto della feature, solo dall'ordinamento relativo.

#### Z-score rolling

```
Domanda:    "Il valore corrente è anomalo rispetto alla media recente?"
Formula:    output_t = (feature_t - mean_W) / std_W
Output:     numero di deviazioni standard, approssimativamente N(0,1)
Parametri:  finestra W ∈ {48, 96, 168}
```

#### Delta

```
Domanda:    "Il valore sta cambiando rapidamente?"
Formula:    output_t = feature_t - feature_{t-lag}
Output:     variazione assoluta nel periodo lag
Parametri:  lag ∈ {1, 3, 6, 12}
```

### Tabella riepilogativa

| Trasformazione | Output | Distribuzione | Parametro | Soglie successive |
|---|---|---|---|---|
| Identità | Feature originale | Dipende dalla feature | — | Distribuzionali (percentili storici) |
| Pctrank | `[0, 1]` uniforme | Uniforme per costruzione | Finestra W | Distribuzionali (coincidono con i percentili) |
| Z-score | `~N(0, 1)` | Quasi-normale | Finestra W | Teoriche (-2.0, -1.5, -1.0, ...) |
| Delta | Centrata su 0 | Simmetrica | Lag | Distribuzionali (percentili della distribuzione dei delta) |

**Esempio su `close_rsi_25` con W=96:**

```
Identità:
  output = close_rsi_25
  range: [10.0, 88.0]  mean=47.5

Pctrank (W=96):
  output = pr_close_rsi_25_96
  range: [0.010, 1.000]  mean=0.509  ← distribuzione quasi uniforme
  barra t=375: rsi25=24.74, finestra [24.0, 52.2], 2/96 valori < 24.74
               pctrank = 0.021

Z-score (W=96):
  output = zscore_close_rsi_25_96
  range: [-4.10, +4.09]  p05=-1.79  p10=-1.41  mean≈0

Delta (lag=6):
  output = delta_close_rsi_25_6
  range: [-35.9, +29.9]  p05=-14.2  p10=-11.0  mean≈0
```

---

## 6. Step 3 — Event Generation

L'Event Generator prende ogni serie prodotta dallo Step 2 e applica una soglia per produrre un evento booleano. È il punto in cui si passa dal dominio continuo al dominio discreto.

### Tipi di evento

#### Threshold (soglia)

```
Formula:    evento_t = (serie_t < soglia)    [per estremi bassi]
            evento_t = (serie_t > soglia)    [per estremi alti]
Output:     booleano — True finché la condizione è vera (stato persistente)
```

#### Crossing (attraversamento)

```
Formula:    evento_t = (serie_t < soglia) AND (serie_{t-1} >= soglia)
Output:     booleano — True solo nel momento del crossing (evento puntuale)
Differenza: Threshold rimane attivo per N barre consecutive.
            Crossing si attiva per 1 sola barra — la transizione.
```

### Il Threshold Catalog

Le soglie non sono valori hardcoded — vengono calcolate sulla distribuzione della **serie trasformata** (non della feature originale). Questo garantisce che ogni soglia produca sempre la stessa frequenza di attivazione, indipendentemente dall'asset.

**Tipo A — Soglie distribuzionali** (calcolate sulla serie trasformata):

| Estremo | Percentili scanditi |
|---|---|
| Basso | p03, p05, p08, p10, p15 |
| Alto | p85, p90, p92, p95, p97 |

**Tipo B — Soglie teoriche** (solo per Z-score, valori fissi):

| Soglia | Significato | Frequenza teorica |
|---|---|---|
| z < −2.0 | 2σ sotto la media | ~2.3% |
| z < −1.5 | 1.5σ sotto la media | ~6.7% |
| z < −1.0 | 1σ sotto la media | ~15.9% |
| z > +1.0 | 1σ sopra la media | ~15.9% |
| z > +1.5 | 1.5σ sopra la media | ~6.7% |
| z > +2.0 | 2σ sopra la media | ~2.3% |

> **Perché le soglie della vista, non della feature?**
> La trasformazione cambia la distribuzione. Il p10 di `close_rsi_25` è 30.5
> (scala [0–100]). Il p10 della sua versione pctrank è esattamente 0.10
> (distribuzione uniforme). Il p10 della versione z-score è circa −1.28.
> Usare i percentili della serie trasformata garantisce che ogni soglia
> produca sempre ~10% di attivazioni su qualsiasi feature e qualsiasi asset.

---

## 7. Step 4 — Consistency Gate

Il Consistency Gate filtra gli eventi basandosi **esclusivamente sulla struttura temporale** delle attivazioni. Non osserva il forward return, non conosce il target.

Un evento con 3 attivazioni all'anno non è interessante — non c'è abbastanza storia per valutarlo. Un evento che si attiva solo in 2 mesi su 12 è probabilmente regime-dipendente. Il gate scarta questi casi prima di passare il controllo ad Alpha Discovery.

### I quattro criteri

```
CRITERIO 1 — Volume minimo
  n_activations >= MIN_ACT
  Default: 50
  Scopo:   prerequisito di stabilità statistica per Alpha Discovery —
           non è un filtro sulla frequenza operativa dell'evento

CRITERIO 2 — Copertura mensile
  n_active_months >= MIN_MONTHS
  Default: 8 su 12
  Scopo:   generalizzabilità cross-regime — l'evento deve presentarsi
           in contesti di mercato diversi, non concentrarsi in un solo periodo

CRITERIO 3 — Concentrazione massima
  max_single_month / n_activations <= MAX_CONC
  Default: 0.40
  Scopo:   nessun mese singolo domina le attivazioni — evita eventi
           che sono l'impronta di un unico episodio di mercato

CRITERIO 4 — Frequenza minima
  n_activations / n_months >= MIN_TPM
  Default: 2.0 attivazioni/mese
  Scopo:   conseguenza degli altri criteri — raramente il vincolo
           determinante. Per eventi rari strutturalmente validi,
           abbassare MIN_TPM è la leva corretta.
```

> **Nota:** il gate filtra per **distribuzione temporale stabile**,
> non per volume. Un evento con 438 attivazioni può fallire (concentrazione
> 58% in un solo mese) mentre uno con 50 attivazioni distribuite su 10 mesi
> passa. La domanda che il gate pone è: *questo evento è una proprietà
> ricorrente del mercato, o è l'impronta di un singolo regime?*

### Esempio — eventi da `close_rsi_25`

La tabella mostra perché alcuni eventi vengono scartati e altri promossi.

**Identità + soglia distribuzionale:**

| Evento | N | Mesi | Conc. | TPM | Esito | Motivo |
|---|---:|---:|---:|---:|---|---|
| `close_rsi_25 < 25.8` (p05) | 438 | 10 | **0.58** | 36.5 | ❌ FAIL | Concentrazione 58% in 1 mese |
| `close_rsi_25 < 29.1` (p08) | 702 | 11 | **0.45** | 58.5 | ❌ FAIL | Concentrazione 45% |
| `close_rsi_25 < 30.5` (p10) | 876 | 12 | 0.40 | 73.0 | ✅ PASS | Tutti i criteri soddisfatti |
| `close_rsi_25 < 33.6` (p15) | 1314 | 12 | 0.30 | 109.5 | ✅ PASS | |

> **Nota:** `close_rsi_25 < 25.8` fallisce non per volume (438 è sufficiente) ma
> per concentrazione: il 58% delle attivazioni cade in un solo mese.
> Questo è il segnale che quella soglia assoluta è regime-dipendente —
> si attiva quasi solo nei periodi di panico acuto, non distribuisce nel tempo.

**Pctrank + soglia distribuzionale:**

| Evento | N | Mesi | Conc. | TPM | Esito |
|---|---:|---:|---:|---:|---|
| `pr_close_rsi_25_96 < 0.05` | 487 | 12 | 0.11 | 40.6 | ✅ PASS |
| `pr_close_rsi_25_96 < 0.08` | 810 | 12 | 0.10 | 67.5 | ✅ PASS |
| `pr_close_rsi_25_96 < 0.10` | 1017 | 12 | 0.10 | 84.8 | ✅ PASS |
| `pr_close_rsi_25_96 < 0.12` | 1211 | 12 | 0.10 | 100.9 | ✅ PASS |
| `pr_close_rsi_25_96 < 0.15` | 1490 | 12 | 0.10 | 124.2 | ✅ PASS |

Tutti i pctrank passano il gate con concentrazione ~0.10. Per costruzione matematica
il pctrank distribuisce le attivazioni uniformemente nel tempo — è il suo punto di forza.

**Z-score + soglia teorica:**

| Evento | N | Mesi | Conc. | TPM | Esito |
|---|---:|---:|---:|---:|---|
| `zscore_close_rsi_25_96 < -1.0` | 1608 | 12 | 0.11 | 134.0 | ✅ PASS |
| `zscore_close_rsi_25_96 < -1.5` | 743 | 12 | 0.10 | 61.9 | ✅ PASS |
| `zscore_close_rsi_25_96 < -2.0` | 270 | 12 | 0.11 | 22.5 | ✅ PASS |

**Delta + soglia distribuzionale:**

| Evento | N | Mesi | Conc. | TPM | Esito |
|---|---:|---:|---:|---:|---|
| `delta_close_rsi_25_6 < -14.2` (p05) | 438 | 12 | 0.14 | 36.5 | ✅ PASS |
| `delta_close_rsi_25_6 < -12.0` (p08) | 701 | 12 | 0.11 | 58.4 | ✅ PASS |
| `delta_close_rsi_25_6 < -11.0` (p10) | 876 | 12 | 0.11 | 73.0 | ✅ PASS |

**Crossing:**

| Evento | N | Mesi | Conc. | TPM | Esito |
|---|---:|---:|---:|---:|---|
| `close_rsi_25 crosses_below 25.8` | 231 | 10 | 0.32 | 19.2 | ✅ PASS |
| `close_rsi_25 crosses_below 30.5` | 428 | 12 | 0.17 | 35.7 | ✅ PASS |
| `close_rsi_25 crosses_below 33.6` | 552 | 12 | 0.16 | 46.0 | ✅ PASS |

### Parametri del gate

| Parametro | Default | Note |
|---|---|---|
| `MIN_ACT` | 50 | Per dataset 12 mesi, 1H |
| `MIN_MONTHS` | 8 | Su 12 mesi totali |
| `MAX_CONC` | 0.40 | Nessun mese > 40% delle attivazioni |
| `MIN_TPM` | 2.0 | Almeno 2 attivazioni/mese in media |

> I parametri del gate non dipendono dall'asset o dalla strategia — dipendono solo
> dall'orizzonte temporale del dataset. Per un dataset di 6 mesi, `MIN_MONTHS`
> scende a 4. Per un dataset di 24 mesi, `MIN_MONTHS` sale a 16.

---

## 8. Step 5 — AND Composition

Dopo il gate sui singoli eventi, il modulo genera composizioni AND tra eventi che passano e riapplica il gate sul risultato. La composizione produce eventi più selettivi — meno attivazioni, ma potenzialmente più significative.

### Regole di composizione

```
AMMESSA:     eventi di trasformazioni diverse sulla stessa feature
             → Identità AND Pctrank (V1 × V3)
             → Identità AND Zscore  (V1 × V2)
             → Pctrank  AND Delta   (V3 × V4)

AMMESSA:     eventi di feature semanticamente diverse
             → close_rsi_25 AND ratio_ema09_25

NON AMMESSA: eventi della stessa trasformazione con soglie diverse
             → pr_close_rsi_25_96 < 0.10 AND pr_close_rsi_25_96 < 0.15
             (il secondo è un soprainsieme del primo — ridondante)

NON AMMESSA: più di tre eventi in AND
             (oltre i tre layer diventa overfitting strutturale)
```

### Perché Identità × Pctrank è la combinazione più potente

Le due trasformazioni rispondono a domande **ortogonali** sulla stessa feature:

```
Identità (soglia assoluta):  "il valore è basso in assoluto nella storia dell'asset?"
                              filtra l'intensità assoluta

Pctrank (soglia relativa):   "il valore è raro nel contesto delle ultime W barre?"
                              filtra la rarità nel regime corrente

AND → "è contemporaneamente basso in assoluto E raro nel contesto recente"
      seleziona solo i momenti dove entrambe le condizioni convergono
      elimina i falsi positivi dove una sola condizione è vera
```

Questo non è un semplice AND tra due filtri — è la combinazione di due scale temporali distinte. La soglia assoluta porta il contesto storico, il pctrank porta il contesto recente.

### Risultati composizioni su `close_rsi_25`

**Identità × Pctrank:**

| Composizione | N | Mesi | Conc. | TPM | Esito |
|---|---:|---:|---:|---:|---|
| `rsi25 < 30.5 AND pr_96 < 0.05` | 186 | 12 | 0.18 | 15.5 | ✅ PASS |
| `rsi25 < 30.5 AND pr_96 < 0.08` | 291 | 12 | 0.20 | 24.2 | ✅ PASS |
| **`rsi25 < 30.5 AND pr_96 < 0.10`** | **329** | **12** | **0.20** | **27.4** | **✅ PASS** |
| `rsi25 < 33.6 AND pr_96 < 0.05` | 228 | 12 | 0.18 | 19.0 | ✅ PASS |
| `rsi25 < 33.6 AND pr_96 < 0.10` | 412 | 12 | 0.17 | 34.3 | ✅ PASS |

**Identità × Zscore:**

| Composizione | N | Mesi | Conc. | TPM | Esito |
|---|---:|---:|---:|---:|---|
| `rsi25 < 30.5 AND zscore < -1.0` | 442 | 12 | 0.23 | 36.8 | ✅ PASS |
| `rsi25 < 30.5 AND zscore < -1.5` | 253 | 12 | 0.18 | 21.1 | ✅ PASS |
| `rsi25 < 33.6 AND zscore < -1.5` | 315 | 12 | 0.15 | 26.2 | ✅ PASS |

**Esempio con feature binaria — `ratio_ema09_25`:**

```
pr_ratio_ema09_25_96 < 0.05    N=773   mesi=12  conc=0.11  → PASS
pr_ratio_ema09_25_96 < 0.10    N=1284  mesi=12  conc=0.11  → PASS
ratio_ema09_25 < 0.9994 (p10)  N=876   mesi=12  conc=0.13  → PASS
```

---

## 9. Output: Event Candidate

Gli eventi che superano il gate al passo 5 diventano **Event Candidate** — il formato di output consegnato ad Alpha Discovery.

```yaml
# EVENT CANDIDATE
# ─────────────────────────────────────────────────────────────
event_id:         "EVT-close_rsi_25-ID×PR-P105-W096-P010"
status:           "CANDIDATE"
generated_date:   "2026-05-23"

# Struttura dell'evento
components:
  - source_feature:  "close_rsi_25"
    transform:       "identity"
    transform_params: {}
    threshold:       30.5          # p10 della distribuzione storica di close_rsi_25
    threshold_type:  "distributional_p10"
    expression:      "close_rsi_25 < 30.5"

  - source_feature:  "close_rsi_25"
    transform:       "rolling_pctrank"
    transform_params: {window: 96}
    threshold:       0.10          # p10 della distribuzione del pctrank (= 0.10 per costruzione)
    threshold_type:  "distributional_p10"
    expression:      "pr_close_rsi_25_96 < 0.10"

# Espressione composta — soglie fisse, propagate invariate ad Alpha Discovery e Rule Discovery
expression: "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"

# Statistiche di attivazione (Consistency Gate superato)
n_activations:      329
n_active_months:    12
zero_months:        0
max_monthly_share:  0.20
mean_tpm:           27.4

consistency_gate:
  result:    "PASS"
  min_act:   50
  min_months: 8
  max_conc:  0.40
  min_tpm:   2.0

# Handoff — compilato da Alpha Discovery
alpha_discovery_response: null
```

---

## 10. Decodifica dei nomi degli eventi

Ogni `expression` restituita da EventDiscovery è costruita a partire da un nome di colonna trasformata e una soglia. La struttura del nome è deterministica e completamente decodificabile — ogni parte porta informazione precisa su cosa si sta misurando e in che contesto.

### Struttura generale

```
{prefisso_trasformata}_{feature}_{param_trasformata} {operatore} {soglia}
```

Le tre parti (prefisso, feature, parametro) si leggono come una frase:
> **"la [trasformata] su [finestra/lag] di [feature] è [sotto/sopra] [soglia]"**

---

### Prefisso della trasformata

Il prefisso indica quale trasformazione temporale è stata applicata alla feature.

| Prefisso | Trasformata | Parametro finale | Esempio |
|---|---|---|---|
| *(nessuno)* | Identità — valore grezzo scale-free | — | `close_rsi_25 < 30.5` |
| `pr_` | Rolling Pctrank — posizione nel range delle ultime W barre | `_{W}` | `pr_close_rsi_25_96 < 0.10` |
| `zs_` | Rolling Z-score — distanza in σ dalla media delle ultime W barre | `_{W}` | `zs_close_rsi_25_48 < -1` |
| `delta_` | Delta — variazione assoluta rispetto a L barre precedenti | `_{L}` | `delta_close_rsi_25_3 < -4.11` |

- **W** (finestra) ∈ {48, 96, 168} — ore di lookback per pctrank e z-score
- **L** (lag) ∈ {1, 3, 6, 12} — barre di ritardo per il delta

---

### Feature: arity 1 (feature native)

Feature native del Variable Catalog, senza operazioni combinatorie.

```
{sorgente}_{indicatore}_{param_indicatore}
```

| Parte | Significato | Esempio |
|---|---|---|
| `sorgente` | Colonna base del KPI (`close`, `volume`, …) | `close` |
| `indicatore` | Tipo di indicatore (`rsi`, `sma`, `ema`, `bb`, …) | `rsi` |
| `param_indicatore` | Parametro dell'indicatore (periodo, lunghezza) | `25` |

**Esempi:**

| Nome colonna | Sorgente | Indicatore | Param |
|---|---|---|---|
| `close_rsi_25` | `close` | `rsi` | 25 periodi |
| `close_sma_09` | `close` | `sma` | 9 periodi |
| `color` | — | intero {-1, 0, 1} (colore candela) | — |

**Esempi di eventi con feature native:**

```
close_rsi_25 < 30.5
    → RSI(25) grezzo sotto 30.5 (p10 della distribuzione storica)
    → Trasformata: identità (scale-free confermato)

pr_close_rsi_25_96 < 0.135
    → RSI(25) negli ultimi 96 valori è nel 13.5° percentile
    → Trasformata: pctrank, finestra W=96

zs_close_rsi_25_48 < -1
    → RSI(25) è 1σ sotto la media delle ultime 48 barre
    → Trasformata: z-score, finestra W=48

delta_close_rsi_25_3 < -4.11
    → RSI(25) è calato di almeno 4.11 punti rispetto a 3 barre fa
    → Trasformata: delta, lag L=3
```

---

### Feature: arity 2 (feature derivate binarie)

Operazioni su **due feature della stessa famiglia semantica** (stessa unità di misura). Il nome è costruito concatenando le due sorgenti.

#### `ratio_` — rapporto tra due indicatori

```
ratio_{base}_{indicatore_A}{param_A}_{indicatore_B}{param_B}
```

Misura **quanto A è sopra o sotto B** in termini moltiplicativi. Valore intorno a 1; > 1 significa A sopra B.

| Nome colonna | Formula | Semantica |
|---|---|---|
| `ratio_close_rsi14_rsi25` | `close_rsi_14 / close_rsi_25` | RSI(14) relativo a RSI(25) |
| `ratio_close_sma09_sma25` | `close_sma_09 / close_sma_25` | SMA veloce / SMA lenta |

**Esempi di eventi:**

```
pr_ratio_close_rsi14_rsi25_96 < 0.135
    → Il rapporto rsi14/rsi25 è nel 13.5° percentile degli ultimi 96 valori
    → Segnala: rsi14 è insolitamente basso rispetto a rsi25

ratio_close_sma09_sma25 < 0.997
    → SMA(9) è almeno 0.3% sotto SMA(25)  [valore assoluto — solo se scale-free]
```

#### `spread_` — distanza percentuale tra prezzo e media mobile

```
spread_{base}_{media_mobile}{param}
```

Formula: `(close - MA) / MA`. Misura quanto il prezzo si discosta dalla sua media mobile. Centrato su 0; negativo = prezzo sotto la MA.

| Nome colonna | Formula | Semantica |
|---|---|---|
| `spread_close_sma09` | `(close - close_sma_09) / close_sma_09` | Distanza % da SMA(9) |
| `spread_close_sma25` | `(close - close_sma_25) / close_sma_25` | Distanza % da SMA(25) |

**Esempi di eventi:**

```
pr_spread_close_sma09_96 > 0.864
    → La distanza % da SMA(9) è nell'86.4° percentile delle ultime 96 barre
    → Segnala: il prezzo è insolitamente sopra la sua media veloce

pr_spread_close_sma25_168 < 0.10
    → La distanza % da SMA(25) è nel 10° percentile delle ultime 168 barre
    → Segnala: il prezzo è insolitamente vicino o sotto la media lenta
```

#### `diffnorm_` — differenza normalizzata tra due indicatori

```
diffnorm_{base}_{indicatore_A}{param_A}_{indicatore_B}{param_B}
```

Formula: `(A - B) / std(A - B)`. È lo z-score della differenza — misura **quanto è anomala la divergenza** tra i due indicatori. Centrato su 0; distribuito circa come N(0,1).

| Nome colonna | Formula | Semantica |
|---|---|---|
| `diffnorm_close_rsi14_rsi25` | `(rsi14 - rsi25) / std(rsi14-rsi25)` | Divergenza normalizzata RSI(14) vs RSI(25) |
| `diffnorm_close_sma09_sma25` | `(sma09 - sma25) / std(sma09-sma25)` | Divergenza normalizzata SMA veloce vs lenta |

**Esempi di eventi:**

```
delta_diffnorm_close_rsi14_rsi25_1 < -0.626
    → La divergenza normalizzata rsi14-rsi25 è calata di 0.626σ in 1 barra
    → Segnala: rsi14 si sta avvicinando rapidamente a rsi25 dal basso

zs_diffnorm_close_rsi14_rsi25_96 > 1
    → La divergenza normalizzata è 1σ sopra la media delle ultime 96 barre
    → Segnala: rsi14 è insolitamente sopra rsi25 nel contesto recente
```

---

### Feature: arity 3 (feature derivate ternarie)

Operazioni su tre feature con struttura `(lower, value, upper)`. Producono una posizione relativa in [0, 1].

#### `bb_pct_b_` — posizione nelle Bande di Bollinger

```
bb_pct_b_{base}_{param_bb}
```

Formula: `(close - bb_lower) / (bb_upper - bb_lower)`. 0 = banda inferiore, 1 = banda superiore.

```
pr_bb_pct_b_close_20 < 0.05
    → Il prezzo è nel 5° percentile della sua posizione nelle BB degli ultimi 96 valori
    → Segnala: prezzo insolitamente vicino (o sotto) la banda inferiore
```

#### `pos_` — posizione nel range min/max rolling

```
pos_{base}_range{param}
```

Formula: `(close - min_N) / (max_N - min_N)`. 0 = minimo delle N barre, 1 = massimo.

```
pr_pos_close_range24_96 < 0.08
    → La posizione del prezzo nel range 24h è nell'8° percentile delle ultime 96 barre
    → Segnala: il prezzo è insolitamente vicino al minimo recente
```

---

### Operatore e soglia

L'operatore deriva dalla **direzione** dell'evento (configurata dal Threshold Catalog):

| Operatore | Direzione | Significato |
|---|---|---|
| `<` | `below` | Evento attivo quando la serie è **sotto** la soglia |
| `>` | `above` | Evento attivo quando la serie è **sopra** la soglia |
| `crosses_below` | crossing `below` | Evento attivo **solo** nella barra di discesa sotto la soglia |

Per la soglia stessa, il suffisso `threshold_type` nel componente indica come è stata calcolata:

| `threshold_type` | Come è calcolata la soglia |
|---|---|
| `distributional_p03` … `distributional_p15` | Percentile della serie trasformata in-sample |
| `distributional_p85` … `distributional_p97` | Percentile della serie trasformata in-sample |
| `theoretical_z-2.0` … `theoretical_z2.0` | Valore fisso (solo per z-score) |
| `binary_native` | Valore massimo della colonna binaria |
| `categorical_onehot` | Sempre 1.0 (one-hot) |

---

### Composizioni AND

Le composizioni AND concatenano due (o tre) espressioni con ` AND `:

```
{espressione_1} AND {espressione_2}
```

**Regole di lettura:**
- Ogni sub-espressione si decodifica autonomamente con le regole sopra
- Le due sub-espressioni sono quasi sempre su **trasformate ortogonali** della stessa feature (identità + pctrank, pctrank + zscore, …) oppure su **feature semanticamente diverse**
- Il gate richiede che la condizione AND sia vera contemporaneamente: entrambe le condizioni devono essere soddisfatte nella stessa barra

**Esempio completo:**

```
pr_color_48 > 0.78125 AND zs_color_48 > 1
```

| Parte | Decodifica |
|---|---|
| `pr_color_48` | Pctrank della colonna `color` su finestra 48 barre |
| `> 0.78125` | La serie è nell'84° percentile (cioè il suo pctrank è alto) |
| `zs_color_48` | Z-score della colonna `color` su finestra 48 barre |
| `> 1` | La serie è oltre 1σ sopra la media recente |
| **AND** | Entrambe le condizioni devono essere vere nella stessa barra |
| **Lettura** | Il colore candela è insolitamente alto sia in termini di rango (top 22%) che di deviazione standard (+1σ) nelle ultime 48 barre |

```
pr_color_48 > 0.78125 AND delta_diffnorm_close_rsi14_rsi25_1 > 0.448
```

| Parte | Decodifica |
|---|---|
| `pr_color_48 > 0.78125` | Colore candela nel top 22% degli ultimi 48 valori |
| `delta_diffnorm_close_rsi14_rsi25_1 > 0.448` | La divergenza normalizzata rsi14-rsi25 è aumentata di almeno 0.448σ in 1 barra |
| **Lettura** | Candela rialzista forte (top 22% di rango) accompagnata da un accelerazione della divergenza RSI veloce/lento |

---

### Cheat sheet rapido

```
pr_close_rsi_25_96 < 0.135
│  │     │      │  └──── W=96 (finestra pctrank)
│  │     │      └─────── param indicatore = 25
│  │     └────────────── indicatore = rsi
│  └──────────────────── sorgente = close
└─────────────────────── trasformata = pctrank (pr_)

                  < 0.135  →  sotto il 13.5° percentile delle ultime 96 barre

─────────────────────────────────────────────────────────────────────────────

delta_diffnorm_close_rsi14_rsi25_1 > 0.448
│      │        │     │      │     └── lag=1 (delta di 1 barra)
│      │        │     │      └──────── indicatore B = rsi25
│      │        │     └─────────────── indicatore A = rsi14
│      │        └───────────────────── sorgente = close
│      └────────────────────────────── operazione arity-2 = diffnorm
└───────────────────────────────────── trasformata = delta

                  > 0.448  →  la differenza normalizzata è salita di 0.448σ in 1 barra

─────────────────────────────────────────────────────────────────────────────

zs_diffnorm_close_rsi14_rsi25_168 < -1
│   │        │     │      │       └── W=168 (finestra zscore)
│   │        │     │      └────────── indicatore B = rsi25
│   │        │     └───────────────── indicatore A = rsi14
│   │        └─────────────────────── sorgente = close
│   └──────────────────────────────── operazione arity-2 = diffnorm
└──────────────────────────────────── trasformata = zscore (zs_)

                  < -1   →  la divergenza è 1σ sotto la media delle ultime 168 barre
```

---

## 11. Esempio end-to-end: da `close_rsi_25` a `rsi25_pr96 < 0.10`

Traccia completa del percorso attraverso i cinque step, usando `close_rsi_25` come feature di partenza. Il modulo non conosce RSI, non conosce oversold — sa solo che ha una serie numerica e un catalogo di trasformazioni da applicare.

```
STEP 1 — Feature Generation
  close_rsi_25 è già normalizzata [0–100]
  → Arietà 1, operazione Identità
  → Nessuna feature derivata necessaria
  → Entra direttamente nel catalogo esteso come close_rsi_25

STEP 2 — Transform Layer
  Il modulo applica tutte le trasformazioni disponibili:

  Identità   → close_rsi_25           range [10, 88]
  Pctrank 48 → pr_close_rsi_25_48     range [0, 1]
  Pctrank 96 → pr_close_rsi_25_96     range [0, 1]   ← questa è rsi25_pr96
  Pctrank 168→ pr_close_rsi_25_168    range [0, 1]
  Zscore  48 → zscore_rsi25_48        range [-4, +4]
  Zscore  96 → zscore_rsi25_96        range [-4, +4]
  Zscore 168 → zscore_rsi25_168       range [-4, +4]
  Delta lag1 → delta_rsi25_lag1       range [-20, +20]
  Delta lag3 → delta_rsi25_lag3       range [-30, +25]
  Delta lag6 → delta_rsi25_lag6       range [-36, +30]
  Delta lag12→ delta_rsi25_lag12      range [-40, +35]

STEP 3 — Event Generation
  Per ogni serie trasformata, applica soglie dal Threshold Catalog:

  Da Identità   → 9 soglie dist. → 9 eventi Threshold + 6 Crossing = 15 eventi
  Da Pctrank×3  → 8 soglie dist. → 24 eventi Threshold
  Da Zscore×3   → 6 soglie teor. → 18 eventi Threshold
  Da Delta×4    → 6 soglie dist. → 24 eventi Threshold
  ─────────────────────────────────────────────────────────────────
  Totale: ~81 eventi candidati grezzi da close_rsi_25

STEP 4 — Consistency Gate
  Applicato a tutti gli 81 eventi:

  close_rsi_25 < 25.8  (p05 di identità)  → ❌ FAIL  conc=0.58
  close_rsi_25 < 29.1  (p08 di identità)  → ❌ FAIL  conc=0.45
  close_rsi_25 < 30.5  (p10 di identità)  → ✅ PASS  N=876  mesi=12  conc=0.40
  pr_rsi25_96 < 0.10                       → ✅ PASS  N=1017 mesi=12  conc=0.10
  zscore_rsi25_96 < -1.5                   → ✅ PASS  N=743  mesi=12  conc=0.10
  delta_rsi25_6 < -14.2                    → ✅ PASS  N=438  mesi=12  conc=0.14
  ...  (altri ~35 eventi passano)

STEP 5 — AND Composition
  Combina gli eventi che passano il gate:

  close_rsi_25 < 30.5  AND  pr_rsi25_96 < 0.10
  → N=329  mesi=12  conc=0.20  tpm=27.4  → ✅ PASS
  → Event Candidate generato   ← proto-RI_01

  close_rsi_25 < 30.5  AND  zscore_rsi25_96 < -1.5
  → N=253  mesi=12  conc=0.18  tpm=21.1  → ✅ PASS
  → Event Candidate generato   (variante)

  ...  (altri ~20 candidati composti passano)

OUTPUT
  ~55 Event Candidates da close_rsi_25
  → tutti passano a Alpha Discovery
  → Alpha Discovery misurerà IC, lift, Cohen's d
  → Solo lì si scoprirà che il proto-RI_01 ha lift +18pp
```

Il modulo ha prodotto `close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10` come uno dei ~55 candidati — senza sapere che il RSI fosse rilevante, senza conoscere il mercato, senza aver mai guardato il prezzo futuro.

È diventato il nucleo di RI_01 solo perché Alpha Discovery ha misurato che, tra tutti i candidati ricevuti, aveva il lift più alto (+18pp) combinato a distribuzione mensile uniforme (zero mesi vuoti). La soglia `30.5` è quella prodotta dal Threshold Catalog di Event Discovery (p10 della distribuzione storica di `close_rsi_25`) e viene propagata invariata nell'Alpha Contract e nella regola finale — Alpha Discovery non modifica le soglie.

---

*Event Discovery Module — FORGE (Feature-Oriented Rule Generation Engine) · Versione 2.2 · Giugno 2026*
*Status: Draft · Parte di FORGE v1.0*
