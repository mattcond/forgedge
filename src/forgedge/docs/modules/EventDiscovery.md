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
│  Criteri: frequenza (tpm),             │
│           potenza (n. episodi),        │
│           dispersione (Poisson-test)   │
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

Il Feature Generator non esplora tutte le combinazioni possibili tra tutte le colonne del catalogo. Il vincolo che tiene lo spazio combinatorio sotto controllo è uno solo:

**Stesso cluster semantico** — le coppie arietà-2 "generiche" (ratio/spread/diffnorm) si generano solo tra colonne che condividono la stessa famiglia (stessa base e stesso indicatore: EMA con EMA, SMA con SMA, RSI con RSI, ecc.), oppure tra un prezzo e una sua media mobile, o tra `volume` e le sue medie mobili. All'interno di ogni famiglia le coppie sono **esaustive** — ogni colonna viene accoppiata con ogni altra colonna della stessa famiglia (ordinate per periodo, "veloce" contro "lenta") — quindi la dimensione dello spazio dipende solo da quante colonne condividono una famiglia (tipicamente 3-6 periodi per indicatore), non da un filtro statistico aggiuntivo a valle del raggruppamento.

**Non esiste un filtro basato su IC/correlazione.** Nessuna soglia `|IC univariato| > threshold` seleziona quali coppie generare — Event Discovery non osserva mai il forward return (invariante #1 della pipeline FORGE), quindi un filtro IC, che richiederebbe di misurare la correlazione con un target, non avrebbe comunque posto in questo modulo.

### Famiglie arietà-2 aggiuntive (sempre attive salvo dove indicato)

Oltre al raggruppamento generico per famiglia semantica sopra, cinque famiglie arietà-2 dedicate e più mirate ampliano il catalogo, ciascuna pensata per una forma di pattern che il raggruppamento generico non può esprimere:

| Famiglia | Metodo | Colonne prodotte | Condizione |
|---|---|---|---|
| Cross-column/cross-time OHLC | `_generate_lag_cross` | `ratio_{a}_{b}_lag{N}`, `spread_{a}_{b}_lag{N}` — confronta una base OHLC con un'altra base OHLC ritardata di N barre (es. "close di oggi vs low di ieri") | Sempre attiva; N ∈ {1,3,6,12} (gli stessi lag del Delta) |
| Indicatore vs base OHLC ritardata | `_generate_indicator_lag_cross` | `ratio_{indicatore}_{base}_lag{N}` — indicatori a scala di prezzo (SMA/EMA/WMA/HMA) contro una base OHLC ritardata | Sempre attiva; lag di default `(1, 3)`, configurabile via `DiscoveryConfig.indicator_lag_cross_lags` |
| MACD vs signal | `_generate_macd_pairs` | `ratio_{base}_macd{fast}_{slow}_signal`, `diffnorm_{base}_macd{fast}_{slow}_signal` | Solo se `close_macd_{fast}_{slow}` e la sua signal line sono presenti nel KPI Table |
| Prezzo vs volume | `_generate_price_volume_pairs` | `ratio_close_ret{N}_volume_ret{N}`, `diffnorm_close_ret{N}_volume_ret{N}` — variazione % prezzo contro variazione % volume, stesso periodo | Solo se il KPI Table include una colonna di return sul volume (non generata di default da `build_features()`) |
| Geometria candela | `_generate_candle_geometry_pairs` | coppie tra `body`/`upper_wick`/`lower_wick`/`close_pos`/`range_pct`/`gap`, e ciascuna contro `close_natr_{N}` | Solo se `candle_features()` è stata applicata al KPI Table (per NATR: solo se l'ATR è abilitato in `build_features()`) |

Le prime due famiglie sono sempre attive e non richiedono colonne opzionali nel KPI Table (agiscono direttamente su `open`/`high`/`low`/`close`); le ultime tre attivano solo se le rispettive colonne prerequisito sono presenti — altrimenti sono no-op silenziosi, non errori.

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

> **Restrizione importante:** gli eventi crossing si generano **esclusivamente**
> per la trasformazione **identità** e **solo per la direzione `below`**
> (`serie_t < soglia`, discesa sotto soglia). Non esiste un `crosses_above`
> nel pool generato, e non esistono crossing su pctrank, z-score o delta.
> La ragione è strutturale, non una scelta arbitraria di scope: una soglia
> assoluta di crossing è priva di senso su pctrank/z-score, che sono
> ri-ancorati ad ogni barra dalla propria finestra rolling (non c'è un
> "attraversamento" stabile da rilevare), e sul delta, che oscilla già
> intorno allo zero per costruzione. Solo l'identità mantiene una soglia
> statica coerente nel tempo, e solo la direzione bassa è stata definita
> come caso d'uso (rilevare l'ingresso in ipervenduto/estremo basso) — la
> direzione alta non produce eventi nel pool.

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

Un evento con 3 attivazioni all'anno non è interessante — non c'è abbastanza storia per valutarlo. Un evento le cui attivazioni sono statisticamente incompatibili con un processo casuale alla propria stessa frequenza è probabilmente regime-dipendente. Il gate scarta questi casi prima di passare il controllo ad Alpha Discovery.

Il gate opera in una di due **modalità di conteggio** (`GateParams.event_counting`), che decidono cosa viene contato per i criteri di frequenza e dispersione:

- **`"episode"`** (default) — conta **episodi**: run massimali di barre attive consecutive (con una tolleranza di `episode_gap` barre di buco). Uno stato persistente di 3-5 barre (es. `RSI < 30` per 4 barre di fila) conta come **un solo episodio**, non quattro attivazioni — questo elimina l'artefatto per cui uno stato persistente gonfia artificialmente la varianza mensile e viene ingiustamente respinto.
- **`"bar"`** — comportamento storico, conta le barre attivate una per una. Riproduce esattamente il comportamento pre-#134, mantenuto per compatibilità.

### I tre criteri (modalità `"episode"`, default)

```
CRITERIO 1 — Frequenza minima (rate)
  n_episodes / n_months >= min_tpm
  Default: 0.5 episodi/mese ("almeno un episodio ogni due mesi")
  Scopo:   filtro di frequenza operativa — un evento troppo raro non ha
           abbastanza storia ricorrente da consegnare ad Alpha Discovery

CRITERIO 2 — Potenza statistica (power)
  n_episodes >= min_episodes
  Default: 10 episodi
  Scopo:   pavimento assoluto e indipendente dal rate sul numero di episodi
           indipendenti disponibili — un evento può soddisfare il rate
           minimo su uno storico lungo senza aver mai accumulato abbastanza
           episodi per una stima statistica affidabile a valle.
           Applicato solo in-sample: su un fold di walk-forward un conteggio
           assoluto equivarrebbe a un requisito di rate inversamente
           proporzionale alla lunghezza del fold — i fold usano invece un
           minimo di Poisson calibrato sul rate osservato dell'evento

CRITERIO 3 — Dispersione episodica (test contro il rumore di Poisson)
  episode_index_of_dispersion <= poisson_floor(n_months) x dispersion_margin
  Default: dispersion_margin = 1.3
  Scopo:   verificare che la distribuzione mensile degli episodi non sia
           statisticamente incompatibile con un processo casuale (Poisson)
           alla frequenza osservata. NON è un tetto assoluto di
           concentrazione: è un confronto contro il rumore atteso a quella
           stessa frequenza e su quell'orizzonte temporale — un evento raro
           su uno storico breve produce naturalmente più varianza mensile
           anche se genuinamente casuale, e il floor lo tiene in conto
```

`episode_index_of_dispersion` è l'Indice di Dispersione (`Var/Mean`) dei conteggi mensili di episodi — per un processo di Poisson puro vale 1 in aspettativa. `poisson_floor(n_months)` è il quantile 95° della distribuzione χ² con `n_months - 1` gradi di libertà, diviso per gli stessi gradi di libertà: il valore più alto di Indice di Dispersione ancora compatibile con la casualità al 5% di significatività. `dispersion_margin` è quanto **oltre** quel floor statisticamente difendibile un preset è disposto a tollerare — un valore vicino a 1 (es. 1.05) resta aderente a un processo quasi-Poisson, un valore alto (es. 3.0) tollera clustering deliberatamente "meno che casuale".

> **Perché un floor invece di una soglia fissa?** Il floor di Poisson dipende
> solo dal numero di mesi nello storico (non dall'asset, non dalla
> frequenza dell'evento), e su qualunque orizzonte realistico da 6 a 60 mesi
> resta nell'intervallo ~1.3–2.2. Una soglia fissa di dispersione — come il
> vecchio tetto di concentrazione — avrebbe respinto sistematicamente gli
> eventi rari su storici brevi anche quando la loro varianza mensile è
> esattamente quella attesa dal caso. Il floor scala con l'orizzonte
> temporale, il tetto fisso no.

**Esempio — soglia effettiva di dispersione per diversi orizzonti** (`dispersion_margin` di default = 1.3):

| Mesi nel dataset | df = mesi − 1 | `poisson_floor` | `eff_max_dispersion` (floor × 1.3) |
|---:|---:|---:|---:|
| 6 | 5 | 2.21 | 2.87 |
| 12 | 11 | 1.79 | 2.32 |
| 24 | 23 | 1.53 | 1.99 |
| 60 | 59 | 1.32 | 1.72 |

**Esempio numerico completo** — un evento con 45 episodi su uno storico di 12 mesi (`eff_max_dispersion` = 2.32 dalla tabella sopra):

- Criterio 1 (rate): `45 / 12 = 3.75` episodi/mese `>= 0.5` → PASS
- Criterio 2 (potenza): `45 >= 10` → PASS
- Criterio 3 (dispersione): se i 45 episodi sono ragionevolmente distribuiti nei 12 mesi, `episode_index_of_dispersion` risulta, poniamo, `1.85` → `1.85 <= 2.32` → PASS. Lo **stesso** evento con gli stessi 45 episodi ma concentrati quasi tutti in 3-4 mesi su 12 produrrebbe invece un `episode_index_of_dispersion` più alto, poniamo `3.10` → `3.10 > 2.32` → **FAIL**, con motivo riportato `"episode dispersion: ID=3.10 > 2.32"`.

La domanda che il terzo criterio pone non è più "quanto è concentrato l'evento in termini assoluti" ma "questa concentrazione è più di quella che il puro caso produrrebbe a questa frequenza, su questo storico".

### Modalità `"bar"` (storica)

Per compatibilità retroattiva, `event_counting="bar"` conta le barre attivate invece degli episodi e usa **due** criteri soltanto, senza floor di Poisson e senza criterio di potenza:

```
CRITERIO 1 — Frequenza (bar-level)
  n_activations / n_months >= min_tpm

CRITERIO 2 — Dispersione (bar-level, tetto assoluto)
  index_of_dispersion <= max_dispersion
  Default: 1.5
```

Questa modalità riproduce esattamente il comportamento storico del gate, incluso l'artefatto per cui uno stato persistente multi-barra gonfia la varianza mensile — per questo non è più la modalità di default.

### Parametri del gate

| Parametro | Default | Modalità | Note |
|---|---|---|---|
| `min_tpm` | 0.5 | entrambe | episodi/mese (`"episode"`) o barre/mese (`"bar"`) |
| `min_episodes` | 10 | `"episode"` | pavimento assoluto — solo in-sample, non applicato su un fold di walk-forward |
| `dispersion_margin` | 1.3 | `"episode"` | moltiplicatore sopra il floor di Poisson; non letto in modalità `"bar"` |
| `max_dispersion` | 1.5 | `"bar"` | tetto assoluto di Indice di Dispersione; non letto in modalità `"episode"` |
| `episode_gap` | 1 | `"episode"` | barre di buco ancora tollerate all'interno dello stesso episodio |
| `event_counting` | `"episode"` | — | `"episode"` o `"bar"` |

> I parametri del gate non dipendono dall'asset — `min_tpm` e `dispersion_margin`
> sono invarianti di rate/rapporto e si trasferiscono senza riscalatura tra
> in-sample e out-of-sample. `min_episodes`, essendo un conteggio assoluto, non
> si trasferisce allo stesso modo: applicato verbatim a un fold di walk-forward
> più corto dello storico di discovery implicherebbe un requisito di rate molto
> più severo — per questo è applicato solo in-sample.

### `event_distribution_report` — diagnostica sempre calcolata

`EventDiscovery.run()` popola sempre un attributo pubblico `event_distribution_report: str` (`None` prima della chiamata a `run()`), indipendentemente dall'esito della ricerca. È un riepilogo testuale della distribuzione di tpm e dispersione osservata su **ogni** candidato grezzo che il gate ha valutato (prima della composizione AND), confrontata con le soglie configurate.

Il problema che risolve: `config_report()` è cieco ai dati per costruzione (deve risolvere la configurazione senza un DataFrame), quindi può rilevare solo un'incoerenza configurazione-contro-configurazione, mai una configurazione-contro-statistiche-reali-del-candidato. Un preset può essere internamente coerente e comunque respingere ogni candidato che un asset specifico produce — e prima di questa diagnostica, l'unico segnale era una riga di log con il solo conteggio finale, indistinguibile da "la pipeline è rotta".

Quando la quota di candidati che superano il gate scende sotto il 15%, il report aggiunge anche un suggerimento di parametri concreti, calcolato sulla mediana osservata:

```
M1 Event Discovery — 2400 candidati generati, 312 superano il Consistency Gate (13.0%).
tpm osservato: mediana=0.35 (soglia min_tpm=0.5, 61.2% sotto soglia).
dispersione osservata: mediana=2.10 (soglia effettiva=2.32, 38.4% sopra soglia).
Meno del 15% dei candidati generati supera il Consistency Gate (312/2400 = 13.0%).
Prova questi parametri (mediana osservata su tpm e dispersione): min_tpm<=0.35, dispersion_margin>=1.17.
```

La riga di log dello stage M1 di `forge()` riporta lo stesso testo — non serve chiamare `EventDiscovery` a mano per vederlo, appare già nei log di un run standard.

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
n_activations:        329
n_active_months:      12
zero_months:          0
max_monthly_share:    0.20
mean_tpm:             27.4
index_of_dispersion:  1.85

consistency_gate:
  result:              "PASS"
  event_counting:      "episode"
  min_tpm:             0.5
  min_episodes:        10
  dispersion_margin:   1.3
  n_episodes:          45
  episode_index_of_dispersion: 1.85
  eff_max_dispersion:  2.32

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

### Feature: arity 2 (famiglie dedicate, sempre attive o condizionali)

Oltre alle coppie generiche sopra, le cinque famiglie arietà-2 dedicate (§4) hanno le proprie convenzioni di nome:

| Prefisso/pattern | Famiglia | Esempio | Semantica |
|---|---|---|---|
| `ratio_{a}_{b}_lag{N}` / `spread_{a}_{b}_lag{N}` | Cross-column/cross-time OHLC | `ratio_close_low_lag1` | `close[t] / low[t-1]` — confronto tra basi OHLC diverse a tempi diversi |
| `ratio_{indicatore}_{base}_lag{N}` | Indicatore vs base OHLC ritardata | `ratio_close_sma_12_low_lag3` | `close_sma_12[t] / low[t-3]` — un indicatore a scala di prezzo contro una base OHLC di N barre prima |
| `ratio_{base}_macd{fast}_{slow}_signal` / `diffnorm_{base}_macd{fast}_{slow}_signal` | MACD vs signal | `ratio_close_macd12_26_signal` | Linea MACD relativa alla propria signal line |
| `ratio_close_ret{N}_volume_ret{N}` / `diffnorm_close_ret{N}_volume_ret{N}` | Prezzo vs volume | `ratio_close_ret12_volume_ret12` | Variazione % prezzo relativa alla variazione % volume, stesso periodo |
| coppie tra `body`/`upper_wick`/`lower_wick`/`close_pos`/`range_pct`/`gap`, e ciascuna vs `close_natr_{N}` | Geometria candela | `ratio_body_upper_wick` | Geometria della candela tra loro o contro la volatilità (NATR) |

Le prime due famiglie sono etichettate con il suffisso `_lag{N}` per distinguerle esplicitamente da un confronto allo stesso istante — il numero indica quante barre indietro è stata presa la seconda serie, non un parametro del Transform Layer (Delta). Restano soggette solo alla trasformazione identità (vedi §4): non vengono ulteriormente combinate con pctrank/zscore/delta.

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

> Non esiste un `crosses_above` nel pool generato — i crossing si generano solo per la trasformazione identità e solo in direzione `below` (vedi §6). Un `>` senza prefisso `crosses_` è sempre un Threshold persistente, mai un crossing.

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
  Applicato a tutti gli 81 eventi (modalità "episode", 12 mesi di storico
  → eff_max_dispersion = 2.32, vedi §7):

  close_rsi_25 < 25.8  (p05 di identità)  → ❌ FAIL  n_attivazioni=438  n_episodi=41  ID_episodico=2.95 (> 2.32)
  close_rsi_25 < 29.1  (p08 di identità)  → ❌ FAIL  n_attivazioni=702  n_episodi=58  ID_episodico=2.68 (> 2.32)
  close_rsi_25 < 30.5  (p10 di identità)  → ✅ PASS  n_attivazioni=876  n_episodi=73  ID_episodico=1.85
  pr_rsi25_96 < 0.10                       → ✅ PASS  n_attivazioni=1017 n_episodi=85  ID_episodico=1.05
  zscore_rsi25_96 < -1.5                   → ✅ PASS  n_attivazioni=743  n_episodi=62  ID_episodico=1.10
  delta_rsi25_6 < -14.2                    → ✅ PASS  n_attivazioni=438  n_episodi=37  ID_episodico=1.40
  ...  (altri ~35 eventi passano)

  Le prime due falliscono non per un tetto di concentrazione ma perché la
  distribuzione mensile dei loro episodi è statisticamente incompatibile
  con un processo casuale alla frequenza osservata (ID_episodico oltre il
  floor di Poisson): la soglia identità p05/p08 si attiva quasi solo nei
  periodi di panico acuto, concentrando gli episodi in meno mesi di quanto
  il caso spiegherebbe a quella frequenza.

STEP 5 — AND Composition
  Combina gli eventi che passano il gate:

  close_rsi_25 < 30.5  AND  pr_rsi25_96 < 0.10
  → n_attivazioni=329  n_episodi=28  ID_episodico=1.35  → ✅ PASS
  → Event Candidate generato   ← proto-RI_01

  close_rsi_25 < 30.5  AND  zscore_rsi25_96 < -1.5
  → n_attivazioni=253  n_episodi=22  ID_episodico=1.20  → ✅ PASS
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
