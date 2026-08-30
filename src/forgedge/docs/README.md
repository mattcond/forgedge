# FORGE
### Feature-Oriented Rule Generation Engine

> Sistema open source per la scoperta, validazione e formalizzazione
> di regole di trading algoritmico su dati di mercato storici.
> FORGE prende in input una tabella di KPI precomputati su candele OHLCV
> e produce regole operative validate, pronte per essere integrate
> in qualsiasi sistema di esecuzione.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

---

## Indice

1. [Cos'è FORGE](#1-cosè-forge)
2. [Architettura della Pipeline](#2-architettura-della-pipeline)
3. [I Moduli](#3-i-moduli)
4. [Input: la KPI Table](#4-input-la-kpi-table)
5. [Output: la Regola Validata](#5-output-la-regola-validata)
6. [Gli Artefatti Intermedi](#6-gli-artefatti-intermedi)
7. [Ciclo di Vita di una Regola](#7-ciclo-di-vita-di-una-regola)
8. [Principi Fondamentali](#8-principi-fondamentali)
9. [Integrazione con Sistemi di Esecuzione](#9-integrazione-con-sistemi-di-esecuzione)
10. [Roadmap](#10-roadmap)
11. [Documentazione di Dettaglio](#11-documentazione-di-dettaglio)
12. [Licenza](#12-licenza)

---

## 1. Cos'è FORGE

FORGE è un motore di ricerca quantitativa che trasforma dati di mercato
storici in regole di trading operative validate statisticamente.

Il nome riflette il processo: come una fucina metallurgica trasforma
il minerale grezzo in uno strumento lavorato attraverso fasi successive,
FORGE trasforma una tabella di indicatori tecnici in regole booleane
operative attraverso quattro moduli in sequenza — senza mai fare assunzioni
sul sistema di esecuzione che le utilizzerà.

```
KPI Table                 FORGE                       Regola validata
──────────    ────────────────────────────────────    ───────────────
rsi, ema,  →  Evento → Alpha → Regola → Registry  →  espressione +
bb, vol, …    (scopri)  (misura) (valida) (cataloga)  parametri operativi
```

### Cosa fa

FORGE risolve il problema della **ricerca sistematica di alpha**. Invece
di costruire regole basandosi su intuizioni di mercato e poi validarle,
FORGE inverte il processo: genera sistematicamente tutti gli eventi
booleani possibili da un catalogo di feature, misura il loro potere
predittivo rispetto a un target economico, e seleziona quelli che
producono edge statisticamente significativo.

### Cosa non fa

FORGE non esegue ordini, non gestisce posizioni, non si connette a exchange.
Produce specifiche — regole con parametri validati — che altri sistemi
possono implementare. È un sistema di **ricerca**, non di **esecuzione**.

---

## 2. Architettura della Pipeline

FORGE è organizzato in cinque step logici distribuiti su cinque moduli.
Ogni step riceve l'output del precedente come input formale.

```
KPI Table (input)
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULO 0 — MARKET CONTEXT MODULE                                   │
│                                                                     │
│  Classifica ogni barra per regime di mercato                        │
│  Aggiunge colonne 'regime' + 'regime_stable' alla KPI Table        │
│  Gira una volta — output immutabile per tutta la sessione           │
│  v1.0: EMAProxyClassifier  |  v2.0+: HMM, KMeans, custom          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ KPI Table arricchita (+ regime)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULO 1 — EVENT DISCOVERY                                         │
│                                                                     │
│  Step 1 — Feature Generation                                        │
│           Costruisce feature derivate normalizzate                  │
│           Arietà 1 (unaria): identità, log, inverso                 │
│           Arietà 2 (binaria): ratio, spread pct tra coppie          │
│           Arietà 3 (ternaria): posizione in un intervallo           │
│                                                                     │
│  Step 2 — Transform Layer                                           │
│           Applica trasformazioni temporali a ogni feature           │
│           Identità | Pctrank rolling | Zscore rolling | Delta       │
│                                                                     │
│  ASSEMBLAGGIO E TEST — nessun forward return osservato              │
│                                                                     │
│  Step 3 — Event Generation                                          │
│           Applica soglie alle serie trasformate → eventi booleani   │
│           Threshold (stato) | Crossing (transizione)               │
│           Soglie: distribuzionali (percentili) o teoriche (z-score) │
│                                                                     │
│  Step 4 — Consistency Gate                                          │
│           Filtra per struttura temporale dell'evento (episodi)      │
│           Frequenza minima (tpm) | N episodi minimo                 │
│           Dispersione episodi entro il floor statistico di Poisson  │
│                                                                     │
│  Step 5 — AND Composition                                           │
│           Combina eventi di trasformazioni diverse sulla            │
│           stessa feature o su feature semanticamente distinte       │
│           Riapplica il Consistency Gate sul composto                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Event Candidates
                           │
                           │  VALUTAZIONE — primo contatto con il target
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULO 2 — ALPHA DISCOVERY                                         │
│                                                                     │
│  Step 1 — Definizione del target economico                          │
│           Holding period, target return, base rate                  │
│                                                                     │
│  Step 2 — Analisi struttura di mercato                              │
│           Hurst exponent, autocorrelazione — contesto               │
│                                                                     │
│  Step 3 — IC Measurement                                            │
│           Spearman IC della feature sottostante vs forward return   │
│                                                                     │
│  Step 4 — Win Rate Analysis                                         │
│           WR, lift vs base rate, Cohen's d, t-test                  │
│           per ogni Event Candidate                                  │
│                                                                     │
│  Step 5 — Regime Sensitivity                                        │
│           IC e WR per regime di mercato                             │
│                                                                     │
│  Step 6 — Alpha Scoring (voto, non gate)                            │
│           Composite score, grade A / B / C / D                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Alpha Contract
                           │
                           │  OPERATIVITÀ — primo contatto con fee e fill
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULO 3 — RULE DISCOVERY                                          │
│                                                                     │
│  Backtest con meccanica di ordine realistica                        │
│  (limit order, buy_drop, buy_delay, fee)                            │
│                                                                     │
│  Selezione e raffinamento parametri operativi                       │
│  Validazione statistica (t-test, Deflated Sharpe Ratio)             │
│  Analisi dipendenza dal regime                                      │
│                                                                     │
│  Risponde: EDGE | NON-EDGE | PARTIAL-EDGE | INSUFFICIENT-DATA       │
│  Produce:  regola con parametri operativi validati                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Regola validata
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULO 4 — RULE REGISTRY                                           │
│                                                                     │
│  Registro in-memory per la sessione corrente                        │
│  Matrice Jaccard sulle attivazioni (sovrapposizione temporale)      │
│  Matrice Spearman sui gain (correlazione rendimenti)                │
│  Deduplicazione e marcatura duplicati                               │
│  Export: tabella piatta (CSV/Excel) + report HTML autocontenuto     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
                    Regola esportata
               (espressione + parametri operativi)
```

### Separazione dei domini

FORGE mantiene una separazione netta tra tre domini operativi.
Ogni confine è un vincolo architetturale, non una convenzione:

| Dominio | Moduli | Cosa osserva |
|---|---|---|
| **Struttura temporale** | Event Discovery | Solo distribuzione degli eventi nel tempo |
| **Predittività statistica** | Alpha Discovery | Forward return, IC, win rate |
| **Operatività** | Rule Discovery, Rule Registry | Fee, fill rate, PF, drawdown |

### Risoluzione dei parametri e coerenza di configurazione

Ogni esecuzione di `forge()` inizia con uno step di risoluzione che precede
persino il Modulo 0: viene costruito un `PipelineContext` — la fonte unica
di verità per timeframe, nomi di colonna dello schema, fee e politica
statistica della sessione (`forgedge.resolver`) — e ogni campo di
configurazione lasciato al valore sentinella `UNSET` da chi chiama viene
risolto rispetto a quel contesto, invece di cadere silenziosamente su un
default calibrato per un altro timeframe.

Subito dopo, `config_report()` valuta il bundle di configurazione *già
risolto* alla ricerca di contraddizioni strutturali (es. una finestra di
selezione di Modulo 3 troppo corta per il tasso di arrivo che le è stato
chiesto di sostenere). Con `strict=True` — il default di `forge()` —
un esito `FAIL` solleva un `ValueError` prima ancora che il Modulo 0
inizi: un muro di scarti a valle è indistinguibile da "il segnale è
debole", per cui rifiutarsi di partire è la risposta onesta, non un bug
da aggirare con `strict=False`.

Questo livello (issue #173–#221 e successive) chiude una classe di bug
reale e misurata in cui due impostazioni singolarmente ragionevoli su
moduli diversi risultavano incompatibili tra loro solo quando combinate —
v. `docs/analysis/pipeline_parameter_coherence.md` per l'audit completo
(17 classi di configurazione, 158 campi, 16 findings) che ha motivato
resolver e `config_report()`.

---

## 3. I Moduli

Ogni modulo espone i propri parametri di configurazione (`DiscoveryConfig`,
`AlphaConfig`, `RuleDiscoveryConfig`, ecc.), ma impostarli a mano modulo per
modulo rischia di produrre soglie incoerenti tra loro — ad esempio un evento
ammesso da Event Discovery con `min_tpm=1.0` che Rule Discovery scarta perché
richiede `min_tpm=2.0`. `forge_preset()` risolve questo problema traducendo
un'intenzione dichiarativa in una tripla di configurazioni già coerenti:

```python
from forgedge import forge, forge_preset, PRESETS

# PRESETS == ["sniper", "balanced", "sweep", "burst"]
disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1H", asset="ADA")
result = forge(df, event_discovery_config=disc_cfg,
                alpha_config=alpha_cfg, rule_discovery_config=rd_cfg)
```

I quattro preset incarnano filosofie di ricerca diverse: `sniper` (alta
precisione, soglie strette), `balanced` (compromesso di default), `sweep`
(esplorazione permissiva, pensata per lavorare insieme a `RotationCalibrator`)
e `burst` (adatto a eventi ad alta frequenza). Ogni preset scala `min_tpm`,
`max_dispersion`/`dispersion_margin`, `fdr_q` e le soglie economiche di Rule
Discovery in modo coerente per il timeframe scelto — restano comunque
liberamente sovrascrivibili passando `**overrides`.

### Modulo 0 — Market Context Module

Classifica ogni barra della KPI Table per regime di mercato prima che
qualsiasi modulo di discovery inizi. Arricchisce la KPI Table con la
colonna `regime` — disponibile a tutti i moduli successivi, immutabile
durante la sessione.

**Domanda:** *"in quale contesto di mercato si trova ogni barra?"*

**Input:** KPI Table grezza

**Output:** KPI Table con colonne `regime` e `regime_stable` aggiunte

**v1.0 — EMAProxyClassifier:** ratio EMA veloce / EMA lenta discretizzato
in cinque label (`STRONG_BEAR` → `STRONG_BULL`). Le colonne EMA vengono
cercate nella KPI Table per nome (`{col}_ema_{period:02d}`) — se presenti
vengono usate direttamente, altrimenti calcolate inline (dal solo `close`)
senza lasciare traccia nella KPI Table.

Le finestre EMA `fast`/`slow` vengono **decise dai dati**: il modulo stima
l'half-life del processo Ornstein-Uhlenbeck (analisi di Hurst) e ne ricava
le due finestre per ogni asset. I valori `9` / `25` restano come **fallback**,
usati solo quando l'half-life non converge (serie trending o storia troppo
corta). La risoluzione effettiva è tracciata in `window_resolution`.

**Estensibilità:** interfaccia `RegimeClassifier` pluggabile — in v2.0
si sostituisce `ema_proxy` con `hmm`, `kmeans` o un classificatore custom
cambiando solo il campo `classifier` nella configurazione.

→ Documento di dettaglio: **[Market Context Module](Market_Context_Module.md)**

---

### Modulo 1 — Event Discovery

Genera sistematicamente tutti gli eventi booleani possibili dalla KPI Table
e filtra quelli con struttura temporale stabile, **senza mai osservare
il forward return**.

**Domanda:** *"questo evento ha struttura temporale stabile e non banale?"*

**Input:** KPI Table — tabella con feature native e serie temporali OHLCV

**Output:** Event Candidates — eventi booleani che hanno superato
il Consistency Gate, con statistiche di attivazione temporale.
`EventDiscovery.event_distribution_report` (issue #215) affianca un
riepilogo sempre calcolato della distribuzione di tpm/dispersione
osservata su *tutti* i candidati grezzi valutati dal gate, confrontata
con le soglie configurate — sotto un tasso di sopravvivenza del gate
del 15% suggerisce anche parametri concreti calibrati sulla mediana
osservata, al posto del solo log "0 candidati" altrimenti indistinguibile
da una pipeline rotta.

**Parametri del Consistency Gate** (`GateParams`, modalità di conteggio
`event_counting="episode"` di default — un "episodio" è un run di barre
attive consecutive, non la singola barra):

| Parametro | Significato | Default |
|---|---|---|
| `min_tpm` | Episodi minimi per mese (rate) | 0.5 |
| `min_episodes` | Numero minimo di episodi nel periodo | 10 |
| `dispersion_margin` | Margine sul floor statistico di Poisson per la dispersione degli episodi | 1.3 |
| `episode_gap` | Barre di buco ancora tollerate dentro lo stesso episodio | 1 |

Un evento passa il gate se: (a) il tasso di episodi/mese ≥ `min_tpm`,
(b) il numero di episodi ≥ `min_episodes`, e (c) l'Index of Dispersion
degli episodi resta entro il floor di Poisson al 95% (dipendente dal
numero di mesi) moltiplicato per `dispersion_margin`. Non esiste più
un criterio di copertura mensile minima né di concentrazione massima
in un singolo mese: la vecchia logica a 4 criteri (`MIN_ACT`/`MIN_MONTHS`/
`MAX_CONC`/`MIN_TPM`) è stata sostituita da questo disegno basato su
episodi (issue #134/#205). Una modalità legacy `event_counting="bar"`
resta disponibile, con i due soli criteri `min_tpm` (barre/mese) e
`max_dispersion` (default 1.5).

→ Documento di dettaglio: **[Event Discovery Module](Event_Discovery_Module.md)**

---

### Modulo 2 — Alpha Discovery

Deriva dai dati il target economico di ciascun Event Candidate (orizzonte
h* = argmax |z_h|, l'excess log-return Δ_h = μ_cond − μ_base standardizzato da un
null a rotazione circolare (autocorrelation-robust);
sell_pct = quantile della Maximum Favorable Excursion a h*; direzione = segno
dell'excess Δ_h*, che esclude il drift), lo replica
out-of-sample e ne misura il potere predittivo. Ogni candidato con
direzione determinata diventa un Alpha Contract con voto A–D — le misure
statistiche alimentano il voto, non scartano: Rule Discovery è l'unico
giudice economico.

**Domanda:** *"questo evento predice un target economico — quale, in che
direzione, e con quanta evidenza statistica?"*

**Input:** Event Candidates (nessun parametro economico: il target è
derivato)

**Output:** Alpha Contract — specifica formale con IC, win rate,
lift, Cohen's d, regime sensitivity, alpha score

**Soglie di promozione:**

| Metrica | Soglia minima |
|---|---|
| Lift vs base rate | ≥ +8pp |
| Cohen's d | ≥ 0.15 |
| Significatività | False Discovery Rate (Benjamini-Hochberg), `fdr_q` di default 0.10 — varia per preset: 0.05 sniper, 0.15 balanced, 0.25 sweep, 0.10 burst |
| N casi (floor statistico) | ≥ 10 — costante interna (`_MIN_STATS_CASES`), non un campo configurabile di `PromotionThresholds` |

Il controllo di significatività di default **non** è un semplice t-test
con soglia fissa p<0.05: `PromotionThresholds.use_fdr=True` applica il
controllo FDR di Benjamini-Hochberg sull'intera famiglia di candidati
testati nella sessione, con `fdr_q` come tasso di falsa scoperta
accettato. Il vecchio `max_p_value` resta presente ma è inerte
(`UNSET`) a meno di impostare esplicitamente `use_fdr=False`.

**Rotation null a livello di ricerca:** `forge()` esegue di default
(`fast_null=True`) un `FastRotationNull` sull'intera famiglia di
Alpha Contract promossi — uno shift circolare della maschera di
attivazione ripetuto su molti offset per stimare quanto la statistica
osservata sia spiegabile dal solo caso, correlation-aware sull'intera
superficie di ricerca. Ogni contratto promosso viene annotato con
`rotation_p` e `rotation_threshold`, che alimentano un gate rigido
in Rule Discovery (`SelectionCriteria.max_rotation_p`): un contratto
con `rotation_p` troppo alto non può ricevere un verdetto `EDGE` pieno,
indipendentemente da quanto siano buone le metriche di backtest.

→ Documento di dettaglio: **[Alpha Discovery Pipeline](Alpha_Discovery_Pipeline.md)**

---

### Modulo 3 — Rule Discovery

Traduce l'alpha in una regola operativa e ne valida l'eseguibilità
al netto di fee, fill rate e parametri di rischio.

**Domanda:** *"questo pattern è operazionalmente sfruttabile?"*

**Input:** Alpha Contract

**Output:** Verdetto (`EDGE / NON-EDGE / PARTIAL-EDGE / INSUFFICIENT-DATA`)
+ regola con parametri operativi (`direction`, `entry_mode`, `buy_drop_pct`,
`buy_delay_bar`, `sell_pct`, `target_h`, `fee`)

Il quarto verdetto, `INSUFFICIENT-DATA`, è governato da
`SelectionCriteria.power_gate` (default `True`): quando l'evidenza
out-of-sample non ha potere statistico sufficiente per sostenere un
giudizio (numero di trade effettivi sotto la soglia minima), Rule
Discovery non forza un `EDGE`/`NON-EDGE` — dichiara che i dati non
bastano, distinguendo esplicitamente "non c'è edge" da "non lo sappiamo".

**`entry_mode` di default è `"auto"`**, non `"limit"`: è una valutazione
a due stadi — Stadio 1 misura la regola a un ingresso a mercato (il
verdetto di questo stadio è definitivo: lo Stadio 2 non può mai trasformare
un `NON-EDGE` in edge), lo Stadio 2 ottimizza opzionalmente un limit order
sopra quel risultato. Questo è un fix comportamentale deliberato (issue
#185), non un rinominare cosmetico: valutare solo in modalità `"limit"`
— l'unica modalità che il documento descriveva in precedenza — misura
il prezzo di ingresso invece del segnale, perché uno stop limit non
riempito filtra silenziosamente i trade peggiori dal campione. `"limit"`
resta disponibile esplicitamente per chi lo desidera.

**Metriche di validazione:**

| Metrica | Target |
|---|---|
| Profit Factor | ≥ 2.0 |
| Win Rate | ≥ 55% (`SelectionCriteria.min_win_rate`, 0.50–0.60 a seconda del preset) |
| Mesi zero trade | ≤ 2 su 12 |
| Deflated Sharpe Ratio | ≥ 1.0 |

**Selezione walk-forward di default:** `RuleDiscoveryConfig.selection_mode`
ha default `"walk_forward"` — il punto operativo pubblicato viene scelto
dalle finestre di train del walk-forward, non da un singolo backtest
in-sample. È il comportamento predefinito, non un'opzione da attivare.

**Strumentazione di sovrapposizione dei trade:** `BacktestSummary` porta
`n_episodes`, `mean_concurrent_positions` e `max_concurrent_positions`
(issue #168) per quantificare quante posizioni sono aperte in
contemporanea. `StatisticalValidation.n_effective` (`total_trades /
mean_concurrent_positions`) sostituisce il conteggio nominale dei trade
nei test di significatività: un mucchio di posizioni aperte sulla stessa
finestra temporale non vale come altrettante osservazioni indipendenti.

→ Documento di dettaglio: **[Rule Discovery Pipeline](Rule_Discovery_Pipeline.md)**

---

### Modulo 4 — Rule Registry

Raccoglie tutte le regole validate della sessione in un registro
in-memory, calcola le matrici di correlazione, identifica i duplicati,
esegue il backtest cross-ticker e produce l'output finale.

**Domanda:** *"tra tutte le regole validate in questa sessione,
quali sono distinte, complementari e generiche su più ticker?"*

**Input:** regole validate da Rule Discovery — una per ogni ticker
presente nel CSV di input

**Output:** due file per sessione
- **Tabella piatta** (CSV / Excel) — una riga per regola, colonne
  per ogni ticker (pf, wr, verdetto), flag duplicato, badge genericità
- **Report HTML** autocontenuto — equity curve, monthly breakdown,
  cross-ticker summary, heatmap correlazioni, trade log

**Natura stateless:** il registro viene costruito da zero a ogni sessione.
Non esiste persistenza. La tabella piatta è l'unico artefatto che
l'utente mantiene tra sessioni.

**Deduplicazione:** Jaccard similarity sulle date di attivazione.
Duplicati marcati con flag visibile — non eliminati.

**Cross-ticker backtest:** ogni regola viene testata su tutti i ticker
diversi da quello di origine. Le soglie assolute vengono ricalcolate
sui percentili corrispondenti della distribuzione locale del ticker
target — la struttura logica della regola rimane invariata.

**Classificazione genericità:** `GENERIC` se supera il PF minimo
sulla maggioranza dei ticker, `PARTIAL`, `SPECIFIC` o `ISOLATED`
altrimenti.

→ Documento di dettaglio: **[Rule Registry Module](Rule_Registry_Module.md)**

---

## 4. Input: la KPI Table

FORGE accetta in input una singola tabella — la **KPI Table** — che contiene
candele OHLCV arricchite con indicatori tecnici precomputati. Non richiede
connessioni a exchange, database operativi o API esterne.

### Costruire la KPI Table da OHLCV grezzo (`kpi_builder`)

FORGE non impone di arrivare con una KPI Table già arricchita a mano: il
sottopacchetto `forgedge.kpi_builder` la costruisce a partire dalle sole
candele OHLCV, esposto come API pubblica di primo livello:

```python
from forgedge import build_features, candle_features, lag_features

kpi = build_features(candles, timestamp_col="open_time")   # indicatori base + open_dt
kpi = candle_features(kpi)                                 # geometria candela scale-free
kpi = lag_features(kpi, "close", like="_ema_", periods=[1, 2, 3])
```

`build_features()` calcola gli indicatori base (RSI, EMA/SMA, bande di
Bollinger, ecc.) e la colonna `open_dt`; `candle_features()` aggiunge
geometria scale-free della candela (corpo, ombre, range); `lag_features()`
crea versioni laggate delle colonne indicate; `pattern_features()` (non
mostrato sopra) aggiunge pattern candlestick booleani. Tutte e quattro sono
esportate direttamente da `forgedge`. Restano comunque valide le colonne
`{base}_{indicator}_{period}` costruite altrove, purché seguano la naming
convention (v. sotto) — `kpi_builder` è la via consigliata, non l'unica.

### Schema minimo

```
KPI Table
───────────────────────────────────────────────────────────────────
Colonne obbligatorie:
  open_dt       datetime    timestamp di apertura della candela
  open          float       prezzo di apertura
  high          float       prezzo massimo
  low           float       prezzo minimo
  close         float       prezzo di chiusura
  volume        float       volume scambiato

Colonne consigliate (almeno una famiglia):
  close_rsi_14  float       RSI 14 periodi
  close_rsi_25  float       RSI 25 periodi
  close_ema_09  float       EMA 9 periodi
  close_ema_25  float       EMA 25 periodi
  close_sma_25  float       SMA 25 periodi
  close_bb_upper_20  float  Banda Bollinger superiore (20, 2σ)
  close_bb_lower_20  float  Banda Bollinger inferiore (20, 2σ)
  volume_sma_25 float       Volume SMA 25 periodi
```

### Formato di caricamento

```python
import pandas as pd
from forgedge import EventDiscovery

# Carica la KPI Table da qualsiasi fonte
df = pd.read_csv("kpi_table.csv", parse_dates=["open_dt"])
df = pd.read_parquet("kpi_table.parquet")
df = pd.read_sql("SELECT * FROM candle_kpi", engine)

# FORGE accetta un DataFrame pandas — nessuna dipendenza dalla sorgente
ed = EventDiscovery(kpi_table=df)
candidates = ed.run()
```

### Famiglie di indicatori supportate

FORGE organizza i KPI in famiglie semantiche, che determinano
quali coppie possono essere combinate nell'arietà 2 (binaria):

| Famiglia | Indicatori tipici | Operazione generata |
|---|---|---|
| **Oscillatori** | RSI 14, RSI 25, RSI 9 | Diretta, pctrank, zscore |
| **EMA** | ema_03, ema_09, ema_12, ema_25 | Ratio breve/lungo |
| **SMA** | sma_03, sma_09, sma_25 | Spread percentuale dal prezzo |
| **Bande** | bb_lower, bb_upper, bb_width | Posizione %B (ternaria) |
| **Volume** | volume, vol_sma_12, vol_sma_25 | Ratio vs media |
| **Rolling min/max** | close_min_24, close_max_48 | Distanza percentuale |
| **Volatilità** | close_vol_12, close_vol_24 | Diretta, zscore |
| **Return** | close_ret_12, close_ret_24 | Diretta, zscore |

### Convenzione dei nomi

FORGE riconosce automaticamente la famiglia di un indicatore
dalla naming convention `{base}_{tipo}_{parametro}`:

```
close_rsi_14     → oscillatore RSI su close, window 14
close_ema_09     → EMA su close, window 9
volume_sma_25    → SMA su volume, window 25
close_bb_lower_20 → Bollinger lower su close, window 20
```

Se la naming convention non corrisponde, le famiglie possono essere
dichiarate esplicitamente nella configurazione.

---

## 5. Output: la Regola Validata

L'output finale di FORGE è una **regola validata** — una specifica
che un sistema di esecuzione può implementare.

### Formato di output

```yaml
# FORGE Output — Regola Validata
# ─────────────────────────────────────────────────────────────────
rule_id:          "RULE-001"
version:          "1.0"
export_date:      "2026-05-23"

# Tracciabilità
alpha_id:         "ALPHA-ADAUSDC-1H-250101-003"
event_candidate_id: "EVT-close_rsi_25-ID×PR-P105-W096-P010"

# Espressione della regola
expression:       "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
feature_dependencies:
  native:         ["close_rsi_25"]
  derived:        ["pr_close_rsi_25_96"]   # pctrank(close_rsi_25, W=96)

# Parametri operativi validati
operational_params:
  direction:      "long"
  entry_mode:     "limit"      # esito di RuleDiscoveryConfig.entry_mode="auto" (default):
                                # qui lo Stadio 2 ha promosso un limit order sopra la
                                # baseline a mercato dello Stadio 1 (vedi Sezione 3, Modulo 3)
  buy_drop_pct:   0.010       # 1% sotto il close al momento del segnale
  buy_delay_bar:  0           # barre di attesa prima di piazzare l'ordine
  sell_pct:       0.040       # +4% target di uscita
  target_h:       24          # barre massime di detenzione
  fee:            0.002       # 0.2% per lato (tailor per exchange target)

# Evidenza statistica
backtest:
  dataset:        "ADAUSDC 1H 2025-01-01/2026-01-01"
  profit_factor:  3.17
  win_rate:       0.814
  total_trades:   102
  zero_months:    0
  deflated_sharpe: 1.31

# Regime
regime_constraints:
  type:           "conditional"
  avoid_in:       ["uptrend_continuous"]
  note: >
    Win rate scende a 12.5% in mercati con EMA9/EMA25 > 1.01
    su base giornaliera. Considerare un regime filter.

# Verdetto
verdict:          "EDGE"
```

### Espressione della regola

L'espressione è una stringa booleana valutabile su ogni nuova barra:

```python
# Pseudo-codice di valutazione in un sistema di esecuzione
def should_generate_signal(candle_kpi_row):
    return eval(rule.expression, candle_kpi_row)

# L'espressione usa solo feature della KPI Table
# (native o derivate pre-calcolate come pr_close_rsi_25_96)
```

Le feature derivate utilizzate nell'espressione (es. `pr_close_rsi_25_96`)
vengono calcolate una volta sola nella fase di preprocessing della KPI Table
e rese disponibili come colonne aggiuntive.

---

## 6. Gli Artefatti Intermedi

FORGE produce tre artefatti formali che fluiscono da un modulo all'altro.
Ogni artefatto è il contratto tra due moduli adiacenti.

### Event Candidate

```yaml
# Prodotto da: Event Discovery
# Consumato da: Alpha Discovery
event_id:         "EVT-close_rsi_25-ID×PR-P105-W096-P010"
expression:       "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
source_feature:   "close_rsi_25"
components:
  - transform:    "identity"
    threshold:    30.5
    threshold_origin: "distributional_p10"
  - transform:    "rolling_pctrank"
    params:       {window: 96}
    threshold:    0.10
    threshold_origin: "distributional_p10"
activation_stats:
  n_activations:  329
  n_episodes:     41           # run di barre attive consecutive
  n_months:       12
  mean_tpm:       3.4          # episodi/mese (event_counting="episode")
  dispersion_index: 1.08       # Index of Dispersion sugli episodi
consistency_gate: "PASS"
# Nessun campo economico — il forward return non è stato osservato
```

### Alpha Contract

```yaml
# Prodotto da: Alpha Discovery
# Consumato da: Rule Discovery
alpha_id:              "ALPHA-ADAUSDC-1H-250101-003"
event_candidate_id:    "EVT-close_rsi_25-ID×PR-P105-W096-P010"
event_expression:      "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
derived_target:                # derivato dai dati, per evento
  holding_period_h:    3       # h* = argmax|z_h| (excess standardizzato, null a rotazione)
  sell_pct:            0.042   # quantile MFE a h*, barre attive IS
  direction:           "long"  # segno dell'excess log-return Δ_h*
  base_rate:           0.235
oos_validation:
  p_value:             0.0011
  passed:              true
statistical_evidence:
  ic:                  -0.0316
  p_value_ic:          0.0029
  win_rate:            0.415
  lift_vs_base:        +0.180
  cohens_d:            0.394
  p_value_ttest:       0.000004
regime_analysis:
  conditional: true
  weak_regimes: ["uptrend_continuous"]
alpha_score:
  composite:   0.68
  grade:       "B"
status:        "HYPOTHESIS"
```

### Regola Validata

```yaml
# Prodotto da: Rule Discovery → Rule Registry
# Output finale — pronto per l'integrazione nel sistema di esecuzione
# (formato completo nella Sezione 5)
rule_id:    "RULE-001"
expression: "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
verdict:    "EDGE"
```

---

## 7. Ciclo di Vita di una Regola

```
FASE 1 — GENERAZIONE                               MODULO
──────────────────────────────────────────────     ──────────────────
Event Discovery riceve la KPI Table                Event Discovery
Genera ~2.400 eventi candidati grezzi
  per ogni feature del catalogo:
  ~81 eventi per feature (4 trasformazioni × soglie)

Consistency Gate filtra:
  close_rsi_25 < 25.8  → FAIL (dispersione episodi oltre il floor di Poisson)
  close_rsi_25 < 30.5  → PASS
  pr_close_rsi_25_96 < 0.10 → PASS (uniforme per costruzione)

AND Composition:
  "close_rsi_25 < 30.5 AND pr_rsi25_96 < 0.10"
  N=329  episodi=41  mesi=12  → PASS

Output: Event Candidate                            ↓

FASE 2 — VALUTAZIONE STATISTICA                    MODULO
──────────────────────────────────────────────     ──────────────────
Alpha Discovery riceve l'Event Candidate           Alpha Discovery
Target: sell_pct=4%, holding=24h, base_rate=23.5%
Struttura mercato: H=0.44 (mean-reverting)

IC measurement: close_rsi_25 → IC=-0.032, p=0.003
Win Rate: WR=41.5%, lift=+18pp, d=0.394, p=0.000004
Regime: valido in 3/4 regimi, debole in uptrend

Score: 0.68 → Grade B

Output: Alpha Contract                             ↓

FASE 3 — VALIDAZIONE OPERATIVA                     MODULO
──────────────────────────────────────────────     ──────────────────
Rule Discovery riceve l'Alpha Contract             Rule Discovery
Backtest con limit order, fee 0.4% RT:
  PF=3.17, WR=81.4%, T=102, zero_months=0
Validazione: DSR=1.31 → edge reale
Parametri: buy_drop=1%, sell_pct=4%, hold=24h

Verdict: EDGE

Output: Regola validata                            ↓

FASE 4 — CATALOGAZIONE ED EXPORT                   MODULO
──────────────────────────────────────────────     ──────────────────
Rule Registry riceve la regola validata            Rule Registry
Deduplicazione: nessuna sovrapposizione
  con regole esistenti nel catalogo
Export: RULE-001.yaml

Output: Regola esportata → sistema di esecuzione
```

---

## 8. Principi Fondamentali

### Il forward return è invisibile fino ad Alpha Discovery

Event Discovery lavora completamente cieco rispetto al target economico.
I parametri `holding_period`, `target_return`, `fee` non esistono
nei primi due step. Questo non è un dettaglio implementativo — è un
vincolo architetturale che elimina una categoria intera di look-ahead bias.

```
Event Discovery  →  nessun target                   (struttura temporale)
Alpha Discovery  →  primo contatto con il return     (predittività)
Rule Discovery   →  primo contatto con fee e fill    (operatività)
```

### Le soglie non si modificano dopo Event Discovery

Ogni soglia dell'evento (es. `close_rsi_25 < 30.5`) viene fissata
da Event Discovery attraverso il Threshold Catalog — calcolato come
percentile della distribuzione della serie trasformata. Tutti i moduli
successivi propagano quella soglia **invariata**.

Se si vuole esplorare una soglia diversa (es. `< 33.6`), bisogna tornare
a Event Discovery e aggiungere il percentile corrispondente al
Threshold Catalog. Questo genera un nuovo Event Candidate distinto
che percorre l'intera pipeline indipendentemente.

### La rigidità di Event Discovery è selettiva, non assoluta

Il vincolo unidirezionale si applica solo alle **decisioni strutturali
sugli eventi**: soglie, finestre di trasformazione, composizioni AND.
Tutto ciò che sta a valle di quelle decisioni rimane liberamente
esplorabile all'interno della stessa sessione:

```
Event Discovery   →  soglie fisse (rigido by design)
                      │
Alpha Discovery   →  target_return, holding_period, fee, base_rate
                     criteri di promozione (lift, Cohen's d, p-value)
                     tutti variabili liberamente          (flessibile)
                      │
Rule Discovery    →  buy_drop_pct, sell_pct, buy_delay_bar, target_h
                     grid parametrico esplorato per ogni Alpha Contract
                                                          (flessibile)
```

La rigidità a monte è la garanzia che ciò che arriva ad Alpha Discovery
sia una misura genuina — non un artefatto di ottimizzazione sul target.
Compra la libertà di esplorare liberamente a valle perché le soglie
sottostanti non sono state contaminate da nessuna scelta economica.

### Le soglie sono distribuzionali, non hardcoded

Le soglie non sono valori fissi come `rsi < 30` — sono percentili
calcolati sulla distribuzione della serie trasformata per l'asset
e il periodo specifico. Questo garantisce la trasferibilità:

```
ADAUSDC 1H 2025:  close_rsi_25.quantile(0.10) = 30.5  → evento: rsi25 < 30.5
BTCUSDC 1H 2025:  close_rsi_25.quantile(0.10) = 27.8  → evento: rsi25 < 27.8
SOLUSDC 1H 2025:  close_rsi_25.quantile(0.10) = 32.1  → evento: rsi25 < 32.1
```

Stesso meccanismo, parametri che si adattano automaticamente all'asset.

### La deduplicazione opera a due livelli

Due espressioni diverse possono attivare sulle stesse barre —
e due espressioni diverse possono produrre posizioni identiche
anche con segnali parzialmente diversi. Rule Registry distingue
questi due casi con due livelli di analisi:

**Livello 1 — Deduplicazione strutturale** (segnale standardizzato):
i parametri operativi vengono proiettati sui quantili della
distribuzione congiunta del pool, e la Jaccard viene calcolata
su questi segnali standardizzati. Risponde alla domanda:
*queste espressioni booleane sono semanticamente equivalenti?*

**Livello 2 — Deduplicazione comportamentale** (fill effettivo):
il confronto avviene sui trade log reali — stesse barre di fill,
gain simili. L'espressione booleana è irrilevante. Risponde alla
domanda: *queste regole sono intercambiabili nel portfolio?*

I due livelli producono tre label distinte:

| Label | Significato | Azione raccomandata |
|---|---|---|
| `DUPLICATE_STRUCTURAL` | Stessa semantica (Livello 1) | Scartare la più debole |
| `DUPLICATE_BEHAVIORAL` | Stesso comportamento operativo (Livello 2) | Scartare o ridurre sizing |
| `INDEPENDENT_CONFIRMATION` | Diversa struttura, stesso comportamento | Tenere entrambe con sizing ridotto |

Una coppia `INDEPENDENT_CONFIRMATION` non è un duplicato da scartare
— è la conferma indipendente dello stesso edge da due meccanismi diversi.

### FORGE è stateless tra sessioni

FORGE non mantiene stato tra una sessione di ricerca e l'altra.
Ogni sessione riceve la KPI Table, genera Event Candidate, valuta,
e produce output. Il catalogo delle regole validate (Rule Registry)
è l'unica persistenza — e si popola progressivamente sessione per sessione.

---

## 9. Integrazione con Sistemi di Esecuzione

FORGE è indipendente dal sistema di esecuzione. L'output — una regola
validata in formato YAML — può essere integrato in qualsiasi architettura.

### Interfaccia minima richiesta

Il sistema di esecuzione deve essere in grado di:

1. **Caricare la KPI Table** aggiornata a ogni nuova candela
   (o tenerla in memoria e appendere le nuove barre)

2. **Calcolare le feature derivate** presenti nella regola
   (es. `pr_close_rsi_25_96` = rolling pctrank con W=96)

3. **Valutare l'espressione booleana** sulla barra corrente

4. **Generare un segnale** se l'espressione è vera e le condizioni
   operative sono soddisfatte (dimensionamento, risk budget, ecc.)

```python
# Pseudocodice di integrazione generica
class ForgeRule:
    def __init__(self, rule_yaml):
        self.expression = rule_yaml['expression']
        self.params     = rule_yaml['operational_params']

    def precompute_derived(self, df):
        """Calcola le feature derivate richieste dalla regola."""
        df['pr_close_rsi_25_96'] = (
            df['close_rsi_25']
            .rolling(96, min_periods=48)
            .rank(pct=True)
        )
        return df

    def evaluate(self, row):
        """True se la regola si attiva sulla barra corrente."""
        return eval(self.expression, row.to_dict())

    def get_entry_price(self, close):
        """Prezzo limite di ingresso."""
        return close * (1 - self.params['buy_drop_pct'])
```

### Compatibilità

| Sistema di esecuzione | Compatibilità | Note |
|---|---|---|
| Sistemi custom Python | ✅ Nativa | Integrazione diretta con pandas |
| Sistemi con ORM (SQLAlchemy, Peewee) | ✅ Nativa | Carica KPI Table dal DB |
| Backtrader, Zipline | ✅ Con adapter | Wrappa l'espressione in un Indicator |
| Trading platform proprietarie | ⚠️ Traduzione | Richiede conversione dell'espressione |

---

## 10. Roadmap

### v1.0 — Feature corrente

- [x] Event Discovery (5 step)
- [x] Alpha Discovery (IC, Win Rate, Regime Sensitivity)
- [x] Rule Discovery (Backtest, validazione statistica)
- [x] Alpha Contract e Rule Candidate come artefatti formali
- [x] Threshold Catalog distribuzionale (soglie adattive)
- [x] Consistency Gate configurabile
- [x] Rule Registry
- [x] Market Context Module (EMAProxyClassifier v1.0)
- [x] Costruzione automatica della KPI Table da OHLCV grezzo
      (`forgedge.kpi_builder`: `build_features`/`lag_features`/
      `candle_features`/`pattern_features` — v. Sezione 4)
- [x] Walk-forward optimization integrata in Rule Discovery
      (`RuleDiscoveryConfig.selection_mode="walk_forward"` è il default:
      il punto operativo pubblicato è scelto dalle finestre di train
      del walk-forward, non da un singolo backtest in-sample)
- [x] Sistema di preset coerenti (`forge_preset()`: sniper/balanced/sweep/burst)
- [x] Resolver di configurazione a livello di sessione
      (`PipelineContext`, `config_report()`, `strict=True` di default — v. Sezione 2)

### v1.x — Deduplicazione a due livelli *(Rule Registry)*

Evoluzione della deduplicazione da binaria a strutturata:

- [ ] **Livello 1** — Jaccard su segnale standardizzato: parametri operativi
      proiettati sui quantili della distribuzione congiunta del pool
      prima del calcolo della sovrapposizione
- [ ] **Livello 2** — Jaccard su fill effettivo con gain similarity:
      confronto sui trade log reali, indipendente dall'espressione booleana
- [ ] **Tre label di output:** `DUPLICATE_STRUCTURAL` / `DUPLICATE_BEHAVIORAL`
      / `INDEPENDENT_CONFIRMATION` in sostituzione del flag binario `is_duplicate`

### v2.0 — Funzionalità avanzate

- [ ] Bayesian search nello spazio dei parametri (via Optuna)
- [ ] Multi-asset discovery parallela
- [ ] Alpha decay monitor (rileva degradazione delle regole in produzione)
- [ ] Dashboard interattiva dei risultati
- [ ] Sessione persistente per utente — confronto cross-sessione automatico,
      rilevamento duplicati tra sessioni diverse, evoluzione del catalogo nel tempo
- [ ] Hint adattivi di Alpha Discovery — suggerimento di soglie alternative
      con stima del lift atteso, senza modificare il contratto formale
      con Event Discovery

---

## 11. Documentazione di Dettaglio

| Documento | Contenuto |
|---|---|
| **[Market Context Module](Market_Context_Module.md)** | Interfaccia RegimeClassifier, EMAProxyClassifier v1.0, configurazione, lookup colonne EMA, output `regime`+`regime_stable`, estensibilità v2.0 (HMM, KMeans, custom) |
| **[Event Discovery Module](Event_Discovery_Module.md)** | Architettura 5 step, Variable Catalog, Feature Generation (arietà 1/2/3), Transform Layer (Identità/Pctrank/Zscore/Delta), Event Generation, Consistency Gate, AND Composition, esempio end-to-end su `close_rsi_25` |
| **[Alpha Discovery Pipeline](Alpha_Discovery_Pipeline.md)** | Alpha Contract format, definizione del target, analisi Hurst/ACF, IC Measurement, Win Rate Analysis, Regime Sensitivity, Alpha Scoring, False Discovery Rate (BH), handoff a Rule Discovery |
| **[Rule Discovery Pipeline](Rule_Discovery_Pipeline.md)** | Parse Alpha Contract, backtest con meccanica limit order, selezione parametri, validazione statistica (t-test, DSR), analisi regime, checklist |
| **[Rule Registry Module](Rule_Registry_Module.md)** | Registro in-memory, input multi-ticker, matrici Jaccard e Spearman, deduplicazione, cross-ticker backtest con ricalcolo soglie, classificazione genericità (GENERIC/PARTIAL/SPECIFIC/ISOLATED), tabella piatta, report HTML |

### Esempio applicato

Il documento **[ADA_USDC_1H_Backtest_Report.md](ADA_USDC_1H_Backtest_Report.md)**
mostra un'applicazione completa di FORGE su ADAUSDC 1H (2025):
10 regole estratte, analisi di distribuzione mensile, analisi del regime,
confronto tra regole assolute e regime-independent.

---

## 12. Licenza

```
MIT License

Copyright (c) 2026 FORGE Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> ⚠️ **Disclaimer:** FORGE è uno strumento di ricerca quantitativa.
> I risultati del backtest non garantiscono performance future.
> Nessuna parte di questo software costituisce consulenza finanziaria.
> L'utilizzo in ambienti di trading live è sotto la piena responsabilità
> dell'utente.

---

*FORGE — Feature-Oriented Rule Generation Engine*
*Versione 1.0 · Maggio 2026 · Licenza MIT*