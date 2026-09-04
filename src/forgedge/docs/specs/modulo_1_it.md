# Modulo 1 — Event Discovery

Il Modulo 1 scopre eventi booleani dalla struttura temporale degli indicatori
nella KPI Table. Un **evento** è una condizione booleana (es. `RSI < 30.5 AND pctrank_96 < 0.10`)
che si attiva su un sottoinsieme di barre. Il modulo lavora **senza mai vedere il
forward return**: valuta solo se un evento ha un pattern di attivazione temporale
stabile e statisticamente plausibile.

L'output è una lista di `EventCandidate`, uno per ogni evento che supera il
ConsistencyGate. Questi candidati vengono poi passati al Modulo 2 per misurare
il loro potere predittivo.

---

## Utilizzo di base

```python
from forgedge import EventDiscovery
from forgedge.event_discovery.discovery import DiscoveryConfig
from forgedge.event_discovery.models import GateParams

ed = EventDiscovery(enriched_kpi)       # kpi già arricchita con regime da Modulo 0
candidates = ed.run()

print(f"Trovati {len(candidates)} candidati")
print(ed.summary().sort_values("mean_tpm", ascending=False).head(10))
```

La configurazione di default usa i parametri di produzione (`min_tpm=0.5`,
`event_counting="episode"`, `min_episodes=10`, `dispersion_margin=1.3`,
ecc. — vedi Step 4 più sotto). Per esplorare con soglie più permissive:

```python
config = DiscoveryConfig(
    gate_params=GateParams(min_tpm=0.3, min_episodes=5, dispersion_margin=1.6),
    max_and_components=2,
)
ed = EventDiscovery(enriched_kpi, config=config)
candidates = ed.run()
```

---

## La pipeline a 5 step

### Step 0 — Classificazione delle colonne (`TypeClassifier`)

Ogni colonna del DataFrame (escluso il timestamp) viene classificata in uno
di tre tipi:

| Tipo | Criterio | Trattamento downstream |
|---|---|---|
| `CONTINUOUS` | Numerica, > 2 valori distinti | Pipeline completa (Step 1–3) |
| `BINARY` | Esattamente 2 valori distinti | Salta Step 1–2, va direttamente a Step 3 |
| `CATEGORICAL` | Non numerica, o ≤ `max_categorical_classes` valori | One-hot in Step 3; se > limite, esclusa |

Per le colonne CONTINUOUS viene rilevata anche la proprietà **scale-free**:
una serie è scale-free se i suoi valori sono intrinsecamente delimitati
(es. RSI in [0,100], percentuali) e non dipendono dal livello di prezzo
dell'asset. Questa proprietà determina se la trasformazione `identity`
viene inclusa nel Step 2.

Il rilevamento è un'euristica asimmetrica e conservativa: preferisce il falso
negativo (classificare come non-scale-free una serie che lo è) al falso positivo
(classificare come scale-free una serie che non lo è, generando eventi con
soglie dipendenti dal livello assoluto del prezzo).

**Override manuale:** se l'euristica produce un risultato errato su una colonna
specifica, è possibile correggerla:

```python
config = DiscoveryConfig(
    scale_free_overrides={"close_rsi_14": True, "volume": False}
)
```

Dopo `run()`, le classificazioni sono ispezionabili via `ed.get_classifications()`:

```python
cls = ed.get_classifications()
for col, c in cls.items():
    print(f"{col}: {c.col_type.value}, scale_free={c.effective_scale_free}")
```

---

### Step 1 — Generazione delle feature (`FeatureGenerator`)

A partire dalle feature native della KPI Table, vengono generate feature derivate
di arietà 1, 2 e 3:

| Arietà | Operazione | Formula | Esempio |
|---|---|---|---|
| 1 | Pass-through | `f` | `close_rsi_25` (solo scale-free) |
| 2 | Ratio | `a / b` | `ratio_close_ema_09_ema_25` |
| 2 | Spread percentuale | `(a - b) / b` | `spread_close_bb_upper_lower` |
| 2 | Diffnorm | `(a - b) / σ(a-b)` | `diffnorm_close_sma_09_sma_25` |
| 3 | %B Bollinger | `(val - lo) / (hi - lo)` | `bb_pct_b_close_bb_lower_upper` |
| 3 | Posizione in range | `(val - min) / (max - min)` | `pos_close_min_24_max_48` |

Le feature di arietà 2 vengono generate solo tra colonne della stessa famiglia
(es. due EMA sullo stesso source, non EMA e RSI). I denominatori nulli producono
`NaN` (non `±inf` o `pd.NA`), preservando il dtype `float64`.

Per la feature `diffnorm`, la deviazione standard `σ(a-b)` viene calcolata
sul periodo in-sample e salvata in `transform_params["diffnorm_std"]`.
Questo valore viene riutilizzato per il replay OOS in modo da preservare
la stessa scala dell'in-sample: la feature OOS viene normalizzata con la
stessa deviazione standard IS, non ricalcolata.

---

### Step 2 — Trasformazioni temporali (`TransformLayer`)

Ogni feature del catalogo riceve le seguenti trasformazioni:

| Transform | Codice | Finestre | Applica a |
|---|---|---|---|
| Identità | `identity` | — | Solo scale-free |
| Rolling percentile rank | `rolling_pctrank` | 48, 96, 168 barre | Tutte le continue |
| Rolling z-score | `rolling_zscore` | 48, 96, 168 barre | Tutte le continue |
| Delta (differenza) | `delta` | 1, 3, 6, 12 barre | Tutte le continue |

`min_periods` per le finestre rolling: `max(2, window // 2)`.
Questo significa che la finestra da 96 barre inizia a produrre valori già dopo
48 barre, anche se la stima è meno stabile.

**Perché queste trasformazioni?**
- `identity`: per feature già scale-free e stazionarie (RSI, %B), il valore
  grezzo è direttamente comparabile su tutto il dataset.
- `rolling_pctrank`: converte qualsiasi serie in un range [0,1] relativo alla
  storia recente. È robusta agli outlier e non richiede stazionarietà.
- `rolling_zscore`: sensibile alla distribuzione locale. Utile per rilevare
  deviazioni statistiche dalla media recente.
- `delta`: cattura le variazioni di breve periodo (momentum o reversal su lag specifici).

---

### Step 3 — Generazione degli eventi (`EventGenerator`)

Ogni serie trasformata viene convertita in eventi booleani applicando soglie.
Il Threshold Catalog distingue due famiglie di soglie:

**Soglie distribuzionali** (basate sui percentili della serie trasformata):
```
p3, p5, p8, p10, p15     (code basse — condizioni estreme al ribasso)
p85, p90, p92, p95, p97  (code alte — condizioni estreme al rialzo)
```

**Soglie teoriche** (per la trasformazione zscore):
```
-2.0, -1.5, -1.0, +1.0, +1.5, +2.0
```

Per ciascuna soglia viene sempre generato un evento `threshold`. Viene generato
anche un evento `crossing`, ma **solo per la trasformazione `identity`, e solo
sulle soglie con `direction="below"`** — le soglie assolute non hanno senso
per rolling pctrank/zscore (ri-ancorate ad ogni barra) o per delta (che
oscilla intorno allo zero), quindi i crossing sono ristretti a `identity`.
All'interno di `identity`, è definita solo la formula del crossing al
ribasso (`serie_t < soglia AND serie_{t-1} >= soglia`): non esiste nel pool
generato un crossing simmetrico al rialzo (`above`) — il caso d'uso a cui
questa parte del generatore è rivolta è la rilevazione di ipervenduto/estremo
basso, e non ha mai ricevuto una controparte al rialzo.

| Tipo | Descrizione | Quando attivo | Generato per |
|---|---|---|---|
| `threshold` | Stato persistente | Ogni barra in cui la condizione è vera | Tutte le trasformazioni, entrambe le direzioni |
| `crossing` | Transizione istantanea | Solo la barra in cui la serie attraversa la soglia | Solo trasformazione `identity`, `direction="below"` |

Gli eventi di tipo `crossing` segnalano "il segnale è appena entrato in zona",
utile per logiche di entry. Gli eventi `threshold` catturano "il segnale è in
zona da un numero arbitrario di barre", più appropriato per filtri di regime.

**Colonne BINARY:** genera un evento `binary_native` per ciascuno dei due valori
(0 e 1) — non richiede trasformazioni.

**Colonne CATEGORICAL:** genera un evento `categorical_onehot` per ogni classe
con `n_distinct ≤ max_categorical_classes`. Le classi con troppi valori
distinti vengono escluse perché assimilabili a identificatori.

---

### Step 4 — ConsistencyGate (`ConsistencyGate`)

Il gate filtra gli eventi in base alla loro distribuzione temporale di attivazione.
La logica è: un evento con struttura temporale instabile (es. tutti i trigger
concentrati in un solo mese, o statisticamente indistinguibile da un processo
casuale) non è candidato affidabile per l'alpha discovery.

Il gate gira in una di due modalità, selezionata da `GateParams.event_counting`.

**Modalità `"episode"` (default, issue #134)** — conta gli *episodi* (run
massimali di attivazioni consecutive, che assorbono gap fino a `episode_gap`
barre) invece delle singole barre. Questo elimina un artefatto di conteggio
per cui uno stato persistente su più barre (es. un tratto di 3–5 barre con
`RSI < 30`) gonfia la varianza mensile e viene rigettato erroneamente. Un
evento **passa** se e solo se soddisfa **tutti e 3** i criteri:

| Criterio | Parametro | Default | Razionale |
|---|---|---|---|
| Frequenza | `min_tpm` | 0.5 | Episodi al mese in media, almeno questo valore |
| Potenza statistica | `min_episodes` | 10 | Floor assoluto sul numero di episodi, così i test di significatività a valle (Moduli 2/3) hanno campione sufficiente |
| Dispersione | `dispersion_margin` | 1.3 | L'Index of Dispersion a livello di episodio non deve superare `poisson_floor(n_months) x dispersion_margin` |

Il criterio di dispersione (issue #205) confronta contro un *floor χ² di
Poisson* — il più grande Index of Dispersion ancora coerente con un processo
puramente casuale al tasso proprio dell'evento — invece di un valore assoluto
fisso. Questo garantisce che il gate non rigetti mai un evento che un processo
casuale avrebbe potuto plausibilmente produrre, pur lasciando che la
tolleranza alla burstiness propria di un preset (`dispersion_margin`) sia
effettivamente vincolante: misurato su tutti e quattro i preset e i quattro
timeframe, una soglia fissa `max_dispersion` non è mai risultata vincolante
in 12 combinazioni su 16, perché il floor da solo era quasi sempre il vincolo
più stringente. `max_dispersion` non ha alcun ruolo in modalità `"episode"`.

**Modalità `"bar"`** — il comportamento storico, 100% retrocompatibile:
conta le singole barre attivate invece degli episodi. Un evento **passa**
se e solo se soddisfa **entrambi** i criteri:

| Criterio | Parametro | Default | Razionale |
|---|---|---|---|
| Frequenza | `min_tpm` | 0.5 | Barre attivate al mese in media, almeno questo valore |
| Dispersione | `max_dispersion` | 1.5 | L'Index of Dispersion per barra (Var/Media dei conteggi mensili) non deve superare questo valore assoluto — nessun floor di Poisson coinvolto |

`min_episodes` e `dispersion_margin` non hanno alcun ruolo in modalità `"bar"`.

Il `GateResult` include il campo `fail_reason` con il primo criterio fallito
(utile per debug e tuning dei parametri), più campi diagnostici sempre
calcolati indipendentemente dalla modalità: `n_episodes` (conteggio episodi),
`episode_index_of_dispersion` (Index of Dispersion a livello di episodio,
calcolato quando è disponibile un indice mensile) e `n_eff` — una dimensione
campionaria effettiva (`n_episodes / max(episode_ID, 1.0)`) pensata per la
deflazione della significatività a valle nei Moduli 2/3.

**`min_episodes` differenziato per preset (issue #206):** `forge_preset()`
non applica un unico default fisso — `sniper`, `balanced` e `burst`
mantengono 10 (il floor di potenza statistica è proprio lo scopo di questi
profili), mentre `sweep` lo abbassa a 5, permissivo per design, perché
delega il rigore statistico al `RotationCalibrator` a valle. Poiché
`min_episodes` è un conteggio assoluto, la finestra di discovery IS
necessaria per raggiungerlo con margine di Poisson al 95% dipende sia da
`min_episodes` sia da `min_tpm` — la finestra naive `min_episodes / min_tpm`
soddisfa il floor solo in aspettativa. Il warning `m1_is_window_too_short`
di `config_report()` segnala lo scarto tra la combinazione configurata e
l'ampiezza di discovery effettivamente disponibile (`span_months x
train_ratio`).

---

### Step 4b — Diversity Gate (opt-in)

**Nome:** Diversity Gate (opt-in)

**Scopo:** dopo il ConsistencyGate e prima della composizione AND, gli eventi singoli
near-duplicate vengono rimossi tramite similarità Jaccard sulla sovrapposizione delle
date di attivazione. Due eventi con Jaccard ≥ `diversity_threshold` sono considerati
near-duplicate; il più debole (meno attivazioni IS) viene scartato.

**Perché è importante:** senza il Diversity Gate, l'AND Composer può produrre un gran
numero di eventi composti strutturalmente ridondanti (es. `RSI < 31 AND RSI < 30` dove
le due componenti hanno >85% di date di attivazione sovrapposte). Deduplicare il pool
di eventi singoli prima della composizione AND riduce sia lo spazio di ricerca sia il
rischio di candidati ridondanti.

**Default:** disabilitato (`diversity_gate_enabled=False` in `DiscoveryConfig`).
Solo opt-in — nessun breaking change.

**Soglia default:** `diversity_threshold=0.85`. Nota empirica: a p99 della distribuzione
Jaccard inter-evento (su 12 mesi di dati 1H), Jaccard=0.47 — valori sopra 0.70 sono
genuine near-duplicate.

**Attivazione:** impostare `diversity_gate_enabled=True` in `DiscoveryConfig`.

```python
from forgedge import EventDiscovery, DiscoveryConfig

config = DiscoveryConfig(
    diversity_gate_enabled=True,    # deduplicazione Jaccard opt-in
    diversity_threshold=0.85,       # rimuovi near-duplicate con Jaccard ≥ 0.85
)
ed = EventDiscovery(enriched, config=config)
candidates = ed.run()
```

---

### Step 5 — AND Composition (`ANDComposer`)

**Disattivato di default dall'issue #254 Fase 8.** Il default di classe di
`DiscoveryConfig()` è ora `max_and_components=1`, che salta del tutto questo
step — il percorso di default di `forge()` (`two_pass_composition=True`)
compone a valle invece, dopo la prima passata di grading del Modulo 2,
usando il voto come criterio di pairing invece di quello puramente
strutturale di questo step (vedi la spec del Modulo 2 e
`forgedge.composition.grade_guided_compose`). Alza `max_and_components`
sopra `1` per riattivare questo step — sia standalone (`EventDiscovery`
usato direttamente, come nell'esempio *Uso base* sopra), sia via
`forge(two_pass_composition=False)`, che riproduce esattamente la pipeline
single-pass pre-#254. Tutto ciò che segue in questa sezione descrive questo
step come si comporta ogni volta che gira — la sua meccanica interna non è
cambiata nella Fase 8.

Il composer combina coppie (e opzionalmente triple) di eventi che passano il gate
con l'operatore AND, cercando combinazioni che mantengano la coerenza temporale.

**Regole di ammissibilità per l'AND:**
- ✅ Stesso feature, trasformazioni diverse (es. `identity` AND `pctrank_96` su RSI)
- ✅ Feature semanticamente distinte (es. RSI AND volume)
- ❌ Stessa trasformazione + soglie diverse sullo stesso feature (una è sottoinsieme dell'altra)

Il composto `A AND B` viene poi ri-sottoposto al ConsistencyGate. Solo le
composizioni che passano anche il gate composto vengono promosse a candidati.

`max_and_components` (default `1` dalla Fase 8, era `2`) limita il numero di
componenti — `1` disattiva questo step. Valori > 3 sono accettati ma
fortemente sconsigliati per rischio di overfitting strutturale.

**Esempio di AND composition valida:**
```
RSI25 < 30.5                          (identity threshold, p10)
AND
pctrank(RSI25, w=96) < 0.10           (rolling pctrank, p10)

→ "RSI è in zona oversold AND lo è stato raramente nelle ultime 96 barre"
→ Combinazione semanticamente ricca, non ridondante
```

---

### Diagnostica sulla distribuzione degli eventi (`event_distribution_report`)

Dopo `run()`, `ed.event_distribution_report` è **sempre** popolato (issue
#215) — un attributo pubblico `str`, indipendente da `config_report()`. I due
sono complementari: `config_report()` risolve la coerenza config-vs-config
prima di vedere qualsiasi dato, quindi non può mai rilevare che i candidati
grezzi di un asset specifico stanno fallendo il gate in modo patologico;
`event_distribution_report` aggrega gli scalari `GateResult` effettivamente
raccolti per ogni candidato grezzo (pre-composizione AND) prodotto dallo
Step 3, indipendentemente dal fatto che `cfg.retain_raw_events` mantenga o
meno l'intera popolazione di candidati.

Riporta:
- il numero totale di candidati grezzi e quanti hanno superato il ConsistencyGate;
- la frequenza (tpm) e la dispersione mediane osservate rispetto alle soglie
  configurate, e quale frazione di candidati cade dal lato sbagliato di
  ciascuna;
- quando la sopravvivenza al gate è **sotto una soglia del 15%**, un
  suggerimento concreto di parametri alla mediana osservata.

```python
print(ed.event_distribution_report)
```

Il testo stesso viene emesso in italiano indipendentemente dal contesto di
chiamata (è anche la riga di log dello stage M1), ad esempio:

```
M1 Event Discovery — 8213 candidati generati, 512 superano il
Consistency Gate (6.2%).
tpm osservato: mediana=0.31 (soglia min_tpm=0.5, 61.4% sotto soglia).
dispersione osservata: mediana=2.10 (soglia effettiva=1.69, 38.0% sopra soglia).
Meno del 15% dei candidati generati supera il Consistency Gate (512/8213 = 6.2%).
Prova questi parametri (mediana osservata su tpm e dispersione): min_tpm<=0.31, dispersion_margin>=1.62.
```

---

## Strutture dati

### `EventCandidate`

Artefatto finale prodotto da `run()`. Ogni candidato rappresenta un evento
booleano che ha superato il ConsistencyGate.

```python
cand = candidates[0]
print(cand.event_id)       # "EVT-close_rsi_25-PR-0042"
print(cand.expression)     # "pr_close_rsi_25_96 < 0.10"
print(cand.event_formula)  # "pctrank(close_rsi_25, w=96) < 0.10"
print(cand.activation_stats.n_activations)    # 329
print(cand.activation_stats.mean_tpm)         # 4.7
print(cand.consistency_gate.max_monthly_share) # 0.18
```

| Campo | Tipo | Descrizione |
|---|---|---|
| `event_id` | `str` | Identificatore unico nel formato `EVT-{feature}-{transform_abbr}-{idx:04d}` |
| `status` | `str` | Sempre `"CANDIDATE"` a questo stadio |
| `components` | `list[EventComponent]` | 1 per eventi semplici, 2-3 per AND composition |
| `expression` | `str` | Espressione booleana leggibile (nomi colonne trasformate) |
| `event_formula` | `str` | Formula in notazione matematica standard |
| `activation_stats` | `ActivationStats` | Statistiche di distribuzione temporale |
| `consistency_gate` | `GateResult` | Sempre `passed=True` per candidati da `run()` |
| `event_series` | `pd.Series` | Serie booleana 0/1/NaN con DatetimeIndex |
| `validation` | `ValidationResult \| None` | Risultato walk-forward OOS; None se non configurato |
| `sql_expression` | `str` (property) | Espressione DuckDB-compatibile |

#### Formato `event_id`

```
EVT-{source_feature[:20]}-{transform_abbrs}-{idx:04d}

Abbreviazioni:
  ID  = identity
  PR  = rolling_pctrank
  ZS  = rolling_zscore
  DL  = delta
  BN  = binary_native
  OH  = categorical_onehot
  AND = and_composition

Esempi:
  EVT-close_rsi_25-PR-0042        (pctrank su RSI)
  EVT-close_rsi_25-IDxPR-0117     (AND: identity × pctrank su RSI)
```

### `EventComponent`

Ogni `EventCandidate` contiene uno o più componenti, uno per ciascuna condizione
booleana nell'espressione AND.

```python
comp = cand.components[0]
print(comp.source_feature)    # "close_rsi_25"
print(comp.transform)         # "rolling_pctrank"
print(comp.transform_params)  # {"window": 96}
print(comp.threshold)         # 0.1
print(comp.threshold_type)    # "distributional_p10"
print(comp.direction)         # "below"
print(comp.event_type)        # "threshold"
print(comp.expression)        # "pr_close_rsi_25_96 < 0.10"
print(comp.event_formula)     # "pctrank(close_rsi_25, w=96) < 0.10"
print(comp.source_cols)       # [] per arity-1, ["col_a", "col_b"] per arity-2
```

| Campo | Descrizione |
|---|---|
| `source_feature` | Nome della feature sorgente (es. `close_rsi_25`, `ratio_close_ema_09_ema_25`) |
| `transform` | Tipo di transform: `identity`, `rolling_pctrank`, `rolling_zscore`, `delta`, `binary_native`, `categorical_onehot` |
| `transform_params` | Parametri del transform: `{"window": 96}` per pctrank/zscore, `{"lag": 3}` per delta, `{"diffnorm_std": 0.023}` per diffnorm |
| `transformed_col` | Nome della colonna trasformata (es. `pr_close_rsi_25_96`) |
| `threshold` | Valore numerico della soglia (es. `0.10`, `30.5`) |
| `threshold_type` | Origine: `"distributional_p10"`, `"theoretical_z-1.5"`, ecc. |
| `direction` | `"below"` o `"above"` |
| `event_type` | `"threshold"` (persistente) o `"crossing"` (transizione) |
| `expression` | Stringa leggibile della singola condizione |
| `event_formula` | Formula in notazione standard (es. `pctrank(close_rsi_25, w=96) < 0.10`) |
| `source_cols` | Colonne native originali (vuoto per arity-1; `[col_a, col_b]` per arity-2; `[val, lo, hi]` per arity-3) |
| `sql_expression` | Espressione SQL DuckDB-compatibile |

### `ActivationStats`

```python
stats = cand.activation_stats
print(stats.n_activations)               # totale attivazioni
print(stats.n_active_months)             # mesi con almeno un'attivazione
print(stats.zero_months)                 # mesi senza attivazioni
print(stats.max_monthly_share)           # quota del mese più concentrato
print(stats.mean_tpm)                    # media attivazioni per mese
print(stats.index_of_dispersion)         # Index of Dispersion per barra (Var/Media dei conteggi mensili)
print(stats.n_episodes)                  # numero di episodi di attivazione distinti
print(stats.episode_index_of_dispersion) # Index of Dispersion dei conteggi mensili di episodi
print(stats.n_eff)                       # dimensione campionaria effettiva: n_episodes / max(episode_ID, 1.0)
```

`index_of_dispersion`, `n_episodes`, `episode_index_of_dispersion` e `n_eff`
sono gli stessi campi diagnostici che `GateResult` porta con sé (vedi Step
4) — sempre calcolati, non solo quando i criteri della modalità attiva li
usano effettivamente.

---

## Formula matematica (`event_formula`)

Il campo `event_formula` su `EventComponent` e la property `event_formula`
su `EventCandidate` forniscono una rappresentazione in notazione matematica
standard, più leggibile rispetto ai nomi di colonna trasformata.

La formula viene costruita in tre fasi da `_build_event_formula`:

| Fase | Esempi di output |
|---|---|
| Feature (`_formula_feature`) | `close_rsi_25`, `close / open`, `(sma_09 - sma_25) / std(sma_09 - sma_25)  [std=0.0032]`, `bb_pct_b(close, bb_lower, bb_upper)` |
| Transform (`_formula_transform`) | `pctrank(close_rsi_25, w=96)`, `zscore(close / open, w=48)`, `Δ(close_rsi_25, lag=1)` |
| Condizione (`_formula_condition`) | `... < P10 [P10=0.10]`, `... > -1.5`, `... crosses ↓ P5 [P5=-0.03]` |

**Convenzione `FORMULA [VALORE]`:** le soglie distribuzionali (derivate dai
percentili in-sample della serie trasformata) sono mostrate come `P10 [P10=0.10]`,
dove `P10` indica il percentile e `0.10` è il valore effettivo su quel dataset.
Le soglie teoriche degli zscore (`-2.0`, `-1.5`, ecc.) sono costanti fisse e
vengono mostrate senza annotazione. La stessa convenzione si applica al denominatore
della feature `diffnorm`: `std(x-y)  [std=0.0032]` indica che la deviazione
standard IS usata per normalizzare vale `0.0032`.

Esempi completi per un AND candidate:

```python
cand.event_formula
# "(pctrank(close_rsi_25, w=48) < P10 [P10=0.10]) AND (zscore(close_rsi_25, w=96) > -1.5)"
```

---

## SQL export (`sql_expression`)

Ogni `EventComponent` contiene un'espressione SQL DuckDB-compatibile che replica
la condizione identicamente alla pipeline pandas, usando le stesse finestre rolling.

```python
import duckdb
rel = duckdb.from_df(ed.df.reset_index())

# Verifica event attivo per ogni barra
query = f"SELECT open_dt, ({cand.sql_expression})::INT AS active FROM df"
result = rel.query("df", query)
```

Caratteristiche dell'espressione SQL:
- `pctrank`: usa `list_filter` lambdas (DuckDB ≥ 0.8) con average-method rank
  per riprodurre esattamente `pandas.rank(pct=True)`.
- `zscore` e `delta`: usa window functions standard (`AVG`, `STDDEV_SAMP`, `LAG`).
- `min_periods = max(2, window // 2)` replicato tramite `CASE WHEN`.
- Ordine garantito tramite `ORDER BY {timestamp_col}`.

---

## Replay su nuovi dati: `EventCandidate.apply(df)`

Il metodo `apply(df)` ricostruisce la serie booleana su un DataFrame OOS
usando esclusivamente i parametri salvati nelle componenti, senza richiedere
una nuova esecuzione della pipeline.

```python
oos_kpi = pd.read_parquet("kpi_oos.parquet")
oos_series = cand.apply(oos_kpi)
print(oos_series.value_counts())
```

Per la ricostruzione corretta le colonne native (`source_cols`) devono essere
presenti nel DataFrame. La funzione `build_feature_series(comp, df)` ricostruisce
la feature continua sottostante (senza trasformazione), utile per passarla
all'Alpha Discovery:

```python
from forgedge.event_discovery.models import build_feature_series

feature_series = build_feature_series(cand.components[0], oos_kpi)
```

### Persistenza su disco: `EventCandidate.persist(path)`

`persist(path)` serializza il candidato completo su disco come file pickle,
preservando l'intera struttura (componenti, soglie, statistiche di attivazione,
validazione walk-forward). Per ricaricare basta il modulo standard `pickle`:

```python
# Salvataggio
cand.persist("contracts/ev_btc_rsi25.pkl")

# Ricaricamento — restituisce un EventCandidate pronto all'uso
import pickle
cand_loaded = pickle.load(open("contracts/ev_btc_rsi25.pkl", "rb"))
cand_loaded.apply(new_df)   # funziona immediatamente
```

`persist()` è il metodo raccomandato per archiviare candidati singoli tra
sessioni o condividerli tra processi. Per archivi di più candidati, la forma
JSON/CSV del `to_dict()` è più leggibile ma non è invertibile (non riproduce
l'oggetto `EventCandidate` completo); `persist()` è l'unico metodo che
garantisce il round-trip completo.

### Rivalutazione OOS: `EventCandidate.update_event(df)`

`update_event(df)` rivaluta il candidato su un nuovo DataFrame **in place**,
usando esclusivamente i parametri IS fissati durante la scoperta (soglie,
finestre di trasformazione) — senza ricalibrarli. Aggiorna:

- `event_series` — nuova serie booleana sul DataFrame passato
- `activation_stats` — ricalcolate dalla nuova serie
- `consistency_gate` — metriche e `passed` ricalcolati con i `gate_params`
  originali (se disponibili; senza `gate_params` il gate è marcato `False`)

```python
# Valutare un candidato scoperto su dati storici su dati più recenti
new_kpi = pd.read_parquet("btc_1h_new.parquet")
cand.update_event(new_kpi)

print(f"Attivazioni sul nuovo periodo: {cand.activation_stats.n_activations}")
print(f"Gate superato: {cand.consistency_gate.passed}")
print(f"Mesi attivi: {cand.activation_stats.n_active_months}")

# Il segnale è ora aggiornato sui nuovi dati
new_signal = cand.event_series
```

`update_event()` è l'alternativa a `apply()` quando si vogliono aggiornare anche
le statistiche e il gate (e non solo la serie booleana): dopo la chiamata,
`cand.event_series` e `cand.activation_stats` riflettono il nuovo periodo.
Le soglie non cambiano mai — sono sempre quelle fissate durante la scoperta IS.

---

### `CustomEvent` — iniezione manuale di eventi

`CustomEvent` permette all'utente di definire una formula-ipotesi e iniettarla
direttamente in Alpha Discovery (Modulo 2) e Rule Discovery (Modulo 3), bypassando
completamente il pipeline automatico di Event Discovery. La formula è valutata con
`pd.DataFrame.eval()` e può riferirsi a qualsiasi colonna del DataFrame — compresi
indicatori proprietari che il FeatureGenerator automatico non produce mai.

Import: `from forgedge import CustomEvent`

**Costruttore:**

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `formula` | str | — | Espressione booleana per `pd.DataFrame.eval()`. Esempi: `"rsi_14 > 70"`, `"close < bb_lower and volume > 1e6"`. |
| `name` | str | `""` | Etichetta leggibile usata nei report. Default: il testo della formula. |

**Metodi:**
- `apply(df)` → `pd.Series[bool]`: valuta la formula su `df`. I risultati NaN sono trattati come inattivi (False).
- `to_event_candidate(df, gate_params=None)` → `EventCandidate`: costruisce un `EventCandidate` completo dalla formula. Il ConsistencyGate e le statistiche di attivazione sono calcolati su `df`; non ci sono soglie distribuzionali (la formula è definita dall'utente).

**Comportamento importante:**
- Quando usato via `forge(manual_events=[...])`, ogni CustomEvent attraversa comunque il ConsistencyGate. Un fallimento emette un `logger.warning` ma non scarta l'evento — l'ipotesi dell'utente è sempre inoltrata.
- La composizione AND non viene eseguita in modalità iniezione manuale.
- Il candidato risultante ha `event_id = "CUSTOM-{name}"`.

**Esempio standalone:**
```python
from forgedge import CustomEvent, AlphaDiscovery, AlphaConfig

# Definire una ipotesi custom
ev = CustomEvent("rsi_14 > 70 and volume > 1e6", name="rsi_overbought_volume")

# Applicare a qualsiasi frame
signal = ev.apply(df)                            # pd.Series bool

# Costruire un EventCandidate completo per M2/M3
cand = ev.to_event_candidate(df)
print(cand.event_id)                             # "CUSTOM-rsi_overbought_volume"
print(cand.activation_stats.n_activations)

# Alpha Discovery direttamente
ad = AlphaDiscovery(df, [cand], AlphaConfig(asset="BTC", timeframe="1H"))
contracts = ad.run()
```

**Esempio via `forge()` con iniezione manuale:**
```python
from forgedge import forge, CustomEvent

events = [
    CustomEvent("close_adj_v2 < 100", name="close_below_100"),
    CustomEvent("rsi_14 < 25 and spread_ema < -0.02", name="rsi_extreme_spread"),
]

# manual_events bypassa il Modulo 1; Moduli 2 e 3 girano normalmente
result = forge(
    kpi,
    ticker="BTCUSDC",
    timeframe="1H",
    manual_events=events,
)
for contract, resp in result.rule_responses:
    print(contract.alpha_id, resp.verdict)
```

Nota: `manual_events` e `event_discovery_config` sono mutualmente esclusivi — passarli
entrambi genera `ValueError`.

---

## Walk-forward OOS validation

La validazione walk-forward è opzionale e serve a verificare che un evento
scoperto in-sample mantenga la stessa struttura di attivazione su dati
mai visti. È una misura di stabilità, non di potere predittivo (quello
è compito del Modulo 2).

```python
from forgedge.event_discovery.models import EventWalkForwardConfig

config = DiscoveryConfig(
    train_ratio=0.80,           # 80% IS, 20% OOS
    walk_forward=EventWalkForwardConfig(
        n_splits=3,             # divide l'OOS in 3 finestre
        min_pass_rate=0.60,     # deve passare il gate in almeno 2/3 finestre
    ),
)
ed = EventDiscovery(enriched_kpi, config=config)
candidates = ed.run()

# Solo i candidati OOS-stabili
stable = ed.validated_candidates()
print(f"{len(stable)} candidati stabili su {len(candidates)}")
```

### Come funziona

1. Il dataset viene diviso: prime `train_ratio` righe = IS, il resto = OOS.
2. L'intera pipeline (Step 0–5) gira **solo sull'IS**.
3. Per ogni candidato, l'OOS viene diviso in `n_splits` finestre uguali (fold).
4. Su ogni fold, `apply()` ricostruisce la serie booleana. Le ultime 168
   barre IS vengono preposte come contesto rolling per evitare warmup NaN
   all'inizio del fold.
5. Il ConsistencyGate viene applicato ad ogni fold con `min_episodes`
   **forzato a 0** invece che scalato (issue #177): essendo un conteggio
   assoluto, `min_episodes` non sopravvive all'applicazione su una finestra
   più corta — su un fold di 2 mesi il default di classe pari a 10 richiede
   5 episodi/mese contro un requisito in-sample che può essere anche
   1/mese, ~5x più severo per costruzione. `min_tpm`, `max_dispersion` e
   `dispersion_margin` sono invarianti rispetto al tasso/rapporto e
   passano **invariati** dai `gate_params` IS (oppure da
   `wf.oos_gate_params` quando impostato esplicitamente — comunque sempre
   con `min_episodes` forzato a 0).
6. Ogni fold viene inoltre verificato per **testabilità**: il tasso IS
   proprio del candidato implica un conteggio atteso di episodi `lam =
   is_rate x n_months` per la durata di quel fold. Quando `lam <
   MIN_FOLD_LAMBDA` (3.0), un fold vuoto ha probabilità ≥5% anche per un
   candidato che ha mantenuto perfettamente il proprio tasso IS — un fold
   simile viene marcato `indeterminate` ed **escluso dal denominatore di
   `pass_rate`**, mai contato come fallimento.
7. Un candidato è OOS-stabile (`passed=True`) quando passa il gate in
   almeno `min_pass_rate` dei fold **testabili** (`n_testable`, non
   `n_folds`). Quando nessun fold è testabile, `passed` vale `None` —
   inconcludente, non fallito; il filtro `only_validated_events` di
   `forge()` tratta `None` in modo distinto da `False`, così un
   walk-forward che non ha potuto girare non scarta silenziosamente tutti
   i candidati.

### Output `ValidationResult`

Campi di `ValidationResult`: `n_folds` (fold totali eseguiti), `n_passed`,
`n_testable` (fold abbastanza lunghi da dire qualcosa — il denominatore di
`pass_rate`), `pass_rate` (`n_passed / n_testable`, `nan` quando nulla era
testabile), `passed` (`bool | None`), `fold_results` (dettaglio per fold,
inclusi quelli indeterminati). Ogni `FoldResult` porta inoltre
`indeterminate` (bool) e `lam` (il conteggio atteso di episodi del fold al
tasso in-sample del candidato).

```python
for cand in candidates:
    if cand.validation:
        v = cand.validation
        print(f"{cand.event_id}: {v.n_passed}/{v.n_testable} fold testabili "
              f"({v.n_folds} totali), pass_rate={v.pass_rate:.2f}, stable={v.passed}")
        # v.passed è None quando nessun fold era testabile — inconcludente, non fallito
        for fold in v.fold_results:
            if fold.indeterminate:
                status = f"INDETERMINATE (lam={fold.lam:.2f} < 3.0)"
            else:
                status = "OK" if fold.passed else fold.gate_result.fail_reason
            print(f"  fold {fold.fold_idx}: {fold.n_rows} barre, {status}")
```

---

## Configurazione completa

### `DiscoveryConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `gate_params` | `GateParams()` | Soglie ConsistencyGate |
| `max_categorical_classes` | 20 | Colonne categoriali con più classi vengono escluse dalla pipeline |
| `scale_free_overrides` | `None` | Override manuale: `{"col": True/False}` |
| `timestamp_col` | `"open_dt"` | Nome della colonna datetime (o nome del DatetimeIndex) |
| `max_and_components` | 1 (era 2 pre-#254 Fase 8) | Massimo componenti per AND composition; `1` disattiva lo Step 5 (la composizione propria di questo modulo), precondizione per `forge(two_pass_composition=True)` — il default. Alzare per il percorso legacy single-pass o un chiamante con una propria fase di composizione |
| `retain_raw_events` | `False` (era `True` pre-#254 Fase 8) | Se `EventDiscovery.raw_events` resta popolato dopo `run()`; serve solo a `TargetOptimizer`, che imposta `True` esplicitamente sulla propria config di fallback |
| `train_ratio` | 1.0 | Frazione IS (1.0 = nessun split OOS, walk-forward disabilitato) |
| `walk_forward` | `None` | Config walk-forward; `None` = nessuna validazione OOS |

### `GateParams`

| Parametro | Default | Descrizione |
|---|---|---|
| `min_tpm` | 0.5 | Minimo medio di trigger al mese, nell'unità scelta da `event_counting` (episodi/mese in modalità `"episode"`, barre/mese in modalità `"bar"`) |
| `max_dispersion` | 1.5 | Massimo Index of Dispersion per barra — **solo modalità `"bar"`**; non letto in modalità `"episode"` |
| `dispersion_margin` | 1.3 | **Solo modalità `"episode"`** — moltiplicatore sopra il floor χ² di Poisson: `eff_max_dispersion = poisson_floor(n_months) x dispersion_margin` |
| `event_counting` | `"episode"` | Unità di conteggio per i criteri di frequenza/dispersione: `"episode"` o `"bar"` |
| `min_episodes` | 10 | Floor assoluto sul numero di episodi (guardia di potenza statistica, solo modalità `"episode"`); applicato **solo in-sample**, forzato a 0 nei fold OOS del walk-forward |
| `episode_gap` | 1 | Gap massimo (in barre) ancora appartenente allo stesso episodio; `0` = run strettamente consecutivi |

### `EventWalkForwardConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `n_splits` | 3 | Numero di finestre OOS |
| `min_pass_rate` | 0.60 | Quota minima di finestre **testabili** che devono passare il gate |
| `oos_gate_params` | `None` | Gate OOS esplicito; se `None`, vengono usati i `gate_params` IS, con `min_episodes` comunque sempre forzato a 0 |

---

## Parsing del timestamp

La colonna timestamp accetta tutti i formati comuni:

| Formato | Comportamento |
|---|---|
| `DatetimeIndex` sul DataFrame | Usato direttamente |
| Colonna `datetime64` | Parsata con `pd.to_datetime` |
| Colonna numerica (Unix epoch) | Unità inferita dal valore mediano: `<1e10`=s, `<1e13`=ms, `<1e16`=µs, altrimenti ns |
| Colonna stringa | Parsata come ISO 8601 |

La colonna timestamp viene **rimossa** da `df` dopo il parsing per non
entrare nel TypeClassifier. `ed.df` ha sempre un `DatetimeIndex`.

---

## Leggere e filtrare i risultati

```python
# Riepilogo tabellare di tutti i candidati
summary = ed.summary()
print(summary.columns.tolist())
# ['event_id', 'status', 'expression', 'n_activations', 'n_active_months',
#  'zero_months', 'max_monthly_share', 'mean_tpm', 'index_of_dispersion',
#  'gate_passed', 'oos_pass_rate', 'oos_n_passed', 'oos_n_folds', 'oos_stable']  # OOS cols solo se walk-forward configurato

# I candidati con arietà 2 (AND composition)
and_candidates = [c for c in candidates if len(c.components) == 2]

# Candidati con pctrank
pctrank_cands = [c for c in candidates if any(comp.transform == "rolling_pctrank"
                                               for comp in c.components)]

# Serie booleana di un candidato
series = cand.event_series   # pd.Series con DatetimeIndex
print(series.resample("ME").sum())  # attivazioni per mese
```
