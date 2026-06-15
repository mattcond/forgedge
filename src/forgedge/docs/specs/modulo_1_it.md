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

La configurazione di default usa i parametri di produzione (`min_act=50`,
`min_months=8`, ecc.). Per esplorare con soglie più permissive:

```python
config = DiscoveryConfig(
    gate_params=GateParams(min_act=30, min_months=6, max_conc=0.50, min_tpm=1.5),
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
p3, p5, p10, p20, p25    (code basse — condizioni estreme al ribasso)
p75, p80, p90, p95, p97  (code alte — condizioni estreme al rialzo)
```

**Soglie teoriche** (per la trasformazione zscore):
```
-2.0, -1.5, -1.0, 0.0, +1.0, +1.5, +2.0
```

Per ciascuna soglia vengono generati due tipi di evento:

| Tipo | Descrizione | Quando attivo |
|---|---|---|
| `threshold` | Stato persistente | Ogni barra in cui la condizione è vera |
| `crossing` | Transizione istantanea | Solo la barra in cui la serie attraversa la soglia |

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
concentrati in un solo mese) non è candidato affidabile per l'alpha discovery.

Un evento **passa** se e solo se soddisfa **tutti** i 4 criteri:

| Criterio | Parametro | Default | Razionale |
|---|---|---|---|
| Volume minimo | `min_act` | 50 | Stima statistica affidabile richiede campione sufficiente |
| Copertura temporale | `min_months` | 8 | L'evento deve essersi attivato in almeno 8 mesi distinti |
| Concentrazione | `max_conc` | 0.40 | Nessun singolo mese può contenere > 40% delle attivazioni |
| Frequenza media | `min_tpm` | 2.0 | Almeno 2 attivazioni al mese in media |

Il `GateResult` include il campo `fail_reason` con il primo criterio fallito
(utile per debug e tuning dei parametri).

```python
# Analisi degli eventi che non passano il gate
for candidate in ed.run():
    pass  # già solo quelli che passano

# Per vedere anche i falliti, inspezionare raw_events (non esposto pubblicamente)
# ma si può abbassare le soglie del gate per esplorare
```

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

Il composer combina coppie (e opzionalmente triple) di eventi che passano il gate
con l'operatore AND, cercando combinazioni che mantengano la coerenza temporale.

**Regole di ammissibilità per l'AND:**
- ✅ Stesso feature, trasformazioni diverse (es. `identity` AND `pctrank_96` su RSI)
- ✅ Feature semanticamente distinte (es. RSI AND volume)
- ❌ Stessa trasformazione + soglie diverse sullo stesso feature (una è sottoinsieme dell'altra)

Il composto `A AND B` viene poi ri-sottoposto al ConsistencyGate. Solo le
composizioni che passano anche il gate composto vengono promosse a candidati.

`max_and_components` (default 2) limita il numero di componenti. Valori > 3
sono accettati ma fortemente sconsigliati per rischio di overfitting strutturale.

**Esempio di AND composition valida:**
```
RSI25 < 30.5                          (identity threshold, p10)
AND
pctrank(RSI25, w=96) < 0.10           (rolling pctrank, p10)

→ "RSI è in zona oversold AND lo è stato raramente nelle ultime 96 barre"
→ Combinazione semanticamente ricca, non ridondante
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
print(stats.n_activations)      # totale attivazioni
print(stats.n_active_months)    # mesi con almeno un'attivazione
print(stats.zero_months)        # mesi senza attivazioni
print(stats.max_monthly_share)  # quota del mese più concentrato
print(stats.mean_tpm)           # media attivazioni per mese
```

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
from forgedge.event_discovery.models import WalkForwardConfig

config = DiscoveryConfig(
    train_ratio=0.80,           # 80% IS, 20% OOS
    walk_forward=WalkForwardConfig(
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
3. Per ogni candidato, l'OOS viene diviso in `n_splits` finestre uguali.
4. Su ogni finestra, `apply()` ricostruisce la serie booleana.
   Le ultime 168 barre IS vengono preposte come contesto rolling per
   evitare warmup NaN all'inizio di ogni fold.
5. Il ConsistencyGate viene applicato con parametri scalati proporzionalmente
   alla dimensione della finestra OOS:
   - `min_act` e `min_months` scalati (floor: 5 e 1 rispettivamente)
   - `max_conc` e `min_tpm` invariati (sono rate, non conteggi)
6. Un candidato è OOS-stabile se passa in almeno `min_pass_rate` delle finestre.

### Output `ValidationResult`

```python
for cand in candidates:
    if cand.validation:
        v = cand.validation
        print(f"{cand.event_id}: {v.n_passed}/{v.n_folds} folds, "
              f"pass_rate={v.pass_rate:.2f}, stable={v.passed}")
        for fold in v.fold_results:
            print(f"  fold {fold.fold_idx}: {fold.n_rows} barre, "
                  f"passed={fold.passed}, "
                  f"{'OK' if fold.passed else fold.gate_result.fail_reason}")
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
| `max_and_components` | 2 | Massimo componenti per AND composition (2 o 3; > 3 sconsigliato) |
| `train_ratio` | 1.0 | Frazione IS (1.0 = nessun split OOS, walk-forward disabilitato) |
| `walk_forward` | `None` | Config walk-forward; `None` = nessuna validazione OOS |

### `GateParams`

| Parametro | Default | Descrizione |
|---|---|---|
| `min_act` | 50 | Attivazioni totali minime nel periodo IS |
| `min_months` | 8 | Mesi di calendario con almeno un'attivazione |
| `max_conc` | 0.40 | Quota massima di attivazioni concentrate in un singolo mese |
| `min_tpm` | 2.0 | Media attivazioni per mese (totale / n_mesi_totali_nel_range) |

### `WalkForwardConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `n_splits` | 3 | Numero di finestre OOS |
| `min_pass_rate` | 0.60 | Quota minima di finestre che devono passare il gate |
| `oos_gate_params` | `None` | Gate OOS esplicito; se `None`, i parametri IS vengono scalati proporzionalmente |

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
#  'zero_months', 'max_monthly_share', 'mean_tpm', 'gate_passed',
#  'oos_pass_rate', 'oos_n_passed', 'oos_n_folds', 'oos_stable']  # OOS cols solo se walk-forward configurato

# I candidati con arietà 2 (AND composition)
and_candidates = [c for c in candidates if len(c.components) == 2]

# Candidati con pctrank
pctrank_cands = [c for c in candidates if any(comp.transform == "rolling_pctrank"
                                               for comp in c.components)]

# Serie booleana di un candidato
series = cand.event_series   # pd.Series con DatetimeIndex
print(series.resample("ME").sum())  # attivazioni per mese
```
