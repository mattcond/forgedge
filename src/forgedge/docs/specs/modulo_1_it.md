# Modulo 1 — Event Discovery (Spec da Codebase)

> **Riferimento codice:** `src/forgedge/event_discovery/`
> **Analisi funzionale:** `docs/modules/EventDiscovery.md`
> **Stato:** ✅ Implementato. Logica core allineata con analisi funzionale;
> presenti funzionalità aggiuntive non documentate (SQL export, apply OOS,
> walk-forward dettagliato).

---

## 1. Posizione nella pipeline

```
KPI Table + regime (da Modulo 0)
        │
        ▼
  EventDiscovery.run()
        │
        ▼
   list[EventCandidate]  ──► Alpha Discovery (Modulo 2)
```

Il Modulo 1 non vede mai il forward return. Opera esclusivamente sulla struttura
temporale degli indicatori nella KPI Table.

---

## 2. Interfaccia pubblica

### `EventDiscovery` (`discovery.py`)

```python
EventDiscovery(kpi_table, config=None)
```

| Metodo / Proprietà | Descrizione |
|---|---|
| `run() → list[EventCandidate]` | Esegue la pipeline completa; restituisce tutti i candidati che passano il gate |
| `summary() → pd.DataFrame` | Riepilogo tabellare di tutti i candidati (post `run()`) |
| `validated_candidates() → list[EventCandidate]` | Solo i candidati che passano la walk-forward OOS validation |
| `get_classifications() → dict` | Classificazioni delle colonne da Step 0 (debug) |
| `is_period → tuple or None` | (inizio, fine) del periodo in-sample |
| `oos_period → tuple or None` | (inizio, fine) del periodo OOS, o None se nessun split |
| `df` | KPI Table post-pipeline (con colonne derivate aggiunte da Step 1) |

---

## 3. Pipeline a 5 step

### Step 0 — TypeClassifier (`classifier.py`)

Classifica ogni colonna non-timestamp del DataFrame:

| Tipo | Criterio |
|---|---|
| `BINARY` | Esattamente 2 valori distinti non-null |
| `CATEGORICAL` | Non numerica, o numerica con valori distinti ≤ `max_categorical_classes` |
| `CONTINUOUS` | Numerica con più di 2 valori distinti |

Per le colonne CONTINUOUS, viene eseguita anche la rilevazione **scale-free**
(euristica asimmetrica conservativa — falso negativo costa meno di falso positivo).
Il risultato può essere sovrascritto via `scale_free_overrides`.

Output per colonna: `ColumnClassification` con:
- `col_type: ColumnType`
- `n_distinct: int`
- `is_scale_free: bool | None`
- `scale_free_override: bool | None`
- `effective_scale_free: bool` (property: override > automatico > False)
- `scale_free_overridden: bool` (property: True se override contraddice l'automatico)

Le colonne CATEGORICAL con `n_distinct > max_categorical_classes` (default: 20)
vengono classificate ma escluse dalla generazione degli eventi.

---

### Step 1 — FeatureGenerator (`feature_generator.py`)

Genera le feature derivate dal catalogo di feature native.

| Arietà | Operazione | Esempio | Condizione |
|---|---|---|---|
| 1 (pass-through) | Identità | `close_rsi_25` | Solo scale-free |
| 2 (ratio) | `a / b` | `ratio_close_ema_09_ema_25` | Stessa famiglia, stesso source |
| 2 (spread%) | `(a - b) / b` | `spread_close_bb_upper_lower` | Stessa famiglia |
| 2 (diffnorm) | `(a - b) / std(a-b)` | `diffnorm_close_sma_09_sma_25` | Stessa famiglia, stesso source |
| 3 (bb_pct_b) | `(val - lo) / (hi - lo)` | `bb_pct_b_close_bb_lower_upper` | Bande di Bollinger |
| 3 (pos) | `(val - min) / (max - min)` | `pos_close_min_24_max_48` | Rolling min/max |

Per le feature `diffnorm`, la deviazione standard viene calcolata sull'insieme
in-sample e salvata in `transform_params["diffnorm_std"]` per il replay OOS.

---

### Step 2 — TransformLayer (`transform_layer.py`)

Applica i 4 transform temporali a ogni feature del catalogo:

| Transform | Codice | Finestre | Applicabile a |
|---|---|---|---|
| Identità | `identity` | — | Solo scale-free |
| Rolling pctrank | `rolling_pctrank` | 48, 96, 168 barre | Tutte le continue |
| Rolling zscore | `rolling_zscore` | 48, 96, 168 barre | Tutte le continue |
| Delta (differenza) | `delta` | 1, 3, 6, 12 barre | Tutte le continue |

`min_periods` per le finestre rolling: `max(2, window // 2)`.

---

### Step 3 — EventGenerator (`event_generator.py`)

Converte ogni serie trasformata in eventi booleani:

| Tipo evento | Descrizione |
|---|---|
| `threshold` | Persistente: serie < soglia o serie > soglia |
| `crossing` | Transizione: la barra t attraversa la soglia rispetto a t-1 |

**Soglie dal Threshold Catalog (distribuzionali):**
- Percentili della serie trasformata: p3, p5, p10, p20, p25, p75, p80, p90, p95, p97
- Soglie teoriche z-score: -2.0, -1.5, -1.0, 0, +1.0, +1.5, +2.0 (per zscore transform)

**Colonne BINARY:** un evento per ogni valore (0 o 1) — transform: `binary_native`.
**Colonne CATEGORICAL:** un evento per classe (one-hot) — transform: `categorical_onehot`.

---

### Step 4 — ConsistencyGate (`consistency_gate.py`)

Filtra gli eventi per la loro distribuzione temporale.
Un evento **passa** se e solo se soddisfa tutti e 4 i criteri:

| Criterio | Parametro | Default | Descrizione |
|---|---|---|---|
| Volume | `min_act` | 50 | Attivazioni totali su tutto il dataset |
| Copertura | `min_months` | 8 | Mesi di calendario con almeno un'attivazione |
| Concentrazione | `max_conc` | 0.40 | Quota massima di attivazioni in un singolo mese |
| Frequenza | `min_tpm` | 2.0 | Media attivazioni per mese (totale / n_mesi) |

Il `fail_reason` nel `GateResult` riporta il primo criterio fallito.

---

### Step 5 — ANDComposer (`and_composer.py`)

Combina gli eventi che passano il gate con AND logico.
La composizione è consentita tra:
- **Stesso feature, transform diversi** (es. identity + pctrank96 sulla stessa RSI)
- **Feature semanticamente distinte**

La composizione è **vietata** tra:
- Stesso transform + soglie diverse sullo stesso feature (una è sottoinsieme dell'altra)

Gli eventi composti vengono sottoposti nuovamente al ConsistencyGate.
`max_and_components` (default: 2) controlla il massimo numero di componenti.

---

## 4. Strutture dati principali

### `EventCandidate` (`models.py`)

Artefatto di output del Modulo 1.

| Campo | Tipo | Descrizione |
|---|---|---|
| `event_id` | `str` | ID nel formato `EVT-{feature}-{transform_abbr}-{idx:04d}` |
| `status` | `str` | Sempre `"CANDIDATE"` a questo stadio |
| `components` | `list[EventComponent]` | 1 per eventi semplici, 2-3 per AND composition |
| `expression` | `str` | Espressione booleana leggibile (componenti unite con ` AND `) |
| `activation_stats` | `ActivationStats` | Statistiche di distribuzione temporale |
| `consistency_gate` | `GateResult` | Sempre `passed=True` per candidati restituiti da `run()` |
| `event_series` | `pd.Series` | Serie booleana 0/1/NaN con DatetimeIndex |
| `validation` | `ValidationResult | None` | Risultato walk-forward OOS; None se non configurato |
| `sql_expression` | `str` (property) | Espressione booleana DuckDB-compatibile |

**Metodo `apply(df)`:** ricostruisce la serie booleana su nuovi dati OOS,
replicando feature construction + temporal transform + threshold usando i
parametri salvati nelle componenti.
Per feature `diffnorm`, usa `transform_params["diffnorm_std"]` dell'in-sample.

### `EventComponent` (`models.py`)

| Campo | Tipo | Descrizione |
|---|---|---|
| `source_feature` | `str` | Nome della feature sorgente (es. `close_rsi_25`) |
| `transform` | `str` | Tipo di transform applicato |
| `transform_params` | `dict` | Parametri del transform (es. `{"window": 96}`) |
| `transformed_col` | `str` | Nome colonna dopo il transform |
| `threshold` | `float` | Soglia di binarizzazione |
| `threshold_type` | `str` | Origine della soglia (es. `"distributional_p10"`) |
| `direction` | `str` | `"below"` o `"above"` |
| `event_type` | `str` | `"threshold"` o `"crossing"` |
| `expression` | `str` | Stringa leggibile della condizione |
| `source_cols` | `list` | Colonne native originali (per feature arity 2/3) |
| `sql_expression` | `str` | Espressione SQL DuckDB-compatibile |

### `GateResult` (`models.py`)

| Campo | Descrizione |
|---|---|
| `passed: bool` | True se tutti i 4 criteri sono soddisfatti |
| `n_activations: int` | Totale attivazioni |
| `n_active_months: int` | Mesi con almeno un'attivazione |
| `max_monthly_share: float` | Quota del mese più concentrato |
| `mean_tpm: float` | Media attivazioni per mese |
| `fail_reason: str | None` | Primo criterio fallito, o None se passato |

### `ActivationStats` (`models.py`)

Come `GateResult` ma include anche `zero_months` (mesi senza attivazioni).

---

## 5. Format dell'event_id

Il codice genera:
```
EVT-{source_feature[:20]}-{transform_abbrs}-{idx:04d}
```

Abbreviazioni transform:
```python
"identity"          → "ID"
"rolling_pctrank"   → "PR"
"rolling_zscore"    → "ZS"
"delta"             → "DL"
"binary_native"     → "BN"
"categorical_onehot"→ "OH"
"and_composition"   → "AND"
```

Esempi:
```
EVT-close_rsi_25-PR-0042          # pctrank su RSI
EVT-close_rsi_25-PRxZS-0117       # AND: pctrank × zscore su RSI
```

---

## 6. Walk-forward OOS validation

Opzionale, configurabile via `DiscoveryConfig(train_ratio=..., walk_forward=WalkForwardConfig(...))`.

### Flusso

1. Il dataset viene diviso temporalmente: prime `train_ratio` righe = IS, il resto = OOS.
2. La pipeline completa (Step 0–5) opera solo sull'IS.
3. Per ogni candidato, il periodo OOS viene diviso in `n_splits` finestre uguali.
4. Su ogni finestra, `EventCandidate.apply()` ricostruisce la serie booleana.
   Le ultime `_MAX_CONTEXT_BARS = 168` barre IS vengono preposte come contesto
   rolling per evitare warmup NaN all'inizio di ogni fold.
5. Il ConsistencyGate viene applicato con parametri scalati proporzionalmente alla dimensione della finestra OOS:
   - `min_act` e `min_months` scalati proporzionalmente (floor: 5 e 1)
   - `max_conc` e `min_tpm` invariati (sono rate, non conteggi)
6. Un candidato è OOS-stabile se passa in almeno `min_pass_rate` (default 0.6) delle finestre.

### `WalkForwardConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `n_splits` | 3 | Numero di finestre OOS |
| `min_pass_rate` | 0.6 | Quota minima di finestre che devono passare |
| `oos_gate_params` | None | Gate OOS esplicito; se None, parametri scalati automaticamente |

### Output in `EventCandidate.validation`

```python
ValidationResult:
    n_folds: int
    n_passed: int
    pass_rate: float
    passed: bool
    fold_results: list[FoldResult]   # dettaglio per fold
```

---

## 7. SQL export (`sql_expression`)

Ogni `EventComponent` contiene un campo `sql_expression`:
un'espressione booleana DuckDB-compatibile che replica la condizione
applicando le stesse finestre rolling, pctrank, zscore, delta.

Caratteristiche:
- Usa `list_filter` lambdas per pctrank (DuckDB ≥ 0.8)
- Usa window functions standard (`AVG`, `STDDEV_SAMP`, `LAG`) per zscore e delta
- `min_periods = max(2, window // 2)` replicato via `CASE WHEN`
- `ORDER BY` usa `DiscoveryConfig.timestamp_col` (default `"open_dt"`)

Utilizzo esempio:
```python
import duckdb
rel = duckdb.from_df(ed.df.reset_index())
rel.query("df", f"SELECT *, ({candidate.sql_expression})::INT AS active FROM df")
```

---

## 8. Configurazione

### `DiscoveryConfig` (`discovery.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `gate_params` | `GateParams()` | Soglie ConsistencyGate (Step 4) |
| `max_categorical_classes` | 20 | Soglia colonne categoriali (> N: escluse dalla pipeline) |
| `scale_free_overrides` | `None` | Override manuali scale-free per colonna |
| `timestamp_col` | `"open_dt"` | Nome della colonna datetime (o nome del DatetimeIndex) |
| `max_and_components` | 2 | Massimo componenti per AND composition |
| `train_ratio` | 1.0 | Frazione IS (1.0 = nessun split OOS) |
| `walk_forward` | `None` | Config walk-forward; None = nessuna validazione OOS |

### `GateParams` (`models.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `min_act` | 50 | Attivazioni totali minime |
| `min_months` | 8 | Mesi attivi minimi |
| `max_conc` | 0.40 | Concentrazione mensile massima |
| `min_tpm` | 2.0 | Attivazioni medie per mese minime |

---

## 9. Parsing del timestamp

Il `DiscoveryConfig.timestamp_col` può provenire da:
1. **DatetimeIndex** — usato direttamente
2. **Colonna datetime64** — parsata con `pd.to_datetime`
3. **Colonna numerica** — unità inferita automaticamente dal valore mediano (s/ms/us/ns)
4. **Colonna stringa** — parsata con `pd.to_datetime` (ISO 8601)

La colonna timestamp viene **rimossa** da `self.df` dopo il parsing
(ridondante, non deve entrare nel TypeClassifier).

---

## 10. Allineamento con l'analisi funzionale

### ✅ Allineato

- 5 step + classificazione tipo (Step 0–5)
- `GateParams` con i 4 criteri (min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)
- `ColumnType`: CONTINUOUS, BINARY, CATEGORICAL
- `ColumnClassification` con `effective_scale_free` e `scale_free_override`
- `max_categorical_classes = 20`
- Generazione feature arity 1/2/3
- 4 transform (identity/pctrank/zscore/delta) con finestre documentate
- Eventi threshold e crossing con soglie distribuzionali e teoriche
- AND composition con regole di ammissibilità
- Gate riapplicato sugli eventi composti
- Walk-forward OOS validation (`train_ratio`, `WalkForwardConfig`)
- `EventCandidate` con tutti i campi documentati

### ➕ Aggiunto nel codice (non nella documentazione funzionale)

- **`sql_expression`** su `EventComponent` e `EventCandidate` — export DuckDB
- **`EventCandidate.apply(df)`** — replay deterministico della serie booleana su nuovi dati
- **`build_feature_series(comp, df)`** — funzione modulo per la feature continua sottostante
- **`_MAX_CONTEXT_BARS = 168`** — coda IS usata come contesto rolling nella validazione OOS
- **`_scale_gate_params()`** — scaling proporzionale dei parametri IS per le finestre OOS
- **`validated_candidates()`** — metodo convenienza per i soli candidati OOS-stabili
- **`is_period` e `oos_period`** — proprietà per le date dei periodi IS/OOS
- **`source_cols`** su `EventComponent` — colonne native originali per feature arity-2/3
- **`diffnorm_std`** salvato in `transform_params` — normalizzatore IS preservato per OOS

### ⚠️ Divergenze rispetto all'analisi funzionale

- **Format `event_id`:** Il codice genera `EVT-close_rsi_25-PR-0042` (indice sequenziale).
  L'analisi funzionale mostra un formato più dettagliato con finestra e soglia
  encoded nell'ID (es. `EVT-close_rsi_25-ID×PR-P105-W096-P010`).
  Il codice usa un indice progressivo invece di encodare i parametri.

- **Separatore AND composition:** Il codice usa `x` minuscolo (`PRxZS`);
  l'analisi funzionale usa `×` (moltiplicazione Unicode, `ID×PR`).
