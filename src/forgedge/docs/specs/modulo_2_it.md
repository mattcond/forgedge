# Modulo 2 — Alpha Discovery (Spec da Codebase)

> **Riferimento codice:** `src/forgedge/alpha_discovery/`
> **Analisi funzionale:** `docs/modules/AlphaDiscovery.md`
> **Stato:** ✅ Implementato. Logica core allineata con analisi funzionale;
> presenti normalizzazioni e campi aggiuntivi non documentati.
> Il campo `rule_discovery_hints` documentato non è presente nel modello.

---

## 1. Posizione nella pipeline

```
KPI Table + regime  (Modulo 0)
list[EventCandidate] (Modulo 1)
        │
        ▼
  AlphaDiscovery.run()
        │
        ▼
  list[AlphaContract]   (tutti: HYPOTHESIS + REJECTED)
  promoted_contracts()  (solo HYPOTHESIS)  ──► Rule Discovery (non implementato)
```

Alpha Discovery è il **primo modulo che vede il forward return**.
Non ricalcola né modifica le soglie degli eventi.

---

## 2. Interfaccia pubblica

### `AlphaDiscovery` (`discovery.py`)

```python
AlphaDiscovery(kpi_table, event_candidates, config=None)
```

| Metodo / Proprietà | Descrizione |
|---|---|
| `run() → list[AlphaContract]` | Valuta tutti i candidati; restituisce tutti i contratti (promossi + rifiutati) |
| `promoted_contracts() → list[AlphaContract]` | Solo i contratti con `status == "HYPOTHESIS"` |
| `summary() → pd.DataFrame` | Riepilogo tabellare, ordinato per `composite_score` decrescente |
| `base_rate` | Win rate senza filtro (popolato da `run()`) |
| `market_structure` | Struttura di mercato (Hurst + ACF), calcolata una sola volta |

`AlphaDiscovery` consuma `ed.df` (post-pipeline di Event Discovery) che contiene
già la colonna `regime` (da Market Context) e le colonne feature derivate.
Quando una feature derivata non è nella tabella, viene ricalcolata deterministicamente
dai parametri salvati nel componente.

---

## 3. Pipeline a 8 step

### Step 1 — Definizione del target (`target.py`)

```python
build_target(close, tgt) → (fwd_return, target_binary, base_rate)
```

- `fwd_return`: rendimento forward massimo su `holding_period_h` barre
  (per direzione `long`: max(close[t+1..t+h]) / close[t] - 1)
- `target_binary`: 1 se `fwd_return >= sell_pct`, 0 altrimenti
- `base_rate`: media di `target_binary` su tutte le barre valide (win rate senza filtro)

`sell_pct` e `holding_period_h` provengono da `TargetDefinition`.
Il `fee_per_side` è solo informativo (non viene sottratto dal target — è compito di Rule Discovery).

---

### Step 2 — Analisi struttura di mercato (`market_structure.py`)

```python
analyse_market_structure(close, fwd_return) → MarketStructure
```

| Campo | Descrizione |
|---|---|
| `hurst` | Esponente di Hurst del prezzo di chiusura (DFA) |
| `hurst_interpretation` | `"mean_reverting"` / `"random_walk"` / `"trending"` |
| `expected_family` | `"mean_reversion"` / `"momentum"` / `"none"` |
| `autocorr` | ACF del rendimento forward per lag selezionati |

Calcolata **una sola volta** per l'intera sessione, non per candidato.

---

### Step 3 — Misura dell'IC (`discovery.py: _measure_ic`)

**Correlazione di Spearman** tra la feature continua sottostante e `fwd_return`.

**Gate di ammissione:** un candidato è ammesso se NON sono entrambe vere:
- `|IC| < ic_min_abs` (default: 0.02)
- `p_value > ic_max_p` (default: 0.05)

Cioè: il candidato è **respinto** al gate IC solo se IC è debole **E** non significativo.
Se è significativo (p basso) pur con IC piccolo, passa l'IC gate.

**Rolling IC stability:** valutata su ≈20 finestre equidistanti di ampiezza `rolling_ic_window`
(default: 60 giorni in barre, inferito da `bars_per_day`).
Un candidato è stabile se la stessa firma (segno IC) si ripete in almeno il 70% delle finestre.

Output: `ICResult` con campi:
- `ic`, `p_value`, `n`: statistiche globali
- `rolling_ic_stable: bool | None`: True se segno coerente in ≥70% delle finestre
- `rolling_ic_mean: float | None`: media degli IC rolling
- `rolling_sign_consistency: float | None`: quota finestre con stesso segno dell'IC globale
- `admitted: bool`: True se passa il gate IC

---

### Step 4 — Analisi win rate (`discovery.py: _measure_event`)

Misura il potere predittivo dell'**evento binario** sul target:

| Metrica | Descrizione |
|---|---|
| `win_rate` | `target_binary.mean()` sulle barre con evento attivo |
| `lift` | `win_rate - base_rate` |
| `fwd_return_mean` | Media dei rendimenti forward sulle barre attive |
| `cohens_d` | Dimensione dell'effetto: differenza standardizzata active vs inactive returns |
| `t_stat`, `p_value` | t-test indipendente (one-sided: `alternative="greater"`) |

Output: `EventStats`.

---

### Step 5 — Sensibilità al regime (`discovery.py: _measure_regimes`)

Per ogni regime con almeno `min_regime_obs` (default: 10) osservazioni:
- IC di Spearman della feature continua vs `fwd_return`
- Win rate condizionale dell'evento

**Classificazione forza regime:**

| Strength | Condizione |
|---|---|
| `"strong"` | `p < 0.05` e `|IC| >= 0.05` |
| `"moderate"` | `p < 0.05` e `|IC| < 0.05` |
| `"negligible"` | `p >= 0.05` (non significativo) |
| `"insufficient"` | Meno di `min_regime_obs` osservazioni |

**Classificazione dipendenza da regime:**

| Dependency type | Condizione |
|---|---|
| `"agnostic"` | Tutti i regimi valutati sono significativi (strong o moderate) e ≥2 |
| `"conditional"` | Più di 1 regime significativo, ma non tutti |
| `"specific"` | Esattamente 1 regime significativo |
| `"broken"` | 0 regimi significativi |
| `"unknown"` | Nessuna colonna regime disponibile |

Se `use_stable_regime_only = True` e `regime_stable` è presente:
vengono usate solo le barre con `regime_stable = True` (no barre di transizione).

Output: `RegimeAnalysis` con `per_regime`, `dependency_type`, `active_regimes`,
`weak_regimes`, `regime_breadth` (quota regimi significativi su valutati).

---

### Step 6 — Alpha scoring (`discovery.py: _score`)

**Score composito** (0–1):

```
composite = Σ(w_i * norm_i) / Σ(w_i)
```

| Termine | Peso default | Normalizzazione |
|---|---|---|
| IC magnitude | 0.25 | `min(|IC| / 0.10, 1.0)` — satura a IC=10% |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` — satura a lift=30% |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` — satura a d=0.80 |
| Regime breadth | 0.20 | `regime_breadth` (già 0–1) |

Se il regime non è disponibile, il termine `regime_breadth` viene escluso
e i pesi rimanenti vengono rinormalizzati (non sostituiti con 0).

**Grade da score composito:**

| Grade | Score |
|---|---|
| `A` | ≥ 0.75 |
| `B+` | ≥ 0.60 |
| `B` | ≥ 0.45 |
| `C` | < 0.45 |

---

### Step 7 — Compilazione Alpha Contract (`discovery.py: _build_contract`)

**Gate di promozione** — un candidato viene promosso (`status = "HYPOTHESIS"`)
solo se soddisfa **tutti** i criteri seguenti:

| Criterio | Parametro | Default |
|---|---|---|
| IC ammesso | `ic_min_abs`, `ic_max_p` | 0.02, 0.05 |
| Lift ≥ soglia | `min_lift` | 0.08 (+8pp) |
| Cohen's d ≥ soglia | `min_cohens_d` | 0.15 |
| Attivazioni ≥ soglia | `min_activations` | 30 |
| FDR Benjamini-Hochberg | `fdr_q` | 0.10 (se `use_fdr=True`) |
| p_value < soglia | `max_p_value` | 0.05 (se `use_fdr=False`) |

Il campo `rejection_reasons` elenca tutti i criteri falliti (non solo il primo).

---

### Step 8 — FDR control (`stats.py: benjamini_hochberg`)

Il controllo FDR viene applicato **su tutti i candidati insieme** prima della compilazione
dei contratti.
Algoritmo: Benjamini-Hochberg (BH).
Target: q = `fdr_q` (default 0.10) → al massimo il 10% di falsi positivi tra i promossi.

Quando `use_fdr = True`, il BH sostituisce la soglia `max_p_value` per la promozione.
Il campo `fdr_promoted` nel contratto indica se il candidato supera il BH,
indipendentemente dall'esito finale della promozione (utile per audit).

---

## 4. Strutture dati principali

### `AlphaContract` (`models.py`)

| Campo | Tipo | Descrizione |
|---|---|---|
| `alpha_id` | `str` | `ALPHA-{asset}-{timeframe}-{stamp}-{idx:03d}` |
| `version` | `str` | `"1.0"` |
| `discovery_date` | `str` | Data ISO (today o `AlphaConfig.discovery_date`) |
| `status` | `str` | `"HYPOTHESIS"` o `"REJECTED"` |
| `asset`, `exchange`, `timeframe`, `direction` | `str` | Metadati target |
| `event_candidate_id` | `str` | Link all'EventCandidate sorgente |
| `event_expression` | `str` | Espressione booleana dell'evento |
| `pattern_family` | `str` | `"mean_reversion"` / `"momentum"` / `"unspecified"` |
| `target_definition` | `TargetDefinition` | Parametri del target |
| `base_rate` | `float` | Win rate senza filtro |
| `underlying_feature` | `ICResult` | Step 3: IC della feature continua |
| `event_stats` | `EventStats` | Step 4: metriche predittive binarie |
| `market_structure` | `MarketStructure` | Step 2: Hurst + ACF |
| `regime_analysis` | `RegimeAnalysis` | Step 5: sensibilità al regime |
| `alpha_score` | `AlphaScore` | Step 6: score composito e grade |
| `promoted` | `bool` | True se tutti i gate sono superati |
| `rejection_reasons` | `list[str]` | Gate falliti (vuoto se promosso) |
| `fdr_promoted` | `bool | None` | True se supera il BH FDR |
| `handoff_status` | `str` | `"PENDING_RULE_DISCOVERY"` (default) |
| `rule_discovery_response` | `dict | None` | Risposta da Rule Discovery (non implementato) |

**Metodi di serializzazione:**
- `to_dict()` — dict piatto per DataFrame (campo `pattern_family` incluso)
- `to_contract_dict()` — dict nidificato per export YAML/JSON

### `ICResult` (`models.py`)

| Campo | Descrizione |
|---|---|
| `ic: float` | Correlazione di Spearman globale |
| `p_value: float` | p-value IC |
| `n: int` | Osservazioni valide |
| `rolling_ic_stable: bool | None` | True se segno coerente in ≥70% finestre |
| `rolling_ic_mean: float | None` | Media IC rolling |
| `rolling_sign_consistency: float | None` | Quota finestre con stesso segno |
| `admitted: bool` | True se passa il gate IC |

### `EventStats` (`models.py`)

| Campo | Descrizione |
|---|---|
| `n_activations: int` | Barre con evento attivo |
| `win_rate: float` | Win rate condizionale |
| `base_rate: float` | Win rate non condizionale |
| `lift: float` | `win_rate - base_rate` |
| `fwd_return_mean: float` | Media rendimenti forward su barre attive |
| `cohens_d: float` | Dimensione effetto (active vs inactive) |
| `t_stat: float` | Statistica t |
| `p_value: float` | p-value t-test (one-sided) |

### `RegimeStat` (`models.py`)

| Campo | Descrizione |
|---|---|
| `regime: str` | Nome del regime |
| `n: int` | Osservazioni nel regime |
| `ic: float` | IC feature nel regime |
| `p_value: float` | p-value IC nel regime |
| `win_rate: float` | Win rate condizionale nel regime |
| `strength: str` | `"strong"` / `"moderate"` / `"negligible"` / `"insufficient"` |

### `AlphaScore` (`models.py`)

| Campo | Descrizione |
|---|---|
| `ic_magnitude: float` | |IC| |
| `lift: float` | Lift |
| `cohens_d: float` | Cohen's d |
| `regime_breadth: float` | Quota regimi significativi |
| `composite_score: float` | Score composito 0–1 |
| `grade: str` | A / B+ / B / C |

---

## 5. Primitive statistiche (`stats.py`)

Il modulo implementa in **puro numpy** (nessuna dipendenza da scipy):

| Funzione | Algoritmo |
|---|---|
| `spearmanr(x, y)` | Correlazione Spearman via numpy rank |
| `cohens_d(group1, group2)` | Differenza medie / deviazione standard pooled |
| `ttest_ind(x, y, alternative)` | t-test indipendente, p-value via funzione beta incompleta |
| `benjamini_hochberg(p_values, q)` | Controllo FDR Benjamini-Hochberg |
| `betai(a, b, x)` | Funzione beta incompleta regolarizzata (Lentz continued-fraction) |

Le probabilità Student-t sono ottenute dalla funzione beta incompleta regolarizzata
implementata con l'algoritmo a frazione continua di Lentz (*Numerical Recipes*).
La precisione è ≈1e-6 rispetto ai valori scipy.

---

## 6. Configurazione

### `TargetDefinition` (`models.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `holding_period_h` | 24 | Orizzonte forward in barre |
| `sell_pct` | 0.04 | Soglia rendimento per target binario (es. 0.04 = +4%) |
| `direction` | `"long"` | `"long"` o `"short"` |
| `fee_per_side` | 0.002 | Solo informativo (non sottratto nel target) |
| `asset` | `"ASSET"` | |
| `exchange` | `""` | |
| `timeframe` | `"1H"` | |

### `PromotionThresholds` (`models.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `ic_min_abs` | 0.02 | |IC| minimo per gate IC |
| `ic_max_p` | 0.05 | p-value massimo per gate IC |
| `min_lift` | 0.08 | Lift minimo (+8pp) |
| `min_cohens_d` | 0.15 | Cohen's d minimo |
| `max_p_value` | 0.05 | p-value massimo (se `use_fdr=False`) |
| `min_activations` | 30 | Attivazioni minime |
| `use_fdr` | `True` | Usa BH invece di `max_p_value` |
| `fdr_q` | 0.10 | Target false-discovery rate BH |

### `AlphaConfig` (`models.py`)

| Parametro | Default | Descrizione |
|---|---|---|
| `target` | `TargetDefinition()` | Target economico |
| `thresholds` | `PromotionThresholds()` | Gate di ammissione e promozione |
| `close_col` | `"close"` | Colonna prezzo chiusura |
| `timestamp_col` | `"open_dt"` | Colonna datetime |
| `regime_col` | `"regime"` | Colonna regime (da Market Context) |
| `regime_stable_col` | `"regime_stable"` | Colonna regime_stable |
| `use_stable_regime_only` | `False` | Usa solo barre stable per analisi regime |
| `min_regime_obs` | 10 | Osservazioni minime per valutare un regime |
| `rolling_ic_window` | `None` | Finestra rolling IC (None → 60 giorni in barre) |
| `bars_per_day` | `None` | Barre per giorno (None → inferito) |
| `score_weights` | `(0.25, 0.30, 0.25, 0.20)` | Pesi: (IC, lift, cohens_d, breadth) |
| `discovery_date` | `None` | Data ISO per i contratti (None → oggi) |

---

## 7. Allineamento con l'analisi funzionale

### ✅ Allineato

- Pipeline a 8 step con corretta sequenza
- `PromotionThresholds` con tutti i valori documentati
- `AlphaContract` con tutti i campi principali
- `ICResult`, `EventStats`, `RegimeStat`, `RegimeAnalysis`, `AlphaScore`
- `MarketStructure` (Hurst + ACF)
- FDR Benjamini-Hochberg con `fdr_q = 0.10`
- Classificazione forza regime (strong/moderate/negligible/insufficient)
- Classificazione dependency type (agnostic/conditional/specific/broken)
- Grade threshold (A≥0.75, B+≥0.60, B≥0.45, C<0.45)
- Pesi score: (0.25, 0.30, 0.25, 0.20)
- `use_stable_regime_only` + `min_regime_obs`
- `handoff_status = "PENDING_RULE_DISCOVERY"` e `rule_discovery_response`

### ➕ Aggiunto nel codice (non nella documentazione funzionale)

- **Caps di normalizzazione nello score:**
  - |IC| normalizzato come `min(|IC|/0.10, 1.0)` — satura a 10%
  - Lift normalizzato come `min(lift/0.30, 1.0)` — satura a 30%
  - Cohen's d normalizzato come `min(d/0.80, 1.0)` — satura a 0.80
- **Rinormalizzazione pesi** quando regime non disponibile (non sostituzione con 0)
- **`rolling_sign_consistency`** su `ICResult` — quota finestre con stesso segno IC
- **`rolling_ic_mean`** su `ICResult` — media IC rolling
- **`pattern_family`** su `AlphaContract` — derivato da `MarketStructure.expected_family`
- **`to_contract_dict()`** — serializzazione YAML/JSON del contratto completo
- **Inferenza `bars_per_day`** da spaziatura mediana del DatetimeIndex
- **Rolling IC** calcolato su ≈20 finestre equidistanti (stride = max(1, (n-w)//20)),
  non su ogni barra — costo piatto indipendentemente dalla lunghezza del dataset
- **`fdr_promoted`** — flag separato dal `promoted` per audit FDR
- **Stats pure numpy** senza dipendenza da scipy (Lentz continued-fraction per p-value)

### ⚠️ Divergenze rispetto all'analisi funzionale

- **Gate IC:** L'analisi funzionale documenta i gate IC e p-value come indipendenti.
  Il codice usa `not (weak_ic AND weak_p)`: un candidato è **ammesso** se IC è forte
  **OPPURE** il p-value è significativo. La rigetto avviene solo quando entrambi sono deboli.

### ❌ Mancante nel codice (documentato nell'analisi funzionale)

- **`rule_discovery_hints`** nel contratto (Section 7 di AlphaDiscovery.md):
  campi come `entry_mode`, `buy_drop_pct_range`, `sell_pct_range`, `min_pf_target`,
  `exclusion_conditions`. Il codice ha solo `rule_discovery_response: dict | None`
  (generico, non strutturato), e il campo rimane `null` perché Rule Discovery
  non è ancora implementato.
