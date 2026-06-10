# Modulo 2 — Alpha Discovery

Alpha Discovery è il terzo modulo della pipeline FORGE e il **primo che vede il
forward return**. Riceve la lista di `EventCandidate` prodotta da Event Discovery
e misura la forza predittiva di ciascun evento rispetto a un target economico
definito dall'utente. L'output è una lista di `AlphaContract` — uno per candidato
— che registra tutte le misure statistiche e stabilisce se il candidato è stato
promosso a ipotesi operabile.

---

## Utilizzo di base

```python
from forgedge import (
    MarketContext,
    EventDiscovery,
    AlphaDiscovery, AlphaConfig, TargetDefinition,
)
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")

# Modulo 0 e 1: regime + scoperta eventi
enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)
candidates = ed.run()

# Modulo 2: misurazione predittiva
config = AlphaConfig(
    target=TargetDefinition(
        holding_period_h=24,
        sell_pct=0.04,
        direction="long",
        asset="BTC",
        timeframe="1H",
    )
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

promoted = ad.promoted_contracts()
print(f"{len(promoted)} promossi su {len(contracts)} valutati")
print(ad.summary().head())
```

`run()` restituisce **tutti** i contratti (promossi e rifiutati), così è
possibile verificare i motivi di rifiuto. `promoted_contracts()` filtra solo i
contratti con `status == "HYPOTHESIS"`.

---

## Posizione nella pipeline

```
KPI Table + regime  (Modulo 0)
list[EventCandidate] (Modulo 1)
        │
        ▼
  AlphaDiscovery.run()
        │
        ▼
  list[AlphaContract]   (tutti: HYPOTHESIS + REJECTED)
  promoted_contracts()  (solo HYPOTHESIS) ──► Rule Discovery (non implementato)
```

Alpha Discovery **non ricalcola** né le soglie degli eventi né le feature
derivate. Legge la colonna `regime` dal DataFrame di Event Discovery (`ed.df`)
e usa la `event_series` salvata in ogni candidato senza ri-derivarla.

---

## Pipeline a 8 step

### Step 1 — Definizione del target

Il target economico è costruito dalla colonna `close` tramite `build_target()`:

- **`fwd_return`**: rendimento forward massimo su `holding_period_h` barre.
  Per `direction="long"`: `max(close[t+1..t+h]) / close[t] - 1`.
  Per `direction="short"`: `1 - min(close[t+1..t+h]) / close[t]`.
- **`target_binary`**: 1 se `fwd_return >= sell_pct`, 0 altrimenti.
- **`base_rate`**: frequenza di `target_binary == 1` sull'intero campione valido
  (win rate senza filtro). Esposto come `ad.base_rate` dopo `run()`.

`fee_per_side` è registrato nel contratto a scopo informativo: Alpha Discovery
non lo detrae dal target — questa responsabilità spetta a Rule Discovery.

---

### Step 2 — Analisi struttura di mercato

Calcolata **una sola volta** per l'intera sessione (non per candidato) su
`close` e `fwd_return`:

```python
ad.market_structure.hurst               # float: esponente di Hurst (DFA)
ad.market_structure.hurst_interpretation  # "mean_reverting" | "random_walk" | "trending"
ad.market_structure.expected_family     # "mean_reversion" | "momentum" | "none"
ad.market_structure.autocorr            # dict[int, float]: ACF per lag selezionati
```

`expected_family` viene copiato in ogni contratto come `pattern_family`
(`"unspecified"` se `"none"`), fornendo un contesto interpretativo sul tipo di
alpha da cui ci si aspetta che l'evento emerga.

La soglia Hurst è `0.5`: sotto → mean-reverting, sopra → trending.

---

### Step 3 — Misura dell'IC (Information Coefficient)

Per ogni candidato, la feature continua sottostante al suo primo componente
(es. il valore RSI prima della sogliatura) viene correlata con `fwd_return`
tramite la correlazione di Spearman:

```
IC = ρ(feature, fwd_return)   [Spearman]
```

Tutti i calcoli statistici sono implementati in **puro numpy** senza dipendenze
da scipy.

#### Gate di ammissione IC

Il candidato supera il gate IC se **non** sono entrambe vere le seguenti
condizioni:

- `|IC| < ic_min_abs` (default: 0.02) — IC debole
- `p_value > ic_max_p` (default: 0.05) — non significativo

In altre parole: il candidato viene **respinto al gate IC solo se l'IC è debole
E il p-value non è significativo**. Se l'IC è piccolo ma statisticamente
significativo (basso p-value), il candidato passa comunque il gate.

#### Rolling IC stability

La stabilità temporale dell'IC è valutata su ≈20 finestre equidistanti di
ampiezza `rolling_ic_window` (default: 60 giorni espressi in barre, inferiti
dalla spaziatura mediana del DatetimeIndex):

```
stride = max(1, (n - window) // 20)
```

Un candidato è stabile (`rolling_ic_stable = True`) se il segno dell'IC rolling
coincide con il segno dell'IC globale in almeno il 70% delle finestre
(`rolling_sign_consistency >= 0.70`). Il numero fisso di ≈20 finestre mantiene
il costo computazionale costante indipendentemente dalla lunghezza del dataset.

Output — `ICResult`:
```python
ic.ic                    # float: IC di Spearman globale
ic.p_value               # float: p-value
ic.n                     # int: osservazioni valide
ic.admitted              # bool: supera il gate IC
ic.rolling_ic_stable     # bool | None: segno coerente in ≥70% finestre
ic.rolling_ic_mean       # float | None: media IC rolling
ic.rolling_sign_consistency  # float | None: quota finestre con stesso segno
```

---

### Step 4 — Analisi win rate

Misura il potere predittivo dell'**evento binario** sul target:

| Metrica | Formula |
|---|---|
| `n_activations` | Numero di barre con evento attivo e target valido |
| `win_rate` | `mean(target_binary)` sulle barre attive |
| `lift` | `win_rate - base_rate` |
| `fwd_return_mean` | Media dei rendimenti forward sulle barre attive |
| `cohens_d` | `(mean_active - mean_inactive) / std_pooled` |
| `t_stat`, `p_value` | t-test indipendente one-sided (`alternative="greater"`) |

Il p-value usa un t-test a due campioni con alternativa `greater`:
"il rendimento medio sulle barre con evento è superiore a quello sulle barre
senza evento". La funzione beta incompleta (algoritmo Lentz *Numerical Recipes*)
produce p-value con precisione ≈1e-6 rispetto a scipy.

Output — `EventStats`:
```python
ev.n_activations   # int
ev.win_rate        # float: es. 0.38 → 38% delle attivazioni raggiunge il target
ev.base_rate       # float: win rate senza filtro (copia di ad.base_rate)
ev.lift            # float: es. 0.06 → +6pp rispetto al base rate
ev.fwd_return_mean # float: rendimento medio sulle barre attive
ev.cohens_d        # float: dimensione dell'effetto
ev.t_stat          # float
ev.p_value         # float
```

---

### Step 5 — Sensibilità al regime

Per ogni regime definito nella colonna `regime` (dal Modulo 0), con almeno
`min_regime_obs` osservazioni (default: 10):

1. Calcola l'IC di Spearman della feature continua vs `fwd_return` nel regime
2. Calcola il win rate condizionale dell'evento nel regime

Se `use_stable_regime_only = True` e `regime_stable` è presente nel DataFrame,
vengono usate solo le barre con `regime_stable = True` (esclude le barre di
transizione dal calcolo dell'IC per regime).

#### Classificazione forza regime

| Strength | Condizione |
|---|---|
| `"strong"` | `p < 0.05` e `\|IC\| ≥ 0.05` |
| `"moderate"` | `p < 0.05` e `\|IC\| < 0.05` |
| `"negligible"` | `p ≥ 0.05` (non significativo) |
| `"insufficient"` | Meno di `min_regime_obs` osservazioni |

#### Classificazione dipendenza dal regime

| `dependency_type` | Condizione |
|---|---|
| `"agnostic"` | Tutti i regimi valutati sono significativi (strong o moderate) e il numero è ≥2 |
| `"conditional"` | Più di 1 regime significativo, ma non tutti |
| `"specific"` | Esattamente 1 regime significativo |
| `"broken"` | 0 regimi significativi |
| `"unknown"` | Nessuna colonna regime disponibile |

Output — `RegimeAnalysis`:
```python
ra.per_regime       # list[RegimeStat]: misure per ciascun regime
ra.dependency_type  # str: classificazione dipendenza
ra.active_regimes   # list[str]: regimi significativi (strong o moderate)
ra.weak_regimes     # list[str]: regimi negligible
ra.regime_breadth   # float: len(active) / len(evaluated)
```

Ogni `RegimeStat`:
```python
rs.regime    # str: nome del regime
rs.n         # int: osservazioni nel regime
rs.ic        # float: IC nel regime
rs.p_value   # float: p-value IC nel regime
rs.win_rate  # float: win rate condizionale nel regime
rs.strength  # str: "strong" | "moderate" | "negligible" | "insufficient"
```

---

### Step 6 — Alpha scoring

Lo **score composito** (0–1) è una media pesata di quattro componenti
normalizzate:

| Componente | Peso default | Normalizzazione |
|---|---|---|
| IC magnitude | 0.25 | `min(\|IC\| / 0.10, 1.0)` — satura a IC=10% |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` — satura a lift=30% |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` — satura a d=0.80 |
| Regime breadth | 0.20 | `regime_breadth` (già 0–1) |

Quando la colonna regime non è disponibile, il termine `regime_breadth` viene
rimosso e i pesi rimanenti sono **rinormalizzati** (non sostituiti con 0):

```
composite = Σ(w_i * norm_i) / Σ(w_i)   [sui soli termini disponibili]
```

#### Grade da score composito

| Grade | Score |
|---|---|
| `A` | ≥ 0.75 |
| `B+` | ≥ 0.60 |
| `B` | ≥ 0.45 |
| `C` | < 0.45 |

Output — `AlphaScore`:
```python
sc.ic_magnitude    # float: |IC| grezzo
sc.lift            # float: lift grezzo
sc.cohens_d        # float: Cohen's d grezzo
sc.regime_breadth  # float: quota regimi significativi
sc.composite_score # float: 0–1, arrotondato a 4 decimali
sc.grade           # str: "A" | "B+" | "B" | "C"
```

---

### Step 7 — Compilazione del contratto

Un candidato viene promosso (`status = "HYPOTHESIS"`) solo se supera **tutti**
i gate seguenti:

| Gate | Parametro | Default | Note |
|---|---|---|---|
| IC ammesso | `ic_min_abs`, `ic_max_p` | 0.02, 0.05 | Logica `not (weak_ic AND weak_p)` |
| Lift ≥ soglia | `min_lift` | 0.08 | +8 punti percentuali rispetto al base rate |
| Cohen's d ≥ soglia | `min_cohens_d` | 0.15 | Dimensione effetto minima |
| Attivazioni ≥ soglia | `min_activations` | 30 | Stima stabile del win rate |
| Significatività statistica | `use_fdr`/`fdr_q` o `max_p_value` | BH q=0.10 | Vedi Step 8 |

`rejection_reasons` elenca **tutti** i gate falliti (non solo il primo), così è
possibile capire in modo completo perché un candidato è stato rifiutato:

```python
for c in contracts:
    if not c.promoted:
        print(c.event_candidate_id, c.rejection_reasons)
```

---

### Step 8 — Controllo FDR (Benjamini-Hochberg)

Quando `use_fdr = True` (default), prima della compilazione dei contratti viene
applicato il controllo FDR di Benjamini-Hochberg (BH) su tutti i p-value dei
t-test contemporaneamente:

```
BH a q = fdr_q (default 0.10) → al massimo 10% di falsi positivi tra i promossi
```

Il BH sostituisce la soglia `max_p_value` come criterio di significatività.

Il campo `fdr_promoted` nel contratto registra se il candidato supera il BH
indipendentemente dall'esito finale della promozione — utile per audit:

```python
# Candidati che hanno superato il BH ma sono stati rifiutati per altri gate
bh_ok_but_rejected = [c for c in contracts if c.fdr_promoted and not c.promoted]
```

Quando `use_fdr = False`, viene usata la soglia `max_p_value` direttamente.

---

## Struttura dati: `AlphaContract`

```python
c = promoted[0]

# Identificatori
c.alpha_id            # str: "ALPHA-BTC-1H-260610-000"
c.version             # str: "1.0"
c.discovery_date      # str: data ISO (oggi o AlphaConfig.discovery_date)
c.status              # str: "HYPOTHESIS" | "REJECTED"
c.pattern_family      # str: "mean_reversion" | "momentum" | "unspecified"

# Origine
c.asset, c.exchange, c.timeframe, c.direction  # str
c.event_candidate_id  # str: link all'EventCandidate sorgente
c.event_expression    # str: es. "rsi_14 < 30.5 AND spread_ema_9_25 < -0.012"

# Target
c.target_definition   # TargetDefinition
c.base_rate           # float: win rate senza filtro

# Misure statistiche
c.market_structure    # MarketStructure: Hurst + ACF (Step 2)
c.underlying_feature  # ICResult: IC della feature continua (Step 3)
c.event_stats         # EventStats: misure binarie (Step 4)
c.regime_analysis     # RegimeAnalysis: sensibilità al regime (Step 5)
c.alpha_score         # AlphaScore: score composito e grade (Step 6)

# Esito promozione
c.promoted            # bool
c.rejection_reasons   # list[str]: gate falliti (vuoto se promosso)
c.fdr_promoted        # bool | None

# Handoff a Rule Discovery
c.handoff_status      # str: "PENDING_RULE_DISCOVERY"
c.rule_discovery_response  # dict | None (Rule Discovery non ancora implementato)
```

#### Formato `alpha_id`

```
ALPHA-{asset}-{timeframe}-{stamp}-{idx:03d}
```

Dove `stamp` è la data di discovery in formato `AAMMGG` (es. `260610` per
2026-06-10) e `idx` è l'indice del candidato (000, 001, ...).

---

## Metodi di output

### `ad.run() → list[AlphaContract]`

Valuta tutti i candidati e restituisce l'intera lista (promossi + rifiutati).
Deve essere chiamato prima di qualunque altro metodo.

### `ad.promoted_contracts() → list[AlphaContract]`

Restituisce solo i contratti con `status == "HYPOTHESIS"`.
Richiede che `run()` sia stato eseguito.

### `ad.summary() → pd.DataFrame`

Riepilogo tabellare, ordinato per `composite_score` decrescente. Ogni riga
corrisponde a un contratto e contiene tutte le metriche chiave in colonne
piatte (utile per analisi esplorative e filtraggio):

```python
df = ad.summary()
df.columns
# alpha_id, status, promoted, event_candidate_id, expression, pattern_family,
# feature, ic, ic_p_value, ic_admitted, rolling_ic_stable, n_activations,
# win_rate, base_rate, lift, fwd_return_mean, cohens_d, t_stat, p_value,
# fdr_promoted, regime_dependency, regime_breadth, composite_score, grade,
# rejection_reasons
```

### `c.to_dict() → dict`

Dizionario piatto equivalente a una riga di `summary()`. Utile per costruire
DataFrame personalizzati da subset di contratti.

### `c.to_contract_dict() → dict`

Dizionario nidificato completo, pronto per serializzazione YAML/JSON:

```python
import json
for c in promoted:
    print(json.dumps(c.to_contract_dict(), indent=2))
```

---

## Configurazione completa

### `TargetDefinition`

| Parametro | Default | Descrizione |
|---|---|---|
| `holding_period_h` | `24` | Orizzonte forward in barre |
| `sell_pct` | `0.04` | Soglia rendimento per target binario (es. 0.04 = +4%) |
| `direction` | `"long"` | `"long"` o `"short"` |
| `fee_per_side` | `0.002` | Informativo; non detratto dal target |
| `asset` | `"ASSET"` | Nome dell'asset |
| `exchange` | `""` | Nome dell'exchange |
| `timeframe` | `"1H"` | Timeframe della barra |

### `PromotionThresholds`

| Parametro | Default | Descrizione |
|---|---|---|
| `ic_min_abs` | `0.02` | \|IC\| minimo per gate IC |
| `ic_max_p` | `0.05` | p-value massimo per gate IC |
| `min_lift` | `0.08` | Lift minimo (+8pp) |
| `min_cohens_d` | `0.15` | Cohen's d minimo |
| `max_p_value` | `0.05` | p-value massimo (se `use_fdr=False`) |
| `min_activations` | `30` | Attivazioni minime per stima stabile |
| `use_fdr` | `True` | Usa BH invece di `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate BH |

### `AlphaConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `target` | `TargetDefinition()` | Target economico |
| `thresholds` | `PromotionThresholds()` | Gate di ammissione e promozione |
| `close_col` | `"close"` | Colonna prezzo di chiusura |
| `timestamp_col` | `"open_dt"` | Colonna datetime (o nome del DatetimeIndex) |
| `regime_col` | `"regime"` | Colonna regime (da Market Context) |
| `regime_stable_col` | `"regime_stable"` | Colonna stabilità regime |
| `use_stable_regime_only` | `False` | Usa solo barre stable per analisi regime |
| `min_regime_obs` | `10` | Osservazioni minime per valutare un regime |
| `rolling_ic_window` | `None` | Finestra rolling IC (None → 60 giorni in barre) |
| `bars_per_day` | `None` | Barre per giorno (None → inferito dalla spaziatura) |
| `score_weights` | `(0.25, 0.30, 0.25, 0.20)` | Pesi: (IC, lift, cohens_d, breadth) |
| `discovery_date` | `None` | Data ISO per i contratti (None → oggi) |

---

## Pattern d'uso avanzati

### Ispezionare i contratti rifiutati

```python
contracts = ad.run()
for c in contracts:
    if not c.promoted:
        print(f"{c.event_candidate_id}: {c.rejection_reasons}")
```

Esempio di output:
```
EV-BTC-1H-001: ['lift 0.0312 < 0.08']
EV-BTC-1H-004: ['IC below admission (|IC|<0.02 and p>0.05)', 'n_activations 18 < 30']
```

### Filtrare per grade

```python
df = ad.summary()
grade_a = df[df["grade"] == "A"]
grade_b_plus = df[df["grade"].isin(["A", "B+"])]
```

### Analisi per regime

```python
for c in promoted:
    print(f"Event: {c.event_candidate_id}")
    print(f"  Regime dependency: {c.regime_analysis.dependency_type}")
    for rs in c.regime_analysis.per_regime:
        print(f"  {rs.regime}: IC={rs.ic:.3f}, win_rate={rs.win_rate:.3f}, strength={rs.strength}")
```

### Export YAML dei contratti promossi

```python
import yaml
for c in promoted:
    with open(f"{c.alpha_id}.yaml", "w") as f:
        yaml.dump(c.to_contract_dict(), f)
```

### Analisi senza Market Context

Se il DataFrame non contiene la colonna `regime`, il modulo salta l'analisi di
regime sensitivity e rinormalizza lo score escludendo il termine `regime_breadth`:

```python
# Funziona anche senza regime — regime_analysis.dependency_type sarà "unknown"
ad = AlphaDiscovery(kpi_without_regime, candidates, config)
contracts = ad.run()
# alpha_score.regime_breadth = NaN, ma composite_score è comunque calcolato
# su (IC, lift, cohens_d) con pesi rinormalizzati a 1.0
```

### Modifica dei gate di promozione

```python
from forgedge.alpha_discovery.models import PromotionThresholds

# Criteri più conservativi
strict = AlphaConfig(
    target=TargetDefinition(holding_period_h=48, sell_pct=0.06),
    thresholds=PromotionThresholds(
        min_lift=0.12,
        min_cohens_d=0.20,
        min_activations=50,
        fdr_q=0.05,
    )
)
ad = AlphaDiscovery(ed.df, candidates, strict)
```

---

## Primitive statistiche (`stats.py`)

Tutte le primitive statistiche sono implementate in **puro numpy** senza
dipendenze da scipy o statsmodels:

| Funzione | Algoritmo |
|---|---|
| `spearmanr(x, y)` | Correlazione di Spearman via ranking numpy |
| `cohens_d(group1, group2)` | `(mean1 - mean2) / std_pooled` |
| `ttest_ind(x, y, alternative)` | t-test indipendente, p-value via beta incompleta |
| `benjamini_hochberg(p_values, q)` | Controllo FDR Benjamini-Hochberg |
| `betai(a, b, x)` | Beta incompleta regolarizzata (frazione continua di Lentz) |

Le probabilità del t-test sono ottenute dalla funzione beta incompleta
regolarizzata implementata con l'algoritmo a frazione continua di Lentz
(*Numerical Recipes*). La precisione è ≈1e-6 rispetto ai valori scipy.

---

## Utilizzo downstream

Alpha Discovery produce contratti con `handoff_status = "PENDING_RULE_DISCOVERY"`.
Rule Discovery (Modulo 3, non ancora implementato) consumerà i contratti
promossi per eseguire un backtest realistico con meccaniche d'ordine, fee, e
validazione operativa dell'edge.

Il campo `rule_discovery_response` è riservato alla risposta di Rule Discovery
e rimane `None` fino all'implementazione del modulo.
