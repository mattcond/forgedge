# Modulo 2 — Alpha Discovery

Alpha Discovery è il terzo modulo della pipeline FORGE e il **primo che vede il
forward return**. Riceve la lista di `EventCandidate` prodotta da Event Discovery
e, per ogni candidato, **deriva il target economico direttamente dai dati** —
scegliendo l'orizzonte temporale che massimizza la separazione statistica tra
barre con evento attivo e barre senza evento. L'output è una lista di
`AlphaContract` — uno per candidato — che registra il target derivato, tutte le
misure statistiche in-sample, la conferma out-of-sample, e l'esito della
promozione.

---

## Utilizzo di base

```python
from forgedge import (
    MarketContext,
    EventDiscovery,
    AlphaDiscovery, AlphaConfig,
)
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")

# Modulo 0 e 1: regime + scoperta eventi
enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)
candidates = ed.run()

# Modulo 2: derivazione target + misurazione predittiva
config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    # orizzonte derivato automaticamente dalla grid; nessun target da specificare
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

promoted = ad.promoted_contracts()
print(f"{len(promoted)} promossi su {len(contracts)} valutati")
print(ad.summary().head())
```

A differenza delle versioni precedenti, `AlphaConfig` **non richiede un
`TargetDefinition`**: horizon, sell_pct e direction sono derivati per ogni
evento dai dati stessi. `asset` e `timeframe` sono metadati di tracciabilità e
non influenzano i calcoli.

---

## Posizione nella pipeline

```
KPI Table + regime  (Modulo 0)
list[EventCandidate] (Modulo 1)
        │
        ▼
  AlphaDiscovery.run()
  ├─ Step 1: derivazione target per evento (IS)
  ├─ Step 2: struttura di mercato (IS)
  ├─ Step 3: IC e rolling stability (IS)
  ├─ Step 4: win rate e Cohen's d (IS)
  ├─ Step 5: sensibilità al regime (IS)
  ├─ OOS: conferma del target derivato (OOS tail)
  ├─ Step 6: alpha scoring
  └─ Step 7: compilazione contratto + FDR
        │
        ▼
  list[AlphaContract]   (tutti: HYPOTHESIS + REJECTED)
  promoted_contracts()  (solo HYPOTHESIS) ──► Rule Discovery (non implementato)
```

Il DataFrame è diviso cronologicamente in **in-sample (IS)** e
**out-of-sample (OOS)** tramite `train_ratio` (default: 70/30). Tutte le misure
statistiche (IC, win rate, regime) sono calcolate sull'IS; il target derivato
viene poi confermato sull'OOS tail.

---

## Pipeline a 8 step

### Step 1 — Derivazione del target per evento

Alpha Discovery **non riceve parametri economici in input**. Per ogni candidato,
scansiona una grid di orizzonti (`horizon_grid`, default: `(1, 2, 3, 4, 6, 8,
12, 16, 24, 36, 48)` barre) sull'IS e seleziona:

- **`holding_period_h`** — l'orizzonte che massimizza `|t-stat|` della
  differenza tra rendimenti forward sulle barre con evento vs senza evento;
- **`mean_advantage`** — il rendimento forward medio orientato delle barre
  attive a quell'orizzonte;
- **`sell_pct`** — `abs(mean_advantage)`: il take-profit baseline;
- **`direction`** — `"long"` se `mean_advantage > 0`, `"short"` se < 0,
  `"undetermined"` se non finito.

Il calcolo è **vettorializzato** su tutta la grid: un insieme di statistiche
sufficienti (count, sum, sum-of-squares) viene procalcolato una volta sola e
riusato per tutti i candidati, mantenendo il costo lineare nel numero di
candidati.

Output — `DerivedTarget`:
```python
dt.holding_period_h  # int: orizzonte selezionato
dt.sell_pct          # float: |mean_advantage| — take-profit baseline
dt.direction         # str: "long" | "short" | "undetermined"
dt.mean_advantage    # float: rendimento medio orientato a h*
dt.advantage_by_h    # dict[int, float]: rendimento medio per ciascun orizzonte
dt.t_stat_by_h       # dict[int, float]: t-stat per ciascun orizzonte
```

Tutte le misure successive (IC, win rate, regime) sono calcolate **al target
derivato** (`h*`, `sell_pct*`, `direction*`).

---

### Step 2 — Analisi struttura di mercato

Calcolata **una sola volta** per l'intera sessione sull'IS, al mediano degli
orizzonti della grid:

```python
ad.market_structure.hurst                 # float: esponente di Hurst (DFA)
ad.market_structure.hurst_interpretation  # "mean_reverting" | "random_walk" | "trending"
ad.market_structure.expected_family       # "mean_reversion" | "momentum" | "none"
ad.market_structure.autocorr              # dict[int, float]: ACF ai lag selezionati
```

`expected_family` viene copiato in ogni contratto come `pattern_family`
(`"unspecified"` se `"none"`). La soglia Hurst è `0.5`: sotto → mean-reverting,
sopra → trending.

---

### Step 3 — Misura dell'IC (Information Coefficient)

**Sull'IS**, la feature continua sottostante al primo componente del candidato
viene correlata con il forward return al derivato `h*` tramite Spearman:

```
IC = ρ(feature, fwd_return_h*)   [Spearman, solo IS]
```

Il risultato è **cached per `(feature, horizon)`**: candidati che condividono la
stessa feature e lo stesso `h*` ottengono lo stesso `ICResult` senza ricalcolo.

#### Gate di ammissione IC

Il candidato supera il gate IC se **non** sono entrambe vere:
- `|IC| < ic_min_abs` (default: 0.02) — IC debole
- `p_value > ic_max_p` (default: 0.05) — non significativo

Se l'IC è piccolo ma statisticamente significativo (basso p-value), il
candidato passa comunque il gate.

#### Rolling IC stability (sull'IS)

Stabilità valutata su ≈20 finestre equidistanti di ampiezza `rolling_ic_window`
(default: 60 giorni in barre):

```
stride = max(1, (n_is - window) // 20)
```

`rolling_ic_stable = True` se il segno IC rolling coincide con il globale in
almeno il 70% delle finestre.

---

### Step 4 — Analisi win rate (IS, al target derivato)

I forward return sono **orientati** prima del calcolo: per direction `"short"`,
il rendimento è moltiplicato per -1 così che "favorevole alla trade" sia sempre
positivo e Cohen's d / t-test leggano nello stesso modo per long e short.

| Metrica | Descrizione |
|---|---|
| `n_activations` | Barre IS con evento attivo e target valido |
| `win_rate` | `mean(target_binary_is)` sulle barre attive |
| `lift` | `win_rate - base_rate_is` |
| `fwd_return_mean` | Media rendimenti orientati sulle barre attive (IS) |
| `cohens_d` | `(mean_active - mean_inactive) / std_pooled` |
| `t_stat`, `p_value` | t-test one-sided (`alternative="greater"`) |

`base_rate_is` è calcolato **sull'IS** al target derivato — ogni candidato ha il
proprio base rate (diverso orizzonte e sell_pct → diverso base rate).

---

### Step 5 — Sensibilità al regime (IS)

Per ogni regime, con almeno `min_regime_obs` osservazioni sull'IS:

1. IC di Spearman feature vs `fwd_return_h*` nel regime (IS)
2. Win rate condizionale nel regime (IS)

I risultati per regime sono **cached per `(feature, horizon, regime)`** in modo
simile all'IC globale.

Se `use_stable_regime_only = True`, vengono usate solo le barre IS con
`regime_stable = True`.

Classificazione forza e dependency type: identica alla versione precedente
(strong/moderate/negligible/insufficient; agnostic/conditional/specific/broken/unknown).

---

### Validazione OOS

Dopo tutte le misure IS, il target derivato `(h*, sell_pct*, direction*)` viene
**replicato sull'OOS tail** (l'ultimo `1 - train_ratio` del dataset):

- I forward return OOS sono orientati per la direction derivata
- Si misura win rate, lift, mean_advantage e t-test active vs inactive sull'OOS
- Il candidato **supera la conferma OOS** (`oos.passed = True`) se:
  1. `n_oos_activations >= min_oos_activations` (default: 10)
  2. `mean_advantage > 0` (l'advantage orientato rimane positivo sull'OOS)
  3. `p_value < oos_max_p` (default: 0.10)

La mancata conferma OOS è un gate di rifiuto: il candidato viene promosso solo
se tutti i gate IS **e** la conferma OOS sono superati.

Output — `OOSValidation`:
```python
oos.n_bars          # int: barre nell'OOS window
oos.n_activations   # int: attivazioni con orizzonte completo nell'OOS
oos.mean_advantage  # float: rendimento medio orientato sull'OOS (>0 = confermato)
oos.t_stat          # float
oos.p_value         # float: t-test one-sided
oos.win_rate        # float: win rate OOS al target derivato
oos.base_rate       # float: base rate OOS al target derivato
oos.lift            # float: lift OOS
oos.passed          # bool: True se supera i 3 criteri
```

Quando `train_ratio = 1.0`, l'IS copre tutto il dataset e `oos_validation` è
`None` (split OOS disabilitato — sconsigliato in produzione).

---

### Step 6 — Alpha scoring

Score composito (0–1) invariato rispetto alla versione precedente:

| Componente | Peso default | Normalizzazione |
|---|---|---|
| IC magnitude | 0.25 | `min(\|IC\| / 0.10, 1.0)` |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` |
| Regime breadth | 0.20 | `regime_breadth` (0–1) |

Quando il regime non è disponibile, il termine breadth è rimosso e i pesi
rimanenti sono rinormalizzati. Grade: A ≥ 0.75, B+ ≥ 0.60, B ≥ 0.45, C < 0.45.

---

### Step 7 — Compilazione del contratto

Un candidato viene promosso (`status = "HYPOTHESIS"`) solo se supera **tutti** i
gate seguenti:

| Gate | Parametro | Default |
|---|---|---|
| Target derivabile | — | direction ≠ "undetermined" |
| IC ammesso | `ic_min_abs`, `ic_max_p` | 0.02, 0.05 |
| Lift ≥ soglia | `min_lift` | 0.08 |
| Cohen's d ≥ soglia | `min_cohens_d` | 0.15 |
| Attivazioni ≥ soglia | `min_activations` | 30 |
| Significatività FDR/p | `use_fdr`/`fdr_q` o `max_p_value` | BH q=0.10 |
| Conferma OOS | `oos_max_p`, `min_oos_activations` | 0.10, 10 |

`rejection_reasons` elenca tutti i gate falliti:
```python
for c in contracts:
    if not c.promoted:
        print(c.event_candidate_id, c.rejection_reasons)
```

Il BH FDR (Step 8) è applicato ai p-value del t-test IS di tutti i candidati
simultaneamente prima della compilazione.

---

## Struttura dati: `AlphaContract`

```python
c = promoted[0]

# Identificatori
c.alpha_id            # str: "ALPHA-BTC-1H-260610-000"
c.version             # str: "1.0"
c.discovery_date      # str: data ISO
c.status              # str: "HYPOTHESIS" | "REJECTED"
c.pattern_family      # str: "mean_reversion" | "momentum" | "unspecified"

# Metadati di scope (tracciabilità)
c.asset               # str
c.exchange            # str
c.timeframe           # str
c.fee_per_side        # float: informativo, non detratto

# Origine e target derivato
c.event_candidate_id  # str: link all'EventCandidate sorgente
c.event_expression    # str: espressione booleana dell'evento
c.direction           # str: "long" | "short" | "undetermined"
c.derived_target      # DerivedTarget: target derivato dai dati (Step 1)
c.base_rate           # float: base rate IS al target derivato

# Misure statistiche IS
c.market_structure    # MarketStructure: Hurst + ACF (Step 2)
c.underlying_feature  # ICResult: IC IS (Step 3)
c.event_stats         # EventStats: win rate IS (Step 4)
c.regime_analysis     # RegimeAnalysis: sensibilità regime IS (Step 5)

# Conferma OOS
c.oos_validation      # OOSValidation | None

# Score e promozione
c.alpha_score         # AlphaScore: score composito e grade (Step 6)
c.promoted            # bool
c.rejection_reasons   # list[str]: gate falliti (vuoto se promosso)
c.fdr_promoted        # bool | None

# Handoff a Rule Discovery
c.handoff_status      # str: "PENDING_RULE_DISCOVERY"
c.rule_discovery_response  # dict | None
```

---

## Metodi di output

### `ad.run() → list[AlphaContract]`

Deriva il target per ogni evento, esegue tutte le misure IS + OOS e restituisce
la lista completa (promossi + rifiutati). Deve essere chiamato per primo.

Proprietà popolate da `run()`:
- `ad.market_structure` — struttura di mercato IS
- `ad.split_idx` — indice di riga del confine IS/OOS

### `ad.promoted_contracts() → list[AlphaContract]`

Solo contratti con `status == "HYPOTHESIS"`.

### `ad.summary() → pd.DataFrame`

Riepilogo tabellare ordinato per `composite_score` decrescente. Include le
colonne OOS rispetto alla versione precedente:

```
alpha_id, status, promoted, event_candidate_id, expression, pattern_family,
holding_period_h, sell_pct, direction, mean_advantage,
feature, ic, ic_p_value, ic_admitted, rolling_ic_stable,
n_activations, win_rate, base_rate, lift, fwd_return_mean, cohens_d, t_stat, p_value,
fdr_promoted, oos_passed, oos_p_value, oos_lift,
regime_dependency, regime_breadth, composite_score, grade, rejection_reasons
```

### `c.to_dict() → dict`

Dizionario piatto (una riga di `summary()`).

### `c.to_contract_dict() → dict`

Dizionario nidificato completo per serializzazione YAML/JSON.

---

## Configurazione completa

### `AlphaConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `horizon_grid` | `(1,2,3,4,6,8,12,16,24,36,48)` | Grid di orizzonti candidati (barre) |
| `train_ratio` | `0.7` | Frazione IS del dataset (0 < x ≤ 1.0) |
| `thresholds` | `PromotionThresholds()` | Gate di ammissione e promozione |
| `asset` | `"ASSET"` | Metadato tracciabilità (copiato nel contratto e alpha_id) |
| `exchange` | `""` | Metadato tracciabilità |
| `timeframe` | `"1H"` | Metadato tracciabilità |
| `fee_per_side` | `0.002` | Informativo; non detratto dal target |
| `close_col` | `"close"` | Colonna prezzo chiusura |
| `timestamp_col` | `"open_dt"` | Colonna datetime (o nome del DatetimeIndex) |
| `regime_col` | `"regime"` | Colonna regime (da Market Context) |
| `regime_stable_col` | `"regime_stable"` | Colonna stabilità regime |
| `use_stable_regime_only` | `False` | Usa solo barre stable per analisi regime |
| `min_regime_obs` | `10` | Osservazioni minime IS per valutare un regime |
| `rolling_ic_window` | `None` | Finestra rolling IC (None → 60 giorni in barre) |
| `bars_per_day` | `None` | Barre per giorno (None → inferito dalla spaziatura) |
| `score_weights` | `(0.25, 0.30, 0.25, 0.20)` | Pesi: (IC, lift, cohens_d, breadth) |
| `discovery_date` | `None` | Data ISO per i contratti (None → oggi) |

### `PromotionThresholds`

| Parametro | Default | Descrizione |
|---|---|---|
| `ic_min_abs` | `0.02` | \|IC\| minimo per gate IC |
| `ic_max_p` | `0.05` | p-value massimo per gate IC |
| `min_lift` | `0.08` | Lift minimo (+8pp) |
| `min_cohens_d` | `0.15` | Cohen's d minimo |
| `max_p_value` | `0.05` | p-value massimo (se `use_fdr=False`) |
| `min_activations` | `30` | Attivazioni IS minime |
| `use_fdr` | `True` | Usa BH invece di `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate BH |
| `oos_max_p` | `0.10` | p-value massimo per la conferma OOS |
| `min_oos_activations` | `10` | Attivazioni OOS minime per la conferma |

---

## Pattern d'uso avanzati

### Configurare la grid e lo split IS/OOS

```python
config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72),   # custom grid
    train_ratio=0.75,                        # 75% IS / 25% OOS
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()
print(f"IS bars: {ad.split_idx}, OOS bars: {len(ad._frame) - ad.split_idx}")
```

### Ispezionare il target derivato

```python
for c in promoted:
    dt = c.derived_target
    print(f"{c.event_candidate_id}: h={dt.holding_period_h}, "
          f"direction={dt.direction}, sell_pct={dt.sell_pct:.4f}")
    # Profilo completo degli orizzonti
    for h, adv in dt.advantage_by_h.items():
        print(f"  h={h}: advantage={adv:.4f}, t={dt.t_stat_by_h[h]:.2f}")
```

### Ispezionare la conferma OOS

```python
for c in contracts:
    oos = c.oos_validation
    if oos is not None:
        print(f"{c.event_candidate_id}: OOS passed={oos.passed}, "
              f"lift={oos.lift:.4f}, p={oos.p_value:.4f}")
```

### Candidati rifiutati solo per OOS

```python
# Promettenti IS ma non confermati OOS
is_ok_oos_fail = [
    c for c in contracts
    if not c.promoted
    and c.oos_validation is not None
    and not c.oos_validation.passed
    and all("OOS" not in r for r in c.rejection_reasons[:5])  # altri gate ok
]
```

### Disabilitare lo split OOS (non raccomandato)

```python
# train_ratio=1.0 disabilita lo split; oos_validation sarà None per ogni contratto
config = AlphaConfig(train_ratio=1.0, asset="BTC")
```

### Inasprire i criteri di promozione

```python
from forgedge.alpha_discovery.models import PromotionThresholds

strict = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    horizon_grid=(6, 12, 24, 48),
    train_ratio=0.75,
    thresholds=PromotionThresholds(
        min_lift=0.12,
        min_cohens_d=0.20,
        min_activations=50,
        fdr_q=0.05,
        oos_max_p=0.05,
        min_oos_activations=15,
    )
)
```

### Export YAML dei contratti promossi

```python
import yaml
for c in promoted:
    with open(f"{c.alpha_id}.yaml", "w") as f:
        yaml.dump(c.to_contract_dict(), f)
```

---

## Primitive statistiche (`stats.py`)

Tutte implementate in **puro numpy** senza dipendenze da scipy o statsmodels:

| Funzione | Algoritmo |
|---|---|
| `spearmanr(x, y)` | Correlazione di Spearman via ranking numpy |
| `cohens_d(group1, group2)` | `(mean1 - mean2) / std_pooled` |
| `ttest_ind(x, y, alternative)` | t-test indipendente, p-value via beta incompleta |
| `benjamini_hochberg(p_values, q)` | Controllo FDR Benjamini-Hochberg |
| `betai(a, b, x)` | Beta incompleta regolarizzata (frazione continua di Lentz) |

---

## Utilizzo downstream

Alpha Discovery produce contratti con `handoff_status = "PENDING_RULE_DISCOVERY"`.
Rule Discovery (Modulo 3, non ancora implementato) consumerà i contratti
promossi per eseguire un backtest realistico con meccaniche d'ordine, fee, e
validazione operativa dell'edge.

Il `derived_target` nei contratti promossi costituisce il punto di partenza per
la calibrazione dei parametri operativi da parte di Rule Discovery:
`holding_period_h` e `sell_pct` sono candidati, non parametri validati — sono
prodotti da un'ottimizzazione IS su grid e la conferma OOS mostra che il segnale
persiste fuori dal campione, ma il dimensionamento preciso spetta a Rule
Discovery.
