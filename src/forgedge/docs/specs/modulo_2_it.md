# Modulo 2 — Alpha Discovery

Alpha Discovery è il terzo modulo della pipeline FORGE e il **primo che vede il
forward return**. Riceve la lista di `EventCandidate` prodotta da Event Discovery
e, per ogni candidato, **deriva il target economico direttamente dai dati** —
scegliendo l'orizzonte temporale con il miglior rapporto segnale/orizzonte,
derivando il take-profit dalla distribuzione delle escursioni favorevoli, e
misurando il potere predittivo rispetto al target derivato. L'output è una lista
di `AlphaContract` — uno per candidato — che registra il target derivato, tutte
le misure statistiche in-sample, la conferma out-of-sample e il grade A–D.

**Principio di promozione:** tutti i contratti con direzione determinata
(`"long"` o `"short"`) vengono promossi a `HYPOTHESIS` e passati a Rule
Discovery. Le misure statistiche (IC, lift, Cohen's d, FDR, OOS) producono
diagnostiche non bloccanti che informano il grade ma non impediscono la
promozione. Rule Discovery è il giudice economico finale.

---

## Utilizzo di base

```python
from forgedge import (
    MarketContext, EventDiscovery,
    AlphaDiscovery, AlphaConfig,
)
import pandas as pd

kpi = pd.read_parquet("kpi_table.parquet")

enriched = MarketContext(kpi).run()
ed = EventDiscovery(enriched)
candidates = ed.run()

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    # orizzonte, sell_pct e direction sono derivati automaticamente
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

# Tutti i contratti con direzione determinata sono promossi
promoted = ad.promoted_contracts()
print(f"{len(promoted)} promossi (HYPOTHESIS) su {len(contracts)} valutati")
print(ad.summary()[["expression", "direction", "holding_period_h",
                    "sell_pct", "grade", "oos_passed"]].head())
```

`AlphaConfig` **non richiede un `TargetDefinition`**: horizon, sell_pct e
direction sono derivati per ogni evento dai dati stessi. `asset` e `timeframe`
sono metadati di tracciabilità e non influenzano i calcoli.

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
  ├─ Step 6: alpha scoring (A–D)
  └─ Step 7: compilazione contratto
        │
        ▼
  list[AlphaContract]   (tutti: HYPOTHESIS + REJECTED)
  promoted_contracts()  (HYPOTHESIS: direction determinata) ──► Rule Discovery
```

Il DataFrame è diviso cronologicamente in **in-sample (IS)** e
**out-of-sample (OOS)** tramite `train_ratio` (default: 70/30). Tutte le misure
statistiche (IC, win rate, regime) sono calcolate sull'IS; il target derivato
viene confermato sull'OOS tail.

---

## Pipeline a 8 step

### Step 1 — Derivazione del target per evento

Alpha Discovery **non riceve parametri economici in input**. Per ogni candidato,
scansiona una grid di orizzonti (`horizon_grid`, default: `(1, 2, 3, 4, 6, 8,
12, 16, 24, 36, 48)` barre) sull'IS e seleziona:

**Selezione dell'orizzonte `h*`:**
```python
score[h] = |mean_advantage[h]| / sqrt(h)
h*       = argmax_h score[h]
```

Il criterio `|mean_advantage| / √h` è una deflessione simil-Sharpe che
bilancia la grandezza del vantaggio con la varianza crescente dell'orizzonte.
Evita il bias sistematico del max-t-stat verso orizzonti brevi, dove il
denominatore del t-test è strutturalmente piccolo.

**Derivazione di `sell_pct`:**
```python
MFE_i    = max escursione favorevole nelle h* barre successive alla barra i attiva
sell_pct = max(quantile(MFE, mfe_quantile), mfe_floor)
```

`sell_pct` è il quantile `mfe_quantile` (default: 0.5 = mediana) della
distribuzione delle Maximum Favorable Excursion delle barre attive IS a `h*`.
Questo àncora il take-profit baseline alla distribuzione reale delle escursioni,
non a una media che include barre perdenti. Il floor `mfe_floor` (default: 50 bp)
garantisce un take-profit operativamente significativo.

**Output — `DerivedTarget`:**
```python
dt.holding_period_h  # int: orizzonte selezionato
dt.sell_pct          # float: quantile MFE a h*
dt.direction         # str: "long" | "short" | "undetermined"
dt.mean_advantage    # float: rendimento medio orientato a h*
dt.advantage_by_h    # dict[int, float]: rendimento medio per ciascun orizzonte
dt.t_stat_by_h       # dict[int, float]: t-stat per ciascun orizzonte
dt.score_by_h        # dict[int, float]: |mean_advantage| / sqrt(h) per orizzonte
```

Se nessun orizzonte produce un vantaggio finito, `direction = "undetermined"` e
il contratto viene rifiutato. Tutte le misure successive (IC, win rate, regime)
sono calcolate **al target derivato** (`h*`, `sell_pct*`, `direction*`).

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
(`"unspecified"` se `"none"`). La soglia Hurst è 0.5: sotto → mean-reverting,
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

#### Gate di ammissione IC (non bloccante)

Se **entrambe** le condizioni sono vere, l'IC è classificato come debole:
- `|IC| < ic_min_abs` (default: 0.02)
- `p_value > ic_max_p` (default: 0.05)

Se l'IC è piccolo ma statisticamente significativo, il candidato è comunque
ammesso (`ic.admitted = True`). Un IC debole produce una diagnostica
`"[diagnostic] IC weak …"` in `rejection_reasons` ma non blocca la promozione.

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
positivo.

| Metrica | Descrizione |
|---|---|
| `n_activations` | Barre IS con evento attivo e target valido |
| `win_rate` | `mean(target_binary_is)` sulle barre attive |
| `lift` | `win_rate - base_rate_is` |
| `fwd_return_mean` | Media rendimenti orientati sulle barre attive (IS) |
| `cohens_d` | `(mean_active - mean_inactive) / std_pooled` |
| `t_stat`, `p_value` | t-test one-sided (`alternative="greater"`) |

`base_rate_is` è calcolato **sull'IS** al target derivato — ogni candidato ha il
proprio base rate. Lift, cohens_d e activations producono diagnostiche non
bloccanti se sotto soglia.

---

### Step 5 — Sensibilità al regime (IS)

Per ogni regime, con almeno `min_regime_obs` osservazioni sull'IS:

1. IC di Spearman feature vs `fwd_return_h*` nel regime (IS)
2. Win rate condizionale nel regime (IS)

I risultati per regime sono **cached per `(feature, horizon, regime)`** in modo
simile all'IC globale. Se `use_stable_regime_only = True`, vengono usate solo le
barre IS con `regime_stable = True`.

Classificazione forza: `strong` / `moderate` / `negligible` / `insufficient`.
Dependency type: `agnostic` / `conditional` / `specific` / `broken` / `unknown`.

---

### Validazione OOS (non bloccante)

Dopo tutte le misure IS, il target derivato `(h*, sell_pct*, direction*)` viene
**replicato sull'OOS tail** (l'ultimo `1 - train_ratio` del dataset):

- I forward return OOS sono orientati per la direction derivata
- Si misura win rate, lift, mean_advantage e t-test active vs inactive sull'OOS
- `oos.passed = True` se:
  1. `n_oos_activations >= min_oos_activations` (default: 10)
  2. `mean_advantage > 0` (l'advantage orientato rimane positivo sull'OOS)
  3. `p_value < oos_max_p` (default: 0.10)

**La mancata conferma OOS produce una diagnostica non bloccante** —
`"[diagnostic] OOS weak …"` in `rejection_reasons` — ma non impedisce la
promozione. Il segnale statistico OOS contribuisce al grade.

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
oos.passed          # bool
```

Quando `train_ratio = 1.0`, `oos_validation` è `None` per ogni contratto
(split OOS disabilitato — non raccomandato in produzione).

---

### Step 6 — Alpha scoring

Score composito (0–1):

| Componente | Peso default | Normalizzazione |
|---|---|---|
| IC magnitude | 0.25 | `min(|IC| / 0.10, 1.0)` |
| Lift | 0.30 | `min(lift / 0.30, 1.0)` |
| Cohen's d | 0.25 | `min(d / 0.80, 1.0)` |
| Regime breadth | 0.20 | `regime_breadth` (0–1) |

Quando il regime non è disponibile, il termine breadth è rimosso e i pesi
rimanenti sono rinormalizzati.

**Grade:** A ≥ 0.75 | B ≥ 0.50 | C ≥ 0.25 | D < 0.25

Tutti e quattro i grade vengono passati a Rule Discovery — il grade indica la
forza dell'evidenza statistica, non l'esclusione.

---

### Step 7 — Compilazione del contratto

L'**unico gate di rifiuto** è l'assenza di una direzione determinata:

| Gate | Condizione |
|---|---|
| **Hard (blocca)** | `direction == "undetermined"` — nessun vantaggio finito su tutta la grid |
| **Diagnostiche (non bloccanti)** | IC debole, lift < soglia, cohens_d < soglia, attivazioni insufficienti, non significativo FDR/p-value, OOS debole |

Le diagnostiche sono annotate con il prefisso `[diagnostic]` in
`rejection_reasons`. Un contratto promosso può avere `rejection_reasons`
non vuoto — la lista documenta le debolezze statistiche rilevate.

```python
for c in contracts:
    print(f"{c.event_candidate_id}: promoted={c.promoted}, grade={c.alpha_score.grade}")
    for r in c.rejection_reasons:
        print(f"  {r}")
# Esempio:
# EVT-rsi_25-PR-0042: promoted=True, grade=C
#   [diagnostic] lift 0.0520 < 0.08
#   [diagnostic] OOS weak (p=0.143 vs 0.10, mean_adv=0.00210, n_act=7)
```

Il BH FDR è applicato ai p-value IS di tutti i candidati simultaneamente;
`fdr_promoted` registra l'esito ma non blocca la promozione.

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
c.promoted            # bool: True se direction è "long" o "short"
c.rejection_reasons   # list[str]: diagnostiche (prefisso [diagnostic] = non bloccante)
c.fdr_promoted        # bool | None: esito BH FDR (non bloccante)

# Handoff a Rule Discovery
c.handoff_status      # str: "PENDING_RULE_DISCOVERY"
c.rule_discovery_response  # dict | None
```

---

## Metodi di output

### `ad.run() → list[AlphaContract]`

Deriva il target per ogni evento, esegue tutte le misure IS + OOS e restituisce
la lista completa. Deve essere chiamato per primo.

Proprietà popolate da `run()`:
- `ad.market_structure` — struttura di mercato IS
- `ad.split_idx` — indice di riga del confine IS/OOS

### `ad.promoted_contracts() → list[AlphaContract]`

Solo contratti con `status == "HYPOTHESIS"` (direction determinata).

### `contract.persist(path)`

Salva l'`AlphaContract` su disco come file pickle. Il contratto può essere
ricaricato in una sessione successiva e passato direttamente a `RuleDiscovery`
senza rieseguire Alpha Discovery.

```python
import pickle, pathlib

pathlib.Path("contracts").mkdir(exist_ok=True)
for c in promoted:
    c.persist(f"contracts/{c.alpha_id}.pkl")

# Ricaricare in una sessione successiva
contract = pickle.load(open("contracts/ALPHA-BTC-1H-000.pkl", "rb"))
```

### `ad.summary() → pd.DataFrame`

Riepilogo tabellare ordinato per `composite_score` decrescente:

```
alpha_id, status, promoted, event_candidate_id, expression, pattern_family,
holding_period_h, sell_pct, direction, mean_advantage,
feature, ic, ic_p_value, ic_admitted, rolling_ic_stable,
n_activations, win_rate, base_rate, lift, fwd_return_mean, cohens_d, t_stat, p_value,
fdr_promoted, oos_passed, oos_p_value, oos_lift,
regime_dependency, regime_breadth, composite_score, grade, rejection_reasons
```

---

## Configurazione completa

### `AlphaConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `horizon_grid` | `(1,2,3,4,6,8,12,16,24,36,48)` | Grid di orizzonti candidati (barre) |
| `mfe_quantile` | `0.5` | Quantile MFE per derivare sell_pct (0.5 = mediana) |
| `mfe_floor` | `0.005` | Floor di sell_pct (50 bp) dopo il quantile |
| `train_ratio` | `0.7` | Frazione IS del dataset (0 < x ≤ 1.0) |
| `thresholds` | `PromotionThresholds()` | Soglie diagnostiche (non gate di promozione) |
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
| `score_weights` | `(0.20, 0.25, 0.15, 0.25, 0.15)` | Pesi: (IC, lift, cohens_d, z, breadth). Accetta anche la 4-tupla legacy (IC, lift, cohens_d, breadth). |
| `statistically_weak_penalty` | `0.6` | Moltiplicatore del composite quando `statistically_weak=True`. |
| `oos_bonus` | `0.05` | Bonus additivo al composite quando la conferma OOS passa. |
| `discovery_date` | `None` | Data ISO per i contratti (None → oggi) |

### `PromotionThresholds`

Questi parametri controllano le **diagnostiche** — non gate di promozione.

| Parametro | Default | Descrizione |
|---|---|---|
| `ic_min_abs` | `0.02` | Soglia \|IC\| per classificare IC come debole |
| `ic_max_p` | `0.05` | p-value massimo per classificare IC come debole |
| `min_lift` | `0.08` | Lift minimo (diagnostica) |
| `min_cohens_d` | `0.15` | Cohen's d minimo (diagnostica) |
| `max_p_value` | `0.05` | p-value massimo (se `use_fdr=False`) (diagnostica) |
| `min_activations` | `30` | Attivazioni IS minime (diagnostica) |
| `use_fdr` | `True` | Usa BH invece di `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate BH |
| `oos_max_p` | `0.10` | p-value massimo per la conferma OOS (diagnostica) |
| `min_oos_activations` | `10` | Attivazioni OOS minime per la conferma (diagnostica) |

---

## Pattern d'uso avanzati

### Configurare la grid e lo split IS/OOS

```python
config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72),
    mfe_quantile=0.6,        # take-profit più aggressivo (60° percentile MFE)
    mfe_floor=0.010,         # floor a 100 bp
    train_ratio=0.75,
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
    # Score per orizzonte (|mean_advantage| / sqrt(h))
    best_h = max(dt.score_by_h, key=dt.score_by_h.get)
    print(f"  h* score={dt.score_by_h[best_h]:.5f}  "
          f"mean_adv={dt.advantage_by_h[best_h]:.4f}")
```

### Leggere le diagnostiche su un promosso

```python
for c in promoted:
    if c.rejection_reasons:
        print(f"{c.event_candidate_id} (grade={c.alpha_score.grade}):")
        for r in c.rejection_reasons:
            print(f"  {r}")
```

### Filtrare per robustezza OOS

```python
# Contratti promossi con conferma OOS
oos_robust = [
    c for c in promoted
    if c.oos_validation is not None and c.oos_validation.passed
]

# Contratti promossi con OOS debole (da usare con cautela in Rule Discovery)
oos_weak = [
    c for c in promoted
    if c.oos_validation is None or not c.oos_validation.passed
]
```

### Disabilitare lo split OOS (non raccomandato)

```python
config = AlphaConfig(train_ratio=1.0, asset="BTC")
# oos_validation sarà None per ogni contratto
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
Rule Discovery (Modulo 3) consuma i contratti promossi per eseguire un backtest
realistico con meccaniche d'ordine, fee, e validazione operativa dell'edge.

Il `derived_target` è il punto di partenza per Rule Discovery: `holding_period_h`
e `sell_pct` sono candidati, non parametri validati — il dimensionamento preciso
spetta a Rule Discovery, che li usa come centro della grid operativa.
