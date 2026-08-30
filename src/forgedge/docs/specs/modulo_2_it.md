# Modulo 2 — Alpha Discovery

Alpha Discovery è il terzo modulo della pipeline FORGE e il **primo che vede il
forward return**. Riceve la lista di `EventCandidate` prodotta da Event Discovery
e, per ogni candidato, **deriva il target economico direttamente dai dati** —
scegliendo l'orizzonte temporale con l'eccesso standardizzato dalla rotation
null più forte, derivando il take-profit dalla distribuzione delle escursioni
favorevoli, e misurando il potere predittivo rispetto al target derivato. L'output è una lista
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
scansiona una grid di orizzonti (`horizon_grid`, risolto dalla sessione in base
alla classe di timeframe — vedi la Configurazione completa) sull'IS e deriva
l'**eccesso di log-return** a ogni orizzonte:

```python
Δ_h = μ_cond_h − μ_base_h   # media condizionale meno la baseline incondizionata
```

`μ_cond_h` è il log-return medio delle barre attive all'orizzonte `h`;
`μ_base_h` è la media *incondizionata* del log-return su tutte le barre valide
allo stesso orizzonte. Sottrarre la baseline elimina il drift proprio
dell'asset dal segnale, così `Δ_h` riflette solo l'edge dell'evento — mai il
trend prevalente. Per questo la stessa regola (es. `RSI > 80`) può derivare
*short* in un bull run e *long* in un mercato laterale: si legge solo l'eccesso
rispetto al drift prevalente, mai un rendimento grezzo che mescola i due.

**Selezione dell'orizzonte `h*`:**
```python
z_h = Δ_h / σ_null,h          # eccesso standardizzato da una circular-rotation null
h*  = argmax_h |z_h|
```

`σ_null,h` **non** è una deflessione `1/√h`. Deriva da una **circular-rotation
null**: ogni shift circolare non banale della maschera di attivazione
dell'evento viene correlato con la stessa serie di forward return (calcolati
tutti insieme via cross-correlazione basata su FFT), e la deviazione standard
di questa distribuzione null empirica è `σ_null,h`.

Questo sostituisce un criterio precedente, più semplice — `score[h] = |Δ_h| /
√h`, una deflessione "simil-Sharpe" — abbandonato perché fallisce sugli
**eventi clusterizzati**. Un t-statistic ingenuo `T_h = Δ_h / (σ_cond / √n)`
tratta le finestre di forward return sovrapposte come osservazioni
indipendenti, quindi il suo denominatore si restringe con `√n` (correlato con
`√h`); per un evento le cui attivazioni arrivano a raffiche — il caso comune —
questo gonfia `|T_h|` sugli orizzonti lunghi e fissa sistematicamente `h*` al
bordo lungo della grid anche in assenza di un edge reale. La circular-rotation
null ricava la standardizzazione dalla struttura di autocorrelazione propria
dei dati invece di assumere indipendenza, eliminando questo bias.

La rotation null produce anche un p-value bidirezionale per orizzonte
(`p_value_by_h`); il controllo Benjamini-Hochberg a `fdr_q` su questo insieme
di p-value produce `h_sig`, gli orizzonti statisticamente distinguibili dalla
null. `h*` viene comunque sempre scelto come `argmax|z_h|` sull'**intera**
grid — `h_sig` non restringe mai la ricerca — ma quando `h*` cade fuori da
`h_sig` il target viene marcato `statistically_weak` (usato nello scoring
dello Step 6 e nel gate di direzione sotto) invece di essere scartato.

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
dt.holding_period_h        # int: orizzonte selezionato h*
dt.sell_pct                # float: quantile MFE a h*
dt.direction               # str: "long" | "short" | "undetermined"
dt.mean_advantage          # float: eccesso di log-return con segno Δ_h* a h*
dt.advantage_by_h          # dict[int, float]: eccesso di log-return Δ_h per orizzonte
dt.t_stat_by_h             # dict[int, float]: z_h standardizzato per orizzonte
dt.score_by_h              # dict[int, float]: score di selezione |z_h| per orizzonte
dt.p_value_by_h            # dict[int, float]: p-value della circular-rotation null per orizzonte (solo diagnostico)
dt.h_sig                   # tuple[int, ...]: orizzonti che superano il controllo BH a fdr_q (solo diagnostico)
dt.statistically_weak      # bool: True quando h* non è in h_sig
dt.fixed_target            # bool: True quando il target è specificato dall'utente (modalità fixed-target) invece che derivato
dt.data_derived_horizon_h  # int | None: solo modalità fixed-target — l'orizzonte che la derivazione dai dati avrebbe scelto
dt.data_derived_sell_pct   # float | None: solo modalità fixed-target — il sell_pct che la derivazione dai dati avrebbe prodotto
```

`direction = "undetermined"` (il contratto viene rifiutato) quando si
verifica **una qualsiasi** di queste condizioni:
- nessun orizzonte produce un eccesso di log-return `Δ_h` finito;
- `|z_h*| < min_direction_t` (default `0.5`) — l'eccesso all'orizzonte
  selezionato non è distinguibile dalla rotation null;
- `require_significant_direction = True` (default, su
  `PromotionThresholds`) **e** `h*` non è in `h_sig` — nessun orizzonte ha
  superato il gate Benjamini-Hochberg, quindi `argmax|z_h|` assegnerebbe
  altrimenti una direction equivalente a un lancio di moneta (spesso il bordo
  lungo della grid guidato dal drift). Impostare
  `require_significant_direction = False` per il comportamento legacy non
  bloccante — una direction viene sempre assegnata soggetta solo a
  `min_direction_t`, con l'evidenza debole segnalata via
  `statistically_weak` invece che bloccata.

Tutte le misure successive (IC, win rate, regime) sono calcolate **al target
derivato** (`h*`, `sell_pct*`, `direction*`).

#### Arricchimento della grid di orizzonti

`AlphaConfig.horizon_enrichment` (default `(0.5, 1.0, 2.0)`) aggiunge, **per
evento**, orizzonti attorno alla scala temporale strutturale propria
dell'evento alla `horizon_grid` di base scansionata sopra — un'unione, mai una
restrizione. Per ogni candidato, `EventCandidate.dominant_window()` restituisce
`w`, la finestra indicatore/trasformazione più lenta incorporata nelle
informazioni di condizionamento proprie dell'evento (es. un evento basato su
`ema_9` ha `w = 9`); per ogni moltiplicatore `m` in `horizon_enrichment`,
`round(m · w)` viene aggiunto agli orizzonti scansionati per quel candidato.
Così un evento `ema_9` scansiona anche `h ≈ 5, 9, 18` anche quando la grid di
base li salta — i moltiplicatori di default coprono la banda empiricamente
supportata (gli eventi di tipo reazione tendono a risolversi in circa metà
finestra; quelli di tipo ciclo in una-due finestre). Gli orizzonti arricchiti
sono limitati a `split // horizon_enrichment_min_obs` (default `20`) così che
una finestra di condizionamento lenta non possa richiedere un periodo di
possesso che i dati in-sample non possono sostenere statisticamente; il limite
non restringe mai la `horizon_grid` di base. Ogni orizzonte aggiunto è
conteggiato dal ledger di sessione e prezzato dalla rotation null a livello di
ricerca come ogni altro. Impostare `horizon_enrichment=None` (o `()`) per
disabilitarlo e scansionare solo la grid di base.

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
`"IC weak …"` in `diagnostics` ma non blocca la promozione.

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
**replicato sull'OOS tail** (l'ultimo `1 - train_ratio` del dataset, spostato
di `embargo_bars` se impostato — vedi la Configurazione completa):

- I forward return OOS sono orientati per la direction derivata
- Si misura win rate, lift, mean_advantage e t-test active vs inactive sull'OOS
- `oos.passed = True` se valgono entrambe le condizioni:
  1. `mean_advantage > 0` (l'advantage orientato rimane positivo sull'OOS)
  2. `p_value < oos_max_p` (default: 0.10, one-sided)

  Non è imposto un conteggio minimo di attivazioni come gate separato — il
  p-value incorpora già la dimensione campionaria. Un floor non
  parametrizzabile di 10 attivazioni innesca invece una diagnostica non
  bloccante sulla bassa affidabilità statistica; non esiste un campo
  `min_oos_activations` da configurare.

**La mancata conferma OOS produce una diagnostica non bloccante** —
`"OOS weak …"` in `diagnostics` — ma non impedisce la
promozione. Il segnale statistico OOS contribuisce al grade.

Output — `OOSValidation`:
```python
oos.n_bars                # int: barre nell'OOS window
oos.n_activations         # int: attivazioni con orizzonte completo nell'OOS
oos.mean_advantage        # float: rendimento medio orientato sull'OOS (>0 = confermato)
oos.t_stat                # float
oos.p_value               # float: t-test one-sided
oos.win_rate              # float: win rate OOS al target derivato
oos.base_rate             # float: base rate OOS al target derivato
oos.lift                  # float: lift OOS
oos.passed                # bool
oos.min_detectable_effect # float: Cohen's d minimo rilevabile a oos_max_p data la dimensione campionaria OOS — confrontare con il cohens_d IS per diagnosticare una finestra OOS sotto-potenziata
```

Quando `train_ratio = 1.0`, `oos_validation` è `None` per ogni contratto
(split OOS disabilitato — non raccomandato in produzione).

---

### Step 6 — Alpha scoring

Score composito (0–1), media pesata di cinque termini di qualità del segnale:

| Componente | Peso default | Normalizzazione |
|---|---|---|
| IC magnitude | 0.20 | `min(|IC| / 0.10, 1.0)` |
| Lift | 0.25 | `min(lift / 0.30, 1.0)` |
| Cohen's d | 0.15 | `clip(d / 0.80, -1.0, 1.0)` — **con segno** |
| `z` (eccesso rotation-null) | 0.25 | `min(|z_h*| / 3.0, 1.0)` |
| Regime breadth | 0.15 | `regime_breadth` (0–1) |

`z` è `|z_h*|`, la statistica di eccesso standardizzata dalla rotation null
all'orizzonte selezionato (Step 1) — il rapporto edge/rumore. Il termine
Cohen's d è normalizzato **con segno** invece che troncato a zero sul lato
basso: un Cohen's d negativo (il gruppo condizionato performa *peggio* del
background) penalizza attivamente il composite invece di contribuire nulla.
Quando il regime non è disponibile, il termine breadth viene rimosso e i pesi
rimanenti sono rinormalizzati. `score_weights` accetta ancora anche una
4-tupla legacy `(ic, lift, cohens_d, breadth)`, aggiornata con un peso `z` di
default.

Due ulteriori aggiustamenti si applicano dopo la somma pesata:
- se il target derivato è `statistically_weak` (`h*` fuori da `h_sig`, Step
  1), il composite viene **moltiplicato** per `statistically_weak_penalty`
  (default `0.6`) — un orizzonte selezionato esattamente dal bias di
  selezione che il controllo FDR esiste per intercettare non può classificarsi
  in alto;
- se la conferma OOS è passata (`oos_validation.passed`), viene **aggiunto**
  `oos_bonus` (default `0.05`), separando gli edge confermati da quelli non
  confermati.

Il risultato viene limitato a `[0, 1]`.

**Grade:** A ≥ 0.75 | B ≥ 0.50 | C ≥ 0.25 | D < 0.25

Tutti e quattro i grade vengono passati a Rule Discovery — il grade indica la
forza dell'evidenza statistica, non l'esclusione.

---

### Step 7 — Compilazione del contratto

L'**unico gate di rifiuto** è l'assenza di una direzione determinata:

| Gate | Condizione |
|---|---|
| **Hard (blocca)** | `direction == "undetermined"` — nessun eccesso di log-return finito su tutta la grid, `\|z_h*\| < min_direction_t`, oppure (con `require_significant_direction=True`, default) `h*` non in `h_sig` (vedi Step 1) |
| **Diagnostiche (non bloccanti)** | IC debole, lift < soglia, cohens_d < soglia, attivazioni insufficienti, non significativo FDR/p-value, OOS debole |

Le diagnostiche vivono in un campo dedicato, `diagnostics`. `rejection_reasons`
contiene solo ciò che ha effettivamente bloccato la promozione, quindi è **vuoto
su un contratto promosso**; `diagnostics` documenta le debolezze statistiche
rilevate ed è normalmente non vuoto sui contratti promossi.

```python
for c in contracts:
    print(f"{c.event_candidate_id}: promoted={c.promoted}, grade={c.alpha_score.grade}")
    for d in c.diagnostics:
        print(f"  {d}")
# Esempio:
# EVT-rsi_25-PR-0042: promoted=True, grade=C
#   lift 0.0520 < 0.08
#   OOS weak (p=0.143 vs 0.10, mean_adv=0.00210, n_act=7)
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
c.fee_per_side        # float: non detratto qui — ma è il costo che
                      # Rule Discovery addebita (risolto dalla sessione)

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
c.rejection_reasons   # list[str]: solo cause bloccanti — vuoto se promosso
c.diagnostics         # list[str]: osservazioni non bloccanti che pesano sul grade
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
regime_dependency, regime_breadth, composite_score, grade, rejection_reasons,
diagnostics
```

---

## Configurazione completa

### `AlphaConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `horizon_grid` | `UNSET` → risolto dalla sessione in base alla classe di timeframe: `(1,2,4,8,12,24)` su 1H/4H, `(1,2,3,5,7,10)` su 1D e oltre, `(1,2,5,10,20,50)` sub-orario (fallback standalone fuori da `forge()`: `(1,2,4,8,12,24)`) | Grid di orizzonti candidati (barre), scansionata nello Step 1 |
| `mfe_quantile` | `0.5` | Quantile MFE per derivare sell_pct (0.5 = mediana) |
| `mfe_floor` | `0.005` | Floor di sell_pct (50 bp) dopo il quantile |
| `train_ratio` | `0.7` | Frazione IS del dataset (0 < x ≤ 1.0) |
| `embargo_bars` | `0` | Barre di quarantena aggiuntive dopo lo split IS/OOS prima che inizi la conferma OOS — protegge dalla correlazione seriale oltre alla purga meccanica della finestra forward |
| `horizon_enrichment` | `(0.5, 1.0, 2.0)` | Moltiplicatori per evento di `EventCandidate.dominant_window()` aggiunti a `horizon_grid` (unione, mai restrizione). `None`/`()` disabilita l'arricchimento |
| `horizon_enrichment_min_obs` | `20` | Limite statistico per gli orizzonti arricchiti: `h <= split // horizon_enrichment_min_obs` |
| `thresholds` | `PromotionThresholds()` | Gate di ammissione/promozione — per lo più diagnostici (vedi sotto) |
| `asset` | `"ASSET"` | Metadato tracciabilità (copiato nel contratto e alpha_id) |
| `exchange` | `""` | Metadato tracciabilità |
| `timeframe` | `"1H"` | Non solo metadato: guida la risoluzione di sessione di `horizon_grid` e degli altri campi che contano barre |
| `fee_per_side` | `0.002` *(risolto dalla sessione)* | Non detratto dal target qui; è la base di costo che M3 addebita, propagata in `BacktestParams.fee` |
| `close_col` | `"close"` *(risolto dalla sessione)* | Colonna prezzo chiusura; si propaga a `BacktestParams.{target_col, buy_price_anchor}` |
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
| `fixed_target` | `None` | `TargetConfig` — se impostato, salta la *derivazione* del target e misura ogni candidato rispetto a questo target specificato dall'utente (vedi Modalità fixed-target sotto) |
| `fixed_target_diagnostic` | `True` | Solo modalità fixed-target: esegue comunque la derivazione dai dati in sola lettura per popolare le diagnostiche di convergenza `data_derived_*` |
| `target_mode` | `"proj"` | Definizione del target binario: `"proj"` misura l'eccesso rispetto al trend locale (PROJ_LOG); `"abs"` è il target legacy a rendimento assoluto. PROJ si applica solo ai long |
| `trend_sma_mult` | `2.0` | Solo PROJ_LOG: finestra SMA del trend = `round(trend_sma_mult · h)` barre |

### `PromotionThresholds`

La maggior parte di questi campi controlla **diagnostiche**, non gate di
promozione — l'unico gate bloccante in tutta Alpha Discovery è la
determinazione della direction (Step 1/7), e `min_direction_t` /
`require_significant_direction` sono i due campi che vi partecipano
effettivamente.

| Parametro | Default | Descrizione |
|---|---|---|
| `ic_min_abs` | `0.02` | Soglia \|IC\| per classificare IC come debole (diagnostica) |
| `ic_max_p` | `0.05` | p-value massimo per classificare IC come debole (diagnostica) |
| `min_lift` | `0.08` | Lift minimo (diagnostica) |
| `min_cohens_d` | `0.15` | Cohen's d minimo (diagnostica) |
| `max_p_value` | `0.05` | p-value massimo, raggiungibile solo con `use_fdr=False` (diagnostica; inerte sotto ogni preset, che impostano tutti `use_fdr=True`) |
| `use_fdr` | `True` | Usa BH invece di `max_p_value` |
| `fdr_q` | `0.10` | Target false-discovery rate BH (guida `h_sig` dello Step 1) |
| `oos_max_p` | `0.10` | p-value massimo one-sided per la conferma OOS |
| `min_direction_t` | `0.5` | \|z_h*\| minimo perché venga assegnata una direction — sotto questa soglia, `direction = "undetermined"` (**blocca la promozione**) |
| `require_significant_direction` | `True` | Una direction viene assegnata solo se `h*` è in `h_sig` (BH-significativo); altrimenti `direction = "undetermined"` (**blocca la promozione**). `False` ripristina il comportamento legacy non bloccante |

`PromotionThresholds` non ha campi `min_activations` o
`min_oos_activations` — non esiste alcun gate di promozione basato su un
conteggio minimo di attivazioni in Alpha Discovery; la dimensione campionaria
è assorbita direttamente nei p-value (i p-value della rotation null dello
Step 1, il t-test OOS).

### `TargetConfig` (modalità fixed-target)

Passato tramite `AlphaConfig.fixed_target` per bypassare la derivazione del
target per evento — vedi Modalità fixed-target sotto.

| Parametro | Default | Descrizione |
|---|---|---|
| `horizon` | *(obbligatorio)* | Periodo di possesso in barre (`> 0`); aggiunto a `horizon_grid` se assente |
| `min_return` | *(obbligatorio)* | Soglia take-profit come frazione (es. `0.02` = 2%), usata come `sell_pct` |
| `side` | *(obbligatorio)* | `"long"` o `"short"` — mai sovrascritto dai dati |
| `min_activations` | `10` | Floor del workflow TargetOptimizer per uno scoring del lift valido; ignorato dalla modalità fixed-target di Alpha Discovery |
| `min_lift_atoms` | `1.0` | Soglia di pruning del **1° passaggio** di TargetOptimizer (eventi atomici); ignorata dalla modalità fixed-target |
| `min_lift_result` | `1.0` | Soglia di pruning del **2° passaggio** di TargetOptimizer (set di risultati finale); ignorata dalla modalità fixed-target |
| `target_mode` | `"proj"` | `"abs"` o `"proj"` (PROJ_LOG) — vedi `AlphaConfig.target_mode` |
| `trend_sma_mult` | `2.0` | Moltiplicatore SMA del trend PROJ_LOG — vedi `AlphaConfig.trend_sma_mult` |

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
    # Score per orizzonte (|z_h|, l'eccesso standardizzato dalla rotation null)
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

## Modalità fixed-target

Il flusso di default *deriva* `(h*, sell_pct*, direction*)` per evento (Step
1). `AlphaConfig.fixed_target` è l'unica eccezione documentata: impostalo a un
`TargetConfig` e Alpha Discovery **salta interamente la derivazione**,
misurando ogni candidato rispetto a un singolo `(horizon, min_return, side)`
specificato dall'utente. Ogni misura downstream — IC, win rate, lift, Cohen's
d, sensibilità al regime, conferma OOS, scoring dello Step 6 — resta invariata
e viene calcolata rispetto a quel target fisso. Questo è il meccanismo che
`TargetOptimizer` usa internamente per valutare molti candidati evento rispetto
a un unico target economico comune, invece di lasciare che ciascuno scelga il
proprio.

```python
from forgedge import AlphaConfig
from forgedge.alpha_discovery.models import TargetConfig

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    fixed_target=TargetConfig(horizon=12, min_return=0.02, side="long"),
)
ad = AlphaDiscovery(ed.df, candidates, config)
contracts = ad.run()

c = contracts[0]
dt = c.derived_target
dt.fixed_target            # True
dt.holding_period_h        # 12 (l'orizzonte dell'utente, aggiunto a horizon_grid se assente)
dt.sell_pct                # 0.02 (da min_return)
dt.direction               # "long" (da side — mai sovrascritto dai dati)
dt.mean_advantage          # nan — non misurato in modalità fixed-target
dt.data_derived_horizon_h  # l'orizzonte che la derivazione dai dati *avrebbe* scelto
dt.data_derived_sell_pct   # il sell_pct che la derivazione dai dati *avrebbe* prodotto
```

Con `fixed_target_diagnostic=True` (default), la derivazione ordinaria dai
dati viene comunque eseguita **in sola lettura** insieme al target fisso,
popolando `data_derived_horizon_h`/`data_derived_sell_pct` e le diagnostiche
per orizzonte sul contratto — un consumer può quindi verificare
`data_derived_horizon_h ≈ holding_period_h` come segnale di convergenza che i
dati confermano indipendentemente l'orizzonte scelto dall'utente. Impostarlo a
`False` per un bypass puro e leggermente più veloce, con quelle diagnostiche
lasciate vuote.

Il target binario stesso (usato per win rate / lift / base rate) è governato
da `target_mode` (presente anche su `TargetConfig`, e replicato su
`AlphaConfig` per il flusso normale a target derivato):

- `"abs"` — il target legacy a rendimento assoluto: il rendimento forward
  grezzo confrontato con `min_return`.
- `"proj"` (default) — **PROJ_LOG**: il rendimento forward in eccesso rispetto
  al trend locale, calcolato contro una SMA di finestra
  `round(trend_sma_mult · h)` barre (moltiplicatore default `2.0`). Questo
  elimina il premio di trend di cui un evento long verrebbe altrimenti
  accreditato in un bull market — la stessa idea di rimozione del drift del
  target derivato (eccesso di log-return, Step 1), applicata qui al target
  specificato dall'utente. PROJ si applica solo ai **long**; un target
  `"short"` torna ad `"abs"` (il trend ribassista *è* l'alpha da catturare,
  non rumore da sottrarre). Ricade su `"abs"` con un warning quando la storia
  è più corta del warmup PROJ (`(trend_sma_mult + 1) · h` barre).

`TargetOptimizer` sta al di fuori del resolver di coerenza dei parametri (il
suo `discover_alpha()` costruisce un `AlphaConfig` interno senza passare dal
resolver di sessione di `forge()`), quindi sui dati giornalieri-o-più-lenti si
applica lo stesso fallback sulla grid oraria che colpisce l'uso standalone di
`AlphaConfig` — passa `horizon_grid` esplicitamente sulla config passata a
`discover_alpha()` per l'uso giornaliero-o-più-lento.

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
