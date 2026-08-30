# FORGE — Rule Discovery Pipeline
> Terzo modulo della pipeline FORGE.
> Riceve un **Alpha Contract** da Alpha Discovery e risponde a una domanda
> operativa: il pattern statisticamente evidenziato è sfruttabile in un
> backtest realistico — con fee, fill rate, entry a mercato o a limite e
> target discreto?
> Per default (`entry_mode="auto"`, #185) la valutazione è a due stadi: uno
> Stadio 1 a **entry di mercato** (fill ≈ 100%) decide il verdetto in modo
> autoritativo; uno Stadio 2 opzionale raffina l'entry con un limit order
> solo se supera tre condizioni out-of-sample. `entry_mode="limit"` resta
> disponibile per i casi in cui il limit order stesso *è* la strategia.
> Emette un verdetto (`EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA`)
> e restituisce l'Alpha Contract aggiornato con parametri operativi validati.

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Input: l'Alpha Contract](#2-input-lalpha-contract)
3. [Step 1 — Setup: Parse e Feature Preparation](#3-step-1--setup-parse-e-feature-preparation)
4. [Step 2 — Backtest e Scoring](#4-step-2--backtest-e-scoring)
5. [Step 3 — Selezione e Raffinamento](#5-step-3--selezione-e-raffinamento)
6. [Step 4 — Validazione Statistica](#6-step-4--validazione-statistica)
7. [Step 5 — Analisi della Dipendenza dal Regime](#7-step-5--analisi-della-dipendenza-dal-regime)
8. [Output: Verdetto e Alpha Contract Aggiornato](#8-output-verdetto-e-alpha-contract-aggiornato)
9. [Riferimento di configurazione](#9-riferimento-di-configurazione)
10. [Checklist Completa](#10-checklist-completa)

---

## 1. Posizionamento e Responsabilità

```
┌──────────────────────────────────────────────────────────────────┐
│                        PIPELINE FORGE                            │
│                                                                  │
│  ┌────────────────────┐                                          │
│  │  EVENT DISCOVERY   │  Genera eventi booleani dal catalogo     │
│  └──────────┬─────────┘                                          │
│             │ Event Candidates                                   │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │  ALPHA DISCOVERY   │  Misura il potere predittivo             │
│  └──────────┬─────────┘                                          │
│             │ Alpha Contract                                     │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │  RULE DISCOVERY    │  ← questo modulo                        │
│  │  Input:  Alpha     │    Valida l'operatività con backtest     │
│  │          Contract  │    EDGE/PARTIAL-EDGE/NON-EDGE/           │
│  │                    │    INSUFFICIENT-DATA                     │
│  └──────────┬─────────┘                                          │
│             │ Regola validata + parametri                        │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │  RULE REGISTRY     │  Deduplica, cross-ticker, export        │
│  └────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Separazione netta dalle responsabilità upstream

| Modulo | Responsabilità | NON fa |
|---|---|---|
| Event Discovery | Genera eventi booleani | Non sa nulla del target |
| Alpha Discovery | Misura IC, WR, lift, regime | Non modifica soglie, non crea regole |
| **Rule Discovery** | **Valida l'operatività con backtest** | **Non genera eventi, non modifica soglie** |
| Rule Registry | Deduplica, cross-ticker, export | Non valuta qualità singola regola |

**Vincolo critico:** Rule Discovery riceve l'espressione dall'Alpha Contract
e la usa così com'è. Non introduce nuove feature, non ottimizza le soglie,
non esplora combinazioni alternative. L'esplorazione è già avvenuta
a monte — qui si valida e si parametrizza l'operatività.

---

## 2. Input: l'Alpha Contract

Rule Discovery legge il campo `event_expression` e i `rule_discovery_hints`
dell'Alpha Contract prodotto da Alpha Discovery.

```yaml
# Estratto dell'Alpha Contract ricevuto
event_expression:   "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"

target_definition:
  holding_period_h: 24
  sell_pct:         0.04
  base_rate:        0.235

rule_discovery_hints:
  entry_mode:         "limit"   # suggerimento upstream — vedi nota sotto
  buy_drop_pct_range: [0.006, 0.015]
  sell_pct_range:     [0.030, 0.050]
  target_h_range:     [12, 48]
  min_pf_target:      2.0
  min_wr_target:      0.70
  exclusion_conditions:
    - "Regime uptrend continuo"
    - "Volume < 0.3× SMA25"

# Statistiche disponibili (da usare come riferimento, non come input del backtest)
event_stats:
  n_activations:  329
  win_rate:       0.415        # win rate statistico (no fill, no fee)
  lift_vs_base:   +0.180
  cohens_d:       0.394
```

> **Nota:** il `win_rate` nell'Alpha Contract è calcolato su tutte le barre
> di attivazione, senza simulare alcun fill e senza dedurre le fee. Il
> backtest di Rule Discovery produrrà un win rate diverso.
>
> **Nota su `entry_mode` dell'hint:** questo campo è un suggerimento
> upstream, non il meccanismo di esecuzione effettivo. Il meccanismo
> realmente usato è governato da `RuleDiscoveryConfig.entry_mode`
> (default `"auto"` da #185) — vedi §4.0. Sotto `"auto"` Rule Discovery
> valuta comunque *entrambi* gli entry (mercato e limit), a prescindere
> da questo hint, e il win rate dello Stadio 1 (mercato, fill ≈ 100%) è
> quello che decide il verdetto.

---

## 3. Step 1 — Setup: Parse dell'Alpha Contract

Rule Discovery non carica dati, non calcola feature, non prepara dataset.
Riceve due input già pronti dalla sessione FORGE corrente:

- **Alpha Contract** — prodotto da Alpha Discovery
- **KPI Table estesa** — la stessa tabella usata da Event Discovery e
  Alpha Discovery, che contiene già tutte le feature derivate
  (pctrank, zscore, delta) calcolate da Event Discovery nel Transform Layer

L'unico lavoro di setup è leggere il contratto e definire il grid.

### 1.1 Parse dell'espressione

```python
# Legge l'espressione dall'Alpha Contract
expression = alpha_contract['event_expression']
# → "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"

# Verifica che tutte le feature richieste siano presenti nella KPI Table
# (devono esserlo — Event Discovery le ha già calcolate nella stessa sessione)
required = parse_required_features(expression)
assert all(f in kpi_table.columns for f in required), \
    f"Feature mancanti: {set(required) - set(kpi_table.columns)}"
```

### 1.2 Definizione del grid parametrico

I parametri operativi da esplorare vengono letti dagli hints del contratto:

```python
BUY_DROP_GRID = np.arange(
    alpha_contract['rule_discovery_hints']['buy_drop_pct_range'][0],
    alpha_contract['rule_discovery_hints']['buy_drop_pct_range'][1],
    step=0.002
)
SELL_PCT_GRID = np.arange(
    alpha_contract['rule_discovery_hints']['sell_pct_range'][0],
    alpha_contract['rule_discovery_hints']['sell_pct_range'][1],
    step=0.005
)

BASE_CONFIG = {
    'buy_type':         'limit',  # meccanica dell'entry a limit — vedi §4.0
    'buy_price_anchor': 'close',
    'buy_drop_pct':     0.010,
    'buy_delay_bar':    6,   # risolto: 6h di ordine vivo (1 su 1D) — #179
    'sell_pct':         0.040,
    'target_h':         24,  # risolto: cima della classe di orizzonti (10 su 1D) — #179
    'fee':              0.002,
    'early_stopping':   True,
    'pf_min_trades':    15,
    'pf_min_tpm':       2,   # risolto da criteria.min_tpm (#178)
}
```

> **Vincolo:** il grid esplora solo i parametri operativi — `buy_drop_pct`,
> `sell_pct`, `target_h`. L'espressione rimane invariata.
>
> **`buy_type` qui è la meccanica di un singolo backtest**, non l'intera
> strategia di valutazione. Con `entry_mode="auto"` (default, §4.0) Rule
> Discovery esegue *due* backtest — uno con `buy_type='market'` e uno con
> `buy_type='limit'` — e il `BASE_CONFIG` sopra descrive solo la seconda
> metà. Con `entry_mode="limit"` esplicito, invece, questo è l'intero
> flusso a singolo stadio.

### 1.3 Ricostruzione del segnale evento

Il backtest e il grid hanno bisogno della serie booleana di attivazione
dell'evento allineata alle candele osservate (la KPI Table ricevuta). Un Event
Candidate **è** una funzione di attivazione: Rule Discovery la *rivaluta* sulle
candele che osserva tramite `EventCandidate.apply(frame)` — un replay
deterministico dei parametri congelati (soglie, finestre, `diffnorm_std`,
etichette di classe), **senza** rifittare nulla. È ricostruzione della feature,
non ri-derivazione.

La `EventCandidate.event_series` pre-calcolata è una **cache** di quella stessa
funzione valutata sulle candele di training. Viene riusata come fast-path
trasparente **solo** quando il suo indice è identico a quello del frame
osservato — dove la cache coincide con `apply()` bit-for-bit:

```python
stored = candidate.event_series
if stored is not None and stored.index.equals(frame.index):
    signal = stored                  # fast-path: indice identico
else:
    signal = candidate.apply(frame)  # rivalutazione onesta + warning
```

> **Vincolo (no-recompute, stesso contratto di Alpha Discovery):** quando i set
> di candele differiscono — diversa unit di timestamp, uno slice, o una finestra
> OOS — la cache è inapplicabile. Un `reindex()` cieco mapperebbe ogni barra non
> sovrapposta a `NaN → inattivo`, nel caso peggiore facendo collassare il segnale
> a tutto-zero e backtestando una regola che non si attiva mai. Per questo
> l'evento viene rivalutato sul frame osservato; le trasformazioni a finestra
> riflettono la storia effettivamente disponibile in quella finestra.

---

## 4. Step 2 — Backtest e Scoring

### 4.0 Entry mode: valutazione "auto" a due stadi (default, #185)

`RuleDiscoveryConfig.entry_mode` governa **come** l'entry viene simulata, e
di default vale `"auto"` (era `"limit"` prima di #185). Tre valori:

| `entry_mode` | Comportamento |
|---|---|
| `"auto"` (default) | Due stadi — vedi sotto. Il verdetto viene dallo Stadio 1 (mercato); lo Stadio 2 (limit) è un raffinamento opzionale dell'entry. |
| `"market"` | Solo lo Stadio 1: entry al prossimo open, fill ≈ 100%. Nessun ottimizzatore di entry — isola l'edge del *segnale* dalla meccanica dell'ordine. |
| `"limit"` | Comportamento pre-#185, a singolo stadio: il grid esplora `buy_drop_pct` e l'entry a limit *è* insieme meccanica d'ordine e ottimizzatore di prezzo d'entrata. Resta pienamente supportato — è la scelta corretta quando il limit order stesso è la strategia, non solo un raffinamento esecutivo. |

**Perché il default è cambiato.** In modalità `"limit"` l'entry svolge un
doppio ruolo: meccanica d'ordine *e* ottimizzatore del prezzo d'entrata. Uno
sconto più profondo si riempie più raramente e — punto cruciale — solo sui
percorsi di prezzo che sono tornati a toccarlo: il profit factor sale su un
sottoinsieme di trade che non è la popolazione effettivamente tradeabile (il
"fill confound"). Il verdetto finisce per misurare il prezzo d'entrata
invece del segnale.

**Il flusso a due stadi di `"auto"`:**

1. **Stadio 1 — baseline di mercato (autoritativo).** Rule Discovery esegue
   l'intera pipeline (grid, walk-forward, validazione statistica, regime)
   con `buy_type="market"`: entry al prossimo open, fill ≈ 100%. Il verdetto
   di questo stadio è **autoritativo** — lo Stadio 2 non può mai trasformare
   un `NON-EDGE` in un edge, può solo scegliere quali parametri pubblicare
   per una regola già promossa.
2. **Stadio 2 — raffinamento opzionale del limit order.** Solo se lo Stadio 1
   ha promosso la regola (EDGE o PARTIAL-EDGE), Rule Discovery esplora
   `buy_drop_pct` in-sample (l'uscita resta quella vincente dello Stadio 1),
   individua il miglior candidato limit e lo **rigioca out-of-sample** sulle
   stesse finestre di test dello Stadio 1. Il punto operativo a limit viene
   adottato solo se supera **tutte e tre** le condizioni seguenti, misurate
   OOS:

   | # | Condizione | Soglia |
   |---|---|---|
   | 1 | `fill_rate >= min_fill_rate_opt` | 0.80 (default) — nessun PF gonfiato da fill rari |
   | 2 | `opportunity_sharpe(limit) >= opportunity_sharpe(mercato)` | uno Sharpe *per unità di tempo* (trade/mese), non quello annualizzato di `StatisticalValidation` — chi trada meno deve guadagnare di più per trade per vincere il confronto |
   | 3 | `net_gain(limit) >= min_net_gain_retention × net_gain(mercato)` | 0.5 (default) — backstop per il caso che lo Sharpe non vede: una media minuscola con varianza minuscola dà uno Sharpe eccellente pur producendo quasi nulla |

   Se una qualunque condizione fallisce, il punto operativo di mercato viene
   mantenuto; il verdetto non cambia in nessun caso.

Il risultato è un nuovo artefatto, `EntryOptimization`
(`RuleDiscoveryResponse.entry_optimization`), che riporta entrambi i punti
operativi con la relativa evidenza:

```python
eo = resp.entry_optimization
eo.selected_entry          # "market" o "limit" — il punto pubblicato
eo.authoritative           # sempre "market" — da dove viene il verdetto
eo.adopted                 # True se il limit ha superato tutte e tre le condizioni
eo.failed_condition        # "fill" | "sharpe" | "net_gain" | None (adottato)
eo.market_summary, eo.limit_summary            # BacktestSummary OOS
eo.market_opportunity_sharpe, eo.limit_opportunity_sharpe
eo.market_oos_net_gain, eo.limit_oos_net_gain
eo.reason                  # spiegazione leggibile della decisione
```

`entry_optimization` è `None` quando `entry_mode` non è `"auto"`, oppure
quando lo Stadio 1 è già `NON-EDGE` (lo Stadio 2 viene saltato).

### 2.1 Meccanica del fill

**Entry di mercato** (Stadio 1 di `"auto"`, o `entry_mode="market"`):

```
Barra t:    segnale attivato
            fill al prossimo open, fill_rate ≈ 100%

Barre fill+1..fill+target_h:  exit window
            Se high_k ≥ fill_price × (1 + sell_pct) → TARGET HIT
            Se k = fill+target_h → chiudi a close_k
```

**Entry a limit** (`entry_mode="limit"`, o lo Stadio 2 di `"auto"`):

```
Barra t:    segnale attivato
            buy_price = close_t × (1 - buy_drop_pct)

Barre t+1..t+6:  fill window
            Se low_k ≤ buy_price → fill a buy_price
            Se nessuna barra tocca buy_price → segnale scartato

Barre fill+1..fill+24:  exit window
            Se high_k ≥ buy_price × (1 + sell_pct) → TARGET HIT
            Se k = fill+24 → chiudi a close_k
```

**Fill rate atteso (limit):** con `buy_drop_pct = 1%` e `delay = 6`, il fill
rate è tipicamente 65–78%. Un fill rate < 50% suggerisce che lo sconto è
troppo profondo per il contesto di mercato. Sotto `entry_mode="auto"` questo
gate (`min_fill_rate`) è di fatto inerte sullo Stadio 1 di mercato (fill
≈ 100%) — il floor che conta davvero è `min_fill_rate_opt` dello Stadio 2.

### 2.2 Metriche di scoring

```python
summary = run_backtest(candle=kpi_table, rule=expression, **BASE_CONFIG)

# Metriche primarie
profit_factor        = summary['profit_factor']       # target: ≥ 2.0
win_rate_pct         = summary['win_rate_pct']        # target: ≥ 0.55
expectancy           = summary['expectancy']          # target: > 0

# Metriche di distribuzione temporale
_tpm_mu              = summary['_tpm_mu']             # trade/mese media
_tpm_C_norm          = summary['_tpm_C_norm']         # consistenza [0,1]
pf_score_tpm         = summary['pf_score_tpm']        # = PF × C_norm

# Diagnostica
fill_rate            = summary['fill_rate']           # % segnali riempiti
zero_months          = summary['zero_months']         # mesi senza trade

# Trade-overlap (issue #168) — quantificano la sovrapposizione di capitale
n_episodes                = summary['n_episodes']                 # episodi di attivazione (barre attive consecutive, gap-bridged)
mean_concurrent_positions = summary['mean_concurrent_positions']  # posizioni aperte in media, quando almeno una è aperta
max_concurrent_positions  = summary['max_concurrent_positions']   # picco — decide se la regola è deployabile con il capitale disponibile
```

> **Nota (#168) — `run_backtest` apre una posizione su ogni barra attiva,
> senza alcun controllo di stato "flat".** È una scelta deliberata e
> compatibile col capitale — le metriche economiche riportate sono
> riproducibili in produzione *a patto di avere il capitale per finanziare
> le posizioni concorrenti* — ma finché non esistevano questi tre campi non
> c'era modo di sapere quanto capitale servisse. Episodi e concorrenza sono
> misure diverse e generalmente in disaccordo: gli episodi raggruppano per
> segnale, la concorrenza per percorso di prezzo, e trade di episodi diversi
> si sovrappongono comunque ogni volta che il periodo di holding supera il
> gap fra un'attivazione e l'altra. Caso di riferimento: 120 barre di
> segnale, 76 episodi, 3.71 posizioni concorrenti in media.

### 2.3 Criteri di eliminazione rapida

`_early_elimination()` (`discovery.py`) verifica esattamente **tre**
condizioni — non c'è alcun controllo su `win_rate_pct` o su `_tpm_mu` in
questo screening rapido. Scartare immediatamente (`NON-EDGE`) se:

| Condizione | Significato |
|---|---|
| `total_trades < max(10, n_months × min_tpm)` | Floor **dinamico**, non un valore fisso: almeno 10 trade in assoluto, oppure `n_months × min_tpm` se maggiore. Cresce con la lunghezza della finestra IS invece di penalizzare finestre corte o essere troppo permissivo su quelle lunghe. |
| `profit_factor < 1.0` | Strategia perdente in-sample |
| `fill_rate < min_fill_rate` (0.40) | Buy_drop troppo profondo, non si riempie (inerte allo Stadio 1 di `entry_mode="auto"`, dove il fill è ≈100%) |

> **Nota (#217) — la prima finestra di train del walk-forward è un caso
> speciale.** Quando `selection_mode="walk_forward"` (default), lo screening
> rapido viene eseguito anche sulla *prima finestra di train* prima di
> lanciare il walk-forward completo — ma lì il floor non è
> `n_months × min_tpm`: è passato esplicitamente come `_MIN_TRADES_ABS`
> (fisso, 10 trade). Il motivo: quella finestra è dimensionata dal resolver
> apposta per raggiungere **esattamente** 10 trade a `min_tpm` (margine di
> Poisson al 95%), non `n_months × min_tpm` trade. Ri-derivare quella
> seconda formula su una finestra così corta pone una domanda più stringente
> e scollegata, e collassa su una soglia irraggiungibile quando `min_tpm` è
> abbastanza alto da far arrotondare la finestra al suo floor minimo di un
> mese (es. 1 mese × 35.2 tpm = 35 trade richiesti, quando la finestra era
> dimensionata per provarne solo 10). Al di fuori di questo caso speciale,
> il floor dinamico standard si applica ovunque — inclusa la stessa
> condizione ripetuta nel gate finale del verdetto (§5.4).

### 2.4 Screening in batch sul grid

```python
results = []
for buy_drop in BUY_DROP_GRID:
    for sell_pct in SELL_PCT_GRID:
        summary, perf, _ = run_backtest(
            candle=kpi_table, rule=expression,
            buy_drop_pct=buy_drop, sell_pct=sell_pct,
            **{k:v for k,v in BASE_CONFIG.items()
               if k not in ['buy_drop_pct','sell_pct']}
        )
        results.append({
            'buy_drop': buy_drop, 'sell_pct': sell_pct,
            'pf': summary['profit_factor'],
            'wr': summary['win_rate_pct'],
            'tpm': summary['_tpm_mu'],
            'sc': summary['pf_score_tpm'],
            'zero_m': summary['zero_months'],
        })

# Ordina per pf_score_tpm — bilancia PF e consistenza temporale
results_df = pd.DataFrame(results).sort_values('sc', ascending=False)
```

---

## 5. Step 3 — Selezione e Raffinamento

### 3.1 Criteri di selezione del miglior set di parametri

`_passes()` (`grid.py`) verifica queste condizioni — valori di default di
`SelectionCriteria` (preset `"balanced"`; vedi §9.1 per il range fra preset):

```
profit_factor    ≥ 2.0    # min_profit_factor
win_rate_pct     ≥ 0.55    # min_win_rate (preset: 0.50 sweep – 0.60 sniper, mai 0.70)
tpm_mu           ≥ 2.0    # min_tpm — sessione-risolto
pf_score_tpm     ≥ 0.30    # min_pf_score_tpm
fill_rate        ≥ 0.40    # min_fill_rate
```

> **Nessun gate assoluto sul numero di trade qui.** `_passes()` non
> controlla `total_trades`: la frequenza è imposta solo in modo relativo
> tramite `min_tpm` (trade/mese). Il floor assoluto dinamico
> (`max(10, n_months × min_tpm)`) vive nei gate del verdetto (§2.3, §5.4),
> non nella selezione del grid.

Il parametro selezionato è quello con `pf_score_tpm` massimo tra tutti
quelli che superano i criteri (se nessuno li supera, si prende comunque il
`pf_score_tpm` più alto in assoluto, per ispezione — verrà scartato a valle).
Non si usa il PF grezzo per evitare overfitting.

Una volta individuata la configurazione migliore, si esegue un ultimo backtest
completo per ottenere il DataFrame dei trade da usare nei passi successivi:

```python
# Configurazione vincente dal grid
BEST_CONFIG = results_df.iloc[0]  # riga con pf_score_tpm massimo

_, perf_trades_raw, _ = run_backtest(
    candle=kpi_table, rule=expression,
    buy_drop_pct=BEST_CONFIG['buy_drop'],
    sell_pct=BEST_CONFIG['sell_pct'],
    **{k:v for k,v in BASE_CONFIG.items()
       if k not in ['buy_drop_pct','sell_pct']}
)

# Solo i trade effettivamente riempiti
perf_trades = perf_trades_raw[perf_trades_raw['buy_signal_bool'] == 1].copy()
```

### 3.2 Analisi del breakdown mensile

```python
perf_trades['month'] = pd.to_datetime(perf_trades['fill_open_dt']).dt.to_period('M')

monthly = perf_trades.groupby('month').agg(
    n_trades = ('net_pct_gain', 'count'),
    win_rate = ('gain_bool', 'mean'),
    cum_gain = ('net_pct_gain', 'sum'),
)

bad_months = monthly[monthly['cum_gain'] < 0]
print(f"Mesi negativi: {len(bad_months)}/12")
print(f"Mesi zero trade: {(monthly['n_trades'] == 0).sum()}/12")
```

**Per RI_01:** distribuzione mensile `[9, 19, 6, 12, 1, 13, 1, 2, 3, 12, 5, 19]`
— zero mesi a zero trade.

### 3.3 Robustezza parametrica

Una regola robusta mantiene PF ≥ 2 anche variando leggermente i parametri:

```python
sensitivity = {}
for delta in [-0.002, -0.001, 0, +0.001, +0.002]:
    s, _, _ = run_backtest(
        candle=kpi_table, rule=expression,
        buy_drop_pct=BEST_DROP + delta, **OTHER_PARAMS
    )
    sensitivity[BEST_DROP + delta] = s['profit_factor']

# Regola fragile: PF varia da 0.8 a 3.5 su delta di ±0.2%
# Regola robusta: PF rimane tra 2.5 e 3.8 su tutto il range
```

---

## 6. Step 4 — Validazione Statistica

### 4.1 Significatività del win rate

```python
from scipy import stats

# H0: win_rate = base_rate (nessun edge)
wins  = perf_trades['gain_bool'].values
base  = alpha_contract['target_definition']['base_rate']

t_stat, p_value = stats.ttest_1samp(wins, popmean=base, alternative='greater')
# p < 0.05 → edge statisticamente significativo
```

> **Nota (#177) — campione nominale vs effettivo.** L'implementazione reale
> (`validation.py`) non usa il conteggio nominale di trade per l'errore
> standard/i gradi di libertà di questo t-test: usa `n_effective` — vedi
> il box a fine §4.3.

### 4.2 Significatività dell'expectancy

```python
# H0: expectancy = 0 (strategia neutrale)
gains = perf_trades['net_pct_gain'].values
t_stat, p_value = stats.ttest_1samp(gains, popmean=0, alternative='greater')
```

### 4.3 Deflated Sharpe Ratio

Il grid al Step 2 ha testato N configurazioni. Il PF in-sample del vincitore
è sovrastimato — il DSR corregge questo bias:

```python
def deflated_sharpe(sr_selected, n_trials, n_obs):
    """
    sr_selected: Sharpe Ratio annualizzato della configurazione selezionata
    n_trials:    N configurazioni testate nel grid
    n_obs:       numero di trade della configurazione selezionata
    """
    import math
    gamma = 0.5772156649
    correction = math.sqrt(1 - gamma * math.log(n_trials) / math.log(n_obs))
    return sr_selected * correction

# DSR > 1.0 → l'edge è reale dopo correzione per multiple testing
```

> **Per RI_01:** ~80 configurazioni testate, SR ≈ 2.1, 102 trade → DSR ≈ 1.3.

> **`n_obs` è `n_effective`, non il conteggio nominale di trade (issue
> #177).** `StatisticalValidation.n_effective = total_trades /
> mean_concurrent_positions` (§2.2 — campo di `BacktestSummary`). Poiché
> `run_backtest` apre una posizione su ogni barra attiva senza controllo
> flat, trade che si sovrappongono condividono lo stesso percorso di prezzo
> e non sono osservazioni indipendenti: `validate()` (`validation.py`)
> alimenta `n_effective`, non il conteggio nominale, sia nell'`n_obs` del
> DSR sia nell'errore standard/gradi di libertà dei t-test di §4.1/4.2. Le
> metriche economiche (profit factor, expectancy, net gain) restano
> **nominali** — sono realtà riproducibile compatibile col capitale, non
> qualcosa da "correggere". L'effetto può essere sostanziale: 118 trade
> nominali contro ≈32 effettivi sovrastimano la significatività di un
> fattore `sqrt(118/32) ≈ 1.93×`.

### 4.4 Stazionarietà temporale

```python
trades = perf_trades.sort_values('fill_open_dt')
midpoint = len(trades) // 2
first_half  = trades.iloc[:midpoint]
second_half = trades.iloc[midpoint:]

print(f"Prima metà:   PF={compute_pf(first_half):.2f}  WR={first_half['gain_bool'].mean():.1%}")
print(f"Seconda metà: PF={compute_pf(second_half):.2f}  WR={second_half['gain_bool'].mean():.1%}")

# Red flag: PF > 3 nella prima metà e < 1.5 nella seconda
```

---

## 7. Step 5 — Analisi della Dipendenza dal Regime

### 5.1 Leggere il regime dalla KPI Table

La colonna `regime` è già presente nella KPI Table — calcolata dal
**Market Context Module** all'inizio della sessione. Rule Discovery
la legge direttamente senza ricalcolarla.

```python
# Legge il regime dalla KPI Table — non ricalcola
regime_col = kpi_table[['open_dt', 'regime', 'regime_stable']]
```

### 5.2 Performance per regime

```python
trades_with_regime = perf_trades.merge(
    regime_col,
    left_on='fill_open_dt', right_on='open_dt'
)

regime_perf = trades_with_regime.groupby('regime').agg(
    n_trades = ('net_pct_gain', 'count'),
    win_rate = ('gain_bool', 'mean'),
    cum_gain = ('net_pct_gain', 'sum'),
    exp      = ('net_pct_gain', 'mean'),
)
```

### 5.3 Metrica di dipendenza

```python
def regime_dependency_score(monthly_trades):
    """
    0.0 = distribuzione uniforme — regime-independent
    1.0 = tutti i trade in un mese — massima dipendenza
    """
    from scipy.stats import entropy
    p = monthly_trades / monthly_trades.sum()
    h = entropy(p + 1e-10)
    h_max = np.log(len(monthly_trades))
    return 1 - (h / h_max)

monthly = perf_trades.groupby('month').size()
dep_score = regime_dependency_score(monthly)
# < 0.2 = bassa dipendenza
# > 0.5 = alta dipendenza → segnalare nel contratto
```

### 5.4 Decisione

La decisione reale (`_decide()` in `discovery.py`) non è la semplice tabella
a tre rami basata su `zero_months`/PF che una versione precedente di questo
documento descriveva. È una cascata a tre fasi: gate duri → requisiti per
l'EDGE pieno → degrado power-aware. Schema (non letterale, ma fedele alla
logica):

**Fase 1 — Gate duri (producono `NON-EDGE` se falliti):**

```
SE PF in-sample < partial_min_profit_factor (1.5):
    → NON-EDGE

SE total_trades < max(10, n_months × min_tpm):
    SE la finestra è strutturalmente troppo corta per raggiungere quel
       floor a min_tpm (n_months × min_tpm < floor stesso — #173):
        → non un NON-EDGE — accantonato come "underpowered", risolto in
          Fase 3 come INSUFFICIENT-DATA (la finestra è il problema, non
          la regola)
    ALTRIMENTI:
        → NON-EDGE (la regola ha davvero fatto troppo pochi trade)

SE expectancy non significativa (p ≥ max_ttest_p, t-test one-sided):
    → NON-EDGE

SE walk-forward OOS profit_factor < 1.0:
    → NON-EDGE
```

Se nessun gate duro scatta ma la finestra era underpowered, il verdetto è
già `INSUFFICIENT-DATA` a questo punto (i gate duri restanti sono comunque
tutti passati).

**Fase 2 — Requisiti per l'EDGE pieno** (ognuno che fallisce degrada a
`PARTIAL-EDGE` invece di bloccare):

```
PF in-sample        ≥ min_profit_factor (2.0)
win_rate_pct        ≥ min_win_rate (0.55)
active_month_rate   ≥ min_active_month_rate (0.80)
                     dove active_month_rate = (n_months - zero_months) / n_months
DSR                  ≥ min_dsr (1.0)
                     — un DSR *indefinito* (radicando dell'haircut negativo,
                       n_trials troppo alto per n_obs) blocca l'EDGE pieno
                       tanto quanto un DSR basso: "selezione troppo severa
                       per essere credibile"
temporal_stability  != "FAIL"
regime_dependency    ≤ max_regime_dependency (0.30)
wf.consistency       ≥ 0.5   (frazione di finestre OOS in profitto)
rotation_p           ≤ max_rotation_p (0.05)  — se il contratto porta
                     un'annotazione rotation-null (FastRotationNull /
                     RotationCalibrator); altrimenti inerte
```

`EDGE` solo se **tutti** questi passano; altrimenti `PARTIAL-EDGE` con
l'elenco di quali sono falliti.

**Fase 3 — Degrado power-aware (`criteria.power_gate`, default `True`):**

Anche un verdetto altrimenti positivo (`EDGE`/`PARTIAL-EDGE`) viene
degradato a `INSUFFICIENT-DATA` quando l'evidenza OOS **pooled** (su tutte
le finestre di test concatenate, mai per singola finestra) non può
confermarlo:

```
SE non esiste alcun walk-forward:
    → INSUFFICIENT-DATA

SE pooled OOS trades < min_oos_trades (10):
    → INSUFFICIENT-DATA

SE l'expectancy minima rilevabile (MDE) sul campione OOS pooled supera
   l'expectancy IS dichiarata:
    → INSUFFICIENT-DATA — l'OOS non potrebbe confermare un effetto di
      quella dimensione anche se fosse reale
```

Un `NON-EDGE` non viene **mai** salvato da questa fase — sotto-alimentato o
no, la conseguenza operativa (non tradare) è identica.

---

## 8. Output: Verdetto e Alpha Contract Aggiornato

Rule Discovery compila il campo `rule_discovery_response`
nell'Alpha Contract e restituisce il documento aggiornato.

```yaml
rule_discovery_response:
  date:    "2026-05-23"
  verdict: "EDGE"     # EDGE | PARTIAL-EDGE | NON-EDGE | INSUFFICIENT-DATA

  # Non-null per EDGE, PARTIAL-EDGE e INSUFFICIENT-DATA (null solo per NON-EDGE)
  validated_rule:
    expression:      "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
    # Espressione invariata rispetto all'Alpha Contract
    entry_mode:      "market"   # o "limit" — vedi entry_optimization sotto
    buy_drop_pct:    0.010
    sell_pct:        0.040
    target_h:        24
    fee:             0.002

  backtest_results:
    profit_factor:   3.17
    win_rate:        0.814
    total_trades:    102
    fill_rate:       0.71
    expectancy:      0.0155
    zero_months:     0
    tpm_mu:          8.5
    pf_score_tpm:    0.61
    n_episodes:               76      # #168
    mean_concurrent_positions: 3.71
    max_concurrent_positions:  7

  statistical_validation:
    ttest_winrate_p:   0.0012
    ttest_expectancy_p: 0.0008
    deflated_sharpe:   1.31
    n_trials_tested:   80
    n_effective:       32.4      # #177 — total_trades / mean_concurrent_positions
    temporal_stability: "PASS"   # prima vs seconda metà

  # Presente solo quando entry_mode="auto" (default) e lo Stadio 1 è EDGE/PARTIAL
  entry_optimization:
    selected_entry:    "market"    # "market" o "limit" — il punto pubblicato
    authoritative:     "market"    # da dove viene il verdetto (sempre "market")
    adopted:           false       # true se il limit ha superato tutte e tre le condizioni OOS
    failed_condition:  "sharpe"    # "fill" | "sharpe" | "net_gain" | null (se adopted)
    reason:            "limit @ buy_drop=0.012 did not hold its risk-adjusted return..."

  regime_analysis:
    dependency_score: 0.18
    zero_months:      0
    avoid_in:         ["STRONG_BULL"]

  # Solo se NON-EDGE, o le ragioni del degrado se INSUFFICIENT-DATA
  rejection_reasons: null

# Aggiorna lo status
status: "VALIDATED"   # o "REJECTED" o "PARTIAL" o "INSUFFICIENT_DATA"
```

### Verdetti

**EDGE** — regola operativa. PF ≥ 2.0, WR ≥ 0.55, DSR ≥ 1.0, active-month
rate ≥ 0.80, stabilità temporale, dipendenza dal regime ≤ 0.30, consistenza
OOS ≥ 0.5, rotation-null superato (se annotato). Pronta per Rule Registry.

**PARTIAL-EDGE** — supera i gate duri (§5.4 Fase 1) ma fallisce almeno uno
dei requisiti per l'EDGE pieno (Fase 2) — es. PF fra `partial_min_profit_factor`
(1.5) e `min_profit_factor` (2.0), win rate sotto soglia, dipendenza dal
regime eccessiva, o rotation-null non superato. Deploy possibile con vincoli
espliciti documentati in `rejection_reasons`.

**NON-EDGE** — fallisce un gate duro: PF in-sample sotto 1.5, floor di trade
non raggiunto (finestra abbastanza lunga da poterlo raggiungere), expectancy
non significativa, o PF OOS del walk-forward sotto 1.0. Il gap tra IC/WR di
Alpha Discovery e PF operativo è normale — Alpha Discovery misura una
relazione, Rule Discovery misura l'operabilità al netto di fee e fill.

**INSUFFICIENT-DATA** — un verdetto che *sarebbe* positivo (EDGE o
PARTIAL-EDGE) ma la cui evidenza OOS pooled non può statisticamente
sostenerlo: nessun walk-forward disponibile, meno di `min_oos_trades` (10)
trade di test pooled, oppure l'expectancy minima rilevabile sul campione OOS
supera l'expectancy IS dichiarata (`criteria.power_gate`, default `True`,
via `_power_assessment()`). Conserva il proprio `validated_rule` per una
futura ri-valutazione ma **non è tradeabile** (`is_edge` è `False`) e non
raggiunge mai la Rule Registry. Un `NON-EDGE` non viene mai promosso a
questo verdetto — la conseguenza operativa è identica in entrambi i casi.

---

## 9. Riferimento di configurazione

Le tabelle sotto elencano i campi delle classi di configurazione più
rilevanti (`rule_discovery/models.py`) con i loro default effettivi. Un
campo marcato *sessione-risolto* è `UNSET` finché `forge()` non costruisce
il `PipelineContext` della sessione — costruendo `RuleDiscovery` a mano
(fuori da `forge()`) quel campo assume il proprio fallback documentato,
tipicamente calibrato su base oraria.

### 9.1 `SelectionCriteria` — soglie di accettazione e verdetto

| Campo | Default | Note |
|---|---|---|
| `min_profit_factor` | 2.0 | preset-parametrizzato: 2.5 sniper, 1.8 sweep, 2.0 balanced/burst |
| `min_win_rate` | 0.55 | preset-parametrizzato: 0.60 sniper, 0.55 balanced/burst, 0.50 sweep — **mai 0.70** |
| `min_tpm` | 2.0 (sessione-risolto) | radice di una catena: dimensiona `walk_forward.min_train_months` e `scoring.pf_min_tpm` |
| `min_pf_score_tpm` | 0.30 | preset-parametrizzato: 0.40 sniper, 0.25 sweep, 0.30 balanced/burst |
| `min_fill_rate` | 0.40 | inerte sotto `entry_mode="auto"` (Stadio 1 è a mercato, fill ≈100%) |
| `min_fill_rate_opt` | 0.80 | il floor che conta davvero sotto `"auto"` — Stadio 2, limit optimizer |
| `min_net_gain_retention` | 0.5 (sessione-risolto) | terza condizione di adozione del limit point |
| `min_sell_pct` | 0.005 (sessione-risolto da `AlphaConfig.mfe_floor`) | floor operativo sul take-profit derivato |
| `partial_min_profit_factor` | 1.5 | soglia inferiore per PARTIAL-EDGE |
| `min_active_month_rate` | 0.80 | rimpiazza i vecchi `max_zero_months_edge`/`max_zero_months_partial` — sostituisce un conteggio assoluto di mesi vuoti con un tasso, quindi funziona sia su base oraria (tasso vicino a 1.0) sia giornaliera |
| `max_regime_dependency` | 0.30 | |
| `min_dsr` | 1.0 | un DSR indefinito blocca l'EDGE pieno tanto quanto un DSR basso |
| `max_ttest_p` | 0.05 (sessione-risolto da `PipelineContext.alpha`) | unico gate hard per-ipotesi — produce NON-EDGE |
| `max_rotation_p` | 0.05 (sessione-risolto) | contro il rotation-null di ricerca (`AlphaContract.rotation_p`); inerte se il contratto non porta l'annotazione |
| `power_gate` | `True` | abilita il degrado a INSUFFICIENT-DATA (§5.4 Fase 3) |
| `min_oos_trades` | 10 | soglia sul conteggio OOS **pooled**, mai per singola finestra |
| `early_elimination` | `True` | scarto anticipato (§2.3) prima del walk-forward |

> **Campi rimossi:** `max_zero_months_edge`/`max_zero_months_partial` non
> esistono più nella classe — vedi `min_active_month_rate` sopra come loro
> sostituto basato su tasso.

### 9.2 `RuleWalkForwardConfig` — validazione walk-forward OOS

| Campo | Default | Note |
|---|---|---|
| `n_splits` | 4 | finestre di test OOS |
| `train_span_months` | `None` | `None` → walk-forward ancorato (la finestra di train cresce); intero → rolling |
| `test_span_months` | `None` | `None` → lo span post-train diviso equamente in `n_splits` |
| `min_train_months` | sessione-risolto | dimensionato da `criteria.min_tpm` con margine di Poisson al 95% |
| `reoptimise` | `True` | re-esegue lo screening del grid su ogni finestra di train |
| `purge_bars` | `None` (auto) | **campo prima non documentato.** `None` dimensiona automaticamente il purge dal `target_h` più grande del grid risolto (più il ritardo di fill) — previene che una entry aperta a fine finestra di train venga valutata su prezzi della finestra di test adiacente; `0` disabilita il purging |
| `embargo_bars` | sessione-risolto da `AlphaConfig.embargo_bars` | **campo prima non documentato.** Quarantena extra all'inizio di ogni finestra di *test*, in barre — stessa policy dell'embargo di Alpha Discovery, applicata al confine di ogni fold |

### 9.3 `RuleDiscoveryConfig` — configurazione top-level

| Campo | Default | Note |
|---|---|---|
| `entry_mode` | `"auto"` | `"auto"` \| `"market"` \| `"limit"` — vedi §4.0 |
| `use_contract_target` | `True` | semina `sell_pct`/`target_h` dal target derivato del contratto |
| `selection_mode` | `"walk_forward"` | **campo prima non documentato.** `"walk_forward"` (default) — il punto operativo pubblicato viene solo dalle finestre di train del walk-forward (§3.4); IS summary, early-elimination, validazione statistica e ogni diagnostica sono calcolati sullo **span di selezione** `[inizio, fine ultima finestra di train)`, mai sulla finestra di test finale. `"full_sample"` — comportamento legacy: screening, selezione e metriche IS sull'intera tabella (il walk-forward gate-a comunque il verdetto, ma i parametri pubblicati sono stati esposti alle sue finestre di test). `"walk_forward"` ripiega su `"full_sample"` (con nota) quando lo span dati è troppo corto per un solo split |
| `wf_param_policy` | `"last"` | **campo prima non documentato.** Come `selection_mode="walk_forward"` sceglie il punto operativo pubblicato tra le selezioni di train per-split: `"last"` — il vincitore della finestra di train più recente; `"consensus"` — il set di parametri più frequente fra gli split, pareggi risolti verso il più recente |
| `n_trials_upstream` | 1 | **campo prima non documentato.** Moltiplicatore piegato nell'`n_trials` del Deflated Sharpe oltre al conteggio di celle del grid — per chi vuole includere nell'haircut analitico un fattore di ricerca upstream esplicito (es. il numero di contratti fratelli che ricevono un verdetto nella stessa sessione) |

---

## 10. Checklist Completa

### Setup

- [ ] Alpha Contract ricevuto con `event_expression` e `rule_discovery_hints`
- [ ] KPI Table estesa verificata — tutte le feature dell'espressione presenti
- [ ] Grid parametrico definito dagli hints del contratto

### Backtest

- [ ] `entry_mode` verificato — `"auto"` (default, due stadi) salvo motivo
      esplicito per forzare `"market"` o `"limit"`
- [ ] Fee incluse (tipicamente 0.2% per lato)
- [ ] `early_stopping=True` per simulazione realistica
- [ ] Fill rate documentato (Stadio 1 di mercato ≈ 100%; Stadio 2 limit
      contro `min_fill_rate_opt`)
- [ ] `entry_optimization` letto quando `entry_mode="auto"` — `adopted`,
      `failed_condition`, punti mercato/limit a confronto

### Selezione

- [ ] `profit_factor ≥ 2.0`
- [ ] `win_rate ≥ 0.55`
- [ ] `tpm_mu ≥ 2.0` (nessun gate assoluto sul conteggio trade in `_passes()`)
- [ ] `pf_score_tpm ≥ 0.30`
- [ ] `fill_rate ≥ 0.40`
- [ ] Breakdown mensile analizzato
- [ ] Robustezza parametrica verificata (±2 step su buy_drop e sell_pct)

### Validazione statistica

- [ ] t-test su win rate vs base rate (p < `max_ttest_p`, default 0.05)
- [ ] t-test su expectancy vs 0 (p < `max_ttest_p`) — gate duro per NON-EDGE
- [ ] Deflated Sharpe Ratio ≥ `min_dsr` (1.0), e non indefinito
- [ ] Stabilità temporale verificata (prima vs seconda metà)
- [ ] `n_effective` (#177) letto invece del conteggio nominale quando si
      valuta la forza statistica dell'evidenza
- [ ] `n_episodes` / `mean_concurrent_positions` / `max_concurrent_positions`
      (#168) controllati per capire il capitale realmente richiesto
- [ ] Verdetto `INSUFFICIENT-DATA` distinto da `NON-EDGE` — non tradeabile
      ma potenzialmente rivalutabile con più dati

### Regime

- [ ] Zero months documentati
- [ ] Regime dependency score calcolato
- [ ] Performance per regime analizzata
- [ ] Vincoli di regime documentati nell'output se PARTIAL-EDGE

### Verdetto

- [ ] Verdetto letto come uno tra `EDGE` / `PARTIAL-EDGE` / `NON-EDGE` /
      `INSUFFICIENT-DATA` — non un enum a tre valori
- [ ] `entry_optimization` ispezionato quando `entry_mode="auto"` per capire
      se e perché il punto a limit è stato adottato

---

*Rule Discovery Pipeline — FORGE (Feature-Oriented Rule Generation Engine)*
*Versione 2.0 · Maggio 2026 · Parte di FORGE v1.0*
*Status: Draft · ⚠️ Documento tecnico di ricerca. Non costituisce consulenza finanziaria.*
