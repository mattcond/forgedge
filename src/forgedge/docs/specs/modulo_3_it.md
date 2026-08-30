# Modulo 3 — Rule Discovery

Rule Discovery è il quarto modulo della pipeline FORGE. Riceve un `AlphaContract`
promosso da Alpha Discovery e risponde alla domanda operativa che il contratto
lascia aperta: **il pattern statisticamente identificato sopravvive a un backtest
realistico — con fee, fill rate finito, ordini limite e target discreto — e tiene
fuori campione?**

L'output è un `RuleDiscoveryResponse` con verdetto `EDGE` / `PARTIAL-EDGE` /
`NON-EDGE` / `INSUFFICIENT-DATA` e, nei primi tre casi, una `ValidatedRule` con
i parametri operativi validati (`INSUFFICIENT-DATA` mantiene la sua
`ValidatedRule` per una futura ri-valutazione ma non è negoziabile — vedi
*Verdetto* più sotto).

---

## Utilizzo di base

```python
from forgedge import (
    MarketContext, EventDiscovery, AlphaDiscovery, AlphaConfig,
    RuleDiscovery, RuleDiscoveryConfig,
)

enriched   = MarketContext(kpi).run()
ed         = EventDiscovery(enriched)
candidates = ed.run()

ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="BTC", timeframe="1H"))
contracts  = ad.run()
promoted   = ad.promoted_contracts()

# Index dei candidati per ID (richiesto da RuleDiscovery)
by_id = {c.event_id: c for c in candidates}

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        vr = resp.validated_rule
        print(f"  entry: {vr.params.buy_type}  drop={vr.params.buy_drop_pct}"
              f"  sell={vr.params.sell_pct}  h={vr.params.target_h}")
```

---

## Posizione nella pipeline

```
list[AlphaContract] promossi (Modulo 2)
EventCandidate corrispondenti (Modulo 1)
        │
        ▼
  RuleDiscovery.run()  [per ogni contratto]
        │
        ▼
  RuleDiscoveryResponse
  ├─ verdict: EDGE | PARTIAL-EDGE | NON-EDGE | INSUFFICIENT-DATA
  ├─ validated_rule   (None solo per NON-EDGE)
  ├─ in_sample_summary
  ├─ execution_envelope + excursion (MAE/MFE)
  ├─ walk_forward OOS
  ├─ statistical_validation
  ├─ regime_analysis
  └─ entry_optimization   (solo entry_mode="auto")
        │
        ▼
  Rule Registry (non implementato)
```

Rule Discovery è **l'unico modulo che usa i prezzi per una simulazione di
esecuzione**. Non ri-ottimizza le soglie degli eventi né modifica i parametri
del target derivato: usa l'espressione dell'evento e `derived_target.sell_pct` /
`derived_target.holding_period_h` come punto di partenza per la grid operativa.

---

## Entry mode: valutazione a due stadi (default `entry_mode="auto"`)

Dal #185, `RuleDiscoveryConfig.entry_mode` ha default `"auto"`, non più
`"limit"`. Questo cambia cosa *misura* un verdetto, non solo come viene
eseguito l'entry:

- **Stage 1 (market, autoritativo).** La regola viene backtestata con un
  entry a mercato — fill all'apertura della barra successiva, fill rate
  ≈100%. Questo verdetto, e ogni diagnostica walk-forward / statistica /
  di regime che lo accompagna, è **autoritativo**: lo Stage 2 non può mai
  trasformare un `NON-EDGE` dello Stage 1 in un edge, può solo scegliere
  quale meccanismo di entry pubblicare.
- **Stage 2 (raffinamento a limite, solo su un edge).** Se lo Stage 1 supera
  `EDGE`/`PARTIAL-EDGE`, uno sweep su `buy_drop_pct` cerca un raffinamento a
  ordine limite del *solo prezzo di entry* — la meccanica di uscita, il
  target e il verdetto restano invariati. Il punto limite vincente viene
  ripetuto out-of-sample sulle stesse finestre di test dello Stage 1 e
  adottato solo se soddisfa tutte e tre le condizioni seguenti, ciascuna
  valutata **out-of-sample**:
  1. `fill_rate >= criteria.min_fill_rate_opt` (default `0.80`) — nessun PF
     gonfiato da fill rari.
  2. `opportunity_sharpe >= quello del market` — uno Sharpe scalato sulla
     *frequenza di trading*, non `StatisticalValidation.sharpe_ratio`: un
     punto che opera meno spesso deve guadagnare di più per trade per
     vincere questo confronto.
  3. `net_gain >= min_net_gain_retention × quello del market` (default
     `0.5`) — un backstop contro il caso che lo Sharpe non può vedere, una
     media piccola con una varianza piccola.

  Questo esiste perché il vecchio default `"limit"`-only permetteva a un
  limite profondo, che si riempie raramente, di gonfiare il profit factor su
  un sottoinsieme di trade non rappresentativo (il "fill confound") — il
  verdetto misurava il prezzo di entry anziché il segnale. `"auto"` mantiene
  entrambe le letture e le separa.

Entrambi i punti operativi, e quale condizione ha bloccato l'adozione, sono
riportati su `resp.entry_optimization` (un `EntryOptimization`):

```python
opt = resp.entry_optimization
opt.selected_entry     # "market" o "limit" — cosa è stato effettivamente pubblicato
opt.authoritative       # sempre "market" — da dove viene il verdetto
opt.adopted             # True se il raffinamento a limite è stato adottato
opt.failed_condition    # "fill" | "sharpe" | "net_gain" | None (adottato, o nessun candidato)
opt.market_opportunity_sharpe, opt.limit_opportunity_sharpe
opt.market_summary, opt.limit_summary   # BacktestSummary, out-of-sample
```

`entry_mode="market"` esegue solo lo Stage 1 — isola l'edge del solo
segnale; `min_fill_rate` è allora di fatto inerte. `entry_mode="limit"` è il
comportamento pre-#185: la grid varia `buy_drop_pct` direttamente e l'entry
a limite funge anche da ottimizzatore del prezzo di entry. È ancora
pienamente supportato ed è la scelta corretta quando l'ordine limite è
davvero **la strategia**, e non un raffinamento di esecuzione sopra un
segnale già valido a mercato.

---

## Pipeline a 5 step

### Step 1 — Setup

**Ricostruzione del segnale.** L'evento booleano viene ricostruito deterministic-
amente sul DataFrame tramite `EventCandidate.apply()` (o dalla `event_series`
memorizzata) e iniettato come colonna `__rule_signal__`.

**Seed dai parametri del contratto.** Con `use_contract_target=True` (default),
i parametri di backtest di partenza vengono inizializzati da:
- `target_h = derived_target.holding_period_h`
- `sell_pct = max(0.01, derived_target.sell_pct)` (clampato a un floor operativo)

Questi diventano il centro della grid di screening, non parametri fissi.

**Propagazione della direzione.** Il motore supporta `"long"` e `"short"`.
La direzione viene letta da `derived_target.direction` e impostata su
`BacktestParams.direction`. Il meccanismo short è il perfetto specchio del long:
il limite di acquisto si trova *sopra* l'anchor (`anchor × (1 + buy_drop_pct)`),
si riempie quando `high` lo raggiunge, e il take-profit è *sotto* il prezzo di
fill (`fill × (1 − sell_pct)`), raggiunto quando il prezzo scende fino a esso.
Il guadagno netto è `(fill − exit) / fill`. Solo direzioni diverse da `"long"`
o `"short"` generano `NON-EDGE` immediato.

---

### Step 2 — Grid screening in-sample

La grid operativa esplora il prodotto cartesiano di:

| Dimensione | Controllata da |
|---|---|
| `buy_drop_pct` | `GridSpec.buy_drop_pct` |
| `sell_pct` | `GridSpec.sell_pct` |
| `target_h` | `GridSpec.target_h` |
| `buy_delay_bar` | `GridSpec.buy_delay_bar` |

Quando `GridSpec` è vuoto (default), FORGE costruisce automaticamente una grid
sensata centrata sui valori derivati dal contratto.

> Quanto segue descrive la meccanica **a ordine limite**, usata direttamente
> sotto `entry_mode="limit"` e dallo Stage 2 del default `entry_mode="auto"`
> (vedi *Entry mode* sopra). Sotto lo Stage 1 del default `"auto"`, e sotto
> `entry_mode="market"`, l'entry è invece un fill a mercato all'apertura
> della barra successiva (fill rate ≈100%); `buy_drop_pct`/`buy_delay_bar`
> sono inerti in quello stadio.

**Meccanica di esecuzione per ogni configurazione:**

1. Al segnale, viene piazzato un ordine limite a `anchor * (1 - buy_drop_pct)`
2. Se il prezzo tocca il limite entro `buy_delay_bar` barre, l'ordine viene
   eseguito (fill). Altrimenti è annullato.
3. Dopo il fill, la posizione viene chiusa:
   - al primo bar che chiude a ≥ `sell_price = fill_price * (1 + sell_pct)`, oppure
   - al close della barra `target_h` (stop a orizzonte)

> **`target_h` conta le barre *dopo* il fill, non l'intero intervallo
> segnale→uscita.** Lo scarto segnale→fill è sempre di 1 barra (correttezza
> point-in-time — non si può agire su un close prima che sia avvenuto), quindi
> l'intervallo totale segnale→uscita è `1 + target_h`. "Tieni la posizione per
> N barre dall'entrata" corrisponde quindi a `target_h = N - 1`. `target_h = 0`
> è un valore legale e significativo — esce al close della stessa barra di
> fill (round-trip nella stessa sessione) — non un placeholder per "nessun
> orizzonte".

La configurazione viene valutata tramite il composite score `pf_score_tpm` che
bilancia Profit Factor, frequenza di trading e consistenza mensile.

**Early elimination (Step 2.3):** le configurazioni vengono scartate prima
della validazione costosa (walk-forward + diagnostiche) quando vale una
delle seguenti:

- trade totali sotto una **soglia dinamica**, `max(10, n_months × min_tpm)`
  — non un conteggio fisso; scala con l'estensione dell'IS e con
  `SelectionCriteria.min_tpm` (spec RD-04), così un periodo IS breve non
  viene penalizzato e uno lungo non viene sotto-richiesto
- PF < 1.0
- `fill_rate < min_fill_rate`

Con `SelectionCriteria(early_elimination=False)` la pipeline gira interamente
anche per queste configurazioni: il verdetto rimane `NON-EDGE`, ma
walk-forward e diagnostiche sono popolati — utile per report uniformi o per
analizzare il comportamento OOS di regole deboli.

> **Fix issue #217 — la prima finestra di train del walk-forward usa invece
> una soglia fissa.** Con `selection_mode="walk_forward"` (default), un
> pre-screen di early elimination gira anche sulla sola prima finestra di
> train del walk-forward, prima del walk-forward vero e proprio. La
> lunghezza di quella finestra (`min_train_months`) è già dimensionata dal
> resolver, con un margine di Poisson al 95%, per raggiungere esattamente la
> soglia assoluta (10 trade) a `criteria.min_tpm` — ri-derivare
> `n_months × min_tpm` su quella stessa finestra breve pone una domanda
> *più severa* e non correlata, e collassa a una soglia irraggiungibile non
> appena `min_tpm` è abbastanza alto da far arrotondare la finestra al suo
> minimo di 1 mese (es. 1 mese × 35.2 tpm = 35 trade richiesti, quando la
> finestra era dimensionata solo per dimostrarne 10). Questo eliminava
> erroneamente contratti legittimi con `min_tpm` alto. La correzione: quel
> pre-screen specifico usa la soglia assoluta fissa (`10`) anziché quella
> dinamica — cambia solo lo screen sulla prima finestra di train; ogni altro
> uso della soglia dinamica (la tabella sopra, lo screen sull'intero
> intervallo) non è interessato.

---

### Step 3 — Selezione e raffinamento

La configurazione migliore è quella con `pf_score_tpm` massimo tra quelle che
superano le soglie di `SelectionCriteria`. Se nessuna configurazione è
selezionabile, il verdetto è `NON-EDGE`.

**Dove avviene la selezione (`RuleDiscoveryConfig.selection_mode`).** Per
default (`selection_mode="walk_forward"`), questo screening — e ogni
metrica "IS" citata altrove in questo documento (`in_sample_summary`, lo
screen di early elimination dello Step 2.3, la validazione statistica dello
Step 4, il breakdown per regime dello Step 5) — viene calcolata solo sulla
**selection span**, `[inizio, fine dell'ultima finestra di train)`: le
finestre di train del walk-forward. Nessuna metrica che alimenta il
verdetto o la `ValidatedRule` pubblicata legge mai l'ultima finestra di test
del walk-forward. Il punto operativo pubblicato viene scelto tra i vincitori
per-split di train secondo `wf_param_policy`: `"last"` (default) — il
vincitore della finestra di train più recente, cioè cosa negozieresti al
prossimo giro; `"consensus"` — il set di parametri più frequente tra gli
split, a parità si preferisce il più recente. `selection_mode="full_sample"`
torna al comportamento legacy — grid screening e ogni metrica IS calcolati
sull'intera tabella, sebbene il walk-forward continui a determinare il
verdetto, quindi i parametri pubblicati sono stati esposti alle sue stesse
finestre di test. `"walk_forward"` ricade su `"full_sample"` (con una nota
sulla response) quando l'estensione dei dati è troppo breve anche per un
solo split di walk-forward.

---

### Step 4 — Validazione statistica

Sulla configurazione selezionata, sull'IS:

| Metrica | Descrizione |
|---|---|
| `ttest_winrate_t/p` | t-test win rate vs base rate del contratto |
| `ttest_expectancy_t/p` | t-test expectancy vs zero |
| `sharpe_ratio` | Sharpe annualizzato |
| `deflated_sharpe` | Sharpe deflato per n_trials (penalizza data snooping) |
| `n_effective` | Sample size effettivo, `total_trades / mean_concurrent_positions` (#168/#177) — uguale al conteggio dei trade quando non c'è overlap. I trade che si sovrappongono condividono lo stesso percorso di prezzo e non sono osservazioni indipendenti, quindi è questo — non il conteggio nominale dei trade — a essere consumato dall'errore standard/gradi di libertà dietro `ttest_*_p` e dall'`n_obs` dietro `deflated_sharpe`; le stime puntuali (media, dispersione) usano comunque ogni trade. `nan` quando non è misurabile, nel qual caso viene usato il conteggio nominale |
| `temporal_stability` | `"PASS"` / `"WARN"` / `"FAIL"`: PF prima metà vs seconda |
| `n_trials_tested` | Numero di configurazioni testate nella grid |

#### Walk-forward OOS

Il timeline viene diviso in `n_splits` finestre di test consecutive. Per ogni
split:
1. La grid viene ri-screenata sul train window precedente (con `reoptimise=True`)
2. I parametri migliori vengono applicati una sola volta sul test window
3. I trade delle finestre di test vengono concatenati → track record OOS onesto

`WalkForwardResult.oos_summary` contiene le metriche aggregate sui trade OOS.
`WalkForwardResult.consistency` = quota di finestre con net gain positivo.

---

### Step 5 — Regime dependency

Per ogni regime presente nel DataFrame, vengono calcolate le metriche di
performance sui trade IS: PF, win rate, expectancy, net gain cumulato.

`dependency_score`: entropia normalizzata della distribuzione mensile dei trade
(0 = trade distribuiti uniformemente; 1 = tutti i trade concentrati in un mese).

`avoid_in`: regimi con ≥ 5 trade e PF < 1.0 — regimi da evitare in produzione.

---

## Verdetto: EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA

### Gate `NON-EDGE` (hard — esclusione immediata)

| Condizione | Parametro |
|---|---|
| PF IS < `partial_min_profit_factor` (1.5) | — |
| Trade IS < `max(10, n_months × min_tpm)` (soglia dinamica) | — |
| t-test expectancy p ≥ `max_ttest_p` (0.05) | — |
| PF OOS walk-forward < 1.0 | — |

Se uno qualsiasi è violato: `NON-EDGE`.

### Gate `EDGE` (tutti richiesti per verdetto pieno)

| Condizione | Parametro |
|---|---|
| PF IS ≥ `min_profit_factor` (2.0) | — |
| Win rate IS ≥ `min_win_rate` (0.55) | — |
| `active_months / n_months` ≥ `min_active_month_rate` (0.80) | — |
| DSR ≥ `min_dsr` (1.0) | — |
| `temporal_stability` ≠ `"FAIL"` | — |
| `dependency_score` ≤ `max_regime_dependency` (0.30) | — |
| OOS consistency ≥ 0.50 | — |
| `AlphaContract.rotation_p` ≤ `max_rotation_p` (0.05), se annotato | — |

Se tutti soddisfatti: `EDGE`. Se solo i gate NON-EDGE sono soddisfatti ma non
tutti i gate EDGE: `PARTIAL-EDGE`.

Il tasso `active_months`/`n_months` sostituisce un tetto fisso su
`zero_months` (campo rimosso): un tasso è timeframe-agnostico, un conteggio
fisso di mesi vuoti tollerati no — su dati 1H (tpm ≫ 1) il tasso attivo sta
vicino a 1.0 senza alcuna calibrazione speciale, mentre su dati 1D
(tpm ~ 1.5–4) un processo di Poisson con dispersione fino a `max_dispersion`
produce naturalmente un tasso attivo nell'intervallo 0.75–0.95, che il
default `0.80` accomoda correttamente.

### `INSUFFICIENT-DATA` (downgrade power-gated, §3.2)

Quando `SelectionCriteria.power_gate` è `True` (default), un verdetto che
sarebbe altrimenti `EDGE`/`PARTIAL-EDGE` viene degradato a
`INSUFFICIENT-DATA` quando l'evidenza OOS **pooled** del walk-forward non può
sostenerlo: nessun walk-forward è stato possibile, il conteggio di trade OOS
pooled su tutte le finestre di test è sotto `min_oos_trades` (default `10`),
oppure l'effetto minimo rilevabile del campione OOS pooled supera
l'expectancy in-sample dichiarata (l'OOS non potrebbe confermare un effetto
di quella dimensione anche se fosse reale). La valutazione legge solo il
ledger concatenato delle finestre di test — mai i conteggi per singola
finestra, dato che le finestre di walk-forward sono corte per design e non
vengono gate-ate individualmente. I verdetti `NON-EDGE` non vengono mai
salvati da questo meccanismo — è un verdetto che sarebbe stato
*positivo* a essere degradato, mai uno negativo, dato che la conseguenza
operativa ("non tradare") è comunque la stessa. `INSUFFICIENT-DATA` mantiene
la sua `validated_rule` per una futura ri-valutazione quando ci saranno più
dati, ma `resp.is_edge` è `False` e non raggiunge mai il Rule Registry.

```python
resp.verdict         # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE" | "INSUFFICIENT-DATA"
resp.is_edge         # True se EDGE o PARTIAL-EDGE (INSUFFICIENT-DATA non è negoziabile)
resp.rejection_reasons  # gate falliti (lista vuota se EDGE)
```

---

## Execution Envelope e MAE/MFE

### `ExecutionEnvelope`

La stessa configurazione viene backtestata con due convenzioni di exit diverse,
definendo un intervallo di esecuzione realistica:

| Variante | `target_hit_col` | Descrizione |
|---|---|---|
| `conservative` | `"close"` | Il target conta solo quando una barra **chiude** oltre sell_price. Sottostima (il limit sell reale potrebbe riempirsi intrabar). Corrisponde al motore di riferimento certificato, sia per long che per short. |
| `optimistic` | `"high"` (long) / `"low"` (short) | Il target conta al primo tocco intrabar. Sovrastima (assume che il limit sell si riempia sempre). |

La performance reale si trova tra i due. Per risolvere la colonna ottimistica
in base alla direzione, usare l'utility `optimistic_hit_col`:

```python
from forgedge.rule_discovery import optimistic_hit_col

col = optimistic_hit_col("short")  # → "low"
col = optimistic_hit_col("long")   # → "high"

env = resp.execution_envelope
print(f"conservative PF: {env.conservative.profit_factor:.2f}")
print(f"optimistic   PF: {env.optimistic.profit_factor:.2f}")
```

### `ExcursionStats` (MAE/MFE)

Per ogni trade eseguito, sulla finestra `[fill+1 .. exit]`:
- **MAE** (Maximum Adverse Excursion): drawdown massimo raggiunto, `(min_low - buy_price) / buy_price` (negativo)
- **MFE** (Maximum Favourable Excursion): run-up massimo raggiunto, `(max_high - buy_price) / buy_price` (positivo)

```python
ex = resp.excursion
print(f"MAE medio: {ex.mae_mean:.4f}, peggiore: {ex.mae_worst:.4f}")
print(f"MFE medio: {ex.mfe_mean:.4f}, ha raggiunto target: {ex.mfe_reached_target_pct:.1f}%")
```

---

## Struttura dati: `RuleDiscoveryResponse`

```python
resp = rd.run()

resp.verdict             # str: "EDGE" | "PARTIAL-EDGE" | "NON-EDGE" | "INSUFFICIENT-DATA"
resp.is_edge             # bool: True se EDGE o PARTIAL-EDGE
resp.alpha_id            # str: ID del contratto sorgente
resp.asset, resp.timeframe  # str

# Regola validata (None solo per NON-EDGE)
resp.validated_rule.expression       # espressione booleana
resp.validated_rule.params           # BacktestParams con la configurazione ottimale
resp.validated_rule.event_candidate_id

# Metriche IS
resp.in_sample_summary.total_trades
resp.in_sample_summary.profit_factor
resp.in_sample_summary.win_rate_pct
resp.in_sample_summary.expectancy
resp.in_sample_summary.tpm_mu        # trade/mese medi
resp.in_sample_summary.zero_months   # mesi senza trade

# Execution envelope + MAE/MFE
resp.execution_envelope   # ExecutionEnvelope | None
resp.excursion            # ExcursionStats | None

# Walk-forward OOS
resp.walk_forward.oos_summary.profit_factor
resp.walk_forward.consistency          # quota finestre con net gain > 0
resp.walk_forward.n_profitable_splits
resp.walk_forward.oos_envelope         # ExecutionEnvelope OOS
resp.walk_forward.oos_excursion        # ExcursionStats OOS

# Validazione statistica
resp.statistical_validation.deflated_sharpe
resp.statistical_validation.ttest_expectancy_p
resp.statistical_validation.n_effective         # sample size effettivo (#168/#177)
resp.statistical_validation.temporal_stability  # "PASS"/"WARN"/"FAIL"

# Regime
resp.regime_analysis.dependency_score
resp.regime_analysis.avoid_in          # list[str] regimi da evitare
resp.regime_analysis.per_regime        # list[dict]

# Valutazione entry-mode (solo entry_mode="auto" — vedi "Entry mode" sopra)
resp.entry_optimization.selected_entry     # "market" | "limit" | None
resp.entry_optimization.adopted            # bool
resp.entry_optimization.failed_condition   # "fill" | "sharpe" | "net_gain" | None

# Audit
resp.rejection_reasons   # list[str]
resp.notes               # list[str]
resp.grid_results        # list[GridResult]
```

### `BacktestSummary` — campi completi

`run_backtest` apre una posizione su ogni barra attiva, senza controllo dello
stato flat — una policy deliberata e capital-permitting (l'economia è
riproducibile in produzione dato abbastanza capitale per finanziare le
posizioni concorrenti), ma fino all'issue #168 non c'era un modo supportato
per sapere quanto capitale serve. I tre campi di concorrenza sotto
rispondono a questo; il trade ledger (`return_trades=True`) porta anche un
`episode_id` per riga quando servono a livello di singolo trade.

| Campo | Descrizione |
|---|---|
| `total_signals` | Segnali totali nel dataset |
| `total_trades` | Trade eseguiti (signal riempito) |
| `fill_rate` | `total_trades / total_signals` |
| `win_rate_pct` | Win rate (0–1) |
| `winning_trades`, `losing_trades` | Conteggi |
| `total_net_gain` | Somma dei rendimenti netti (fee incluse) |
| `expectancy` | `total_net_gain / total_trades` |
| `std_net_gain` | Dev. standard rendimenti netti |
| `profit_factor` | `sum(vincite) / sum(perdite)` |
| `best_trade`, `worst_trade` | Rendimento migliore/peggiore |
| `target_hit_rate_pct` | % trade che hanno raggiunto il target |
| `n_months` | Mesi totali nel dataset |
| `active_months` | Mesi con almeno un trade |
| `zero_months` | `n_months - active_months` |
| `tpm_mu`, `tpm_sigma` | Media e dev. std. trade/mese |
| `c_norm` | Regolarità mensile: `min(1, 1 / max(indice_di_dispersione, 1))`. Scale-free — un processo di Poisson prende 1 a *qualunque* tasso, e solo la varianza in eccesso rispetto al Poisson viene penalizzata (#178). |
| `n_episodes` | *Episodi* di attivazione dietro i trade — barre attive consecutive (con gap collegati) contate come un unico evento. Risponde a "quanto spesso spara questo segnale" (#168) |
| `mean_concurrent_positions` | Posizioni aperte medie sulle barre in cui almeno una è aperta. Risponde a "quando questa regola funziona, quante posizioni sto finanziando" — `total_trades / mean_concurrent_positions` è la dimensione campionaria che l'overlap effettivamente supporta (#168) |
| `max_concurrent_positions` | Picco di posizioni concorrenti aperte — decide se la regola è dispiegabile su un dato conto (#168) |
| `pf_score`, `pf_score_tpm` | Score composito: `pf × sigmoid(numero di trade)` e `pf × c_norm` |
| `exp_score_tpm` | Stessa penalità di regolarità applicata all'expectancy: `expectancy × c_norm` |
| `sharpe_raw` | Sharpe grezzo (non annualizzato) |

---

## Metodi di output

### `rd.run() → RuleDiscoveryResponse`

Esegue l'intera pipeline. Deve essere chiamato prima di qualsiasi altro metodo.

### `rd.grid_summary() → pd.DataFrame`

DataFrame piatto di tutte le configurazioni testate nella grid IS, ordinabile per
`pf_score_tpm`:

```python
grid_df = rd.grid_summary()
print(grid_df.sort_values("pf_score_tpm", ascending=False).head())
# Colonne: buy_drop_pct, sell_pct, target_h, buy_delay_bar,
#          profit_factor, win_rate_pct, total_trades, expectancy,
#          tpm_mu, fill_rate, zero_months, pf_score_tpm
```

### `text_report(resp) → str` e `html_report(resp) → str`

```python
from forgedge.rule_discovery import text_report, html_report

# Report testuale compatto
print(text_report(resp))

# Report HTML self-contained (no CDN, funziona offline)
with open(f"{resp.alpha_id}.html", "w") as f:
    f.write(html_report(resp))
```

L'HTML include: sezione verdict, parametri della regola validata, metriche IS,
execution envelope, MAE/MFE, walk-forward per split, validazione statistica,
breakdown per regime. Tutto implementato con CSS inline senza dipendenze esterne.

### `resp.to_dict() → dict`

Dizionario nidificato completo per serializzazione YAML/JSON:

```python
import json
with open(f"{resp.alpha_id}_rule_discovery.json", "w") as f:
    json.dump(resp.to_dict(), f, indent=2)
```

### `resp.persist(path)`

Salva la `RuleDiscoveryResponse` completa su disco come file pickle. Utile per
archiviare l'intera regola — verdetto, metriche, trade log, risultati
walk-forward — senza dover rieseguire il backtest.

```python
import pickle, pathlib

pathlib.Path("rules").mkdir(exist_ok=True)
for contract, resp in rule_responses:
    if resp.is_edge:
        resp.persist(f"rules/{resp.alpha_id}.pkl")

# Ricaricare in una sessione successiva
resp = pickle.load(open("rules/ALPHA-BTC-1H-000.pkl", "rb"))
print(resp.verdict, resp.in_sample_summary.profit_factor)
```

---

## Configurazione completa

### `BacktestParams`

| Parametro | Default | Descrizione |
|---|---|---|
| `direction` | `"long"` | `"long"` o `"short"`. Short = specchio simmetrico del long: entry sopra anchor, take-profit sotto fill |
| `buy_type` | `"limit"` | `"limit"` o `"market"` |
| `buy_drop_pct` | `0.010` | Entità dello scostamento dal anchor (es. 0.01 = 1%): sconto per long, premio per short |
| `buy_delay_bar` | `6` | Barre di vita dell'ordine limite |
| `buy_price_anchor` | `"close"` *(risolto dalla sessione)* | Colonna a cui si applica l'offset del limite — qualsiasi colonna numerica, anche un indicatore derivato (`"close_sma_3"` con `buy_drop_pct=0.10` = un limite al 90% della SMA a 3 barre). Riempita da `close_col` perché una colonna prezzo rinominata si porti dietro l'anchor di default; un anchor esplicito è un livello a sé |
| `sell_pct` | `0.040` | Take-profit come frazione del prezzo di fill |
| `target_h` | `24` | Barre tenute *dopo* la barra di fill prima dell'uscita a orizzonte (l'intervallo segnale→uscita è sempre `1 + target_h`). `0` = round-trip nella stessa sessione (close della barra di fill) |
| `target_col` | `"close"` *(risolto dalla sessione)* | Colonna per l'uscita a orizzonte. Deve nominare la stessa serie di `close_col` |
| `target_hit_col` | `"close"` | Colonna per rilevare il take-profit. Conservative = `"close"` per entrambe le direzioni; ottimistico = `"high"` per long, `"low"` per short (usare `optimistic_hit_col(direction)`) |
| `fee` | `0.002` *(risolto dalla sessione)* | Fee per lato, derivata da `AlphaConfig.fee_per_side` — la base di costo del contratto è il costo addebitato |
| `early_stopping` | `True` | Esci al take-profit; se False, sempre a orizzonte |

### `GridSpec`

| Parametro | Default | Descrizione |
|---|---|---|
| `buy_drop_pct` | `None` | Lista di sconti da testare (None = auto) |
| `sell_pct` | `None` | Lista di target da testare (None = auto) |
| `target_h` | `None` | Lista di orizzonti da testare (None = auto) |
| `buy_delay_bar` | `None` | Lista di delay da testare (None = auto) |

### `RuleWalkForwardConfig` (Rule Discovery)

| Parametro | Default | Descrizione |
|---|---|---|
| `n_splits` | `4` | Numero di finestre test OOS |
| `train_span_months` | `None` | Mesi di train (None = anchored, cresce) |
| `test_span_months` | `None` | Mesi di test (None = divisi ugualmente) |
| `min_train_months` | `6` *(risolto dalla sessione)* | Train minimo prima della prima finestra test — dimensionato da `criteria.min_tpm` con un margine di Poisson al 95% per raggiungere `_MIN_TRADES_ABS` (10) trade |
| `reoptimise` | `True` | Re-ottimizza la grid su ogni train window |
| `purge_bars` | `None` *(auto)* | Larghezza di purge, in barre, alla fine di ogni train window — le entrate aperte lì potrebbero riempirsi/uscire dentro la finestra di test adiacente, facendo trapelare la selezione di train nei prezzi di test. `None` (default) dimensiona automaticamente in base al `target_h` più grande della grid risolta più il delay di fill; `0` disabilita il purge |
| `embargo_bars` | `0` *(risolto dalla sessione)* | Quarantena extra all'inizio di ogni finestra test, in barre — risolto dalla sessione a partire da `AlphaConfig.embargo_bars` (stessa policy "quanta correlazione seriale mettere in quarantena dopo un confine", applicata al confine di fold anziché allo split IS/OOS di sessione); un valore esplicito qui prevale comunque |

### `SelectionCriteria`

| Parametro | Default | Descrizione |
|---|---|---|
| `min_profit_factor` | `2.0` | PF minimo per EDGE |
| `min_win_rate` | `0.55` | Win rate minimo (0–1) |
| `min_tpm` | `2.0` | Trade/mese minimi — definisce anche la soglia dinamica di trade `max(10, n_months × min_tpm)` |
| `min_pf_score_tpm` | `0.30` | Score composito minimo |
| `min_fill_rate` | `0.40` | Fill rate minimo. Inerte sotto il default `entry_mode="auto"` (lo Stage 1 è un entry a mercato, fill ≈100%) — significativo sotto `entry_mode="limit"` |
| `min_fill_rate_opt` | `0.80` | Soglia di fill rate per lo stadio di ottimizzazione limite di `entry_mode="auto"` — la soglia di fill che vincola davvero per default (vedi *Entry mode* sopra) |
| `min_net_gain_retention` | `0.5` *(risolto dalla sessione)* | Frazione del net gain OOS del punto market che il punto limite deve mantenere per essere adottato sotto `entry_mode="auto"` — la terza e più permissiva condizione di adozione (backstop per uno Sharpe con media e varianza minuscole) |
| `min_sell_pct` | `0.005` *(risolto dalla sessione da `AlphaConfig.mfe_floor`)* | Floor operativo sul take-profit derivato dal target del contratto, così un target irraggiungibile intrabar non viene mai pubblicato |
| `partial_min_profit_factor` | `1.5` | PF minimo per PARTIAL-EDGE |
| `min_active_month_rate` | `0.80` | Frazione minima di mesi IS con ≥1 trade per un EDGE pieno (`active_months / n_months >= min_active_month_rate`) — sostituto rate-based, timeframe-agnostico dei tetti fissi rimossi `max_zero_months_edge`/`max_zero_months_partial` |
| `max_regime_dependency` | `0.30` | Score dipendenza regime massimo per EDGE |
| `min_dsr` | `1.0` | Deflated Sharpe minimo per EDGE |
| `max_ttest_p` | `0.05` *(risolto dalla sessione da `PipelineContext.alpha`)* | p-value massimo t-test expectancy — l'unico gate hard per-ipotesi della pipeline |
| `max_rotation_p` | `0.05` *(risolto dalla sessione da `PipelineContext.alpha`)* | p-value massimo della rotation-null a livello di ricerca (`AlphaContract.rotation_p`) per un EDGE pieno — limita a PARTIAL-EDGE le regole che hanno solo vinto la lotteria del multiple-testing. Inerte quando il contratto non porta annotazione di rotazione |
| `power_gate` | `True` | Se `True`, degrada un `EDGE`/`PARTIAL-EDGE` che sarebbe tale a `INSUFFICIENT-DATA` se l'evidenza OOS pooled non può sostenerlo (vedi *Verdetto* sopra) |
| `min_oos_trades` | `10` | Trade di test walk-forward pooled minimi (su tutte le finestre di test) per un verdetto positivo con fiducia; sotto questa soglia, `INSUFFICIENT-DATA` quando `power_gate=True` |
| `early_elimination` | `True` | Se `True`, regole che falliscono lo screen rapido (Step 2.3) vengono rifiutate prima di walk-forward e diagnostiche; se `False`, la pipeline gira per intero — il verdetto rimane `NON-EDGE` ma con tutti i diagnostici popolati |

### `RuleDiscoveryConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `base_params` | `BacktestParams()` | Parametri fissi e centro della grid |
| `scoring` | `ScoringParams()` | Parametri del composite score |
| `grid` | `GridSpec()` | Grid operativa (vuota = auto) |
| `walk_forward` | `RuleWalkForwardConfig()` | Impostazioni walk-forward OOS |
| `criteria` | `SelectionCriteria()` | Soglie di selezione e verdetto |
| `entry_mode` | `"auto"` | `"auto"` (due stadi market→raffinamento limite, default dal #185), `"market"` (solo Stage 1) o `"limit"` (comportamento pre-#185 — l'entry a limite funge anche da ottimizzatore del prezzo di entry). Vedi *Entry mode* sopra |
| `use_contract_target` | `True` | Seed sell_pct/target_h dal contratto |
| `timestamp_col` | `"open_dt"` | Colonna datetime |
| `signal_col` | `"__rule_signal__"` | Nome colonna segnale iniettata |
| `discovery_date` | `None` | Data ISO per la response (None = oggi) |
| `selection_mode` | `"walk_forward"` | `"walk_forward"` (default) — il punto operativo e le metriche IS vengono solo dalle finestre di train del walk-forward; `"full_sample"` — comportamento legacy, intera tabella. Vedi Step 3 sopra |
| `wf_param_policy` | `"last"` | Come `selection_mode="walk_forward"` sceglie il punto pubblicato tra i vincitori per-split di train: `"last"` (default) o `"consensus"` |
| `n_trials_upstream` | `1` | Moltiplicatore incorporato nell'`n_trials` del Deflated Sharpe oltre al conteggio delle celle della grid, per un fattore di ricerca a monte esplicito (es. contratti fratelli che ricevono anch'essi un verdetto). Default `1` = solo celle della grid |

---

## Pattern d'uso avanzati

### Grid personalizzata

```python
from forgedge.rule_discovery import GridSpec

config = RuleDiscoveryConfig(
    grid=GridSpec(
        buy_drop_pct=[0.005, 0.010, 0.015, 0.020],
        sell_pct=[0.03, 0.04, 0.05, 0.06],
        target_h=[12, 24, 36, 48],
        buy_delay_bar=[3, 6, 12],
    )
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
```

### Analisi grid completa

```python
# Tutte le configurazioni testate, ordinate per score
grid_df = rd.grid_summary()
top10 = grid_df.nlargest(10, "pf_score_tpm")
print(top10[["sell_pct", "buy_drop_pct", "target_h", "profit_factor",
             "win_rate_pct", "tpm_mu", "pf_score_tpm"]])
```

### Export report per tutti i PARTIAL-EDGE+

```python
from forgedge.rule_discovery import html_report, text_report

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    if resp.is_edge:
        # HTML per review umana
        with open(f"reports/{resp.alpha_id}.html", "w") as f:
            f.write(html_report(resp))
        # JSON per integrazione
        import json
        with open(f"reports/{resp.alpha_id}.json", "w") as f:
            json.dump(resp.to_dict(), f, indent=2)
    else:
        print(f"NON-EDGE {resp.alpha_id}: {resp.rejection_reasons}")
```

### Collegare la response al contratto

`AlphaContract.rule_discovery_response` è il campo riservato a Rule Discovery:

```python
import json
contract.rule_discovery_response = resp.to_dict()
# Il contratto ora porta con sé la risposta operativa per il Rule Registry
```

---

## Note operative

- **`target_hit_col`:** il default `"close"` riproduce il motore certificato di
  riferimento (conservativo). Per la convenzione intrabar ottimistica, usare
  `optimistic_hit_col(direction)` — `"high"` per long, `"low"` per short.
- **Rule Registry:** Modulo 4 non ancora implementato. La `ValidatedRule`
  prodotta da Rule Discovery è già nella forma pronta per l'ingestion nel
  registry.
