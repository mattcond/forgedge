# FORGE — Dal Mercato al Segnale: Evento, Alpha e Regola

Questo documento spiega come FORGE produce segnali di trading a partire da dati
storici, costruendo progressivamente i tre concetti fondamentali del sistema:
l'**evento**, l'**alpha** e la **regola**. Ogni concetto risponde a una domanda
precisa e produce un artefatto formale che il successivo riceve in input — senza
mai retrocedere dati, riaprire ottimizzazioni chiuse, o mescolare i domini.

---

## Il problema che FORGE affronta

La ricerca di trading edge sistematica è infestata da un paradosso: è sempre
possibile trovare pattern che "avrebbero funzionato" nel passato. Basta
ottimizzare le soglie abbastanza a lungo, scegliere i periodi giusti, guardare
il rendimento mentre si costruisce il segnale. Il risultato è un backtest
attraente e una strategia che in produzione non funziona.

I tre vettori principali di questa distorsione sono:

1. **Look-ahead bias nella selezione del segnale.** Se la soglia dell'evento
   viene scelta perché produce rendimenti positivi, la soglia già "sa" il futuro.
   Qualsiasi backtest successivo è circolare.

2. **Ottimizzazione sullo stesso campione della valutazione.** Se si cerca il
   miglior orizzonte, il miglior take-profit, il miglior filtro di regime sugli
   stessi dati su cui si misura il Profit Factor, il PF riflette il rumore del
   campione, non l'edge strutturale.

3. **Assenza di separazione tra fase statistica e fase operativa.** Sapere che
   un evento predice statisticamente un rendimento positivo non equivale a sapere
   che è profittevole con fee, slippage e meccaniche d'ordine realistici.

FORGE risponde a questi tre problemi con una pipeline a tre moduli separati da
confini formali: nessun modulo può accedere ai dati del successivo, nessuna
soglia può essere ricalibrata dopo la scoperta.

---

## 1. L'Evento: Osservare il Mercato senza Pregiudizi

### Definizione

Un **evento** è una condizione booleana su barre storiche: per ogni barra della
KPI Table, l'evento è `True` (attivo) o `False` (inattivo). Non è un segnale
di trading — è un'osservazione strutturata del mercato.

```
Evento: RSI_14 < 31.2
         │
         ├─ Barra 4712 (RSI=29.8): True  ← evento attivo
         ├─ Barra 4713 (RSI=33.1): False
         ├─ Barra 4714 (RSI=30.0): True  ← evento attivo
         └─ ...
```

Un evento può essere composto da più condizioni in AND:

```
Evento: RSI_14 < 31.2  AND  spread_ema_9_25 < -0.0118
```

Il nome formale nella codebase è `EventCandidate`. L'espressione booleana è
salvata in `c.expression` e `c.event_formula`.

### Cosa rende un evento interessante

Non tutti gli eventi sono utili. Un evento che si attiva una volta sola, o
sempre nello stesso mese, o con una frequenza del 0.1% delle barre, non
permette di misurare nulla di affidabile. I criteri di selezione sono
puramente **strutturali**. L'unità di conteggio di default è l'**episodio**
(una sequenza di attivazioni consecutive), non la singola barra attivata
— un segnale che resta attivo per più barre di fila conta una volta sola,
non una volta per barra:

| Criterio | Significato |
|---|---|
| Frequenza | Episodi al mese ≥ `min_tpm` (default 0.5 episodi/mese) |
| Potenza | Almeno `min_episodes` episodi nel campione (default 10) |
| Dispersione | Indice di Dispersione a livello di episodio (Var/Media dei conteggi mensili di episodi) ≤ una soglia Poisson-χ² scalata da `dispersion_margin` (default 1.3) — `eff_max_dispersion = poisson_floor(n_months) × dispersion_margin` |

Il gate non rigetta mai un evento statisticamente coerente con un processo
casuale alla sua stessa frequenza osservata, pur lasciando che la tolleranza
di un preset per la burstiness continui davvero a vincolare. Non esiste più
una regola fissa "nessun mese sopra il 40% delle attivazioni" —
`max_monthly_share` è ancora riportato, ma solo come diagnostica; il vero
controllo di dispersione è il test Poisson-χ² sopra. (Una modalità di
conteggio legacy `"bar"`, più vicina alla vecchia semantica, resta
disponibile per retrocompatibilità.)

Il ConsistencyGate di Modulo 1 filtra tutti gli eventi che non rispettano questi
criteri. Quello che supera il gate è un evento con **struttura temporale stabile**
— indipendentemente dal fatto che produca rendimenti positivi o negativi.

### Soglie distribuzionali, non assolute

La soglia `RSI_14 < 31.2` non è scelta perché 31.2 è un valore magico. È il
**10° percentile della distribuzione IS del RSI su quell'asset specifico**.
Questo è chiamato "soglia distribuzionale": rappresenta uno stato estremo
dell'indicatore *per quell'asset*, non un valore assoluto universale.

Il risultato è che lo stesso pattern strutturale ("RSI in zona bassa") produce
soglie diverse su asset diversi:

| Asset | RSI p10 | Significato |
|---|---|---|
| BTC | 27.8 | BTC entra in ipervenduto a 27.8 |
| ADA | 31.2 | ADA entra in ipervenduto a 31.2 |
| ETH | 29.1 | ETH entra in ipervenduto a 29.1 |

La struttura è la stessa (RSI al 10° percentile), la soglia è adattiva.

### L'isolamento dal forward return

Il principio fondamentale di Modulo 1 è che **non vede mai il rendimento
futuro**. Le feature vengono classificate, le soglie vengono fissate sulla
distribuzione IS, la stabilità temporale viene misurata — tutto senza mai
calcolare un singolo forward return.

Questo garantisce che le soglie dell'evento non siano contaminate dalla
conoscenza di ciò che succede dopo. Non c'è look-ahead bias nella selezione
del segnale.

### Le soglie sono immutabili

Una volta che l'Event Discovery fissa le soglie — `RSI_14 < 31.2`,
`spread_ema_9_25 < -0.0118` — queste non cambiano mai, nemmeno se i moduli
successivi scoprissero che una soglia diversa produce un Profit Factor più alto.
Cambiare la soglia richiederebbe una nuova sessione di Event Discovery, con
un nuovo campione IS.

Questo previene la forma più sottile di look-ahead bias: ottimizzare le soglie
dopo aver visto i rendimenti.

### Artefatto: `EventCandidate`

```python
c = candidates[0]
c.expression      # "rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118"
c.event_formula   # versione human-readable con notazione percentile
c.activation_stats.n_activations  # 87 attivazioni IS
c.activation_stats.n_active_months # attivo in 18 mesi diversi

# Applicare l'evento su qualsiasi DataFrame con le stesse colonne native
signal = c.apply(new_kpi_table)   # pd.Series bool
```

L'EventCandidate è portabile: può essere applicato a dati futuri senza
richiedere la sessione di scoperta. Le soglie sono salvate nelle componenti.

---

## 2. L'Alpha: Misurare il Potere Predittivo

### Il passaggio al forward return

Un evento descrive una configurazione di mercato. Ma non dice nulla su cosa
succede dopo. È normale che RSI sia al 10° percentile — può essere un segnale
di rimbalzo o l'inizio di un crollo ulteriore.

L'**alpha** è la risposta empirica alla domanda: *dato che l'evento si è
attivato, che cosa succede statisticamente nei prossimi h bar?*

Questa è la **prima esposizione al rendimento futuro** dell'intera pipeline. Il
forward return non è mai stato calcolato prima di questo punto.

### Derivare il target senza assunzioni

Alpha Discovery non riceve dall'utente un orizzonte temporale, una direzione di
trading, o un take-profit. Li **deriva dai dati** per ogni evento.

**Selezione dell'orizzonte `h*`:**

La grid di orizzonti stessa non è una costante fissa — è risolta a livello di
sessione per classe di timeframe da `PipelineContext`: `(1, 2, 3, 5, 7, 10)`
barre sui timeframe giornalieri, `(1, 2, 4, 8, 12, 24)` sull'intraday,
`(1, 2, 5, 10, 20, 50)` sull'HFT. (Costruire un `AlphaConfig` standalone,
fuori da `forge()`, ricade sulla grid oraria/intraday `(1, 2, 4, 8, 12, 24)`
a prescindere dal timeframe dichiarato.)

Per ogni evento, Alpha Discovery calcola il rendimento log in eccesso a ogni
orizzonte della grid, `Δ_h = mean_advantage[h]` — il rendimento log medio
delle barre attive meno la baseline *incondizionata* su tutte le barre valide
a quell'orizzonte — e poi lo standardizza rispetto a una **null di rotazione
circolare**: il pattern di attivazione dell'evento viene ruotato rispetto
alla serie dei rendimenti reali molte volte, per costruire una distribuzione
nulla di come apparirebbe `Δ_h` in assenza di una relazione reale, ottenendo
uno score standardizzato `z_h = Δ_h / σ_null,h` e un p-value per ogni
orizzonte. L'orizzonte selezionato è:

```
h* = argmax |z_h|
```

Questo ha sostituito una precedente deflazione naive `Δ_h / (σ_cond / √n)`
(divisione per l'errore standard naive, all'incirca una correzione a forma
`1/√h`) perché quell'approccio tratta le finestre di forward-return
sovrapposte di un evento clusterizzato o episodico come campioni
indipendenti: il suo denominatore si riduce con l'orizzonte a prescindere dal
fatto che l'evento si concentri davvero in raffiche, il che gonfia lo score
agli orizzonti lunghi e spinge `h*` verso l'estremo della grid anche in
assenza di un vero edge. Standardizzare rispetto a una null costruita sul
pattern di attivazione dell'evento stesso rimuove questa distorsione.

Viene poi applicato un gate FDR di Benjamini-Hochberg ai p-value su tutta la
grid, producendo l'insieme di orizzonti `h_sig` che superano la
significatività. `h*` è sempre scelto come `argmax |z_h|` sull'*intera* grid
— ma se `h*` cade fuori da `h_sig`, il target viene marcato
`statistically_weak` (questo penalizza, non scarta, lo score alpha risultante
— vedi lo score composito più sotto).

**Derivazione della direzione:**

`direction` è `"undetermined"` — e l'evento viene rigettato — in uno
qualsiasi dei tre casi seguenti:

- nessun orizzonte produce un eccesso finito `Δ_h` (non finito su tutta la
  grid);
- l'eccesso standardizzato all'orizzonte selezionato è troppo piccolo,
  `|z_h*| < min_direction_t` (default 0.5);
- (comportamento di default, `require_significant_direction=True`) `h*`
  stesso non è BH-significativo (`statistically_weak`) — l'eccesso è
  statisticamente indistinguibile dalla null di rotazione ovunque, quindi
  leggere una direzione da `argmax|z_h|` equivarrebbe a un lancio di moneta,
  spesso distorto verso il drift dell'asset stesso all'estremo lungo della
  grid.

Altrimenti la direzione è `"long"` quando `Δ_h*` (`mean_advantage[h*]`) è
positivo, `"short"` quando è negativo.

**Derivazione del `sell_pct`:**

Il take-profit baseline non è la media dei rendimenti (che include le barre
perdenti), ma il **quantile della distribuzione delle Maximum Favorable
Excursion (MFE)** delle barre attive all'orizzonte `h*`:

```
sell_pct = max(quantile(MFE_active_bars, 0.5), 0.005)
```

La MFE di una barra attiva è la massima escursione favorevole raggiunta nelle
`h*` barre successive. Il 50° percentile di questa distribuzione è la stima
conservativa del take-profit: metà delle attivazioni IS ha raggiunto o superato
questa escursione.

### Misurare l'evidenza statistica

Con il target derivato `(h*, direction*, sell_pct*)`, Alpha Discovery misura
l'evidenza statistica su IS:

| Misura | Significato |
|---|---|
| **IC** (Information Coefficient) | Correlazione di Spearman tra la feature continua e il rendimento forward a `h*`. Misura la forza del segnale continuo prima della soglia. |
| **Win rate e lift** | Frequenza di barre attive con rendimento orientato positivo; lift = win_rate - base_rate. Misura quante volte l'evento "ha ragione". |
| **Cohen's d** | Separazione tra la distribuzione dei rendimenti sulle barre attive e sulle barre inattive. Misura la grandezza dell'effetto. |
| **IC rolling** | Stabilità del segno IC su ≈20 finestre scorrevoli. Misura se la relazione è stabile nel tempo. |

Tutte le misure IS vengono calcolate con primitive statistiche implementate in
**puro numpy** — Spearman, t-test, FDR di Benjamini-Hochberg, beta incompleta.

### Conferma out-of-sample

Dopo le misure IS, il target derivato `(h*, direction*, sell_pct*)` viene
**replicato sull'OOS tail** (l'ultimo 30% del dataset, non mai toccato da IS):

- I rendimenti OOS vengono orientati per la direction derivata
- Si misura win rate, lift, mean_advantage e t-test sull'OOS
- L'OOS viene considerato confermato se: mean_advantage > 0 e p-value < 0.10
  (`oos_max_p`). Non esiste una soglia separata sul numero di attivazioni —
  il commento nel codice sorgente è esplicito: "solo il p-value determina il
  passed: la dimensione del campione è già incorporata nel p" (un campione
  OOS piccolo rende semplicemente il p-value più difficile da superare).

La conferma OOS è una **diagnostica non bloccante**: un contratto OOS-debole
viene comunque promosso se ha direzione determinata, ma il suo grade riflette
la mancanza di conferma. Questo evita di rigettare eventi rari ma strutturalmente
solidi per i quali l'OOS ha poche attivazioni.

### Il grade A–D

Lo score composito (0–1) integra le misure IS. I pesi di default
(`AlphaConfig.score_weights`) sono `(0.20, 0.25, 0.15, 0.25, 0.15)` per
`(ic, lift, cohens_d, z, breadth)` — una formula a **cinque** termini:

```
score = 0.20 × IC_norm + 0.25 × lift_norm + 0.15 × d_norm
      + 0.25 × z_norm  + 0.15 × regime_breadth
```

`z_norm` è l'eccesso standardizzato rispetto alla null di rotazione a `h*`
normalizzato (`|z_h*|`, il rapporto edge-rumore calcolato sopra) — un termine
che la formula naive omette del tutto. `d_norm` è **con segno** in
`[-1, 1]`: un Cohen's d negativo (il gruppo condizionato performa peggio del
background) penalizza attivamente lo score invece di essere troncato a zero.

Vengono poi applicati due ulteriori aggiustamenti dopo la somma pesata:

- se l'orizzonte selezionato è `statistically_weak` (fuori dall'insieme
  BH-significativo), il composito viene moltiplicato per
  `statistically_weak_penalty` (default `0.6`) — un orizzonte scelto proprio
  a causa del bias di selezione che il controllo FDR vuole prevenire non può
  ottenere uno score alto;
- se la conferma OOS passa, viene aggiunto `oos_bonus` (default `0.05`).

Il risultato è troncato a `[0, 1]`. Ogni componente grezza è normalizzata su
una scala 0–1 prima della ponderazione. Il grade:

| Grade | Score | Significato |
|---|---|---|
| A | ≥ 0.75 | Evidenza statistica forte |
| B | ≥ 0.50 | Evidenza solida |
| C | ≥ 0.25 | Evidenza moderata |
| D | < 0.25 | Evidenza debole — potenzialmente solo rumore |

Tutti e quattro i grade passano a Rule Discovery. Il grade non è un filtro di
promozione: è un'indicazione della forza dell'evidenza statistica.

### Artefatto: `AlphaContract`

```python
c = promoted[0]

# Il target derivato dai dati
dt = c.derived_target
print(f"Orizzonte: {dt.holding_period_h}h")
print(f"Direzione: {dt.direction}")        # "long" o "short"
print(f"sell_pct:  {dt.sell_pct:.4f}")     # es. 0.0312 = 3.12%

# L'evidenza statistica IS
print(f"IC: {c.underlying_feature.ic:.4f}, p={c.underlying_feature.p_value:.4f}")
print(f"Lift: {c.event_stats.lift:.4f}")
print(f"Grade: {c.alpha_score.grade}")     # "A", "B", "C" o "D"

# Conferma OOS
oos = c.oos_validation
print(f"OOS passed: {oos.passed}, lift OOS: {oos.lift:.4f}")

# Status
print(f"Status: {c.status}")               # "HYPOTHESIS" o "REJECTED"
```

Un AlphaContract con `status="HYPOTHESIS"` è l'ipotesi formale che l'evento
ha potere predittivo nella direzione indicata all'orizzonte indicato. Non è una
certezza — è un'ipotesi da testare operativamente.

---

## 3. La Regola: Tradare in Modo Realistico

### Perché l'alpha non è sufficiente

Un evento con grade A, IC=0.08, lift=0.12, confermato OOS — è un risultato
notevole. Ma non risponde alla domanda che conta in produzione: *posso guadagnare
soldi tradando questo segnale, con fee reali, con ordini limite che potrebbero
non venire eseguiti, su un orizzonte che devo chiudere anche se il mercato non
ha mosso abbastanza?*

L'evidenza statistica misura la separazione delle distribuzioni. Non misura
la redditività operativa. Questa è la responsabilità di Modulo 3.

### La meccanica dell'entrata: una valutazione auto a due stadi

Rule Discovery traduce l'AlphaContract in un backtest con meccaniche d'ordine
realistiche. Dall'issue #185, `RuleDiscoveryConfig.entry_mode` ha default
`"auto"`, che esegue la valutazione in **due stadi** invece di assumere
un'entrata a limite fin dall'inizio:

**Stadio 1 — entrata a mercato (autoritativa per il verdetto).** La regola
viene backtestata entrando all'apertura della barra successiva (fill ≈
100%). Questo isola l'edge del *segnale* da qualsiasi ottimizzazione del
prezzo d'entrata, e il suo verdetto è definitivo: lo Stadio 2 può raffinare
quali parametri vengono pubblicati, ma non può mai trasformare un NON-EDGE
dello Stadio 1 in un edge.

**Stadio 2 — sweep opzionale del prezzo limite.** Su un sopravvissuto dello
Stadio 1, Rule Discovery esplora opzionalmente `buy_drop_pct` (un ordine
limite a `fill_price = close × (1 - buy_drop_pct)`, valido per
`buy_delay_bar` barre, eseguito quando il prezzo scende a quel livello nelle
barre successive usando il close come approssimazione conservativa) e
replica il candidato vincente out-of-sample. Il prezzo limite viene
**adottato** — pubblicato al posto dell'entrata a mercato — solo se supera
tutte e tre le condizioni OOS:

1. `fill_rate >= min_fill_rate_opt` — nessun PF gonfiato da fill rari.
2. `opportunity_sharpe >= quello del mercato` — uno Sharpe *per frequenza di
   trade*, quindi un punto che tratta meno spesso deve guadagnare di più per
   trade per compensare.
3. `net_gain >= min_net_gain_retention × quello del mercato` — una rete di
   sicurezza per il caso che lo Sharpe non vede (una mu minuscola con una
   sigma minuscola).

`RuleDiscoveryResponse.entry_optimization.failed_condition` indica quale
delle tre ha bloccato l'adozione (`"fill"` / `"sharpe"` / `"net_gain"`), o
`None` quando è stata adottata. Questo esiste perché il vecchio default
solo-limite lasciava che un limite profondo e raramente eseguito gonfiasse
il profit factor su un sottoinsieme di trade non rappresentativo (il "fill
confound") — l'entrata svolgeva un doppio ruolo, meccanica d'ordine e
ottimizzatore del prezzo d'entrata, e il verdetto finiva per misurare il
prezzo d'entrata invece del segnale.

**Uscita (in entrambe le modalità d'entrata):**
- **Take-profit:** si esce quando il prezzo sale a
  `take_profit = fill_price × (1 + sell_pct)`
- **Stop a orizzonte:** se il take-profit non viene raggiunto entro `target_h`
  barre dal fill, si chiude al close di quella barra
- **Short:** specchio simmetrico — entrata sopra anchor, take-profit sotto fill

**Fee:** dedotte sia all'entrata sia all'uscita su ogni trade (`fee_per_side`).

`entry_mode="limit"` — il default pre-#185 — resta pienamente supportato: la
grid varia direttamente `buy_drop_pct` e l'entrata a limite svolge il doppio
ruolo di ottimizzatore del prezzo d'entrata, senza una baseline a mercato.
Rimane la scelta giusta quando l'ordine limite *è* la strategia, non una mera
raffinatura dell'esecuzione. Una terza modalità, `entry_mode="market"`,
esegue solo lo Stadio 1 senza alcun ottimizzatore d'entrata.

### La grid dei parametri

Rule Discovery non assume i valori ottimali di `buy_drop_pct`, `sell_pct` e
`target_h`. Invece di un menu fisso di valori candidati, `build_grid()`
costruisce un piccolo **ventaglio simmetrico** aritmeticamente attorno a
ciascun valore base derivato dal contratto:

```
buy_drop_pct = [d − 0.005, d − 0.002, d, d + 0.002, d + 0.005]   (floor a 0.001)
sell_pct     = [s − 0.02,  s − 0.01,  s, s + 0.01,  s + 0.02]    (floor a 0.005)
target_h     = {round(h × 0.5), round(h × 1.0), round(h × 2.0)}
buy_delay_bar = [valore base]   — un unico valore, non esplorato, a meno che
                                  il chiamante imposti esplicitamente
                                  GridSpec.buy_delay_bar
```

dove `d`, `s` e `h` sono `buy_drop_pct`, `sell_pct` e `target_h` presi dal
target derivato dell'AlphaContract (via `base.resolved()`). Qualsiasi asse
che il chiamante imposti esplicitamente su `GridSpec` sovrascrive questo
ventaglio automatico.

Per ogni configurazione viene calcolato il composite score `pf_score_tpm` che
bilancia Profit Factor, frequenza di trading e consistenza mensile. Le
configurazioni vengono filtrate contro una soglia dinamica sul numero di
trade, `max(pf_min_trades, n_months × pf_min_tpm)` — `pf_min_trades` ha
default 15 e `pf_min_tpm` ha default 2 trade/mese, quindi la soglia cresce
con la lunghezza della finestra di selezione invece di restare fissa a un
conteggio piatto. Le configurazioni sotto questa soglia, o con PF < 1,
vengono scartate immediatamente.

La configurazione migliore tra quelle che superano le soglie di selezione viene
passata alla validazione statistica e walk-forward.

### La validazione walk-forward

Il backtest IS da solo sarebbe ancora ottimistico: si è scelto il miglior set
di parametri su IS, quindi il risultato IS è ottimisticamente biased.

Il walk-forward OOS divide la storia in finestre scorrevoli (train + test):

```
  [────────── train 1 ──────────] [test 1]
        [────────── train 2 ──────────] [test 2]
              [────────── train 3 ──────────] [test 3]
```

Su ogni finestra di train, i parametri vengono re-ottimizzati. Sul test
successivo si misura il PF con quei parametri. Il `wf.consistency` è la
quota di finestre test con PF > 1 — la misura più diretta di robustezza
out-of-sample.

### Il verdetto EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA

Il verdetto finale integra gate hard e gate soft:

**NON-EDGE (gate hard):**
- PF < 1.5 su IS (`partial_min_profit_factor`)
- Tutti i mesi con risultato negativo o zero (troppo irregolare)
- Nessuna configurazione con fill rate adeguato

**EDGE** richiede tutti i seguenti:
- PF ≥ 2.0 su IS (`min_profit_factor`)
- Win rate ≥ 55%
- Deflated Sharpe ≥ 1.0 (penalizzato per il numero di configurazioni testate)
- Dipendenza regime < 30% (l'edge non è concentrato in un solo regime)
- Stabilità temporale: PF prima metà ≈ PF seconda metà
- Walk-forward `consistency ≥ 0.5` (almeno metà delle finestre di test OOS
  hanno PF > 1)
- La null di rotazione a livello di ricerca è superata: `rotation_p ≤
  max_rotation_p`. È il meccanismo `fast_null`/`FastRotationNull` che
  `forge()` esegue di default, che quota la superficie di multiple-testing
  dell'intera sessione di scoperta — un contratto che vince solo quella
  lotteria resta limitato sotto il pieno EDGE anche se supera ogni altro
  gate.

**PARTIAL-EDGE:** supera i gate hard (NON-EDGE) ma non tutti i gate EDGE.

**INSUFFICIENT-DATA:** un verdetto che sarebbe `EDGE`/`PARTIAL-EDGE` viene
declassato a `INSUFFICIENT-DATA` quando l'evidenza out-of-sample aggregata è
troppo esigua da supportare un giudizio positivo con fiducia (governato da
`SelectionCriteria.power_gate`, default `True`, tramite
`_power_assessment()`). Un `NON-EDGE` non viene mai salvato in questo modo —
sotto-potenziato o meno, la conseguenza operativa è la stessa. Il verdetto
è quindi uno dei **quattro** valori:
`"EDGE" | "PARTIAL-EDGE" | "NON-EDGE" | "INSUFFICIENT-DATA"`
(`RuleDiscoveryResponse.verdict`).

### Artefatto: `ValidatedRule` (dentro `RuleDiscoveryResponse`)

```python
resp = rd.run()

print(f"Verdetto: {resp.verdict}")   # "EDGE", "PARTIAL-EDGE", "NON-EDGE" o "INSUFFICIENT-DATA"
print(f"È edge: {resp.is_edge}")     # True per EDGE e PARTIAL-EDGE

if resp.is_edge:
    vr = resp.validated_rule
    params = vr.params
    print(f"Entrata: limite a -{params.buy_drop_pct:.2%} dal close")
    print(f"Take-profit: +{params.sell_pct:.2%} dal fill")
    print(f"Orizzonte massimo: {params.target_h} barre")
    print(f"Direzione: {params.direction}")

# Metriche IS
s = resp.in_sample_summary
print(f"Trade: {s.total_trades}, PF: {s.profit_factor:.2f}, WR: {s.win_rate_pct:.2%}")

# Walk-forward
wf = resp.walk_forward
print(f"Consistenza OOS: {wf.consistency:.0%}")  # es. 75% delle finestre con PF>1
```

Una `ValidatedRule` contiene i parametri operativi precisi — non stime
statistiche, ma valori pronti per un sistema di esecuzione.

---

## 4. Dal Contratto al Segnale di Trading

### Quando l'evento si riattiva su nuovi dati

Una volta che il sistema ha prodotto un `AlphaContract` (HYPOTHESIS) e una
`ValidatedRule` (EDGE o PARTIAL-EDGE), la logica operativa è semplice:

> Quando l'evento si attiva su nuovi dati → genera un segnale di trading.

```python
# Nuovi dati in arrivo (live o batch)
new_kpi = fetch_latest_bars(asset="BTC", timeframe="1H")

# Il segnale è l'evento applicato ai nuovi dati
signal = event_candidate.apply(new_kpi)    # pd.Series bool

# Le barre con segnale True sono candidati all'ingresso
signal_bars = new_kpi[signal]
```

### I parametri dell'ordine

La `ValidatedRule` traduce il segnale in istruzioni d'ordine precise:

```python
params = validated_rule.params

for ts, row in signal_bars.iterrows():
    # Calcolo del prezzo limite (long)
    limit_price = row["close"] * (1 - params.buy_drop_pct)
    take_profit = limit_price * (1 + params.sell_pct)
    max_bars    = params.target_h   # chiudiamo entro questo numero di barre

    # Istruzione al sistema di esecuzione
    place_limit_order(
        direction   = params.direction,    # "long" o "short"
        limit_price = limit_price,
        take_profit = take_profit,
        max_bars    = max_bars,
        fee         = params.fee,
    )
```

Ogni parametro è il risultato di un processo empirico strutturato:
- `direction` — derivato da Alpha Discovery (sign del mean_advantage)
- `buy_drop_pct` — ottimizzato da Rule Discovery su IS, validato su OOS
- `sell_pct` — radicato nella distribuzione MFE IS, raffinato da Rule Discovery
- `target_h` — radicato nell'orizzonte Alpha, raffinato da Rule Discovery

### Filtrare per regime

Un edge statisticamente valido può essere più robusto in certi regimi. La
`RuleDiscoveryResponse` espone l'analisi per regime:

```python
ra = resp.regime_analysis
print(f"Regime da evitare: {ra.avoid_in}")
print(f"Score di dipendenza dal regime: {ra.dependency_score:.2f}")

# Filtrare il segnale per regime
enriched = MarketContext(new_kpi).run()
regime_now = enriched["regime"].iloc[-1]

if regime_now not in ra.avoid_in:
    # L'edge è presente in questo regime → procedi con l'ordine
    pass
```

### Il flusso completo dal dato al segnale

```
Dati storici (KPI Table)
       │
       ▼
[Modulo 0] Classifica ogni barra per regime
       │
       ▼
[Modulo 1] Scopre eventi con struttura temporale stabile
           → EventCandidate  (soglie distribuzionali, immutabili)
       │
       ▼
[Modulo 2] Misura il potere predittivo (senza assunzioni a priori)
           → AlphaContract   (orizzonte, direzione, sell_pct derivati)
       │
       ▼
[Modulo 3] Valida operativamente con meccaniche d'ordine realistiche
           → ValidatedRule   (parametri precisi per il sistema di esecuzione)
       │
       ▼
Nuovi dati (live)
       │
       ▼
EventCandidate.apply()  →  segnale bool per barra
       │ True
       ▼
ValidatedRule.params    →  limit_price, take_profit, max_bars, direction
       │
       ▼
Sistema di esecuzione   →  ordine
```

---

## 5. Perché Questa Separazione è Necessaria

### I tre confini formali

La pipeline FORGE impone tre confini che non possono essere attraversati in
nessuna direzione:

**Confine 1: Modulo 1 non vede il forward return.**
Se l'evento venisse scoperto guardando i rendimenti, le soglie sarebbero
calibrate per massimizzare il PF — non per catturare una struttura di mercato
reale. Le soglie immutabili sono la conseguenza diretta di questo confine.

**Confine 2: Alpha Discovery non riceve parametri economici in input.**
L'orizzonte e la direzione non sono assunzioni dell'utente: sono misure
empiriche. Se l'utente potesse specificare "ho ipotesi che questo evento sia
un segnale long a 24 barre", la misura confermerebbe l'ipotesi anche quando
i dati la contraddicono parzialmente.

**Confine 3: Rule Discovery valuta su OOS scorrevole, non su IS.**
La configurazione ottimale dei parametri operativi trovata su IS sarebbe
sovra-adattata. Il walk-forward separa l'ottimizzazione dalla misurazione su
ogni finestra.

### Le soglie immutabili come garanzia

Il principio che le soglie degli eventi non cambiano mai — nemmeno se una soglia
diversa producesse un PF migliore — è la protezione contro la forma più sottile
di look-ahead bias: ottimizzare le soglie dell'evento *dopo* aver visto i
rendimenti.

Recalibrarle richiederebbe una nuova sessione di Event Discovery, su un nuovo
campione IS, con una nuova serie storica.

### Il target derivato come misura, non come assunzione

Il fatto che Alpha Discovery derivi l'orizzonte, la direzione e il sell_pct dai
dati — invece di riceverli come parametri — significa che ogni AlphaContract
è una risposta empirica, non la verifica di un'ipotesi preconcetta.

Un utente potrebbe intuire che "RSI basso è un segnale di rimbalzo a 24 ore".
FORGE misura se l'evento ha un vantaggio più forte a 6, 12, 24 o 36 ore, e
seleziona l'orizzonte con il miglior rapporto segnale/orizzonte — che potrebbe
essere diverso dall'intuizione.

### L'OOS come gate formale, non come check post-hoc

L'OOS di Alpha Discovery e il walk-forward di Rule Discovery non sono analisi
opzionali eseguite dopo la promozione: sono condizioni formali della pipeline.
I dati OOS non partecipano a nessun calcolo IS — sono un osservatore
indipendente che non ha mai "visto" le soglie, il target derivato, o la
configurazione operativa ottimale.

### Due strati che mantengono onesti i confini lungo tutta la sessione

Due meccanismi, entrambi invisibili nelle descrizioni dei singoli moduli qui
sopra, si trovano sotto ogni esecuzione di `forge()` e fanno rispettare la
separazione nella pratica:

**Un resolver centrale dei parametri.** `forge()` costruisce un
`PipelineContext` (la fonte unica di verità della sessione per timeframe,
nomi delle colonne dello schema, fee e policy statistica — `forgedge.resolver`)
e risolve ogni campo di configurazione lasciato non impostato dal chiamante
contro di esso, prima ancora che il Modulo 0 venga eseguito. Il bundle
risolto viene poi controllato per contraddizioni interne da `config_report()`
(`ConfigReport`): `forge(strict=True)` — il default — solleva un
`ValueError` invece di eseguire una sessione la cui configurazione non può
strutturalmente produrre un verdetto (es. una finestra di selezione di
Modulo 3 troppo corta per la frequenza di arrivo che le è stato chiesto di
richiedere). Un muro di rigetti silenziosi è indistinguibile da "il segnale
è cattivo"; rifiutarsi di partire è la risposta onesta.

**Calibrazione della null di rotazione sulla ricerca stessa.** Ogni singolo
`AlphaContract` è già standardizzato rispetto alla propria null di rotazione
circolare (§2 sopra). Di default `forge()` esegue inoltre una null di
rotazione *a livello di ricerca* (`fast_null=True`,
`calibration.fast_null.FastRotationNull`) che quota la superficie di
multiple-testing dell'intera sessione di scoperta — quanti candidati sono
stati tentati, non solo quanto appare forte un candidato isolatamente — e
annota il risultato su `AlphaContract.rotation_p`. Il gate EDGE di Rule
Discovery legge questo valore direttamente (§3 sopra): un contratto che vince
solo la lotteria del multiple-testing resta limitato a `PARTIAL-EDGE` anche
se supera ogni altro gate.

---

## Riepilogo dei tre concetti

| Concetto | Domanda | Modulo | Artefatto | Gate |
|---|---|---|---|---|
| **Evento** | "Questa configurazione di mercato è stabile e ripetibile?" | Modulo 1 | `EventCandidate` | ConsistencyGate (strutturale, no forward return) |
| **Alpha** | "Questo evento predice statisticamente un rendimento orientato?" | Modulo 2 | `AlphaContract` | direction ≠ "undetermined" (unico gate hard) |
| **Regola** | "Questo alpha è redditizio con meccaniche d'ordine realistiche?" | Modulo 3 | `ValidatedRule` | EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA |

Un segnale di trading nasce dall'intersezione di questi tre accertamenti: una
configurazione di mercato strutturalmente stabile (`Evento`), con evidenza
empirica di potere predittivo (`Alpha`), che reggere sotto condizioni operative
realistiche (`Regola`).

---

## Riferimenti

| File | Contenuto |
|---|---|
| `index_it.md` | Panoramica del sistema e quick start |
| `how_to_use_it.md` | Guida pratica end-to-end con configurazione completa |
| `configuration_it.md` | Riferimento globale della configurazione — ogni dataclass e campo |
| `modulo_0_it.md` | Market Context: classificazione dei regimi |
| `modulo_1_it.md` | Event Discovery: pipeline, ConsistencyGate, EventCandidate |
| `modulo_2_it.md` | Alpha Discovery: derivazione target, IC, OOS, AlphaContract |
| `modulo_3_it.md` | Rule Discovery: backtest, verdetto EDGE, walk-forward, report |
| `modulo_4_it.md` | Rule Registry: dedup, replay cross-ticker, genericità |
