# Proposta funzionale — modalità ranged per `min_tpm` (M1) e diagnostica di allineamento al mercato (M2)

**Stato del documento:** bozza di specifiche funzionali, non di design tecnico né di
implementazione. Raccoglie tre idee emerse da una sessione di esplorazione sul
pipeline FORGE, con lo stato attuale del codice verificato riga per riga e le
decisioni di indirizzo già prese. Nessun codice è stato scritto: questo documento è il
punto di partenza per un eventuale design tecnico successivo.

**Idee tracciate:**

- **A.** Event Discovery (M1): modalità "ranged" per `GateParams.min_tpm` — da floor a
  target con banda di tolleranza.
- **B.** Alpha Discovery (M2): etichettatura momentum / mean-reversion / idiosincratica
  per contratto, basata sulla correlazione tra l'esito della regola e il rendimento di
  mercato allo stesso orizzonte `h`.
- **C.** Alpha Discovery (M2): diagnostica di sufficienza della griglia degli orizzonti
  (`horizon_grid`) — emersa come prerequisito dell'idea B, ma tracciata come idea
  indipendente perché utile a prescindere da essa.

Le tre idee sono indipendenti nell'implementazione (toccano moduli/file diversi) ma B
dipende concettualmente da C per essere affidabile (si veda §4).

---

## 1. Contesto comune

FORGE separa il pipeline in domini che non si devono mescolare (vedi `CLAUDE.md`/skill
`forgedge`, invarianti #1 e #4): M1 vede solo struttura temporale delle attivazioni,
mai il rendimento; M2 deriva il target economico (orizzonte, direzione, size) dai dati,
per evento, mai a priori; M3 valuta l'operatività (fee, fill, backtest). Tutte e tre le
idee qui proposte rispettano questa separazione: nessuna delle tre introduce
un'osservazione del rendimento in M1, né fissa un target economico a priori in M2.

---

## 2. Idea A — Consistency Gate in modalità "ranged" (M1)

### 2.1 Motivazione

Oggi `min_tpm` è un floor a senso unico ovunque nel gate: si accetta un evento se il suo
tasso di attivazione è **almeno** `min_tpm`. Non esiste un modo per dire "voglio eventi
che si attivano *in media* a questo ritmo", ammettendo sia eventi più rari sia eventi
più frequenti solo entro la variabilità che il processo stesso mostra. Un caso d'uso
tipico: selezionare pattern "regolari" a una cadenza attesa, scartando sia i pattern
troppo rari (poco potere statistico) sia quelli quasi-sempre-attivi (spesso condizioni
banali/degenerate, es. un indicatore quasi sempre sopra soglia).

### 2.2 Stato attuale (verificato)

- `ConsistencyGate._gate_pass` (`event_discovery/consistency_gate.py:351-357`) implementa
  il criterio come disequazione a un lato:
  - modalità `"bar"`: `mean_tpm >= min_tpm` e `id_score <= max_dispersion`;
  - modalità `"episode"` (default): `episode_tpm >= min_tpm`, `n_episodes >= min_episodes`,
    `episode_id <= eff_max_dispersion`.
- La dispersione è già misurata come **indice di dispersione** (`Var/Mean` dei conteggi
  mensili) e confrontata con un **tetto** statisticamente motivato:
  `eff_max_dispersion = poisson_floor(n_months) × dispersion_margin`
  (`consistency_gate.py:290-304`) — il tetto oltre il quale un processo non è più
  compatibile con la casualità (chi-quadro al 95%, approssimazione di Wilson-Hilferty).
- `min_tpm` è la radice di una catena di derivazioni nel resolver
  (`resolver.py`, funzione `_derive_pf_min_tpm` e dintorni, righe ~928+): da esso si
  derivano `min_train_months`, `pf_min_tpm`, e vengono verificati vincoli come
  `oos_span_too_short`, `m3_stricter_than_m1`. Tutta questa catena assume la semantica di
  floor con margine di Poisson al 95% ("almeno N trade attesi al tasso minimo").

### 2.3 Proposta funzionale

Aggiungere una modalità di selezione alternativa al gate, **attiva solo su richiesta
esplicita** e **confinata a M1**:

- Nuovo campo (nome indicativo) `GateParams.tpm_mode: Literal["floor", "ranged"] = "floor"`.
  Default invariato → nessuna regressione sul comportamento esistente.
- In modalità `"ranged"`, `min_tpm` è interpretato come **centro** di una banda anziché
  come soglia minima. La larghezza della banda si deriva dalla dispersione *già
  misurata* dal gate per quell'evento — non un nuovo parametro statistico, ma la stessa
  quantità (`episode_id` / `eff_max_dispersion`) letta in chiave bidirezionale invece che
  come solo tetto.
- Criterio di accettazione (bozza): l'evento passa se il suo tasso osservato
  (`episode_tpm` o `mean_tpm`, a seconda di `event_counting`) cade entro
  `min_tpm ± k · σ_tpm`, dove `σ_tpm` è la deviazione standard implicita nella
  dispersione osservata a quel numero di mesi, e `k` gioca lo stesso ruolo di
  `dispersion_margin` oggi (quanta tolleranza oltre il minimo statisticamente
  difendibile).

### 2.4 Decisione già presa

**La banda ranged resta un filtro di selezione locale a M1.** Gli eventi che superano il
gate — in modalità `"floor"` o `"ranged"` — arrivano a M2 nella stessa forma
(`EventCandidate` con la sua serie di attivazione), indistinguibili dal punto di vista di
M2 in poi. Questo significa **esplicitamente**:

- Nessuna modifica alla catena di derivazione del resolver (`min_train_months`,
  `pf_min_tpm`, i controlli di coerenza `oos_span_too_short` / `m3_stricter_than_m1`
  restano ancorati alla semantica floor di `min_tpm`, invariata).
- La modalità ranged non introduce un nuovo "significato" di `min_tpm` visibile a valle:
  è un criterio di ammissione in più al gate, non una ridefinizione del parametro.
- Se in futuro servisse che anche M3/resolver ragionassero in termini di banda, sarà una
  proposta separata e esplicita, non un effetto collaterale di questa modalità.

### 2.5 Rischi / vincoli noti

- Va scelta con cura la stima di `σ_tpm` per non reintrodurre un nuovo parametro
  "magico" scollegato da `dispersion_margin` — l'intento è riusare la stessa
  informazione statistica già presente nel gate (si veda §2.2), non aggiungerne una
  indipendente.
- `min_episodes` (potere statistico minimo) resta comunque un vincolo assoluto separato,
  come oggi — la modalità ranged non lo sostituisce.
- Va deciso se la modalità ranged è compatibile con `event_counting="bar"` oltre che con
  `"episode"` (quest'ultimo è il default e il caso più naturale, dato che l'indice di
  dispersione episodico è già la quantità pensata per non essere gonfiata da stati
  persistenti multi-barra).

### 2.6 Domande aperte per il design tecnico

- Formula esatta di `σ_tpm` a partire da `episode_id`/`eff_max_dispersion` e
  `n_total_months`.
- Se `k` è un nuovo campo (`tpm_margin`) o se si riusa `dispersion_margin` anche per la
  banda.
- Comportamento quando `n_total_months` è troppo piccolo per stimare la dispersione
  (stesso caso limite già gestito in modalità floor, `consistency_gate.py:118-126`).

---

## 3. Idea B — Etichettatura momentum / mean-reversion / idiosincratica (M2)

### 3.1 Motivazione

Si vuole distinguere, tra i contratti con un vantaggio statistico confermato, quelli il
cui edge è "market beta mascherato" (la regola vince perché il mercato si muoveva già in
quella direzione) da quelli il cui edge è più idiosincratico (vince a prescindere dalla
direzione che il mercato ha effettivamente preso nello stesso periodo). Questa
informazione può servire sia come etichetta descrittiva (mean-reversion / momentum) sia
come indicazione di quanto l'edge sia "vantaggioso rispetto al mercato" in un senso più
ricco della sola media.

### 3.2 Stato attuale (verificato)

- `AlphaContract` deriva già `direction` e `mean_advantage` come eccesso rispetto alla
  baseline **incondizionata**: `Δ_h = μ_cond_h − μ_base_h`, dove `μ_base_h` è il
  rendimento medio a `h` barre su **tutte** le barre (`alpha_discovery/discovery.py:450-454`,
  `models.py:47-48`). Questo È già, nella sostanza, "il vantaggio rispetto al rendimento
  medio di mercato allo stesso orizzonte" — non va reinventato.
  - Conseguenza pratica: la parte "capire se la regola è più vantaggiosa del mercato"
    (una delle due letture proposte per questa idea) è in buona parte già coperta da
    `mean_advantage`/`direction`. Un confronto ulteriore — la regola batte il buy&hold
    cumulato sull'intero periodo OOS — è un confronto di **livello aggregato** diverso
    (media per-barra vs. rendimento composto sull'intera finestra) e più naturale come
    arricchimento del report M3/M4 (accanto a `BacktestSummary.net_gain`), non come
    nuovo criterio di selezione in M2. Non è oggetto di questo documento ma va tenuto
    presente come idea collegata, più leggera, per evitare di sovrapporla per errore a B.
  - Quello che **manca** è una misura di **co-movimento** (correlazione), non di livello:
    `mean_advantage` non dice se l'edge si muove *con* il mercato o *contro* di esso,
    solo *quanto* lo batte in media.
- Esiste già un'infrastruttura di correlazione riusabile, ma con un target diverso:
  `rule_registry/correlation.py` — `_spearman()` e `gain_correlation_by_date()`
  calcolano correlazione di Spearman tra i *gain* di **coppie di regole** (M4,
  de-duplicazione/diversificazione di portafoglio), non tra una regola e il rendimento
  del mercato sottostante.
- `AlphaContract.market_structure` (`alpha_discovery/models.py:788`, popolato da
  `alpha_discovery/market_structure.py`) calcola Hurst/ACF **sul prezzo/mercato**, come
  contesto interpretativo generale (il mercato nel suo complesso è mean-reverting o
  trending) — non è una proprietà della singola regola, e non risponde alla domanda "la
  *mia* regola si allinea al movimento di mercato quando si attiva?".

### 3.3 Proposta funzionale

Per ogni contratto (o candidato, a seconda di dove si decide di calcolarla), calcolare
la correlazione (Spearman, riusando `_spearman` — da estrarre in un modulo condiviso
invece di duplicarla) tra:

- il rendimento per-attivazione già calcolato da M2 per derivare `direction`/`lift`
  (nessun nuovo dato da produrre: la serie esiste già nella scansione della griglia
  degli orizzonti), e
- il rendimento di mercato realizzato sullo stesso orizzonte `h*`, sulle stesse barre di
  attivazione.

Dal segno e dall'intensità della correlazione, derivare un'etichetta a tre valori sul
contratto (nome di campo indicativo: `market_alignment`):

- **correlazione positiva forte → "momentum-aligned"**: la regola vince quando il
  mercato si muove nella stessa direzione — il suo edge si sovrappone (parzialmente) al
  movimento del mercato nello stesso periodo.
- **correlazione negativa forte → "mean-reversion-aligned"**: vince quando il mercato si
  muove nella direzione opposta.
- **vicino a zero → "idiosyncratic"**: l'edge non dipende dalla direzione che il mercato
  ha effettivamente preso nello stesso periodo.

Questa etichetta è un **diagnostico aggiuntivo**, non un nuovo gate di promozione: non
cambia `direction`, `lift`, né i criteri di `promoted_contracts()` esistenti — coerente
con l'impostazione "additiva, non invasiva" già scelta per l'idea A.

### 3.4 Dipendenza dall'idea C

**Questo è il punto critico emerso in discussione.** L'etichetta si calcola a
`h = holding_period_h` (h\*, derivato come `argmax|z_h|` sulla griglia
`AlphaConfig.horizon_grid`). Per un edge trend-following genuino, `|z_h|` può crescere
**monotonicamente** con `h` invece di avere un picco interno — h\* finisce allora pinnato
al bordo superiore della griglia non perché lì ci sia l'orizzonte ottimale, ma perché la
griglia non è stata scansionata abbastanza lontano. In quel caso:

- l'etichetta "momentum-aligned" verrebbe calcolata a un orizzonte arbitrario (il
  bordo della griglia configurata), non a un orizzonte realmente identificato dai dati;
- due regole trend-following vere, con griglie di ampiezza diversa, prenderebbero
  correlazioni/etichette diverse per un dettaglio di configurazione, non per una
  differenza reale di mercato.

**Conseguenza per il design:** l'etichetta "momentum-aligned" va assegnata con piena
fiducia solo quando h\* è un punto **interno** della griglia (picco genuino di
`|z_h|`); quando h\* è pinnato al bordo con `|z_h|` ancora crescente, l'etichetta va
declassata (es. `"momentum-aligned (horizon possibly under-scanned)"`) invece di essere
affermata come se fosse ben identificata. Il meccanismo di rilevamento è l'oggetto
dell'idea C (§4) — B dipende da C per essere affidabile sui casi più interessanti (i
trend-follower veri), anche se le due idee restano separate nell'implementazione.

### 3.5 Rischi / vincoli noti

- Serve chiarire se "rendimento di mercato a h" è il rendimento realizzato puntuale
  `log(close[t+h]/close[t])` (stessa unità già usata per `Δ_h`) o qualcos'altro — per
  coerenza con il resto di M2 si userebbe la prima definizione, in log-space (stessa
  scelta motivata in `target.py:114-119` per la robustezza alle code pesanti crypto).
- Va deciso un criterio quantitativo di soglia per "correlazione forte" vs "vicino a
  zero" (analogo alle soglie già esistenti per `min_direction_t`, `fdr_q`, ecc. — non un
  numero arbitrario scollegato dal resto della calibrazione statistica di M2).
- La correlazione andrebbe calcolata sul periodo di stima coerente con la validazione
  già esistente (IS vs. OOS confermato) per non introdurre una nuova forma di
  overfitting non controllata dalla walk-forward OOS di M1/rotazione già presente.

### 3.6 Domande aperte

- L'etichetta va calcolata anche sui candidati non promossi (diagnostica generale) o
  solo sui contratti promossi/EDGE (più economico, ma meno utile per capire perché un
  candidato non è stato promosso)?
- Va esposta anche una versione quantitativa (il coefficiente di correlazione grezzo)
  oltre all'etichetta categorica, per chi vuole soglie proprie?

---

## 4. Idea C — Diagnostica di sufficienza della griglia degli orizzonti (M2)

### 4.1 Motivazione

Emersa discutendo l'idea B, ma **tracciata come idea a sé stante**: è utile a
prescindere dall'etichettatura momentum/mean-reversion, ogni volta che si legge
`holding_period_h` come "l'orizzonte migliore" per un evento.

### 4.2 Stato attuale (verificato)

- Il codice ha già un guardrail per un problema **imparentato ma diverso**:
  `require_significant_direction` (`alpha_discovery/models.py:239-243`,
  `discovery.py:526-529`) declassa `direction` a `"undetermined"` quando **nessun**
  orizzonte della griglia è statisticamente significativo (BH-corrected) — il commento
  nel codice lo motiva esplicitamente: *"argmax|z_h| would assign a direction off a
  coin-flip (often the drift-driven long edge of the grid)"*.
- Questo guardrail protegge dal caso "segnale debole/rumore che casualmente si accumula
  verso il bordo della griglia". **Non protegge** dal caso opposto: un edge trend-following
  **vero e significativo**, dove `|z_h|` è ancora crescente all'ultimo punto scansionato —
  h\* pinnato al bordo non per rumore ma perché la griglia è strutturalmente troppo
  corta per quell'evento.
- `holding_period_h` alimenta a cascata: `purge_bars = max(horizon_grid)` in
  `TimeBudget.build` (`discovery.py:162`), e attraverso il resolver il sizing di
  `min_train_months` e altri campi bar-counting di M3 (`target_h`, `buy_delay_bar`).
  Un h\* artefatto si propaga quindi oltre M2.
- Esiste già un meccanismo di allargamento mirato della griglia — `EventCandidate
  .dominant_window()` più i moltiplicatori `(0.5, 1.0, 2.0)` (`models.py:447-451`) — ma è
  agganciato alla finestra dell'indicatore che ha generato l'evento, non a un test sulla
  forma di `|z_h|` osservata; non garantisce che un trend-follower genuino trovi un
  picco interno.

### 4.3 Proposta funzionale

Due livelli, dal più economico al più completo:

1. **Rilevamento (diagnostico, a costo pressoché zero).** Flag su `AlphaContract
   .diagnostics` (lo stesso campo non bloccante usato oggi per altre annotazioni — vedi
   pitfall #12 della skill `forgedge`, non `rejection_reasons`) quando
   `h* == max(horizon_grid)`. Si legge direttamente da `advantage_by_h`/i valori di
   `z_h` già calcolati per l'intera griglia — nessun ricalcolo.
2. **Distinzione fine (monotonicità).** Non basta sapere che h\* è al bordo: bisogna
   distinguere il caso patologico (`|z_h|` ancora in crescita all'ultimo punto — griglia
   sotto-scansionata) dal caso innocuo (il bordo coincide per caso con un vero massimo
   interno, `|z_h|` già in discesa prima del bordo o piatto). Controllo derivato dagli
   stessi dati (coerente con l'invariante #3 — nessuna assunzione a priori tipo "le
   regole trend-following usano h=X"), non un valore hardcoded.

Il risultato (diagnostico + esito del controllo di monotonicità) è la base su cui
l'idea B declassa la propria etichetta (§3.4).

### 4.4 Rischi / vincoli noti

- Allargare meccanicamente `horizon_grid` ogni volta che si rileva il pinning **non è
  gratis**: `purge_bars` cresce con `max(horizon_grid)`, riducendo la finestra di
  training utile; il sizing del resolver (`min_train_months` e la catena collegata)
  andrebbe rivalutato. La proposta qui è **solo diagnostica** (segnalare, non
  correggere automaticamente allargando la griglia) — un'eventuale politica di
  allargamento automatico è una proposta successiva e separata, con i suoi trade-off da
  valutare a parte.
- Il controllo di monotonicità va reso robusto al rumore sugli ultimi punti della
  griglia (differenze finite su pochi orizzonti vicini possono essere instabili) —
  probabilmente serve una tolleranza, non un confronto esatto punto-a-punto.

### 4.5 Domande aperte

- Il diagnostico si applica solo a `direction != "undetermined"` (ha senso solo dove
  esiste già un h\* significativo) o anche ai candidati scartati, per capire *perché*
  erano scartati?
- La soglia di tolleranza per "ancora in crescita" al bordo — assoluta o relativa alla
  scala di `|z_h|` osservata?

---

## 5. Relazioni tra le idee e sequenza consigliata

```
A (M1, ranged tpm)         — indipendente, nessuna dipendenza dalle altre due
C (M2, grid sufficiency)   — indipendente nell'implementazione, prerequisito
                              concettuale per usare B con fiducia sui trend-follower
B (M2, market alignment)   — usa l'esito di C per calibrare la confidenza dell'etichetta
```

Non c'è un obbligo di ordine di implementazione tra A e {B, C} (moduli diversi, nessuna
interazione). Tra B e C, C dovrebbe precedere o accompagnare B: implementare B da sola
produrrebbe etichette "momentum-aligned" sistematicamente meno affidabili proprio sui
casi più interessanti (i trend-follower veri con h\* al bordo).

## 6. Prossimi passi

Questo documento resta a livello di specifica funzionale. I passi successivi, da
decidere con l'utente:

- Congelare le formule esatte (§2.6, §3.6, §4.5) in una specifica tecnica per modulo.
- Decidere le soglie quantitative (bande di correlazione, tolleranza di monotonicità,
  `k`/`tpm_margin` per la banda ranged) con un audit empirico sui fixture esistenti
  (`tests/fixtures/ADA_1D_TRAIN.parquet`), sul modello di
  `docs/analysis/pipeline_parameter_coherence.md`.
- Solo dopo: apertura di branch/issue separati per A, B, C.
