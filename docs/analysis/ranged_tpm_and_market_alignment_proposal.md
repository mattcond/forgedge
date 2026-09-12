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

### 2.3 Proposta funzionale (formule congelate)

Aggiungere una modalità di selezione alternativa al gate, **attiva solo su richiesta
esplicita** e **confinata a M1**. Due nuovi campi su `GateParams`:

```python
tpm_mode: Literal["floor", "ranged"] = "floor"   # default invariato, nessuna regressione
tpm_tolerance: float = UNSET                      # stessa unità di min_tpm
```

In modalità `"ranged"`, `min_tpm` è il **centro** della banda. `tpm_tolerance` è la
mezza-larghezza, in unità assolute (episodi/mese o barre/mese, come `min_tpm`), con due
comportamenti a seconda che sia lasciato `UNSET` o fissato dall'utente — lo stesso
idioma `UNSET`→derivato usato ovunque nel resolver, ma risolto **localmente dentro il
gate**, non nella tabella `CONSTRAINTS` di `resolver.py` (si veda il perché in §2.4):

- **`UNSET` (default) → derivato dalla dispersione tollerata di sessione:**

  ```
  σ_tpm = sqrt(eff_max_dispersion · min_tpm / n_total_months)
  tpm_tolerance_effettivo = z · σ_tpm            (z = 1.959964, normale a due code, 95%)
  ```

  `eff_max_dispersion` è il tetto **di sessione** già calcolato dal gate (§2.2), non
  l'`episode_id` del singolo evento. È una scelta deliberata, non la più ovvia — si veda
  §2.4 per il perché.

- **Fissato esplicitamente dall'utente → banda letterale**, nessuna statistica coinvolta:
  `banda = [min_tpm − tpm_tolerance, min_tpm + tpm_tolerance]`. Risolve direttamente il
  caso d'uso "voglio eventi con mediamente 4 episodi al mese, tolleranza 2": si scrive
  `tpm_tolerance=2.0` (con `min_tpm=4.0`, anch'esso in override) e si ottiene esattamente
  quella banda, senza che nessuna formula derivata la modifichi.

In entrambi i casi: `banda = [max(0, centro − mezza_larghezza), centro + mezza_larghezza]`
(un tasso non può essere negativo).

**Precondizione strutturale, non parte della banda:** un evento con `n_episodes == 0`
(non si attiva mai) è scartato a monte, prima di valutare la banda — non è "un evento con
tasso 0 che eventualmente rientra nella banda per effetto del troncamento a zero", è
semplicemente non un evento. Non richiede una nuova verifica: `min_episodes` (default 10)
lo esclude già in pratica, ma la precondizione va resa esplicita nel codice invece di
dipendere da quel default.

Il criterio di burstiness (`episode_id <= eff_max_dispersion`, invariato) resta
**indipendente** dal criterio di banda — la modalità ranged aggiunge un vincolo sul
livello del tasso, non ne rimpiazza o modifica uno esistente su forma/regolarità.

### 2.4 Perché il default usa `eff_max_dispersion` (di sessione) e non `episode_id` (dell'evento)

Prima scelta scartata: usare `episode_id` proprio dell'evento al posto di
`eff_max_dispersion` renderebbe la banda adattiva per evento (più stretta per un evento
regolare, più larga per uno vicino al tetto di burstiness ammesso). Sembra più
"principiato" statisticamente, ma ha un difetto concreto: **accoppia nella direzione
sbagliata** i due criteri del gate. Un evento già borderline sulla burstiness (`ID`
vicino al tetto) riceverebbe automaticamente *più* tolleranza sul tasso — l'opposto di
quanto si vorrebbe da un filtro di qualità (più scrutinio, non meno, dove la struttura
temporale è già al limite). In più, `episode_id` è una varianza campionaria stimata su
pochi conteggi mensili: con `n_total_months` piccolo (il regime più comune per i preset
più selettivi, es. `"sniper"`) la stima è essa stessa rumorosa, e quel rumore si
propagherebbe due volte nella larghezza della banda.

Usare `eff_max_dispersion` (il tetto di sessione, identico per ogni evento) risolve
entrambi i problemi: è una funzione chiusa di `n_total_months` e `dispersion_margin`,
nessuna stima campionaria per-evento coinvolta, e — punto chiave — equivale a valutare la
formula "adattiva" nel caso peggiore ammissibile (`episode_id ≤ eff_max_dispersion` per
ogni evento che supera comunque il gate), quindi è **sempre almeno larga quanto** lo
sarebbe stata la versione adattiva per quello specifico evento, mai più stretta. Il
costo è la perdita di adattività per-evento (un evento molto regolare non viene premiato
con una banda più stretta) — un costo di precisione, non di sicurezza, e comunque
aggirabile fissando `tpm_tolerance` esplicitamente quando serve.

### 2.5 Decisioni già prese

- **La banda ranged resta un filtro di selezione locale a M1.** Gli eventi che superano
  il gate — in modalità `"floor"` o `"ranged"` — arrivano a M2 nella stessa forma
  (`EventCandidate` con la sua serie di attivazione), indistinguibili dal punto di vista
  di M2 in poi. Nessuna modifica alla catena di derivazione del resolver
  (`min_train_months`, `pf_min_tpm`, i controlli `oos_span_too_short` /
  `m3_stricter_than_m1` restano ancorati alla semantica floor di `min_tpm`, invariata).
- **La derivazione di `tpm_tolerance` non può vivere nel resolver di sessione.**
  `eff_max_dispersion` dipende da `n_total_months`, una quantità **nota solo dal
  dataset reale**. Il resolver (`resolver.py`) è per costruzione cieco ai dati — deriva
  tutto da configurazione + `PipelineContext`, mai da `n_bars`/`span_months` — e
  `config_report()`/`forge_preset()` costruiscono le configurazioni senza mai vedere la
  tabella KPI. `eff_max_dispersion` infatti **già oggi** si calcola dentro
  `ConsistencyGate.evaluate()` (`consistency_gate.py:143`), non nel resolver — la
  derivazione di `tpm_tolerance` va nello stesso posto, per lo stesso motivo.
  Conseguenza pratica: a differenza di `min_train_months`/`pf_min_tpm`,
  `tpm_tolerance` derivato **non comparirà** in `ResolutionTrace`
  (`result.resolution.describe()`) — è locale al gate, non alla sessione.
- **Integrazione con `forge_preset()`.** I quattro preset (`sniper`/`balanced`/`sweep`/
  `burst`) non impostano oggi `tpm_mode`/`tpm_tolerance` — comportamento invariato per
  chi non li usa esplicitamente. Per attivarli tramite preset serve aggiungerli alla
  whitelist di override di `forge_preset()` (`presets.py:352-362`, che oggi rifiuta con
  `TypeError` ogni chiave non elencata) esattamente come `min_tpm`/`dispersion_margin`:
  `tpm_mode = overrides.pop("tpm_mode", "floor")`,
  `tpm_tolerance = overrides.pop("tpm_tolerance", UNSET)`, passati a `GateParams(...)`.
  Nessuna voce di calibrazione per-preset serve per `tpm_tolerance` (a differenza di
  `min_tpm`, che ha un valore `daily_min_tpm_episode` da scalare per timeframe) perché
  resta `UNSET` di default su tutti e quattro. Il default derivato eredita comunque la
  filosofia del preset "gratis", perché `eff_max_dispersion` dipende da
  `dispersion_margin`, che ogni preset già fissa in modo diverso (`sniper` quasi zero
  slack → banda stretta; `burst` clustering deliberatamente Poisson-implausibile →
  banda larga).
- **Incompatibilità di preset segnalata:** `"burst"` è disegnato esplicitamente per
  ammettere concentrazione temporale estrema (`dispersion_margin=3.00`, commento in
  `presets.py:300`: *"clustering Poisson-implausibile, di proposito"*) — l'opposto
  filosofico di "seleziona una cadenza tipica". Combinare `tpm_mode="ranged"` con
  `"burst"` è sconsigliato, sullo stesso piano della nota già esistente "non abbinare
  `sniper` al `RotationCalibrator`" (pitfall #6 della skill `forgedge`).

### 2.6 Rischi / vincoli noti

- `min_episodes` (potere statistico minimo) resta un vincolo assoluto separato, come
  oggi — la modalità ranged non lo sostituisce, si somma ad esso.
- Va deciso se la modalità ranged è utilizzabile anche con `event_counting="bar"` oltre
  che con `"episode"` (default) — in `"bar"` mode non esiste un `eff_max_dispersion` con
  floor (`max_dispersion` è lì una soglia assoluta senza floor, si veda §2.2): la scelta
  più coerente sarebbe usare `max_dispersion` stesso come input di `σ_tpm` in quel caso,
  non un'osservazione per-evento.
- Ambiguità di unità già presente su `min_tpm` (episodi/mese vs. barre/mese a seconda di
  `event_counting`) si propaga a `tpm_tolerance`: un valore letterale impostato pensando
  a un modo conta-eventi e poi riletto sotto l'altro cambia scala silenziosamente. Non è
  un problema nuovo introdotto da questa feature, ma va documentato esplicitamente perché
  qui riguarda sia il centro sia il raggio della banda.
- Vincolo implementativo per l'eventuale composizione AND: `ANDComposer.compose()` valuta
  il gate in forma vettorizzata via `_gate_pass` (`consistency_gate.py:307-357`), condivisa
  col path a singolo evento per costruzione (fix #226) — il criterio ranged deve entrare
  nello stesso punto condiviso, non essere reimplementato separatamente nel path batch,
  per non riaprire la stessa classe di bug.

### 2.7 Domande aperte residue

- Se estendere la modalità ranged a `event_counting="bar"` ora o rimandarlo (vedi §2.6).
- Se `z=1.959964` (95% a due code) deve restare fisso nel default derivato o esporre un
  modo per cambiarlo senza dover per forza passare a `tpm_tolerance` letterale.

### 2.8 Validazione empirica — `"balanced"` e `"sniper"`, fixture reale

Metodo: `tpm_mode="ranged"` non esiste ancora nel codice (nessuna sorgente è stata
modificata). Per testare il criterio proposto sui path reali di Event Discovery —
inclusi Diversity Gate e `ANDComposer`, non solo il gate a singolo evento — lo script di
verifica sostituisce a runtime (monkeypatch, solo nel processo dello script, mai nel
pacchetto) l'unico chokepoint condiviso già identificato in §2.6:
`consistency_gate._gate_pass`, importato con lo stesso nome anche da `and_composer.py`
(fix #226 lo rende l'unica fonte di verità sia per il path a singolo evento sia per
quello batch dei compositi). Sostituire quella funzione per la durata di una run è
comportamentalmente equivalente ad avere implementato la modalità per davvero, ai fini
del test. Dataset: `tests/fixtures/ADA_1D_TRAIN.parquet` (882 barre, 29 mesi).

**`"balanced"` (`min_tpm=1.0`, `dispersion_margin=1.30`) — banda derivata `[0.496, 1.504]`:**

| | grezzi (pre-gate) | dopo Diversity Gate + `max_and=2` |
|---|---|---|
| floor (oggi) | 1662 atomici | 1288 atomici + 2000 composti = 3288 |
| ranged (proposto) | 2571 atomici | 2033 atomici + 2000 composti = 4033 |

Risultato inatteso: su questo asset, a default, **ranged è netto più permissivo del
floor attuale** (2571 vs 1662 grezzi; 2033 vs 1288 dopo dedup), non più selettivo.
Il lato "ammetti chi è appena sotto il target" (1694 candidati grezzi tra 0.496 e 1.0
tpm/mese) è molto più popolato del lato "escludi chi è molto sopra" (785 candidati
sopra 1.504, tipicamente 3.5-3.9 tpm/mese — questi sì genuinamente "troppo frequenti",
il caso che l'idea A intende intercettare). Ipotesi iniziale scartata dai dati: il
surplus non è quasi-duplicazione — il tasso di deduplica (~21%) è quasi identico a
quello del floor (~22.5%), quindi la maggioranza dei 1694 sono pattern distinti, non
soglie percentili ridondanti sulla stessa feature.

**`"sniper"` (`min_tpm=0.3`, `dispersion_margin=1.05`) — banda derivata `[0.052, 0.548]`:**

| | grezzi (pre-gate) | dopo Diversity Gate + `max_and=2` |
|---|---|---|
| floor (oggi) | 3809 atomici | 2992 atomici + 2000 composti = 4992 |
| ranged (proposto) | 681 atomici | 561 atomici + 2000 composti = 2561 |

Qui l'effetto si **inverte completamente**: ranged è drasticamente più restrittivo del
floor (681 vs 3809 grezzi, −82%). Con `min_tpm=0.3` il floor di oggi è quasi un
non-filtro (la stragrande maggioranza delle feature si attiva più spesso di una volta
ogni ~3 mesi), mentre il bordo superiore della banda derivata (0.548, appena 1.8× il
target) fa quasi tutto il lavoro di selezione. Nessun candidato rientra nel caso
"ammesso solo da ranged" (`ranged-only = 0`): a `n_total_months=29`, il pavimento di
potenza statistica `min_episodes >= 10` implica già da solo un tasso minimo di
`10/29 ≈ 0.345`, sopra il bordo inferiore della banda (0.052) — quel meccanismo di
"estensione verso il basso" non ha mai la possibilità di attivarsi su questo preset.

**Conclusione della validazione:** la formula (§2.3, tenuta semplice e interpretabile
come deciso) è confermata **corretta e stabile** — nessun bug, nessun comportamento
degenere. Ma il suo **effetto netto sul volume di candidati non è una proprietà fissa
della modalità ranged**: dipende da dove il `min_tpm` del preset cade rispetto alla
distribuzione reale dei tassi delle feature su quell'asset. Un preset con `min_tpm`
basso (`sniper`) vede dominare il bordo superiore (più restrittivo); un preset con
`min_tpm` moderato (`balanced`) vede dominare il bordo inferiore (più permissivo). Va
letto come comportamento atteso e specifico di preset/asset, non come difetto della
formula — coerente con la scelta already presa di non complicarla per inseguire un
effetto netto uniforme.

**Limite noto del metodo:** il conteggio dei compositi satura a 2000 in tutti e quattro
i casi testati — quasi certamente `_MAX_PAIRS`, il tetto duro di `and_composer.py`, non
un segnale di "nessuna differenza a valle". Con pool atomiche anche solo nell'ordine
delle centinaia il numero di coppie possibili supera ampiamente quel tetto, quindi il
conteggio dei compositi non è informativo per confrontare la pressione computazionale
reale tra le configurazioni — solo la dimensione della pool atomica pre-composizione lo è.

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

- **Idea A: validazione empirica chiusa** (§2.8) — formula confermata corretta e
  stabile su due preset (`"balanced"`, `"sniper"`) sul fixture di riferimento; l'effetto
  netto sul volume di candidati varia per preset/asset per una ragione capita e
  spiegata, non per un difetto della formula. Resta da congelare solo il dettaglio
  tecnico residuo di §2.7 (`"bar"` mode, se esporre `z`).
- Congelare le formule esatte per B (§3.6) e C (§4.5) in una specifica tecnica per
  modulo, con lo stesso tipo di audit empirico appena fatto per A.
- Solo dopo: apertura di branch/issue separati per A, B, C.
