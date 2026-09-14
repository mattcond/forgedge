# Copertura esaustiva della composizione AND grade-guided — analisi funzionale

## 1. Osservazione che origina l'analisi

Due esecuzioni separate della stessa configurazione (DAAX 1H, preset
`balanced`, stessa finestra di discovery/holdout) hanno prodotto **due
regole `AND`-composte `PARTIAL-EDGE` diverse**, con orizzonti di
detenzione differenti (una con `h*=192` barre, l'altra proveniente da una
coppia composta differente). Nessuna delle due esecuzioni ha cambiato
dati, configurazione o codice — solo il processo Python è stato
riavviato tra un run e l'altro.

Questo solleva la domanda: **la composizione AND-composta di FORGE
garantisce di aver scansionato tutte le coppie/triple realmente
potenziali, o il risultato dipende da un campionamento che può variare
run su run?** Se la seconda, ogni "regola trovata" andrebbe letta come
"una delle regole che *sarebbe potuta* emergere", non "la migliore
regola disponibile in quel pool" — una distinzione che conta quando si
decide se investigare ulteriormente un `PARTIAL-EDGE` o scartarlo.

## 2. Meccanismo realmente in gioco: `grade_guided_compose`

Va chiarita subito una distinzione: FORGE ha **due** meccanismi di
composizione AND, non uno:

1. **`ANDComposer` "strutturale"** (`event_discovery/and_composer.py`),
   invocato da `EventDiscovery.run()` quando `max_and_components > 1`.
   Appaia eventi in base a tpm/dispersione strutturali, con un pool
   ridotto a `_MAX_PER_SLOT=3` eventi per (feature, transform, params) e
   un cap di `_MAX_PAIRS=2000`/`_MAX_TRIPLES=500`.
2. **`grade_guided_compose`** (`composition/grade_pairing.py`, issue
   #254), invocato da `forge()` quando `two_pass_composition=True`
   (il default). Appaia eventi in base al voto A-D che la prima passata
   di Alpha Discovery assegna a ciascun candidato 1D — un segnale di
   pairing empiricamente molto più forte di quello strutturale (l'issue
   #254 riporta 122 contratti `PARTIAL-EDGE`/`EDGE` contro 5, sullo
   stesso dataset).

**Tutti i test di questa sessione su DAAX 1H hanno usato `forge()` con le
impostazioni di default — quindi il meccanismo realmente responsabile
delle due regole diverse è `grade_guided_compose`, non l'`ANDComposer`
strutturale.** Il resto del documento si concentra su questo.

`grade_guided_compose` funziona così:

- **Pool**: ogni candidato 1D con un voto (`pool_selector=lambda pool:
  pool` — nessuna riduzione per-slot, a differenza del composer
  strutturale).
- **Strati**: le coppie vengono raggruppate per compatibilità di grado
  (`A_same`, `A_B`, `B_same`, `B_C`, `C_same`, `C_D`; `D` non è mai
  radice, solo partner di `B`/`C`) via `_stratum_key`.
- **Budget**: `effective_max_pairs = n_strata * per_stratum_pair_cap`,
  con `per_stratum_pair_cap=100` di default — quindi tipicamente
  300-600 coppie totali materializzate, indipendentemente da quante ne
  esistano davvero.
- **Selezione**: `ANDComposer.compose()` riceve uno `stratify_fn` che
  assegna ogni coppia valida al suo stratum, mescola ciascuno stratum con
  un seed *fisso* (`_shuffle_order`, issue #230), e interleaving
  round-robin tra strati fino a riempire il budget.

Il seed fisso di `_shuffle_order` garantisce che, **a parità di ordine
del pool in ingresso**, lo stesso shuffle produca sempre lo stesso
risultato. Il problema è che l'ordine del pool stesso **non è garantito
stabile tra processi separati**: `EventDiscovery`/`FeatureGenerator`
iterano internamente su `set()` di nomi di colonna (documentato nella
skill del progetto come limite noto, issue #24 — dipende da
`PYTHONHASHSEED`, randomizzato per processo di default in Python). Il
seed fisso protegge quindi la riproducibilità *entro* un processo, non
*tra* processi diversi — esattamente il sintomo osservato.

## 3. Misure empiriche (DAAX 1H, preset `balanced`, finestra 2021-01-03 →
2026-03-03, 30529 barre)

| Quantità | Valore |
|---|---|
| Candidati M1 (gate-passati) | 3133 |
| Distribuzione voti M2 pass-1 | D=2947, C=183, B=3 (nessun A) |
| **Coppie strutturalmente valide (pre-stratify, pre-gate)** | **4.899.803** |
| Coppie ammissibili per grado (post-stratify, pre-gate) | 4.890.974 |
| Coppie materializzate oggi (cap) | 300-500 |
| **% dello spazio realmente coperto oggi** | **≈0.01%** |
| Tempo Pass-1 esaustivo (AND uint8 + soglia volume) su tutte le 4.9M coppie | 139.5s |
| Coppie che superano il pre-filtro di volume | 177.982 (3.6%) |
| Tempo Pass-2 completo (matmul mensile + statistiche episodio + gate) sulle 177.982 volume-passing | 302.3s |
| **Coppie che superano il gate COMPLETO** | **112.169** |
| **Tempo totale scansione esaustiva (Pass-1+Pass-2)** | **441.8s (~7.4 min)** |
| Tempo M1+M2 pass-1 (già speso oggi, invariato) | 291s (~4.9 min) |

Il dato chiave: **il sistema vede meno dello 0.5% delle composizioni che
realmente supererebbero il gate** (112.169 trovate contro 300-500
materializzate), e il costo per scoprirle **tutte** è di soli ~7.4
minuti aggiuntivi — un fattore ~2.5× sul tempo che M1+M2-pass1 già
impiegano, non un ordine di grandezza. La computazione Pass-1/Pass-2 è
già vettorizzata e a chunk limitato in memoria (issue #228) per
costruzione — il limite odierno non è di potenza di calcolo, è
esclusivamente nella soglia di *quante* composizioni vengono
materializzate come oggetti `EventCandidate` prima di essere passate a
M2 pass-2.

## 4. Perché non si può semplicemente "alzare il cap" a 112.169

M2 pass-2 (Alpha Discovery) è per-candidato molto più costoso del gate
di composizione: già 3133 candidati singoli costano 152s in fase di
grading. Centomila candidati composti, ciascuno con la propria serie
booleana da mantenere in memoria e la propria scansione di orizzonti +
rotation-null, scalerebbero a ore di calcolo e a un footprint di memoria
proibitivo (coerente con gli OOM già osservati in questa sessione su
pool anche solo di 8000 candidati a 5 minuti). Il cap a valle di M2 resta
necessario — il problema non è "il cap esiste", è "il cap taglia una
popolazione che non abbiamo mai visto per intero, con un criterio
(l'ordine di shuffle) che non riflette il merito della composizione".

## 5. Proposta: disaccoppiare scansione da materializzazione

1. **Scansione esaustiva** (Pass-1 + Pass-2 completi, come sopra) su
   tutto lo spazio di coppie ammissibili per stratum — nessun
   early-break sul cap. Costo dimostrato: ~7.4 minuti su DAAX 1H.
   Raccoglie, per ogni coppia che supera il gate, le statistiche già
   calcolate gratuitamente dal gate stesso (`episode_tpm`,
   `episode_index_of_dispersion`, `n_episodes`) — senza costruire ancora
   l'oggetto `EventCandidate` completo (il passo davvero costoso).
2. **Selezione per merito**, non per ordine di scoperta: dalla
   popolazione *nota per intero*, ordinare ogni stratum per un punteggio
   composito calcolato da quelle stesse statistiche gratuite (vedi §6),
   e materializzare solo le top-N per stratum (stesso ordine di
   grandezza del cap odierno, 100-500).
3. **Diagnostica di copertura**: esporre "N coppie esaminate, M
   gate-passanti, K materializzate" per stratum — oggi questo
   troncamento è invisibile; renderlo visibile come già fa
   `event_distribution_report` per il gate M1 (issue #215).

Risultato: la selezione diventa **deterministica per costruzione**
(basata su punteggio calcolato sulla popolazione completa, non su un
ordine di iterazione dipendente dall'hash-seed del processo) —
risolvendo la causa radice del sintomo osservato al §1, non solo il suo
effetto.

## 6. Punteggio di merito proposto — e simulazione

Un punteggio composito costruito solo dalle statistiche che il gate
stesso già calcola per ogni composizione (nessun costo aggiuntivo):

- **Vicinanza al target di tpm**: `-|log(episode_tpm / min_tpm)|` — le
  composizioni vicine al centro della banda tpm configurata sono
  preferite a quelle appena sopra la soglia minima o eccessivamente
  frequenti.
- **Dispersione**: `-episode_index_of_dispersion` — composizioni più
  "regolari" (meno bursty) sono preferite.
- **Numero di episodi**: `n_episodes` — più evidenza statistica.

I tre criteri sono su scale diverse e non comparabili direttamente;
per combinarli senza pesi arbitrari si usa il **rango percentile entro
lo stesso stratum** di ciascun criterio, poi la loro media —
un `merit_score` in `[0, 1]`, confrontabile tra strati di dimensione
diversa.

**Risultati della simulazione** (stesso pool DAAX 1H di cui sopra, scansione
esaustiva completata in 276s, 112.122 coppie gate-passanti — coerente con
la misura precedente):

Distribuzione dei gate-passer per stratum:

| Stratum | Gate-passer |
|---|---|
| D_same | 107.223 |
| C_D | 4.424 |
| C_same | 472 |
| B_C | 3 |

`D_same` domina (95.6%) — coerente con la distribuzione dei voti M2
(2947/3133 candidati votati `D`).

Confronto tra la selezione per merito (top-100/stratum per `merit_score`)
e un campionamento casuale della stessa dimensione dallo stesso pool
(proxy della dipendenza-da-ordine odierna — 303 coppie in entrambi i
casi, capped da `B_C` che ne ha solo 3 disponibili):

| Metrica | Merito | Casuale (proxy) | Target/soglia |
|---|---|---|---|
| coppie source uniche | **49** | **149** | — |
| tpm mediano | 26.89 | 28.73 | 24.00 |
| dispersione mediana | **0.642** | 0.987 | ≤1.706 |
| n_episodi mediano | 1694 | 1810 | — |
| n_episodi p10 (coda peggiore) | 1559.8 | 1561.4 | — |
| stabilità (stesso pool, ordine mescolato) | **303/303 identiche (100%)** | — | — |

**Il punteggio di merito funziona come previsto su due dei tre criteri**
(dispersione nettamente più bassa, tpm più vicino al target) **ed è
perfettamente stabile/deterministico** (100% delle coppie selezionate
sono le stesse indipendentemente dall'ordine con cui il pool viene
attraversato — risolve esattamente l'instabilità del §1).

**Ma introduce un problema nuovo, non anticipato**: la selezione per
merito copre solo **49 combinazioni di feature sorgente distinte**,
contro le **149** di un campionamento casuale della stessa dimensione.
Ordinare per punteggio, senza vincoli di diversità, converge sulle
poche feature "migliori" per tpm/dispersione/episodi e le ripropone con
soglie leggermente diverse — esattamente il comportamento che l'issue
#230 aveva già dovuto correggere per il composer strutturale (dominanza
di un singolo componente "radice"). Il campionamento casuale, pur
instabile, offriva involontariamente più diversità.

**Implicazione per il design**: il punteggio di merito da solo non
basta — va combinato con un vincolo di diversità esplicito, non lasciato
come effetto collaterale della casualità. Due strade compatibili con
quanto già esiste nel codice:

- Applicare il filtro `max_constituent_jaccard` (già presente in
  `ANDComposer.compose()`, oggi opt-in e disattivato di default) prima
  del ranking per merito, scartando coppie troppo simili tra loro.
- Raggruppare per "feature sorgente coinvolta" oltre che per grade
  stratum, e fare round-robin *tra* i gruppi di feature prima di
  applicare il merito *dentro* ciascun gruppo — la stessa logica
  dell'interleaving round-robin che `_stratified_pair_order` già usa tra
  grade strata, applicata a un secondo livello di stratificazione.

## 7. Raccomandazione

1. Disaccoppiare scansione (esaustiva, ~7.4 min dimostrati) da
   materializzazione (cap invariato, 100-500 per stratum).
2. Selezionare non per punteggio grezzo, ma per **punteggio di merito
   dentro una quota di diversità per feature sorgente** — la sola
   ottimizzazione statistica, come mostra la simulazione, rischia di
   restringere involontariamente la varietà strutturale delle regole
   scoperte.
3. Esporre la diagnostica di copertura (§5, punto 3) così un utente vede
   sempre quanto dello spazio ammissibile è stato davvero scansionato e
   quanta diversità la selezione finale conserva rispetto al pool
   completo (non solo il conteggio ammesso oggi).

## 8. Politica a due fasce: esaustivo per A/B, campionato per C/D

Una raffinazione che nasce direttamente dai numeri del §3: dato che i
gradi A e B sono per costruzione rari (in questo pool: B=3, A=0, contro
D=2947), qualunque stratum "radicato" solo in A/B genera per costruzione
pochissime coppie — il cap da 100 non li tocca nemmeno. La combinatoria
esplosiva è concentrata esclusivamente dove compaiono C o D.

Regola generale: **esaustivo quando entrambi i gradi della coppia sono
in {A, B}; campionato quando almeno uno dei due è C o D.**

| Stratum | Entrambi in {A,B}? | Trattamento |
|---|---|---|
| `A_same`, `A_B`, `B_same` | sì | **esaustivo** (nessun campionamento — il volume naturale è già sotto controllo) |
| `B_C` | no | campionato (merito + diversità, §6) |
| `C_same`, `C_D` | no | campionato (merito + diversità, §6) |
| `D_same` | no | campionato — **da trattare col budget più stretto**, essendo lo stratum più popoloso in assoluto (4.335.094 coppie ammissibili, 107.223 gate-passanti, il 97.6% dell'intero spazio ammissibile e il 95.6% dei gate-passer totali misurati) |

Non è una regola arbitraria sovrapposta al design: è la stessa
scansione esaustiva del §5 che, applicata a valle, rivela quali strati
sono già sotto al cap (quindi "esaustivo" è semplicemente ciò che
`min(cap, len(gruppo))` produce da solo, senza logica dedicata) e quali
lo superano di ordini di grandezza (dove serve davvero merito+diversità).
Il costo di scansione resta lo stesso indipendentemente da questa
distinzione — cambia solo, a valle, quanti oggetti `EventCandidate`
vengono infine costruiti per stratum.

## 9. Prototipo su DAAX — risultati

Implementato e testato sullo stesso pool DAAX 1H (3133 candidati, cache
riusata per evitare di ripetere la discovery/grading da 5 minuti):
scansione esaustiva completata in 434s, 112.122 coppie gate-passanti —
coerente con le misure precedenti. Per ogni stratum "campionato" (tutti
tranne `B_same`, che qui non ha gate-passer), prima di applicare il
punteggio di merito si raggruppano le coppie per `source_pair` (le due
feature sorgente coinvolte) e si tiene **un solo rappresentante — il
migliore per punteggio — per ciascuna combinazione distinta**; solo dopo
si prendono le top-100 tra questi rappresentanti.

| Stratum | Gate-passer | Source-pair distinte | Tenute | Source-pair tenute |
|---|---|---|---|---|
| `B_C` | 3 | 2 | 2 | 2 (100%) |
| `C_D` | 4.424 | 476 | 100 | 100 (100%) |
| `C_same` | 472 | 74 | 74 (tutte — sotto cap) | 74 (100%) |
| `D_same` | 107.223 | 2.234 | 100 | 100 (100%) |

**Totale**: 276 composizioni materializzate, **231 combinazioni di
feature sorgente distinte (83.7%)** — contro le 49/303 (16.2%) del
punteggio di merito "nudo" testato al §6. Il passo di deduplicazione
per `source_pair` prima del ranking risolve quasi del tutto il problema
di concentrazione osservato in precedenza, mantenendo la qualità:

- tpm mediano 25.43 (target 24.00 — vicino quanto la versione senza
  deduplicazione, 26.89)
- dispersione mediana 0.673 (in linea con 0.642 di prima, ben sotto la
  soglia effettiva 1.706)
- n_episodi mediano 1602

**La famiglia di feature della regola originale (§1) sopravvive alla
nuova selezione**: la combinazione `ratio_close_ret03_ret06` (soglia
diretta) × `ratio_close_ret03_ret06` (stessa feature, trasformata in
percentile-rank a finestra 96 — la stessa coppia dietro `h*=192`
dell'analisi precedente) compare **sia in `C_D` che in `C_same`**, con
punteggi di merito tra i più alti dei rispettivi strati (0.663 e 0.622).
Non è garantito che sopravviva sempre — dipende dai suoi numeri relativi
agli altri candidati del pool in quella specifica run — ma qui lo fa,
ed è esattamente il comportamento cercato: una regola con buone
statistiche non viene più persa per sfortuna d'ordine.

**Effetto collaterale positivo non pianificato**: per `C_same` (474
gate-passer, 74 combinazioni distinte) la deduplicazione fa sì che
*tutte* le combinazioni vengano tenute automaticamente, perché sono meno
del cap — lo stesso "esaustivo emergente" già visto per gli strati
A/B al §8, qui ottenuto senza bisogno di codificare esplicitamente
`C_same` come caso speciale.

**Conclusione**: la politica a due fasce + deduplicazione per feature
sorgente è pronta per un'implementazione reale in `grade_pairing.py`.
Il costo aggiuntivo osservato (~7.2 minuti sulla scansione esaustiva,
invariato rispetto alle misure precedenti) è indipendente dalla
politica di selezione a valle, che nella pratica costa frazioni di
secondo (operazioni pandas su ~112mila righe già in memoria).


