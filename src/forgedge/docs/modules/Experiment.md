# FORGE — Experiment Module
> Non è un modulo della pipeline (M0-M4) e non è né `forgedge.playground` né
> `forgedge.deployment`: è un terzo tipo di modulo gemello che **esegue una
> propria pipeline multi-fase sopra `forge()`**, invece di limitarsi a
> leggerne i risultati (playground) o ad agire su di essi (deployment).
> `StepWiseDiscovery`, la sua prima classe, risponde a una domanda più
> esplorativa di una singola chiamata a `forge()`: partendo dalle regole a
> condizione singola più forti e confermate sull'hold-out, farle crescere
> una condizione AND alla volta cercando solo nella sotto-popolazione che
> ciascuna già seleziona, invece dell'unico passaggio globale che `forge()`
> fa su tutti i candidati insieme.
>
> Questo documento descrive **perché** il modulo è progettato così e
> **come funziona internamente** il suo algoritmo. Per la guida
> all'utilizzo — firme, parametri, esempi eseguibili, output verificato —
> vedi `docs/specs/experiment_en.md` (`experiment_it.md` per l'italiano).

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Principi di design](#2-principi-di-design)
3. [Logica interna — l'algoritmo passo per passo](#3-logica-interna--lalgoritmo-passo-per-passo)
4. [`StepWiseDiscoveryConfig` — le leve](#4-stepwisediscoveryconfig--le-leve)
5. [Provenienza: la ricerca manuale che ha preceduto il modulo](#5-provenienza-la-ricerca-manuale-che-ha-preceduto-il-modulo)
6. [Stabilità e garanzie](#6-stabilità-e-garanzie)

---

## 1. Posizionamento e Responsabilità

```
                     KPI Table (di sola forma, non di valore, per il
                     dimensionamento dello split — vedi §3.1)
                                │
                                ▼
              ┌─────────────────────────────────────────────┐
              │           forgedge.experiment                 │
              │                                                │
              │  1. risolvi le config UNA VOLTA (§3.1)         │
              │  2. split esterno search_df / holdout_df       │
              │  3. iterazione 1: un forge() su search_df      │
              │  4. selezione semi (diversi, confermati)       │
              │  5. per ogni seme: partition & compose         │
              │     (EventDiscovery/AlphaDiscovery/            │
              │      RuleDiscovery diretti, config condivise)  │
              └──────────────────┬─────────────────────────────┘
                                │
                                ▼
                    StepWiseDiscoveryResult
                    (una ChainResult per seme + audit dei semi
                     scartati + il DataFrame di riepilogo)
```

**Domanda a cui risponde questo modulo:** non "quali regole a singola condizione sono valide" (a quello risponde già `forge()`) ma *"una di quelle regole regge anche una seconda condizione, se cerco quella seconda condizione solo nella sotto-popolazione dove la prima è già attiva?"* — una domanda che nessuna singola chiamata a `forge()` pone, perché il suo compositore a due passate (issue #254) accoppia candidati sull'intero pool globale in base al grade, non seme per seme sulla partizione attiva di ciascuno.

**Confine netto rispetto a `forgedge.playground`/`forgedge.deployment`:** entrambi quei moduli non rilanciano mai la pipeline — leggono (playground) o agiscono su (deployment) un `ForgeResult` già prodotto. `forgedge.experiment` fa l'opposto: il suo intero scopo è orchestrare **nuove** chiamate a `forge()`/`EventDiscovery`/`AlphaDiscovery`/`RuleDiscovery`, secondo un algoritmo che nessuna di quelle chiamate da sola implementa. Non è nemmeno un preset (`forgedge.presets`) — un preset sceglie solo dei numeri per una singola esecuzione della pipeline; questo modulo orchestra *più* esecuzioni in una sequenza con stato (semi, catene, hold-out condiviso).

`from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig` è l'import previsto; come `playground`/`deployment`, il modulo **non** è ri-esportato dal pacchetto top-level `forgedge`. Il codice sorgente segue il pattern standard dei moduli della libreria: `models.py` (le dataclass di configurazione e risultato), `redundancy.py` (le funzioni pure di controllo ridondanza, riusabili e testabili senza alcuna pipeline), `step_wise_discovery.py` (l'orchestrazione).

---

## 2. Principi di design

1. **Generico per costruzione, non per un asset o timeframe specifico.** `StepWiseDiscoveryConfig` non ha alcun campo che nomini un asset, una valuta o un timeframe — ogni calibrazione economica (fee, `mfe_floor`, quali periodi indicatore sono entrati nella KPI table) resta responsabilità del chiamante, passata già risolta tramite `event_discovery_config`/`alpha_config`/`rule_discovery_config`. Questo modulo possiede solo l'algoritmo di ricerca, mai la conoscenza di dominio necessaria a calibrarlo per un asset particolare — la stessa separazione che la libreria intera mantiene tra "come si cerca" (M1-M3) e "cosa conviene cercare per questo mercato" (compito del chiamante).
2. **Le configurazioni si risolvono una volta sola, mai una seconda volta a metà ricerca.** Un pitfall documentato per l'uso standalone di `EventDiscovery`/`AlphaDiscovery` fuori da `forge()` è che un campo che conta le barre (`horizon_grid` e simili), se lasciato `UNSET`, ricade silenziosamente sulla calibrazione oraria indipendentemente dal `timeframe` reale — perché la resa esplicita di quella calibrazione avviene solo dentro il `PipelineContext` che `forge()` costruisce internamente. `StepWiseDiscovery` risolve le configurazioni una volta, all'inizio (`config_report()`, §3.1), e ogni chiamata successiva a `EventDiscovery`/`AlphaDiscovery`/`RuleDiscovery` nel loop di profondità riusa quelle stesse configurazioni già risolte — mai una nuova istanza parzialmente `UNSET`.
3. **Un hold-out esterno, mai visto durante la ricerca, distinto da qualunque split interno.** Le configurazioni del chiamante possono già avere un proprio `train_ratio`/walk-forward interno (per M1 o M3) — questo modulo aggiunge un secondo taglio, più esterno, che nessuna chiamata di discovery vede mai: solo `_confirms_on_holdout` lo legge, e solo per re-valutare un candidato già scelto, mai per sceglierlo.
4. **La ridondanza si controlla con lo stesso meccanismo in due punti diversi.** Un candidato può essere ridondante con un componente già scelto in due modi che la sola uguaglianza di nome non cattura: stesso insieme di attivazione booleano (Jaccard) o stessa informazione continua sottostante nonostante soglie diverse (correlazione). Lo stesso `is_redundant()` (`redundancy.py`) applica entrambi i controlli sia *dentro* una catena (un figlio non deve ripetere un componente del padre) sia *tra* i semi selezionati all'inizio (un secondo seme non deve essere la stessa informazione di un primo sotto un nome diverso) — un'unica implementazione, due punti di chiamata, non due controlli scritti separatamente che potrebbero divergere.
5. **Un tentativo fallito non sovrascrive mai uno stato genuinamente confermato.** Il risultato di ogni catena riporta l'ULTIMO stato che ha superato `_confirms_on_holdout` — il seme da solo, o la composizione più profonda che ha effettivamente confermato — mai un tentativo di composizione più profondo il cui verdetto sull'hold-out è risultato negativo. Questo principio è nato da un bug osservato durante lo sviluppo (§5): senza di esso, un seme genuinamente valido può sparire dal risultato finale, sostituito silenziosamente da un tentativo di composizione fallito.

---

## 3. Logica interna — l'algoritmo passo per passo

### 3.1 Risoluzione delle configurazioni

`_resolve()` chiama `forgedge.config_report(disc_cfg, alpha_cfg, rd_cfg, kpi=self.kpi, timeframe=self.timeframe)` **una sola volta**, passando l'intera KPI table. Questo non introduce look-ahead: `config_report()` è cieco ai valori dei dati per costruzione (legge solo `len(kpi)` e lo span temporale, mai una colonna) — esattamente la stessa garanzia che permette a `forge()` di validare la coerenza della configurazione prima ancora di eseguire Modulo 0. `event_discovery_config.max_and_components` viene forzato a `1` indipendentemente da cosa passa il chiamante: la composizione è compito esclusivo di questo modulo, mai di M1.

### 3.2 Split esterno

`_split_holdout()` costruisce un `TimeBudget` con `horizon_bars`/`embargo_bars` derivati, di default, dallo stesso `alpha_config.horizon_grid`/`embargo_bars` **risolto** al passo precedente — la stessa quantità che `forge()` userebbe internamente per il proprio budget (`StepWiseDiscoveryConfig.outer_horizon_bars`/`outer_embargo_bars`, entrambi sovrascrivibili). `search_df` è tutto ciò che precede lo split; `holdout_df` tutto ciò che segue l'embargo — nessuna chiamata di discovery in questo modulo vede mai `holdout_df` prima di `_confirms_on_holdout`.

### 3.3 Iterazione 1

Una singola chiamata a `forge(search_df, ..., two_pass_composition=False, run_registry=False)`, con `event_discovery_config` già forzato a `max_and_components=1` (§3.1). `two_pass_composition=False` è necessario anche con `max_and_components=1`: altrimenti `forge()` eseguirebbe comunque la propria composizione a due passate guidata dal grade (issue #254) dopo Modulo 2, vanificando lo scopo di un'iterazione 1 a condizione singola pulita. `result1.candidates`/`.rule_responses` alimentano direttamente la selezione dei semi.

### 3.4 Precomputo della transform layer

Prima della selezione dei semi (non dopo, a differenza di una prima versione interna dello script di ricerca da cui questo modulo deriva — vedi §5), `_precompute_transforms()` calcola `TypeClassifier`/`FeatureGenerator`/`TransformLayer` **una volta** sull'intera `search_df`, producendo `_series_by_col` — un dizionario `{nome_colonna: serie_continua}` che copre sia le feature identity sia quelle a trasformazione rolling (pctrank/zscore/delta). Serve a due cose: recuperare i figli rolling in modo corretto rispetto al look-ahead (§3.6) e calcolare la correlazione continua sia nella selezione dei semi sia dentro ogni catena — spostato prima della selezione dei semi proprio perché quest'ultima ne ha bisogno.

### 3.5 Selezione dei semi

Per ogni contratto classificato (EDGE/PARTIAL-EDGE prima, poi per `composite_score` decrescente): salta se la famiglia (`family_key`, periodi numerici rimossi dal nome) è già stata usata; salta se `is_redundant()` rispetto a ogni seme già accettato eccede la soglia di Jaccard o di correlazione continua (controllo fatto PRIMA della costosa `_confirms_on_holdout`, per non sprecare una RuleDiscovery su un candidato che verrebbe comunque scartato); altrimenti prova `_confirms_on_holdout` e accetta solo se conferma. Continua finché `config.n_seeds` sono stati accettati o i candidati classificati sono esauriti.

### 3.6 Il loop di profondità, per seme

Ad ogni profondità: calcola `retain_ratio_min_k` (il floor adattivo — §4) dalla dimensione reale della partizione corrente; genera figli puntuali con un `EventDiscovery` costruito direttamente sulla partizione (sicuro: non dipende dalla contiguità delle righe); genera figli a trasformazione rolling con `_rolling_children_on_partition` — che non ricalcola mai la trasformazione sulla partizione, la legge da `_series_by_col` (già calcolata sull'intera `search_df`, §3.4) e usa la partizione solo per calibrare la soglia locale, evitando che "N barre fa" diventi silenziosamente "N righe attive fa" su una partizione non contigua; scarta ogni candidato ridondante (nome, Jaccard, correlazione continua) con un componente già nella catena; valuta i sopravvissuti con `AlphaDiscovery`+`RuleDiscovery` su `search_df` intera (mai sulla partizione); tra i confermati sull'hold-out, prova a comporre col padre in ordine di classifica finché uno produce un target derivabile da M2 con almeno `min_composed_activations` attivazioni; committa il nuovo stato SOLO se la composizione stessa riconferma sull'hold-out (§2, principio 5) — altrimenti la catena si ferma riportando l'ultimo stato confermato.

---

## 4. `StepWiseDiscoveryConfig` — le leve

| Campo | Default | Effetto |
|---|---|---|
| `n_seeds` | `3` | Numero di semi distinti da cui far partire una catena indipendente. |
| `depth` | `2` | Profondità massima di composizione per catena. |
| `train_ratio` | `0.80` | Frazione della KPI table riservata allo split esterno — indipendente da qualunque split interno delle configurazioni passate. |
| `outer_horizon_bars` / `outer_embargo_bars` | `None` / `None` | Purge/embargo dello split esterno. `None` li deriva da `alpha_config.horizon_grid`/`embargo_bars` risolti — la stessa quantità che userebbe `forge()` stesso; un valore esplicito ha sempre priorità (es. una scala derivata dalla semivita OU, se disponibile). |
| `retain_ratio_floor` | `0.2` | Floor assoluto su `retain_ratio_min_k = max(floor, min_trades_M3 / n_righe_partizione)`, ricalcolato ad ogni profondità dalla dimensione reale della partizione corrente — sostituisce un floor fisso di righe con uno che si adatta al floor statistico di M3 (`rule_discovery_config.criteria.min_oos_trades`). |
| `max_constituent_jaccard` | `0.85` | Stessa soglia di default di `forgedge.event_discovery.diversity_gate`/`ANDComposer(max_constituent_jaccard=...)` — vedi `forgedge.experiment.redundancy`. |
| `max_constituent_abs_corr` | `0.95` | Deliberatamente più severa della soglia Jaccard — deve catturare solo quasi-duplicati, non feature semplicemente correlate ma genuinamente distinte. |
| `child_gate` | `None` | Consistency Gate applicato ai figli valutati su una partizione. `None` usa un gate deliberatamente più permissivo di quello tipico di sessione (`min_tpm=0.25, dispersion_margin=3.0, min_episodes=4`) — una ricerca su sotto-popolazione richiede un gate locale permissivo, ma non così permissivo da affamare di attivazioni la regola composta prima ancora che Alpha Discovery possa derivarne un target. |
| `min_composed_activations` | `20` | Un candidato padre-AND-figlio sotto questa soglia di attivazioni viene scartato prima ancora di provare Alpha Discovery, con una ragione esplicita invece di un opaco "no derivable target" di M2. |
| `min_local_partition_rows` | `10` | Un figlio a trasformazione rolling la cui serie locale (ristretta alla partizione) ha meno di queste righe non-NaN viene saltato — troppo pochi dati per calibrare una soglia. |
| `strict` | `True` | Passato a `forge()`/`config_report()` — solleva su un'incoerenza di livello `FAIL` invece di procedere silenziosamente (stesso invariante di `forge(strict=True)`). |

---

## 5. Provenienza: la ricerca manuale che ha preceduto il modulo

`StepWiseDiscovery` generalizza uno script di ricerca "partition & compose" sviluppato iterativamente, a mano, su più asset (EURUSD, COPPER.CMDUSD, DAAX, ADAUSDC) a timeframe diversi (1D, 1H) prima di essere promosso a modulo. Ogni principio del §2 corrisponde a un bug osservato concretamente durante quello sviluppo, non a una scelta teorica:

- **L'esclusione totale delle trasformazioni rolling dalla ricerca locale** era la scelta iniziale (per evitare esattamente il bug di look-ahead che §3.6 descrive) — un controllo con il compositore nativo a due passate di `forge()` sugli stessi dati ha mostrato che *ogni* regola composta che quel compositore trovava usava esattamente le trasformazioni rolling escluse. Il fix (calcolare la trasformazione sulla serie continua intera, calibrare solo la soglia sulla partizione) è il meccanismo di `_rolling_children_on_partition` oggi.
- **La ridondanza tra semi** (principio 4, §2) non era controllata nella prima versione: due semi costruiti su feature diverse per nome ma quasi identiche per informazione (un rapporto su ATR e il suo analogo su NATR = ATR/close) venivano entrambi accettati come "famiglie distinte" e convergevano poi sulla stessa identica regola composta finale — uno spreco di un intero seme, non una seconda dimensione genuina.
- **Il bug di reporting che ha motivato il principio 5** (§2): in una run osservata, tre semi genuinamente confermati `PARTIAL-EDGE` sono stati sovrascritti da tentativi di composizione con `composite_score` di ricerca fino a 1.000 ma 0% di generalizzazione sull'hold-out, perché lo stato della catena veniva aggiornato PRIMA di verificare `_confirms_on_holdout` sulla composizione, non dopo.

Questa provenienza non è un dettaglio storico isolato: è la ragione per cui ogni principio del §2 esiste nella forma in cui esiste, e per cui i controlli di ridondanza (Jaccard + correlazione continua, non solo per nome) sono stati validati specificamente contro casi reali osservati su questi asset, non ipotizzati a tavolino.

---

## 6. Stabilità e garanzie

- **Non fa parte dell'API core.** Come `forgedge.playground`/`forgedge.deployment`, è uno strato costruito sopra `forge()`, non un componente M0-M4.
- **Nessuna conoscenza di dominio incorporata.** Nessun default di `StepWiseDiscoveryConfig` dipende da un asset o timeframe specifico (§2, principio 1) — la generalità è un requisito di design esplicito, non un effetto collaterale.
- **Config risolte una sola volta, mai ricostruite a metà ricerca.** §2, principio 2 e §3.1 — ogni `EventDiscovery`/`AlphaDiscovery`/`RuleDiscovery` che questo modulo costruisce direttamente condivide le stesse configurazioni risolte dall'iterazione 1.
- **Nessuna dipendenza aggiuntiva.** Come il resto della libreria, solo `numpy`/`pandas`; usa solo API pubbliche e alcuni helper interni già esistenti di `forgedge.event_discovery` (`_apply_component`, `TypeClassifier`, `FeatureGenerator`, `TransformLayer`, `EventGenerator`), mai una reimplementazione parallela.
- Per il comportamento esatto verificato contro il codice — firme, parametri, un esempio eseguibile end-to-end — vedi sempre `docs/specs/experiment_en.md`/`experiment_it.md`, non questo documento: questo descrive il design, quello descrive l'API.
