# FORGE — Deployment Module
> Non è un modulo della pipeline (M0-M4) e non è `forgedge.playground`: è lo
> strato che porta una regola **già scoperta e promossa** da `forge()` verso
> la produzione — decidere se è abbastanza solida per andare live, scriverla
> su disco in un formato replicabile, e indicizzarla per un job di
> monitoraggio periodico. A differenza di `forgedge.playground`, questo
> modulo **ha effetti reali**: `promotion_gate()` decide cosa può essere
> promosso, `export_rules()` scrive file sul filesystem.
>
> Nato dentro `forgedge.playground` (issue #245), spostato nel proprio
> pacchetto top-level da PR #247 proprio perché "playground" smetteva di
> descrivere onestamente un modulo con effetti reali — vedi
> `docs/modules/Playground.md` §10 per la storia completa dello spostamento.
>
> Questo documento descrive **perché** il modulo è progettato così e
> **come funzionano internamente** le sue funzioni. Per la guida
> all'utilizzo — firme, parametri, esempi eseguibili, output verificato —
> vedi `docs/specs/deployment_en.md` (`deployment_it.md` per l'italiano).

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Principi di design](#2-principi-di-design)
3. [Logica interna — la sequenza produttiva](#3-logica-interna--la-sequenza-produttiva)
4. [`PromotionGateConfig` — le sei leve](#4-promotiongateconfig--le-sei-leve)
5. [Stabilità e garanzie](#5-stabilità-e-garanzie)

---

## 1. Posizionamento e Responsabilità

```
              forge() / forge_multi()  (+ forge_multi()'s pooled RuleRegistry)
                          │
                          ▼
              list[ForgeResult] (+ list[RuleRegistry], opzionale)
                          │
                          ▼
              ┌───────────────────────────────────────────┐
              │            forgedge.deployment              │
              │                                             │
              │  promotion_gate()   → pd.DataFrame (flag)   │
              │        │                                    │
              │        ▼                                    │
              │  export_rules()     → .pkl + .yaml su disco │
              │        │                                    │
              │        ▼                                    │
              │  monitoring_manifest() → pd.DataFrame        │
              └───────────────────────────────────────────┘
                          │
                          ▼
        job di monitoraggio periodico: RuleDiscovery su candele fresche
        (mai AlphaDiscovery — vedi manuale §9, pattern 5)
```

**Domanda a cui risponde questo modulo:** non "quali regole sono statisticamente valide" (a quello rispondono M2/M3 con `alpha_score`/`verdict`) ma *"quali di queste regole già valide sono abbastanza solide da esporre a un sistema di esecuzione esterno, e come si tiene un registro replicabile di cosa è stato esposto?"* — una domanda operativa, non statistica, che si pone solo dopo che `forge()` ha già finito di girare.

**Confine netto rispetto a `forgedge.playground`:** entrambi i moduli leggono `ForgeResult` senza mai rilanciare la pipeline, ma solo `forgedge.deployment` ha effetti osservabili all'esterno del processo Python che lo chiama — un file scritto su disco, una decisione che (in un sistema a valle) determina se una regola viene eseguita con soldi veri. `promotion_gate()` e `monitoring_manifest()` restano puri (nessun I/O); `export_rules()` è l'unica funzione dell'intera libreria — playground incluso — il cui unico scopo è un side effect deliberato.

`from forgedge.deployment import *` è l'import previsto (`forgedge/deployment/__init__.py` definisce esplicitamente `__all__`, quattro nomi: `PromotionGateConfig`, `promotion_gate`, `export_rules`, `monitoring_manifest`); come `forgedge.playground`, non è ri-esportato dal pacchetto top-level `forgedge`. Tutto il codice vive in un unico file, `forgedge/deployment/rules.py` — a differenza di `playground/`, non c'è bisogno di separare per modulo di origine dei dati, perché le tre funzioni condividono lo stesso identico input (`ForgeResult.rule_responses` + `RuleRegistry.documents`, opzionale) e la stessa sequenza logica.

---

## 2. Principi di design

1. **Le tre funzioni sono pensate per girare in sequenza, mai isolate.** `forge() → promotion_gate() [filtra] → export_rules() [scrive, sulle sole regole promosse] → monitoring_manifest() [indicizza l'export]`. `export_rules()` non è un wrapper leggero attorno a `promotion_gate()`: **ricalcola internamente la stessa identica logica di gate** (stesso `_compute_rows`/`_promotable_mask`) invece di richiedere in input il `pd.DataFrame` che `promotion_gate()` ha già prodotto — scelta deliberata per garantire che le due funzioni non possano mai disaccordare su cosa sia promuovibile, al costo di ricalcolare il filtro due volte se un chiamante le invoca entrambe.
2. **Ogni flag è sempre calcolato e sempre riportato, indipendentemente da cosa `PromotionGateConfig` blocca davvero.** `rotation_only`, `is_duplicate`, `is_isolated`, `consistency`, `fold_stability_score` compaiono come colonne del `DataFrame` anche quando il corrispondente `block_*`/`require_consistency`/`min_fold_stability_score` non blocca nulla (rispettivamente `False` o `None`) — solo `promotable`, la colonna finale, riflette la configurazione. Questo separa nettamente "cosa osserviamo" da "cosa decidiamo di bloccare oggi": disattivare un controllo non fa sparire il dato che avrebbe prodotto, così un audit successivo può sempre chiedersi "e se avessimo bloccato anche X?" senza dover rieseguire nulla.
3. **`registries` è sempre opzionale, mai un requisito silenzioso.** `is_duplicate`/`is_isolated` richiedono un `RuleRegistry` per essere calcolati (vengono letti da `RuleDocument`, non da `ForgeResult`); quando `registries=None` quelle due colonne restano semplicemente `None` per ogni riga, e i corrispondenti `block_duplicate`/`block_isolated` non bloccano mai nulla (un `None` non soddisfa mai la condizione di blocco). Un chiamante con un solo `forge()` single-ticker senza registro esplicito ottiene comunque un `DataFrame` utilizzabile, solo con due colonne meno informative — mai un errore per l'assenza di un input opzionale.
4. **Solo `export_rules()` tocca il filesystem, ed è isolato apposta.** Le altre due funzioni sono `Iterable[...] -> pd.DataFrame` pure, testabili senza `tmp_path`/mock — lo stesso principio "nessuna dipendenza aggiuntiva, nessun effetto collaterale non dichiarato" di `forgedge.playground`, applicato qui a due funzioni su tre invece che a tutte.

---

## 3. Logica interna — la sequenza produttiva

### Helper condiviso: `_compute_rows`

Sia `promotion_gate` sia `export_rules` passano prima da `_compute_rows(results, registries)`, che itera `result.rule_responses` per ogni `ForgeResult`, tiene solo i contratti con `response.is_edge` (cioè `EDGE`/`PARTIAL-EDGE` — `NON-EDGE`/`INSUFFICIENT-DATA` non arrivano mai a questo modulo), e calcola per ciascuno:

- **`rotation_only`** — stessa identica logica di `forgedge.playground.lottery_only_winners` (§6 di `Playground.md`): vero se `rejection_reasons` ha esattamente un elemento e inizia con `"search-level rotation null not cleared"`. La duplicazione è intenzionale — questo modulo non importa da `forgedge.playground` per restare architetturalmente indipendente dal suo sibling, anche a costo di ripetere ~5 righe di logica.
- **`is_duplicate`/`is_isolated`** — letti da `RuleDocument` via un indice `{alpha_id: RuleDocument}` costruito da `_document_index(registries)`; `None` per entrambi se nessun `RuleDocument` risolve o se `registries=None`.
- **`consistency`** — `response.walk_forward.consistency` se `walk_forward` non è `None`, altrimenti `None`.

Il dizionario per riga porta anche `_contract`/`_response`/`_candidate` (gli oggetti live, non solo i loro campi scalari) — colonne "private" che `promotion_gate()` scarta prima di restituire il `DataFrame` ma che `export_rules()` usa direttamente per scrivere i file senza dover ricalcolare o ripescare nulla.

### `promotion_gate`

Applica `_promotable_mask` (vedi §4) alle righe di `_compute_rows` e restituisce le sole colonne pubbliche più `promotable`. Nessun I/O, nessuno stato — una funzione pura di reporting/filtro.

### `export_rules`

Ricalcola `_compute_rows`/`_promotable_mask` (stessa configurazione, stesso identico algoritmo di `promotion_gate` — vedi principio di design #1), poi per ogni riga selezionata (promuovibile, o tutte se `promotable_only=False`) scrive due file in `output_dir`:

- **`{alpha_id}.pkl`** — l'intero oggetto `EventCandidate` serializzato via `pickle`, non solo l'espressione testuale: porta con sé la funzione di attivazione deterministica (`EventCandidate.apply`), quindi rieseguire l'evento su dati futuri non richiede di ricostruire nulla a mano.
- **`{alpha_id}.yaml`** — `ValidatedRule.to_dict()` (direzione, entry mode, parametri buy/sell, holding period, fee — il punto operativo pubblicato) più `ticker`/`alpha_id`/`verdict` per contesto. Scritto con un writer YAML minimale scritto ad hoc (`_dump_yaml_mapping`/`_yaml_scalar`), non con una libreria esterna: ogni valore che un `ValidatedRule`/manifest porta è uno scalare piatto (str/int/float/bool/None), quindi round-trippare correttamente non richiede la dipendenza di un parser YAML completo — coerente con la scelta dell'intera libreria di restare su sole `numpy`/`pandas` come dipendenze runtime.

Una riga il cui `_candidate` non risolve o il cui `response.validated_rule` è `None` viene saltata silenziosamente (nessun file scritto, nessuna eccezione) — coerente con lo stile "salta, non sollevare" del sibling `forgedge.playground`, anche se qui la conseguenza di uno skip è "un file in meno sul disco", non "una riga in meno in un DataFrame".

### `monitoring_manifest`

Applica `RuleSpec.from_forge_result` (già esistente in `forgedge.rule_report`, usata anche da `rule_performance_report()` — manuale §9, pattern 5) a ogni `ForgeResult` di R, appiattendo il risultato in un unico `DataFrame` con `event_candidate_id`, `is_end`, `verdict`, `oos_expectancy` per ogni regola tradeable. Il join naturale con l'output di `export_rules()` è su `event_candidate_id`, per restringere il manifest alle sole regole effettivamente esportate su disco.

*Perché esiste, come motivazione di design:* un job di monitoraggio periodico deve rigiocare ogni regola pubblicata su candele fresche via `RuleDiscovery` (mai `AlphaDiscovery` — manuale §9, pattern 5, e §21 per il perché) senza dover ricostruire a mano il riferimento a ciascuna. Questa funzione produce l'indice pronto, la stessa lista che un cron/scheduler esterno leggerebbe per sapere cosa ricontrollare.

---

## 4. `PromotionGateConfig` — le sei leve

Dataclass con sei campi di gating (più un settimo, `fold_pf_cap`, che non blocca nulla da solo ma calibra uno di essi), ciascuno indipendente dagli altri (nessuna combinazione è mutuamente esclusiva):

| Campo | Default | Effetto |
|---|---|---|
| `min_consistency` | `0.5` | Soglia su `consistency` — la stessa soglia che la pipeline usa internamente per un verdetto positivo, non un numero nuovo inventato per questo modulo. |
| `require_consistency` | `True` | Se `False`, `min_consistency` smette di partecipare a `promotable` (la colonna `consistency` resta comunque popolata). |
| `block_rotation_only` | `False` | Se `True`, blocca un `PARTIAL-EDGE` il cui unico ostacolo era il rotation null. Default `False` perché — come `forgedge.playground.lottery_only_winners` documenta — un rotation-only miss è tipicamente un compromesso accettabile, non un segnale di debolezza reale. |
| `block_duplicate` | `True` | Blocca una regola che il Rule Registry ha marcato `is_duplicate=True`. |
| `block_isolated` | `True` | Blocca una regola classificata `"ISOLATED"` sul replay cross-ticker. Nessun effetto (`is_isolated` resta `None`) se `registries` non è stato passato. |
| `min_fold_stability_score` | `None` | Soglia sul `fold_stability_score` (#253): `mean(fold_pf) - std(fold_pf)` sui fold walk-forward di M3. `None` per default — il gate è spento finché non lo si attiva esplicitamente. |
| `fold_pf_cap` | `10.0` | Non è un gate: è il cap applicato a ciascun `test_summary.profit_factor` di fold prima di calcolare `fold_stability_score`, per evitare che il sentinel `9999.0` "zero perdite" domini media e deviazione standard. |

*Perché questi campi e non altri:* i primi cinque corrispondono esattamente a un segnale che uno degli altri playground/moduli della pipeline già calcola (`lottery_only_winners`, `duplicate_clusters`, `classification_by_grade`, `RuleDiscoveryResponse.walk_forward.consistency`) — fino a #253, `PromotionGateConfig` non introduceva alcuna nuova soglia statistica, ma combinava soltanto giudizi già esistenti in una singola decisione binaria di andare o non andare in produzione. `min_fold_stability_score` è la prima eccezione deliberata: nessun altro modulo calcola già la dispersione fold-per-fold del profit factor walk-forward, quindi questo è un segnale nuovo, motivato dal caso concreto in #253 (una regola con PF aggregato alto ma un solo fold "fortunato" che collassa su dati OOS freschi). I due default `True` (`block_duplicate`, `block_isolated`) restano la scelta conservativa: bloccare per default ciò che la pipeline stessa ha già segnalato come ridondante o non generalizzabile; `min_fold_stability_score` resta invece `None` per default — a differenza di quelli, non c'è ancora una soglia "giusta" nota a priori, quindi va scelta esplicitamente da chi la usa.

---

## 5. Stabilità e garanzie

- **Non fa parte dell'API core.** Come `forgedge.playground`, è uno strato costruito sopra `forge()`/`RuleRegistry`, non un componente M0-M4.
- **Effetti reali, dichiarati esplicitamente.** A differenza di ogni altra funzione diagnostica della libreria, `export_rules()` scrive su disco per design — questo è il motivo dell'intera esistenza di questo modulo separato da `forgedge.playground` (§1). `promotion_gate()`/`monitoring_manifest()` restano pure.
- **Nessuna dipendenza aggiuntiva.** Il writer YAML è scritto ad hoc proprio per evitare di introdurre una dipendenza runtime che il resto della libreria non ha (§3).
- **Nessun cambio di comportamento dallo spostamento da `forgedge.playground`.** Stesse firme, stessi default, stesso algoritmo — solo il percorso di import è cambiato (issue #245, PR #247; vedi `docs/modules/Playground.md` §10).
- Per il comportamento esatto verificato contro il codice — firme, parametri, esempi eseguibili — vedi sempre `docs/specs/deployment_en.md`/`deployment_it.md`, non questo documento: questo descrive il design, quello descrive l'API.
