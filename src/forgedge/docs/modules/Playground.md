# FORGE — Playground Module
> Non è un modulo della pipeline (M0-M4): è un livello di analisi che gira
> **dopo**, sopra l'insieme di risultati che una sessione di ricerca produce.
> `forgedge.playground` raccoglie funzioni di sola lettura che prendono **R**
> — uno o più `ForgeResult` (`forge()`/`forge_multi()`) — e rispondono a
> domande trasversali sul comportamento della pipeline stessa: dove il
> funnel perde candidati, quali famiglie di feature non si orientano mai,
> quanto sono "nervosi" i confini di regime. Non rilancia mai `forge()`,
> non modifica nulla che gli viene passato, e non fa parte dell'API core:
> è tracciato da una checklist ora completa (issue #237, 11/11 casi d'uso),
> ma resta uno strumento diagnostico, non uno strato stabile al livello di
> `forge()` o `RuleDiscovery`.
>
> Le funzioni che portano una regola scoperta **in produzione** (gate di
> promozione, export su disco, manifest di monitoraggio) non vivono più qui:
> sono state spostate nel modulo gemello `forgedge.deployment` (issue #245,
> spostamento in PR #247) perché hanno effetti reali — un "playground" che
> include la decisione di cosa va in produzione era un nome fuorviante. Vedi
> `docs/modules/Deployment.md`.
>
> Questo documento descrive **perché** il modulo è progettato così e
> **come funzionano internamente** le sue funzioni. Per la guida
> all'utilizzo — firme, parametri, esempi eseguibili, output verificato —
> vedi `docs/specs/playground_en.md` (`playground_it.md` per l'italiano).

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Principi di design](#2-principi-di-design)
3. [Logica interna — M0 Market Context](#3-logica-interna--m0-market-context)
4. [Logica interna — M1 Event Discovery](#4-logica-interna--m1-event-discovery)
5. [Logica interna — M2 Alpha Discovery](#5-logica-interna--m2-alpha-discovery)
6. [Logica interna — M3 Rule Discovery](#6-logica-interna--m3-rule-discovery)
7. [Logica interna — M4 Rule Registry](#7-logica-interna--m4-rule-registry)
8. [Logica interna — caso trasversale](#8-logica-interna--caso-trasversale)
9. [Roadmap — issue #237](#9-roadmap--issue-237)
10. [Il modulo gemello: `forgedge.deployment`](#10-il-modulo-gemello-forgedgedeployment)
11. [Stabilità e garanzie](#11-stabilità-e-garanzie)

---

## 1. Posizionamento e Responsabilità

```
                     forge() / forge_multi()  (una o più volte, anche nel tempo)
                                │
                                ▼
                     list[ForgeResult]  ("R" — l'insieme accumulato da una sessione)
                                │
                                ▼
              ┌─────────────────────────────────────────────┐
              │           forgedge.playground                │
              │  legge SOLO attributi già presenti su R —    │
              │  .enriched, .rule_responses, .candidates,     │
              │  .contracts, .ticker, …                       │
              │  non chiama mai forge()/RuleDiscovery/ecc.    │
              └──────────────────┬────────────────────────────┘
                                │
                                ▼
                pd.DataFrame long-format (una riga per osservazione elementare)
                                │
                                ▼
                groupby / aggregazione a valle, a cura del chiamante
```

**Domanda a cui risponde questo modulo, in generale:** *non* "cosa ha prodotto questa run" (a quello risponde `ForgeResult` stesso — vedi il manuale, §9) ma *"cosa dice il comportamento della pipeline, osservato su più run, sulla pipeline stessa?"* — dove il funnel perde candidati, quali confini di regime sono instabili, quali famiglie di feature Alpha Discovery non riesce mai a orientare. Sono domande che nessun singolo campo di un singolo `ForgeResult` può rispondere da solo, perché richiedono di mettere in pool più run e riformattare i loro artefatti in una forma comune.

**Confine netto:** `forgedge.playground` non tocca mai la pipeline core (M0-M4) e non ha side effect — ogni funzione è una pura trasformazione `Iterable[ForgeResult] -> pd.DataFrame`. Non è un modulo di discovery, non produce verdetti, non persiste nulla. È uno strato di analisi **sopra** ciò che la pipeline ha già prodotto — la stessa distinzione architetturale tracciata altrove nel repository tra i moduli M0-M4 (che producono artefatti formali) e gli strumenti diagnostici opt-in come `summary_report()`/`config_report()` (che leggono artefatti già prodotti senza mai influenzarli).

`from forgedge.playground import *` è l'import previsto (`forgedge/playground/__init__.py` definisce esplicitamente `__all__`); il modulo **non** è ri-esportato dal pacchetto top-level `forgedge` — va importato dal proprio sottopacchetto. Il codice sorgente è organizzato per modulo di origine dei dati che ciascuna funzione legge principalmente: `forgedge/playground/m0.py` (Market Context), `m1.py` (Event Discovery), `m2.py` (Alpha Discovery), `m3.py` (Rule Discovery), `m4.py` (Rule Registry — le due sole funzioni che prendono `RuleRegistry` invece di `ForgeResult`, vedi §7), più `funnel.py` per l'unico caso trasversale a tutti i moduli (§8).

---

## 2. Principi di design

Tre scelte, applicate a ogni funzione del modulo senza eccezioni:

1. **Input sempre `Iterable[ForgeResult]`.** Mai un singolo `ForgeResult` nudo, anche quando si analizza una sola run — così lo stesso codice chiamante scala naturalmente da un ticker a un'intera sessione multi-asset senza cambiare firma. Nessuna funzione rilancia `forge()`, `forge_multi()` o alcun modulo M0-M4: se il dato che serve non è già su `ForgeResult`, la funzione non esiste ancora (vedi §9, Roadmap; eccezione dichiarata: le due funzioni M4 del §7 prendono `RuleRegistry`, non `ForgeResult`).
2. **Output sempre in formato long.** Una riga per osservazione elementare — una transizione di regime, una ragione di rigetto, un componente di un evento — mai una tabella già aggregata. L'aggregazione (`groupby`, ranking, percentili) è responsabilità del chiamante, non della funzione: la stessa tabella lunga deve poter rispondere a domande diverse da quella che ha motivato la funzione, e nessuna funzione qui dentro decide a priori quale sia "la" statistica di sintesi giusta.
3. **"Salta, non sollevare" per gli input parziali — ma sempre dichiarato esplicitamente.** Una run senza colonna `regime` (Market Context disabilitato), un contratto il cui `event_candidate_id` non risolve più contro `result.candidates`, un `AlphaContract` senza `alpha_score` (mai gradato): nessuno di questi casi solleva un'eccezione, tutti vengono saltati silenziosamente. Questo è coerente con lo stile del resto della libreria (manuale §11: le funzioni di analisi qualitativa non sono validazioni dei dati), ma non è mai un default implicito senza documentazione — ogni funzione dichiara nel proprio docstring esattamente cosa salta e perché (vedi `docs/specs/playground_en.md` per l'elenco completo caso per caso).

Conseguenza pratica dei primi due principi: chiamare due volte la stessa funzione playground su due sottoinsiemi di `R` e concatenare i risultati (`pd.concat`) dà lo stesso risultato di chiamarla una volta sola sull'unione — le funzioni non mantengono stato tra le chiamate e non normalizzano nulla che dipenda dall'insieme completo.

---

## 3. Logica interna — M0 Market Context

Entrambe le funzioni di questa sezione (`regime_transitions`, `regime_time_share`, in `m0.py`) leggono `result.enriched` — la KPI Table dopo Market Context, con le colonne `regime`/`regime_stable` aggiunte — e saltano qualunque risultato il cui `enriched` non abbia una colonna `regime`.

**`regime_transitions` — algoritmo.** Un flip è definito come `regime[t] != regime[t-1]` con entrambi i valori non-NaN (una barra NaN che segue l'inizio della serie non genera mai una transizione fantasma). La lunghezza del run che precede ogni flip è calcolata con l'idioma standard "raggruppa per run consecutivo": un contatore cumulativo dei soli punti di cambiamento (`changed.cumsum()`) assegna un `run_id` a ogni tratto costante, poi `groupby(run_id).cumcount() + 1` dà la posizione (1-based) di ogni barra dentro il proprio run — la funzione legge quella posizione all'ultima barra del run precedente al flip. La risoluzione del timestamp segue una catena di fallback deliberata: `DatetimeIndex` se presente, altrimenti la colonna `open_dt`, altrimenti la posizione intera stessa — mai un errore, perché questa funzione è pensata per girare anche su frame costruiti a mano nei test o in un notebook esplorativo.

*Perché esiste, come motivazione di design:* un confine di regime "nervoso" — che flippa dopo un run di 1-2 barre invece di restare stabile per un tratto ragionevole — inquina silenziosamente qualunque evento M1 condizionato al regime a valle: se il classificatore oscilla rapidamente tra due etichette, le statistiche calcolate "durante il regime X" mescolano di fatto più regimi. Questo è esattamente il fenomeno che `regime_stable` (con la sua finestra `stable_window`, risolta dalla sessione — vedi il manuale §8, Market Context) è pensata per filtrare a valle; `regime_transitions` rende visibile *quanto* nervosismo c'è da filtrare, prima ancora di guardare `regime_stable`.

**`regime_time_share` — algoritmo.** Un semplice `value_counts(dropna=True)` sulla colonna `regime` di ciascun risultato, normalizzato sul totale delle sole barre classificate (le barre `NaN` — tipicamente il warm-up prima che il classificatore abbia abbastanza storia — sono escluse sia dal numeratore sia dal denominatore, non trattate come un regime a parte).

*Perché esiste, come motivazione di design:* un asset che vive per la stragrande maggioranza della propria storia in un solo regime è un candidato a "regole scoperte su di esso sembrano generiche ma sono in realtà regime-specifiche" — non c'è mai stato abbastanza di un regime alternativo presente nel campione per dimostrare il contrario. Questa funzione rende quel rischio misurabile prima di fidarsi della genericità di una regola scoperta su quell'asset.

---

## 4. Logica interna — M1 Event Discovery

Entrambe le funzioni di questa sezione vivono in `m1.py` e guardano al confine tra M1 e M2 da lati diversi: `dead_event_candidates` guarda a valle (cosa produce ogni candidato sopravvissuto al gate), `gate_survival_observed` guarda a monte (cosa succede *prima* del gate, a ogni candidato grezzo).

### `dead_event_candidates`

Per ogni `ForgeResult`, costruisce prima un indice `{event_candidate_id: [contratti]}` da `result.contracts`, poi itera su `result.candidates` — ogni `EventCandidate` che ha già superato il Consistency Gate — e classifica ciascuno in base a quanti contratti M2 ne sono derivati e quanti di quelli sono `direction="undetermined"`:

- **`"dead"`** — zero contratti: il candidato è sopravvissuto al gate ma non ha mai raggiunto Alpha Discovery con un target derivabile, o semplicemente non è mai stato scelto per la valutazione M2.
- **`"undetermined_only"`** — almeno un contratto, ma tutti `"undetermined"`: M2 ha provato a derivare un target ma non c'è mai riuscito per questo candidato specifico.
- **`"actionable"`** — almeno un contratto con una `direction` derivata: il candidato ha prodotto valore, indipendentemente dal verdetto che riceverà poi in M3.

*Perché esiste, come motivazione di design:* il Consistency Gate (manuale §8) è progettato per lasciar passare solo candidati con struttura temporale stabile — ma "stabile" non implica "utile". Un candidato può superare il gate e non produrre mai nulla di actionable a valle, per ragioni che il gate stesso non misura (es. l'evento non correla mai con un movimento di prezzo abbastanza pulito da derivare un target). Questa funzione quantifica quanto della capacità di calcolo spesa da M1 si traduce poi in materiale utilizzabile da M2 — lo spreco M1→M2, non visibile guardando `result.candidates` da solo.

### `gate_survival_observed`

Legge `result.event_discovery.raw_events` — la popolazione grezza pre-gate, disponibile solo quando `DiscoveryConfig.retain_raw_events=True` (il default) — e per ciascun `RawEvent` estrae il proprio `GateResult` (statistiche osservate: `mean_tpm`, `index_of_dispersion`, `episode_index_of_dispersion`, `n_episodes`, `passed`, `fail_reason`) affiancato ai `GateParams` configurati (`min_tpm`, `max_dispersion`, `dispersion_margin`, `event_counting`) — questi ultimi ripetuti identici su ogni riga dello stesso risultato, per permettere un confronto diretto osservato-vs-soglia riga per riga senza dover riandare a recuperare la configurazione a parte. Salta silenziosamente qualunque risultato senza `event_discovery` o con `raw_events=None` (Event Discovery non eseguita, o eseguita con `retain_raw_events=False`).

*Perché esiste, come motivazione di design:* `EventDiscovery.event_distribution_report()` (manuale §9, issue #215) racconta già in prosa perché un preset è troppo permissivo o troppo restrittivo per un dato asset — questa funzione espone gli stessi identici ingredienti come dati, componibili con `groupby`/aggregazioni proprie invece che nella forma fissa di un report testuale. Il caso d'uso esplicito è diagnosticare un mismatch preset/asset (es. `min_tpm` troppo alto per un asset a bassa frequenza di eventi) *prima* che si manifesti a valle come "0 candidati" in M2 — un sintomo che, senza questi dati, richiederebbe di indovinare quale delle tre soglie del gate (tpm, dispersione aggregata, dispersione per episodio) sia quella che sta effettivamente bocciando la popolazione.

---

## 5. Logica interna — M2 Alpha Discovery

### `discard_reasons_by_grade`

Legge `result.rule_responses` — **ogni** contratto promosso abbinato al proprio verdetto di Rule Discovery, deliberatamente non `result.edges()` (che per costruzione non contiene mai `NON-EDGE`, l'unico verdetto che questa funzione filtra). Il filtro sul grade è case-insensitive su entrambi i lati (`grade.strip().upper()` contro `alpha_score.grade` normalizzato allo stesso modo); un contratto senza `alpha_score` — mai gradato — non soddisfa mai alcun filtro di grade, per costruzione. Per ogni contratto che passa il filtro, `rejection_reasons` viene esploso una riga per elemento — una lista vuota produce comunque una riga con `reason=None`, non viene scartata, cosicché il conteggio totale delle righe corrisponda sempre al numero di contratti filtrati quando nessuno ha più di una ragione.

*Perché esiste, come motivazione di design:* un grade alto (A/B) da Alpha Discovery misura solo il potere predittivo statistico — non garantisce nulla sull'esito di Rule Discovery, che è l'unico giudice economico (invariante §8 del manuale). Capire *perché* Rule Discovery scarta sistematicamente contratti di grado alto (soglie di trade insufficienti? rotation null non superato? win rate sotto floor?) è la domanda diagnostica più diretta per giudicare se un preset è troppo permissivo a monte o troppo severo a valle.

### `undetermined_direction_by_family`

Per ogni contratto in `result.contracts` (promossi e rigettati indistintamente — non solo `result.promoted`), risolve l'`EventCandidate` di origine via `event_candidate_id` (un dizionario `{event_id: candidate}` costruito una volta per risultato) e itera sui componenti dell'evento, emettendo una riga per componente con la `direction` del contratto (inclusa `"undetermined"`).

**Classificazione della famiglia (`_feature_family`, interna)** — il dispatch avviene su `len(source_cols)`, non sulla sua truthiness:
- `len(source_cols) == 2` (coppie cross-colonna — cross-OHLC, MACD-vs-signal, prezzo-vs-volume, …) → `"cross_pair"`.
- `len(source_cols) == 3` (triple — es. geometria candela) → `"cross_triple"`.
- Qualunque altra lunghezza (incluse `0`/assente, e le feature native — per cui `source_cols` in pratica **non è affidabilmente vuoto**, vedi nota storica sotto) → ricade sul regex `{base}_{indicatore}_{periodo}` (es. `close_rsi_25` → `"rsi"`); `"other"` è il fallback finale solo se il regex non trova corrispondenza.

**Nota storica sul bug corretto in `_feature_family`:** una prima versione dispatchava su `bool(source_cols)` invece che sulla sua lunghezza. `EventComponent.source_cols` risulta però non vuoto (lunghezza 1) anche per feature native di arità 1, nonostante il proprio docstring dichiari il contrario — quindi ogni feature nativa veniva instradata nel ramo cross-feature prima ancora che il regex sul nome potesse mai girare, e ogni famiglia nativa (`rsi`, `ema`, `ret`, …) spariva silenziosamente dentro `"other"`. Corretto dispatchando su `len(source_cols)`; verificato di nuovo contro una run reale dopo il fix (vedi `docs/specs/playground_en.md` per i numeri aggiornati). È l'esempio concreto, in questo repository, del motivo per cui §11 tratta questo modulo come diagnostico e non come API stabile, checklist completa o no.

*Perché esiste, come motivazione di design:* `direction == "undetermined"` è l'unico gate di rigetto rigido di Alpha Discovery (manuale §8/§9) — capire se certe famiglie di feature sorgente alimentano eventi che M2 non riesce quasi mai a orientare, sistematicamente, è la domanda diagnostica diretta per decidere se investire nel migliorare quella famiglia di feature o abbandonarla. Un evento composto (AND) contribuisce una riga per componente proprio per questo: una famiglia che compare solo dentro eventi composti non deve sparire dietro l'`event_id` dell'intero composto.

---

## 6. Logica interna — M3 Rule Discovery

Entrambe le funzioni di `m3.py` condividono un piccolo helper interno, `_contract_grade`, identico a quello usato in `m2.py`/`forgedge.deployment` (normalizza `alpha_score.grade` a stringa maiuscola, `None` se il contratto non è mai stato gradato).

### `diagnostics_vs_verdict`

Itera su `result.rule_responses` ed esplode `AlphaContract.diagnostics` — osservazioni di Alpha Discovery che informano il grade ma non bloccano nulla in M2 (a differenza di `direction="undetermined"`, che è un gate rigido) — una riga per diagnostic, ciascuna abbinata al `verdict` che M3 ha assegnato più tardi allo stesso contratto. Un contratto senza diagnostics emette comunque una riga con `diagnostic=None`, con lo stesso principio "salta zero righe, sempre una riga per contratto filtrato" già visto in `discard_reasons_by_grade`.

*Perché esiste, come motivazione di design:* un `diagnostic` non bloccante è per definizione qualcosa che Alpha Discovery ha notato ma non ha ritenuto abbastanza grave da bocciare il contratto. Questa funzione rende misurabile se quella scelta è giustificata: un diagnostic che compare sproporzionatamente spesso sulle righe `NON-EDGE` è un candidato concreto a essere promosso da semplice annotazione FYI a un vero gate M2 — la stessa domanda, generalizzata, che ha originato `discard_reasons_by_grade`.

### `lottery_only_winners`

Filtra `result.rule_responses` ai soli `verdict == "PARTIAL-EDGE"`, poi controlla `response.rejection_reasons`: un verdetto `PARTIAL-EDGE` significa che almeno un `edge_block` ha impedito il pieno `EDGE`, quindi la lista non è mai vuota per queste righe. `rotation_only` è vero quando quella lista ha **esattamente un elemento** e il suo testo comincia con `"search-level rotation null not cleared"` — il prefisso letterale del messaggio che il rotation null (manuale §14-15) genera quando un contratto perde la lotteria del multiple-testing pur avendo superato ogni altro gate economico/statistico.

*Perché esiste, come motivazione di design:* preset permissivi come `"sweep"` (manuale §7) sono pensati per essere accoppiati al `RotationCalibrator` proprio perché generano molti `PARTIAL-EDGE`, alcuni dei quali sono "quasi vincitori" bloccati solo dalla correzione per test multipli, altri genuinamente deboli su PF/DSR/consistenza OOS. Questa funzione separa i due casi senza dover rileggere `rejection_reasons` a mano contratto per contratto — un `rotation_only=True` in alta proporzione è il segnale che il preset sta funzionando come da manuale, non che sta producendo rumore.

---

## 7. Logica interna — M4 Rule Registry

Le due funzioni di `m4.py` sono le uniche del modulo a **non** prendere `Iterable[ForgeResult]` come input, ma `Iterable[RuleRegistry]` — una deviazione deliberata dal principio #1 del §2, spiegata nel docstring del file stesso: la classificazione cross-ticker (`GENERIC`/`PARTIAL`/`SPECIFIC`/`ISOLATED`) vive sul registro **pooled** che `forge_multi()` restituisce separatamente dai `ForgeResult` per-ticker (ognuno dei quali ha `.registry = None` in quel percorso, proprio per evitare un registro parziale fuorviante). Per una singola run `forge()` (dove il cross-ticker è per definizione banale — un solo ticker), si passa semplicemente `[result.registry]`.

### `classification_by_grade`

Un semplice passaggio su `registry.documents`, che tiene solo i `RuleDocument` con `classification` non `None` (cioè quelli per cui lo Step 4 del Rule Registry — il backtest cross-ticker con ricalcolo soglie, manuale §11 — è effettivamente girato) ed emette una riga per documento con `grade`/`classification` affiancati.

*Perché esiste, come motivazione di design:* il grade alpha (A/B/C/…) misura potere predittivo statistico su un singolo asset; la classificazione di genericità misura tutt'altro — se la stessa regola tiene su asset diversi. Non c'è alcuna garanzia a priori che le due cose correlino, e questa funzione rende la domanda direttamente verificabile invece che assunta: un grade alto che produce prevalentemente regole `ISOLATED` sarebbe un segnale che il grading di M2 sta misurando qualcosa che non generalizza.

### `duplicate_clusters`

Un passaggio ancora più diretto su `registry.documents`, senza filtro: una riga per documento con `is_duplicate`/`duplicate_of` esposti così come il Rule Registry li ha calcolati (Step 2-3, matrici Jaccard/Spearman — manuale §11).

*Perché esiste, come motivazione di design:* la quota di deduplicazione da sola (`is_duplicate.mean()`) dice quanto M1/M2 ridondano; raggruppare per `duplicate_of` dice **come** ridondano — un cluster grande convergente su un unico sopravvissuto è la prova diretta che più soglie/espressioni distinte scoperte da Event Discovery stanno codificando la stessa idea economica, non 3-4 edge realmente indipendenti.

---

## 8. Logica interna — caso trasversale

### `conversion_funnel`

Vive in `funnel.py`, non in un file `mN.py`, perché non è ancorato a un singolo modulo: per ogni `ForgeResult` conta la popolazione a quattro tappe del funnel — `len(result.candidates)` (M1, sopravvissuti al gate), `len(result.contracts)` (M2, ogni valutazione, promossa o no), `len(result.promoted)` (ipotesi passate a M3), `len(result.edges())` (M3, verdetti `EDGE`/`PARTIAL-EDGE`) — ed emette una riga per `(ticker, stage)`.

*Perché esiste, come motivazione di design:* nessun singolo campo di `ForgeResult` mostra il funnel end-to-end in una forma direttamente comparabile tra ticker o tra run — bisognerebbe leggere quattro attributi diversi e assemblarli a mano ogni volta. `df.pivot(index="ticker", columns="stage", values="n")` dà la tabella funnel pronta; dividere tappe adiacenti dà il tasso di conversione per step, la metrica che l'issue #237 originale chiamava esplicitamente come caso d'uso "Extra" trasversale a tutti i moduli.

---

## 9. Roadmap — issue #237

`forgedge.playground` seguiva una checklist di 11 casi d'uso unici tracciata su GitHub come issue #237 — **oggi tutti e 11 sono implementati** (§3-8 di questo documento ne coprono la logica interna; `docs/specs/playground_en.md` ne copre l'utilizzo):

| Modulo | Caso d'uso | Stato | PR |
|---|---|---|---|
| M0 | Nervosismo dei confini di regime | ✅ `regime_transitions()` | #239 |
| M0 | Asset "prigionieri" di un regime | ✅ `regime_time_share()` | #239 |
| M1 | Eventi "morti" — sopravvivono al gate ma non producono mai un `AlphaContract`, o solo con `direction="undetermined"` | ✅ `dead_event_candidates()` | #243 |
| M1 | Gate-survival atteso vs osservato per preset/asset (confronto strutturato di `EventDiscovery.event_distribution_report` tra run) | ✅ `gate_survival_observed()` | #243 |
| M2 | Perché M3 scarta contratti alpha di grado A | ✅ `discard_reasons_by_grade()` | #238 |
| M2 | Famiglie di indicatori che non targettizzano mai | ✅ `undetermined_direction_by_family()` | #238 (fix classificazione: #241) |
| M3 | Diagnostics M2 non bloccanti che correlano con un `NON-EDGE` in M3, per grado | ✅ `diagnostics_vs_verdict()` | #244 |
| M3 | Contratti `PARTIAL-EDGE` solo perché il rotation null non è stato superato, per preset/grado | ✅ `lottery_only_winners()` | #244 |
| M4 | Le regole di grado alto generalizzano meglio cross-ticker? (distribuzione `classification` per grade alpha di origine) | ✅ `classification_by_grade()` | #244 |
| M4 | Peso e cluster della deduplicazione (`is_duplicate`/`duplicate_of`) | ✅ `duplicate_clusters()` | #244 |
| Trasversale | Tasso di conversione end-to-end `candidati → contratti → promossi → edge` per asset | ✅ `conversion_funnel()` | #244 |

L'issue #237 resta aperta su GitHub nonostante la checklist sia completa — non traccia più lavoro futuro su `forgedge.playground` in sé, ma è il riferimento storico su design e provenienza di ogni funzione. Il seguito naturale della checklist — portare le regole scoperte in produzione — è tracciato separatamente dall'issue #245 e non vive più in questo modulo (§10).

---

## 10. Il modulo gemello: `forgedge.deployment`

Nato come seguito naturale di `forgedge.playground` (issue #245: "come portare in produzione le regole che `forge()` promuove"), inizialmente implementato dentro `forgedge/playground/production.py`. Spostato nel proprio pacchetto top-level `forgedge/deployment/` da PR #247, con motivazione esplicita nel changelog dell'issue: le tre funzioni che ci vivevano (`promotion_gate`, `export_rules`, `monitoring_manifest`) hanno **effetti reali** — un gate che decide cosa può andare in produzione, una scrittura su filesystem — che il nome "playground" (sinonimo, nel resto di questo documento, di "sola lettura, nessun effetto collaterale", §2/§11) non descriveva più onestamente una volta che quelle funzioni facevano più che guardare.

Nessun cambio di comportamento nello spostamento — stesse firme, stessa `PromotionGateConfig`, solo il percorso di import è cambiato:

```python
# prima (issue #245, ora obsoleto)
from forgedge.playground import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest

# dopo (PR #247)
from forgedge.deployment import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest
```

`forgedge.playground` non importa più nulla da quel modulo — il suo `__all__` è tornato a coprire solo le 10 funzioni di sola analisi del §9. Per la motivazione di design e la logica interna di `forgedge.deployment` stesso (perché tre funzioni in sequenza, cosa fa `PromotionGateConfig`, perché solo `export_rules` tocca il filesystem), vedi il documento gemello `docs/modules/Deployment.md`; per la guida all'utilizzo, `docs/specs/deployment_en.md`/`deployment_it.md`.

---

## 11. Stabilità e garanzie

- **Non fa parte dell'API core.** `forge()`, `RuleDiscovery`, `RuleRegistry` e gli altri componenti M0-M4 sono l'API stabile della libreria; `forgedge.playground` è uno strato diagnostico costruito sopra di essa. La checklist dell'issue #237 è completa (§9), ma questo non lo promuove ad API stabile: le firme esistenti possono ancora affinarsi mano a mano che emergono nuovi casi d'uso reali (§5 documenta un esempio concreto già accaduto: un bug di classificazione corretto dopo la prima release del modulo).
- **Confine di scopo esplicito verso `forgedge.deployment`.** Da PR #247, `forgedge.playground` è tornato a coprire *solo* funzioni di sola analisi — niente che decida cosa va in produzione o scriva su disco. Una futura funzione con effetti reali va in `forgedge.deployment`, non qui (§10).
- **Nessun effetto collaterale.** Ogni funzione è una pura trasformazione di dati già calcolati; non rilancia mai `forge()` né alcun modulo M0-M4, non muta i `ForgeResult`/`RuleRegistry` che riceve, non ha stato tra una chiamata e l'altra.
- **Nessuna dipendenza aggiuntiva.** Come il resto della libreria, solo `numpy`/`pandas`.
- Per il comportamento esatto verificato contro il codice — firme, parametri, esempi eseguibili — vedi sempre `docs/specs/playground_en.md`/`playground_it.md`, non questo documento: questo descrive il design, quello descrive l'API.
