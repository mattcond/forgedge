# FORGE — Playground Module
> Non è un modulo della pipeline (M0-M4): è un livello di analisi che gira
> **dopo**, sopra l'insieme di risultati che una sessione di ricerca produce.
> `forgedge.playground` raccoglie funzioni di sola lettura che prendono **R**
> — uno o più `ForgeResult` (`forge()`/`forge_multi()`) — e rispondono a
> domande trasversali sul comportamento della pipeline stessa: dove il
> funnel perde candidati, quali famiglie di feature non si orientano mai,
> quanto sono "nervosi" i confini di regime. Non rilancia mai `forge()`,
> non modifica nulla che gli viene passato, e non fa parte dell'API core:
> è deliberatamente in evoluzione, tracciato da una checklist aperta
> (issue #237), non uno strumento stabile al livello di `forge()` o
> `RuleDiscovery`.
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
4. [Logica interna — M2 Alpha Discovery](#4-logica-interna--m2-alpha-discovery)
5. [Roadmap — issue #237](#5-roadmap--issue-237)
6. [Stabilità e garanzie](#6-stabilità-e-garanzie)

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

`from forgedge.playground import *` è l'import previsto (`forgedge/playground/__init__.py` definisce esplicitamente `__all__`); il modulo **non** è ri-esportato dal pacchetto top-level `forgedge` — va importato dal proprio sottopacchetto. Il codice sorgente è organizzato per modulo di origine dei dati che ciascuna funzione legge principalmente: `forgedge/playground/m0.py` per le funzioni ancorate a Market Context, `m2.py` per quelle ancorate ad Alpha Discovery, e così via man mano che la checklist del §5 si completa.

---

## 2. Principi di design

Tre scelte, applicate a ogni funzione del modulo senza eccezioni:

1. **Input sempre `Iterable[ForgeResult]`.** Mai un singolo `ForgeResult` nudo, anche quando si analizza una sola run — così lo stesso codice chiamante scala naturalmente da un ticker a un'intera sessione multi-asset senza cambiare firma. Nessuna funzione rilancia `forge()`, `forge_multi()` o alcun modulo M0-M4: se il dato che serve non è già su `ForgeResult`, la funzione non esiste ancora (vedi §5, Roadmap).
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

## 4. Logica interna — M2 Alpha Discovery

### `discard_reasons_by_grade`

Legge `result.rule_responses` — **ogni** contratto promosso abbinato al proprio verdetto di Rule Discovery, deliberatamente non `result.edges()` (che per costruzione non contiene mai `NON-EDGE`, l'unico verdetto che questa funzione filtra). Il filtro sul grade è case-insensitive su entrambi i lati (`grade.strip().upper()` contro `alpha_score.grade` normalizzato allo stesso modo); un contratto senza `alpha_score` — mai gradato — non soddisfa mai alcun filtro di grade, per costruzione. Per ogni contratto che passa il filtro, `rejection_reasons` viene esploso una riga per elemento — una lista vuota produce comunque una riga con `reason=None`, non viene scartata, cosicché il conteggio totale delle righe corrisponda sempre al numero di contratti filtrati quando nessuno ha più di una ragione.

*Perché esiste, come motivazione di design:* un grade alto (A/B) da Alpha Discovery misura solo il potere predittivo statistico — non garantisce nulla sull'esito di Rule Discovery, che è l'unico giudice economico (invariante §8 del manuale). Capire *perché* Rule Discovery scarta sistematicamente contratti di grado alto (soglie di trade insufficienti? rotation null non superato? win rate sotto floor?) è la domanda diagnostica più diretta per giudicare se un preset è troppo permissivo a monte o troppo severo a valle.

### `undetermined_direction_by_family`

Per ogni contratto in `result.contracts` (promossi e rigettati indistintamente — non solo `result.promoted`), risolve l'`EventCandidate` di origine via `event_candidate_id` (un dizionario `{event_id: candidate}` costruito una volta per risultato) e itera sui componenti dell'evento, emettendo una riga per componente con la `direction` del contratto (inclusa `"undetermined"`).

**Classificazione della famiglia (`_feature_family`, interna)** — il dispatch avviene su `len(source_cols)`, non sulla sua truthiness:
- `len(source_cols) == 2` (coppie cross-colonna — cross-OHLC, MACD-vs-signal, prezzo-vs-volume, …) → `"cross_pair"`.
- `len(source_cols) == 3` (triple — es. geometria candela) → `"cross_triple"`.
- Qualunque altra lunghezza (incluse `0`/assente, e le feature native — per cui `source_cols` in pratica **non è affidabilmente vuoto**, vedi nota storica sotto) → ricade sul regex `{base}_{indicatore}_{periodo}` (es. `close_rsi_25` → `"rsi"`); `"other"` è il fallback finale solo se il regex non trova corrispondenza.

**Nota storica sul bug corretto in `_feature_family`:** una prima versione dispatchava su `bool(source_cols)` invece che sulla sua lunghezza. `EventComponent.source_cols` risulta però non vuoto (lunghezza 1) anche per feature native di arità 1, nonostante il proprio docstring dichiari il contrario — quindi ogni feature nativa veniva instradata nel ramo cross-feature prima ancora che il regex sul nome potesse mai girare, e ogni famiglia nativa (`rsi`, `ema`, `ret`, …) spariva silenziosamente dentro `"other"`. Corretto dispatchando su `len(source_cols)`; verificato di nuovo contro una run reale dopo il fix (vedi `docs/specs/playground_en.md` per i numeri aggiornati). È l'esempio concreto, in questo repository, del motivo per cui §6 classifica questo modulo come ancora in evoluzione.

*Perché esiste, come motivazione di design:* `direction == "undetermined"` è l'unico gate di rigetto rigido di Alpha Discovery (manuale §8/§9) — capire se certe famiglie di feature sorgente alimentano eventi che M2 non riesce quasi mai a orientare, sistematicamente, è la domanda diagnostica diretta per decidere se investire nel migliorare quella famiglia di feature o abbandonarla. Un evento composto (AND) contribuisce una riga per componente proprio per questo: una famiglia che compare solo dentro eventi composti non deve sparire dietro l'`event_id` dell'intero composto.

---

## 5. Roadmap — issue #237

`forgedge.playground` segue una checklist aperta di 11 casi d'uso unici, tracciata su GitHub come issue #237. Quattro sono implementati (§3-4 di questo documento ne coprono la logica interna; `docs/specs/playground_en.md` ne copre l'utilizzo); sette restano da progettare e implementare:

| Modulo | Caso d'uso | Stato |
|---|---|---|
| M0 | Nervosismo dei confini di regime | ✅ `regime_transitions()` |
| M0 | Asset "prigionieri" di un regime | ✅ `regime_time_share()` |
| M1 | Eventi "morti" — sopravvivono al gate ma non producono mai un `AlphaContract`, o solo con `direction="undetermined"` | ⬜ non implementato |
| M1 | Gate-survival atteso vs osservato per preset/asset (confronto strutturato di `EventDiscovery.event_distribution_report` tra run) | ⬜ non implementato |
| M2 | Perché M3 scarta contratti alpha di grado A | ✅ `discard_reasons_by_grade()` |
| M2 | Famiglie di indicatori che non targettizzano mai | ✅ `undetermined_direction_by_family()` |
| M3 | Diagnostics M2 non bloccanti che correlano con un `NON-EDGE` in M3, per grado | ⬜ non implementato |
| M3 | Contratti `PARTIAL-EDGE` solo perché il rotation null non è stato superato, per preset/grado | ⬜ non implementato |
| M4 | Le regole di grado alto generalizzano meglio cross-ticker? (distribuzione `classification` per grade alpha di origine) | ⬜ non implementato |
| M4 | Peso e cluster della deduplicazione (`is_duplicate`/`duplicate_of`) | ⬜ non implementato |
| Trasversale | Tasso di conversione end-to-end `candidati → contratti → promossi → edge` per asset | ⬜ non implementato |

Chi implementa una voce di questa checklist dovrebbe seguire gli stessi tre principi del §2 (input `Iterable[ForgeResult]`, output long-format, "salta non sollevare" dichiarato esplicitamente), aggiungere la funzione al file `m{N}.py` corrispondente al modulo di origine dei dati che legge principalmente — così come `m0.py`/`m2.py` fanno oggi — e aggiornare sia questo documento (logica interna, roadmap) sia `docs/specs/playground_en.md`/`playground_it.md` (guida all'utilizzo, esempio verificato).

---

## 6. Stabilità e garanzie

- **Non fa parte dell'API core.** `forge()`, `RuleDiscovery`, `RuleRegistry` e gli altri componenti M0-M4 sono l'API stabile della libreria; `forgedge.playground` è uno strato diagnostico costruito sopra di essa, esplicitamente più giovane e più soggetto a evolvere — nuove funzioni si aggiungono seguendo la checklist del §5, e le firme di quelle esistenti possono affinarsi mano a mano che emergono nuovi casi d'uso reali (§4 documenta un esempio concreto già accaduto: un bug di classificazione corretto dopo la prima release del modulo).
- **Nessun effetto collaterale.** Ogni funzione è una pura trasformazione di dati già calcolati; non rilancia mai `forge()` né alcun modulo M0-M4, non muta i `ForgeResult` che riceve, non ha stato tra una chiamata e l'altra.
- **Nessuna dipendenza aggiuntiva.** Come il resto della libreria, solo `numpy`/`pandas`.
- Per il comportamento esatto verificato contro il codice — firme, parametri, esempi eseguibili — vedi sempre `docs/specs/playground_en.md`/`playground_it.md`, non questo documento: questo descrive il design, quello descrive l'API.
