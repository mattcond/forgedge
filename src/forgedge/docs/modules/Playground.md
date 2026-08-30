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

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Principi di design](#2-principi-di-design)
3. [Funzioni — M0 Market Context](#3-funzioni--m0-market-context)
4. [Funzioni — M2 Alpha Discovery](#4-funzioni--m2-alpha-discovery)
5. [Esempio completo, verificato](#5-esempio-completo-verificato)
6. [Roadmap — issue #237](#6-roadmap--issue-237)
7. [Stabilità e garanzie](#7-stabilità-e-garanzie)

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

**Confine netto:** `forgedge.playground` non tocca mai la pipeline core (M0-M4) e non ha side effect — ogni funzione è una pura trasformazione `Iterable[ForgeResult] -> pd.DataFrame`. Non è un modulo di discovery, non produce verdetti, non persiste nulla. È uno strato di analisi **sopra** ciò che la pipeline ha già prodotto.

`from forgedge.playground import *` è l'import previsto (`forgedge/playground/__init__.py` definisce esplicitamente `__all__`); il modulo **non** è ri-esportato dal pacchetto top-level `forgedge` — va importato dal proprio sottopacchetto.

---

## 2. Principi di design

Tre scelte, applicate a ogni funzione del modulo senza eccezioni:

1. **Input sempre `Iterable[ForgeResult]`.** Mai un singolo `ForgeResult` nudo, anche quando si analizza una sola run — così lo stesso codice chiamante scala naturalmente da un ticker a un'intera sessione multi-asset senza cambiare firma. Nessuna funzione rilancia `forge()`, `forge_multi()` o alcun modulo M0-M4: se il dato che serve non è già su `ForgeResult`, la funzione non esiste ancora (vedi §6, Roadmap).
2. **Output sempre in formato long.** Una riga per osservazione elementare — una transizione di regime, una ragione di rigetto, un componente di un evento — mai una tabella già aggregata. Ogni funzione documenta nel proprio docstring il `groupby`/l'aggregazione tipica da concatenare a valle, ma non la esegue essa stessa: la stessa tabella lunga deve poter rispondere a domande diverse da quella che ha motivato la funzione.
3. **"Salta, non sollevare" per gli input parziali — ma sempre dichiarato esplicitamente.** Una run senza colonna `regime` (Market Context disabilitato), un contratto il cui `event_candidate_id` non risolve più contro `result.candidates`, un `AlphaContract` senza `alpha_score` (mai gradato): nessuno di questi casi solleva un'eccezione, tutti vengono saltati silenziosamente. Questo è coerente con lo stile del resto della libreria (§11 del manuale: le funzioni di analisi qualitativa non sono validazioni dei dati), ma **non è mai un default implicito senza documentazione** — ogni funzione qui sotto elenca esattamente cosa salta e perché.

Conseguenza pratica dei primi due principi: chiamare due volte la stessa funzione playground su due sottoinsiemi di `R` e concatenare i risultati (`pd.concat`) dà lo stesso risultato di chiamarla una volta sola sull'unione — le funzioni non mantengono stato tra le chiamate e non normalizzano nulla che dipenda dall'insieme completo (percentili, ranking, ecc. sono sempre lasciati al chiamante via `groupby`).

---

## 3. Funzioni — M0 Market Context

Entrambe le funzioni di questa sezione leggono `result.enriched` — la KPI Table dopo Market Context, con le colonne `regime`/`regime_stable` aggiunte — e **saltano** silenziosamente qualunque risultato il cui `enriched` non abbia una colonna `regime` (sessione con `run_market_context=False`, o `ForgeResult` costruito a mano senza passare da Market Context), invece di sollevare.

### `regime_transitions(results: Iterable[ForgeResult]) -> pd.DataFrame`

Log in formato long di **ogni** cambio di regime osservato, con la lunghezza del run che lo ha preceduto.

**Perché esiste:** un confine di regime "nervoso" — che flippa dopo un run di 1-2 barre invece di restare stabile per un tratto ragionevole — inquina silenziosamente qualunque evento M1 condizionato al regime a valle: se il classificatore oscilla rapidamente tra due etichette, le statistiche calcolate "durante il regime X" mescolano di fatto più regimi. Questo è esattamente il fenomeno che `regime_stable` (con la sua finestra `stable_window`, risolta dalla sessione — vedi il manuale §8, Market Context) è pensata per filtrare a valle; `regime_transitions` rende visibile *quanto* nervosismo c'è da filtrare, prima ancora di guardare `regime_stable`.

**Colonne restituite:** `ticker`, `bar_index` (posizione intera della barra in cui avviene il flip), `timestamp`, `from_regime`, `to_regime`, `run_length_before` (numero di barre consecutive nel regime di partenza, incluso l'ultimo bar prima del flip).

**Risoluzione del timestamp**, in ordine di preferenza (`_timestamp_at`, interna): `DatetimeIndex` del frame, se presente; altrimenti la colonna `open_dt`, se presente; altrimenti la posizione intera stessa (nessun errore, nessun warning — un fallback silenzioso e deliberato).

**Cosa non conta come transizione:** una barra con `regime` non classificato (`NaN`) subito dopo l'inizio della serie non genera una transizione "fantasma" dal `NaN` al primo regime osservato — il confronto `prev_regime.notna()` esclude esplicitamente questo caso. Una serie a regime costante produce un `DataFrame` vuoto (ma con le colonne corrette, mai `None`).

**Uso tipico a valle** (esplicitamente suggerito dal docstring):
```python
df = regime_transitions(results)
df[df["run_length_before"] <= 2].groupby("ticker").size()   # ranking dei ticker per "nervosismo"
```

### `regime_time_share(results: Iterable[ForgeResult]) -> pd.DataFrame`

Quota di barre (sulle sole barre classificate, cioè non-`NaN`) che ciascun ticker passa in ciascun regime.

**Perché esiste:** un asset che vive per la stragrande maggioranza della propria storia in un solo regime è un candidato a "regole scoperte su di esso sembrano generiche ma sono in realtà regime-specifiche" — non c'è mai stato abbastanza di un regime alternativo presente nel campione per dimostrare il contrario. Questa funzione rende quel rischio misurabile prima di fidarsi della genericità di una regola scoperta su quell'asset.

**Colonne restituite:** `ticker`, `regime`, `n_bars` (conteggio assoluto), `share` (frazione in `[0, 1]`, calcolata sul totale delle barre classificate di quel risultato — non sul totale assoluto della serie, quindi le barre `NaN` sono escluse sia dal numeratore sia dal denominatore).

**Uso tipico a valle:**
```python
df = regime_time_share(results)
df.sort_values("share", ascending=False).groupby("ticker").head(1)   # regime dominante per asset
```

---

## 4. Funzioni — M2 Alpha Discovery

### `discard_reasons_by_grade(results: Iterable[ForgeResult], grade: str = "A") -> pd.DataFrame`

Per i contratti del `grade` richiesto (case-insensitive) il cui verdetto di Rule Discovery è `NON-EDGE`, esplode `rejection_reasons` una riga per ragione — così una regola con più ragioni concorrenti (es. `profit_factor` basso *e* `expectancy` non significativa) contribuisce più righe, componibili con un `groupby("reason")` a valle.

**Perché esiste:** un grade alto (A/B) da Alpha Discovery misura solo il potere predittivo statistico — non garantisce nulla sull'esito di Rule Discovery, che è l'unico giudice economico (invariante §8 del manuale). Capire *perché* Rule Discovery scarta sistematicamente contratti di grado alto (soglie di trade insufficienti? rotation null non superato? win rate sotto floor?) è la domanda diagnostica più diretta per giudicare se un preset è troppo permissivo a monte o troppo severo a valle.

**Legge:** `result.rule_responses` — **ogni** contratto promosso abbinato al proprio verdetto di Rule Discovery, non solo i tradeable (`result.edges()` sarebbe la scelta sbagliata qui: per costruzione non contiene mai `NON-EDGE`).

**Colonne restituite:** `ticker`, `alpha_id`, `event_candidate_id`, `reason` (una singola stringa da `rejection_reasons`, o `None` se la lista è vuota — una riga viene comunque emessa, non scartata), `failed_condition` (da `response.entry_optimization.failed_condition` quando l'oggetto esiste, altrimenti `None` — vedi il manuale §9 per quando `entry_optimization` è popolato).

**Filtro sul grade:** case-insensitive (`grade.strip().upper()` contro `alpha_score.grade` normalizzato allo stesso modo); un contratto senza `alpha_score` (mai gradato) non soddisfa mai alcun filtro di grade, per nessun valore di `grade` richiesto.

**Uso tipico a valle:**
```python
df = discard_reasons_by_grade(results, grade="A")
df["reason"].value_counts()                       # quali ragioni dominano
pd.crosstab(df["failed_condition"], df["reason"])  # incrocio con l'esito di entry_mode="auto"
```

### `undetermined_direction_by_family(results: Iterable[ForgeResult]) -> pd.DataFrame`

Per **ogni** contratto valutato (`result.contracts` — promossi e rigettati indistintamente, non solo `result.promoted`), risolve l'`EventCandidate` di origine via `event_candidate_id` ed emette una riga per ciascun componente dell'evento, con la famiglia semantica del componente e la `direction` finale del contratto (inclusa `"undetermined"`).

**Perché esiste:** `direction == "undetermined"` è l'unico gate di rigetto rigido di Alpha Discovery (manuale §8/§9) — capire se certe famiglie di feature sorgente (RSI, EMA, coppie cross-colonna, …) alimentano eventi che M2 non riesce quasi mai a orientare, sistematicamente, è la domanda diagnostica diretta per decidere se investire nel migliorare quella famiglia di feature o abbandonarla.

**Classificazione della famiglia** (`_feature_family`, interna):
- Colonne native che seguono la convenzione `{base}_{indicatore}_{periodo}` (es. `close_rsi_25`) → il gruppo indicatore (`"rsi"`).
- Feature composte con `source_cols` popolato (coppie/triple cross-colonna — cross-OHLC, MACD-vs-signal, prezzo-vs-volume, geometria candela, …), che non seguono quella convenzione sul proprio nome sintetico → bucket per arità: `"cross_pair"` (2 colonne sorgente), `"cross_triple"` (3 colonne sorgente), `"other"` per ogni altra arità.
- Qualunque altro nome non conforme → `"other"`.

**Eventi composti (AND):** un evento con più componenti contribuisce **una riga per componente**, tutte con la stessa `direction` del contratto — così una famiglia che compare solo all'interno di un evento composto viene comunque contata, invece di sparire dietro l'`event_id` dell'intero composto.

**Contratti senza candidato risolvibile:** se `event_candidate_id` non trova corrispondenza in `result.candidates` (es. un `ForgeResult` assemblato a mano con liste disallineate), il contratto viene saltato silenziosamente — nessuna riga emessa, nessuna eccezione.

**Colonne restituite:** `ticker`, `alpha_id`, `event_candidate_id`, `family`, `direction`.

**Uso tipico a valle:**
```python
df = undetermined_direction_by_family(results)
df.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())   # tasso undetermined per famiglia
```

---

## 5. Esempio completo, verificato

Eseguito su due run `forge()` reali — `ADAUSDC` (il fixture di riferimento del repository) e una seconda serie sintetica etichettata `BTCUSDC` — messe in pool in un'unica lista `results`:

```python
import pandas as pd
from forgedge import forge
from forgedge.playground import (
    regime_transitions, regime_time_share,
    discard_reasons_by_grade, undetermined_direction_by_family,
)

result_ada = forge(kpi_ada, ticker="ADAUSDC", timeframe="1D", progress=False)
result_btc = forge(kpi_btc, ticker="BTCUSDC", timeframe="1D", progress=False)
results = [result_ada, result_btc]

rt = regime_transitions(results)
print(rt.shape)   # (236, 6)
print(rt[rt["run_length_before"] <= 2].groupby("ticker").size())
```

**Output verificato:**

```
(236, 6)
ticker
ADAUSDC    45
BTCUSDC    41
dtype: int64
```

```python
share = regime_time_share(results)
top = share.sort_values("share", ascending=False).groupby("ticker").head(1)
print(top[["ticker", "regime", "share"]].to_string(index=False))
```

```
 ticker      regime    share
BTCUSDC STRONG_BEAR 0.438776
ADAUSDC STRONG_BEAR 0.407029
```

Entrambi i ticker di questo esempio risultano dominati da `STRONG_BEAR` per oltre il 40% della propria storia — esattamente il tipo di segnale per cui questa funzione esiste: qualunque regola scoperta su questi due asset merita una verifica esplicita di quanto sia stata condizionata da un solo regime.

```python
d = discard_reasons_by_grade(results, grade="B")
print(d.shape)                          # (561, 5)
print(d["reason"].value_counts().head(3))
```

```
total_trades 4 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    34
total_trades 9 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    20
total_trades 6 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    19
Name: count, dtype: int64
```

Il floor sul numero di trade nella prima finestra di walk-forward (issue #217, vedi `modulo_3_it.md` §9) domina nettamente le ragioni di scarto dei contratti di grado B su questo fixture — un'informazione che nessun singolo `RuleDiscoveryResponse.rejection_reasons` isolato rende visibile con la stessa chiarezza.

```python
fam = undetermined_direction_by_family([result_ada])
rate = fam.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())
print(rate.sort_values(ascending=False))
```

```
family
cross_triple    0.945455
cross_pair      0.915001
other           0.908046
Name: direction, dtype: float64
```

Nota onesta su questo specifico fixture: nessun componente arrivato fino a un contratto valutato appartiene a una famiglia nativa semplice (`rsi`, `ema`, …) — tutti i 7356 componenti osservati ricadono in `cross_pair`/`cross_triple`/`other`. Non è un difetto della funzione: è esattamente il tipo di fatto — invisibile guardando un `AlphaContract` alla volta — che mettere in pool tutti i componenti di tutti i contratti e interrogarli in formato long rende visibile.

---

## 6. Roadmap — issue #237

`forgedge.playground` segue una checklist aperta di 11 casi d'uso unici, tracciata su GitHub come issue #237. Quattro sono implementati (questo documento li copre per intero, §3-4); sette restano da progettare e implementare:

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

Chi implementa una voce di questa checklist dovrebbe seguire gli stessi tre principi del §2 (input `Iterable[ForgeResult]`, output long-format, "salta non sollevare" dichiarato esplicitamente) e aggiungere la funzione al file `m{N}.py` corrispondente al modulo di origine dei dati che legge principalmente — così come `m0.py`/`m2.py` fanno oggi.

---

## 7. Stabilità e garanzie

- **Non fa parte dell'API core.** `forge()`, `RuleDiscovery`, `RuleRegistry` e gli altri componenti M0-M4 sono l'API stabile della libreria; `forgedge.playground` è uno strato diagnostico costruito sopra di essa, esplicitamente più giovane e più soggetto a evolvere — nuove funzioni si aggiungono seguendo la checklist del §6, e le firme di quelle esistenti possono affinarsi mano a mano che emergono nuovi casi d'uso reali.
- **Nessun effetto collaterale.** Ogni funzione è una pura trasformazione di dati già calcolati; non rilancia mai `forge()` né alcun modulo M0-M4, non muta i `ForgeResult` che riceve, non ha stato tra una chiamata e l'altra.
- **Nessuna dipendenza aggiuntiva.** Come il resto della libreria, solo `numpy`/`pandas`.
- **Nessuna colonna/campo qui documentato è garantito stabile allo stesso livello dei campi documentati nel manuale principale** (`docs/manual-en.md` §9) — questo documento riflette lo stato del modulo al momento della scrittura; per il comportamento esatto verificare sempre il docstring della funzione nella versione installata.
