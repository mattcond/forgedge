# FORGE — Playground: helper di analisi sopra ForgeResult

`forgedge.playground` è un piccolo strato di funzioni di sola lettura che
prendono **R** — uno o più oggetti `ForgeResult`, l'output di
`forge()`/`forge_multi()` messo in pool da un'intera sessione di ricerca —
e li trasformano in `pandas.DataFrame` in formato long che rispondono a
domande trasversali sul *comportamento della pipeline stessa*: quanto è
nervoso un confine di regime, perché Rule Discovery scarta i contratti di
grado A, quali famiglie di feature Alpha Discovery non riesce mai a
orientare. Nessuna funzione qui dentro chiama mai `forge()`, `RuleDiscovery`
o alcun altro componente M0–M4 — ogni funzione legge solo attributi già
presenti sugli oggetti `ForgeResult` che le passi.

Questa è una guida all'utilizzo: firme, parametri, colonne restituite ed
esempi verificati. Per la motivazione di design (perché il formato long,
perché `Iterable[ForgeResult]`, l'algoritmo interno di ciascuna funzione)
vedi `src/forgedge/docs/modules/Playground.md`.

**Questo modulo è esplicitamente uno strato diagnostico, non un'API core
stabile al livello di `forge()` o `RuleDiscovery`.** La sua checklist di
tracciamento (issue GitHub #237) di 11 casi d'uso pianificati è ora completa
— tutte e 10 le funzioni sotto più il trasversale `conversion_funnel` sono
implementate — ma le firme possono ancora affinarsi man mano che emergono
nuovi casi d'uso reali (vedi la storia del bug di bucketing delle famiglie
sotto `undetermined_direction_by_family`). Controlla i docstring della
versione installata se una firma qui sotto sembra disallineata.

---

## Utilizzo di base

```python
from forgedge import forge
from forgedge.playground import *   # l'import previsto — __all__ è esplicito

result_ada = forge(kpi_ada, ticker="ADAUSDC", timeframe="1D")
result_btc = forge(kpi_btc, ticker="BTCUSDC", timeframe="1D")

results = [result_ada, result_btc]   # metti in pool tutti i ForgeResult che hai — "R"

regime_transitions(results)
regime_time_share(results)
discard_reasons_by_grade(results, grade="A")
undetermined_direction_by_family(results)
```

Ogni funzione prende la stessa lista `results` — costruiscila una volta per
sessione (tra ticker, tra run ripetute nel tempo) e passala a qualunque
funzione risponda alla domanda che hai.

---

## `regime_transitions(results: Iterable[ForgeResult]) -> pd.DataFrame`

Log in formato long di **ogni** flip di regime osservato, con la lunghezza
del run che lo ha preceduto.

Legge `result.enriched` (la KPI Table dopo Market Context); **salta**
silenziosamente qualunque risultato il cui `enriched` non abbia una colonna
`regime` (Market Context disabilitato) invece di sollevare.

**Colonne restituite:** `ticker`, `bar_index` (posizione intera di riga del
flip), `timestamp`, `from_regime`, `to_regime`, `run_length_before` (barre
consecutive nel regime di partenza, incluso l'ultimo bar prima del flip).

**Risoluzione del timestamp**, in ordine di preferenza: il `DatetimeIndex`
del frame se presente, altrimenti la colonna `open_dt`, altrimenti la
posizione intera del bar stessa — mai un errore, così funziona anche su
frame costruiti a mano in un test o in un notebook.

Una serie a regime costante restituisce un `DataFrame` vuoto (con le colonne
corrette, mai `None`). Un regime `NaN` subito dopo l'inizio della serie non
conta mai come flip *dal* `NaN`.

```python
df = regime_transitions(results)
df[df["run_length_before"] <= 2].groupby("ticker").size()   # ranking dei ticker per nervosismo dei confini
```

**Verificato**, mettendo in pool una run `forge()` reale su `ADAUSDC` (il
fixture di riferimento del repository) con una seconda serie sintetica
etichettata `BTCUSDC`:

```
df.shape == (236, 6)
df[df["run_length_before"] <= 2].groupby("ticker").size()
# ADAUSDC    45
# BTCUSDC    41
```

---

## `regime_time_share(results: Iterable[ForgeResult]) -> pd.DataFrame`

Quota in formato long di barre che ciascun ticker passa in ciascun regime —
sulle sole barre classificate (non-`NaN`) di quel risultato.

Stessa regola di skip di `regime_transitions`: qualunque risultato senza
colonna `regime` viene saltato silenziosamente.

**Colonne restituite:** `ticker`, `regime`, `n_bars` (conteggio assoluto),
`share` (in `[0, 1]`).

```python
df = regime_time_share(results)
df.sort_values("share", ascending=False).groupby("ticker").head(1)   # regime dominante per ticker
```

**Verificato**, stesso pool a due ticker:

```
top = share.sort_values("share", ascending=False).groupby("ticker").head(1)
top[["ticker", "regime", "share"]]
#  ticker      regime    share
# BTCUSDC STRONG_BEAR 0.438776
# ADAUSDC STRONG_BEAR 0.407029
```

Entrambi i ticker di questo esempio passano oltre il 40% della propria
storia in `STRONG_BEAR` — esattamente il tipo di segnale per cui questa
funzione esiste: qualunque regola scoperta su questi due asset merita una
verifica esplicita di quanto sia stata condizionata da un solo regime.

---

## `discard_reasons_by_grade(results: Iterable[ForgeResult], grade: str = "A") -> pd.DataFrame`

Scomposizione in formato long del *perché* Rule Discovery emette verdetto
`NON-EDGE` sui contratti alpha di un dato grade.

Legge `result.rule_responses` (ogni contratto promosso abbinato al proprio
verdetto di Rule Discovery — **non** `result.edges()`, che per costruzione
non contiene mai `NON-EDGE`). Mantiene i contratti il cui grade combacia
(case-insensitive) e il cui `response.verdict == "NON-EDGE"`, poi esplode
`rejection_reasons` una riga per ragione. Un contratto con
`rejection_reasons` vuota emette comunque una riga, con `reason=None` — non
viene mai scartato silenziosamente. Un contratto senza `alpha_score` (mai
gradato) non soddisfa mai alcun filtro di grade.

**Colonne restituite:** `ticker`, `alpha_id`, `event_candidate_id`, `reason`,
`failed_condition` (da `response.entry_optimization.failed_condition` quando
quell'oggetto esiste, altrimenti `None`).

```python
df = discard_reasons_by_grade(results, grade="A")
df["reason"].value_counts()                        # quali ragioni dominano
pd.crosstab(df["failed_condition"], df["reason"])   # incrocio con l'esito di entry_mode="auto"
```

**Verificato**, stesso pool a due ticker, `grade="B"`:

```
df.shape == (561, 5)
df["reason"].value_counts().head(3)
# total_trades 4 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    34
# total_trades 9 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    20
# total_trades 6 < 10 (first train window sized for 10 trades at min_tpm=2 (95% Poisson margin), not significant)    19
```

Il floor sul numero di trade nella prima finestra di walk-forward (issue
#217 — vedi il manuale, §9, "Un floor irraggiungibile è una finestra, non
un verdetto") domina nettamente le ragioni di scarto dei contratti di grado
B su questo fixture — un fatto che nessun singolo elenco
`RuleDiscoveryResponse.rejection_reasons` isolato rende visibile da solo.

---

## `undetermined_direction_by_family(results: Iterable[ForgeResult]) -> pd.DataFrame`

Legame in formato long tra la famiglia semantica di una feature sorgente e
la `direction` derivata del contratto risultante (inclusa
`"undetermined"`).

Legge `result.contracts` (ogni contratto valutato, promosso e rigettato
indistintamente — **non** `result.promoted`). Risolve l'`EventCandidate` di
origine di ciascun contratto via `event_candidate_id`; un contratto il cui
id non risolve contro `result.candidates` viene saltato silenziosamente
(nessuna riga, nessuna eccezione). Emette **una riga per componente**
dell'espressione del candidato, così una famiglia che compare solo dentro
un evento AND composto viene comunque contata, invece di sparire dietro
l'`event_id` del composto.

**Classificazione della famiglia**, esatta e sensibile all'ordine: un
componente con `len(source_cols) == 2` è `"cross_pair"`; `== 3` è
`"cross_triple"`; qualunque altra lunghezza (incluse le feature native, di
arità 1) ricade su un match del nome `{base}_{indicatore}_{periodo}` (es.
`close_rsi_25` → `"rsi"`), con `"other"` come fallback finale solo quando
quel regex non trova corrispondenza.

**Colonne restituite:** `ticker`, `alpha_id`, `event_candidate_id`,
`family`, `direction`.

```python
df = undetermined_direction_by_family(results)
df.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())
```

**Verificato**, su `ADAUSDC` da solo:

```
fam = undetermined_direction_by_family([result_ada])
fam.shape == (7356, 5)
rate = fam.groupby("family")["direction"].apply(lambda s: (s == "undetermined").mean())
rate.sort_values(ascending=False)
# family
# cross_triple    0.945455
# ret             0.915367
# cross_pair      0.915001
# vol             0.903101
# other           0.892857
# mdd             0.850000
```

Il tasso di `undetermined` su questo fixture sta in una fascia stretta
85-95% su ogni famiglia raggiunta — nessuna famiglia spicca come
affidabilmente orientabile su questi dati. (Una versione precedente di
questo documento riportava qui solo `cross_pair`/`cross_triple`/`other`,
senza mai una famiglia nativa — era il sintomo di un bug reale di
classificazione, da allora corretto; vedi
`src/forgedge/docs/modules/Playground.md` §4 per la storia completa. I
numeri sopra sono della funzione corretta, verificati di nuovo contro una
run reale.)

---

## `dead_event_candidates(results: Iterable[ForgeResult]) -> pd.DataFrame`

Classificazione in formato long del destino di ogni candidato sopravvissuto al gate in M2.

Legge `result.contracts` (indicizzato per `event_candidate_id`) e
`result.candidates` (ogni `EventCandidate` già passato attraverso il
Consistency Gate), poi etichetta ogni candidato `"dead"` (zero contratti
derivati), `"undetermined_only"` (contratti presenti, ma tutti con
`direction == "undetermined"`), o `"actionable"` (almeno un contratto con
una direzione derivata).

**Colonne restituite:** `ticker`, `event_candidate_id`, `expression`,
`n_contracts`, `n_undetermined`, `status`.

```python
df = dead_event_candidates(results)
df[df["status"] != "actionable"].groupby("ticker").size()   # spreco M1->M2 per asset
```

**Verificato**, su un nuovo pool a due ticker — `ADAUSDC` (il fixture di
riferimento del repository) più una serie sintetica `BTCUSDC`, costruito
stavolta via `forge_multi()` (non un `forge()` per ticker come negli esempi
M0/M2 sopra) così da avere un `RuleRegistry` cross-ticker pooled genuino per
gli esempi M4 sotto. Ogni blocco "Verificato" da qui alla fine di questo
documento usa lo stesso pool `forge_multi()`:

```
dead.shape == (11550, 6)
dead.groupby(["ticker", "status"]).size()
# ticker    status
# ADAUSDC   actionable             468
#           undetermined_only    4888
# BTCUSDC   actionable            3238
#           undetermined_only    2956
```

Nessuna riga `"dead"` compare per nessuno dei due ticker su questo fixture:
`len(result.contracts) == len(result.candidates)` vale esattamente per
entrambi (Alpha Discovery valuta ogni candidato sopravvissuto al gate
esattamente una volta), quindi l'unica vera separazione osservata qui è se
quell'unico contratto abbia mai ottenuto una direzione derivata.

---

## `gate_survival_observed(results: Iterable[ForgeResult]) -> pd.DataFrame`

Esito del Consistency Gate in formato long per ogni candidato grezzo valutato — passati e falliti indistintamente.

Legge `result.event_discovery.raw_events` (l'intera popolazione pre-gate,
presente quando `DiscoveryConfig.retain_raw_events=True`, il default),
ciascuno annotato con il proprio `GateResult`, insieme ai `GateParams` che
hanno deciso l'esito. Salta silenziosamente qualunque risultato in cui Event
Discovery non è stata eseguita, o è stata eseguita con
`retain_raw_events=False`.

**Colonne restituite:** `ticker`, `mean_tpm`, `index_of_dispersion`,
`episode_index_of_dispersion`, `n_episodes`, `passed`, `fail_reason`,
`min_tpm`, `max_dispersion`, `dispersion_margin`, `event_counting` — le
ultime quattro ripetono le soglie configurate su ogni riga per un confronto
diretto osservato-vs-soglia riga per riga.

```python
df = gate_survival_observed(results)
df.groupby("ticker")["passed"].mean()                                          # tasso di sopravvivenza osservato per asset
df.groupby("ticker").apply(lambda g: (g["mean_tpm"] < g["min_tpm"]).mean())     # quanto del rigetto è guidato dal tpm
```

**Verificato**, stesso pool:

```
gs.shape == (10535, 11)
gs.groupby("ticker")["passed"].mean()
# ADAUSDC    0.746607
# BTCUSDC    0.694371
gs.groupby("ticker").apply(lambda g: (g["mean_tpm"] < g["min_tpm"]).mean())
# ADAUSDC    0.044271
# BTCUSDC    0.057450
```

---

## `diagnostics_vs_verdict(results: Iterable[ForgeResult]) -> pd.DataFrame`

Legame in formato long tra i diagnostics non bloccanti di M2 e il verdetto di M3.

Esplode `AlphaContract.diagnostics` — osservazioni che informano il grade
alpha ma non bloccano nulla in M2 — contro il `RuleDiscoveryResponse.verdict`
che M3 ha assegnato più tardi allo stesso contratto. Un contratto senza
diagnostics emette comunque una riga, con `diagnostic=None`.

**Colonne restituite:** `ticker`, `alpha_id`, `grade`, `diagnostic`, `verdict`.

```python
df = diagnostics_vs_verdict(results)
pd.crosstab(df["diagnostic"], df["verdict"], normalize="index")   # quali diagnostics inclinano verso NON-EDGE
```

**Verificato**, stesso pool:

```
dv.shape == (5145, 5)
dv["diagnostic"].value_counts(dropna=False).head(3)
# NaN                                                                     1883
# OOS sample too small for reliable statistics (n_oos_activations=7 < 10)  143
# OOS sample too small for reliable statistics (n_oos_activations=8 < 10)  142
```

Ogni diagnostic non `NaN` su questo fixture è una variante dello stesso
avviso sulla dimensione del campione OOS — un'illustrazione concreta del
tipo di pattern che questa funzione esiste per far emergere: questa
formulazione esatta, a questa frequenza, sarebbe una candidata forte a
diventare un vero gate M2.

---

## `lottery_only_winners(results: Iterable[ForgeResult]) -> pd.DataFrame`

Flag in formato long per i contratti `PARTIAL-EDGE` bloccati solo dal rotation null a livello di ricerca.

Filtrato ai soli contratti `verdict == "PARTIAL-EDGE"`. `rotation_only` è
vero quando `rejection_reasons` ha esattamente un elemento e inizia con
`"search-level rotation null not cleared"` — un contratto che ha superato
ogni gate economico/statistico e ha perso solo la lotteria del
multiple-testing, a differenza di uno che fallisce ancora su PF, DSR,
consistenza OOS, ecc.

**Colonne restituite:** `ticker`, `alpha_id`, `grade`, `rotation_p`,
`rotation_threshold`, `n_reasons`, `rotation_only`.

```python
df = lottery_only_winners(results)
df.groupby("grade")["rotation_only"].mean()
```

**Verificato**, stesso pool:

```
lw.shape == (96, 7)
lw["rotation_only"].value_counts()
# False    89
# True      7
```

`lw.shape[0] == 96` combacia con il conteggio totale di edge su entrambi i
ticker (`44 + 52`, vedi `conversion_funnel` sotto) — su questo fixture, con
configurazione di default (senza preset), **ogni** contratto tradeable
finisce a `PARTIAL-EDGE`, mai a un `EDGE` pieno (vedi `monitoring_manifest`
in `deployment_it.md` per lo stesso fatto dal lato deployment).

---

## `classification_by_grade(registries: Iterable[RuleRegistry]) -> pd.DataFrame`

Legame in formato long tra il grade alpha di origine di una regola e la sua classificazione cross-ticker.

A differenza di ogni altra funzione di questo documento, questa (e
`duplicate_clusters` sotto) prende oggetti `RuleRegistry` direttamente, non
`ForgeResult` — la classificazione cross-ticker vive sul registro **pooled**
che `forge_multi()` restituisce separatamente (ogni `ForgeResult` per-ticker
ha `.registry = None` su quel percorso). Passa `[result.registry]` per una
singola run `forge()`, oppure `[registry]` per un registro pooled da
`forge_multi()`. Una riga per `RuleDocument` con `classification` non
`None` (`None` quando lo Step 4 non è mai girato).

**Colonne restituite:** `rule_id`, `source_ticker`, `grade`, `classification`.

```python
df = classification_by_grade(registries)
pd.crosstab(df["grade"], df["classification"], normalize="index")   # le regole di grado A inclinano verso GENERIC?
```

**Verificato**, sul registro pooled `forge_multi()` su ADAUSDC + BTCUSDC:

```
cbg.shape == (96, 4)
pd.crosstab(cbg["grade"], cbg["classification"])
# classification  GENERIC  ISOLATED
# grade
# A                     8        46
# B                    13        23
# C                     2         4
```

Su questo fixture compaiono solo `GENERIC`/`ISOLATED` — nessuna regola
finisce nella via di mezzo (`PARTIAL`/`SPECIFIC`) tra "tiene su ogni altro
ticker" e "non tiene su nessuno". Il grado A *non* inclina di più verso
`GENERIC` qui (8/54 ≈ 15%) rispetto al grado B (13/36 ≈ 36%) — semmai il
contrario, su questo pool a due ticker — un fatto visibile solo ponendo la
domanda che questa funzione esiste per porre, non qualcosa che il grade
alpha da solo avrebbe previsto.

---

## `duplicate_clusters(registries: Iterable[RuleRegistry]) -> pd.DataFrame`

Esito della deduplicazione in formato long per ogni regola nel registro.

Stesso input `Iterable[RuleRegistry]` di `classification_by_grade` sopra
(vedi quella sezione per il perché). Una riga per `RuleDocument`, senza
filtro, che segnala se è stata marcata duplicata e, in caso affermativo,
in quale `rule_id` sopravvissuto è stata assorbita.

**Colonne restituite:** `rule_id`, `source_ticker`, `grade`, `is_duplicate`,
`duplicate_of`.

```python
df = duplicate_clusters(registries)
df["is_duplicate"].mean()                                                              # tasso di dedup complessivo
df[df["is_duplicate"]].groupby("duplicate_of").size().sort_values(ascending=False)     # cluster di assorbimento più grandi
```

**Verificato**, stesso registro pooled:

```
dc.shape == (96, 5)
dc["is_duplicate"].mean()
# 0.5104166666666666
dc[dc["is_duplicate"]].groupby("duplicate_of").size().sort_values(ascending=False).head(5)
# duplicate_of
# RULE_ADA_23    3
# RULE_ADA_16    2
# RULE_BTC_28    2
# RULE_BTC_29    2
# RULE_ADA_41    2
```

Poco più della metà di ogni regola tradeable messa in pool tra i due ticker
(`51%`) è marcata duplicata su questo fixture — coerente con il default
`block_duplicate=True` di `promotion_gate` in `forgedge.deployment` che
blocca la maggioranza dei contratti da solo con questo flag (vedi
`src/forgedge/docs/specs/deployment_it.md`).

---

## `conversion_funnel(results: Iterable[ForgeResult]) -> pd.DataFrame`

Conteggio in formato long del funnel end-to-end per asset, attraverso ogni modulo — l'unico caso d'uso non ancorato a un singolo modulo.

Una riga per `(ticker, stage)` con la dimensione della popolazione a quella
tappa: `candidates` (sopravvissuti al gate M1), `contracts` (ogni
valutazione M2, promossa o rigettata), `promoted` (ipotesi passate a M3),
`edges` (verdetti `EDGE`/`PARTIAL-EDGE` di M3, `result.edges()`).

**Colonne restituite:** `ticker`, `stage`, `n`.

```python
df = conversion_funnel(results)
df.pivot(index="ticker", columns="stage", values="n")   # la tabella funnel, una riga per ticker
```

**Verificato**, stesso pool:

```
cf.pivot(index="ticker", columns="stage", values="n")
# stage    candidates  contracts  edges  promoted
# ADAUSDC        5356       5356     44      468
# BTCUSDC        6194       6194     52     3238
```

`candidates == contracts` esattamente per entrambi i ticker — lo stesso
fatto su cui si appoggia `dead_event_candidates` sopra (ogni candidato
sopravvissuto al gate è valutato da Alpha Discovery esattamente una volta).
Il tasso di conversione da `promoted` a `edges` è molto più basso su
`BTCUSDC` (3238 → 52, ~1.6%) che su `ADAUSDC` (468 → 44, ~9.4%) nonostante
`BTCUSDC` promuova quasi sette volte più contratti — un esempio visibile del
perché un conteggio grezzo di `promoted` da solo sovrastima quanto della
ricerca di una sessione paghi davvero.

---

## Cosa manca ancora

La checklist di tracciamento dell'issue #237 è ora completa — tutti gli 11
casi d'uso tra M0-M4 più il funnel trasversale sopra sono implementati. Il
seguito è una questione diversa: portare una regola scoperta in
**produzione** (gate di qualità, export su disco, indicizzazione per il
monitoraggio) vive nel modulo gemello `forgedge.deployment`, non qui — vedi
`src/forgedge/docs/specs/deployment_it.md` per la guida all'utilizzo e
`src/forgedge/docs/modules/Playground.md` §10 / `modules/Deployment.md` per
il perché dello spostamento (issue #245, PR #247).
