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

**Questo modulo è esplicitamente ancora in evoluzione, non un'API core
stabile al livello di `forge()` o `RuleDiscovery`.** Segue una checklist
aperta di 11 casi d'uso pianificati (issue GitHub #237); quattro sono
implementati oggi. Controlla i docstring della versione installata se una
firma qui sotto sembra disallineata.

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

## Cosa manca ancora

Altri sette casi d'uso restano aperti sulla stessa checklist di tracciamento
(issue #237) — due per M1, due per M3, due per M4, e un caso trasversale sul
tasso di conversione end-to-end che attraversa ogni modulo. Vedi
`src/forgedge/docs/modules/Playground.md` §5 per la tabella roadmap completa
e i principi di design che un nuovo caso d'uso deve seguire.
