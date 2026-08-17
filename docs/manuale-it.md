# Il Manuale di forgedge

*Una guida completa e pratica alla libreria Python FORGE (Feature-Oriented Rule Generation Engine).*

Versione coperta: `forgedge==0.1.3`. Ogni esempio di codice in questo manuale è stato eseguito realmente contro la libreria presente in questo repository; ogni numero citato da un'esecuzione è un valore reale e verificato, non un'illustrazione inventata. Quando il manuale afferma qualcosa che gli autori stessi hanno documentato (a differenza di qualcosa dedotto dalla lettura del codice), lo dichiara esplicitamente.

---

## Indice

1. [Introduzione](#1-introduzione)
2. [Perché esiste FORGE](#2-perché-esiste-forge)
3. [Quando usarla — e quando no](#3-quando-usarla--e-quando-no)
4. [Concetti fondamentali](#4-concetti-fondamentali)
5. [Installazione](#5-installazione)
6. [Il primo esempio funzionante](#6-il-primo-esempio-funzionante)
7. [Quick start: un'esecuzione completa della pipeline](#7-quick-start-unesecuzione-completa-della-pipeline)
8. [Anatomia del workflow](#8-anatomia-del-workflow)
9. [API e componenti principali](#9-api-e-componenti-principali)
10. [Configurazione](#10-configurazione)
11. [Gestione degli errori](#11-gestione-degli-errori)
12. [Casi d'uso progressivi](#12-casi-duso-progressivi)
13. [Lavorare con i dati presenti in questo repository](#13-lavorare-con-i-dati-presenti-in-questo-repository)
14. [Scelte di design](#14-scelte-di-design)
15. [Comportamenti opt-in](#15-comportamenti-opt-in)
16. [Trade-off](#16-trade-off)
17. [Performance e scalabilità](#17-performance-e-scalabilità)
18. [Testing](#18-testing)
19. [Integrare forgedge in un'applicazione reale](#19-integrare-forgedge-in-unapplicazione-reale)
20. [Un'architettura production-ready](#20-unarchitettura-production-ready)
21. [Troubleshooting](#21-troubleshooting)
22. [Best practice](#22-best-practice)
23. [Anti-pattern](#23-anti-pattern)
24. [FAQ](#24-faq)
25. [Glossario](#25-glossario)
26. [Reference API (consultazione rapida)](#26-reference-api-consultazione-rapida)

---

## 1. Introduzione

`forgedge` è una libreria Python che prende una tabella di dati di prezzo storici più indicatori tecnici (una **KPI Table**) e scopre sistematicamente **regole di trading booleane** — espressioni come `rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118` — che hanno un potere predittivo misurabile e confermato out-of-sample sul movimento futuro del prezzo.

Lo fa attraverso una pipeline a cinque stadi, ognuno dei quali risponde a esattamente una domanda e passa allo stadio successivo un oggetto formale e ispezionabile:

```
KPI Table  →  [Market Context]  →  [Event Discovery]  →  [Alpha Discovery]  →  [Rule Discovery]  →  [Rule Registry]
 (input)         tag di regime       eventi booleani       target economico     EDGE / NON-EDGE      catalogo + report
```

L'output di una sessione `forgedge` non è una previsione e non è un modello addestrato. È un insieme di **specifiche formali di regole** — una condizione booleana, una direzione (long/short), un periodo di detenzione, una percentuale di take-profit, e un verdetto sostenuto da statistiche walk-forward out-of-sample. Cosa farne di questa specifica (costruire un generatore di segnali, collegarla a un motore di esecuzione, revisionarla manualmente) è interamente a tua discrezione; `forgedge` non piazza ordini, non parla con un exchange e non conserva alcuno stato di posizione.

Questo manuale presuppone che tu sia uno sviluppatore Python competente che non ha mai visto questa libreria prima. Non presuppone che tu conosca già il gergo della finanza quantitativa — ogni termine (KPI Table, evento, alpha, walk-forward, ConsistencyGate, rotation null...) viene introdotto prima di essere usato. Alla fine, dovresti essere in grado di installare la libreria, eseguire una sessione di discovery completa su dati reali, comprendere ogni parametro di configurazione principale e cosa costa cambiarlo, e avere un piano concreto per integrare `forgedge` in un'applicazione più grande.

---

## 2. Perché esiste FORGE

La documentazione stessa della libreria ([`README.md`](../README.md)) enuncia il problema direttamente:

> "La ricerca sistematica di edge soffre di tre problemi ricorrenti: **look-ahead bias** — soglie di evento calibrate osservando rendimenti che già 'conoscono' il futuro prima della discovery; **ottimizzazione in-sample** — soglie e orizzonti tarati sulla stessa finestra usata per la valutazione producono backtest circolari; **separazione operativa mancante** — l'evidenza statistica di potere predittivo non è la stessa cosa della profittabilità sotto commissioni e meccaniche d'ordine reali."

L'architettura è la risposta a questi tre problemi, non un effetto collaterale. La guida architetturale approfondita inclusa nel repository (`src/forgedge/docs/README.md`, in italiano) lo dichiara come un vincolo esplicito e non negoziabile, non come una convenzione di codifica:

> "FORGE mantiene una separazione netta tra tre domini operativi. **Ogni confine è un vincolo architetturale, non una convenzione.**" I tre domini sono: **struttura temporale** (Event Discovery osserva solo il pattern temporale degli indicatori), **potere predittivo statistico** (Alpha Discovery è il primo stadio che legge un rendimento forward), e **operatività** (Rule Discovery e Rule Registry sono gli unici stadi che conoscono commissioni, fill e drawdown).

Concretamente, questo significa:

- **Event Discovery (Modulo 1) non calcola mai un rendimento forward.** Le espressioni booleane che estrae sono scelte esclusivamente da come il valore di un indicatore si è comportato storicamente — mai da cosa è successo dopo. È dichiarato nel codice sorgente come un fatto architetturale, non una scelta di tuning: *"Event Discovery lavora completamente cieco rispetto al target economico... Questo non è un dettaglio implementativo — è un vincolo architetturale che elimina una categoria intera di look-ahead bias"* (`src/forgedge/docs/README.md`).
- **Le soglie sono fissate una volta scoperte e non vengono mai ritarate sui risultati.** Volere una soglia "migliore" significa rieseguire Event Discovery su una finestra in-sample diversa — mai modificare un candidato già scoperto.
- **Il target economico (per quanto tempo tenere, in che direzione, quale take-profit) è derivato dai dati da Alpha Discovery, mai assunto dall'utente.** Questo impedisce che un'intuizione economica dell'utente (es. "sicuramente un movimento del 2% in 10 barre") polarizzi silenziosamente quali eventi sembrano buoni.
- **Rule Discovery è l'unico giudice economico.** Un'espressione booleana può apparire statisticamente eccellente nel Modulo 2 ed essere comunque rigettata nel Modulo 3 perché non è profittevole una volta simulate meccaniche d'ordine realistiche (fill a limite, commissioni, ritardi).

### Il nome e la metafora

La guida architetturale del repository è esplicita sul fatto che il nome sia una metafora deliberata, non un gioco di parole su un acronimo: "Come una fucina metallurgica trasforma il minerale grezzo in uno strumento lavorato attraverso fasi successive, FORGE trasforma una tabella di indicatori tecnici in regole booleane operative attraverso quattro moduli in sequenza — senza mai fare assunzioni sul sistema di esecuzione che le utilizzerà." (`src/forgedge/docs/README.md`)

---

## 3. Quando usarla — e quando no

### Usa `forgedge` quando...

- Hai (o puoi costruire) una tabella di barre OHLCV storiche più indicatori tecnici per uno o più ticker, e vuoi **cercare sistematicamente** condizioni booleane che storicamente hanno preceduto un movimento di prezzo statisticamente significativo — invece di scegliere a mano soglie di indicatori dall'intuizione.
- Vuoi che questa ricerca porti **garanzie statistiche oneste contro l'overfitting** per costruzione: conferma out-of-sample, validazione walk-forward, una correzione per multiple-testing a livello di ricerca (la rotation null, §14-15), e un controllo Deflated Sharpe Ratio — non aggiunti a posteriori.
- Vuoi un **output verificabile**: ogni candidato porta le proprie ragioni di rifiuto, ogni contratto registra perché è stato o non è stato promosso, e gli artefatti intermedi della pipeline (`ForgeResult.candidates`, `.contracts`, `.event_frame`) restano ispezionabili dopo l'esecuzione.
- Il tuo budget di dipendenze è limitato. `forgedge` dipende **solo da `numpy` e `pandas`** — nessun `scipy`, nessun `statsmodels`, nessun framework ML. Tutte le primitive statistiche (correlazione di Spearman, t-test, funzione beta incompleta, controllo FDR di Benjamini-Hochberg, regressione dell'half-life di un processo OU) sono reimplementate in numpy puro.

### NON usare `forgedge` per...

- **Esecuzione di ordini o gestione di posizioni.** Non ha connettività verso exchange, non piazza ordini, non ha stato di portfolio. È dichiarato esplicitamente e ripetutamente nella documentazione del repository: *"FORGE non esegue ordini, non gestisce posizioni, non si connette a exchange... È un sistema di ricerca, non di esecuzione"* (`src/forgedge/docs/README.md`). Se ti serve questo, `forgedge` produce la *specifica* (una `ValidatedRule`, un insieme di parametri entry/exit) che un sistema di esecuzione separato implementa.
- **Generazione di segnali basata su machine learning.** I documenti di design (`docs/analysis/forge2_functional_analysis.md`) elencano esplicitamente questa direzione come rifiutata, con una motivazione dichiarata: "Niente ML/feature learning nella discovery. Il valore differenziante è che ogni regola è un'espressione booleana leggibile e auditabile; un modello addestrato romperebbe il contratto molto più di qualunque bug." Se cerchi una libreria che adatta un classificatore o una rete neurale ai dati di prezzo, questa non lo è, per scelta di design.
- **Un singolo indicatore/soglia che sai già di voler testare.** Il valore di `forgedge` sta nella ricerca sistematica più le garanzie statistiche; se hai già un'ipotesi specifica (es. "verifica se RSI < 30 predice un rimbalzo"), il meccanismo `CustomEvent` della libreria (§9, §12 Caso d'uso 5) permette di iniettarla direttamente, ma useresti solo una piccola frazione di ciò che la libreria fa.
- **Ricerca su microstruttura sub-secondo/tick-level.** Ogni esempio funzionante, ogni calibrazione statistica, e lo studio di robustezza a bassa frequenza della libreria stessa (`docs/analysis/lowfreq_robustness.md`) sono costruiti e testati su barre orarie-giornaliere. Nulla nel codice vieta tecnicamente altre frequenze, ma vedi §16 (Trade-off) e §21 (Troubleshooting) per i problemi molto reali di potere statistico che compaiono su finestre in-sample brevi.
- **Qualsiasi cosa richieda un catalogo persistente e cross-sessione di regole scoperte già pronto all'uso.** Il Modulo 4 (Rule Registry) è esplicitamente stateless e ricostruito da zero a ogni sessione — la persistenza è responsabilità dell'*applicazione host* (§19-20), non della libreria.

---

## 4. Concetti fondamentali

Prima di toccare qualsiasi API, costruisci questo modello mentale. `forgedge` è organizzata attorno a tre concetti formali, ciascuno il deliverable di uno stadio della pipeline, più un concetto di input e uno di supporto:

| Concetto | Prodotto da | Risponde a |
|---|---|---|
| **KPI Table** | tu (o `forgedge.kpi_builder`) | "Quali dati sto passando alla pipeline?" |
| **Regime** | Modulo 0 — Market Context | "In che tipo di condizione di mercato si trova la barra *t*?" |
| **Evento** (`EventCandidate`) | Modulo 1 — Event Discovery | "Questa condizione sull'indicatore è stabile e ripetibile nel tempo?" |
| **Alpha** (`AlphaContract`) | Modulo 2 — Alpha Discovery | "Dato che l'evento si è attivato, cosa succede statisticamente nelle barre successive?" |
| **Regola** (`RuleDiscoveryResponse`) | Modulo 3 — Rule Discovery | "Questo alpha è effettivamente profittevole con meccaniche d'ordine realistiche?" |

### KPI Table

Un `pandas.DataFrame` con:

- una colonna **`close`** (float, richiesta da ogni modulo),
- una **sorgente datetime** — o una colonna (nome default `open_dt`) o un `DatetimeIndex`,
- un numero qualsiasi di **colonne feature**: RSI, EMA, Bollinger Bands, volatilità, spread, geometria delle candele — qualunque cosa vuoi che `forgedge` consideri.

`forgedge` classifica automaticamente ogni colonna non-timestamp (numerica continua, discreta/binaria, categoriale) e decide come trasformarla — non le dici tu quali colonne usare come feature.

### Evento

Un **evento** è una condizione booleana su barre storiche — es. `rsi_14 < 31.2` o la composizione AND `rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118` — scoperta puramente dalla struttura temporale di una o più colonne indicatore, **senza mai guardare un rendimento forward**. Le sue soglie sono *distribuzionali* (percentili della storia stessa dell'indicatore su quello specifico asset) piuttosto che costanti hardcoded, ed è ciò che rende trasferibile la *procedura* di discovery tra asset diversi anche se la soglia *numerica* risultante differisce per asset. Le soglie di un evento sono immutabili nel momento in cui Event Discovery le ha fissate.

### Alpha

Un **alpha** è la risposta empirica a: *dato che questo evento si è appena attivato, cosa succede, in media, nelle prossime h barre?* Alpha Discovery deriva — mai assume — tre cose per ogni evento: il miglior periodo di detenzione `h*`, una direzione (long o short), e un livello di take-profit `sell_pct`. Misura il potere predittivo (Information Coefficient, lift sopra il base rate, Cohen's d) e lo conferma su una coda out-of-sample. L'unica cosa che può rigettare in modo definitivo un alpha è una **direzione indeterminata** — nessun orizzonte nella griglia scansionata produce un vantaggio finito e a segno determinato. Ogni altra debolezza statistica (IC basso, lift basso, conferma OOS fallita) diventa un diagnostico non bloccante che abbassa il voto in lettere (A-D) del contratto invece di rigettarlo del tutto.

### Regola

Una **regola** è il verdetto operativo su un contratto alpha, prodotto simulando un backtest realistico — ingresso a ordine limite, uscita a take-profit, uno stop basato sull'orizzonte, commissioni per lato — e poi validando il set di parametri selezionato su uno split **walk-forward** out-of-sample rolling. Il verdetto è una di quattro stringhe: `"EDGE"`, `"PARTIAL-EDGE"`, `"NON-EDGE"`, o `"INSUFFICIENT-DATA"` (§8, §9, §15). Rule Discovery è esplicitamente l'unico giudice economico della pipeline — un alpha con un profilo statistico perfetto può comunque risultare `NON-EDGE`.

### Flusso, input e invarianti in sintesi

```
   KPI Table (fornita da te, o costruita con kpi_builder)
        │
        ▼
   Modulo 0 — classifica il regime di mercato di ogni barra
        │  output: + 'regime' (categoriale ordinata a 5 livelli), + 'regime_stable'
        ▼
   Modulo 1 — estrae eventi booleani SOLO dalla storia degli indicatori
        │  output: list[EventCandidate]   (soglie ormai congelate)
        ▼
   Modulo 2 — deriva un target economico per evento, misura il potere predittivo
        │  output: list[AlphaContract]    (h*, direzione, sell_pct — tutti derivati)
        ▼
   Modulo 3 — backtest realistico + validazione walk-forward OOS
        │  output: EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA
        ▼
   Modulo 4 — dedup, test di generalizzazione cross-ticker, catalogo + report HTML
```

I dati fluiscono sempre solo in avanti. Nessun modulo successivo torna a toccare gli interni di un modulo precedente, e — questa è l'invariante da interiorizzare prima di scrivere qualsiasi codice contro questa libreria — **nessun modulo può accedere a informazioni che, cronologicamente, il suo predecessore non avrebbe potuto avere**. Questa singola regola è ciò per cui esiste l'intera architettura.

---

## 5. Installazione

`forgedge` richiede **Python ≥ 3.9** e dipende solo da `pandas>=1.5` e `numpy>=1.23` (da `pyproject.toml`). Non ha alcuna dipendenza opzionale necessaria per la pipeline principale.

```bash
pip install forgedge
```

Per eseguire la suite di test o contribuire, installa l'extra `dev`:

```bash
pip install "forgedge[dev]"     # aggiunge pytest>=7.0
```

Se lavori da un clone di questo repository (come sono stati verificati gli esempi di questo manuale):

```bash
git clone https://github.com/mattcond/forgedge
cd forgedge
pip install -e ".[dev]"
```

Alcune cose che `forgedge` **non** richiede, che potresti aspettarti: nessun `scipy`, nessun `statsmodels`, nessun database, nessun file di configurazione, nessuna variabile d'ambiente, nessun accesso di rete, nessuna GPU. È una libreria stateless, di puro calcolo — tutto ciò di cui ha bisogno è ciò che le passi in memoria.

Se prevedi di leggere il fixture parquet usato in questo manuale (`tests/fixtures/ADA_1D_TRAIN.parquet`), ti serve anche un motore parquet — `forgedge` stessa non lo richiede, ma `pandas.read_parquet` sì:

```bash
pip install pyarrow
```

**Verificare l'installazione:**

```python
import forgedge
print(forgedge.__version__)   # "0.1.3" al momento della scrittura
```

---

## 6. Il primo esempio funzionante

Questo è il più piccolo pezzo di codice possibile che dimostra l'idea centrale della libreria: scoprire un *evento booleano dalla storia di un indicatore*, senza alcun rendimento forward coinvolto. Usa il Modulo 1 (Event Discovery) in isolamento, su una piccola tabella sintetica, così puoi vedere esattamente com'è fatto un `EventCandidate` prima che entri in gioco qualunque nozione di profittabilità.

```python
import numpy as np
import pandas as pd
from forgedge import EventDiscovery

# Una KPI Table minima: 500 barre orarie con un oscillatore simile a RSI.
rng = np.random.default_rng(42)
n = 500
kpi = pd.DataFrame({
    "open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
    "close": 100 * np.cumprod(1 + rng.normal(0, 0.004, n)),
    "close_rsi_14": np.clip(50 + rng.normal(0, 15, n).cumsum() * 0.05, 1, 99),
})

ed = EventDiscovery(kpi)
candidates = ed.run()

print(f"{len(candidates)} candidati evento hanno superato il Consistency Gate")

c = candidates[0]
print("expression:", c.expression)
print("attivazioni:", c.activation_stats.n_activations)
print("trade/mese medi:", round(c.activation_stats.mean_tpm, 3))
```

**Output verificato** (l'autore di questo manuale ha eseguito esattamente questo codice contro esattamente questo repository — i tuoi numeri potrebbero differire leggermente con una versione diversa di numpy, dato che il flusso dell'RNG può cambiare tra versioni, ma la *forma* è rappresentativa):

```
309 candidati evento hanno superato il Consistency Gate
expression: close crosses_below 95.45
attivazioni: 10
trade/mese medi: 10.0
```

`candidates[0]` è un evento di tipo "crossing" costruito direttamente sulla colonna grezza `close`, non — come si potrebbe immaginare — uno costruito su `close_rsi_14`. Controllare `ed.get_classifications()` su questa esatta esecuzione rivela il perché, ed è una sorpresa genuinamente utile:

```python
print(ed.get_classifications())
# {'close': ColumnClassification(..., is_scale_free=True, ...),
#  'close_rsi_14': ColumnClassification(..., is_scale_free=False, ...)}
```

Questo è l'**opposto** dell'intuizione naturale (di sicuro un oscillatore limitato è "scale-free" e una serie di prezzo grezza no?). È un risultato reale e verificato, e vale la pena usarlo per correggere quell'intuizione invece di nasconderlo: il test di scale-free (`TypeClassifier._is_scale_free`, `event_discovery/classifier.py`) non guarda se una serie è limitata in `[0,100]`. Divide la serie in blocchi e controlla se il *range di valori* di ogni blocco si sovrappone abbastanza con quello degli altri (`support_overlap_threshold`, default 0.5) — cioè "questa serie occupa più o meno lo stesso range per tutto il campione, o deriva verso nuovi livelli nel tempo?" Il `close_rsi_14` sintetico di questo esempio giocattolo è stato costruito come `50 + rng.normal(0,15,n).cumsum()*0.05` — un wobble **cumulativo** lento che deriva attraverso il suo range limitato in modo a blocchi, non stazionario — quindi ha fallito quel test, mentre il random walk geometrico lievemente derivante di `close` è risultato passarlo su questo specifico seed. **La lezione si generalizza oltre questo esempio giocattolo:** la classificazione scale-free è un vero test statistico sui tuoi dati effettivi, non un lookup basato sul nome della colonna o sui suoi limiti nominali — un vero RSI calcolato correttamente probabilmente si classificherebbe diversamente, ma non dovresti assumere la classificazione di nessuna colonna senza controllare `ed.get_classifications()`.

### Cosa è successo, internamente

1. `EventDiscovery(kpi)` ha classificato entrambe le colonne come `CONTINUOUS`, poi ha eseguito il test scale-free su ciascuna indipendentemente — vedi il riquadro sopra per cosa misura effettivamente quel test. Il flag scale-free governa se una colonna è idonea a costruire feature *ratio/spread* contro altre colonne della stessa famiglia (Step 2 sotto); non determina se una colonna può avere propri eventi diretti di soglia/crossing, ed è per questo che `close` — pur non essendo ciò che chiameresti casualmente "scale-free" — ha comunque prodotto il candidato `close crosses_below 95.45` in cima alla lista di questa esecuzione.
2. Ha generato **versioni trasformate** di ciascuna colonna: rank percentile mobile, z-score mobile e delta semplici, su diverse lunghezze di finestra.
3. Per ogni serie (base o trasformata), ha provato un catalogo di **soglie distribuzionali** (es. il 10° percentile della storia stessa di quella serie) e **soglie teoriche** (livelli z-score fissi come −2.0), ognuna delle quali produce un'espressione booleana candidata — un evento "crossing" sul valore grezzo di una colonna (come con `close` qui), oppure qualcosa nella forma `close_rsi_14 < P10 [P10=...]` per una serie trasformata (presente tra i 309 candidati di questa esecuzione — filtra con `[c for c in candidates if "rsi" in c.expression]` per trovarne uno).
4. Ogni candidato è passato attraverso il **Consistency Gate**: si attiva abbastanza spesso (`min_tpm`), in modo abbastanza consistente tra i mesi (non bursty — `max_dispersion`), e con abbastanza osservazioni totali da essere statisticamente significativo? I candidati che falliscono vengono scartati silenziosamente — non diventano mai oggetti `EventCandidate`.
5. **In nessun punto di tutto questo si è guardato il valore futuro di `close`.** Il gate ispeziona solo *quando* l'evento si è attivato, mai *cosa è successo dopo*.

### Configurazione implicita da notare

Hai chiamato `EventDiscovery(kpi)` senza argomento `config=`. Questo significa che sono stati usati silenziosamente i default di `DiscoveryConfig()`:

- `train_ratio=1.0` — è stata usata l'*intera* tabella per la discovery, nessuno split OOS riservato (non c'è validazione walk-forward in corso qui; vedi §10 per come abilitarla).
- `gate_params=GateParams()` — le soglie di default del Consistency Gate: `min_tpm=0.5` (almeno 0.5 "episodi" qualificanti al mese), `max_dispersion=1.5` (le attivazioni non devono essere troppo bursty), `event_counting="episode"` (§15).
- `max_and_components=2` — Event Discovery ha anche provato a comporre coppie di eventi a singola colonna con AND, soggette allo stesso gate.

Nessuna di queste scelte ha coinvolto il rendimento forward di `close` — quel concetto non esiste ancora a questo stadio della pipeline.

---

## 7. Quick start: un'esecuzione completa della pipeline

La sezione 6 ha mostrato un modulo in isolamento. Questa sezione esegue l'**intera pipeline** con la singola funzione orchestratrice, `forge()`, su dati reali inclusi in questo repository: `tests/fixtures/ADA_1D_TRAIN.parquet` — 882 barre giornaliere OHLCV di ADA (Cardano) più 22 colonne di indicatori tecnici precalcolate, che coprono dal 2024-01-01 al 2026-05-31. Questo è lo stesso fixture su cui la suite di regressione golden della libreria stessa fissa i propri valori attesi.

```python
import pandas as pd
from forgedge import forge

kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
print(kpi.shape)                      # (882, 26)
print(list(kpi.columns)[:6])          # ['open_dt', 'high', 'low', 'close', 'open', 'close_ret_03']

result = forge(kpi, ticker="ADAUSDC", timeframe="1D", progress=False)

print("Candidati M1:", len(result.candidates))
print("Promossi M2: ", len(result.promoted))
print("Risposte M3: ", len(result.rule_responses))
print("tradabili (edges()):", len(result.edges()))
```

**Output verificato** (l'autore di questo manuale ha eseguito esattamente questo codice contro esattamente questo repository):

```
(882, 26)
['open_dt', 'high', 'low', 'close', 'open', 'close_ret_03']
Candidati M1: 5241
Promossi M2:   370
Risposte M3:   370
tradabili (edges()): 54
```

### Interpretare l'output

- **5241 candidati evento** hanno superato il Consistency Gate di Event Discovery — ricorda, nessuno di questi è stato ancora verificato contro un rendimento forward.
- **370 di essi sono stati promossi** da Alpha Discovery a oggetti `AlphaContract` con stato "HYPOTHESIS" — cioè ciascuno ha una direzione determinata (long o short) e un periodo di detenzione/take-profit derivato.
- **370 risposte di regola** — ogni contratto promosso è passato attraverso il backtest realistico e la validazione walk-forward di Rule Discovery (questo fixture ha `run_rule_discovery=True` di default).
- **54 sono tradabili** (`result.edges()` — verdetto `EDGE` o `PARTIAL-EDGE`). Su questo specifico dataset con impostazioni di default, scavando un livello più a fondo si scopre che *tutti e 54* sono `PARTIAL-EDGE`, non `EDGE` pieno:

```python
from collections import Counter
print(Counter(r.verdict for _, r in result.rule_responses))
# Counter({'NON-EDGE': 314, 'PARTIAL-EDGE': 54, 'INSUFFICIENT-DATA': 2})
```

Zero verdetti `EDGE` pieni non è un bug e non è un segno che la libreria "non funzioni" — è il gate della rotation null di default (§14-15) che fa esattamente ciò per cui è progettato. Guarda il singolo miglior candidato `PARTIAL-EDGE` su questi dati:

```python
partial = [(c, r) for c, r in result.rule_responses if r.verdict == "PARTIAL-EDGE"]
partial.sort(key=lambda x: x[1].in_sample_summary.profit_factor, reverse=True)
c, r = partial[0]
print(c.event_expression, "|", c.direction)
print(r.in_sample_summary.profit_factor, r.in_sample_summary.total_trades)
print(r.walk_forward.consistency, r.walk_forward.oos_summary.profit_factor)
print(r.rejection_reasons)
```

Output verificato:

```
delta_diffnorm_close_vol12_vol24_6 < -0.899244 | short
16.882 46
1.0 9.721
['active_months 11/20 = 55% < 80%', 'search-level rotation null not cleared (rotation_p=1.0000 > 0.05)']
```

Questa regola ha un profit factor in-sample eccezionale (16.9), un walk-forward *positivo nel 100% delle finestre di test*, e un profit factor OOS di 9.7 — ed è comunque limitata a `PARTIAL-EDGE`. Le `rejection_reasons` dicono esattamente perché: è attiva solo nel 55% dei mesi della sua finestra (sotto la soglia di copertura dell'80%), e — la ragione più importante — il suo p-value della rotation null a livello di ricerca è 1.0, cioè il test null di rotazione randomizzata proprio di FORGE (§14) ha scoperto che una versione ruotata e disaccoppiata dall'esito della stessa ricerca fa altrettanto bene o meglio. Questo è la libreria che è onesta sulla dimensione del proprio spazio di ricerca, non un falso negativo.

### Cosa ha fatto `forge()`, che non le hai chiesto esplicitamente

Questo è importante, ed è la prima cosa che sorprende i nuovi utilizzatori. `forge(kpi, ticker="ADAUSDC", timeframe="1D")` senza ulteriore configurazione ha fatto silenziosamente tutto quanto segue:

1. Eseguito il Modulo 0 (Market Context) perché non hai passato `run_market_context=False` e la tabella non aveva già una colonna `regime`.
2. Sostituito una `horizon_grid` **calibrata sul giornaliero**, per Alpha Discovery, perché `timeframe="1D"` è giornaliero-o-più-lento e non hai passato un `AlphaConfig` esplicito — la griglia di default della classe `(1,2,3,4,6,8,12,16,24,36,48)` è calibrata per barre approssimativamente orarie, e usarla invariata su dati giornalieri scansionerebbe periodi di detenzione fino a 48 *giorni*.
3. **Arricchito** la griglia di orizzonti di ogni evento con punti aggiuntivi attorno a 0.5×/1×/2× la finestra dell'indicatore dominante di quell'evento (`AlphaConfig.horizon_enrichment`, attivo di default) — un'unione con la griglia base, mai una restrizione.
4. Eseguito la **rotation null veloce a livello di ricerca** (`fast_null=True` di default) e annotato `rotation_p`/`rotation_threshold` su ogni contratto promosso — questo è esattamente ciò che ha prodotto il tetto `PARTIAL-EDGE` appena visto.
5. Costruito un `TimeBudget` condiviso e **purgato** per Event/Alpha Discovery (§15) anche se non hai passato alcun argomento `time_budget=`.
6. Registrato un `HypothesisLedger` su `result.ledger`, contando quanto fosse effettivamente ampia la superficie di ricerca della sessione.
7. Eseguito Rule Discovery su tutti i 370 contratti promossi con `selection_mode="walk_forward"` (il default) — cioè i parametri operativi pubblicati provengono solo dall'interno delle finestre di train walk-forward, mai da uno sguardo alla finestra di test finale.
8. Saltato il Modulo 4 (Rule Registry) — non perché tu l'abbia disabilitato, ma perché `RuleRegistry.from_forge_results` ha bisogno di più ticker per dire qualcosa sulla generalizzazione cross-ticker; con una chiamata `forge()` a singolo ticker viene comunque eseguito e produce un registry, ma ogni regola è classificata `ISOLATED` (§9).

Nessuno dei punti 2-7 è qualcosa che hai configurato tu. Sono tutti default scelti dagli autori della libreria specificamente affinché il percorso "quick start" e il percorso "configurato a mano" non divergano silenziosamente in onestà. §14-15 spiegano ciascuno di questi in profondità, incluso quali puoi disattivare e cosa costa farlo.

---

## 8. Anatomia del workflow

Questa sezione percorre i cinque moduli end-to-end, con abbastanza dettaglio meccanico da poter prevedere cosa farà cambiare un input — non solo come si chiama il metodo pubblico di ogni modulo.

### Modulo 0 — Market Context

**Input:** la KPI Table. **Output:** la stessa tabella più `regime` (una categoriale ordinata: `STRONG_BEAR < BEAR < NEUTRAL < BULL < STRONG_BULL`) e `regime_stable` (bool).

Internamente, il classificatore di default (`EMAProxyClassifier`) calcola `ratio = ema_short / ema_long` e lo colloca in bucket contro quattro soglie (default in modalità fissa `[0.975, 0.990, 1.010, 1.025]`). La parte interessante è *da dove vengono le finestre EMA*: di default (`auto_window=True`), il modulo stima l'half-life locale di un processo mean-reverting Ornstein-Uhlenbeck dalla serie di prezzo stessa (via regressione Hurst/OU: `dP_t = const + kappa·P_{t-1} + ε`, `half_life = -log(2)/log(1+kappa)`), poi deriva `long_period = round(half_life)` e `short_period = round(half_life × 0.435)`. Solo se questa stima non converge, ricade sui default fissi 9/25. `mc.window_resolution["source"]` dice cosa è successo: `"hurst_ou"` (convergenza), `"fallback"`, o `"configured"` (hai impostato `auto_window=False` e dato finestre esplicite).

`regime_stable` è `True` solo una volta che il regime corrente si è mantenuto per almeno `stable_window` (default 12) barre consecutive — le prime 11 barre di una transizione di regime fresca sono `regime_stable=False`.

**Importante:** Event Discovery (Modulo 1) non legge affatto `regime` — è presente nella tabella ma ignorato durante gli Step 0-5. Solo l'analisi di sensibilità al regime di Alpha Discovery (Step 5 della propria pipeline) lo legge.

### Modulo 1 — Event Discovery

È una pipeline interna a cinque step (`EventDiscovery.run()`), ed è utile conoscere gli step per nome perché i messaggi di errore e i parametri di configurazione vi fanno riferimento:

1. **Classificazione delle colonne** (`TypeClassifier`) — ogni colonna non-timestamp viene etichettata `CONTINUOUS`, `BINARY` (esattamente 2 valori distinti), o `CATEGORICAL` (non numerica, o numerica con ≤ `max_categorical_classes` valori distinti, default 20). Le colonne continue ricevono un ulteriore flag "scale-free": questa serie è limitata/intrinseca (RSI, una percentuale) o dipendente dal livello di prezzo (una EMA grezza)? L'euristica è deliberatamente conservativa — preferisce perdere una serie scale-free (falso negativo) piuttosto che etichettare erroneamente come scale-free una serie dipendente dal prezzo (falso positivo), perché quest'ultima genererebbe soglie contaminate dal livello di prezzo assoluto.
2. **Generazione di feature** (`FeatureGenerator`) — deriva nuove colonne di arità 2-3: ratio (`a/b`), spread percentuali (`(a-b)/b`), differenze normalizzate (`(a-b)/σ(a-b)`), %B di Bollinger, e posizione-nel-range. La maggior parte delle combinazioni di arità 2 accoppia colonne della stessa famiglia semantica (due EMA, non una EMA e un RSI), ma il generator ha anche diversi **pairing di arità 2 dedicati e a scope ristretto** oltre a questa regola generale, ciascuno aggiunto per colmare uno specifico limite che la regola di famiglia non poteva raggiungere:
   - **coppie OHLC cross-colonna, cross-tempo** — es. "la chiusura di oggi sopra il minimo di ieri" (ogni coppia ordinata di basi OHLC grezze presenti, contro una copia laggata, agli stessi lag usati altrove per le trasformazioni delta); sempre attivo.
   - **una linea MACD contro la propria signal line** — abbinate per `(base, fast, slow)` condivisi, non dal raggruppamento generico same-family, perché una linea deve accoppiarsi con la *propria* signal line, mai con quella di una configurazione MACD non correlata; si attiva solo se hai abilitato `"macd"` in `build_features()` (disabilitato di default, §9).
   - **variazione % di prezzo contro variazione % di volume allo stesso lookback** — un segnale di divergenza prezzo-volume (es. "nuovo massimo di prezzo su volume in calo"); si attiva solo se la tua KPI Table porta una colonna di rendimento basata sul volume, che non fa parte della config default di `kpi_builder`.
   - **le sei colonne di geometria di `candle_features()` tra loro, e contro `close_natr_N`** — queste colonne dal nome nudo (`body`, `gap`, …) non rispettano affatto la convenzione `{base}_{indicatore}_{periodo}`, quindi senza questo pairing dedicato potrebbero essere usate solo standalone; accoppiate contro l'ATR *normalizzato* (`natr`), mai quello grezzo, per evitare di reintrodurre una dipendenza dal livello di prezzo.
   - **un indicatore price-scale (solo SMA/EMA/WMA/HMA) contro una base OHLC grezza laggata** — es. `close_sma_12[t] > low[t-3]`; deliberatamente ristretto alle famiglie di indicatori price-scale (un ratio contro colonne RSI/volatilità/rendimento/drawdown/Bollinger non sarebbe dimensionalmente sensato) e alla sola operazione `ratio_`. Attivo di default, con un proprio set di lag — `DiscoveryConfig.indicator_lag_cross_lags`, default `(1, 3)` (§10) — distinto dal set di lag generale delle trasformazioni delta.

   Tutte queste tranne la prima sono aggiunte genuinamente recenti (arrivate dopo che gran parte del walkthrough del Modulo 1 di questo manuale era già stata redatta, e corrette qui dopo aver riletto direttamente il sorgente attuale, secondo la disciplina di verifica propria di questo manuale, §21). Quelle vincolate a una config `kpi_builder` non-default (MACD, rendimenti di volume, ATR/NATR) non costano nulla su una KPI Table di default; il pairing indicatore-vs-OHLC-base-laggata è incondizionato e aggiunge un costo misurabile — quantificato in §17.
3. **Trasformazioni temporali** (`TransformLayer`) — ogni feature dagli step 1-2 riceve anche versioni con rank percentile mobile, z-score mobile e delta semplici, su diverse lunghezze di finestra (48/96/168 barre per pctrank/zscore; 1/3/6/12 barre per delta).
4. **Generazione eventi** (`EventGenerator`) — per ogni coppia (feature, trasformazione), prova un catalogo di soglie distribuzionali (percentili p3…p97 della storia stessa di quella serie) e — solo per le serie z-scored — soglie teoriche fisse (±1.0, ±1.5, ±2.0). Ogni soglia produce sia un evento "persistente" (attivo su ogni barra in cui vale la condizione) sia un evento "crossing" (attivo solo sulla barra in cui la condizione diventa vera per la prima volta).
5. **Consistency Gate** (§10, §15) — ogni candidato evento grezzo deve superare un filtro di rate/dispersione prima di diventare un `EventCandidate`. Gli eventi singoli che superano il gate vengono poi combinati a coppie (o triple, se `max_and_components=3`) con AND, e ogni composizione viene ripresentata allo stesso gate — solo le composizioni che a loro volta lo superano diventano candidati AND-composti.

**Nulla in questi cinque step legge un rendimento forward.** Questo è il singolo fatto più importante su questo modulo.

### Modulo 2 — Alpha Discovery

Data la lista di candidati dal Modulo 1, e *per la prima volta nella pipeline*, questo modulo legge il percorso futuro del prezzo. Per ogni candidato evento:

1. **Derivazione di orizzonte e direzione.** Per ogni orizzonte `h` nella griglia (eventualmente arricchita), calcola `mean_advantage[h]` (il vantaggio medio orientato del rendimento forward delle barre attive rispetto a tutte le barre) e un punteggio di selezione dell'orizzonte `|mean_advantage[h]| / sqrt(h)` — una deflazione "Sharpe-like" che evita di favorire orizzonti brevi solo perché il denominatore della loro t-statistic è più piccolo. `h* = argmax_h score[h]`. La direzione è `"long"` se `mean_advantage[h*] > 0`, `"short"` se `< 0`, e `"undetermined"` — l'unico gate di rigetto rigido della pipeline — se nessun orizzonte nella griglia produce affatto un vantaggio finito e a segno determinato.
2. **Derivazione del take-profit.** `sell_pct = max(quantile(MFE, mfe_quantile=0.5), mfe_floor=0.005)`, dove MFE è la massima escursione favorevole entro `h*` barre da ogni barra attiva. Questo è un numero *derivato*, non un input di configurazione.
3. **Misurazione del potere predittivo (IS):** Information Coefficient (correlazione di Spearman tra la feature grezza e il rendimento forward, calcolata una volta e messa in cache per `(feature, orizzonte)`), win rate/lift sopra il base rate, Cohen's d, un t-test a una coda.
4. **Conferma OOS** (quando `train_ratio < 1.0`, default 0.7): lo stesso target derivato viene replicato sulla coda tenuta fuori, e passa se ha abbastanza attivazioni, un vantaggio orientato positivo, e un p-value abbastanza basso. Fallire qui è un **diagnostico non bloccante**, non un rigetto.
5. **Sensibilità al regime** — IC per regime e win rate, con una classificazione `dependency_type` (`agnostic`/`conditional`/`specific`/`broken`/`unknown`).
6. **Scoring composito** — una combinazione pesata delle metriche sopra in un `composite_score` 0-1, mappato su un voto in lettere da A (≥0.75) a D (<0.25).
7. **Compilazione del contratto.** Tutti i candidati con una direzione determinata diventano oggetti `AlphaContract` con `status="HYPOTHESIS"`; ogni altra metrica sopra aggiunge solo una stringa a `diagnostics` — non blocca mai la promozione, e `rejection_reasons` resta vuoto su un contratto promosso. Questo è dichiarato come un principio di design deliberato: le debolezze statistiche "alimentano il voto, non scartano — Rule Discovery è l'unico giudice economico" (`src/forgedge/docs/README.md`).

### Modulo 3 — Rule Discovery

Dato un `AlphaContract` promosso e il suo `EventCandidate` di origine, Rule Discovery simula un backtest realistico di esecuzione ordini e lo valida out of sample. **Non** ri-ottimizza le soglie dell'evento né sovrascrive il target derivato — usa `derived_target.holding_period_h`/`sell_pct` solo come *centro* di una griglia di parametri operativi da esplorare.

Le meccaniche di esecuzione, esattamente:

1. Su una barra di segnale, viene piazzato un ordine limite a `anchor × (1 − buy_drop_pct)` (long) o `anchor × (1 + buy_drop_pct)` (short).
2. Se il prezzo tocca quel limite entro `buy_delay_bar` barre, si riempie; altrimenti l'ordine viene cancellato (da qui viene il fill rate).
3. Dopo un fill, la posizione si chiude a qualunque cosa avvenga prima: la prima barra che chiude oltre il livello di take-profit, o la chiusura della barra `target_h` dopo il fill (lo "stop di orizzonte").

Una sottigliezza da interiorizzare con precisione: **`target_h` conta le barre *dopo* la barra di fill**, e il gap segnale→fill è sempre esattamente 1 barra (non puoi agire sulla chiusura di una barra prima che sia avvenuta) — quindi lo span totale segnale-uscita è `1 + target_h` barre. `target_h=0` è legale e significa "esci alla chiusura stessa della barra di fill", non "nessun orizzonte".

La griglia di combinazioni `(buy_drop_pct, sell_pct, target_h, buy_delay_bar)` viene vagliata in-sample, la combinazione con il punteggio migliore diventa il punto operativo, e quel punto operativo fisso viene poi rivalidato su uno split walk-forward rolling (finestre di train che si espandono o scorrono, `n_splits` finestre di test concatenate in un unico "track record OOS onesto"). La validazione statistica sopra questo include un Deflated Sharpe Ratio, un controllo di stabilità temporale (PF prima metà vs. seconda metà), e una scomposizione per dipendenza dal regime.

La logica del verdetto (gate esatti, dai default di `SelectionCriteria`) è coperta in §9 e §15 — versione breve: `NON-EDGE` è un qualsiasi fallimento rigido (troppo pochi trade, PF IS sotto 1.5, PF OOS sotto 1.0, expectancy non significativa); `EDGE` richiede il superamento di ogni gate di un insieme più stringente incluso il controllo della rotation null; qualsiasi cosa superi il pavimento `NON-EDGE` ma non ogni gate `EDGE` è `PARTIAL-EDGE`; e `INSUFFICIENT-DATA` (§15) è un quarto verdetto che declassa un `EDGE`/`PARTIAL-EDGE` altrimenti valido quando l'evidenza OOS aggregata è statisticamente troppo debole per sostenere l'affermazione, indipendentemente da quanto buona appaia la stima puntuale.

### Modulo 4 — Rule Registry

Ingerisce ogni submission di regola `EDGE`/`PARTIAL-EDGE` (NON-EDGE viene silenziosamente saltato, non è un errore) in un `RuleDocument`. Calcola due matrici di correlazione tra tutti i documenti (overlap di Jaccard delle date di attivazione; correlazione di Spearman dei gain su un asse di date condiviso e zero-paddato), segnala — ma non elimina mai — il più debole di ogni coppia di regole quasi-duplicate (Jaccard ≥ `overlap_threshold`, default 0.70), e riesegue ogni regola su ogni *altro* ticker della sessione con le sue soglie assolute ri-percentilate sulla distribuzione di quel ticker. Una regola è `GENERIC` se supera la barra di profit factor cross-ticker su almeno `generic_ratio_threshold` (default esattamente `2/3`) degli altri ticker testati; altrimenti `PARTIAL`, `SPECIFIC`, o (se segnalata come duplicato) `ISOLATED`. Il registry è interamente stateless tra sessioni — la tabella piatta esportata o il report HTML sono gli unici artefatti di persistenza che produce.

---

## 9. API e componenti principali

Questa sezione copre le classi e funzioni principali rivolte all'utente — non un elenco esaustivo (quello è §26), ma abbastanza per scrivere codice reale contro ogni modulo. Ogni firma qui sotto è citata dal sorgente attuale, non parafrasata.

### L'orchestratore

```python
forge(
    kpi_table: pd.DataFrame, *,
    ticker: str | None = None, asset: str = "ASSET", timeframe: str = "1H",
    market_context_config: MarketContextConfig | None = None,
    event_discovery_config: DiscoveryConfig | None = None,
    alpha_config: AlphaConfig | None = None,
    rotation_calibration: RotationConfig | None = None,
    fast_null: bool = True,
    time_budget: TimeBudget | None = None,
    rule_discovery_config: RuleDiscoveryConfig | None = None,
    registry_config: RegistryConfig | None = None,
    manual_events: list[CustomEvent] | None = None,
    run_market_context: bool = True,
    run_rule_discovery: bool = True,
    run_registry: bool = True,
    only_validated_events: bool = False,
    rule_discovery_grades: Iterable[str] | None = None,
    progress: bool = True,
) -> ForgeResult
```

`manual_events` e `event_discovery_config` sono mutuamente esclusivi — passare entrambi solleva `ValueError`. `ticker` ricade su `alpha_config.asset`, poi su `asset`, quando non passato esplicitamente.

`ForgeResult` — il valore di ritorno — porta ogni artefatto intermedio:

| Campo | Tipo | Significato |
|---|---|---|
| `enriched` | `pd.DataFrame` | KPI Table dopo Market Context |
| `event_frame` | `pd.DataFrame` | frame post-pipeline di Event Discovery — **passa questo, non `enriched`**, a qualunque cosa richieda feature derivate |
| `candidates` | `list[EventCandidate]` | ogni output del Modulo 1 |
| `contracts` | `list[AlphaContract]` | ogni output del Modulo 2, promossi *e* rifiutati |
| `promoted` | `list[AlphaContract]` | il sottoinsieme con `status="HYPOTHESIS"` |
| `rule_responses` | `list[tuple[AlphaContract, RuleDiscoveryResponse]]` | una coppia per contratto promosso, se M3 è stato eseguito |
| `registry` | `RuleRegistry \| None` | output del Modulo 4, se eseguito |
| `calibration` | `CalibrationReport \| None` | il report della rotation null |
| `ledger` | `HypothesisLedger \| None` | contabilità della superficie di ricerca |
| `time_budget` | `TimeBudget \| None` | lo split IS/OOS effettivo usato |
| `market_context`, `event_discovery`, `alpha_discovery` | istanze modulo | oggetti live per drill-down (`.distribution()`, `.summary()`, …) |

Metodi: `.edges()` → coppie `(contract, response)` dove `response.is_edge` è vero; `.validated_rules()`; `.submissions()`; `.summary()` (un `pd.DataFrame`, una riga per candidato, arricchito con `rule_verdict`).

`forge_multi(frames_by_ticker: dict[str, pd.DataFrame], *, registry_config=None, progress=True, **forge_kwargs) -> tuple[dict[str, ForgeResult], RuleRegistry]` esegue `forge()` una volta per ticker e mette in comune ogni regola tradabile in un unico registry cross-ticker — è il modo naturale per ottenere dal Modulo 4 regole genuinamente classificate `GENERIC`.

### Preset

```python
forge_preset(preset: str, timeframe: str, asset: str = "ASSET",
             train_ratio: float = 0.70, **overrides
             ) -> tuple[DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig]
```

`preset` è uno tra `PRESETS = ["sniper", "balanced", "sweep", "burst"]`. Restituisce una *tripla* — un oggetto config calibrato per modulo (M1/M2/M3) — pretarato per un *profilo* di ricerca, e scalato per il timeframe che passi. `preset_info(preset=None)` stampa i parametri numerici risolti per uno o tutti i preset.

### Modulo 0 — `MarketContext`

```python
MarketContext(kpi_table: pd.DataFrame, config: MarketContextConfig | None = None)
mc.run() -> pd.DataFrame
mc.distribution()          # quota di barre per regime, per diagnostica
mc.window_resolution       # {"source": "hurst_ou" | "fallback" | "configured", ...}
```

### Modulo 1 — `EventDiscovery`, `EventCandidate`, `CustomEvent`

```python
EventDiscovery(kpi_table: pd.DataFrame, config: DiscoveryConfig | None = None,
                time_budget: TimeBudget | None = None)
ed.run() -> list[EventCandidate]
ed.df           # frame post-pipeline — passa questo ad AlphaDiscovery, non kpi_table
ed.summary()    # pd.DataFrame, una riga per candidato
```

Gli attributi importanti di un `EventCandidate`: `event_id`, `expression` (la condizione booleana, come stringa), `event_formula` (una resa human-readable), `sql_expression` (una traduzione SQL compatibile con DuckDB della stessa condizione, utile se vuoi valutare l'evento fuori da Python), `components`, `activation_stats` (`n_activations`, `n_active_months`, `zero_months`, `max_monthly_share`, `mean_tpm`), `consistency_gate` (un `GateResult`), `validation` (un `ValidationResult`, solo se il walk-forward era configurato). Il suo metodo `.apply(df) -> pd.Series[bool]` rivaluta deterministicamente le soglie *memorizzate* su qualsiasi nuovo frame — nessuna ricalibrazione, nessun look-ahead — e `.persist(path)` dà un round-trip pickle completo (l'unico metodo che la documentazione descrive come completamente invertibile; la forma JSON di `.to_dict()` non lo è).

```python
CustomEvent(formula: str, name: str = "")
```

Per iniettare manualmente una tua ipotesi (es. `CustomEvent("close_rsi_14 < 30")`) invece di eseguire la discovery automatica. Le formule sono valutate con `pandas.DataFrame.eval()`. Un `CustomEvent` attraversa comunque il Consistency Gate, ma un fallimento registra solo un warning — non viene mai scartato. La composizione AND non viene eseguita sugli eventi iniettati manualmente. Si usa via `forge(..., manual_events=[...])` — mutuamente esclusivo con `event_discovery_config`.

### Modulo 2 — `AlphaDiscovery`, `AlphaContract`

```python
AlphaDiscovery(kpi_table_or_ed_df: pd.DataFrame, candidates: list[EventCandidate],
                config: AlphaConfig, time_budget: TimeBudget | None = None)
ad.run() -> list[AlphaContract]
ad.promoted_contracts(min_lift: float | None = None) -> list[AlphaContract]
ad.summary()   # pd.DataFrame, ordinato per composite score
```

**Fondamentale:** passa `ed.df` (il frame post-pipeline di Event Discovery), non la KPI Table originale — porta già le colonne ratio/spread/trasformazione derivate a cui le espressioni degli eventi fanno riferimento.

Gli attributi importanti di un `AlphaContract`: `alpha_id`, `status` (`"HYPOTHESIS"`/`"REJECTED"`), `event_candidate_id` (rimanda all'`EventCandidate` di origine), `derived_target` (`holding_period_h`, `sell_pct`, `direction`, `base_rate`, `mean_advantage`), `oos_validation`, `event_stats` (`win_rate`, `lift`, `cohens_d`, `p_value`), `regime_analysis`, `alpha_score` (`composite_score`, `grade`), `rejection_reasons` (solo cause bloccanti — vuoto su un contratto promosso), `diagnostics` (osservazioni non bloccanti che pesano sul grade; normalmente non vuoto sui contratti promossi), `rotation_p`/`rotation_threshold` (impostati dalla rotation null a livello di ricerca).

### Modulo 3 — `RuleDiscovery`, `RuleDiscoveryResponse`

```python
RuleDiscovery(event_frame: pd.DataFrame, contract: AlphaContract, candidate: EventCandidate,
               config: RuleDiscoveryConfig | None = None)
resp = rd.run() -> RuleDiscoveryResponse
```

L'`event_candidate` che passi deve essere quello a cui `contract.event_candidate_id` effettivamente punta, o il costruttore solleva `ValueError`.

#### Sovrapposizione — quanto capitale costano questi numeri

`run_backtest` apre una posizione su **ogni** barra attiva, senza controllo di
stato flat. È deliberato e resta così: è una politica legittima che presuppone
capitale, e l'economia riportata è riproducibile dal vivo *dato abbastanza
capitale per finanziare le posizioni concorrenti*. Quello che mancava era un
modo per sapere quanto capitale sia (issue #168) — i report portavano una frase
fissa sulle «posizioni sovrapposte» senza alcun numero, quindi una regola che
richiede 1× il capitale di una posizione e una che ne richiede 12× si leggevano
identiche.

`BacktestSummary` ora lo misura sul ledger che i parametri pubblicati producono
davvero:

| campo | a quale domanda risponde |
|---|---|
| `n_episodes` | quanto spesso *scatta* questo segnale? |
| `mean_concurrent_positions` | quando lavora, quante posizioni sto finanziando? |
| `max_concurrent_positions` | posso proprio metterla in produzione sul mio conto? |

`trades` (da `return_trades=True`) porta un `episode_id` per riga, quindi
`trades.groupby("episode_id").size()` è disponibile senza reimplementare il
raggruppamento.

**Episodi e concorrenza sono misure diverse e in generale non coincidono.**
Gli episodi raggruppano per *segnale* — una serie di cinque barre con `RSI < 30`
è una cosa che accade, non cinque. La concorrenza raggruppa per *percorso di
prezzo* — e trade di episodi nettamente distinti si sovrappongono comunque
quando l'orizzonte di holding supera il gap fra loro. Nel caso della #168: 120
barre di segnale, 76 episodi, e una media di 3.71 posizioni concorrenti.

Quale serva dipende dalla domanda: dimensionare il capitale → concorrenza;
quanto spesso scatta un segnale → episodi; inferenza statistica → concorrenza,
perché trade sovrapposti condividono un percorso di prezzo e non sono
osservazioni indipendenti. `total_trades / mean_concurrent_positions` è la
dimensione campionaria che la sovrapposizione sostiene davvero (118 nominali →
≈32 effettivi in quel caso). Le conseguenze inferenziali sono affare della
#177; questa è la misura che le serve.

`forgedge.episodes` espone le primitive — `episode_starts`, `episode_ids`,
`concurrency` — per chi le vuole direttamente.


#### Modalità d'ingresso — cosa misura il verdetto

`entry_mode` ha default **`"auto"`** (era `"limit"` prima della #185), e vale la
pena capire il cambio perché muove i verdetti.

In modalità `"limit"` la griglia varia `buy_drop_pct`, quindi l'entrata a limite
fa due lavori insieme: meccanica d'ordine *e* ottimizzatore del prezzo
d'ingresso. Uno sconto più profondo riempie di rado e — qui sta il punto —
**solo sui percorsi che sono tornati giù a prenderlo**. Il profit factor sale su
un sottoinsieme di trade che non è la popolazione tradeable. È il *fill
confound*, e sotto `"limit"` il verdetto misura in parte il prezzo d'ingresso
invece del segnale.

`"auto"` separa le due letture:

- **Stage 1** valuta la regola con entrata market (fill al next-open, ≈100%).
  Questo verdetto è autoritativo. Lo Stage 2 non può mai trasformare un
  `NON-EDGE` in un edge.
- **Stage 2** sweepa `buy_drop_pct` sui sopravvissuti, **replaya** il vincitore
  fuori campione sulle stesse finestre di test dello Stage 1, e lo pubblica solo
  se supera tutte e tre le condizioni di adozione.

Le condizioni, tutte misurate su quel replay:

| # | condizione | cosa impedisce |
|---|---|---|
| 1 | `fill_rate >= min_fill_rate_opt` | un PF gonfiato da fill rari |
| 2 | `opportunity_sharpe >= quello del market` | un punto che trada meno per un edge poco migliore |
| 3 | `net_gain >= min_net_gain_retention × quello del market` | un µ minuscolo con un σ minuscolo |

La condizione 2 usa uno **Sharpe diverso da quello su
`StatisticalValidation`**, e la differenza è tutto il criterio. `validate()`
annualizza per *capacità* — `bars_per_year / avg_holding_bars`, quanti periodi
di holding non sovrapposti stanno in un anno. È il denominatore giusto per
«quanto è buona questa regola», perché non la premia per l'accidente di essersi
attivata spesso. Ma i due punti operativi stanno sulla *stessa* regola e tengono
la posizione per la stessa durata, quindi la capacità è **identica per entrambi**
e il fattore `sqrt` si semplifica: il confronto collassa sullo Sharpe per trade,
esattamente la metrica cieca all'opportunità da cui il criterio vuole uscire.

`opportunity_sharpe` conta invece i trade realizzati — `(µ/σ) × sqrt(trade per
anno)` — quindi dimezzare i trade costa `sqrt(2) ≈ 1.41×` che la qualità per
trade deve battere. Sull'esempio della issue:

| | market | limite | rapporto |
|---|---|---|---|
| Sharpe per trade | 0.267 | 0.375 | 1.41× |
| annualizzato per capacità | 5.095 | 7.164 | 1.41× — invariato |
| `opportunity_sharpe` | 1.461 | 1.299 | **0.89×** |
| rendimento totale | 48% | 36% | **0.75×** |

La lettura per capacità adotta un punto che rende un quarto in meno.

Entrambi i punti sono riportati per intero su
`RuleDiscoveryResponse.entry_optimization` — ciascuno con la propria regola,
il proprio summary fuori campione e le proprie statistiche, più
`failed_condition` che nomina quale condizione ha fermato l'adozione. Il
walk-forward del punto limite è un *replay* (`reoptimise=False`), quindi non
aggiunge selezione né `n_trials`; la sua DSR porta il proprio conteggio di
tentativi più alto (celle Stage 1 + Stage 2) come metrica assoluta, mentre il
gate `min_dsr` legge sempre quella del punto market — il verdetto non paga mai
per lo Stage 2.

`"limit"` resta pienamente supportata ed è la scelta giusta quando l'ordine a
limite *è* la strategia, non un raffinamento dell'esecuzione.


Gli attributi importanti di una `RuleDiscoveryResponse`: `verdict` (`"EDGE"|"PARTIAL-EDGE"|"NON-EDGE"|"INSUFFICIENT-DATA"`), `is_edge` (vero per i primi due), `rejection_reasons`, `validated_rule` (porta `.params`, un `BacktestParams`), `in_sample_summary` (`total_trades`, `profit_factor`, `win_rate_pct`, `expectancy`, `tpm_mu`), `execution_envelope` (`.conservative`/`.optimistic` — vedi §17), `walk_forward` (`.oos_summary`, `.consistency`), `statistical_validation` (`.temporal_stability`, `.deflated_sharpe`), `regime_analysis`, `excursion` (MAE/MFE), `entry_optimization` (popolato solo quando `entry_mode="auto"`, §15).

`from forgedge.rule_discovery import text_report, html_report` costruiscono report human-readable/HTML da una response; `resp.to_dict()` dà una forma serializzabile in JSON.

### Modulo 4 — `RuleRegistry`

```python
RuleRegistry(submissions: list[RuleSubmission], frames: dict[str, pd.DataFrame],
              config: RegistryConfig | None = None)
RuleRegistry.from_forge_results(results: dict[str, ForgeResult], config=None)   # entry point preferito
reg = registry.run()
reg.summary(); reg.flat_table(); reg.documents; reg.matrices   # .jaccard, .spearman
reg.export("rules.xlsx")             # o .csv, secondo RegistryConfig.export_format
reg.html_report(timeframe="1H")      # HTML autocontenuto, SVG inline, nessuna CDN
```

`frames` deve contenere i frame *post-Event-Discovery* (cioè `ForgeResult.event_frame` per ticker), non KPI Table grezze — il replay cross-ticker ha bisogno delle colonne feature derivate a cui le espressioni delle regole fanno riferimento.

### KPI Builder — costruire una KPI Table da candele grezze

```python
from forgedge import build_features, candle_features, lag_features, pattern_features

kpi = build_features(candles, config=None, *, timestamp_col, output_timestamp_col="open_dt",
                      timestamp_unit="ms", add_color=True, sort_output=True) -> pd.DataFrame
kpi = candle_features(kpi, *, order_on="open_dt", add_gap=True, round_to=5) -> pd.DataFrame
kpi = lag_features(kpi, *cols, periods=(1,2,3), like=None, order_on="open_dt") -> pd.DataFrame
```

`build_features` calcola un insieme configurabile di indicatori base (SMA, EMA, RSI, Bollinger Bands, ATR, MACD, min/max mobile, rendimenti, volatilità, max-drawdown) da OHLCV grezzo, e deriva la colonna timestamp `open_dt` che `forge()` si aspetta. `config` accetta un `dict`, un percorso a file YAML, o `None` per il default pacchettizzato. Gli indicatori che referenziano colonne assenti in `candles` (es. `volume`) vengono saltati silenziosamente con un `logger.warning`, non un errore — input solo-OHLC è sicuro. `"atr"` e `"macd"` sono **disabilitati di default** in quella configurazione pacchettizzata.

`candle_features` aggiunge sei colonne di geometria candlestick scale-free (`body`, `upper_wick`, `lower_wick`, `close_pos`, `range_pct`, `gap`), tutte in `[-1,1]`/`[0,1]` indipendentemente dal livello di prezzo. `lag_features` aggiunge copie shiftate `{col}_prev_{NN}` di colonne nominate o matchate per pattern.

`pattern_features(df, *, patterns=None, order_on="open_dt", col="candle_pattern")` è una quarta funzione, deliberatamente **opt-in** — vedi §15.

**Convenzione di naming delle colonne, importante:** perché una colonna sia riconosciuta come parte di una *coppia ratio same-family* dal feature generator di Event Discovery, il nome deve seguire `{base}_{indicatore}_{periodo}` con `base ∈ {close, high, low, open, volume}` e `indicatore ∈ {ema, sma, rsi, dema, tema, wma, hma, mdd, atr, natr}` (o i pattern di naming dedicati per Bollinger/volatilità/rendimento/MACD). Una colonna che non rispetta questa convenzione funziona comunque come feature standalone e può comunque essere raggiunta da uno degli *altri* pairing di arità 2 a scope più ristretto descritti in §8 (coppie OHLC cross-tempo, MACD-vs-signal, rendimento prezzo-vs-volume, coppie di geometria di `candle_features()`, indicatore-vs-OHLC-base-laggata) — è specificamente dal raggruppamento generico same-family che resta esclusa. Se un indicatore custom che hai aggiunto non compare mai composto con nulla nei candidati di Event Discovery, controlla prima il nome contro questa convenzione.

### Coerenza della configurazione — `config_report`

```python
config_report(event_discovery=None, alpha=None, rule_discovery=None,
              registry=None, market_context=None, *, ctx=None, kpi=None,
              timeframe="1H") -> ConfigReport
```

`summary_report` valida i **dati**; `config_report` valida la
**configurazione**, con lo stesso vocabolario `Finding`. Risponde a due domande
in un solo output, perché hanno senso solo insieme: *con che configurazione sto
per girare* e *quella configurazione è internamente soddisfacibile*.

Sia lui sia `forge()` passano per lo **stesso resolver**, quindi ciò che il
report mostra è per costruzione ciò che la pipeline eseguirà — `rep.configs`
sono gli oggetti veri, non una ricostruzione. Non solleva mai, non avvisa mai e
non muta mai ciò che riceve.

```python
rep = config_report(disc, alpha, rd, kpi=kpi, timeframe="1D")
print(rep.to_text())          # la resolution trace, poi la diagnostica
if rep.has_critical:
    raise ValueError(rep.one_line())
```

Tredici vincoli, ciascuno una relazione fra materializzazioni di un parametro
latente. Tre sono `FAIL` — riservati a una configurazione che rende uno stage
**strutturalmente incapace** di produrre un verdetto: `wf_bucket_too_short`
(issue #173), `m1_oos_fold_too_short`, `oos_span_too_short`. Gli altri dieci
sono `WARN`. Ogni messaggio porta il valore da impostare, non solo il
fallimento.

> **`forge(strict=True)` è il default, ed è un cambiamento di comportamento.**
> Un `FAIL` ora solleva `ValueError` invece di girare. Un run del genere non può
> dirti nulla: ogni candidato viene eliminato per ragioni di configurazione, e il
> muro di rejection che ne esce è indistinguibile da "il segnale è brutto", che è
> proprio ciò che stavi cercando di misurare. Passa `strict=False` per degradare
> tutto a `UserWarning` e girare comunque. Le incoerenze non critiche sono sempre
> avvisi, mai errori. **Nessun verdetto cambia** — cambia che alcuni run non
> partono più.

> **Noto: `forge_preset("balanced", "1D")` è attualmente segnalato.** Ai valori
> standard del preset `min_train_months=6 × criteria.min_tpm=0.80 = 4.8` contro
> un floor di 10 — la F2 dell'audit, e il motivo per cui su dati giornalieri si
> vede l'early-elimination di massa. Viene risolto derivando `min_train_months`
> dal tasso (issue #177); fino ad allora i run col preset su daily richiedono
> `strict=False`.

#### Cosa riempie il resolver

Un campo lasciato stare non è un valore: è una domanda a cui risponde la
sessione. Questi sono i campi il cui default arriva ora dalla sessione invece
che dal corpo di una classe — impostane uno e ogni modulo che legge la stessa
grandezza lo segue.

| parametro latente | campi in cui si materializzava | default risolto |
|---|---|---|
| la colonna timestamp | `timestamp_col` su M1 / M2 / M3 / M4 | `"open_dt"` |
| la serie dei prezzi | `AlphaConfig.close_col`, `BacktestParams.{target_col, buy_price_anchor}` | `"close"` |
| le colonne di regime | `AlphaConfig.{regime_col, regime_stable_col}` | `"regime"` / `"regime_stable"` |
| la base di costo | `AlphaConfig.fee_per_side`, `BacktestParams.fee` | `0.002` |
| l'asticella di genericità | `RegistryConfig.{cross_pf_threshold, min_cross_pf_retention}` | `1.5` / `0.8` |

Due di questi erano bug veri, non ordine formale. `AlphaConfig.fee_per_side`
stampigliava il contratto mentre `BacktestParams.fee` addebitava il backtest, e
niente li collegava: `AlphaConfig(fee_per_side=0.0005)` produceva contratti che
documentavano 5 bp e backtest che ne addebitavano 20 — in silenzio, perché i due
concordavano solo condividendo un default. E `forge_preset(timestamp_col="ts")`
configurava solo M1, quindi M2 falliva più avanti chiedendo un valore che
credevi di aver già dato.

La propagazione non è simmetrica alla raccolta, in un punto deliberato, e la
ragione va detta con precisione. `buy_price_anchor` **non è un campo di schema**:
nomina il *livello di riferimento* a cui si applica l'offset del limite —
`buy_price = anchor × (1 ∓ buy_drop_pct)` — e qualsiasi colonna numerica della
tabella candele è ammessa, indicatori derivati compresi. `buy_price_anchor=
"close_sma_3"` con `buy_drop_pct=0.10` è il modo di dire *«metti un limite al 90%
della SMA a 3 barre»*; il motore non ha altro modo di esprimerlo.

Quindi l'anchor viene *riempito* dalla colonna prezzo — il suo livello di
riferimento di default è il close, e rinominare la colonna deve portarselo
dietro — ma non *semina* mai il contesto, e non viene confrontato con `close_col`.
Seminare da lì rispingerebbe `"close_sma_3"` dentro `AlphaConfig.close_col` e
farebbe misurare a M2 i rendimenti futuri su una media mobile. `target_col` è
diverso — l'uscita a orizzonte dev'essere prezzata sulla serie su cui M2 ha
misurato i rendimenti — quindi lì un disaccordo viene segnalato.

> **La genericità è ora un test di trasferibilità, non di qualità.**
> `cross_pf_threshold` aveva default `2.0` indipendente da M3, mentre
> `partial_min_profit_factor` ammette le regole a `1.5` — quindi una regola
> `PARTIAL-EDGE` avrebbe dovuto fare *meglio* fuori casa che in casa per essere
> generica, e l'intera classe era esclusa dalla genericità per costruzione. Il
> verdetto è ora `PASS ⟺ pf ≥ floor AND pf ≥ retention × pf_casa`: la metà
> assoluta chiede *è tradeable là*, quella relativa chiede *trasferisce*. La
> qualità resta sul verdetto M3 e sul grade, dove il registro già la registra.
> `CrossTickerResult.bar` riporta il numero contro cui ogni verdetto è stato
> misurato.

### Qualità dei dati — `summary_report`

```python
summary_report(df: pd.DataFrame, *, timestamp_col="open_dt",
                price_cols=("open","high","low","close"), timeframe=None,
                return_high_move=0.5, top_n=5, verbose=True,
                return_report=False) -> DataQualityReport | None
```

Un controllo diagnostico economico e **puramente consultivo** sulle colonne di prezzo — non solleva mai eccezioni e non blocca mai la pipeline. Controlla schema/NaN/infiniti, coerenza di scala del prezzo (magnitudini miste — un sintomo comune di un bug nel feed dati), coerenza interna OHLC, outlier sui rendimenti (uno z-score robusto basato su MAD più una soglia sul movimento assoluto), e continuità temporale (gap, timestamp duplicati o fuori ordine). Ogni finding è un `Finding(level, code, message)` con `level ∈ {"OK","WARN","FAIL"}`; `DataQualityReport` espone `.worst`, `.has_critical`, `.has_warnings`, `.one_line()`, `.to_text()`, e la lista completa `.findings`.

### Time budget, hypothesis ledger, calibrazione

```python
TimeBudget.build(n_bars: int, train_ratio: float = 0.7, horizon_bars: int = 0,
                  purge_bars: int | None = None, embargo_bars: int = 0) -> TimeBudget
```

`HypothesisLedger` (`result.ledger`) — semplice contabilità (`m1_candidates`, `m2_horizons`, `m2_promoted`, `m3_grid_cells`, `m2_surface`, `total_surface`), non un meccanismo di correzione.

```python
FastRotationNull(event_frame, candidates, alpha_config, time_budget=None).run(promoted) -> CalibrationReport
RotationCalibrator(event_frame, candidates, alpha_config, time_budget=None).run(promoted, RotationConfig(...)) -> CalibrationReport
```

Entrambi coperti in profondità in §14-15.

---

## 10. Configurazione

Ogni modulo accetta una dataclass che porta i suoi parametri. Questa sezione copre quelli che più probabilmente andrai effettivamente a tarare, con i default citati esattamente dal sorgente.

### `DiscoveryConfig` (Modulo 1)

| Campo | Default | Significato |
|---|---|---|
| `gate_params` | `GateParams()` | soglie del Consistency Gate — vedi sotto |
| `max_categorical_classes` | `20` | oltre questo numero di valori distinti, una colonna non numerica viene scartata, non one-hot-encoded |
| `timestamp_col` | `"open_dt"` | |
| `max_and_components` | `2` | `1`=solo singoli, `2`=+coppie, `3`=+coppie+triple |
| `train_ratio` | `1.0` | `<1.0` riserva una coda per la validazione walk-forward opzionale del Modulo 1 stesso |
| `walk_forward` | `None` | imposta un `EventWalkForwardConfig` per abilitare la validazione OOS a livello di evento (§15 — opt-in) |
| `diversity_gate_enabled` | `False` | soppressione opt-in dei quasi-duplicati (§15) |
| `diversity_threshold` | `0.85` | soglia Jaccard per il diversity gate, quando abilitato |
| `indicator_lag_cross_lags` | `(1, 3)` | set di lag per il pairing indicatore-price-scale-vs-OHLC-base-laggata (§8); passa `()` per disabilitare del tutto questo pairing |

`GateParams` (il Consistency Gate):

| Campo | Default | Significato |
|---|---|---|
| `min_tpm` | `0.5` | minimo di trigger medi al mese (l'unità dipende da `event_counting`) |
| `max_dispersion` | `1.5` | massimo Index of Dispersion consentito (Var/Mean dei conteggi mensili) |
| `event_counting` | `"episode"` | `"episode"` conta run massimali di attivazioni consecutive; `"bar"` conta ogni singola barra (§15) |
| `min_episodes` | `10` | floor assoluto sul conteggio di episodi, solo modalità `"episode"` |
| `episode_gap` | `1` | gap massimo in barre che appartiene ancora allo stesso episodio |

### `AlphaConfig` (Modulo 2)

| Campo | Default | Significato |
|---|---|---|
| `horizon_grid` | `(1,2,3,4,6,8,12,16,24,36,48)` | **calibrata sull'orario** — vedi la trappola sui dati giornalieri in §21 |
| `train_ratio` | `0.7` | split IS/OOS per la conferma propria di Alpha Discovery |
| `embargo_bars` | `0` | buffer OOS extra opt-in, §15 |
| `horizon_enrichment` | `(0.5, 1.0, 2.0)` | attivo di default; aggiunge orizzonti attorno alla finestra dominante propria di ogni evento, §15 |
| `thresholds` | `PromotionThresholds()` | soglie statistiche che guidano il voto, non un gate rigido (eccetto la direzione) |
| `fee_per_side` | `0.002` | registrata per Rule Discovery **e addebitata da lui** — non applicata qui (M2 non nettizza le fee), ma è lo stesso valore, non più una copia indipendente |
| `target_mode` | `"proj"` | scoring eccesso-sopra-trend di default per eventi long, §15 |
| `trend_sma_mult` | `2.0` | moltiplicatore della finestra SMA di trend per `target_mode="proj"` |
| `use_stable_regime_only` | `False` | opt-in, restringe l'analisi di regime alle barre `regime_stable=True` |

`PromotionThresholds`: `ic_min_abs=0.02`, `ic_max_p=0.05`, `min_lift=0.08`, `min_cohens_d=0.15`, `use_fdr=True`, `fdr_q=0.10`, `oos_max_p=0.10`, `min_direction_t=0.5`, `require_significant_direction=True`. **Solo `direction=="undetermined"` blocca la promozione** — ogni altra soglia qui informa il voto A-D.

### `RuleDiscoveryConfig` (Modulo 3)

| Campo | Default | Significato |
|---|---|---|
| `base_params` | `BacktestParams()` | punto operativo seme |
| `grid` | costruita automaticamente | griglia di ricerca attorno al target derivato del contratto quando lasciata vuota |
| `walk_forward` | `RuleWalkForwardConfig()` | `n_splits=4`, `min_train_months=6`, `purge_bars=None` (→ orizzonte testato) |
| `criteria` | `SelectionCriteria()` | gate del verdetto — vedi §15 |
| `entry_mode` | `"auto"` | `"auto"`/`"market"`/`"limit"`, §15 |
| `selection_mode` | `"walk_forward"` | punto operativo scelto solo dentro le finestre di train; `"full_sample"` è il comportamento legacy |

Default di `BacktestParams`: `direction="long"`, `buy_type="limit"`, `buy_drop_pct=0.010`, `buy_delay_bar=6`, `sell_pct=0.040`, `target_h=24`, `fee=0.002`, `early_stopping=True`.

Default di `SelectionCriteria` che più probabilmente toccherai: `min_profit_factor=2.0`, `min_win_rate=0.55`, `min_tpm=2.0` (l'unico gate di frequenza — il floor sui trade eseguiti è `max(10, n_months × min_tpm)`, non un conteggio fisso), `min_fill_rate=0.40`, `min_dsr=1.0`, `max_rotation_p=0.05`, `power_gate=True` (§15), `early_elimination=True` (imposta `False` per forzare la diagnostica completa anche su un `NON-EDGE` filtrato velocemente).

### `RegistryConfig` (Modulo 4)

`overlap_threshold=0.70` (soglia Jaccard di dedup), `cross_pf_threshold=1.5` + `min_cross_pf_retention=0.8` (le due metà del `PASS` cross-ticker — floor assoluto, e frazione del PF di casa mantenuta), `generic_ratio_threshold=2/3` (**passalo come `2/3`, non `0.67`** — una regola che passa esattamente 2-su-3 ticker ha rapporto `0.6666...`, che supera `>= 2/3` ma non `>= 0.67`; la documentazione segnala esplicitamente questo problema di precisione), `export_format="excel"`.

### Configurare via preset invece

Invece di assemblare `DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig` a mano, `forge_preset(preset, timeframe, asset, train_ratio=0.70, **overrides)` restituisce tutti e tre pretarati per un *profilo* di ricerca nominato:

```python
from forgedge import forge, forge_preset

disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="ADAUSDC")
result = forge(kpi, ticker="ADAUSDC", timeframe="1D",
               event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
               rule_discovery_config=rd_cfg)
```

| Preset | Profilo |
|---|---|
| `"sniper"` | Eventi rari, regolari, ad alta precisione, regole semplici. Richiede una finestra IS lunga (≥2 anni su 1D). Da **non** abbinare al rotation calibrator (troppo pochi eventi per calibrare). |
| `"balanced"` | Frequenza moderata, default sensato per la maggior parte di asset/timeframe. |
| `"sweep"` | Ricerca ampia e permissiva — progettata per abbinarsi a `rotation_calibration=RotationConfig(k>=100)` e a un filtro `min_lift` a valle. |
| `"burst"` | Eventi concentrati nel tempo (cambio di regime, momentum). Alta dispersione esplicitamente tollerata. |

Override accettati per nome: lato M1 — `min_tpm`, `max_dispersion`, `max_and_components`, `timestamp_col`, `event_counting`; lato M2 — `min_lift`, `min_cohens_d`, `fdr_q`, `oos_max_p`, `horizon_grid`, `bars_per_day`; lato M3 — `rd_min_tpm`. Una chiave di override non riconosciuta solleva `TypeError`.

---

## 11. Gestione degli errori

`forgedge` **non ha una gerarchia di eccezioni custom** — ogni errore sollevato è un built-in Python semplice (`ValueError`, `KeyError`, `RuntimeError`, `TypeError`, `ImportError`, `FileNotFoundError`). Non c'è nulla di specifico a `forgedge` da catturare selettivamente; cattura i built-in.

### Filosofia di validazione

Non c'è **nessuna singola validazione di schema anticipata** della KPI Table. `forge()` stessa non chiama `summary_report` né controlla le colonne prima di iniziare — la validazione è lazy e distribuita: ogni modulo valida le specifiche colonne/sorgente timestamp di cui ha bisogno, esattamente quando gli servono, e fallisce rapidamente con `ValueError`/`KeyError` se mancano. L'unica eccezione deliberata è `kpi_builder.build_features()`, che salta silenziosamente (con un `logger.warning`, non un'eccezione) qualunque indicatore le cui colonne di input richieste non siano presenti — così le candele solo-OHLC sono sempre sicure da passare, anche contro una config che chiede anche indicatori basati sul volume.

`summary_report()` (§9) è la risposta della libreria a "voglio validare prima di impegnarmi in una run" — ma è interamente **opt-in**: non solleva mai, non blocca mai, e non viene mai chiamata automaticamente. Se vuoi uno stop rigido su dati cattivi, lo scrivi tu:

```python
rep = summary_report(kpi, return_report=True, verbose=False)
if rep.has_critical:
    raise ValueError(f"Risolvi i problemi sui dati prima: {rep.one_line()}")
```

### Eccezioni comuni, verificate

Non sono parafrasate — ognuna è stata effettivamente scatenata e il suo messaggio catturato.

```python
from forgedge import forge, CustomEvent, DiscoveryConfig

forge(kpi, manual_events=[CustomEvent("close < 50")], event_discovery_config=DiscoveryConfig())
# ValueError: manual_events and event_discovery_config are mutually exclusive.
# Pass one or the other, not both.
```

```python
from forgedge import AlphaDiscovery, AlphaConfig

ad = AlphaDiscovery(kpi, [], AlphaConfig())
ad.promoted_contracts()
# RuntimeError: Call run() before promoted_contracts().
```

Questo pattern `RuntimeError` "chiama prima `.run()`" è consistente tra `MarketContext.distribution()`, `EventDiscovery.summary()`, `AlphaDiscovery.summary()`/`.promoted_contracts()`, `RuleDiscovery.grid_summary()`, e `TargetOptimizer.validate_oos()`/`.discover_alpha()` — se lo vedi, hai chiamato un accessor prima del corrispondente `.run()`.

```python
from forgedge.event_discovery.models import GateParams
GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)
# TypeError: GateParams.__init__() got an unexpected keyword argument 'min_act'
```

Questo caso merita una menzione specifica: `min_act`/`min_months`/`max_conc` erano campi di una API **precedente** di `GateParams`. Diversi script sotto `examples/` in questo repository (`alpha_discovery_usage.py`, `extended_usage.py`, `kpi_table_1d.py`, `search_rotation_calibration.py`, `lowfreq_null_diagnostic.py`, `lowfreq_endpoint_diagnostic.py`) costruiscono ancora `GateParams` in questo modo vecchio e **sollevano esattamente questo `TypeError` se eseguiti contro la versione attuale della libreria, come installata in questo repository**. Questo manuale lo segnala esplicitamente invece di aggirarlo silenziosamente: se copi codice da quegli script di esempio, traduci `GateParams(min_act=..., min_months=..., max_conc=..., min_tpm=...)` ai campi attuali (`min_tpm`, `max_dispersion`, `event_counting`, `min_episodes`, `episode_gap` — §10). `examples/kpi_builder_usage.py` **non** ha questo problema — è stato verificato funzionare correttamente end-to-end contro l'API attuale (§13).

### Tabella riassuntiva di cosa solleva cosa

| Eccezione | Trigger tipico |
|---|---|
| `ValueError` | stringa enum-like non valida (`direction`, `target_mode`, `buy_type`, `entry_mode`, `selection_mode`, `threshold_mode`, `timeframe`, `preset`, …), campo di config numerico fuori range, argomenti mutuamente esclusivi di `forge()`, coppia contratto/candidato non corrispondente passata a `RuleDiscovery` |
| `KeyError` | una colonna richiesta manca — colonne OHLC, `timestamp_col`, `source_col`, un nome di pattern candlestick sconosciuto |
| `RuntimeError` | un accessor chiamato prima di `.run()` |
| `TypeError` | tipo di input errato a `build_features`/`lag_features`, o una chiave `forge_preset(**overrides)` non riconosciuta |
| `ImportError` | `load_kpi_config()` chiamata con un percorso YAML ma PyYAML non è installato |
| `FileNotFoundError` | `load_kpi_config()` con un percorso che non esiste |

### Warning da non ignorare

`forgedge` usa `warnings.warn` (non eccezioni) per situazioni valide ma probabilmente non volute:

- **`UserWarning` — `horizon_grid` oraria obsoleta su dati giornalieri-o-più-lenti.** Sollevata da `forge()` quando passi un `AlphaConfig` esplicito che porta ancora la griglia default calibrata sull'orario non modificata su un `timeframe` di un giorno o più.
- **`UserWarning` — mismatch dell'indice delle candele osservate.** Sollevata da `AlphaDiscovery`, `RuleDiscovery`, e dall'ingestione del Rule Registry quando il frame che passi loro ha un indice diverso dalla serie di attivazione di training memorizzata dell'evento — cioè l'evento viene rivalutato via `.apply()` invece di riusare la cache. Una seconda variante, più seria, di questo stesso warning si attiva quando il conteggio di attivazioni rivalutato collassa sotto il 10% del conteggio di training — è un forte segnale che stai per ottenere `direction="undetermined"` perché le baseline delle trasformazioni mobili (pctrank, z-score) si sono spostate (§21).
- **`DeprecationWarning`** — il campo legacy `TargetConfig.min_lift` (superato da `min_lift_atoms`/`min_lift_result`) e un argomento legacy del costruttore di `TypeClassifier` (`scale_free_drift_threshold`).

### Comportamenti degradati ma non fatali da conoscere

Alcune condizioni non sono né errori né warning — cambiano il comportamento silenziosamente, registrate solo a livello INFO/DEBUG o come stringa diagnostica:

- Un `CustomEvent` che fallisce il Consistency Gate viene **mantenuto**, non scartato — solo un `logger.warning`.
- `binary_target(..., target_mode="proj")` ricade su `"abs"` quando non c'è abbastanza storia per il warmup della SMA di trend, registrato a livello `WARNING`.
- `RuleDiscoveryConfig(selection_mode="walk_forward")` ricade silenziosamente sulla selezione full-sample quando lo span di dati è troppo corto anche per un singolo split walk-forward — registrato come nota nella response, non sollevato.

---

## 12. Casi d'uso progressivi

### Caso d'uso 1 — Hello World: scoprire un evento

Già coperto per intero in §6. Il paradigma da portare via: **Event Discovery trova struttura senza mai guardare cosa succede dopo.** Tutto il resto della libreria si costruisce sopra quel singolo evento.

### Caso d'uso 2 — Un workflow realistico a singolo asset, da candele grezze

Uno scenario più completo: hai candele OHLCV grezze (non ancora una KPI Table), vuoi che gli indicatori siano calcolati per te, e vuoi eseguire la pipeline statistica completa fino ad Alpha Discovery. Questo è `examples/kpi_builder_usage.py` di questo repository, **verificato funzionare correttamente, senza modifiche, contro la libreria attuale**:

```python
from forgedge import (
    build_features, candle_features, lag_features,
    summary_report, forge, forge_preset,
)

# candles: un DataFrame con open_time (epoch ms), open, high, low, close, volume
summary_report(candles, timeframe="1D")   # check pre-flight opt-in, non blocca mai

DEMO_CONFIG = {
    "ema":             {"enabled": True, "params": {"periods": [9, 25, 50], "columns": ["close"]}},
    "rsi":             {"enabled": True, "params": {"periods": [14], "columns": ["close"]}},
    "bollinger_bands": {"enabled": True, "params": {"periods": [20], "columns": ["close"]}},
    "min":             {"enabled": True, "params": {"periods": [24], "columns": ["close"]}},
    "max":             {"enabled": True, "params": {"periods": [24], "columns": ["close"]}},
    "return":          {"enabled": True, "params": {"periods": [1, 6, 24], "columns": ["close"]}},
}

kpi = build_features(candles, DEMO_CONFIG, timestamp_col="open_time")
kpi = candle_features(kpi)
kpi = lag_features(kpi, "close", "color", like="_ema_", periods=[1, 2, 3])

disc, alpha, rd = forge_preset("balanced", timeframe="1D", asset="DEMO")
result = forge(kpi, ticker="DEMO", timeframe="1D",
               event_discovery_config=disc, alpha_config=alpha,
               run_rule_discovery=False, progress=False)

print(f"M1 candidati = {len(result.candidates)}  M2 promossi = {len(result.promoted)}")
```

**Output verificato** (eseguendo questo script così com'è, su 2000 barre di candele sintetiche generate dallo script stesso):

```
build_features  : (2000, 21)  (indicatori base + open_dt + color)
candle_features : (2000, 27)  (+ body, upper_wick, lower_wick, close_pos, range_pct, gap)
lag_features    : (2000, 42)  (+ 15 colonne *_prev_NN)
forge(kpi)      : M1 candidati = 5015  M2 promossi = 1091
```

Interpretazione: `build_features` ha trasformato 6 colonne grezze in 21 (gli indicatori richiesti più `open_dt` e `color`). `candle_features` ha aggiunto 6 colonne di geometria scale-free. `lag_features` ha aggiunto 15 copie shiftate. La KPI Table risultante a 42 colonne ha prodotto oltre 5000 candidati evento e poco più di mille contratti promossi — su **dati random-walk sintetici**, il che è a sua volta un risultato istruttivo (§16, §21): un random walk *dovrebbe* produrre un gran numero di candidati statisticamente "significativi in apparenza" ma economicamente privi di senso, ed è esattamente per questo che a valle esistono Rule Discovery e la rotation null.

### Caso d'uso 3 — Dati reali del repository: ADA e AMZN

Coperto per intero in §7 (quick start ADA) e §13 (entrambi i dataset in profondità, incluso il workflow di pulizia del CSV grezzo di AMZN).

### Caso d'uso 4 — Configurazione avanzata

Quattro assi di configurazione che è probabile ti servano davvero, ognuno mostrato come diff reale rispetto al default:

**a) Allentare il Consistency Gate per un dataset più corto o a frequenza minore.**

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import GateParams

# Il default GateParams(min_tpm=0.5, max_dispersion=1.5) è già piuttosto
# permissivo; alzare min_tpm scambia via eventi rari/marginali per
# potere statistico per evento (vedi il trade-off frequenza-vs-selettività di §16).
config = DiscoveryConfig(gate_params=GateParams(min_tpm=1.5, max_dispersion=2.0))
ed = EventDiscovery(kpi, config=config)
```

**b) Passare il modello di ingresso di Rule Discovery da limite a mercato.**

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig

# "limit" (default) può soffrire di un "fill confound": un limite profondo,
# raramente riempito, può mostrare un PF gonfiato su un piccolo sottoinsieme
# non rappresentativo di trade. "market" isola l'edge del segnale dal
# meccanismo di ingresso interamente.
config = RuleDiscoveryConfig(entry_mode="market")
resp = RuleDiscovery(event_frame, contract, candidate, config=config).run()
```

**c) Eseguire il calibratore di rotazione completo e campionato invece di quello veloce di default — con il preset `"sweep"`, come la documentazione raccomanda di abbinarli.**

```python
from forgedge import forge, RotationConfig

disc, alpha, rd = forge_preset("sweep", timeframe="1D", asset="ADAUSDC")
result = forge(kpi, ticker="ADAUSDC", timeframe="1D",
               event_discovery_config=disc, alpha_config=alpha, rule_discovery_config=rd,
               rotation_calibration=RotationConfig(k=100))   # sostituisce fast_null
promoted = result.alpha_discovery.promoted_contracts(min_lift=0.05)
```

**d) Passare lo scoring del target binario di Alpha Discovery da eccesso-sul-trend (`"proj"`, il default) a rendimento assoluto (`"abs"`).**

```python
from forgedge import AlphaConfig
config = AlphaConfig(asset="ADAUSDC", timeframe="1D", target_mode="abs")
```

Per ognuno: cosa cambia, perché esiste, e il costo di cambiarlo sono coperti punto per punto in §15 (comportamenti opt-in) e §16 (trade-off) — questa sezione è il "come", quelle sono il "perché".

### Caso d'uso 5 — Errori e dati problematici

**Dati non validi: una tabella vuota.**

```python
import pandas as pd
from forgedge import summary_report

rep = summary_report(pd.DataFrame(columns=["open", "high", "low", "close"]),
                      verbose=False, return_report=True)
print(rep.worst, [f.code for f in rep.findings])
# FAIL ['empty']
```

`summary_report` non solleva mai — riporta. Se la tua applicazione ha bisogno di uno stop rigido, lo scrivi tu: `if rep.has_critical: raise ValueError(rep.one_line())`.

**Una configurazione genuinamente obsoleta e incompatibile (un bug reale riproducibile oggi).**

```python
from forgedge.event_discovery.models import GateParams
GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0)
# TypeError: GateParams.__init__() got an unexpected keyword argument 'min_act'
```

Questa è la vecchia API di `GateParams`, che diversi script `examples/*.py` di questo repository usano ancora. Vedi §11 per la spiegazione completa e quali script sono coinvolti.

**Configurazione mutuamente esclusiva.**

```python
from forgedge import forge, CustomEvent, DiscoveryConfig
forge(kpi, manual_events=[CustomEvent("close < 50")], event_discovery_config=DiscoveryConfig())
# ValueError: manual_events and event_discovery_config are mutually exclusive.
```

**La trappola silenziosa "nessun risultato" più comune: `direction="undetermined"`.**

Non è un'eccezione — è un contratto con `status="REJECTED"` e una ragione di rifiuto specifica. È l'unico gate di rigetto rigido in Alpha Discovery (§8, §9), e di gran lunga la ragione più comune per cui un evento apparentemente promettente non diventa mai una regola utilizzabile. Succede quando nessun orizzonte nella griglia produce un vantaggio forward medio finito e a segno determinato — comunemente perché:

- la griglia di orizzonti genuinamente non copre la scala temporale a cui l'evento ha un effetto (mitigato, ma non eliminato, da `AlphaConfig.horizon_enrichment`, attivo di default — §15);
- hai passato ad Alpha Discovery un frame il cui indice osservato differisce dalla serie di attivazione di training dell'evento (il secondo `UserWarning` di §11), quindi il conteggio di attivazioni rivalutato è collassato e non resta potere statistico sufficiente per determinare affatto un segno — uno scenario coperto concretamente in §21.

```python
rejected = [c for c in result.contracts if not c.promoted]
for c in rejected[:3]:
    print(c.event_candidate_id, c.rejection_reasons)
# ogni contratto rifiutato ha "no derivable target" nelle rejection_reasons
# — l'unica ragione di rigetto a gate rigido che questo modulo produce
```

**Un nome di colonna feature non conforme.**

```python
kpi["my_custom_signal"] = ...   # non corrisponde a {base}_{indicatore}_{periodo}
```

Questo **non** solleva né avverte. `my_custom_signal` funziona comunque come feature standalone di Event Discovery — semplicemente non viene mai accoppiata in una feature ratio/spread con un'altra colonna (§9). Se ti aspettavi di vedere `ratio_my_custom_signal_something` nella lista di candidati e non c'è, questa convenzione di naming è la prima cosa da controllare, non un bug da segnalare.

### Caso d'uso 6 — Un'applicazione realistica: un servizio di monitoraggio attorno a una sessione di discovery

Questo abbozza come l'output di una sessione diventa qualcosa che un sistema più grande consuma, usando solo chiamate `forgedge` reali e verificate (il codice "wrapper" — `MonitoringService`, la classe di storage — è codice applicativo illustrativo, non parte della libreria; §19 è esplicita su questo confine).

```python
import pandas as pd
from forgedge import forge, RuleSpec, rule_performance_report

class MonitoringService:
    """Avvolge una sessione di discovery forge() e la espone per il monitoraggio continuo."""

    def __init__(self, kpi_table: pd.DataFrame, ticker: str, timeframe: str):
        self.result = forge(kpi_table, ticker=ticker, timeframe=timeframe, progress=False)
        # RuleSpec.from_forge_result(): una spec per ogni regola tradabile (EDGE/PARTIAL-EDGE),
        # ciascuna con i params e il candidate necessari per riprodurla in seguito.
        self.specs = RuleSpec.from_forge_result(self.result)

    def published_rules(self) -> list[str]:
        return [s.name for s in self.specs]

    def health_report_html(self, fresh_candles: pd.DataFrame) -> str:
        # Riproduce ogni regola pubblicata deterministicamente su fresh_candles
        # tramite lo stesso percorso EventCandidate.apply() che usa Rule Discovery —
        # fresh_candles non deve essere la tabella di scoperta.
        return rule_performance_report(self.specs, fresh_candles,
                                        title=f"{self.result.ticker} monitoring")


svc = MonitoringService(kpi, ticker="ADAUSDC", timeframe="1D")
print(f"{len(svc.published_rules())} regole pubblicate: {svc.published_rules()[:3]} ...")
html = svc.health_report_html(kpi)   # in produzione: tabella di scoperta + barre genuinamente nuove
```

**Output verificato** sul fixture ADA: `54 regole pubblicate: ['RULE_ADA_01', 'RULE_ADA_02', 'RULE_ADA_03'] ...`, e un report HTML generato di ~5 MB (grafici SVG inline, nessuna risorsa esterna — sicuro da salvare o inviare via email come singolo file). I nomi `RuleSpec` seguono la stessa convenzione `RULE_{TICKER}_{NN}` che il Modulo 4 usa internamente, anche quando non passi affatto per `RuleRegistry`.

---

## 13. Lavorare con i dati presenti in questo repository

Due dataset reali sono inclusi in questo repository, più un artefatto derivato. Nessuno dei due è un giocattolo — entrambi sono dati di mercato genuini.

### `tests/fixtures/ADA_1D_TRAIN.parquet`

882 barre giornaliere OHLCV di ADAUSDC (Cardano), dal 2024-01-01 al 2026-05-31, con 22 colonne indicatore precalcolate già allegate: `close_ret_{03,12,96}` (rendimenti), Bollinger Bands (`close_bb_{mid,upper,lower,width}_20`), un max-drawdown mobile (`close_mdd_48`), EMA su due finestre sia su `close` sia su `low` (con tre copie shiftate ciascuna, `_prev_01..03`), e volatilità mobile (`close_vol_{05,12,24}`). Questo è esattamente il fixture su cui la suite di regressione golden della libreria stessa (`tests/test_golden.py`) fissa i propri valori attesi — cioè se esegui `forge()` con gli argomenti mostrati in §7, stai riproducendo a mano parte della suite di test della libreria stessa.

```python
import pandas as pd
kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
```

Nessun preprocessing necessario — questa tabella è già una KPI Table valida (ha `close`, ha `open_dt` come `datetime64[ns]`, ordinata cronologicamente). Questo è il dataset usato in §7 e nei Casi d'uso 4-6 di §12.

### `examples/data/AMZN_1D.csv`

1378 barre giornaliere di AMZN (Amazon), un export da un provider di dati finanziari in un formato comune ma **non direttamente compatibile con `forge()`**:

```python
import pandas as pd
raw = pd.read_csv("examples/data/AMZN_1D.csv")
print(raw.columns.tolist())
# ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
print(raw.head(2))
#          Date   Price    Open    High     Low     Vol. Change %
# 0  06/30/2026  238.71  237.50  241.53  237.57   31.50M   -0.60%
# 1  06/29/2026  240.14  234.22  249.71  233.80   77.62M    3.20%
```

Nota cosa non va in questa tabella, dal punto di vista di `forgedge`: il prezzo di chiusura si chiama `"Price"`, non `"close"`; il volume è una **stringa** con un suffisso di unità (`"31.50M"`); le date sono in ordine **decrescente** (più recente per prima) e memorizzate come stringhe `MM/DD/YYYY`, non una colonna o un indice `datetime64`. Questo è realistico, ed è esattamente il tipo di problema che `summary_report` è pensata per catturare prima di sprecare una run di `forge()`. Il walkthrough completo del repository stesso (`examples/forge_amzn_walkthrough.ipynb`), che questo manuale ha verificato cella per cella, mostra la pulizia necessaria:

```python
raw.columns = [c.strip().lower().replace(".", "").replace(" ", "_") for c in raw.columns]
raw = raw.rename(columns={"price": "close", "vol": "volume", "change_%": "chg_pct"})

raw["open_dt"] = pd.to_datetime(raw["date"], format="%m/%d/%Y")
raw = raw.sort_values("open_dt").reset_index(drop=True)   # ordine ascendente — richiesto

for col in ["open", "high", "low", "close"]:
    raw[col] = pd.to_numeric(raw[col].astype(str).str.replace(",", ""), errors="coerce")

raw["volume"] = (raw["volume"].astype(str).str.replace("M", "e6").str.replace("B", "e9"))
raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")

candles = raw[["open_dt", "open", "high", "low", "close", "volume"]]
```

Dopo la pulizia, il notebook esegue `summary_report(candles, timestamp_col="open_dt", timeframe="1D", return_report=True)` e — vale la pena citarlo direttamente perché è un riscontro genuino e reale, non un esempio didattico costruito — fa emergere un'anomalia reale: "l'**ultima barra** del dataset (2026-06-30, la barra 'di oggi' al momento dell'estrazione) ha `open` leggermente sotto il `low` registrato — sintomo tipico di una barra ancora **incompleta** (sessione di mercato in corso al momento dello scarico del CSV)." Questa è un'illustrazione genuinamente utile e reale del perché si esegue `summary_report` prima, non dopo, una sessione di discovery: una barra finale incompleta non è una ragione per diffidare dell'intero dataset, ma è una ragione per considerare di scartare quella singola riga prima di eseguire Event Discovery, dato che la relazione OHLC di una barra parziale viola l'assunzione che ogni altra barra della tabella soddisfa.

Il notebook procede poi attraverso `build_features()` (con `atr`/`macd` esplicitamente abilitati, a differenza della config demo del Caso d'uso 2), `forge_preset("sweep", ...)`, e una chiamata `forge()` completa, arrivando — su dati AMZN reali — a esattamente **2** segnali `PARTIAL-EDGE` e **zero** `EDGE` pieni, su 3035 candidati M1 e 508 contratti promossi M2. L'interpretazione dello stesso notebook (§14 copre il "perché" in profondità) è esplicita e vale la pena citarla: "Il risultato onesto: quasi nessun edge, ed è la cosa giusta."

### `kpi_table_1d.csv` (radice del repository)

Un **artefatto di output** pre-generato, non un dataset di input — è ciò che `examples/kpi_table_1d.py` produce confrontando il percorso di event-discovery proprio di FORGE contro il `TargetOptimizer` standalone (§9) su dati giornalieri ADA attraverso tre finestre temporali (2024 in-sample, 2025 e 2026 out-of-sample). Le sue 19 colonne (`ID, Pipeline, EVENT_ID, EXPRESSION, IS 2024_WR, IS 2024_PF, ...`) sono un esempio lavorato di confronto tra due strategie di discovery e il loro decadimento out-of-sample fianco a fianco — utile da leggere come *forma di output di riferimento*, non qualcosa che devi rigenerare per usare questo manuale.

### Notebook

Da `notebooks/01_event_discovery.ipynb` a `06_rule_registry.ipynb` percorrono ogni modulo individualmente e in profondità; `notebooks/hurst.ipynb` è un approfondimento dedicato sulla stima dell'half-life Hurst/OU che Market Context usa per la selezione automatica delle finestre EMA (§8). `examples/forge_amzn_walkthrough.ipynb` (usato per tutta questa sezione) è l'unico costruito interamente attorno a un dataset incluso in questo repository ed è la cosa più vicina a un walkthrough canonico, completamente riproducibile, su dati reali.

---

## 14. Scelte di design

Questa sezione distingue tre categorie in modo esplicito: **(documentato)** — il ragionamento dichiarato dagli autori stessi, citato o parafrasato da vicino dalla documentazione del repository; **(misurato)** — un risultato reale, riportato da un'esecuzione su dati reali, non un benchmark inventato; **(inferito)** — la lettura del codice fatta da questo manuale, chiaramente segnalata come tale, mai presentata come un fatto dichiarato dagli autori.

### La separazione a tre domini è architetturale, non convenzionale (documentato)

Già introdotta in §2. La guida architetturale del repository è esplicita sul fatto che sia un vincolo, non una scelta di stile, e dà la ragione specifica: tenere il rendimento forward fuori da Event Discovery *"elimina una categoria intera di look-ahead bias"* — per costruzione, non per disciplina.

### "La rigidità a monte compra libertà a valle" (documentato)

Un trade-off dichiarato esplicitamente, nelle parole degli autori stessi, in `src/forgedge/docs/README.md`: "La rigidità a monte è la garanzia che ciò che arriva ad Alpha Discovery sia una misura genuina — non un artefatto di ottimizzazione sul target. **Compra la libertà di esplorare liberamente a valle** perché le soglie sottostanti non sono state contaminate da nessuna scelta economica." Questa è la motivazione per cui le soglie di Event Discovery sono immutabili: il costo (non puoi tarare una soglia a posteriori, anche se sei sicuro che una leggermente diversa funzionerebbe meglio) è ciò che rende tutto a valle affidabile.

### Le misure statistiche informano il voto; solo la direzione blocca la promozione (documentato)

Anche questo esplicito nella stessa fonte: le debolezze statistiche in Alpha Discovery "alimentano il voto, non scartano — Rule Discovery è l'unico giudice economico". Questa è una scelta di design genuinamente non ovvia su cui vale la pena soffermarsi: un'implementazione ingenua rigetterebbe un candidato per IC debole o lift basso. `forgedge` invece lascia passare al Modulo 3 tutto ciò che ha una direzione determinata, e tratta le debolezze statistiche puramente come informazione che (a) abbassa il voto in lettere e (b) compare verbatim in `rejection_reasons` — anche su contratti che *sono stati* promossi. Il risultato, verificato in §7: il miglior candidato `PARTIAL-EDGE` del quick start ADA porta un voto `A` e una lista di diagnostici non bloccanti, insieme all'unico gate (rotation null) che ha effettivamente limitato il suo verdetto.

### Anti-goal — cosa gli autori hanno esplicitamente deciso di non costruire (documentato)

Il documento di analisi funzionale (`docs/analysis/forge2_functional_analysis.md`) dichiara quattro direzioni rifiutate con motivazioni, verbatim:

- **"Niente ML/feature learning nella discovery."** Motivazione data: "Il valore differenziante è che ogni regola è un'espressione booleana leggibile e auditabile; un modello addestrato romperebbe il contratto molto più di qualunque bug."
- **"Niente registry persistente/DB."** Motivazione data: "In-memory + export (flat table, HTML) è il livello giusto di ambizione; la persistenza è un problema dell'host." — questo è il motivo per cui §19-20 di questo manuale trattano la persistenza come una responsabilità dell'*applicazione*, non una feature della libreria.
- **"Niente verdetti probabilistici al posto della triade."** Motivazione data: "La triade (+ INSUFFICIENT-DATA) È il contratto; la confidenza va accanto al verdetto, non al suo posto."
- **"Niente dipendenze statistiche esterne."** Motivazione data: "Le primitive in numpy puro sono un asset di auditabilità, non un debito."

### La rotation null esiste perché l'audit degli stessi autori ha trovato che i conteggi di promozione non erano affidabili a bassa frequenza (documentato + misurato)

Questa è la scelta di design più importante da capire se vuoi fidarti dell'output di `forgedge`, ed è sostenuta da numeri reali che gli autori riportano dal proprio audit (`docs/analysis/lowfreq_robustness.md`, su `ADA_1D_FULL.parquet`, 901 barre giornaliere):

> "**ADA REALE** ha promosso **58** alpha da 2542 candidati su una finestra in-sample 2024; il **rumore phase-randomized** (5 run, stessa ricetta, stesse statistiche, autocorrelazione preservata ma qualsiasi predittività reale distrutta) ha promosso **146 ± 45** (range 101-207). *Il rumore viene promosso più spesso dell'asset reale.*"

E, uno stadio più a valle, al livello dei verdetti tradabili `EDGE`/`PARTIAL-EDGE`:

> "Il rumore puro guadagna comunque ~2-3 EDGE su 12 testati (**~20% di pavimento di falsi positivi per regola**)."

Questa è la diagnosi degli stessi autori di una debolezza reale in una versione della pipeline **prima** che la rotation null esistesse come default. La correzione — `FastRotationNull`, da allora attiva di default — è descritta nello stesso documento con il proprio costo ed effetto misurato: "calcola la rotation null esatta a livello di ricerca su ogni offset circolare (cross-correlazione FFT, ~1 s su questo dataset — niente K, niente seed)... Su questo dataset ADA il search p è ≈ 0.70: ogni EDGE precedente viene onestamente limitato a PARTIAL-EDGE." Quest'ultima frase è un riscontro genuinamente notevole e autocritico: gli autori hanno eseguito la propria correzione contro i propri risultati migliori precedenti e hanno riportato che li ha declassati tutti.

### Il design è stato scelto per mantenere identico il contratto rivolto all'utente correggendo l'onestà interna (documentato)

Il documento di analisi funzionale inquadra questo esplicitamente come il vincolo guida dell'intero redesign: "ripartendo da zero, cosa va tenuto e cosa va riprogettato — **senza cambiare il contratto con l'utente**?" E la sintesi finale dichiara la diagnosi in modo netto: "il sistema conta benissimo le proprie prove dentro ogni modulo, ma nessuno conta le prove dell'intera catena — e sui dati lenti questo trasforma il multiple testing in verdetti EDGE che il rumore puro sa replicare 1 volta su 5." La conseguenza pratica per te come utente: ogni default descritto in §15 sotto che sembra rendere la pipeline *più conservativa* (la rotation null, il power gate, gli split temporali purgati) esiste specificamente per chiudere quel divario, e nessuno di essi è stato aggiunto cambiando la firma di chiamata di `forge()` in modo incompatibile.

### Perché la griglia di orizzonti doveva diventare consapevole del timeframe (documentato + misurato)

`lowfreq_robustness.md` identifica questo come un punto debole specifico, referenziato nel codice: "`horizon_grid` non è scalata sulla frequenza... `forge(..., timeframe="1D")` la usa invariata → periodi di detenzione fino a 48 giorni. A differenza di MarketContext / Hurst / rolling-IC / bars-per-year, che si auto-scalano, la griglia di orizzonti è una trappola silenziosa." Questo è il motivo per cui §7 di questo manuale dedica un paragrafo al fatto che `forge()` sostituisca una griglia giornaliera per te — è una correzione per un bug documentato e precedentemente reale, non una feature cosmetica.

### Perché la deduplicazione solo segnala, non elimina mai (documentato)

`src/forgedge/docs/README.md` dà una ragione specifica e non ovvia per cui due regole strutturalmente sovrapposte potrebbero valere la pena di essere entrambe tenute: "Una coppia `INDEPENDENT_CONFIRMATION` non è un duplicato da scartare — è la conferma indipendente dello stesso edge da due meccanismi diversi." (Nota: la classificazione a due livelli che questo passaggio descrive — `DUPLICATE_STRUCTURAL`/`DUPLICATE_BEHAVIORAL`/`INDEPENDENT_CONFIRMATION` — è elencata nella roadmap dello stesso repository come **non ancora implementata**; il comportamento attuale, effettivamente distribuito, è il più semplice flag binario `is_duplicate` descritto in §8/§9. Questo manuale segnala esplicitamente questo divario invece di descrivere una feature pianificata come attuale.)

---

## 15. Comportamenti opt-in

Questa è la sezione che il compito ha chiesto di trattare con particolare cura. "Opt-in" qui significa specificamente: un comportamento che **non** avviene a meno che tu non imposti esplicitamente un valore non-default. Diversi default sotto (la rotation null, il purging, il power gate, l'arricchimento degli orizzonti) sono essi stessi *attivi* di default e devono essere esplicitamente *disattivati* — sono segnalati come "default-on, opt-out" invece di essere impropriamente classificati come opt-in.

| Feature | Default | Come abilitarla | Vantaggio | Costo / trade-off | Quando usarla |
|---|---|---|---|---|---|
| **Walk-forward OOS a livello di evento** (`DiscoveryConfig.walk_forward`) | `None` (off) | `DiscoveryConfig(train_ratio=0.8, walk_forward=EventWalkForwardConfig(n_splits=3, min_pass_rate=0.6))` | Conferma che la *struttura temporale* di un evento — non ancora il suo potere predittivo — è stabile su più finestre OOS prima ancora di arrivare ad Alpha Discovery | Meno barre disponibili per la mining IS di Event Discovery; aggiunge tempo di esecuzione al Modulo 1 | Sospetti che il tuo catalogo di indicatori possa fare overfitting su un regime e vuoi un primo filtro prima degli stadi (molto più costosi) di Alpha/Rule Discovery |
| **Diversity Gate** (`DiscoveryConfig.diversity_gate_enabled`) | `False` | `DiscoveryConfig(diversity_gate_enabled=True, diversity_threshold=0.85)` | Scarta eventi singoli quasi-duplicati (Jaccard ≥ soglia sulle date di attivazione) prima della composizione AND, così i candidati composti non vengono sprecati su coppie ridondanti | Può scartare un evento genuinamente distinto che risulta correlato strutturalmente a un altro su questa specifica finestra; aggiunge un passaggio di confronto O(n²)-simile sui singoli che superano il gate | Cataloghi di indicatori ampi dove hai osservato molti candidati quasi-identici sopravvivere al gate |
| **`event_counting="bar"`** (`GateParams`) | `"episode"` | `GateParams(event_counting="bar")` | Riproduce esattamente il comportamento del gate pre-#134 (conteggio per episodio), barra per barra | Stati persistenti multi-barra (es. un tratto di 3-5 barre con RSI<30) gonfiano la varianza dei conteggi mensili e possono venire rigettati a torto — questa è la ragione documentata per cui `"episode"` è diventato il default | Ti serve riproducibilità byte-per-byte con una sessione pre-episode-counting, o hai una ragione specifica per pesare per conteggio di barre grezze |
| **`target_mode="abs"`** (`AlphaConfig`, `TargetConfig`) | `"proj"` | `AlphaConfig(target_mode="abs")` | Valuta il target binario sul rendimento forward grezzo, non sull'eccesso-sopra-trend — più semplice, corrisponde alle convenzioni di backtest "da manuale" | `"proj"` esiste specificamente perché un evento long che cavalca una deriva di bull market non venga accreditato del trend del mercato come se fosse l'edge dell'evento; `"abs"` reintroduce quel rischio | Confronto con una baseline pre-`"proj"`, o lavoro su un asset/periodo dove il trend-following è esplicitamente la strategia che vuoi misurare, non filtrare via |
| **`AlphaConfig.embargo_bars` / `RuleWalkForwardConfig.embargo_bars`** | `0` | `AlphaConfig(embargo_bars=5)` | Aggiunge un buffer di quarantena dell'autocorrelazione seriale all'inizio della finestra OOS, oltre a ciò che il purging da solo rimuove | Riduce ulteriormente il campione OOS, che — combinato con una coda OOS giornaliera già sottile — può spingere un contratto verso `INSUFFICIENT-DATA` | Asset con autocorrelazione a breve lag nota e forte oltre a ciò che la larghezza di purge (= orizzonte massimo) già copre |
| **`RotationCalibrator` (esplicito, via `rotation_calibration=`)** | `None` (sostituito dal `FastRotationNull` di default) | `forge(..., rotation_calibration=RotationConfig(k=100))` | Calibrazione multi-yardstick (composite, lift, t-stat) via una combinazione di Tippett min-p — cattura statistiche discriminanti che il singolo yardstick `abs_z` di `FastRotationNull` può perdere | `~K ×` il costo di un passaggio di Alpha Discovery (dell'ordine di secondi-minuti per estrazione su dati reali, secondo la misura dello stesso documento di design "~4 s/estrazione") vs. i ~1 secondo totali di `FastRotationNull` | Abbinato esplicitamente al preset `"sweep"`; ogni volta che vuoi l'oggetto report completo, non solo l'annotazione pass/fail che `fast_null` lascia su ogni contratto |
| **`fast_null=False`** | `fast_null=True` (default-on, opt-out) | `forge(..., fast_null=False)` | Salta interamente il controllo rotation null di default — più veloce, e un contratto altrimenti limitato può raggiungere un verdetto `EDGE` pieno | Reintroduce esattamente il problema "il rumore viene promosso quasi quanto il segnale" che §14 documenta con numeri reali — non è un parametro di performance da attivare/disattivare con leggerezza | Solo per debug/prototipazione, o quando eseguirai comunque il `RotationCalibrator` completo separatamente e non vuoi l'annotazione del passaggio veloce |
| **`time_budget=` (`TimeBudget` esplicito)** | `None` — ma il purging resta attivo di default anche senza | `forge(..., time_budget=TimeBudget.build(n_bars=len(kpi), horizon_bars=48, embargo_bars=5))` | Un unico asse IS/OOS condiviso ed esplicito tra Event e Alpha Discovery, con un embargo controllabile | Più da configurare correttamente; sbagliare l'argomento `horizon_bars` sbaglia anche la larghezza di purge | Pipeline multi-modulo costruite a mano (non via `forge()`) dove serve che gli split di Event e Alpha Discovery coincidano esattamente |
| **`purge_bars=0`** (opt-out dal purging attivo di default) | larghezza di purge = `max(horizon_grid)` (default-on) | `TimeBudget.build(..., purge_bars=0)`, o `RuleWalkForwardConfig(purge_bars=0)` per M3 | Riproduce esattamente i risultati numerici pre-purging (utile per confrontare con run vecchie, o i vecchi valori golden-test della libreria stessa) | Reintroduce un look-ahead reale, anche se di solito piccolo: le barre IS il cui orizzonte forward attraversa nel OOS non vengono più escluse | Solo quando serve specificamente riproducibilità storica, non per uso normale |
| **`power_gate=False`** (`SelectionCriteria`) | `True` (default-on, opt-out) | `RuleDiscoveryConfig(criteria=SelectionCriteria(power_gate=False))` | Un verdetto che altrimenti verrebbe declassato a `INSUFFICIENT-DATA` per mancanza di potere statistico OOS viene lasciato passare come `EDGE`/`PARTIAL-EDGE` | Perdi il segnale proprio della pipeline che l'evidenza OOS è troppo debole per fidarsi della stima puntuale — un contratto può sembrare tradabile puramente perché nessuno ha controllato se la dimensione del campione potesse sostenere l'affermazione | Praticamente mai per una decisione live; solo per ispezionare quale sarebbe stato il verdetto *non filtrato* |
| **`entry_mode="limit"`** (uscire dalla valutazione a due stadi di default) | `"auto"` (default dalla #185) | `RuleDiscoveryConfig(entry_mode="limit")` | La griglia ottimizza `buy_drop_pct` come parte del verdetto, quindi una strategia il cui *edge è l'ordine a limite stesso* viene misurata come una cosa sola invece che divisa in segnale + esecuzione | Reintroduce il fill confound nel verdetto: uno sconto più profondo riempie solo sui percorsi che sono tornati a prenderlo, quindi il PF sale su un sottoinsieme che non è la popolazione tradeable | Quando l'ordine a limite *è* davvero la strategia. Per un segnale che stai cercando di misurare, `"auto"` separa le due letture e pubblica comunque il punto limite quando se lo merita |
| **`selection_mode="full_sample"`** | `"walk_forward"` | `RuleDiscoveryConfig(selection_mode="full_sample")` | Ripristina il comportamento legacy: il punto operativo viene scelto vagliando l'*intera* tabella, non solo le finestre di train walk-forward | Il profit factor in-sample e i parametri operativi pubblicati possono essere influenzati da dati che poi diventano parte della finestra di test OOS — una fuga reale e documentata che la modalità walk-forward esiste per chiudere | Solo compatibilità legacy; la ragione documentata per evitarlo in lavoro nuovo è esplicita nel sorgente |
| **`AlphaConfig.horizon_enrichment=None`** (opt-out) | `(0.5, 1.0, 2.0)` (default-on) | `AlphaConfig(horizon_enrichment=None)` | Restringe Alpha Discovery strettamente alla `horizon_grid` base, senza aggiunte per-evento | Secondo la misurazione degli stessi autori, **34 su 247** alpha promossi sul dataset ADA hanno trovato il loro orizzonte migliore *solo* grazie a questo arricchimento — disattivarlo li perderebbe silenziosamente | Riprodurre una baseline pre-arricchimento, o quando hai una ragione specifica per cui la griglia base non deve essere estesa |
| **`pattern_features()`** (una chiamata a funzione separata, non un flag di config) | non chiamata | `from forgedge import pattern_features; kpi = pattern_features(kpi)` | Aggiunge un'unica colonna categoriale `candle_pattern` (dieci formazioni con nome: HAMMER, DOJI, pattern di engulfing, …) che attraversa `forge()` end-to-end come eventi one-hot | I pattern con nome codificano soglie fisse scelte dall'uomo; la geometria continua di `candle_features()` è preferita per la discovery automatica specificamente perché FORGE deriva le proprie soglie asset-adattive invece | Lavoro manuale/esplorativo dove vuoi specificamente testare ipotesi su pattern con nome, non per il percorso di discovery automatica di default |
| **`use_stable_regime_only=True`** (`AlphaConfig`) | `False` | `AlphaConfig(use_stable_regime_only=True)` | L'analisi di sensibilità al regime (Step 5 di Alpha Discovery) considera solo le barre in cui il regime si è mantenuto per ≥ `stable_window` barre — statistiche per-regime più pulite | Meno osservazioni per regime, che può portare `min_regime_obs` (default 10) fuori portata per i regimi meno comuni | Hai osservato barre di transizione di regime contaminare la scomposizione per-regime con rumore del regime *precedente* |
| **`early_elimination=False`** (`SelectionCriteria`) | `True` | `RuleDiscoveryConfig(criteria=SelectionCriteria(early_elimination=False))` | Forza l'esecuzione della pipeline completa di walk-forward e diagnostica anche su una regola che fallisce lo screen rapido in-sample — utile per report uniformi anche sulle regole NON-EDGE | Compute significativamente maggiore per regola (l'intero scopo dell'eliminazione precoce è saltare quel lavoro sulle regole che falliranno comunque) | Scenari di audit/reporting dove vuoi il comportamento OOS di ogni regola popolato, non solo quello dei sopravvissuti |

---

## 16. Trade-off

Questi sono i trade-off effettivamente visibili nel codice e nella documentazione di design — non una lista generica.

**Automazione vs. controllo.** `forge()` prende un gran numero di decisioni per te di default (scalatura della griglia di orizzonti, la rotation null, il purging, l'arricchimento degli orizzonti) specificamente affinché una run "quick start" e una run "curata attentamente" non divergano silenziosamente in onestà statistica (§14). Il costo è che un lettore alla prima esperienza non può davvero prevedere il comportamento completo di `forge()` dalla sola firma di chiamata — la lista "cosa ha fatto `forge()` che non hai chiesto" di §7 esiste perché quel divario è reale, non perché questo manuale sia insolitamente meticoloso.

**Frequenza vs. selettività, nel Consistency Gate.** L'analisi di calibrazione della libreria stessa (`docs/analysis/search_rotation_calibration.md`) riporta questo esplicitamente, con numeri reali, come un trade-off genuino piuttosto che un bug da correggere: alzare `min_tpm` da 1.5 a 3.0 ha fatto scendere i candidati estratti da 2621 a 584 e il lift del miglior alpha reale da 0.707 a 0.287 — ma il candidato *sopravvissuto* è passato da 9 attivazioni in-sample (che fallivano OOS: win rate 0.78 IS → 0.40 OOS) a 29 attivazioni (che passavano OOS, p=0.000). Nelle loro stesse parole: "`min_tpm` più alto abbassa il lift... scambi 'estremo ma fragile' per 'modesto ma confermabile'." Non esiste un valore di default di `min_tpm` semplicemente "corretto" — è una manopola tra "eventi rari con edge dall'aspetto drammatico ma statisticamente fragili" ed "eventi frequenti con edge modesti ma ben confermati", e quale estremo vuoi dipende da quanti dati in-sample hai effettivamente.

**Rigore statistico vs. costo in tempo di calcolo.** Il `FastRotationNull` di default è stato progettato specificamente per rendere questo trade-off quasi gratuito (≈1 secondo, secondo la misura dello stesso documento di design, contro i ≈4 secondi *per estrazione* del `RotationCalibrator` a `K=100`) — ma paga quella velocità coprendo un solo yardstick (`abs_z`), dove il calibratore completo ne copre diversi combinati via il metodo di Tippett. Se `abs_z` non è la statistica che avrebbe discriminato i tuoi dati specifici (il documento di calibrazione mostra che questo cambia in base a `min_tpm`, cioè in base alla tua stessa configurazione), la copertura a singolo yardstick della null veloce è un divario reale, non ipotetico.

**Validazione eager vs. lazy.** Non c'è un singolo passaggio di validazione dello schema sulla tua KPI Table (§11). Questo compra la capacità di passare una tabella arbitraria ed evolutiva senza pre-registrarne lo schema da nessuna parte — ma significa che una colonna malformata emerge come `KeyError`/`ValueError` potenzialmente diversi moduli dentro una chiamata `forge()`, non alla porta. `summary_report()` esiste specificamente per lasciarti scegliere la validazione eager quando la vuoi, senza forzarla su ogni chiamata.

**Riproducibilità vs. performance in Rule Discovery.** `SelectionCriteria.early_elimination=True` (default) scarta una regola da ulteriore elaborazione nel momento in cui fallisce uno screen in-sample economico — è un vantaggio di performance reale (l'intero scopo è evitare di eseguire walk-forward e validazione statistica completa su regole che falliranno comunque), ma significa che le regole `NON-EDGE` portano di default diagnostiche incomplete (nessun `walk_forward` popolato). La voce `early_elimination=False` di §15 è esattamente la via d'uscita per quando il reporting uniforme conta più della velocità.

**Purezza delle dipendenze vs. rischio di reimplementazione.** L'anti-goal esplicito "niente dipendenze statistiche esterne" (§14) significa che correlazione di Spearman, t-test, funzione beta incompleta, e controllo FDR di Benjamini-Hochberg sono tutti scritti a mano in numpy invece di essere delegati a `scipy.stats`/`statsmodels`. Il beneficio dichiarato è l'auditabilità — puoi leggere esattamente cosa calcola `forgedge` senza addentrarti nel sorgente di una libreria esterna molto più grande. Il rischio corrispondente (questa è un'inferenza chiaramente propria di questo manuale, non dichiarata) è che queste primitive non ereditano automaticamente i decenni di irrobustimento sui casi limite di `scipy`; l'estesa suite di test (§18) è presumibilmente la mitigazione, anche se i documenti di design non lo inquadrano esplicitamente in questi termini.

---

## 17. Performance e scalabilità

Nulla in questa sezione è un benchmark inventato. Ogni numero è misurato e riportato nella documentazione del repository stesso, oppure è un'osservazione qualitativa di complessità tratta direttamente dal codice.

**Misurazioni riportate (da `docs/analysis/` e dalle esecuzioni verificate da questo manuale):**

- `FastRotationNull` su dati giornalieri ADA reali: **~1 secondo**, calcolando la rotation null esatta su ogni offset circolare via FFT — riportato dal documento di design, e coerente con l'output quasi istantaneo di `result.calibration.summary()` catturato da questo manuale in §7.
- Il `RotationCalibrator` (più pesante), campionato: **~4 secondi per estrazione** sugli stessi dati, secondo `docs/analysis/search_rotation_calibration.md` — cioè `K=100` è dell'ordine di diversi minuti, non secondi. Questo è il costo diretto del trade-off rigore-statistico-vs-velocità di §16.
- L'esecuzione `forge()` verificata da questo manuale sul fixture ADA di 882 barre, single-thread, ha prodotto 5241 candidati → 370 promossi → 370 risposte di rule-discovery. I documenti di design della libreria segnalano separatamente il Modulo 3 come lo stadio compute-intensivo su scala: "M3 sequenziale (**255 contratti × ~0.4 s** su dati piccoli)" — cioè il backtest walk-forward per contratto di Rule Discovery è il modulo il cui costo scala più direttamente con quanti contratti Alpha Discovery ha promosso.
- Lo stesso documento riporta che la suite di test stessa richiede **~8.5 minuti**, "dominata da pipeline complete ripetute" — un fatto più rilevante per contribuire alla libreria (§18) che per usarla, ma indicativo di quanto calcolo rappresenti una chiamata `forge()` completa.

**Osservazioni di complessità dal codice (lettura propria di questo manuale, non un benchmark dichiarato):**

- Gli Step 1-3 di Event Discovery (generazione feature × trasformazioni temporali × catalogo soglie) sono combinatori nel numero di colonne feature native: ogni colonna continua può produrre diverse varianti di trasformazione, ogni variante di trasformazione viene testata contro circa una dozzina di soglie, e le combinazioni di feature di arità 2 vengono tentate sia tra colonne della stessa famiglia sia via i diversi pairing dedicati descritti in §8. Una KPI Table con molte colonne indicatore dal nome simile (es. dieci diversi periodi RSI) genererà un pool di candidati corrispondentemente più grande prima che il Consistency Gate lo potatura — è esattamente per questo che `max_categorical_classes`, `max_and_components`, `indicator_lag_cross_lags`, e le stesse soglie di rate/dispersione del gate esistono come leve pratiche per controllare la dimensione del pool di candidati (§10).
- **Una cifra di costo reale e misurata per uno dei pairing di §8** (dal commit che ha introdotto `indicator_lag_cross_lags`, non un benchmark inventato): su una KPI Table EMA/SMA a 36 colonne (3 basi OHLC × 6 periodi × 2 famiglie di indicatori), abilitare il pairing indicatore-vs-OHLC-base-laggata (attivo di default) ha misurato **+24% di tempo di `EventDiscovery.run()` e +21% di conteggio candidati** rispetto alla stessa tabella con quel pairing disabilitato (73.8 s / 23179 candidati vs. 59.6 s / 19205 candidati su quel fixture). Il messaggio del commit è esplicito sul fatto che questo abbia superato la stima iniziale a spanne perché "famiglia" si è rivelato significare più di una colonna rappresentativa una volta tenuta in conto la molteplicità dei periodi — utile da sapere se stai tarando `indicator_lag_cross_lags` o `max_and_components` su una KPI Table con molti periodi di indicatori.
- Lo screening a griglia di Rule Discovery è, per contratto, una piccola ricerca a griglia (`buy_drop_pct × sell_pct × target_h × buy_delay_bar`) eseguita una volta in-sample e poi rieseguita una volta per ogni split walk-forward — quindi il suo costo scala come `(dimensione griglia) × (n_splits + 1) × (costo backtest per configurazione)`, indipendentemente per ogni contratto promosso. L'audit degli stessi autori definisce esplicitamente questo "imbarazzantemente parallelo" tra contratti (§14) — ma la libreria non lo parallelizza da sé; sarebbe una responsabilità a livello applicativo (§19).
- Costo a livello `pandas`: diversi passaggi interni di generazione feature inseriscono colonne una alla volta in un DataFrame in crescita invece di concatenarle tutte insieme, il che l'esecuzione di test verificata da questo manuale ha fatto emergere direttamente come un `pandas.errors.PerformanceWarning` ("DataFrame is highly fragmented... Consider joining all columns at once using pd.concat(axis=1)") durante la suite golden. È un dettaglio implementativo interno, non qualcosa che puoi configurare, ma vale la pena sapere che il warning è atteso e non un segno che il tuo codice abbia fatto qualcosa di sbagliato.

**Nessuna GPU, nessun calcolo distribuito, nessun I/O asincrono da nessuna parte nella libreria** — ogni modulo è codice numpy/pandas sincrono, single-process, single-thread. Se ti serve eseguire la discovery su molti ticker, `forge_multi()` li esegue comunque sequenzialmente, un ticker alla volta (§9); parallelizzare tra ticker o tra i backtest per-contratto di Rule Discovery è squisitamente una responsabilità a livello applicativo (§19-20), non qualcosa che la libreria fa per te.

---

## 18. Testing

La suite di test del repository stesso è il miglior modello per testare codice che usa `forgedge` — è sostanziale (**586 funzioni di test su 15 file**, `testpaths = ["tests"]` in `pyproject.toml`) e le sue convenzioni sono abbastanza coerenti da valere la pena adottarle direttamente.

### Eseguirla

```bash
pip install -e ".[dev]"      # pytest>=7.0
pytest                        # l'intera suite
pytest tests/test_rule_discovery.py                          # un modulo
pytest tests/test_forge.py::TestForgeManualEvents             # una classe
pytest tests/test_forge.py::TestForgeManualEvents::test_mutual_exclusion_raises   # un test
pytest tests/test_golden.py                                    # solo i pin di regressione
pytest -k golden                                                # per keyword
```

### Lo stile della casa, osservato direttamente nella suite

- **Un file di test per modulo sorgente**, nominato `test_<modulo>.py` — `test_event_discovery.py` ↔ `event_discovery/`, `test_rule_discovery.py` ↔ `rule_discovery/`, e così via, più due file trasversali: `test_forge.py` (wiring dell'orchestratore) e `test_golden.py` (fissaggio di regressione end-to-end).
- **Nessun mocking da nessuna parte**, tranne due usi di `monkeypatch` di pytest in `test_alpha_discovery.py`, entrambi usati come "tripwire di non-deve-essere-chiamato" piuttosto che come stub:

  ```python
  def test_events_come_from_stored_series_not_apply(self, monkeypatch):
      """Percorso veloce: la serie di attivazione in cache deve essere riusata, apply() non deve girare."""
      def _boom(self, frame):
          raise AssertionError("Alpha Discovery ha ricalcolato un evento via apply()")
      monkeypatch.setattr(EventCandidate, "apply", _boom)
      ...
  ```

  Ogni altro test costruisce serie di prezzo sintetiche reali e seedate con `np.random.default_rng(seed)` ed esegue il codice reale della pipeline contro di esse — non c'è I/O finto da mockare, perché la libreria non ne ha.
- **Dati sintetici deterministici, seedati e documentati nel motivo** sono l'idioma dominante delle fixture — quasi ogni file di test definisce il proprio helper locale in stile `_make_kpi_table()`/`_ohlc_kpi_table()`, ciascuno con una docstring che spiega *perché* è stata scelta quella specifica forma di segnale (es. "`feat` basso predice un rendimento positivo alla barra successiva, quindi gli eventi su `feat` dovrebbero alzare il win rate"). `tests/conftest.py` fornisce esattamente due fixture condivise, module-scoped (`kpi_4380`, `kpi_8760` — tabelle orarie sintetiche di ~6 e ~12 mesi) per i test che non hanno bisogno di una forma di segnale su misura.
- **Un unico file fixture reale** per il test di regressione: `tests/fixtures/ADA_1D_TRAIN.parquet` (§13), usato dalla fixture session-scoped `forge_result` di `test_golden.py`, che esegue `forge()` esattamente una volta e deriva decine di asserzioni a livello di singolo campo da quell'unica esecuzione — isolando quale stadio della pipeline si è rotto senza rieseguire la pipeline per ogni asserzione.
- **`pytest.approx(..., rel=...)`** è l'idioma standard per fissare i float — mai uguaglianza nuda su un float. **`pytest.raises(..., match=...)`** e **`pytest.warns(...)`** sono usati in modo consistente (~53 usi su 8 file) per fissare messaggi esatti di errore/warning, non solo i tipi di eccezione.
- **`@pytest.mark.parametrize` è usato esattamente una volta** nell'intera suite (`test_target_optimizer.py`) — la convenzione stessa del codebase favorisce fortemente un metodo di test esplicito e nominato per comportamento invece di tabelle parametrizzate, anche per casi strutturalmente simili.

### Un esempio lavorato di golden test, che illustra il pattern

```python
# tests/test_golden.py (struttura, condensata)
@pytest.fixture(scope="session")
def forge_result():
    kpi = pd.read_parquet(FIXTURE_PATH)   # tests/fixtures/ADA_1D_TRAIN.parquet
    return forge(kpi, ticker="ADAUSDC", timeframe="1D",
                 run_rule_discovery=True, run_registry=False, progress=False)

class TestGoldenEventDiscovery:
    def test_n_activations(self, golden_candidate):
        assert golden_candidate.activation_stats.n_activations == 27  # int, pytest.approx non serve
    def test_mean_tpm(self, golden_candidate):
        assert golden_candidate.activation_stats.mean_tpm == pytest.approx(0.931034, rel=1e-4)
```

I commenti inline dello stesso file documentano la *storia* dei valori golden: sono stati ri-fissati più volte man mano che cambiamenti legittimi della pipeline venivano introdotti (il default del conteggio per episodio, la griglia di orizzonti scalata sul timeframe, il cambio di selezione walk-forward, il power gate). Questo è il workflow inteso, e vale la pena adottarlo per i tuoi test costruiti sopra `forgedge`: **un golden test che si rompe è atteso su un cambiamento di comportamento legittimo** — la risposta corretta è ri-fissare il valore con un commento che spiega perché, non assumere che il test (o il cambiamento) sia sbagliato.

### Cosa testare nel tuo codice che chiama `forgedge`

- **Il wiring della configurazione**, sul modello di `test_forge.py`: la tua applicazione assembla correttamente `DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig` da qualunque sorgente di configurazione tu usi (variabili d'ambiente, un file di settings, un preset), e fallisce come ti aspetti su combinazioni non valide?
- **Casi limite con dati sintetici che controlli interamente**, sul modello del resto della suite: un evento che non si attiva mai, una tabella con un solo regime per tutta la sua durata, una lista `promoted` vuota che arriva a Rule Discovery, un verdetto della rotation null che limita tutto a `PARTIAL-EDGE`. Sono economici da costruire con l'RNG seedato di `numpy` e non richiedono dati di mercato reali.
- **Fissa in regressione la tua stessa logica a valle**, come `test_golden.py` fissa quella della libreria: se la tua applicazione deriva una decisione (es. "agisci solo su regole EDGE di voto A") da `ForgeResult`, scrivi un test che esegue un input fisso attraverso `forge()` una volta e verifica la decisione derivata, non solo l'output grezzo di `forgedge` — quello è lo strato più probabile che vada silenziosamente alla deriva mentre tari la tua stessa configurazione.
- **Non mockare `forge()` stessa** nei test pensati per validare la tua logica di integrazione — la libreria non ha I/O da fingere, e l'intero valore della pipeline sta nel suo comportamento statistico reale; mockarla via non testa nulla di vero. Usa una KPI Table sintetica piccola e veloce (vedi il pattern di `conftest.py`) invece di un dataset reale multi-anno per mantenere veloce la tua stessa suite di test.

---

## 19. Integrare forgedge in un'applicazione reale

`forgedge` è una **libreria di ricerca**, e questa sezione è esplicita, per tutta la sua estensione, sul confine tra cosa dà la libreria e cosa la tua applicazione deve costruire sopra di essa. Nulla sotto è un'API `forgedge` — è la guida architetturale di questo manuale, chiaramente separata dalla reference API di §9.

### Cosa dà forgedge

- Una funzione pura dei suoi input: `forge(kpi_table, ...) -> ForgeResult`. Nessuno stato nascosto, nessun setup/teardown richiesto, nessun thread in background.
- Ogni artefatto intermedio di una run, ispezionabile a posteriori (`ForgeResult.candidates`, `.contracts`, `.event_frame`, `.calibration`, `.ledger`).
- Riproduzione deterministica di una regola scoperta contro nuovi dati (`EventCandidate.apply()`, `RuleDiscovery` su candele fresche, `rule_performance_report()`) — questo è il meccanismo su cui si costruiscono sia il Caso d'uso 6 di §12 sia il componente di "monitoraggio" di questa sezione.
- Export serializzabili in JSON a ogni stadio (`AlphaContract.to_contract_dict()`, `RuleDiscoveryResponse.to_dict()`) e un round-trip pickle completo per `EventCandidate` (`.persist(path)`).

### Cosa deve costruire la tua applicazione

- **Persistenza.** `forgedge` (in particolare il Modulo 4) è esplicitamente stateless tra sessioni (gli anti-goal di §14). Se ti serve un catalogo durevole di regole scoperte attraverso molte run di discovery nel tempo, quel catalogo vive nel *tuo* database, popolato dall'output di `RuleRegistry.flat_table()` / `.export()` o dalla tua stessa serializzazione dei campi di `ForgeResult`.
- **Scheduling.** Nulla in `forgedge` decide *quando* rieseguire la discovery, o quando ricontrollare una regola pubblicata contro dati freschi. È uno scheduler esterno (cron, un DAG Airflow/Prefect, un worker di coda) che chiama il codice della tua applicazione, che a sua volta chiama `forgedge`.
- **Segreti e acquisizione dati.** `forgedge` non parla mai con un exchange o un vendor di dati — portare una KPI Table in memoria (chiavi API, rate limit, retry) è interamente compito della tua applicazione, a monte dello step KPI Builder.
- **Esecuzione.** Il confine singolarmente più importante. I `BacktestParams` di una `ValidatedRule` (`buy_drop_pct`, `sell_pct`, `target_h`, `buy_delay_bar`, `fee`) sono una *specifica* di come un ordine dovrebbe comportarsi sotto la simulazione di backtest propria della pipeline — non un ordine live. Trasformare quella specifica in un ordine reale contro un exchange reale, con la propria latenza, slippage e comportamento di fill parziale, è lavoro di sistema di esecuzione che sta interamente fuori da questa libreria, per scelta esplicita di design (§2, §3).
- **Monitoraggio e alerting.** `rule_performance_report()` (§9, §12) produce uno snapshot HTML statico — non invia una notifica, non fa polling su una schedulazione, e non conosce il tuo sistema di alerting. Collegare "il segnale è ora attivo" (un flag reale che il report calcola: `EventCandidate.apply(latest_bars).iloc[-1]`) a una notifica Slack/email/webhook è il tuo strato applicativo.

### Uno schizzo di servizio minimale

Questo compone solo chiamate `forgedge` reali e verificate, avvolte in codice applicativo illustrativo (non di libreria):

```python
import pandas as pd
from dataclasses import dataclass
from forgedge import forge, ForgeResult, RuleSpec, RuleDiscovery

@dataclass
class DiscoverySession:
    """Wrapper a livello applicativo: una run forge() più ciò che serve per monitorarla in seguito."""
    ticker: str
    timeframe: str
    result: ForgeResult
    specs: list[RuleSpec]

    @classmethod
    def run(cls, kpi_table: pd.DataFrame, ticker: str, timeframe: str) -> "DiscoverySession":
        result = forge(kpi_table, ticker=ticker, timeframe=timeframe, progress=False)
        specs = RuleSpec.from_forge_result(result)
        return cls(ticker=ticker, timeframe=timeframe, result=result, specs=specs)

    def persist(self, store) -> None:
        # 'store' è IL TUO strato di persistenza — forgedge non ne ha uno.
        # AlphaContract.to_contract_dict() / RuleDiscoveryResponse.to_dict()
        # sono i mattoni serializzabili in JSON.
        rows = [
            {"ticker": self.ticker, "rule_id": spec.name,
             "contract": next(c for c in self.result.promoted
                               if c.event_candidate_id == spec.candidate.event_id).to_contract_dict()}
            for spec in self.specs
        ]
        store.save_rules(self.ticker, rows)

    def check_signals(self, fresh_bars: pd.DataFrame) -> list[str]:
        """Quali regole pubblicate si stanno attivando sulle barre più recenti, ora."""
        active = []
        for spec in self.specs:
            fires = spec.candidate.apply(fresh_bars).fillna(0).astype(bool)
            if len(fires) and fires.iloc[-1]:
                active.append(spec.name)
        return active

    def revalidate(self, eval_df: pd.DataFrame) -> dict[str, str]:
        """Ricontrolla se ogni regola pubblicata regge ancora, su dati di scoperta + nuovi.
        Usa RuleDiscovery (non AlphaDiscovery) — vedi §21 per perché questa distinzione conta."""
        by_id = {c.event_id: c for c in self.result.candidates}
        verdicts = {}
        for spec in self.specs:
            contract = next(c for c in self.result.promoted
                             if c.event_candidate_id == spec.candidate.event_id)
            cand = by_id[contract.event_candidate_id]
            resp = RuleDiscovery(eval_df, contract, cand).run()
            verdicts[spec.name] = resp.verdict
        return verdicts
```

Questo è genuinamente eseguibile contro il fixture ADA:

```python
kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
session = DiscoverySession.run(kpi, ticker="ADAUSDC", timeframe="1D")
print(f"{len(session.specs)} regole pubblicate")     # 54
print(session.check_signals(kpi))                     # regole attive sull'ultima barra del fixture
```

---

## 20. Un'architettura production-ready

Adattando il pattern generale a ciò che questo codebase effettivamente supporta — uno stadio di ricerca/discovery stateless, che alimenta uno strato di persistenza e monitoraggio che la libreria non fornisce:

```
                     ┌───────────────────────────┐
                     │  Servizio di ingestione    │   ← responsabilità applicativa:
                     │  (client exchange/vendor)  │     chiavi API, retry, rate limit
                     └─────────────┬─────────────┘
                                   │  candele OHLCV grezze
                                   ▼
                     ┌───────────────────────────┐
                     │  forgedge.kpi_builder      │   build_features / candle_features /
                     │  (feature engineering)     │   lag_features  →  KPI Table
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  forgedge.summary_report   │   gate qualità dati opt-in
                     │  (qualità dati)            │   (l'app decide: bloccare o avvisare)
                     └─────────────┬─────────────┘
                                   │  KPI Table validata
                                   ▼
              ┌────────────────────────────────────────────┐
              │        forgedge.forge()  (discovery)        │   job schedulato — NON sul path
              │  M0 Market Context → M1 Event Discovery →   │   di richiesta; una run per ticker
              │  M2 Alpha Discovery → M3 Rule Discovery →   │   per ciclo di ri-discovery
              │  M4 Rule Registry (forge_multi per molti)   │   (settimanale/mensile, non per-richiesta)
              └─────────────────────┬────────────────────────┘
                                   │  ForgeResult (candidates, contracts,
                                   │  rule_responses, calibration, ledger)
                                   ▼
                     ┌───────────────────────────┐
                     │ Persistenza applicativa    │   ← IL TUO database. forgedge non ne ha.
                     │ (catalogo regole, versioni)│     Salva dict RuleSpec/AlphaContract,
                     └─────────────┬─────────────┘     pickle .persist() dei candidati se serve.
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
        ┌─────────────────────┐      ┌─────────────────────────┐
        │  Worker di            │      │  UI di review/reporting  │
        │  monitoraggio          │      │  rule_performance_report │
        │  (schedulato, esamina  │      │  → HTML, servito o       │
        │  candele fresche,      │      │    inviato a un umano    │
        │  EventCandidate        │      └─────────────────────────┘
        │  .apply() +            │
        │  replay RuleDiscovery) │
        └──────────┬────────────┘
                    │  eventi "segnale attivo" / verdetto-cambiato
                    ▼
        ┌─────────────────────┐
        │ Sistema di alerting/  │   ← interamente esterno a forgedge, per
        │ esecuzione (review    │     scelta esplicita di design (§2, §19)
        │ umana, o un sistema    │
        │ di order management   │
        │ separato)              │
        └─────────────────────┘
```

### Flusso end-to-end, e cosa è libreria vs. applicazione a ogni passo

1. **Ingestione** (applicazione) → candele grezze.
2. **Feature engineering** (`forgedge.kpi_builder` — libreria) → KPI Table.
3. **Gate di qualità dati** (`forgedge.summary_report` — funzione di libreria, ma la *decisione* di bloccare su `has_critical` è logica applicativa; la libreria non blocca mai da sé).
4. **Discovery** (`forge()`/`forge_multi()` — libreria) — questo è lo stadio costoso, con un'apparenza di stato ma in realtà puro. Dovrebbe girare come un **job schedulato in background**, non inline in un path di richiesta: §17 ha stabilito che non c'è I/O asincrono né parallelismo interno alla libreria, quindi una chiamata `forge()` blocca il thread chiamante per tutto il tempo che i backtest walk-forward per-contratto di Rule Discovery richiedono.
5. **Persistenza** (applicazione) — i contratti promossi, le risposte di regola, e il report ledger/calibrazione di un `ForgeResult` sono il tuo record durevole. `AlphaContract.to_contract_dict()` e `RuleDiscoveryResponse.to_dict()` danno dict pronti per JSON da salvare; `EventCandidate.persist(path)` dà un round-trip pickle completo se serve ricostruire l'oggetto live in seguito (il suo metodo `.apply()`) piuttosto che solo i suoi numeri.
6. **Monitoraggio** (applicazione, costruita su primitive di libreria) — un processo schedulato separato che, su ogni nuovo batch di candele, chiama `EventCandidate.apply()` per il flag "si sta attivando proprio ora" e `RuleDiscovery(...).run()` (non `AlphaDiscovery` — §21) per "il verdetto regge ancora". `rule_performance_report()` è l'unico artefatto built-in della libreria per una versione human-readable di questo.
7. **Alerting/esecuzione** (interamente applicazione/esterno) — fuori dallo scope della libreria per scelta esplicita di design.

### Osservabilità, versioning e gestione delle risorse — supporto libreria vs. responsabilità applicativa

| Aspetto | Supporto libreria | Responsabilità applicativa |
|---|---|---|
| Logging | `forge()` logga ogni stadio della pipeline a livello `INFO` via il modulo `logging` standard indipendentemente dal flag `progress` — `logging.basicConfig(level=logging.INFO)` lo espone | Instradare quei log al tuo aggregatore; correlare i log di una run con il suo `ForgeResult` persistito |
| Report di avanzamento | `progress=True` (default) stampa un tracker di stadio e una progress bar `tqdm` (o un fallback senza dipendenze) su `stderr` | Esporre l'avanzamento in una UI, se presente |
| Errori | Eccezioni built-in semplici, nessuna gerarchia custom (§11) | Catturarle al confine del tuo job di discovery; decidere la policy retry vs. fail |
| Metriche | Nessuna built-in — `HypothesisLedger`/`CalibrationReport` sono dati strutturati che puoi trasformare tu stesso in metriche | Esportare `ledger.m2_surface`, `calibration.tippett_p`, la distribuzione dei conteggi per verdetto, ecc. al tuo sistema di metriche |
| Timeout | Nessuno — una chiamata `forge()` gira fino al completamento o solleva un'eccezione | Avvolgere la chiamata nel tuo timeout/cancellazione se uno scheduler di job lo richiede |
| Retry | Nessuno — la libreria è deterministica dato lo stesso input, quindi un retry ingenuo è sicuro ma inutile a meno che l'input stesso non fosse il problema | La logica di retry appartiene allo strato di ingestione dati, dove i fallimenti I/O transitori avvengono realmente |
| Caching | Nessuno — ogni chiamata `forge()` ricalcola tutto dalla KPI Table che passi | Cachare `ForgeResult` intermedi (o almeno `event_frame`) chiave per hash dell'input, se rieseguí la discovery su dati in gran parte invariati |
| Segreti | Nessuno — la libreria non prende credenziali di alcun tipo | Le chiavi API per la tua sorgente dati vivono interamente nel tuo strato di ingestione, mai vicino a `forgedge` |
| Versioning | `forgedge.__version__`; i golden test della libreria stessa dimostrano che il comportamento *può* cambiare tra versioni anche a config fissa (§14, §18) | Fissa la tua versione di `forgedge`; riesegui i tuoi test di regressione (§18) all'upgrade, aspettandoti che alcuni valori fissati necessitino legittimamente un ri-fissaggio |
| Rollback | Nessuno — stateless, nulla da annullare dentro la libreria | Il rollback significa tornare a un catalogo di regole precedentemente persistito nel *tuo* database |

---

## 21. Troubleshooting

Ogni voce: sintomo → causa probabile → come verificarla → correzione → come prevenirla in futuro.

### "Zero contratti promossi, o ogni contratto è `direction='undetermined'`"

- **Causa A — genuinamente nessun evento predittivo in questa combinazione dati/config.** Non tutto ciò che una KPI Table produce avrà potere predittivo; questo può essere un risultato corretto e onesto.
- **Causa B — l'indice del frame osservato non corrisponde alla serie di attivazione di training dell'evento.** Verifica: `AlphaDiscovery`/`RuleDiscovery` ha emesso l'`UserWarning` su "candles whose index differs from the event's stored activation series", forse seguito dalla variante più forte sul conteggio di attivazioni che collassa sotto il 10% del conteggio di training (§11)? Se sì, è questa la causa. **Correzione:** se intendevi estendere la finestra di training con nuove barre, passa `pd.concat([train_df, new_bars_df])`, non solo `new_bars_df` da sola — le baseline delle trasformazioni mobili (pctrank, z-score) hanno bisogno della storia precedente per significare la stessa cosa che significavano durante la discovery. **Prevenzione:** tratta quello specifico `UserWarning` come un errore in CI per qualsiasi percorso di codice che dovrebbe estendere, non sostituire, la finestra di discovery.
- **Causa C — la griglia di orizzonti genuinamente non copre la scala temporale dell'evento.** Verifica via `c.derived_target.score_by_h` / `.advantage_by_h` — la curva del punteggio sta ancora salendo al bordo della griglia? `AlphaConfig.horizon_enrichment` (attivo di default) mitiga questo ma non lo elimina per eventi con scala temporale genuinamente lunga. **Correzione:** allarga esplicitamente `horizon_grid`.

### "Ho eseguito uno script `examples/*.py` e ho ottenuto `TypeError: GateParams.__init__() got an unexpected keyword argument 'min_act'`"

- **Causa:** confermata, reale, riproducibile (§11) — diversi script di esempio in questo repository precedono un cambio di API di `GateParams` e lo costruiscono ancora con i vecchi nomi di campo (`min_act`, `min_months`, `max_conc`). **Correzione:** traduci ai campi attuali — `GateParams(min_tpm=2.0, max_dispersion=2.5, event_counting="bar")` riproduce l'intento dello script vecchio nel modo più fedele (`event_counting="bar"` ripristina la semantica di conteggio pre-refactor che i vecchi campi `min_act`/`min_months`/`max_conc` implicavano). **Verifica quali script sono coinvolti:** `alpha_discovery_usage.py`, `extended_usage.py`, `kpi_table_1d.py`, `search_rotation_calibration.py`, `lowfreq_null_diagnostic.py`, `lowfreq_endpoint_diagnostic.py`. `kpi_builder_usage.py` non è coinvolto (verificato, §13).

### "Tutti i miei verdetti sono `PARTIAL-EDGE`, mai `EDGE` pieno"

- **Causa:** quasi certamente il gate rotation null di default (`rotation_p > SelectionCriteria.max_rotation_p`, default 0.05), non un bug. Verifica: controlla `rejection_reasons` per `"search-level rotation null not cleared"`. Questo è atteso, comune, e — secondo l'audit della libreria stessa — il comportamento *inteso* su dataset dove la superficie di ricerca è ampia rispetto alla dimensione del campione (§14, e gli esempi reali ADA/AMZN in §7/§13, dove è successo su entrambi).
- **Correzione, se hai confermato che vuoi davvero vedere oltre il gate per scopi ispettivi:** `forge(..., fast_null=False)` o alza `SelectionCriteria.max_rotation_p` — ma leggi la voce costo/rischio di §15 per entrambi prima di farlo per qualunque scopo diverso dal debug.

### "L'output di `RuleRegistry` ha colonne `classification`/`is_generic` vuote"

- **Causa:** una sessione a singolo ticker. La classificazione cross-ticker è matematicamente indefinita senza nulla contro cui testarla — ogni documento diventa `ISOLATED` per mancanza di ticker di test, non perché la regola abbia fallito un test. Verifica: `reg.documents[i].cross_ticker_total == 0`.
- **Correzione:** usa `forge_multi()` su almeno due ticker se vuoi una classificazione genuina `GENERIC`/`PARTIAL`/`SPECIFIC`.

### "Una colonna feature custom non compare mai combinata con nulla"

- **Causa:** il nome della colonna non corrisponde alla convenzione di riconoscimento di famiglia `{base}_{indicatore}_{periodo}` di Event Discovery (§9). Funziona comunque come feature standalone — semplicemente non viene mai accoppiata in un ratio/spread. Verifica: rinominala per corrispondere alla convenzione e riesegui; se ora compaiono candidati che la coinvolgono come *coppia*, questa era la causa.

### "`summary_report` segnala `scale_mixed` su dati di cui sono sicuro"

- **Causa:** `summary_report` segnala un salto di magnitudine ≥ 2 ordini di 10 tra barre come probabile bug del feed dati (es. alcune barre in centesimi, altre in dollari). Una storia di prezzo genuina su più ordini di magnitudine (es. un asset passato da $0.01 a $50) può attivare questo legittimamente.
- **Verifica:** ispeziona `[f for f in rep.findings if f.code == "scale_mixed"].message` — riporta il min/mediana/max effettivi, così puoi giudicare se lo spread è un range di prezzo storico reale o un bug di conversione di unità.
- **Correzione:** se è una storia di prezzo reale, questo finding è un falso positivo che puoi tranquillamente notare e superare — `summary_report` non blocca mai nulla da sé (§11); non c'è nulla da "correggere" nei tuoi dati.

### "`RuleDiscovery.__init__` solleva `ValueError` su un candidato/contratto non corrispondente"

- **Causa:** hai passato un `EventCandidate` il cui `event_id` non corrisponde a `contract.event_candidate_id` — un errore comune quando si costruisce male il dizionario di lookup `by_id = {c.event_id: c for c in candidates}`, o si riusa un candidato da una diversa esecuzione di `forge()`.
- **Correzione:** deriva sempre il candidato dallo *stesso* output `ForgeResult`/`EventDiscovery.run()` del contratto: `by_id[contract.event_candidate_id]`, come mostrato in tutto §9/§12.

### "Rieseguire Alpha Discovery su nuovi dati per 'verificare se l'edge regge ancora' dà risultati peggiori o insensati"

- **Causa:** questo è un **errore di metodologia**, non un bug — `AlphaDiscovery` *ri-deriva* direzione e orizzonte da qualunque dato tu le dia. Su una storia che copre regimi di mercato incompatibili, la stessa condizione booleana può aver preceduto rendimenti di segno opposto in regimi diversi, e la ri-derivazione può ribaltare la direzione o collassare a `undetermined`, anche se l'edge *originale* scoperto è intatto.
- **Correzione:** lo strumento corretto per "una regola pubblicata regge ancora" è **`RuleDiscovery`**, non `AlphaDiscovery` — riproduce il target *fisso*, precedentemente derivato, invece di ri-derivarlo. Vedi l'API del Modulo 3 in §9, il Caso d'uso 6 in §12, e lo schizzo di `revalidate()` in §19.

---

## 22. Best practice

- **Esegui `summary_report()` prima di ogni sessione di discovery, e decidi esplicitamente cosa fare con `has_critical`/`has_warnings`.** Costa quasi nulla e la libreria non lo farà mai per te (§11).
- **Passa `ed.df` (o `ForgeResult.event_frame`), non la KPI Table originale, a qualunque cosa sia a valle di Event Discovery quando costruisci la pipeline a mano.** Questa è la fonte singola più comune di `KeyError` su una colonna feature derivata (§9).
- **Tratta un risultato solo-`PARTIAL-EDGE` come informativo, non come un fallimento da correggere.** Data la storia documentata della libreria stessa su una pipeline precedentemente troppo permissiva (§14), una run che produce prevalentemente `PARTIAL-EDGE` e pochi o zero `EDGE` è molto spesso la pipeline che è onesta su un dataset genuinamente difficile — controlla `rejection_reasons` prima di assumere che qualcosa sia mal configurato.
- **Usa `forge_preset()` invece di assemblare a mano tre oggetti config, a meno che tu non abbia una ragione specifica per non farlo.** I preset esistono specificamente per mantenere coerenti tra loro le impostazioni di M1/M2/M3 (§10) — un modo comune di ottenere un risultato sottilmente sbagliato è tarare la config di un modulo senza aggiustare corrispondentemente quella di un altro.
- **Fissa la tua versione di `forgedge` e tratta un test in stile golden sulla tua stessa logica a valle come parte di prima classe della tua suite di test (§18).** Il comportamento della libreria stessa è legittimamente cambiato tra versioni (l'intera narrazione di §14 è una serie di tali cambiamenti); il codice che dipende da `forgedge` dovrebbe accorgersene quando succede, secondo la tua schedulazione, non silenziosamente.
- **Usa `RuleDiscovery`, mai `AlphaDiscovery`, per verificare se un edge precedentemente scoperto regge ancora su nuovi dati.** Questa è la singola regola di correttezza più consequenziale di questo manuale (§21).
- **Logga `result.ledger.describe()` e `result.calibration.summary()` (o `.tippett_p`) insieme a ogni run di discovery che persisti.** Sono economici, già calcolati, e sono esattamente i numeri che vorrai in seguito se mai dovrai spiegare *perché* un dato verdetto è stato o non è stato raggiunto.

---

## 23. Anti-pattern

- **Ri-derivare il target di Alpha Discovery su dati freschi per "monitorare" una regola pubblicata.** Coperto in profondità in §21 — lo strumento corretto è `RuleDiscovery`. Farlo in modo sbagliato non causa un crash; produce silenziosamente un risultato dall'aspetto peggiore (o `undetermined`) che si legge come "l'edge è decaduto" quando la causa reale è metodologica.
- **Disattivare la rotation null (`fast_null=False`) per "ottenere più verdetti EDGE".** Questo non rende le tue regole migliori — rimuove il controllo che è stato aggiunto specificamente, con giustificazione misurata (§14), perché il comportamento di default precedente della pipeline promuoveva il rumore quasi quanto il segnale su dati a bassa frequenza. Se ti trovi a fare questo per far apparire migliore un risultato, è esattamente la situazione che il gate esiste per catturare.
- **Copiare `GateParams(min_act=..., min_months=..., max_conc=..., min_tpm=...)` dagli script `examples/*.py` di questo repository senza controllare che siano aggiornati.** Non lo sono (§11, §21) — questo specifico anti-pattern non è ipotetico, è riproducibile oggi contro esattamente questo codebase.
- **Trattare un verdetto non-`EDGE` come "nessun segnale" senza leggere `rejection_reasons`.** Una response `PARTIAL-EDGE` o persino `NON-EDGE` porta spesso stringhe diagnostiche specifiche e attuabili (un p-value della rotation null, un rapporto di mesi attivi, un numero di fill rate) che dicono esattamente cosa è marginale — scartare l'oggetto response e controllare solo `verdict` butta via quell'informazione.
- **Costruire `AlphaDiscovery` direttamente (bypassando `forge()`) su dati giornalieri-o-più-lenti senza impostare esplicitamente `horizon_grid`.** La sostituzione automatica della griglia giornaliera di `forge()` (§7) avviene solo dentro `forge()` stessa — costruire `AlphaDiscovery` a mano non la ottiene, e scansionerai silenziosamente periodi di detenzione fino a 48 giorni di default (§9, §21).
- **Assumere che un `AlphaContract` promosso (`status="HYPOTHESIS"`) sia esso stesso un segnale di trading.** È l'output del Modulo 2, non del Modulo 3 — non è ancora stato backtestato con meccaniche d'ordine realistiche né validato out of sample da Rule Discovery. Tratta `AlphaContract.status == "HYPOTHESIS"` come "vale la pena backtestarlo", non "vale la pena tradarlo".
- **Mockare `forge()` nei test della tua stessa applicazione.** Come coperto in §18 — non c'è I/O da fingere, e l'intero valore della chiamata sta nel suo output statistico reale; un mock non dice nulla sulla correttezza del tuo codice di integrazione.

---

## 24. FAQ

**`forgedge` piazza ordini o si connette a un exchange?**
No. Non ha alcuna capacità di esecuzione, per scelta esplicita di design (§2, §3, §19). Produce specifiche di regole; un sistema di esecuzione che costruisci separatamente le implementa.

**Perché la mia run di `forge()` ha promosso molti contratti ma è finita con pochissimi (o zero) verdetti `EDGE` pieni?**
È molto probabilmente il controllo rotation null di default che fa esattamente ciò per cui è pensato (§14, §21) — controlla `rejection_reasons` per `"search-level rotation null not cleared"` prima di assumere che qualcosa sia sbagliato.

**Posso usare `forgedge` su dati a 1 minuto o tick-level?**
Nulla lo impedisce tecnicamente, ma ogni esempio funzionante, analisi di calibrazione, e lo studio di robustezza a bassa frequenza della libreria stessa sono costruiti e testati su barre orarie-giornaliere. Finestre in-sample molto brevi a frequenza molto alta introducono gli stessi problemi di potere statistico che §16 e §21 descrivono per i dati giornalieri, probabilmente in modo più acuto. Questo manuale non può verificare il comportamento a quella frequenza perché nulla nel repository lo testa direttamente.

**Perché la `horizon_grid` di default della classe `AlphaConfig` sembra sbagliata per i miei dati giornalieri?**
Perché è calibrata sull'orario per progetto, e `forge()` (non `AlphaConfig` stessa) sostituisce automaticamente una griglia calibrata sul giornaliero per te quando `timeframe` è un giorno o più lento (§7, §9, §14). Costruire `AlphaDiscovery` direttamente bypassa quella sostituzione.

**L'output di `forge()` è deterministico?**
Dato lo stesso DataFrame di input e la stessa configurazione, sì — non c'è casualità nella logica propria di Event/Alpha/Rule Discovery. `RotationCalibrator` (non il `FastRotationNull` di default, che è esatto/esaustivo) usa un RNG seedato (`RotationConfig.seed`, default `20260624`) per le sue estrazioni campionate, quindi anch'esso è riproducibile dato lo stesso seed.

**`forgedge` persiste qualcosa tra le run?**
No — esplicitamente e per scelta di design (gli anti-goal di §14, §19). Ogni chiamata `forge()` è una funzione pura del suo input; la persistenza è interamente responsabilità della tua applicazione.

**Qual è la differenza tra un `EventCandidate`, un `AlphaContract`, e una `RuleDiscoveryResponse`?**
Sono gli output rispettivamente dei Moduli 1, 2 e 3 (§4, §8) — una condizione booleana senza ancora significato economico; la stessa condizione più un target economico derivato e un voto statistico; e infine un verdetto backtestato e validato walk-forward. Una `RuleSpec` (§9, §12) è un quarto oggetto, più leggero, che raggruppa ciò che serve per *riprodurre* in seguito una regola tradabile.

**Perché il `min_tpm` di default di `GateParams` è così basso (0.5)?**
È deliberatamente permissivo così che il Consistency Gate non scarti eventi genuinamente rari-ma-reali prima ancora che raggiungano i controlli statistici molto più potenti di Alpha Discovery (§10, §16). Se scopri che Event Discovery produce troppi candidati di bassa qualità, alzare `min_tpm` è la leva prevista (il trade-off frequenza-vs-selettività di §16) — ma capisci il trade-off prima di farlo.

---

## 25. Glossario

- **AND composition** — combinare due (o tre) eventi a singola colonna con un AND booleano per formare un evento composto più specifico, esso stesso riverificato contro il Consistency Gate.
- **AlphaContract** — l'oggetto di output del Modulo 2: un candidato evento più un target economico derivato (direzione, periodo di detenzione, take-profit) e un voto statistico.
- **base rate** — il win rate incondizionato del target binario derivato, misurato su *tutte* le barre, non solo quelle attive — la baseline contro cui si misura il `lift`.
- **Consistency Gate** — il filtro del Modulo 1 sui candidati evento grezzi: rate minimo di attivazione, dispersione massima (burstiness), e (in modalità `"episode"`) un conteggio minimo di episodi.
- **DerivedTarget** — la tripla `(holding_period_h, sell_pct, direction)` che Alpha Discovery calcola per un candidato evento, dai dati, mai assunta.
- **dispersione (Index of Dispersion)** — Varianza/Media dei conteggi mensili di attivazione di un evento; 1.0 per un processo Poisson (senza memoria), più alto per attivazioni bursty/raggruppate.
- **EDGE / PARTIAL-EDGE / NON-EDGE / INSUFFICIENT-DATA** — i quattro possibili verdetti di Rule Discovery (§8, §9, §15); `is_edge` è vero per i primi due.
- **embargo** — un buffer aggiuntivo opzionale di barre all'inizio di una finestra OOS, oltre a ciò che il purging rimuove, per mettere ulteriormente in quarantena l'autocorrelazione seriale. Opt-in, default 0 (§15).
- **EventCandidate** — l'oggetto di output del Modulo 1: un'espressione booleana immutabile più le sue statistiche di attivazione e il risultato del gate.
- **fill rate** — la frazione di segnali in cui l'ordine limite simulato si è effettivamente riempito entro `buy_delay_bar` barre.
- **HypothesisLedger** — un oggetto `ForgeResult.ledger` che registra quanto fosse ampia la superficie di ricerca di una sessione (candidati × orizzonti × celle di griglia) — contabilità, non una correzione.
- **IC (Information Coefficient)** — la correlazione di Spearman tra il valore grezzo di una feature e il rendimento forward all'orizzonte derivato, calcolata in-sample.
- **KPI Table** — il `pandas.DataFrame` di input: `close` + una sorgente datetime + un numero qualsiasi di colonne feature.
- **lift** — win rate sulle barre attive meno il base rate.
- **MAE / MFE** — Maximum Adverse / Favorable Excursion — il rendimento non realizzato peggiore/migliore sperimentato da un trade tra fill e uscita.
- **regime** — la classificazione ordinata a 5 livelli della condizione di mercato del Modulo 0 (`STRONG_BEAR`…`STRONG_BULL`).
- **rotation null** — una calibrazione statistica a livello di ricerca che ruota circolarmente la colonna `close` (disaccoppiando il timing dell'evento dall'esito) per costruire una distribuzione null empirica per la migliore statistica propria della pipeline, correggendo per l'esposizione al multiple-testing dell'intera ricerca (§14, §15).
- **RuleDiscoveryResponse** — l'oggetto di output del Modulo 3: il verdetto più ogni statistica di supporto (summary IS, risultato walk-forward, validazione statistica, execution envelope).
- **RuleSpec** — un pacchetto leggero (nome, candidato, parametri, verdetto) per riprodurre una regola tradabile contro nuovi dati, indipendentemente dal Rule Registry.
- **TimeBudget** — lo split condiviso, purgato (ed eventualmente embargato) di indici di barra IS/OOS usato da Event e Alpha Discovery.
- **walk-forward** — validare un punto operativo fisso su una sequenza di finestre di test out-of-sample rolling, ciascuna preceduta dalla propria finestra di train, concatenate in un unico track record OOS.

---

## 26. Reference API (consultazione rapida)

Questo è un indice compatto, non un sostituto della trattazione più completa di §9-10. Ogni voce nomina il modulo in cui vive.

| Simbolo | Modulo | Scopo in una riga |
|---|---|---|
| `forge()` | `forgedge` | esegue l'intera pipeline end to end |
| `forge_multi()` | `forgedge` | `forge()` per ticker + un registry cross-ticker unificato |
| `ForgeResult` | `forgedge` | il valore di ritorno di `forge()` — ogni artefatto intermedio |
| `forge_preset()` / `preset_info()` / `PRESETS` | `forgedge` | triple pretarate `(DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig)` |
| `MarketContext` / `MarketContextConfig` / `EMAProxyConfig` | `forgedge` | Modulo 0 — classificazione del regime |
| `EventDiscovery` / `DiscoveryConfig` / `EventCandidate` / `CustomEvent` | `forgedge` | Modulo 1 — estrazione eventi |
| `AlphaDiscovery` / `AlphaConfig` / `AlphaContract` / `PromotionThresholds` | `forgedge` | Modulo 2 — derivazione del target e misura del potere predittivo |
| `RuleDiscovery` / `RuleDiscoveryConfig` / `RuleDiscoveryResponse` / `BacktestParams` / `SelectionCriteria` | `forgedge` | Modulo 3 — backtest realistico e verdetto walk-forward |
| `RuleRegistry` / `RegistryConfig` / `RuleSubmission` / `RuleDocument` | `forgedge` | Modulo 4 — dedup, cross-ticker, catalogo |
| `build_features()` / `candle_features()` / `lag_features()` / `pattern_features()` | `forgedge` | da candele grezze a KPI Table |
| `summary_report()` / `DataQualityReport` / `Finding` | `forgedge` | diagnostica qualità dati opt-in |
| `TimeBudget` | `forgedge` | split IS/OOS purgato/embargato condiviso |
| `HypothesisLedger` | `forgedge` | contabilità della superficie di ricerca |
| `FastRotationNull` / `RotationCalibrator` / `RotationConfig` / `CalibrationReport` | `forgedge` | calibrazione per multiple-testing a livello di ricerca |
| `TargetOptimizer` / `TargetConfig` | `forgedge` | workflow alternativo standalone target-first |
| `RuleSpec` / `rule_performance_report()` | `forgedge` | riproduce regole pubblicate su nuove candele, report HTML |
| `text_report()` / `html_report()` | `forgedge.rule_discovery` | report human/HTML da una `RuleDiscoveryResponse` |
| `run_backtest()` | `forgedge.rule_discovery` | il motore di backtest a singola configurazione sottostante |

Per la lista completa dei campi e default di ogni dataclass, vedi §10 (quelli che più probabilmente andrai a tarare) o direttamente il sorgente — ogni default citato in questo manuale è stato verificato contro il pacchetto installato al momento della scrittura.

---

*Questo manuale è stato scritto leggendo il codice sorgente di `forgedge`, la sua suite di test, la sua stessa documentazione e le note di analisi di design, ed eseguendo codice reale contro la versione della libreria installata da questo repository. Dove un'affermazione riflette il ragionamento dichiarato dagli autori stessi piuttosto che la lettura del codice fatta da questo manuale, è citata e attribuita come tale in tutto il testo. L'edizione inglese gemella, `docs/manual-en.md`, copre contenuti identici.*

