# FORGE — Feature-Oriented Rule Generation Engine

FORGE è un sistema di ricerca quantitativa per la **scoperta sistematica di regole
di trading algoritmico** da dati storici di mercato. A partire da una KPI Table
(OHLCV + indicatori tecnici), FORGE identifica eventi booleani con struttura
temporale stabile, ne misura il potere predittivo rispetto a un target economico
derivato dai dati, e produce contratti formali pronti per la validazione operativa.

---

## Cos'è FORGE

La ricerca sistematica di edge di trading soffre di tre problemi ricorrenti:
look-ahead bias nella selezione dei segnali, ottimizzazione delle soglie sullo
stesso campione usato per valutare il segnale, e mancanza di separazione tra la
fase statistica e quella operativa.

FORGE affronta questi problemi con un'architettura a pipeline rigidamente
separata:

- **Modulo 1 non vede mai il forward return.** Gli eventi vengono scoperti
  unicamente dalla struttura temporale degli indicatori — distribuzione, frequenza,
  stabilità — senza alcuna esposizione al rendimento futuro.
- **Le soglie sono immutabili.** Una volta fissate da Event Discovery, le soglie
  degli eventi (`RSI < 30.5`, `spread_ema < -0.012`) attraversano la pipeline
  invariate. Nessun modulo successivo può ricalibrarle.
- **Il target è derivato dai dati per evento.** Alpha Discovery non riceve
  parametri economici in input: per ogni evento scansiona una grid di orizzonti
  e seleziona quello che massimizza la separazione statistica tra barre attive
  e inattive. Il target non è un'assunzione, è una misura.
- **La conferma out-of-sample è un gate, non un check.** La validazione OOS
  sull'ultimo 30% del dataset è un requisito formale per la promozione — non un
  check opzionale a posteriori.

---

## Pipeline

```
KPI Table (OHLCV + indicatori tecnici)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 0 — Market Context                                       │
│  Classifica ogni barra per regime di mercato (5 livelli).        │
│  Output: KPI Table + colonne 'regime' e 'regime_stable'          │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 1 — Event Discovery                                      │
│  Scopre eventi booleani dalla struttura temporale degli          │
│  indicatori. Non vede mai il forward return.                     │
│  Output: list[EventCandidate]                                    │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 2 — Alpha Discovery                                      │
│  Deriva il target per evento, misura il potere predittivo IS,    │
│  e confirma sull'OOS tail. Prima esposizione al forward return.  │
│  Output: list[AlphaContract]                                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 3 — Rule Discovery                                       │
│  Backtest realistico con order mechanics (limit order, fee).     │
│  Output: Edge/Non-Edge + parametri operativi                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 4 — Rule Registry                                        │
│  Deduplicazione, backtest cross-ticker, export report.           │
│  Output: Tabella piatta + report HTML con badge genericity        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stato implementazione

| Modulo | Stato |
|---|---|
| 0 — Market Context | ✅ Implementato |
| 1 — Event Discovery | ✅ Implementato |
| 2 — Alpha Discovery | ✅ Implementato |
| 3 — Rule Discovery | ✅ Implementato |
| 4 — Rule Registry | ✅ Implementato |

---

## Installazione

FORGE dipende unicamente da `numpy` e `pandas`. Nessuna dipendenza da `scipy`,
`statsmodels` o librerie di ML: tutte le primitive statistiche (Spearman, t-test,
regressione OU, FDR Benjamini-Hochberg, funzione beta incompleta) sono
implementate in puro numpy.

```bash
pip install numpy pandas
```

---

## Quick start

```python
import pandas as pd
from forgedge import forge

# KPI Table con OHLCV + indicatori tecnici (colonna 'close' richiesta)
kpi = pd.read_parquet("kpi_table.parquet")

# Pipeline completa: da KPI Table a regole validate in una chiamata
result = forge(kpi, asset="BTC", timeframe="1H")

print(result.summary())                         # una riga per candidato + rule_verdict
for contract, response in result.edges():       # solo EDGE / PARTIAL-EDGE
    print(contract.alpha_id, response.verdict)
```

L'orchestratore accetta la configurazione di ogni modulo come argomento
opzionale. Per la pipeline step-by-step con configurazione completa, vedere
`how_to_use_it.md`.

---

## Principi di design

**Separazione dei domini.** Ogni modulo risponde a una sola domanda. Modulo 1:
"questo evento ha struttura temporale stabile?" Modulo 2: "questo evento predice
il target?" Modulo 3: "questo pattern è operativamente tradabile?" Nessun modulo
risponde alla domanda del successivo, nessuno accede ai dati del precedente al
di là dell'artefatto formale prodotto.

**Soglie immutabili.** Le soglie degli eventi vengono fissate da Event Discovery
sulla base della distribuzione in-sample dell'asset e non vengono mai modificate
dai moduli a valle. Ricalibrarle richiederebbe una nuova sessione di scoperta.

**Soglie distribuzionali.** Le soglie non sono valori assoluti ma percentili della
distribuzione della feature sull'asset specifico. Lo stesso pattern strutturale
produce soglie diverse su asset diversi (`RSI p10 = 30.5` su ADA, `27.8` su BTC)
mantenendo la stessa semantica statistica.

**No look-ahead bias.** Il forward return non entra mai in Event Discovery.
Il target economico è derivato da Alpha Discovery **per ogni evento** su una
finestra IS, e confermato su un tail OOS che non ha partecipato a nessun calcolo
precedente.

---

## Documentazione

| File | Contenuto |
|---|---|
| `concepts_it.md` | Guida concettuale: evento, alpha e regola — dal mercato al segnale |
| `how_to_use_it.md` | Guida pratica alla pipeline end-to-end per produzione |
| `modulo_0_it.md` | Market Context: regime, EMAProxy, configurazione |
| `modulo_1_it.md` | Event Discovery: pipeline 5-step, EventCandidate, walk-forward |
| `modulo_2_it.md` | Alpha Discovery: target derivato, OOS, AlphaContract |
| `modulo_3_it.md` | Rule Discovery: backtest, verdetto EDGE, walk-forward OOS, report |
| `modulo_4_it.md` | Rule Registry: deduplicazione, cross-ticker, genericity, export |

Le versioni inglesi sono nei file corrispondenti `*_en.md`.
