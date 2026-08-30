# FORGE — Feature-Oriented Rule Generation Engine

FORGE è un sistema di ricerca quantitativa per la **scoperta sistematica di regole
di trading algoritmico** da dati storici di mercato. A partire da una KPI Table
(OHLCV + indicatori tecnici), FORGE identifica eventi booleani con struttura
temporale stabile, ne misura il potere predittivo rispetto a un target economico
derivato dai dati, e produce contratti formali pronti per la validazione operativa.

---

> **Cerchi un riferimento unico e completo?** Il
> [manuale di forgedge](../../../../docs/manuale-it.md) è una guida completa,
> verificata negli esempi, che copre l'installazione, l'architettura di
> produzione, la risoluzione dei problemi, una FAQ e un glossario. È il punto
> di partenza consigliato; questo indice e le specifiche dei moduli qui sotto
> approfondiscono i dettagli interni di ciascuna fase della pipeline.

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
result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.summary())                         # una riga per candidato + rule_verdict
for contract, response in result.edges():       # solo EDGE / PARTIAL-EDGE
    print(contract.alpha_id, response.verdict)
print(result.registry.summary())                # Modulo 4 — regole catalogate
```

Sessioni multi-ticker con `forge_multi`:

```python
from forgedge import forge_multi

frames = {"BTCUSDC": btc_kpi, "ETHUSDC": eth_kpi, "ADAUSDC": ada_kpi}
results, registry = forge_multi(frames, timeframe="1H")

# GENERIC: la regola si generalizza su ≥ 2/3 dei ticker testati
df = registry.flat_table()
print(df[["rule_id", "classification", "pf", "cross_ticker_score"]])

# Report HTML autocontenuto (SVG inline, nessuna CDN)
html = registry.html_report(timeframe="1H")
with open("report.html", "w") as f:
    f.write(html)
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

**Configurazione centralizzata e validata.** Ogni run di `forge()` risolve la
configurazione dei singoli moduli tramite un `PipelineContext` centrale
(`resolver.py`), che completa i campi non impostati e poi valida l'intero
bundle con `config_report()`. Per default (`strict=True`) una configurazione
strutturalmente incoerente solleva `ValueError` **prima** che la pipeline
venga eseguita, invece di girare fino in fondo producendo silenziosamente un
muro di candidati respinti. Passare `strict=False` degrada questi rilievi a
warning e consente comunque l'esecuzione.

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
| `configuration_it.md` | Riferimento completo alla configurazione: ogni campo dataclass, tipo, default e descrizione |
| `playground_it.md` | Playground: guida all'utilizzo degli helper di analisi sopra i `ForgeResult` messi in pool (tutti gli 11 casi d'uso, issue #237) |
| `deployment_it.md` | Deployment: guida all'utilizzo delle funzioni di gate/export/monitoraggio che portano le regole scoperte in produzione |

`forgedge.playground` — un livello di analisi di sola lettura sopra i `ForgeResult` messi in pool — è coperto nel manuale principale (`docs/manuale-it.md`, §9). La sua checklist di tracciamento (issue #237) è completa (10 funzioni più il trasversale `conversion_funnel`), ma resta uno strato diagnostico, non un'API core stabile. [`playground_it.md`](playground_it.md) è il suo riferimento dettagliato all'utilizzo (firme, parametri, esempi verificati); [`modules/Playground.md`](../modules/Playground.md) copre invece la motivazione di design e gli algoritmi interni.

`forgedge.deployment` — il suo modulo gemello per portare le regole promosse in produzione (gate di qualità, export su disco, manifest di monitoraggio) — è stato separato da `forgedge.playground` da PR #247 (issue #245) perché quelle funzioni hanno effetti reali che un nome di sola lettura non descriveva più onestamente. [`deployment_it.md`](deployment_it.md) è il suo riferimento all'utilizzo; [`modules/Deployment.md`](../modules/Deployment.md) copre la motivazione di design.

Le versioni inglesi sono nei file corrispondenti `*_en.md`.

Lavori su questo codebase con Claude Code? La skill `forgedge` in
`.claude/skills/forgedge/` copre l'API della libreria, i verdetti e gli
invarianti della pipeline in una forma che Claude può usare direttamente.
