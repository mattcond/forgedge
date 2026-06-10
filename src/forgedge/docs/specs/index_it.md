# FORGE — Manuale Tecnico della Libreria (v0.1.0)

FORGE (**Feature-Oriented Rule Generation Engine**) è un sistema di ricerca
quantitativa che scopre, valida e formalizza regole di trading algoritmico a
partire da dati storici di mercato.

Il sistema è organizzato come una pipeline modulare a 5 stadi. I moduli si
eseguono in sequenza: ogni stadio consuma l'output del precedente e produce
un artefatto formale che diventa l'input del successivo.

---

## Pipeline e moduli

```
KPI Table (OHLCV + indicatori tecnici)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modulo 0 — Market Context                                      │
│  Classifica ogni barra per regime di mercato.                   │
│  Output: KPI Table + colonne 'regime' e 'regime_stable'         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modulo 1 — Event Discovery                                     │
│  Scopre eventi booleani dalla struttura temporale degli         │
│  indicatori. Non vede mai il forward return.                    │
│  Output: list[EventCandidate]                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modulo 2 — Alpha Discovery                                     │
│  Misura il potere predittivo degli eventi rispetto a un         │
│  target economico. Prima esposizione al forward return.         │
│  Output: list[AlphaContract]                                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modulo 3 — Rule Discovery            [non implementato v0.1.0] │
│  Backtest realistico con order mechanics (limit order, fee).    │
│  Output: Edge/Non-Edge + parametri operativi                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modulo 4 — Rule Registry             [non implementato v0.1.0] │
│  Deduplicazione, backtest cross-asset, export CSV/report HTML.  │
│  Output: Regole validate pronte per il sistema di esecuzione    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installazione e dipendenze

FORGE dipende unicamente da `numpy` e `pandas`.
Non è richiesto `scipy` o `statsmodels`: tutte le primitive statistiche
(Spearman, t-test, regressione OU, FDR Benjamini-Hochberg) sono implementate
in puro numpy.

```python
pip install numpy pandas
```

---

## Esempio minimo end-to-end

```python
import pandas as pd
from forgedge import (
    MarketContext,
    EventDiscovery,
    AlphaDiscovery, AlphaConfig, TargetDefinition,
)

# KPI Table con OHLCV + indicatori tecnici
kpi = pd.read_parquet("kpi_table.parquet")   # deve avere colonna 'close'

# Modulo 0 — regime di mercato
enriched = MarketContext(kpi).run()            # aggiunge 'regime' e 'regime_stable'

# Modulo 1 — scoperta eventi
ed = EventDiscovery(enriched)
candidates = ed.run()                          # list[EventCandidate]
print(f"{len(candidates)} candidati trovati")
print(ed.summary().head())

# Modulo 2 — misura alpha
ad = AlphaDiscovery(
    ed.df,
    candidates,
    AlphaConfig(target=TargetDefinition(holding_period_h=24, sell_pct=0.04)),
)
contracts = ad.run()
promoted = ad.promoted_contracts()             # list[AlphaContract] con status "HYPOTHESIS"
print(f"{len(promoted)} candidati promossi su {len(contracts)}")
print(ad.summary().head())
```

---

## Struttura della documentazione

| File | Modulo | Argomenti trattati |
|---|---|---|
| `modulo_0_it.md` | Market Context (0) | Classificazione regime, EMAProxy, configurazione, metodi di output |
| `modulo_1_it.md` | Event Discovery (1) | Pipeline 5-step, strutture dati, walk-forward, SQL/formula export |
| `modulo_2_it.md` | Alpha Discovery (2) | Misurazione IC, win rate, regime, scoring, Alpha Contract |

Le versioni inglesi sono nei file `modulo_0_en.md`, `modulo_1_en.md`, `modulo_2_en.md`.

---

## Principi architetturali fondamentali

**Separazione dei domini.** Ogni modulo risponde a una sola domanda:
il Modulo 1 chiede solo "questo evento ha struttura temporale stabile?"
senza mai vedere il forward return. Il Modulo 2 chiede "questo evento predice
il target?" senza ottimizzare le soglie. Il Modulo 3 chiede "questo pattern è
operativamente tradabile?" senza riapplicare la ricerca statistica.

**Soglie immutabili.** Le soglie degli eventi (es. `RSI < 30.5`) vengono fissate
dal Modulo 1 e propagate invariate attraverso tutta la pipeline. Né il Modulo 2
né il Modulo 3 possono modificarle. Una nuova soglia richiede una nuova sessione
di Event Discovery.

**Soglie distribuzionali.** Le soglie non sono valori fissi ma percentili della
distribuzione della serie trasformata sull'asset specifico. Lo stesso pattern
strutturale genera soglie diverse su asset diversi (es. RSI p10 = 30.5 su ADA,
27.8 su BTC) mantenendo la stessa semantica statistica.

**No look-ahead bias.** Il forward return non entra mai nel Modulo 1.
La classificazione scale-free è volutamente conservativa (il falso negativo
costa meno del falso positivo). Le soglie distribuzionali sono calcolate sulla
serie in-sample.
