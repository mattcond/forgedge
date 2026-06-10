# FORGE — Specifiche da Codebase (v0.1.0)

> **Scopo di questa directory**
> I file in `specs/` descrivono il comportamento implementato nella codebase,
> derivato direttamente dall'ispezione del codice sorgente.
> Servono da riferimento per verificare quanto l'analisi funzionale
> (`docs/README.md` e `docs/modules/`) corrisponda allo stato attuale del codice.

---

## Struttura

| File | Contenuto |
|---|---|
| `modulo_0_it.md` / `modulo_0_en.md` | Market Context — regime classifier (Modulo 0) |
| `modulo_1_it.md` / `modulo_1_en.md` | Event Discovery — pipeline 5 step (Modulo 1) |
| `modulo_2_it.md` / `modulo_2_en.md` | Alpha Discovery — misura predittività (Modulo 2) |

---

## Stato implementativo (v0.1.0)

| Modulo | Implementato | Note |
|---|---|---|
| Modulo 0 — Market Context | ✅ Completo | `EMAProxyClassifier`, auto-window Hurst/OU |
| Modulo 1 — Event Discovery | ✅ Completo | Step 0–5, walk-forward, `sql_expression` |
| Modulo 2 — Alpha Discovery | ✅ Completo | 8 step, FDR, scoring, pattern family |
| Modulo 3 — Rule Discovery | ❌ Non implementato | Solo documentato in analisi funzionale |
| Modulo 4 — Rule Registry | ❌ Non implementato | Solo documentato in analisi funzionale |

---

## Convenzioni di allineamento

In ogni specifica di modulo, le sezioni evidenziate indicano:

- ✅ **Allineato** — implementazione coincide con l'analisi funzionale
- ➕ **Aggiunto nel codice** — presente nel codice, non documentato nell'analisi funzionale
- ⚠️ **Divergenza** — comportamento diverso da quanto descritto nell'analisi funzionale
- ❌ **Mancante nel codice** — documentato nell'analisi funzionale, non ancora implementato

---

## Dipendenze runtime

Il pacchetto dipende unicamente da `numpy` e `pandas`.
Le primitive statistiche (Spearman, t-test, regressione OU, FDR Benjamini-Hochberg)
sono implementate in puro numpy — nessuna dipendenza da `scipy` o `statsmodels`.
