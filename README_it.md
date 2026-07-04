# FORGE — Feature-Oriented Rule Generation Engine

FORGE è un sistema di ricerca quantitativa per la **scoperta sistematica di regole di trading algoritmico** da dati storici di mercato. A partire da una KPI Table (OHLCV + indicatori tecnici), FORGE identifica eventi booleani con struttura temporale stabile, ne misura il potere predittivo rispetto a un target economico derivato dai dati, e produce contratti formali pronti per la validazione operativa.

[🇬🇧 English version](README.md)

---

## Perché FORGE

La ricerca sistematica di edge di trading soffre di tre problemi ricorrenti:

- **Look-ahead bias** — le soglie degli eventi calibrate osservando i rendimenti "conoscono" già il futuro prima della scoperta
- **Ottimizzazione in-sample** — soglie e orizzonti calibrati sulla stessa finestra usata per la valutazione producono backtest circolari
- **Mancanza di separazione operativa** — l'evidenza statistica del potere predittivo non equivale alla profittabilità sotto commissioni reali e meccaniche di esecuzione degli ordini

FORGE affronta tutti e tre con un'**architettura a pipeline rigidamente separata**: ogni modulo risponde a una sola domanda e passa al successivo solo un artefatto formale. Nessun modulo può accedere ai dati del modulo successivo; nessuna soglia può essere ricalibrata dopo la scoperta.

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
│  conferma sull'OOS tail. Prima esposizione al forward return.    │
│  Output: list[AlphaContract]                                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 3 — Rule Discovery                                       │
│  Backtest realistico con order mechanics (limit order, fee).     │
│  Output: EDGE / PARTIAL-EDGE / NON-EDGE + parametri operativi   │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Modulo 4 — Rule Registry                                        │
│  Deduplicazione, backtest cross-ticker, classificazione genericity│
│  Output: tabella piatta + report HTML autocontenuto              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Invarianti fondamentali

| Invariante | Cosa previene |
|---|---|
| Modulo 1 non vede mai il forward return | Look-ahead bias nella selezione degli eventi |
| Le soglie sono immutabili dopo la scoperta | Ottimizzazione delle soglie sul campione di valutazione |
| Orizzonte, direzione e take-profit sono derivati dai dati per ogni evento | Assunzioni economiche che pre-cuociono il risultato |
| Il verdetto operativo è vincolato al walk-forward OOS (Modulo 3); la conferma OOS di Alpha è registrata su ogni contratto | Conferma a posteriori di una conclusione già decisa |

---

## Installazione

FORGE dipende unicamente da `numpy` e `pandas`. Nessuna dipendenza da `scipy`, `statsmodels` o librerie ML: tutte le primitive statistiche (Spearman, t-test, regressione OU, FDR Benjamini-Hochberg, funzione beta incompleta) sono implementate in puro numpy.

```bash
pip install forgedge
```

---

## Quick start

```python
import pandas as pd
from forgedge import forge

# KPI Table con OHLCV + indicatori tecnici (colonna 'close' richiesta)
kpi = pd.read_parquet("kpi_table.parquet")

# Pipeline completa: da KPI Table a regole validate in una singola chiamata
result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.summary())                         # una riga per candidato + rule_verdict
for contract, response in result.edges():       # solo EDGE / PARTIAL-EDGE
    print(contract.alpha_id, response.verdict)
print(result.registry.summary())                # Modulo 4 — regole catalogate
```

Su un `timeframe` daily o più lento la griglia di holding period di default
viene sostituita automaticamente da una calibrata sul daily (la griglia
standard è calibrata su barre ~orarie); passando un proprio `AlphaConfig` si
mantiene il pieno controllo.  Per una configurazione per-modulo coerente in
frequenza usare `forgedge.presets.forge_preset("balanced", timeframe="1D",
asset=...)`.

Di default `forge()` esegue anche il **fast rotation null a livello di
ricerca** — la distribuzione nulla esatta del miglior eccesso standardizzato
su ogni offset circolare, calcolata via FFT in ~secondi — e un verdetto `EDGE`
pieno richiede in aggiunta di batterla (`rotation_p <= 0.05`): una regola che
ha solo vinto la lotteria del multiple testing della propria sessione di
discovery viene declassata a `PARTIAL-EDGE`.  La superficie di ipotesi della
sessione è registrata su `ForgeResult.ledger`; disattivabile con
`fast_null=False`.

Gli split temporali sono **purged**: le barre in-sample la cui finestra
forward attraversa il confine IS/OOS sono escluse da ogni misura di Alpha
Discovery, e le finestre train del walk-forward terminano un worst-case trade
span prima della finestra test (vedi `forgedge.timebudget.TimeBudget` —
passandone uno a `forge()` tutti i moduli condividono un unico asse, con
embargo opzionale).

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

---

## I tre concetti

FORGE struttura il processo di scoperta attorno a tre concetti formali che rispondono ciascuno a una domanda distinta e producono un artefatto distinto.

### Evento — osservare il mercato senza bias

Un **evento** è una condizione booleana sulle barre storiche, scoperta dalla struttura temporale degli indicatori senza mai calcolare un forward return. Le soglie sono distribuzionali (percentili specifici per asset) e immutabili una volta fissate.

```python
c = candidates[0]
print(c.expression)            # "rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118"
signal = c.apply(new_kpi)      # pd.Series bool — deterministico, senza look-ahead
```

Il ConsistencyGate filtra gli eventi con struttura temporale instabile (attivazioni insufficienti, clustering stagionale, bassa frequenza mensile) prima che venga calcolato qualsiasi rendimento.

### Alpha — misurare il potere predittivo

Un **alpha** è la risposta empirica alla domanda: *dato che l'evento si è attivato, cosa succede statisticamente nelle prossime h barre?* Orizzonte, direzione e take-profit (`sell_pct`) sono tutti **derivati dai dati** — mai assunti — scansionando `|mean_advantage|/√h` su una grid di orizzonti e prendendo il quantile MFE delle barre attive.

```python
c = promoted[0]
dt = c.derived_target
print(f"{dt.direction} a h={dt.holding_period_h}h  sell_pct={dt.sell_pct:.4f}")
print(f"Grade {c.alpha_score.grade}  |  OOS lift: {c.oos_validation.lift:.4f}")
```

L'unico gate di rigetto rigido è la direzione indeterminata (nessun vantaggio finito su nessun orizzonte). Tutte le altre metriche statistiche (IC, Cohen's d, lift, FDR) contribuiscono al grade A–D senza bloccare la promozione.

### Regola — tradare in modo realistico

Una **regola** è il verdetto operativo su un contratto alpha. Rule Discovery esegue un backtest realistico con ingresso a limit order, uscita a take-profit, stop a orizzonte e commissioni per lato — poi valida la migliore configurazione di parametri su un walk-forward OOS rolling.

```python
resp = RuleDiscovery(ed.df, contract, cand).run()
print(resp.verdict)                               # "EDGE", "PARTIAL-EDGE", "NON-EDGE"
if resp.is_edge:
    p = resp.validated_rule.params
    print(f"Entry: limit -{p.buy_drop_pct:.2%}  TP: +{p.sell_pct:.2%}  h={p.target_h}")
    print(f"IS PF: {resp.in_sample_summary.profit_factor:.2f}"
          f"  WF consistency: {resp.walk_forward.consistency:.0%}")
```

---

## Panoramica dei moduli

| Modulo | Domanda cui risponde | Output principale |
|---|---|---|
| 0 — Market Context | In che regime si trova questa barra? | Colonna `regime` (5 livelli), `regime_stable` |
| 1 — Event Discovery | Questa configurazione dell'indicatore è stabile e ripetibile? | `EventCandidate` — soglie immutabili, `apply()` |
| 2 — Alpha Discovery | L'evento predice un rendimento orientato? | `AlphaContract` — target derivato, grade A–D |
| 3 — Rule Discovery | Questo alpha è profittevole con order mechanics reali? | `RuleDiscoveryResponse` — verdetto EDGE, `ValidatedRule` |
| 4 — Rule Registry | Questa regola si generalizza su altri ticker? | Tabella piatta, report HTML — GENERIC / PARTIAL / SPECIFIC |

---

## Stato implementazione

| Modulo | Stato |
|---|---|
| 0 — Market Context | ✅ Implementato |
| 1 — Event Discovery | ✅ Implementato |
| 2 — Alpha Discovery | ✅ Implementato |
| 3 — Rule Discovery | ✅ Implementato |
| 4 — Rule Registry | 🚧 WIP |

---

## Documentazione

| File | Contenuto |
|---|---|
| [`concepts_it.md`](src/forgedge/docs/specs/concepts_it.md) | Guida concettuale: evento, alpha e regola — dal mercato al segnale |
| [`how_to_use_it.md`](src/forgedge/docs/specs/how_to_use_it.md) | Guida pratica alla pipeline end-to-end per produzione |
| [`modulo_0_it.md`](src/forgedge/docs/specs/modulo_0_it.md) | Market Context: classificazione regime, EMAProxy, configurazione |
| [`modulo_1_it.md`](src/forgedge/docs/specs/modulo_1_it.md) | Event Discovery: pipeline 5-step, ConsistencyGate, EventCandidate |
| [`modulo_2_it.md`](src/forgedge/docs/specs/modulo_2_it.md) | Alpha Discovery: target derivato, IC, OOS, AlphaContract |
| [`modulo_3_it.md`](src/forgedge/docs/specs/modulo_3_it.md) | Rule Discovery: backtest, verdetto EDGE, walk-forward, report |
| [`modulo_4_it.md`](src/forgedge/docs/specs/modulo_4_it.md) | Rule Registry: deduplicazione, cross-ticker, genericity, export |

---

## Licenza

MIT
