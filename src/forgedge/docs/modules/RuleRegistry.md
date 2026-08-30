# FORGE — Rule Registry Module
> Quarto e ultimo modulo della pipeline FORGE.
> Riceve le regole validate da Rule Discovery, le raccoglie in un
> registro in-memory per la sessione corrente, calcola le matrici
> di correlazione, identifica i duplicati, esegue il backtest
> cross-ticker e produce l'output finale — una tabella piatta
> e un report HTML autocontenuto.

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Input Multi-Ticker](#2-input-multi-ticker)
3. [Natura Stateless](#3-natura-stateless)
4. [Il Registro in Memoria](#4-il-registro-in-memoria)
5. [Step 1 — Ingestion](#5-step-1--ingestion)
6. [Step 2 — Matrici di Correlazione](#6-step-2--matrici-di-correlazione)
7. [Step 3 — Deduplicazione](#7-step-3--deduplicazione)
8. [Step 4 — Cross-Ticker Backtest](#8-step-4--cross-ticker-backtest)
9. [Step 5 — Export](#9-step-5--export)
10. [Formato della Tabella Piatta](#10-formato-della-tabella-piatta)
11. [Formato del Report HTML](#11-formato-del-report-html)
12. [Esempio end-to-end](#12-esempio-end-to-end)
13. [Parametri configurabili](#13-parametri-configurabili)

---

## 1. Posizionamento e Responsabilità

```
CSV multi-ticker
      │
      ├── TICKER_A → Event Discovery → Alpha Discovery → Rule Discovery
      ├── TICKER_B → Event Discovery → Alpha Discovery → Rule Discovery
      └── TICKER_C → Event Discovery → Alpha Discovery → Rule Discovery
                                │
                         Pool regole validate
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  RULE REGISTRY                                                   │
│                                                                  │
│  Step 1 — Ingestion                                             │
│           Aggiunge ogni regola al registro in-memory            │
│                                                                  │
│  Step 2 — Correlation Matrices                                  │
│           Jaccard sulle attivazioni                             │
│           Spearman sui gain                                     │
│                                                                  │
│  Step 3 — Deduplicazione                                        │
│           Identifica e marca i duplicati (per-ticker)           │
│                                                                  │
│  Step 4 — Cross-Ticker Backtest                                 │
│           Ogni regola viene testata su tutti i ticker           │
│           con soglie ricalcolate sulla distribuzione locale     │
│                                                                  │
│  Step 5 — Export                                                │
│           Tabella piatta (CSV / Excel)                          │
│           Report HTML autocontenuto                             │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
  flat_table.csv  +  report.html
```

**Domanda del modulo:** *"tra tutte le regole validate in questa sessione,
quali sono distinte, complementari e generiche? Come le presento
all'utente?"*

**Input:** regole validate da Rule Discovery — una per ogni ticker
presente nel CSV di input.

**Output:** due file per sessione — tabella piatta e report HTML.

**Vincolo critico:** Rule Registry non valuta la qualità delle singole
regole sul ticker di origine — lo ha già fatto Rule Discovery. Valuta
le relazioni tra regole e la loro generalizzabilità su altri ticker.

---

## 2. Input Multi-Ticker

Quando il CSV di input contiene più ticker accodati, FORGE esegue
la pipeline completa separatamente per ogni ticker. Rule Registry
riceve il pool aggregato di tutte le regole validate.

### Separazione per ticker

```python
# FORGE separa il CSV per ticker prima di avviare la pipeline
tickers = df['ticker'].unique()   # es. ['ADAUSDC', 'SOLUSDC', 'BTCUSDC']

forge_results = {}
for ticker in tickers:
    ticker_df = df[df['ticker'] == ticker].copy()
    forge_results[ticker] = run_forge_pipeline(ticker_df, config)  # ForgeResult

# Entry point consigliato: costruisce submissions + frames dai ForgeResult
rule_registry = RuleRegistry.from_forge_results(forge_results).run()
```

`RuleRegistry` non accetta una lista piatta di regole: il costruttore
diretto richiede `RuleRegistry(submissions, frames, config=None)`, dove
`submissions` è una lista di `RuleSubmission` (ticker + `RuleDiscoveryResponse`
+ `EventCandidate`) e `frames` è un dizionario `{ticker: kpi_table}` — serve
per ricalcolare le soglie nel backtest cross-ticker (Step 4). Quando si parte
già da un `ForgeResult` per ticker, `RuleRegistry.from_forge_results(...)`
costruisce entrambi automaticamente.

### Perché per ticker e non tutto insieme

Mescolare ticker diversi nella stessa pipeline senza stratificazione
produce artefatti statistici. Ogni ticker ha distribuzioni di prezzo,
volatilità e oscillatori diverse — il Feature Generator costruirebbe
ratio e spread su serie non comparabili. Le soglie distribuzionali
del Threshold Catalog devono essere calcolate sulla distribuzione
del singolo ticker, non su quella aggregata.

La generalizzabilità cross-ticker è una proprietà **misurata
a posteriori** nel Rule Registry, non costruita a priori mescolando
i dati.

---

## 3. Natura Stateless

Il registro viene costruito da zero a ogni sessione FORGE.
Non esiste un catalogo persistente che cresce nel tempo.

```
Sessione A (gennaio):   Produce RULE_01, RULE_02, RULE_03
Sessione B (marzo):     Produce RULE_04, RULE_05 — non sa nulla di A

Conseguenza:
  - Le matrici di correlazione sono calcolate solo sulle regole
    della sessione corrente
  - Il cross-ticker backtest usa solo i ticker presenti nel CSV
    della sessione corrente
  - Il confronto tra sessioni diverse è responsabilità dell'utente
    (può farlo sulla tabella piatta esportata)
```

**Razionale:** mantiene FORGE semplice, riproducibile e privo di
dipendenze da storage persistente. La tabella piatta esportata è
l'unico artefatto di persistenza — l'utente la gestisce come preferisce.

> **Nota per versioni future:** un catalogo persistente permetterebbe
> confronti cross-sessione. Non è in scope per v1.0.

---

## 4. Il Registro in Memoria

Il registro è una lista di documenti. Ogni documento corrisponde
a una regola validata su un ticker specifico.

### Schema del documento

```python
RuleDocument = {
    # Identificazione
    "rule_id":          str,    # es. "RULE_ADA_01"
    "expression":       str,    # es. "close_rsi_25 < 30.5 AND pr_96 < 0.10"
    "source_ticker":    str,    # ticker su cui la regola è stata estratta
    "source_alpha_id":  str,    # tracciabilità verso Alpha Discovery
    "verdict":          str,    # "EDGE" | "PARTIAL-EDGE" (verdetto di Rule Discovery)

    # Array paralleli indicizzati sulla KPI Table del ticker sorgente
    # Lunghezza = n. di trade effettivi (barre con fill confermato)
    "activation_idx":   list[int],    # indici di riga nella KPI Table
    "activation_dates": list[str],    # date corrispondenti (ISO 8601)
    "gains":            list[float],  # net_pct_gain per ogni trade

    # Parametri operativi
    "params": {
        "direction":      str,    # "long" | "short"
        "entry_mode":     str,    # "limit" | "market"
        "buy_drop_pct":   float,
        "buy_delay_bar":  int,
        "sell_pct":       float,
        "target_h":       int,
        "fee":            float,
    },

    # Statistiche aggregate sul ticker sorgente (da Rule Discovery)
    "stats": {
        "pf":            float,
        "win_rate":      float,
        "total_trades":  int,
        "expectancy":    float,
        "zero_months":   int,
        "dsr":           float,
        "grade":         str,       # "A" | "B+" | "B" | "C"
        "monthly_gains": list[float],
    },

    # Regime
    "regime": {
        "type":     str,         # "agnostic" | "conditional" | "specific"
        "avoid_in": list[str],
    },

    # Campi popolati da Rule Registry — Step 2 e 3 (inizialmente None)
    "overlap_max":      float | None,   # Jaccard max con altre regole
    "gain_corr_max":    float | None,   # Spearman max con altre regole
    "is_duplicate":     bool  | None,
    "duplicate_of":     str   | None,   # rule_id della regola dominante

    # Campi popolati da Rule Registry — Step 4 (inizialmente vuoto)
    "cross_ticker": {
        # Un sotto-documento (CrossTickerResult) per ogni ticker diverso
        # da source_ticker. Campi: ticker, expression_adapted, pf, win_rate,
        # total_trades, zero_months, verdict, bar.
        # "bar" è la soglia PF che QUESTA regola doveva superare su QUESTO
        # ticker (dipende dal pf_home della regola — vedi Step 4) — registrata
        # per rendere leggibile perché un ticker è FAIL: soglia non raggiunta
        # in assoluto, o edge non trasferito a sufficienza.
        # es. "SOLUSDC": { "pf": 2.41, "wr": 0.79, "bar": 1.5, "verdict": "PASS" }
    },
    "cross_ticker_score":  int   | None,   # n. ticker su cui PF >= soglia
    "cross_ticker_total":  int   | None,   # n. ticker testati
    "is_generic":          bool  | None,   # True se cross_score / total >= threshold
    "classification":      str   | None,   # "GENERIC" | "PARTIAL" | "SPECIFIC" | "ISOLATED"
}
```

**Gli array `activation_idx` e `gains` sono la struttura dati chiave**
per le matrici di correlazione. I sotto-documenti `cross_ticker`
contengono i risultati del backtest su ogni ticker alternativo.

---

## 5. Step 1 — Ingestion

Per ogni regola emessa da Rule Discovery con verdetto `EDGE`
o `PARTIAL-EDGE`, il Registry crea un documento e lo aggiunge
alla lista in memoria.

```
FUNCTION ingest(validated_rule):
    doc = build_document(validated_rule)
    registry.append(doc)
    RETURN doc.rule_id
```

Le regole con verdetto `NON-EDGE` non entrano nel registro.

---

## 6. Step 2 — Matrici di Correlazione

Le matrici vengono calcolate al termine dell'ingestion.
Operano sull'intero pool di regole indipendentemente dal ticker
sorgente — due regole estratte su ticker diversi possono essere
correlate se catturano lo stesso pattern di mercato.

### Matrice A — Jaccard sulle attivazioni

Misura la **sovrapposizione temporale**: quante barre (per data)
sono coperte da entrambe le regole?

Per regole estratte su ticker diversi, il confronto avviene
per data — non per indice di riga, che cambia tra KPI Table diverse.

```
J(R_i, R_j) = |date_i ∩ date_j|
              ──────────────────
              |date_i ∪ date_j|
```

```python
def jaccard_by_date(dates_a, dates_b):
    set_a = set(dates_a)
    set_b = set(dates_b)
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
```

### Matrice B — Spearman sui gain

Misura la **correlazione dei rendimenti**: le due regole guadagnano
e perdono nelle stesse condizioni di mercato?

Allineamento per data — le barre senza trade ricevono valore `0`.

```python
def gain_correlation_by_date(doc_a, doc_b, all_dates):
    series_a = pd.Series(0.0, index=all_dates)
    series_b = pd.Series(0.0, index=all_dates)

    for date, gain in zip(doc_a['activation_dates'], doc_a['gains']):
        if date in series_a.index:
            series_a[date] = gain
    for date, gain in zip(doc_b['activation_dates'], doc_b['gains']):
        if date in series_b.index:
            series_b[date] = gain

    active = (series_a != 0) | (series_b != 0)
    if active.sum() < 10:
        return 0.0

    corr, _ = spearmanr(series_a[active], series_b[active])
    return corr
```

### Lettura combinata

| Jaccard | Spearman | Interpretazione | Azione |
|---|---|---|---|
| ✅ > soglia | ✅ > 0.70 | Duplicate — stessa logica, stesse barre | Scarta la più debole |
| ❌ < 0.30 | ✅ > 0.70 | Stessa esposizione al regime, barre diverse | Segnala — decide l'utente |
| ❌ < 0.30 | ❌ < 0.30 | Complementari | Tieni entrambe |

---

## 7. Step 3 — Deduplicazione

Dopo aver calcolato le matrici, il Registry marca i duplicati.

### Logica

```
PER OGNI coppia (R_i, R_j) con i < j:
    SE jaccard(R_i, R_j) > OVERLAP_THRESHOLD:
        → marca come duplicato quella con PF minore
        → popola duplicate_of con il rule_id della dominante

Nessuna regola viene eliminata — viene solo marcata is_duplicate = True.
```

### Comportamento in caso di catena

```
R01 (PF 3.17) ← R02 (PF 2.90) ← R03 (PF 2.45)
     dominante     dup di R01       dup di R02
```

Il registro marca `R02.duplicate_of = "R01"` e `R03.duplicate_of = "R02"`.
L'utente vede la catena e decide autonomamente.

> **Nota implementativa:** `OVERLAP_THRESHOLD` è un parametro
> configurabile. Il valore di default verrà calibrato in fase
> di implementazione.

---

## 8. Step 4 — Cross-Ticker Backtest

Per ogni regola nel registro, Rule Registry esegue un backtest
su tutti i ticker diversi da quello di origine. La struttura
della regola rimane invariata — cambiano le soglie assolute,
ricalcolate sulla distribuzione locale del ticker target.

### Principio del ricalcolo delle soglie

Le soglie dell'espressione sono state generate da Event Discovery
come percentili della distribuzione del ticker sorgente.
Per testare la stessa regola su un ticker diverso, le soglie
assolute vengono ricalcolate sui percentili corrispondenti
della distribuzione del ticker target.

```
Regola estratta su ADA:
  expression: "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
  soglia_abs: 30.5  = quantile(rsi25_ADA, 0.10)
  soglia_rel: 0.10  = percentile fisso (invariante)

Test su SOL:
  soglia_abs: quantile(rsi25_SOL, 0.10) = 27.8
  expression adattata: "close_rsi_25 < 27.8 AND pr_close_rsi_25_96 < 0.10"

Test su BTC:
  soglia_abs: quantile(rsi25_BTC, 0.10) = 24.1
  expression adattata: "close_rsi_25 < 24.1 AND pr_close_rsi_25_96 < 0.10"
```

La struttura logica della regola è identica su tutti i ticker.
Solo i valori assoluti cambiano — e cambiano in modo sistematico,
non arbitrario, perché derivano dallo stesso metodo percentile
che ha generato la soglia originale.

### Pseudocodice

```
PER OGNI regola R nel registro:
    PER OGNI ticker T diverso da R.source_ticker:

        kpi_table_T = get_kpi_table(ticker=T)

        // Ricalcola le soglie assolute sulla distribuzione di T
        expression_T = recalculate_absolute_thresholds(
            expression = R.expression,
            threshold_catalog = R.threshold_catalog,
            kpi_table = kpi_table_T
        )

        // Esegui il backtest con i parametri operativi invariati
        result = run_backtest(
            kpi_table  = kpi_table_T,
            rule       = expression_T,
            params     = R.params       // buy_drop, sell_pct, target_h, fee
        )

        // Calcola la soglia (bar) che QUESTA regola deve superare su T:
        // il maggiore tra il floor assoluto e la frazione di ritenzione
        // del PF sul ticker sorgente (pf_home) — vedi nota sotto.
        pf_home = R.stats.pf
        bar = max(CROSS_PF_THRESHOLD, MIN_CROSS_PF_RETENTION * pf_home)

        // Salva il risultato nel documento
        R.cross_ticker[T] = {
            "expression_adapted": expression_T,
            "pf":           result.profit_factor,
            "win_rate":     result.win_rate,
            "total_trades": result.total_trades,
            "zero_months":  result.zero_months,
            "bar":          bar,
            "verdict":      "PASS" if result.pf >= bar else "FAIL"
        }

    // Calcola lo score aggregato
    R.cross_ticker_score  = count(T where verdict == "PASS")
    R.cross_ticker_total  = len(altri ticker)
    R.is_generic          = (R.cross_ticker_score / R.cross_ticker_total
                             >= GENERIC_RATIO_THRESHOLD)
```

### Classificazione della genericità

| cross_score / total | Classificazione | Significato |
|---|---|---|
| 3/3 (100%) | **GENERIC** | La regola funziona su tutti i ticker testati |
| 2/3 (67%) | **PARTIAL** | La regola funziona sulla maggioranza |
| 1/3 (33%) | **SPECIFIC** | La regola funziona solo sul ticker sorgente |
| 0/3 (0%) | **ISOLATED** | La regola non si generalizza |

### Criterio di PASS — due condizioni, non una

Il verdetto `PASS` non dipende da una singola soglia assoluta: dipende
da **due** condizioni, entrambe necessarie:

```
PASS ⟺ pf_other >= CROSS_PF_THRESHOLD              (è negoziabile sul ticker target)
        AND pf_other >= MIN_CROSS_PF_RETENTION * pf_home   (l'edge si trasferisce davvero)
```

Una sola soglia assoluta confonderebbe due domande diverse — "la regola
è buona anche altrove?" e "la regola *trasferisce* il suo edge?" — che
possono divergere in entrambe le direzioni: una regola che trasferisce
perfettamente un PF modesto (1.6 → 1.6) fallirebbe un bar assoluto di 2.0,
mentre una regola che perde un terzo del suo edge (3.0 → 2.05) lo
supererebbe. Il floor assoluto da solo darebbe alle regole più deboli
il test di genericità più facile; il floor relativo da solo
promuoverebbe a "generica" una regola che ha perso un terzo del suo
vantaggio. Le due condizioni insieme misurano il trasferimento — non
la qualità, che resta compito del verdetto di Rule Discovery e del grade.

`CROSS_PF_THRESHOLD` e `MIN_CROSS_PF_RETENTION` sono parametri
configurabili con default risolti e già in uso (non più "TBD"):

- `cross_pf_threshold`: risolto a sessione dal `SelectionCriteria.partial_min_profit_factor`
  — la stessa soglia che ha ammesso la regola in patria è quella che deve
  superare altrove. Fallback `1.5` quando non risolvibile.
- `min_cross_pf_retention`: `0.8` — la regola deve conservare almeno l'80%
  del proprio PF sul ticker sorgente.
- `generic_ratio_threshold`: `2/3` esatto (non `0.67` arrotondato), così
  che una regola che passa su 2 ticker su 3 legga correttamente `PARTIAL`.

Il valore restituito da questo calcolo (`bar`) viene salvato in ogni
sotto-documento `cross_ticker[T]` — vedi Sezione 4 — proprio perché varia
per regola (dipende dal `pf_home` di quella specifica regola).

---

## 9. Step 5 — Export

L'export produce due file al termine di ogni sessione FORGE.

### Decisione di export

```
DEFAULT:  esporta tutte le regole
          (duplicate e non-generiche incluse, con flag visibili)
OPZIONE:  esporta solo le regole non-duplicate
OPZIONE:  esporta solo le regole generiche (cross_score >= soglia)
```

La filosofia di FORGE è **non nascondere la complessità**:
mostrare anche i duplicati e le regole non-generiche con flag
espliciti permette all'utente di capire perché certe regole
non sono state promosse.

---

## 10. Formato della Tabella Piatta

Una riga per regola, tutti i parametri appiattiti.
Formato: CSV o Excel (configurabile).

### Schema colonne

```
IDENTIFICAZIONE
  rule_id               es. "RULE_ADA_01"
  expression            es. "close_rsi_25 < 30.5 AND pr_96 < 0.10"
  source_ticker         es. "ADAUSDC"
  grade                 es. "B+"
  source_alpha_id       tracciabilità
  verdict               "EDGE" | "PARTIAL-EDGE" (verdetto di Rule Discovery)

STATISTICHE SUL TICKER SORGENTE
  pf                    Profit Factor
  win_rate              Win Rate (0–1)
  total_trades          Numero di trade
  expectancy            Gain netto medio per trade
  zero_months           Mesi senza trade
  dsr                   Deflated Sharpe Ratio

PARAMETRI OPERATIVI
  entry_mode            "limit" | "market"
  direction             "long" | "short"
  buy_drop_pct          es. 0.010
  sell_pct              es. 0.040
  target_h              es. 24
  fee                   es. 0.002

REGIME
  regime_type           "agnostic" | "conditional" | "specific"
  regime_avoid          Regimi da evitare (stringa separata da virgole)

CORRELAZIONE (Step 2 e 3)
  overlap_max           Jaccard massimo con qualsiasi altra regola
  gain_corr_max         Spearman massimo con qualsiasi altra regola
  is_duplicate          True | False
  duplicate_of          rule_id della regola dominante (se duplicato)

CROSS-TICKER (Step 4 — una colonna per ticker per ogni metrica)
  pf_{TICKER}           es. pf_SOLUSDC, pf_BTCUSDC
  wr_{TICKER}           es. wr_SOLUSDC, wr_BTCUSDC
  trades_{TICKER}       es. trades_SOLUSDC, trades_BTCUSDC
  verdict_{TICKER}      "PASS" | "FAIL" per ogni ticker
  cross_pf_bar          soglia PF richiesta per PASS (una colonna sola —
                        dipende dal pf_home della regola, non dal ticker)
  cross_ticker_score    n. ticker con verdetto PASS
  cross_ticker_total    n. ticker testati
  is_generic            True | False
```

### Esempio di riga (3 ticker: ADA, SOL, BTC)

```
rule_id        | RULE_ADA_01
expression     | close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10
source_ticker  | ADAUSDC
grade          | B+
verdict        | EDGE
pf             | 3.17
win_rate       | 0.814
total_trades   | 102
zero_months    | 0
dsr            | 1.31
entry_mode     | limit
direction      | long
buy_drop_pct   | 0.010
sell_pct       | 0.040
target_h       | 24
regime_type    | conditional
avoid_in       | uptrend_continuous
overlap_max    | 0.12
gain_corr_max  | 0.34
is_duplicate   | False
duplicate_of   |
pf_SOLUSDC     | 2.41
wr_SOLUSDC     | 0.790
trades_SOLUSDC | 87
verdict_SOLUSDC| PASS
pf_BTCUSDC     | 1.89
wr_BTCUSDC     | 0.714
trades_BTCUSDC | 61
verdict_BTCUSDC| FAIL
cross_pf_bar   | 2.536
cross_score    | 1
cross_total    | 2
is_generic     | False
```

---

## 11. Formato del Report HTML

Un singolo file HTML autocontenuto. Nessuna dipendenza esterna —
grafici inline come SVG, nessuna CDN, apre in qualsiasi browser.

### Struttura del documento

```
┌───────────────────────────────────────────────────────────────┐
│  FORGE Report                                                 │
│  Ticker: ADAUSDC, SOLUSDC, BTCUSDC · Timeframe: 1H           │
│  Periodo: 2025-01-01 → 2026-01-01                            │
│  Sessione: 2026-05-23 · Regole: 18 validate → 6 nel registry │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  SEZIONE 1 — OVERVIEW                                         │
│    Funnel per ticker:                                         │
│      ADA: 847 candidati → 18 → 6 → 4 validate               │
│      SOL: 791 candidati → 15 → 5 → 3 validate               │
│      BTC: 702 candidati → 12 → 4 → 2 validate               │
│    Tabella riassuntiva tutte le regole con cross_score        │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  SEZIONE 2 — HEATMAP CORRELAZIONI                             │
│    Matrice Jaccard (attivazioni per data)                     │
│    Matrice Spearman (gain)                                    │
│    Note sui duplicati identificati                            │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  SEZIONE 3 — CROSS-TICKER SUMMARY                             │
│    Tabella: regola × ticker → PF, WR, PASS/FAIL              │
│    Badge [GENERIC] [PARTIAL] [SPECIFIC] [ISOLATED]           │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  SEZIONE 4 — DETTAGLIO PER REGOLA                             │
│  (una scheda per ogni regola nel registro)                    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  RULE_ADA_01  [B+]  [EDGE]  [PARTIAL]                │   │
│  │  close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10   │   │
│  │  Estratta su: ADAUSDC                                 │   │
│  │                                                       │   │
│  │  PF 3.17 · WR 81.4% · T 102 · DSR 1.31              │   │
│  │                                                       │   │
│  │  [Equity curve mensile — SVG]                        │   │
│  │  [Monthly breakdown table]                           │   │
│  │  [Win/Lose bar chart — SVG]                          │   │
│  │                                                       │   │
│  │  Cross-ticker:                                        │   │
│  │    SOLUSDC  → PF 2.41  WR 79.0%  [PASS]             │   │
│  │    BTCUSDC  → PF 1.89  WR 71.4%  [FAIL]             │   │
│  │                                                       │   │
│  │  Regime: conditional · Evitare: uptrend_continuous   │   │
│  │  Overlap max: 0.12 (RULE_ADA_03)                     │   │
│  │  Tracciabilità: ALPHA-001 → EVT-close_rsi_25-...     │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  SEZIONE 5 — TRADE LOG                                        │
│    Barre con almeno un trade attivo (tutti i ticker)          │
│    Colonne: data, ticker, rule_id, entry_price, exit_price,  │
│             result_pct, won, regime                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Principi del report HTML

**Autocontenuto:** tutti i grafici sono SVG inline. Nessuna CDN,
nessun JavaScript esterno. Apre offline in qualsiasi browser.

**Onestà visiva:** i duplicati compaiono con banner `[DUPLICATO DI RULE_XX]`.
Le regole non-generiche compaiono con badge `[SPECIFIC]` o `[ISOLATED]`.
Nulla viene nascosto — l'utente vede tutto e decide.

**Tracciabilità completa:** ogni regola mostra la catena
`rule_id → alpha_id → event_candidate_id` risalendo fino
all'evento originale di Event Discovery.

---

## 12. Esempio end-to-end

Sessione FORGE con 3 ticker (ADA, SOL, BTC) e 6 regole validate
totali nel pool (4 da ADA, 1 da SOL, 1 da BTC).

### Input ricevuto

```
RULE_ADA_01  expr_A  PF=3.17  T=102  source=ADA
RULE_ADA_02  expr_B  PF=2.74  T=148  source=ADA
RULE_ADA_03  expr_C  PF=4.42  T=99   source=ADA
RULE_ADA_04  expr_D  PF=5.16  T=90   source=ADA
RULE_SOL_01  expr_E  PF=3.81  T=94   source=SOL
RULE_BTC_01  expr_F  PF=2.95  T=73   source=BTC
```

### Step 1 — Ingestion

6 documenti aggiunti al registro in-memory.

### Step 2 — Matrici

```
Matrice Jaccard (per data):

            ADA_01 ADA_02 ADA_03 ADA_04 SOL_01 BTC_01
  ADA_01   1.00   0.08   0.61   0.72   0.18   0.09
  ADA_02   0.08   1.00   0.09   0.07   0.11   0.06
  ADA_03   0.61   0.09   1.00   0.68   0.21   0.12
  ADA_04   0.72   0.07   0.68   1.00   0.19   0.10
  SOL_01   0.18   0.11   0.21   0.19   1.00   0.14
  BTC_01   0.09   0.06   0.12   0.10   0.14   1.00
```

### Step 3 — Deduplicazione (soglia = 0.70)

```
ADA_01 – ADA_04: Jaccard=0.72 → ADA_01 marcata duplicato di ADA_04
Nessun'altra coppia supera la soglia.
```

### Step 4 — Cross-Ticker Backtest

```
RULE_ADA_01 testata su SOL e BTC:
  SOL: soglie ricalcolate → PF=2.41 PASS
  BTC: soglie ricalcolate → PF=1.89 FAIL
  cross_score=1/2 → is_generic=False → [SPECIFIC]

RULE_ADA_03 testata su SOL e BTC:
  SOL: PF=3.12 PASS
  BTC: PF=2.78 PASS
  cross_score=2/2 → is_generic=True → [GENERIC]

RULE_ADA_04 testata su SOL e BTC:
  SOL: PF=3.55 PASS
  BTC: PF=2.91 PASS
  cross_score=2/2 → is_generic=True → [GENERIC]

RULE_SOL_01 testata su ADA e BTC:
  ADA: PF=2.87 PASS
  BTC: PF=1.71 FAIL
  cross_score=1/2 → [SPECIFIC]

RULE_BTC_01 testata su ADA e SOL:
  ADA: PF=2.54 PASS
  SOL: PF=2.33 PASS
  cross_score=2/2 → [GENERIC]
```

### Step 5 — Export

```
flat_table.csv:
  6 righe · ADA_01 con is_duplicate=True
  Colonne cross-ticker per ADA, SOL, BTC

report.html:
  Overview: 3 ticker, 6 regole (1 duplicato, 3 generiche)
  Heatmap correlazioni
  Cross-ticker summary table
  5 schede dettaglio (ADA_02, ADA_03, ADA_04, SOL_01, BTC_01)
  1 scheda con banner DUPLICATO (ADA_01 → di ADA_04)
  Trade log: tutte le barre con trade, tutti i ticker
```

---

## 13. Parametri configurabili

| Parametro | Default | Descrizione | Nota implementativa |
|---|---|---|---|
| `OVERLAP_THRESHOLD` | TBD | Soglia Jaccard per deduplicazione | Calibrare su dati reali |
| `cross_pf_threshold` | `1.5` (fallback) | PF minimo assoluto per PASS nel cross-ticker — prima metà del criterio | Risolto a sessione da `SelectionCriteria.partial_min_profit_factor`; non più TBD |
| `min_cross_pf_retention` | `0.8` | Frazione minima del PF sorgente da conservare sul ticker target — seconda metà del criterio | Non più TBD — vedi Sezione 8 |
| `generic_ratio_threshold` | `2/3` (≈0.667) | Frazione minima PASS per badge GENERIC/PARTIAL | Esatto, non `0.67` arrotondato — non più TBD |
| `export_format` | `"excel"` | Formato tabella piatta: `"csv"` / `"excel"` | |
| `export_duplicates` | `True` | Se includere i duplicati nell'export | |
| `export_non_generic` | `True` | Se includere le regole non-generiche | |
| `html_include_tradelog` | `True` | Se includere il Trade Log nel report | |
| `html_charts` | `True` | Se includere i grafici SVG inline | |

---

*Rule Registry Module — FORGE (Feature-Oriented Rule Generation Engine)*
*Versione 1.1 · Maggio 2026 · Parte di FORGE v1.0*
*Status: Implementato — `forgedge.rule_registry`*
