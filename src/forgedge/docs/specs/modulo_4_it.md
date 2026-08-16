# Modulo 4 — Rule Registry

Rule Registry è il quinto e ultimo modulo della pipeline FORGE (quarto per
ordine di elaborazione dopo i moduli 0–3). Riceve le regole validate da Rule
Discovery — un pool per ogni ticker presente nella sessione — e le raccoglie
in un registro in-memory per produrre gli artefatti finali: una tabella piatta
e un report HTML autocontenuto.

Il suo compito è duplice: valutare le **relazioni** tra le regole (sovrapposizione
temporale, correlazione dei guadagni, deduplicazione) e la loro
**generalizzabilità** — ogni regola viene rieseguita su tutti gli altri ticker
della sessione, con le soglie ricalibrate sulla distribuzione locale.

**Registro stateless.** Il registro viene costruito da zero a ogni sessione FORGE.
Non esiste un catalogo persistente: l'unico artefatto di persistenza è la tabella
piatta esportata. La gestione di quel file (archiviazione, versioning, confronto
tra sessioni) è di competenza dell'utente.

**Principio di promozione.** Solo le submission con verdetto `EDGE` o
`PARTIAL-EDGE` entrano nel registro. Le submission `NON-EDGE` sono ignorate
silenziosamente da `ingest()` — non viene lanciata nessuna eccezione.

---

## Utilizzo di base

### Percorso 1 — costruzione manuale

```python
from forgedge import RuleRegistry, RuleSubmission, RegistryConfig

# submissions: lista di regole validate da Rule Discovery
submissions = [
    RuleSubmission(ticker="ADAUSDC", response=ada_resp, candidate=ada_cand),
    RuleSubmission(ticker="SOLUSDC", response=sol_resp, candidate=sol_cand),
    RuleSubmission(ticker="BTCUSDC", response=btc_resp, candidate=btc_cand),
]

# frames: dizionario ticker → DataFrame post-pipeline di Event Discovery
frames = {
    "ADAUSDC": ada_kpi_df,
    "SOLUSDC": sol_kpi_df,
    "BTCUSDC": btc_kpi_df,
}

registry = RuleRegistry(submissions, frames).run()

df   = registry.flat_table()
html = registry.html_report()
```

### Percorso 2 — da ForgeResult per ticker

```python
from forgedge import forge, RuleRegistry, RegistryConfig
from forgedge.event_discovery import DiscoveryConfig

ed_cfg = DiscoveryConfig(timestamp_col="open_dt")
results = {}
for ticker in ["ADAUSDC", "SOLUSDC", "BTCUSDC"]:
    df = raw[raw["symbol"] == ticker].copy().sort_values("open_dt")
    results[ticker] = forge(df, asset=ticker, timeframe="1H",
                            event_discovery_config=ed_cfg)

registry = RuleRegistry.from_forge_results(
    results,
    RegistryConfig(
        overlap_threshold=0.70,
        cross_pf_threshold=1.5,          # floor assoluto (derivato da M3)
        min_cross_pf_retention=0.8,     # e la frazione di PF di casa da mantenere
        generic_ratio_threshold=2 / 3,
        export_format="excel",
    ),
).run()

print(registry.summary().to_string(index=False))

flat_path = registry.export("forge_flat_table.xlsx")
with open("forge_report.html", "w", encoding="utf-8") as fh:
    fh.write(registry.html_report(timeframe="1H"))
```

`from_forge_results` estrae automaticamente le submission EDGE/PARTIAL-EDGE,
gli `EventCandidate` corrispondenti, il grade dall'`AlphaContract` e i frame
arricchiti dall'`EventDiscovery` — nessuna manipolazione manuale richiesta.

---

## Posizione nella pipeline

```
Modulo 0  MarketContext
      │
      ▼
Modulo 1  EventDiscovery
      │
      ▼
Modulo 2  AlphaDiscovery
      │
      ▼
Modulo 3  RuleDiscovery  [per ogni contratto promosso]
      │   ─── produce RuleDiscoveryResponse per ticker ───
      ▼
Modulo 4  RuleRegistry   ◄── input: list[RuleSubmission]
      │                             dict[ticker, kpi_df]
      ├── Tabella piatta (CSV / Excel)
      └── Report HTML autocontenuto
```

Rule Registry è l'**unico** modulo che opera su più ticker
simultaneamente. Tutti i moduli precedenti girano in isolamento su un singolo
ticker; la generalizzabilità cross-ticker è una proprietà *misurata a
posteriori* da Rule Registry, non costruita a priori mescolando i dati.

---

## Pipeline a 5 step

### Step 1 — Ingestion

Per ogni submission con verdetto `EDGE` o `PARTIAL-EDGE`, `ingest()` costruisce
un `RuleDocument`. Le submission `NON-EDGE` sono silenziosamente saltate.

**Formato `rule_id`.**

```
RULE_{SHORT_TICKER}_{NN:02d}
```

`SHORT_TICKER` è l'asset di base con il suffisso della valuta di quotazione
rimosso. FORGE riconosce i suffissi `USDC`, `USDT`, `BUSD`, `USD`, `EUR`,
`BTC`, `ETH`:

| Ticker sorgente | SHORT_TICKER | Esempi rule_id |
|---|---|---|
| `ADAUSDC` | `ADA` | `RULE_ADA_01`, `RULE_ADA_02` |
| `SOLUSDC` | `SOL` | `RULE_SOL_01` |
| `BTCUSDC` | `BTC` | `RULE_BTC_01` |
| `ETHBTC` | `ETH` | `RULE_ETH_01` |

Il contatore `NN` è indipendente per ogni `SHORT_TICKER` e inizia da `01` a
ogni sessione.

**Array di attivazione.**

I tre array paralleli `activation_idx`, `activation_dates` e `gains` vengono
recuperati dal replay del backtest sul ticker sorgente:

- `activation_idx` — indici interi delle barre in cui il trade è stato
  eseguito (il fill ha avuto successo)
- `activation_dates` — date ISO corrispondenti (una data per trade, stringa)
- `gains` — rendimento netto di ogni trade (fee incluse), parallelo ai due
  array precedenti

La lunghezza comune dei tre array è `n_eseguiti` — il numero di trade
effettivamente riempiti, non il numero di segnali.

---

### Step 2 — Matrici di correlazione

Dopo l'ingestion, `compute_correlations()` produce due matrici `N × N`
(N = numero di regole nel registro) e popola i campi `overlap_max` e
`gain_corr_max` di ogni documento.

#### Matrice A — Jaccard (sovrapposizione temporale)

Per ogni coppia di regole `(A, B)`, il coefficiente di Jaccard misura la
sovrapposizione delle **date di attivazione**:

```
jaccard(A, B) = |dates_A ∩ dates_B| / |dates_A ∪ dates_B|
```

Il risultato è in `[0, 1]`: 0 = nessun giorno in comune, 1 = identico
calendario di trade. La matrice è simmetrica con diagonale 1.

#### Matrice B — Spearman (correlazione dei guadagni)

Per ogni coppia `(A, B)`, i gain vengono allineati sull'asse delle date: ogni
data che compare in almeno uno dei due calendari diventa una riga; le date
senza trade per una delle regole contribuiscono con `0.0` (nessun trade,
rendimento zero).

La correlazione di Spearman viene calcolata senza dipendenze da scipy: si
applicano i ranghi alla serie allineata e poi si calcola il coefficiente di
Pearson sui ranghi. La matrice è simmetrica con diagonale 1.

**Soglia `min_active` (default 10).** Se il numero di barre con almeno un
trade in `A ∪ B` è inferiore a `cross_min_active`, la correlazione Spearman
viene riportata come `0.0` invece di essere calcolata — troppo pochi dati per
una stima affidabile.

**Annotazione per-documento.** Al termine, ogni `RuleDocument` riceve:

- `overlap_max` — il massimo coefficiente di Jaccard con qualsiasi altra
  regola del registro
- `gain_corr_max` — il massimo coefficiente di Spearman con qualsiasi altra
  regola del registro

---

### Step 3 — Deduplicazione

`deduplicate()` **marca** (non elimina) la regola più debole di ogni coppia
sovrapposta. Le regole rimangono nel registro; il flag `is_duplicate` le
identifica per l'eventuale esclusione in fase di export.

**Definizione di sovrapposizione.**

Una coppia `(A, B)` è sovrapposta se il coefficiente di Jaccard supera la
soglia `overlap_threshold` (default `0.70`).

**Definizione di regola più debole.**

La regola più debole è quella con il Profit Factor inferiore sul ticker
sorgente. In caso di parità, FORGE marca la seconda in ordine di ingestion.

**Propagazione a catena.**

La deduplicazione è chain-aware: se A domina B e B domina C, C viene marcato
come duplicato di B (il dominante immediato), non di A.

```
A (PF 3.2) → B (PF 2.8) → C (PF 2.1)

B: is_duplicate = True, duplicate_of = "RULE_ADA_01"   # A
C: is_duplicate = True, duplicate_of = "RULE_ADA_02"   # B
```

Campi popolati da questo step:

| Campo | Tipo | Descrizione |
|---|---|---|
| `is_duplicate` | `bool` | `True` se la regola è stata marcata come duplicato |
| `duplicate_of` | `str \| None` | `rule_id` del dominante immediato |

---

### Step 4 — Cross-ticker backtest

`cross_ticker()` rigioca ogni regola su tutti gli **altri** ticker della
sessione. Il risultato classifica ogni regola come `GENERIC`, `PARTIAL`,
`SPECIFIC` o `ISOLATED`.

#### Ricalibrazione delle soglie

L'espressione di una regola contiene due tipi di soglie:

- **Soglie assolute** (es. `RSI < 31.2`, `close < 45000`): dipendono dalla
  distribuzione del ticker sorgente e devono essere ricalibrate.
  Procedura: la soglia viene convertita nel suo percentile sulla distribuzione
  IS del ticker sorgente; quel percentile viene poi usato per ricavare la
  soglia equivalente sulla distribuzione del ticker target.
- **Soglie relative** (es. `pctrank < 0.20`, `zscore > 1.5`): sono
  approssimativamente invarianti per costruzione (già normalizzate) e vengono
  mantenute invariate.

L'espressione adattata è memorizzata in `CrossTickerResult.expression_adapted`.

#### Verdetto per coppia (regola, ticker target)

Per ogni coppia viene eseguito un backtest completo con i parametri operativi
della regola sorgente (buy_drop_pct, sell_pct, target_h, ecc.).

Il verdetto richiede **entrambe** le metà del criterio di trasferibilità:

```
PASS  ⟺  pf >= cross_pf_threshold                    (è tradeable là)
         AND  pf >= min_cross_pf_retention · pf_casa   (trasferisce davvero)
```

Un'unica asticella assoluta risponde a *«è buona altrove?»*, mentre ogni parola
del vocabolario `GENERIC`/`PARTIAL`/`SPECIFIC`/`ISOLATED` chiede *«trasferisce?»*.
Le due cose divergono in entrambe le direzioni:

| regola | PF casa | PF fuori | asticella unica `2.0` |
|---|---|---|---|
| trasferisce perfettamente | 1.6 | 1.6 | FAIL — e su ogni ticker, quindi `ISOLATED` |
| degradata di un terzo | 3.0 | 2.05 | PASS — con un terzo dell'edge perso |

La prima riga non era un caso limite: `partial_min_profit_factor` ammette le
regole a `1.5`, quindi **l'intera classe `PARTIAL-EDGE` era strutturalmente
esclusa dalla genericità**. Abbassare l'asticella unica al PF di casa sistema
quella riga e peggiora la seconda — le regole *più deboli* avrebbero il test di
genericità *più facile*. Con due metà il floor resta un floor e il rapporto
misura la trasferibilità; la qualità resta dove il registro già la registra, sul
verdetto M3 e sul grade.

Il valore effettivamente richiesto è riportato in `CrossTickerResult.bar`.

#### Classificazione di genericità

Al termine del cross-ticker, ogni regola viene classificata:

| `cross_ticker_score` | `cross_ticker_total` | `is_generic` | `classification` |
|---|---|---|---|
| qualsiasi | qualsiasi | — | `ISOLATED` (se `is_duplicate = True`) |
| `score / total >= generic_ratio_threshold` | — | `True` | `GENERIC` |
| `0 < score / total < generic_ratio_threshold` | — | `False` | `PARTIAL` |
| `score = 0` | — | `False` | `SPECIFIC` |

- `cross_ticker_score` — numero di ticker target con verdetto `PASS`
- `cross_ticker_total` — numero totale di ticker target (altri ticker della
  sessione)
- `generic_ratio_threshold` — default `2/3` (non `0.67`): con 3 ticker
  aggiuntivi bastano 2 `PASS` per essere `GENERIC`

**Sessione a ticker singolo.** Se il registro contiene un solo ticker, non
esiste nessun "altro ticker" su cui testare. In questo caso Step 4 non produce
`cross_ticker_results` e ogni regola riceve `classification = "ISOLATED"`.

---

### Step 5 — Export

Gli artefatti finali sono disponibili su richiesta esplicita dopo `run()`.

- `flat_table(apply_filters=False)` — DataFrame piatto con tutte le regole
- `export(path)` — scrive su disco in formato CSV o Excel
- `html_report(**kwargs)` — HTML autocontenuto con grafici SVG inline
- `summary()` — panoramica compatta, una riga per regola

---

## Struttura dati

### `RuleDocument`

Il documento centrale del registro. Ogni submission EDGE/PARTIAL-EDGE produce
esattamente un `RuleDocument`.

#### Campi di identificazione

| Campo | Tipo | Descrizione |
|---|---|---|
| `rule_id` | `str` | Identificatore univoco della sessione (es. `RULE_ADA_01`) |
| `expression` | `str` | Espressione booleana della regola |
| `source_ticker` | `str` | Ticker su cui la regola è stata scoperta (es. `"ADAUSDC"`) |
| `source_alpha_id` | `str` | ID del contratto Alpha Discovery sorgente |
| `verdict` | `str` | `"EDGE"` o `"PARTIAL-EDGE"` |
| `grade` | `str` | Grade (lettera) portato dall'Alpha Contract, o derivato dall'evidenza di Rule Discovery |

#### Array di attivazione

| Campo | Tipo | Descrizione |
|---|---|---|
| `activation_idx` | `list[int]` | Indici di barra dei trade eseguiti |
| `activation_dates` | `list[str]` | Date ISO dei trade eseguiti |
| `gains` | `list[float]` | Rendimento netto di ogni trade (fee incluse) |

I tre array sono paralleli: `activation_idx[i]`, `activation_dates[i]` e
`gains[i]` descrivono lo stesso trade.

#### Parametri operativi e statistiche

| Campo | Tipo | Descrizione |
|---|---|---|
| `params` | `dict` | Parametri `BacktestParams` della configurazione ottimale (buy_drop_pct, sell_pct, target_h, ecc.) |
| `stats` | `dict` | Metriche IS: `pf`, `win_rate`, `total_trades`, `tpm`, `zero_months`, `expectancy`, ecc. |
| `regime` | `dict` | Analisi di regime da Rule Discovery (dependency_score, avoid_in, per_regime) |

#### Campi popolati da Step 2 / 3

| Campo | Tipo | Default | Descrizione |
|---|---|---|---|
| `overlap_max` | `float \| None` | `None` | Massimo Jaccard con qualsiasi altra regola |
| `gain_corr_max` | `float \| None` | `None` | Massima Spearman con qualsiasi altra regola |
| `is_duplicate` | `bool \| None` | `None` | `True` se marcata come duplicato |
| `duplicate_of` | `str \| None` | `None` | `rule_id` del dominante immediato |

#### Campi popolati da Step 4

| Campo | Tipo | Default | Descrizione |
|---|---|---|---|
| `cross_ticker` | `dict[str, CrossTickerResult]` | `{}` | Risultati per ogni ticker target |
| `cross_ticker_score` | `int \| None` | `None` | Numero di `PASS` cross-ticker |
| `cross_ticker_total` | `int \| None` | `None` | Numero di ticker target testati |
| `is_generic` | `bool \| None` | `None` | `True` se `score/total >= generic_ratio_threshold` |
| `classification` | `str \| None` | `None` | `GENERIC` / `PARTIAL` / `SPECIFIC` / `ISOLATED` |

#### Proprietà di convenienza

```python
doc.pf  # float — profit factor sul ticker sorgente (da doc.stats["pf"])
```

#### Serializzazione

```python
doc.to_dict()  # dizionario nidificato completo (tutti i campi pubblici)
```

---

### `CrossTickerResult`

Esito del replay di una regola su un ticker alternativo.

| Campo | Tipo | Descrizione |
|---|---|---|
| `ticker` | `str` | Ticker target |
| `expression_adapted` | `str` | Espressione con soglie assolute ricalibrate sulla distribuzione target |
| `pf` | `float` | Profit factor sul ticker target |
| `win_rate` | `float` | Win rate sul ticker target (0–1) |
| `total_trades` | `int` | Trade eseguiti sul ticker target |
| `zero_months` | `int` | Mesi senza trade sul ticker target |
| `verdict` | `str` | `"PASS"` o `"FAIL"` |

```python
result = doc.cross_ticker["SOLUSDC"]
print(result.pf, result.verdict)
result.to_dict()  # serializzazione
```

---

### `CorrelationMatrices`

| Campo | Tipo | Descrizione |
|---|---|---|
| `rule_ids` | `list[str]` | Etichette riga/colonna, in ordine di registro |
| `jaccard` | `pd.DataFrame` | Matrice di Jaccard (N×N, simmetrica) |
| `spearman` | `pd.DataFrame` | Matrice di Spearman (N×N, simmetrica) |

```python
matrices = registry.matrices

# Matrice Jaccard come DataFrame
print(matrices.jaccard)

# Coppia specifica
j = matrices.jaccard.loc["RULE_ADA_01", "RULE_ADA_02"]
s = matrices.spearman.loc["RULE_ADA_01", "RULE_ADA_02"]
```

---

## Metodi di output

### `registry.run() → RuleRegistry`

Esegue in sequenza i Step 1–4 e restituisce `self`. Permette la concatenazione:

```python
registry = RuleRegistry(submissions, frames).run()
```

`run()` chiama `ingest()`, `compute_correlations()`, `deduplicate()` e
`cross_ticker()` nell'ordine corretto. L'export è separato e su richiesta.

---

### `registry.ingest() → list[RuleDocument]`

**Step 1.** Costruisce i documenti dalle submission. Restituisce la lista di
`RuleDocument` inseriti nel registro (le submission `NON-EDGE` sono escluse).
Popola `registry.documents`.

---

### `registry.compute_correlations() → CorrelationMatrices`

**Step 2.** Calcola le matrici di Jaccard e Spearman e annota `overlap_max` /
`gain_corr_max` su ogni documento. Restituisce il `CorrelationMatrices` e lo
memorizza in `registry.matrices`.

---

### `registry.deduplicate() → None`

**Step 3.** Marca i duplicati. Se `registry.matrices` è `None`, chiama
automaticamente `compute_correlations()` prima di procedere. Non ha valore
di ritorno; modifica i documenti in-place (`is_duplicate`, `duplicate_of`).

---

### `registry.cross_ticker() → None`

**Step 4.** Esegue il backtest cross-ticker e classifica ogni regola. Non ha
valore di ritorno; popola `cross_ticker`, `cross_ticker_score`,
`cross_ticker_total`, `is_generic` e `classification` su ogni documento.

---

### `registry.flat_table(apply_filters=False) → pd.DataFrame`

Restituisce un DataFrame con una riga per ogni `RuleDocument` nel registro.

Con `apply_filters=False` (default), tutte le regole sono incluse — duplicati
e non-generiche compresi. FORGE preferisce mostrare la complessità piuttosto
che nasconderla.

Con `apply_filters=True`, i filtri `export_duplicates` e `export_non_generic`
della `RegistryConfig` vengono applicati:

```python
# Tutte le regole
df_all = registry.flat_table()

# Solo le regole che passerebbero l'export
df_filtered = registry.flat_table(apply_filters=True)
```

---

### `registry.export(path) → str`

Scrive la tabella piatta su disco nel formato specificato da
`RegistryConfig.export_format` (`"excel"` o `"csv"`) e restituisce il percorso
del file scritto.

```python
flat_path = registry.export("forge_flat_table.xlsx")
print(f"Tabella scritta in: {flat_path}")
```

L'export applica i filtri `export_duplicates` / `export_non_generic` della
configurazione. Per esportare tutte le regole indipendentemente dalla
configurazione, usare `flat_table()` e scrivere il DataFrame manualmente.

---

### `registry.html_report(**kwargs) → str`

Genera un report HTML autocontenuto (stringa). Il report non dipende da
risorse esterne: CSS inline, grafici SVG inline, nessuna CDN.

Contenuto del report (controllato da `RegistryConfig`):
- Banner di sessione con data, ticker, numero di regole
- Sezione per ogni regola: expression, stats IS, classification badge
- Heatmap SVG della matrice Jaccard e Spearman (se `html_charts=True`)
- Equity curve per regola (se `html_charts=True`)
- Riepilogo cross-ticker con badge `GENERIC` / `PARTIAL` / `SPECIFIC` /
  `ISOLATED` per ogni combinazione (regola, ticker target)
- Banner per i duplicati con indicazione del dominante
- Trade log cross-sessione (se `html_include_tradelog=True`)

```python
html = registry.html_report(timeframe="1H")
with open("forge_report.html", "w", encoding="utf-8") as fh:
    fh.write(html)
```

Se `registry.matrices` è `None` al momento della chiamata,
`html_report()` chiama automaticamente `compute_correlations()`.

---

### `registry.summary() → pd.DataFrame`

Panoramica compatta: una riga per regola, una colonna per campo chiave.

```python
print(registry.summary().to_string(index=False))
```

Colonne restituite:

| Colonna | Descrizione |
|---|---|
| `rule_id` | Identificatore della regola |
| `source_ticker` | Ticker sorgente |
| `grade` | Grade lettera |
| `pf` | Profit factor IS |
| `win_rate` | Win rate IS (0–1) |
| `total_trades` | Trade eseguiti IS |
| `overlap_max` | Massimo Jaccard con altre regole |
| `is_duplicate` | Flag duplicato |
| `duplicate_of` | `rule_id` del dominante |
| `cross_ticker_score` | PASS cross-ticker |
| `cross_ticker_total` | Ticker target testati |
| `is_generic` | Flag genericità |
| `classification` | `GENERIC` / `PARTIAL` / `SPECIFIC` / `ISOLATED` |

---

## Configurazione completa — `RegistryConfig`

```python
from forgedge import RegistryConfig

config = RegistryConfig(
    overlap_threshold=0.70,
    cross_pf_threshold=1.5,
    min_cross_pf_retention=0.8,
    generic_ratio_threshold=2 / 3,
    export_format="excel",
)
```

| Parametro | Default | Descrizione |
|---|---|---|
| `overlap_threshold` | `0.70` | Soglia Jaccard sopra la quale due regole sono considerate duplicate (Step 3). La regola con PF inferiore viene marcata. |
| `gain_corr_threshold` | `0.70` | Soglia Spearman usata a fini di reporting ("stessa esposizione di regime"). Non guida la deduplicazione. |
| `cross_pf_threshold` | `1.5` | Floor assoluto di profit factor per un verdetto `PASS` (Step 4) — metà del criterio. Risolto dalla sessione a partire da `SelectionCriteria.partial_min_profit_factor`: l'asticella che ha ammesso la regola in casa. |
| `min_cross_pf_retention` | `0.8` | L'altra metà: frazione del profit factor **di casa** che la regola deve mantenere sul ticker target. |
| `generic_ratio_threshold` | `2/3` | Frazione minima di `PASS` per il flag `is_generic` e il badge `GENERIC`. Usare `2/3` (non `0.67`) per evitare errori di arrotondamento al limite. |
| `cross_min_active` | `10` | Numero minimo di barre attive allineate per il calcolo della Spearman. Sotto questa soglia, la Spearman è riportata come `0.0`. |
| `export_format` | `"excel"` | Formato di export della tabella piatta: `"excel"` o `"csv"`. |
| `export_duplicates` | `True` | Includi i duplicati nell'export (applicato solo quando `apply_filters=True` o con `export()`). |
| `export_non_generic` | `True` | Includi le regole non-generiche nell'export. |
| `html_include_tradelog` | `True` | Aggiungi il trade log cross-sessione al report HTML. |
| `html_charts` | `True` | Incorpora i grafici SVG inline (equity curve, heatmap). |
| `timestamp_col` | `"open_dt"` | Colonna datetime sui DataFrame di ogni ticker. |
| `session_date` | `None` | Data ISO stampata sul report. `None` → data odierna. |

---

## `RuleSubmission`

Il tipo di ingresso del registro: una regola validata con il suo contesto di
ricostruzione.

```python
from forgedge import RuleSubmission

sub = RuleSubmission(
    ticker="ADAUSDC",
    response=ada_resp,      # RuleDiscoveryResponse
    candidate=ada_cand,     # EventCandidate
    grade="A",              # opzionale
)
```

| Campo | Tipo | Descrizione |
|---|---|---|
| `ticker` | `str` | Ticker sorgente (es. `"ADAUSDC"`) |
| `response` | `RuleDiscoveryResponse` | Verdetto di Rule Discovery. Solo `EDGE` / `PARTIAL-EDGE` entrano nel registro. |
| `candidate` | `EventCandidate` | L'Event Candidate referenziato dalla regola validata. Fornisce il percorso di replay deterministico (`apply`) usato dal backtest cross-ticker. |
| `grade` | `str \| None` | Grade lettera portato dall'upstream (es. il grade di Alpha Discovery). Se `None`, il registro deriva un grade dall'evidenza di Rule Discovery. |

---

## Pattern d'uso avanzati

### 1. Costruzione manuale con submissions esplicite

```python
from forgedge import RuleRegistry, RuleSubmission, RegistryConfig

submissions = []
for contract, response in rule_discovery_results:
    if response.is_edge:
        cand = candidates_by_id[response.validated_rule.event_candidate_id]
        submissions.append(
            RuleSubmission(
                ticker="ADAUSDC",
                response=response,
                candidate=cand,
                grade=contract.alpha_score.grade if contract.alpha_score else None,
            )
        )

registry = RuleRegistry(
    submissions,
    frames={"ADAUSDC": ada_kpi_df},
    config=RegistryConfig(overlap_threshold=0.65),
).run()
```

### 2. Filtrare solo le regole non-duplicate e generiche dalla flat table

```python
df = registry.flat_table()

# Regole operative: non duplicate e GENERIC
df_operative = df[
    (df["is_duplicate"] == False) &
    (df["classification"] == "GENERIC")
]

# Regole operative allargato: non duplicate e almeno PARTIAL
df_candidate = df[
    (df["is_duplicate"] == False) &
    (df["classification"].isin(["GENERIC", "PARTIAL"]))
]
```

### 3. Ispezionare le matrici di correlazione

```python
matrices = registry.matrices

# Coppia con sovrapposizione massima
j = matrices.jaccard
j_no_diag = j.where(~pd.concat(
    [pd.Series(True, index=[i]) for i in j.index],
    axis=1
).T.values)
max_pair = j.stack().idxmax()
print(f"Coppia più sovrapposta: {max_pair} — Jaccard={j.stack().max():.3f}")

# Correlazione Spearman tra due regole specifiche
corr = matrices.spearman.loc["RULE_ADA_01", "RULE_SOL_01"]
print(f"Spearman ADA_01 ↔ SOL_01: {corr:.3f}")
```

### 4. Accedere ai risultati cross-ticker per una regola

```python
doc = next(d for d in registry.documents if d.rule_id == "RULE_ADA_01")

print(f"Classificazione: {doc.classification}")
print(f"Score: {doc.cross_ticker_score}/{doc.cross_ticker_total}")

for ticker, result in doc.cross_ticker.items():
    print(f"  {ticker}: PF={result.pf:.2f}  WR={result.win_rate:.1%}"
          f"  trade={result.total_trades}  {result.verdict}")
    print(f"    expression adattata: {result.expression_adapted}")
```

### 5. Export selettivo — solo non-duplicate e generiche

```python
# Opzione A: configurare il registro per escludere dal CSV/Excel
config = RegistryConfig(
    export_duplicates=False,
    export_non_generic=False,
    export_format="excel",
)
registry = RuleRegistry(submissions, frames, config=config).run()
path = registry.export("forge_only_generic.xlsx")

# Opzione B: export manuale con filtro applicato a posteriori
df = registry.flat_table(apply_filters=True)
df.to_excel("forge_filtered.xlsx", index=False)

# Opzione C: filtro personalizzato
df_all = registry.flat_table()
mask = (
    (~df_all["is_duplicate"]) &
    (df_all["pf"] >= 2.5)
)
df_all[mask].to_csv("forge_custom.csv", index=False)
```

---

## Note operative

- **Il registro è stateless.** Nessun catalogo persistente tra sessioni. La
  tabella piatta esportata è l'unico artefatto di persistenza; la gestione
  storica delle regole (confronto tra sessioni, versioning, catalogo a lungo
  termine) è di responsabilità dell'utente.

- **Frames da passare.** I DataFrame passati in `frames` devono essere i
  DataFrame post-pipeline di Event Discovery — non la KPI Table originale
  grezza. Devono contenere le colonne native dei prezzi (`open`, `high`, `low`,
  `close`) e tutte le feature referenziate dai candidati. `RuleRegistry.__init__`
  ordina cronologicamente i frame e imposta il `DatetimeIndex`, ma non
  ricalcola le feature mancanti.

- **Submissions NON-EDGE.** `ingest()` salta silenziosamente le submission con
  `response.is_edge == False`. Non viene lanciato nessun avviso né eccezione.
  Se si vuole verificare quante submission sono state scartate, confrontare
  `len(registry.submissions)` con `len(registry.documents)` dopo `ingest()`.

- **Sessione a ticker singolo.** Se tutti le submission appartengono allo
  stesso ticker (o c'è un solo ticker in `frames`), Step 4 non ha altri ticker
  su cui testare. Ogni regola riceve `cross_ticker_score = 0`,
  `cross_ticker_total = 0`, `cross_ticker = {}` e
  `classification = "ISOLATED"`. Questo non indica una regola debole — solo
  che la generalizzabilità non è misurabile in questa sessione.

- **Ricalibro delle soglie.** Il ricalibro usa la distribuzione IS del ticker
  target — non la distribuzione live o OOS. Questo è coerente con il principio
  di Rule Discovery: tutto ciò che riguarda la calibrazione si basa sul periodo
  in-sample del dataset fornito.

- **Dipendenza da openpyxl.** L'export Excel richiede `openpyxl`. Se non
  installato, usare `export_format="csv"` oppure installare con
  `pip install openpyxl`.
