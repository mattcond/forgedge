# Ricetta KPI minimale per regola — manifest di redeploy ridotto e verificato

> **Status: Fasi 1 e 2 implementate** in `src/forgedge/deployment/kpi_recipe.py`
> (`KpiRecipe`, `KpiRecipeVerification`, `minimal_kpi_recipe()`,
> `verify_kpi_recipe()`) e integrate in `export_rules()`/`monitoring_manifest()`;
> test in `tests/test_kpi_recipe.py` e `tests/test_deployment.py`. Riferimento
> d'uso: `src/forgedge/docs/specs/deployment_en.md` (`_it.md`), §*Minimal KPI
> recipe*. La Fase 3 (aggregazione del manifest a scala, §5.1.3) resta aperta.
> Il resto di questo documento è la proposta originale, lasciata com'era; le
> deviazioni dall'API proposta sono elencate in §7. Issue:
> [#296](https://github.com/mattcond/forgedge/issues/296).

**Punto di partenza:** mentre si costruiva una pipeline di validazione hold-out in un
notebook (PR #295), è emersa l'esigenza di salvare, per ogni regola persistita, non
solo i parametri operativi e le performance ma anche *il minimo indispensabile* per
ricalcolare le colonne KPI che l'evento referenzia — senza portarsi dietro l'intero
set di feature usato in fase di discovery (nella fixture di test, 51 colonne per
un evento che ne usa 1-2).

**Domanda a cui risponde questo documento:** l'idea è buona, ma dove va il codice
nella libreria, quali sono i confini esatti di cosa può e non può ricostruire, e quali
bug concreti si sono già incontrati prototipandola che l'implementazione in libreria
deve evitare fin dall'inizio?

**Metodo:** il prototipo nel notebook è stato scritto, eseguito e fatto fallire
deliberatamente più volte (smoke test su 15 candidati reali della fixture
`tests/fixtures/ADA_1D_TRAIN.parquet`) prima di scrivere questo documento — i due bug
descritti in §4 sono stati riprodotti e risolti lì, non sono ipotetici.

---

## 1. Perché oggi non basta il pickle di `export_rules()`

`forgedge.deployment.export_rules()` (`deployment/rules.py`) già persiste, per ogni
contratto esportato, l'`EventCandidate` intero via `pickle` (`{alpha_id}.pkl`) più i
parametri operativi in YAML. Funziona — un pickle ricostruisce l'oggetto esatto,
inclusa la logica di `.apply()` — ma ha tre limiti che una ricetta minima risolve:

1. **Fragilità di versione.** Un pickle creato con una versione di `forgedge` non è
   garantito deserializzarsi correttamente con un'altra: i dataclass di
   `event_discovery` non hanno un contratto di stabilità binaria (né dovrebbero
   averlo — non è quello il loro scopo).
2. **Opacità operativa.** Un pickle non dice a un sistema di monitoraggio *quali
   colonne della pipeline di feature engineering* servono per far rivivere quella
   specifica regola su candele fresche. Il sistema a valle deve ricalcolare l'intero
   set di KPI (spesso 50+ colonne quando sono attivi gli indicatori opt-in) anche per
   tenere in vita una singola regola a una o due feature.
3. **Non ispezionabilità/portabilità.** Un pickle Python non è leggibile in sicurezza
   da un consumatore non-Python, né diffabile in un sistema di revisione testuale.

Una ricetta minima — esempio reale dal prototipo:

```json
{
  "build_features_config": {
    "volatility": {"enabled": true, "params": {"periods": [12, 24], "columns": ["close"]}}
  },
  "lag_features": [],
  "candle_features": false,
  "pattern_features": false,
  "unresolved_columns": []
}
```

risolve tutti e tre: è JSON piatto, leggibile, diffabile, e permette di calcolare in
produzione solo le colonne che contano per quella regola. Non sostituisce il pickle —
lo affianca come manifest leggibile, con il pickle che resta il fallback per i casi che
la ricetta non sa ricostruire (§5).

## 2. Il vincolo architetturale che decide dove va il codice

`event_discovery` **non dipende da `kpi_builder`**, per scelta esplicita e documentata
nel codice stesso. Da `event_discovery/feature_generator.py`:

> "the same default is documented in kpi_builder's default_enricher.yaml as ... —
> duplicating it from kpi_builder's config would need a new runtime dependency..."
>
> "...would require a new kpi_builder -> event_discovery coupling..."

Ricostruire una ricetta minima richiede necessariamente di chiamare
`kpi_builder.build_features()` per sapere quali colonne produce una data combinazione
indicatore/periodo/colonna. Questo significa che **la funzionalità non può vivere come
metodo su `EventCandidate`** (`event_discovery/models.py`) senza introdurre esattamente
l'accoppiamento che quel modulo evita deliberatamente. Deve vivere in un modulo che già
dipende legittimamente da entrambi i lati: `event_discovery` (per leggere
`EventCandidate.components`) e `kpi_builder` (per `build_features`/`DEFAULT_CONFIG`/
`candle_features`/`pattern_features`/`lag_features`).

`forgedge.deployment` è esattamente questo modulo — importa già `..forge`,
`..rule_registry`, `..rule_report`; è il livello di orchestrazione a valle, non di
dominio, ed è lì che vive già l'unico altro punto di persistenza per-regola
(`export_rules()`). Non esiste nel codice un modulo chiamato "persist"; il punto più
vicino a quell'idea è `export_rules()` stesso.

`forgedge.playground` (il sibling read-only di `deployment`) è stato considerato e
scartato come sede primaria: il suo contratto, dichiarato nel proprio `__init__.py`, è
"una `DataFrame` long-format, una riga per osservazione elementare". Una ricetta per
regola è invece un dict annidato più un booleano di verifica — una forma che non si
adatta a quel pattern senza forzarlo artificialmente.

## 3. API proposta

Nuovo sottomodulo `forgedge/deployment/kpi_recipe.py`, esportato da
`forgedge.deployment.__init__`:

```python
@dataclass
class KpiRecipe:
    build_features_config: dict    # {indicator: {"enabled": True, "params": {"periods": [...], "columns": [...]}}}
    lag_features: list[dict]       # [{"column": ..., "periods": [...]}]
    candle_features: bool
    pattern_features: bool
    unresolved_columns: list[str]  # non vuoto = ricetta incompleta, vedi §5

    def to_dict(self) -> dict: ...

    def rebuild(self, candles: pd.DataFrame, timestamp_col: str = "open_dt") -> pd.DataFrame:
        """Ricostruisce una KPI Table ridotta dalle sole candele OHLC(V) + questa ricetta."""


@dataclass
class KpiRecipeVerification:
    matches: bool
    n_mismatched_bars: int
    first_mismatch_at: Optional[pd.Timestamp]


def minimal_kpi_recipe(
    candidate: EventCandidate,
    kpi_config: dict = DEFAULT_CONFIG,
    base_cols: Sequence[str] = ("open_dt", "open", "high", "low", "close", "volume"),
) -> KpiRecipe:
    """Ricetta minima per ricalcolare le sole colonne KPI che `candidate` referenzia.

    Legge `candidate.components[*].source_cols` (o `.source_feature` per i
    componenti arity-1) — mai la naming convention per induzione/regex — e risale,
    per ciascuna colonna nativa non già in `base_cols`, a quale voce di `kpi_config`
    la produce, costruendo la mappa inversa una volta sola (§4).
    """


def verify_kpi_recipe(
    candidate: EventCandidate,
    recipe: KpiRecipe,
    candles: pd.DataFrame,
    evaluation_mask: Optional[pd.Series] = None,
) -> KpiRecipeVerification:
    """candidate.apply(recipe.rebuild(candles)) deve riprodurre esattamente
    candidate.apply(candles) sulla finestra indicata da `evaluation_mask`
    (default: l'intera serie — il chiamante che conosce un confine train/hold-out
    dovrebbe passare esplicitamente la maschera corretta, vedi §5.1)."""
```

**Integrazione in `export_rules()`** (`deployment/rules.py`): nuovo parametro
`include_kpi_recipe: bool = True`. Per ogni contratto esportato, scrive anche
`{alpha_id}.kpi_recipe.json`, usando `result.event_frame` — già disponibile su
`ForgeResult`, nessun parametro nuovo richiesto al chiamante — sia come sorgente per
`minimal_kpi_recipe()` sia come riferimento per `verify_kpi_recipe()`. La colonna
`kpi_recipe_verified` si aggiunge al DataFrame-manifest già restituito da
`export_rules()`.

## 4. Strategia di implementazione — diversa (e migliore) del prototipo

Il prototipo nel notebook fa probing **brute-force**: per ogni colonna nativa mancante,
prova ogni combinazione `(indicatore, colonna, periodo)` del catalogo chiamando
`build_features()` una volta per combinazione — costo O(componenti × catalogo).
Accettabile su un notebook con poche decine di regole; non scala a un `export_rules()`
chiamato su centinaia o migliaia di contratti promossi (uno scenario reale: la skill
`forgedge` documenta run con `243`+ regole tradeable su una singola fixture).

**Strategia corretta per l'implementazione in libreria:** costruire *una sola volta*
una mappa inversa `{nome_colonna_prodotta: (indicatore, periodo_o_tripla, colonna_sorgente)}`
chiamando `build_features()` *una sola volta* con `kpi_config` per intero (non voce per
voce) su un frame sintetico minimo, poi per ogni evento fare solo lookup O(1) nella
mappa già costruita. Stesso risultato del prototipo, O(1) chiamate a `build_features()`
invece di O(n) sull'intero export.

## 5. Confini — dentro e fuori

### Dentro

- Eventi (singoli o AND-composti) i cui componenti referenziano colonne prodotte da
  `kpi_builder.build_features()` — inclusi tutti gli indicatori opt-in (ATR, MACD,
  CCI, Williams %R, Stochastic, WMA, TRIMA, ADX, Aroon, Williams A/D) —,
  `candle_features()`, `pattern_features()`, o loro lag via `lag_features()`
  (suffisso `_prev_NN`).
- Verifica round-trip su dati reali, riusando `ForgeResult.event_frame` già
  disponibile, senza richiedere parametri aggiuntivi al chiamante di `export_rules()`.

### Fuori (esplicitamente, per questa prima versione)

- **`CustomEvent`/`manual_events`.** Un `pd.DataFrame.eval()` su formula libera può
  referenziare colonne arbitrarie, anche proprietarie, mai riconducibili a
  `kpi_builder`. Il campo `unresolved_columns` le segnala correttamente come non
  risolvibili — non è un bug da correggere, è un limite strutturale del dominio: non
  esiste, in generale, un modo per risalire da un nome di colonna arbitrario a come
  ricalcolarlo.
- **Naming convention non standard.** Una feature che non segue
  `{base}_{indicator}_{period}` esce già dal radar del `FeatureGenerator` (pitfall #4
  della skill `forgedge`); esce allo stesso modo, per lo stesso motivo, dal radar di
  `minimal_kpi_recipe()`.
- **Qualunque modifica a `event_discovery`, `rule_discovery`, `alpha_discovery`.**
  Zero rischio sugli invarianti della pipeline: la feature è additiva, a valle,
  interamente nel layer di deployment, e non altera in alcun modo come un evento viene
  scoperto o validato.
- **Sostituire l'export via `pickle`.** `export_rules()` continua a scrivere sempre il
  `.pkl` — resta il meccanismo di fallback per gli eventi che la ricetta non sa
  ricostruire.

### 5.1 Limiti noti e domande aperte

1. **Warm-up numerico degli indicatori ricorsivi.** Verificato empiricamente durante
   il prototipo: ricostruendo `close_ema_12` da zero sulla fixture
   `tests/fixtures/ADA_1D_TRAIN.parquet` (882 bar), la serie differisce dall'originale
   (costruito su uno storico più lungo, poi troncato) solo nei primi **86 bar**, poi
   diventa identica bit a bit per sempre. Un confronto round-trip sull'**intera** serie
   fallirebbe per un motivo puramente numerico (il transitorio dello stato ricorsivo
   dell'EMA), non per una ricetta sbagliata — il prototipo è passato da 7/15 a 15/15
   candidati corretti nello smoke test proprio limitando il confronto alla finestra
   di hold-out invece che all'intera serie train+hold-out.

   Domanda aperta: `verify_kpi_recipe()` deve calcolare da sé una finestra di default
   sicura (es. "tutto tranne i primi `candidate.dominant_window() × k` bar", con `k` da
   calibrare su più fixture), o deve richiedere sempre una maschera esplicita a carico
   del chiamante? Si propone la seconda opzione per la v1 — nessuna euristica non
   validata su un solo asset/timeframe — con la prima come follow-up una volta
   raccolta evidenza su più casi.

2. **Parametri posizionali (MACD e simili).** `periods` per MACD è la tripla
   posizionale `(fast, slow, signal)`, non un insieme di finestre indipendenti.
   **Bug reale riprodotto nel prototipo:** accumulare questi valori in un `set()` e poi
   ordinarli numericamente (`sorted({12, 26, 9})` → `[9, 12, 26]`) genera una config
   che produce colonne con nomi sbagliati (`close_macd_9_12_signal_26` invece di
   `close_macd_12_26_...signal_09`), con un `KeyError` silenzioso a valle in
   `EventCandidate.apply()` nonostante la ricetta sembrasse corretta (nessuna colonna
   in `unresolved_columns`). L'implementazione in libreria deve trattare MACD (e
   qualunque futuro indicatore con semantica posizionale) come caso esplicito nella
   mappa inversa, mai come un insieme ordinabile.

3. **Dimensione del manifest a scala.** Con migliaia di contratti promossi, migliaia
   di file `{alpha_id}.kpi_recipe.json` individuali potrebbero diventare scomodi
   quanto i `.pkl` attuali. Da valutare in Fase 3 (§6): se `monitoring_manifest()`
   debba aggregare le ricette in un unico file indicizzato per `alpha_id` invece di
   un file per regola.

## 6. Piano di implementazione a fasi

- **Fase 1** — `KpiRecipe`, `KpiRecipeVerification`, `minimal_kpi_recipe()`,
  `verify_kpi_recipe()` in `deployment/kpi_recipe.py`, con la mappa-inversa O(1)
  (§4), non il probing brute-force del prototipo. Test unitari (`tests/test_deployment.py`
  o un nuovo `tests/test_kpi_recipe.py`, nello stile già in uso nel repo — dati
  sintetici seedati, nessun mocking) che coprano: feature arity-1 e arity-2+, lag,
  MACD (parametri posizionali, §5.1.2), colonne non risolvibili
  (`unresolved_columns` popolato correttamente su un `CustomEvent`). Nessuna
  integrazione con `export_rules()` ancora — API isolata e testabile da sola.
- **Fase 2** — integrazione in `export_rules()` (`include_kpi_recipe: bool = True`) e
  `monitoring_manifest()` (colonne `kpi_recipe_path`/`kpi_recipe_verified`).
- **Fase 3** (follow-up, non bloccante) — decisione sull'aggregazione del manifest a
  scala (§5.1.3) una volta misurato un caso reale con un numero di regole promosse
  rappresentativo.

## Riferimenti

- Prototipo e verifica empirica: PR #295,
  `notebooks/07_mysql_holdout_pipeline.ipynb` §8 (funzioni
  `extract_minimal_kpi_recipe`/`rebuild_from_recipe`/`verify_minimal_recipe`) — 15/15
  su uno smoke test di candidati reali dopo i due fix di §5.1.1-2.
- `src/forgedge/deployment/rules.py` — `export_rules()`, punto di integrazione
  proposto.
- `src/forgedge/event_discovery/feature_generator.py` — commenti che documentano il
  confine `event_discovery` ⇏ `kpi_builder` (§2).
- Skill `forgedge`, pitfall #4 — naming convention non standard fuori dal radar del
  `FeatureGenerator` (stesso limite si applica qui, §5).
- Issue [#296](https://github.com/mattcond/forgedge/issues/296).

## 7. Note di implementazione (deviazioni dalla proposta)

- **Mappa inversa:** non una singola chiamata `build_features()` con `kpi_config`
  per intero — da quella non si può attribuire ogni colonna prodotta alla voce che
  l'ha prodotta. È invece un probe per ogni combinazione indicatore × colonna ×
  gruppo-di-periodi su un frame sintetico di 8 barre, costruito **una volta per
  config** (`lru_cache`): ~130 probe / ~0,2 s per `DEFAULT_CONFIG`, poi solo lookup
  per ogni candidato. L'obiettivo di §4 (costo indipendente dal numero di regole
  esportate) è rispettato. Le voci vengono sondate ignorando `enabled`, così gli
  indicatori opt-in si risolvono anche se la config passata li ha disabilitati.
- **Campi aggiunti a `KpiRecipe`:** `color` (la colonna `color` di
  `build_features` — il prototipo non la ricostruiva se la config risultava vuota)
  e `base_columns`; `is_complete`, `to_json()`/`from_dict()`/`from_json()` per
  rileggere il file esportato.
- **`KpiRecipeVerification`** riporta anche `n_compared_bars`, `last_mismatch_at`
  (mismatch confinati all'inizio = firma del warm-up) ed `error` (ricetta
  incompleta, colonna mancante) invece di sollevare.
- **Finestra di verifica (§5.1.1):** come proposto, nessuna euristica in v1 —
  `verify_kpi_recipe(evaluation_mask=...)` e
  `export_rules(kpi_recipe_warmup_bars=...)`, default l'intera serie. Misurato:
  quando la KPI Table è costruita da `kpi_builder` sulle stesse candele il
  round-trip è esatto su 23 488 / 23 488 candidati (fixture ADA 1D ricostruita con
  `DEFAULT_CONFIG` + MACD/ATR/Stochastic/Aroon + candle features + lag). Sulla
  fixture così com'è (costruita su uno storico più lungo) il warm-up si propaga
  attraverso le finestre pctrank fino a ~260 barre; in più le colonne non
  arrotondate `close_bb_mid_*`/`close_bb_width_*` differiscono per rumore
  floating-point lungo tutta la serie, che in rari casi ribalta un confronto
  esattamente sulla soglia — non eliminabile con una maschera, visibile da
  `n_mismatched_bars`.
- **`CustomEvent`:** la formula viene riportata testualmente in
  `unresolved_columns` (non si tenta di estrarne gli identificatori).
- **`monitoring_manifest(results, exported=None)`:** colonne
  `kpi_recipe_path`/`kpi_recipe_verified` sempre presenti (schema stabile),
  valorizzate solo passando l'output di `export_rules()`.
