# FORGE — Deployment: portare le regole scoperte in produzione

`forgedge.deployment` è il modulo gemello di `forgedge.playground` che
prosegue da dove finisce il toolkit di analisi: dati i contratti tradeable
(`EDGE`/`PARTIAL-EDGE`) prodotti da una sessione `forge()`, decide quali sono
abbastanza solidi da andare live, li scrive su disco in un formato
replicabile, e indicizza cosa è stato esportato per un job di monitoraggio
periodico. A differenza di `forgedge.playground`, questo modulo ha
**effetti reali** — `promotion_gate()` prende una decisione di go/no-go,
`export_rules()` scrive file.

Questa è una guida all'utilizzo: firme, parametri, colonne restituite ed
esempi verificati. Per la motivazione di design (perché il modulo è stato
separato da `forgedge.playground`, perché le tre funzioni girano in
sequenza fissa, perché solo `export_rules` tocca il filesystem) vedi
`src/forgedge/docs/modules/Deployment.md`.

**Storia del naming:** queste tre funzioni vivevano originariamente dentro
`forgedge.playground` (issue #245). Sono state spostate nel proprio modulo
top-level da PR #247 perché hanno effetti reali che un nome "playground" di
sola lettura non descriveva più onestamente — vedi
`src/forgedge/docs/modules/Playground.md` §10 per la storia completa.
Nessun cambio di comportamento, solo il percorso di import:

```python
# prima (issue #245, ora obsoleto)
from forgedge.playground import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest

# dopo (PR #247, attuale)
from forgedge.deployment import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest
```

---

## Utilizzo di base

```python
from forgedge import forge
from forgedge.deployment import PromotionGateConfig, promotion_gate, export_rules, monitoring_manifest

result_ada = forge(kpi_ada, ticker="ADAUSDC", timeframe="1D")
result_btc = forge(kpi_btc, ticker="BTCUSDC", timeframe="1D")
results = [result_ada, result_btc]

# i registries sono opzionali — passali per gatire anche su is_duplicate/is_isolated
gate = promotion_gate(results, registries=[result_ada.registry, result_btc.registry])
exported = export_rules(results, "exported_rules/", registries=[result_ada.registry, result_btc.registry])
manifest = monitoring_manifest(results)
```

La sequenza prevista è `forge() -> promotion_gate() [filtra] -> export_rules()
[scrive, sulle sole regole promuovibili] -> monitoring_manifest() [indicizza
l'export]` — `export_rules()` riesegue internamente lo stesso calcolo del
gate (vedi il documento di design), quindi non è mai in disaccordo con
`promotion_gate()` su cosa sia promuovibile.

---

## `PromotionGateConfig`

Dataclass che porta la policy di promozione condivisa da `promotion_gate()`
ed `export_rules()`. Ogni flag sotto è sempre calcolato e riportato su ogni
riga indipendentemente da questi settaggi — i campi `block_*`/
`require_consistency` decidono solo quali flag confluiscono nella colonna
finale `promotable`, così disattivare un controllo non fa mai perdere
visibilità su cosa avrebbe segnalato.

| Campo | Default | Effetto |
|---|---|---|
| `min_consistency` | `0.5` | Soglia su `RuleDiscoveryResponse.walk_forward.consistency` (frazione di fold walk-forward OOS profittevoli) — la stessa soglia che la pipeline usa internamente per un verdetto positivo. |
| `require_consistency` | `True` | Se `min_consistency` partecipa a `promotable`. |
| `block_rotation_only` | `False` | Blocca un `PARTIAL-EDGE` il cui unico ostacolo al pieno `EDGE` era il rotation null a livello di ricerca. Default `False` — un rotation-only miss è di solito un compromesso accettabile, non un campanello d'allarme. |
| `block_duplicate` | `True` | Blocca una regola che il Rule Registry ha marcato `is_duplicate=True`. |
| `block_isolated` | `True` | Blocca una regola classificata `"ISOLATED"` sul replay cross-ticker. Nessun effetto (`is_isolated` resta `None`) se non sono stati passati `registries`. |
| `min_fold_stability_score` | `None` | Soglia sullo stability score penalizzato per varianza tra fold (#253): `mean(fold_pf) - std(fold_pf)` calcolato sui `test_summary.profit_factor` per-fold di `RuleDiscoveryResponse.walk_forward.splits` (ciascuno prima limitato a `fold_pf_cap`). Cattura una regola il cui PF walk-forward aggregato sembra forte solo perché un singolo fold ad alta varianza (spesso il sentinel `9999.0` "zero trade in perdita") domina l'aggregato. `None` disattiva il gate; una regola con meno di due split walk-forward lo supera sempre (la std campionaria è indefinita su un solo fold). |
| `fold_pf_cap` | `10.0` | Cap applicato al `test_summary.profit_factor` di ciascun fold prima di calcolare `fold_stability_score`, così un singolo fold col valore sentinel non può dominare media/std. |

```python
config = PromotionGateConfig(block_rotation_only=True, min_consistency=0.6)
config = PromotionGateConfig(min_fold_stability_score=1.0)   # #253
```

---

## `promotion_gate(results, registries=None, config=PromotionGateConfig()) -> pd.DataFrame`

Gate di qualità in formato long su ogni contratto tradeable (`EDGE`/`PARTIAL-EDGE`).

Calcola, per contratto, gli stessi flag che le funzioni playground M3/M4
espongono singolarmente (`rotation_only` di `lottery_only_winners`,
`is_duplicate` di `duplicate_clusters`, la classificazione `"ISOLATED"` di
`classification_by_grade`, la `consistency` del walk-forward, e lo stability
score penalizzato per varianza tra fold `fold_stability_score`, #253), poi
li combina in `promotable` secondo `config`. Pura — nessun I/O su filesystem.

**Parametri:**
- `results: Iterable[ForgeResult]` — R, uno o più output di `forge()`/`forge_multi()`.
- `registries: Iterable[RuleRegistry], opzionale` — sorgente per `is_duplicate`/`classification` (vedi `modules/Deployment.md` per il perché è separato da `results`, stessa motivazione delle funzioni M4 di `forgedge.playground`). `None` salta questi due controlli (le colonne restano `None`) invece di fallire.
- `config: PromotionGateConfig` — quali controlli bloccano la promozione, e a quale soglia.

**Colonne restituite:** `ticker`, `alpha_id`, `grade`, `verdict`,
`rotation_only`, `is_duplicate`, `is_isolated`, `consistency`,
`fold_stability_score`, `promotable`.

```python
gate = promotion_gate(results, registries=[result_ada.registry, result_btc.registry])
gate[gate["promotable"]].groupby("ticker").size()   # quante regole superano il gate, per ticker
```

**Verificato**, mettendo in pool `forge_multi()` su `ADAUSDC` (il fixture di
riferimento del repository) e una seconda serie sintetica etichettata
`BTCUSDC` — lo stesso pool usato da ogni esempio "Verificato" nelle sezioni
M1/M3/M4 di `playground_it.md` e sotto:

```
pg.shape == (96, 10)
pg["promotable"].value_counts()
# False    91
# True      5
pg.groupby("ticker")["promotable"].sum()
# ADAUSDC    0
# BTCUSDC    5
```

`min_fold_stability_score` è `None` di default, quindi non contribuisce a
`promotable` qui — i conteggi sopra sono invariati rispetto a prima di
#253; la nuova colonna `fold_stability_score` è comunque popolata per
l'audit.

Ogni contratto `ADAUSDC` è bloccato su questo fixture — `duplicate_clusters`
(`forgedge.playground`) ha già mostrato che il 51% dei contratti in pool è
duplicato e `classification_by_grade` ha mostrato che la maggior parte
delle regole si classifica `"ISOLATED"`, e i due blocchi di default a
`True` (`block_duplicate`, `block_isolated`) si sommano su un ticker dove
entrambi sono comuni. È il gate che fa il suo lavoro conservativo di
default, non un bug.

---

## `export_rules(results, output_dir, *, registries=None, config=PromotionGateConfig(), promotable_only=True, include_kpi_recipe=True, kpi_config=None, kpi_recipe_warmup_bars=0) -> pd.DataFrame`

Scrive un `.pkl` (evento) + un `.yaml` (parametri della regola) per ogni
contratto esportato. L'unica funzione di questo modulo — e dell'intera
libreria, a parte gli helper espliciti di scrittura report — il cui scopo è
un effetto collaterale sul filesystem.

Esegue lo stesso calcolo di `promotion_gate()` internamente (così le due non
sono mai in disaccordo su cosa sia promuovibile) e, per ogni contratto
selezionato, scrive:

- **`{output_dir}/{alpha_id}.pkl`** — l'`EventCandidate` via `pickle`, che porta con sé la propria funzione di attivazione deterministica (`EventCandidate.apply`) — nessuna ricostruzione manuale necessaria per rigiocare l'evento in seguito.
- **`{output_dir}/{alpha_id}.yaml`** — `ValidatedRule.to_dict()` (il punto operativo pubblicato: direzione, entry mode, parametri buy/sell, holding period, fee) più `ticker`/`alpha_id`/`verdict` per contesto, scritto con un piccolo writer YAML senza dipendenze (ogni valore è uno scalare piatto, quindi non serve una libreria YAML).
- **`{output_dir}/{alpha_id}.kpi_recipe.json`** (`include_kpi_recipe=True`, issue #296) — la ricetta `kpi_builder` minima che ricalcola solo le colonne KPI lette dall'evento, più un blocco `"verification"` (il controllo di round-trip su `result.event_frame`). Vedi [*Ricetta KPI minima*](#ricetta-kpi-minima-issue-296) sotto. Il `.pkl` viene comunque sempre scritto: resta il fallback fedele per gli eventi che la ricetta non sa ricostruire.

**Parametri:**
- `results: Iterable[ForgeResult]` — R.
- `output_dir: str | Path` — directory in cui scrivere; creata se mancante.
- `registries: Iterable[RuleRegistry], opzionale` — inoltrato al calcolo del gate sottostante.
- `config: PromotionGateConfig` — inoltrato al calcolo del gate sottostante.
- `promotable_only: bool, default True` — esporta solo i contratti che il gate marca `promotable`. Imposta `False` per esportare ogni contratto tradeable indipendentemente dal gate (le colonne del gate sono comunque riportate per audit).
- `include_kpi_recipe: bool, default True` — scrive anche `{alpha_id}.kpi_recipe.json`.
- `kpi_config: dict | str | Path | None` — la config `kpi_builder` con cui è stata costruita la KPI Table (`None` = `DEFAULT_CONFIG`), inoltrata a `minimal_kpi_recipe()`.
- `kpi_recipe_warmup_bars: int, default 0` — barre escluse dall'inizio di ogni `result.event_frame` nella verifica della ricetta. Lascia `0` quando la KPI Table è stata costruita con `kpi_builder` dalle stesse candele passate a `forge()`; impostalo quando è stata costruita su uno storico più lungo e poi troncata (vedi *Warm-up* sotto). Nessun default euristico.

Una riga il cui candidato non risolve o il cui `response.validated_rule` è
`None` viene saltata silenziosamente — nessun file scritto, nessuna eccezione.

**Colonne restituite:** `ticker`, `alpha_id`, `event_candidate_id`,
`verdict`, `promotable`, `pkl_path`, `yaml_path`, `kpi_recipe_path`,
`kpi_recipe_verified` — una riga per ogni contratto effettivamente
esportato. `kpi_recipe_verified` è `None` quando non è stata scritta alcuna
ricetta o quando il risultato non ha un `event_frame` su cui verificarla.

```python
exported = export_rules(results, "exported_rules/", registries=[result_ada.registry, result_btc.registry])
len(exported)   # quanti contratti sono stati effettivamente scritti su disco
```

**Verificato**, stesso pool, config di default (`promotable_only=True`) —
misurato prima che la #296 aggiungesse `kpi_recipe_path`/`kpi_recipe_verified`
(e un `.kpi_recipe.json` per contratto), quindi oggi il frame è `(5, 9)` e i
file scritti sono 15:

```
exp.shape == (5, 7)
# 10 file scritti (5 .pkl + 5 .yaml)
exp.iloc[0][["ticker", "alpha_id", "verdict", "promotable"]]
# ticker       BTCUSDC
# alpha_id     ALPHA-BTCUSDC-1D-260830-1196
# verdict      PARTIAL-EDGE
# promotable       True
```

Tutti e 5 i contratti esportati sono `BTCUSDC` — coerente con
`promotion_gate` sopra, che su questo fixture non trova nessun contratto
`ADAUSDC` promuovibile.

---

## `monitoring_manifest(results: Iterable[ForgeResult], exported: pd.DataFrame | None = None) -> pd.DataFrame`

Indice in formato long di ogni regola tradeable, per un job di ricontrollo periodico.

Applica `RuleSpec.from_forge_result` (già fornita da `forgedge.rule_report`
per una singola run — vedi il manuale, §9, pattern 5) su tutto R, così un
job di monitoraggio ha un unico file che elenca ogni regola da rigiocare su
candele fresche via `RuleDiscovery` — mai `AlphaDiscovery` — invece di
ricostruire a mano il riferimento a ciascuna.

**Colonne restituite:** `ticker`, `rule_name`, `event_candidate_id`,
`is_end`, `verdict`, `oos_expectancy`, `kpi_recipe_path`,
`kpi_recipe_verified`. Fai il join su `event_candidate_id` contro l'output
di `export_rules` per restringerti alle sole regole effettivamente
esportate.

Passa `exported=` (il frame restituito da `export_rules`) per valorizzare
`kpi_recipe_path`/`kpi_recipe_verified` (join su `(ticker,
event_candidate_id)`): il job di ricontrollo sa così quali regole può tenere
in vita dalla sola ricetta minima. Senza, le due colonne sono `None`; le
righe non vengono mai filtrate da questo parametro.

```python
manifest = monitoring_manifest(results, exported=exported)
manifest.merge(exported[["event_candidate_id"]], on="event_candidate_id")   # restringi alle sole regole esportate
```

**Verificato**, stesso pool:

```
mm.shape == (96, 6)   # misurato prima della #296; ora (96, 8)
mm["verdict"].value_counts()
# PARTIAL-EDGE    96
```

Ogni regola tradeable su questo fixture è `PARTIAL-EDGE` — lo stesso fatto
che `lottery_only_winners` in `forgedge.playground` osserva dal lato
dell'analisi (`src/forgedge/docs/specs/playground_it.md`).

---

## Ricetta KPI minima (issue #296)

```python
from forgedge.deployment import KpiRecipe, minimal_kpi_recipe, verify_kpi_recipe

recipe = minimal_kpi_recipe(candidate)                 # kpi_config=None -> DEFAULT_CONFIG
recipe.to_dict()
# {'build_features_config': {'volatility': {'enabled': True,
#                                           'params': {'periods': [12, 24], 'columns': ['close']}}},
#  'lag_features': [], 'candle_features': False, 'pattern_features': False,
#  'color': False, 'unresolved_columns': [], 'base_columns': [...]}

reduced = recipe.rebuild(candles)                      # solo le colonne che servono a questa regola
check = verify_kpi_recipe(candidate, recipe, result.event_frame)
check.matches, check.n_mismatched_bars, check.last_mismatch_at

KpiRecipe.from_json(open("exported_rules/ALPHA-....kpi_recipe.json").read())  # riletta da disco
```

Il pickle dell'`EventCandidate` riproduce l'evento esattamente, ma è fragile
tra versioni della libreria, opaco (non dice quali colonne KPI un job di
monitoraggio deve calcolare) e illeggibile fuori da Python. Una `KpiRecipe`
è il complemento leggibile: la ricetta `kpi_builder` più piccola — config di
`build_features`, flag `candle_features`/`pattern_features`/`color`,
richieste `lag_features` ordinate — che ricalcola esattamente le colonne
native lette dai componenti dell'evento (`source_cols`, o `source_feature`
per i componenti arity-1) a partire dalle sole candele OHLC(V).

- **Perché qui e non su `EventCandidate`.** `event_discovery` non dipende da
  `kpi_builder` per scelta; risolvere una ricetta richiede di chiamare
  `build_features()`. `deployment` è il livello a valle che può dipendere da
  entrambi.
- **Risoluzione per costruzione.** Ogni indicatore/colonna/periodo di
  `kpi_config` viene costruito una volta su un piccolo frame sintetico e i
  nomi delle colonne prodotte vengono indicizzati verso la voce che li
  produce — in cache per config, quindi un intero export costa un numero
  fisso di probe (~130 per `DEFAULT_CONFIG`, ~0,2 s) e ogni candidato solo
  lookup in un dizionario. Ogni voce viene sondata indipendentemente da
  `enabled`, quindi anche gli indicatori opt-in (ATR, MACD, Aroon, …) si
  risolvono; `aroon_up_NN`, senza prefisso, non richiede casi speciali.
- **Periodi posizionali.** I `periods` di MACD sono triple `(fast, slow,
  signal)`: la ricetta mantiene ogni tripla necessaria intatta e in ordine
  (ordinare `[12, 26, 9]` in `[9, 12, 26]` costruisce un MACD diverso — bug
  riprodotto nel prototipo della #295). Ogni futuro indicatore posizionale
  va aggiunto a `kpi_recipe._POSITIONAL_PERIOD_GROUPS`.
- **Superset per indicatore.** `build_features` fa `periods × columns` per
  indicatore, quindi un evento che usa `close_ema_12` e `low_ema_03` ottiene
  anche `close_ema_03`/`low_ema_12`. Mai una colonna mancante.
- **Fuori perimetro → `unresolved_columns`.** Le formule `CustomEvent`
  (riportate testualmente), le colonne `regime` di M0, colonne non standard
  o fornite dall'utente, e grafie di lag che `lag_features()` non scrive mai
  (`close_prev_2`). `rebuild()` rifiuta una ricetta incompleta
  (`ValueError`); `verify_kpi_recipe()` la riporta (`matches=False`,
  `error=...`).

**Warm-up.** `verify_kpi_recipe()` confronta le attivazioni dell'evento barra
per barra (per timestamp, NaN letto come inattivo) sulla tabella di
riferimento e su quella ricostruita. Quando la KPI Table di riferimento è
stata costruita da `kpi_builder` dalle stesse candele, il round-trip è
esatto: **23 488 / 23 488** candidati di una Event Discovery sull'OHLC della
fixture ADA 1D ricostruita con `DEFAULT_CONFIG` + MACD/ATR/Stochastic/Aroon
+ `candle_features` + lag e un lag-di-lag. Quando il riferimento è stato
costruito su uno storico più lungo e poi troncato — come
`tests/fixtures/ADA_1D_TRAIN.parquet` stessa — gli indicatori ricorsivi e a
finestra mobile ricostruiti sullo span più corto differiscono durante un
transitorio (EMA-12: prime 86 barre su 882; propagato attraverso una
finestra pctrank di 168 barre, fino a ~260 barre), e le colonne non
arrotondate `close_bb_mid_*`/`close_bb_width_*` possono differire per rumore
floating-point ovunque (le somme mobili dipendono dal percorso), cosa che
molto raramente ribalta un confronto che cade esattamente sulla soglia.
Restringi il confronto con `evaluation_mask=` (o
`export_rules(kpi_recipe_warmup_bars=...)`) e leggi
`n_mismatched_bars`/`first_mismatch_at`/`last_mismatch_at` — in questa prima
versione non viene applicata alcuna finestra di default euristica.

Ancora aperto (issue #296, fase 3): se `monitoring_manifest()` debba
aggregare le ricette in un unico file indicizzato per `alpha_id` invece di
un JSON per regola, una volta misurate dimensioni di export reali.

---

## Cosa manca ancora

I tre casi d'uso di questo modulo (issue #245) sono tutti implementati — non
c'è una checklist aperta qui come per `forgedge.playground` (o come c'era).
Eventuali futuri casi d'uso sulla messa in produzione verrebbero tracciati
come nuove issue GitHub su questo modulo, non riaprendo la #245.
