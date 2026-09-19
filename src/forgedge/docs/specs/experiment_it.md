# FORGE — Experiment: pipeline di esperimenti costruite su `forge()`

`forgedge.experiment` è un terzo modulo gemello accanto a `forgedge.playground`
(analisi di sola lettura) e `forgedge.deployment` (produzione) — **esegue una
propria pipeline multi-fase sopra `forge()`**, invece di limitarsi a leggere o
agire su un `ForgeResult` già esistente. `StepWiseDiscovery`, la sua prima
classe, fa crescere regole a condizione singola confermate sull'hold-out una
condizione AND alla volta, cercando nella sotto-popolazione attiva di
ciascuna regola una seconda dimensione — una ricerca locale per seme,
distinta dall'unico passaggio globale e guidato dal grade della
composizione a due passate di `forge()` stesso.

Questa è una guida all'utilizzo: firma, parametri, campi restituiti ed
esempi verificati. Per la motivazione di design (perché la risoluzione
della configurazione avviene una sola volta, perché la ridondanza si
controlla sia dentro una catena sia tra i semi, la storia della ricerca
manuale che ha motivato ogni fix) vedi
`src/forgedge/docs/modules/Experiment.md`.

```python
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig
```

---

## Utilizzo di base

```python
import pandas as pd
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig

kpi = pd.read_parquet("kpi_table.parquet")   # 'close' + una colonna timestamp, come forge()

engine = StepWiseDiscovery(
    kpi, ticker="BTCUSDC", timeframe="1D",
    config=StepWiseDiscoveryConfig(n_seeds=3, depth=2),
)
result = engine.run()

print(result.summary())          # una riga per catena: expression, depth_reached, verdict, ...
for chain in result.edges():     # catene il cui stato finale (ultimo confermato) è EDGE/PARTIAL-EDGE
    print(chain.chain_label, chain.expression, chain.verdict)
```

Qualunque calibrazione specifica dell'asset (`AlphaConfig.fee_per_side`,
`mfe_floor`, soglie dei gate, ...) è responsabilità del chiamante, passata
già risolta tramite `event_discovery_config`/`alpha_config`/
`rule_discovery_config` — `StepWiseDiscovery` le tratta come date e non le
rimette mai in discussione. Lasciare tutte e tre a `None` (il default) le
risolve allo stesso modo di una chiamata a `forge()` non configurata.

---

## `StepWiseDiscovery(kpi, *, ticker=None, asset="ASSET", timeframe, event_discovery_config=None, alpha_config=None, rule_discovery_config=None, config=None, run_market_context=True)`

**Parametri:**
- `kpi: pd.DataFrame` — la KPI Table completa. Mai modificata.
- `ticker: str, opzionale` — inoltrato a `forge()` e a ogni chiamata diretta ad `AlphaDiscovery`/`RuleDiscovery` di questa classe, così gli ID restano coerenti tra l'iterazione 1 e ogni catena.
- `asset: str` — fallback di tracciabilità, stessa semantica del parametro `asset` di `forge()` (usato solo quando né `ticker` né un `alpha_config` esplicito ne impostano uno).
- `timeframe: str` — dimensione della barra, es. `"1D"`, `"1H"`. Keyword obbligatoria.
- `event_discovery_config: DiscoveryConfig, opzionale` — `max_and_components` viene forzato a `1` indipendentemente da cosa viene passato; la composizione è compito esclusivo di questa classe.
- `alpha_config: AlphaConfig, opzionale`
- `rule_discovery_config: RuleDiscoveryConfig, opzionale`
- `config: StepWiseDiscoveryConfig, opzionale` — parametri dell'algoritmo di ricerca (vedi tabella sotto). Default `StepWiseDiscoveryConfig()`.
- `run_market_context: bool, default True` — inoltrato alla chiamata interna a `forge()` per l'iterazione 1.

**`.run() -> StepWiseDiscoveryResult`** — esegue l'intero algoritmo (vedi il docstring del modulo / `modules/Experiment.md` §3) e restituisce il risultato. Solleva `ValueError` prima di eseguire qualunque cosa se `config.strict` (default `True`) e la configurazione risolta ha un esito di coerenza di livello `FAIL` — lo stesso contratto fail-fast di `forge(strict=True)`.

**`.log_lines: list[str]`** — ogni riga diagnostica registrata durante `.run()` (riepilogo del TimeBudget, accettazione/scarto dei semi con motivazione, conteggi dei candidati per profondità, tentativi di regola composta), in ordine. Utile da stampare insieme a `result.summary()`, allo stesso modo in cui il manuale raccomanda di loggare `result.ledger.describe()`/`result.resolution.describe()` accanto a una normale run di `forge()`.

---

## `StepWiseDiscoveryConfig`

Nessuno di questi campi è specifico per asset o timeframe — quella calibrazione va sulle configurazioni passate al costruttore.

| Campo | Default | Effetto |
|---|---|---|
| `n_seeds` | `3` | Numero di semi distinti da cui far partire una catena indipendente. |
| `depth` | `2` | Profondità massima di composizione AND per catena. |
| `train_ratio` | `0.80` | Frazione della KPI table riservata allo split esterno di hold-out — indipendente da, e in aggiunta a, qualunque split interno usato dalle configurazioni passate. |
| `outer_horizon_bars` / `outer_embargo_bars` | `None` / `None` | Larghezza di purge/embargo per lo split esterno. `None` li deriva dal `alpha_config.horizon_grid`/`embargo_bars` RISOLTO — la stessa quantità che userebbe `forge()` stesso; un valore esplicito ha sempre priorità (es. una scala derivata dalla semivita OU, se disponibile). |
| `retain_ratio_floor` | `0.2` | Floor assoluto su `retain_ratio_min_k = max(floor, min_trades_M3 / n_righe)`, ricalcolato ad ogni profondità dalla dimensione reale della partizione CORRENTE — un floor adattivo legato al requisito statistico di M3 (`rule_discovery_config.criteria.min_oos_trades`), non un conteggio fisso di righe. |
| `max_constituent_jaccard` | `0.85` | Stesso default di `forgedge.event_discovery.diversity_gate`/`ANDComposer(max_constituent_jaccard=...)` — vedi `forgedge.experiment.redundancy`. |
| `max_constituent_abs_corr` | `0.95` | Deliberatamente più severa della soglia Jaccard — deve catturare solo quasi-duplicati, non segnali semplicemente correlati ma genuinamente distinti. |
| `child_gate` | `None` | Consistency Gate per i figli valutati su una partizione. `None` usa `GateParams(min_tpm=0.25, dispersion_margin=3.0, min_episodes=4, event_counting="episode", episode_gap=1)` — più permissivo di un gate di sessione tipico (una ricerca su sotto-popolazione ne ha bisogno), ma non così permissivo da affamare di attivazioni la regola composta prima che Alpha Discovery possa derivarne un target. |
| `min_composed_activations` | `20` | Un candidato padre-AND-figlio sotto questa soglia di attivazioni sulla search frame completa viene scartato prima ancora di provare Alpha Discovery — fallisce velocemente con una ragione esplicita invece di un opaco "no derivable target" di M2. |
| `min_local_partition_rows` | `10` | Un figlio a trasformazione rolling la cui serie locale (ristretta alla partizione) ha meno righe non-NaN di questa viene saltato. |
| `strict` | `True` | Inoltrato a `forge()`/`config_report()` — solleva su un'incoerenza di livello `FAIL` invece di procedere. |

```python
config = StepWiseDiscoveryConfig(n_seeds=5, depth=3, max_constituent_abs_corr=0.90)
```

---

## `StepWiseDiscoveryResult`

- **`chains: list[ChainResult]`** — uno per seme, in ordine di classifica dei semi.
- **`seed_attempts: list[SeedAttempt]`** — ogni candidato esaminato durante la selezione dei semi, in ordine di classifica, sia diventato seme o no (`family`, `alpha_id`, `verdict`, `composite_score`, `accepted`, `reason`, `jaccard_vs_seeds`, `abs_corr_vs_seeds`) — permette di verificare *perché* un candidato ben classificato è stato scartato (mancata conferma sull'hold-out vs. ridondanza inter-seme).
- **`feature_recurrence: dict[str, list[str]]`** — solo osservazione, non un gate: quale famiglia di feature è ricorsa tra quali catene/profondità durante la ricerca locale dei figli.
- **`search_df`, `holdout_df`: pd.DataFrame** — lo split esterno effettivamente usato. `holdout_df` non è mai stato toccato da alcuna chiamata di discovery/alpha/rule-discovery in questa run.
- **`time_budget: TimeBudget`** — il budget dello split esterno.
- **`coherence: ConfigReport`** — la configurazione risolta con cui questa run è stata eseguita, stessa forma di `ForgeResult.coherence`.
- **`config: StepWiseDiscoveryConfig`** — i parametri dell'algoritmo usati in questa run.
- **`.summary() -> pd.DataFrame`** — una riga per catena: `chain`, `seed_family`, `expression`, `depth_reached`, `composite_score`, `verdict`, `confirm_reason`, `stop_reason`.
- **`.edges() -> list[ChainResult]`** — catene il cui stato finale (ultimo confermato) è `EDGE`/`PARTIAL-EDGE`.

## `ChainResult`

`chain_label`, `seed_family`, `candidate: EventCandidate`, `contract: AlphaContract`, `response: RuleDiscoveryResponse`, `depth_reached: int`, `confirm_reason: str`, `stop_reason: str`, più le proprietà di comodo `.expression`, `.composite_score`, `.verdict`. `candidate`/`contract`/`response` descrivono l'ULTIMO STATO CONFERMATO della catena — il seme da solo se nessuna composizione ha mai confermato sull'hold-out, o la regola composta più profonda che lo ha fatto; un tentativo di composizione che fallisce la conferma sull'hold-out non viene mai scritto qui (vedi `modules/Experiment.md` §2, principio 5).

---

## `forgedge.experiment.redundancy` — i controlli di diversità

Funzioni pure, senza dipendenza dalla pipeline — usabili e testabili in autonomia:

- **`family_key(candidate) -> str`** — raggruppa un candidato per famiglia di feature, togliendo i periodi numerici da `source_feature` (basato sul nome, più grezzo dei due controlli sotto).
- **`max_jaccard(bool_series, used_series_list) -> float`** — `J(A, B) = |A∩B| / |A∪B|` contro ogni serie della lista; stessa definizione di `forgedge.event_discovery.diversity_gate`.
- **`max_abs_corr(components, used_cols, series_by_col) -> float`** — massima `|correlazione di Pearson|` della serie CONTINUA pre-soglia di ciascun componente (cercata in `series_by_col`, indicizzata per `transformed_col`) contro ogni colonna in `used_cols`. Chiude un limite che il solo Jaccard lascia aperto: due componenti la cui informazione continua è quasi identica (un rapporto su ATR e il suo analogo su NATR) possono comunque avere attivazioni booleane divergenti solo per dove è caduta ciascuna soglia.
- **`is_redundant(*, bool_series, used_bool_series, components, used_cols, series_by_col, max_jaccard_threshold=0.85, max_abs_corr_threshold=0.95) -> (bool, float, float)`** — combina entrambi i controlli; restituisce i due punteggi anche quando nessuno supera la soglia, per il logging.

---

## Esempio verificato

Sul fixture di riferimento di questo repository `tests/fixtures/ADA_1D_TRAIN.parquet`
(882 barre), `DiscoveryConfig`/`AlphaConfig`/`RuleDiscoveryConfig` di default,
`StepWiseDiscoveryConfig(n_seeds=2, depth=2)`:

```python
import pandas as pd
from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig

kpi = pd.read_parquet("tests/fixtures/ADA_1D_TRAIN.parquet")
engine = StepWiseDiscovery(kpi, ticker="ADAUSDC", timeframe="1D",
                            config=StepWiseDiscoveryConfig(n_seeds=2, depth=2))
result = engine.run()
print(result.summary()[["chain", "expression", "depth_reached", "verdict"]].to_string(index=False))
```

```
                                        chain                                                            expression  depth_reached  verdict
seed1:diffnorm_close_vol_vol|rolling_pctrank                          pr_diffnorm_close_vol12_vol24_168 < 0.172619              0  PARTIAL-EDGE
          seed2:ratio_close_ema_ema|identity  (ratio_close_ema03_ema12 < 0.952059) AND (zs_close_ema_03_96 < -1.5)              1  NON-EDGE
```

`seed2` raggiunge la profondità 1 — una regola composta a due condizioni
genuinamente confermata (verdetto di ricerca `NON-EDGE`, ma expectancy
media positiva sui fold walk-forward post-holdout sulla valutazione
concatenata search+holdout). `seed1` resta a condizione singola: l'unica
composizione tentata a profondità 0 non ha superato la conferma
sull'hold-out, quindi la catena riporta correttamente l'ultimo stato
confermato (il seme) invece del tentativo fallito. Nessuna delle due
catene raggiunge la profondità 2 su questo fixture.

Un secondo esempio eseguibile sui dati OHLCV bundled di questo repository
(EURUSD 1D, con la diagnostica del fee-per-side che un asset non-crypto
richiede prima di fidarsi di un verdetto): `examples/step_wise_discovery_usage.py`.

---

## Prossimi passi

`StepWiseDiscovery` è la prima classe di questo modulo — future pipeline di
esperimenti costruite su `forge()` (es. un orchestratore di ricerca
cross-ticker o cross-timeframe) andrebbero aggiunte come nuove classi qui,
non ampliando lo scopo di questa.
