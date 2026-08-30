# FORGE — Riferimento completo dei configuratori

Tutti i configuratori di FORGE sono dataclass Python standard: mutabili dopo l'istanziazione, componibili per nesting e dotati di default calibrati su dati crypto 1H liquidi. Non è necessario specificare tutti i parametri: l'utente tocca solo ciò che vuole cambiare rispetto ai default. I configuratori di alto livello (es. `MarketContextConfig`, `DiscoveryConfig`, `RuleDiscoveryConfig`) accettano i sotto-configuratori come argomenti nidificati, seguendo esattamente la gerarchia descritta in questo documento.

---

## Modulo 0

### EMAProxyConfig

Configura il classificatore EMA-proxy: sorgente dati, calcolo automatico delle finestre, soglie di regime e calibrazione adattiva. Viene passato come campo `ema_proxy` di `MarketContextConfig`.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `source_col` | str | `"close"` | Colonna OHLCV su cui calcolare le EMA. Non serve che le EMA siano già nella tabella. |
| `auto_window` | bool | `True` | Se True, ricava le finestre EMA dall'analisi OU/Hurst dei dati. Se False, usa `short_period`/`long_period` fissi. |
| `short_period` | int | `9` | Span dell'EMA veloce (usato come fallback o quando `auto_window=False`). |
| `long_period` | int | `25` | Span dell'EMA lenta (usato come fallback o quando `auto_window=False`). |
| `thresholds` | list[float] | `[0.975, 0.990, 1.010, 1.025]` | 4 cut point sul ratio `ema_short/ema_long` in modalità `"fixed"`. |
| `threshold_mode` | str | `"fixed"` | `"fixed"`: usa i threshold assoluti. `"balanced"`: ricalcola i threshold come quantili per avvicinarsi a `target_distribution`. |
| `target_distribution` | list[float] | `[0.10, 0.20, 0.40, 0.20, 0.10]` | Frequenze target dei 5 regimi (usato in modalità `"balanced"`). |
| `threshold_basis` | str | `"global"` | `"global"`: quantili calcolati sull'intera serie (non causale). `"expanding"`: quantili causali (look-ahead free, accuratezza approssimata). |
| `threshold_warmup` | int | `200` | Barre iniziali in cui si usano i threshold fissi prima che `"expanding"` sia stabile. |
| `window_unit` | str | `"day"` | Unità per `window_estimation`/`window_stride`: `"day"` (giorni calendario) o `"bar"` (barre). |
| `window_estimation` | float | `168` | Ampiezza della finestra di stima delle EMA, nell'unità scelta. 168 giorni = ~24 settimane. |
| `window_stride` | float | `1` | Passo tra stime successive, nella stessa unità. |
| `bar_hours` | float \| None | *(risolto: da `timeframe`)* | Durata esplicita della candela in ore (es. `4.0` per 4H). Risolto a livello di sessione; senza sessione viene inferito dal DatetimeIndex. |
| `fast_ratio` | float | `1/2.3 ≈ 0.435` | Rapporto span_veloce/span_lento per la derivazione automatica. |
| `min_window_estimates` | int | `10` | Numero minimo di stime OU convergenti per fidarsi della derivazione automatica. |

```python
from forgedge import MarketContext, MarketContextConfig, EMAProxyConfig

config = MarketContextConfig(
    ema_proxy=EMAProxyConfig(
        auto_window=True,
        window_unit="day",
        bar_hours=4.0,                # candele da 4H
        threshold_mode="balanced",
        target_distribution=[0.10, 0.20, 0.40, 0.20, 0.10],
        threshold_basis="expanding",  # causale, no look-ahead
    )
)
```

---

### MarketContextConfig

Contenitore di primo livello per la configurazione del Modulo 0. Aggrega la scelta del classificatore, i suoi parametri e le opzioni di stabilità del regime.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `classifier` | str | `"ema_proxy"` | Implementazione del classificatore. In v1.0 solo `"ema_proxy"` è disponibile. |
| `ema_proxy` | EMAProxyConfig | `EMAProxyConfig()` | Configurazione del classificatore EMA-proxy. |
| `labels` | list[str] | `["STRONG_BEAR","BEAR","NEUTRAL","BULL","STRONG_BULL"]` | Etichette dei 5 regimi, dal più ribassista al più rialzista. |
| `stable_window` | int | *(risolto: 12h di regime invariato, minimo 2 barre)* | Numero di barre consecutive identiche richieste per `regime_stable=True`. 12 su 1H, 3 su 4H, 2 su 1D, 48 su 15m — un 12 fisso chiedeva dodici *giorni* su candele daily. |

```python
from forgedge import MarketContext, MarketContextConfig, EMAProxyConfig

enriched = MarketContext(
    kpi,
    config=MarketContextConfig(
        stable_window=6,          # più reattivo al cambio di regime
        ema_proxy=EMAProxyConfig(
            auto_window=True,
            window_unit="day",
            bar_hours=1.0,
        ),
    ),
).run()
```

---

## Modulo 1

### GateParams

Soglie del ConsistencyGate — il filtro strutturale che verifica se un evento ha struttura temporale stabile. Viene passato come campo `gate_params` di `DiscoveryConfig`.

Dall'issue #134 il gate opera in una di due modalità di conteggio, selezionate
da `event_counting`: **`"episode"`** (default) conta i run massimali di
attivazioni consecutive invece delle barre grezze, così uno stato persistente
multi-barra (un tratto di 3–5 barre con `RSI < 30`) non viene penalizzato
come se fossero più trigger indipendenti; **`"bar"`** riproduce esattamente
il conteggio pre-#134, barra per barra. Le due modalità coincidono per gli
eventi impulsivi (crossover, pattern candlestick — un episodio per barra).

La doppia modalità cambia quale campo di dispersione legge davvero il gate
(issue #205): `max_dispersion` è la soglia grezza sull'Index of Dispersion,
ma viene letta **solo in modalità `"bar"`**. In modalità `"episode"` il gate
confronta invece contro `poisson_floor(n_months) × dispersion_margin` — un
margine sopra il floor statisticamente difendibile di Poisson, non un ID
assoluto — perché il floor dominava quasi sempre un `max_dispersion` fisso
in pratica (misurato: 12 combinazioni preset×timeframe su 16 non avevano mai
`max_dispersion` vincolante). Quindi con il default `event_counting="episode"`
va calibrato `dispersion_margin`, non `max_dispersion`.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `min_tpm` | float | `0.5` | Frequenza media minima di *trigger* al mese, nell'unità scelta da `event_counting` (episodi/mese in modalità `"episode"`, barre/mese in modalità `"bar"`). Default 0.5 ≈ "almeno un episodio ogni due mesi". |
| `max_dispersion` | float | `1.5` | Index of Dispersion massimo ammesso (`Var/Mean` dei conteggi mensili). **Solo modalità `"bar"`** — in modalità `"episode"` (default) questo campo non viene letto affatto dal gate; vedi `dispersion_margin`. |
| `dispersion_margin` | float | `1.3` | **Tolleranza di dispersione della modalità `"episode"`** — un margine sopra il floor χ² di Poisson (`eff_max_dispersion = poisson_floor(n_months) × dispersion_margin`), non un Index of Dispersion assoluto. `1.05` resta vicino a quanto produrrebbe un processo di Poisson; `3.0` tollera deliberatamente clustering Poisson-implausibile. Non letto in modalità `"bar"`. |
| `event_counting` | `"episode"` \| `"bar"` | `"episode"` | Unità di conteggio per i criteri di frequenza/dispersione — vedi sopra. |
| `min_episodes` | int | `10` | Floor assoluto sul numero di episodi richiesto per passare in modalità `"episode"` (guardia di potenza statistica). Ignorato in modalità `"bar"`, e applicato solo in-sample. `forge_preset()` lo abbassa a `5` su `"sweep"` (permissivo per design); gli altri preset mantengono `10`. |
| `episode_gap` | int | `1` | Gap massimo, in barre, che appartiene ancora allo stesso episodio. Con il default `1`, una singola barra mancante dentro un run non apre un nuovo episodio. `0` impone run strettamente consecutivi. |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import GateParams

config = DiscoveryConfig(
    gate_params=GateParams(
        min_tpm=0.3,             # meno restrittivo per dataset più corti
        dispersion_margin=1.6,   # più margine sopra il floor di Poisson
        min_episodes=5,
        event_counting="episode",  # default; "bar" riproduce il comportamento pre-#134
    )
)
```

> I campi sopra hanno sostituito uno schema `GateParams(min_act, min_months,
> max_conc, min_tpm)` più vecchio; nessuno tra `min_act`/`min_months`/
> `max_conc` esiste più, e costruire `GateParams` con questi nomi solleva
> oggi un `TypeError`. Alcuni script `examples/*.py` di questo repo usano
> ancora lo schema vecchio — vedi la lista dei pitfall della skill `forgedge`
> prima di copiarli.

---

### EventWalkForwardConfig (Modulo 1)

Configura la validazione walk-forward OOS del Modulo 1. Prima si chiamava `WalkForwardConfig` e collideva con l'omonima classe — diversa — del Modulo 3; il vecchio nome resta come alias in `event_discovery.models`, ma è preferibile quello esplicito.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `n_splits` | int | `3` | Numero di finestre OOS uguali su cui riprodurre ogni evento. |
| `min_pass_rate` | float | `0.6` | Frazione minima di finestre in cui l'evento deve superare il gate per essere marcato OOS-stabile. |
| `oos_gate_params` | GateParams \| None | `None` | Soglie del gate specifiche per la valutazione OOS. Se None, i parametri vengono scalati automaticamente proporzionalmente alla durata della finestra OOS rispetto all'IS. |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams

config = DiscoveryConfig(
    train_ratio=0.80,
    walk_forward=EventWalkForwardConfig(
        n_splits=4,
        min_pass_rate=0.75,   # più severo: 3 su 4 finestre devono passare
    ),
)
```

---

### DiscoveryConfig

Configurazione principale del Modulo 1. Controlla gate, composizione degli eventi AND, split IS/OOS e walk-forward.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `gate_params` | GateParams | `GateParams()` | Soglie del ConsistencyGate. |
| `max_categorical_classes` | int | `20` | Colonne categoriche con più valori distinti di questo limite sono classificate ma escluse dalla generazione di eventi. |
| `scale_free_overrides` | dict[str,bool] \| None | `None` | Override manuali del flag scale-free per colonne specifiche (es. `{"rsi_14": True}`). Utile quando l'euristica automatica fallisce su serie corte. |
| `timestamp_col` | str | `"open_dt"` *(risolto dalla sessione)* | Colonna datetime nella KPI Table (o nome dell'indice DatetimeIndex). |
| `max_and_components` | int | `2` | Numero massimo di singoli eventi da combinare in un AND. Valori > 3 sono tecnicamente accettati ma sconsigliati (overfitting strutturale). |
| `train_ratio` | float | `1.0` | Frazione di barre IS (0 < train_ratio ≤ 1.0). Default 1.0 = tutto IS (nessun split). |
| `walk_forward` | EventWalkForwardConfig \| None | `None` | Configurazione walk-forward OOS. Attivo solo se anche `train_ratio < 1.0`. |
| `diversity_gate_enabled` | bool | `False` | Se True, applica una deduplicazione Jaccard degli eventi singoli dopo il ConsistencyGate e prima della composizione AND. Opt-in — nessun breaking change. |
| `diversity_threshold` | float | `0.85` | Similarità Jaccard massima tollerata tra due eventi conservati. Usato solo con `diversity_gate_enabled=True`. A p99 della distribuzione Jaccard inter-evento (12 mesi di dati 1H), Jaccard=0.47 — valori sopra 0.70 sono genuine near-duplicate. |
| `indicator_lag_cross_lags` | tuple[int,...] | `(1, 3)` | Set di lag per la famiglia di feature cross-time indicatore × base OHLC (issue #165, es. `close_sma_12[t] > low[t-3]`), ristretta agli indicatori price-scale (SMA/EMA/WMA/HMA) contro una base OHLC grezza. Passare `()` disabilita interamente questa famiglia sempre attiva. |
| `retain_raw_events` | bool | `True` | Se `EventDiscovery.raw_events` (l'intera popolazione di candidati pre-gate, ognuno con la propria serie di attivazione a piena lunghezza) resta in memoria dopo `.run()` (issue #232). Tenere `True` per `TargetOptimizer`, che legge `.raw_events` direttamente; una config solo per `forge()` (che non lo legge mai) può impostare `False` per una riduzione misurata di 4.2x della memoria trattenuta. |

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import GateParams, EventWalkForwardConfig

ed = EventDiscovery(
    enriched,
    config=DiscoveryConfig(
        train_ratio=0.80,
        max_and_components=2,
        gate_params=GateParams(min_tpm=0.5, dispersion_margin=1.3, min_episodes=10),
        walk_forward=EventWalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        scale_free_overrides={"rsi_14": True},  # forza scale-free su RSI
        diversity_gate_enabled=True,            # deduplicazione Jaccard opt-in
        diversity_threshold=0.85,
        retain_raw_events=False,                # config solo forge(): risparmia la memoria
    ),
)
candidates = ed.run()
```

---

## Modulo 2

### PromotionThresholds

Soglie statistiche IS che contribuiscono al grade A–D. Non bloccano la promozione (eccetto casi estremi): informano il grade. Viene passato come campo `thresholds` di `AlphaConfig`.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `ic_min_abs` | float | `0.02` | IC (Spearman) minimo in valore assoluto. Sotto questa soglia la metrica IC viene registrata in `AlphaContract.diagnostics` (non bloccante). |
| `ic_max_p` | float | *(risolto: `ctx.alpha` = 0.05)* | P-value massimo per l'IC. Uno dei cinque alpha per-ipotesi, risolto a livello di sessione (#182). Alimenta una diagnostica non bloccante che pesa solo sul grade. |
| `min_lift` | float | `0.08` | Lift minimo (win_rate − base_rate). |
| `min_cohens_d` | float | `0.15` | Cohen's d minimo (separazione tra distribuzioni active/inactive). |
| `max_p_value` | float | *(risolto: `ctx.alpha` = 0.05)* | P-value massimo del t-test sul vantaggio medio. Risolto a livello di sessione (#182), ma **raggiungibile solo con `use_fdr=False`** — ogni preset e il default di classe impostano `use_fdr=True`, quindi sotto qualunque preset è inerte. |
| `use_fdr` | bool | `True` | Applica correzione FDR Benjamini-Hochberg sulla famiglia di test. |
| `fdr_q` | float | `0.10` | Livello FDR (q) target. **Non** legato a `ctx.alpha`: un `q` è un tasso di falsa scoperta su una famiglia, un alpha è un tasso di errore per singolo test. Lo sceglie il preset, perché il `q` giusto dipende da quanto è ampia la ricerca (#182). |
| `oos_max_p` | float | `0.10` | P-value massimo per la conferma OOS. **Non** legato a `ctx.alpha`, e legittimamente più lasco: è un livello di *conferma* di un'ipotesi già selezionata — un test singolo pre-specificato, senza molteplicità, su un campione piccolo per costruzione (#182). |
| `min_direction_t` | float | `0.5` | `\|z_h*\|` minimo (excess standardizzato dalla rotazione) per assegnare una direzione; sotto → `undetermined`. |
| `require_significant_direction` | bool | `True` | Se True, la direzione è assegnata solo se `h*` supera Benjamini-Hochberg (non `statistically_weak`); altrimenti → `undetermined`. False = comportamento legacy non-bloccante. |

> `min_activations`/`min_oos_activations` **non esistono più** su questa
> dataclass — il controllo sulla numerosità del campione IS/OOS è ora una
> costante di modulo hardcoded, `_MIN_STATS_CASES = 10`, in
> `alpha_discovery/discovery.py` (una diagnostica non bloccante, non un
> campo configurabile). Costruire `PromotionThresholds` con uno di questi
> nomi solleva oggi un `TypeError`.

```python
from forgedge import AlphaConfig, PromotionThresholds

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    thresholds=PromotionThresholds(
        ic_min_abs=0.03,
        min_lift=0.10,
        min_cohens_d=0.20,
        oos_max_p=0.05,
    ),
)
```

---

### AlphaConfig

Configurazione principale del Modulo 2. Controlla la grid degli orizzonti, la derivazione del target, la suddivisione IS/OOS e i metadati di tracciabilità.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `horizon_grid` | tuple[int,...] | *(risolto: la classe di orizzonti della sessione)* | Grid di orizzonti (in barre) scansionata per derivare `h*`. `(1,2,4,8,12,24)` su orarie e 4H, `(1,2,3,5,7,10)` su daily e più lente, `(1,2,5,10,20,50)` su sotto-orarie — calibrato per classe come `BacktestParams.target_h`, non convertito in wall-clock (#196). Prima veniva sostituito solo quando non si passava alcun `AlphaConfig`, quindi una config esplicita su candele daily scandagliava fino a 48 *giorni*. |
| `mfe_quantile` | float | `0.5` | Quantile della distribuzione MFE delle barre attive usato come `sell_pct` di base. |
| `mfe_floor` | float | `0.005` | Floor per `sell_pct`: il take-profit non può essere < 0.5% indipendentemente dal MFE. |
| `train_ratio` | float | `0.7` | Frazione IS per la misurazione statistica. Il restante `1 - train_ratio` è l'OOS tail. |
| `embargo_bars` | int | `0` | Quarantena extra dopo lo split IS/OOS: la conferma OOS inizia `embargo_bars` barre dopo lo split. Default `0` — il purge rimuove già la sovrapposizione meccanica della finestra forward; l'embargo protegge in più dalla correlazione seriale. |
| `horizon_enrichment` | tuple[float,...] \| None | `(0.5, 1.0, 2.0)` | Arricchimento della grid di orizzonti per-evento dalla scala temporale strutturale dell'evento stesso: per ogni candidato, `round(m · w)` per ogni moltiplicatore `m` (dove `w` è `EventCandidate.dominant_window()`) viene **aggiunto** all'`horizon_grid` base (unione, mai una restrizione), limitato da `horizon_enrichment_min_obs`. `None`/`()` disabilita l'arricchimento. |
| `horizon_enrichment_min_obs` | int | `20` | Limite statistico per gli orizzonti arricchiti: un `h` aggiunto deve lasciare almeno questo numero di finestre forward non sovrapposte nello span IS (`h <= split // min_obs`). Non restringe mai l'`horizon_grid` base. |
| `thresholds` | PromotionThresholds | `PromotionThresholds()` | Soglie IS per le metriche statistiche. |
| `asset` | str | `"ASSET"` | Nome dell'asset (tracciabilità negli AlphaContract). |
| `exchange` | str | `""` | Exchange/mercato (opzionale, tracciabilità). |
| `timeframe` | str | `"1H"` | Timeframe (tracciabilità). |
| `fee_per_side` | float | `0.002` *(risolto dalla sessione)* | Commissione per lato (0.2%), registrata nel contratto **e addebitata dal backtest**: ora si propaga in `BacktestParams.fee` invece di essere una copia indipendente. |
| `close_col` | str | `"close"` *(risolto dalla sessione)* | Colonna del prezzo di chiusura. Si propaga a `BacktestParams.{target_col, buy_price_anchor}`. |
| `timestamp_col` | str | `"open_dt"` *(risolto dalla sessione)* | Colonna datetime. |
| `regime_col` | str | `"regime"` *(risolto dalla sessione)* | Colonna regime (da Modulo 0). |
| `regime_stable_col` | str | `"regime_stable"` *(risolto dalla sessione)* | Colonna regime_stable (da Modulo 0). |
| `use_stable_regime_only` | bool | `False` | Se True, esclude le barre con `regime_stable=False` dall'analisi dei regimi. |
| `min_regime_obs` | int | `10` | Osservazioni minime per regime per calcolare metriche per-regime attendibili. |
| `rolling_ic_window` | int \| None | `None` | Ampiezza della finestra per il rolling IC. Se None, calcolata automaticamente (≈ n/20). |
| `bars_per_day` | float \| None | *(risolto: da `timeframe`)* | Barre per giorno, dimensiona la finestra di rolling IC. Risolto a livello di sessione; senza sessione viene inferito dai timestamp. |
| `score_weights` | tuple[float,...] | `(0.20, 0.25, 0.15, 0.25, 0.15)` | Pesi del composite score (IC, lift, Cohen's d, z, regime breadth). Accetta anche la 4-tupla legacy (IC, lift, Cohen's d, breadth). |
| `statistically_weak_penalty` | float | `0.6` | Moltiplicatore del composite score quando il target è `statistically_weak`. `1.0` disabilita. |
| `oos_bonus` | float | `0.05` | Bonus additivo al composite score quando la conferma OOS passa. `0.0` disabilita. |
| `discovery_date` | str \| None | `None` | Data di scoperta (ISO, es. `"2026-01-15"`). Se None, usa la data corrente. |
| `fixed_target` | TargetConfig \| None | `None` | Se impostato, **salta la derivazione del target** e misura ogni candidato sul `(horizon, min_return, side)` dell'utente. L'orizzonte è aggiunto alla grid dei forward return se assente. |
| `fixed_target_diagnostic` | bool | `True` | Solo in fixed-target: esegue comunque la derivazione in read-only per popolare i diagnostici per-orizzonte e i campi di convergenza `data_derived_*`. `False` = bypass puro. |
| `target_mode` | `"abs"` \| `"proj"` | `"proj"` | Definizione del target binario per win rate / lift / base rate. `"proj"` (PROJ_LOG) misura il forward return long in **eccesso sul trend locale** (`log(fwd_max/close) − log(SMA_w/SMA_w[−h]) ≥ log(1+sell_pct)`): un long che cavalca il trend non viene accreditato del premio del trend — molto più stabile IS→OOS. `"abs"` = rendimento assoluto legacy. PROJ vale solo per long (short → abs); reverte ad abs se la storia è `< (trend_sma_mult+1)·h`. |
| `trend_sma_mult` | float | `2.0` | Solo PROJ_LOG: finestra SMA del trend = `round(trend_sma_mult·h)` barre. Relativa alle barre (auto-scala su ogni timeframe); più bassa segue l'orizzonte più da vicino, più alta leviga il trend. |

#### TargetConfig

Target economico specificato dall'utente per `AlphaConfig.fixed_target` e il workflow TargetOptimizer.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `horizon` | int | — | Holding period in barre (`> 0`). |
| `min_return` | float | — | Soglia take-profit come frazione (es. `0.02` = 2%), usata come `sell_pct` (`> 0`). |
| `side` | str | — | `"long"` o `"short"` — mai sovrascritto dai dati. |
| `min_activations` | int | `10` | TargetOptimizer: attivazioni minime per lo scoring del lift. I candidati che scattano su meno barre vengono saltati (win rate condizionato troppo rumoroso). Ignorato dalla modalità fixed-target di Alpha Discovery. |
| `min_lift_atoms` | float | `1.0` | **1° passo** del TargetOptimizer (eventi atomici, pre-AND): soglia di prune sul lift condizionato. La proprietà "lossless" del pruning vale solo al default `1.0` — valori superiori sopprimono attivamente le composizioni AND con lift emergente. |
| `min_lift_result` | float | `1.0` | **2° passo** del TargetOptimizer: soglia di prune sul set di risultati finale (atomi sopravvissuti *e* composizioni). Alzarla accorcia la lista dei risultati senza toccare la discovery AND. |
| `min_lift` | float \| None | `None` | **Deprecato** — usare `min_lift_atoms`/`min_lift_result`. Se impostato, si applica a entrambi i passi (comportamento legacy a soglia unica) e solleva `DeprecationWarning`; un valore sopra `1.0` sopprime allora anche la discovery AND, dato che il campo legacy guidava anche il 1° passo. |
| `target_mode` | `"abs"` \| `"proj"` | `"proj"` | Definizione del target binario (vedi `AlphaConfig.target_mode`). |
| `trend_sma_mult` | float | `2.0` | Moltiplicatore finestra SMA del trend PROJ_LOG (vedi `AlphaConfig.trend_sma_mult`). |

```python
from forgedge import AlphaDiscovery, AlphaConfig, PromotionThresholds

ad = AlphaDiscovery(
    ed.df,
    candidates,
    AlphaConfig(
        asset="ADAUSDC",
        timeframe="4H",
        horizon_grid=(4, 8, 12, 24, 48, 72, 96),   # orizzonti in barre 4H
        train_ratio=0.75,
        fee_per_side=0.001,
        thresholds=PromotionThresholds(
            min_lift=0.08,
            min_cohens_d=0.15,
            oos_max_p=0.10,
        ),
    ),
)
contracts = ad.run()
```

---

## Modulo 3

### BacktestParams

Parametri dell'esecuzione di un singolo backtest: direzione, tipo di ordine, livelli di ingresso/uscita e fee. Raramente usato direttamente dall'utente — la grid viene costruita da `GridSpec` e `RuleDiscovery` deriva `direction`, `sell_pct` e `target_h` di partenza dall'`AlphaContract`.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `direction` | str | `"long"` | Direzione del trade: `"long"` o `"short"`. Di norma derivato dall'AlphaContract. |
| `buy_type` | str | `"limit"` | Tipo di ordine di ingresso. In v1.0 solo `"limit"`. |
| `buy_drop_pct` | float | `0.010` | Distanza percentuale sotto il close a cui si piazza il limit order (1%). |
| `buy_delay_bar` | int | *(risolto: 6h di ordine vivo)* | Barre in cui l'ordine limite resta vivo. 6 su 1H, 2 su 4H, 1 su 1D, 24 su 15m — un 6 fisso lasciava l'ordine appeso sei *giorni* su candele daily. |
| `buy_price_anchor` | str | `"close"` *(risolto dalla sessione)* | Colonna a cui si applica l'offset del limite: `buy_price = anchor × (1 ∓ buy_drop_pct)`. **Qualsiasi colonna numerica è ammessa**, anche un indicatore derivato — `buy_price_anchor="close_sma_3", buy_drop_pct=0.10` significa "un limite al 90% della SMA a 3 barre". Viene riempita da `close_col` perché rinominare la colonna prezzo deve portarsi dietro l'anchor *di default*; un anchor esplicito è un livello di riferimento a sé e **non** ridefinisce la colonna prezzo della sessione. |
| `sell_pct` | float | `0.040` | Take-profit come percentuale dal fill price (4%). |
| `target_h` | int | *(risolto: cima della classe di orizzonti della sessione)* | Orizzonte massimo in barre: se il TP non viene raggiunto entro questo numero di barre, si chiude al close. 24 su orarie, 10 su daily, 50 su sotto-orarie — calibrato per classe come `horizon_grid`, non convertito in wall-clock. Di norma viene seminato dall'`holding_period_h` del contratto prima che questo default si applichi. |
| `target_col` | str | `"close"` *(risolto dalla sessione)* | Colonna usata per verificare il raggiungimento dello stop a orizzonte. Deve nominare la stessa serie di `close_col`; un disaccordo viene segnalato. |
| `target_hit_col` | str | `"close"` | Colonna usata per verificare il raggiungimento del take-profit. |
| `fee` | float | `0.002` *(risolto dalla sessione)* | Commissione per lato (0.2%), derivata da `AlphaConfig.fee_per_side`. |
| `early_stopping` | bool | `True` | Se True, la grid search si interrompe quando il top-K è stabile (ottimizzazione). |

---

### ScoringParams

Soglie usate dalla funzione di scoring della grid per combinare il Profit Factor con la regolarità degli arrivi dei trade. `pf_score_tpm = profit_factor × c_norm`, dove `c_norm` è l'inverso dell'indice di dispersione (`μ/σ²`, troncato a 1): è scale-free, quindi una regola non viene penalizzata perché opera più spesso, ma solo perché opera a raffiche. "Abbastanza trade" è una domanda separata, presidiata da `criteria.min_tpm` e dal floor dinamico qui sotto.

Entrambi i campi sono **risolti a livello di sessione**: lasciandoli non impostati li riempie `resolve()`, e `config_report()` mostra il valore che verrà eseguito.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `pf_min_trades` | int | *(risolto: `15`)* | Floor assoluto del conteggio dinamico `max(pf_min_trades, n_months × pf_min_tpm)` che alimenta `pf_score`. |
| `pf_min_tpm` | float | *(risolto: `criteria.min_tpm`)* | Floor di frequenza dello stesso conteggio dinamico. Segue la frequenza richiesta dal gate — 0.8/mese su 1D, 76.8 su 15m — invece di un 2 fisso che non coincideva con il gate su nessun timeframe. |

---

### GridSpec

Definisce lo spazio di ricerca della grid search dei parametri operativi. I campi lasciati a `None` usano la grid di default predefinita da `RuleDiscovery`.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `buy_drop_pct` | list[float] \| None | `None` | Lista di valori `buy_drop_pct` da esplorare. Se None, la grid di default è usata (es. `[0.005, 0.010, 0.015, 0.020]`). |
| `sell_pct` | list[float] \| None | `None` | Lista di valori `sell_pct` da esplorare. |
| `target_h` | list[int] \| None | `None` | Lista di orizzonti in barre da esplorare. |
| `buy_delay_bar` | list[int] \| None | `None` | Lista di delay bar da esplorare. |

```python
from forgedge.rule_discovery.models import GridSpec

grid = GridSpec(
    buy_drop_pct=[0.005, 0.010, 0.015],
    sell_pct=[0.030, 0.040, 0.050, 0.060],
    target_h=[12, 24, 36],
    buy_delay_bar=[3, 6],
)
```

---

### RuleWalkForwardConfig (Modulo 3)

Configura la validazione walk-forward OOS del Modulo 3. Prima si chiamava `WalkForwardConfig`; il vecchio nome resta come alias, sia in `rule_discovery.models` sia al livello superiore (`forgedge.WalkForwardConfig` ha sempre risolto a questa classe).

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `n_splits` | int | `4` | Numero di finestre rolling train+test. |
| `train_span_months` | int \| None | `None` | Ampiezza della finestra di training in mesi. Se None, calcolato automaticamente. |
| `test_span_months` | int \| None | `None` | Ampiezza della finestra di test in mesi. Se None, calcolato automaticamente. |
| `min_train_months` | int | *derivato* *(risolto dalla sessione)* | Mesi minimi per la finestra di training — lo span su cui gira lo screen di early elimination. **Derivato da `criteria.min_tpm`** con margine di Poisson al 95 %, così la finestra può davvero fornire il floor di trade che sta per pretendere (#173): 20 mesi a `min_tpm=0.80`, non il vecchio 6 fisso. Il `floor / rate` ingenuo dà 12.5 e resta corto circa il 44 % delle volte. |
| `reoptimise` | bool | `True` | Se True, riottimizza i parametri su ogni finestra di training. Se False, usa la configurazione IS fissa. |
| `purge_bars` | int \| None | `None` | Ampiezza del purge, in barre, alla fine di ogni finestra di **train**: le entrate aperte nelle ultime `purge_bars` barre hanno le finestre di fill/uscita che ricadono nella finestra di test adiacente, quindi la selezione dei parametri verrebbe altrimenti valutata su prezzi di test. `None` (default) dimensiona il purge automaticamente dalla grid risolta (il `target_h` più grande più il ritardo di fill); `0` disabilita il purge (comportamento pre-`TimeBudget`). Deliberatamente **non** unificato con `TimeBudget.purge_bars` (F6, #180) — quello è l'orizzonte del forward return, questo è lo span di trade nel caso peggiore; attraversamenti diversi di confini diversi. |
| `embargo_bars` | int | `UNSET` *(risolto dalla sessione)* | Quarantena extra all'inizio di ogni finestra di **test**, in barre. Stessa policy di `AlphaConfig.embargo_bars` ("quante barre di correlazione seriale mettere in quarantena dopo un confine"), quindi è risolto a livello di sessione da esso — un valore esplicito qui vince comunque. |

---

### SelectionCriteria

Gate di promozione del Modulo 3. Definisce le condizioni per i verdetti `EDGE`, `PARTIAL-EDGE` e `NON-EDGE`.

Quattro di questi campi sono **preset-parametrizzati** (#207) invece di
default universali fissi — `forge_preset()` li imposta per profilo perché le
descrizioni dei preset stessi divergono esplicitamente su precisione-vs-volume:

| campo | default di classe | `sniper` | `balanced` | `sweep` | `burst` |
|---|---|---|---|---|---|
| `min_profit_factor` | `2.0` | `2.5` | `2.0` | `1.8` | `2.0` |
| `min_win_rate` | `0.55` | `0.60` | `0.55` | `0.50` | `0.55` |
| `min_pf_score_tpm` | `0.30` | `0.40` | `0.30` | `0.25` | `0.30` |
| `min_fill_rate_opt` | `0.80` | `0.80` | `0.80` | `0.70` | `0.80` |

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `min_profit_factor` | float | `2.0` | PF IS minimo per EDGE. Preset-parametrizzato — vedi tabella sopra. |
| `min_win_rate` | float | `0.55` | Win rate IS minimo per EDGE (55%). Preset-parametrizzato — vedi tabella sopra. |
| `min_tpm` | float | `2.0` *(risolto dalla sessione)* | Frequenza media minima (trades/mese) per EDGE. È anche l'unico gate sul numero di trade: la soglia minima di trade eseguiti è dinamica, `max(10, n_months × min_tpm)`, e scala con la lunghezza dell'IS (spec RD-04) invece di una soglia assoluta fissa. Risolto da `PipelineContext.target_rate_tpm × rate_retention` quando la frequenza di M1 è stata dichiarata; altrimenti resta il default documentato `2.0`. |
| `min_pf_score_tpm` | float | `0.30` | Score composito minimo PF×TPM per includere una configurazione nella selezione. Preset-parametrizzato — vedi tabella sopra. |
| `min_fill_rate` | float | `0.40` | Fill rate minimo del limit order: almeno il 40% degli eventi deve tradursi in un trade. **Inerte sotto il default `entry_mode="auto"`** (lo Stage 1 è un'entrata market, fill ≈ 100%) — significativo solo con `entry_mode="limit"`; il floor che conta davvero sotto `"auto"` è `min_fill_rate_opt`. |
| `min_sell_pct` | float | `0.005` *(risolto dalla sessione)* | Floor operativo sul take-profit seminato dal target derivato del contratto. Risolto da `AlphaConfig.mfe_floor`; era un `max(0.01, …)` cablato dentro `_seed_base_params`, quindi il vincolo che legava era quello che il chiamante non poteva configurare (F11). |
| `min_fill_rate_opt` | float | `0.80` | Prima condizione di adozione sotto `entry_mode="auto"`: il punto limite può essere pubblicato solo se fila ancora a ≥ questa soglia **fuori campione**, evitando il confound del fill-collasso. Preset-parametrizzato — vedi tabella sopra. |
| `min_net_gain_retention` | float | `0.5` *(risolto dalla sessione)* | Terza condizione di adozione: frazione del net gain OOS del punto market che il limite deve mantenere. Deliberatamente larga — è un backstop contro il caso di μ minuscolo e σ minuscolo, che lo Sharpe non vede perché è scale-free in μ. |
| `partial_min_profit_factor` | float | `1.5` | PF IS minimo per PARTIAL-EDGE (non raggiunge EDGE ma non è NON-EDGE). |
| `min_active_month_rate` | float | `0.80` | Frazione minima di mesi IS che devono contenere almeno un trade per un EDGE pieno: `active_months / n_months >= min_active_month_rate`. Basato su tasso (sostituisce i vecchi campi assoluti `max_zero_months_edge`/`max_zero_months_partial`, che non esistono più), quindi timeframe-agnostico: su dati 1H il tasso è naturalmente vicino a 1.0, su dati 1D un processo di Poisson con dispersione fino a `max_dispersion` produce 0.75–0.95, che il default accomoda correttamente. |
| `max_regime_dependency` | float | `0.30` | Dipendenza di regime massima ammessa: se > 30% dei trade è concentrato in un singolo regime, scatta come gate soft. |
| `min_dsr` | float | `1.0` | Deflated Sharpe Ratio minimo (corretto per il numero di configurazioni testate). Un DSR indefinito (il radicando dell'haircut di selezione è diventato negativo — selection bias troppo severo per essere credibile) blocca anch'esso un EDGE pieno. |
| `max_ttest_p` | float | *(risolto: `ctx.alpha` = 0.05)* | P-value massimo del t-test sul net gain medio. Risolto a livello di sessione (#182). È l'**unico gate per-ipotesi hard** della pipeline — produce `NON-EDGE` in `_decide`. Nessun preset lo ha mai toccato. |
| `max_rotation_p` | float | *(risolto: `ctx.alpha` = 0.05)* | P-value massimo della rotation-null a livello di ricerca (`AlphaContract.rotation_p`) per un EDGE pieno — prezza l'intera superficie di discovery, così una regola che ha solo vinto la lotteria del multiple-testing resta limitata a PARTIAL-EDGE. Risolto a livello di sessione (#182); inerte quando il contratto non porta l'annotazione della rotation-null. Un valore stretto sotto `"sweep"` è intenzionale: la permissività a monte di `"sweep"` (`fdr_q=0.25`) presuppone che questo gate filtri a valle, abbinato a `RotationConfig(k>=100)`. |
| `power_gate` | bool | `True` | §3.2 — verdetti power-aware: se True, un EDGE/PARTIAL-EDGE viene degradato a `INSUFFICIENT-DATA` quando l'evidenza OOS non può sostenerlo — nessun walk-forward è stato possibile, i trade OOS pooled sono sotto `min_oos_trades`, oppure l'expectancy minima rilevabile del campione OOS pooled supera l'expectancy IS dichiarata. I verdetti `NON-EDGE` non vengono mai salvati. |
| `min_oos_trades` | int | `10` | Trade minimi di test walk-forward pooled (su tutte le finestre di test) per un verdetto positivo affidabile sotto `power_gate`. Sotto → `INSUFFICIENT-DATA`. Mai applicato per singola finestra. |
| `early_elimination` | bool | `True` | Se True (default), scarta velocemente le configurazioni che non passano i fast screen IS (< 20 trade, PF < 1, fill rate insufficiente) senza eseguire il walk-forward. Se False, la pipeline completa è sempre eseguita (utile per diagnostica uniforme su regole NON-EDGE). |

> `max_zero_months_edge`/`max_zero_months_partial` **non esistono più** su
> questa dataclass — sostituiti dal campo basato su tasso
> `min_active_month_rate` sopra. Costruire `SelectionCriteria` con uno di
> questi nomi solleva oggi un `TypeError`.

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig, SelectionCriteria

config = RuleDiscoveryConfig(
    criteria=SelectionCriteria(
        min_profit_factor=1.8,   # meno severo per asset ad alta volatilità
        min_win_rate=0.50,
        early_elimination=False, # diagnostica completa anche su NON-EDGE
    ),
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
```

---

### RuleDiscoveryConfig

Configurazione principale del Modulo 3. Aggrega tutti i sotto-configuratori: parametri base del backtest, scoring, grid di ricerca, walk-forward e criteri di selezione.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `base_params` | BacktestParams | `BacktestParams()` | Parametri base del backtest (punto di partenza per la grid search). |
| `scoring` | ScoringParams | `ScoringParams()` | Pesi dello scoring della grid. |
| `grid` | GridSpec | `GridSpec()` | Spazio di ricerca della grid. Se tutti i campi sono None, si usa la grid di default. |
| `walk_forward` | RuleWalkForwardConfig | `RuleWalkForwardConfig()` | Configurazione walk-forward OOS. |
| `criteria` | SelectionCriteria | `SelectionCriteria()` | Criteri EDGE/PARTIAL-EDGE/NON-EDGE. |
| `entry_mode` | str | `"auto"` | Modalità di valutazione dell'ingresso: `"auto"` (default — lo Stage 1 a entrata market decide il verdetto, lo Stage 2 sweepa `buy_drop_pct`, replaya il vincitore fuori campione e lo pubblica solo se supera tutte e tre le condizioni di adozione), `"market"` (solo la baseline al next-open, fill ≈ 100%, nessun ottimizzatore) o `"limit"` (il default pre-#185: la griglia ottimizza `buy_drop_pct`, quindi l'ingresso fa anche da ottimizzatore del prezzo di ingresso). |
| `use_contract_target` | bool | `True` | Se True, usa `direction`, `sell_pct` e `target_h` dall'AlphaContract come punto di partenza per la grid. |
| `timestamp_col` | str | `"open_dt"` | Colonna datetime. |
| `signal_col` | str | `"__rule_signal__"` | Colonna interna temporanea per il segnale. |
| `discovery_date` | str \| None | `None` | Data di scoperta (ISO). |
| `selection_mode` | `"walk_forward"` \| `"full_sample"` | `"walk_forward"` | Dove viene *selezionato* il punto operativo pubblicato (§3.4). `"walk_forward"` (default): il punto operativo viene dalle sole finestre di train del walk-forward (secondo `wf_param_policy`); nessuna metrica che alimenta il verdetto o la `ValidatedRule` pubblicata legge mai la finestra di test finale. `"full_sample"` ricade sulla selezione pre-#217 sull'intero span IS. |
| `wf_param_policy` | `"last"` \| `"consensus"` | `"last"` | Come `selection_mode="walk_forward"` sceglie il punto operativo pubblicato dalle selezioni di train per-split: `"last"` (default) — il vincitore della finestra di train più recente (ciò che si traderebbe subito dopo); `"consensus"` — il set di parametri più frequente tra gli split, pareggi risolti verso il più recente. |
| `n_trials_upstream` | int | `1` | Moltiplicatore incorporato nell'`n_trials` del Deflated Sharpe oltre al conteggio delle celle della grid, per chi vuole che l'haircut analitico includa un fattore di ricerca a monte esplicito (es. il numero di contratti fratelli che ricevono un verdetto). Default `1` = solo celle della grid (comportamento storico). |

```python
from forgedge import (
    RuleDiscovery, RuleDiscoveryConfig, BacktestParams,
    SelectionCriteria, RuleWalkForwardConfig,
)
from forgedge.rule_discovery.models import GridSpec, ScoringParams

config = RuleDiscoveryConfig(
    base_params=BacktestParams(fee=0.001),
    grid=GridSpec(
        buy_drop_pct=[0.005, 0.010, 0.015, 0.020],
        sell_pct=[0.030, 0.040, 0.050, 0.060],
        target_h=[12, 24, 36, 48],
        buy_delay_bar=[3, 6],
    ),
    walk_forward=RuleWalkForwardConfig(n_splits=5, min_train_months=8),
    criteria=SelectionCriteria(min_profit_factor=2.0, min_win_rate=0.55),
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
```

---

## Modulo 4

### RegistryConfig

Configurazione del Modulo 4. Controlla deduplicazione, classificazione genericity e opzioni di export.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `overlap_threshold` | float | `0.70` | Soglia Jaccard sopra cui due regole sono considerate duplicate (≥ 70% di sovrapposizione nelle date di attivazione). |
| `gain_corr_threshold` | float | `0.70` | Soglia Spearman sopra cui due regole hanno gain correlati. Usata come metrica secondaria nella matrice di correlazione. |
| `cross_pf_threshold` | float | `1.5` *(risolto dalla sessione)* | Floor assoluto di PF su un ticker esterno — metà del criterio di PASS. Derivato da `SelectionCriteria.partial_min_profit_factor`: l'asticella che ha ammesso la regola in casa. Era un `2.0` indipendente, che escludeva per costruzione ogni regola PARTIAL-EDGE dalla genericità. |
| `min_cross_pf_retention` | float | `0.8` *(risolto dalla sessione)* | L'altra metà: frazione del PF **di casa** che la regola deve mantenere sul ticker esterno. `PASS ⟺ pf ≥ cross_pf_threshold AND pf ≥ retention × pf_casa`. |
| `generic_ratio_threshold` | float | `2/3 ≈ 0.667` | Frazione minima di ticker esterni PASS per classificare la regola come GENERIC. PARTIAL se ≥ 1 PASS ma < 2/3. Il valore è esattamente 2/3: su 3 ticker esterni, 2 PASS → GENERIC, 1 PASS → PARTIAL. |
| `cross_min_active` | int | `10` | Attivazioni minime su un ticker esterno per includerlo nel conteggio cross-ticker. |
| `export_format` | str | `"excel"` | Formato di export della tabella piatta: `"excel"` o `"csv"`. |
| `export_duplicates` | bool | `True` | Se True, include le regole duplicate nella tabella esportata (con flag `is_duplicate=True`). |
| `export_non_generic` | bool | `True` | Se True, include SPECIFIC e ISOLATED nella tabella esportata. |
| `html_include_tradelog` | bool | `True` | Se True, include il trade log per-regola nel report HTML. |
| `html_charts` | bool | `True` | Se True, genera SVG inline (equity curve, heatmap) nel report HTML. |
| `timestamp_col` | str | `"open_dt"` | Colonna datetime nei frame forniti. |
| `session_date` | str \| None | `None` | Data di sessione (ISO). Se None, usa la data corrente. |

```python
from forgedge import RuleRegistry, RegistryConfig

config = RegistryConfig(
    overlap_threshold=0.65,         # deduplication più aggressiva
    cross_pf_threshold=1.8,         # alza il floor assoluto per asset illiquidi
    min_cross_pf_retention=0.7,     # e tollera un po' più di decadimento fuori casa
    generic_ratio_threshold=0.5,    # GENERIC se ≥ 50% ticker PASS
    export_format="csv",
    html_charts=True,
    export_duplicates=False,        # escludi i duplicati dall'export
)
registry = RuleRegistry.from_forge_results(results, config=config).run()
```

---

## Preset — `forge_preset()`

I default documentati sopra (`GateParams`, `SelectionCriteria`, e diversi
campi di `AlphaConfig`/`PromotionThresholds`) vengono regolarmente sovrascritti
in blocco da quattro preset nominati, scelti per **profilo di ricerca**, non
per asset:

| Preset | Profilo |
|---|---|
| `"sniper"` | Eventi rari/regolari, alta precisione. Richiede una finestra IS lunga. Non abbinare al rotation calibrator. |
| `"balanced"` | Default sensato — frequenza moderata, buon equilibrio IS/OOS. |
| `"sweep"` | Ricerca ampia/permissiva, molti candidati. Abbinare a `rotation_calibration=RotationConfig(k>=100)` e a un filtro `min_lift` su `promoted_contracts()`. |
| `"burst"` | Eventi concentrati nel tempo (momentum, regime-change) — dispersione alta tollerata di proposito. |

```python
from forgedge import forge, forge_preset

disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="BTC")
result = forge(kpi, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
                rule_discovery_config=rd_cfg)
```

`forge_preset(preset, timeframe, asset="ASSET", train_ratio=0.70, **overrides)`
imposta i criteri di frequenza di M1/M2/M3 (e diversi campi dipendenti) in
modo coerente per il timeframe scelto, e restituisce la tripla pronta all'uso
`(DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig)`. Ogni parametro che
calcola può essere sovrascritto per nome via `**overrides`:

| Modulo | Chiavi di override accettate |
|---|---|
| M1 (Event Discovery) | `min_tpm`, `max_dispersion`, `dispersion_margin`, `min_episodes`, `max_and_components`, `timestamp_col`, `event_counting` |
| M2 (Alpha Discovery) | `min_lift`, `min_cohens_d`, `fdr_q`, `oos_max_p`, `horizon_grid`, `bars_per_day` |
| M3 (Rule Discovery) | `rd_min_tpm`, `min_profit_factor`, `min_win_rate`, `min_pf_score_tpm`, `min_fill_rate_opt` |

`forgedge.presets.preset_info()` stampa i parametri numerici risolti per uno
o tutti i preset, utile per verificare a cosa risolve davvero un preset su un
dato timeframe prima di lanciarlo. Vedi la lista dei pitfall della skill
`forgedge` per due insidie reali e timeframe-specifiche: i preset su `"1D"`
possono far scattare `oos_span_too_short` in `config_report()` su una storia
modesta (il controllo che fa il suo lavoro, non un preset rotto), e
`"sniper"` non va combinato con il rotation calibrator.

---

## Riepilogo delle importazioni

| Classe | Import | Modulo |
|---|---|---|
| `EMAProxyConfig` | `from forgedge import EMAProxyConfig` | 0 — Market Context |
| `MarketContextConfig` | `from forgedge import MarketContextConfig` | 0 — Market Context |
| `GateParams` | `from forgedge.event_discovery.models import GateParams` | 1 — Event Discovery |
| `EventWalkForwardConfig` | `from forgedge import EventWalkForwardConfig` | 1 — Event Discovery |
| `DiscoveryConfig` | `from forgedge import DiscoveryConfig` | 1 — Event Discovery |
| `PromotionThresholds` | `from forgedge import PromotionThresholds` | 2 — Alpha Discovery |
| `AlphaConfig` | `from forgedge import AlphaConfig` | 2 — Alpha Discovery |
| `BacktestParams` | `from forgedge import BacktestParams` | 3 — Rule Discovery |
| `ScoringParams` | `from forgedge.rule_discovery.models import ScoringParams` | 3 — Rule Discovery |
| `GridSpec` | `from forgedge.rule_discovery.models import GridSpec` | 3 — Rule Discovery |
| `RuleWalkForwardConfig` | `from forgedge import RuleWalkForwardConfig` | 3 — Rule Discovery |
| `SelectionCriteria` | `from forgedge import SelectionCriteria` | 3 — Rule Discovery |
| `RuleDiscoveryConfig` | `from forgedge import RuleDiscoveryConfig` | 3 — Rule Discovery |
| `RegistryConfig` | `from forgedge import RegistryConfig` | 4 — Rule Registry |

> **Nota:** le due config walk-forward hanno ora nomi espliciti — `EventWalkForwardConfig` (Modulo 1) e `RuleWalkForwardConfig` (Modulo 3), entrambe importabili da `forgedge`. L'alias `WalkForwardConfig` continua a funzionare: al livello superiore risolve alla classe del Modulo 3, dentro `event_discovery.models` / `rule_discovery.models` a quella del modulo corrispondente.

---

## Risoluzione dei parametri — `UNSET` e il resolver

I campi di configurazione hanno tre stati, non due:

| stato | significato | chi lo scrive |
|---|---|---|
| esplicito | il chiamante ha scelto questo valore | mai sovrascritto; è un *input* alla derivazione |
| `UNSET` | "decide il resolver" | derivato dai vincoli che lo legano a ciò che è stato impostato |
| nessun vincolo | né il preset né la sessione hanno un'opinione | il default di classe documentato, esattamente come prima |

`UNSET` esiste perché un default normale non può rispondere alla domanda che un
resolver deve porsi: `AlphaConfig.timestamp_col == "open_dt"` è identico sia che
l'utente l'abbia scritto sia che l'abbia ereditato.

```python
from forgedge import UNSET, PipelineContext, collect_context, resolve

bundle = {"event_discovery": disc, "alpha": alpha, "rule_discovery": rd}
ctx = collect_context(bundle, PipelineContext.from_frame(kpi, timeframe="1D"))
resolved, trace, violations = resolve(bundle, ctx)
print(trace.to_text())
```

`resolve()` restituisce **copie** — ispezionare una configurazione non è mai un
effetto collaterale su di essa — ed è idempotente. `forge()` lo fa una volta
all'avvio ed espone il risultato su `ForgeResult.context` / `.resolution` /
`.coherence`.

La derivazione legge il timeframe, lo schema e i valori delle config; non legge
mai i dati (`n_bars`, `span_months`), visibili solo alla metà *check*. Questo
rende `resolve()` totale senza il frame — ed è ciò che permette a un'ispezione di
mostrare esattamente quello che girerà — e toglie la tentazione di limitare un
requisito alla storia disponibile invece di segnalare che non ci sta.

Precedenza, dalla più forte: un `PipelineContext` esplicito → argomenti di
`forge()` → campi impostati in una qualsiasi config → default di classe. Due
config in disaccordo sullo stesso valore vengono **segnalate**, mai riconciliate
in silenzio.

