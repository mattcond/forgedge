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
| `bar_hours` | float \| None | `None` | Durata esplicita della candela in ore (es. `4.0` per 4H). Se None, derivato dal DatetimeIndex. |
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
| `stable_window` | int | `12` | Numero di barre consecutive identiche richieste per `regime_stable=True`. |

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

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `min_act` | int | `50` | Numero minimo di attivazioni IS. |
| `min_months` | int | `8` | Numero minimo di mesi distinti con almeno un'attivazione. |
| `max_conc` | float | `0.40` | Concentrazione massima ammessa in un singolo mese (frazione sul totale). |
| `min_tpm` | float | `2.0` | Frequenza media minima di attivazione (trades per mese). |

```python
from forgedge import DiscoveryConfig
from forgedge.event_discovery.models import GateParams

config = DiscoveryConfig(
    gate_params=GateParams(
        min_act=30,      # meno restrittivo per dataset più corti
        min_months=6,
        max_conc=0.50,
        min_tpm=1.5,
    )
)
```

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
| `timestamp_col` | str | `"open_dt"` | Colonna datetime nella KPI Table (o nome dell'indice DatetimeIndex). |
| `max_and_components` | int | `2` | Numero massimo di singoli eventi da combinare in un AND. Valori > 3 sono tecnicamente accettati ma sconsigliati (overfitting strutturale). |
| `train_ratio` | float | `1.0` | Frazione di barre IS (0 < train_ratio ≤ 1.0). Default 1.0 = tutto IS (nessun split). |
| `walk_forward` | EventWalkForwardConfig \| None | `None` | Configurazione walk-forward OOS. Attivo solo se anche `train_ratio < 1.0`. |
| `diversity_gate_enabled` | bool | `False` | Se True, applica una deduplicazione Jaccard degli eventi singoli dopo il ConsistencyGate e prima della composizione AND. Opt-in — nessun breaking change. |
| `diversity_threshold` | float | `0.85` | Similarità Jaccard massima tollerata tra due eventi conservati. Usato solo con `diversity_gate_enabled=True`. A p99 della distribuzione Jaccard inter-evento (12 mesi di dati 1H), Jaccard=0.47 — valori sopra 0.70 sono genuine near-duplicate. |

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import GateParams, EventWalkForwardConfig

ed = EventDiscovery(
    enriched,
    config=DiscoveryConfig(
        train_ratio=0.80,
        max_and_components=2,
        gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
        walk_forward=EventWalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        scale_free_overrides={"rsi_14": True},  # forza scale-free su RSI
        diversity_gate_enabled=True,            # deduplicazione Jaccard opt-in
        diversity_threshold=0.85,
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
| `ic_max_p` | float | `0.05` | P-value massimo per l'IC. |
| `min_lift` | float | `0.08` | Lift minimo (win_rate − base_rate). |
| `min_cohens_d` | float | `0.15` | Cohen's d minimo (separazione tra distribuzioni active/inactive). |
| `max_p_value` | float | `0.05` | P-value massimo del t-test sul vantaggio medio. |
| `min_activations` | int | `30` | Attivazioni IS minime perché il contratto sia promosso. |
| `use_fdr` | bool | `True` | Applica correzione FDR Benjamini-Hochberg sulla famiglia di test. |
| `fdr_q` | float | `0.10` | Livello FDR (q) target. |
| `oos_max_p` | float | `0.10` | P-value massimo per la conferma OOS. |
| `min_oos_activations` | int | `10` | Attivazioni OOS minime per considerare la conferma OOS attendibile. |
| `min_direction_t` | float | `0.5` | `\|z_h*\|` minimo (excess standardizzato dalla rotazione) per assegnare una direzione; sotto → `undetermined`. |
| `require_significant_direction` | bool | `True` | Se True, la direzione è assegnata solo se `h*` supera Benjamini-Hochberg (non `statistically_weak`); altrimenti → `undetermined`. False = comportamento legacy non-bloccante. |

```python
from forgedge import AlphaConfig, PromotionThresholds

config = AlphaConfig(
    asset="BTC",
    timeframe="1H",
    thresholds=PromotionThresholds(
        ic_min_abs=0.03,
        min_lift=0.10,
        min_cohens_d=0.20,
        min_activations=40,
        oos_max_p=0.05,
    ),
)
```

---

### AlphaConfig

Configurazione principale del Modulo 2. Controlla la grid degli orizzonti, la derivazione del target, la suddivisione IS/OOS e i metadati di tracciabilità.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `horizon_grid` | tuple[int,...] | `(1,2,3,4,6,8,12,16,24,36,48)` | Grid di orizzonti (in barre) scansionata per derivare `h*`. |
| `mfe_quantile` | float | `0.5` | Quantile della distribuzione MFE delle barre attive usato come `sell_pct` di base. |
| `mfe_floor` | float | `0.005` | Floor per `sell_pct`: il take-profit non può essere < 0.5% indipendentemente dal MFE. |
| `train_ratio` | float | `0.7` | Frazione IS per la misurazione statistica. Il restante `1 - train_ratio` è l'OOS tail. |
| `thresholds` | PromotionThresholds | `PromotionThresholds()` | Soglie IS per le metriche statistiche. |
| `asset` | str | `"ASSET"` | Nome dell'asset (tracciabilità negli AlphaContract). |
| `exchange` | str | `""` | Exchange/mercato (opzionale, tracciabilità). |
| `timeframe` | str | `"1H"` | Timeframe (tracciabilità). |
| `fee_per_side` | float | `0.002` | Commissione per lato (0.2%), registrata nel contratto per Rule Discovery. |
| `close_col` | str | `"close"` | Colonna del prezzo di chiusura. |
| `timestamp_col` | str | `"open_dt"` | Colonna datetime. |
| `regime_col` | str | `"regime"` | Colonna regime (da Modulo 0). |
| `regime_stable_col` | str | `"regime_stable"` | Colonna regime_stable (da Modulo 0). |
| `use_stable_regime_only` | bool | `False` | Se True, esclude le barre con `regime_stable=False` dall'analisi dei regimi. |
| `min_regime_obs` | int | `10` | Osservazioni minime per regime per calcolare metriche per-regime attendibili. |
| `rolling_ic_window` | int \| None | `None` | Ampiezza della finestra per il rolling IC. Se None, calcolata automaticamente (≈ n/20). |
| `bars_per_day` | float \| None | `None` | Barre per giorno per il calcolo del Deflated Sharpe. Se None, derivato dal timestamp. |
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
| `min_activations` | int | `10` | TargetOptimizer: attivazioni minime per lo scoring del lift. |
| `min_lift` | float | `1.0` | TargetOptimizer: soglia di prune sul lift condizionato. |
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
            min_activations=30,
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
| `buy_delay_bar` | int | `6` | Numero massimo di barre successive all'evento in cui il limit può essere eseguito. |
| `buy_price_anchor` | str | `"close"` | Colonna usata come anchor per il prezzo di ingresso. |
| `sell_pct` | float | `0.040` | Take-profit come percentuale dal fill price (4%). |
| `target_h` | int | `24` | Orizzonte massimo in barre: se il TP non viene raggiunto entro questo numero di barre, si chiude al close. |
| `target_col` | str | `"close"` | Colonna usata per verificare il raggiungimento dello stop a orizzonte. |
| `target_hit_col` | str | `"close"` | Colonna usata per verificare il raggiungimento del take-profit. |
| `fee` | float | `0.002` | Commissione per lato (0.2%). |
| `early_stopping` | bool | `True` | Se True, la grid search si interrompe quando il top-K è stabile (ottimizzazione). |

---

### ScoringParams

Pesi e soglie usati dalla funzione di scoring della grid (`pf_score_tpm`) per bilanciare Profit Factor e frequenza di trading.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `pf_min_trades` | int | `15` | Numero minimo di trade per includere una configurazione nel ranking. |
| `pf_min_tpm` | int | `2` | Frequenza minima (trades/mese) perché la frequenza contribuisca positivamente allo score. |
| `pf_tpm_target` | int | `3` | Frequenza target (trades/mese): raggiunto questo livello lo score di frequenza è massimo. |

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
| `min_train_months` | int | `6` | Mesi minimi richiesti per la finestra di training. |
| `reoptimise` | bool | `True` | Se True, riottimizza i parametri su ogni finestra di training. Se False, usa la configurazione IS fissa. |

---

### SelectionCriteria

Gate di promozione del Modulo 3. Definisce le condizioni per i verdetti `EDGE`, `PARTIAL-EDGE` e `NON-EDGE`.

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `min_profit_factor` | float | `2.0` | PF IS minimo per EDGE. |
| `min_win_rate` | float | `0.55` | Win rate IS minimo per EDGE (55%). |
| `min_tpm` | float | `2.0` | Frequenza media minima (trades/mese) per EDGE. È anche l'unico gate sul numero di trade: la soglia minima di trade eseguiti è dinamica, `max(10, n_months × min_tpm)`, e scala con la lunghezza dell'IS (spec RD-04) invece di una soglia assoluta fissa. |
| `min_pf_score_tpm` | float | `0.30` | Score composito minimo PF×TPM per includere una configurazione nella selezione. |
| `min_fill_rate` | float | `0.40` | Fill rate minimo del limit order: almeno il 40% degli eventi deve tradursi in un trade. |
| `min_fill_rate_opt` | float | `0.80` | Floor di fill per lo stadio di ottimizzazione limit di `entry_mode="auto"`: il limit può migliorare l'operating point solo se fila ancora a ≥ 80%, evitando il confound del fill-collasso. |
| `partial_min_profit_factor` | float | `1.5` | PF IS minimo per PARTIAL-EDGE (non raggiunge EDGE ma non è NON-EDGE). |
| `max_zero_months_edge` | int | `1` | Mesi a zero o negativi massimi ammessi per EDGE. |
| `max_zero_months_partial` | int | `4` | Mesi a zero o negativi massimi ammessi per PARTIAL-EDGE. |
| `max_regime_dependency` | float | `0.30` | Dipendenza di regime massima ammessa: se > 30% dei trade è concentrato in un singolo regime, scatta come gate soft. |
| `min_dsr` | float | `1.0` | Deflated Sharpe Ratio minimo (corretto per il numero di configurazioni testate). |
| `max_ttest_p` | float | `0.05` | P-value massimo del t-test sul net gain medio. |
| `early_elimination` | bool | `True` | Se True (default), scarta velocemente le configurazioni che non passano i fast screen IS (< 20 trade, PF < 1, fill rate insufficiente) senza eseguire il walk-forward. Se False, la pipeline completa è sempre eseguita (utile per diagnostica uniforme su regole NON-EDGE). |

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
| `entry_mode` | str | `"limit"` | Modalità di valutazione dell'ingresso: `"limit"` (default, retro-compatibile), `"market"` (baseline al next-open, fill ≈ 100%, nessun ottimizzatore) o `"auto"` (pipeline a due stadi: il **market** decide il verdetto, il **limit** ottimizza l'operating point dei soli sopravvissuti a fill ≥ `min_fill_rate_opt`). |
| `use_contract_target` | bool | `True` | Se True, usa `direction`, `sell_pct` e `target_h` dall'AlphaContract come punto di partenza per la grid. |
| `timestamp_col` | str | `"open_dt"` | Colonna datetime. |
| `signal_col` | str | `"__rule_signal__"` | Colonna interna temporanea per il segnale. |
| `discovery_date` | str \| None | `None` | Data di scoperta (ISO). |

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
    scoring=ScoringParams(pf_tpm_target=4),
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
| `cross_pf_threshold` | float | `2.0` | PF minimo su un ticker esterno per contare come PASS nel backtest cross-ticker. |
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
    cross_pf_threshold=1.8,         # meno severo per asset illiquidi
    generic_ratio_threshold=0.5,    # GENERIC se ≥ 50% ticker PASS
    export_format="csv",
    html_charts=True,
    export_duplicates=False,        # escludi i duplicati dall'export
)
registry = RuleRegistry.from_forge_results(results, config=config).run()
```

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

