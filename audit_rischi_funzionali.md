# Audit — Rischi Funzionali

> Codebase: `mattcond/forgedge` — branch `main`
> Data: 2026-06-15
> Scope: `market_context`, `event_discovery`, `alpha_discovery`, `rule_discovery`, `forge`

---

## Legenda

| Simbolo | Significato |
|---------|-------------|
| 🔴 ALTA | Compromette la validità statistica o l'interpretabilità dei risultati |
| 🟠 MEDIA | Introduce distorsioni sistematiche gestibili con accorgimenti |
| 🟡 BASSA | Limitazione di design che riduce l'efficacia ma non la correttezza |

I rischi funzionali sono scelte di design che non causano crash né dati corrotti, ma che producono risultati sistematicamente distorti, non rappresentativi, o difficilmente interpretabili. Il fix richiede decisioni di prodotto, non solo codice.

---

## RF-01 🔴 — `consistency_gate.py` — Gate non rileva run consecutivi, stagionalità né non-stazionarietà del rate

**File:** `src/forgedge/event_discovery/consistency_gate.py`

### Comportamento attuale

Il gate verifica 4 criteri sulla distribuzione mensile degli eventi:

1. `volume` — numero minimo di attivazioni totali (`min_act`)
2. `coverage` — numero minimo di mesi attivi (`min_months`)
3. `concentration` — nessun mese concentra più di `max_conc` delle attivazioni totali
4. `frequency` — media di attivazioni per mese (`min_tpm`)

### Problema

Tre aspetti rilevanti non vengono rilevati:

**a) Run consecutivi lunghi (segnale ridondante)**

Un evento che rimane attivo per 50 barre consecutive contribuisce 50 attivazioni ma contiene **un solo segnale informativo** (il crossing iniziale). Passa il gate con alta `volume` e buona `coverage`, ma il numero di segnali indipendenti è di fatto 1. La durata media del run non è misurata né limitata.

*Esempio critico:* Un evento threshold (es. `pctrank_rsi > 0.95`) in un trend forte può restare attivo per settimane. Con 8760 barre orarie e 3 run da 17 barre ciascuno, il gate vede 51 attivazioni su 3 mesi e promuove l'evento. Il segnale reale è solo "3 volte è entrato in zona overbought".

**b) Stagionalità annuale**

Un evento che si attiva ogni anno esclusivamente in Q4 (es. fine anno fiscale, rally natalizio) può passare il gate se il dataset copre 2+ anni:
- `n_active_months = 6` (3 mesi Q4 per 2 anni) ≥ `min_months = 8` — *fallisce* per `min_months` default, ma se abbassato per fold corti, passa

Non esiste alcun test di uniformità cross-semestre o cross-anno.

**c) Non-stazionarietà del rate**

Un evento attivo nei primi 12 mesi e poi sparito può passare il gate se il totale di attivazioni supera `min_act`. Il rate nel secondo periodo è 0, ma la media complessiva rimane sopra `min_tpm`. Nessun test di Chow o confronto first-half / second-half del rate.

### Impatto

- Segnali con scarso numero di osservazioni indipendenti vengono promossi ad AlphaDiscovery e RuleDiscovery inflazionando le metriche IS
- Segnali stagionali appaiono robustamente distribuiti se il dataset è sufficientemente lungo
- Segnali regime-dipendenti che cambiano stato non vengono rilevati come instabili

### Fix proposto

Aggiungere tre criteri opzionali al `GateParams`:

```python
# 5. Lunghezza massima del run medio (in barre)
max_avg_run_length: float = 10.0

# 6. Test di stazionarietà del rate: il tasso nel secondo semestre
#    deve essere >= fraction del tasso nel primo semestre
min_second_half_rate_ratio: float = 0.25

# 7. Massima concentrazione in un singolo trimestre (Q4 protection)
max_quarterly_conc: float = 0.60
```

Calcolare il run medio: `run_lengths = [lunghezza di ogni run consecutivo True]`, `avg_run = mean(run_lengths)`.

---

## RF-02 🔴 — `ema_proxy.py` — Look-ahead bias nel mode default `balanced` + `global`

**File:** `src/forgedge/market_context/ema_proxy.py`
**Righe:** 191–209

### Comportamento attuale

Con `threshold_mode="balanced"` e `threshold_basis="global"` (il **default**), le soglie sono calcolate sui quantili dell'**intera** serie storica, incluse le barre future rispetto a ogni bar classificata. Il commento lo riconosce:

```python
# not causal: a bar's label depends on the full sample, including future bars
```

### Problema

Ogni bar viene classificata con informazioni future. La colonna `regime` risultante:

1. **Non è riproducibile in live trading**: classificare una nuova barra richiede di rivedere tutti i quantili storici.
2. **Introduce look-ahead implicito** in tutti i moduli a valle che usano `regime`:
   - `EventDiscovery`: il gate di coverage per regime usa la colonna `regime` non-causal
   - `AlphaDiscovery`: `_measure_regimes()` calcola IC per regime usando la colonna non-causal
   - `RuleDiscovery`: `_regime_breakdown()` usa la colonna non-causal

3. **Il default non-causal è pericoloso**: un utente che usa `EventDiscovery` pensando di fare una discovery causale ottiene in realtà label di regime contaminati da informazioni future su ogni bar.

### Impatto

Risultati in-sample sistematicamente più favorevoli nei regimi "correttamente" classificati con look-ahead. La validazione OOS testa segnali su regime classificati causalmente (expanding mode) se il MarketContext viene ri-applicato, ma se si riusa la colonna IS non-causal anche per l'OOS, il bias persiste.

### Fix proposto

Due opzioni (scegliere una):

**Opzione A — Cambiare il default a `expanding`:**

```python
# models.py
threshold_basis: str = "expanding"   # era "global"
```

L'expanding mode è causale: ogni bar usa solo i quantili della storia fino a quel momento. Richiede `threshold_warmup` (default 200 barre) durante il quale si usano soglie fisse.

**Opzione B — Mantenere il default ma aggiungere un warning:**

```python
# ema_proxy.py — in fit() o classify()
if self.threshold_mode == "balanced" and self.threshold_basis == "global":
    import warnings
    warnings.warn(
        "EMAProxyClassifier: threshold_basis='global' uses future data for label assignment. "
        "Use threshold_basis='expanding' for causal (live-safe) classification.",
        UserWarning, stacklevel=2
    )
```

---

## RF-03 🔴 — `alpha_discovery/discovery.py` — Circolarità parziale nel validation OOS

**File:** `src/forgedge/alpha_discovery/discovery.py`
**Righe:** 327–340, 660–718

### Comportamento attuale

Il target `(h*, sell_pct*, direction*)` è derivato interamente sull'IS (70% dei dati). La validazione OOS poi testa se il segnale ha vantaggio sul restante 30% **usando lo stesso target IS**.

### Problema

Non è una circolarità formale (i dati OOS non influenzano la derivazione IS), ma introduce tre distorsioni:

1. **`h*` sovra-adattato al regime IS**: l'orizzonte ottimale IS può non essere ottimale OOS. Se il regime OOS è più lento, `h*` breve può essere troppo breve; se è più veloce, troppo lungo.

2. **`sell_pct*` sovra-adattato al regime IS**: `sell_pct` è la mediana delle MFE IS positive. In un regime OOS più volatile, le MFE sono più grandi → base rate OOS più alto → OOS sembra "più facile" del dovuto. In un regime meno volatile, MFE più piccole → base rate OOS più basso → OOS sembra "più difficile", rigettando segnali validi.

3. **OOS statisticamente insufficiente su dataset corti**: con 6 mesi IS (1H data = ~4320 barre) e 30% OOS = ~1296 barre (~54 giorni), il numero atteso di attivazioni OOS per un evento con `min_act = 5` può essere < 2 per evento raro, rendendo il test OOS statisticamente privo di potere.

### Fix proposto

**Breve termine**: documentare esplicitamente nei diagnostics del contratto il numero di barre OOS e di attivazioni OOS attese, così l'utente può valutare l'affidabilità del test. Aggiungere a `EventStats.oos`:

```python
@dataclass
class OosStats:
    ...
    n_oos_bars: int          # numero di barre OOS totali
    n_expected_activations: float  # n_oos_bars * (n_is_activations / n_is_bars)
    statistical_power: str   # "low" se n_oos_activations < 10, "medium" < 30, "high" >= 30
```

**Lungo termine**: implementare un walk-forward per AlphaDiscovery (simile a RuleDiscovery) con re-derivazione del target per ogni fold.

---

## RF-04 🟠 — `alpha_discovery/discovery.py` — `sell_pct` sovrastimato per costruzione

**File:** `src/forgedge/alpha_discovery/discovery.py`
**Righe:** 369–395

### Comportamento attuale

```python
active_mask = active_is[:max_bars] & (mfe > 0) & np.isfinite(mfe)
active_mfe = mfe[active_mask]
...
sell_pct = float(np.quantile(active_mfe, q=cfg.mfe_quantile))  # default q=0.50
```

### Problema

La distribuzione da cui si estrae `sell_pct` è **già filtrata per MFE > 0**: vengono escluse tutte le barre attive dove il prezzo non si muove favorevolmente. Questo sistematicamente:

1. Sovrastima il `sell_pct` perché esclude il 50% peggiore delle attivazioni
2. Produce un base rate IS artificialmente alto (~50% delle attivazioni IS **favorevoli** raggiungono il target), mentre il base rate effettivo include anche le attivazioni con MFE ≤ 0 che non possono mai raggiungere il target
3. Per asset ad alta volatilità (crypto), il `sell_pct` può essere 3-5%, un target quasi irraggiungibile in OOS in regimi diversi

Il `mfe_floor = 0.005` (50bp) protegge da target troppo piccoli ma non da target troppo grandi.

### Fix proposto

**Opzione A — Includere le MFE ≤ 0 come 0 nella distribuzione:**

```python
# Includere tutte le barre attive, non solo quelle con MFE > 0
all_mfe = mfe[:max_bars][active_is[:max_bars] & np.isfinite(mfe[:max_bars])]
# Sostituire MFE negative con 0 (il minimo target raggiungibile)
all_mfe_floored = np.maximum(all_mfe, 0.0)
if len(all_mfe_floored) < 2:
    return float("nan")
sell_pct = float(np.quantile(all_mfe_floored, q=cfg.mfe_quantile))
```

**Opzione B — Aggiungere un cap superiore (`mfe_cap`):**

```python
mfe_cap: float = 0.05   # 5% massimo sell_pct
sell_pct = min(float(np.quantile(active_mfe, q=cfg.mfe_quantile)), cfg.mfe_cap)
```

---

## RF-05 🟠 — `alpha_discovery/stats.py` — BH usa `m = candidati con p-value valido`, non totale candidati

**File:** `src/forgedge/alpha_discovery/stats.py`
**Righe:** 290–305

### Comportamento attuale

```python
valid_idx = np.where(np.isfinite(p))[0]
m = valid_idx.size  # solo i candidati con p-value finito
thresholds = (np.arange(1, m + 1) / m) * q
```

### Problema

Il BH standard usa `m = numero totale di test` (inclusi quelli con p-value `nan`). Usare solo i candidati con p-value finito abbassa `m`, alza le soglie BH e rende la procedura **più liberale** dello standard dichiarato:

- Con 100 candidati di cui 30 con `p = nan`: BH applicato come se ci fossero 70 test
- La FDR effettiva è `> q = 0.10` (più falsi positivi del 10% dichiarato)

La scelta è difendibile se i candidati `nan` sono "non testati" (direzione non determinabile), ma non è documentata come deviazione dal BH standard.

### Fix proposto

**Opzione A (più conservativa — BH standard):**

```python
m_total = len(p)  # tutti i test, inclusi i nan
thresholds = (np.arange(1, m_total + 1) / m_total) * q
# applicare le soglie solo ai valid_idx
```

**Opzione B (mantenere comportamento attuale ma documentare):**

Aggiungere un commento e un parametro esplicito:

```python
def benjamini_hochberg(p: np.ndarray, q: float = 0.10,
                       exclude_nan: bool = True) -> np.ndarray:
    """
    ...
    Parameters
    ----------
    exclude_nan : bool
        Se True (default), i candidati con p-value nan vengono esclusi dal
        conteggio m (procedura più liberale del BH standard ma adatta per
        la discovery dove i test non-eseguiti non sono ipotesi false).
        Se False, usa m = len(p) (BH standard).
    """
```

---

## RF-06 🟠 — `validation.py` — Win rate testato con t-test su variabile Bernoulli

**File:** `src/forgedge/rule_discovery/validation.py`
**Righe:** 145–164

### Comportamento attuale

```python
wins = (net > 0).astype(float)  # serie binaria 0/1
t_wr, p_wr = _ttest_1samp_greater(wins, base_rate)
```

### Problema

Il t-test assume la normalità del campione. Per una variabile Bernoulli (win = 0 o 1), la distribuzione campionaria della media converge alla normale solo per `n ≥ 30` (TCL). Con campioni piccoli (20–30 trade, il minimo ammesso da `validate()`), il p-value del t-test può essere significativamente inaccurato.

*Esempio:* Con `n = 25 trade` e `win_rate = 0.72 > base_rate = 0.50`, il p-value t-test può essere 0.03 mentre il p-value esatto binomiale è 0.07 — la differenza può cambiare il verdetto.

### Fix proposto

Sostituire il t-test con un test esatto di proporzione (binomiale):

```python
from scipy.stats import binomtest

def _binomial_greater(wins: np.ndarray, base_rate: float) -> tuple[float, float]:
    n = len(wins)
    k = int(wins.sum())
    result = binomtest(k, n, base_rate, alternative="greater")
    return float("nan"), result.pvalue  # il t-stat non ha senso per binomiale

# In validate():
_, p_wr = _binomial_greater(wins, base_rate)
```

Per compatibilità, mantenere la firma `(stat, pvalue)` e restituire `float("nan")` come statistic.

---

## RF-07 🟠 — `walkforward.py:53` — Dataset corto bypassa OOS completamente

**File:** `src/forgedge/rule_discovery/walkforward.py`
**Riga:** 53

### Comportamento attuale

```python
if total_months <= min_train:
    return []
```

Se il dataset copre ≤ `min_train_months = 6` mesi, `_build_splits` ritorna lista vuota, la WF è saltata, e `discovery.py:174` aggiunge la nota `"walk-forward skipped — data span too short"`. Il candidato può essere promosso a `EDGE` basandosi solo sull'IS.

### Problema

Un candidato promosso `EDGE` senza validazione OOS ha la stessa etichetta di uno validato OOS correttamente. L'utente non ha modo di distinguere i due casi a meno di leggere le `notes` del contratto.

### Fix proposto

**Opzione A — Aggiungere un flag esplicito al contratto:**

```python
# In RuleContract o BacktestSummary:
oos_validated: bool = False  # True solo se walk-forward ha prodotto almeno uno split
```

**Opzione B — Impedire la promozione a EDGE senza OOS:**

In `_decide()` di `rule_discovery/discovery.py`:

```python
if wf is None:
    # Dataset troppo corto: non può essere EDGE, solo PARTIAL-EDGE
    edge_block.append("walk-forward skipped: OOS validation not available")
```

Questo abbassa il massimo label raggiungibile da `EDGE` a `PARTIAL-EDGE` per dataset corti.

---

## RF-08 🟠 — `transform_layer.py` — `min_periods = window // 2` produce pctrank/zscore instabili durante il warmup

**File:** `src/forgedge/event_discovery/transform_layer.py`
**Righe:** 240–241, 264–265

### Comportamento attuale

```python
min_p = max(2, window // 2)
return series.rolling(window, min_periods=min_p).rank(pct=True)
```

Per `window = 168`, `min_periods = 84`: il pctrank è emesso già con 84 osservazioni (metà finestra).

### Problema

Durante il warmup parziale (barre 2–83):

- **pctrank** su 10–83 osservazioni ha alta varianza: il rank di un valore su 15 barre è molto meno stabile del rank su 168. In pratica le prime barre tendono a produrre pctrank più estremi (la distribuzione empirica di un campione piccolo è più uniforme di quella a regime).
- **zscore** con std calcolata su pochi punti è artificialmente grande per piccole deviazioni: su 10 barre con std = 0.001, uno zscore di 3 corrisponde a 3 tick, ben al di sopra di qualsiasi threshold ragionevole.

Queste attivazioni di warmup fanno sì che il `ConsistencyGate` riceva eventi che appaiono frequenti nelle prime barre, potenzialmente gonfiando `n_activations` se il dataset è corto o se il warmup copre una frazione significativa del dataset IS.

### Fix proposto

**Opzione A — Aumentare `min_periods` a `window` (warmup pieno):**

```python
min_p = window  # nessuna emissione parziale
```

Svantaggio: perde le prime `window - 1` barre del dataset IS, riducendo il dataset effettivo.

**Opzione B — Escludere il periodo di warmup dal gate (preferita):**

Nel `ConsistencyGate`, non contare le attivazioni nelle prime `max(windows)` barre del dataset:

```python
# Aggiungere a GateParams:
warmup_bars: int = 168  # barre iniziali da escludere dal conteggio

# In _count_by_month, azzerare le attivazioni prima del warmup:
active_vals[:warmup_bars] = False
```

---

## RF-09 🟠 — `feature_generator.py` — Feature ridondanti per costruzione non filtrate

**File:** `src/forgedge/event_discovery/feature_generator.py`
**Righe:** 326–352

### Comportamento attuale

Per ogni coppia `(col_a, col_b)` vengono generate **entrambe**:
- `ratio_colA_colB = colA / colB`
- `diffnorm_colA_colB = (colA - colB) / std(colA - colB)`

### Problema

Su coppie EMA (dove `colA` e `colB` sono dello stesso ordine di grandezza), il ratio e il diffnorm sono **monotonicamente correlati**:

```
ratio = colA / colB ≈ 1 + (colA - colB) / colB
```

Se `colB` è approssimativamente costante in un rolling window, `ratio - 1` e `diffnorm` sono quasi linearmente correlate. Dopo le transform pctrank/zscore, le loro distribuzioni sono quasi identiche. Producono EventCandidate quasi-identici che:

1. Passano entrambi il gate (il gate non filtra per correlazione tra feature)
2. Vengono composti in AND dando composizioni false-ridondanti
3. Gonflano il numero di candidati portati ad AlphaDiscovery

Stesso problema per `bb_pct_b_close_N` e `pos_close_range_N` (quando la finestra è la stessa): entrambi misurano la posizione del prezzo nel suo range recente.

### Fix proposto

Aggiungere un filtro di de-correlazione post-generazione in `FeatureGenerator.generate()`. Per ogni coppia di feature con correlazione > soglia (es. 0.95 sulla serie IS), mantenere solo quella con minor numero di NaN:

```python
# In feature_generator.py — dopo aver generato tutte le derived features
corr_matrix = extended[derived_cols].corr().abs()
to_drop = set()
for i, col_i in enumerate(derived_cols):
    for col_j in derived_cols[i+1:]:
        if corr_matrix.loc[col_i, col_j] > 0.95:
            # Drop quella con più NaN
            if extended[col_i].isna().sum() > extended[col_j].isna().sum():
                to_drop.add(col_i)
            else:
                to_drop.add(col_j)
extended = extended.drop(columns=list(to_drop))
```

---

## RF-10 🟠 — `backtest.py` — Fill model ottimistico: nessuno slippage, nessun gap risk

**File:** `src/forgedge/rule_discovery/backtest.py`
**Righe:** 228–241

### Comportamento attuale

```python
# Long limit: fill quando low <= buy_price
fill_probe = high if is_short else low
mask = (window >= bp) if is_short else (window <= bp)
```

Se il `low` della barra scende sotto il `buy_price` del limit order, il fill avviene sempre a `buy_price` esatto.

### Problema

1. **Nessun slippage**: in pratica, se il prezzo scende sotto il limit, il fill può avvenire sotto `buy_price` (gap down, illiquidità, priority queue). Il modello sovrastima il prezzo di entrata.

2. **Nessun gap risk post-fill**: se il prezzo scende drasticamente subito dopo il fill (momentum avverso), il modello non lo penalizza diversamente da un'entrata normale.

3. **Implicazione per la strategia**: i profitti netti stimati in IS sono sistematicamente più alti del reale, specialmente per `buy_drop_pct` piccolo (limit vicino al prezzo corrente) o per asset illiquidi.

### Fix proposto

**Opzione A — Slippage proporzionale:**

```python
# In config o RuleDiscovery:
slippage_pct: float = 0.001   # 10bp per-side come default

# In _scan_fill(), dopo aver trovato il fill:
buy_price = np.where(
    fill_rn >= 0,
    buy_price * (1.0 + slippage_pct) if not is_short else buy_price * (1.0 - slippage_pct),
    np.nan
)
```

**Opzione B — Fill al prezzo di apertura della barra successiva al tocco:**

Se il `low` tocca il limit alla barra `t`, il fill avviene all'`open` della barra `t+1` anziché a `buy_price`. Questo è più conservativo e più realistico per order flow models.

---

## RF-11 🟠 — `ema_proxy.py` — Discontinuità al warmup per `balanced` + `expanding`

**File:** `src/forgedge/market_context/ema_proxy.py`
**Righe:** 222–233

### Comportamento attuale

Durante il warmup (prime 200 barre), le soglie fisse `config.thresholds` vengono usate come fallback. Al bar 200, le soglie passano ai quantili espansi della storia corrente.

### Problema

Se il dataset inizia in un periodo anomalo (es. crash, bull run estremo), i quantili espansi post-warmup sono molto diversi dalle soglie fisse calibrate su dati "normali". Si crea una discontinuità visibile nella serie di regime: barre immediatamente prima e dopo il bar 200 possono avere label opposti pur avendo lo stesso valore del ratio EMA.

Questa discontinuità si traduce in un run reset nel `_rolling_stability` e in label `regime_stable = False` per un'altra finestra di stabilità dopo il bar 200.

### Fix proposto

Interpolare le soglie durante il warmup (blending da `thresholds_fixed` a `thresholds_expanding`):

```python
# Nelle ultime N barre del warmup, blend lineare
blend_window = 50   # barre di transizione
for t in range(warm - blend_window, warm):
    alpha = (t - (warm - blend_window)) / blend_window  # 0 → 1
    thresholds_t = (1 - alpha) * fixed_thresholds + alpha * expanding_thresholds[t]
```

---

## RF-12 🟡 — `ema_proxy.py` — Soglie fisse calibrate su crypto 1H non trasferibili ad altri asset

**File:** `src/forgedge/market_context/ema_proxy.py` / `models.py`

### Comportamento attuale

```python
thresholds: list[float] = field(
    default_factory=lambda: [0.975, 0.990, 1.010, 1.025]
)
```

Le soglie fisse `[0.975, 0.990, 1.010, 1.025]` sono calibrate su dati crypto 1H (ADAUSDC, DOGEUSDC) con volatilità storica tipica di 3-5% annualizzata su 1H. Su asset con volatilità diversa:

- Forex (EUR/USD 1H): volatilità ~6% annualizzata → soglie così strette classificano quasi tutto come `STRONG_BULL` o `STRONG_BEAR`
- Blue chip equity 1H: volatilità ~20% annualizzata → stesso problema

### Fix proposto

Derivare automaticamente le soglie in funzione della volatilità storica del ratio EMA:

```python
def _auto_thresholds(ratio: pd.Series, target_distribution: list[float]) -> list[float]:
    """Derivare le soglie dai quantili storici del ratio EMA."""
    clean = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    cum = np.cumsum(target_distribution)[:-1]
    return [float(clean.quantile(q)) for q in cum]
```

Questo è equivalente a `threshold_basis="global"` (con il look-ahead associato), ma rende esplicita la calibrazione per-asset.

---

## RF-13 🟡 — `and_composer.py` — Sparsità strutturale dell'AND per eventi rari

**File:** `src/forgedge/event_discovery/and_composer.py`

### Problema

Con due eventi a `activation_rate ≈ 5%` (P5 o P95 threshold), l'AND ha `rate ≈ 0.25%` assumendo indipendenza. Con 8760 barre (1 anno H1), si ottengono ~22 attivazioni — sotto `min_act = 50`.

Il gate AND richiede quindi che i due eventi siano **correlati** per passare (co-attivazione > indipendenza). Questo fa sì che:
- Le coppie AND promosse siano quasi sempre coppie di eventi altamente correlati (stessa feature, stesso regime)
- Le composizioni realmente ortogonali (es. volume + momentum) vengono scartate per sparsità
- Il catalogo AND è dominato da composizioni ridondanti che passano il gate per alta correlazione

### Fix proposto

**Opzione A — Ridurre `min_act` per le AND compositions:**

```python
gate_params_composed = GateParams(
    min_act=max(10, gate_params.min_act // 3),   # 1/3 del minimo singolo evento
    min_months=max(3, gate_params.min_months // 2),
    ...
)
```

**Opzione B — Sostituire AND bar-level con co-occorrenza in finestra:**

Invece di richiedere che i due eventi siano attivi **sulla stessa barra**, richiedere che si attivino entrambi dentro una finestra di `N` barre (es. 12H window per dati orari). Questo cattura segnali complementari che non sono simultanei ma sequenziali.

---

## RF-14 🟡 — `alpha_discovery/discovery.py` — `market_structure` calcolata su orizzonte mediano fisso, non su `h*`

**File:** `src/forgedge/alpha_discovery/discovery.py`
**Righe:** 143–148

### Comportamento attuale

```python
med_h = horizons[len(horizons) // 2]
self.market_structure = analyse_market_structure(
    close.iloc[:split], fwd[med_h].iloc[:split]
)
```

La struttura di mercato (Hurst, ACF) è calcolata **una volta** sull'orizzonte mediano della griglia e condivisa da tutti i contratti.

### Problema

L'ACF del return al `med_h` può non essere rappresentativa dell'ACF al `h*` derivato per ogni candidato. Un candidato con `h* = 2` (mean-reversion veloce) e uno con `h* = 48` (trend lento) ricevono la stessa `market_structure` calcolata su `med_h = 12` (se la griglia è `[1,2,3,6,12,24,48]`).

La classificazione `expected_family` (mean_reversion / momentum) derivata da `market_structure` può essere errata per candidati con `h*` molto diverso dalla mediana.

### Fix proposto

Calcolare `market_structure` per-candidato al momento di `_build_contract()`:

```python
# In _build_contract():
h_star = derived.h_star
ms = analyse_market_structure(
    close.iloc[:split],
    fwd[h_star].iloc[:split]
)
```

Il costo computazionale è accettabile: `analyse_market_structure` calcola Hurst (già calcolato una volta) e ACF (poco costosa). In alternativa, pre-calcolare per ogni `h` nel grid e cachare.

---

## RF-15 🟡 — `alpha_discovery/discovery.py` — `train_ratio=1.0` promuove senza OOS, senza warning

**File:** `src/forgedge/alpha_discovery/discovery.py`
**Righe:** 143, 660–665

### Comportamento attuale

`AlphaConfig` permette `train_ratio=1.0` (riga 261 di `models.py`). Con questo valore, `split = n`, `n_oos = 0`, e `_validate_oos()` ritorna `None`. Il contratto viene promosso senza alcuna validazione OOS. Nei diagnostics non appare nessun warning esplicito.

### Fix proposto

Aggiungere nei diagnostics del contratto un warning esplicito:

```python
if oos is None:
    contract_diagnostics.append(
        "WARNING: train_ratio=1.0 — no OOS validation performed. "
        "This contract has not been tested out-of-sample."
    )
```

E/o aggiungere un campo `oos_validated: bool` al contratto.

---

## RF-16 🟡 — `discovery.py (RuleDiscovery)` — `zero_months` gate valuta IS, non OOS

**File:** `src/forgedge/rule_discovery/discovery.py`
**Riga:** 399

### Comportamento attuale

```python
if s.zero_months > cr.max_zero_months_edge:
    edge_block.append(...)
```

`s = is_summary` — la summary in-sample. Il gate di `zero_months` valuta la distribuzione temporale IS, non OOS.

### Problema

Un alpha ottimamente distribuito IS (zero_months IS basso) ma concentrato in un regime specifico OOS (zero_months OOS alto) passa il gate di `zero_months`. La stagionalità OOS non viene misurata.

### Fix proposto

Applicare il check `zero_months` anche alla OOS summary (se disponibile):

```python
if wf is not None and wf.oos_summary is not None:
    oos_zero = wf.oos_summary.zero_months
    if oos_zero > cr.max_zero_months_edge:
        edge_block.append(f"OOS zero_months {oos_zero} > {cr.max_zero_months_edge}")
```

---

## RF-17 🟡 — `rule_discovery/grid.py` — Grid `sell_pct` può collassare a valore unico se il contratto ha `sell_pct` minimo

**File:** `src/forgedge/rule_discovery/grid.py`
**Righe:** 40–41

### Comportamento attuale

```python
sell = [round(max(0.005, s + k), 4) for k in (-0.02, -0.01, 0.0, 0.01, 0.02)]
```

Il floor `max(0.005, ...)` clampga tutti i valori di `sell_pct` inferiori a 50bp a 50bp.

### Problema

Se `s = sell_pct = 0.006` (vicinissimo al `mfe_floor`), il grid produce:
- `max(0.005, 0.006 - 0.02) = 0.005`
- `max(0.005, 0.006 - 0.01) = 0.005`
- `max(0.005, 0.006 + 0.00) = 0.006`
- `max(0.005, 0.006 + 0.01) = 0.016`
- `max(0.005, 0.006 + 0.02) = 0.026`

Risultato: 3 valori distinti invece di 5. Il grid è parzialmente degenere senza segnalazione. In `select_best()`, vengono testati meno punti del previsto.

### Fix proposto

Dopo aver generato il grid, de-duplicare i valori e loggare se il numero è inferiore al previsto:

```python
sell = sorted(set([round(max(0.005, s + k), 4) for k in (-0.02, -0.01, 0.0, 0.01, 0.02)]))
if len(sell) < 5:
    logger.debug(f"sell_pct grid degenerate: {len(sell)} distinct values (expected 5) for base sell_pct={s:.4f}")
```
