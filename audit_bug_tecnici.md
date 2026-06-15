# Audit — Bug Tecnici

> Codebase: `mattcond/forgedge` — branch `main`
> Data: 2026-06-15
> Scope: `market_context`, `event_discovery`, `alpha_discovery`, `rule_discovery`, `forge`

---

## Legenda priorità

| Simbolo | Significato |
|---------|-------------|
| 🔴 ALTA | Causa dati errati o crash in produzione |
| 🟠 MEDIA | Produce risultati sbagliati in edge case reali |
| 🟡 BASSA | Viola un contratto interno o è fonte di confusione, ma non cambia l'output nel flusso normale |

---

## BUG-01 🔴 — `feature_generator.py` — `bb_pct_b` usa sempre `close` indipendentemente dal `base` della banda

**File:** `src/forgedge/event_discovery/feature_generator.py`
**Righe:** 461–481

### Codice attuale (difettoso)

```python
close_cols = [col for col, pf in parsed.items() if pf.family == "price" and pf.base == "close"]
close_col = close_cols[0] if close_cols else None

for key in set(bb_lower) & set(bb_upper):
    if close_col is None:
        break
    base, param = key
    lower_col = bb_lower[key]
    upper_col = bb_upper[key]
    new_col = f"bb_pct_b_{base}_{param:02d}"
    if new_col not in extended.columns:
        series = _safe_position(df[close_col], df[lower_col], df[upper_col])
```

### Problema

La variabile `close_col` è selezionata una sola volta prima del loop, e viene usata come numeratore per **tutte** le varianti di `bb_pct_b`, incluse quelle costruite su `high` o `low`. Di conseguenza:

- `bb_pct_b_high_20` = `(close - high_bb_lower_20) / (high_bb_upper_20 - high_bb_lower_20)` — **errato**
- La colonna dovrebbe misurare la posizione di `high` nelle bande di `high`, non di `close` nelle bande di `high`
- Quando le bande di `high` hanno un centro superiore al close (il caso tipico), il numeratore `close - high_bb_lower` è spesso negativo → valori sistematicamente fuori `[0, 1]`

### Propagazione a valle

**AlphaDiscovery** (`alpha_discovery/discovery.py:917–920`) legge la colonna direttamente da `ed.df`:

```python
name = comp.source_feature   # e.g. "bb_pct_b_high_20"
if name in self._frame.columns:
    return self._frame[name]  # ← restituisce il valore ERRATO
```

Tutte le metriche di AlphaDiscovery (IC, win rate, Cohen's d, regime analysis) vengono quindi calcolate sulla feature sbagliata. **RuleDiscovery non è affetto** perché usa `build_feature_series()` via replay, che legge `source_cols` dallo storage del componente (che contiene il base corretto).

### Fix richiesto

Nel loop, il numeratore deve essere la colonna del `base` corrispondente, non sempre `close_col`:

```python
for key in set(bb_lower) & set(bb_upper):
    base, param = key
    lower_col = bb_lower[key]
    upper_col = bb_upper[key]
    new_col = f"bb_pct_b_{base}_{param:02d}"
    if new_col not in extended.columns:
        # Trova la colonna del base corretto (es. "high", "low", "close")
        base_cols = [col for col, pf in parsed.items() if pf.family == "price" and pf.base == base]
        if not base_cols:
            continue
        base_col = base_cols[0]
        series = _safe_position(df[base_col], df[lower_col], df[upper_col])
        extended[new_col] = series
```

Se la colonna `{base}` non è presente nel frame, saltare silenziosamente (come già fa con `close_col is None`).

---

## BUG-02 🔴 — `and_composer.py` — `_components` come attributo dinamico non persistente

**File:** `src/forgedge/event_discovery/and_composer.py`
**Riga:** 578

**File impattato:** `src/forgedge/event_discovery/discovery.py`
**Riga:** 475

**File impattato:** `src/forgedge/alpha_discovery/discovery.py`
**Righe:** 917–920

### Codice attuale (difettoso)

```python
# and_composer.py:578
comp._components = components  # type: ignore[attr-defined]
```

```python
# event_discovery/discovery.py:475
comps = getattr(comp, "_components", [comp])
```

### Problema

`EventComponent` è un `@dataclass`. Impostare `_components` come attributo dinamico non incluso nella definizione del dataclass ha tre conseguenze:

1. **`dataclasses.fields(comp)` non include `_components`**: qualsiasi framework che itera i campi (pickle, JSON, deepcopy) perderà l'attributo.
2. **Dopo serializzazione/deserializzazione**, `getattr(comp, "_components", [comp])` ritorna il fallback `[comp]`, ovvero il componente `and_composition` stesso invece dei componenti reali. Il replay delle feature AND-composte produce risultati sbagliati silenziosamente.
3. **AlphaDiscovery** (`discovery.py:917`) usa `cand.components[0]` per derivare la feature continua per IC e win rate. Per eventi AND-composti, solo il primo componente viene analizzato, non la congiunzione.

### Propagazione a valle

| Modulo | Impatto |
|--------|---------|
| `event_discovery/discovery.py:475` | Lettura corretta solo in-session; fallisce dopo deserializzazione |
| `alpha_discovery/discovery.py:917` | IC e win rate calcolati solo sul primo componente dell'AND |
| `rule_discovery` | Non affetto: usa `event_series` pre-calcolata |

### Fix richiesto

Aggiungere il campo `components` direttamente al dataclass `EventComponent` in `models.py`:

```python
# models.py — EventComponent
@dataclass
class EventComponent:
    ...
    components: list["EventComponent"] = field(default_factory=list)
```

In `and_composer.py:_make_composed_event`, sostituire l'assegnazione dinamica:

```python
# PRIMA (difettoso):
comp._components = components

# DOPO (corretto):
comp.components = components
```

In `event_discovery/discovery.py:475`, sostituire:

```python
# PRIMA:
comps = getattr(comp, "_components", [comp])

# DOPO:
comps = comp.components if comp.components else [comp]
```

---

## BUG-03 🔴 — `consistency_gate.py` — `KeyError` su `NaT` in `_build_month_index`

**File:** `src/forgedge/event_discovery/consistency_gate.py`
**Righe:** 263–267

### Codice attuale (difettoso)

```python
periods = timestamps.dt.to_period("M")
unique_months = periods.sort_values().unique()
month_to_idx = {m: i for i, m in enumerate(unique_months)}
month_index = np.array([month_to_idx[p] for p in periods], dtype=np.int32)
return month_index, len(unique_months)
```

### Problema

Se `timestamps` contiene almeno un `NaT`, `timestamps.dt.to_period("M")` produce un `NaT` Period. `sort_values().unique()` include `NaT` come elemento. La list comprehension `[month_to_idx[p] for p in periods]` non crasha perché `NaT` è in `month_to_idx`, ma `np.array([...], dtype=np.int32)` converte `NaT` in un valore sentinel indefinito (tipicamente `-9223372036854775808` troncato a `int32` → valore out-of-bounds). Il successivo accesso `counts[month_index]` in `_count_by_month` può generare `IndexError` se l'indice sentinel supera `n_months`.

Il bug già segnalato in `_count_zero_months` di `discovery.py` (guardia NaT già presente) **non protegge** questo percorso interno al gate.

### Fix richiesto

Aggiungere una guardia NaT all'inizio di `_build_month_index` e filtrare i `NaT` dalla serie:

```python
def _build_month_index(timestamps: pd.Series) -> tuple[np.ndarray, int]:
    periods = timestamps.dt.to_period("M")
    # Rimuovi NaT prima di indicizzare
    valid_mask = ~periods.isna()
    valid_periods = periods[valid_mask]
    if valid_periods.empty:
        return np.array([], dtype=np.int32), 0
    unique_months = valid_periods.sort_values().unique()
    month_to_idx = {m: i for i, m in enumerate(unique_months)}
    # Assegna -1 ai NaT (verranno ignorati nel conteggio)
    month_index = np.where(
        valid_mask,
        np.array([month_to_idx.get(p, -1) for p in periods], dtype=np.int32),
        -1,
    )
    return month_index, len(unique_months)
```

In `_count_by_month`, aggiungere il filtro degli indici `-1`:

```python
mask = (month_index >= 0) & (month_index < n_months)
# usare solo month_index[mask] nei conteggi
```

---

## BUG-04 🔴 — `analysis.py:98` — `is_short = any()` invece di `all()`

**File:** `src/forgedge/rule_discovery/analysis.py`
**Riga:** 98

### Codice attuale (difettoso)

```python
is_short = "direction" in trades.columns and (trades["direction"] == "short").any()
```

### Problema

`any()` restituisce `True` se **almeno una** trade è short. Se per qualsiasi motivo (bug nel backtest engine o concatenazione di trade da contratti diversi) un frame misto contiene sia trade long che short, `is_short = True` forza la formula MAE/MFE short su **tutte** le trade, producendo:

- Per i long: MAE e MFE calcolati con il segno invertito
- I valori risultanti sono sistematicamente sbagliati senza nessun avviso

Il caso misto non si verifica nel flusso normale (ogni contratto ha direzione fissa), ma la funzione è pubblica e non ha protezione contrattuale.

### Propagazione a valle

`excursion_stats()` è chiamata in:
- `rule_discovery/discovery.py:181` — IS excursion
- `rule_discovery/walkforward.py:183` — OOS excursion (su `oos_concat`, concatenazione multi-split)

### Fix richiesto

Sostituire `.any()` con `.all()`:

```python
# PRIMA (difettoso):
is_short = "direction" in trades.columns and (trades["direction"] == "short").any()

# DOPO (corretto):
is_short = "direction" in trades.columns and (trades["direction"] == "short").all()
```

Aggiungere opzionalmente una guardia per frame misti:

```python
if "direction" in trades.columns:
    directions = trades["direction"].unique()
    if len(directions) > 1:
        raise ValueError(f"excursion_stats: mixed directions {directions}, expected single direction")
    is_short = directions[0] == "short"
else:
    is_short = False
```

---

## BUG-05 🟠 — `backtest.py:384-386` — `_month_index` esclude l'ultimo mese quando `ts_to=None`

**File:** `src/forgedge/rule_discovery/backtest.py`
**Righe:** 384–386

### Codice attuale (difettoso)

```python
n = max((end.year - start.year) * 12 + (end.month - start.month), 1)
return pd.period_range(start, periods=n, freq="M")
```

### Problema

Quando `ts_to=None`, `end = pd.Timestamp(dt.max()).to_period("M")`. La differenza `(end.year - start.year) * 12 + (end.month - start.month)` conta i mesi **escludendo** `end`. Esempio: da gennaio 2023 a dicembre 2023 → `n = 11`, il mese di dicembre è escluso.

La funzione analoga in `rule_discovery/discovery.py:355` usa correttamente `+ 1`:

```python
# discovery.py:355 — versione CORRETTA
n_months = max((end.year - start.year) * 12 + (end.month - start.month) + 1, 1)
```

### Impatto sulle metriche

Quando `ts_to=None` (ovvero in **tutta la grid search IS**):

| Metrica | Effetto |
|---------|---------|
| `n_months` | Undercounted di 1 |
| `zero_months` | Sovrastimato di 1 (l'ultimo mese appare "inattivo") |
| `tpm_mu` | Gonfiato: `n_trade / (n_months - 1)` anziché `/ n_months` |
| `pf_score_tpm` | Sovrastimato |
| `c_norm` | Potenzialmente distorto |

**Walk-forward OOS non è affetto** perché in `walkforward.py` i parametri `timerange_from` e `timerange_to` vengono passati esplicitamente.

### Fix richiesto

Aggiungere `+ 1` nella formula:

```python
# PRIMA (difettoso):
n = max((end.year - start.year) * 12 + (end.month - start.month), 1)

# DOPO (corretto):
n = max((end.year - start.year) * 12 + (end.month - start.month) + 1, 1)
```

---

## BUG-06 🟠 — `alpha_discovery/discovery.py:287-288` — varianza online può essere negativa per cancellazione numerica

**File:** `src/forgedge/alpha_discovery/discovery.py`
**Righe:** 287–288, 291–295

### Codice attuale (difettoso)

```python
var_a = (sumsq_a - cnt_a * mean_a ** 2) / np.maximum(cnt_a - 1, 1)
var_b = (sumsq_b - cnt_b * mean_b ** 2) / np.maximum(cnt_b - 1, 1)
...
sp2 = ((cnt_a - 1) * var_a + (cnt_b - 1) * var_b) / np.maximum(cnt_a + cnt_b - 2, 1)
denom = np.sqrt(sp2 * (1.0 / cnt_a + 1.0 / cnt_b))
usable = (cnt_a >= 2) & (cnt_b >= 2) & (denom > 0)
t = np.where(usable, (mean_a - mean_b) / np.maximum(denom, 1e-9), np.nan)
```

### Problema

La formula `sumsq - n * mean²` è nota per produrre valori leggermente negativi per cancellazione catastrofica quando i dati sono quasi costanti (forward return quasi tutti identici). Se `var_a < 0` o `var_b < 0`, allora `sp2 < 0` e `sqrt(sp2)` produce `nan`. La condizione `usable = ... & (denom > 0)` maschera i `nan` come "non usable", quindi:

- L'orizzonte con quasi-zero varianza (che può essere statisticamente valido) viene scartato silenziosamente
- `h*` viene selezionato tra gli orizzonti rimanenti, potenzialmente subottimale

### Fix richiesto

Clampare `var_a` e `var_b` a zero prima di usarle:

```python
var_a = np.maximum(
    (sumsq_a - cnt_a * mean_a ** 2) / np.maximum(cnt_a - 1, 1),
    0.0
)
var_b = np.maximum(
    (sumsq_b - cnt_b * mean_b ** 2) / np.maximum(cnt_b - 1, 1),
    0.0
)
```

Il clamp a 0 è corretto: una varianza numericamente negativa di piccola magnitudine rappresenta varianza reale prossima a zero, e `sqrt(0) = 0` produce `denom = 0` → `usable = False`, che è il comportamento corretto per distribuzioni degeneri.

---

## BUG-07 🟠 — `validation.py:119` — DSR deflaziona lo Sharpe annualizzato invece di quello per-trade

**File:** `src/forgedge/rule_discovery/validation.py`
**Righe:** 100–121

### Codice attuale (difettoso)

```python
# Riga 100-104: calcolo sharpe_annual
sharpe_trade = exp / max(std, 1e-9)
periods_per_year = 365 * 24 / holding_h if holding_h else 1.0
sharpe_annual = sharpe_trade * math.sqrt(periods_per_year)

# Riga 119: DSR
dsr = deflated_sharpe(sharpe_annual, n_trials, n)
```

```python
# deflated_sharpe (righe 30-51):
def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    ...
    radicand = 1.0 - gamma * math.log(n_trials) / math.log(n_obs)
    correction = math.sqrt(max(radicand, 0.0))
    return norm.cdf((sharpe * correction - sharpe_star) / se)
```

### Problema

La formula DSR di Bailey & de Prado (2014) opera sullo Sharpe **per-osservazione** (per-trade), non sullo Sharpe annualizzato. Deflare `sharpe_annual` (già moltiplicato per `√periods_per_year ≈ √8760 ≈ 93` per dati orari) produce una doppia scalatura. Il DSR risultante è:

- **Più alto** del corretto quando `sharpe_annual >> sharpe_trade` (asset con alta `periods_per_year`)
- Paradossalmente più conservativo per contratti con molti trade

Combinato con l'uso di `n` (numero di trade) invece di `n_barre` come `n_obs`, i due errori si compensano parzialmente ma non si annullano.

### Fix richiesto

Passare `sharpe_trade` (non annualizzato) al DSR:

```python
# PRIMA (difettoso):
dsr = deflated_sharpe(sharpe_annual, n_trials, n)

# DOPO (corretto):
dsr = deflated_sharpe(sharpe_trade, n_trials, n)
```

Se si vuole anche correggere `n_obs`, passare il numero di barre nel training window anziché il numero di trade. Il numero di barre può essere derivato da `len(frame)` (già disponibile in `validate()` come `n = net.size` per i trade, ma `len(trades_frame)` per le barre). Valutare se questo cambiamento è nel scope del fix o da trattare separatamente.

---

## BUG-08 🟠 — `feature_generator.py` — `pd.NA` vs `float("nan")` in colonne nullable `Float64`

**File:** `src/forgedge/event_discovery/feature_generator.py`
**Righe:** 522–577 (`_safe_ratio`, `_safe_spread_pct`, `_safe_position`)

### Codice attuale (potenzialmente difettoso)

```python
def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return result.replace([float("inf"), float("-inf")], float("nan"))
```

### Problema

Se `a` o `b` hanno dtype `Float64` (nullable, es. dopo `pd.read_parquet` con metadati pandas abilitati), la divisione per zero produce `pd.NA` invece di `float("nan")`. Il metodo `.replace([inf, -inf], float("nan"))` non sostituisce `pd.NA`. La serie risultante è di dtype `object` con mix di `float` e `pd.NA`, causando:

- `fillna(0).values.astype(bool)` in `and_composer.py:517` → `TypeError` su `pd.NA`
- `active = event.fillna(0).astype(bool).to_numpy()` in `alpha_discovery/discovery.py:162` → crash

### Impatto attuale

Non ci sono prove che la KPI table usata in produzione abbia colonne `Float64`. Il bug è **latente** ma si attiverebbe con qualsiasi input da `pd.read_parquet` senza `dtype_backend="numpy_nullable"` esplicito.

### Fix richiesto

In tutte e tre le funzioni, aggiungere la conversione esplicita a `float64` prima delle operazioni:

```python
def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.astype("float64")
    b = b.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return result.replace([float("inf"), float("-inf")], float("nan"))
```

Applicare lo stesso pattern a `_safe_spread_pct` e `_safe_position`.

---

## BUG-09 🟡 — `hurst.py:164` — `ou_halflife` ritorna `nan` float invece di `None`

**File:** `src/forgedge/market_context/hurst.py`
**Riga:** 154–167

### Codice attuale (difettoso)

```python
log_p = np.log(series)
delta = np.diff(log_p)
...
kappa = float(reg.coef_[0])
hl = -np.log(2) / np.log(1 + kappa)
if hl <= 0 or hl > len(series):
    return None
return hl
```

### Problema

Quando `kappa ∈ (-2, -1)`, `1 + kappa < 0`, e `np.log(1 + kappa)` produce `nan` (con `RuntimeWarning`). La condizione `hl <= 0` non cattura `nan` (perché `nan <= 0` è `False` in Python). La funzione ritorna `nan` (float), violando il tipo di ritorno dichiarato (`float | None`).

Il chiamante `rolling_halflife` usa:

```python
np.nan if hl is None else hl
```

che non gestisce il caso `nan` float — tratta il `nan` come valore valido. `hl_series.dropna()` a valle lo filtra correttamente, quindi non causa crash, ma viola il contratto della funzione e genera `RuntimeWarning` su ogni finestra con `kappa` in quel range.

### Fix richiesto

Aggiungere il controllo `not np.isfinite(hl)` prima degli altri check:

```python
hl = -np.log(2) / np.log(1 + kappa)
if not np.isfinite(hl) or hl <= 0 or hl > len(series) - 1:
    return None
return hl
```

Il bound `len(series) - 1` corregge anche l'off-by-one segnalato (la regressione OU opera su `n-1` punti, quindi la half-life massima plausibile è `n-1`, non `n`).

---

## BUG-10 🟡 — `backtest.py:284-286` — `exit_price` non validato quando `sell_price` è NaN

**File:** `src/forgedge/rule_discovery/backtest.py`
**Righe:** 284–286

### Codice attuale (difettoso)

```python
exit_price[i] = sp   # sp = sell_price[i]
exit_rn[i] = hit_rn
target_hit[i] = True
```

### Problema

`sell_price` è calcolato come `buy_price * (1 + sell_pct)`. Se `buy_price[i]` è `nan` o `inf` (caso possibile quando `_scan_fill` non esclude questi prima della computazione del prezzo), `sell_price[i]` è `nan` e `exit_price[i] = nan`. In `_summarise()`, `net = (ep - bp) / bp` produce `nan` che viene silenziosamente escluso da `net[net > 0]`, causando la perdita silenziosa di quella trade nei conteggi.

### Fix richiesto

In `_scan_fill`, verificare che `buy_price` sia finito prima di calcolare `sell_price`:

```python
# Aggiungere dopo il calcolo di buy_price:
valid_fill = np.isfinite(buy_price) & (fill_rn >= 0)
sell_price = np.where(valid_fill, buy_price * (1.0 + sell_pct), np.nan)
```

---

## BUG-11 🟡 — `walkforward.py:209-214` — `_oos_months` fragile se `test_to` è l'ultimo giorno del mese

**File:** `src/forgedge/rule_discovery/walkforward.py`
**Righe:** 209–214

### Codice attuale (fragile)

```python
def _oos_months(splits) -> pd.PeriodIndex:
    return pd.period_range(
        pd.Timestamp(splits[0].test_from).to_period("M"),
        pd.Timestamp(splits[-1].test_to).to_period("M") - 1,
        freq="M",
    )
```

### Problema

`test_to` è il **primo giorno del mese successivo** all'ultimo periodo di test (es. `"2024-09-01"` se il test finisce ad agosto). `to_period("M") - 1` porta correttamente al mese di agosto.

Il bug si attiva se `test_to` è invece l'ultimo giorno del mese (es. `"2024-08-31"`): `to_period("M")` dà agosto, e `- 1` porta a luglio, escludendo l'intero mese di agosto dall'`_oos_months`.

Nel flusso attuale `test_to` è sempre costruito come primo del mese successivo (`DateOffset(months=N)` da inizio mese), quindi il bug non si innesca. La fragilità è latente per qualsiasi modifica a come vengono costruiti i `splits`.

### Fix richiesto

Rendere il calcolo esplicito usando `test_to` come limite esclusivo:

```python
def _oos_months(splits) -> pd.PeriodIndex:
    start = pd.Timestamp(splits[0].test_from).to_period("M")
    # test_to è esclusivo: sottrarre 1 giorno per ottenere l'ultimo giorno incluso
    end_ts = pd.Timestamp(splits[-1].test_to) - pd.Timedelta(days=1)
    end = end_ts.to_period("M")
    return pd.period_range(start, end, freq="M")
```

---

## BUG-12 🟡 — `and_composer.py` — triple composition può produrre triple duplicate

**File:** `src/forgedge/event_discovery/and_composer.py`
**Righe:** 247–304

### Problema

La condizione `valid_k > idx_b` previene le triple con indici non crescenti, ma non previene che la stessa tripla `{a, b, c}` appaia due volte se viene generata con seed pair `(a, b)` e `(a, c)` in iterazioni diverse:

- Seed `(a=0, b=1)` genera terzo `k=2` → tripla `{0, 1, 2}`
- Seed `(a=0, b=2)` tenta terzo `k > 2`, ma `k=1` è già stato generato con `idx_b=1` in un'iterazione diversa

In realtà, poiché il terzo componente richiede `valid_k > idx_b`, la tripla `{0, 1, 2}` viene generata **solo** dal seed `(a=0, b=1)` con `k=2`, non dal seed `(a=0, b=2)` (che richiederebbe `k > 2`). Quindi le triple non si duplicano nella struttura attuale. Il rischio residuo è se il seed enumerava già `(0,2)` con `k=1`, ma `valid_k > idx_b = 2` esclude `k=1`. **Nessun bug attivo**, ma manca un commento che espliciti questa invariante.

### Azione richiesta

Aggiungere un commento alla riga 247 che espliciti perché `valid_k > idx_b` garantisce l'unicità delle triple:

```python
# valid_k > idx_b garantisce unicità: la tripla {idx_a, idx_b, k} ha indici strettamente
# crescenti (idx_a < idx_b < k), quindi ogni tripla non ordinata appare esattamente una volta.
valid_k = valid_k[valid_k > idx_b]
```

---

## BUG-13 🟡 — `ema_proxy.py` — `resolved_threshold_basis` è `None` per mode `fixed`

**File:** `src/forgedge/market_context/ema_proxy.py`
**Righe:** 138–141

### Codice attuale (difettoso)

```python
self.resolved_threshold_basis = (
    "global" if self.threshold_mode == "balanced" else None
)
```

### Problema

Quando `threshold_mode="fixed"`, `resolved_threshold_basis` è `None`. In `get_config()` viene esposto come `"resolved_threshold_basis": None`, ambiguo rispetto a un errore di risoluzione o a un campo non applicabile.

### Fix richiesto

```python
self.resolved_threshold_basis = (
    "global" if self.threshold_mode == "balanced" else
    "fixed" if self.threshold_mode == "fixed" else
    None
)
```

---

## BUG-14 🟡 — `context.py` — NaN resetta artificialmente i run consecutivi in `_rolling_stability`

**File:** `src/forgedge/market_context/context.py`
**Righe:** 424–441

### Problema

```python
labels = regime.astype("object")
changed = labels.ne(labels.shift(1))
run_id = changed.cumsum()
run_len = labels.groupby(run_id).cumcount() + 1
stable = (run_len >= window) & labels.notna()
```

`labels.ne(NaN)` è sempre `True` per ogni NaN confrontato con qualsiasi valore. Quindi ogni bar NaN (durante il warmup EMA) avvia un nuovo `run_id`, resettando il `cumcount()`. Dopo il warmup (200+ bar), il regime potrebbe essere stabile, ma il contatore `run_len` riparte da 1 e `regime_stable` rimane `False` per altre `window` barre, perdendo potenzialmente un periodo di stabilità valido.

### Fix richiesto

Costruire i `run_id` solo sui bar non-NaN, lasciando NaN nei gap:

```python
non_nan = labels.notna()
# Calcola il run solo sui valori non-NaN
changed = labels.where(non_nan).ne(labels.where(non_nan).shift(1))
run_id = changed.cumsum()
run_len = labels.groupby(run_id).cumcount() + 1
stable = (run_len >= window) & non_nan
```

In alternativa, propagare il `run_id` precedente attraverso i NaN (forward-fill del run su NaN), ma questa scelta dipende dalla semantica desiderata.

---

## BUG-15 🟡 — `feature_generator.py` — diffnorm su serie sparse senza controllo minimo di punti

**File:** `src/forgedge/event_discovery/feature_generator.py`
**Funzione:** `_safe_diff_norm`

### Codice attuale

```python
def _safe_diff_norm(a: pd.Series, b: pd.Series) -> tuple[pd.Series, float]:
    diff = a - b
    std = float(diff.std())
    if std == 0 or pd.isna(std):
        return pd.Series(float("nan"), index=a.index, dtype=float), 0.0
    return diff / std, std
```

### Problema

`diff.std()` con `skipna=True` ignora i NaN leading. Se le serie hanno molti NaN (EMA con period lungo su dataset corto), `std` può essere calcolato su troppo pochi punti (es. 5 osservazioni su 250 barre), producendo un normalizzatore instabile. Non c'è controllo sul numero minimo di punti validi.

### Fix richiesto

Aggiungere un check sul numero minimo di osservazioni valide:

```python
def _safe_diff_norm(a: pd.Series, b: pd.Series, min_obs: int = 30) -> tuple[pd.Series, float]:
    diff = a - b
    n_valid = diff.notna().sum()
    if n_valid < min_obs:
        return pd.Series(float("nan"), index=a.index, dtype=float), 0.0
    std = float(diff.std())
    if std == 0 or pd.isna(std):
        return pd.Series(float("nan"), index=a.index, dtype=float), 0.0
    return diff / std, std
```

---

## BUG-16 🟡 — `discovery.py` — `_count_zero_months` usa branch DatetimeIndex ignorando `timestamps` del caller

**File:** `src/forgedge/event_discovery/discovery.py`
**Righe:** 632–645

### Problema

```python
if isinstance(series.index, pd.DatetimeIndex):
    periods = series.index.to_period("M")
else:
    periods = pd.DatetimeIndex(timestamps.values).to_period("M")
```

Se `series` ha un `DatetimeIndex`, il branch usa l'indice di `series` ignorando il parametro `timestamps` passato dal caller. Se i due indici divergessero (uso esterno non previsto), i conteggi mensili sarebbero calcolati su indici diversi senza segnalazione.

### Fix richiesto

Eliminare il branch e usare sempre `timestamps`:

```python
periods = pd.DatetimeIndex(timestamps.values).to_period("M")
```

Verificare che tutti i caller passino `timestamps` correttamente (nel flusso corrente lo fanno).
