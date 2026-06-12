# Modulo 3 — Rule Discovery

Rule Discovery è il quarto modulo della pipeline FORGE. Riceve un `AlphaContract`
promosso da Alpha Discovery e risponde alla domanda operativa che il contratto
lascia aperta: **il pattern statisticamente identificato sopravvive a un backtest
realistico — con fee, fill rate finito, ordini limite e target discreto — e tiene
fuori campione?**

L'output è un `RuleDiscoveryResponse` con verdetto `EDGE` / `PARTIAL-EDGE` /
`NON-EDGE` e, nei primi due casi, una `ValidatedRule` con i parametri operativi
validati.

---

## Utilizzo di base

```python
from forgedge import (
    MarketContext, EventDiscovery, AlphaDiscovery, AlphaConfig,
    RuleDiscovery, RuleDiscoveryConfig,
)

enriched   = MarketContext(kpi).run()
ed         = EventDiscovery(enriched)
candidates = ed.run()

ad = AlphaDiscovery(ed.df, candidates, AlphaConfig(asset="BTC", timeframe="1H"))
contracts  = ad.run()
promoted   = ad.promoted_contracts()

# Index dei candidati per ID (richiesto da RuleDiscovery)
by_id = {c.event_id: c for c in candidates}

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        vr = resp.validated_rule
        print(f"  entry: {vr.params.buy_type}  drop={vr.params.buy_drop_pct}"
              f"  sell={vr.params.sell_pct}  h={vr.params.target_h}")
```

---

## Posizione nella pipeline

```
list[AlphaContract] promossi (Modulo 2)
EventCandidate corrispondenti (Modulo 1)
        │
        ▼
  RuleDiscovery.run()  [per ogni contratto]
        │
        ▼
  RuleDiscoveryResponse
  ├─ verdict: EDGE | PARTIAL-EDGE | NON-EDGE
  ├─ validated_rule   (se EDGE o PARTIAL-EDGE)
  ├─ in_sample_summary
  ├─ execution_envelope + excursion (MAE/MFE)
  ├─ walk_forward OOS
  ├─ statistical_validation
  └─ regime_analysis
        │
        ▼
  Rule Registry (non implementato)
```

Rule Discovery è **l'unico modulo che usa i prezzi per una simulazione di
esecuzione**. Non ri-ottimizza le soglie degli eventi né modifica i parametri
del target derivato: usa l'espressione dell'evento e `derived_target.sell_pct` /
`derived_target.holding_period_h` come punto di partenza per la grid operativa.

---

## Pipeline a 5 step

### Step 1 — Setup

**Ricostruzione del segnale.** L'evento booleano viene ricostruito deterministic-
amente sul DataFrame tramite `EventCandidate.apply()` (o dalla `event_series`
memorizzata) e iniettato come colonna `__rule_signal__`.

**Seed dai parametri del contratto.** Con `use_contract_target=True` (default),
i parametri di backtest di partenza vengono inizializzati da:
- `target_h = derived_target.holding_period_h`
- `sell_pct = max(0.01, derived_target.sell_pct)` (clampato a un floor operativo)

Questi diventano il centro della grid di screening, non parametri fissi.

**Propagazione della direzione.** Il motore supporta `"long"` e `"short"`.
La direzione viene letta da `derived_target.direction` e impostata su
`BacktestParams.direction`. Il meccanismo short è il perfetto specchio del long:
il limite di acquisto si trova *sopra* l'anchor (`anchor × (1 + buy_drop_pct)`),
si riempie quando `high` lo raggiunge, e il take-profit è *sotto* il prezzo di
fill (`fill × (1 − sell_pct)`), raggiunto quando il prezzo scende fino a esso.
Il guadagno netto è `(fill − exit) / fill`. Solo direzioni diverse da `"long"`
o `"short"` generano `NON-EDGE` immediato.

---

### Step 2 — Grid screening in-sample

La grid operativa esplora il prodotto cartesiano di:

| Dimensione | Controllata da |
|---|---|
| `buy_drop_pct` | `GridSpec.buy_drop_pct` |
| `sell_pct` | `GridSpec.sell_pct` |
| `target_h` | `GridSpec.target_h` |
| `buy_delay_bar` | `GridSpec.buy_delay_bar` |

Quando `GridSpec` è vuoto (default), FORGE costruisce automaticamente una grid
sensata centrata sui valori derivati dal contratto.

**Meccanica di esecuzione per ogni configurazione:**

1. Al segnale, viene piazzato un ordine limite a `anchor * (1 - buy_drop_pct)`
2. Se il prezzo tocca il limite entro `buy_delay_bar` barre, l'ordine viene
   eseguito (fill). Altrimenti è annullato.
3. Dopo il fill, la posizione viene chiusa:
   - al primo bar che chiude a ≥ `sell_price = fill_price * (1 + sell_pct)`, oppure
   - al close della barra `target_h` (stop a orizzonte)

La configurazione viene valutata tramite il composite score `pf_score_tpm` che
bilancia Profit Factor, frequenza di trading e consistenza mensile.

**Early elimination (Step 2.3):** configurazioni con < 20 trade, PF < 1.0, o
`fill_rate < min_fill_rate` vengono scartate prima della validazione.

---

### Step 3 — Selezione e raffinamento

La configurazione migliore è quella con `pf_score_tpm` massimo tra quelle che
superano le soglie di `SelectionCriteria`. Se nessuna configurazione è
selezionabile, il verdetto è `NON-EDGE`.

---

### Step 4 — Validazione statistica

Sulla configurazione selezionata, sull'IS:

| Metrica | Descrizione |
|---|---|
| `ttest_winrate_t/p` | t-test win rate vs base rate del contratto |
| `ttest_expectancy_t/p` | t-test expectancy vs zero |
| `sharpe_ratio` | Sharpe annualizzato |
| `deflated_sharpe` | Sharpe deflato per n_trials (penalizza data snooping) |
| `temporal_stability` | `"PASS"` / `"WARN"` / `"FAIL"`: PF prima metà vs seconda |
| `n_trials_tested` | Numero di configurazioni testate nella grid |

#### Walk-forward OOS

Il timeline viene diviso in `n_splits` finestre di test consecutive. Per ogni
split:
1. La grid viene ri-screenata sul train window precedente (con `reoptimise=True`)
2. I parametri migliori vengono applicati una sola volta sul test window
3. I trade delle finestre di test vengono concatenati → track record OOS onesto

`WalkForwardResult.oos_summary` contiene le metriche aggregate sui trade OOS.
`WalkForwardResult.consistency` = quota di finestre con net gain positivo.

---

### Step 5 — Regime dependency

Per ogni regime presente nel DataFrame, vengono calcolate le metriche di
performance sui trade IS: PF, win rate, expectancy, net gain cumulato.

`dependency_score`: entropia normalizzata della distribuzione mensile dei trade
(0 = trade distribuiti uniformemente; 1 = tutti i trade concentrati in un mese).

`avoid_in`: regimi con ≥ 5 trade e PF < 1.0 — regimi da evitare in produzione.

---

## Verdetto: EDGE / PARTIAL-EDGE / NON-EDGE

### Gate `NON-EDGE` (hard — esclusione immediata)

| Condizione | Parametro |
|---|---|
| PF IS < `partial_min_profit_factor` (1.5) | — |
| Trade IS < `min_trades` (30) | — |
| t-test expectancy p ≥ `max_ttest_p` (0.05) | — |
| PF OOS walk-forward < 1.0 | — |

Se uno qualsiasi è violato: `NON-EDGE`.

### Gate `EDGE` (tutti richiesti per verdetto pieno)

| Condizione | Parametro |
|---|---|
| PF IS ≥ `min_profit_factor` (2.0) | — |
| Win rate IS ≥ `min_win_rate` (0.55) | — |
| `zero_months` ≤ `max_zero_months_edge` (1) | — |
| DSR ≥ `min_dsr` (1.0) | — |
| `temporal_stability` ≠ `"FAIL"` | — |
| `dependency_score` ≤ `max_regime_dependency` (0.30) | — |
| OOS consistency ≥ 0.50 | — |

Se tutti soddisfatti: `EDGE`. Se solo i gate NON-EDGE sono soddisfatti ma non
tutti i gate EDGE: `PARTIAL-EDGE`.

```python
resp.verdict         # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
resp.is_edge         # True se EDGE o PARTIAL-EDGE
resp.rejection_reasons  # gate falliti (lista vuota se EDGE)
```

---

## Execution Envelope e MAE/MFE

### `ExecutionEnvelope`

La stessa configurazione viene backtestata con due convenzioni di exit diverse,
definendo un intervallo di esecuzione realistica:

| Variante | `target_hit_col` | Descrizione |
|---|---|---|
| `conservative` | `"close"` | Il target conta solo quando una barra **chiude** oltre sell_price. Sottostima (il limit sell reale potrebbe riempirsi intrabar). Corrisponde al motore di riferimento certificato, sia per long che per short. |
| `optimistic` | `"high"` (long) / `"low"` (short) | Il target conta al primo tocco intrabar. Sovrastima (assume che il limit sell si riempia sempre). |

La performance reale si trova tra i due. Per risolvere la colonna ottimistica
in base alla direzione, usare l'utility `optimistic_hit_col`:

```python
from forgedge.rule_discovery import optimistic_hit_col

col = optimistic_hit_col("short")  # → "low"
col = optimistic_hit_col("long")   # → "high"

env = resp.execution_envelope
print(f"conservative PF: {env.conservative.profit_factor:.2f}")
print(f"optimistic   PF: {env.optimistic.profit_factor:.2f}")
```

### `ExcursionStats` (MAE/MFE)

Per ogni trade eseguito, sulla finestra `[fill+1 .. exit]`:
- **MAE** (Maximum Adverse Excursion): drawdown massimo raggiunto, `(min_low - buy_price) / buy_price` (negativo)
- **MFE** (Maximum Favourable Excursion): run-up massimo raggiunto, `(max_high - buy_price) / buy_price` (positivo)

```python
ex = resp.excursion
print(f"MAE medio: {ex.mae_mean:.4f}, peggiore: {ex.mae_worst:.4f}")
print(f"MFE medio: {ex.mfe_mean:.4f}, ha raggiunto target: {ex.mfe_reached_target_pct:.1f}%")
```

---

## Struttura dati: `RuleDiscoveryResponse`

```python
resp = rd.run()

resp.verdict             # str: "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
resp.is_edge             # bool: True se EDGE o PARTIAL-EDGE
resp.alpha_id            # str: ID del contratto sorgente
resp.asset, resp.timeframe  # str

# Regola validata (None se NON-EDGE)
resp.validated_rule.expression       # espressione booleana
resp.validated_rule.params           # BacktestParams con la configurazione ottimale
resp.validated_rule.event_candidate_id

# Metriche IS
resp.in_sample_summary.total_trades
resp.in_sample_summary.profit_factor
resp.in_sample_summary.win_rate_pct
resp.in_sample_summary.expectancy
resp.in_sample_summary.tpm_mu        # trade/mese medi
resp.in_sample_summary.zero_months   # mesi senza trade

# Execution envelope + MAE/MFE
resp.execution_envelope   # ExecutionEnvelope | None
resp.excursion            # ExcursionStats | None

# Walk-forward OOS
resp.walk_forward.oos_summary.profit_factor
resp.walk_forward.consistency          # quota finestre con net gain > 0
resp.walk_forward.n_profitable_splits
resp.walk_forward.oos_envelope         # ExecutionEnvelope OOS
resp.walk_forward.oos_excursion        # ExcursionStats OOS

# Validazione statistica
resp.statistical_validation.deflated_sharpe
resp.statistical_validation.ttest_expectancy_p
resp.statistical_validation.temporal_stability  # "PASS"/"WARN"/"FAIL"

# Regime
resp.regime_analysis.dependency_score
resp.regime_analysis.avoid_in          # list[str] regimi da evitare
resp.regime_analysis.per_regime        # list[dict]

# Audit
resp.rejection_reasons   # list[str]
resp.notes               # list[str]
resp.grid_results        # list[GridResult]
```

### `BacktestSummary` — campi completi

| Campo | Descrizione |
|---|---|
| `total_signals` | Segnali totali nel dataset |
| `total_trades` | Trade eseguiti (signal riempito) |
| `fill_rate` | `total_trades / total_signals` |
| `win_rate_pct` | Win rate (0–1) |
| `winning_trades`, `losing_trades` | Conteggi |
| `total_net_gain` | Somma dei rendimenti netti (fee incluse) |
| `expectancy` | `total_net_gain / total_trades` |
| `std_net_gain` | Dev. standard rendimenti netti |
| `profit_factor` | `sum(vincite) / sum(perdite)` |
| `best_trade`, `worst_trade` | Rendimento migliore/peggiore |
| `target_hit_rate_pct` | % trade che hanno raggiunto il target |
| `n_months` | Mesi totali nel dataset |
| `active_months` | Mesi con almeno un trade |
| `zero_months` | `n_months - active_months` |
| `tpm_mu`, `tpm_sigma` | Media e dev. std. trade/mese |
| `c_norm` | Consistenza mensile normalizzata |
| `pf_score`, `pf_score_tpm` | Score composito (con e senza penalità frequenza) |
| `exp_score_tpm` | Score basato su expectancy e frequenza |
| `sharpe_raw` | Sharpe grezzo (non annualizzato) |

---

## Metodi di output

### `rd.run() → RuleDiscoveryResponse`

Esegue l'intera pipeline. Deve essere chiamato prima di qualsiasi altro metodo.

### `rd.grid_summary() → pd.DataFrame`

DataFrame piatto di tutte le configurazioni testate nella grid IS, ordinabile per
`pf_score_tpm`:

```python
grid_df = rd.grid_summary()
print(grid_df.sort_values("pf_score_tpm", ascending=False).head())
# Colonne: buy_drop_pct, sell_pct, target_h, buy_delay_bar,
#          profit_factor, win_rate_pct, total_trades, expectancy,
#          tpm_mu, fill_rate, zero_months, pf_score_tpm
```

### `text_report(resp) → str` e `html_report(resp) → str`

```python
from forgedge.rule_discovery import text_report, html_report

# Report testuale compatto
print(text_report(resp))

# Report HTML self-contained (no CDN, funziona offline)
with open(f"{resp.alpha_id}.html", "w") as f:
    f.write(html_report(resp))
```

L'HTML include: sezione verdict, parametri della regola validata, metriche IS,
execution envelope, MAE/MFE, walk-forward per split, validazione statistica,
breakdown per regime. Tutto implementato con CSS inline senza dipendenze esterne.

### `resp.to_dict() → dict`

Dizionario nidificato completo per serializzazione YAML/JSON:

```python
import json
with open(f"{resp.alpha_id}_rule_discovery.json", "w") as f:
    json.dump(resp.to_dict(), f, indent=2)
```

---

## Configurazione completa

### `BacktestParams`

| Parametro | Default | Descrizione |
|---|---|---|
| `direction` | `"long"` | `"long"` o `"short"`. Short = specchio simmetrico del long: entry sopra anchor, take-profit sotto fill |
| `buy_type` | `"limit"` | `"limit"` o `"market"` |
| `buy_drop_pct` | `0.010` | Entità dello scostamento dal anchor (es. 0.01 = 1%): sconto per long, premio per short |
| `buy_delay_bar` | `6` | Barre di vita dell'ordine limite |
| `buy_price_anchor` | `"close"` | Colonna usata come anchor del limite |
| `sell_pct` | `0.040` | Take-profit come frazione del prezzo di fill |
| `target_h` | `24` | Orizzonte massimo in barre (close-at-horizon) |
| `target_col` | `"close"` | Colonna per l'uscita a orizzonte |
| `target_hit_col` | `"close"` | Colonna per rilevare il take-profit. Conservative = `"close"` per entrambe le direzioni; ottimistico = `"high"` per long, `"low"` per short (usare `optimistic_hit_col(direction)`) |
| `fee` | `0.002` | Fee per lato |
| `early_stopping` | `True` | Esci al take-profit; se False, sempre a orizzonte |

### `GridSpec`

| Parametro | Default | Descrizione |
|---|---|---|
| `buy_drop_pct` | `None` | Lista di sconti da testare (None = auto) |
| `sell_pct` | `None` | Lista di target da testare (None = auto) |
| `target_h` | `None` | Lista di orizzonti da testare (None = auto) |
| `buy_delay_bar` | `None` | Lista di delay da testare (None = auto) |

### `WalkForwardConfig` (Rule Discovery)

| Parametro | Default | Descrizione |
|---|---|---|
| `n_splits` | `4` | Numero di finestre test OOS |
| `train_span_months` | `None` | Mesi di train (None = anchored, cresce) |
| `test_span_months` | `None` | Mesi di test (None = divisi ugualmente) |
| `min_train_months` | `6` | Train minimo prima della prima finestra test |
| `reoptimise` | `True` | Re-ottimizza la grid su ogni train window |

### `SelectionCriteria`

| Parametro | Default | Descrizione |
|---|---|---|
| `min_profit_factor` | `2.0` | PF minimo per EDGE |
| `min_win_rate` | `0.55` | Win rate minimo (0–1) |
| `min_trades` | `30` | Trade minimi |
| `min_tpm` | `2.0` | Trade/mese minimi |
| `min_pf_score_tpm` | `0.30` | Score composito minimo |
| `min_fill_rate` | `0.40` | Fill rate minimo |
| `partial_min_profit_factor` | `1.5` | PF minimo per PARTIAL-EDGE |
| `max_zero_months_edge` | `1` | Mesi zero massimi per EDGE |
| `max_zero_months_partial` | `4` | Mesi zero massimi per PARTIAL-EDGE |
| `max_regime_dependency` | `0.30` | Score dipendenza regime massimo per EDGE |
| `min_dsr` | `1.0` | Deflated Sharpe minimo per EDGE |
| `max_ttest_p` | `0.05` | p-value massimo t-test expectancy |

### `RuleDiscoveryConfig`

| Parametro | Default | Descrizione |
|---|---|---|
| `base_params` | `BacktestParams()` | Parametri fissi e centro della grid |
| `scoring` | `ScoringParams()` | Parametri del composite score |
| `grid` | `GridSpec()` | Grid operativa (vuota = auto) |
| `walk_forward` | `WalkForwardConfig()` | Impostazioni walk-forward OOS |
| `criteria` | `SelectionCriteria()` | Soglie di selezione e verdetto |
| `use_contract_target` | `True` | Seed sell_pct/target_h dal contratto |
| `timestamp_col` | `"open_dt"` | Colonna datetime |
| `signal_col` | `"__rule_signal__"` | Nome colonna segnale iniettata |
| `discovery_date` | `None` | Data ISO per la response (None = oggi) |

---

## Pattern d'uso avanzati

### Grid personalizzata

```python
from forgedge.rule_discovery import GridSpec

config = RuleDiscoveryConfig(
    grid=GridSpec(
        buy_drop_pct=[0.005, 0.010, 0.015, 0.020],
        sell_pct=[0.03, 0.04, 0.05, 0.06],
        target_h=[12, 24, 36, 48],
        buy_delay_bar=[3, 6, 12],
    )
)
rd = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
```

### Analisi grid completa

```python
# Tutte le configurazioni testate, ordinate per score
grid_df = rd.grid_summary()
top10 = grid_df.nlargest(10, "pf_score_tpm")
print(top10[["sell_pct", "buy_drop_pct", "target_h", "profit_factor",
             "win_rate_pct", "tpm_mu", "pf_score_tpm"]])
```

### Export report per tutti i PARTIAL-EDGE+

```python
from forgedge.rule_discovery import html_report, text_report

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    if resp.is_edge:
        # HTML per review umana
        with open(f"reports/{resp.alpha_id}.html", "w") as f:
            f.write(html_report(resp))
        # JSON per integrazione
        import json
        with open(f"reports/{resp.alpha_id}.json", "w") as f:
            json.dump(resp.to_dict(), f, indent=2)
    else:
        print(f"NON-EDGE {resp.alpha_id}: {resp.rejection_reasons}")
```

### Collegare la response al contratto

`AlphaContract.rule_discovery_response` è il campo riservato a Rule Discovery:

```python
import json
contract.rule_discovery_response = resp.to_dict()
# Il contratto ora porta con sé la risposta operativa per il Rule Registry
```

---

## Note operative

- **`target_hit_col`:** il default `"close"` riproduce il motore certificato di
  riferimento (conservativo). Per la convenzione intrabar ottimistica, usare
  `optimistic_hit_col(direction)` — `"high"` per long, `"low"` per short.
- **Rule Registry:** Modulo 4 non ancora implementato. La `ValidatedRule`
  prodotta da Rule Discovery è già nella forma pronta per l'ingestion nel
  registry.
