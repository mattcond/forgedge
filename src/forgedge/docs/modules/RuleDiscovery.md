# FORGE — Rule Discovery Pipeline
> Terzo modulo della pipeline FORGE.
> Riceve un **Alpha Contract** da Alpha Discovery e risponde a una domanda
> operativa: il pattern statisticamente evidenziato è sfruttabile in un
> backtest realistico — con fee, fill rate, limit order e target discreto?
> Emette un verdetto (`EDGE / NON-EDGE / PARTIAL-EDGE`) e restituisce
> l'Alpha Contract aggiornato con parametri operativi validati.

---

## Indice

1. [Posizionamento e Responsabilità](#1-posizionamento-e-responsabilità)
2. [Input: l'Alpha Contract](#2-input-lalpha-contract)
3. [Step 1 — Setup: Parse e Feature Preparation](#3-step-1--setup-parse-e-feature-preparation)
4. [Step 2 — Backtest e Scoring](#4-step-2--backtest-e-scoring)
5. [Step 3 — Selezione e Raffinamento](#5-step-3--selezione-e-raffinamento)
6. [Step 4 — Validazione Statistica](#6-step-4--validazione-statistica)
7. [Step 5 — Analisi della Dipendenza dal Regime](#7-step-5--analisi-della-dipendenza-dal-regime)
8. [Output: Verdetto e Alpha Contract Aggiornato](#8-output-verdetto-e-alpha-contract-aggiornato)
9. [Checklist Completa](#9-checklist-completa)

---

## 1. Posizionamento e Responsabilità

```
┌──────────────────────────────────────────────────────────────────┐
│                        PIPELINE FORGE                            │
│                                                                  │
│  ┌────────────────────┐                                          │
│  │  EVENT DISCOVERY   │  Genera eventi booleani dal catalogo     │
│  └──────────┬─────────┘                                          │
│             │ Event Candidates                                   │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │  ALPHA DISCOVERY   │  Misura il potere predittivo             │
│  └──────────┬─────────┘                                          │
│             │ Alpha Contract                                     │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │  RULE DISCOVERY    │  ← questo modulo                        │
│  │  Input:  Alpha     │    Valida l'operatività con backtest     │
│  │          Contract  │    Emette EDGE / NON-EDGE                │
│  └──────────┬─────────┘                                          │
│             │ Regola validata + parametri                        │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │  RULE REGISTRY     │  Deduplica, cross-ticker, export        │
│  └────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Separazione netta dalle responsabilità upstream

| Modulo | Responsabilità | NON fa |
|---|---|---|
| Event Discovery | Genera eventi booleani | Non sa nulla del target |
| Alpha Discovery | Misura IC, WR, lift, regime | Non modifica soglie, non crea regole |
| **Rule Discovery** | **Valida l'operatività con backtest** | **Non genera eventi, non modifica soglie** |
| Rule Registry | Deduplica, cross-ticker, export | Non valuta qualità singola regola |

**Vincolo critico:** Rule Discovery riceve l'espressione dall'Alpha Contract
e la usa così com'è. Non introduce nuove feature, non ottimizza le soglie,
non esplora combinazioni alternative. L'esplorazione è già avvenuta
a monte — qui si valida e si parametrizza l'operatività.

---

## 2. Input: l'Alpha Contract

Rule Discovery legge il campo `event_expression` e i `rule_discovery_hints`
dell'Alpha Contract prodotto da Alpha Discovery.

```yaml
# Estratto dell'Alpha Contract ricevuto
event_expression:   "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"

target_definition:
  holding_period_h: 24
  sell_pct:         0.04
  base_rate:        0.235

rule_discovery_hints:
  entry_mode:         "limit"
  buy_drop_pct_range: [0.006, 0.015]
  sell_pct_range:     [0.030, 0.050]
  target_h_range:     [12, 48]
  min_pf_target:      2.0
  min_wr_target:      0.70
  exclusion_conditions:
    - "Regime uptrend continuo"
    - "Volume < 0.3× SMA25"

# Statistiche disponibili (da usare come riferimento, non come input del backtest)
event_stats:
  n_activations:  329
  win_rate:       0.415        # win rate statistico (no fill, no fee)
  lift_vs_base:   +0.180
  cohens_d:       0.394
```

> **Nota:** il `win_rate` nell'Alpha Contract è calcolato su tutte le barre
> di attivazione, senza simulare il fill del limit order e senza dedurre le fee.
> Il backtest di Rule Discovery produrrà un win rate diverso (tipicamente più
> basso) perché solo una parte delle attivazioni viene effettivamente riempita.

---

## 3. Step 1 — Setup: Parse dell'Alpha Contract

Rule Discovery non carica dati, non calcola feature, non prepara dataset.
Riceve due input già pronti dalla sessione FORGE corrente:

- **Alpha Contract** — prodotto da Alpha Discovery
- **KPI Table estesa** — la stessa tabella usata da Event Discovery e
  Alpha Discovery, che contiene già tutte le feature derivate
  (pctrank, zscore, delta) calcolate da Event Discovery nel Transform Layer

L'unico lavoro di setup è leggere il contratto e definire il grid.

### 1.1 Parse dell'espressione

```python
# Legge l'espressione dall'Alpha Contract
expression = alpha_contract['event_expression']
# → "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"

# Verifica che tutte le feature richieste siano presenti nella KPI Table
# (devono esserlo — Event Discovery le ha già calcolate nella stessa sessione)
required = parse_required_features(expression)
assert all(f in kpi_table.columns for f in required), \
    f"Feature mancanti: {set(required) - set(kpi_table.columns)}"
```

### 1.2 Definizione del grid parametrico

I parametri operativi da esplorare vengono letti dagli hints del contratto:

```python
BUY_DROP_GRID = np.arange(
    alpha_contract['rule_discovery_hints']['buy_drop_pct_range'][0],
    alpha_contract['rule_discovery_hints']['buy_drop_pct_range'][1],
    step=0.002
)
SELL_PCT_GRID = np.arange(
    alpha_contract['rule_discovery_hints']['sell_pct_range'][0],
    alpha_contract['rule_discovery_hints']['sell_pct_range'][1],
    step=0.005
)

BASE_CONFIG = {
    'buy_type':         'limit',
    'buy_price_anchor': 'close',
    'buy_drop_pct':     0.010,
    'buy_delay_bar':    6,
    'sell_pct':         0.040,
    'target_h':         24,
    'fee':              0.002,
    'early_stopping':   True,
    'pf_min_trades':    15,
    'pf_min_tpm':       2,
    'pf_tpm_target':    3,
}
```

> **Vincolo:** il grid esplora solo i parametri operativi — `buy_drop_pct`,
> `sell_pct`, `target_h`. L'espressione rimane invariata.

---

## 4. Step 2 — Backtest e Scoring

### 2.1 Meccanica del fill limit

```
Barra t:    segnale attivato
            buy_price = close_t × (1 - buy_drop_pct)

Barre t+1..t+6:  fill window
            Se low_k ≤ buy_price → fill a buy_price
            Se nessuna barra tocca buy_price → segnale scartato

Barre fill+1..fill+24:  exit window
            Se high_k ≥ buy_price × (1 + sell_pct) → TARGET HIT
            Se k = fill+24 → chiudi a close_k
```

**Fill rate atteso:** con `buy_drop_pct = 1%` e `delay = 6`, il fill rate
è tipicamente 65–78%. Un fill rate < 50% suggerisce che lo sconto è troppo
profondo per il contesto di mercato.

### 2.2 Metriche di scoring

```python
summary = run_backtest(candle=kpi_table, rule=expression, **BASE_CONFIG)

# Metriche primarie
profit_factor        = summary['profit_factor']       # target: ≥ 2.0
win_rate_pct         = summary['win_rate_pct']        # target: ≥ 0.70
expectancy           = summary['expectancy']          # target: > 0

# Metriche di distribuzione temporale
_tpm_mu              = summary['_tpm_mu']             # trade/mese media
_tpm_C_norm          = summary['_tpm_C_norm']         # consistenza [0,1]
pf_score_tpm         = summary['pf_score_tpm']        # = PF × C_norm

# Diagnostica
fill_rate            = summary['fill_rate']           # % segnali riempiti
zero_months          = summary['zero_months']         # mesi senza trade
```

### 2.3 Criteri di eliminazione rapida

Scartare immediatamente (`NON-EDGE`) se:

| Condizione | Significato |
|---|---|
| `total_trades < 20` | Troppo pochi trade — non significativo |
| `profit_factor < 1.0` | Strategia perdente in-sample |
| `win_rate_pct < 0.50` | Meno di metà trade vincenti |
| `_tpm_mu < 1.5` | Meno di 1.5 trade/mese — troppo raro |
| `fill_rate < 0.40` | Buy_drop troppo profondo, non si riempie |

### 2.4 Screening in batch sul grid

```python
results = []
for buy_drop in BUY_DROP_GRID:
    for sell_pct in SELL_PCT_GRID:
        summary, perf, _ = run_backtest(
            candle=kpi_table, rule=expression,
            buy_drop_pct=buy_drop, sell_pct=sell_pct,
            **{k:v for k,v in BASE_CONFIG.items()
               if k not in ['buy_drop_pct','sell_pct']}
        )
        results.append({
            'buy_drop': buy_drop, 'sell_pct': sell_pct,
            'pf': summary['profit_factor'],
            'wr': summary['win_rate_pct'],
            'tpm': summary['_tpm_mu'],
            'sc': summary['pf_score_tpm'],
            'zero_m': summary['zero_months'],
        })

# Ordina per pf_score_tpm — bilancia PF e consistenza temporale
results_df = pd.DataFrame(results).sort_values('sc', ascending=False)
```

---

## 5. Step 3 — Selezione e Raffinamento

### 3.1 Criteri di selezione del miglior set di parametri

```
profit_factor    ≥ 2.0
win_rate_pct     ≥ 0.70
total_trades     ≥ 30
_tpm_mu          ≥ 3.0
pf_score_tpm     ≥ 0.30
```

Il parametro selezionato è quello con `pf_score_tpm` massimo tra tutti
quelli che superano i criteri. Non si usa il PF grezzo per evitare overfitting.

Una volta individuata la configurazione migliore, si esegue un ultimo backtest
completo per ottenere il DataFrame dei trade da usare nei passi successivi:

```python
# Configurazione vincente dal grid
BEST_CONFIG = results_df.iloc[0]  # riga con pf_score_tpm massimo

_, perf_trades_raw, _ = run_backtest(
    candle=kpi_table, rule=expression,
    buy_drop_pct=BEST_CONFIG['buy_drop'],
    sell_pct=BEST_CONFIG['sell_pct'],
    **{k:v for k,v in BASE_CONFIG.items()
       if k not in ['buy_drop_pct','sell_pct']}
)

# Solo i trade effettivamente riempiti
perf_trades = perf_trades_raw[perf_trades_raw['buy_signal_bool'] == 1].copy()
```

### 3.2 Analisi del breakdown mensile

```python
perf_trades['month'] = pd.to_datetime(perf_trades['fill_open_dt']).dt.to_period('M')

monthly = perf_trades.groupby('month').agg(
    n_trades = ('net_pct_gain', 'count'),
    win_rate = ('gain_bool', 'mean'),
    cum_gain = ('net_pct_gain', 'sum'),
)

bad_months = monthly[monthly['cum_gain'] < 0]
print(f"Mesi negativi: {len(bad_months)}/12")
print(f"Mesi zero trade: {(monthly['n_trades'] == 0).sum()}/12")
```

**Per RI_01:** distribuzione mensile `[9, 19, 6, 12, 1, 13, 1, 2, 3, 12, 5, 19]`
— zero mesi a zero trade.

### 3.3 Robustezza parametrica

Una regola robusta mantiene PF ≥ 2 anche variando leggermente i parametri:

```python
sensitivity = {}
for delta in [-0.002, -0.001, 0, +0.001, +0.002]:
    s, _, _ = run_backtest(
        candle=kpi_table, rule=expression,
        buy_drop_pct=BEST_DROP + delta, **OTHER_PARAMS
    )
    sensitivity[BEST_DROP + delta] = s['profit_factor']

# Regola fragile: PF varia da 0.8 a 3.5 su delta di ±0.2%
# Regola robusta: PF rimane tra 2.5 e 3.8 su tutto il range
```

---

## 6. Step 4 — Validazione Statistica

### 4.1 Significatività del win rate

```python
from scipy import stats

# H0: win_rate = base_rate (nessun edge)
wins  = perf_trades['gain_bool'].values
base  = alpha_contract['target_definition']['base_rate']

t_stat, p_value = stats.ttest_1samp(wins, popmean=base, alternative='greater')
# p < 0.05 → edge statisticamente significativo
```

### 4.2 Significatività dell'expectancy

```python
# H0: expectancy = 0 (strategia neutrale)
gains = perf_trades['net_pct_gain'].values
t_stat, p_value = stats.ttest_1samp(gains, popmean=0, alternative='greater')
```

### 4.3 Deflated Sharpe Ratio

Il grid al Step 2 ha testato N configurazioni. Il PF in-sample del vincitore
è sovrastimato — il DSR corregge questo bias:

```python
def deflated_sharpe(sr_selected, n_trials, n_obs):
    """
    sr_selected: Sharpe Ratio annualizzato della configurazione selezionata
    n_trials:    N configurazioni testate nel grid
    n_obs:       numero di trade della configurazione selezionata
    """
    import math
    gamma = 0.5772156649
    correction = math.sqrt(1 - gamma * math.log(n_trials) / math.log(n_obs))
    return sr_selected * correction

# DSR > 1.0 → l'edge è reale dopo correzione per multiple testing
```

> **Per RI_01:** ~80 configurazioni testate, SR ≈ 2.1, 102 trade → DSR ≈ 1.3.

### 4.4 Stazionarietà temporale

```python
trades = perf_trades.sort_values('fill_open_dt')
midpoint = len(trades) // 2
first_half  = trades.iloc[:midpoint]
second_half = trades.iloc[midpoint:]

print(f"Prima metà:   PF={compute_pf(first_half):.2f}  WR={first_half['gain_bool'].mean():.1%}")
print(f"Seconda metà: PF={compute_pf(second_half):.2f}  WR={second_half['gain_bool'].mean():.1%}")

# Red flag: PF > 3 nella prima metà e < 1.5 nella seconda
```

---

## 7. Step 5 — Analisi della Dipendenza dal Regime

### 5.1 Leggere il regime dalla KPI Table

La colonna `regime` è già presente nella KPI Table — calcolata dal
**Market Context Module** all'inizio della sessione. Rule Discovery
la legge direttamente senza ricalcolarla.

```python
# Legge il regime dalla KPI Table — non ricalcola
regime_col = kpi_table[['open_dt', 'regime', 'regime_stable']]
```

### 5.2 Performance per regime

```python
trades_with_regime = perf_trades.merge(
    regime_col,
    left_on='fill_open_dt', right_on='open_dt'
)

regime_perf = trades_with_regime.groupby('regime').agg(
    n_trades = ('net_pct_gain', 'count'),
    win_rate = ('gain_bool', 'mean'),
    cum_gain = ('net_pct_gain', 'sum'),
    exp      = ('net_pct_gain', 'mean'),
)
```

### 5.3 Metrica di dipendenza

```python
def regime_dependency_score(monthly_trades):
    """
    0.0 = distribuzione uniforme — regime-independent
    1.0 = tutti i trade in un mese — massima dipendenza
    """
    from scipy.stats import entropy
    p = monthly_trades / monthly_trades.sum()
    h = entropy(p + 1e-10)
    h_max = np.log(len(monthly_trades))
    return 1 - (h / h_max)

monthly = perf_trades.groupby('month').size()
dep_score = regime_dependency_score(monthly)
# < 0.2 = bassa dipendenza
# > 0.5 = alta dipendenza → segnalare nel contratto
```

### 5.4 Decisione

```
SE zero_months ≤ 1 E regime_dependency < 0.3:
    → EDGE — regola accettabile as-is

SE zero_months ∈ [2, 4] E PF > 3:
    → PARTIAL-EDGE — deploy con Regime Filter
    → documentare quale regime evitare

SE zero_months > 4:
    → NON-EDGE — dipendenza dal regime eccessiva
    → documentare il motivo nel campo rejection_reasons dell'Alpha Contract
```

---

## 8. Output: Verdetto e Alpha Contract Aggiornato

Rule Discovery compila il campo `rule_discovery_response`
nell'Alpha Contract e restituisce il documento aggiornato.

```yaml
rule_discovery_response:
  date:    "2026-05-23"
  verdict: "EDGE"     # EDGE | NON-EDGE | PARTIAL-EDGE

  # Solo se EDGE o PARTIAL-EDGE
  validated_rule:
    expression:      "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
    # Espressione invariata rispetto all'Alpha Contract
    entry_mode:      "limit"
    buy_drop_pct:    0.010
    sell_pct:        0.040
    target_h:        24
    fee:             0.002

  backtest_results:
    profit_factor:   3.17
    win_rate:        0.814
    total_trades:    102
    fill_rate:       0.71
    expectancy:      0.0155
    zero_months:     0
    tpm_mu:          8.5
    pf_score_tpm:    0.61

  statistical_validation:
    ttest_winrate_p:   0.0012
    ttest_expectancy_p: 0.0008
    deflated_sharpe:   1.31
    n_trials_tested:   80
    temporal_stability: "PASS"   # prima vs seconda metà

  regime_analysis:
    dependency_score: 0.18
    zero_months:      0
    avoid_in:         ["STRONG_BULL"]

  # Solo se NON-EDGE
  rejection_reasons: null

# Aggiorna lo status
status: "VALIDATED"   # o "REJECTED" o "PARTIAL"
```

### Verdetti

**EDGE** — regola operativa. PF ≥ 2, WR ≥ 70%, DSR ≥ 1.0,
distribuzione mensile accettabile. Pronta per Rule Registry.

**PARTIAL-EDGE** — funziona con vincoli espliciti. PF 1.5–2.0, o
mesi vuoti, o instabilità parametrica. Deploy possibile con flag di cautela.

**NON-EDGE** — il backtest non conferma l'alpha statisticamente evidenziato.
Il gap tra IC/WR di Alpha Discovery e PF operativo è normale — Alpha Discovery
misura una relazione, Rule Discovery misura l'operabilità al netto di fee e fill.

---

## 9. Checklist Completa

### Setup

- [ ] Alpha Contract ricevuto con `event_expression` e `rule_discovery_hints`
- [ ] KPI Table estesa verificata — tutte le feature dell'espressione presenti
- [ ] Grid parametrico definito dagli hints del contratto

### Backtest

- [ ] Configurazione `buy_type=limit` con parametri esplicitati
- [ ] Fee incluse (tipicamente 0.2% per lato)
- [ ] `early_stopping=True` per simulazione realistica
- [ ] Fill rate documentato

### Selezione

- [ ] `profit_factor ≥ 2.0`
- [ ] `win_rate ≥ 0.70`
- [ ] `total_trades ≥ 30`
- [ ] `_tpm_mu ≥ 3.0`
- [ ] `pf_score_tpm ≥ 0.30`
- [ ] Breakdown mensile analizzato
- [ ] Robustezza parametrica verificata (±2 step su buy_drop e sell_pct)

### Validazione statistica

- [ ] t-test su win rate vs base rate (p < 0.05)
- [ ] t-test su expectancy vs 0 (p < 0.05)
- [ ] Deflated Sharpe Ratio > 1.0
- [ ] Stabilità temporale verificata (prima vs seconda metà)

### Regime

- [ ] Zero months documentati
- [ ] Regime dependency score calcolato
- [ ] Performance per regime analizzata
- [ ] Vincoli di regime documentati nell'output se PARTIAL-EDGE

---

*Rule Discovery Pipeline — FORGE (Feature-Oriented Rule Generation Engine)*
*Versione 2.0 · Maggio 2026 · Parte di FORGE v1.0*
*Status: Draft · ⚠️ Documento tecnico di ricerca. Non costituisce consulenza finanziaria.*
