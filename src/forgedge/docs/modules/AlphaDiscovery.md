# FORGE — Alpha Discovery Pipeline
> Secondo modulo della pipeline di ricerca quantitativa FORGE.
> Riceve gli **Event Candidate** da Event Discovery, **deriva dai dati** il
> target economico di ciascun evento (orizzonte, sell_pct, direzione), lo
> conferma out-of-sample e ne misura il potere predittivo. Seleziona i
> candidati con evidenza statistica sufficiente e li formalizza in un
> **Alpha Contract** — l'artefatto che Rule Discovery consuma per rispondere
> "Edge / Non-Edge".

---

## Indice

1. [Posizionamento nella Pipeline](#1-posizionamento-nella-pipeline)
2. [Il Contratto Alpha — Formato e Semantica](#2-il-contratto-alpha--formato-e-semantica)
3. [Step 1 — Derivare il Target di Valutazione](#3-step-1--derivare-il-target-di-valutazione)
4. [Step 2 — Analisi della Struttura di Mercato](#4-step-2--analisi-della-struttura-di-mercato)
5. [Step 3 — IC Measurement: Misurare il Potere Predittivo](#5-step-3--ic-measurement-misurare-il-potere-predittivo)
6. [Step 4 — Win Rate Analysis: Quantificare l'Edge](#6-step-4--win-rate-analysis-quantificare-ledge)
7. [Step 5 — Regime Sensitivity Analysis](#7-step-5--regime-sensitivity-analysis)
8. [Step 6 — Alpha Scoring e Ranking](#8-step-6--alpha-scoring-e-ranking)
9. [Step 7 — Compilare l'Alpha Contract](#9-step-7--compilare-lalpha-contract)
10. [Step 8 — Handoff e Risposta di Rule Discovery](#10-step-8--handoff-e-risposta-di-rule-discovery)
11. [Esempio Concreto: Valutazione di proto-RI_01](#11-esempio-concreto-valutazione-di-proto-ri_01)
12. [Generalizzazione ad Altri Ticker](#12-generalizzazione-ad-altri-ticker)
13. [Anti-pattern e Falsi Alpha](#13-anti-pattern-e-falsi-alpha)
    - [Multiple Testing e False Discovery Rate](#-overfitting-per-multiple-testing)

---

## 1. Posizionamento nella Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE FORGE                                │
│                                                                     │
│  ┌─────────────────────┐                                            │
│  │  EVENT DISCOVERY    │  Costruisce e filtra eventi booleani       │
│  │  Input:  Variable   │  senza guardare il forward return          │
│  │          Catalog    │                                            │
│  │  Output: Event      │                                            │
│  │          Candidates │                                            │
│  └──────────┬──────────┘                                            │
│             │ Event Candidates                                      │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  ALPHA DISCOVERY    │  ← questo modulo                          │
│  │  Input:  Event      │  Misura il potere predittivo               │
│  │          Candidates │  Seleziona i candidati validi              │
│  │  Output: Alpha      │  Formalizza l'Alpha Contract               │
│  │          Contract   │                                            │
│  └──────────┬──────────┘                                            │
│             │ Alpha Contract                                        │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  RULE DISCOVERY     │  Traduce in regola operativa               │
│  │  Input:  Alpha      │  Verifica eseguibilità (fee, fill rate,    │
│  │          Contract   │  PF, WR)                                   │
│  │  Output: Edge /     │                                            │
│  │          Non-Edge   │                                            │
│  └──────────┬──────────┘                                            │
│             │ Regola validata + parametri                           │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  RULE REGISTRY      │  Deduplica e gestisce il portfolio         │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  STRATEGIST (QHF)   │  Deploy in produzione                     │
│  └─────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Responsabilità di ciascun modulo

| Modulo | Responsabilità | Input | Output |
|---|---|---|---|
| **Event Discovery** | Generare tutti gli eventi booleani possibili dal catalogo e filtrare per struttura temporale | Variable Catalog | Event Candidates |
| **Alpha Discovery** | Misurare il potere predittivo degli Event Candidate rispetto al target economico | Event Candidates | Alpha Contract |
| **Rule Discovery** | Tradurre l'alpha in regola operativa e validarne l'eseguibilità | Alpha Contract | Edge/NonEdge + parametri |
| **Rule Registry** | Deduplicare, testare cross-ticker, esportare il portfolio di regole | Regole validate | Tabella piatta + report HTML |

### Principio di separazione

**Event Discovery** risponde alla domanda: _"questo evento ha struttura temporale stabile e non banale?"_
**Alpha Discovery** risponde alla domanda: _"questo evento predice il target economico con significatività statistica?"_
**Rule Discovery** risponde alla domanda: _"questo pattern è operazionalmente sfruttabile al netto di fee, fill e rischio?"_
**Rule Registry** risponde alla domanda: _"questa regola è distinta dalle esistenti e si generalizza su altri ticker?"_

Ogni modulo opera su un dominio distinto. Alpha Discovery **non genera nuovi eventi, non modifica soglie, non combina feature**. Riceve Event Candidate da Event Discovery e li misura rispetto al forward return.

---

## 2. Il Contratto Alpha — Formato e Semantica

L'Alpha Contract è l'artefatto di output: un documento strutturato che rappresenta
l'**interfaccia formale** tra Alpha Discovery e Rule Discovery.
Contiene esclusivamente l'Event Candidate selezionato e le sue misure statistiche
rispetto al target — nessun parametro operativo, nessuna soglia ottimizzata.

### Formato del contratto

```yaml
# ─────────────────────────────────────────────────────────────────────
# ALPHA CONTRACT
# ─────────────────────────────────────────────────────────────────────

alpha_id:          "ALPHA-ADAUSDC-1H-250101-003"
version:           "1.0"
discovery_date:    "2026-05-23"
status:            "HYPOTHESIS"   # HYPOTHESIS | VALIDATED | REJECTED | PARTIAL

# ── SCOPE ────────────────────────────────────────────────────────────
asset:             "ADAUSDC"
exchange:          "binance_spot"
timeframe:         "1H"
direction:         "long"

# ── ORIGINE ──────────────────────────────────────────────────────────
event_candidate_id: "EVT-close_rsi_25-ID×PR-P105-W096-P010"
event_expression:   "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"

# ── TARGET DERIVATO ──────────────────────────────────────────────────
# Alpha Discovery non riceve parametri economici: il target è derivato
# in-sample per ciascun evento e replicato out-of-sample (vedi Step 1).
derived_target:
  holding_period_h: 24         # h* = argmax(|vantaggio medio| / √h) sulla griglia
  sell_pct:         0.0420     # quantile MFE (mfe_quantile) a h*, barre attive IS
  direction:        "long"     # segno del vantaggio medio
  mean_advantage:   +0.0218    # vantaggio medio firmato (convenzione long)
  base_rate:        0.235      # win rate senza filtro a (h*, sell_pct*)
  advantage_by_h:   {1: 0.004, 6: 0.011, 12: 0.017, 24: 0.0218, 48: 0.019}
  t_stat_by_h:      {1: 1.8,   6: 3.1,   12: 4.6,   24: 5.34,   48: 4.2}
  score_by_h:       {1: 0.004, 6: 0.0045, 12: 0.0049, 24: 0.00445, 48: 0.0027}

# ── REPLAY OUT-OF-SAMPLE DEL TARGET DERIVATO ─────────────────────────
oos_validation:
  n_bars:          2690        # coda temporale mai usata nella derivazione
  n_activations:   97
  mean_advantage:  +0.0184     # orientato: positivo = favorevole al trade
  t_stat:          3.12
  p_value:         0.0011      # one-sided, attivi vs inattivi
  win_rate:        0.392
  base_rate:       0.241
  lift:            +0.151
  passed:          true        # diagnostica non bloccante — pesa sul voto A–D

# ── DESCRIZIONE ──────────────────────────────────────────────────────
pattern_family:    "mean_reversion_oversold"
description: >
  Quando il RSI a 25 periodi tocca un nuovo minimo relativo delle ultime
  96h (percentile rank < 10%) e il valore assoluto scende sotto 30.5
  (p10 storico), il prezzo tende a recuperare nelle successive 24h con
  win rate significativamente superiore al base rate.

economic_rationale: >
  Il RSI25 in oversold su 4 giorni indica esaurimento del momentum
  ribassista su scala swing. I venditori a breve termine sono esauriti,
  il book si svuota di sell orders e il rimbalzo avviene per ripristino
  della liquidità. La controparte è il retail panic seller che scarica
  ai minimi tecnici.

# ── EVIDENZA STATISTICA ──────────────────────────────────────────────
statistical_evidence:
  dataset_period:   "2025-01-01 to 2026-01-01"
  n_observations:   8964

  # IC della feature continua sottostante vs forward return
  underlying_feature:
    name:       "close_rsi_25"
    ic:         -0.0316
    p_value:    0.0029
    ic_rolling_stable: true   # IC non cambia segno nel tempo

  # Misure sull'Event Candidate binario
  event_stats:
    n_activations:  329
    win_rate:       0.415
    lift_vs_base:   +0.180
    fwd_return_mean: +0.0218
    cohens_d:       0.394
    t_stat:         5.34
    p_value:        0.000004

# ── REGIME SENSITIVITY ───────────────────────────────────────────────
regime_analysis:
  Q1_2025:
    regime:   "bear_correction"
    ic:       -0.0788
    p_value:  0.0003
    wr_cond:  0.535
    strength: "strong"

  Q2_2025:
    regime:   "mixed_volatile"
    ic:       -0.0456
    p_value:  0.0330
    wr_cond:  0.412
    strength: "moderate"

  Q3_2025:
    regime:   "uptrend"
    ic:       -0.0270
    p_value:  0.2049        # NON significativo
    wr_cond:  0.125
    strength: "negligible"

  Q4_2025:
    regime:   "bear_correction"
    ic:       -0.0772
    p_value:  0.0003
    wr_cond:  0.371
    strength: "strong"

regime_dependency:
  type:            "conditional"   # agnostic | conditional | specific
  active_regimes:  ["bear_correction", "mixed_volatile"]
  weak_regimes:    ["uptrend_continuous"]
  zero_months:     0

# ── HINTS PER RULE DISCOVERY ─────────────────────────────────────────
rule_discovery_hints:
  entry_mode:         "limit"
  buy_drop_pct_range: [0.006, 0.015]
  sell_pct_range:     [0.030, 0.050]
  target_h_range:     [12, 48]
  min_pf_target:      2.0
  min_wr_target:      0.70
  exclusion_conditions:
    - "Regime uptrend continuo (EMA9/25 > 1.01 su 1D)"
    - "Volume < 0.3× SMA25 (mercato illiquido)"

# ── ALPHA QUALITY SCORE ──────────────────────────────────────────────
alpha_score:
  ic_magnitude:    0.032
  lift:            0.180
  cohens_d:        0.394
  regime_breadth:  0.75
  composite_score: 0.68
  grade:           "B"          # A ≥ 0.75 | B ≥ 0.50 | C ≥ 0.25 | D < 0.25

# ── HANDOFF ──────────────────────────────────────────────────────────
handoff_status:            "PENDING_RULE_DISCOVERY"
rule_discovery_response:   null
```

### Campi obbligatori vs opzionali

| Campo | Obbligatorio | Note |
|---|---|---|
| `event_candidate_id`, `event_expression` | ✅ | Tracciabilità verso Event Discovery |
| `derived_target` (holding_period, sell_pct, direction, base_rate) | ✅ | Target derivato dai dati — baseline per Rule Discovery |
| `oos_validation` | ✅ | Conferma out-of-sample del target derivato |
| `description`, `economic_rationale` | ✅ | Leggibilità e falsifiability |
| `event_stats` (WR, lift, p-value) | ✅ | Evidenza statistica minima |
| `regime_analysis` | ✅ | Critico per capire quando l'alpha funziona |
| `rule_discovery_hints` | ✅ | Guida il modulo successivo |
| `underlying_feature.ic` | ⚠️ | Raccomandato — misura la forza del segnale continuo |
| `cohens_d` | ⚠️ | Raccomandato — effect size indipendente dalla scala |
| `alpha_score` | ⚠️ | Utile per priorizzare tra più candidati |

---

## 3. Step 1 — Derivare il Target di Valutazione

Alpha Discovery **non riceve parametri economici in input**: orizzonte,
sell_pct e direzione sono derivati dai dati, per ciascun Event Candidate.
Imporre un target a priori (es. "+4% in 24 barre") significherebbe misurare
ogni evento contro un'ipotesi arbitraria; derivarlo significa chiedere ai
dati *quando* e *quanto* l'evento sposta i rendimenti — e poi verificare
out-of-sample che la risposta non sia un artefatto della ricerca.

### 1.1 Split temporale IS/OOS

La tabella, ordinata cronologicamente, è divisa in due finestre
(`train_ratio`, default 0.7):

```
IN-SAMPLE  (primi 70%):  derivazione del target + tutte le misure (Step 1–6)
OUT-OF-SAMPLE (ultimi 30%): mai toccato dalla derivazione — replay del
                            target derivato (diagnostica per il voto A–D)
```

### 1.2 Derivazione per evento — vantaggio deflazionato per √h

Per ogni Event Candidate, su una griglia di orizzonti candidati
(`horizon_grid`, default 1–48 barre):

```python
# fwd_ret[h] = close.shift(-h) / close - 1   (convenzione long, firmato)
for h in horizon_grid:
    mean_adv[h] = fwd_ret[h][attive].mean()       # vantaggio medio firmato, IS
    score[h]    = abs(mean_adv[h]) / sqrt(h)      # deflazione della varianza

h_star         = argmax_h score[h]
mean_advantage = mean_adv[h_star]
direction      = "long" if mean_advantage > 0 else "short"

# sell_pct: quantile della Maximum Favorable Excursion a h* (barre attive IS)
# MFE_i = max escursione favorevole nelle h* barre successive alla barra i
sell_pct = max(quantile(MFE[attive], mfe_quantile), mfe_floor)
```

- **`h*`** massimizza il vantaggio medio deflazionato per `√h`. La deflazione
  compensa solo la crescita della varianza con l'orizzonte (∝ √h): a
  differenza dell'argmax sul |t-stat| — il cui denominatore si gonfia con h e
  penalizza sistematicamente gli orizzonti lunghi — questo criterio lascia
  emergere gli edge persistenti (es. un rimbalzo oversold che si realizza in
  24 barre).
- **`sell_pct`** è il quantile (`mfe_quantile`, default 0.5 = mediana, con
  floor `mfe_floor`, default 0.5%) della **Maximum Favorable Excursion** a h*
  sulle barre attive IS: la migliore escursione *raggiungibile* dal trade, non
  il rendimento medio a scadenza. È il candidato take-profit realistico che
  Rule Discovery raffinerà con la meccanica reale degli ordini.
- **`direction`** è il segno del vantaggio: un evento che anticipa ribassi
  diventa uno short candidate, senza configurazione manuale.

Il profilo completo (`advantage_by_h`, `t_stat_by_h`, `score_by_h`) viene
scritto nel contratto per trasparenza.

### 1.3 Target binario e base rate al target derivato

```python
# Estremo favorevole nelle successive h* barre (max per long, min per short)
fwd_ext = close.rolling(H_STAR).max().shift(-H_STAR)        # long
target  = (fwd_ext / close - 1 >= SELL_PCT).astype(float)

# Base rate: win rate senza alcun filtro a (h*, sell_pct*) — benchmark
base_rate = target[:split].mean()      # calcolato solo in-sample
```

> **Il base rate è il benchmark fondamentale.** Un Event Candidate con WR = 40%
> ha un lift di +16.5pp se il base rate è 23.5%, ma è inutile se il base rate
> è 38%. Poiché il target è per-evento, anche il base rate è per-evento.
> Tutti gli step successivi misurano la distanza dal base rate,
> non il WR assoluto.

### 1.4 Conferma out-of-sample del target derivato

Selezionare `h*` ottimizzando sulla griglia è comunque un'ottimizzazione
in-sample (horizon snooping): senza contromisure il vantaggio misurato è
sistematicamente sovrastimato. La contromisura è strutturale: il target
derivato viene replicato sulla coda OOS, che non ha avuto alcun ruolo nella
derivazione:

```
mean_advantage_oos  orientato (positivo = favorevole alla direzione derivata)
t-test one-sided    attivi vs inattivi sulla coda OOS
win_rate / base_rate / lift   a (h*, sell_pct*, direction*) su OOS
```

`passed = true` quando:

```
n_activations_oos >= min_oos_activations   (default 10)
mean_advantage_oos > 0                     (il segno si conferma)
p_value_oos < oos_max_p                    (default 0.10)
```

La conferma OOS **non è un gate di promozione**: una conferma mancata viene
registrata come diagnostica non bloccante (`[diagnostic] OOS weak …`) e pesa
sul voto A–D. Il contratto passa comunque al Modulo 3 — Rule Discovery, con
walk-forward e meccanica reale degli ordini, è l'unico giudice economico.

---

## 4. Step 2 — Analisi della Struttura di Mercato

Prima di interpretare le misure degli Event Candidate, verificare che il mercato
abbia le proprietà strutturali compatibili con la famiglia di alpha attesa.
Questo step non seleziona né genera eventi — fornisce il **contesto interpretativo**
per capire perché certi candidati funzionano e in quali condizioni.

### 2.1 Hurst Exponent — il mercato è mean-reverting?

```python
def hurst_exponent(ts, min_lag=2, max_lag=100):
    """
    H < 0.5  → mean-reverting  → candidati oversold/overbought avranno lift positivo
    H = 0.5  → random walk     → nessun edge strutturale atteso
    H > 0.5  → trending        → candidati momentum avranno lift positivo
    """
    lags = range(min_lag, max_lag)
    tau  = [np.sqrt(np.std(np.subtract(ts[l:], ts[:-l]))) for l in lags]
    reg  = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0] * 2.0
```

**Risultato su ADAUSDC 1H 2025:**

| Serie | H | Interpretazione |
|---|---|---|
| Prezzo close | **0.44** | Mean-reverting — i candidati oversold dovrebbero avere lift > 0 |
| Return 12h | **0.11** | Fortemente mean-reverting — inversione attesa entro 12–24h |

> H = 0.44 è il prerequisito strutturale che giustifica di promuovere
> Event Candidate della famiglia mean-reversion. Se H fosse > 0.55
> (trending), ci si aspetterebbe lift positivo dai candidati momentum,
> non da quelli oversold.

### 2.2 Autocorrelazione — a quale lag si inverte il mercato?

```python
rets = df['close_ret_12'].dropna()
for lag in [1, 2, 3, 6, 12, 24]:
    print(f"  lag={lag:2d}h:  ACF={rets.autocorr(lag=lag):+.4f}")
```

**Risultato su ADAUSDC 1H 2025:**

| Lag | ACF return 12h | Interpretazione |
|---|---|---|
| 1h | +0.9247 | Persistenza a brevissimo (inerziale) |
| 6h | +0.4859 | Momentum ancora presente |
| 12h | −0.0510 | Inversione debole inizia |
| 24h | −0.0819 | Anticorrelazione — inversione a 24h |

La struttura ACF è la **lettura interpretativa** dell'orizzonte derivato allo
Step 1: un `h* = 24` derivato per massima separazione è coerente con
un'inversione che si manifesta a 24h. Se la derivazione restituisse `h*`
incompatibili con la struttura ACF (es. h* lunghi su un mercato che inverte a
6h), il contrasto è un segnale d'allarme da annotare nel contratto.

---

## 5. Step 3 — IC Measurement: Misurare il Potere Predittivo

Per ogni Event Candidate ricevuto da Event Discovery, Alpha Discovery calcola
l'**Information Coefficient** (IC) della feature continua sottostante rispetto
al forward return. L'IC misura se esiste una relazione monotona tra la feature
e i rendimenti futuri — indipendentemente dall'evento booleano specifico.

### 3.1 Formula dell'IC

```
IC = Spearman rank correlation(feature_t, fwd_return_{t+h})

IC < 0  → feature alta → return futuro basso (segnale mean-reversion)
IC > 0  → feature alta → return futuro alto  (segnale momentum)
|IC| > 0.03  → debole ma potenzialmente utile
|IC| > 0.05  → moderato
|IC| > 0.10  → forte (raro sui mercati finanziari)
```

### 3.2 Calcolo per gli Event Candidate ricevuti

```python
# Input: lista di Event Candidates da Event Discovery
# Ogni EC ha: event_id, expression, source_feature, n_activations, ...

from scipy.stats import spearmanr

ic_results = []
for ec in event_candidates:
    feature = ec['source_feature']    # feature continua sottostante
    sub = df[[feature, 'fwd_return_24']].dropna()
    ic, p = spearmanr(sub[feature], sub['fwd_return_24'])
    ic_results.append({
        'event_id': ec['event_id'],
        'expression': ec['expression'],
        'feature':  feature,
        'ic':       ic,
        'abs_ic':   abs(ic),
        'p_value':  p,
    })

# Ordina per |IC| decrescente
ic_df = pd.DataFrame(ic_results).sort_values('abs_ic', ascending=False)
```

**Esempio — Event Candidates da `close_rsi_25` su ADAUSDC 1H 2025:**

| Event Candidate | Feature sottostante | IC | p-value |
|---|---|---:|---:|
| `close_rsi_25 < 30.5` | `close_rsi_25` | −0.0316 | 0.0029 |
| `close_rsi_25 < 30.5 AND pr_96 < 0.10` | `close_rsi_25` | −0.0316 | 0.0029 |
| `zscore_close_rsi_25_96 < -1.5` | `close_rsi_25` | −0.0316 | 0.0029 |
| `pr_close_rsi_25_96 < 0.10` | `close_rsi_25` | +0.0014 | 0.8921 |

> **Nota importante:** Event Candidate che derivano dalla stessa feature sottostante
> hanno lo stesso IC — l'IC misura la feature continua, non l'evento booleano.
> La differenza tra candidati della stessa feature emerge alla Win Rate Analysis
> (Step 4), non qui. L'IC è utile per confrontare candidati di feature diverse.

### 3.3 IC rolling nel tempo — il segnale è stabile?

```python
window_bars = 60 * 24  # 60 giorni su timeframe 1H

rolling_ic = []
for i in range(window_bars, len(df)):
    sub = df.iloc[i-window_bars:i][[feature, 'fwd_return_24']].dropna()
    ic, _ = spearmanr(sub[feature], sub['fwd_return_24'])
    rolling_ic.append({'date': df['open_dt'].iloc[i], 'ic': ic})

# Segnale stabile  → IC mantiene stesso segno nel tempo
# Segnale instabile → IC cambia segno → alpha probabilmente illusorio
```

### 3.4 Soglia di ammissione

Gli Event Candidate la cui feature sottostante ha `|IC| < 0.02` e `p > 0.05`
vengono scartati — la relazione tra feature e forward return è troppo debole
per giustificare la valutazione successiva.

---

## 6. Step 4 — Win Rate Analysis: Quantificare l'Edge

Per ogni Event Candidate che ha superato la soglia IC, misurare il potere
predittivo dell'evento **binario** rispetto al target. Questo è il momento
in cui si misura il lift effettivo dell'evento specifico, al netto di tutti
i parametri scelti da Event Discovery (trasformazione, finestra, soglia).

### 4.1 Misure sull'evento binario

```python
def evaluate_event_candidate(kpi_table, event_mask, target_col, fwd_return_col, base_rate):
    """
    Calcola le misure di qualità dell'Event Candidate.
    event_mask: Serie booleana — True quando l'evento è attivo
    """
    active   = kpi_table.loc[event_mask, fwd_return_col].dropna()
    inactive = kpi_table.loc[~event_mask, fwd_return_col].dropna()

    win_rate  = kpi_table.loc[event_mask, target_col].mean()
    lift      = win_rate - base_rate
    fwd_mean  = active.mean()

    # Effect size
    pooled_std = np.sqrt((active.std()**2 + inactive.std()**2) / 2)
    cohens_d   = (active.mean() - inactive.mean()) / pooled_std

    # Significatività statistica
    t_stat, p_val = stats.ttest_ind(active, inactive, alternative='greater')

    return {
        'n_activations': int(event_mask.sum()),
        'win_rate':      round(win_rate, 4),
        'lift':          round(lift, 4),
        'fwd_mean':      round(fwd_mean, 4),
        'cohens_d':      round(cohens_d, 4),
        't_stat':        round(t_stat, 4),
        'p_value':       round(p_val, 6),
    }
```

### 4.2 Profilo della feature sottostante — analisi diagnostica

Alpha Discovery analizza la forma del win rate lungo l'intera distribuzione
della **feature continua sottostante** — non dell'evento già binarizzato.
Questa analisi è **puramente diagnostica**: non serve a modificare la soglia
(già fissata da Event Discovery) ma a capire la forma del segnale e
identificare potenziali alpha simmetrici.

> **Distinzione importante:** l'evento `close_rsi_25 < 30.5 AND pr_96 < 0.10`
> ha WR = 41.5%. L'analisi per decili sotto mostra la WR di **tutte le barre**
> in ciascun decile di RSI25, indipendentemente dal pctrank. Sono due cose
> diverse: la prima misura l'evento composto, la seconda mostra la struttura
> della feature continua sottostante.

```python
# Analisi sulla feature continua — non sull'evento
kpi_table['q'] = pd.qcut(
    kpi_table['close_rsi_25'], q=10,
    labels=False, duplicates='drop'
)
quantile_profile = kpi_table.groupby('q').agg(
    win_rate = ('target', 'mean'),
    n        = ('target', 'count'),
    rsi_mean = ('close_rsi_25', 'mean'),
)
```

**Risultati su close_rsi_25 — ADAUSDC 1H 2025:**

| Decile | RSI25 medio | WR barr. nel decile | Note |
|---|---:|---:|---|
| Q0 (p0–p10) | 26.8 | 28.0% | Zona RSI25 < 30.5 — l'evento aggiunge il filtro pctrank (+13.5pp di lift) |
| Q1–Q8 | 40–60 | 20–25% | Zona neutrale |
| Q9 (p90–p100) | 66.2 | 29.2% | Potenziale alpha momentum separato |

**Lettura della forma:**

| Forma | Descrizione | Implicazione |
|---|---|---|
| **U** | Q0 e Q9 entrambi alti | Due alpha distinti: mean-reversion e momentum |
| **L** | Solo Q0 alto | Mean-reversion pura |
| **J** | Solo Q9 alto | Momentum puro |
| **Piatta** | Nessuna variazione | Feature non predittiva — scartare |

> **Per close_rsi_25 su ADAUSDC:** forma a U debole. Il Q0 mostra WR=28%
> sulla feature raw — il filtro pctrank dell'evento porta questa WR a 41.5%,
> aggiungendo +13.5pp di lift selezionando i momenti dove il RSI è basso
> **anche** nel contesto recente. Il Q9 segnala un potenziale alpha momentum
> separato da documentare negli hints del contratto per sessioni future.

### 4.3 Soglie diagnostiche (non bloccanti)

Le soglie alimentano il voto A–D ma **non scartano** il candidato — ogni
violazione viene scritta come `[diagnostic]` nel contratto:

```
lift      >= 0.08    (almeno +8pp sopra il base rate)
cohens_d  >= 0.15    (effect size almeno piccolo)
p_value   <  0.05    (significatività statistica / FDR)
n_activations >= 30  (base statistica minima)
```

Tutti i contratti con direzione determinata (long/short) passano al Modulo 3:
Rule Discovery, con la meccanica reale degli ordini, è l'unico giudice
economico. Solo i candidati senza vantaggio finito sulla griglia
(direzione indeterminata) restano `REJECTED`.

---

## 7. Step 5 — Regime Sensitivity Analysis

Un Event Candidate con buone misure aggregate potrebbe funzionare solo in
certi regimi di mercato. Prima di promuoverlo, misurare le sue performance
per regime usando la colonna `regime` già presente nella KPI Table —
prodotta dal **Market Context Module** all'inizio della sessione.

### 5.1 Leggere il regime dalla KPI Table

```python
# La colonna 'regime' è già calcolata dal Market Context Module
# Alpha Discovery la legge — non la ricalcola
regimes = kpi_table['regime'].cat.categories
# → ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
```

### 5.2 Misure per regime

```python
for regime_label in regimes:
    regime_mask  = kpi_table['regime'] == regime_label
    ec_in_regime = event_mask & regime_mask
    if ec_in_regime.sum() < 10: continue

    ic_r, p_r = spearmanr(
        kpi_table.loc[regime_mask, source_feature],
        kpi_table.loc[regime_mask, 'fwd_return_24']
    )
    wr_r = kpi_table.loc[ec_in_regime, 'target'].mean()
    print(f"{regime_label}: IC={ic_r:+.4f} p={p_r:.4f}  WR={wr_r:.1%}")
```

**Risultati su proto-RI_01 per regime:**

| Regime | IC | p-value | WR evento | Classificazione |
|---|---:|---:|---:|---|
| STRONG_BEAR | −0.079 | 0.0003 | 53.5% | ✅ Strong |
| BEAR | −0.046 | 0.033 | 41.2% | ✅ Moderate |
| NEUTRAL | −0.038 | 0.044 | 34.8% | ✅ Moderate |
| **BULL** | −0.027 | **0.205** | **18.4%** | ❌ Negligible |
| **STRONG_BULL** | −0.021 | **0.312** | **12.5%** | ❌ Negligible |

### 5.3 Classificazione della regime-sensitivity

| Tipo | Criteri | Implicazione per il contratto |
|---|---|---|
| **Agnostic** | IC significativo in tutti i regimi | Deploy diretto |
| **Conditional** | IC significativo in 2–3 regimi su 5 | Deploy con Regime Filter |
| **Specific** | IC significativo in 1 solo regime | Deploy solo se regime identificabile in real-time |
| **Broken** | IC non significativo in nessun regime | Scartare |

> **Per proto-RI_01:** Conditional. Valido in STRONG_BEAR, BEAR, NEUTRAL.
> Debole in BULL e STRONG_BULL — il mercato in uptrend continuo assorbe
> il rimbalzo RSI senza inversione significativa.
> La classificazione viene scritta nel campo `regime_dependency.type`
> del contratto.

---

## 8. Step 6 — Alpha Scoring e Ranking

Quando più Event Candidate superano la soglia di promozione, è necessario
priorizarli. Il Composite Alpha Score combina le misure dei passi precedenti.

### 6.1 Formula

```python
def alpha_score(ic_abs, lift, cohens_d, regime_breadth):
    """
    ic_abs:         |IC| della feature sottostante
    lift:           WR_evento - base_rate  (es. 0.18 = +18pp)
    cohens_d:       effect size dell'evento binario
    regime_breadth: frazione dei regimi con IC significativo [0, 1]
    """
    # Normalizzazione su valori tipici crypto 1H
    ic_norm      = min(ic_abs / 0.10, 1.0)
    lift_norm    = min(lift   / 0.30, 1.0)
    d_norm       = min(cohens_d / 0.80, 1.0)
    breadth_norm = regime_breadth

    weights = [0.25, 0.30, 0.25, 0.20]
    return sum(w*v for w,v in zip(weights, [ic_norm, lift_norm, d_norm, breadth_norm]))
```

### 6.2 Griglia di prioritizzazione

| Grade | Score | Lettura |
|---|---|---|
| **A** | ≥ 0.75 | Evidenza forte — priorità alta in Rule Discovery |
| **B** | 0.50–0.74 | Evidenza solida — priorità media |
| **C** | 0.25–0.49 | Evidenza moderata — esplorare con cautela |
| **D** | < 0.25 | Evidenza debole — comunque consegnato al Modulo 3, bassa priorità |

Il voto **non è un gate**: tutti i contratti A–D vengono consegnati a Rule
Discovery, che li ordina per priorità e li giudica economicamente.

**Esempio — proto-RI_01:**

```
ic_abs = 0.032   → ic_norm = 0.32
lift   = 0.180   → lift_norm = 0.60
cohens_d = 0.394 → d_norm = 0.49
regime_breadth = 0.75 (3/4 regimi significativi)

Score = 0.25×0.32 + 0.30×0.60 + 0.25×0.49 + 0.20×0.75 = 0.68 → Grade B
```

---

## 9. Step 7 — Compilare l'Alpha Contract

Raccogliere le misure degli step precedenti e compilare il contratto nel formato
definito nella Sezione 2. Il contratto referenzia **l'Event Candidate ID esatto**
ricevuto da Event Discovery — non crea nuove espressioni, non modifica soglie.

### Checklist di compilazione

- [ ] `event_candidate_id` copiato esattamente da Event Discovery
- [ ] `event_expression` copiata esattamente — nessuna modifica alle soglie
- [ ] `derived_target` compilato (h*, sell_pct, direction, base_rate, profili per h)
- [ ] `oos_validation` compilato — diagnostica non bloccante per il voto A–D
- [ ] `description` in forma narrativa
- [ ] `economic_rationale` con controparte identificata
- [ ] `underlying_feature.ic` con p-value
- [ ] `event_stats` (WR, lift, Cohen's d, p-value, n_activations)
- [ ] `regime_analysis` per ogni regime testato
- [ ] `regime_dependency.type` classificato
- [ ] `rule_discovery_hints` compilati con range di parametri
- [ ] `alpha_score` calcolato e grade assegnato
- [ ] `status` = "HYPOTHESIS"

---

## 10. Step 8 — Handoff e Risposta di Rule Discovery

### Il protocollo di handoff

```
Alpha Discovery:
  → Compila Alpha Contract
  → status = "HYPOTHESIS"
  → Passa a Rule Discovery

Rule Discovery:
  → Legge l'Alpha Contract
  → Esegue la Rule Discovery Pipeline
  → Compila rule_discovery_response nel contratto
  → Restituisce con status finale
```

### Formato della risposta di Rule Discovery

```yaml
rule_discovery_response:
  date:    "2026-05-23"
  verdict: "EDGE"      # EDGE | NON-EDGE | PARTIAL-EDGE

  # Solo se verdict = EDGE o PARTIAL-EDGE
  validated_rule:
    expression:      "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
    buy_type:        "limit"
    buy_drop_pct:    0.010
    sell_pct:        0.040
    target_h:        24
    fee:             0.002

  backtest_results:
    profit_factor:     3.17
    win_rate_pct:      0.814
    total_trades:      102
    expectancy_pct:    0.0155
    zero_months:       0

  statistical_validation:
    ttest_winrate_p:  0.0012
    deflated_sharpe:  1.31

  regime_constraints:
    deploy_in:   ["bear_correction", "neutral", "mixed"]
    avoid_in:    ["uptrend_continuous"]

  # Solo se verdict = NON-EDGE
  rejection_reasons: null

status: "VALIDATED"
```

### Verdetti possibili

**EDGE** — regola operazionalmente valida. PF ≥ 2, WR ≥ 70%, significatività statistica, distribuzione accettabile.

**PARTIAL-EDGE** — funziona con limitazioni. PF 1.5–2.0, o mesi vuoti, o instabilità parametrica. Deploy con vincoli espliciti.

**NON-EDGE** — il backtest non conferma l'alpha. Motivi tipici: fee troppo alte, fill rate basso, PF < 1.5 nonostante IC significativo. L'IC misura una relazione, non l'operabilità — il gap è normale.

---

## 11. Esempio Concreto: Valutazione di proto-RI_01

Traccia passo-passo di come Alpha Discovery riceve e valuta il candidato
`EVT-close_rsi_25-ID×PR-P105-W096-P010`, che diventerà la base di RI_01.

### Input ricevuto da Event Discovery

```yaml
event_id:    "EVT-close_rsi_25-ID×PR-P105-W096-P010"
expression:  "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
n_activations: 329
n_months:    12
zero_months: 0
mean_tpm:    27.4
consistency_gate: PASS
```

### Step 1 — Derivazione del target (in-sample)

```
Griglia orizzonti: 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48
score(h) = |vantaggio_medio(h)| / √h  massimo a h = 24  →  h* = 24
mean_advantage(24) = +2.18%  →  direction = long
sell_pct = mediana MFE(24h) barre attive IS = 0.042   (quantile configurabile)
base_rate a (24, 0.042, long) = 23.5%   (in-sample)
```

### Step 2 — Struttura di mercato

```
H(prezzo) = 0.44 → mean-reverting → candidati oversold attesi con lift > 0 ✅
ACF lag=24h: -0.0819 → inversione attesa a 24h ✅
Contesto compatibile con la famiglia mean-reversion dell'evento ricevuto.
```

### Step 3 — IC Measurement

```
Feature sottostante: close_rsi_25
IC vs fwd_return_24h = -0.0316
p-value = 0.0029  (**) → significativo
IC rolling: stabile, segno negativo costante in tutti i 12 mesi

→ Evento supera la soglia IC (|IC| = 0.032 > 0.02)
```

### Step 4 — Win Rate Analysis

```
Evento: close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10
N attivazioni:  329
Win rate:       41.5%
Lift:           +18.0pp vs base rate 23.5%
Cohen's d:      0.394
t-stat:         5.34
p-value:        0.000004

Profilo curva win rate (close_rsi_25 per decili):
  Q0 (RSI~34): WR=28.0% → lift=+4.5pp   (evento individuale debole)
  Q9 (RSI~66): WR=29.2% → lift=+5.7pp   (potenziale alpha momentum separato)
  Forma a U → segnalato per future sessioni di discovery

→ Nessuna diagnostica: lift 18pp >> soglia 8pp, Cohen's d = 0.394 >> 0.15
```

### Step 5 — Regime Sensitivity

```
Q1 (bear):    IC=-0.079  p=0.0003  WR=53.5%  → Strong ✅
Q2 (mixed):   IC=-0.046  p=0.033   WR=41.2%  → Moderate ✅
Q3 (uptrend): IC=-0.027  p=0.205   WR=12.5%  → Negligible ❌
Q4 (bear):    IC=-0.077  p=0.0003  WR=37.1%  → Strong ✅

→ Regime-conditional: valido in 3/4 regimi
→ Esclusione raccomandata: uptrend continuo
```

### Step 6 — Score

```
Score = 0.68 → Grade B
→ Priorità media → Alpha Contract compilato con nota di regime
```

### Step 7 — Alpha Contract compilato

```yaml
alpha_id:         "ALPHA-ADAUSDC-1H-250101-003"
event_candidate_id: "EVT-close_rsi_25-ID×PR-P105-W096-P010"
event_expression:  "close_rsi_25 < 30.5 AND pr_close_rsi_25_96 < 0.10"
status:           "HYPOTHESIS"
→ inviato a Rule Discovery
```

### Step 8 — Risposta di Rule Discovery

```
verdict:       EDGE
PF:            3.17   WR: 81.4%   T: 102   zero_months: 0
DSR:           1.31   (edge reale dopo correzione multiple testing)
status:        VALIDATED → deploy in Strategist
```

---

## 12. Generalizzazione ad Altri Ticker

### Cosa cambia per ogni asset

| Elemento | Trasferibile | Da ricalcolare |
|---|---|---|
| Struttura della pipeline | ✅ Identica | — |
| Formula IC e win rate analysis | ✅ Identica | — |
| Formato Alpha Contract | ✅ Identico | — |
| Base rate del target | ❌ | Dipende da volatilità e sell_pct dell'asset |
| H (Hurst exponent) | ❌ | Verificare compatibilità con la famiglia alpha |
| Parametri hints per Rule Discovery | ❌ | Dipendono da spread e volatilità dell'asset |

> **Nota:** Alpha Discovery non calibra soglie — quelle sono già fissate
> da Event Discovery e vengono ereditate nell'Alpha Contract senza modifica.
> Se per un nuovo asset gli Event Candidate ricevuti producono zero attivazioni
> o lift nullo, il problema è a monte: occorre avviare una nuova sessione FORGE
> su quel ticker, in modo che Event Discovery rigeneri il catalogo di eventi
> sulla distribuzione specifica dell'asset.

---

## 13. Anti-pattern e Falsi Alpha

### ❌ Modificare le soglie degli Event Candidate

```
SBAGLIATO: l'IC su close_rsi_25 suggerisce che 32 funziona meglio di 30.5
           → Alpha Discovery aggiusta la soglia a 32

CORRETTO:  Alpha Discovery misura il lift dell'evento ricevuto (soglia 30.5)
           e lo riporta nel contratto invariato.
           Se si vuole esplorare la soglia 32, bisogna tornare a Event Discovery
           e aggiungere p12 al Threshold Catalog — così viene generato
           un nuovo Event Candidate distinto da valutare in una nuova sessione.
```

### ❌ Generare nuove combinazioni AND in Alpha Discovery

```
SBAGLIATO: Alpha Discovery vede che close_rsi_25 e ema_ratio hanno entrambi
           IC significativo → combina i due eventi in un nuovo AND
           direttamente in questa sessione

CORRETTO:  Alpha Discovery documenta l'osservazione negli hints del contratto:
           "i candidati RSI e EMA hanno entrambi lift positivo —
           valutare la composizione AND in una sessione futura".
           La sessione successiva di Event Discovery può includere
           esplicitamente questa combinazione nel catalogo di AND Composition.
           Alpha Discovery poi valuta i candidati composti quando li riceve.
```

### ❌ Look-ahead bias nel target

```python
# SBAGLIATO
df['fwd_max'] = df['close'].rolling(24).max()   # include dati futuri!

# CORRETTO
df['fwd_max'] = df['close'].rolling(24).max().shift(-24)
```

### ❌ Confusione tra IC e operabilità

Un IC significativo non implica operabilità. Esempi di gap:

| IC significativo | Ma non operabile perché |
|---|---|
| IC = −0.04 su return 1h | Fee 0.4% > gain atteso per trade |
| IC = −0.06 su close_rsi_25 | Fill rate basso — il limite non viene eseguito |
| IC = −0.08 su N=15 trade | Coincidenza statistica — troppo pochi trade |

La distanza tra IC e operabilità è normale — è esattamente il compito di Rule Discovery colmarla.

### ❌ Overfitting per multiple testing

Alpha Discovery riceve da Event Discovery un numero elevato di Event Candidate —
potenzialmente 200 o più su un singolo asset. Testare ciascuno con soglia p < 0.05
senza correzione produce un numero atteso di falsi positivi pari a `N × 0.05`.
Con 200 candidati: circa 10 false promozioni anche in assenza di qualsiasi edge reale.

FORGE adotta il metodo **Benjamini-Hochberg (BH)** per controllare il
**False Discovery Rate (FDR)** — la proporzione attesa di falsi positivi
tra tutti i candidati promossi.

**Differenza rispetto a Bonferroni:**

| Metodo | Controllo | Effetto pratico |
|---|---|---|
| Bonferroni | FWER — probabilità di *almeno un* falso positivo | Molto conservativo — poche promozioni |
| Benjamini-Hochberg | FDR — proporzione di falsi positivi tra i promossi | Bilanciato — ammette al massimo `q%` di falsi |
| Soglia grezza p < 0.05 | Nessuno | ~5% dei test risulta positivo per caso |

**Come funziona BH:**

```python
def benjamini_hochberg(p_values, q):
    """
    p_values: lista di p-value ordinati per candidato
    q:        FDR target (es. 0.10 = max 10% falsi positivi tra i promossi)

    Restituisce: maschera booleana dei candidati da promuovere
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]

    # Soglia BH per ogni candidato: (i / m) × q
    bh_thresholds = (np.arange(1, m + 1) / m) * q

    # Trova il più grande i per cui p_i ≤ soglia_i
    below = sorted_p <= bh_thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)

    cutoff = np.where(below)[0].max()
    promoted = np.zeros(m, dtype=bool)
    promoted[sorted_idx[:cutoff + 1]] = True
    return promoted
```

**Il parametro `q`:**

Il valore di `q` (FDR target) è una scelta metodologica che determina
quanti falsi positivi FORGE è disposto ad ammettere tra i candidati promossi.

| `q` | Significato | Profilo |
|---|---|---|
| 0.05 | Max 5% falsi tra i promossi | Conservativo — poche promozioni, alta affidabilità |
| 0.10 | Max 10% falsi tra i promossi | Bilanciato — standard nella letteratura statistica |
| 0.20 | Max 20% falsi tra i promossi | Permissivo — più output, più cautela richiesta all'utente |

> **Nota implementativa:** il valore di default di `q` e la possibilità
> di configurarlo come parametro verranno definiti durante la realizzazione
> del modulo. La scelta dipende dal profilo dell'utente target e dal numero
> medio di candidati che FORGE genera sul dataset di riferimento.

### ❌ Data snooping — derivare il target e valutarlo sugli stessi dati

La derivazione di `h*` per massima separazione è un'ottimizzazione sulla
griglia di orizzonti: valutarla sugli stessi dati che l'hanno selezionata
gonfia sistematicamente le misure (horizon snooping). Per questo lo split
IS/OOS è interno al modulo:

```
IN-SAMPLE  (train_ratio, default 70%):
    derivazione del target + IC, win rate, regime analysis, BH-FDR
OUT-OF-SAMPLE (coda temporale, 30%):
    mai toccato dalla derivazione → replay del target derivato
    → la promozione richiede oos_validation.passed = true
```

Rule Discovery effettua comunque la propria validazione finale con la
meccanica reale degli ordini — la conferma OOS di Alpha Discovery protegge
la *derivazione del target*, non sostituisce il backtest.

---

*Alpha Discovery Pipeline — FORGE — Feature-Oriented Rule Generation Engine · Versione 2.0 · Maggio 2026*
*Status: Draft · Parte di FORGE v1.0 · Dipendenze: Event Discovery Module v2.0*
