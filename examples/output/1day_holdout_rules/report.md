# Top 10 regole 1D — Validazione Hold-Out (ultimi 6 mesi censurati)

> **ERRATA (2026-09-08):** i PF di hold-out riportati sotto sono calcolati sui trade
> nominali sovrapposti e sono fuorvianti — il caso studiato in dettaglio (regola #1,
> 25 trade nominali da appena 5 episodi indipendenti) è nella diagnosi completa in
> **[`report_v2.md`](./report_v2.md)**, che sostituisce questa top 10 con due nuovi set
> di regole (una diversificata per ticker, una filtrata su PF walk-forward moderato
> 1.5-2.5) ri-verificati con una metodologia a livello di episodio. Vedi `report_v2.md`
> prima di usare i numeri di questo file.

**Data esecuzione:** 2026-09-08
**Pipeline:** FORGE (`forge()`, preset `"balanced"`, `timeframe="1D"`)
**Ticker analizzati:** 8 dataset `*_1DAY.csv` in `examples/data/`
**Metodo di validazione:** hold-out temporale — ultimi 6 mesi di ogni serie censurati durante la discovery, poi usati come test "completamente buio"

---

## 1. Obiettivo

Estrarre le migliori regole di trading discovery-driven dai dati giornalieri disponibili nel
repository, applicando un controllo aggiuntivo oltre alla walk-forward OOS interna di FORGE
(Modulo 3): un **hold-out temporale** in cui gli ultimi 6 mesi di ciascuna serie sono stati
rimossi *prima* di far girare `forge()`, e riutilizzati solo alla fine per testare — senza
alcun refit — le regole selezionate.

Una regola è **"selezionabile"** solo se, replicata con i suoi parametri esatti (soglia
dell'evento, direzione, holding period, sell_pct, modalità di entrata) sul periodo di
hold-out, **si attiva** (almeno un trade) **e produce Profit Factor ≥ 1**. In caso contrario
resta comunque loggata con un disclaimer esplicito, come richiesto.

## 2. Dati e split train / hold-out

| Ticker | Barre train | Range train | Barre hold-out | Range hold-out |
|---|---:|---|---:|---|
| BTCEUR | 1046 | 2023-03-14 → 2026-03-01 | 185 | 2026-03-02 → 2026-09-02 |
| ETHEUR | 1047 | 2023-03-14 → 2026-03-01 | 185 | 2026-03-02 → 2026-09-02 |
| COPPER.CMDUSD | 1605 | 2021-01-03 → 2026-03-01 | 158 | 2026-03-02 → 2026-09-02 |
| EURUSD | 1614 | 2021-01-03 → 2026-03-01 | 159 | 2026-03-02 → 2026-09-02 |
| E_Brent | 1316 | 2021-01-04 → 2026-02-27 | 132 | 2026-03-02 → 2026-09-02 |
| E_DAAX | 1601 | 2021-01-03 → 2026-03-01 | 158 | 2026-03-02 → 2026-09-02 |
| E_SandP-500 | 1607 | 2021-01-03 → 2026-03-01 | 158 | 2026-03-02 → 2026-09-02 |
| GBPUSD | 1613 | 2021-01-03 → 2026-03-01 | 159 | 2026-03-02 → 2026-09-02 |

Il cutoff è fissato a **6 mesi prima dell'ultima barra disponibile** (`open_dt.max() - 6 mesi`).
Il training set (M0→M3 di FORGE) non ha mai visto le barre successive al cutoff — coerente con
l'invariante #1 di FORGE (Event Discovery non osserva mai il forward return) esteso qui anche
al controllo economico di M3.

## 3. Pipeline e configurazione

Per ciascun ticker, sul solo train set:

1. `build_features()` → `candle_features()` → `lag_features()` per costruire la KPI Table
   (180 colonne circa per ticker, indicatori EMA/SMA/RSI/Bollinger + geometria candela + lag).
2. `forge_preset("balanced", timeframe="1D", asset=<ticker>)` per ottenere config M1/M2/M3
   internamente coerenti (verificato con `config_report()` prima del run: `"balanced"` è
   l'unico preset OK su tutti gli 8 ticker dopo la censura degli ultimi 6 mesi — `"sniper"` e
   `"sweep"` falliscono `oos_span_too_short` su BTCEUR/ETHEUR con la storia più corta).
3. `forge(train, ticker=..., timeframe="1D", ...)` — pipeline completa M0→M4, composizione a
   due passaggi di default (`two_pass_composition=True`), rotazione-null di default
   (`fast_null=True`).
4. Dalle regole con verdetto tradeable (`result.edges()`, cioè `EDGE`/`PARTIAL-EDGE`) si
   estrae `validated_rule` (espressione + parametri di backtest **fissati** in training).

Per la valutazione hold-out (Sezione 4), l'`EventCandidate` di ciascuna regola è stato
riapplicato sull'**intera serie** (train+hold-out concatenati, mai sulle sole barre nuove —
per non collassare le baseline rolling usate da alcuni pattern), e il backtest è stato
rieseguito con `run_backtest()` **usando esattamente gli stessi `BacktestParams` trovati in
training**, filtrato al solo periodo `[cutoff, fine serie]`. Nessun parametro è stato
ri-ottimizzato sull'hold-out.

## 4. Risultati per ticker

| Ticker | Regole tradeable (M3) | Sopravvissute hold-out | Tasso sopravvivenza |
|---|---:|---:|---:|
| E_Brent | 441 | 88 | 20.0% |
| BTCEUR | 109 | 30 | 27.5% |
| E_DAAX | 67 | 54 | **80.6%** |
| E_SandP-500 | 43 | 33 | 76.7% |
| ETHEUR | 24 | 8 | 33.3% |
| COPPER.CMDUSD | 19 | 16 | **84.2%** |
| EURUSD | 0 | 0 | — |
| GBPUSD | 0 | 0 | — |

**Totale: 703 regole tradeable valutate, 229 sopravvivono all'hold-out (32.6%).**

Note:
- **EURUSD e GBPUSD non hanno prodotto nessuna regola `EDGE`/`PARTIAL-EDGE`** in Modulo 3 col
  preset `"balanced"` — coerente con la maggiore efficienza/rumorosità dei cambi FX rispetto a
  crypto/indici/commodity su base giornaliera; nessuna regola quindi da validare per questi due
  ticker (0/0, non un fallimento dell'hold-out).
- **Tutte** le 474 regole non sopravvissute sono state scartate per lo stesso motivo:
  **Profit Factor hold-out < 1**. Nessuna è stata scartata per "non si attiva" — la frequenza
  di attivazione giornaliera è risultata sempre sufficiente sui 132-185 giorni di hold-out
  disponibili.
- Indici (E_DAAX, E_SandP-500) e commodity (COPPER) mostrano un tasso di sopravvivenza molto
  più alto (77-84%) di crypto (BTCEUR 27.5%, ETHEUR 33.3%) e di Brent (20%) — leggibile come
  un segnale di maggiore stabilità di regime nel periodo di hold-out (2026 H1-H2) per questi
  asset.

## 5. Le 10 migliori regole (sopravvissute all'hold-out)

Ranking: punteggio composito `√PF_train × PF_holdout(cap 10) × wf_consistency × ln(1+n_trade_holdout) × bonus_grade`,
poi deduplicate per (ticker, direzione, pattern strutturale) tenendo la variante di soglia
migliore — per evitare che 5 soglie leggermente diverse dello stesso indicatore occupassero
tutta la classifica.

| # | Ticker | Espressione | Dir. | Grade | PF train (n) | WF consist. | **PF hold-out (n, WR)** |
|---|---|---|---|---|---|---|---|
| 1 | E_DAAX | `zs_spread_close_ema25_48 < -1` | long | B | 5.93 (279) | 4/4 | **31.09 (25, 92%)** |
| 2 | E_DAAX | `pr_spread_close_ema12_48 < 0.125` | long | B | 5.71 (166) | 4/4 | **20.74 (17, 88%)** |
| 3 | E_DAAX | `zs_diffnorm_volume_sma03_sma25_48 > 1` | long | B | 4.81 (199) | 4/4 | **∞ (15, 100%)** |
| 4 | E_DAAX | `zs_diffnorm_volume_sma09_sma12_48 > 1` | long | C | 2.80 (192) | 4/4 | **132.88 (36, 94%)** |
| 5 | E_SandP-500 | `pr_ratio_close_ret03_ret06_48 < 0.145833` | long | C | 4.78 (158) | 4/4 | **26.78 (14, 86%)** |
| 6 | COPPER.CMDUSD | `diffnorm_volume_sma03_sma25 > 0.912647` | long | B | 3.16 (191) | 4/4 | **8.59 (32, 78%)** |
| 7 | E_DAAX | `zs_diffnorm_gap_range_pct_48 < -1.5` | long | C | 5.23 (91) | 4/4 | **102.26 (10, 90%)** |
| 8 | E_DAAX | `delta_diffnorm_close_sma03_sma09_1 < -0.499617` | long | C | 2.58 (77) | 4/4 | **20.03 (27, 85%)** |
| 9 | E_SandP-500 | `ratio_close_ret01_ret12 > 0.589114` | long | B | 3.83 (182) | 4/4 | **15.46 (11, 91%)** |
| 10 | COPPER.CMDUSD | `diffnorm_volume_sma03_sma25 > 0.912647 AND zs_diffnorm_volume_sma03_sma25_48 > 1` | long | A | 5.09 (154) | 4/4 | **∞ (6, 100%)** |

Tutte e 10 sono verdetto `PARTIAL-EDGE` in training (nessuna raggiunge `EDGE` pieno — coerente
col fatto che il rotation-null di default di FORGE cappa a `PARTIAL-EDGE` qualsiasi contratto
che vinca "solo" la lotteria della ricerca multi-test; vedi §7 per il significato operativo).
Tutte e 10 hanno **walk-forward consistency 4/4** (ogni fold di test OOS interno a M3 è stato
profittevole) e **temporal_stability PASS/WARN con t-test win-rate altamente significativo**
(p < 0.001 su tutte).

### Parametri operativi (fissati in training, mai ri-ottimizzati sull'hold-out)

| # | target_h (giorni) | sell_pct | buy_type | buy_delay | fee |
|---|---:|---:|---|---:|---:|
| 1 | 48 | 7.36% | market | 1 | 0.20% |
| 2 | 48 | 7.28% | market | 1 | 0.20% |
| 3 | 96 | 5.38% | market | 1 | 0.20% |
| 4 | 48 | 4.93% | market | 1 | 0.20% |
| 5 | 48 | 5.56% | market | 1 | 0.20% |
| 6 | 50 | 7.56% | market | 1 | 0.20% |
| 7 | 48 | 5.33% | market | 1 | 0.20% |
| 8 | 36 | 5.00% | market | 1 | 0.20% |
| 9 | 48 | 7.01% | market | 1 | 0.20% |
| 10 | 96 | 10.53% | market | 1 | 0.20% |

In tutte e 10 le regole, Rule Discovery (`entry_mode="auto"`) ha scelto il punto operativo
**market** rispetto al limite: nessuna variante limit ha raggiunto `fill_rate ≥ 0.80` in
sample, quindi il market entry — l'unico verdetto autoritativo — è il valore pubblicato.

## 6. Un esempio di overfitting che l'hold-out ha correttamente catturato

Non tutte le regole con statistiche di training eccellenti sopravvivono. Il caso più netto:

> **BTCEUR — `zs_spread_close_ema12_48 < -1.5`**: Profit Factor **7.99** in training (55
> trade), collassato a Profit Factor **0.02** sui 12 trade attivati nell'hold-out —
> un capovolgimento quasi totale, nonostante fosse la seconda miglior regola per PF di
> training tra le 703 valutate. Un caso da manuale di pattern che ha smesso di funzionare
> nel regime 2026 H1-H2, che sarebbe stato impossibile scartare guardando solo le metriche di
> training/walk-forward interno.

Questo è esattamente lo scenario per cui l'hold-out temporale aggiuntivo è stato richiesto:
la walk-forward OOS di Modulo 3 valida solo entro la finestra di training, e non protegge da
un cambio di regime *successivo* all'intera finestra osservata in discovery.

## 7. Regole scartate — log con disclaimer

Delle 703 regole tradeable valutate, **474 (67.4%) non sopravvivono** all'hold-out. Il log
completo (tutte e 703, con motivazione per le scartate) è esportato in
`full_rule_log_1day_holdout.pkl` (§9). Motivazione univoca in questo run:

- **474 / 474** scartate per `PF hold-out < 1` (nessuna per mancata attivazione).

Esempio rappresentativo (vedi anche §6):

| Ticker | Espressione | PF train (n) | PF hold-out (n) | Disclaimer |
|---|---|---|---|---|
| BTCEUR | `pr_close_ret_06_48 < 0.145833 AND zs_spread_close_ema12_48 < -1.5` | 13.09 (45) | 0.03 (10) | non sopravvive hold-out: PF hold-out 0.03 < 1 |
| BTCEUR | `zs_spread_close_ema12_48 < -1.5` | 8.00 (55) | 0.02 (12) | non sopravvive hold-out: PF hold-out 0.02 < 1 |
| COPPER.CMDUSD | `ratio_volume_sma03_sma96 < 0.537541` | 7.41 (111) | 0.15 (14) | non sopravvive hold-out: PF hold-out 0.15 < 1 |
| BTCEUR | `ratio_close_ret03_ret168 > 0.213578` | 7.56 (96) | 0.78 (17) | non sopravvive hold-out: PF hold-out 0.78 < 1 |

## 8. Limitazioni e avvertenze

- **Nessuna regola raggiunge il verdetto `EDGE` pieno** — solo `PARTIAL-EDGE`. La sopravvivenza
  all'hold-out qui misurata è quindi un controllo *aggiuntivo* rispetto a quanto FORGE stesso
  certifica, non un upgrade del verdetto ufficiale del framework.
- **Guadagni nominali, non "effettivi"**: `run_backtest` apre una posizione su ogni barra
  attiva senza controllo di stato flat — con `target_h` di 36-96 giorni, i trade delle regole
  #1-10 si sovrappongono nel tempo. I guadagni riportati assumono capitale sufficiente a
  finanziare le posizioni concorrenti (vedi `docs/manual-en.md`, sezione *Nominal vs. effective
  sample size*); il conteggio `n_effective` usato nei t-test di training (§5, colonna implicita)
  è già corretto per questo effetto, i PF/gain di hold-out riportati qui **non** lo sono.
- **Campioni di hold-out contenuti**: 6-36 trade per regola su 132-185 giorni. Un PF infinito
  (nessun trade perdente, righe #3 e #10) va letto con cautela extra quando `n` è a una cifra —
  è per questo che il ranking limita il contributo del PF a un cap di 10 nel punteggio.
- **Un solo hold-out, non walk-forward ripetuto**: la censura è stata applicata una volta sola
  (ultimi 6 mesi); non è stato fatto rolling dell'hold-out su più finestre per limitare il
  tempo di calcolo (~75 minuti di CPU totali per gli 8 ticker). Un rolling multi-finestra
  fornirebbe una stima più robusta della stabilità futura.
- **Preset unico (`"balanced"`)**: non sono stati esplorati `"sniper"`/`"sweep"`/`"burst"` per
  tutti i ticker — `"balanced"` è l'unico risultato internamente coerente su tutti gli 8
  ticker dopo la censura (verificato con `config_report()`, §3).
- **FORGE resta uno strumento di ricerca**: nessuna delle regole qui riportate implica
  esecuzione automatica — sono specifiche di backtest (espressione booleana + parametri
  d'ordine) da validare ulteriormente prima di qualsiasi uso operativo.

## 9. File esportati

Nella stessa cartella di questo report (`examples/output/1day_holdout_rules/`):

- **`top10_rules_1day_holdout.pkl`** — lista di 10 dict Python (nessuna dipendenza da
  `forgedge` per il caricamento) con espressione, parametri di backtest, statistiche di
  training e di hold-out, e disclaimer per ognuna delle 10 regole selezionate.
- **`full_rule_log_1day_holdout.pkl`** — log completo delle 703 regole tradeable valutate
  (229 sopravvissute + 474 scartate, ciascuna con motivazione).

```python
import pickle
top10 = pickle.load(open("top10_rules_1day_holdout.pkl", "rb"))
top10[0]["expression"], top10[0]["holdout_pf"], top10[0]["params"]
```
