# Regole 1D — Report v2: diagnosi PF gonfiati + Set A/B ri-verificati

**Data esecuzione:** 2026-09-08
**Supersede/corregge:** `report.md` (v1) nella stessa cartella — i numeri di PF hold-out
riportati in v1 erano tecnicamente corretti ma **fuorvianti**: questo report ne spiega il
motivo e fornisce due nuove selezioni di regole con una verifica più onesta.

---

## 1. Perché i PF di hold-out del primo report erano troppo alti

Il sospetto era fondato. Non è un bug di calcolo, ma la combinazione di due effetti che il
primo report non isolava:

### 1.1 — Trade nominali fortemente sovrapposti ("episodi" vs "trade")

`target_h` (holding period) delle regole migliori era spesso di **36-96 giorni**, contro una
finestra di hold-out di soli **132-185 giorni**. Con un evento che si attiva su barre
consecutive, questo produce decine di posizioni aperte contemporaneamente sullo stesso
movimento di prezzo — non scommesse indipendenti.

Esempio concreto, la regola #1 del report v1 (`E_DAAX — zs_spread_close_ema25_48 < -1`):

| Metrica | Valore |
|---|---|
| Trade nominali in hold-out | 25 |
| **Episodi realmente indipendenti** (cluster di attivazione consecutiva) | **5** |
| PF calcolato sui 25 trade nominali | 31.09 |
| PF calcolato su 1 osservazione per episodio (media dei trade dello stesso episodio) | ∞ (5/5 episodi vincenti) |

I 25 "trade" derivano da appena 5 episodi: 12 trade nel cluster di marzo (stesso segnale
attivo per ~2 settimane consecutive), altri nel cluster di giugno. Il PF=31 non misura 25
scommesse — ne misura essenzialmente 5, ciascuna contata 5-6 volte.

### 1.2 — Regime di mercato fortemente direzionale nella finestra di hold-out

La finestra di hold-out (2026-03-02 → 2026-09-02) è stata un rally quasi ininterrotto su
**tutti** gli asset con regole tradeable:

| Ticker | Var. % close nel periodo di hold-out |
|---|---:|
| BTCEUR | +13.4% |
| E_SandP-500 | +11.7% |
| COPPER.CMDUSD | +10.6% |
| E_DAAX | +5.2% |

**652 delle 703 regole tradeable (92.8%) sono `direction="long"`** — coerente con quello che
Event/Alpha Discovery hanno trovato profittevole *in training* — e in un semestre
prevalentemente rialzista quasi ogni segnale long, indipendentemente dalla sua vera qualità
predittiva, tende a chiudere in target. Prova diretta:

| Direzione | N regole | Tasso sopravvivenza (episodio-based) | PF episodio medio (sopravvissute) |
|---|---:|---:|---:|
| long | 652 | 65.2% | 3.43 |
| short | 51 | 11.8% | 1.79 |

Le regole **short** — che in un mercato rialzista dovrebbero perdere sistematicamente — sopravvivono
molto meno spesso (11.8% contro 65.2%) e con PF nettamente più basso quando sopravvivono. Questo
conferma che gran parte del PF elevato osservato in v1 riflette **l'allineamento fortuito fra
direzione della regola e regime del periodo di hold-out**, non necessariamente un edge robusto
a qualunque regime.

**Conclusione**: i calcoli non erano sbagliati, ma il PF sui trade nominali sovrapposti in
una finestra di 6 mesi a trend forte è una metrica ottimisticamente distorta. Questo report
la sostituisce con una verifica a livello di episodio (§2) e propone due nuovi set di regole
più difendibili (§3, §4).

## 2. Metodologia rivista — verifica a livello di episodio

Per ogni regola, i trade nominali di hold-out sono raggruppati in **episodi** (cluster di
barre attive consecutive, gap ≤ 1 — stessa definizione di `forgedge.episodes` usata
internamente da Rule Discovery). All'interno di ogni episodio si calcola il rendimento medio
dei trade che lo compongono: un solo punto dato per episodio.

- `PF_episodio` = somma rendimenti medi degli episodi vincenti / |somma rendimenti medi degli episodi perdenti|
- **Sopravvive** se: ≥ 2 episodi **e** `PF_episodio ≥ 1` (un PF infinito richiede almeno 3
  episodi vincenti per essere accettato, altrimenti è "troppo pochi episodi per valutare").

Effetto sulla popolazione delle 703 regole tradeable:

| | Trade-based (v1) | Episodio-based (v2) |
|---|---:|---:|
| Sopravvissute | 229 (32.6%) | 431 (61.3%) |

Il numero di sopravvissute *aumenta* con la verifica per episodio (229→431): la volatilità
trade-per-trade nasconde episodi complessivamente vincenti; il punto qui non è "quante
sopravvivono" ma **quanto è affidabile il numero di PF riportato per ciascuna**, ora basato
su un conteggio di scommesse realmente indipendenti (mediana 10 episodi contro mediana 18
trade nominali del report v1).

Per ticker:

| Ticker | Tradeable | Sopravvissute v1 (trade) | Sopravvissute v2 (episodio) |
|---|---:|---:|---:|
| BTCEUR | 109 | 30 (27.5%) | 28 (25.7%) |
| COPPER.CMDUSD | 19 | 16 (84.2%) | 16 (84.2%) |
| ETHEUR | 24 | 8 (33.3%) | 4 (16.7%) |
| E_Brent | 441 | 88 (20.0%) | 293 (66.4%) |
| E_DAAX | 67 | 54 (80.6%) | 52 (77.6%) |
| E_SandP-500 | 43 | 33 (76.7%) | 38 (88.4%) |

## 3. Set A — 10 regole, almeno una per ticker, minima sovrapposizione temporale

Selezione richiesta: almeno una regola per ciascuno dei 6 ticker con regole tradeable
(EURUSD e GBPUSD restano a 0 — nessuna regola `EDGE`/`PARTIAL-EDGE` trovata in M3 per
nessuno dei due, indipendentemente dall'hold-out; non inclusi "se possibile" perché non
esiste nulla da includere). Dove sono state aggiunte più regole per lo stesso ticker (per
arrivare a 10), è stata verificata la sovrapposizione temporale reciproca — quota di giorni
di hold-out coperti in comune da due regole (indice di Jaccard sulle giornate coperte da
almeno un trade) — scartando candidate con overlap > 35% rispetto a quanto già scelto sullo
stesso ticker.

| # | Ticker | Espressione | Grade | PF train (n) | WF oos PF (M3) | **Episodi hold-out** | **PF episodio** | WR episodio | Overlap max (stesso ticker) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | BTCEUR | `delta_ratio_close_ema03_ema12_3 < -0.0185241` | C | 3.80 (123) | 1.27 | 6 | 6.47 | 83% | 0.28 |
| 2 | COPPER.CMDUSD | `delta_volume_sma_25_3 > 0.93338` | B | 2.98 (198) | 3.06 | 10 | 15.06 | 90% | 0.22 |
| 3 | ETHEUR | `ratio_open_close_lag1 crosses_below 0.99988` | C | 3.53 (62) | 1.37 | 23 | 1.50 | 70% | 0.00 |
| 4 | E_Brent | `delta_ratio_close_ema12_ema96_1 < -0.00421629` | B | 1.73 (162) | 1.64 | 7 | 4.92 | 86% | 0.28 |
| 5 | E_DAAX | `delta_diffnorm_close_sma03_sma09_1 < -0.499617` | C | 2.58 (77) | 2.28 | 11 | 67.55 | 91% | 0.34 |
| 6 | E_SandP-500 | `pr_ratio_close_ret03_ret06_48 < 0.145833` | C | 4.78 (158) | 2.78 | 12 | 21.98 | 83% | 0.00 |
| 7 | COPPER.CMDUSD | `zs_diffnorm_volume_sma03_sma25_96 > 1` | A | 4.09 (191) | 4.01 | 4 | ∞ | 100% | 0.22 |
| 8 | E_DAAX | `delta_diffnorm_close_mdd12_mdd24_1 < -0.0253197` | B | 1.64 (183) | 1.54 | 8 | 3.47 | 88% | 0.34 |
| 9 | BTCEUR | `delta_ratio_close_ema09_ema12_1 < -0.00134449 AND delta_diffnorm_close_rsi14_rsi25_12 < -1.36832` | C | 4.85 (51) | 1.59 | 5 | 9.12 | 80% | 0.28 |
| 10 | E_Brent | `delta_upper_wick_12 > 0.372947` | B | 1.86 (98) | 1.42 | 8 | 2.43 | 75% | 0.28 |

Tutte e 10 direzione `long` (coerente con §1.2: nel pool sopravvissuto le regole long
dominano nettamente). La regola #3 (ETHEUR) è la più robusta in termini di campione — 23
episodi indipendenti — pur con il PF più contenuto (1.50) del gruppo: un trade-off
"più dati, meno spettacolare" tipico quando si privilegia la numerosità campionaria.
La #7 (COPPER, PF infinito) ha solo 4 episodi: da leggere con cautela nonostante il filtro
di accettazione (≥3 episodi vincenti).

## 4. Set B — 10 regole con PF walk-forward (M3) in [1.5, 2.5], ri-verificate in hold-out

Filtro sulla popolazione delle 703 regole tradeable: `walk_forward.oos_summary.profit_factor`
(il PF sui fold di test out-of-sample **interni al training di Modulo 3** — out-of-sample
rispetto alla selezione dei parametri, ma comunque interno alla finestra pre-censura) compreso
tra 1.5 e 2.5. Sono un range di PF "moderato", non estremo, per verificare se regole meno
appariscenti in training generalizzano meglio sull'hold-out reale.

346 regole cadono in questo intervallo; 227 sopravvivono alla ri-verifica episodio-based in
hold-out (§2). Le 10 selezionate (punteggio = PF episodio (cap 4) × √episodi × wf_consistency,
con un tetto massimo di 3 per ticker per non farlo collassare su un solo asset):

| # | Ticker | Espressione | WF oos PF (M3, n) | **Episodi hold-out** | **PF episodio** | WR episodio |
|---|---|---|---|---:|---:|---:|
| 1 | E_SandP-500 | `delta_ratio_gap_upper_wick_1 < -0.00482201` | 1.70 (133) | 20 | 5.80 | 75% |
| 2 | E_DAAX | `delta_diffnorm_close_sma03_sma09_1 < -0.499617` | 2.28 (112) | 11 | 67.55 | 91% |
| 3 | E_DAAX | `zs_diffnorm_volume_sma09_sma12_48 > 1` | 2.24 (212) | 11 | 135.01 | 91% |
| 4 | E_SandP-500 | `delta_ratio_gap_upper_wick_3 < -0.00702697` | 2.21 (111) | 18 | 5.33 | 78% |
| 5 | COPPER.CMDUSD | `diffnorm_volume_sma09_sma12 > 1.21359` | 2.09 (139) | 10 | 4.90 | 70% |
| 6 | E_DAAX | `delta_ratio_volume_sma03_sma25_1 > 0.411509` | 2.40 (97) | 10 | 19.66 | 90% |
| 7 | E_SandP-500 | `ratio_close_ret01_ret12 > 0.589114` | 2.48 (183) | 8 | 9.59 | 88% |
| 8 | E_Brent | `zs_spread_close_sma09_168 < -1` | 1.60 (164) | 9 | 3.53 | 89% |
| 9 | E_Brent | `delta_ratio_close_ema12_ema96_1 < -0.00421629` | 1.64 (163) | 7 | 4.92 | 86% |
| 10 | E_Brent | `spread_close_ema09 < -0.0227804` | 1.83 (177) | 7 | 3.74 | 86% |

**Osservazione importante**: anche partendo da un PF walk-forward *moderato* (1.5-2.5, non
i valori estremi del report v1), il PF osservato in hold-out resta comunque molto alto
(3.5-135). Questo conferma la diagnosi di §1.2: l'inflazione **non nasce dall'aver scelto in
partenza regole con PF di training estremo** — nasce dal regime fortemente direzionale della
finestra di hold-out stessa, che premia qualunque regola long sufficientemente attiva,
indipendentemente da quanto "moderata" fosse la sua statistica di selezione. Nessun taglio
del preset di selezione risolve questo effetto: serve una finestra di hold-out che copra
anche un regime laterale o ribassista per una verifica davvero probante (vedi §5).

## 5. Limitazioni (aggiornate rispetto a v1)

- **Il regime della finestra di hold-out domina il risultato.** Con un rally del 5-13% su
  quasi tutti gli asset testati, qualunque insieme di regole prevalentemente long ottiene PF
  elevati in questa specifica finestra. Non è possibile, con un solo hold-out di 6 mesi,
  distinguere "regola con edge robusto a più regimi" da "regola che semplicemente compra e
  il mercato sale". Le regole short, che *dovrebbero* soffrire in questo regime e infatti
  soffrono (§1.2), sono la controprova più diretta che l'effetto è di regime.
- **Anche la verifica episodio-based resta un solo campione temporale.** Un hold-out rolling
  su più finestre (es. 6 finestre di hold-out da 1 mese ciascuna, o walk-forward ripetuto
  sull'intera serie) darebbe una stima molto più difendibile della stabilità cross-regime —
  non eseguito qui per limiti di tempo di calcolo.
- **`PF_episodio` resta comunque calcolato su poche osservazioni** (mediana 10, minimo 2 per
  definizione di sopravvivenza) — un t-test o un intervallo di confidenza esplicito sui
  rendimenti per episodio (non presente in questo report) sarebbe il passo naturale successivo
  prima di qualunque uso operativo.
- **`fee=0.2%` per lato e nessuno slippage modellato oltre l'esecuzione a mercato** — invariato
  rispetto a v1.
- Valgono ancora le limitazioni generali di v1 (§8 del report originale): FORGE resta uno
  strumento di ricerca, nessuna esecuzione automatica, nessuna regola qui raggiunge il
  verdetto `EDGE` pieno (solo `PARTIAL-EDGE`).

## 6. File esportati

In `examples/output/1day_holdout_rules/`:

- **`setA_diversificata_1day_holdout.pkl`** — le 10 regole del Set A (§3), dict Python puri.
- **`setB_wf_pf_1.5_2.5_1day_holdout.pkl`** — le 10 regole del Set B (§4), dict Python puri.
- **`full_rule_log_v2_episode_based.pkl`** — log delle 703 regole con **entrambe** le
  metriche (trade-based v1 e episodio-based v2) per confronto diretto, disclaimer incluso per
  le non sopravvissute.

Ogni dict include sia le metriche nominali (`holdout_pf_nominal`, calcolate come in v1) sia
quelle episodio-based (`holdout_pf_episode`) per permettere il confronto diretto tra le due
metodologie.
