# Regole 1D — Report v3: hold-out rolling su 3 finestre (18 mesi)

**Data esecuzione:** 2026-09-08
**Estende:** `report_v2.md` — qui la censura passa da 6 a **18 mesi**, suddivisi in **3 finestre
rolling non sovrapposte da 6 mesi ciascuna**, per verificare se una regola sopravvive a più di
un singolo regime di mercato invece che a uno solo (il limite esplicitamente segnalato in
`report_v2.md` §5).

---

## 1. Metodologia

Per ciascun ticker, il training (`forge()`, preset `"balanced"`) usa solo i dati fino a
**18 mesi prima della fine serie** (`2025-03-02`, uguale per tutti gli 8 ticker — stessa data
di fine serie). Le regole tradeable scoperte vengono poi ri-testate — **stessi parametri
esatti, nessun refit** — separatamente su tre finestre consecutive che coprono l'intero
periodo censurato:

| Finestra | Da | A | Note |
|---|---|---|---|
| W3 (più vecchia) | 2025-03-02 | 2025-09-02 | prima metà del periodo censurato |
| W2 | 2025-09-02 | 2026-03-02 | seconda metà |
| W1 (più recente) | 2026-03-02 | 2026-09-02 | la stessa finestra usata come hold-out unico in `report_v2.md` |

Per ogni finestra si applica la stessa verifica episodio-based di `report_v2.md` (PF calcolato
su un punto per episodio, non per trade nominale sovrapposto — vedi quel report per la
motivazione). Una regola **sopravvive a una finestra** se ha ≥ 2 episodi e PF episodio ≥ 1.
Il nuovo indicatore di questo report è **quante delle 3 finestre** una regola supera:
`n_windows_survived` da 0 a 3.

## 2. Il regime di ciascuna finestra — perché conta

| Ticker | W3 (mar-set 2025) | W2 (set 2025-mar 2026) | W1 (mar-set 2026) |
|---|---:|---:|---:|
| BTCEUR | +3.1% | **-41.5%** | +13.5% |
| ETHEUR | +52.4% | **-55.7%** | +20.3% |
| COPPER.CMDUSD | -0.2% | +30.2% | +9.7% |
| EURUSD | +12.5% | +1.1% | -0.9% |
| E_Brent | -4.6% | +5.9% | +22.5% |
| E_DAAX | +6.1% | +5.7% | +5.1% |
| E_SandP-500 | +8.3% | +6.0% | +11.2% |
| GBPUSD | +7.5% | +0.2% | +0.8% |

**W2 è un crollo del -41.5% su BTCEUR e -55.7% su ETHEUR** — un vero e proprio crash. Questo è
esattamente il "regime scuro" che un hold-out singolo di 6 mesi (come in `report_v1`/`v2`, che
copriva solo un periodo rialzista) non può rivelare. Indici (E_DAAX, E_SandP-500) e Brent
mostrano invece tre finestre tutte moderatamente positive — un regime molto più stabile — che
spiega perché le loro regole si dimostrano più robuste in §3.

## 3. Risultati per ticker

| Ticker | Regole tradeable | Sopravvivono **3/3** finestre | Sopravvivono ≥ 2/3 |
|---|---:|---:|---:|
| E_Brent | 421 | **94 (22.3%)** | 281 (66.7%) |
| E_DAAX | 30 | 1 (3.3%) | 16 (53.3%) |
| E_SandP-500 | 19 | 4 (21.1%) | 15 (78.9%) |
| COPPER.CMDUSD | 13 | 3 (23.1%) | 10 (76.9%) |
| BTCEUR | 61 | **0 (0.0%)** | 12 (19.7%) |
| ETHEUR | 19 | **0 (0.0%)** | 4 (21.1%) |
| EURUSD | 8 | 0 (0.0%) | 2 (25.0%) |
| GBPUSD | 0 | — | — |

**Nessuna regola BTCEUR o ETHEUR sopravvive a tutte e 3 le finestre** — coerente al 100% con
il crollo del -41/-56% in W2 di §2: qualunque regola prevalentemente `long` (il 90%+ del pool,
vedi `report_v2.md` §1.2) perde in un crash di quella entità, indipendentemente da quanto
buona fosse nelle altre due finestre. Indici e commodity, con regimi molto più stabili nelle 3
finestre, producono un pool di regole realmente robuste (94 su E_Brent da solo).

**Totale: 571 regole tradeable valutate (18 mesi di training, invece dei 703 di `report_v2.md`
su 6 mesi in meno di censura), 102 sopravvivono a tutte e 3 le finestre, 340 ad almeno 2 su 3.**

## 4. Le 10 regole selezionate — sopravvivono a tutte e 3 le finestre

Selezione: solo regole con `n_windows_survived == 3`, almeno una per ciascuno dei 4 ticker che
ne hanno prodotta almeno una (COPPER.CMDUSD, E_Brent, E_DAAX, E_SandP-500 — **BTCEUR, ETHEUR,
EURUSD e GBPUSD non hanno nessuna regola pienamente robusta da offrire**, non un'omissione).
Punteggio: `min(PF episodio delle 3 finestre, cap 5) × √(min episodi delle 3 finestre)` — usa
il **minimo**, non la media, per premiare la consistenza sul caso peggiore piuttosto che una
finestra eccezionale che compensa una mediocre.

| # | Ticker | Espressione | PF episodio (W3 → W2 → W1) | Episodi (W3 → W2 → W1) | PF minimo | target_h (gg) |
|---|---|---|---|---|---:|---:|
| 1 | COPPER.CMDUSD | `pr_bb_pct_b_close_20_48 < 0.145833 AND zs_pos_close_range12...` | 2.56 → ∞ → 1.88 | 5 → 6 → 7 | 1.88 | 5 |
| 2 | E_Brent | `ratio_close_high_lag1 < 0.962564` | 3.66 → 2.34 → 2.09 | 11 → 6 → 13 | 2.09 | 6 |
| 3 | E_DAAX | `zs_ratio_close_vol05_vol12_48 < -1 AND delta_ratio_close_vol05_vol96_3 < -0.379128` | 1.39 → 1.29 → 1.27 | 4 → 8 → 7 | 1.27 | 10 |
| 4 | E_SandP-500 | `delta_ratio_close_ret01_ret12_12 > 0.944154` | 1.90 → 1.09 → 1.52 | 15 → 19 → 17 | 1.09 | 12 |
| 5 | E_Brent | `spread_close_high_lag1 < -0.0374358` | 3.66 → 2.34 → 2.09 | 11 → 6 → 13 | 2.09 | 6 |
| 6 | E_Brent | `pos_close_range12 < 0.0493224` | ∞ → 1.93 → 4.17 | 8 → 7 → 7 | 1.93 | 6 |
| 7 | COPPER.CMDUSD | `pr_pos_close_range12_48 < 0.145833` | 1.36 → ∞ → 43.86 | 7 → 9 → 10 | 1.36 | 7 |
| 8 | E_SandP-500 | `delta_bb_pct_b_close_20_3 < -0.361134` | ∞ → 2.04 → ∞ | 3 → 7 → 3 | 2.04 | 80 |
| 9 | E_SandP-500 | `delta_ratio_close_ret01_ret12_6 < -0.986643` | 10.91 → 1.20 → ∞ | 18 → 20 → 7 | 1.20 | 48 |
| 10 | COPPER.CMDUSD | `delta_volume_sma_25_3 > 0.9548` | 1.31 → 8.10 → 2.43 | 3 → 7 → 11 | 1.31 | 14 |

**Nota sulle righe #1 e #2**: sono lo **stesso segnale espresso in due rappresentazioni
matematicamente equivalenti** (`ratio_close_high` e `spread_close_high` sono trasformazioni
monotone della stessa quantità sottostante, il rapporto/differenza tra `close` e l'`high` del
giorno precedente) — non due scoperte indipendenti. Lasciate entrambe per trasparenza, ma vanno
contate come un solo segnale, non due, quando si valuta la diversità del set.

Rispetto ai set di `report_v2.md`, questo set ha **PF molto più contenuti e credibili
(1.09–2.09 nel caso peggiore, contro i 5–135 di v1/v2)** e **holding period molto più corti**
(5–14 giorni per 8 delle 10 regole, contro 36–96 giorni di v1) — entrambi gli effetti secondari
attesi del filtro di robustezza multi-regime: le regole sopravvissute non dipendono da un trend
lungo e ininterrotto per funzionare.

## 5. Confronto con i report precedenti

| | v1 (hold-out singolo 6 mesi) | v2 (stesso hold-out, verifica episodio) | **v3 (rolling 18 mesi, 3 finestre)** |
|---|---|---|---|
| Censura in training | 6 mesi | 6 mesi | **18 mesi** |
| Regole tradeable valutate | 703 | 703 | 571 |
| Regole "robuste" | 229 (PF trade-based ≥1) | 431 (PF episodio ≥1, 1 finestra) | **102 (PF episodio ≥1, tutte e 3 le finestre)** |
| PF tipico delle migliori 10 | 8.6–9999 | 1.5–135 | **1.09–2.09** |
| Copertura regimi | 1 finestra (rialzista) | 1 finestra (rialzista) | **3 finestre, incluso un crash -41/-56% su crypto** |
| BTCEUR/ETHEUR robuste? | Sì (30/8 regole "sopravvivono") | Sì (28/4 regole) | **No — 0 regole sopravvivono a tutte e 3 le finestre** |

Il confronto rende esplicito il punto sollevato inizialmente: le regole di v1/v2 che
sembravano ottime lo erano *in quella specifica finestra*. Sottoposte a un regime diverso (il
crash di W2), le regole BTCEUR/ETHEUR non generalizzano affatto — un risultato che un hold-out
singolo, per quanto "completamente buio", non può mostrare per costruzione.

## 6. Limitazioni

- **3 finestre non sono un numero enorme di regimi** — sufficienti a rivelare che una regola
  non sopravvive a un crash, ma non a stimare con precisione una probabilità di sopravvivenza
  futura. Un vero walk-forward rolling con finestre più numerose (es. mensili anziché
  semestrali) darebbe una stima più fine, al costo di un tempo di calcolo molto maggiore (qui
  già ~95 minuti di CPU per gli 8 ticker con 3 finestre).
- **Il training resta fisso alle stesse 3 finestre per tutti i ticker** (stessa data di fine
  serie) — non è stato fatto un rolling anche del training stesso (walk-forward completo con
  ri-discovery per ogni finestra), che sarebbe il passo naturale successivo ma richiede N volte
  il tempo di calcolo già speso.
- **Tutte le 10 regole selezionate restano `long`** — nessuna regola `short` è risultata
  sufficientemente robusta da comparire nel set, coerente con quanto osservato in
  `report_v2.md` §1.2 (le regole short sopravvivono molto meno spesso in generale).
- Valgono le limitazioni generali già discusse in `report_v1`/`v2`: nessuna regola raggiunge
  `EDGE` pieno (solo `PARTIAL-EDGE`), i guadagni restano nominali (trade sovrapposti), e FORGE
  resta uno strumento di ricerca, non di esecuzione.

## 7. File esportati

In `examples/output/1day_holdout_rules/`:

- **`setC_rolling18m_1day_holdout.pkl`** — le 10 regole di §4 (dict Python puri).
- **`full_rule_log_rolling18m.pkl`** — log delle 571 regole valutate con l'esito per ciascuna
  delle 3 finestre e il conteggio `n_windows_survived`.
