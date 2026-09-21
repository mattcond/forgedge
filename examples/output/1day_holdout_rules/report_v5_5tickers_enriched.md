# Report v5: indicatori arricchiti (MACD + ATR/NATR) sui 5 ticker restanti

**Data esecuzione:** 2026-09-08
**Segue:** `report_v3_rolling.md` (hold-out rolling 18 mesi, 3 finestre) e `report_v4_btc_eth_eurusd_attempt.md`
(stesso arricchimento indicatori, applicato lì a BTCEUR/ETHEUR/EURUSD). Qui lo stesso trattamento —
KPI Table con MACD e ATR/NATR abilitati (disattivati di default in `DEFAULT_CONFIG`, 187 colonne
contro le 180 originali) — viene applicato ai 5 ticker rimanenti: **COPPER.CMDUSD, E_Brent, E_DAAX,
E_SandP-500, GBPUSD**. A differenza di BTC/ETH/EURUSD, questi 5 ticker (tranne GBPUSD) avevano già
prodotto regole robuste 3/3 in `report_v3_rolling.md` — qui la domanda è se l'arricchimento ne
aumenta numero o qualità, non se li sblocca da zero.

Preset usato: `"balanced"` per tutti e 5 (stesso preset di v3, per isolare l'effetto dei soli
indicatori). Metodologia di validazione: identica a v3 — censura 18 mesi, 3 finestre rolling da 6
mesi (W3 2025-03-02→2025-09-02, W2 2025-09-02→2026-03-02, W1 2026-03-02→2026-09-02), PF su base
episodio, sopravvivenza = PF episodio ≥ 1 e ≥2 episodi in ciascuna finestra.

---

## 1. Risultati aggregati — confronto v3 (indicatori base) → v5 (arricchiti)

| Ticker | Tradeable (v3→v5) | Sopravvivono 3/3 (v3→v5) | Sopravvivono ≥2/3 (v3→v5) |
|---|---:|---:|---:|
| E_Brent | 421 → 397 | 94 (22.3%) → **97 (24.4%)** | 281 (66.7%) → 270 (68.0%) |
| E_DAAX | 30 → 28 | 1 (3.3%) → **1 (3.6%)** | 16 (53.3%) → 14 (50.0%) |
| E_SandP-500 | 19 → 21 | 4 (21.1%) → **4 (19.0%)** | 15 (78.9%) → 18 (85.7%) |
| COPPER.CMDUSD | 13 → **40** | 3 (23.1%) → **3 (7.5%)** | 10 (76.9%) → 9 (22.5%) |
| GBPUSD | 0 → 0 | — → **—** | — → — |

**Lettura ticker per ticker:**

- **E_Brent, E_DAAX, E_SandP-500**: numeri sostanzialmente invariati (differenze di 1-4 regole,
  dentro il rumore di ricerca stocastica tra run). Nessun miglioramento sistematico.
- **COPPER.CMDUSD**: il pool di regole tradeable triplica (13→40), ma il numero assoluto di
  sopravvissute 3/3 resta identico (3) — quindi la percentuale *peggiora* (23.1%→7.5%). Il motivo
  è direzionale: **29 dei 40 candidati arricchiti usano feature ATR/NATR, e sono quasi tutti
  `short`** (31/40 short contro una minoranza `long` nell'originale) — nessuno di questi nuovi
  candidati short sopravvive alle 3 finestre. Le 3 regole 3/3 restano **le stesse identiche
  espressioni** già trovate in v3 (`pr_bb_pct_b_close_20_48 < ...`, `pr_pos_close_range12_48 <
  0.145833`, `delta_volume_sma_25_3 > 0.9548`) — non usano MACD/ATR.
- **GBPUSD**: **0 edge anche con indicatori arricchiti** (`forge()` termina con 1874 candidati
  promossi in M2 ma 0 archi in M3, identico esito nullo di v3). L'assenza di regole non è un
  problema di indicatori mancanti — è un limite più profondo (probabilmente nei gate di
  composizione o nella dinamica della serie stessa) fuori dallo scopo di questa leva.

## 2. Il risultato interessante: E_Brent guadagna regole NATR-based genuinamente nuove

A differenza di COPPER.CMDUSD, in **E_Brent 4 regole tra le 97 sopravvissute 3/3 usano
esplicitamente feature NATR** (indisponibili in v3) — non erano quindi semplicemente le stesse
regole ritrovate:

| Espressione | PF episodio (W3 → W2 → W1) | Episodi (W3 → W2 → W1) |
|---|---|---|
| `delta_diffnorm_close_pos_upper_wick_6 < -1.94024 AND delta_diffnorm_close_pos_close_natr_14_6 < -1.6585` | 2.11 → 4.49 → 6.99 | 10 → 6 → 7 |
| `delta_diffnorm_close_pos_close_natr_28_12 < -1.61845` | 1.35 → 1.12 → 1.01 | 15 → 9 → 14 |
| `delta_diffnorm_upper_wick_close_natr_14_12 > 2.08858` | 1.15 → 1.00 → 1.46 | 11 → 9 → 6 |
| `delta_diffnorm_upper_wick_close_natr_28_12 > 2.08763` | 1.15 → 1.00 → 1.46 | 11 → 9 → 6 |

La prima riga è l'unica solida: PF **crescente e ben sopra 1** in tutte e 3 le finestre (2.11 →
4.49 → 6.99), combina un pattern di candela (posizione della chiusura vs upper wick) con volatilità
scale-free (NATR a 14 periodi) — un segnale che senza ATR/NATR abilitato non sarebbe mai stato
candidato. Le altre 3 sono marginali (PF vicino a 1 in almeno una finestra) e aggiungono poco.

Nessuna regola basata su **MACD** sopravvive 3/3 in nessuno dei 5 ticker (una sola, `EBrent`,
`delta_close_macd_12_26_1 < -0.00248`, arriva a 2/3). MACD non produce guadagni robusti qui,
coerentemente con quanto osservato anche in `report_v4` su BTC/ETH/EURUSD.

## 3. Conclusione

L'arricchimento MACD/ATR **non cambia il quadro qualitativo** per questi 5 ticker: dove v3 aveva
già trovato regole robuste (E_Brent, E_DAAX, E_SandP-500, COPPER.CMDUSD) i conteggi restano
sostanzialmente gli stessi; dove non ne aveva trovate (GBPUSD) continua a non trovarne. L'unico
guadagno concreto è **una regola aggiuntiva di qualità per E_Brent** (§2, riga 1) che sfrutta NATR
in modo che gli indicatori di base non permettevano — un'estensione utile ma marginale rispetto al
pool di 94-97 regole già disponibili su quel ticker. Il **Set C** di `report_v3_rolling.md`
(10 regole 3/3-robuste, min 1 per ticker) resta la selezione di riferimento: non viene rivisto,
perché nessuno dei nuovi risultati arricchiti giustifica una sostituzione (le regole COPPER/E_DAAX/
E_SandP-500 sono le stesse, e la nuova regola E_Brent non è chiaramente superiore alle 3 già in
Set C su quello stesso ticker). La nuova regola NATR di E_Brent è comunque loggata ed esportata
qui sotto come candidato aggiuntivo per un futuro Set C ampliato.

## 4. File esportati

In `examples/output/1day_holdout_rules/`:

- **`full_rule_log_v5_5tickers_enriched.pkl`** — tutte le 486 regole valutate (40+397+28+21+0) con
  l'esito su ciascuna delle 3 finestre, flag `uses_macd_or_atr` per isolare le regole che sfruttano
  gli indicatori arricchiti, e flag `is_new_vs_v3` (solo per le regole 3/3 di E_Brent che usano
  NATR — le uniche genuinamente nuove rispetto a `full_rule_log_rolling18m.pkl`).
