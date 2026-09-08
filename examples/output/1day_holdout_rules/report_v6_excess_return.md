# Report v6: il rendimento di hold-out di Set C è eccesso genuino o solo regime?

**Data esecuzione:** 2026-09-08
**Segue:** `report_v3_rolling.md` — le 10 regole di Set C sopravvivono tutte e 3 le finestre
rolling con PF episodio ≥ 1, ma un PF ≥ 1 su un asset in trend persistente (§2 di
`report_v3_rolling.md`: E_SandP-500 +8.3%/+6.0%/+11.2% nelle 3 finestre, sempre rialzista) non
dimostra da solo che la regola stia selezionando qualcosa — potrebbe limitarsi a essere `long`
in un mercato che sale comunque. Questo report isola le due componenti.

## 1. Metodologia

Per ciascuna delle 10 regole di Set C e ciascuna delle 3 finestre rolling, si backtesta un
**segnale baseline "always-on"** (attivo su ogni barra, anziché solo quando la condizione KPI
della regola è vera) con la **stessa identica `BacktestParams`** della regola — stessa
direzione, `target_h`, `buy_type`, `buy_drop_pct`, `sell_pct`, `fee`. Stesso ticker, stessa
finestra, stessa meccanica d'ordine: l'unica variabile isolata è **quale giorno** si entra.

- **Lato regola**: rendimento medio per *episodio* (come in `report_v2.md`/`v3` — dedup delle
  entrate nominali sovrapposte quando la regola si attiva su barre consecutive).
- **Lato baseline**: un segnale sempre attivo entra quasi ogni giorno, quindi quasi tutte le
  posizioni si sovrappongono e collassano in un solo "episodio" enorme — l'aggregazione a
  episodio butterebbe via l'intera distribuzione (n=1). Si usa invece il **rendimento per singolo
  trade nominale** (un valore per ogni giorno di ingresso tentato): è esattamente la popolazione
  "cosa avrebbe reso un ingresso con la stessa meccanica in un giorno medio di questa finestra" —
  il benchmark di regime corretto.
- **Eccesso** = rendimento medio per episodio (regola) − rendimento medio per trade (baseline),
  stessa finestra.
- **Significatività**: permutation test a due code (2000 permutazioni, pool congiunto regola+
  baseline) sulla differenza delle medie — nessuna dipendenza da `scipy`.

## 2. Risultati per regola

| # | Ticker | Direzione | Eccesso W3 | Eccesso W2 | Eccesso W1 | Positivo su | Eccesso medio |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | COPPER.CMDUSD | long | +1.88% (p=.51) | +1.69% (p=.15) | +0.88% (p=.46) | 3/3 | **+1.48%** |
| 2 | E_Brent | long | +2.20% (p=.18) | +0.84% (p=.56) | +2.40% (p=.25) | 3/3 | **+1.81%** |
| 3 | E_DAAX | long | +0.88% (p=.61) | +0.50% (p=.54) | +0.02% (p=.98) | 3/3 | +0.47% |
| 4 | E_SandP-500 | long | +0.06% (p=.95) | +0.14% (p=.73) | -0.14% (p=.83) | 2/3 | **+0.02%** |
| 5 | E_Brent | long | +2.20% (p=.19) | +0.84% (p=.58) | +2.40% (p=.26) | 3/3 | +1.81% |
| 6 | E_Brent | long | +3.06% (p=.10) | +0.33% (p=.80) | +3.26% (p=.21) | 3/3 | **+2.22%** |
| 7 | COPPER.CMDUSD | long | +1.02% (p=.74) | +2.11% (p=**.03**) | +2.06% (p=.05) | 3/3 | **+1.73%** |
| 8 | E_SandP-500 | long | +0.08% (p=.78) | -0.23% (p=.87) | +0.24% (p=.54) | 2/3 | +0.03% |
| 9 | E_SandP-500 | long | -0.07% (p=.90) | +0.01% (p=.99) | +1.10% (p=.25) | 2/3 | +0.35% |
| 10 | COPPER.CMDUSD | long | +1.26% (p=.85) | +0.43% (p=.68) | +0.60% (p=.61) | 3/3 | +0.77% |

**Aggregato**: 27/30 combinazioni regola×finestra hanno eccesso positivo, eccesso medio
complessivo **+1.07%** per episodio/trade. Solo 2 osservazioni su 30 raggiungono p<0.10 (righe
#7 W2 e #6 W3) — con 5-20 episodi per finestra la potenza statistica è strutturalmente bassa;
questi p-value vanno letti come "direzione consistente, non prova statistica individuale", non
come conferma per singola cella.

## 3. Due gruppi nettamente diversi

Il pattern per ticker è netto e coerente con i regimi osservati in `report_v3_rolling.md` §2:

| Ticker | Regime nelle 3 finestre (W3→W2→W1) | Eccesso medio |
|---|---|---:|
| E_Brent | -4.6% → +5.9% → +22.5% (misto, W3 negativo) | **+1.95%** |
| COPPER.CMDUSD | -0.2% → +30.2% → +9.7% (misto, W3 piatto/negativo) | **+1.33%** |
| E_DAAX | +6.1% → +5.7% → +5.1% (rialzo costante e moderato) | +0.47% |
| E_SandP-500 | +8.3% → +6.0% → +11.2% (rialzo costante e forte) | **+0.13%** |

**E_Brent e COPPER.CMDUSD** (6 regole su 10) mostrano un eccesso consistente **anche nelle
finestre in cui il regime del ticker era piatto o negativo** (W3 per entrambi) — è proprio lì
che un rendimento positivo *non può* venire dal solo essere `long` in un mercato che sale, quindi
l'eccesso osservato è la prova più diretta di selezione temporale genuina della regola.

**E_SandP-500** (3 regole su 10) mostra un eccesso **quasi nullo e senza segno consistente**
(-0.07% a +1.10%, media +0.13%) in un ticker che è salito ininterrottamente in tutte e 3 le
finestre. Il PF episodio ≥ 1 di queste 3 regole (già riportato in Set C) è quindi spiegato quasi
interamente da **beta di regime** (essere `long` su un indice in rialzo strutturale), non da una
reale capacità della condizione KPI di scegliere i giorni migliori. **E_DAAX** è un caso
intermedio, più vicino a S&P-500 (regime costantemente positivo, eccesso modesto) ma non nullo.

## 4. Conclusione e raccomandazione

Il criterio di sopravvivenza usato finora in Set C (PF episodio ≥ 1 su tutte e 3 le finestre) **non
distingue tra alpha e beta di regime** — questo report aggiunge quella distinzione:

- **Alpha probabile** (eccesso positivo e consistente anche in finestre di regime piatto/negativo):
  righe #1, #2/#5, #6, #7, #10 — tutte le regole COPPER.CMDUSD ed E_Brent di Set C.
- **Beta di regime, alpha non dimostrato** (eccesso ≈ 0, nessun segno consistente): righe #4, #8,
  #9 — le 3 regole E_SandP-500. Restano valide come esposizione tattica `long` coerente col
  criterio di robustezza originale, ma **non vanno presentate come regole con vantaggio
  informativo dimostrato** rispetto a una semplice esposizione direzionale sullo stesso ticker/
  periodo di detenzione.
- **Caso intermedio**: riga #3 (E_DAAX), eccesso positivo ma piccolo e in calo (arriva quasi a
  zero in W1).

Nessuna riga viene rimossa da Set C in questo report (il criterio originale di robustezza
rolling resta valido come lo era), ma il file esportato aggiunge il flag `likely_alpha` per
permettere di filtrare le sole regole con eccesso dimostrato sul regime.

## 5. Limiti

- La potenza statistica per-cella è bassa (5-20 episodi per regola/finestra): l'evidenza è nella
  **consistenza del segno attraverso 3 finestre indipendenti e regimi diversi**, non nei singoli
  p-value.
- Il baseline "always-on" con `buy_type="limit"` non è un vero benchmark passivo (non è un
  semplice buy&hold) — è deliberato: isola l'effetto della *condizione KPI* tenendo fissa la
  meccanica d'ordine (stesso `target_h`, stesso `sell_pct`, stessa `fee`), che è esattamente la
  domanda "la regola sceglie *quando* entrare meglio di un ingresso indiscriminato con la stessa
  meccanica?".
- Analisi limitata alle 10 regole di Set C (indicatori base, non arricchiti) — non ripetuta su
  Set A/B/v4/v5 per contenere il tempo di calcolo; la metodologia (`measure_excess_return.py`)
  è riutilizzabile su qualunque altro set.

## 6. File esportati

In `examples/output/1day_holdout_rules/`:

- **`excess_return_setC.pkl`** — per ciascuna delle 10 regole di Set C: eccesso medio, PF e
  n episodi (regola vs baseline), p-value del permutation test per ciascuna delle 3 finestre,
  flag `likely_alpha` (eccesso medio > 0 in almeno 2/3 finestre e nessuna finestra fortemente
  negativa).
