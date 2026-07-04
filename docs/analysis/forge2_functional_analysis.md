# FORGE 2.0 — analisi funzionale a parità di contratto

**Domanda:** ripartendo da zero, cosa va tenuto e cosa va riprogettato — senza
cambiare il contratto con l'utente?
**Base empirica:** sessione di troubleshooting/audit su `ADA_1D_FULL.parquet`
(901 barre daily, run completo + audit del codice, PR #145) e il null test di
`lowfreq_robustness.md` (surrogati phase-randomized e IID sullo stesso dataset).

---

## 1. Il contratto con l'utente (ciò che non si tocca)

Il contratto è la parte migliore del prodotto e va difeso letteralmente:

1. **Input:** una KPI Table (OHLCV + indicatori, `close` obbligatoria) e una
   sola chiamata — `forge(kpi, ticker, timeframe)` / `forge_multi`.
2. **Output:** artefatti formali per stadio — `EventCandidate` →
   `AlphaContract` → verdetto `EDGE / PARTIAL-EDGE / NON-EDGE` →
   registry con report autosufficiente.
3. **Invarianti dichiarate:** M1 non vede mai il forward return; soglie
   immutabili dopo la discovery; target (h*, direzione, take-profit) derivato
   dai dati per evento; evidenza statistica separata dal giudizio economico.
4. **Auditabilità:** ogni contratto porta le proprie `rejection_reasons`;
   `ForgeResult` conserva tutti gli artefatti intermedi.
5. **Dipendenze:** solo numpy/pandas, primitive statistiche in-house.

Tutto ciò che segue migliora *il modo in cui il contratto viene onorato*, non
il contratto.

---

## 2. Cosa terrei (punti di forza verificati, non dichiarati)

| Cosa | Perché va tenuto (evidenza) |
|---|---|
| **Pipeline a moduli con artefatti formali** | La separazione è reale nel codice: M1 non importa nulla del forward return; il replay degli eventi in M3 è bit-for-bit quello di M1. Non è marketing architetturale. |
| **Soglie distribuzionali immutabili** | I percentili asset-specifici congelati alla discovery eliminano davvero la ricalibrazione post-hoc; `EventCandidate.apply()` è deterministico su dati nuovi. |
| **Target derivato per evento** | h*, direzione e `sell_pct` dalla MFE evitano che un'assunzione economica pre-cuocia il risultato — è l'idea più distintiva del sistema. |
| **M3 come giudice economico** | Backtest a meccanica ordini reale, walk-forward ri-ottimizzato per finestra come gate formale (OOS PF ≥ 1.0 = NON-EDGE duro), DSR, stabilità temporale. La parte "onesta" della pipeline è questa. |
| **Diagnostica trasparente** | I contratti respinti restano ispezionabili; le `[diagnostic]` sull'OOS debole sono un pattern giusto (il problema è cosa se ne fa il verdetto — v. §3.2). |
| **Golden test come contratto comportamentale** | La suite golden ha reso il re-pin del fix horizon-grid un'operazione controllata, non un salto nel buio. |
| **Preset come traduzione di intenzione** | `forge_preset("balanced", "1D")` che allinea min_tpm/dispersione/orizzonti tra M1-M2-M3 è la direzione giusta (il problema è che non è il default — v. §3.5). |
| **Zero dipendenze pesanti** | Spearman/t-test/BH-FDR/OU in numpy puro: installabilità e auditabilità delle primitive. |

---

## 3. Cosa riprogetterei (in ordine di impatto funzionale)

### 3.1 Il budget statistico come cittadino di prima classe — *il* difetto strutturale

**Osservato.** Su ADA 1D: 2.755 candidati da ~630 barre IS. Nel null test il
conteggio di alpha promossi sul rumore supera quello del dato reale (146±45 vs
58) e il rumore puro incassa ~20 % di falsi EDGE. Ogni modulo controlla il
proprio multiple testing (BH-FDR in M2, DSR in M3) ma **il numero di prove non
si propaga attraverso i moduli**: `validation.py` deflaziona lo Sharpe con
`n_trials = len(grid_results)` (~200 configurazioni operative), ignorando le
migliaia di eventi × orizzonti testati a monte. Il RotationCalibrator — che è
esattamente il correttivo giusto — è opzionale e spento di default.

**Forge 2.0.** Un *ledger delle ipotesi*: ogni modulo dichiara quante ipotesi
ha consumato (eventi generati, orizzonti scansionati, celle di griglia), e il
verdetto finale è deflazionato sull'intera catena, non sull'ultimo miglio. Il
null empirico (rotazioni/surrogati, K adattivo alla lunghezza del dato) diventa
il default della promozione: si promuove l'*eccesso sopra il null*, non il
superamento di soglie assolute. Il contratto non cambia: stesso input, stessi
verdetti — ma EDGE torna a significare EDGE.

**[IMPLEMENTATO]** `forgedge.ledger.HypothesisLedger` registra la superficie
della sessione su `ForgeResult.ledger`; `calibration.FastRotationNull` calcola
il null search-level *esatto* su tutti gli offset circolari (via FFT, ~1 s —
niente K, niente seed), gira di default dentro `forge()` e il verdetto EDGE
pieno di Rule Discovery ora richiede di batterlo
(`SelectionCriteria.max_rotation_p`); un DSR indefinito (haircut con radicando
negativo) blocca anch'esso l'EDGE pieno invece di saltare il gate.

### 3.2 Verdetti con potenza dichiarata — degradazione onesta, non fiducia di default

**Osservato.** Su 2 anni di daily i gate OOS di M2 lavorano su n_act < 10; il
sistema lo *sa* (emette `[diagnostic] OOS sample too small`, calcola la MDE) ma
il verdetto esce comunque come EDGE pieno. La conoscenza sull'affidabilità
esiste e viene buttata via al momento della sintesi.

**Forge 2.0.** Il vocabolario dei verdetti si arricchisce di uno stato:
`INSUFFICIENT-DATA` (o un campo `confidence` obbligatorio sul verdetto), emesso
quando MDE > effetto stimato o quando trades-per-finestra-WF scende sotto un
floor. È un'estensione compatibile: chi consuma `is_edge` oggi continua a
funzionare; chi vuole sapere *quanto fidarsi* finalmente può.

**[IMPLEMENTATO]** Verdetto `INSUFFICIENT-DATA`
(`SelectionCriteria.power_gate`, on di default): un verdetto che sarebbe
EDGE/PARTIAL-EDGE viene degradato quando l'evidenza OOS non può sostenerlo —
nessun walk-forward possibile, trade OOS **aggregati** sotto
`min_oos_trades`, oppure MDE dell'expectancy sul campione OOS aggregato
superiore all'expectancy IS dichiarata. La valutazione legge *solo* il ledger
concatenato delle finestre test, mai i conteggi per-finestra (le finestre WF
sono corte by design e non vengono mai gateate individualmente). NON-EDGE non
viene mai riscattato; INSUFFICIENT-DATA conserva la `ValidatedRule` per
ri-valutazione futura ma non è tradeable (`is_edge=False`, mai nel registry).

### 3.3 Un solo asse temporale, purged & embargoed

**Osservato.** Tre split indipendenti e non coordinati: `train_ratio` di M1,
`train_ratio` di M2 (0.70), walk-forward di M3 — ciascuno taglia il tempo per
conto suo. Gli orizzonti forward (fino a h* barre) attraversano i confini delle
finestre adiacenti: il return che "conferma" in OOS è parzialmente lo stesso
che ha derivato il target in IS. Il null test attribuisce a questo il
paradosso del punto 3.1 (finestre corte regime-confuse: il reale fallisce la
conferma, il rumore stazionario la passa).

**Forge 2.0.** Un `TimeBudget` centrale costruito una volta da `forge()` e
passato a tutti i moduli: assegna IS/conferma/WF su un unico asse, con purging
ed embargo automatici pari a `max(horizon_grid)` barre ai confini. Nessun
modulo taglia il tempo da solo. (È anche il punto 4 di
`lowfreq_robustness.md`, lì marcato "higher effort" — in una 2.0 è il posto
dove metterlo, perché retrofittarlo su tre split indipendenti è più costoso
che nascerci.)

**[IMPLEMENTATO]** `forgedge.timebudget.TimeBudget`: asse unico opzionale
(`forge(time_budget=...)` lo impone a M1+M2; ogni modulo lo accetta anche
standalone). Il purge è **on di default** anche senza budget esplicito: in M2
le ultime `h` barre IS per orizzonte sono escluse da ogni misura (la loro
finestra forward attraversa lo split), in M3 le finestre train del
walk-forward sono accorciate del worst-case trade span della griglia (le
uscite leggevano prezzi della finestra test). Embargo opzionale
(`AlphaConfig.embargo_bars`, `WalkForwardConfig.embargo_bars`), default 0.

### 3.4 La selezione operativa di M3 fuori dal full-sample

**Osservato.** `RuleDiscovery._run_stage` esegue lo screening di griglia e il
"IS summary" **sull'intera tabella** (nessun timerange): il walk-forward poi
rifà la selezione onestamente per finestra, ma i gate di early-elimination, il
PF in-sample riportato e la configurazione `ValidatedRule` pubblicata guardano
dati che includono anche la coda OOS del contratto.

**Forge 2.0.** La configurazione operativa si seleziona solo dentro le finestre
train del walk-forward; l'"in-sample summary" esposto è la concatenazione dei
train, il parametro pubblicato è quello dell'ultima finestra (o il consenso).
Stessa API di risposta, numeri più onesti.

**[IMPLEMENTATO]** `RuleDiscoveryConfig.selection_mode="walk_forward"`
(default): i parametri pubblicati vengono dalle sole finestre train del WF
(`wf_param_policy`: ultima finestra o consenso), e IS summary,
early-elimination, validazione statistica, envelope, MAE/MFE e regime sono
tutti calcolati sullo *span di selezione* `[start, fine ultima finestra
train)` — l'ultima finestra test non è mai letta da nulla che alimenti il
verdetto o la `ValidatedRule`. Il pre-screen di eliminazione gira sulla prima
finestra train (dati solo selection-side). `"full_sample"` resta come
escape hatch legacy; fallback automatico (annotato) quando lo span non
consente neppure uno split.

### 3.5 I default derivano tutti dal dato, o non sono default

**Osservato.** Il fix della horizon grid (PR #145) è il sintomo di un problema
generale: `forge()` liscio e `forge_preset` incarnano due filosofie diverse.
Il default di M1 è "no split, nessun walk-forward"; la griglia orizzonti era
oraria su qualunque timeframe; `min_tpm` ha unità diverse (bar vs episodi) a
seconda di dove lo si imposta. L'utente del Quick start e l'utente dei preset
ottengono due prodotti diversi.

**Forge 2.0.** Ogni parametro frequenza-dipendente ha una sola sorgente di
verità, derivata da `timeframe` + lunghezza del dato (la logica di
`_TFClass` promossa a servizio centrale). `forge(kpi, intent="balanced")`
diventa la strada maestra; gli oggetti config restano come override esperti.
Un `forge.explain_config()` stampa la configurazione risolta e *perché* —
coerente con la filosofia di auditabilità del resto del sistema.

### 3.6 Identità contenutistica degli artefatti

**Osservato.** `alpha_id` incorpora la data di run (`ALPHA-ADA-1D-260704-…`):
due run identici in giorni diversi producono ID diversi; il dedup del registry
e i confronti tra sessioni si appoggiano a euristiche su espressione/parametri.

**Forge 2.0.** ID = hash stabile di (espressione canonica, target derivato,
config rilevante); `discovery_date` resta come metadato. Riproducibilità
byte-for-byte tra run e dedup del registry esatti, gratis.

### 3.7 Registry: perimetro di validità esplicito

**Osservato.** In sessione single-ticker `classification` / `is_generic`
escono `None` senza spiegazione (50 documenti su ADA, tutte le colonne di
genericità vuote); il consumatore non distingue "non generalizzata" da "non
testabile".

**Forge 2.0.** La genericità diventa un verdetto a tre stati espliciti
(`GENERIC / SPECIFIC / NOT-TESTED(motivo)`) e ogni regola catalogata espone il
proprio *perimetro di validità*: ticker testati, finestra temporale, regimi in
cui l'edge esiste, data oltre la quale la regola non è stata validata. Il
report HTML lo mostra come prima cosa, non come nota a piè di pagina.

### 3.8 Ergonomia e costo (minore, ma composto)

- **Suite test 8,5 min** dominata da pipeline complete ripetute → fixture di
  sessione condivise e cache degli stadi (`ed.df` ricomputato identico decine
  di volte).
- **M3 sequenziale** (255 contratti × ~0,4 s su dati piccoli) → il backtest per
  contratto è imbarazzantemente parallelo; un `n_jobs` senza cambiare API.
- **Progress/logging** già ben fatto (reporter a stadi + tqdm opzionale):
  tenere.

---

### 3.9 (addendum) — Griglia orizzonti arricchita dallo span degli indicatori

**Osservato (analisi su ADA 1D, griglia diagnostica 1–40 barre).** La relazione
tra la finestra dominante `w` dell'indicatore e l'orizzonte di picco `h*` esiste
ma non è una legge unica: per w ≤ ~30 circa metà dei picchi cade in `[w/2, 2w]`
(eventi "ciclici"); per w > 30 (pctrank_168, zscore_96 — eventi "di stato") il
100% dei picchi sta sotto `w/2`: la finestra lunga definisce quanto raro è lo
stato, non la scala della reazione. Una banda dura `[w, 2w]` avrebbe escluso
l'orizzonte vero per la maggioranza degli eventi promossi.

**[IMPLEMENTATO]** Arricchimento per **unione**, mai restrizione
(`AlphaConfig.horizon_enrichment=(0.5, 1.0, 2.0)`, on di default): per ogni
candidato gli orizzonti `round(m·w)` — con `w =
EventCandidate.dominant_window()`, proprietà strutturale outcome-independent —
vengono aggiunti alla griglia base, con cap statistico `h ≤
split/horizon_enrichment_min_obs` (default 20 finestre forward non sovrapposte
minime). Le ipotesi aggiunte sono contate esattamente dal ledger
(`m2_return_tests`) e prezzate dal fast rotation null con la stessa griglia
per-evento (fedeltà bit-for-bit preservata). Su ADA: 34/247 alpha promossi
trovano h* su un orizzonte arricchito (6–24 giorni) che la griglia base non
copriva.

## 4. Cosa NON farei (anti-goal espliciti)

- **Niente ML/feature learning nella discovery.** Il valore differenziante è
  che ogni regola è un'espressione booleana leggibile e auditabile; un modello
  addestrato romperebbe il contratto molto più di qualunque bug.
- **Niente registry persistente/DB.** In-memory + export (flat table, HTML) è
  il livello giusto di ambizione; la persistenza è un problema dell'host.
- **Niente verdetti probabilistici al posto della triade.** La triade (+
  `INSUFFICIENT-DATA`) *è* il contratto; la confidenza va accanto al verdetto,
  non al suo posto.
- **Niente dipendenze statistiche esterne.** Le primitive in numpy puro sono
  un asset di auditabilità, non un debito.

---

## 5. Sintesi: dove va l'investimento

A parità di contratto, FORGE 1.x ha già risolto il problema *architetturale*
(separazione, artefatti, auditabilità) e quello *meccanico* (backtest, replay
deterministico). Il debito è concentrato in un punto solo, ed è *statistico*:

> il sistema conta benissimo le proprie prove dentro ogni modulo, ma nessuno
> conta le prove dell'intera catena — e sui dati lenti questo trasforma il
> multiple testing in verdetti EDGE che il rumore puro sa replicare 1 volta
> su 5.

Le priorità 3.1–3.4 attaccano esattamente questo, in ordine di rapporto
valore/rischio; 3.5–3.7 rendono il prodotto coerente con la sua stessa
filosofia; 3.8 è manutenzione. Un ipotetico forge 2.0 che implementasse solo
3.1 e 3.2 sarebbe già un prodotto sostanzialmente più onesto dello stato
attuale — senza che l'utente debba cambiare una riga del proprio codice.
