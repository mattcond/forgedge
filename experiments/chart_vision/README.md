# Chart vision — "gli occhi del trader" come spazio delle feature

**Domanda:** si può prevedere il target a *h* barre usando un grafico di 50 candele, prima disegnato come immagine e poi codificato da un encoder (CNN o autoencoder)?

**Risposta breve:** no, almeno su dati orari a *h* = 6 e *h* = 24. Le feature estratte dall'immagine non prevedono la direzione meglio del caso e non fanno meglio delle stesse 50 candele passate come semplici numeri. L'encoder però funziona: sulla volatilità, che è un target prevedibile, le stesse feature arrivano ad AUC di circa 0.79.

## Setup

| | |
|---|---|
| Dati | 8 asset 1H (`examples/data`): S&P500, DAX, EURUSD, GBPUSD, Brent, Copper, BTCEUR, ETHEUR. 81 164 campioni (stride 3 barre) |
| Immagine | 2 canali (candele rialziste / ribassiste) × 32 px × 50 candele. Asse Y autoscalato sulla finestra, come su una piattaforma. Corpo = 1, ombra = 0.5 (`results/render_examples.png`) |
| Encoder | `rconv`: CNN a kernel casuali 5×5, non addestrata (336 feature) · `pca`: autoencoder lineare a 32 dimensioni · `ae`: autoencoder MLP 3200→256→**32**→256→3200 · `pixels+scale`: rete supervisionata end-to-end sui pixel |
| Baseline | `raw`: le stesse 50 candele come numeri (OHLC / ultimo close) · `kpi`: indicatori di `forgedge.build_features`, resi scale-free |
| Target | `dir` = segno del log-return a *h* barre (la domanda vera) · `vol` = \|ret\| sopra la mediana del train (controllo positivo) |
| Teste | regressione logistica (L2), HistGradientBoosting |
| Validazione | walk-forward espandente su tempo di calendario, 4 fold, pool multi-asset. Purge: le label del train finiscono prima del test. Encoder fittati solo sul train |
| Nullo | label del test shiftate circolarmente all'interno di ogni asset: l'autocorrelazione resta, l'allineamento con le feature viene distrutto. 20 shift per fold |

## Risultati (`results/results.csv`)

**Direzione (media dei 4 fold):**

| feature | AUC h=6 | AUC h=24 | z vs nullo (h=6) | long-short top30–bottom30 (bps, h=6) |
|---|---|---|---|---|
| immagine `rconv` + HGB | 0.504 | 0.505 | 0.8 | −1.9 |
| immagine `ae` + logit | 0.509 | 0.507 | 1.5 | +0.1 |
| immagine `pca` + logit | 0.508 | 0.508 | 1.2 | −0.6 |
| pixel end-to-end | 0.501 | 0.499 | 0.0 | +0.6 |
| candele numeriche `raw` + HGB | 0.510 | 0.508 | 1.7 | −0.4 |
| KPI forgedge + HGB | 0.512 | 0.507 | 2.2 | +0.6 |

- Tutte le AUC stanno tra 0.50 e 0.51. La deviazione standard del nullo è circa 0.005 per fold a h=6 e circa 0.01 a h=24. Ci sono 34 configurazioni testate senza correzione per test multipli.
- Lo spread long-short è circa 0 o negativo. L'hit rate (≈0.51) è **sotto** il base rate (0.517 / 0.525): un banale "sempre long" fa meglio di ogni modello.
- La piccola AUC sopra 0.5 è compatibile con le differenze di drift tra asset (alcuni asset salgono più spesso di altri) più che con un timing reale. Lo conferma il long-short nullo.

**Volatilità (controllo):**

| feature | AUC h=6 | AUC h=24 |
|---|---|---|
| `rconv` (solo immagine) | 0.61 | 0.56 |
| `ae` (solo immagine) | 0.54 | 0.51 |
| `rconv+scale` | 0.79 | 0.78 |
| `ae+scale` | 0.79 | 0.78 |
| `raw` / `kpi` | 0.80 | 0.79 |

Il grafico autoscalato **perde l'informazione di scala**, cioè quanto è ampia la finestra in percentuale. Quell'informazione il trader la legge sui numeri dell'asse Y. Basta aggiungerla come singolo scalare e l'immagine contiene praticamente tutta l'informazione delle candele numeriche. Quindi il limite non è l'encoder: nella *forma* del grafico non c'è informazione direzionale sfruttabile oltre a quella già presente nei numeri.

## Conclusioni

1. **La rappresentazione visiva non aggiunge nulla.** Nel migliore dei casi l'immagine eguaglia le candele numeriche, ed è ovvio: ne è una trasformazione con perdita di informazione (quantizzazione a 32 livelli e perdita della scala).
2. **L'encoder addestrato non supervisionato ottimizza la cosa sbagliata.** L'autoencoder impara a ricostruire il grafico (R² ≈ 0.23) e non ciò che predice il futuro. La rete supervisionata sui pixel ha più parametri che segnale e va a 0.50.
3. **Il collo di bottiglia è il segnale, non lo spazio delle feature.** Con circa 80k campioni orari sovrapposti, la direzione a 6–24 ore resta quasi un random walk per qualunque rappresentazione della stessa finestra.

## Possibili sviluppi

- **Target condizionato invece che incondizionato:** usare l'embedding come *filtro di contesto* sugli eventi FORGE (il grafico "somiglia" a quelli in cui l'evento X ha funzionato?) anziché come previsore diretto.
- **Target diversi:** volatilità, breakout del range, MFE/MAE. Qui l'immagine funziona, ma non fa meglio dei numeri.
- **CNN vera addestrata (PyTorch)** con augmentation e contrastive learning (SimCLR sui grafici). Nell'ambiente cloud torch non era installabile. È il test mancante più onesto, ma i risultati dei baseline `raw` suggeriscono un tetto basso.
- **Timeframe giornaliero**, con finestre più lunghe e pattern "classici" (testa-spalle ecc.).

Riprodurre: `python experiments/chart_vision/chart_vision.py` (~18 min, 4 core) oppure `--quick` (20k campioni, ~5 min).

---

# Parte 2 — CNN addestrata: target long +1% dopo 10 barre (`train_cnn.py`)

**Domanda:** una CNN addestrata a guardare un grafico di 20 candele riconosce le situazioni in cui `close[t+10] / close[t] − 1 ≥ +1%`?

**Risposta breve:** la CNN impara qualcosa (AUC 0.72), ma impara **la volatilità, non la direzione**. Un modello che guarda solo l'ampiezza della finestra fa esattamente lo stesso (AUC 0.722), e combinare i due modelli non aggiunge nulla.

## Setup

| | |
|---|---|
| Immagine | 20 candele → 2 canali × 32 × 40 px (2 px per candela), asse Y autoscalato |
| Target | `close[t+10] / close[t] − 1 ≥ 1%`, dove t è l'ultima candela visibile |
| Split | per ogni asset: primo 70% = train, con finestre **non sovrapposte** da 20 barre (l'ultimo 15% delle finestre di train serve per l'early stopping); restante 30% = test, anch'esso non sovrapposto. L'ultima label di train finisce prima del taglio |
| Dati | EURUSD da solo ha **6 positivi su 1 237** finestre di train (base rate 0,5%) → uso in pool tutti i 33 asset `*_1HOUR`: 17 107 train, 3 034 val, 8 615 test, base rate 17% |
| CNN | JAX: conv 3×3 (16→32→64) + max-pool ×3, media sull'asse prezzo mantenendo 5 colonne temporali, dense 32 → 1. AdamW, loss bilanciata, early stopping sull'AUC di validazione, 3 seed in ensemble |
| Varianti | `cnn image only` (solo pixel) e `cnn image + y-axis scale` (pixel + range/volatilità della finestra, cioè "i numeri sull'asse Y") |
| Baseline | `vol`: regressione logistica su 3 numeri (range della finestra, volatilità realizzata, range medio delle candele) · `raw`: 20 candele numeriche + vol con HGB |

## Risultati sul test (30% finale, `results/train_cnn_results.csv`)

| modello | AUC (pool) | AUC entro l'asset | z vs nullo | precisione top 10% (entro asset) |
|---|---|---|---|---|
| vol (3 numeri) | **0.722** | 0.604 | 9.8 | 0.265 |
| raw candele + vol (HGB) | 0.719 | 0.602 | 9.6 | 0.271 |
| CNN solo immagine | 0.585 | 0.526 | 2.4 | 0.199 |
| CNN immagine + scala | 0.720 | 0.603 | 10.0 | 0.269 |
| stack vol + CNN | 0.720 | 0.603 | 10.1 | 0.274 |

Il base rate è 0.172. "Entro l'asset" significa: AUC calcolata dentro ogni asset e poi mediata. Il nullo si ottiene permutando le label all'interno di ogni asset.

**Perché è volatilità e non direzione:** nel 10% delle finestre a punteggio più alto (dentro ogni asset):

| | P(≥ +1%) | P(≤ −1%) | P(chiude su) |
|---|---|---|---|
| tutto il test | 0.172 | 0.146 | 0.531 |
| top 10% del punteggio | 0.265 | **0.219** | 0.539 |

La probabilità di un −1% cresce quanto quella di un +1%. Il modello riconosce "sta per muoversi molto", non "sta per salire". La direzione (0.539 contro 0.531) non cambia.

## Conclusioni

1. **L'addestramento funziona:** la CNN supera il caso in modo netto (z ≈ 10).
2. **Ma tutto il segnale è la volatilità.** Per un target a soglia fissa, +1% è prima di tutto una domanda su *quanto* si muove il prezzo. Tre numeri (range e volatilità) bastano per ottenere lo stesso risultato. Con l'immagine autoscalata e senza scala, la CNN deve ricostruire la volatilità indirettamente dalle forme, e arriva solo a 0.585.
3. **La forma del grafico non aggiunge nulla oltre la volatilità:** lo stack vol + CNN è uguale al solo vol (0.603 contro 0.604 entro l'asset).
4. **EURUSD da solo non è addestrabile** su questo target: con 6 positivi un +1% in 10 ore è un evento raro per il cambio.

**Prossimo test sensato:** rendere il target indipendente dalla volatilità, per esempio `fwd ≥ +k·σ` (con σ la volatilità della finestra) oppure "+1% prima di −1%" (triple barrier). In questo modo l'unica cosa che resta da prevedere è la direzione, cioè la vera domanda sugli "occhi del trader".

Riprodurre: `python experiments/chart_vision/train_cnn.py` (~15 min su 4 core CPU, richiede `jax optax`).

---

# Parte 3 — CNN addestrata: target "+1% prima di −1%" (`train_cnn.py --target barrier`)

**Domanda:** tolta la volatilità dal target, la CNN che guarda il grafico sa dire **da che parte** si muoverà il prezzo?

**Risposta breve: no.** Sul test la CNN ha AUC **0.497**, sotto il caso, con z ≈ −0.5 rispetto al nullo. Nel 10% di finestre a punteggio più alto vince il 52.4% dei trade, contro una frequenza di base del 52.1%.

## Setup

Uguale alla Parte 2 (20 candele → immagine 2×32×40, finestre non sovrapposte, split 70/30 per asset, 33 asset 1H), con queste differenze:

| | |
|---|---|
| Target | ingresso al close dell'ultima candela visibile; **1** se l'high tocca +1% prima che il low tocchi −1%, **0** altrimenti |
| Esclusi | finestre in cui entrambe le soglie sono toccate nella stessa barra (ordine non conoscibile su OHLC) e finestre non risolte entro 200 barre |
| Purge | una finestra di train è tenuta solo se la barriera si risolve **prima** del taglio del 70% |
| Campioni | 16 533 train / 2 933 val / 8 298 test. Frequenza di base 0.51 nel train e 0.52 nel test. Mediana di 8 barre per arrivare a una delle due soglie |
| EURUSD | ora è utilizzabile: 913 finestre di train (430 positive) e 377 di test |
| Baseline in più | `asset prior`: il tasso di vittoria di ogni asset nel train (cattura solo il drift dell'asset, nessun timing) |

## Risultati sul test (`results/train_cnn_results_barrier.csv`)

| modello | AUC (pool) | AUC entro l'asset | z vs nullo | vittorie nel top 10% (entro asset) | P&L medio top 10% (bps) |
|---|---|---|---|---|---|
| asset prior | 0.487 | 0.500 | — | 0.521 | 4.3 |
| vol (3 numeri) | 0.511 | 0.511 | 1.6 | 0.531 | 6.3 |
| raw candele + vol (HGB) | 0.509 | 0.511 | 1.6 | 0.546 | 9.1 |
| **CNN solo immagine** | **0.498** | **0.497** | **−0.5** | 0.503 | 0.6 |
| **CNN immagine + scala** | **0.497** | **0.498** | **−0.5** | 0.524 | 4.9 |
| stack vol + CNN | 0.505 | 0.505 | 0.8 | 0.514 | 2.7 |
| *tutti i trade* | | | | *0.521* | *4.3* |

Il P&L medio è in bps per trade: +100 se vince, −100 se perde, costi esclusi.

- **Già in validazione** la CNN si ferma a 0.51–0.52 (con la soglia fissa della Parte 2 arrivava a 0.74). L'early stopping si attiva dopo circa 13 epoche: il modello non trova nulla da imparare che generalizzi.
- **Nessun baseline è significativo.** Il migliore (raw + vol, z = 1.6) resta sotto la soglia del 5% anche prima di correggere per i 6 modelli testati.
- **Sul piano economico:** il 10% di trade selezionati dalla CNN rende +4.9 bps lordi, praticamente come prenderli tutti (+4.3 bps, cioè il drift). Dopo i costi è negativo.
- **Per asset** (`results/train_cnn_per_asset_barrier.csv`): l'AUC della CNN va da 0.43 a 0.56 e si distribuisce simmetricamente intorno a 0.5, come ci si aspetta dal rumore su 150–500 campioni.

## Conclusioni sulle tre parti

| | target | cosa ha imparato il modello visivo |
|---|---|---|
| Parte 1 | direzione a 6/24 barre | niente (AUC 0.50–0.51) |
| Parte 2 | close +1% a 10 barre | **volatilità** (AUC 0.72), identica a 3 numeri su range e volatilità |
| Parte 3 | +1% prima di −1% | niente (AUC 0.497) |

L'idea degli "occhi del trader" funziona *tecnicamente*: la CNN impara davvero dai pixel quando nel target c'è qualcosa da imparare (Parte 2). Ma l'unica informazione che il grafico porta è *quanto* si muove il prezzo. La *direzione* a breve, su dati orari, non è contenuta nella forma delle ultime 20–50 candele, per nessun encoder provato (CNN addestrata, CNN casuale, PCA, autoencoder) e nemmeno nei numeri grezzi.

Se si vuole continuare, ha più senso usare il grafico come **contesto** invece che come previsore autonomo:
- condizionare un evento FORGE già validato: "l'evento X funziona meglio quando il grafico ha questa forma?";
- prevedere la volatilità per dimensionare stop e target (Parte 2).
