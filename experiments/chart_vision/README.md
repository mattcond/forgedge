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
