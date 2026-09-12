# Proposta funzionale — modalità ranged per `min_tpm` (M1) e diagnostica di allineamento al mercato (M2)

**Stato del documento:** bozza di specifiche funzionali, non di design tecnico né di
implementazione. Raccoglie tre idee emerse da una sessione di esplorazione sul
pipeline FORGE, con lo stato attuale del codice verificato riga per riga e le
decisioni di indirizzo già prese. Nessun codice è stato scritto: questo documento è il
punto di partenza per un eventuale design tecnico successivo.

**Idee tracciate:**

- **A.** Event Discovery (M1): modalità "ranged" per `GateParams.min_tpm` — da floor a
  target con banda di tolleranza.
- **B.** Alpha Discovery (M2): etichettatura momentum / mean-reversion / idiosincratica
  per contratto, in due stadi — (a) un test di significatività integrato sull'intera
  griglia degli orizzonti (riusa il rotation-null esistente), (b) la pendenza del tasso
  per-barra dell'eccesso lungo la griglia, solo se (a) è positivo.
- **C.** Alpha Discovery (M2): diagnostica di sufficienza della griglia degli orizzonti
  (`horizon_grid`) — emersa come prerequisito dell'idea B, ma tracciata come idea
  indipendente perché utile a prescindere da essa.

Le tre idee sono indipendenti nell'implementazione (toccano moduli/file diversi). Una
dipendenza concettuale B→C era stata ipotizzata in una prima stesura ma la validazione
empirica di B (§3.4) l'ha esclusa come causa dei problemi trovati — resta solo una nota
diagnostica accessoria (si veda §3.5).

---

## 1. Contesto comune

FORGE separa il pipeline in domini che non si devono mescolare (vedi `CLAUDE.md`/skill
`forgedge`, invarianti #1 e #4): M1 vede solo struttura temporale delle attivazioni,
mai il rendimento; M2 deriva il target economico (orizzonte, direzione, size) dai dati,
per evento, mai a priori; M3 valuta l'operatività (fee, fill, backtest). Tutte e tre le
idee qui proposte rispettano questa separazione: nessuna delle tre introduce
un'osservazione del rendimento in M1, né fissa un target economico a priori in M2.

---

## 2. Idea A — Consistency Gate in modalità "ranged" (M1)

### 2.1 Motivazione

Oggi `min_tpm` è un floor a senso unico ovunque nel gate: si accetta un evento se il suo
tasso di attivazione è **almeno** `min_tpm`. Non esiste un modo per dire "voglio eventi
che si attivano *in media* a questo ritmo", ammettendo sia eventi più rari sia eventi
più frequenti solo entro la variabilità che il processo stesso mostra. Un caso d'uso
tipico: selezionare pattern "regolari" a una cadenza attesa, scartando sia i pattern
troppo rari (poco potere statistico) sia quelli quasi-sempre-attivi (spesso condizioni
banali/degenerate, es. un indicatore quasi sempre sopra soglia).

### 2.2 Stato attuale (verificato)

- `ConsistencyGate._gate_pass` (`event_discovery/consistency_gate.py:351-357`) implementa
  il criterio come disequazione a un lato:
  - modalità `"bar"`: `mean_tpm >= min_tpm` e `id_score <= max_dispersion`;
  - modalità `"episode"` (default): `episode_tpm >= min_tpm`, `n_episodes >= min_episodes`,
    `episode_id <= eff_max_dispersion`.
- La dispersione è già misurata come **indice di dispersione** (`Var/Mean` dei conteggi
  mensili) e confrontata con un **tetto** statisticamente motivato:
  `eff_max_dispersion = poisson_floor(n_months) × dispersion_margin`
  (`consistency_gate.py:290-304`) — il tetto oltre il quale un processo non è più
  compatibile con la casualità (chi-quadro al 95%, approssimazione di Wilson-Hilferty).
- `min_tpm` è la radice di una catena di derivazioni nel resolver
  (`resolver.py`, funzione `_derive_pf_min_tpm` e dintorni, righe ~928+): da esso si
  derivano `min_train_months`, `pf_min_tpm`, e vengono verificati vincoli come
  `oos_span_too_short`, `m3_stricter_than_m1`. Tutta questa catena assume la semantica di
  floor con margine di Poisson al 95% ("almeno N trade attesi al tasso minimo").

### 2.3 Proposta funzionale (formule congelate)

Aggiungere una modalità di selezione alternativa al gate, **attiva solo su richiesta
esplicita** e **confinata a M1**. Due nuovi campi su `GateParams`:

```python
tpm_mode: Literal["floor", "ranged"] = "floor"   # default invariato, nessuna regressione
tpm_tolerance: float = UNSET                      # stessa unità di min_tpm
```

In modalità `"ranged"`, `min_tpm` è il **centro** della banda. `tpm_tolerance` è la
mezza-larghezza, in unità assolute (episodi/mese o barre/mese, come `min_tpm`), con due
comportamenti a seconda che sia lasciato `UNSET` o fissato dall'utente — lo stesso
idioma `UNSET`→derivato usato ovunque nel resolver, ma risolto **localmente dentro il
gate**, non nella tabella `CONSTRAINTS` di `resolver.py` (si veda il perché in §2.4):

- **`UNSET` (default) → derivato dalla dispersione tollerata di sessione:**

  ```
  σ_tpm = sqrt(eff_max_dispersion · min_tpm / n_total_months)
  tpm_tolerance_effettivo = z · σ_tpm            (z = 1.959964, normale a due code, 95%)
  ```

  `eff_max_dispersion` è il tetto **di sessione** già calcolato dal gate (§2.2), non
  l'`episode_id` del singolo evento. È una scelta deliberata, non la più ovvia — si veda
  §2.4 per il perché.

- **Fissato esplicitamente dall'utente → banda letterale**, nessuna statistica coinvolta:
  `banda = [min_tpm − tpm_tolerance, min_tpm + tpm_tolerance]`. Risolve direttamente il
  caso d'uso "voglio eventi con mediamente 4 episodi al mese, tolleranza 2": si scrive
  `tpm_tolerance=2.0` (con `min_tpm=4.0`, anch'esso in override) e si ottiene esattamente
  quella banda, senza che nessuna formula derivata la modifichi.

In entrambi i casi: `banda = [max(0, centro − mezza_larghezza), centro + mezza_larghezza]`
(un tasso non può essere negativo).

**Precondizione strutturale, non parte della banda:** un evento con `n_episodes == 0`
(non si attiva mai) è scartato a monte, prima di valutare la banda — non è "un evento con
tasso 0 che eventualmente rientra nella banda per effetto del troncamento a zero", è
semplicemente non un evento. Non richiede una nuova verifica: `min_episodes` (default 10)
lo esclude già in pratica, ma la precondizione va resa esplicita nel codice invece di
dipendere da quel default.

Il criterio di burstiness (`episode_id <= eff_max_dispersion`, invariato) resta
**indipendente** dal criterio di banda — la modalità ranged aggiunge un vincolo sul
livello del tasso, non ne rimpiazza o modifica uno esistente su forma/regolarità.

### 2.4 Perché il default usa `eff_max_dispersion` (di sessione) e non `episode_id` (dell'evento)

Prima scelta scartata: usare `episode_id` proprio dell'evento al posto di
`eff_max_dispersion` renderebbe la banda adattiva per evento (più stretta per un evento
regolare, più larga per uno vicino al tetto di burstiness ammesso). Sembra più
"principiato" statisticamente, ma ha un difetto concreto: **accoppia nella direzione
sbagliata** i due criteri del gate. Un evento già borderline sulla burstiness (`ID`
vicino al tetto) riceverebbe automaticamente *più* tolleranza sul tasso — l'opposto di
quanto si vorrebbe da un filtro di qualità (più scrutinio, non meno, dove la struttura
temporale è già al limite). In più, `episode_id` è una varianza campionaria stimata su
pochi conteggi mensili: con `n_total_months` piccolo (il regime più comune per i preset
più selettivi, es. `"sniper"`) la stima è essa stessa rumorosa, e quel rumore si
propagherebbe due volte nella larghezza della banda.

Usare `eff_max_dispersion` (il tetto di sessione, identico per ogni evento) risolve
entrambi i problemi: è una funzione chiusa di `n_total_months` e `dispersion_margin`,
nessuna stima campionaria per-evento coinvolta, e — punto chiave — equivale a valutare la
formula "adattiva" nel caso peggiore ammissibile (`episode_id ≤ eff_max_dispersion` per
ogni evento che supera comunque il gate), quindi è **sempre almeno larga quanto** lo
sarebbe stata la versione adattiva per quello specifico evento, mai più stretta. Il
costo è la perdita di adattività per-evento (un evento molto regolare non viene premiato
con una banda più stretta) — un costo di precisione, non di sicurezza, e comunque
aggirabile fissando `tpm_tolerance` esplicitamente quando serve.

### 2.5 Decisioni già prese

- **La banda ranged resta un filtro di selezione locale a M1.** Gli eventi che superano
  il gate — in modalità `"floor"` o `"ranged"` — arrivano a M2 nella stessa forma
  (`EventCandidate` con la sua serie di attivazione), indistinguibili dal punto di vista
  di M2 in poi. Nessuna modifica alla catena di derivazione del resolver
  (`min_train_months`, `pf_min_tpm`, i controlli `oos_span_too_short` /
  `m3_stricter_than_m1` restano ancorati alla semantica floor di `min_tpm`, invariata).
- **La derivazione di `tpm_tolerance` non può vivere nel resolver di sessione.**
  `eff_max_dispersion` dipende da `n_total_months`, una quantità **nota solo dal
  dataset reale**. Il resolver (`resolver.py`) è per costruzione cieco ai dati — deriva
  tutto da configurazione + `PipelineContext`, mai da `n_bars`/`span_months` — e
  `config_report()`/`forge_preset()` costruiscono le configurazioni senza mai vedere la
  tabella KPI. `eff_max_dispersion` infatti **già oggi** si calcola dentro
  `ConsistencyGate.evaluate()` (`consistency_gate.py:143`), non nel resolver — la
  derivazione di `tpm_tolerance` va nello stesso posto, per lo stesso motivo.
  Conseguenza pratica: a differenza di `min_train_months`/`pf_min_tpm`,
  `tpm_tolerance` derivato **non comparirà** in `ResolutionTrace`
  (`result.resolution.describe()`) — è locale al gate, non alla sessione.
- **Integrazione con `forge_preset()`.** I quattro preset (`sniper`/`balanced`/`sweep`/
  `burst`) non impostano oggi `tpm_mode`/`tpm_tolerance` — comportamento invariato per
  chi non li usa esplicitamente. Per attivarli tramite preset serve aggiungerli alla
  whitelist di override di `forge_preset()` (`presets.py:352-362`, che oggi rifiuta con
  `TypeError` ogni chiave non elencata) esattamente come `min_tpm`/`dispersion_margin`:
  `tpm_mode = overrides.pop("tpm_mode", "floor")`,
  `tpm_tolerance = overrides.pop("tpm_tolerance", UNSET)`, passati a `GateParams(...)`.
  Nessuna voce di calibrazione per-preset serve per `tpm_tolerance` (a differenza di
  `min_tpm`, che ha un valore `daily_min_tpm_episode` da scalare per timeframe) perché
  resta `UNSET` di default su tutti e quattro. Il default derivato eredita comunque la
  filosofia del preset "gratis", perché `eff_max_dispersion` dipende da
  `dispersion_margin`, che ogni preset già fissa in modo diverso (`sniper` quasi zero
  slack → banda stretta; `burst` clustering deliberatamente Poisson-implausibile →
  banda larga).
- **Incompatibilità di preset segnalata:** `"burst"` è disegnato esplicitamente per
  ammettere concentrazione temporale estrema (`dispersion_margin=3.00`, commento in
  `presets.py:300`: *"clustering Poisson-implausibile, di proposito"*) — l'opposto
  filosofico di "seleziona una cadenza tipica". Combinare `tpm_mode="ranged"` con
  `"burst"` è sconsigliato, sullo stesso piano della nota già esistente "non abbinare
  `sniper` al `RotationCalibrator`" (pitfall #6 della skill `forgedge`).

### 2.6 Rischi / vincoli noti

- `min_episodes` (potere statistico minimo) resta un vincolo assoluto separato, come
  oggi — la modalità ranged non lo sostituisce, si somma ad esso.
- Va deciso se la modalità ranged è utilizzabile anche con `event_counting="bar"` oltre
  che con `"episode"` (default) — in `"bar"` mode non esiste un `eff_max_dispersion` con
  floor (`max_dispersion` è lì una soglia assoluta senza floor, si veda §2.2): la scelta
  più coerente sarebbe usare `max_dispersion` stesso come input di `σ_tpm` in quel caso,
  non un'osservazione per-evento.
- Ambiguità di unità già presente su `min_tpm` (episodi/mese vs. barre/mese a seconda di
  `event_counting`) si propaga a `tpm_tolerance`: un valore letterale impostato pensando
  a un modo conta-eventi e poi riletto sotto l'altro cambia scala silenziosamente. Non è
  un problema nuovo introdotto da questa feature, ma va documentato esplicitamente perché
  qui riguarda sia il centro sia il raggio della banda.
- Vincolo implementativo per l'eventuale composizione AND: `ANDComposer.compose()` valuta
  il gate in forma vettorizzata via `_gate_pass` (`consistency_gate.py:307-357`), condivisa
  col path a singolo evento per costruzione (fix #226) — il criterio ranged deve entrare
  nello stesso punto condiviso, non essere reimplementato separatamente nel path batch,
  per non riaprire la stessa classe di bug.

### 2.7 Domande aperte residue

- Se estendere la modalità ranged a `event_counting="bar"` ora o rimandarlo (vedi §2.6).
- Se `z=1.959964` (95% a due code) deve restare fisso nel default derivato o esporre un
  modo per cambiarlo senza dover per forza passare a `tpm_tolerance` letterale.

### 2.8 Validazione empirica — `"balanced"` e `"sniper"`, fixture reale

Metodo: `tpm_mode="ranged"` non esiste ancora nel codice (nessuna sorgente è stata
modificata). Per testare il criterio proposto sui path reali di Event Discovery —
inclusi Diversity Gate e `ANDComposer`, non solo il gate a singolo evento — lo script di
verifica sostituisce a runtime (monkeypatch, solo nel processo dello script, mai nel
pacchetto) l'unico chokepoint condiviso già identificato in §2.6:
`consistency_gate._gate_pass`, importato con lo stesso nome anche da `and_composer.py`
(fix #226 lo rende l'unica fonte di verità sia per il path a singolo evento sia per
quello batch dei compositi). Sostituire quella funzione per la durata di una run è
comportamentalmente equivalente ad avere implementato la modalità per davvero, ai fini
del test. Dataset: `tests/fixtures/ADA_1D_TRAIN.parquet` (882 barre, 29 mesi).

**`"balanced"` (`min_tpm=1.0`, `dispersion_margin=1.30`) — banda derivata `[0.496, 1.504]`:**

| | grezzi (pre-gate) | dopo Diversity Gate + `max_and=2` |
|---|---|---|
| floor (oggi) | 1662 atomici | 1288 atomici + 2000 composti = 3288 |
| ranged (proposto) | 2571 atomici | 2033 atomici + 2000 composti = 4033 |

Risultato inatteso: su questo asset, a default, **ranged è netto più permissivo del
floor attuale** (2571 vs 1662 grezzi; 2033 vs 1288 dopo dedup), non più selettivo.
Il lato "ammetti chi è appena sotto il target" (1694 candidati grezzi tra 0.496 e 1.0
tpm/mese) è molto più popolato del lato "escludi chi è molto sopra" (785 candidati
sopra 1.504, tipicamente 3.5-3.9 tpm/mese — questi sì genuinamente "troppo frequenti",
il caso che l'idea A intende intercettare). Ipotesi iniziale scartata dai dati: il
surplus non è quasi-duplicazione — il tasso di deduplica (~21%) è quasi identico a
quello del floor (~22.5%), quindi la maggioranza dei 1694 sono pattern distinti, non
soglie percentili ridondanti sulla stessa feature.

**`"sniper"` (`min_tpm=0.3`, `dispersion_margin=1.05`) — banda derivata `[0.052, 0.548]`:**

| | grezzi (pre-gate) | dopo Diversity Gate + `max_and=2` |
|---|---|---|
| floor (oggi) | 3809 atomici | 2992 atomici + 2000 composti = 4992 |
| ranged (proposto) | 681 atomici | 561 atomici + 2000 composti = 2561 |

Qui l'effetto si **inverte completamente**: ranged è drasticamente più restrittivo del
floor (681 vs 3809 grezzi, −82%). Con `min_tpm=0.3` il floor di oggi è quasi un
non-filtro (la stragrande maggioranza delle feature si attiva più spesso di una volta
ogni ~3 mesi), mentre il bordo superiore della banda derivata (0.548, appena 1.8× il
target) fa quasi tutto il lavoro di selezione. Nessun candidato rientra nel caso
"ammesso solo da ranged" (`ranged-only = 0`): a `n_total_months=29`, il pavimento di
potenza statistica `min_episodes >= 10` implica già da solo un tasso minimo di
`10/29 ≈ 0.345`, sopra il bordo inferiore della banda (0.052) — quel meccanismo di
"estensione verso il basso" non ha mai la possibilità di attivarsi su questo preset.

**Conclusione della validazione:** la formula (§2.3, tenuta semplice e interpretabile
come deciso) è confermata **corretta e stabile** — nessun bug, nessun comportamento
degenere. Ma il suo **effetto netto sul volume di candidati non è una proprietà fissa
della modalità ranged**: dipende da dove il `min_tpm` del preset cade rispetto alla
distribuzione reale dei tassi delle feature su quell'asset. Un preset con `min_tpm`
basso (`sniper`) vede dominare il bordo superiore (più restrittivo); un preset con
`min_tpm` moderato (`balanced`) vede dominare il bordo inferiore (più permissivo). Va
letto come comportamento atteso e specifico di preset/asset, non come difetto della
formula — coerente con la scelta already presa di non complicarla per inseguire un
effetto netto uniforme.

**Limite noto del metodo:** il conteggio dei compositi satura a 2000 in tutti e quattro
i casi testati — quasi certamente `_MAX_PAIRS`, il tetto duro di `and_composer.py`, non
un segnale di "nessuna differenza a valle". Con pool atomiche anche solo nell'ordine
delle centinaia il numero di coppie possibili supera ampiamente quel tetto, quindi il
conteggio dei compositi non è informativo per confrontare la pressione computazionale
reale tra le configurazioni — solo la dimensione della pool atomica pre-composizione lo è.

---

## 3. Idea B — Etichettatura momentum / mean-reversion / idiosincratica (M2)

### 3.1 Motivazione

Si vuole distinguere, tra i contratti con un vantaggio statistico confermato, quelli il
cui edge è "market beta mascherato" (la regola vince perché il mercato si muoveva già in
quella direzione) da quelli il cui edge è più idiosincratico (vince a prescindere dalla
direzione che il mercato ha effettivamente preso nello stesso periodo). Questa
informazione può servire sia come etichetta descrittiva (mean-reversion / momentum) sia
come indicazione di quanto l'edge sia "vantaggioso rispetto al mercato" in un senso più
ricco della sola media.

### 3.2 Stato attuale (verificato) e due false partenze

- `AlphaContract` deriva già `direction` e `mean_advantage` come eccesso rispetto alla
  baseline **incondizionata**: `Δ_h = μ_cond_h − μ_base_h`, dove `μ_base_h` è il
  rendimento medio a `h` barre su **tutte** le barre (`alpha_discovery/discovery.py:450-454`,
  `models.py:47-48`). Questo È già, nella sostanza, "il vantaggio rispetto al rendimento
  medio di mercato allo stesso orizzonte" — non va reinventato. Un confronto di livello
  aggregato più grezzo — la regola batte il buy&hold cumulato sull'intero periodo OOS —
  resta un'idea collegata più leggera, più naturale come arricchimento del report M3/M4
  (accanto a `BacktestSummary.net_gain`) che come criterio di selezione in M2; non è
  oggetto di questo documento.
- **Prima falsa partenza (corretta in questa revisione): correlare col rendimento di
  mercato sulle stesse barre è degenere.** A livello di M2 non esiste un rendimento
  della regola distinto dal rendimento di mercato: sulla barra attiva, "il rendimento
  della regola" è letteralmente `L0[t, h]` (il forward log-return del mercato), al più
  con segno invertito da `direction`. Correlare le due serie sulle stesse barre allo
  stesso `h` dà quindi **esattamente 1** per un evento long ed **esattamente -1** per uno
  short — codifica solo `direction`, non un'informazione nuova. La formula
  originariamente proposta in questa sezione (Spearman tra rendimento per-attivazione e
  rendimento di mercato "sullo stesso orizzonte h\*, sulle stesse barre") va scartata per
  questo motivo, non per preferenza di design.
- **Seconda considerazione, non uno scarto ma una precisazione trovata per strada:**
  `AlphaContract.regime_analysis` (`models.py:714-722`, `discovery.py:990-1049`) misura
  già, per ogni regime di mercato di M0 (mean-reverting/trending/random-walk),
  l'Information Coefficient dell'evento **dentro quel regime**, classificando
  `dependency_type` come agnostic/conditional/specific/broken. Risponde a "questa regola
  generalizza tra regimi di mercato diversi", non alla domanda di B ("quando questa
  regola si attiva, il suo vantaggio è coerente con la direzione che il mercato aveva
  preso, o no") — non è ridondante, ma è stata la base di due alternative (correlazione
  con lo stato di mercato pregresso; riuso di `RegimeAnalysis` con bucket direzionali)
  **entrambe superate** dalla proposta definitiva in §3.3, più semplice ed economica.
- `AlphaContract.market_structure` (Hurst/ACF) resta una caratterizzazione del mercato
  nel suo complesso, non della singola regola — nessuna sovrapposizione con B.

### 3.3 Proposta funzionale (formule congelate, in due stadi)

Nessuna correlazione con una serie di mercato esterna: tutto deriva dalla **forma del
profilo `Δ_h` dell'evento stesso lungo la griglia degli orizzonti** — un profilo già
corretto per il drift di mercato per costruzione (`Δ_h = μ_cond_h − μ_base_h`).

**Stadio (a) — l'edge è distinguibile dal mercato, sull'intera griglia?** Non un
confronto di magnitudine ("l'area sotto la curva condizionata è più grande di quella di
mercato quindi sono diverse") ma un test di significatività che riusa l'infrastruttura
del rotation-null già esistente (`_rotation_null`, `discovery.py:618-674`), che calcola
**già** — per ogni shift circolare e ogni orizzonte insieme — la matrice
`null[shift, h]` di `Δ_h` sotto la nulla "il timing dell'evento è scorrelato dai
rendimenti". Integrare quella matrice invece di ridurla subito a `(z_h, p_h)` per
orizzonte costa una proiezione lineare, nessun ricalcolo pesante:

```
w = pesi trapezoidali sulla griglia reale (rispetta spaziature non uniformi, es. H={1,2,5,10})
AUC_Δ = Σ_i w_i · (Δ_{h_i} / h_i)                          # tasso per-barra, NON Δ_h grezzo — vedi §3.4
AUC_Δ^(shift) = Σ_i w_i · (null[shift, h_i] / h_i)          # nulla costruita sulla stessa quantità
p_AUC = (1 + #{|AUC_Δ^(shift)| >= |AUC_Δ|}) / (1 + n_shift_validi)   # stessa formula già usata per-orizzonte

edge_distinguibile = p_AUC < soglia
```

Scartata la combinazione alla Fisher dei `p_value_by_h` esistenti come alternativa più
economica: i p-value per-orizzonte sono correlati tra loro (orizzonti vicini condividono
finestre sovrapposte), quindi Fisher li tratterebbe come indipendenti quando non lo
sono, sovrastimando la significatività. Costruire la nulla sulla statistica già
aggregata (come sopra) evita il problema perché la correlazione tra orizzonti è
automaticamente preservata in ogni shift.

**Nota di consistenza:** sia lo stadio (a) sia lo stadio (b) operano ora sulla stessa
quantità corretta, `Δ_h / h` — non solo (b). La prima stesura usava `Δ_h` grezzo in (a);
la validazione empirica (§3.4) ha mostrato che era anch'essa distorta, per lo stesso
motivo aritmetico scoperto correggendo (b).

**Stadio (b) — solo se (a) è positivo: direzione del profilo sulla griglia.**

```
ρ = spearman(h, (segno_direction · Δ_h) / h)    # tasso per-barra, NON Δ_h grezzo

ρ > soglia   → "momentum-aligned"        (il tasso per-barra cresce con l'orizzonte)
ρ < -soglia  → "mean-reversion-aligned"  (il tasso per-barra si affievolisce/inverte)
altrimenti   → "idiosyncratic"           (nessuna pendenza chiara)
```

**Perché `Δ_h / h` e non `Δ_h`: una seconda falsa partenza, questa volta trovata solo
validando sui dati (§3.4).** `Δ_h` è una quantità *cumulata* su una finestra di `h`
barre: qualunque edge persistente ma **costante per barra** mostra `Δ_h` crescente con
`h` per pura aritmetica del log-return cumulato, non perché il fenomeno sia
trend-following. La versione con `Δ_h` grezzo etichettava quasi ogni edge genuino come
momentum. Dividere per `h` (esatto in log-space, dove i rendimenti si sommano
linearmente — non un'approssimazione) dà il tasso medio per barra: costante per un edge
piatto (`ρ≈0`, idiosyncratic, corretto), crescente per un edge che si autoalimenta
(`ρ>0`, momentum), decrescente per una reazione che si affievolisce (`ρ<0`,
mean-reversion).

Entrambi gli stadi producono un **diagnostico aggiuntivo**, non un nuovo gate di
promozione — non toccano `direction`, `lift`, né `promoted_contracts()`.

### 3.4 Validazione empirica — fixture reale, preset `"balanced"`

Metodo: nessuna sorgente modificata. `AlphaDiscovery._derive_target` è avvolto
(monkeypatch solo nel processo dello script, stessa tecnica usata per l'idea A) al solo
scopo di catturare i suoi stessi argomenti in ingresso (`active_is, valid_is, L0, cnt_t,
sum_t, horizons`) — la funzione originale viene comunque chiamata e il suo risultato
restituito invariato, quindi i contratti prodotti sono identici a una run non
modificata. Da quegli argomenti lo script ricalcola `mu_base`/`delta` e la matrice
`null` con la stessa identica formula FFT già in `_rotation_null` (righe 653-663),
senza toccare il pacchetto. Dataset: `tests/fixtures/ADA_1D_TRAIN.parquet`, griglia
giornaliera `(1,2,3,5,7,10)`.

**Prima passata — perché anche lo stadio (a) andava corretto.** La prima versione dello
stadio (a) integrava `Δ_h` grezzo, non il tasso. Decomponendo la somma pesata
termine-per-termine su tutti i 1662 candidati:

| | quota media di \|AUC\| dal termine dominante | il termine dominante è l'ultimo orizzonte |
|---|---|---|
| `Δ_h` grezzo | 40% | **40.4%** dei casi |
| `Δ_h / h` (corretto) | 31% | **10.7%** dei casi |

Con `Δ_h` grezzo, in 4 casi su 10 la "significatività integrata sulla griglia" era in
realtà quasi interamente il contributo di un solo punto — quasi sempre il più lungo
scansionato: lo stesso problema aritmetico che (b) aveva già rivelato (`Δ_h` è cumulato,
cresce con `h` per qualunque edge persistente anche solo costante per barra),
ripresentato in una forma diversa. Confrontando la selezione "solo AUC" (candidati che
l'AUC promuove e il metodo attuale no) tra le due versioni: con `Δ_h` grezzo, `h*`
mediano = 10 e 43% dei casi al bordo della griglia; con `Δ_h/h`, `h*` mediano = 5 e
solo 6.7% al bordo. La correzione elimina quasi del tutto la concentrazione al bordo —
non era un segnale reale, era l'artefatto aritmetico.

**Numeri finali, entrambi gli stadi corretti su `Δ_h/h`.** Selezione — metodo attuale
(direzione derivabile) vs metodo AUC, sugli stessi 1662 candidati atomici. `h*`,
`direction` e `sell_pct` vengono dalla *stessa* derivazione in entrambi i casi — cambia
solo **quali** candidati ciascun metodo lascia passare:

| | AUC non significativo | AUC significativo (`p_AUC<0.10`) |
|---|---|---|
| **attuale: undetermined** | 1437 | 45 — *"solo AUC"* |
| **attuale: direzione derivata** | 69 — *"solo attuale"* | 111 — *entrambi* |

- **Entrambi (111):** `h*` mediano = 3, 18.9% al bordo.
- **Solo attuale (69):** `h*` mediano = **2**, 10.1% al bordo — il test a singolo
  orizzonte (BH-FDR) trova un picco di significatività a orizzonte breve che non regge
  sul profilo integrato. Rischio del metodo attuale: falsi positivi su orizzonti
  brevi/rumorosi.
- **Solo AUC (45):** `h*` mediano = 5, **solo 6.7% al bordo** — nessun singolo orizzonte
  supera il BH-FDR (`direction` oggi `"undetermined"`, `sell_pct` mai calcolato), ma
  l'effetto integrato su più orizzonti è significativo: un contributo moderato e
  distribuito, non concentrato in un punto. A differenza della prima passata, questi
  casi **non** si concentrano più al bordo della griglia — la nota di collegamento con
  l'idea C ipotizzata nella prima stesura era anch'essa un riflesso dell'artefatto
  aritmetico, non un fenomeno reale.

Non è un rapporto di sottoinsieme: il metodo AUC seleziona un insieme **diverso**, con un
profilo di falsi positivi/negativi diverso, non semplicemente "più" o "meno" del metodo
attuale.

**Etichetta dello stadio (b) sui 156 candidati (111+45) che superano lo stadio (a)
corretto:**

| | n | % |
|---|---|---|
| mean-reversion-aligned | 107 | 68.6% |
| idiosyncratic | 44 | 28.2% |
| momentum-aligned | 5 | 3.2% |

Distribuzione plausibile per pattern basati su indicatori tecnici (incroci, soglie di
percentile) su questo asset/timeframe: prevalentemente reazioni che si affievolisce con
l'orizzonte, non trend che si autoalimentano.

### 3.5 Rischi / vincoli noti

- La soglia di `p_AUC` (qui 0.10, illustrativa) e la soglia di `ρ` (qui 0.5,
  illustrativa) vanno agganciate alla calibrazione statistica esistente di M2
  (`PromotionThresholds`), non lasciate come numeri liberi.
- Il test (a) va calcolato sul periodo di stima coerente con la validazione IS/OOS già
  esistente, per non introdurre una forma di overfitting non controllata dalla
  walk-forward/rotazione già presente.
- Il legame con l'idea C ipotizzato nella prima stesura non regge nemmeno dopo la
  seconda correzione: la concentrazione al bordo osservata nella prima passata era un
  artefatto della formula (`Δ_h` grezzo in entrambi gli stadi), non un fenomeno che
  l'idea C intercetterebbe. Nessuna dipendenza stretta tra B e C.
- Prima di congelare le soglie definitive va ripetuta la stessa doppia verifica (bontà
  della correzione + confronto di selezione) su un secondo preset/asset, sullo stesso
  modello di §2.8 per l'idea A — un'unica validazione ha già portato a due correzioni
  successive, motivo in più per non fermarsi a un solo caso.

### 3.6 Domande aperte

- Calcolare gli stadi (a)/(b) anche sui candidati con `direction` oggi `"undetermined"`
  (necessario per catturare i casi "solo AUC" osservati in §3.4) o solo su quelli già
  diretti — la validazione mostra che la prima opzione è quella che cattura il valore
  aggiunto reale del metodo.
- Esporre anche i valori quantitativi (`p_AUC`, `ρ`) oltre alle etichette categoriche.
- Ripetere la validazione su un secondo preset/asset prima di congelare le soglie
  definitive, sullo stesso modello di §2.8 per l'idea A.

---

## 4. Idea C — Diagnostica di sufficienza della griglia degli orizzonti (M2)

### 4.1 Motivazione

Emersa discutendo l'idea B, ma **tracciata come idea a sé stante**: è utile a
prescindere dall'etichettatura momentum/mean-reversion, ogni volta che si legge
`holding_period_h` come "l'orizzonte migliore" per un evento.

### 4.2 Stato attuale (verificato)

- Il codice ha già un guardrail per un problema **imparentato ma diverso**:
  `require_significant_direction` (`alpha_discovery/models.py:239-243`,
  `discovery.py:526-529`) declassa `direction` a `"undetermined"` quando **nessun**
  orizzonte della griglia è statisticamente significativo (BH-corrected) — il commento
  nel codice lo motiva esplicitamente: *"argmax|z_h| would assign a direction off a
  coin-flip (often the drift-driven long edge of the grid)"*.
- Questo guardrail protegge dal caso "segnale debole/rumore che casualmente si accumula
  verso il bordo della griglia". **Non protegge** dal caso opposto: un edge trend-following
  **vero e significativo**, dove `|z_h|` è ancora crescente all'ultimo punto scansionato —
  h\* pinnato al bordo non per rumore ma perché la griglia è strutturalmente troppo
  corta per quell'evento.
- `holding_period_h` alimenta a cascata: `purge_bars = max(horizon_grid)` in
  `TimeBudget.build` (`discovery.py:162`), e attraverso il resolver il sizing di
  `min_train_months` e altri campi bar-counting di M3 (`target_h`, `buy_delay_bar`).
  Un h\* artefatto si propaga quindi oltre M2.
- Esiste già un meccanismo di allargamento mirato della griglia — `EventCandidate
  .dominant_window()` più i moltiplicatori `(0.5, 1.0, 2.0)` (`models.py:447-451`) — ma è
  agganciato alla finestra dell'indicatore che ha generato l'evento, non a un test sulla
  forma di `|z_h|` osservata; non garantisce che un trend-follower genuino trovi un
  picco interno.

### 4.3 Proposta funzionale (formule congelate)

Tutto il necessario è **già calcolato e già esposto** su `AlphaContract.derived_target`
(`DerivedTarget.score_by_h: Dict[int, float]`, `models.py:85-87,119` — il punteggio
`|z_h|` per orizzonte, la stessa quantità con cui `h*` viene scelto via
`argmax|z_h|`, `discovery.py:496`) — nessun nuovo dato, nessuna nuova statistica.

**Rilevamento del bordo.** Il confronto va fatto contro il massimo degli orizzonti con
punteggio **finito**, non contro `max(AlphaConfig.horizon_grid)`: un orizzonte scartato
per `cnt_a < 2` (troppo pochi bar attivi validi) ha `score_by_h[h] = NaN` ed esce
dall'`argmax` comunque — confrontare col grid configurato tratterebbe come "bordo" anche
un caso in cui l'ultimo punto realmente valutato era prima.

```python
horizons_validi = sorted(h for h, s in derived_target.score_by_h.items() if isfinite(s))
h_star = derived_target.holding_period_h
al_bordo = h_star == horizons_validi[-1]
```

**Distinzione fine — tre stati, non due.** Un confronto a un solo passo è rumoroso
(§4.4 originale). Servono almeno due incrementi consecutivi di `|z_h|` per dichiarare il
caso ad alta confidenza:

```python
scores = [derived_target.score_by_h[h] for h in horizons_validi]  # ordine crescente di h

if not al_bordo:
    stato = "interno"                     # picco genuino, nessun problema
elif len(scores) < 3:
    stato = "bordo_grid_troppo_corta"      # non abbastanza punti per giudicare il trend
elif scores[-1] > scores[-2] > scores[-3]:
    stato = "bordo_in_salita"              # due incrementi consecutivi — alta confidenza
elif scores[-1] > scores[-2]:
    stato = "bordo_ambiguo"                # un solo incremento — potrebbe essere rumore
else:
    stato = "bordo_plateau"                # ultimo punto non superiore al precedente
```

Uguaglianza esatta (`scores[-1] == scores[-2]`) è trattata come **non** salita — un
plateau all'ultimo punto suggerisce di aver catturato l'asintoto, non di essere a metà
di una crescita. Richiedere due incrementi consecutivi (non uno) usa metà di una griglia
tipica (griglie standard a 5-6 punti: `DAILY=(1,2,3,5,7,10)`, `INTRADAY=(1,2,4,8,12,24)`)
— una soglia deliberatamente severa, per non segnalare "sotto-scansionato" su un singolo
rialzo che potrebbe essere rumore del rotation-null.

**Rappresentazione.** Due nuovi valori non-bloccanti in `AlphaContract.diagnostics`
(stesso campo usato oggi per altre annotazioni — pitfall #12 della skill `forgedge`, non
`rejection_reasons`), non un flag booleano:
- `"horizon_at_grid_boundary_climbing"` — stato `bordo_in_salita` (alta confidenza).
- `"horizon_at_grid_boundary_ambiguous"` — stato `bordo_ambiguo` o `bordo_grid_troppo_corta`.

Due livelli invece di uno darebbero all'idea B due gradi di declassamento della propria
etichetta invece di un taglio netto sì/no — nota storica: la validazione empirica di B
(§3.4) ha poi escluso il pinning al bordo come causa dei problemi trovati in quella sede,
quindi questo collegamento resta un affinamento accessorio, non una dipendenza stretta.

**Esclusione per `fixed_target=True`.** Quando il target è fissato dall'utente
(`TargetOptimizer`/`AlphaConfig.fixed_target`), `holding_period_h` non viene da
`argmax|z_h|` — viene dall'utente. Applicare la diagnostica a `holding_period_h` in quel
caso segnalerebbe "la tua scelta esplicita è al bordo", che non è lo stesso tipo di
problema (non un artefatto di ricerca, una decisione dell'utente). La diagnostica va
applicata invece a `data_derived_horizon_h` (il valore diagnostico "cosa avrebbe scelto
la derivazione dai dati", già documentato per verificare la convergenza) solo quando
quel campo non è `None`.

**Nota, non azionabile:** lo stesso problema esiste simmetricamente al bordo
**inferiore** della griglia (punteggio ancora in crescita mentre `h` scende verso il
minimo scansionato) — ma lì generalmente non è correggibile: le griglie iniziano già a 1
barra, la finezza più fine possibile per quel timeframe.

Il risultato (`stato`, e quindi il valore in `diagnostics`) è la base su cui l'idea B
declassa la propria etichetta (§3.4).

### 4.4 Rischi / vincoli noti

- Allargare meccanicamente `horizon_grid` ogni volta che si rileva il pinning **non è
  gratis**: `purge_bars` cresce con `max(horizon_grid)`, riducendo la finestra di
  training utile; il sizing del resolver (`min_train_months` e la catena collegata)
  andrebbe rivalutato. La proposta qui resta **solo diagnostica** (segnalare, non
  correggere automaticamente allargando la griglia) — un'eventuale politica di
  allargamento automatico è una proposta successiva e separata, con i suoi trade-off da
  valutare a parte.
- Il criterio "due incrementi consecutivi" presuppone almeno 3 orizzonti validi vicino al
  bordo — su una griglia molto corta o molto diradata dall'enrichment (§4.2,
  `dominant_window` + moltiplicatori) lo stato `bordo_grid_troppo_corta` sarà frequente:
  è un esito onesto ("non posso giudicare"), non un difetto della soglia.

### 4.5 Domande aperte residue

- Se calcolare la diagnostica solo su `direction != "undetermined"` (dove esiste un h\*
  con un senso economico) o anche sui candidati scartati, per capire *perché* — decisione
  a basso rischio, rimandabile alla specifica tecnica: il calcolo è comunque a costo
  quasi zero in entrambi i casi.
- Validazione empirica sul fixture di riferimento (stesso metodo usato per l'idea A,
  §2.8): quanti contratti reali cadono in ciascuno dei quattro stati, e se
  `bordo_in_salita` si concentra sugli eventi già etichettabili come trend-following da
  altri segnali (Hurst/`market_structure` alto).

---

## 5. Relazioni tra le idee e sequenza consigliata

```
A (M1, ranged tpm)         — indipendente, nessuna dipendenza dalle altre due
B (M2, market alignment)   — indipendente da C (dipendenza ipotizzata, poi esclusa
                              empiricamente in §3.4); usa solo il proprio profilo Δ_h
C (M2, grid sufficiency)   — indipendente nell'implementazione; resta una nota
                              diagnostica accessoria per i casi "solo AUC" di B (§3.4)
```

Le tre idee non hanno più un ordine di implementazione obbligato: A è indipendente per
costruzione; B, dopo la validazione empirica, si è rivelata indipendente da C nella
pratica (una dipendenza concettuale era stata ipotizzata in una prima stesura, ma i dati
mostrano che la causa dei problemi trovati in B era nella formula di B stessa, non nel
pinning al bordo che C rileva).

## 6. Prossimi passi

Questo documento resta a livello di specifica funzionale. I passi successivi, da
decidere con l'utente:

- **Idea A: validazione empirica chiusa** (§2.8) — formula confermata corretta e
  stabile su due preset (`"balanced"`, `"sniper"`) sul fixture di riferimento; l'effetto
  netto sul volume di candidati varia per preset/asset per una ragione capita e
  spiegata, non per un difetto della formula. Resta da congelare solo il dettaglio
  tecnico residuo di §2.7 (`"bar"` mode, se esporre `z`).
- **Idea C: formule congelate** (§4.3) — meccanismo a tre/quattro stati derivato
  interamente da `DerivedTarget.score_by_h`, già esposto sul contratto, nessun nuovo
  dato. Resta da fare la validazione empirica (§4.5, stesso metodo di §2.8) prima di
  considerarla chiusa come A.
- **Idea B: formule congelate e validazione empirica su un preset** (§3.3-§3.4) — due
  stadi (gate AUC via rotation-null riusato, poi pendenza di `Δ_h/h` sulla griglia). La
  validazione ha già corretto una formula sbagliata (`Δ_h` grezzo) e mostrato una
  differenza di selezione concreta rispetto al metodo attuale. Restano aperte le soglie
  (§3.6) e la ripetizione su un secondo preset/asset, come già fatto per A.
- Solo dopo: apertura di branch/issue separati per A, B, C.
