# Proposta funzionale — modalità ranged per `min_tpm` (M1) e diagnostica di allineamento al mercato (M2)

**Stato del documento:** specifica funzionale con **metodologia congelata su tutte e tre
le idee** (A, B, C) — non design tecnico né implementazione. Ogni formula è stata
validata empiricamente sul fixture di riferimento (ADA) e su almeno due asset aggiuntivi
dove rilevante (BTC, EURUSD); i tentativi di raffinamento che non hanno retto alla
validazione sono documentati come tali, non nascosti. Restano solo dettagli di
calibrazione (soglie numeriche, un secondo preset da ripetere) rimandati alla specifica
tecnica — nessuna domanda aperta sul *metodo* in sé. Nessun codice è stato scritto:
questo documento è il punto di partenza per il design tecnico successivo.

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

A è indipendente nell'implementazione dalle altre due (modulo diverso, M1 vs M2). Una
dipendenza concettuale B→C era stata ipotizzata in una prima stesura per il criterio
BH-FDR esistente, ma la validazione empirica di B (§3.4) l'ha esclusa come causa dei
problemi trovati lì (si veda §3.5). Il rafforzamento OR della selezione di B (§3.9),
però, ha fatto emergere un'unificazione reale e diversa: per i candidati promossi via il
nuovo test AUC, la scelta di `h*` (§3.9) e la diagnostica di sufficienza della griglia
di C (§4.5) sono **la stessa domanda posta sulla stessa quantità** (il profilo
cumulato `Δ_h`), non due meccanismi paralleli.

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

### 2.9 Validazione multi-asset — BTC ed EURUSD

Estensione di §2.8 a due asset ulteriori, costruiti da OHLCV grezzo
(`examples/data/BTCEUR_1DAY.csv`, `examples/data/EURUSD_1DAY.csv`) via
`build_features`/`candle_features` — nessuna sorgente forgedge modificata, stesso
metodo di §2.8. **Limite del confronto**: `ADA_1D_TRAIN.parquet` è un fixture
pre-costruito con una configurazione di feature diversa da quella di default usata qui
per BTC/EURUSD (4495 candidati grezzi contro ~47000) — il confronto BTC↔EURUSD è pulito
(stessa config), quello con ADA no.

| asset | mesi | `σ_tpm` (balanced) | rapporto ranged/floor, grezzo | dopo dedup |
|---|---|---|---|---|
| ADA | 29 | 0.257 | 1.55× | 1.58× |
| BTC | 43 | 0.205 | 1.26× | 1.31× |
| EURUSD | 69 | 0.156 | **0.97×** | 0.99× |

| asset | mesi | `σ_tpm` (sniper) | rapporto ranged/floor, grezzo | dopo dedup |
|---|---|---|---|---|
| ADA | 29 | 0.127 | 0.18× | 0.19× |
| BTC | 43 | 0.101 | 0.39× | 0.37× |
| EURUSD | 69 | 0.077 | 0.47× | 0.40× |

**Osservazione di troubleshooting.** Per `"balanced"` il rapporto scende
monotonicamente all'aumentare dei mesi di storia — coerente con `σ_tpm ∝ 1/√n_mesi`
(più dati → banda più stretta → meno ammissione dal basso): il meccanismo si comporta
come progettato. Ma per `"sniper"` il rapporto **sale** con più mesi — direzione
opposta. Non è quindi una relazione generale e pulita con `n_mesi`: la forma della
distribuzione dei tassi grezzi propria di ciascun asset pesa almeno quanto l'ampiezza
della banda, e l'interazione tra le due non si riduce a una regola unica. Nessun bug
individuato — la formula resta corretta e stabile, il suo effetto netto è
genuinamente specifico di asset e preset insieme, non solo di preset come la sola
verifica su ADA avrebbe potuto suggerire.

Il conteggio dei compositi satura a 2000 in **tutte e 12** le combinazioni
asset×preset×modalità testate — conferma ulteriore, non più solo su ADA, che è
`_MAX_PAIRS` a saturare, non un segnale sostanziale.

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

### 3.5 Validazione multi-asset — BTC, EURUSD e DAX

Stesso metodo di §3.4, preset `"balanced"`, sugli asset di §2.9 (BTC ed EURUSD, KPI
table costruite da OHLCV grezzo — stesso limite di confronto con ADA già segnalato lì),
più un quarto asset aggiunto appositamente: **DAX 1D**
(`examples/data/E_DAAX_1DAY.csv`, 1759 barre, 2021-01-03 → 2026-09-02) — l'unico dei
quattro con un **trend forte e pressoché costante** per tutto il campione (rendimento
log cumulato +63%, drift medio ~0.00036/barra, nessun ritracciamento prolungato). Serve
da banco di prova mirato per il momentum: sugli altri tre asset i casi momentum-aligned
erano troppo pochi (1-10) per dire granché.

**Selezione — metodo attuale vs metodo AUC corretto:**

| asset | entrambi (h\* med) | solo attuale (h\* med) | solo AUC (h\* med, % bordo) |
|---|---|---|---|
| ADA | 111 (3) | 69 (2) | 45 (5, 6.7%) |
| BTC | 491 (2) | 209 (1) | 237 (5, 13.1%) |
| EURUSD | 356 (5) | 391 (1) | 184 (7, 13.6%) |
| DAX | 557 (3) | 450 (1) | 182 (7, 13.7%) |

Pattern identico sui quattro asset: "solo attuale" ha sempre l'h\* mediano più basso
(1-2) — il rischio di falsi positivi a orizzonte breve del metodo attuale non è
specifico di ADA. "Solo AUC" resta lontano dal bordo su tutti e quattro (6.7-13.7%, mai
vicino al 43% della formula sbagliata di §3.4) — la correzione tiene anche sull'asset
più trending testato finora.

**Etichetta stadio (b):**

| asset | n | mean-reversion-aligned | idiosyncratic | momentum-aligned |
|---|---|---|---|---|
| ADA | 156 | 68.6% | 28.2% | 3.2% |
| BTC | 728 | **83.4%** | 15.2% | 1.4% |
| EURUSD | 540 | 60.7% | 31.7% | **7.6%** |
| DAX | 739 | 72.8% | 23.0% | **4.2% (n=31)** |

Il mean-reversion domina **anche su DAX** (72.8%) — non è un artefatto di asset "poco
trendy": nemmeno un drift fortissimo e costante rende il pattern medio scoperto da M2
prevalentemente trend-following. Ma il campione momentum sale a 31 casi (contro 1-10
sugli altri tre) — il primo abbastanza grande da vedere cosa succede a `ρ` in coda: il
valore massimo osservato è **0.976**, un profilo quasi puramente trend-following mai
visto negli altri asset. Nessun terzo bug trovato in questo giro: la doppia correzione
di §3.3-§3.4 regge su dati indipendenti, incluso l'asset scelto apposta per essere il
caso più sfavorevole per il mean-reversion.

### 3.6 Casi studio — perché i due metodi divergono, orizzonte per orizzonte

Analisi di dettaglio su ADA (`"balanced"`): profilo completo per-orizzonte di sei
regole rappresentative per ciascuno dei due gruppi che dividono i metodi (§3.4),
per capire il meccanismo, non solo il conteggio aggregato.

**Gruppo "solo attuale" (BH-FDR promuove, AUC no) — tre meccanismi distinti:**

1. *Decadimento veloce diluito dai pesi trapezoidali* (il più frequente):

   ```
   ALPHA-ADA-1D-260912-007  (close_ret_03 > 0.089)
   h    1       2       3       5       7       10
   rate 0.0136  0.0088  0.0068  0.0027  0.0014  0.0025
   z    2.20*   1.62*   1.38    0.64    0.38    0.79
   ```
   `h_sig=(1,2)` — due orizzonti individualmente significativi, non un colpo di
   fortuna isolato — ma `p_AUC=0.169`, non passa. Il tasso decade rapidamente dopo
   h=2 (snap-back tipico), mentre i pesi trapezoidali su `(1,2,3,5,7,10)` **crescono**
   verso il centro/coda (`w(7)=2.5`, il massimo, contro `w(1)=0.5`): il test pesa di
   più gli orizzonti lunghi dove il segnale si è già spento, diluendo un effetto reale
   ma concentrato all'inizio.

2. *Vero picco isolato* (il caso che lo stadio (a) dell'idea B doveva intercettare):

   ```
   ALPHA-ADA-1D-260912-037  (delta_close_ret_03_1 > 0.053)
   h    1      2      3      5      7      10
   z    1.08   1.47   2.17*  1.18   0.16   0.01
   ```
   Solo h=3 è significativo, isolato tra rumore prima e dopo. `p_AUC=0.165`. Qui il
   test funziona esattamente come previsto.

3. *Orizzonte diradato dall'enrichment che domina la somma pesata* (non previsto in
   fase di design):

   ```
   ALPHA-ADA-1D-260912-996  (ratio_high_low_lag12 < 0.920)
   h_sig = (1, 2, 3, 5, 6)  — CINQUE orizzonti su 9 individualmente significativi
   p_AUC = 0.216 — non passa comunque
   ```
   Nonostante un profilo di significatività ampio, non concentrato in un punto,
   l'evento ha orizzonti aggiunti dall'enrichment (6, 12, 24 oltre alla griglia base)
   — a `h=24` il peso trapezoidale è enorme (`w(24)=(24-12)/2=6`, contro `w(1)=0.5`),
   e lì il tasso è quasi nullo (0.0018) ma con quel peso pesa quanto l'intero gruppo
   di orizzonti brevi significativi messi insieme.

**Gruppo "solo AUC" (AUC promuove, BH-FDR no) — un pattern unico e pulito:**

Tutte e sei le regole esaminate hanno `h_sig=()` — **nessun** orizzonte
individualmente BH-significativo — ma mostrano segno **stabile su tutta la griglia
scansionata, mai un'inversione**:

```
ALPHA-ADA-1D-260912-256  (pr_ratio_close_ret03_ret96_96 < 0.083)  direzione implicita: short
h    1        2        3        5        7        10
rate -0.0049  -0.0066  -0.0090  -0.0087*  -0.0051  -0.0039
z    -0.73    -1.22    -1.85    -2.18     -1.49    -1.39
```
Negativo su ogni singolo orizzonte, mai un cambio di segno, eppure nessun punto
supera BH-FDR individualmente — `p_AUC=0.065`. Lo stesso schema si ripete identico
nelle altre cinque regole esaminate (028, 088, 057, 136, 048): z tipicamente tra 1.0
e 2.0, mai abbastanza forte in un punto isolato, sempre presente ovunque si guardi —
l'opposto esatto del gruppo "solo attuale" (forte-ma-solo-qui vs. debole-ma-ovunque).
Nessuna delle sei mostra il pattern "picco poi inversione" del gruppo precedente.

**Verifica del vantaggio reale (gruppo "solo AUC"): confronto con la baseline di
mercato, globale e locale.** Prima di fidarsi delle sei regole sopra, verifica diretta
se misurano un vantaggio proprio dell'evento o solo drift di mercato che trapela.

*Confronto globale* — `μ_base_h` (rendimento medio non condizionato, l'intera storia
IS) è identico per le sei regole a ogni h (~0.0005-0.0008/barra — coerente con un
drift rialzista pressoché costante di ADA nel periodo campionato) mentre il tasso
condizionato all'evento è **6-11× più grande**:

| regola | \|tasso evento\| medio | \|tasso mercato\| medio | rapporto |
|---|---|---|---|
| 028 | 0.00704 | 0.00062 | 11.4× |
| 088 | 0.00591 | 0.00061 | 9.7× |
| 057 | 0.00362 | 0.00059 | 6.2× |
| 256 | 0.00637 | 0.00058 | 10.9× |
| 136 | 0.00430 | 0.00062 | 6.9× |
| 048 | 0.00666 | 0.00059 | 11.3× |

Il caso più convincente è la regola 256 (`pr_ratio_close_ret03_ret96_96 < 0.083`,
direzione implicita short): `mu_cond_rate` è **negativo su ogni orizzonte** mentre
`mu_base_rate` è **sempre positivo** (il mercato saliva in media) — l'evento spinge il
percorso condizionato nella direzione opposta a dove il mercato tende naturalmente,
non un pezzo di drift travestito.

*Confronto locale* — stesso confronto, ma con la baseline ristretta all'unione
deduplicata delle finestre di ±15 barre intorno a ogni attivazione (attivazione
inclusa), invece che l'intera storia: controlla "era genericamente un buon/cattivo
momento vicino a queste attivazioni", non solo "rispetto a tutta la storia
pluriennale".

| regola | attivazioni | copertura finestra locale | rapporto vs globale | rapporto vs locale |
|---|---|---|---|---|
| 028 | 50 | 78.6% | 11.4× | **1.9×** |
| 088 | 66 | 79.9% | 9.7× | 5.8× |
| 057 | 101 | **99.8%** | 6.2× | 5.6× |
| 256 | 49 | 70.0% | 10.9× | **23.8×** |
| 136 | 53 | 96.3% | 6.9× | 5.0× |
| 048 | 58 | 92.5% | 11.3× | 9.0× |

**Avvertimento sulla copertura**: su una storia IS di ~617 barre, con eventi che si
attivano 49-101 volte, l'unione delle finestre ±15 copre già dal 70% al 99.8%
dell'intera serie (per la regola 057, la finestra "locale" *è* quasi tutta la serie)
— il confronto locale aggiunge poco di indipendente rispetto al globale quando
l'evento non è raro rispetto alla lunghezza della storia disponibile. Nonostante
questo, il rapporto resta **sempre sopra 1×** (1.9×-23.8×): il tasso dell'evento
supera comunque il tasso del solo vicinato locale, non solo quello dell'intera
storia. Il caso 256 diventa persino più convincente in locale (23.8× — vicino alle
sue attivazioni il mercato non aveva quasi drift, eppure l'evento produce un
rendimento condizionato negativo netto). Il caso più debole sotto questa lente più
severa è il 028 (rapporto crolla da 11.4× a 1.9×) — resta sopra 1× ma è il meno
robusto del gruppo.

**Conclusione della verifica**: le regole "solo AUC" misurano un vantaggio reale
specifico dell'evento, non un artefatto di deriva di mercato — confermato sia contro
la baseline globale sia (con il caveat sulla copertura) contro quella locale.

**Sintesi.** I due gruppi non sono simmetrici per caso: il metodo attuale è
strutturalmente sensibile a un picco concentrato (anche isolato, meccanismo 2 sopra);
il metodo AUC è strutturalmente sensibile alla coerenza di segno distribuita
(gruppo "solo AUC"), ma — meccanismi 1 e 3 sopra — può **mancare** un effetto reale se
è concentrato all'inizio della griglia e diluito dai pesi lunghi, o se un singolo
orizzonte diradato dall'enrichment domina la somma anche a fronte di un profilo già
ampiamente significativo. Nessuno dei due metodi è strettamente superiore in ogni
caso: misurano cose diverse, con vulnerabilità diverse.

### 3.7 Rischi / vincoli noti

- **Nuovo, dai casi studio §3.6: i pesi trapezoidali penalizzano sistematicamente gli
  edge veloci e concentrati.** Non è un errore nel calcolo — l'integrale pesato è
  esattamente quello specificato — ma è un limite di *design* dello schema di pesi:
  su una griglia con salti crescenti (`1,2,3,5,7,10`), i punti centrali/finali pesano
  più di quelli iniziali, quindi un effetto reale ma a decadimento rapido (proprio il
  pattern mean-reversion che §3.4-§3.5 trovano dominante tra ciò che *supera* lo
  stadio (a)) rischia di essere diluito quando è anche più forte all'inizio della
  griglia rispetto alla coda. Un singolo orizzonte aggiunto dall'enrichment e molto
  distante dai vicini (es. h=24 dopo h=12) può ricevere un peso sproporzionato e
  dominare la somma anche a fronte di un profilo altrimenti ampio e coerente
  (caso 996). Da rivalutare in una revisione successiva (es. pesi legati alla
  varianza attesa a ciascun orizzonte invece che alla sola spaziatura) — non
  affrontato in questa specifica.
- La soglia di `p_AUC` (qui 0.10, illustrativa) e la soglia di `ρ` (qui 0.5,
  illustrativa) vanno agganciate alla calibrazione statistica esistente di M2
  (`PromotionThresholds`), non lasciate come numeri liberi.
- Il test (a) va calcolato sul periodo di stima coerente con la validazione IS/OOS già
  esistente, per non introdurre una forma di overfitting non controllata dalla
  walk-forward/rotazione già presente.
- Il legame con l'idea C ipotizzato nella prima stesura non regge, per la selezione
  BH-FDR esistente, nemmeno dopo la seconda correzione: la concentrazione al bordo
  osservata nella prima passata era un artefatto della formula (`Δ_h` grezzo in entrambi
  gli stadi), non un fenomeno che l'idea C intercetterebbe lì. Per il nuovo percorso di
  promozione via AUC (§3.9), invece, B e C sono **la stessa diagnostica** — si veda §3.9
  e §4.5.
- Resta da ripetere la validazione anche sul preset `"sniper"` (qui testato solo per
  l'idea A, §2.9) prima di considerare le soglie definitive.

### 3.8 Domande aperte

- Calcolare gli stadi (a)/(b) anche sui candidati con `direction` oggi `"undetermined"`
  (necessario per catturare i casi "solo AUC" osservati in §3.4) o solo su quelli già
  diretti — la validazione mostra che la prima opzione è quella che cattura il valore
  aggiunto reale del metodo.
- Esporre anche i valori quantitativi (`p_AUC`, `ρ`) oltre alle etichette categoriche.
- Ripetere la validazione anche sul preset `"sniper"` prima di congelare le soglie
  definitive.

### 3.9 Rafforzamento OR della selezione e scelta di `h*`/`direction` per i candidati "solo AUC" (formula congelata) — unificazione con l'idea C

Risponde alla prima domanda aperta di §3.8: come promuovere i candidati "solo AUC"
(§3.4) mantenendo, non sostituendo, il criterio BH-FDR esistente.

**La modifica è minima.** `_derive_target` (`discovery.py:479-566`) calcola già
`h*`, `direction` e `sell_pct` per **ogni** candidato — la clausola che oggi li scarta è
una sola:

```python
undetermined = (
    not isfinite(adv) or adv == 0.0
    or not isfinite(z_star) or abs(z_star) < min_direction_t
    or (require_significant and statistically_weak)   # <- h_sig=() sempre per "solo AUC"
)
```

Per il gruppo "solo AUC", `h_sig=()` per definizione, quindi `statistically_weak=True`
sempre e questa clausola scatta a prescindere dalla forza del segnale. Il rafforzamento
OR è un cambio di una riga:

```python
or (require_significant and statistically_weak and not auc_significant)
```

`direction` e `sell_pct` non richiedono una nuova derivazione — escono dallo stesso
codice già eseguito oggi, semplicemente non più scartati.

**`direction` è robusta per questo gruppo, per costruzione.** `adv = delta[j_star]` è il
delta a un solo orizzonte, ma i casi studio di §3.6 mostrano che per "solo AUC" il segno
è stabile su tutta la griglia, mai un'inversione — è la stessa proprietà che rende
`AUC_Δ` significativo. Qualunque orizzonte scelga `j_star`, il segno di `adv` sarebbe lo
stesso: nessun rischio "coin-flip" per questo gruppo, a differenza del caso che la
clausola originale vuole prevenire (profilo a picco isolato, gruppo "solo attuale").

**`h*` non lo è.** Con `h_sig=()`, `h*` oggi viene scelto da `argmax|z_h|` su tutta la
griglia — ma il profilo "solo AUC" è per costruzione piatto e diffuso (z moderati
ovunque, mai un picco), quindi l'argmax vince per un margine spesso minimo e può cadere
su un punto poco rappresentativo. `sell_pct` (quantile MFE calcolato *a* `h*`) eredita
quell'instabilità anche quando `direction` non ne soffre.

**Tre alternative validate sul fixture ADA (`"balanced"`, 225 candidati nei tre gruppi
rilevanti: 111 "entrambi", 69 "solo attuale", 45 "solo AUC"):**

```python
w = pesi trapezoidali sulla griglia reale        # gli stessi di §3.3
rate_i = Δ_hi / hi

# 1) invariato — argmax|z_h| (oggi)
# 2) centroide pesato: h*_centroid = round_to_grid( Σ w_i·h_i·|rate_i| / Σ w_i·|rate_i| )
# 3) elbow — marginale che si annulla:
marginal_1 = Δ_h1 / h1
marginal_i = (Δ_hi − Δ_h(i-1)) / (hi − h(i-1))      # eccesso nelle sole barre aggiuntive
h*_elbow = h(i-1) al primo i con segno(AUC_Δ)·marginal_i <= 0
         = ultimo orizzonte della griglia se non si annulla mai (boundary_monotone)
```

| gruppo | n | `h*` mediano (argmax / centroide / elbow) | % al bordo (argmax / centroide / elbow) |
|---|---|---|---|
| entrambi | 111 | 3 / 5 / 3 | 18.9% / 0.0% / 12.6% |
| solo attuale | 69 | 2 / 6 / 2 | 10.1% / 0.0% / 0.0% |
| solo AUC | 45 | 5 / 10 / **6** | 6.7% / 0.0% / **28.9%** |

Concordanza tra i tre metodi sul gruppo "solo AUC": solo 20-33% — non è rumore, scelgono
sistematicamente orizzonti diversi. Esempi concreti (stesse regole di §3.6):

- **028**: il tasso decade (0.0150→0.0021) ma il cumulato `Δ_h` cresce ininterrottamente
  fino a h=24. `argmax` sceglie **h=1**, il punto più corto — economicamente il meno
  sensato, visto che tenere di più continua a pagare. Centroide ed elbow concordano su
  **h=10**.
- **088**: `Δ_h` cresce **monotonicamente su tutta la griglia testata**, fino a h=20
  (arricchito). `argmax` sceglie h=2, ignorando che il vantaggio continua a crescere per
  altri 18 giorni oltre quel punto; l'elbow lo segnala onestamente come
  `boundary_monotone` invece di inventare un punto di stop che non c'è.
- **136**: il centroide sceglie h=10 nonostante il profilo culmini e ridiscenda già a
  h=6 — perché **eredita lo stesso bias dei pesi trapezoidali** già documentato come
  rischio dello stadio (a) in §3.7 (i punti tardi/arricchiti pesano sproporzionatamente).
  `argmax` ed elbow concordano correttamente su h=6.

**Il centroide è scartato**: eredita il bias dei pesi trapezoidali verso gli orizzonti
tardi/arricchiti (caso 136) — non è un metodo pulito finché quei pesi non vengono
rivisti (rischio già aperto in §3.7).

**L'elbow (formula sopra) è il candidato scelto**, con un limite noto: su un profilo non
monotono con un singolo calo isolato poi recuperato (es. regola 256 — negativo ovunque,
un solo punto di parziale reversione a h=7, poi si riappiattisce), la regola grezza si
ferma al primo marginale che cambia segno, anche se non è un vero punto di stop.
**Corretto testando "due marginali consecutivi non positivi"** (mutuato da §4.3, idea
C): il risultato è peggiore, non migliore — su griglie di sole 6-9 orizzonti la soglia a
due passi lascia raramente spazio a due conferme e il metodo collassa quasi sempre al
bordo (82.2% per "solo AUC" contro il 28.9% della regola a un solo marginale; 55.0%
anche su "entrambi", dove il profilo è tipicamente a picco netto). La soglia di idea C
era calibrata per un'altra domanda (una salita sospetta vicino al bordo su griglie più
dense) e non si trapianta pari pari al criterio di arresto sul marginale.

**Anche un secondo raffinamento, a soglia di materialità invece che di conteggio, è
stato provato e scartato.** Idea: scalare il marginale osservato per la deviazione
standard del marginale nullo (stesse rotazioni già usate per `p_AUC`), fermandosi solo
quando il calo eccede un multiplo di quel rumore (`z_marginale <= -soglia`, con
`soglia=0` che riproduce esattamente la regola grezza come controllo). Risultato: già a
`soglia=0.5` il bordo esplode ovunque, incluso il gruppo "entrambi" (picco netto, dove
non dovrebbe succedere) — dal 12.6% al 38.7%, poi 74.8% a `soglia=1.0`. Il motivo è
statistico, non un bug: la differenza tra due cumulate consecutive (`Δ_hi − Δ_h(i-1)`) è
molto più rumorosa della cumulata stessa — differenziare amplifica la varianza — quindi
il null per-segmento ha poca potenza e quasi nessun marginale supera anche un solo sigma
di rumore. Sul caso 057 (quello che il raffinamento doveva correggere) non risolve
nemmeno il problema originale: a `soglia=0.5` si ferma ancora a h=5 (il calo è appena
sopra soglia).

**Resta quindi in vigore la regola grezza a un solo marginale**, col rischio dei falsi
stop residuo (caso 057) accettato come limite noto e documentato, non risolto da nessuno
dei due raffinamenti tentati.

**Validazione su un secondo asset — DAX (§3.5).** Stesso confronto a tre metodi,
sull'asset scelto apposta per essere il più trending dei quattro:

| gruppo | n | `h*` mediano (centroide / elbow) | % al bordo (centroide / elbow) |
|---|---|---|---|
| entrambi | 557 | 5 / 5 | 0.0% / 11.8% |
| solo attuale | 450 | 7 / 2 | 0.4% / 0.7% |
| solo AUC | 182 | 10 / 3 | 0.0% / **13.7%** |

Il tasso di bordo dell'elbow su "solo AUC" scende dal 28.9% di ADA al 13.7% di DAX —
coerente, non sorprendente: il mean-reversion (che tende a un picco interno genuino,
§3.5) resta la maggioranza anche su un asset fortemente trending, quindi l'elbow trova
un punto di arresto interno più spesso. Nessuna instabilità strutturale emersa
sull'asset più sfavorevole al mean-reversion testato finora.

**Unificazione con l'idea C.** L'esito `boundary_monotone` dell'elbow **è** la stessa
domanda che l'idea C pone per il percorso BH-FDR ("la griglia è abbastanza lunga da
vedere dove finisce l'edge?"), solo misurata sul cumulato `Δ_h` invece che su
`score_by_h`/z. Non sono due meccanismi paralleli: sono la stessa diagnostica di
sufficienza della griglia, con due implementazioni a seconda di quale via ha promosso il
candidato — si veda §4.5 per il dettaglio.

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

Nota storica: la prima stesura di questa proposta immaginava questi due valori come un
**declassamento** della fiducia nell'etichetta di idea B. §4.6 mostra, con dati alla
mano, che è l'opposto: il bordo correla **di più**, non di meno, con le regole
genuinamente trend-following. Il collegamento con B resta reale, ma cambia natura — si
veda §4.6 per il disegno finale a due campi.

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

Il risultato (`stato`, e quindi il valore in `diagnostics`) è l'input del disegno a due
campi con l'idea B — si veda §4.6.

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

### 4.5 Unificazione con l'idea B: due percorsi di promozione, un'unica diagnostica

§4.3 descrive la diagnostica per il percorso di promozione **BH-FDR** (`h_sig`
non vuoto): confronta `score_by_h` (cioè `|z_h|`) contro il bordo della griglia. Il
rafforzamento OR di §3.9 introduce un secondo percorso — promozione via test **AUC**
(`h_sig=()` ma `p_AUC` significativo) — dove `h*` non viene più da `argmax|z_h|` ma
dalla regola elbow di §3.9. Per questo percorso **non serve una diagnostica separata**:
lo stato `boundary_monotone` che l'elbow produce mentre sceglie `h*` **è già** la
risposta alla stessa domanda ("la griglia è abbastanza lunga?"), calcolata sul cumulato
`Δ_h` invece che su `score_by_h`.

```python
if via_promozione == "bh_fdr":
    stato = diagnostica_su_score_by_h(derived_target)        # §4.3, invariata
elif via_promozione == "auc":
    stato = "bordo_in_salita" if h_elbow_status == "boundary_monotone" else "interno"
    # nessun ricalcolo: h_elbow_status è già un sottoprodotto della scelta di h* (§3.9)
```

Non ci sono i quattro stati intermedi (`bordo_ambiguo`, `bordo_plateau`,
`bordo_grid_troppo_corta`) sul percorso AUC — la regola elbow a un solo marginale non li
distingue, e i due raffinamenti tentati per introdurli (conteggio a due marginali,
soglia di materialità sul rumore null — entrambi in §3.9) sono stati validati e
scartati: nessuno dei due migliora la regola grezza. Se un raffinamento futuro li
introdurrà, si potranno mappare 1:1 sugli stessi due valori di `AlphaContract
.diagnostics` già definiti in §4.3 (`"horizon_at_grid_boundary_climbing"` /
`"horizon_at_grid_boundary_ambiguous"`), aggiungendo solo un terzo campo
(`horizon_selection_method: "argmax_z" | "elbow_auc"`) per distinguere da quale via
proviene la diagnostica — nessuna nuova struttura dati, un solo campo di provenienza.

Sul fixture ADA (`"balanced"`, gruppo "solo AUC", n=45, §3.9): 37/45 (82.2%) risultano
`boundary_monotone` con la regola a due marginali consecutivi, 13/45 (28.9%) con quella a
un solo marginale (in vigore) — numeri alti in entrambi i casi rispetto al 6.7% del
bordo su `argmax|z_h`, ma è atteso: un profilo piatto e diffuso come quello del gruppo
"solo AUC" (§3.6) è per costruzione più incline a "non aver ancora visto la fine
dell'edge entro la griglia testata" di un profilo a picco netto.

### 4.6 Disegno finale: due campi indipendenti, non una gerarchia

Prima formulazione di questa proposta: il bordo come segnale di **sfiducia** — le due
voci di `diagnostics` di §4.3 "declassano" l'etichetta di idea B. Validandolo sui dati
(ADA e, soprattutto, **DAX** — l'asset con un trend forte e pressoché costante scelto
apposta per avere un campione momentum non trascurabile, §3.5) il risultato è
**l'opposto**.

La domanda giusta non è "tra le regole al bordo, quante sono momentum?" (dominata dalle
frequenze di base — il mean-reversion è comune, il momentum è raro, quindi in valore
assoluto il bordo è quasi sempre pieno di regole non-momentum). La domanda giusta è
**tra le regole di ciascuna etichetta, quante finiscono al bordo?**

| etichetta (idea B) | ADA — n | ADA — % al bordo | DAX — n | DAX — % al bordo |
|---|---|---|---|---|
| momentum-aligned | 5 | **60.0%** | 31 | **35.5%** |
| idiosyncratic | 44 | 43.2% | 170 | 22.9% |
| non_significativo | 69 | 10.1% | 450 | 15.1% |
| mean-reversion-aligned | 107 | 11.2% | 538 | 8.7% |

Su entrambi gli asset il momentum ha il tasso di bordo più alto — coerente con
l'intuizione originale: un vero trend-follower è, per natura, un profilo che non ha
ancora smesso di crescere quando la griglia finisce. Il mean-reversion ha il tasso più
basso: un effetto che si esaurisce/inverte tende a mostrare un picco interno genuino,
prima che la griglia termini. (Campione ADA piccolo, n=5 — il risultato DAX, n=31, è
quello statisticamente più solido, ma punta nella stessa direzione.)

**Questo esclude un declassamento.** Se il bordo correlasse col rumore, "declassare"
l'etichetta al bordo avrebbe senso. Ma il bordo correla **di più** con le regole
genuinamente trend-following — declassarle proprio lì cancellerebbe l'informazione più
utile: "momentum-aligned al bordo" è probabilmente la firma più pulita di un
trend-follower ancora in corsa, non un caso da mettere in dubbio.

**Disegno adottato: due campi ortogonali**, non una quarta categoria che sovrascrive le
prime tre:

```python
nature: Literal["momentum-aligned", "mean-reversion-aligned", "idiosyncratic",
                "non_significativo"]           # idea B, §3.3, invariata
horizon_at_boundary: bool                       # idea C, §4.3/§4.5, invariata
```

Una regola può quindi essere `nature="momentum-aligned", horizon_at_boundary=True` (il
caso da manuale) oppure `nature="mean-reversion-aligned", horizon_at_boundary=False` (il
caso più comune) — mai una perde l'altra. `horizon_at_boundary` copre entrambi i
percorsi di promozione con la stessa logica di §4.5 (stato `bordo_*` per BH-FDR,
`boundary_monotone` per AUC) — cambia solo il fatto che ora è un secondo campo, non un
override del primo. Le due voci fini di `diagnostics` (`"horizon_at_grid_boundary
_climbing"` / `"_ambiguous"`, §4.3) restano come dettaglio interno per chi vuole la
granularità originale a quattro stati sul percorso BH-FDR; `horizon_at_boundary` è la
lettura booleana semplificata pensata per l'uso combinato con `nature`.

### 4.7 Domande aperte residue

- Se calcolare la diagnostica solo su `direction != "undetermined"` (dove esiste un h\*
  con un senso economico) o anche sui candidati scartati, per capire *perché* — decisione
  a basso rischio, rimandabile alla specifica tecnica: il calcolo è comunque a costo
  quasi zero in entrambi i casi.
- **Risolta in §4.6**: `horizon_at_boundary` correla con `nature`, ma nel senso opposto a
  quello ipotizzato in origine (si concentra sul momentum, non lo esclude) — confermato
  su ADA e DAX. Resta da ripetere la stessa verifica su BTC/EURUSD e con un segnale
  indipendente da idea B (Hurst/`market_structure`) per un controllo incrociato più
  forte, ma non è più una domanda aperta sul verso dell'effetto.
- Se e come estendere la regola elbow del percorso AUC ai quattro stati fini del
  percorso BH-FDR — i due raffinamenti tentati in §3.9 sono stati scartati; serve
  un'idea diversa da quelle esplorate finora (conteggio, soglia di materialità sul
  rumore null). A bassa priorità ora che `horizon_at_boundary` (booleano, §4.6) copre
  l'uso pratico su entrambi i percorsi senza bisogno dei quattro stati fini sul percorso
  AUC.

---

## 5. Relazioni tra le idee e sequenza consigliata

```
A (M1, ranged tpm)         — indipendente, nessuna dipendenza dalle altre due

percorso BH-FDR (esistente):
B (M2, h* = argmax|z_h|)   — indipendente da C per questo percorso (dipendenza
                              ipotizzata, poi esclusa empiricamente in §3.4)
C (M2, grid sufficiency)   — diagnostica su score_by_h/z, §4.3, invariata

percorso AUC (nuovo, §3.9 — rafforzamento OR della selezione):
B (M2, h* = elbow su Δ_h)  ─┐
C (M2, grid sufficiency)   ─┴─ stessa diagnostica, stesso calcolo (§4.5): lo stato
                                boundary_monotone dell'elbow È l'esito di C, non un
                                secondo passo
```

A resta indipendente per costruzione. Per il percorso di promozione BH-FDR esistente, B
e C restano indipendenti come la validazione empirica ha mostrato (§3.4-§3.5): la causa
dei problemi trovati in B era nella formula di B stessa, non nel pinning al bordo che C
rileva. Per il nuovo percorso AUC (§3.9), invece, B e C **si fondono**: la scelta di
`h*` e la diagnostica di sufficienza della griglia sono lo stesso calcolo sullo stesso
profilo `Δ_h` (§4.5) — non due idee da sequenziare, ma un unico meccanismo con due letture.

## 6. Prossimi passi

**Metodologia congelata su A, B e C** — questo documento resta a livello di specifica
funzionale, ma l'esplorazione è chiusa: ogni formula è stata scritta, validata sui dati
reali, corretta dove i dati mostravano un problema, e le strade alternative scartate
sono documentate insieme alla ragione dello scarto invece di essere solo omesse.

- **Idea A — congelata.** Formula confermata corretta e stabile su due preset
  (`"balanced"`, `"sniper"`) e tre asset (ADA, BTC, EURUSD) (§2.8-§2.9); l'effetto netto
  sul volume di candidati varia per preset×asset insieme per una ragione capita, non per
  un difetto della formula. Dettaglio tecnico residuo (non metodologico): compatibilità
  `"bar"` mode, se esporre `z` (§2.7) — rimandato alla specifica tecnica.
- **Idea B — congelata.** Due stadi (gate AUC via rotation-null riusato, poi pendenza di
  `Δ_h/h` sulla griglia), entrambi corretti dallo stesso artefatto aritmetico (`Δ_h`
  grezzo) e confermati su ADA, BTC ed EURUSD (§3.3-§3.6). Rafforzamento OR della
  selezione e derivazione di `h*`/`direction` per i "solo AUC" validati su ADA (§3.9):
  regola elbow a un solo marginale scelta; centroide scartato (eredita il bias dei pesi
  trapezoidali); **due raffinamenti dell'elbow tentati e scartati** (conteggio a due
  marginali — troppo severo su griglie corte; soglia di materialità sul rumore
  null — statisticamente sotto-potenziata, differenziare due cumulate amplifica la
  varianza) — il rischio residuo (falsi stop su profili non monotoni, caso 057) resta
  documentato, non risolto. Dettaglio tecnico residuo: ripetere su `"sniper"` e fissare i
  valori numerici delle soglie `p_AUC`/`ρ` (oggi illustrative) (§3.7-§3.8) — rimandato
  alla specifica tecnica.
- **Idea C — congelata.** Meccanismo a tre/quattro stati derivato interamente da
  `DerivedTarget.score_by_h`, nessun nuovo dato (§4.3), validato su 180 contratti
  BH-FDR-promossi su ADA. Unificata con B su entrambi i percorsi di promozione (§4.5) e
  ridisegnata come **due campi indipendenti** — `nature` (idea B, invariata) e
  `horizon_at_boundary` (booleano) — invece di un declassamento, dopo che la
  validazione su ADA e DAX ha mostrato che il bordo correla *di più*, non di meno, con
  le regole genuinamente momentum (§4.6). Dettaglio tecnico residuo: ripetere la stessa
  verifica su BTC/EURUSD e con un segnale indipendente (Hurst/`market_structure`) (§4.7)
  — rimandato alla specifica tecnica, non blocca il congelamento della formula.
- Solo dopo: apertura di branch/issue separati per A, B, C.
