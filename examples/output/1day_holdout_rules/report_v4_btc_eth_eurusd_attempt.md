# Report v4: tentativo mirato su BTCEUR/ETHEUR/EURUSD — indicatori arricchiti + preset alternativi

**Data esecuzione:** 2026-09-08
**Segue:** `report_v3_rolling.md` — qui si tenta di ottenere regole robuste su tutte e 3 le
finestre rolling (18 mesi) per i 3 ticker che in v3 non ne avevano nessuna, con due leve:
indicatori arricchiti (MACD + ATR, disattivati di default) e preset alternativi.

---

## 1. Cosa è stato provato

### 1.1 — Indicatori arricchiti

`build_features()` disattiva **MACD** e **ATR/NATR** di default (`DEFAULT_CONFIG`,
`src/forgedge/kpi_builder/config.py`). Sono state ricostruite le KPI Table di BTCEUR, ETHEUR,
EURUSD con entrambi abilitati (187 colonne contro le 180 originali, +3 colonne MACD/signal/hist,
+4 colonne ATR/NATR a 14 e 28 periodi) — famiglie di indicatori pensate per inversioni di trend
e volatilità scale-free, motivazione diretta per il problema osservato in v3 (nessuna regola
sopravvive al crash di W2).

### 1.2 — Preset alternativi

Verificato con `config_report()` che **`"sniper"` non è utilizzabile su BTCEUR/ETHEUR** con 18
mesi di censura (`oos_span_too_short`: servono 31 mesi di storia, ne restano 22-24) — un limite
strutturale dei dati disponibili, non un problema di configurazione. Al suo posto è stato usato
**`"burst"`** (eventi concentrati nel tempo/cambio di regime — la descrizione del preset stesso
lo rende il candidato più naturale per catturare un'inversione di trend come il crash di W2).

Su **EURUSD**, `"sniper"` risultava fattibile (solo `WARN`, non `FAIL`) ed è stato tentato per
primo.

## 2. Incidente operativo: OOM risk con `sniper` + indicatori arricchiti su EURUSD

Il tentativo `EURUSD/sniper` con la KPI Table arricchita (187 colonne, 1301 barre di training)
ha fatto crescere la memoria del processo **da ~200 MB a oltre 11 GB in circa 3 minuti**, senza
segni di stabilizzazione, su un container da 15 GB totali. È stato interrotto preventivamente
(`SIGTERM`) prima di un OOM-kill incontrollato. Questo è coerente con un rischio di memoria già
documentato per `ANDComposer.compose()` sotto gate params permissivi su una KPI Table ricca (più
colonne → più coppie candidate per la composizione AND) — qui aggravato dalla combinazione
specifica preset-KPI, non osservato con `"burst"`/`"balanced"` sulle stesse tabelle arricchite.
**EURUSD è stato quindi rieseguito con `"balanced"`** (indicatori arricchiti, preset invariato
rispetto a v3) invece di `"sniper"`.

## 3. Risultati — nessuna regola pienamente robusta, in nessuno dei tre casi

| Ticker | Preset (v3 → v4) | Regole tradeable (v3 → v4) | Sopravvivono 3/3 (v3 → v4) | Sopravvivono ≥2/3 (v3 → v4) |
|---|---|---:|---:|---:|
| BTCEUR | balanced → **burst** | 61 → 17 | 0 → **0** | 12 (19.7%) → **0 (0.0%)** |
| ETHEUR | balanced → **burst** | 19 → 17 | 0 → **0** | 4 (21.1%) → **3 (17.6%)** |
| EURUSD | balanced → balanced (arricchito) | 8 → 7 | 0 → **0** | 2 (25.0%) → **1 (14.3%)** |

**Nessuna delle due leve produce una regola con `n_windows_survived == 3` su nessuno dei tre
ticker.** Per BTCEUR il cambio di preset è stato controproducente anche sul criterio più
permissivo (≥2/3): `"burst"` produce un pool più piccolo (17 contro 61) e nessuna delle 17
regola nemmeno 2 finestre su 3.

## 4. Perché — la causa resta di regime, non di indicatori

Il problema diagnosticato in `report_v3_rolling.md` §2 non è mai stato "servono indicatori
migliori": è che **W2 (set 2025 - mar 2026) è stato un crollo del -41.5% su BTCEUR e -55.7% su
ETHEUR**, e nessuna regola a direzione fissa — `long` o `short` — può sopravvivere a
un'inversione di quella entità mantenendo la stessa direzione anche in W1 e W3 (entrambe
rialziste per crypto). MACD e ATR aggiungono segnali di inversione più tempestivi, ma il limite
non è la sensibilità del segnale: è che un'unica regola a direzione fissa non può essere
contemporaneamente "giusta" in un rally (W1, W3) e in un crollo (W2) dello stesso asset.

Un'osservazione più sottile emerge comunque dal confronto **long vs short** con gli indicatori
arricchiti:

| Ticker | Miglior regola *long* in W2 (il crollo) | Miglior regola *short* in W2 |
|---|---:|---:|
| ETHEUR | PF ≈ 0.2–0.4 (perdita netta) | **PF 0.65–0.67** (quasi pareggio) |
| BTCEUR | PF ≈ 0.2–0.5 (perdita netta) | dati insufficienti (solo 1 regola short trovata) |

Le migliori regole *short* di ETHEUR (`delta_spread_close_sma03_1 > ... AND ...`) sopravvivono
**2 finestre su 3** — falliscono solo W3 (rialzista, PF 1.20-1.48 quindi non un tracollo) — ma
restano vicine al pareggio anche in W2, molto meglio delle regole long nello stesso periodo.
Non sopravvivono comunque a tutte e 3 le finestre: nessuna regola short di questo pool riesce a
essere allo stesso tempo redditizia in W2 (crollo) *e* in W1/W3 (entrambe rialziste per
ETHEUR).

## 5. Conclusione

Con i dati e la finestra temporale disponibili, **non esiste una regola a direzione fissa,
scoperta da questa pipeline, che sia robusta su tutte e 3 le finestre rolling per BTCEUR,
ETHEUR o EURUSD** — né con gli indicatori di base (`report_v3_rolling.md`), né con MACD/ATR
aggiunti, né cambiando preset di ricerca (`"burst"` per i due crypto, `"balanced"` per EURUSD
dopo l'incidente di memoria con `"sniper"`). Il limite è strutturale al problema (un'unica
regola direzionale contro un'inversione di regime del -41/-56%), non alla configurazione della
pipeline. Le uniche vie plausibili per progredire, entrambe fuori dallo scopo di un singolo
`forge()` — che produce sempre e solo regole a soglia/direzione fissa — sarebbero: (a) un
meccanismo di *regime switching* esplicito che alterni tra un pool di regole long e uno di
regole short in base a un indicatore di regime separato, o (b) accettare esplicitamente che
questi tre ticker restino fuori dal portafoglio di regole "robuste multi-regime" e vengano
validati solo con un hold-out singolo (come in `report.md`/`report_v2.md`), con la consapevolezza
esplicita — non un'omissione — che quel risultato non regge a un cambio di regime della
dimensione osservata in W2.

## 6. File esportati

In `examples/output/1day_holdout_rules/`:

- **`full_rule_log_v4_btc_eth_eurusd.pkl`** — le 41 regole valutate (17+17+7) con l'esito su
  ciascuna delle 3 finestre, incluso il confronto long/short di §4. Nessuna riga rappresenta una
  regola "selezionabile" nel senso di `report_v3_rolling.md` §4 (nessuna raggiunge 3/3) — il
  file è un log diagnostico, non un set di regole raccomandate.
