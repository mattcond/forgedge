# FORGE — Guida alla Pipeline End-to-End per Produzione

Questa guida descrive come costruire e configurare una pipeline FORGE completa,
da una KPI Table grezza fino ai contratti alpha pronti per Rule Discovery.
Copre la preparazione dei dati, la configurazione di ogni modulo per un contesto
di produzione, l'interpretazione dei risultati e i pattern ricorrenti per
workflow multi-asset.

---

## 1. Requisiti dei dati di input

### Formato della KPI Table

FORGE riceve un `pd.DataFrame` con:

- **Colonna `close`** — prezzo di chiusura, richiesta da tutti e tre i moduli
- **Colonna datetime** — default `open_dt`; accettata come colonna o come nome
  del DatetimeIndex. Deve essere ordinabile cronologicamente.
- **Colonne feature** — qualsiasi indicatore tecnico (RSI, EMA, volume, spread,
  ratio, oscillatori). FORGE classifica automaticamente ogni colonna per tipo
  (numerica continua, numerica discreta, categoriale).

```python
# Schema minimo accettato
# Index: DatetimeIndex  oppure  colonna 'open_dt' con dtype datetime64
# Colonne richieste: 'close'
# Colonne opzionali: qualsiasi feature tecnica

import pandas as pd

kpi = pd.DataFrame({
    "open_dt": pd.date_range("2022-01-01", periods=5000, freq="1h"),
    "close":   [...],          # float, richiesto
    "rsi_14":  [...],          # esempio feature
    "volume":  [...],          # esempio feature
    "macd":    [...],          # esempio feature
})
```

### Convenzione naming per le EMA

Market Context cerca le EMA nella tabella usando il pattern
`{source_col}_ema_{period:02d}` (es. `close_ema_09`, `close_ema_25`). Se le
colonne sono presenti le usa direttamente; altrimenti le calcola inline.
Se la tua tabella già produce queste colonne (es. tramite un framework come
`CandleKPI`), il Modulo 0 è zero-copy per le EMA.

### Quantità di dati raccomandata

| Timeframe | Minimo consigliato | Ottimale |
|---|---|---|
| 1H | 6 mesi (≈4 000 barre) | 2 anni (≈17 000 barre) |
| 4H | 1 anno (≈2 200 barre) | 3 anni (≈6 500 barre) |
| 1D | 2 anni (≈730 barre) | 5 anni (≈1 800 barre) |

Più dati migliorano sia la stima delle half-life OU (Modulo 0) sia la stabilità
del ConsistencyGate (Modulo 1). Con meno di 2 000 barre i gate del Modulo 1
potrebbero essere troppo stringenti per asset poco liquidi.

---

## 2. Modulo 0 — Market Context

### Configurazione minima (produzione)

```python
from forgedge import MarketContext

enriched = MarketContext(kpi).run()
# aggiunge 'regime' (Categorical ordinata) e 'regime_stable' (bool)
```

La configurazione di default (`auto_window=True`, `threshold_mode="fixed"`,
`stable_window=12`) è calibrata per crypto 1H e va bene come punto di partenza.

### Configurazione per timeframe non standard

```python
from forgedge import MarketContext, MarketContextConfig, EMAProxyConfig

config = MarketContextConfig(
    ema_proxy=EMAProxyConfig(
        auto_window=True,
        window_unit="day",          # raccomandato: stessa quantità di storia su qualsiasi TF
        window_estimation=168.0,    # finestra OU di 168 giorni
        bar_hours=4.0,              # esplicito se non c'è DatetimeIndex (es. TF 4H)
        stable_window=12,           # barre consecutive per regime_stable=True
    )
)
mc = MarketContext(kpi, config=config)
enriched = mc.run()
```

### Verificare la qualità del regime

```python
print(mc.distribution())
# Distribuzione equilibrata su 5 regimi è un buon segnale.
# Se un regime occupa >50% delle barre, considera threshold_mode="balanced".

print(mc.window_resolution)
# source="hurst_ou" → EMA derivate dai dati  (ottimale)
# source="fallback" → OU non ha convergito; usate le EMA default (9/25)
# source="configured" → auto_window=False; usate le EMA configurate manualmente
```

### Quando usare `threshold_mode="balanced"`

Se la distribuzione dei regimi è molto sbilanciata (es. asset fortemente
trending come BTC in certi periodi), considera la modalità bilanciata:

```python
config = MarketContextConfig(
    ema_proxy=EMAProxyConfig(
        threshold_mode="balanced",
        threshold_basis="expanding",  # causale: no look-ahead
        target_distribution=[0.15, 0.20, 0.30, 0.20, 0.15],
    )
)
```

---

## 3. Modulo 1 — Event Discovery

### Configurazione minima (produzione)

```python
from forgedge import EventDiscovery

ed = EventDiscovery(enriched)
candidates = ed.run()
print(f"{len(candidates)} eventi candidati")
print(ed.summary().head(20))
```

### Configurazione con walk-forward OOS

Per produzione, è fortemente raccomandato abilitare la validazione walk-forward:

```python
from forgedge import EventDiscovery, DiscoveryConfig
from forgedge.event_discovery.models import WalkForwardConfig

config = DiscoveryConfig(
    train_ratio=0.80,           # 80% IS per la scoperta, 20% riservato all'OOS
    walk_forward=WalkForwardConfig(
        n_splits=4,             # dividi l'OOS in 4 finestre
        min_pass_rate=0.75,     # l'evento deve passare il gate in ≥75% delle finestre
    ),
    gate_params=GateParams(     # from forgedge.event_discovery.models
        min_tpm=2.0,            # ≥2.0 attivazioni/mese in media
        max_dispersion=2.5,     # Index of Dispersion ≤ 2.5 (Var/Mean dei conteggi mensili)
    ),
    max_and_components=2,       # max 2 componenti in AND (default conservativo)
)
ed = EventDiscovery(enriched, config=config)
candidates = ed.run()

# Filtrare solo gli eventi che hanno superato il walk-forward
wf_stable = [c for c in candidates if c.validation and c.validation.passed]
print(f"{len(wf_stable)} eventi stabili OOS su {len(candidates)} totali")
```

Con `train_ratio < 1.0` e `walk_forward` attivo, ogni candidato espone
`c.validation` (un `ValidationResult`) con:
- `c.validation.passed` — True se l'evento supera in ≥ `min_pass_rate` finestre
- `c.validation.pass_rate` — quota di finestre superate
- `c.validation.fold_results` — dettaglio per ogni finestra OOS

### Interpretare un EventCandidate

```python
c = candidates[0]
print(c.event_id)         # "EV-BTC-1H-260610-000"
print(c.expression)       # "rsi_14 < 31.2 AND spread_ema_9_25 < -0.0118"
print(c.event_formula)    # versione human-readable della formula
print(c.sql_expression)   # query DuckDB/SQL per filtrare barre attive

# Statistiche di attivazione
s = c.activation_stats
print(f"Attivazioni: {s.n_activations}, Mesi attivi: {s.n_active_months}")
print(f"Quota mensile max: {s.max_monthly_share:.2%}")

# Usare l'evento su nuovi dati (produzione, no look-ahead)
new_bars_active = c.apply(new_kpi_table)   # pd.Series bool
```

### Controllare la qualità del ConsistencyGate

```python
# Tutti i candidati con gate dettagliato
for c in candidates:
    gate = c.consistency_gate
    print(f"{c.event_id}: acts={gate.n_activations}, "
          f"months={gate.n_active_months}, passed={gate.passed}")
```

### Iniezione manuale di eventi: `CustomEvent`

Quando si vuole testare un'ipotesi definita dall'utente senza eseguire la scoperta automatica, si può bypassare il Modulo 1 tramite `CustomEvent` e l'argomento `manual_events` di `forge()`:

```python
from forgedge import forge, CustomEvent

events = [
    CustomEvent("rsi_14 < 25 and spread_ema < -0.02", name="rsi_extreme_spread"),
    CustomEvent("volume > volume_ema_20 * 2", name="volume_spike"),
]

# manual_events bypassa il Modulo 1; il ConsistencyGate è ancora applicato
# (un fallimento emette un warning ma non scarta l'evento)
result = forge(
    kpi,
    ticker="BTCUSDC",
    timeframe="1H",
    manual_events=events,
)

for contract, resp in result.rule_responses:
    print(contract.alpha_id, resp.verdict)
```

Note importanti:
- `manual_events` e `event_discovery_config` sono mutualmente esclusivi (passarli entrambi genera `ValueError`).
- La composizione AND non viene eseguita in modalità iniezione manuale.
- Il candidato risultante ha `event_id = "CUSTOM-{name}"`.
- Per un uso standalone (senza `forge()`), usare `CustomEvent.apply(df)` o `CustomEvent.to_event_candidate(df)`.

---

## 4. Modulo 2 — Alpha Discovery

### Configurazione minima (produzione)

```python
from forgedge import AlphaDiscovery, AlphaConfig

ad = AlphaDiscovery(
    ed.df,                          # DataFrame post-pipeline di Event Discovery
    candidates,                     # list[EventCandidate]
    AlphaConfig(
        asset="BTC",
        timeframe="1H",
        train_ratio=0.70,           # 70% IS / 30% OOS (default)
    ),
)
contracts = ad.run()
promoted = ad.promoted_contracts()
```

**Nota critica:** passa `ed.df` (non la KPI Table originale). `ed.df` contiene
già le feature derivate (ratio, spread) calcolate durante Event Discovery; se si
passa la tabella originale, le feature vengono ricalcolate deterministicamente
dai parametri salvati nei componenti — il risultato è identico ma leggermente
più lento.

### Configurazione avanzata con grid custom

```python
from forgedge import AlphaConfig
from forgedge.alpha_discovery.models import PromotionThresholds

config = AlphaConfig(
    asset="ADAUSDC",
    timeframe="4H",
    horizon_grid=(4, 8, 12, 24, 48, 72, 96),  # orizzonti in barre 4H
    train_ratio=0.75,
    thresholds=PromotionThresholds(
        ic_min_abs=0.02,
        ic_max_p=0.05,
        min_lift=0.08,
        min_cohens_d=0.15,
        use_fdr=True,
        fdr_q=0.10,
        oos_max_p=0.10,
    ),
    fee_per_side=0.001,             # registrato nel contratto per Rule Discovery
)
ad = AlphaDiscovery(ed.df, candidates, config)
```

### Interpretare i risultati

```python
# Riepilogo ordinato per score composito
df = ad.summary()
print(df[["expression", "direction", "holding_period_h", "sell_pct",
          "lift", "cohens_d", "oos_passed", "grade"]].head(10))

# Dettaglio su un contratto promosso
c = promoted[0]

# Target derivato
dt = c.derived_target
print(f"Orizzonte: {dt.holding_period_h}h, Direction: {dt.direction}")
print(f"sell_pct: {dt.sell_pct:.4f}, mean_advantage: {dt.mean_advantage:.4f}")

# Conferma OOS
oos = c.oos_validation
print(f"OOS: passed={oos.passed}, lift={oos.lift:.4f}, p={oos.p_value:.4f}")

# Sensibilità al regime
for rs in c.regime_analysis.per_regime:
    print(f"  {rs.regime}: IC={rs.ic:.3f}, win_rate={rs.win_rate:.3f}, {rs.strength}")
print(f"Dependency: {c.regime_analysis.dependency_type}")
```

### Diagnostica su contratti rifiutati e promossi

```python
# L'unico rifiuto bloccante è la direzione indeterminata
rejected = [c for c in contracts if not c.promoted]
for c in rejected:
    print(f"REJECTED {c.event_candidate_id}: {c.rejection_reasons}")
# Unica causa: "no derivable target" → nessun orizzonte produce un vantaggio finito

# I contratti promossi possono avere diagnostiche non bloccanti
for c in promoted:
    if c.rejection_reasons:
        print(f"PROMOTED {c.event_candidate_id} (grade={c.alpha_score.grade}):")
        for r in c.rejection_reasons:
            print(f"  {r}")
# Esempi di diagnostiche su contratti promossi (prefisso [diagnostic]):
# "[diagnostic] IC weak (|IC|=0.012 < 0.02, p=0.083)"
# "[diagnostic] lift 0.052 < 0.08"
# "[diagnostic] OOS weak (p=0.143 vs 0.10, mean_adv=0.0021, n_act=7)"
# "[diagnostic] not significant under BH FDR"
# Queste debolezze statistiche informano il grade (A–D) ma non bloccano la promozione.
```

---

## 5. Modulo 3 — Rule Discovery

### Configurazione minima (produzione)

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig

# by_id costruito dopo ed.run() e ad.run()
by_id = {c.event_id: c for c in candidates}

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        vr = resp.validated_rule
        print(f"  drop={vr.params.buy_drop_pct}  sell={vr.params.sell_pct}"
              f"  h={vr.params.target_h}  PF={resp.in_sample_summary.profit_factor:.2f}")
```

### Interpretare la risposta

```python
resp = rd.run()

# Verdetto principale
print(resp.verdict)          # "EDGE" | "PARTIAL-EDGE" | "NON-EDGE"
print(resp.is_edge)          # True per EDGE e PARTIAL-EDGE
print(resp.rejection_reasons) # lista dei gate falliti (vuota se EDGE)

# Metriche IS
s = resp.in_sample_summary
print(f"Trades: {s.total_trades}, PF: {s.profit_factor:.2f}, WR: {s.win_rate_pct:.2%}")
print(f"Expectancy: {s.expectancy:.4f}, tpm: {s.tpm_mu:.1f}")

# Execution envelope (conservativo vs ottimistico)
env = resp.execution_envelope
print(f"PF conservative: {env.conservative.profit_factor:.2f}")
print(f"PF optimistic:   {env.optimistic.profit_factor:.2f}")

# Walk-forward OOS
wf = resp.walk_forward
print(f"OOS PF: {wf.oos_summary.profit_factor:.2f}, consistency: {wf.consistency:.0%}")

# Report testuale compatto
from forgedge.rule_discovery import text_report
print(text_report(resp))
```

### Generare report HTML per review

```python
from forgedge.rule_discovery import html_report
import json

for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    rd   = RuleDiscovery(ed.df, contract, cand)
    resp = rd.run()

    if resp.is_edge:
        with open(f"reports/{resp.alpha_id}.html", "w") as f:
            f.write(html_report(resp))
        with open(f"reports/{resp.alpha_id}.json", "w") as f:
            json.dump(resp.to_dict(), f, indent=2)
    else:
        print(f"NON-EDGE: {resp.alpha_id} — {resp.rejection_reasons}")
```

### Diagnostica completa su NON-EDGE (`early_elimination=False`)

Per default, una regola che fallisce lo screen rapido IS (< 20 trade, PF < 1,
fill rate insufficiente) viene rifiutata immediatamente senza eseguire
walk-forward e diagnostiche — risparmio di calcolo. Con
`early_elimination=False` la pipeline gira per intero: il verdetto rimane
`NON-EDGE` ma walk-forward, regime e MAE/MFE sono popolati, utile per
confrontare report uniformi o analizzare il comportamento OOS di regole deboli:

```python
from forgedge import RuleDiscovery, RuleDiscoveryConfig, SelectionCriteria

config = RuleDiscoveryConfig(
    criteria=SelectionCriteria(early_elimination=False),
)
rd   = RuleDiscovery(ed.df, contract, cand, config=config)
resp = rd.run()
# resp.walk_forward è ora popolato anche se resp.verdict == "NON-EDGE"
print(f"OOS PF: {resp.walk_forward.oos_summary.profit_factor:.2f}")
```

---

## 6. Modulo 4 — Rule Registry

### Configurazione minima (produzione)

Rule Registry riceve le regole validate da tutti i ticker della sessione e
produce la tabella piatta + report HTML. Il percorso più rapido è
`from_forge_results`, che costruisce automaticamente le submission e i frame
dai `ForgeResult` dei singoli ticker:

```python
from forgedge import RuleRegistry, RegistryConfig

# Eseguire forge() separatamente per ogni ticker
results = {}
for ticker, kpi in kpi_tables.items():
    results[ticker] = forge(kpi, asset=ticker, timeframe="1H")

# Modulo 4: una riga per ogni EDGE / PARTIAL-EDGE
registry = RuleRegistry.from_forge_results(results).run()

print(registry.summary().to_string(index=False))
```

### Configurazione con parametri custom

```python
config = RegistryConfig(
    overlap_threshold=0.70,        # Jaccard ≥ 0.70 → duplicato
    cross_pf_threshold=2.0,        # PF minimo per PASS cross-ticker
    generic_ratio_threshold=2/3,   # ≥ 2/3 dei ticker PASS → GENERIC
    export_format="excel",         # "csv" o "excel"
    html_charts=True,              # SVG inline nel report
)
registry = RuleRegistry.from_forge_results(results, config=config).run()
```

### Leggere i risultati

```python
# Tabella piatta completa (una riga per documento)
df = registry.flat_table()
print(df[["rule_id", "source_ticker", "pf", "is_duplicate",
          "classification", "cross_ticker_score"]].to_string())

# Solo non-duplicate e generiche
df_clean = df[~df["is_duplicate"] & df["classification"].isin(["GENERIC", "PARTIAL"])]

# Accedere ai risultati cross-ticker di una regola
doc = registry.documents[0]
for ticker, ct in doc.cross_ticker.items():
    print(f"  {ticker}: PF={ct.pf:.2f}, {ct.verdict}")
print(f"Classificazione: {doc.classification}")  # GENERIC / PARTIAL / SPECIFIC / ISOLATED

# Matrici di correlazione (Jaccard e Spearman)
m = registry.matrices
print(m.jaccard.round(2))
print(m.spearman.round(2))
```

### Export tabella e report HTML

```python
# Tabella piatta CSV o Excel
flat_path = registry.export("forge_flat_table.xlsx")

# Report HTML autocontenuto (SVG inline, nessuna CDN)
html = registry.html_report(timeframe="1H")
with open("forge_report.html", "w", encoding="utf-8") as f:
    f.write(html)
```

### Costruzione manuale con RuleSubmission

Quando non si usa `forge()` come orchestratore, le submission possono essere
costruite esplicitamente:

```python
from forgedge import RuleDiscovery, RuleRegistry, RuleSubmission

submissions = []
for contract in promoted:
    cand = by_id[contract.event_candidate_id]
    resp = RuleDiscovery(ed.df, contract, cand).run()
    if resp.is_edge:
        submissions.append(RuleSubmission(
            ticker="BTC",
            response=resp,
            candidate=cand,
            grade=contract.alpha_score.grade,
        ))

frames = {"BTC": ed.df}
registry = RuleRegistry(submissions, frames).run()
```

---

## 7. Pipeline completa per produzione

### Orchestratore one-call: `forge`

I quattro moduli sono cablati insieme dall'orchestratore `forge`, che riduce
l'intera sessione a una singola chiamata. Accetta la KPI Table più la
configurazione di ogni modulo e restituisce un `ForgeResult` con tutti gli
artefatti:

```python
from forgedge import forge

result = forge(kpi, ticker="BTCUSDC", timeframe="1H")

print(result.summary())                         # una riga per candidato + rule_verdict
for contract, response in result.edges():       # solo EDGE / PARTIAL-EDGE
    print(contract.alpha_id, response.verdict)
print(result.registry.summary())                # Modulo 4 — regole catalogate
```

La configurazione per modulo si passa come argomento keyword dedicato:

```python
from forgedge import (
    forge, MarketContextConfig, EMAProxyConfig,
    DiscoveryConfig, AlphaConfig, RegistryConfig,
)
from forgedge.event_discovery.models import WalkForwardConfig, GateParams
from forgedge.alpha_discovery.models import PromotionThresholds

result = forge(
    kpi,
    asset="BTC",
    timeframe="1H",
    market_context_config=MarketContextConfig(
        ema_proxy=EMAProxyConfig(auto_window=True, window_unit="day", bar_hours=1.0),
    ),
    event_discovery_config=DiscoveryConfig(
        train_ratio=0.80,
        walk_forward=WalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
    ),
    alpha_config=AlphaConfig(
        train_ratio=0.70,
        thresholds=PromotionThresholds(min_lift=0.08, min_cohens_d=0.15),
    ),
    registry_config=RegistryConfig(),
)
```

Switch utili:

- `ticker="BTCUSDC"` — etichetta usata per il pool del Rule Registry e i metadati degli AlphaContract (fallback su `alpha_config.asset` o `asset`).
- `run_market_context=False` — alimenta una tabella che porta già `regime`
  (Modulo 0 viene saltato anche automaticamente quando la colonna `regime` è
  presente).
- `run_rule_discovery=False` — fermati dopo Alpha Discovery per ottenere le
  ipotesi promosse (`result.promoted`) senza backtestare; il Modulo 4 viene saltato automaticamente.
- `run_registry=False` — ferma dopo Rule Discovery, senza costruire il registry del Modulo 4.
- `only_validated_events=True` — passa ad Alpha Discovery solo i candidati
  walk-forward-validati (quando Event Discovery ha usato `walk_forward`).
- `rule_discovery_grades=("A", "B")` — limita il costoso backtest di Rule Discovery ai soli
  Alpha Contract il cui letter grade (`A` / `B` / `C` / `D`) è nell'insieme fornito.
  I contratti esclusi restano visibili in `result.contracts` / `result.promoted` per audit
  ma non ottengono una rule response e non raggiungono il Rule Registry.
  Il confronto è case-insensitive.  Se omesso, tutti i contratti promossi vengono backtestati.
- `progress=True` — stampa lo stato di avanzamento per ogni stadio e una barra di progresso
  di Rule Discovery su `stderr` (utile per run lunghe). Indipendentemente dal flag, ogni
  milestone viene emessa a livello `INFO` sul logger `forgedge.forge`, quindi
  `logging.basicConfig(level=logging.INFO)` espone le stesse informazioni.

`ForgeResult` espone le istanze vive dei moduli (`result.market_context`,
`result.event_discovery`, `result.alpha_discovery`) per drill-down, più
`result.candidates`, `result.contracts`, `result.promoted`,
`result.rule_responses`, `result.event_frame` (il frame post-pipeline) e
`result.registry` (il `RuleRegistry` del Modulo 4).

### Costruirla passo per passo

Per un controllo più fine, lo stesso flusso può essere scritto esplicitamente —
è esattamente ciò che `forge` esegue internamente:

```python
import pandas as pd
from forgedge import (
    MarketContext, MarketContextConfig, EMAProxyConfig,
    EventDiscovery, DiscoveryConfig,
    AlphaDiscovery, AlphaConfig,
    RuleDiscovery,
)
from forgedge.event_discovery.models import WalkForwardConfig, GateParams
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.rule_discovery import html_report


def run_forge_pipeline(
    kpi: pd.DataFrame,
    asset: str,
    timeframe: str,
    bar_hours: float,
) -> tuple:
    """Esegue la pipeline FORGE completa e restituisce (rule_responses, ad, ed).

    Parameters
    ----------
    kpi : pd.DataFrame
        KPI Table con colonna 'close' e colonna 'open_dt' (o DatetimeIndex).
    asset, timeframe : str
        Metadati di tracciabilità per gli AlphaContract.
    bar_hours : float
        Durata della barra in ore (es. 1.0 per 1H, 4.0 per 4H).

    Returns
    -------
    rule_responses : list[tuple[AlphaContract, RuleDiscoveryResponse]]
        Coppie (contratto, risposta) per ogni contratto promosso.
    ad : AlphaDiscovery
        Istanza con accesso a summary(), market_structure, split_idx.
    ed : EventDiscovery
        Istanza con accesso a df (DataFrame arricchito) e candidati.
    """

    # ── Modulo 0: regime ────────────────────────────────────────────────
    mc_config = MarketContextConfig(
        ema_proxy=EMAProxyConfig(
            auto_window=True,
            window_unit="day",
            bar_hours=bar_hours,
            stable_window=12,
        )
    )
    enriched = MarketContext(kpi, config=mc_config).run()

    # ── Modulo 1: eventi ────────────────────────────────────────────────
    ed_config = DiscoveryConfig(
        train_ratio=0.80,
        walk_forward=WalkForwardConfig(n_splits=4, min_pass_rate=0.75),
        gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
        max_and_components=2,
    )
    ed = EventDiscovery(enriched, config=ed_config)
    candidates = ed.run()

    # ── Modulo 2: alpha ─────────────────────────────────────────────────
    alpha_config = AlphaConfig(
        asset=asset,
        timeframe=timeframe,
        horizon_grid=(1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48),
        train_ratio=0.70,
        thresholds=PromotionThresholds(
            min_lift=0.08,
            min_cohens_d=0.15,
            use_fdr=True,
            fdr_q=0.10,
            oos_max_p=0.10,
        ),
    )
    ad = AlphaDiscovery(ed.df, candidates, alpha_config)
    ad.run()
    promoted = ad.promoted_contracts()

    # ── Modulo 3: rule discovery ─────────────────────────────────────────
    by_id = {c.event_id: c for c in candidates}
    rule_responses = []
    for contract in promoted:
        cand = by_id[contract.event_candidate_id]
        rd   = RuleDiscovery(ed.df, contract, cand)
        resp = rd.run()
        rule_responses.append((contract, resp))

    return rule_responses, ad, ed


# Utilizzo
kpi = pd.read_parquet("btc_1h.parquet")
rule_responses, ad, ed = run_forge_pipeline(kpi, asset="BTC", timeframe="1H", bar_hours=1.0)

for contract, resp in rule_responses:
    print(f"{contract.alpha_id}: {resp.verdict}")
    if resp.is_edge:
        print(f"  PF={resp.in_sample_summary.profit_factor:.2f}"
              f"  OOS={resp.walk_forward.oos_summary.profit_factor:.2f}")
```

### Sessioni multi-ticker: `forge_multi`

Il backtest cross-ticker del Rule Registry ha altri ticker su cui riprodurre i segnali solo quando la sessione copre più ticker. `forge_multi` esegue `forge` per ogni ticker e raccoglie tutte le regole tradabili in un unico registry cross-ticker:

```python
from forgedge import forge_multi

frames = {
    "BTCUSDC": btc_kpi,
    "ETHUSDC": eth_kpi,
    "ADAUSDC": ada_kpi,
}
results, registry = forge_multi(frames, timeframe="1H")

print(registry.summary())            # rule_id, source_ticker, cross-ticker score, class
registry.export("rules.xlsx")        # tabella piatta (artefatto di persistenza del Modulo 4)
html = registry.html_report()        # report HTML autocontenuto

# Drill-down per ticker ancora disponibile
for ticker, res in results.items():
    print(ticker, len(res.edges()), "regole tradabili")
```

Passare gli oggetti di configurazione per modulo (`event_discovery_config`, `alpha_config`, …) come keyword argument — vengono inoltrati a ogni run per-ticker. Non passare `ticker` / `asset` (impostati automaticamente per ticker) né `run_registry` (il registry pooled sostituisce quelli per-ticker).

---

## 8. Workflow multi-asset

Per applicare FORGE a più asset in parallelo, esegui una sessione indipendente
per asset. Le sessioni non condividono stato: ogni asset ha le proprie soglie
distribuzionali, le proprie half-life OU, e i propri contratti.

```python
assets = {
    "BTC":  ("btc_1h.parquet",  1.0),
    "ADA":  ("ada_1h.parquet",  1.0),
    "ETH":  ("eth_4h.parquet",  4.0),
}

all_promoted = {}
for asset, (path, bar_hours) in assets.items():
    kpi = pd.read_parquet(path)
    promoted, ad = run_forge_pipeline(kpi, asset=asset, timeframe="1H", bar_hours=bar_hours)
    all_promoted[asset] = promoted
    print(f"{asset}: {len(promoted)} ipotesi promosse")

# Riepilogo cross-asset
rows = []
for asset, contracts in all_promoted.items():
    for c in contracts:
        rows.append({
            "asset": asset,
            "alpha_id": c.alpha_id,
            "expression": c.event_expression,
            "grade": c.alpha_score.grade,
            "direction": c.direction,
            "holding_period_h": c.derived_target.holding_period_h,
        })
import pandas as pd
cross = pd.DataFrame(rows).sort_values("grade")
print(cross)
```

---

## 9. Replay OOS: applicare eventi scoperti a nuovi dati

`EventCandidate.apply(df)` riproduce deterministicamente l'evento su qualsiasi
DataFrame con le stesse colonne native, usando le soglie fissate in fase di
scoperta. Questo è il meccanismo per il forward testing e per l'uso in produzione:

```python
# Dati nuovi (arrivati dopo la sessione di scoperta)
new_data = pd.read_parquet("btc_1h_new.parquet")

for c in promoted:
    active_mask = c.apply(new_data)   # pd.Series bool
    n_fires = active_mask.sum()
    print(f"{c.event_id}: {n_fires} attivazioni sui nuovi dati")
    if active_mask.any():
        print(new_data[active_mask][["open_dt", "close"]].tail(3))
```

`apply()` usa l'espressione e le soglie salvate nel `EventCandidate` — non
richiede accesso alla sessione originale e non ri-ottimizza nulla.

### Monitorare un edge scoperto su nuovi dati

Quando nuovi dati di mercato si rendono disponibili dopo una sessione di scoperta,
il modulo corretto per verificare che l'edge regga è **Rule Discovery**, non Alpha
Discovery.

Alpha Discovery ri-deriva direzione e orizzonte ottimale da qualunque dato gli si
passi. Eseguirlo su un dataset che si estende prima del periodo di training mescola
attivazioni da regimi di mercato incompatibili — ad esempio, un evento vol-spike
scoperto come LONG nel 2024–2026 potrebbe aver scattato frequentemente durante il
crash del 2022, dove la stessa condizione era seguita da rendimenti fortemente
*negativi*. La media dei vantaggi tra quelle due popolazioni opposte tende a zero,
e Alpha Discovery restituisce `direction="undetermined"` ("no derivable target") —
non perché l'edge sia scomparso, ma perché la domanda posta è sbagliata.

Rule Discovery non ri-deriva nulla. Utilizza l'`AlphaContract` con `direction`,
`holding_period_h` e `sell_pct` fissi e misura se la regola di trading produce ancora
valore atteso positivo sulle nuove barre:

```python
import pandas as pd
from forgedge import RuleDiscovery

# Aggiunge solo le barre genuinamente nuove — non estendere verso la storia pre-training
new_bars = full_df[full_df["open_dt"] > train_df["open_dt"].max()]
eval_df  = pd.concat([train_df, new_bars]).drop_duplicates("open_dt")

for contract, cand in discovered_rules:
    resp = RuleDiscovery(eval_df, contract, cand).run()
    print(f"{contract.alpha_id}: {resp.verdict}")
    print(f"  PF={resp.in_sample_summary.profit_factor:.2f}"
          f"  WR={resp.in_sample_summary.win_rate_pct:.0%}"
          f"  OOS-consistency={resp.walk_forward.consistency:.0%}")
```

**Cosa aspettarsi sui nuovi dati**

Un calo moderato del profit factor (5–15%) è normale con l'espansione del periodo IS.
La metrica `walk_forward.consistency` è più informativa per il monitoraggio: un calo
di 25 punti percentuali o più segnala un edge che si sta indebolendo. Un cambio di
verdetto da PARTIAL-EDGE a NON-EDGE è un segnale per verificare se il regime di
mercato è mutato.

**Perché AlphaDiscovery su dati storici completi produce "no derivable target"**

Eseguire `AlphaDiscovery(full_df, events, AlphaConfig(train_ratio=1))` su dati che
precedono il periodo di training pone la domanda sbagliata: "questo evento ha un edge
misurabile su *tutta* la storia disponibile?". La risposta dipende dal regime. Le
attivazioni nella finestra di scoperta (es. 2024–2026) e quelle in un ciclo di crash
precedente (es. 2022) appartengono a due popolazioni con caratteristiche di forward
return opposte. La media dei vantaggi combinata può cancellarsi a ogni orizzonte,
portando `_derive_target` a restituire `direction="undetermined"` anche quando il
numero di attivazioni è ben al di sopra della soglia minima.

`AlphaDiscovery._event_series()` emette un `UserWarning` nel caso specifico in cui
il conteggio di attivazioni sul frame osservato scende quasi a zero (meno di 2
attivazioni e meno del 10% del conteggio di training). Il più comune errore
"no derivable target", tuttavia, emerge a un livello superiore — dalla cancellazione
di segnali opposti tra regimi di mercato incompatibili — e si risolve usando Rule
Discovery invece di rieseguire Alpha Discovery.

---

## 10. Persistenza degli artefatti

### Salvare i contratti promossi

```python
import json, yaml

# JSON (tutti i contratti)
with open("alpha_contracts.json", "w") as f:
    json.dump([c.to_contract_dict() for c in promoted], f, indent=2)

# YAML (contratto singolo — leggibile)
for c in promoted:
    with open(f"contracts/{c.alpha_id}.yaml", "w") as f:
        yaml.dump(c.to_contract_dict(), f, allow_unicode=True)
```

### Salvare il summary come CSV

```python
ad.summary().to_csv("alpha_summary.csv", index=False)
```

### Salvare i candidati (lista completa pre-promozione)

Il metodo raccomandato per archiviare candidati completi è `persist()`, che
esegue un round-trip pickle completo (componenti, soglie, statistiche,
validazione walk-forward incluse):

```python
import pathlib

pathlib.Path("candidates").mkdir(exist_ok=True)
for c in candidates:
    c.persist(f"candidates/{c.event_id}.pkl")

# Ricaricare in una sessione successiva
import pickle
cand = pickle.load(open("candidates/EV-BTC-1H-260610-000.pkl", "rb"))
cand.apply(new_kpi)   # pronto all'uso immediato
```

Per archivi tabulari leggibili (meno completi, non invertibili):

```python
candidates_data = [
    {
        "event_id": c.event_id,
        "expression": c.expression,
        "event_formula": c.event_formula,
        "sql_expression": c.sql_expression,
    }
    for c in candidates
]
pd.DataFrame(candidates_data).to_csv("event_candidates.csv", index=False)
```

---

## 11. Checklist pre-produzione

Prima di portare una `ValidatedRule` nel Rule Registry, verificare:

**Modulo 0**
- [ ] `mc.window_resolution["source"] == "hurst_ou"` — le EMA sono adattive
- [ ] `mc.distribution()` mostra almeno 3 regimi con share > 5% — distribuzione non degenere

**Modulo 1**
- [ ] Tutti i candidati passati ad AlphaDiscovery hanno `c.validation.passed == True` (walk-forward)

**Modulo 2**
- [ ] `len(promoted) >= 3` — abbastanza ipotesi per diversificazione; se 0, le feature non producono vantaggi finiti — aggiungere feature o allargare la grid
- [ ] Nessun promosso con `oos.n_activations < 15` — stima OOS instabile
- [ ] Score grade ≥ B (`composite_score >= 0.50`) su almeno uno dei promossi
- [ ] Verificare `regime_analysis.dependency_type` — evitare `"broken"` se si vuole robustezza cross-regime

**Modulo 3**
- [ ] Almeno un contratto con `resp.verdict == "EDGE"` — se solo PARTIAL-EDGE, investigare i gate falliti
- [ ] `resp.walk_forward.consistency >= 0.50` — metà delle finestre OOS con net gain positivo
- [ ] `resp.statistical_validation.temporal_stability != "FAIL"` — l'edge non è concentrato temporalmente
- [ ] `resp.regime_analysis.avoid_in` è una lista corta — l'edge funziona su più regimi
- [ ] Controllare l'intervallo execution envelope: `env.conservative.profit_factor > 1.5` — l'edge regge anche nel caso conservativo

**Modulo 4**
- [ ] Almeno una regola con `classification in ("GENERIC", "PARTIAL")` — l'edge non è specifico del solo ticker di discovery
- [ ] Controllare `df[df["is_duplicate"]]["duplicate_of"]` — le regole flaggate come duplicate non devono essere promosse in produzione
- [ ] `doc.cross_ticker_score > 0` per le regole GENERIC — la generalizzazione è confermata su almeno un ticker esterno
- [ ] Rivedere la matrice Jaccard (`registry.matrices.jaccard`) — coppie con overlap ≥ 0.85 indicano ridondanza strutturale tra segnali
