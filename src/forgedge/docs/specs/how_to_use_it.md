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
        min_act=50,             # ≥50 attivazioni IS
        min_months=8,           # attivo in ≥8 mesi diversi
        max_conc=0.40,          # ≤40% delle attivazioni in un solo mese
        min_tpm=2.0,            # ≥2.0 attivazioni/mese in media
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
print(f"Concentrazione max: {s.max_monthly_concentration:.2%}")

# Usare l'evento su nuovi dati (produzione, no look-ahead)
new_bars_active = c.apply(new_kpi_table)   # pd.Series bool
```

### Controllare la qualità del ConsistencyGate

```python
# Tutti i candidati con gate dettagliato
for c in candidates:
    gate = c.gate_result
    print(f"{c.event_id}: acts={gate.n_activations}, "
          f"months={gate.n_active_months}, passed={gate.passed}")
```

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
        min_activations=30,
        use_fdr=True,
        fdr_q=0.10,
        oos_max_p=0.10,
        min_oos_activations=10,
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

### Diagnostica sui rifiutati

```python
for c in contracts:
    if not c.promoted:
        print(f"{c.event_candidate_id}: {c.rejection_reasons}")

# Categorie di rifiuto frequenti:
# "no derivable target" → evento troppo raro o ritorno medio nullo su tutti gli orizzonti
# "lift X < 0.08" → win rate non supera il base rate di abbastanza
# "derived target not confirmed OOS" → segnale IS non si replica sull'OOS tail
# "not significant under BH FDR" → troppi candidati; il BH abbassa la soglia effettiva
```

---

## 5. Pipeline completa per produzione

Il pattern raccomandato è un'unica funzione che incapsula l'intera sessione:

```python
import pandas as pd
from forgedge import (
    MarketContext, MarketContextConfig, EMAProxyConfig,
    EventDiscovery, DiscoveryConfig,
    AlphaDiscovery, AlphaConfig,
)
from forgedge.event_discovery.models import WalkForwardConfig, GateParams
from forgedge.alpha_discovery.models import PromotionThresholds


def run_forge_pipeline(
    kpi: pd.DataFrame,
    asset: str,
    timeframe: str,
    bar_hours: float,
) -> tuple:
    """Esegue la pipeline FORGE completa e restituisce (promoted_contracts, ad).

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
    promoted : list[AlphaContract]
        Contratti promossi (status="HYPOTHESIS").
    ad : AlphaDiscovery
        Istanza con accesso a summary(), market_structure, split_idx.
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
        gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
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
            min_activations=30,
            use_fdr=True,
            fdr_q=0.10,
            oos_max_p=0.10,
            min_oos_activations=10,
        ),
    )
    ad = AlphaDiscovery(ed.df, candidates, alpha_config)
    ad.run()

    promoted = ad.promoted_contracts()
    return promoted, ad


# Utilizzo
kpi = pd.read_parquet("btc_1h.parquet")
promoted, ad = run_forge_pipeline(kpi, asset="BTC", timeframe="1H", bar_hours=1.0)

print(f"{len(promoted)} ipotesi promosse")
print(ad.summary()[["expression", "direction", "holding_period_h", "grade"]].head())
```

---

## 6. Workflow multi-asset

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

## 7. Replay OOS: applicare eventi scoperti a nuovi dati

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

---

## 8. Persistenza degli artefatti

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

I candidati non sono serializzabili direttamente via `to_contract_dict()`, ma
l'`event_formula` e l'`expression` salvate in ogni `EventCandidate` contengono
tutto ciò che serve per riapplicarli:

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

## 9. Checklist pre-produzione

Prima di portare un risultato in Rule Discovery (non ancora implementato),
verificare:

- [ ] `mc.window_resolution["source"] == "hurst_ou"` — le EMA sono adattive
- [ ] `mc.distribution()` mostra almeno 3 regimi con share > 5% — distribuzione non degenere
- [ ] Tutti i candidati passati ad AlphaDiscovery hanno `c.validation.passed == True` (walk-forward)
- [ ] `len(promoted) >= 3` — abbastanza ipotesi per diversificazione; se 0, abbassare i gate o aggiungere feature
- [ ] Nessun promosso con `oos.n_activations < 15` — stima OOS instabile
- [ ] Score grade ≥ B (`composite_score >= 0.45`) su almeno uno dei promossi
- [ ] Verificare `regime_analysis.dependency_type` — evitare `"broken"` se si vuole robustezza cross-regime
