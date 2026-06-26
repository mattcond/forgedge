"""Preset configurations for forge() — opinionated starting points.

Traduce un'intenzione di alto livello ("sniper", "balanced", "sweep", "burst")
in una quadrupla (MarketContextConfig, DiscoveryConfig, AlphaConfig,
RuleDiscoveryConfig) pronta da passare a forge(), con M0/M1/M2/M3 allineati
sulla stessa filosofia di ricerca.

Coerenza M0–M1–M2–M3
---------------------
* ``min_tpm`` (M1 e M3) usa lo stesso floor di frequenza per evitare che eventi
  ammessi da M1 vengano rigettati da M3.
* ``MarketContextConfig`` (M0) usa ``threshold_mode="balanced"`` su tutti i
  preset, ma con ``target_distribution`` e ``stable_window`` calibrati sulla
  filosofia del preset: eventi rari vogliono un regime quasi sempre NEUTRAL;
  eventi burst vogliono più barre nei regimi estremi.

Scaling dei parametri per timeframe
------------------------------------
* ``min_tpm`` scala linearmente con le barre/mese.
* ``max_dispersion`` scala verso il basso sui timeframe corti (CLT stabilizza
  i conteggi mensili all'aumentare delle barre).
* ``horizon_grid`` e ``bars_per_day`` sono scalati al timeframe.

Utilizzo
---------
    from forgedge.presets import forge_preset
    from forgedge import forge, RotationConfig

    mc_cfg, disc_cfg, alpha_cfg, rd_cfg = forge_preset(
        "balanced", timeframe="1D", asset="ADA"
    )
    result = forge(
        df,
        market_context_config=mc_cfg,
        event_discovery_config=disc_cfg,
        alpha_config=alpha_cfg,
        rule_discovery_config=rd_cfg,
    )

    # Con RotationCalibrator (raccomandato per "sweep")
    mc_cfg, disc_cfg, alpha_cfg, rd_cfg = forge_preset("sweep", "1H", asset="ETH")
    result = forge(
        df,
        market_context_config=mc_cfg,
        event_discovery_config=disc_cfg,
        alpha_config=alpha_cfg,
        rule_discovery_config=rd_cfg,
        rotation_calibration=RotationConfig(k=100),
    )
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .alpha_discovery.models import AlphaConfig, PromotionThresholds
from .event_discovery.discovery import DiscoveryConfig
from .event_discovery.models import GateParams
from .market_context.models import EMAProxyConfig, MarketContextConfig
from .rule_discovery.models import RuleDiscoveryConfig, SelectionCriteria

__all__ = ["forge_preset", "preset_info", "PRESETS"]


# ── Timeframe parsing ─────────────────────────────────────────────────────────

_TF_RE = re.compile(r"^(\d+)\s*([a-zA-Z]+)$")

_UNIT_MINUTES = {
    "m": 1, "min": 1,
    "h": 60, "H": 60,
    "d": 1440, "D": 1440,
    "w": 10080, "W": 10080,
}


def _tf_minutes(timeframe: str) -> int:
    """Parse a timeframe string and return its duration in minutes."""
    m = _TF_RE.match(timeframe.strip())
    if m is None:
        raise ValueError(
            f"Unrecognised timeframe '{timeframe}'. "
            "Expected format: '1D', '4H', '15m', '1W', etc."
        )
    n, unit = int(m.group(1)), m.group(2)
    if unit not in _UNIT_MINUTES:
        raise ValueError(
            f"Unknown timeframe unit '{unit}' in '{timeframe}'. "
            f"Supported: {list(_UNIT_MINUTES)}"
        )
    return n * _UNIT_MINUTES[unit]


class _TFClass:
    """Classify a timeframe into daily / intraday / hft and expose scaling."""

    DAILY = "daily"
    INTRADAY = "intraday"
    HFT = "hft"

    def __init__(self, timeframe: str) -> None:
        self.timeframe = timeframe
        mins = _tf_minutes(timeframe)
        if mins >= 1440:
            self.cls = self.DAILY
            self.bars_per_month = 1440 / mins * 30
            self.bars_per_day = max(1, round(1440 / mins))
            self.horizon_grid = (1, 2, 3, 5, 7, 10)
        elif mins >= 60:
            self.cls = self.INTRADAY
            self.bars_per_month = (1440 / mins) * 30
            self.bars_per_day = round(1440 / mins)
            self.horizon_grid = (1, 2, 4, 8, 12, 24)
        else:
            self.cls = self.HFT
            self.bars_per_month = (1440 / mins) * 30
            self.bars_per_day = round(1440 / mins)
            self.horizon_grid = (1, 2, 5, 10, 20, 50)

    def scale_tpm(self, daily_tpm: float) -> float:
        """Scale a daily-calibrated min_tpm to this timeframe."""
        daily_bars_per_month = 30.0
        return daily_tpm * (self.bars_per_month / daily_bars_per_month)

    def scale_dispersion(self, daily_dispersion: float) -> float:
        """Scale max_dispersion down for shorter timeframes."""
        if self.cls == self.DAILY:
            return daily_dispersion
        if self.cls == self.INTRADAY:
            return round(daily_dispersion * 0.45, 2)
        return round(daily_dispersion * 0.20, 2)


# ── Preset definitions ────────────────────────────────────────────────────────
#
# Campi per modulo:
#   M0  mc_target_distribution  — target regime frequencies (5 valori, somma=1)
#   M0  mc_stable_window        — barre consecutive per regime_stable=True
#   M1  daily_min_tpm           — gate EventDiscovery (scala col TF)
#   M1  daily_max_dispersion    — gate EventDiscovery (scala col TF)
#   M1  max_and_components      — numero massimo di componenti AND
#   M2  min_lift / min_cohens_d / fdr_q / oos_max_p — gate AlphaDiscovery
#   M3  daily_rd_min_tpm        — gate RuleDiscovery (scala col TF)

_PRESET_SPECS: dict = {
    "sniper": {
        # ── descrizione bilingue ─────────────────────────────────────────────
        "description_it": (
            "Rari e regolari. Cerca segnali poco frequenti ma ad alta precisione "
            "statistica, con regole semplici (max 1 condizione AND). "
            "Richiede un IS lungo (≥2 anni su 1D) per avere abbastanza attivazioni. "
            "Non abbinare al RotationCalibrator: con così pochi eventi il null di "
            "rotazione non ha potere statistico sufficiente. "
            "Regime: quasi tutte le barre etichettate NEUTRAL (5/15/60/15/5 %), "
            "così solo i veri estremi di trend ricevono un'etichetta di regime — "
            "utile per isolare segnali rari in contesti di mercato netti."
        ),
        "description_en": (
            "Rare and precise. Targets low-frequency signals with a high statistical "
            "bar and simple rules (max 1 AND component). "
            "Needs a long IS window (≥2 years on 1D) for enough activations. "
            "Do not pair with RotationCalibrator: too few events give the rotation "
            "null insufficient power. "
            "Regime: almost all bars labelled NEUTRAL (5/15/60/15/5 %), so only "
            "genuine trend extremes get a regime tag — isolates rare signals in "
            "clear-cut market conditions."
        ),
        # ── M0: MarketContext ────────────────────────────────────────────────
        # NEUTRAL-dominato: solo i veri estremi ricevono BULL/BEAR.
        # Finestra stabile lunga: evita di confondere transizioni brevi con
        # cambi di regime reali quando gli eventi sono pochi.
        "mc_target_distribution": [0.05, 0.15, 0.60, 0.15, 0.05],
        "mc_stable_window": 20,
        # ── M1: EventDiscovery ───────────────────────────────────────────────
        "daily_min_tpm": 1.0,
        "daily_max_dispersion": 1.0,
        "max_and_components": 1,
        # ── M2: AlphaDiscovery ───────────────────────────────────────────────
        "min_lift": 0.10,
        "min_cohens_d": 0.15,
        "fdr_q": 0.05,
        "oos_max_p": 0.10,
        # ── M3: RuleDiscovery ────────────────────────────────────────────────
        "daily_rd_min_tpm": 1.0,
    },
    "balanced": {
        # ── descrizione bilingue ─────────────────────────────────────────────
        "description_it": (
            "Compromesso ragionato. Frequenza moderata con buon equilibrio IS/OOS. "
            "Default sensato per la maggior parte degli asset e dei timeframe. "
            "Regime: distribuzione a campana simmetrica (10/20/40/20/10 %), "
            "finestra di stabilità standard — un buon punto di partenza per "
            "qualsiasi analisi senza assunzioni forti sul comportamento del mercato."
        ),
        "description_en": (
            "Sensible trade-off. Moderate frequency with a good IS/OOS balance. "
            "A solid default for most assets and timeframes. "
            "Regime: symmetric bell distribution (10/20/40/20/10 %), standard "
            "stability window — a good starting point with no strong assumptions "
            "about market behaviour."
        ),
        # ── M0: MarketContext ────────────────────────────────────────────────
        # Campana simmetrica: frequenze moderate su tutti i regimi.
        "mc_target_distribution": [0.10, 0.20, 0.40, 0.20, 0.10],
        "mc_stable_window": 12,
        # ── M1: EventDiscovery ───────────────────────────────────────────────
        "daily_min_tpm": 3.0,
        "daily_max_dispersion": 2.0,
        "max_and_components": 2,
        # ── M2: AlphaDiscovery ───────────────────────────────────────────────
        "min_lift": 0.06,
        "min_cohens_d": 0.10,
        "fdr_q": 0.15,
        "oos_max_p": 0.20,
        # ── M3: RuleDiscovery ────────────────────────────────────────────────
        "daily_rd_min_tpm": 2.5,
    },
    "sweep": {
        # ── descrizione bilingue ─────────────────────────────────────────────
        "description_it": (
            "Ampio sweep, molti candidati. Soglie permissive progettate per lavorare "
            "col RotationCalibrator (k≥100), che si occupa di filtrare i falsi "
            "positivi prodotti dalla ricerca estesa. Usare sempre con "
            "promoted_contracts(min_lift=0.05) per escludere i contratti a lift "
            "negativa o nulla. "
            "Regime: stessa distribuzione di balanced, ma finestra di stabilità "
            "corta — include anche gli eventi vicini alle transizioni di regime, "
            "che il RotationCalibrator valuterà se robusti o casuali."
        ),
        "description_en": (
            "Wide sweep, many candidates. Permissive thresholds designed to pair "
            "with RotationCalibrator (k≥100), which filters false positives produced "
            "by the broad search. Always use with promoted_contracts(min_lift=0.05) "
            "to drop zero/negative lift contracts. "
            "Regime: same distribution as balanced, but shorter stability window — "
            "includes events near regime transitions; RotationCalibrator decides "
            "whether they are robust or spurious."
        ),
        # ── M0: MarketContext ────────────────────────────────────────────────
        # Finestra corta: include eventi vicino alle transizioni, che il
        # RotationCalibrator si incarica di valutare.
        "mc_target_distribution": [0.10, 0.20, 0.40, 0.20, 0.10],
        "mc_stable_window": 6,
        # ── M1: EventDiscovery ───────────────────────────────────────────────
        "daily_min_tpm": 1.0,
        "daily_max_dispersion": 2.5,
        "max_and_components": 2,
        # ── M2: AlphaDiscovery ───────────────────────────────────────────────
        "min_lift": 0.05,
        "min_cohens_d": 0.05,
        "fdr_q": 0.25,
        "oos_max_p": 0.20,
        # ── M3: RuleDiscovery ────────────────────────────────────────────────
        "daily_rd_min_tpm": 1.0,
    },
    "burst": {
        # ── descrizione bilingue ─────────────────────────────────────────────
        "description_it": (
            "Eventi concentrati nel tempo: regime-change, momentum, spike di volume. "
            "Dispersione alta esplicitamente permessa — gli eventi burst per "
            "definizione non si distribuiscono uniformemente nel tempo. "
            "Ottimo su mercati con forte stagionalità o run direzionali prolungati. "
            "Regime: distribuzione bimodale (20/25/10/25/20 %), più barre nei regimi "
            "estremi (STRONG_BULL / STRONG_BEAR), meno nel centro — il mercato viene "
            "visto come tendenzialmente direzionale. Finestra di stabilità corta per "
            "catturare eventi durante le transizioni di regime."
        ),
        "description_en": (
            "Temporally concentrated events: regime changes, momentum, volume spikes. "
            "High dispersion is explicitly allowed — burst events by definition do "
            "not spread uniformly over time. "
            "Best for assets with strong seasonality or sustained directional runs. "
            "Regime: bimodal distribution (20/25/10/25/20 %), more bars in extreme "
            "regimes (STRONG_BULL / STRONG_BEAR), less in the centre — the market "
            "is seen as inherently directional. Short stability window to capture "
            "events during regime transitions."
        ),
        # ── M0: MarketContext ────────────────────────────────────────────────
        # Bimodale: enfatizza i regimi estremi rispetto al NEUTRAL.
        # Finestra corta: gli eventi burst spesso si trovano a cavallo delle
        # transizioni di regime.
        "mc_target_distribution": [0.20, 0.25, 0.10, 0.25, 0.20],
        "mc_stable_window": 6,
        # ── M1: EventDiscovery ───────────────────────────────────────────────
        "daily_min_tpm": 2.5,
        "daily_max_dispersion": 5.0,
        "max_and_components": 2,
        # ── M2: AlphaDiscovery ───────────────────────────────────────────────
        "min_lift": 0.08,
        "min_cohens_d": 0.12,
        "fdr_q": 0.10,
        "oos_max_p": 0.10,
        # ── M3: RuleDiscovery ────────────────────────────────────────────────
        "daily_rd_min_tpm": 2.0,
    },
}

PRESETS = list(_PRESET_SPECS.keys())


# ── Public API ────────────────────────────────────────────────────────────────

def forge_preset(
    preset: str,
    timeframe: str,
    asset: str = "ASSET",
    train_ratio: float = 0.70,
    **overrides,
) -> Tuple[MarketContextConfig, DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig]:
    """Return a ``(MarketContextConfig, DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig)`` quadruple.

    M0 (MarketContext), M1 (EventDiscovery), M2 (AlphaDiscovery) e M3
    (RuleDiscovery) sono configurati coerentemente: stesso criterio di frequenza
    ``min_tpm`` tra M1 e M3; ``MarketContextConfig`` con ``threshold_mode="balanced"``
    e distribuzione di regime calibrata sulla filosofia del preset.

    Parameters
    ----------
    preset : str
        One of ``"sniper"``, ``"balanced"``, ``"sweep"``, ``"burst"``.
    timeframe : str
        Bar size, e.g. ``"1D"``, ``"4H"``, ``"1H"``, ``"15m"``, ``"5m"``.
        Controls the scaling of ``min_tpm``, ``max_dispersion``,
        ``horizon_grid`` and ``bars_per_day``.
    asset : str
        Asset label forwarded to ``AlphaConfig``.
    train_ratio : float
        IS/OOS split.  Default 0.70 (70 % training, 30 % OOS).
    **overrides
        Override any computed parameter by name.  Supported keys:

        M0: ``mc_target_distribution`` (list[float], 5 elements),
        ``mc_stable_window`` (int).

        M1: ``min_tpm``, ``max_dispersion``, ``max_and_components``,
        ``timestamp_col``.

        M2: ``min_lift``, ``min_cohens_d``, ``fdr_q``, ``oos_max_p``,
        ``horizon_grid``, ``bars_per_day``.

        M3: ``rd_min_tpm``.

    Returns
    -------
    mc_cfg : MarketContextConfig
    disc_cfg : DiscoveryConfig
    alpha_cfg : AlphaConfig
    rd_cfg : RuleDiscoveryConfig

    Examples
    --------
    >>> mc_cfg, disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", "1D", asset="BTC")
    >>> result = forge(df,
    ...               market_context_config=mc_cfg,
    ...               event_discovery_config=disc_cfg,
    ...               alpha_config=alpha_cfg,
    ...               rule_discovery_config=rd_cfg)

    >>> # Sweep + RotationCalibrator
    >>> mc_cfg, disc_cfg, alpha_cfg, rd_cfg = forge_preset("sweep", "4H", asset="ETH")
    >>> result = forge(df,
    ...               market_context_config=mc_cfg,
    ...               event_discovery_config=disc_cfg,
    ...               alpha_config=alpha_cfg,
    ...               rule_discovery_config=rd_cfg,
    ...               rotation_calibration=RotationConfig(k=100))
    """
    if preset not in _PRESET_SPECS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {PRESETS}")

    spec = _PRESET_SPECS[preset]
    tf = _TFClass(timeframe)

    # ── M0: MarketContext ────────────────────────────────────────────────────
    # threshold_mode="balanced" su tutti i preset: le soglie si adattano alla
    # distribuzione reale dell'EMA ratio dell'asset, evitando cut fissi che
    # producono regimi sbilanciati su asset con trend strutturale.
    mc_target_distribution: List[float] = overrides.pop(
        "mc_target_distribution", spec["mc_target_distribution"]
    )
    mc_stable_window: int = overrides.pop(
        "mc_stable_window", spec["mc_stable_window"]
    )
    mc_cfg = MarketContextConfig(
        ema_proxy=EMAProxyConfig(
            threshold_mode="balanced",
            target_distribution=mc_target_distribution,
        ),
        stable_window=mc_stable_window,
    )

    # ── M1: EventDiscovery ───────────────────────────────────────────────────
    min_tpm = overrides.pop("min_tpm", tf.scale_tpm(spec["daily_min_tpm"]))
    max_dispersion = overrides.pop(
        "max_dispersion", tf.scale_dispersion(spec["daily_max_dispersion"])
    )
    max_and = overrides.pop("max_and_components", spec["max_and_components"])
    timestamp_col = overrides.pop("timestamp_col", "open_dt")

    disc_cfg = DiscoveryConfig(
        gate_params=GateParams(min_tpm=min_tpm, max_dispersion=max_dispersion),
        timestamp_col=timestamp_col,
        max_and_components=max_and,
        train_ratio=1.0,
    )

    # ── M2: AlphaDiscovery ───────────────────────────────────────────────────
    horizon_grid = overrides.pop("horizon_grid", tf.horizon_grid)
    bars_per_day = overrides.pop("bars_per_day", tf.bars_per_day)
    min_lift = overrides.pop("min_lift", spec["min_lift"])
    min_cohens_d = overrides.pop("min_cohens_d", spec["min_cohens_d"])
    fdr_q = overrides.pop("fdr_q", spec["fdr_q"])
    oos_max_p = overrides.pop("oos_max_p", spec["oos_max_p"])

    alpha_cfg = AlphaConfig(
        asset=asset,
        timeframe=timeframe,
        horizon_grid=horizon_grid,
        bars_per_day=bars_per_day if tf.cls != _TFClass.DAILY else None,
        train_ratio=train_ratio,
        thresholds=PromotionThresholds(
            min_lift=min_lift,
            min_cohens_d=min_cohens_d,
            use_fdr=True,
            fdr_q=fdr_q,
            oos_max_p=oos_max_p,
        ),
    )

    # ── M3: RuleDiscovery ────────────────────────────────────────────────────
    # rd_min_tpm leggermente più permissivo di M1: non tutti i segnali
    # AlphaDiscovery si traducono in trade eseguiti (fill rate, overlap regole).
    rd_min_tpm = overrides.pop(
        "rd_min_tpm", tf.scale_tpm(spec["daily_rd_min_tpm"])
    )

    if overrides:
        raise TypeError(f"Unexpected override keys: {list(overrides)}")

    rd_cfg = RuleDiscoveryConfig(
        criteria=SelectionCriteria(min_tpm=rd_min_tpm),
    )

    return mc_cfg, disc_cfg, alpha_cfg, rd_cfg


def preset_info(preset: Optional[str] = None, lang: str = "it") -> None:
    """Print a human-readable description of one or all presets.

    Parameters
    ----------
    preset : str, optional
        Name of the preset to describe.  ``None`` prints all presets.
    lang : str
        Language for the description: ``"it"`` (Italian, default) or ``"en"``
        (English).  The parameter table is always shown in full.
    """
    if lang not in ("it", "en"):
        raise ValueError(f"lang must be 'it' or 'en', got {lang!r}")
    targets = [preset] if preset else PRESETS
    for name in targets:
        if name not in _PRESET_SPECS:
            raise ValueError(f"Unknown preset '{name}'. Available: {PRESETS}")
        spec = _PRESET_SPECS[name]
        desc = spec[f"description_{lang}"]
        dist = spec["mc_target_distribution"]
        dist_str = "/".join(str(int(v * 100)) for v in dist)
        print(f"\n{'─'*65}")
        print(f"  {name.upper()}")
        print()
        for line in _wrap(desc, width=63, indent="  "):
            print(line)
        print()
        print(f"  M0 regime  : balanced  dist={dist_str}%  "
              f"stable_window={spec['mc_stable_window']}")
        print(f"  M1 gate    : min_tpm={spec['daily_min_tpm']}  "
              f"max_dispersion={spec['daily_max_dispersion']}  "
              f"max_and={spec['max_and_components']}")
        print(f"  M2 alpha   : min_lift={spec['min_lift']}  "
              f"cohens_d={spec['min_cohens_d']}  "
              f"fdr_q={spec['fdr_q']}  oos_max_p={spec['oos_max_p']}")
        print(f"  M3 gate    : rd_min_tpm={spec['daily_rd_min_tpm']}")


def _wrap(text: str, width: int = 65, indent: str = "") -> list:
    """Simple word-wrap for preset_info output."""
    words = text.split()
    lines, current = [], indent
    for word in words:
        if current == indent:
            current += word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = indent + word
    if current != indent:
        lines.append(current)
    return lines
