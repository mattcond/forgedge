"""Preset configurations for forge() — opinionated starting points.

Traduce un'intenzione di alto livello ("sniper", "balanced", "sweep", "burst")
in una tripla (DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig) pronta da
passare a forge(), con M1/M2/M3 allineati sullo stesso criterio di frequenza.

Coerenza M1–M2–M3
------------------
Il ``min_tpm`` usato da EventDiscovery (M1) e RuleDiscovery (M3) devono essere
consistenti: un evento ammesso da M1 con tpm=1.0 non può essere rifiutato da M3
che richiede tpm=2.0 per default.  forge_preset() imposta lo stesso floor di
frequenza per tutti e tre i moduli, scalato al timeframe.

Scaling dei parametri per timeframe
------------------------------------
* ``min_tpm`` scala linearmente con le barre/mese: lo stesso numero minimo
  di attivazioni IS richiesto per potere statistico (~30–50) si traduce in
  tpm molto diversi a seconda della densità di barre.
* ``max_dispersion`` scala verso il basso sui timeframe corti: la legge dei
  grandi numeri stabilizza i conteggi mensili all'aumentare delle barre,
  quindi lo stesso ID naturale dei "buoni" eventi è più basso su intraday/HFT
  rispetto al daily.
* ``horizon_grid`` e ``bars_per_day`` sono scalati al timeframe.

Utilizzo
---------
    from forgedge.presets import forge_preset
    from forgedge import forge, RotationConfig

    disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", timeframe="1D", asset="ADA")
    result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
                   rule_discovery_config=rd_cfg)

    # Con RotationCalibrator (raccomandato per "sweep")
    disc_cfg, alpha_cfg, rd_cfg = forge_preset("sweep", timeframe="1H", asset="ETH")
    result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
                   rule_discovery_config=rd_cfg,
                   rotation_calibration=RotationConfig(k=100))
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from .alpha_discovery.models import AlphaConfig, PromotionThresholds
from .event_discovery.discovery import DiscoveryConfig
from .event_discovery.models import GateParams
from .rule_discovery.models import RuleDiscoveryConfig, SelectionCriteria

__all__ = ["forge_preset", "preset_info", "default_horizon_grid", "PRESETS"]


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


def default_horizon_grid(timeframe: str) -> Optional[Tuple[int, ...]]:
    """Timeframe-calibrated horizon grid for ``forge()``'s default AlphaConfig.

    ``AlphaConfig.horizon_grid`` defaults to a grid calibrated on ~hourly bars
    (up to 48 bars ≈ 48 hours).  On a daily-or-slower timeframe the same grid
    silently means holding periods of up to 48 *days* — the "silent footgun"
    documented in ``docs/analysis/lowfreq_robustness.md``.  This helper returns
    the presets' class-calibrated grid when the bar duration is one day or
    longer, and ``None`` otherwise (including unparseable timeframes), meaning
    "keep the AlphaConfig default".
    """
    try:
        tf = _TFClass(timeframe)
    except ValueError:
        return None
    return tf.horizon_grid if tf.cls == _TFClass.DAILY else None


# ── Preset definitions (daily-calibrated) ─────────────────────────────────────

# Ogni preset definisce due famiglie di parametri per min_tpm:
#   daily_min_tpm         — modalità "bar": barre attive/mese (retrocompatibile)
#   daily_min_tpm_episode — modalità "episode": episodi/mese ≈ trade/mese
#                           (default da GateParams post-fix #134)
#
# I due valori NON sono intercambiabili: in modalità episode min_tpm misura
# episodi (eventi distinti), non barre, quindi un RSI<30 con 3 barre per
# episodio ha 0.57 episodi/mese ma 1.70 barre/mese.
#
# Altri campi:
#   daily_rd_min_tpm(_episode) — gate M3 (stesso conteggio del gate M1)
#   daily_max_dispersion        — gate M1 (uguale in entrambe le modalità;
#                                 il Poisson-floor nel ConsistencyGate protegge
#                                 già dalla dispersione artificiale in episode mode)
#   parametri alpha             — gate M2 (AlphaDiscovery)
_PRESET_SPECS: dict = {
    "sniper": {
        "description": (
            "Rari e regolari. Alta precisione statistica, regole semplici. "
            "Richiede IS lungo (>=2 anni su 1D). "
            "Non abbinare a RotationCalibrator."
        ),
        "daily_min_tpm": 1.0,
        "daily_min_tpm_episode": 0.3,   # ≈1 episodio ogni 3 mesi
        "daily_rd_min_tpm": 1.0,
        "daily_rd_min_tpm_episode": 0.3,
        "daily_max_dispersion": 1.0,
        "min_lift": 0.10,
        "min_cohens_d": 0.15,
        "fdr_q": 0.05,
        "oos_max_p": 0.10,
        "max_and_components": 1,
    },
    "balanced": {
        "description": (
            "Compromesso ragionato. Frequenza moderata, buon equilibrio IS/OOS. "
            "Default sensato per la maggior parte degli asset e timeframe."
        ),
        "daily_min_tpm": 3.0,
        "daily_min_tpm_episode": 1.0,   # ≈1 episodio/mese (trade/mese minimo)
        "daily_rd_min_tpm": 2.5,
        "daily_rd_min_tpm_episode": 0.8,
        "daily_max_dispersion": 2.0,
        "min_lift": 0.06,
        "min_cohens_d": 0.10,
        "fdr_q": 0.15,
        "oos_max_p": 0.20,
        "max_and_components": 2,
    },
    "sweep": {
        "description": (
            "Ampio sweep, molti candidati. Soglie permissive. "
            "Progettato per lavorare col RotationCalibrator: "
            "usare sempre rotation_calibration=RotationConfig(k>=100) e "
            "promoted_contracts(min_lift=0.05)."
        ),
        "daily_min_tpm": 1.0,
        "daily_min_tpm_episode": 0.3,   # molto permissivo: il filtro è il RotationCalibrator
        "daily_rd_min_tpm": 1.0,
        "daily_rd_min_tpm_episode": 0.3,
        "daily_max_dispersion": 2.5,
        "min_lift": 0.05,
        "min_cohens_d": 0.05,
        "fdr_q": 0.25,
        "oos_max_p": 0.20,
        "max_and_components": 2,
    },
    "burst": {
        "description": (
            "Eventi concentrati nel tempo (regime-change, momentum, spike di "
            "volume). Dispersione alta esplicitamente permessa. "
            "Ottimo su mercati con forte stagionalità o bull/bear run."
        ),
        "daily_min_tpm": 2.5,
        "daily_min_tpm_episode": 1.5,   # episodi frequenti per natura del burst
        "daily_rd_min_tpm": 2.0,
        "daily_rd_min_tpm_episode": 1.2,
        "daily_max_dispersion": 5.0,
        "min_lift": 0.08,
        "min_cohens_d": 0.12,
        "fdr_q": 0.10,
        "oos_max_p": 0.10,
        "max_and_components": 2,
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
) -> Tuple[DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig]:
    """Return a ``(DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig)`` triple.

    M1 (EventDiscovery), M2 (AlphaDiscovery) e M3 (RuleDiscovery) sono
    configurati con lo stesso criterio di frequenza (``min_tpm``) per evitare
    inconsistenze lungo la pipeline.

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

        M1: ``min_tpm``, ``max_dispersion``, ``max_and_components``,
        ``timestamp_col``, ``event_counting``.

        M2: ``min_lift``, ``min_cohens_d``, ``fdr_q``, ``oos_max_p``,
        ``horizon_grid``, ``bars_per_day``.

        M3: ``rd_min_tpm``.

    Returns
    -------
    disc_cfg : DiscoveryConfig
    alpha_cfg : AlphaConfig
    rd_cfg : RuleDiscoveryConfig

    Examples
    --------
    >>> disc_cfg, alpha_cfg, rd_cfg = forge_preset("balanced", "1D", asset="BTC")
    >>> result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
    ...               rule_discovery_config=rd_cfg)

    >>> # Sweep + RotationCalibrator
    >>> disc_cfg, alpha_cfg, rd_cfg = forge_preset("sweep", "4H", asset="ETH")
    >>> result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
    ...               rule_discovery_config=rd_cfg,
    ...               rotation_calibration=RotationConfig(k=100))
    """
    if preset not in _PRESET_SPECS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {PRESETS}")

    spec = _PRESET_SPECS[preset]
    tf = _TFClass(timeframe)

    # ── M1: EventDiscovery ───────────────────────────────────────────────
    # event_counting selects which daily_min_tpm calibration to use:
    # "episode" → daily_min_tpm_episode (episodi/mese ≈ trade/mese)
    # "bar"     → daily_min_tpm (barre/mese, retrocompatibile)
    event_counting = overrides.pop("event_counting", "episode")
    if event_counting == "episode":
        daily_tpm_key = "daily_min_tpm_episode"
    else:
        daily_tpm_key = "daily_min_tpm"
    min_tpm = overrides.pop("min_tpm", tf.scale_tpm(spec[daily_tpm_key]))
    max_dispersion = overrides.pop(
        "max_dispersion", tf.scale_dispersion(spec["daily_max_dispersion"])
    )
    max_and = overrides.pop("max_and_components", spec["max_and_components"])
    timestamp_col = overrides.pop("timestamp_col", "open_dt")

    disc_cfg = DiscoveryConfig(
        gate_params=GateParams(
            min_tpm=min_tpm,
            max_dispersion=max_dispersion,
            event_counting=event_counting,
        ),
        timestamp_col=timestamp_col,
        max_and_components=max_and,
        train_ratio=1.0,
    )

    # ── M2: AlphaDiscovery ───────────────────────────────────────────────
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

    # ── M3: RuleDiscovery ────────────────────────────────────────────────
    # rd_min_tpm è leggermente più largo di min_tpm M1: non tutti i segnali
    # prodotti da AlphaDiscovery si traducono in trade eseguiti (fill rate,
    # overlap delle regole), quindi il gate M3 può essere appena più permissivo
    # senza creare incoerenza.  Usa la stessa unità (bar/episode) del gate M1.
    if event_counting == "episode":
        daily_rd_tpm_key = "daily_rd_min_tpm_episode"
    else:
        daily_rd_tpm_key = "daily_rd_min_tpm"
    rd_min_tpm = overrides.pop(
        "rd_min_tpm", tf.scale_tpm(spec[daily_rd_tpm_key])
    )

    if overrides:
        raise TypeError(f"Unexpected override keys: {list(overrides)}")

    rd_cfg = RuleDiscoveryConfig(
        criteria=SelectionCriteria(min_tpm=rd_min_tpm),
    )

    return disc_cfg, alpha_cfg, rd_cfg


def preset_info(preset: Optional[str] = None) -> None:
    """Print a human-readable description of one or all presets."""
    targets = [preset] if preset else PRESETS
    for name in targets:
        if name not in _PRESET_SPECS:
            raise ValueError(f"Unknown preset '{name}'. Available: {PRESETS}")
        spec = _PRESET_SPECS[name]
        print(f"\n{'─'*60}")
        print(f"  {name.upper()}")
        print(f"  {spec['description']}")
        print(f"  M1 gate : min_tpm(episode)={spec['daily_min_tpm_episode']}  "
              f"min_tpm(bar)={spec['daily_min_tpm']}  "
              f"max_dispersion={spec['daily_max_dispersion']}  "
              f"max_and={spec['max_and_components']}")
        print(f"  M2 alpha: min_lift={spec['min_lift']}  "
              f"cohens_d={spec['min_cohens_d']}  "
              f"fdr_q={spec['fdr_q']}  oos_max_p={spec['oos_max_p']}")
        print(f"  M3 gate : rd_min_tpm(episode)={spec['daily_rd_min_tpm_episode']}  "
              f"rd_min_tpm(bar)={spec['daily_rd_min_tpm']}")
