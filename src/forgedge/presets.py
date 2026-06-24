"""Preset configurations for forge() — opinionated starting points.

Traduce un'intenzione di alto livello ("sniper", "balanced", "sweep", "burst")
in una coppia (DiscoveryConfig, AlphaConfig) pronta da passare a forge().

I parametri core (min_tpm, max_dispersion, soglie alpha) sono validati
empiricamente su ADA 1D 2024 e scalati al timeframe tramite la classe
interna ``_TFClass``.

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

    disc_cfg, alpha_cfg = forge_preset("balanced", timeframe="1D", asset="ADA")
    result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg)

    # Con RotationCalibrator (raccomandato per "sweep")
    disc_cfg, alpha_cfg = forge_preset("sweep", timeframe="1H", asset="ETH")
    result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
                   rotation_calibration=RotationConfig(k=100))
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional, Tuple

from .alpha_discovery.models import AlphaConfig, PromotionThresholds
from .event_discovery.discovery import DiscoveryConfig
from .event_discovery.models import GateParams

__all__ = ["forge_preset", "PRESETS"]


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
        # daily: >= 1D (1440 min); intraday: >= 1H (60 min); hft: < 1H
        if mins >= 1440:
            self.cls = self.DAILY
            self.bars_per_month = 1440 / mins * 30  # ~30 for 1D
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
        """Scale a daily-calibrated min_tpm to this timeframe.

        Preserves the target absolute activation count: ~30 activations/year
        on daily becomes the same count on intraday, but expressed as a
        higher tpm because there are more bars per month.
        """
        daily_bars_per_month = 30.0
        return daily_tpm * (self.bars_per_month / daily_bars_per_month)

    def scale_dispersion(self, daily_dispersion: float) -> float:
        """Scale max_dispersion down for shorter timeframes.

        On intraday/HFT, the law of large numbers stabilises monthly counts
        (more bars → lower natural ID), so the same discriminatory power
        requires a tighter threshold.
        """
        if self.cls == self.DAILY:
            return daily_dispersion
        if self.cls == self.INTRADAY:
            return round(daily_dispersion * 0.45, 2)
        # HFT
        return round(daily_dispersion * 0.20, 2)


# ── Preset definitions (daily-calibrated) ─────────────────────────────────────

# Ogni preset è definito con valori calibrati su daily (1D).
# Lo scaling al timeframe avviene in forge_preset().
_PRESET_SPECS: dict = {
    "sniper": {
        "description": (
            "Rari e regolari. Alta precisione statistica, regole semplici. "
            "Richiede IS lungo (>=2 anni su 1D). "
            "Non abbinare a RotationCalibrator."
        ),
        "daily_min_tpm": 1.0,
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
) -> Tuple[DiscoveryConfig, AlphaConfig]:
    """Return a (DiscoveryConfig, AlphaConfig) pair for the given preset.

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

        DiscoveryConfig-level: ``min_tpm``, ``max_dispersion``,
        ``max_and_components``, ``timestamp_col``.

        AlphaConfig/PromotionThresholds-level: ``min_lift``,
        ``min_cohens_d``, ``fdr_q``, ``oos_max_p``, ``horizon_grid``,
        ``bars_per_day``.

    Returns
    -------
    disc_cfg : DiscoveryConfig
    alpha_cfg : AlphaConfig

    Examples
    --------
    >>> disc_cfg, alpha_cfg = forge_preset("balanced", "1D", asset="BTC")
    >>> result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg)

    >>> # Sweep + RotationCalibrator
    >>> disc_cfg, alpha_cfg = forge_preset("sweep", "4H", asset="ETH")
    >>> result = forge(df, event_discovery_config=disc_cfg, alpha_config=alpha_cfg,
    ...               rotation_calibration=RotationConfig(k=100))
    >>> contracts = result.alpha_discovery.promoted_contracts(min_lift=0.05)
    """
    if preset not in _PRESET_SPECS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {PRESETS}"
        )

    spec = _PRESET_SPECS[preset]
    tf = _TFClass(timeframe)

    # ── Scale gate parameters ────────────────────────────────────────────
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

    # ── Scale alpha parameters ───────────────────────────────────────────
    horizon_grid = overrides.pop("horizon_grid", tf.horizon_grid)
    bars_per_day = overrides.pop("bars_per_day", tf.bars_per_day)
    min_lift = overrides.pop("min_lift", spec["min_lift"])
    min_cohens_d = overrides.pop("min_cohens_d", spec["min_cohens_d"])
    fdr_q = overrides.pop("fdr_q", spec["fdr_q"])
    oos_max_p = overrides.pop("oos_max_p", spec["oos_max_p"])

    if overrides:
        raise TypeError(f"Unexpected override keys: {list(overrides)}")

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

    return disc_cfg, alpha_cfg


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
        print(f"  daily gate  : min_tpm={spec['daily_min_tpm']}  "
              f"max_dispersion={spec['daily_max_dispersion']}  "
              f"max_and={spec['max_and_components']}")
        print(f"  alpha       : min_lift={spec['min_lift']}  "
              f"cohens_d={spec['min_cohens_d']}  "
              f"fdr_q={spec['fdr_q']}  oos_max_p={spec['oos_max_p']}")
