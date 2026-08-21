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
* ``horizon_grid`` e ``bars_per_day`` sono risolti dal timeframe (#179, #196).

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

from typing import Optional, Tuple

from .alpha_discovery.models import AlphaConfig, PromotionThresholds
from .event_discovery.discovery import DiscoveryConfig
from .event_discovery.models import GateParams
from .resolver import PipelineContext, resolve_config, timeframe_minutes
from .rule_discovery.models import RuleDiscoveryConfig, SelectionCriteria
from .unset import UNSET

__all__ = ["forge_preset", "preset_info", "default_horizon_grid", "PRESETS"]


# ── Timeframe parsing ─────────────────────────────────────────────────────────

#: Timeframe parsing lives in :mod:`forgedge.resolver` — a single
#: implementation shared with the session context, rather than one copy here
#: and one there.
_tf_minutes = timeframe_minutes


class _TFClass:
    """Classify a timeframe into daily / intraday / hft and expose scaling.

    The bar arithmetic itself is **not** reimplemented here: it comes from
    :class:`~forgedge.resolver.PipelineContext`, which is the session's single
    source of bar duration (F5).  What this class adds on top is the *class*
    — daily / intraday / hft — and the calibrations that hang off it, which
    are genuinely preset knowledge and not arithmetic.
    """

    DAILY = "daily"
    INTRADAY = "intraday"
    HFT = "hft"

    #: Holding-period grids per class.  These are calibrations, not
    #: conversions: an hourly session scans up to 24 hours, a daily one up to
    #: 10 days — converting the hourly grid to daily bars would collapse it to
    #: a single bar, which is not the same question.
    _HORIZON_GRIDS = {
        DAILY: (1, 2, 3, 5, 7, 10),
        INTRADAY: (1, 2, 4, 8, 12, 24),
        HFT: (1, 2, 5, 10, 20, 50),
    }

    def __init__(self, timeframe: str) -> None:
        self.timeframe = timeframe
        self.ctx = PipelineContext(timeframe=timeframe)
        mins = _tf_minutes(timeframe)
        if mins >= 1440:
            self.cls = self.DAILY
        elif mins >= 60:
            self.cls = self.INTRADAY
        else:
            self.cls = self.HFT
        self.horizon_grid = self._HORIZON_GRIDS[self.cls]

    @property
    def bars_per_month(self) -> float:
        return self.ctx.bars_per_month

    @property
    def bars_per_day(self) -> int:
        """Rounded, and floored at 1.

        The floor only bites on weekly-or-slower bars, where the exact value
        is below 1 and this is used as a divisor.
        """
        return max(1, round(self.ctx.bars_per_day))

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

    Kept as a public helper, but no longer the mechanism: since #196
    ``AlphaConfig.horizon_grid`` is session-resolved from the same
    ``_TFClass`` table, on every path rather than only when ``forge()`` built
    the config itself.  This returns the daily class grid when the bar
    duration is one day or longer and ``None`` otherwise (including
    unparseable timeframes) — a question about *classes*, useful for a caller
    deciding whether a timeframe is in the slow class at all.
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
        The session's IS/OOS split.  Default 0.70 (70 % training, 30 % OOS).

        It governs **M2 and M3**: Alpha Discovery cuts here, and under
        :func:`forgedge.forge` the same cut becomes the session's one temporal
        axis, which Rule Discovery measures its walk-forward folds against
        (F6, #180).  Event Discovery keeps ``train_ratio=1.0`` — see
        ``DiscoveryConfig.train_ratio`` below for why that is a decision and
        not an omission.
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
    # Left UNSET unless the caller asks for one: the schema is a session
    # fact, not a profile choice, and the resolver propagates it to every
    # module instead of the preset setting M1's copy alone (F10).
    timestamp_col = overrides.pop("timestamp_col", UNSET)

    disc_cfg = DiscoveryConfig(
        gate_params=GateParams(
            min_tpm=min_tpm,
            max_dispersion=max_dispersion,
            event_counting=event_counting,
        ),
        timestamp_col=timestamp_col,
        max_and_components=max_and,
        # Deliberate, and *not* what `train_ratio` above governs (F6, #180).
        # By the pipeline's first invariant M1 never observes the forward
        # return: thresholds come from the temporal structure of the
        # indicators alone. Reserving an OOS tail from it would therefore
        # protect nothing about returns — what crosses the boundary is
        # distributional, the percentiles being computed over the whole span,
        # and that is the weaker form the seventh invariant asks for anyway
        # (thresholds distributional per asset and period).
        #
        # Withholding 30% here costs real structure — the rarest events are
        # the ones a shorter window fails to see at all — for a guarantee M1
        # does not need. The session budget records this choice explicitly so
        # `ForgeResult.time_budget` states M1's axis instead of leaving it to
        # be inferred from a split M1 does not use.
        train_ratio=1.0,
    )

    # ── M2: AlphaDiscovery ───────────────────────────────────────────────
    # Left UNSET by default: the resolver derives it from the same declared
    # timeframe this preset was built for, from the same `_TFClass` table, so
    # there is no second copy of the calibration here (#196 — the same move
    # `bars_per_day` made in #179).  An explicit override still wins.
    horizon_grid = overrides.pop("horizon_grid", UNSET)
    # Left UNSET by default: the resolver derives it from the same declared
    # timeframe this preset was built for, so there is no third copy of the
    # conversion here (F5, #179).  An explicit override still wins.
    bars_per_day = overrides.pop("bars_per_day", UNSET)
    min_lift = overrides.pop("min_lift", spec["min_lift"])
    min_cohens_d = overrides.pop("min_cohens_d", spec["min_cohens_d"])
    fdr_q = overrides.pop("fdr_q", spec["fdr_q"])
    oos_max_p = overrides.pop("oos_max_p", spec["oos_max_p"])

    alpha_cfg = AlphaConfig(
        asset=asset,
        timeframe=timeframe,
        horizon_grid=horizon_grid,
        bars_per_day=bars_per_day,
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
        # The scoring knobs decide which operating point is published, so they
        # belong next to the gates rather than being invisible (F3, #178).
        # Resolved for a daily session — they track `criteria.min_tpm`, and
        # `preset_info()` is timeframe-agnostic, so the row states which.
        _d, alpha_cfg, rd = forge_preset(name, "1D")
        _ctx = PipelineContext(timeframe="1D")
        rd = resolve_config(rd, "rule_discovery", _ctx)
        alpha_cfg = resolve_config(alpha_cfg, "alpha", _ctx)
        print(f"  M3 score: pf_min_tpm={rd.scoring.pf_min_tpm:g}  "
              f"pf_min_trades={rd.scoring.pf_min_trades}  "
              f"min_pf_score_tpm={rd.criteria.min_pf_score_tpm:g}   "
              f"[resolved for 1D; pf_min_tpm tracks criteria.min_tpm]")
        # Every significance threshold, resolved — not only the two that
        # happened to be in the preset spec (F9, #182).  The five that *are*
        # a per-hypothesis alpha come from one place; the two that are not
        # are printed next to them saying so, since the whole confusion this
        # fixes was reading seven numbers as one quantity.
        th = alpha_cfg.thresholds
        print(f"  alpha   : max_p_value={th.max_p_value:g}  "
              f"ic_max_p={th.ic_max_p:g}  "
              f"max_ttest_p={rd.criteria.max_ttest_p:g}  "
              f"max_rotation_p={rd.criteria.max_rotation_p:g}   "
              f"[all from ctx.alpha]")
        print(f"  not-α   : fdr_q={th.fdr_q:g} (false-discovery rate over a "
              f"family)  oos_max_p={th.oos_max_p:g} (confirmation, not "
              f"discovery)  min_pass_rate={spec.get('min_pass_rate', 0.6):g} "
              f"(a vote, not a probability)")
