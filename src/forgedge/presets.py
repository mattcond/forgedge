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
  rispetto al daily.  Vale solo in ``event_counting="bar"`` — in ``"episode"``
  (il default) questo campo non è letto dal gate, vedi sotto.
* ``dispersion_margin`` — la controparte di ``max_dispersion`` in
  ``"episode"`` mode (#205) — NON scala per timeframe: è un moltiplicatore
  sopra il floor statistico di Poisson, già scale-free per costruzione.
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
#   daily_rd_min_tpm     — gate M3, calibrato in barre/mese (M3 non ha nozione
#                          di episodi: apre un trade su ogni barra attiva).
#                          `daily_rd_min_tpm / daily_min_tpm` è il fill ratio
#                          del preset; in modalità "episode" quel rapporto va
#                          applicato al tasso in barre, non a quello in
#                          episodi — la conversione episodi→barre è
#                          PipelineContext.bars_per_episode, non un'altra
#                          coppia di chiavi qui (#204: la coppia
#                          `daily_rd_min_tpm_episode` che c'era prima
#                          applicava il fill ratio senza convertire l'unità,
#                          sottostimando il floor di M3 di ~1.8x).
#   daily_max_dispersion — gate M1, "bar" mode ONLY: soglia assoluta,
#                          nessun floor.  In "episode" mode questo campo non
#                          è letto dal gate — vedi dispersion_margin (#205:
#                          confrontare un valore assoluto contro
#                          max(daily_max_dispersion, poisson_floor) rendeva
#                          il valore del preset codice morto su 12 delle 16
#                          combinazioni preset×timeframe misurate, perché il
#                          floor di Poisson non scala col timeframe mentre
#                          daily_max_dispersion sì — scala_dispersion() lo
#                          riduce, il floor no, quindi vince sempre più
#                          spesso quanto più il timeframe è veloce).
#   dispersion_margin    — gate M1, "episode" mode ONLY: moltiplicatore sopra
#                          il floor di Poisson (eff = floor × margin), non
#                          scalato per timeframe (un moltiplicatore è già
#                          scale-free).  È qui che i preset si differenziano
#                          davvero sulla dispersione in episode mode.
#   min_episodes         — gate M1, "episode" mode ONLY: floor assoluto di
#                          potenza statistica (Criterion 2).  Indipendente dal
#                          tasso, quindi la finestra IS che serve per
#                          raggiungerlo con margine di Poisson al 95% dipende
#                          da entrambi (#206): a min_episodes=10 sniper/sweep
#                          (0.3 episodi/mese) servono 53.3 mesi (4.44 anni),
#                          non i "≥2 anni" che sniper prometteva prima di
#                          questa misura.  sniper tiene 10 — il rigore
#                          statistico è il suo scopo dichiarato, la
#                          descrizione va corretta, non il numero.  sweep lo
#                          abbassa a 5 — è esplicitamente permissivo per
#                          design e delega il rigore al RotationCalibrator a
#                          valle, coerente con la propria filosofia.
#   parametri alpha      — gate M2 (AlphaDiscovery)
_PRESET_SPECS: dict = {
    "sniper": {
        "description": (
            "Rari e regolari. Alta precisione statistica, regole semplici. "
            "Richiede IS molto lungo (~4.5 anni su 1D al proprio tasso minimo, "
            "min_episodes=10 con margine di Poisson al 95% — #206). "
            "Non abbinare a RotationCalibrator."
        ),
        "daily_min_tpm": 1.0,
        "daily_min_tpm_episode": 0.3,   # ≈1 episodio ogni 3 mesi
        "daily_rd_min_tpm": 1.0,
        "daily_max_dispersion": 1.0,
        "dispersion_margin": 1.05,      # quasi zero slack sopra il rumore Poisson
        "min_episodes": 10,             # invariato: il rigore statistico è lo scopo
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
        "daily_max_dispersion": 2.0,
        "dispersion_margin": 1.30,      # margine moderato sopra il floor
        "min_episodes": 10,             # invariato: già coerente al proprio tasso (16 mesi)
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
        "daily_max_dispersion": 2.5,
        "dispersion_margin": 1.60,      # permissivo: il filtro è il RotationCalibrator
        "min_episodes": 5,              # abbassato (#206): permissivo per design, il
                                         # rigore statistico è delegato al RotationCalibrator
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
        "daily_max_dispersion": 5.0,
        "dispersion_margin": 3.00,      # clustering Poisson-implausibile, di proposito
        "min_episodes": 10,             # invariato: già coerente al proprio tasso (10.7 mesi)
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

        M1: ``min_tpm``, ``max_dispersion``, ``dispersion_margin``,
        ``min_episodes``, ``max_and_components``, ``timestamp_col``,
        ``event_counting``.

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
    # Not timeframe-scaled (#205) — a multiplier over the Poisson floor is
    # already scale-free, unlike the absolute `max_dispersion` above.  Reaches
    # the gate only in `event_counting="episode"`; `max_dispersion` above is
    # `"bar"`-mode-only.  Both live on `GateParams` at once because a caller
    # can switch `event_counting` after construction.
    dispersion_margin = overrides.pop("dispersion_margin", spec["dispersion_margin"])
    # Not scaled by timeframe — an absolute count of episodes, and the
    # timeframe's own effect on how long that takes is already carried by
    # `min_tpm`'s scaling.  Per-preset, not a flat class default (#206):
    # sniper/burst/balanced keep the class default 10, sweep lowers it to 5
    # because it is permissive by design and defers rigor to the
    # RotationCalibrator downstream.
    min_episodes = overrides.pop("min_episodes", spec["min_episodes"])
    max_and = overrides.pop("max_and_components", spec["max_and_components"])
    # Left UNSET unless the caller asks for one: the schema is a session
    # fact, not a profile choice, and the resolver propagates it to every
    # module instead of the preset setting M1's copy alone (F10).
    timestamp_col = overrides.pop("timestamp_col", UNSET)

    disc_cfg = DiscoveryConfig(
        gate_params=GateParams(
            min_tpm=min_tpm,
            max_dispersion=max_dispersion,
            dispersion_margin=dispersion_margin,
            min_episodes=min_episodes,
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
    # senza creare incoerenza.
    #
    # M3's rate follows M1's at *this preset's* ratio, not at a flat default
    # (#200).  Overriding `min_tpm` alone used to move M1 and leave M3 on the
    # spec value, which silently mis-sized the walk-forward: `min_train_months`
    # is derived from `criteria.min_tpm` with a Poisson margin (#177), so on the
    # reference fixture `forge_preset("balanced", "1D", min_tpm=2)` kept a
    # 20-month training window built for 0.8 trades/month and left 9 months of
    # OOS span where 19 were available.
    #
    # The ratio is read from the spec's bar-mode pair — `daily_rd_min_tpm /
    # daily_min_tpm` — because that is a pure fill ratio (same unit on both
    # sides).  M3 has no notion of episodes (it opens a trade on every active
    # bar), so a *declared episode* rate is converted to the bar rate M3 will
    # actually see via `PipelineContext.bars_per_episode` before the fill
    # ratio is applied — applying the fill ratio directly to the episode rate
    # was #204: it understated M3's floor by the same factor, which inflated
    # `min_train_months` for no reason.
    #
    # The ratio is read from the spec rather than from the context's flat
    # `rate_retention` because the presets disagree about it — 1.00 on sniper
    # and sweep, 0.80 on balanced and burst — and flattening them would either
    # loosen two profiles or make the other two demand as many trades as
    # episodes, which `m3_stricter_than_m1` exists to prevent.  An explicit
    # `rd_min_tpm` still wins, and with neither override the value is
    # bit-for-bit what it was.
    _fill_ratio = spec["daily_rd_min_tpm"] / spec["daily_min_tpm"]
    _bars_per_episode = tf.ctx.bars_per_episode if event_counting == "episode" else 1.0
    rd_min_tpm = overrides.pop(
        "rd_min_tpm", round(min_tpm * _bars_per_episode * _fill_ratio, 4)
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
              f"max_dispersion(bar-mode only)={spec['daily_max_dispersion']}  "
              f"dispersion_margin(episode-mode only)={spec['dispersion_margin']}  "
              f"min_episodes={spec['min_episodes']}  "
              f"max_and={spec['max_and_components']}")
        from forgedge.resolver import poisson_min_window
        _is_window = poisson_min_window(spec["min_episodes"], spec["daily_min_tpm_episode"])
        print(f"            IS window needed @ 95% Poisson margin (1D, episode "
              f"mode, own rate): {_is_window:.1f} mo ({_is_window / 12:.2f} y)")
        print(f"  M2 alpha: min_lift={spec['min_lift']}  "
              f"cohens_d={spec['min_cohens_d']}  "
              f"fdr_q={spec['fdr_q']}  oos_max_p={spec['oos_max_p']}")
        # No `daily_rd_min_tpm_episode` key any more (#204) — the episode-mode
        # value is *computed*, the same way `forge_preset` computes it: M1's
        # episode rate converted to M3's bar rate via `bars_per_episode`, then
        # the spec's own bar-mode fill ratio.
        _fill_ratio = spec["daily_rd_min_tpm"] / spec["daily_min_tpm"]
        _rd_min_tpm_episode = round(
            spec["daily_min_tpm_episode"] * PipelineContext().bars_per_episode
            * _fill_ratio, 4,
        )
        print(f"  M3 gate : rd_min_tpm(episode)={_rd_min_tpm_episode}  "
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
