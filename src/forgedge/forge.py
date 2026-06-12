"""``forge`` — the end-to-end FORGE pipeline orchestrator.

A single entry point that chains the four FORGE modules — Market Context
(Modulo 0), Event Discovery (Modulo 1), Alpha Discovery (Modulo 2) and Rule
Discovery (Modulo 3) — into one call that turns a raw KPI Table into validated
trading rules.

Usage
-----
    from forgedge import forge

    kpi = pd.read_parquet("btc_1h.parquet")
    result = forge(kpi, asset="BTC", timeframe="1H")

    print(result.summary())                 # one row per evaluated candidate
    for contract, response in result.edges():
        print(contract.alpha_id, response.verdict)

Every stage accepts the configuration object of its module, so the whole
pipeline is tunable through one call::

    from forgedge import (
        forge, MarketContextConfig, EMAProxyConfig,
        DiscoveryConfig, AlphaConfig, RuleDiscoveryConfig,
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
            gate_params=GateParams(min_act=50, min_months=8, max_conc=0.40, min_tpm=2.0),
        ),
        alpha_config=AlphaConfig(
            train_ratio=0.70,
            thresholds=PromotionThresholds(min_lift=0.08, min_cohens_d=0.15),
        ),
        rule_discovery_config=RuleDiscoveryConfig(),
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .alpha_discovery.discovery import AlphaDiscovery
from .alpha_discovery.models import AlphaConfig, AlphaContract
from .event_discovery.discovery import DiscoveryConfig, EventDiscovery
from .event_discovery.models import EventCandidate
from .market_context.context import MarketContext
from .market_context.models import REGIME_COL, MarketContextConfig
from .rule_discovery.discovery import RuleDiscovery
from .rule_discovery.models import RuleDiscoveryConfig, RuleDiscoveryResponse


@dataclass
class ForgeResult:
    """The complete output of a :func:`forge` run.

    Holds every artefact produced along the pipeline so callers can both read
    the final verdicts *and* drill back into any intermediate stage for audit.

    Attributes
    ----------
    enriched : pd.DataFrame
        KPI Table after Market Context — the original columns plus ``regime``
        and ``regime_stable``.  When Market Context is skipped this is the
        table actually fed to Event Discovery.
    candidates : list[EventCandidate]
        Every Event Candidate that passed the Consistency Gate (Modulo 1).
    contracts : list[AlphaContract]
        Every evaluated Alpha Contract (promoted *and* rejected), so rejections
        can be audited via ``contract.rejection_reasons``.
    promoted : list[AlphaContract]
        The subset of ``contracts`` that cleared every promotion gate
        (``status == "HYPOTHESIS"``) and were handed to Rule Discovery.
    rule_responses : list[tuple[AlphaContract, RuleDiscoveryResponse]]
        One ``(contract, response)`` pair per promoted contract, in promotion
        order.  The verdict lives on ``response.verdict``.
    market_context, event_discovery, alpha_discovery : module instances
        The live module objects, for introspection — e.g.
        ``market_context.distribution()``, ``event_discovery.summary()``,
        ``alpha_discovery.summary()``.  ``market_context`` is ``None`` when the
        stage was skipped.
    """

    enriched: pd.DataFrame
    candidates: List[EventCandidate]
    contracts: List[AlphaContract]
    promoted: List[AlphaContract]
    rule_responses: List[Tuple[AlphaContract, RuleDiscoveryResponse]]
    market_context: Optional[MarketContext] = None
    event_discovery: Optional[EventDiscovery] = None
    alpha_discovery: Optional[AlphaDiscovery] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def edges(self) -> List[Tuple[AlphaContract, RuleDiscoveryResponse]]:
        """Return ``(contract, response)`` pairs whose verdict is tradeable.

        Tradeable means ``EDGE`` or ``PARTIAL-EDGE`` (``response.is_edge``).
        """
        return [(c, r) for c, r in self.rule_responses if r.is_edge]

    def validated_rules(self) -> List[RuleDiscoveryResponse]:
        """Return the responses that carry a non-null ``validated_rule``."""
        return [r for _, r in self.rule_responses if r.validated_rule is not None]

    def summary(self) -> pd.DataFrame:
        """Return Alpha Discovery's flat candidate summary, with the rule verdict.

        One row per evaluated candidate (sorted by composite score), augmented
        with a ``rule_verdict`` column for the promoted contracts that reached
        Rule Discovery.  Returns an empty frame when no candidates were
        evaluated.
        """
        if self.alpha_discovery is None:
            return pd.DataFrame()
        df = self.alpha_discovery.summary()
        if df.empty:
            return df
        verdicts = {c.alpha_id: r.verdict for c, r in self.rule_responses}
        df = df.copy()
        df["rule_verdict"] = df["alpha_id"].map(verdicts)
        return df


def forge(
    kpi_table: pd.DataFrame,
    *,
    asset: str = "ASSET",
    timeframe: str = "1H",
    market_context_config: Optional[MarketContextConfig] = None,
    event_discovery_config: Optional[DiscoveryConfig] = None,
    alpha_config: Optional[AlphaConfig] = None,
    rule_discovery_config: Optional[RuleDiscoveryConfig] = None,
    run_market_context: bool = True,
    run_rule_discovery: bool = True,
    only_validated_events: bool = False,
) -> ForgeResult:
    """Run the full FORGE rule-extraction pipeline end to end.

    Chains the four modules and returns every artefact in a :class:`ForgeResult`:

    0. **Market Context** classifies each bar by regime and appends ``regime`` /
       ``regime_stable`` (skipped when ``run_market_context=False`` or the
       table already carries a ``regime`` column).
    1. **Event Discovery** mines structurally-consistent boolean events.
    2. **Alpha Discovery** derives a target per event, measures its edge and
       promotes the candidates that confirm out-of-sample.
    3. **Rule Discovery** backtests each promoted contract under realistic order
       mechanics and emits an ``EDGE`` / ``PARTIAL-EDGE`` / ``NON-EDGE`` verdict.

    Parameters
    ----------
    kpi_table : pd.DataFrame
        Raw KPI Table — a ``close`` column plus any technical features, and a
        datetime source (``open_dt`` column or a DatetimeIndex).
    asset, timeframe : str
        Traceability metadata for the Alpha Contracts.  Used only when
        ``alpha_config`` does not already set them (an explicit ``alpha_config``
        keeps its own ``asset``/``timeframe``).
    market_context_config : MarketContextConfig, optional
        Modulo 0 configuration.  Defaults to the production EMA-proxy settings.
    event_discovery_config : DiscoveryConfig, optional
        Modulo 1 configuration.  Defaults to a no-split, single-component run;
        for production pass ``train_ratio < 1`` with a ``walk_forward`` block.
    alpha_config : AlphaConfig, optional
        Modulo 2 configuration.  When omitted, a default is built carrying
        ``asset`` / ``timeframe``.
    rule_discovery_config : RuleDiscoveryConfig, optional
        Modulo 3 configuration.  Defaults to the standard grid and acceptance
        gates.
    run_market_context : bool, default True
        Run Modulo 0.  Set ``False`` to feed a table that already carries the
        ``regime`` columns (or to skip regime analysis entirely).
    run_rule_discovery : bool, default True
        Run Modulo 3.  Set ``False`` to stop after Alpha Discovery and obtain
        the promoted contracts (hypotheses) without backtesting them; the
        returned ``rule_responses`` is then empty.
    only_validated_events : bool, default False
        When Event Discovery ran with walk-forward validation, hand Alpha
        Discovery only the candidates with ``validation.passed == True``.  Has
        no effect when walk-forward was not configured.

    Returns
    -------
    ForgeResult
        All pipeline artefacts (see the class docstring).
    """
    # ── Modulo 0 — Market Context ─────────────────────────────────────────
    mc: Optional[MarketContext] = None
    already_enriched = REGIME_COL in kpi_table.columns
    if run_market_context and not already_enriched:
        mc = MarketContext(kpi_table, config=market_context_config)
        enriched = mc.run()
    else:
        enriched = kpi_table

    # ── Modulo 1 — Event Discovery ────────────────────────────────────────
    ed = EventDiscovery(enriched, config=event_discovery_config)
    candidates = ed.run()

    alpha_candidates = candidates
    if only_validated_events:
        validated = [
            c for c in candidates if c.validation is not None and c.validation.passed
        ]
        # Only narrow the set when validation actually ran; otherwise keep all
        # candidates so the pipeline does not silently drop everything.
        if any(c.validation is not None for c in candidates):
            alpha_candidates = validated

    # ── Modulo 2 — Alpha Discovery ────────────────────────────────────────
    cfg = alpha_config
    if cfg is None:
        cfg = AlphaConfig(asset=asset, timeframe=timeframe)
    ad = AlphaDiscovery(ed.df, alpha_candidates, cfg)
    contracts = ad.run()
    promoted = ad.promoted_contracts()

    # ── Modulo 3 — Rule Discovery ─────────────────────────────────────────
    by_id = {c.event_id: c for c in candidates}
    rule_responses: List[Tuple[AlphaContract, RuleDiscoveryResponse]] = []
    for contract in promoted if run_rule_discovery else []:
        cand = by_id.get(contract.event_candidate_id)
        if cand is None:
            continue
        rd = RuleDiscovery(ed.df, contract, cand, config=rule_discovery_config)
        response = rd.run()
        rule_responses.append((contract, response))

    return ForgeResult(
        enriched=enriched,
        candidates=candidates,
        contracts=contracts,
        promoted=promoted,
        rule_responses=rule_responses,
        market_context=mc,
        event_discovery=ed,
        alpha_discovery=ad,
    )
