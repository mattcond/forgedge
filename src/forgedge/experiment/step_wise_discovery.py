"""StepWiseDiscovery — partition & compose search built on top of ``forge()``.

Where this fits: ``forge()`` (and its own ``two_pass_composition``) answers
"what single-condition and grade-paired events does this KPI table support,
end to end, in one pass?". ``StepWiseDiscovery`` answers a narrower, more
exploratory question instead: "starting from the strongest hold-out-confirmed
single-condition rules, can we grow each one, one AND-condition at a time,
by searching *only inside the sub-population it already selects* for a
second dimension?" — a step-wise, per-seed local search, as opposed to
``forge()``'s single global pass over every candidate.

Algorithm, in order:

1. **Resolve** the caller's (or default) discovery/alpha/rule-discovery
   configs once, via :func:`forgedge.config_report`, against the FULL KPI
   table's *shape* only (length/span — never its values, so no look-ahead is
   introduced by resolving before the hold-out split. See
   :func:`forgedge.config_report`'s own docs: "kpi ... only its length and
   span are read"). This closes the "hourly-fallback" pitfall the
   ``forgedge`` skill documents for any standalone (non-``forge()``) use of
   these configs — every one of this module's own ``EventDiscovery``/
   ``AlphaDiscovery``/``RuleDiscovery`` calls below reuses these same
   resolved configs, never a fresh, partially-UNSET one.
2. **Split** an outer hold-out from the KPI table — ``holdout_df`` is never
   read by any discovery call below, only by :meth:`_confirms_on_holdout`,
   which re-evaluates a candidate on ``search_df + holdout_df`` concatenated.
   This is a *stricter*, additional control on top of whatever internal
   train/test split the passed-in configs already use — see
   ``StepWiseDiscoveryConfig``'s docstring for how its width is sized.
3. **Iteration 1** — a single ``forge()`` call on ``search_df`` alone
   (composition forced off; that is this module's own job, not M1/M2's),
   giving every single-condition candidate/contract/verdict this KPI table
   supports.
4. **Seed selection** — rank iteration 1's results, and accept up to
   ``config.n_seeds`` of the best, requiring for each: a *distinct* feature
   family (:func:`~forgedge.experiment.redundancy.family_key`), hold-out
   confirmation (:meth:`_confirms_on_holdout`), AND diversity from every
   seed already accepted (:mod:`~forgedge.experiment.redundancy` — this is
   what stops two seeds that are really the same underlying signal under
   different names, e.g. an ATR-ratio and its NATR analogue, from each
   spending a full independent search chain on the same information).
5. **Partition & compose**, independently per seed, up to ``config.depth``:
   partition ``search_df`` to the seed's (or the current chain's) active
   rows, search that partition for a second condition among BOTH pointwise
   children (built directly on the partition — safe, since they don't
   depend on row contiguity) and rolling-transform children (pctrank/
   zscore/delta — recovered via :meth:`_rolling_children_on_partition`,
   which always computes the transform on the FULL ``search_df`` and uses
   the partition only to calibrate the local threshold, avoiding a
   look-ahead bug a naive "compute on the partition" approach would
   introduce for a non-contiguous partition), reject anything redundant
   with a component already in the chain (name, Jaccard, or continuous
   correlation), retry every hold-out-confirmed child in ranked order until
   one composes into an Alpha-Discovery-derivable target, and only commit
   the composed state once it *itself* re-confirms on the hold-out — a
   composition attempt that fails hold-out confirmation never overwrites
   the chain's last genuinely confirmed state (seed alone, or a shallower
   successful composition).

None of this is asset- or timeframe-specific: fee/mfe_floor calibration,
which indicator periods to build into the KPI table, and how much outer
embargo to reserve are all caller decisions (via
``event_discovery_config``/``alpha_config``/``rule_discovery_config`` and
``StepWiseDiscoveryConfig.outer_horizon_bars``/``outer_embargo_bars``) — this
class only owns the search algorithm.
"""
from __future__ import annotations

import dataclasses
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..alpha_discovery import AlphaConfig, AlphaContract, AlphaDiscovery
from ..config_report import config_report
from ..event_discovery import (
    ActivationStats,
    DiscoveryConfig,
    EventCandidate,
    EventComponent,
    EventDiscovery,
    GateParams,
    GateResult,
)
from ..event_discovery.classifier import TypeClassifier
from ..event_discovery.consistency_gate import ConsistencyGate, _build_month_index
from ..event_discovery.event_generator import (
    EventGenerator,
    _apply_threshold,
    _build_event_formula,
    _build_sql_expression,
    _make_expr,
)
from ..event_discovery.feature_generator import FeatureGenerator
from ..event_discovery.models import _apply_component
from ..event_discovery.transform_layer import TransformLayer
from ..forge import forge
from ..rule_discovery import RuleDiscovery, RuleDiscoveryConfig, RuleDiscoveryResponse
from ..timebudget import TimeBudget
from .models import ChainResult, SeedAttempt, StepWiseDiscoveryConfig, StepWiseDiscoveryResult
from .redundancy import family_key, is_redundant

__all__ = ["StepWiseDiscovery"]

POINTWISE_TRANSFORMS = frozenset({"identity", "binary_native", "categorical_onehot"})


def is_pointwise(candidate: EventCandidate) -> bool:
    """Whether every component of ``candidate`` is safe to build directly on a partition.

    Rolling transforms (pctrank/zscore/delta) are excluded here on purpose —
    computed on a *partition* they would silently read "N bars ago" as
    "N ACTIVE rows ago" (wrong on a non-contiguous partition). They are not
    lost: :meth:`StepWiseDiscovery._rolling_children_on_partition` recovers
    them the correct way (transform on the full frame, threshold on the
    partition). Lag features (``"_lag"`` in the source feature name) are
    excluded and, unlike rolling transforms, NOT yet recovered by an
    equivalent fix — they carry the same partition-discontinuity risk but
    this module only closes that gap for rolling transforms so far. This is
    a known, documented limitation, not an oversight.
    """
    for c in candidate.components:
        if c.transform not in POINTWISE_TRANSFORMS:
            return False
        if c.event_type != "threshold":
            return False
        if "_lag" in c.source_feature:
            return False
    return True


def _rank_key(item: Tuple[AlphaContract, RuleDiscoveryResponse]):
    """(contract, response) sort key: EDGE/PARTIAL-EDGE first, then composite_score desc."""
    contract, resp = item
    verdict_rank = 0 if resp.verdict in ("EDGE", "PARTIAL-EDGE") else 1
    return (verdict_rank, -contract.alpha_score.composite_score)


class StepWiseDiscovery:
    """Run the partition & compose search described in the module docstring.

    Parameters
    ----------
    kpi : pd.DataFrame
        The full KPI Table (indicators already built — see
        ``forgedge.build_features``/``candle_features``). Never mutated.
    ticker : str, optional
        Forwarded to ``forge()`` and used in every Alpha/Rule Discovery call
        this class makes directly, so contract/alpha IDs are consistent
        across iteration 1 and every chain.
    asset : str
        Traceability fallback used only when neither ``ticker`` nor an
        explicit ``alpha_config`` sets one — same semantics as ``forge()``.
    timeframe : str
        Bar size, e.g. ``"1D"``, ``"1H"`` — see ``forge()``'s own docs for
        why declaring this matters even though it has a fallback.
    event_discovery_config, alpha_config, rule_discovery_config : optional
        Passed straight through to the resolver (:func:`forgedge.config_report`)
        exactly as they would be to ``forge()``. Any asset-specific
        calibration (fee, mfe_floor, gate thresholds, ...) belongs here,
        already applied by the caller — this class treats them as given and
        never second-guesses them. ``event_discovery_config.max_and_components``
        is forced to ``1`` regardless of what's passed: AND-composition is
        this class's own job.
    config : StepWiseDiscoveryConfig, optional
        Search-algorithm parameters (seeds, depth, redundancy thresholds, ...).
        Defaults to ``StepWiseDiscoveryConfig()``.
    run_market_context : bool
        Forwarded to the internal ``forge()`` call for iteration 1. Default
        ``True``, matching ``forge()``'s own default.
    """

    def __init__(
        self,
        kpi: pd.DataFrame,
        *,
        ticker: Optional[str] = None,
        asset: str = "ASSET",
        timeframe: str,
        event_discovery_config: Optional[DiscoveryConfig] = None,
        alpha_config: Optional[AlphaConfig] = None,
        rule_discovery_config: Optional[RuleDiscoveryConfig] = None,
        config: Optional[StepWiseDiscoveryConfig] = None,
        run_market_context: bool = True,
    ):
        self.kpi = kpi
        self.ticker = ticker
        self.asset = asset
        self.timeframe = timeframe
        self._event_discovery_config = event_discovery_config
        self._alpha_config = alpha_config
        self._rule_discovery_config = rule_discovery_config
        self.config = config or StepWiseDiscoveryConfig()
        self.run_market_context = run_market_context
        self._log: List[str] = []

    def log(self, msg: str) -> None:
        self._log.append(str(msg))

    @property
    def log_lines(self) -> List[str]:
        return list(self._log)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> StepWiseDiscoveryResult:
        self._resolve()
        self._split_holdout()
        result1 = self._run_iteration1()
        self._precompute_transforms()
        seeds, seed_attempts = self._select_seeds(result1)

        chains: List[ChainResult] = []
        feature_recurrence: Dict[str, List[str]] = {}
        for seed_idx, seed in enumerate(seeds):
            chains.append(self._run_chain(seed_idx, seed, feature_recurrence))

        return StepWiseDiscoveryResult(
            chains=chains,
            seed_attempts=seed_attempts,
            feature_recurrence=feature_recurrence,
            search_df=self.search_df,
            holdout_df=self.holdout_df,
            time_budget=self.time_budget,
            coherence=self.coherence,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Step 1 — resolve configs once, against the full KPI table's shape
    # ------------------------------------------------------------------

    def _resolve(self) -> None:
        disc_cfg = self._event_discovery_config or DiscoveryConfig()
        alpha_cfg = self._alpha_config or AlphaConfig(asset=self.ticker or self.asset, timeframe=self.timeframe)
        rd_cfg = self._rule_discovery_config or RuleDiscoveryConfig()
        disc_cfg = dataclasses.replace(disc_cfg, max_and_components=1)

        rep = config_report(disc_cfg, alpha_cfg, rd_cfg, kpi=self.kpi, timeframe=self.timeframe)
        if self.config.strict and rep.has_critical:
            raise ValueError(rep.one_line())
        self.coherence = rep
        self.context = rep.context
        self.disc_cfg = rep.configs["event_discovery"]
        self.alpha_cfg = rep.configs["alpha"]
        self.rd_cfg = rep.configs["rule_discovery"]

    # ------------------------------------------------------------------
    # Step 2 — outer hold-out, never touched during search
    # ------------------------------------------------------------------

    def _split_holdout(self) -> None:
        horizon_bars = self.config.outer_horizon_bars
        if horizon_bars is None:
            horizon_bars = max(self.alpha_cfg.horizon_grid) if self.alpha_cfg.horizon_grid else 0
        embargo_bars = self.config.outer_embargo_bars
        if embargo_bars is None:
            embargo_bars = self.alpha_cfg.embargo_bars or 0

        n = len(self.kpi)
        budget = TimeBudget.build(
            n_bars=n, train_ratio=self.config.train_ratio,
            horizon_bars=horizon_bars, embargo_bars=embargo_bars,
        )
        self.time_budget = budget
        self.search_df = self.kpi.iloc[:budget.split].reset_index(drop=True)
        self.holdout_df = self.kpi.iloc[budget.oos_start:].reset_index(drop=True)
        ts_col = self.context.timestamp_col
        self.holdout_start_ts = (
            pd.Timestamp(self.holdout_df[ts_col].min()) if len(self.holdout_df) else None
        )
        self.log(
            f"TimeBudget: n={n} split={budget.split} oos_start={budget.oos_start} "
            f"purge_bars={budget.purge_bars} embargo_bars={budget.embargo_bars}"
        )

    # ------------------------------------------------------------------
    # Step 3 — iteration 1, via forge(), no composition
    # ------------------------------------------------------------------

    def _run_iteration1(self):
        result1 = forge(
            self.search_df,
            ticker=self.ticker,
            asset=self.asset,
            timeframe=self.timeframe,
            event_discovery_config=self.disc_cfg,
            alpha_config=self.alpha_cfg,
            rule_discovery_config=self.rd_cfg,
            two_pass_composition=False,
            run_market_context=self.run_market_context,
            run_rule_discovery=True,
            run_registry=False,
            strict=self.config.strict,
            progress=False,
        )
        self._candidates_by_id = {c.event_id: c for c in result1.candidates}
        self.log(
            f"Iterazione 1: {len(result1.candidates)} candidati M1, "
            f"{len(result1.contracts)} contratti M2 ({len(result1.promoted)} promossi), "
            f"{len(result1.rule_responses)} risposte M3."
        )
        return result1

    # ------------------------------------------------------------------
    # Precompute FeatureGenerator+TransformLayer ONCE on search_df, so
    # rolling-transform children can be recovered (see module docstring
    # point 5) and so continuous-correlation redundancy checks (both for
    # seed selection and within a chain) have a series to compare against.
    # ------------------------------------------------------------------

    def _precompute_transforms(self) -> None:
        classifier = TypeClassifier(
            max_categorical_classes=self.disc_cfg.max_categorical_classes,
            scale_free_overrides=self.disc_cfg.scale_free_overrides or {},
        )
        classifications = classifier.fit(self.search_df)
        extended_df, derived_meta = FeatureGenerator().generate(
            self.search_df, classifications,
            indicator_lag_cross_lags=self.disc_cfg.indicator_lag_cross_lags,
        )
        extended_df.index = self.search_df.index
        transformed_all = TransformLayer().transform_all(extended_df, derived_meta)
        self._rolling_transformed = [ts for ts in transformed_all if ts.transform != "identity"]
        self._series_by_col = {
            ts.col: (ts.series.values if hasattr(ts.series, "values") else ts.series)
            for ts in transformed_all
        }
        self._event_generator = EventGenerator()

        child_gate = self.config.child_gate or GateParams(
            min_tpm=0.25, dispersion_margin=3.0, min_episodes=4,
            event_counting="episode", episode_gap=1,
        )
        self.child_gate = child_gate
        self.child_disc_cfg = dataclasses.replace(
            self.disc_cfg, gate_params=child_gate, max_and_components=1, train_ratio=1.0,
        )
        self._min_trades_m3 = self.rd_cfg.criteria.min_oos_trades
        self.log(
            f"Transform layer precomputata su search_df: {len(transformed_all)} serie totali, "
            f"{len(self._rolling_transformed)} rolling disponibili per la ricerca locale."
        )

    # ------------------------------------------------------------------
    # Step 4 — seed selection: ranked, distinct family, hold-out confirmed,
    # diverse from every seed already accepted.
    # ------------------------------------------------------------------

    def _select_seeds(self, result1) -> Tuple[List[dict], List[SeedAttempt]]:
        ranked = sorted(result1.rule_responses, key=_rank_key)
        seeds: List[dict] = []
        seen_families = set()
        seed_bool_series: List[np.ndarray] = []
        seed_transformed_cols: List[str] = []
        attempts: List[SeedAttempt] = []

        for contract, resp in ranked:
            if len(seeds) >= self.config.n_seeds:
                break
            cand = self._candidates_by_id.get(contract.event_candidate_id)
            if cand is None:
                continue
            fam = family_key(cand)
            if fam in seen_families:
                continue

            cand_series = cand.apply(self.search_df).fillna(0).astype(bool).values
            redundant, j, corr = is_redundant(
                bool_series=cand_series, used_bool_series=seed_bool_series,
                components=cand.components, used_cols=seed_transformed_cols,
                series_by_col=self._series_by_col,
                max_jaccard_threshold=self.config.max_constituent_jaccard,
                max_abs_corr_threshold=self.config.max_constituent_abs_corr,
            )
            if redundant:
                reason = f"ridondante con un seme gia' accettato (jaccard={j:.3f}, corr_continua={corr:.3f})"
                attempts.append(SeedAttempt(fam, contract.alpha_id, resp.verdict,
                                             contract.alpha_score.composite_score, False, reason, j, corr))
                self.log(f"seme scartato (ridondanza inter-seme): famiglia={fam} alpha_id={contract.alpha_id} -> {reason}")
                continue

            ok, _, reason = self._confirms_on_holdout(contract, cand)
            attempts.append(SeedAttempt(fam, contract.alpha_id, resp.verdict,
                                         contract.alpha_score.composite_score, ok, reason, j, corr))
            if ok:
                seen_families.add(fam)
                seed_bool_series.append(cand_series)
                seed_transformed_cols += [c.transformed_col for c in cand.components]
                seeds.append({"family": fam, "contract": contract, "candidate": cand,
                              "resp": resp, "confirm_reason": reason})
                self.log(f"SEME accettato: famiglia={fam} alpha_id={contract.alpha_id} conferma_holdout={reason}")
            else:
                self.log(f"seme scartato (non confermato su hold-out): famiglia={fam} alpha_id={contract.alpha_id} -> {reason}")

        self.log(f"Semi trovati: {len(seeds)}/{self.config.n_seeds} richiesti, "
                 f"su {len(attempts)} candidati esaminati ({len(ranked)} disponibili in totale).")
        return seeds, attempts

    # ------------------------------------------------------------------
    # Hold-out confirmation: re-evaluate on search_df + holdout_df.
    # ------------------------------------------------------------------

    def _confirms_on_holdout(self, contract, candidate) -> Tuple[bool, Optional[RuleDiscoveryResponse], str]:
        if len(self.holdout_df) == 0:
            return False, None, "holdout_df vuoto (train_ratio=1.0?) — nessuna conferma possibile"
        ts_col = self.context.timestamp_col
        eval_df = pd.concat([self.search_df, self.holdout_df]).drop_duplicates(ts_col).reset_index(drop=True)
        try:
            resp = RuleDiscovery(eval_df, contract, candidate, config=self.rd_cfg,
                                  time_budget=self.time_budget).run()
        except Exception as exc:
            return False, None, f"RuleDiscovery su eval_df fallita: {exc!r}"

        if resp.verdict in ("EDGE", "PARTIAL-EDGE"):
            return True, resp, f"verdetto={resp.verdict} su eval_df (search+holdout)"

        if resp.walk_forward is not None and resp.walk_forward.splits and self.holdout_start_ts is not None:
            holdout_splits = [
                s for s in resp.walk_forward.splits
                if pd.Timestamp(s.test_to) > self.holdout_start_ts and s.test_summary.total_trades > 0
            ]
            if holdout_splits:
                exp_vals = [s.test_summary.expectancy for s in holdout_splits
                            if np.isfinite(s.test_summary.expectancy)]
                if exp_vals:
                    mean_exp = float(np.mean(exp_vals))
                    if mean_exp > 0:
                        return True, resp, (f"verdetto={resp.verdict} ma expectancy media OOS "
                                             f"sui fold post-holdout={mean_exp:.5f} > 0 ({len(holdout_splits)} fold)")
                    return False, resp, (f"verdetto={resp.verdict}, expectancy media OOS "
                                          f"post-holdout={mean_exp:.5f} <= 0")
        return False, resp, f"verdetto={resp.verdict}, nessuna evidenza positiva sull'hold-out"

    def _run_rule_discovery_safe(self, contract, candidate):
        try:
            resp = RuleDiscovery(self.search_df, contract, candidate, config=self.rd_cfg,
                                  time_budget=self.time_budget).run()
            return resp, None
        except Exception as exc:
            return None, f"RuleDiscovery fallita: {exc!r}"

    def _build_candidate_stats(self, candidate: EventCandidate, gate_params: GateParams) -> EventCandidate:
        """Populate activation_stats/consistency_gate by replaying on search_df (never a partition)."""
        series = candidate.apply(self.search_df)
        timestamps = pd.to_datetime(self.search_df[self.context.timestamp_col])
        month_index, n_total_months = _build_month_index(timestamps)
        gate_result = ConsistencyGate(gate_params).evaluate_series(series, month_index, n_total_months)
        zero_months = n_total_months - gate_result.n_active_months
        stats = ActivationStats(
            n_activations=gate_result.n_activations,
            n_active_months=gate_result.n_active_months,
            zero_months=zero_months,
            max_monthly_share=gate_result.max_monthly_share,
            mean_tpm=gate_result.mean_tpm,
            index_of_dispersion=gate_result.index_of_dispersion,
            n_episodes=gate_result.n_episodes,
            episode_index_of_dispersion=gate_result.episode_index_of_dispersion,
            n_eff=gate_result.n_eff,
        )
        candidate.event_series = series
        candidate.activation_stats = stats
        candidate.consistency_gate = gate_result
        return candidate

    def _retain_ratio_min_for(self, n_rows: int) -> float:
        """max(floor, min_trades_M3 / n_rows) for the CURRENT partition size."""
        if n_rows <= 0:
            return float("inf")
        return max(self.config.retain_ratio_floor, self._min_trades_m3 / n_rows)

    # ------------------------------------------------------------------
    # Rolling-transform children — see module docstring point 5.
    # ------------------------------------------------------------------

    def _rolling_children_on_partition(
        self, mask: np.ndarray, n_active: int, retain_ratio_min_k: float,
        gate_params: GateParams, exclude_source_features: frozenset = frozenset(),
    ) -> List[EventCandidate]:
        survivors: List[EventCandidate] = []
        ts_col = self.context.timestamp_col
        for ts in self._rolling_transformed:
            if ts.source_feature in exclude_source_features:
                continue
            local_series = ts.series[mask].dropna()
            if len(local_series) < self.config.min_local_partition_rows:
                continue
            thresholds = self._event_generator._compute_thresholds(local_series, ts.is_zscore)
            merged_params = {**ts.transform_params, **ts.feature_params}
            for threshold, t_type, direction in thresholds:
                bool_series = _apply_threshold(ts.series, threshold, direction)
                local_active = bool_series[mask]
                retained = float(local_active.sum()) / n_active if n_active else 0.0
                if retained < retain_ratio_min_k:
                    continue
                comp = EventComponent(
                    source_feature=ts.source_feature, transform=ts.transform,
                    transform_params=merged_params, transformed_col=ts.col,
                    threshold=threshold, threshold_type=t_type, direction=direction,
                    event_type="threshold", expression=_make_expr(ts.col, direction, threshold),
                    source_cols=ts.source_cols,
                    event_formula=_build_event_formula(
                        ts.source_feature, ts.source_cols, ts.transform, merged_params,
                        threshold, direction, "threshold", threshold_type=t_type),
                    sql_expression=_build_sql_expression(
                        ts.source_feature, ts.source_cols, ts.transform, merged_params,
                        threshold, direction, "threshold", ts_col=ts_col),
                )
                cand = EventCandidate(
                    event_id=f"PART-{ts.col}-{t_type}-{direction}-{len(survivors)}",
                    status="CANDIDATE", components=[comp], expression=comp.expression,
                    activation_stats=ActivationStats(0, 0, 0, float("nan"), float("nan")),
                    consistency_gate=GateResult(False, 0, 0, float("nan"), float("nan")),
                    gate_params=gate_params,
                )
                cand = self._build_candidate_stats(cand, gate_params)
                if cand.consistency_gate.passed:
                    survivors.append(cand)
        return survivors

    # ------------------------------------------------------------------
    # Step 5 — one seed's partition & compose chain.
    # ------------------------------------------------------------------

    def _run_chain(self, seed_idx: int, seed: dict, feature_recurrence: Dict[str, List[str]]) -> ChainResult:
        chain_label = f"seed{seed_idx + 1}:{seed['family']}"
        self.log(f"\n=== Chain {chain_label} — depth budget={self.config.depth} ===")
        current_cand = seed["candidate"]
        current_contract = seed["contract"]
        current_resp = seed["resp"]
        current_confirm_reason = seed["confirm_reason"]
        stop_reason = ""
        depth_reached = 0

        used_source_features = {c.source_feature for c in current_cand.components}
        used_component_series = [
            _apply_component(c, self.search_df).fillna(0).astype(bool).values
            for c in current_cand.components
        ]
        used_transformed_cols = [c.transformed_col for c in current_cand.components]

        for depth_i in range(self.config.depth):
            mask = current_cand.apply(self.search_df).fillna(0).astype(bool).values
            n_active = int(mask.sum())
            retain_ratio_min_k = self._retain_ratio_min_for(n_active)
            self.log(f"[{chain_label} depth={depth_i}] retain_ratio_min_k={retain_ratio_min_k:.3f} "
                     f"(partizione_corrente={n_active} barre)")
            if retain_ratio_min_k >= 1.0:
                stop_reason = (f"partizione troppo piccola ({n_active} barre) per il floor statistico M3 "
                               f"alla profondita {depth_i}")
                break
            partition_df = self.search_df[mask].reset_index(drop=True)

            try:
                child_ed = EventDiscovery(partition_df, config=self.child_disc_cfg)
                child_candidates_raw = child_ed.run()
            except Exception as exc:
                stop_reason = f"EventDiscovery sulla partizione fallita alla profondita {depth_i}: {exc!r}"
                break

            pointwise_children_all = [
                c for c in child_candidates_raw
                if is_pointwise(c) and c.components[0].source_feature not in used_source_features
            ]
            pointwise_children = [
                c for c in pointwise_children_all
                if (c.activation_stats.n_activations / n_active) >= retain_ratio_min_k
            ]
            rolling_children = self._rolling_children_on_partition(
                mask, n_active, retain_ratio_min_k, self.child_gate,
                exclude_source_features=used_source_features,
            )
            pre_redundancy = pointwise_children + rolling_children

            survivors = []
            n_jaccard_rejected = 0
            n_corr_rejected = 0
            for c in pre_redundancy:
                c_series = c.apply(self.search_df).fillna(0).astype(bool).values
                redundant, j, corr = is_redundant(
                    bool_series=c_series, used_bool_series=used_component_series,
                    components=c.components, used_cols=used_transformed_cols,
                    series_by_col=self._series_by_col,
                    max_jaccard_threshold=self.config.max_constituent_jaccard,
                    max_abs_corr_threshold=self.config.max_constituent_abs_corr,
                )
                if redundant:
                    if j >= self.config.max_constituent_jaccard:
                        n_jaccard_rejected += 1
                    else:
                        n_corr_rejected += 1
                    continue
                survivors.append(c)
            self.log(f"[{chain_label} depth={depth_i}] {len(pre_redundancy)} candidati -> "
                     f"{n_jaccard_rejected} scartati per Jaccard, {n_corr_rejected} scartati per "
                     f"correlazione continua -> {len(survivors)} residui")

            if not survivors:
                stop_reason = (f"nessun candidato non ridondante soddisfa retain_ratio_min_k="
                               f"{retain_ratio_min_k:.3f} alla profondita {depth_i}")
                break

            for c in survivors:
                comp = c.components[0]
                base = re.sub(r"_?\d+", "", comp.source_feature)
                feature_recurrence.setdefault(base, []).append(f"{chain_label}@d{depth_i}")
                c.event_series = None

            try:
                child_ad = AlphaDiscovery(self.search_df, survivors, self.alpha_cfg)
                child_contracts = child_ad.run()
            except Exception as exc:
                stop_reason = f"AlphaDiscovery dei figli fallita alla profondita {depth_i}: {exc!r}"
                break

            child_by_event_id = {c.event_id: c for c in survivors}
            child_promoted = [ct for ct in child_contracts if ct.status != "REJECTED"]
            if not child_promoted:
                stop_reason = f"nessun figlio promosso da M2 alla profondita {depth_i}"
                break

            child_responses = []
            for ct in child_promoted:
                cand_c = child_by_event_id.get(ct.event_candidate_id)
                if cand_c is None:
                    continue
                resp_c, err = self._run_rule_discovery_safe(ct, cand_c)
                if resp_c is None:
                    continue
                child_responses.append((ct, resp_c))
            if not child_responses:
                stop_reason = f"RuleDiscovery non ha prodotto risposte valide alla profondita {depth_i}"
                break

            child_ranked = sorted(child_responses, key=_rank_key)
            confirmed_children = []
            for ct, resp_c in child_ranked:
                cand_c = child_by_event_id[ct.event_candidate_id]
                ok, _, reason = self._confirms_on_holdout(ct, cand_c)
                if ok:
                    confirmed_children.append((ct, resp_c, cand_c, reason))
            if not confirmed_children:
                stop_reason = f"nessun figlio ha superato la conferma hold-out alla profondita {depth_i}"
                break

            composed = None
            composed_contract = None
            chosen_reason = None
            best_child_cand = None
            compose_attempts = []
            for cand_child, resp_child, cand_c, reason in confirmed_children:
                candidate_composed = EventCandidate(
                    event_id=f"{current_cand.event_id}+{cand_c.event_id}",
                    status="CANDIDATE",
                    components=list(current_cand.components) + list(cand_c.components),
                    expression=f"({current_cand.expression}) AND ({cand_c.expression})",
                    activation_stats=ActivationStats(0, 0, 0, float("nan"), float("nan")),
                    consistency_gate=GateResult(False, 0, 0, float("nan"), float("nan")),
                    gate_params=self.disc_cfg.gate_params,
                )
                candidate_composed = self._build_candidate_stats(candidate_composed, self.disc_cfg.gate_params)
                n_act = candidate_composed.activation_stats.n_activations
                if n_act < self.config.min_composed_activations:
                    compose_attempts.append((cand_child.alpha_id, f"solo {n_act} attivazioni"))
                    continue
                try:
                    composed_ad = AlphaDiscovery(self.search_df, [candidate_composed], self.alpha_cfg)
                    composed_contracts = composed_ad.run()
                except Exception as exc:
                    compose_attempts.append((cand_child.alpha_id, f"AlphaDiscovery fallita: {exc!r}"))
                    continue
                cc = composed_contracts[0] if composed_contracts else None
                if cc is None or cc.status == "REJECTED":
                    compose_attempts.append((cand_child.alpha_id, "M2 no derivable target"))
                    continue
                composed, composed_contract, chosen_reason, best_child_cand = (
                    candidate_composed, cc, reason, cand_c,
                )
                break

            if composed is None:
                detail = "; ".join(f"{aid}: {why}" for aid, why in compose_attempts[:5])
                stop_reason = (f"nessuna delle {len(confirmed_children)} composizioni confermate produce "
                               f"un target M2 derivabile alla profondita {depth_i} — {detail}")
                break

            composed_resp, err = self._run_rule_discovery_safe(composed_contract, composed)
            if composed_resp is None:
                stop_reason = f"RuleDiscovery sulla regola composta fallita alla profondita {depth_i}: {err}"
                break

            ok, _, reason = self._confirms_on_holdout(composed_contract, composed)
            self.log(f"[{chain_label} depth={depth_i}] regola composta: {composed.expression} "
                     f"verdetto_search={composed_resp.verdict} conferma_holdout({ok})={reason}")
            if not ok:
                # Do NOT commit the failed composition — the chain stops but
                # reports the last CONFIRMED state (seed, or a shallower
                # successful composition), never a disconfirmed attempt.
                stop_reason = f"regola composta non confermata sull'hold-out alla profondita {depth_i}: {reason}"
                break

            current_cand, current_contract, current_resp = composed, composed_contract, composed_resp
            current_confirm_reason = reason
            depth_reached = depth_i + 1
            used_source_features |= {c.source_feature for c in best_child_cand.components}
            used_component_series.append(best_child_cand.apply(self.search_df).fillna(0).astype(bool).values)
            used_transformed_cols += [c.transformed_col for c in best_child_cand.components]

        return ChainResult(
            chain_label=chain_label,
            seed_family=seed["family"],
            candidate=current_cand,
            contract=current_contract,
            response=current_resp,
            depth_reached=depth_reached,
            confirm_reason=current_confirm_reason,
            stop_reason=stop_reason or "profondita massima raggiunta",
        )
