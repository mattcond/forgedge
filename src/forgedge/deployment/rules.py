"""forgedge.deployment — putting discovered rules into production (#245).

Split out of ``forgedge.playground`` (see #245's discussion): unlike the
read-only analysis functions in ``forgedge.playground`` — built for
exploring *why* forge() decided what it decided — this module has real
effects: it gates which rules are allowed to go live and writes files to
disk. "Playground" stopped fitting once the module did more than look.

Three use cases, meant to run in sequence:

    forge() -> promotion_gate() [filter] -> export_rules() [write to disk]
            -> monitoring_manifest() [index the export for periodic re-checks]

``export_rules`` is the only function that performs filesystem I/O —
deliberately isolated so the rest of the sequence stays pure and easy to
test.

Usage::

    from forgedge.deployment import *
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from ..forge import ForgeResult
from ..rule_registry import RuleRegistry
from ..rule_report import RuleSpec
from .kpi_recipe import KpiRecipe, KpiRecipeVerification, minimal_kpi_recipe, verify_kpi_recipe

__all__ = [
    "PromotionGateConfig",
    "promotion_gate",
    "export_rules",
    "monitoring_manifest",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _contract_grade(contract) -> Optional[str]:
    score = getattr(contract, "alpha_score", None)
    if score is None or getattr(score, "grade", None) is None:
        return None
    return str(score.grade).strip().upper()


def _fold_stability_score(response, fold_pf_cap: float) -> Optional[float]:
    """Mean walk-forward-fold profit factor minus its std (#253).

    Per-fold ``test_summary.profit_factor`` is capped at ``fold_pf_cap``
    first to blunt the ``9999.0`` "zero losing trades" sentinel a single
    lucky fold can hit, which would otherwise dominate both the mean and the
    std. A pooled walk-forward PF can look strong purely because one such
    fold inflates the aggregate while the rest are mediocre or losing; this
    score penalises that dispersion directly instead of only looking at the
    pooled number.

    ``None`` when there's no walk-forward result or fewer than two splits —
    sample std (``ddof=1``) is undefined for a single fold, so the gate that
    reads this score skips rather than blocking on absent evidence.
    """
    wf = response.walk_forward
    if wf is None or not wf.splits or len(wf.splits) < 2:
        return None
    fold_pfs = [min(sp.test_summary.profit_factor, fold_pf_cap) for sp in wf.splits]
    mean_pf = float(np.mean(fold_pfs))
    std_pf = float(np.std(fold_pfs, ddof=1))
    return mean_pf - std_pf


def _document_index(registries: Optional[Iterable[RuleRegistry]]) -> Dict[str, object]:
    """Map ``AlphaContract.alpha_id`` -> its ``RuleDocument``, across registries."""
    index: Dict[str, object] = {}
    if registries is None:
        return index
    for registry in registries:
        for doc in registry.documents:
            index[doc.source_alpha_id] = doc
    return index


def _compute_rows(
    results: Iterable[ForgeResult],
    registries: Optional[Iterable[RuleRegistry]],
    fold_pf_cap: float = 10.0,
) -> List[dict]:
    """One dict per tradeable (EDGE/PARTIAL-EDGE) contract, flags plus live refs.

    Internal — carries the actual ``contract``/``response``/``candidate``
    objects (under ``_contract``/``_response``/``_candidate``/``_result``) so
    ``export_rules`` can write files without recomputing anything;
    ``promotion_gate`` strips those columns before returning.
    """
    doc_by_alpha_id = _document_index(registries)
    rows: List[dict] = []

    for result in results:
        candidates_by_id = {c.event_id: c for c in result.candidates}

        for contract, response in result.rule_responses:
            if not response.is_edge:
                continue

            reasons = response.rejection_reasons or []
            rotation_only = (
                len(reasons) == 1
                and reasons[0].startswith("search-level rotation null not cleared")
            )

            doc = doc_by_alpha_id.get(contract.alpha_id)
            is_duplicate = doc.is_duplicate if doc is not None else None
            is_isolated = (doc.classification == "ISOLATED") if doc is not None and doc.classification is not None else None

            consistency = None
            if response.walk_forward is not None:
                consistency = response.walk_forward.consistency

            fold_stability_score = _fold_stability_score(response, fold_pf_cap)

            candidate = candidates_by_id.get(contract.event_candidate_id)

            rows.append(
                {
                    "ticker": result.ticker,
                    "alpha_id": contract.alpha_id,
                    "grade": _contract_grade(contract),
                    "verdict": response.verdict,
                    "rotation_only": rotation_only,
                    "is_duplicate": is_duplicate,
                    "is_isolated": is_isolated,
                    "consistency": consistency,
                    "fold_stability_score": fold_stability_score,
                    "_contract": contract,
                    "_response": response,
                    "_candidate": candidate,
                    "_result": result,
                }
            )

    return rows


def _promotable_mask(rows: List[dict], config: "PromotionGateConfig") -> List[bool]:
    mask = []
    for row in rows:
        blocked = False
        if config.block_rotation_only and row["rotation_only"]:
            blocked = True
        if config.block_duplicate and row["is_duplicate"] is True:
            blocked = True
        if config.block_isolated and row["is_isolated"] is True:
            blocked = True
        if config.require_consistency and row["consistency"] is not None and row["consistency"] < config.min_consistency:
            blocked = True
        if (
            config.min_fold_stability_score is not None
            and row["fold_stability_score"] is not None
            and row["fold_stability_score"] < config.min_fold_stability_score
        ):
            blocked = True
        mask.append(not blocked)
    return mask


# ---------------------------------------------------------------------------
# 1. Promotion gate
# ---------------------------------------------------------------------------

@dataclass
class PromotionGateConfig:
    """Policy for :func:`promotion_gate` / :func:`export_rules`.

    Every flag is always computed and reported regardless of these settings
    — the ``block_*``/``require_consistency`` fields only decide which flags
    feed into the final ``promotable`` column, so a check can be turned off
    without losing visibility into it.

    Attributes
    ----------
    min_consistency : float
        Floor on ``RuleDiscoveryResponse.walk_forward.consistency`` (fraction
        of OOS walk-forward folds that were profitable). Default 0.5 — the
        same floor the pipeline itself uses internally to gate a positive
        verdict.
    block_rotation_only : bool
        Block a ``PARTIAL-EDGE`` whose *only* obstacle to a full ``EDGE`` was
        the search-level rotation null. Default ``False`` — a rotation-only
        miss is usually an acceptable trade-off, not a red flag.
    block_duplicate : bool
        Block a rule the Rule Registry marked ``is_duplicate=True``. Default
        ``True``.
    block_isolated : bool
        Block a rule classified ``"ISOLATED"`` on cross-ticker replay.
        Default ``True``. Has no effect (``is_isolated`` stays ``None``) when
        no ``registries`` were supplied.
    require_consistency : bool
        Whether the ``min_consistency`` floor participates in ``promotable``
        at all. Default ``True``.
    min_fold_stability_score : float or None
        Floor on the fold-variance-penalized stability score (#253):
        ``mean(fold_pf) - std(fold_pf)`` over
        ``RuleDiscoveryResponse.walk_forward.splits``' per-fold
        ``test_summary.profit_factor``, each capped at ``fold_pf_cap`` first.
        Catches a rule whose pooled walk-forward PF looks strong only
        because one high-variance fold (often the ``9999.0`` "zero losing
        trades" sentinel) dominates the aggregate, while its fold-to-fold
        performance is actually inconsistent and prone to collapsing
        out-of-sample. Default ``None`` — the gate is off; a rule with fewer
        than two walk-forward splits always passes it (sample std is
        undefined for one fold, so there's no variance evidence to penalize
        on).
    fold_pf_cap : float
        Cap applied to each fold's ``test_summary.profit_factor`` before
        computing ``fold_stability_score``, blunting the ``9999.0``
        "zero losing trades" sentinel so it doesn't dominate the mean/std.
        Default 10.0.
    """

    min_consistency: float = 0.5
    block_rotation_only: bool = False
    block_duplicate: bool = True
    block_isolated: bool = True
    require_consistency: bool = True
    min_fold_stability_score: Optional[float] = None
    fold_pf_cap: float = 10.0


def promotion_gate(
    results: Iterable[ForgeResult],
    registries: Optional[Iterable[RuleRegistry]] = None,
    config: PromotionGateConfig = PromotionGateConfig(),
) -> pd.DataFrame:
    """Long-format quality gate over every tradeable (EDGE/PARTIAL-EDGE) contract.

    Computes, per contract, the same flags the M3/M4 playground functions
    expose individually (:func:`~forgedge.playground.lottery_only_winners`'s
    ``rotation_only``, :func:`~forgedge.playground.duplicate_clusters`'s
    ``is_duplicate``, :func:`~forgedge.playground.classification_by_grade`'s
    ``ISOLATED`` classification, walk-forward ``consistency``, and the
    fold-variance-penalized ``fold_stability_score``, #253), then combines
    them into ``promotable`` per ``config``.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.
    registries : Iterable[RuleRegistry], optional
        Rule Registries to source ``is_duplicate``/``classification`` from
        (see :mod:`forgedge.playground.m4` for why this is separate from
        ``results``). ``None`` skips those two checks (columns stay
        ``None``) rather than failing.
    config : PromotionGateConfig
        Which checks block promotion, and at what threshold.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``alpha_id``, ``grade``, ``verdict``,
        ``rotation_only``, ``is_duplicate``, ``is_isolated``,
        ``consistency``, ``fold_stability_score``, ``promotable``.
    """
    rows = _compute_rows(results, registries, fold_pf_cap=config.fold_pf_cap)
    promotable = _promotable_mask(rows, config)

    columns = [
        "ticker",
        "alpha_id",
        "grade",
        "verdict",
        "rotation_only",
        "is_duplicate",
        "is_isolated",
        "consistency",
        "fold_stability_score",
    ]
    df = pd.DataFrame(
        [{k: row[k] for k in columns} for row in rows],
        columns=columns,
    )
    df["promotable"] = promotable
    return df


# ---------------------------------------------------------------------------
# 2. Export
# ---------------------------------------------------------------------------

def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if any(ch in text for ch in ":#\"'\n") or text.strip() != text or text == "":
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _dump_yaml_mapping(data: dict) -> str:
    """Minimal, dependency-free YAML block-mapping writer for a flat scalar dict.

    ``forgedge`` deliberately keeps its runtime dependencies to numpy/pandas
    only; every value a ``ValidatedRule``/export manifest carries is a flat
    scalar (str/int/float/bool/None), so a full YAML library is not needed
    to round-trip it correctly.
    """
    lines = [f"{key}: {_yaml_scalar(value)}" for key, value in data.items()]
    return "\n".join(lines) + "\n"


def export_rules(
    results: Iterable[ForgeResult],
    output_dir: Union[str, Path],
    *,
    registries: Optional[Iterable[RuleRegistry]] = None,
    config: PromotionGateConfig = PromotionGateConfig(),
    promotable_only: bool = True,
    include_kpi_recipe: bool = True,
    kpi_config=None,
    kpi_recipe_warmup_bars: int = 0,
) -> pd.DataFrame:
    """Write one ``.pkl`` (event) + one ``.yaml`` (rule params) per exported contract.

    Runs the same computation as :func:`promotion_gate` internally (so the
    two never disagree on what is promotable) and, for every selected
    contract, writes:

    - ``{output_dir}/{alpha_id}.pkl`` — the ``EventCandidate`` via
      :mod:`pickle` (its deterministic activation function, see
      ``EventCandidate.apply``).
    - ``{output_dir}/{alpha_id}.yaml`` — ``ValidatedRule.to_dict()`` (the
      published operating point: direction, entry mode, buy/sell
      parameters, horizon, fee) plus ``ticker``/``alpha_id``/``verdict`` for
      context.
    - ``{output_dir}/{alpha_id}.kpi_recipe.json`` (``include_kpi_recipe``,
      #296) — the :class:`~forgedge.deployment.KpiRecipe` recomputing only
      the KPI columns the event reads from raw candles, plus a
      ``"verification"`` block: the event replayed on
      ``KpiRecipe.rebuild(result.event_frame)`` vs on ``result.event_frame``
      itself (:func:`~forgedge.deployment.verify_kpi_recipe`). The pickle is
      always written too — it stays the faithful fallback whenever the
      recipe is incomplete (``unresolved_columns``) or unverified.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.
    output_dir : str or Path
        Directory to write into; created if missing.
    registries : Iterable[RuleRegistry], optional
        Forwarded to the underlying gate computation.
    config : PromotionGateConfig
        Forwarded to the underlying gate computation.
    promotable_only : bool, default True
        Export only contracts the gate marks ``promotable``. Set ``False``
        to export every tradeable contract regardless of the gate (the
        columns are still reported for audit).
    include_kpi_recipe : bool, default True
        Also write ``{alpha_id}.kpi_recipe.json`` per contract.
    kpi_config : dict, str, Path or None
        ``kpi_builder`` configuration the KPI Table was built with, forwarded
        to :func:`~forgedge.deployment.minimal_kpi_recipe` (``None`` =
        ``DEFAULT_CONFIG``).
    kpi_recipe_warmup_bars : int, default 0
        Bars excluded from the start of each ``result.event_frame`` when
        verifying the recipe. Leave at ``0`` when the KPI Table was built
        with ``kpi_builder`` from the very candles passed to ``forge()``
        (the round-trip is then exact). When it was built on a longer
        history and truncated, recursive/rolling indicators rebuilt from the
        shorter span differ during a warm-up transient — exclude it
        explicitly; no heuristic default is applied (#296).

    Returns
    -------
    pd.DataFrame
        One row per exported contract: ``ticker``, ``alpha_id``,
        ``event_candidate_id``, ``verdict``, ``promotable``, ``pkl_path``,
        ``yaml_path``, ``kpi_recipe_path``, ``kpi_recipe_verified``
        (``None`` when no recipe was written, or when ``result.event_frame``
        is unavailable to verify against).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _compute_rows(results, registries, fold_pf_cap=config.fold_pf_cap)
    promotable = _promotable_mask(rows, config)

    manifest_rows: List[dict] = []
    recipe_cache: Dict[tuple, tuple] = {}

    for row, is_promotable in zip(rows, promotable):
        if promotable_only and not is_promotable:
            continue

        contract = row["_contract"]
        response = row["_response"]
        candidate = row["_candidate"]
        validated_rule = response.validated_rule
        if candidate is None or validated_rule is None:
            continue

        alpha_id = contract.alpha_id
        pkl_path = output_dir / f"{alpha_id}.pkl"
        yaml_path = output_dir / f"{alpha_id}.yaml"

        with open(pkl_path, "wb") as fh:
            pickle.dump(candidate, fh)

        params = validated_rule.to_dict()
        params.update(
            {
                "ticker": row["ticker"],
                "alpha_id": alpha_id,
                "verdict": row["verdict"],
            }
        )
        yaml_path.write_text(_dump_yaml_mapping(params))

        recipe_path = None
        recipe_verified = None
        if include_kpi_recipe and getattr(candidate, "components", None):
            key = (id(row["_result"]), candidate.event_id)
            if key not in recipe_cache:
                recipe_cache[key] = _recipe_and_verification(
                    candidate, row["_result"], kpi_config, kpi_recipe_warmup_bars
                )
            recipe, verification = recipe_cache[key]
            recipe_path = output_dir / f"{alpha_id}.kpi_recipe.json"
            payload = recipe.to_dict()
            payload["verification"] = verification.to_dict() if verification is not None else None
            recipe_path.write_text(json.dumps(payload, indent=2) + "\n")
            recipe_path = str(recipe_path)
            recipe_verified = verification.matches if verification is not None else None

        manifest_rows.append(
            {
                "ticker": row["ticker"],
                "alpha_id": alpha_id,
                "event_candidate_id": contract.event_candidate_id,
                "verdict": row["verdict"],
                "promotable": is_promotable,
                "pkl_path": str(pkl_path),
                "yaml_path": str(yaml_path),
                "kpi_recipe_path": recipe_path,
                "kpi_recipe_verified": recipe_verified,
            }
        )

    return pd.DataFrame(manifest_rows, columns=_EXPORT_COLUMNS)


_EXPORT_COLUMNS = [
    "ticker",
    "alpha_id",
    "event_candidate_id",
    "verdict",
    "promotable",
    "pkl_path",
    "yaml_path",
    "kpi_recipe_path",
    "kpi_recipe_verified",
]


def _recipe_and_verification(
    candidate, result, kpi_config, warmup_bars: int
) -> "tuple[KpiRecipe, Optional[KpiRecipeVerification]]":
    """Recipe for ``candidate`` + its round-trip check on ``result.event_frame``.

    The verification is ``None`` when the result carries no ``event_frame``
    (nothing to verify against) — distinct from a failed check.
    """
    recipe = minimal_kpi_recipe(candidate, kpi_config=kpi_config)
    frame = getattr(result, "event_frame", None)
    if frame is None:
        return recipe, None
    timestamp_col = getattr(getattr(result, "context", None), "timestamp_col", "open_dt")
    mask = None
    if warmup_bars > 0:
        mask = pd.Series(np.arange(len(frame)) >= warmup_bars, index=frame.index)
    verification = verify_kpi_recipe(
        candidate, recipe, frame, evaluation_mask=mask, timestamp_col=timestamp_col
    )
    return recipe, verification


# ---------------------------------------------------------------------------
# 3. Monitoring manifest
# ---------------------------------------------------------------------------

def monitoring_manifest(
    results: Iterable[ForgeResult],
    exported: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Long-format index of every tradeable rule, for a periodic re-check job.

    Applies :meth:`RuleSpec.from_forge_result` (already provided by
    ``forgedge.rule_report`` for a single run) across all of R, so a
    monitoring job has one file listing every rule to replay on fresh
    candles via ``RuleDiscovery`` — never ``AlphaDiscovery`` — instead of
    reconstructing the reference to each rule by hand.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.
    exported : pd.DataFrame, optional
        :func:`export_rules`'s returned manifest. When given, its
        ``kpi_recipe_path``/``kpi_recipe_verified`` are joined in on
        ``(ticker, event_candidate_id)`` (#296), so the re-check job knows
        which rules it can keep alive from the minimal KPI recipe alone and
        which need the full KPI Table / pickled event. Rows are *not*
        filtered — rules that were not exported keep ``None`` there.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``rule_name``, ``event_candidate_id``,
        ``is_end``, ``verdict``, ``oos_expectancy``, ``kpi_recipe_path``,
        ``kpi_recipe_verified`` (the last two ``None`` without
        ``exported``). Join on ``event_candidate_id`` against
        :func:`export_rules`'s output to restrict to rules that were
        actually exported.
    """
    rows: List[dict] = []

    for result in results:
        for spec in RuleSpec.from_forge_result(result):
            rows.append(
                {
                    "ticker": result.ticker,
                    "rule_name": spec.name,
                    "event_candidate_id": spec.candidate.event_id,
                    "is_end": spec.is_end,
                    "verdict": spec.verdict,
                    "oos_expectancy": spec.oos_expectancy,
                }
            )

    recipes: Dict[tuple, tuple] = {}
    if exported is not None and "kpi_recipe_path" in exported.columns:
        for rec in exported.itertuples(index=False):
            key = (rec.ticker, rec.event_candidate_id)
            # Every contract built on the same event shares one recipe; keep
            # the first exported one.
            if pd.notna(rec.kpi_recipe_path) and key not in recipes:
                recipes[key] = (rec.kpi_recipe_path, rec.kpi_recipe_verified)
    for row in rows:
        path, verified = recipes.get((row["ticker"], row["event_candidate_id"]), (None, None))
        row["kpi_recipe_path"] = path
        row["kpi_recipe_verified"] = verified

    columns = [
        "ticker",
        "rule_name",
        "event_candidate_id",
        "is_end",
        "verdict",
        "oos_expectancy",
        "kpi_recipe_path",
        "kpi_recipe_verified",
    ]
    return pd.DataFrame(rows, columns=columns)
