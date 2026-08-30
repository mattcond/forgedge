"""Playground use cases anchored on Modulo 1 — Event Discovery.

Two use cases from the tracking issue:

1. ``dead_event_candidates`` — which gate-surviving ``EventCandidate``s never
   turn into an actionable contract downstream (M1->M2 waste).
2. ``gate_survival_observed`` — the raw, per-candidate Consistency Gate
   statistics (pass/fail, observed tpm/dispersion) alongside the configured
   thresholds, so a preset's fit to a given asset can be judged before it
   shows up as "0 candidates" in M2.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from ..forge import ForgeResult

__all__ = ["dead_event_candidates", "gate_survival_observed"]


def dead_event_candidates(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format classification of every gate-surviving candidate's fate in M2.

    For each ``EventCandidate`` in ``result.candidates`` (every candidate that
    passed the Consistency Gate), counts how many of ``result.contracts`` were
    derived from it and how many of those came back with
    ``direction == "undetermined"``, then labels it:

    - ``"dead"`` — produced zero contracts at all.
    - ``"undetermined_only"`` — produced contracts, but every one is
      undetermined (M2 could never derive an oriented target from it).
    - ``"actionable"`` — produced at least one contract with a derived
      direction.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``event_candidate_id``, ``expression``,
        ``n_contracts``, ``n_undetermined``, ``status``. Downstream, e.g.
        ``df[df["status"] != "actionable"].groupby("ticker").size()`` to
        quantify M1->M2 waste per asset.
    """
    rows: List[dict] = []

    for result in results:
        contracts_by_candidate: dict = {}
        for contract in result.contracts:
            contracts_by_candidate.setdefault(contract.event_candidate_id, []).append(contract)

        for candidate in result.candidates:
            contracts = contracts_by_candidate.get(candidate.event_id, [])
            n_contracts = len(contracts)
            n_undetermined = sum(1 for c in contracts if c.direction == "undetermined")

            if n_contracts == 0:
                status = "dead"
            elif n_undetermined == n_contracts:
                status = "undetermined_only"
            else:
                status = "actionable"

            rows.append(
                {
                    "ticker": result.ticker,
                    "event_candidate_id": candidate.event_id,
                    "expression": candidate.expression,
                    "n_contracts": n_contracts,
                    "n_undetermined": n_undetermined,
                    "status": status,
                }
            )

    columns = [
        "ticker",
        "event_candidate_id",
        "expression",
        "n_contracts",
        "n_undetermined",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)


def gate_survival_observed(results: Iterable[ForgeResult]) -> pd.DataFrame:
    """Long-format Consistency Gate outcome for every raw candidate evaluated.

    Reads ``result.event_discovery.raw_events`` — the full pre-gate
    population (``DiscoveryConfig.retain_raw_events``, default ``True``),
    each annotated with its ``GateResult`` — alongside the configured
    ``GateParams`` that decided pass/fail. Gives the caller the same
    ingredients ``EventDiscovery.event_distribution_report`` narrates in
    prose, as data: e.g. ``df.groupby("ticker")["passed"].mean()`` for the
    observed survival rate per asset, or
    ``df.groupby("ticker").apply(lambda g: (g["mean_tpm"] < g["min_tpm"]).mean())``
    to see how much of the rejection is tpm-driven versus dispersion-driven,
    before a preset/asset mismatch surfaces downstream as "0 candidates" in
    M2. Skips any result where Event Discovery was not run, or was run with
    ``retain_raw_events=False``.

    Parameters
    ----------
    results : Iterable[ForgeResult]
        R — one or more ``forge()``/``forge_multi()`` outputs.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``mean_tpm``, ``index_of_dispersion``,
        ``episode_index_of_dispersion``, ``n_episodes``, ``passed``,
        ``fail_reason``, ``min_tpm``, ``max_dispersion``,
        ``dispersion_margin``, ``event_counting`` — the last four repeat the
        configured thresholds on every row for easy per-group comparison.
    """
    rows: List[dict] = []

    for result in results:
        event_discovery = getattr(result, "event_discovery", None)
        if event_discovery is None:
            continue

        raw_events = event_discovery.raw_events
        if raw_events is None:
            continue

        gate_params = event_discovery.config.gate_params

        for raw_event in raw_events:
            gate_result = raw_event.gate_result
            if gate_result is None:
                continue

            rows.append(
                {
                    "ticker": result.ticker,
                    "mean_tpm": gate_result.mean_tpm,
                    "index_of_dispersion": gate_result.index_of_dispersion,
                    "episode_index_of_dispersion": gate_result.episode_index_of_dispersion,
                    "n_episodes": gate_result.n_episodes,
                    "passed": gate_result.passed,
                    "fail_reason": gate_result.fail_reason,
                    "min_tpm": gate_params.min_tpm,
                    "max_dispersion": gate_params.max_dispersion,
                    "dispersion_margin": gate_params.dispersion_margin,
                    "event_counting": gate_params.event_counting,
                }
            )

    columns = [
        "ticker",
        "mean_tpm",
        "index_of_dispersion",
        "episode_index_of_dispersion",
        "n_episodes",
        "passed",
        "fail_reason",
        "min_tpm",
        "max_dispersion",
        "dispersion_margin",
        "event_counting",
    ]
    return pd.DataFrame(rows, columns=columns)
