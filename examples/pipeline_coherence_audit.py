"""Diagnostic for the pipeline-wide parameter-coherence audit (issue #173 family).

Reproduces the numbers quoted in ``docs/analysis/pipeline_parameter_coherence.md``.
Every stage measures one *latent parameter* — a quantity that has a single
meaning for the whole pipeline but is materialised as several independent
config fields, each with its own default, in different modules.

Usage
-----
    python examples/pipeline_coherence_audit.py [stage ...]

Stages (default: ``rates m1``):

``rates``
    Pure arithmetic, no data.  Prints the minimum event/trade rate every gate
    *implicitly* requires under ``forge_preset("balanced", tf)``, per timeframe,
    plus the ``c_norm`` transfer curve of ``pf_score_tpm``.
``m1``
    Event Discovery's walk-forward: how often the **absolute** ``min_episodes``
    floor rejects an OOS fold that the **rate** gate would have accepted.
``m2``
    Alpha Discovery: how often the absolute ``_MIN_STATS_CASES`` floor fires on
    the OOS tail, as a function of ``train_ratio``.
``m3``
    Rule Discovery: reproduces issue #173's early-elimination on this repo's own
    fixture, and measures the realised trade rate against the fixed
    ``ScoringParams.pf_tpm_target``.

The dataset defaults to ``tests/fixtures/ADA_1D_TRAIN.parquet``; override it
with ``FORGEDGE_PARQUET=/path/to/kpi.parquet``.
"""
from __future__ import annotations

import collections
import os
import re
import sys
import warnings

import pandas as pd

from forgedge import (
    AlphaDiscovery,
    DiscoveryConfig,
    EventDiscovery,
    MarketContext,
    forge,
)
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams
from forgedge.presets import _TFClass, forge_preset

PARQUET = os.environ.get("FORGEDGE_PARQUET", "tests/fixtures/ADA_1D_TRAIN.parquet")


def _load() -> pd.DataFrame:
    kpi = pd.read_parquet(PARQUET)
    print(f"dataset: {PARQUET} — {len(kpi)} bars, "
          f"{kpi['open_dt'].min().date()} → {kpi['open_dt'].max().date()}")
    return kpi


def _family(reason: str, width: int = 78) -> str:
    """Collapse a rejection reason to its numeric-free family key."""
    return re.sub(r"[-+]?\d*\.?\d+", "N", reason)[:width]


# ---------------------------------------------------------------------------
# Stage "rates" — the implied rate ladder
# ---------------------------------------------------------------------------

def stage_rates() -> None:
    print("\n" + "=" * 78)
    print("  LATENT PARAMETER: minimum event/trade rate — what each gate implies")
    print("=" * 78)
    for tf in ("1D", "4H", "1H", "15m"):
        disc, _alpha, rd = forge_preset("balanced", tf, asset="X")
        t = _TFClass(tf)
        gate, cr, sc, wf = disc.gate_params, rd.criteria, rd.scoring, rd.walk_forward
        print(f"\n--- forge_preset('balanced', {tf!r}) — {t.bars_per_month:.0f} bars/month ---")
        print(f"  M1 GateParams.min_tpm          {gate.min_tpm:8.2f} ep/month   [scaled by preset]")
        print(f"  M1 GateParams.min_episodes     {gate.min_episodes:8d}          "
              f"[ABSOLUTE — same value on every timeframe and every fold length]")
        print(f"  M3 SelectionCriteria.min_tpm   {cr.min_tpm:8.2f} tr/month   [scaled by preset]")
        for months in (wf.min_train_months, 12, 24):
            floor = max(10, months * cr.min_tpm)
            print(f"       early-elim floor over {months:>2}mo = max(10, {months}×{cr.min_tpm:.2f})"
                  f" = {floor:6.1f} trades → {floor / months:5.2f} tr/month implied")
        print(f"  M3 ScoringParams.pf_min_trades {sc.pf_min_trades:8d}          [ABSOLUTE — not scaled]")
        print(f"  M3 ScoringParams.pf_min_tpm    {sc.pf_min_tpm:8d}          [ABSOLUTE — not scaled]")
        print(f"  M3 ScoringParams.pf_tpm_target {sc.pf_tpm_target:8d}          [ABSOLUTE — not scaled]")
        print(f"  M3 criteria.min_oos_trades     {cr.min_oos_trades:8d}          [ABSOLUTE — not scaled]")

    print("\n" + "=" * 78)
    print("  pf_score_tpm = PF × c_norm — the selection objective is a band-pass")
    print("  centred on pf_tpm_target = 3 trades/month, on EVERY timeframe")
    print("=" * 78)
    print(f"  {'tpm_mu':>8}  {'sigma':>6}  {'c_norm':>7}  {'PF needed for min_pf_score_tpm=0.30':>36}")
    for mu in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0):
        sigma = mu ** 0.5                       # Poisson arrivals
        f_r = min(3.0 / mu, 1.0)
        c_norm = max(0.0, min(1.0, (mu / (sigma + 1.0)) * f_r / 3.0))
        need = 0.30 / c_norm if c_norm > 0 else float("inf")
        print(f"  {mu:8.1f}  {sigma:6.2f}  {c_norm:7.3f}  {need:36.2f}")


# ---------------------------------------------------------------------------
# Stage "m1" — absolute episode floor vs. fold length
# ---------------------------------------------------------------------------

def stage_m1(kpi: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  M1 — GateParams.min_episodes (absolute) applied to short OOS folds")
    print("=" * 78)
    enriched = MarketContext(kpi).run()
    n, train_ratio, n_splits = len(enriched), 0.80, 3
    n_oos = n - int(n * train_ratio)
    fold_months = n_oos / n_splits / 30.0
    gate = GateParams(min_tpm=1.0, max_dispersion=2.0, event_counting="episode")
    print(f"  bars={n}  IS={int(n * train_ratio)}  OOS={n_oos}  "
          f"fold={n_oos // n_splits} bars ≈ {fold_months:.1f} months")
    print(f"  IS gate requires {gate.min_tpm:.2f} episodes/month;")
    print(f"  the same GateParams requires {gate.min_episodes} episodes inside a "
          f"{fold_months:.1f}-month fold")
    print(f"  → the OOS gate is {gate.min_episodes / fold_months / gate.min_tpm:.1f}× "
          f"stricter than the IS gate, by construction")

    ed = EventDiscovery(enriched, config=DiscoveryConfig(
        gate_params=gate, train_ratio=train_ratio,
        walk_forward=EventWalkForwardConfig(n_splits=n_splits), max_and_components=1,
    ))
    cands = ed.run()
    reasons: collections.Counter = collections.Counter()
    n_pass = n_eval = 0
    for cand in cands:
        if cand.validation is None:
            continue
        for fold in cand.validation.fold_results:
            n_eval += 1
            if fold.gate_result.passed:
                n_pass += 1
            else:
                reasons[fold.gate_result.fail_reason.split(":")[0]] += 1
    stable = sum(1 for c in cands if c.validation and c.validation.passed)
    print(f"\n  candidates passing the IS gate : {len(cands)}")
    print(f"  OOS fold evaluations           : {n_eval}  passed {n_pass} "
          f"({n_pass / max(n_eval, 1):.1%})")
    for key, count in reasons.most_common():
        print(f"     {key:<20} {count:6d}  ({count / max(n_eval, 1):5.1%} of all fold evaluations)")
    print(f"  OOS-stable candidates          : {stable}/{len(cands)} "
          f"({stable / max(len(cands), 1):.1%})")


# ---------------------------------------------------------------------------
# Stage "m2" — absolute activation floor vs. the OOS tail
# ---------------------------------------------------------------------------

def stage_m2(kpi: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  M2 — _MIN_STATS_CASES (absolute) applied to the train_ratio OOS tail")
    print("=" * 78)
    enriched = MarketContext(kpi).run()
    ed = EventDiscovery(enriched, config=DiscoveryConfig(
        gate_params=GateParams(min_tpm=1.0, max_dispersion=2.0, event_counting="episode"),
        train_ratio=1.0, max_and_components=1,
    ))
    cands = ed.run()
    print(f"  M1 candidates: {len(cands)}")
    for train_ratio in (0.70, 0.85):
        _disc, alpha_cfg, _rd = forge_preset("balanced", "1D", asset="ADA")
        alpha_cfg.train_ratio = train_ratio
        ad = AlphaDiscovery(ed.df, cands, alpha_cfg)
        contracts = ad.run()
        n_oos = len(ed.df) - ad.split_idx
        fams: collections.Counter = collections.Counter()
        for contract in contracts:
            # Blocking causes and non-blocking diagnostics live in separate
            # fields; this stage is about what M2 *reports*, so count both.
            for reason in (contract.rejection_reasons or []) + (contract.diagnostics or []):
                fams[_family(reason, 70)] += 1
        print(f"\n  train_ratio={train_ratio}  OOS={n_oos} bars (~{n_oos / 30:.1f} months)  "
              f"promoted={len(ad.promoted_contracts())}/{len(contracts)}")
        for key, count in fams.most_common(5):
            print(f"     {count:6d}  {key}")


# ---------------------------------------------------------------------------
# Stage "m3" — issue #173 on this repo's own fixture
# ---------------------------------------------------------------------------

def stage_m3(kpi: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  M3 — issue #173: min_tpm permissive upstream, min_train_months fixed")
    print("=" * 78)
    disc, alpha_cfg, rd = forge_preset(
        "balanced", "1D", asset="ADA", min_tpm=0.30, rd_min_tpm=0.25
    )
    disc.max_and_components = 1
    print(f"  M1 min_tpm={disc.gate_params.min_tpm}  M3 criteria.min_tpm={rd.criteria.min_tpm}  "
          f"walk_forward.min_train_months={rd.walk_forward.min_train_months}")
    print(f"  ScoringParams left at the class default: {rd.scoring}")

    result = forge(kpi, ticker="ADA", timeframe="1D",
                   event_discovery_config=disc, alpha_config=alpha_cfg,
                   rule_discovery_config=rd, rule_discovery_grades=("A", "B"),
                   progress=False)
    responses = result.rule_responses
    print(f"\n  M1 {len(result.candidates)} candidates | "
          f"M2 {len(result.promoted)}/{len(result.contracts)} promoted | "
          f"M3 {len(responses)} backtested")
    print("  verdicts:", dict(collections.Counter(r.verdict for _, r in responses)))

    fams: collections.Counter = collections.Counter()
    for _contract, response in responses:
        for reason in response.rejection_reasons or []:
            fams[_family(reason)] += 1
    print("\n  M3 rejection reasons:")
    for key, count in fams.most_common(8):
        print(f"     {count:5d}  ({count / max(len(responses), 1):5.1%})  {key}")

    summaries = [r.in_sample_summary for _, r in responses if r.in_sample_summary]
    if summaries:
        df = pd.DataFrame({
            "tpm_mu": [s.tpm_mu for s in summaries],
            "c_norm": [s.c_norm for s in summaries],
            "pf_score_tpm": [s.pf_score_tpm for s in summaries],
            "n_months": [s.n_months for s in summaries],
        })
        print(f"\n  selection-span length: median {df.n_months.median():.0f} months "
              f"(= walk_forward.min_train_months, NOT the in-sample span)")
        print(f"  realised tpm_mu      : median {df.tpm_mu.median():.2f}, "
              f"p95 {df.tpm_mu.quantile(0.95):.2f}  vs pf_tpm_target=3 (fixed)")
        print(f"  c_norm               : median {df.c_norm.median():.3f}; "
              f"{(df.c_norm < 0.30).mean():.1%} of rules sit below 0.30")


# ---------------------------------------------------------------------------

_STAGES = {"rates": None, "m1": stage_m1, "m2": stage_m2, "m3": stage_m3}


def main(argv: list) -> None:
    warnings.filterwarnings("ignore")
    stages = argv[1:] or ["rates", "m1"]
    unknown = [s for s in stages if s not in _STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(_STAGES)}")
    kpi = _load() if any(s != "rates" for s in stages) else None
    for stage in stages:
        if stage == "rates":
            stage_rates()
        else:
            _STAGES[stage](kpi)


if __name__ == "__main__":
    main(sys.argv)
