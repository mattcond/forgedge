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
``alpha``
    Pure arithmetic, no data.  The seven significance thresholds side by side:
    which five are a per-hypothesis alpha and now derive from one, and which
    two are a different quantity and deliberately do not.
``m3``
    Rule Discovery: reproduces issue #173's early-elimination on this repo's own
    fixture, and measures the realised trade rate against the consistency term
    that gates the grid's operating point.

The dataset defaults to ``tests/fixtures/ADA_1D_TRAIN.parquet``; override it
with ``FORGEDGE_PARQUET=/path/to/kpi.parquet``.
"""
from __future__ import annotations

import collections
import math
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
from forgedge.event_discovery.discovery import MIN_FOLD_LAMBDA
from forgedge.event_discovery.models import EventWalkForwardConfig, GateParams
from forgedge.alpha_discovery.models import AlphaConfig
from forgedge.calibration.models import RotationConfig
from forgedge.market_context.models import MarketContextConfig
from forgedge.presets import _TFClass, forge_preset
from forgedge.resolver import PipelineContext, resolve_config
from forgedge.rule_discovery.models import RuleDiscoveryConfig

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
        # Several of these fields are session-resolved now (#173, #178), so the
        # audit has to read the values that *run*, not the sentinels.
        rd = resolve_config(rd, "rule_discovery", PipelineContext(timeframe=tf))
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
        print(f"  M3 ScoringParams.pf_min_tpm    {sc.pf_min_tpm:8.2f}          "
              f"[resolved from criteria.min_tpm — #178]")
        print(f"  M3 criteria.min_oos_trades     {cr.min_oos_trades:8d}          [ABSOLUTE — not scaled]")

    print("\n  #200 — the declared rate now reaches M3, and the walk-forward with it:")
    print(f"  {'declared min_tpm':>16}  {'M1':>6}  {'M3':>6}  {'min_train_months':>16}  "
          f"{'OOS span on 29 months':>22}")
    for override in (0.5, 1.0, 2.0, 4.0):
        disc, _a, rd = forge_preset("balanced", "1D", asset="X", min_tpm=override)
        rd = resolve_config(rd, "rule_discovery", PipelineContext(timeframe="1D"))
        oos = 29 - rd.walk_forward.min_train_months
        span = f"{oos:d} mo" if oos > 0 else "no verdict possible"
        print(f"  {override:16.2f}  {disc.gate_params.min_tpm:6.2f}  "
              f"{rd.criteria.min_tpm:6.2f}  {rd.walk_forward.min_train_months:16d}  "
              f"{span:>22}")
    print("  before #200 the M3 column stayed at 0.80 whatever M1 was told, so")
    print("  min_train_months stayed at 20 and the OOS span at 9 months — raising")
    print("  the session's rate *degraded* the walk-forward instead of tightening it.")
    print("  the rate propagates unchanged (rate_retention=1.0): a margin below 1")
    print("  costs history twice — it lengthens min_train_months via the Poisson")
    print("  margin AND shrinks the pooled OOS count (test_months × min_tpm), so a")
    print("  25% cut in the floor demands 25% more data (13.0 → 16.2 months).")
    print("  the fill margin is a preset judgement instead: 1.00 on sniper/sweep,")
    print("  0.80 on balanced/burst, each spec's own ratio carried through.")

    print("\n  #204 — M3 has no notion of episodes; \"episode\" mode has to say so:")
    print(f"  {'event_counting':>15}  {'M1 declared':>12}  {'M3 (old, wrong)':>16}  "
          f"{'M3 (fixed)':>11}")
    for ec, old_ratio in (("bar", 2.5 / 3.0), ("episode", 0.8)):
        disc, _a, rd = forge_preset("balanced", "1D", asset="X", event_counting=ec)
        m1 = disc.gate_params.min_tpm
        old = round(m1 * old_ratio, 4)
        print(f"  {ec:>15}  {m1:12.4f}  {old:16.4f}  {rd.criteria.min_tpm:11.4f}")
    print("  the \"old\" column is the fill ratio (0.83) applied straight to the")
    print("  declared rate, which is correct in bar mode — M1 and M3 already share")
    print("  a unit there. In episode mode it was applied to the wrong quantity:")
    print("  M3 counts bars (\"a trade opens on every active bar, no flat-state")
    print("  check\"), and episodes are runs of *several* bars — median 1.76 on")
    print("  ADA_1D_TRAIN, 3664 candidates. The fixed column converts episodes to")
    print("  bars first (PipelineContext.bars_per_episode), then applies the same")
    print("  fill ratio: 1.0 × 1.76 × 0.83 = 1.47, not 0.8 — M3's floor was ~1.8x")
    print("  too low, which inflated min_train_months for no reason (same shape")
    print("  as #200's fill-margin fix, one factor upstream of it).")

    print("\n  #205 — a preset's own dispersion tolerance, before and after:")
    from forgedge.event_discovery.consistency_gate import _chi2_ppf_095

    def _poisson_floor(n_months):
        df = n_months - 1
        return _chi2_ppf_095(df) / df if df > 0 else 0.0

    print(f"  {'preset':>9}  {'daily_max_dispersion':>21}  {'eff @ 24mo (old)':>17}  "
          f"{'dispersion_margin':>18}  {'eff @ 24mo (new)':>17}")
    floor_24 = _poisson_floor(24)
    for preset in ("sniper", "balanced", "sweep", "burst"):
        disc, _a, _r = forge_preset(preset, "1D", asset="X")
        old_configured = disc.gate_params.max_dispersion
        old_eff = max(old_configured, floor_24)
        new_eff = floor_24 * disc.gate_params.dispersion_margin
        print(f"  {preset:>9}  {old_configured:21.2f}  {old_eff:17.3f}  "
              f"{disc.gate_params.dispersion_margin:18.2f}  {new_eff:17.3f}")
    print("  at 1D sniper's own max_dispersion=1.0 already collapses to the 24-month")
    print("  Poisson floor (1.53) — and on faster timeframes it gets worse: scale_")
    print("  dispersion() shrinks the configured value further while the floor (a")
    print("  function of calendar months only) does not move, so the floor wins more")
    print("  often the faster the timeframe gets. Measured across all four presets and")
    print("  four timeframes: 12 of 16 combinations had max_dispersion never binding,")
    print("  and sniper — the preset built for \"regular\" events — never bound on any")
    print("  of them (#205). dispersion_margin expresses the tolerance as slack *above*")
    print("  the floor instead, which the floor mechanism can no longer swallow.")

    print("\n  #206 — min_episodes is absolute, so the IS window it needs depends")
    print("  on the rate too, the same shape as F1 one level up (discovery, not a fold):")
    print(f"  {'preset':>9}  {'rate (ep/mo)':>13}  {'min_episodes':>13}  "
          f"{'window @ 95% (mo)':>18}  {'(years)':>8}  {'promised':>28}")
    from forgedge.resolver import poisson_min_window
    promised = {"sniper": "was \"≥2 anni\"", "balanced": "—", "sweep": "—", "burst": "—"}
    for preset in ("sniper", "balanced", "sweep", "burst"):
        disc, _a, _r = forge_preset(preset, "1D", asset="X")
        gp = disc.gate_params
        window = poisson_min_window(gp.min_episodes, gp.min_tpm)
        print(f"  {preset:>9}  {gp.min_tpm:13.2f}  {gp.min_episodes:13d}  "
              f"{window:18.1f}  {window / 12:8.2f}  {promised[preset]:>28}")
    print("  sniper and sweep share the same 0.3 ep/mo rate; before #206 both also")
    print("  shared min_episodes=10 and needed 53.3 months (4.44 years) regardless —")
    print("  more than double what sniper's own description promised. sniper keeps")
    print("  min_episodes=10 (statistical rigor is the point) and the description is")
    print("  corrected instead; sweep lowers it to 5, consistent with being permissive")
    print("  by design and deferring rigor to the RotationCalibrator downstream.")
    print("  config_report() now names the gap directly (m1_is_window_too_short, WARN —")
    print("  not FAIL: a candidate above the floor rate can still clear it on less data).")

    print("\n" + "=" * 78)
    print("  LATENT PARAMETER: bar duration — the fields that mean \"N bars\"")
    print("=" * 78)
    print(f"  {'timeframe':>9}  {'bar_hours':>9}  {'buy_delay_bar':>13}  "
          f"{'target_h':>9}  {'stable_window':>13}  {'bars/day':>9}  "
          f"{'horizon_grid':>22}")
    for tf in ("1D", "4H", "1H", "15m"):
        ctx = PipelineContext(timeframe=tf)
        rd = resolve_config(RuleDiscoveryConfig(), "rule_discovery", ctx)
        mc = resolve_config(MarketContextConfig(), "market_context", ctx)
        al = resolve_config(AlphaConfig(timeframe=tf), "alpha", ctx)
        print(f"  {tf:>9}  {ctx.bar_hours:9.2f}  {rd.base_params.buy_delay_bar:13d}  "
              f"{rd.base_params.target_h:9d}  {mc.stable_window:13d}  "
              f"{ctx.bars_per_day:9.2f}  {str(al.horizon_grid):>22}")
    print("  before #179 every column but the first was frozen at its 1H value,")
    print("  so a daily session rested its limit orders for six *days* (F5).")
    print("  `horizon_grid` was the last to follow (#196): until then it was")
    print("  converted only when forge() built the AlphaConfig itself, so an")
    print("  explicit config on daily candles still scanned up to 48 *days*.")
    print("  note target_h == max(horizon_grid) — one calibration, two readers")

    print("\n" + "=" * 78)
    print("  pf_score_tpm = PF × c_norm — the selection objective, on a *Poisson*")
    print("  process (index of dispersion 1, i.e. perfect regularity at any rate)")
    print("=" * 78)
    print(f"  {'tpm_mu':>8}  {'sigma':>6}  {'c_norm old':>10}  {'c_norm new':>10}  "
          f"{'PF needed old':>13}  {'PF needed new':>13}")
    for mu in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0):
        sigma = mu ** 0.5                       # Poisson arrivals
        f_r = min(3.0 / mu, 1.0)
        old = max(0.0, min(1.0, (mu / (sigma + 1.0)) * f_r / 3.0))
        new = min(1.0, 1.0 / max((sigma * sigma) / mu, 1.0))
        need_old = 0.30 / old if old > 0 else float("inf")
        need_new = 0.30 / new if new > 0 else float("inf")
        print(f"  {mu:8.1f}  {sigma:6.2f}  {old:10.3f}  {new:10.3f}  "
              f"{need_old:13.2f}  {need_new:13.2f}")
    print("  the old column is the same process scored 2.4x worse for trading more")
    print("  often; the new one is flat, as a scale-free measure has to be (F3)")


# ---------------------------------------------------------------------------
# Stage "m1" — absolute episode floor vs. fold length
# ---------------------------------------------------------------------------

def stage_m1(kpi: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  M1 — walk-forward folds: testable, indeterminate, or neither")
    print("=" * 78)
    enriched = MarketContext(kpi).run()
    n, train_ratio, n_splits = len(enriched), 0.80, 3
    n_oos = n - int(n * train_ratio)
    fold_months = n_oos / n_splits / 30.0
    gate = GateParams(min_tpm=1.0, event_counting="episode")
    print(f"  bars={n}  IS={int(n * train_ratio)}  OOS={n_oos}  "
          f"fold={n_oos // n_splits} bars ≈ {fold_months:.1f} months")
    print(f"  IS gate requires {gate.min_tpm:.2f} episodes/month")
    print(f"  before #177 the same GateParams also demanded {gate.min_episodes} "
          f"episodes inside a {fold_months:.1f}-month fold")
    print(f"     → an OOS gate {gate.min_episodes / fold_months / gate.min_tpm:.1f}× "
          f"stricter than the IS one, by construction (F1)")
    print(f"  now: `min_episodes` is in-sample only, and a fold is *testable*")
    print(f"     when its expected episode count reaches λ = {MIN_FOLD_LAMBDA:g}")
    print(f"     (P(empty fold) = e^-λ = {math.exp(-MIN_FOLD_LAMBDA):.0%} for a "
          f"healthy candidate)")

    ed = EventDiscovery(enriched, config=DiscoveryConfig(
        gate_params=gate, train_ratio=train_ratio,
        walk_forward=EventWalkForwardConfig(n_splits=n_splits), max_and_components=1,
    ))
    cands = ed.run()
    reasons: collections.Counter = collections.Counter()
    n_pass = n_eval = n_indet = 0
    for cand in cands:
        if cand.validation is None:
            continue
        for fold in cand.validation.fold_results:
            if fold.indeterminate:
                n_indet += 1
                continue
            n_eval += 1
            if fold.gate_result.passed:
                n_pass += 1
            else:
                reasons[fold.gate_result.fail_reason.split(":")[0]] += 1
    stable = sum(1 for c in cands if c.validation and c.validation.passed)
    unresolved = sum(1 for c in cands
                     if c.validation is not None and c.validation.passed is None)
    print(f"\n  candidates passing the IS gate : {len(cands)}")
    print(f"  folds too short to conclude    : {n_indet}  (INDETERMINATE — "
          f"excluded from the denominator, not counted as failures)")
    print(f"  testable fold evaluations      : {n_eval}  passed {n_pass} "
          f"({n_pass / max(n_eval, 1):.1%})")
    for key, count in reasons.most_common():
        print(f"     {key:<20} {count:6d}  ({count / max(n_eval, 1):5.1%} of testable folds)")
    print(f"  OOS-stable candidates          : {stable}/{len(cands)} "
          f"({stable / max(len(cands), 1):.1%})")
    print(f"  inconclusive (passed=None)     : {unresolved}  — kept, not dropped: "
          f"a window that could not answer is not a candidate that failed")


# ---------------------------------------------------------------------------
# Stage "m2" — absolute activation floor vs. the OOS tail
# ---------------------------------------------------------------------------

def stage_m2(kpi: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  M2 — _MIN_STATS_CASES (absolute) applied to the train_ratio OOS tail")
    print("=" * 78)
    enriched = MarketContext(kpi).run()
    ed = EventDiscovery(enriched, config=DiscoveryConfig(
        gate_params=GateParams(min_tpm=1.0, event_counting="episode"),
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
            "tpm_sigma": [s.tpm_sigma for s in summaries],
            "c_norm": [s.c_norm for s in summaries],
            "pf_score_tpm": [s.pf_score_tpm for s in summaries],
            "n_months": [s.n_months for s in summaries],
        })
        print(f"\n  selection-span length: median {df.n_months.median():.0f} months "
              f"(= walk_forward.min_train_months, NOT the in-sample span)")
        print(f"  realised tpm_mu      : median {df.tpm_mu.median():.2f}, "
              f"p95 {df.tpm_mu.quantile(0.95):.2f}")
        print(f"  c_norm               : median {df.c_norm.median():.3f}; "
              f"{(df.c_norm < 0.30).mean():.1%} of rules sit below 0.30")
        # `min_pf_score_tpm` is a gate in `_passes` alongside `min_profit_factor`,
        # so `min_pf_score_tpm / c_norm` is a *second*, undeclared PF threshold.
        # It only matters where it exceeds the declared one (#178, F3).
        live = df[df.c_norm > 0]
        hidden = 0.30 / live.c_norm
        print(f"  hidden PF threshold  : min_pf_score_tpm=0.30 / c_norm → median "
              f"{hidden.median():.2f}  p90 {hidden.quantile(0.90):.2f}; binding over "
              f"the declared min_profit_factor=2.0 for {(hidden > 2.0).mean():.1%} of rules")


# ---------------------------------------------------------------------------

def stage_alpha() -> None:
    """F9 — the seven thresholds, and which of them are actually one thing."""
    print("\n" + "=" * 78)
    print("  LATENT PARAMETER: significance — seven thresholds, five of them alpha")
    print("=" * 78)
    ctx = PipelineContext(timeframe="1D")
    alpha_cfg = resolve_config(AlphaConfig(), "alpha", ctx)
    rd = resolve_config(RuleDiscoveryConfig(), "rule_discovery", ctx)
    th = alpha_cfg.thresholds

    print(f"  {'threshold':<18} {'module':<8} {'value':>7}  what it gates")
    rows = [
        ("max_p_value", "M2", th.max_p_value,
         "t-test on the excess return — INERT under use_fdr=True"),
        ("ic_max_p", "M2", th.ic_max_p,
         "feature IC — non-blocking, weighs on the grade only"),
        ("max_ttest_p", "M3", rd.criteria.max_ttest_p,
         "expectancy on the ledger — the ONLY hard per-hypothesis gate"),
        ("max_rotation_p", "M3", rd.criteria.max_rotation_p,
         "the search surface — a different null, still an alpha"),
        ("RotationConfig.alpha", "calib", RotationConfig().resolved(ctx.alpha).alpha,
         "the survivor bar"),
    ]
    for name, mod, val, what in rows:
        print(f"  {name:<18} {mod:<8} {val:>7.3f}  {what}")
    print("  ^ all five now derive from ctx.alpha — one number, not five (#182)")

    print()
    for name, mod, val, what in [
        ("fdr_q", "M2", th.fdr_q,
         "false DISCOVERY RATE over the horizon family — not an alpha"),
        ("oos_max_p", "M2", th.oos_max_p,
         "CONFIRMATION of a selected hypothesis — no multiplicity, small n"),
        ("min_pass_rate", "M1", 0.6,
         "a VOTE: how many folds must agree — not a probability at all"),
    ]:
        print(f"  {name:<18} {mod:<8} {val:>7.3f}  {what}")
    print("  ^ deliberately NOT tied to alpha: tying them would be a category error")

    print()
    print("  a different regime is now one number:")
    for a in (0.10, 0.05, 0.01):
        c = PipelineContext(timeframe="1D", alpha=a)
        r = resolve_config(RuleDiscoveryConfig(), "rule_discovery", c)
        t = resolve_config(AlphaConfig(), "alpha", c).thresholds
        print(f"    ctx.alpha={a:<5} -> max_p_value={t.max_p_value:g} "
              f"ic_max_p={t.ic_max_p:g} max_ttest_p={r.criteria.max_ttest_p:g} "
              f"max_rotation_p={r.criteria.max_rotation_p:g}  "
              f"(fdr_q stays {t.fdr_q:g}, oos_max_p stays {t.oos_max_p:g})")


# ---------------------------------------------------------------------------

_STAGES = {"rates": None, "alpha": None, "m1": stage_m1, "m2": stage_m2,
           "m3": stage_m3}


def main(argv: list) -> None:
    warnings.filterwarnings("ignore")
    stages = argv[1:] or ["rates", "m1"]
    unknown = [s for s in stages if s not in _STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(_STAGES)}")
    dataless = {"rates", "alpha"}
    kpi = _load() if any(s not in dataless for s in stages) else None
    for stage in stages:
        if stage == "rates":
            stage_rates()
        elif stage == "alpha":
            stage_alpha()
        else:
            _STAGES[stage](kpi)


if __name__ == "__main__":
    main(sys.argv)
