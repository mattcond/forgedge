"""Measure what switching ``entry_mode`` does to the verdicts (issue #185).

The step-5 change makes ``"auto"`` the default entry mode, which moves the
goldens.  A golden that moves needs a *reason*, and the reason has to be a
measurement, not an argument — so this script runs the same fixture under both
modes and reports the deltas the issue's acceptance criteria ask for:

    verdict distribution · median PF · fill rate · median DSR
    grid cells · rules adopting the limit point
    the two DSR channels, separately

The two DSR channels matter independently and are easy to conflate:

  1. **A smaller haircut.**  ``deflated_sharpe`` applies
     ``sqrt(1 - gamma*ln(n_trials)/ln(n_obs))``.  Stage 1 of ``"auto"`` collapses
     ``buy_drop_pct`` and ``buy_delay_bar`` to one value each (they are inert at
     next-open fill), so the grid goes 75 cells -> 15 and the haircut shrinks.

  2. **A defined DSR where there was none.**  The radicand goes negative — DSR
     undefined, which blocks a full EDGE in ``_decide`` — when
     ``n_obs <= n_trials ** gamma``.  At 75 trials that is every rule with
     <= 12 trades; at 15 it is <= 4.  A rule unblocked this way did not get a
     *better* statistic, it got one at all.

     On this fixture the channel turns out to be **empty**, and the reason is
     worth knowing: a rule with 12 trades is eliminated by the dynamic
     trade-count floor (``max(10, n_months * min_tpm)``) before ``validate()``
     is ever called, so it never has a Deflated Sharpe to be undefined.  The
     two mechanisms overlap almost exactly, and the floor gets there first.

Run::

    python examples/entry_mode_impact.py
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import median

import pandas as pd

from forgedge import RuleDiscoveryConfig, forge

_GAMMA = 0.5772156649  # Euler-Mascheroni, as in rule_discovery.validation

_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ADA_1D_TRAIN.parquet"


def _run(kpi: pd.DataFrame, entry_mode: str):
    return forge(
        kpi,
        ticker="ADAUSDC",
        timeframe="1D",
        rule_discovery_config=RuleDiscoveryConfig(entry_mode=entry_mode),
        run_rule_discovery=True,
        run_registry=False,
        progress=False,
        strict=False,
    )


def _dsr_undefined_below(n_trials: int) -> float:
    """Trade count at or below which the DSR radicand goes negative.

    ``1 - gamma*ln(n_trials)/ln(n_obs) <= 0``
      <=>  ``ln(n_obs) <= gamma*ln(n_trials)``
      <=>  ``n_obs <= n_trials ** gamma``

    Note the exponent is ``gamma``, not ``1/gamma``: 75 trials makes the DSR
    undefined for rules with <= 12 trades, 15 trials for <= 4.  (Inverting it
    gives 1772 and 109, which "explains" every rule on any realistic fixture
    and is how you end up reporting a channel that is not there.)
    """
    if n_trials <= 1:
        return 0.0
    return math.exp(_GAMMA * math.log(n_trials))


def _profile(result, label: str) -> dict:
    rows = []
    for contract, resp in result.rule_responses:
        sv = resp.statistical_validation
        s = resp.in_sample_summary
        opt = resp.entry_optimization
        rows.append({
            "alpha_id": contract.alpha_id,
            "verdict": resp.verdict,
            "pf": s.profit_factor if s else float("nan"),
            "fill_rate": s.fill_rate if s else float("nan"),
            "trades": s.total_trades if s else 0,
            # `validated` separates the two reasons a DSR can be missing.  A
            # rule rejected by the early screen never had one computed; only a
            # rule that reached `validate()` and came back with nan has the
            # *undefined* DSR channel 2 is about.  Conflating them would report
            # the early-elimination rate as a DSR effect.
            "validated": sv is not None,
            "dsr": sv.deflated_sharpe if sv else float("nan"),
            "n_trials": sv.n_trials_tested if sv else float("nan"),
            "grid_cells": len(resp.grid_results or []),
            "adopted_limit": bool(opt and opt.adopted),
            # What the *old* criterion would have decided, from the numbers this
            # run already recorded: adopt whenever a candidate cleared the fill
            # floor in-sample and beat the market point's `pf_score_tpm`.
            # Computed inside the same run rather than by diffing against a
            # previous build — nothing here depends on trusting two versions of
            # the code to have been identical everywhere else.
            "old_would_adopt": bool(
                opt is not None
                and opt.limit_pf_score_tpm is not None
                and opt.limit_pf_score_tpm > opt.market_pf_score_tpm
            ),
            "reached_stage_two": opt is not None,
            "failed_condition": (opt.failed_condition if opt else None),
        })
    df = pd.DataFrame(rows)
    val = df[df["validated"]]
    finite = lambda d, col: [v for v in d[col] if v == v]  # noqa: E731 — drop NaN
    med = lambda xs: median(xs) if len(xs) else float("nan")  # noqa: E731
    prof = {
        "label": label,
        "n_rules": len(df),
        "n_validated": len(val),
        "verdicts": dict(Counter(df["verdict"])),
        "median_pf": med(finite(df, "pf")),
        "median_fill": med(finite(df, "fill_rate")),
        "median_trades": med(list(df["trades"])),
        "median_dsr": med(finite(val, "dsr")),
        "dsr_undefined": int((val["dsr"] != val["dsr"]).sum()),
        "median_grid_cells": med(list(df["grid_cells"])),
        "median_n_trials": med(finite(val, "n_trials")),
        "adopted_limit": int(df["adopted_limit"].sum()),
    }
    return prof, df


def main() -> None:
    kpi = pd.read_parquet(_FIXTURE)

    print("=" * 78)
    print("  entry_mode impact — ADA 1D fixture (issue #185)")
    print("=" * 78)

    profiles = {}
    frames = {}
    for mode in ("limit", "auto"):
        res = _run(kpi, mode)
        profiles[mode], frames[mode] = _profile(res, mode)
        print(f"\n[{mode}] {profiles[mode]['n_rules']} rules evaluated")

    print("\n─── verdicts ───")
    keys = sorted({k for p in profiles.values() for k in p["verdicts"]})
    print(f"{'verdict':<20}{'limit':>10}{'auto':>10}{'delta':>10}")
    for k in keys:
        a = profiles["limit"]["verdicts"].get(k, 0)
        b = profiles["auto"]["verdicts"].get(k, 0)
        print(f"{k:<20}{a:>10}{b:>10}{b - a:>+10}")

    print("\n─── metrics (median over evaluated rules) ───")
    print(f"{'metric':<20}{'limit':>12}{'auto':>12}")
    for key in ("median_pf", "median_fill", "median_trades", "median_dsr",
                "median_grid_cells", "median_n_trials", "dsr_undefined",
                "adopted_limit"):
        a, b = profiles["limit"][key], profiles["auto"][key]
        fa = f"{a:.4f}" if isinstance(a, float) else str(a)
        fb = f"{b:.4f}" if isinstance(b, float) else str(b)
        print(f"{key:<20}{fa:>12}{fb:>12}")

    print("\n─── the two DSR channels, separately ───")
    for mode in ("limit", "auto"):
        df = frames[mode][frames[mode]["validated"]]
        nt = profiles[mode]["median_n_trials"]
        cutoff = _dsr_undefined_below(int(nt)) if nt == nt else 0.0
        blocked = int((df["trades"] <= cutoff).sum())
        print(f"[{mode}] n_trials={nt:.0f} → DSR undefined for rules with "
              f"<= {cutoff:.0f} trades; {blocked}/{len(df)} validated rules are at or below")

    both = frames["limit"].merge(frames["auto"], on="alpha_id", suffixes=("_l", "_a"))
    both = both[both["validated_l"] & both["validated_a"]]
    defined_both = both[(both["dsr_l"] == both["dsr_l"]) & (both["dsr_a"] == both["dsr_a"])]
    if len(defined_both):
        lift = (defined_both["dsr_a"] - defined_both["dsr_l"])
        print(f"\nChannel 1 (smaller haircut): {len(defined_both)} rules with a DSR "
              f"in both modes, median lift {lift.median():+.4f}")
    newly = both[(both["dsr_l"] != both["dsr_l"]) & (both["dsr_a"] == both["dsr_a"])]
    print(f"Channel 2 (undefined → defined): {len(newly)} rules, among "
          f"{len(both)} validated in both modes")

    auto = frames["auto"]
    stage_two = auto[auto["reached_stage_two"]]
    if len(stage_two):
        old_yes = stage_two["old_would_adopt"]
        new_yes = stage_two["adopted_limit"]
        print(f"\n─── adoption criterion, old vs new (within the same run) ───")
        print(f"  reached Stage 2                : {len(stage_two)}")
        print(f"  old criterion would adopt      : {int(old_yes.sum())}")
        print(f"  new criterion adopts           : {int(new_yes.sum())}")
        print(f"  old yes → new no               : {int((old_yes & ~new_yes).sum())}")
        print(f"  old no  → new yes              : {int((~old_yes & new_yes).sum())}")
        blocked = stage_two[old_yes & ~new_yes]["failed_condition"]
        if len(blocked):
            print("  of those, stopped by:")
            for cond, k in blocked.value_counts().items():
                print(f"    {cond:<10} {k}")

    changed = both[both["verdict_l"] != both["verdict_a"]]
    print(f"\n─── {len(changed)} verdict(s) changed ───")
    for _, r in changed.iterrows():
        print(f"  {r['alpha_id']}: {r['verdict_l']} → {r['verdict_a']} "
              f"(PF {r['pf_l']:.2f}→{r['pf_a']:.2f}, fill {r['fill_rate_l']:.2f}→"
              f"{r['fill_rate_a']:.2f}, trades {r['trades_l']}→{r['trades_a']})")


if __name__ == "__main__":
    main()
