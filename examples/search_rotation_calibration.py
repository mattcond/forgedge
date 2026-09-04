"""Search-level rotation null for FORGE — contract-preserving alpha calibration.

Question: is the best alpha FORGE finds on the real asset stronger than the best
it can find when the SAME search runs against a decoupled outcome?  This tells a
real edge from one manufactured by the multiple-testing surface (thousands of
mined candidates + horizon scan + AND composition) without ever rebuilding the
features — so it never breaks FORGE's contract (the user passes a pre-built
feature table and nothing else).

How it works
------------
1. Run the real pipeline once: MarketContext -> EventDiscovery -> AlphaDiscovery.
   Keep the candidate set and the real promoted contracts (sorted by
   composite_score).
2. K times, circularly ROTATE only the ``close`` column of the post-ED table by
   a random offset and re-run *only* AlphaDiscovery, reusing the real candidate
   set.  Rotation keeps the exact marginal distribution and autocorrelation of
   the returns (one wrap seam), but breaks the alignment between each event's
   (real, cached) activations and the forward outcome.  Record the MAXIMUM
   composite_score over all directed contracts in that run.
3. The K maxima form the null distribution of "the best alpha the pipeline can
   manufacture when event and outcome are decoupled".  A real alpha is credible
   only if its composite_score sits above that distribution; we report an
   empirical p-value (1 + #{null_max >= score}) / (1 + K).

Why reusing the real candidate set is legitimate (and cheap): in FORGE mode
EventDiscovery mines events from FEATURES and gates on activation
frequency/dispersion — it is outcome-independent.  All outcome-dependent
selection (target derivation, scoring, OOS) lives in AlphaDiscovery, which we
re-run against the rotated outcome.  So the candidate set is the same under real
and null, and we only pay K AlphaDiscovery runs, not K full pipelines.

Usage:
    FORGEDGE_PARQUET=/path/ADA_1D_FULL.parquet \
        python examples/search_rotation_calibration.py [K]
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from forgedge import AlphaConfig, AlphaDiscovery, DiscoveryConfig, EventDiscovery, MarketContext
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.event_discovery.models import GateParams

PARQUET = os.environ.get(
    "FORGEDGE_PARQUET",
    "/root/.claude/uploads/97a39385-fb76-5589-adb0-bb331ce97505/39e65776-ADA_1D_FULL.parquet",
)
CLOSE_COL = "close"
MIN_TPM = float(os.environ.get("FORGEDGE_MIN_TPM", "1.5"))


def build_alpha_config(label: str) -> AlphaConfig:
    """Same config used in the 1D diagnostics — daily-scaled horizon grid."""
    return AlphaConfig(
        horizon_grid=(1, 2, 3, 5, 7, 10), train_ratio=0.70,
        asset=label, timeframe="1D",
        thresholds=PromotionThresholds(min_lift=0.06, min_cohens_d=0.10,
                                       use_fdr=True, fdr_q=0.15, oos_max_p=0.15),
    )


def rotate_close(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return a copy with ONLY the close column circularly rotated by k bars.

    Features, regime and the index are left untouched, so each event's cached
    activation series (real) is paired with a time-shifted outcome.  The single
    wrap seam corrupts ~h forward returns; with a random offset per draw it adds
    noise to the null max but, if anything, *inflates* it (large seam jumps make
    the null harder to beat) — a conservative direction.
    """
    out = frame.copy()
    out[CLOSE_COL] = np.roll(frame[CLOSE_COL].to_numpy(), k)
    return out


def max_composite(contracts) -> float:
    scores = [c.alpha_score.composite_score for c in contracts
              if c.direction in ("long", "short")]
    return float(max(scores)) if scores else float("nan")


# Several candidate yardsticks, all "higher = stronger".  composite_score is
# clamped to [0, 1] so it saturates at the top (a real 1.0 is indistinguishable
# from a noise 1.0); the others are UNBOUNDED, so the best real alpha can pull
# clear of the null.  We record the max over directed contracts of each.
def max_statistics(contracts) -> dict:
    directed = [c for c in contracts if c.direction in ("long", "short")]
    def _max(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(max(vals)) if vals else float("nan")
    def _zstar(c):
        return abs(c.derived_target.t_stat_by_h.get(
            c.derived_target.holding_period_h, float("nan")))
    return {
        "composite": _max([c.alpha_score.composite_score for c in directed]),
        "abs_z":     _max([_zstar(c) for c in directed]),
        "is_lift":   _max([c.event_stats.lift for c in directed]),
        "is_t":      _max([c.event_stats.t_stat for c in directed]),
        "oos_t":     _max([c.oos_validation.t_stat for c in directed
                           if c.oos_validation is not None]),
    }


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    raw = pd.read_parquet(PARQUET).sort_values("open_dt").reset_index(drop=True)
    raw["open_dt"] = pd.to_datetime(raw["open_dt"])
    df_is = raw[raw["open_dt"] < "2025-01-01"].copy().reset_index(drop=True)
    print(f"IS window: {len(df_is)} bars "
          f"({df_is['open_dt'].iloc[0].date()} -> {df_is['open_dt'].iloc[-1].date()})")

    # ── 1. Real pipeline (once) ──────────────────────────────────────────────
    t0 = time.time()
    enriched = MarketContext(df_is.copy()).run()
    # max_and_components=2: this script chains EventDiscovery/AlphaDiscovery
    # standalone (not forge()), so M1's own legacy single-pass composition
    # applies here unconditionally -- unaffected by forge()'s two_pass_
    # composition default (issue #254 Phase 8).
    ed = EventDiscovery(enriched.copy(), DiscoveryConfig(
        gate_params=GateParams(min_tpm=MIN_TPM, max_dispersion=3.0),
        timestamp_col="open_dt", max_and_components=2, train_ratio=1.0))
    cands = ed.run()
    base_frame = ed.df  # post-ED table: features + regime + close, DatetimeIndex
    print(f"min_tpm = {MIN_TPM}  ->  activation floor ~ {MIN_TPM*12:.0f} over 12 IS months")

    ad_real = AlphaDiscovery(base_frame, cands, build_alpha_config("ADA_REAL"))
    ad_real.run()
    real_promoted = sorted(ad_real.promoted_contracts(),
                           key=lambda c: c.alpha_score.composite_score, reverse=True)
    real_stats = max_statistics(real_promoted)
    print(f"\nReal pipeline: {len(cands)} candidates, "
          f"{len(real_promoted)} directed contracts  ({time.time()-t0:.0f}s)")
    print("Real best per statistic:")
    for name, val in real_stats.items():
        print(f"  {name:10s} = {val:.4f}")
    print("Real top-5 promoted (by composite):")
    for c in real_promoted[:5]:
        dt = c.derived_target
        zc = abs(dt.t_stat_by_h.get(dt.holding_period_h, float("nan")))
        print(f"  comp={c.alpha_score.composite_score:.3f} |z|={zc:.2f} "
              f"is_t={c.event_stats.t_stat:.2f} {c.direction:5s} h={dt.holding_period_h:>2} "
              f"lift={c.event_stats.lift:+.3f}  {c.event_expression[:48]}")

    # ── 2. K search-level rotations of the outcome ───────────────────────────
    n = len(base_frame)
    rng = np.random.default_rng(20260624)
    lo, hi = 10, n - 10  # non-trivial rotation both ways
    null = {k: [] for k in real_stats}
    print(f"\nRunning {K} rotation draws ...")
    for j in range(K):
        k = int(rng.integers(lo, hi))
        t0 = time.time()
        rot = rotate_close(base_frame, k)
        ad = AlphaDiscovery(rot, cands, build_alpha_config(f"ROT{j}"))
        ad.run()
        st = max_statistics(ad.promoted_contracts())
        for name, v in st.items():
            null[name].append(v)
        print(f"  draw {j:>2}  shift={k:>4}  "
              f"comp={st['composite']:.3f} lift={st['is_lift']:.3f} "
              f"|z|={st['abs_z']:.2f} is_t={st['is_t']:.2f} oos_t={st['oos_t']:.2f}  "
              f"({time.time()-t0:.0f}s)")

    # ── 3. Verdict, per statistic ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SEARCH-LEVEL ROTATION NULL — which yardstick separates real from noise?")
    print("=" * 70)
    print(f"{'statistic':10s} {'real':>9} {'null mean':>10} {'null sd':>8} "
          f"{'null q95':>9} {'emp p':>7}  verdict")
    for name in real_stats:
        arr = np.array([x for x in null[name] if np.isfinite(x)])
        real_v = real_stats[name]
        if arr.size == 0 or not np.isfinite(real_v):
            print(f"{name:10s}   (insufficient data)")
            continue
        q95 = np.quantile(arr, 0.95)
        ge = int(np.sum(arr >= real_v))
        p = (1 + ge) / (1 + arr.size)
        flag = "SEPARATES" if p < 0.05 else ("borderline" if p < 0.10 else "saturated/weak")
        print(f"{name:10s} {real_v:9.4f} {arr.mean():10.4f} {arr.std():8.4f} "
              f"{q95:9.4f} {p:7.4f}  {flag}")

    # Detail on the best discriminating statistic (largest real-vs-null gap in sds)
    gaps = {}
    for name in real_stats:
        arr = np.array([x for x in null[name] if np.isfinite(x)])
        if arr.size and arr.std() > 0 and np.isfinite(real_stats[name]):
            gaps[name] = (real_stats[name] - arr.mean()) / arr.std()
    if gaps:
        best = max(gaps, key=gaps.get)
        arr = np.array([x for x in null[best] if np.isfinite(x)])
        bar = np.quantile(arr, 0.95)
        print(f"\nBest discriminator: '{best}'  (real is {gaps[best]:+.1f} sd above null mean)")
        def _stat(c, name):
            if name == "abs_z":
                return abs(c.derived_target.t_stat_by_h.get(
                    c.derived_target.holding_period_h, float("nan")))
            if name == "is_lift": return c.event_stats.lift
            if name == "is_t":    return c.event_stats.t_stat
            if name == "oos_t":   return (c.oos_validation.t_stat
                                          if c.oos_validation else float("nan"))
            return c.alpha_score.composite_score
        survivors = [c for c in real_promoted
                     if np.isfinite(_stat(c, best)) and _stat(c, best) > bar]
        print(f"Real promoted above q95 null bar ({best} > {bar:.3f}): "
              f"{len(survivors)} / {len(real_promoted)}")
        for c in sorted(survivors, key=lambda c: _stat(c, best), reverse=True)[:12]:
            es = c.event_stats
            oos = c.oos_validation
            oos_s = (f"OOS n={oos.n_activations} WR={oos.win_rate:.2f} "
                     f"p={oos.p_value:.3f} pass={oos.passed}") if oos else "OOS n/a"
            print(f"  {best}={_stat(c, best):.3f} IS n_act={es.n_activations:>2} "
                  f"WR={es.win_rate:.2f} | {oos_s} | {c.direction:5s} "
                  f"{c.event_expression[:42]}")

    # Stage 1 (selection): Tippett over in-sample yardsticks only.
    tippett_combination(real_stats, null, which=IN_SAMPLE_STATS,
                        label="in-sample only — OOS held out")
    # For contrast: the (incorrect) version that mixes OOS into selection.
    tippett_combination(real_stats, null,
                        which=("composite", "abs_z", "is_lift", "is_t", "oos_t"),
                        label="all 5 incl. oos_t — double-dips OOS")


# OOS is held out of the SELECTION stage on purpose: using it to choose the
# discriminant would burn the holdout's independence (model selection on the
# holdout = the holdout stops being a holdout).  The Tippett combination runs
# over IN-SAMPLE yardsticks only; OOS is applied once afterwards as a fixed
# independent confirmation gate (two-stage test on disjoint data partitions).
IN_SAMPLE_STATS = ("composite", "abs_z", "is_lift", "is_t")


def tippett_combination(real_stats: dict, null: dict,
                        which=IN_SAMPLE_STATS, label: str = "in-sample only") -> None:
    """Tippett (min-p) combination across the given yardsticks.

    For each draw j, compute the empirical p-value of null[s][j] against the
    full null array of s (inclusive, conservative), then take the minimum over
    statistics.  The real Tippett statistic is min_s p_s(real).  The null
    distribution of the combined statistic is the K minima computed the same
    way.  Empirical p = (1 + #{min_p_null <= min_p_real}) / (1+K).

    Why this is the honest version of "pick the best statistic": the selection
    is paid for by the combination step — null draws that look good on *any*
    single yardstick also produce small min-p and compete against the real.
    No threshold on n_act, no choice of discriminant: the data picks.

    ``which`` restricts the combination to a subset of statistics; by default
    only the in-sample yardsticks, so OOS stays a sealed confirmation gate.
    """
    valid = {s: np.array([x for x in null[s] if np.isfinite(x)])
             for s in which
             if s in real_stats
             and any(np.isfinite(x) for x in null[s]) and np.isfinite(real_stats[s])}
    if not valid:
        return

    def ep(val, arr):
        return (1 + int(np.sum(arr >= val))) / (1 + len(arr))

    # Per-statistic p-values for the real run
    p_real = {s: ep(real_stats[s], arr) for s, arr in valid.items()}
    min_p_real = min(p_real.values())
    best_s = min(p_real, key=p_real.get)

    # Null distribution of min-p: for each draw j take min over statistics
    K_eff = min(len(arr) for arr in valid.values())
    min_p_null = []
    for j in range(K_eff):
        p_j = {s: ep(arr[j], arr) for s, arr in valid.items() if j < len(arr)}
        if p_j:
            min_p_null.append(min(p_j.values()))
    min_p_null = np.array(min_p_null)

    # Count null draws at least as extreme (min-p is "smaller = stronger")
    p_tippett = (1 + int(np.sum(min_p_null <= min_p_real))) / (1 + len(min_p_null))
    q05 = np.quantile(min_p_null, 0.05)

    print("\n" + "=" * 70)
    print(f"TIPPETT (min-p) COMBINATION — adaptive multi-statistic test [{label}]")
    print("=" * 70)
    print("Per-statistic empirical p (real vs rotation null):")
    for s, p in sorted(p_real.items(), key=lambda x: x[1]):
        flag = " <-- drives min-p" if s == best_s else ""
        print(f"  {s:10s}  p={p:.4f}{flag}")
    print(f"\nTippett min-p (real) : {min_p_real:.4f}  (driven by '{best_s}')")
    print(f"Null min-p  mean     : {min_p_null.mean():.4f}   q05={q05:.4f}")
    print(f"Tippett empirical p  : {p_tippett:.4f}  "
          f"({'SEPARATES' if p_tippett < 0.05 else 'borderline' if p_tippett < 0.10 else 'weak'})")
    print(f"({int(np.sum(min_p_null <= min_p_real))}/{len(min_p_null)} null draws "
          f"produced a min-p <= {min_p_real:.4f})")


if __name__ == "__main__":
    main()
