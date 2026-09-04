"""Null-data / overfitting diagnostic for FORGE on low-frequency (1D) data.

Hypothesis under test: on ~1 year of daily bars, FORGE mines thousands of
candidate events from a few hundred in-sample rows. Even with BH-FDR and a
rotation null, the multiple-testing surface may be large enough that the
pipeline promotes a comparable number of "alphas" on PURE NOISE as on real
data — i.e. it cannot distinguish ADA from a spectrum-matched random series.

Method: Fourier phase randomization of daily log-returns produces surrogates
with the SAME power spectrum (hence linear autocorrelation) and (approx.)
marginal as ADA, but with any genuine feature->forward-return predictability
destroyed. Features are rebuilt with ONE recipe applied identically to the real
close and to every surrogate, so the comparison is apples-to-apples.

We report, real vs surrogates: #event candidates, #promoted alphas.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from forgedge import AlphaConfig, AlphaDiscovery, DiscoveryConfig, EventDiscovery, MarketContext
from forgedge.alpha_discovery.models import PromotionThresholds
from forgedge.event_discovery.models import GateParams

import os
PARQUET = os.environ.get("FORGEDGE_PARQUET", "ADA_1D_FULL.parquet")
RAW = pd.read_parquet(PARQUET).sort_values("open_dt").reset_index(drop=True)
RAW["open_dt"] = pd.to_datetime(RAW["open_dt"])


# ── Feature builder: rebuild the 26 base columns from OHLC ───────────────────
def build_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    d = ohlc[["open_dt", "open", "high", "low", "close"]].copy()
    c, l = d["close"], d["low"]
    for n in (3, 12, 96):
        d[f"close_ret_{n:02d}"] = c.pct_change(n)
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    d["close_bb_mid_20"] = mid
    d["close_bb_upper_20"] = mid + 2 * sd
    d["close_bb_lower_20"] = mid - 2 * sd
    d["close_bb_width_20"] = 4 * sd / mid
    # rolling max-drawdown over 48 (approx; consistent across real/surrogate)
    cv = c.values; mdd = np.full(len(cv), np.nan)
    for i in range(47, len(cv)):
        w = cv[i - 47:i + 1]; pk = np.maximum.accumulate(w)
        mdd[i] = ((pk - w) / pk).max()
    d["close_mdd_48"] = mdd
    for span in (3, 12):
        d[f"close_ema_{span:02d}"] = c.ewm(span=span, adjust=False).mean()
        d[f"low_ema_{span:02d}"] = l.ewm(span=span, adjust=False).mean()
    for p in (1, 2, 3):
        d[f"close_ema_03_prev_{p:02d}"] = d["close_ema_03"].shift(p)
        d[f"close_ema_12_prev_{p:02d}"] = d["close_ema_12"].shift(p)
    r = c.pct_change()
    for n in (5, 12, 24):
        d[f"close_vol_{n:02d}"] = r.rolling(n).std()
    return d


def fidelity_check():
    rebuilt = build_features(RAW)
    cols = [x for x in RAW.columns if x not in ("open_dt", "open", "high", "low", "close")]
    print("Feature rebuild fidelity (max abs diff vs parquet):")
    for x in cols:
        md = np.nanmax(np.abs(rebuilt[x].values - RAW[x].values))
        flag = "  <-- approx" if md > 1e-3 else ""
        print(f"  {x:28s} {md:.2e}{flag}")


# ── Surrogate: Fourier phase randomization of log returns ────────────────────
def phase_randomized_close(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lr = np.diff(np.log(close))
    n = len(lr)
    F = np.fft.rfft(lr)
    mag = np.abs(F)
    phases = rng.uniform(0, 2 * np.pi, size=len(F))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    surr_lr = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    surr_lr = surr_lr * (lr.std() / surr_lr.std())  # preserve realized vol
    p0 = close[0]
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(surr_lr)]))


def iid_shuffled_close(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IID shuffle of log-returns: destroys ALL autocorrelation (strict null)."""
    lr = np.diff(np.log(close))
    sl = rng.permutation(lr)
    return close[0] * np.exp(np.concatenate([[0.0], np.cumsum(sl)]))


def make_surrogate_ohlc(rng: np.random.Generator, mode: str = "phase") -> pd.DataFrame:
    if mode == "iid":
        sc = iid_shuffled_close(RAW["close"].values, rng)
    else:
        sc = phase_randomized_close(RAW["close"].values, rng)
    # intrabar ranges: reuse real (high/close-1) and (1-low/close) shuffled
    hi_r = (RAW["high"].values / RAW["close"].values - 1.0)
    lo_r = (1.0 - RAW["low"].values / RAW["close"].values)
    op_r = (RAW["open"].values / RAW["close"].values - 1.0)
    perm = rng.permutation(len(sc))
    out = pd.DataFrame({
        "open_dt": RAW["open_dt"].values,
        "close": sc,
        "high": sc * (1.0 + np.abs(hi_r[perm])),
        "low": sc * (1.0 - np.abs(lo_r[perm])),
        "open": sc * (1.0 + op_r[perm]),
    })
    return out[["open_dt", "open", "high", "low", "close"]]


def count_promotions(feat_df: pd.DataFrame, label: str) -> dict:
    df_is = feat_df[feat_df["open_dt"] < "2025-01-01"].copy().reset_index(drop=True)
    enriched = MarketContext(df_is.copy()).run()
    # max_and_components=2: this script chains EventDiscovery/AlphaDiscovery
    # standalone (not forge()), so M1's own legacy single-pass composition
    # applies here unconditionally -- unaffected by forge()'s two_pass_
    # composition default (issue #254 Phase 8).
    ed = EventDiscovery(enriched.copy(), DiscoveryConfig(
        gate_params=GateParams(min_tpm=1.5, max_dispersion=3.0),
        timestamp_col="open_dt", max_and_components=2, train_ratio=1.0))
    cands = ed.run()
    ad = AlphaDiscovery(ed.df, cands, AlphaConfig(
        horizon_grid=(1, 2, 3, 5, 7, 10), train_ratio=0.70, asset=label, timeframe="1D",
        thresholds=PromotionThresholds(min_lift=0.06, min_cohens_d=0.10,
                                       use_fdr=True, fdr_q=0.15, oos_max_p=0.15)))
    contracts = ad.run()
    promoted = ad.promoted_contracts()
    return {"candidates": len(cands), "contracts": len(contracts), "promoted": len(promoted)}


if __name__ == "__main__":
    fidelity_check()
    print("\n" + "=" * 64)
    t0 = time.time()
    real = count_promotions(build_features(RAW), "REAL")
    print(f"REAL      : {real}   ({time.time()-t0:.0f}s)")
    n_sur = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sur_stats = []
    for s in range(n_sur):
        t0 = time.time()
        rng = np.random.default_rng(1000 + s)
        st = count_promotions(build_features(make_surrogate_ohlc(rng)), f"SURR{s}")
        sur_stats.append(st)
        print(f"SURR[{s}]   : {st}   ({time.time()-t0:.0f}s)")
    prom = np.array([s["promoted"] for s in sur_stats])
    cand = np.array([s["candidates"] for s in sur_stats])
    print("\n" + "=" * 64)
    print("SUMMARY  (promoted alphas)")
    print(f"  REAL                 : {real['promoted']}")
    print(f"  SURROGATE mean±sd    : {prom.mean():.1f} ± {prom.std():.1f}  (range {prom.min()}–{prom.max()})")
    print(f"  candidates REAL/SURR : {real['candidates']} / {cand.mean():.0f}")
    excess = real["promoted"] - prom.mean()
    print(f"  EXCESS (real-noise)  : {excess:+.1f}")
    if prom.mean() > 0:
        print(f"  real / noise ratio   : {real['promoted']/max(prom.mean(),1e-9):.2f}×")
