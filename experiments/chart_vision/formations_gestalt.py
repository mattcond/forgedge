"""Bottom-up formation IDs with a *gestalt* similarity (step 1, option 2).

Two charts share an ID if their overall drawing is the same, even when single
candles differ and phases last a bit longer in one than in the other.

Representation
--------------
* Price path of the 20-candle window: candle midpoint (high+low)/2, scaled
  to the window's own low..high (what an autoscaled chart shows).
* Piecewise aggregate approximation to P=10 points (one point per 2 candles):
  candle-level detail is gone, the shape stays.

Similarity
----------
* ``dtw``  Dynamic Time Warping with a Sakoe-Chiba band of R=2 points: a phase
           may stretch or shrink by up to 2 points (4 candles) — a wider or
           narrower head-and-shoulders is still the same formation — but a
           rise cannot be warped into a fall.
* ``euc``  plain Euclidean on the same 10 points (no elasticity), to isolate
           what warping adds.

Grouping (as in ``formations.py``)
----------------------------------
complete linkage cut at tau = a percentile of the random-pair distance,
ID only with >= MIN_SIZE members, everything else unlabelled; fitted on the
first 70% of each of the 33 1HOUR series, non-overlapping windows.

Is a formation more than chance?
--------------------------------
Every chart — real fit, real holdout, and the same windows on candle-shuffled
series (candle shapes kept, order destroyed) — is assigned to the nearest
ID medoid if within that ID's radius.  For each ID:

  excess = freq(real) / freq(shuffled)

tested with a one-sided binomial test, BH-FDR across IDs, *on the fit
period only*.  IDs that pass are then checked on the holdout (last 30%),
which played no part in building or selecting them.

Run:  python experiments/chart_vision/formations_gestalt.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist
from scipy.stats import binomtest

import formations as fm   # load_all, shuffled_candles, draw

OUT = fm.HERE / "results" / "formations_gestalt"
W = 20
P = 10
R = 2
FIT_FRAC = 0.70
MIN_SIZE = 10
TAU_PCTS = [0.5, 1.0, 2.0, 5.0]
TAU_MAINS = [1.0, 2.0, 5.0]
FDR_Q = 0.05
MIN_EXCESS = 1.5


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

def windows(df):
    n = len(df)
    cut = int(FIT_FRAC * n)
    ends = np.arange(W - 1, n, W)
    split = np.where(ends < cut, "fit", np.where(ends - W + 1 >= cut, "holdout", "gap"))
    keep = split != "gap"
    ends, split = ends[keep], split[keep]
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]
    O, H, L, C = o[idx], h[idx], l[idx], c[idx]
    lo, hi = L.min(1, keepdims=True), H.max(1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    mid = ((H + L) / 2 - lo) / rng
    paa = mid.reshape(len(ends), P, W // P).mean(2)
    ohlc = np.stack([(O - lo) / rng, (H - lo) / rng, (L - lo) / rng, (C - lo) / rng], -1)
    return paa.astype(np.float64), ends, split, ohlc


def build(dfs, shuffle):
    X, meta, paths = [], [], []
    for i, (a, df) in enumerate(dfs.items()):
        d = fm.shuffled_candles(df, seed=i) if shuffle else df
        x, ends, split, ohlc = windows(d)
        X.append(x)
        paths.append(ohlc)
        meta.append(pd.DataFrame({"asset": a, "split": split,
                                  "end_time": d["timestamp"].to_numpy()[ends]}))
    return np.concatenate(X), pd.concat(meta, ignore_index=True), np.concatenate(paths)


# ---------------------------------------------------------------------------
# DTW (numba)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _dtw(a, b, r):
    n = a.shape[0]
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(n, i + r) + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, n])


@nb.njit(parallel=True, cache=True)
def dtw_pdist(X, r):
    n = X.shape[0]
    out = np.empty(n * (n - 1) // 2)
    for i in nb.prange(n - 1):
        base = i * n - i * (i + 1) // 2
        for j in range(i + 1, n):
            out[base + j - i - 1] = _dtw(X[i], X[j], r)
    return out


@nb.njit(parallel=True, cache=True)
def dtw_cdist(A, B, r):
    out = np.empty((A.shape[0], B.shape[0]))
    for i in nb.prange(A.shape[0]):
        for j in range(B.shape[0]):
            out[i, j] = _dtw(A[i], B[j], r)
    return out


def dist_fns(metric):
    if metric == "dtw":
        return (lambda X: dtw_pdist(X, R)), (lambda A, B: dtw_cdist(A, B, R))
    return (lambda X: pdist(X)), (lambda A, B: cdist(A, B))


# ---------------------------------------------------------------------------
# Grouping + evaluation
# ---------------------------------------------------------------------------

def formation_ids(Z, tau):
    lab = fcluster(Z, t=tau, criterion="distance")
    size = np.bincount(lab)
    kept = sorted({k for k in lab if size[k] >= MIN_SIZE}, key=lambda k: -size[k])
    ren = {k: i + 1 for i, k in enumerate(kept)}
    return np.array([ren.get(k, 0) for k in lab])


def medoids(X, ids, cd):
    out = {}
    for k in range(1, ids.max() + 1):
        m = np.where(ids == k)[0]
        D = cd(X[m], X[m])
        j = D.sum(1).argmin()
        out[k] = dict(medoid=m[j], radius=float(D[j].max()), members=m)
    return out


def assign(Xn, M, rad, cd):
    out = np.zeros(len(Xn), int)
    for i in range(0, len(Xn), 5000):
        D = cd(Xn[i:i + 5000], M)
        j = D.argmin(1)
        ok = D[np.arange(len(j)), j] <= rad[j]
        out[i:i + 5000] = np.where(ok, j + 1, 0)
    return out


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def excess_table(lab_real, n_real, lab_null, n_null, K):
    cr = np.bincount(lab_real, minlength=K + 1)[1:]
    cn = np.bincount(lab_null, minlength=K + 1)[1:]
    share = n_real / (n_real + n_null)       # expected share of real hits under H0
    p = np.array([binomtest(int(a), int(a + b), share, alternative="greater").pvalue if a + b else 1.0
                  for a, b in zip(cr, cn)])
    ratio = (cr / n_real) / np.maximum(cn / n_null, 0.5 / n_null)
    return cr, cn, ratio, p


def run_metric(metric, R_, N_):
    pd_fn, cd_fn = dist_fns(metric)
    t0 = time.time()
    Xf = R_["X"][R_["fit"]]
    Z = linkage(pd_fn(Xf), method="complete")
    Zn = linkage(pd_fn(N_["X"][N_["fit"]]), method="complete")
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(Xf), 20000), rng.integers(0, len(Xf), 20000)
    rp = np.array([cd_fn(Xf[a:a + 1], Xf[b:b + 1])[0, 0] for a, b in zip(i[:4000], j[:4000]) if a != b])
    taus = {p: float(np.percentile(rp, p)) for p in TAU_PCTS}
    print(f"[{metric}] linkage done {time.time()-t0:.0f}s", flush=True)

    grid = []
    for p, tau in taus.items():
        for kind, ZZ, n in (("real", Z, len(Xf)), ("shuffled", Zn, N_["fit"].sum())):
            ids = formation_ids(ZZ, tau)
            sizes = np.bincount(ids)[1:]
            grid.append(dict(metric=metric, tau_pct=p, tau=round(tau, 4), data=kind, n_ids=len(sizes),
                             coverage=(ids > 0).mean(), largest=int(sizes.max()) if len(sizes) else 0))
    grid = pd.DataFrame(grid)

    results = {}
    for tm in TAU_MAINS:
        # main tau: IDs built on real fit, every chart assigned with the same rule
        ids = formation_ids(Z, taus[tm])
        med = medoids(Xf, ids, cd_fn)
        K = len(med)
        if K == 0:
            continue
        M = Xf[[med[k]["medoid"] for k in range(1, K + 1)]]
        rad = np.array([med[k]["radius"] for k in range(1, K + 1)])
        lab = {}
        for kind, S in (("real", R_), ("shuffled", N_)):
            for sp in ("fit", "holdout"):
                m = (S["meta"]["split"] == sp).to_numpy()
                lab[(kind, sp)] = (assign(S["X"][m], M, rad, cd_fn), m.sum())
        cr, cn, ratio, p = excess_table(*lab[("real", "fit")], *lab[("shuffled", "fit")], K)
        crh, cnh, ratio_h, p_h = excess_table(*lab[("real", "holdout")], *lab[("shuffled", "holdout")], K)
        q = bh(p)
        selected = (q < FDR_Q) & (ratio >= MIN_EXCESS)
        mf = R_["meta"][R_["fit"]].reset_index(drop=True)
        cat = pd.DataFrame(dict(
            formation=[f"G{k:03d}" for k in range(1, K + 1)],
            size=[len(med[k]["members"]) for k in range(1, K + 1)],
            n_assets=[mf.loc[med[k]["members"], "asset"].nunique() for k in range(1, K + 1)],
            radius=rad,
            hits_real_fit=cr, hits_shuf_fit=cn, excess_fit=ratio, p_fit=p, q_fit=q, selected=selected,
            hits_real_hold=crh, hits_shuf_hold=cnh, excess_hold=ratio_h, p_hold=p_h))
        cov = {f"{k[0]}_{k[1]}": float((v[0] > 0).mean()) for k, v in lab.items()}
        sel = cat[cat.selected]
        summary = dict(metric=metric, tau_pct=tm, tau=taus[tm], n_ids=K, coverage=cov,
                       n_selected=int(selected.sum()),
                       selected_confirmed_holdout=int(((sel.excess_hold > 1) & (sel.p_hold < 0.05)).sum()),
                       selected_excess_hold_median=float(sel.excess_hold.median()) if len(sel) else None,
                       all_ids_excess_fit_median=float(np.median(ratio)))
        results[tm] = (cat, summary, med)
    mf = R_["meta"][R_["fit"]].reset_index(drop=True)
    return grid, results, mf


def gallery(paths, meta, med, cat, fname, title, n_show=24, n_ex=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = cat.sort_values(["selected", "excess_fit"], ascending=False).head(n_show)
    fig, axes = plt.subplots(len(rows), n_ex + 1, figsize=(1.9 * (n_ex + 1), 1.15 * len(rows)), squeeze=False)
    r = np.random.default_rng(0)
    for row, (_, c) in enumerate(rows.iterrows()):
        k = int(c.formation[1:])
        m = med[k]["members"]
        ax = axes[row, 0]
        mids = (paths[m][:, :, 1] + paths[m][:, :, 2]) / 2
        for y in mids:
            ax.plot(y, color="#3b6ea5", alpha=min(0.15, 3 / len(m)), lw=0.8)
        ax.plot(np.median(mids, 0), color="k", lw=1.4)
        tag = "★ " if c.selected else ""
        ax.set_ylabel(f"{tag}{c.formation}\nn={c['size']} · {c.n_assets} asset\nexcess fit {c.excess_fit:.1f}×"
                      f"\nexcess hold {c.excess_hold:.1f}×", rotation=0, labelpad=36, fontsize=6, va="center")
        pick = [med[k]["medoid"]] + list(r.choice(m[m != med[k]["medoid"]], min(n_ex - 1, len(m) - 1), replace=False))
        for j, i in enumerate(pick):
            fm.draw(axes[row, j + 1], paths[i])
            axes[row, j + 1].set_title(f"{meta.at[i, 'asset'][:10]} {str(meta.at[i, 'end_time'])[:10]}", fontsize=5, pad=1)
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=8)
    plt.tight_layout(h_pad=0.2, w_pad=0.2)
    plt.savefig(fname, dpi=85)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dfs = fm.load_all()
    S = {}
    for kind in ("real", "shuffled"):
        X, meta, paths = build(dfs, shuffle=(kind == "shuffled"))
        S[kind] = dict(X=X, meta=meta, paths=paths, fit=(meta["split"] == "fit").to_numpy())
    print(f"charts: real fit={S['real']['fit'].sum()} holdout={(~S['real']['fit']).sum()} ({time.time()-t0:.0f}s)", flush=True)

    grids, summaries = [], []
    for metric in ("dtw", "euc"):
        grid, results, mf = run_metric(metric, S["real"], S["shuffled"])
        grids.append(grid)
        pf = S["real"]["paths"][S["real"]["fit"]]
        for tm, (cat, summary, med) in results.items():
            summaries.append(summary)
            cat.to_csv(OUT / f"catalogue_{metric}_tau{tm:g}.csv", index=False)
            gallery(pf, mf, med, cat, OUT / f"gallery_{metric}_tau{tm:g}.png",
                    f"{metric.upper()} tau@{tm:g}% — ★ = excess over shuffled significant on fit "
                    f"(BH q<{FDR_Q}, ≥{MIN_EXCESS}×)")
            print(json.dumps(summary, indent=1, default=float), flush=True)
            pd.set_option("display.width", 220)
            print(cat.sort_values("excess_fit", ascending=False).head(12).round(3).to_string(index=False), flush=True)
    grid = pd.concat(grids)
    grid.to_csv(OUT / "tau_grid.csv", index=False)
    print(grid.round(4).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=1, default=float))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
