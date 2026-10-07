"""Bottom-up formation IDs: "these charts look alike, so they share one ID".

Step 1 of the chart-setup research, deliberately isolated: no target, no
returns, no FORGE — only *which charts look alike*.

Method
------
* Unit: a 20-candle chart (2x32x40 px, autoscaled — ``train_cnn.render``).
  Windows are NON-overlapping (stride 20); overlapping windows shifted by
  one bar would look alike by construction and fake "formations".
* Similarity: Euclidean distance between slightly blurred images (Gaussian
  sigma 0.8 px) so a candle one pixel off still counts as the same drawing.
* Bottom-up grouping: agglomerative clustering, COMPLETE linkage, cut at a
  distance tau.  Complete linkage means every pair of members is within tau
  of each other ("all look like all"), not just chained neighbours.
* tau is expressed as a percentile of the distance between random pairs of
  charts: tau@1% = "more alike than 99% of random pairs".
* An ID needs >= MIN_SIZE members; every other chart stays UNLABELLED.
  Unlike KMeans nothing is forced into a group.
* Fit on the first 70% of every 1HOUR series; the last 30% is only used to
  check whether the same formations recur (nearest-medoid assignment within
  the formation's own radius).

Control
-------
The same procedure on synthetic series built by shuffling each asset's
candles in time (each candle's open/high/low/close relative to the previous
close is kept; their ORDER is destroyed).  Single-candle shapes survive,
multi-candle formations do not.  If the shuffled data yields as many IDs,
the "formations" are what noise looks like at that tau.

Run:  python experiments/chart_vision/formations.py [--tau-pct 1.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist, pdist

import train_cnn as tc   # render(): (N, 32, 40, 2) NHWC, 2 px per candle

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "formations"
W = tc.W                 # 20 candles
FIT_FRAC = 0.70
MIN_SIZE = 10
TAU_PCTS = [0.25, 0.5, 1.0, 2.0, 5.0]
SIGMA = 0.8


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_all():
    out = {}
    for f in sorted(glob.glob(str(tc.DATA / "*_1HOUR.csv"))):
        df = pd.read_csv(f, sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.sort_values("timestamp").reset_index(drop=True)
        out[Path(f).name.replace("_1HOUR.csv", "")] = df
    return out


def shuffled_candles(df, seed):
    """Same candles, random order: rebuild prices from candles expressed vs previous close."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    pc = c[:-1]
    rel = np.stack([o[1:] / pc, h[1:] / pc, l[1:] / pc, c[1:] / pc], 1)
    rel = rel[np.random.default_rng(seed).permutation(len(rel))]
    close = c[0] * np.cumprod(rel[:, 3])
    prev = np.r_[c[0], close[:-1]]
    return pd.DataFrame({"timestamp": df["timestamp"].iloc[1:].to_numpy(),
                         "open": prev * rel[:, 0], "high": prev * rel[:, 1],
                         "low": prev * rel[:, 2], "close": close})


def windows(df):
    n = len(df)
    cut = int(FIT_FRAC * n)
    ends = np.arange(W - 1, n, W)
    split = np.where(ends < cut, "fit", np.where(ends - W + 1 >= cut, "holdout", "gap"))
    keep = split != "gap"
    ends, split = ends[keep], split[keep]
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]
    img = tc.render(o[idx], h[idx], l[idx], c[idx])                 # N,32,40,2
    img = gaussian_filter(img, sigma=(0, SIGMA, SIGMA, 0))
    return img.reshape(len(ends), -1).astype(np.float32), ends, split, idx


def build(dfs, shuffle=False):
    X, meta, paths = [], [], []
    for i, (a, df) in enumerate(dfs.items()):
        d = shuffled_candles(df, seed=i) if shuffle else df
        x, ends, split, idx = windows(d)
        X.append(x)
        meta.append(pd.DataFrame({"asset": a, "end": ends, "split": split,
                                  "end_time": d["timestamp"].to_numpy()[ends]}))
        # OHLC paths normalised to the window range, for plotting only
        o, h, l, c = (d[k].to_numpy(float)[idx] for k in ("open", "high", "low", "close"))
        lo, hi = l.min(1, keepdims=True), h.max(1, keepdims=True)
        rng = np.where(hi > lo, hi - lo, 1)
        paths.append(np.stack([(o - lo) / rng, (h - lo) / rng, (l - lo) / rng, (c - lo) / rng], -1))
    return np.concatenate(X), pd.concat(meta, ignore_index=True), np.concatenate(paths)


# ---------------------------------------------------------------------------
# Bottom-up grouping
# ---------------------------------------------------------------------------

def random_pair_distances(X, n=200_000, seed=0):
    r = np.random.default_rng(seed)
    i, j = r.integers(0, len(X), n), r.integers(0, len(X), n)
    m = i != j
    return np.linalg.norm(X[i[m]] - X[j[m]], axis=1)


def formation_ids(Z, tau):
    lab = fcluster(Z, t=tau, criterion="distance")
    size = np.bincount(lab)
    ok = size[lab] >= MIN_SIZE
    # renumber kept clusters by size: F001 = largest
    kept = sorted(set(lab[ok]), key=lambda k: -size[k])
    ren = {k: i + 1 for i, k in enumerate(kept)}
    return np.array([ren.get(k, 0) for k in lab])      # 0 = unlabelled


def describe(ids, n):
    sizes = np.bincount(ids)[1:]
    return dict(n_ids=int(len(sizes)), coverage=float((ids > 0).mean()),
                largest=int(sizes.max()) if len(sizes) else 0,
                median_size=float(np.median(sizes)) if len(sizes) else 0.0, n=n)


def medoids(X, ids):
    out = {}
    for k in range(1, ids.max() + 1):
        m = np.where(ids == k)[0]
        D = cdist(X[m], X[m])
        j = D.sum(1).argmin()
        out[k] = dict(medoid=m[j], radius=float(D[j].max()), diameter=float(D.max()),
                      members=m)
    return out


def assign(Xh, X, med):
    ks = sorted(med)
    M = X[[med[k]["medoid"] for k in ks]]
    rad = np.array([med[k]["radius"] for k in ks])
    lab = np.zeros(len(Xh), int)
    D = cdist(Xh, M)
    j = D.argmin(1)
    within = D[np.arange(len(j)), j] <= rad[j]
    lab[:] = np.where(within, np.array(ks)[j], 0)
    return lab


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw(ax, p, alpha=1.0, lw=2.2):
    for q in range(len(p)):
        o, h, l, c = p[q]
        col = "#2a9d55" if c >= o else "#d1495b"
        ax.vlines(q, l, h, color=col, lw=0.7, alpha=alpha)
        ax.vlines(q, min(o, c), max(o, c), color=col, lw=lw, alpha=alpha)


def gallery(paths, meta, med, fname, n_show=30, n_ex=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(med)[:n_show]
    fig, axes = plt.subplots(len(ks), n_ex + 1, figsize=(1.9 * (n_ex + 1), 1.15 * len(ks)))
    r = np.random.default_rng(0)
    for row, k in enumerate(ks):
        m = med[k]["members"]
        ax = axes[row, 0]
        for p in paths[m]:
            ax.plot(p[:, 3], color="#3b6ea5", alpha=min(0.15, 3 / len(m)), lw=0.8)
        ax.plot(np.median(paths[m][:, :, 3], 0), color="k", lw=1.4)
        ax.set_ylabel(f"F{k:03d}\nn={len(m)}\n{meta.loc[m, 'asset'].nunique()} asset",
                      rotation=0, labelpad=28, fontsize=7, va="center")
        pick = [med[k]["medoid"]] + list(r.choice(m[m != med[k]["medoid"]], min(n_ex - 1, len(m) - 1), replace=False))
        for j, i in enumerate(pick):
            draw(axes[row, j + 1], paths[i])
            axes[row, j + 1].set_title(f"{meta.at[i, 'asset'][:10]} {str(meta.at[i, 'end_time'])[:10]}",
                                       fontsize=5, pad=1)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("all closes + median", fontsize=7)
    plt.tight_layout(h_pad=0.2, w_pad=0.2)
    plt.savefig(fname, dpi=85)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(tau_pct_main):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dfs = load_all()
    res = {}
    for kind in ("real", "shuffled"):
        X, meta, paths = build(dfs, shuffle=(kind == "shuffled"))
        fit = (meta["split"] == "fit").to_numpy()
        Xf = X[fit]
        if kind == "real":
            rp = random_pair_distances(Xf)
            taus = {p: float(np.percentile(rp, p)) for p in TAU_PCTS}   # same absolute tau for both
        Z = linkage(pdist(Xf), method="complete")
        print(f"[{kind}] {fit.sum()} fit charts, {(~fit).sum()} holdout, linkage {time.time()-t0:.0f}s", flush=True)
        res[kind] = dict(X=X, meta=meta, paths=paths, fit=fit, Z=Z)

    rows = []
    for p, tau in taus.items():
        for kind in ("real", "shuffled"):
            ids = formation_ids(res[kind]["Z"], tau)
            rows.append(dict(tau_pct=p, tau=round(tau, 3), data=kind, **describe(ids, len(ids))))
            res[kind][p] = ids
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "tau_grid.csv", index=False)
    pd.set_option("display.width", 200)
    print(grid.to_string(index=False))

    # main tau: catalogue, recurrence on holdout, galleries
    p = tau_pct_main
    summary = {"tau_pct": p, "tau": taus[p], "min_size": MIN_SIZE}
    for kind in ("real", "shuffled"):
        R = res[kind]
        X, meta, paths, fit = R["X"], R["meta"], R["paths"], R["fit"]
        ids_fit = R[p]
        Xf, mf, pf = X[fit], meta[fit].reset_index(drop=True), paths[fit]
        med = medoids(Xf, ids_fit)
        lab_h = assign(X[~fit], Xf, med)
        f_fit = np.bincount(ids_fit, minlength=len(med) + 1)[1:] / len(ids_fit)
        f_hold = np.bincount(lab_h, minlength=len(med) + 1)[1:] / len(lab_h)
        corr = float(np.corrcoef(f_fit, f_hold)[0, 1]) if len(med) > 2 else float("nan")
        summary[kind] = dict(n_ids=len(med), coverage_fit=float((ids_fit > 0).mean()),
                             coverage_holdout=float((lab_h > 0).mean()), freq_corr_fit_vs_holdout=corr)
        if kind == "real":
            cat = pd.DataFrame([dict(formation=f"F{k:03d}", size=len(v["members"]),
                                     n_assets=mf.loc[v["members"], "asset"].nunique(),
                                     top_asset_share=mf.loc[v["members"], "asset"].value_counts(normalize=True).iloc[0],
                                     radius=v["radius"], diameter=v["diameter"],
                                     freq_fit=f_fit[k - 1], freq_holdout=f_hold[k - 1],
                                     win_ret_median=float(np.median(pf[v["members"], -1, 3] - pf[v["members"], 0, 0])),
                                     medoid_asset=mf.at[v["medoid"], "asset"],
                                     medoid_end=str(mf.at[v["medoid"], "end_time"]))
                                for k, v in med.items()])
            cat.to_csv(OUT / "catalogue.csv", index=False)
            lab = pd.concat([mf.assign(formation=[f"F{k:03d}" if k else "" for k in ids_fit]),
                             meta[~fit].reset_index(drop=True).assign(
                                 formation=[f"F{k:03d}" if k else "" for k in lab_h])])
            lab.drop(columns=["end"]).to_csv(OUT / "labels.csv", index=False)
            print(cat.head(30).round(3).to_string(index=False))
        gallery(pf, mf, med, OUT / f"gallery_{kind}.png", n_show=30 if kind == "real" else 12)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-pct", type=float, default=1.0)
    main(ap.parse_args().tau_pct)
