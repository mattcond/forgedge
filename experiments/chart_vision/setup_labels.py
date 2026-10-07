"""Unsupervised setup labelling: cluster chart images into "setups".

Idea
----
A discretionary trader names what they see ("flag", "V-bottom", "range") long
before they know what happens next.  Here the vocabulary is learnt without
any target:

  20-candle chart image (2x32x20, autoscaled)
    -> random-kernel CNN features (untrained, so nothing can leak)
    -> standardise -> PCA
    -> KMeans  ->  setup label "S00".."S{k-1}"

* The encoder pipeline (scaler, PCA, KMeans) is fitted on the first
  ``FIT_FRAC`` (50%) of every series, pooled across assets, so it never sees
  data from FORGE's out-of-sample tail (Alpha Discovery's split is at 70%).
* Each bar t gets the label of the chart ending at t (known at t's close).
* The label is a pure function of past prices: it never looks at returns,
  so it satisfies FORGE's invariant #1 and can enter Event Discovery as a
  categorical KPI column (``setup``), which M1 one-hot expands into
  ``is_setup_Sxx`` events.

Outputs (``results/setups/``): one KPI parquet per asset with the ``setup``
column, a prototype figure per setup, and a descriptive table.

Run:  python experiments/chart_vision/setup_labels.py [--k 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from forgedge import build_features, candle_features

import chart_vision as cv   # render() + RandomConvEncoder

DATA = cv.DATA
OUT = Path(__file__).resolve().parent / "results" / "setups"

ASSETS = ["E_SandP-500", "E_DAAX", "EURUSD", "GBPUSD", "E_Brent", "BTCEUR"]
W = 20
FIT_FRAC = 0.5
N_PCA = 10
FIT_SAMPLE = 60000


def load_kpi(asset):
    df = pd.read_csv(DATA / f"{asset}_1HOUR.csv", sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    kpi = candle_features(build_features(df, timestamp_col="timestamp"))
    return kpi.reset_index(drop=True)


def chart_features(kpi, enc):
    o, h, l, c = (kpi[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    ends = np.arange(W - 1, len(kpi))
    idx = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]
    feats = []
    for i in range(0, len(ends), 20000):
        sl = slice(i, i + 20000)
        img = cv.render(o[idx[sl]], h[idx[sl]], l[idx[sl]], c[idx[sl]])
        feats.append(enc.transform(img))
    # simple shape descriptors, used only to *describe* the clusters
    C, H, L = c[idx], h[idx], l[idx]
    lo, hi = L.min(1), H.max(1)
    desc = pd.DataFrame({
        "win_ret": C[:, -1] / C[:, 0] - 1,
        "pos_in_range": (C[:, -1] - lo) / np.where(hi > lo, hi - lo, np.nan),
        "argmax_frac": H.argmax(1) / (W - 1),
        "argmin_frac": L.argmin(1) / (W - 1),
        "range_pct": (hi - lo) / C[:, -1],
    })
    return ends, np.concatenate(feats), desc


def main(k):
    OUT.mkdir(parents=True, exist_ok=True)
    enc = cv.RandomConvEncoder(seed=0)
    per = {}
    for a in ASSETS:
        kpi = load_kpi(a)
        ends, F, desc = chart_features(kpi, enc)
        fit = ends < int(FIT_FRAC * len(kpi))
        per[a] = dict(kpi=kpi, ends=ends, F=F, desc=desc, fit=fit)
        print(f"  {a:12s} bars={len(kpi)} fit={fit.sum()}")

    rng = np.random.default_rng(0)
    Ffit = np.concatenate([p["F"][p["fit"]] for p in per.values()])
    Ffit = Ffit[rng.choice(len(Ffit), min(FIT_SAMPLE, len(Ffit)), replace=False)]
    sc = StandardScaler().fit(Ffit)
    pca = PCA(N_PCA, random_state=0).fit(sc.transform(Ffit))
    Z = pca.transform(sc.transform(Ffit))
    print(f"PCA {N_PCA} comps explain {pca.explained_variance_ratio_.sum():.2f}")

    # silhouette for a few k, for reference (k itself is a user choice)
    sil = {}
    for kk in (6, 8, 10, 12, 16):
        km_ = KMeans(kk, n_init=5, random_state=0).fit(Z)
        sil[kk] = float(silhouette_score(Z[:10000], km_.labels_[:10000]))
    print("silhouette:", {kk: round(v, 3) for kk, v in sil.items()})
    km = KMeans(k, n_init=10, random_state=0).fit(Z)

    rows, protos = [], {}
    for a, p in per.items():
        lab = km.predict(pca.transform(sc.transform(p["F"])))
        setup = np.full(len(p["kpi"]), None, dtype=object)
        setup[p["ends"]] = [f"S{x:02d}" for x in lab]
        kpi = p["kpi"].copy()
        kpi["setup"] = setup
        kpi.to_parquet(OUT / f"{a}_1H_setup.parquet")
        d = p["desc"].assign(setup=lab, asset=a, fit=p["fit"])
        rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    table = D.groupby("setup").agg(
        freq=("win_ret", "size"), win_ret_med=("win_ret", "median"),
        pos_in_range=("pos_in_range", "median"), high_at=("argmax_frac", "median"),
        low_at=("argmin_frac", "median"), range_pct_med=("range_pct", "median"))
    table["freq"] = table["freq"] / table["freq"].sum()
    # stability of the vocabulary: frequency in fit period vs later period
    table["freq_fit"] = D[D.fit].groupby("setup").size() / D.fit.sum()
    table["freq_later"] = D[~D.fit].groupby("setup").size() / (~D.fit).sum()
    table.index = [f"S{i:02d}" for i in table.index]
    table.to_csv(OUT / "setup_table.csv")
    (OUT / "setup_meta.json").write_text(json.dumps(dict(k=k, W=W, fit_frac=FIT_FRAC,
                                                         silhouette=sil, assets=ASSETS), indent=1))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    plot_prototypes(per, km, sc, pca, k)


def plot_prototypes(per, km, sc, pca, k, n_ex=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    a = "E_SandP-500"
    p = per[a]
    lab = km.predict(pca.transform(sc.transform(p["F"])))
    kpi = p["kpi"]
    o, h, l, c = (kpi[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    fig, axes = plt.subplots(k, n_ex, figsize=(2.2 * n_ex, 1.3 * k))
    for s in range(k):
        members = np.where(lab == s)[0]
        # examples closest to the centroid = most typical members
        dist = np.linalg.norm(pca.transform(sc.transform(p["F"][members])) - km.cluster_centers_[s], axis=1)
        pick = members[np.argsort(dist)[:n_ex * 5]]
        pick = rng.choice(pick, min(n_ex, len(pick)), replace=False)
        for j, m in enumerate(pick):
            ax = axes[s, j]
            e = p["ends"][m]
            for q in range(W):
                t = e - W + 1 + q
                col = "#2a9d55" if c[t] >= o[t] else "#d1495b"
                ax.vlines(q, l[t], h[t], color=col, lw=0.7)
                ax.vlines(q, min(o[t], c[t]), max(o[t], c[t]), color=col, lw=2.5)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"S{s:02d}", rotation=0, labelpad=18, fontsize=9)
    plt.suptitle("Setup prototypes (S&P500 1H, members nearest to centroid)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "setup_prototypes.png", dpi=80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    main(ap.parse_args().k)
