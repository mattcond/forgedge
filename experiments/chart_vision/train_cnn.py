"""Trained CNN on chart images — target: long +1% after 10 bars.

Setup (as requested)
--------------------
* Each sample is a chart of W=20 candles, rendered as a 2x32x40 image
  (up/down channels, 2 px per candle, y-axis autoscaled to the window).
* Target: close[t+10] / close[t] - 1 >= +1%, t = last candle in the image.
* Train: first 70% of each series, cut into NON-overlapping 20-bar windows
  (stride 20).  The last 15% of the train windows (chronologically, per asset)
  is used only for early stopping.
* Test: the remaining 30%, also non-overlapping, starting 10 bars after the
  70% cut so that no training label overlaps the test period.
* EURUSD alone has ~7 positives in its 1 238 train windows (base rate 0.6%),
  so the model is trained on the pool of every 1HOUR series in the repo.

Because +1% is a fixed threshold, the base rate differs enormously by asset
(DAX ~5%, NVDA ~41%): a model can score well just by recognising *how
volatile* an asset is.  So we report, next to the pooled AUC:

* ``vol``      baseline: logistic regression on the window's range and
               realised volatility only (what the trader reads off the y-axis)
* ``raw``      the same 20 candles as numbers, gradient boosting
* ``cnn_img``  CNN on the image only (autoscaled -> no scale information)
* ``cnn_img+scale``  CNN on the image + the y-axis scale scalar
* ``cnn+vol``  stacking: does the CNN add anything on top of ``vol``?

and the *within-asset* AUC (mean of per-asset AUCs), which removes the
cross-asset volatility effect and measures timing only.

Target variants (``--target``)
------------------------------
* ``fixed``   (default) close[t+10] / close[t] - 1 >= +1%
* ``barrier`` "+1% before -1%": from the close of the last visible candle,
              walk forward bar by bar; 1 if the high reaches +1% before the
              low reaches -1%, 0 if the opposite.  Bars that touch both levels
              (order unknowable on OHLC) and windows unresolved within
              ``MAX_BARRIER`` bars are dropped.  Train windows must resolve
              before the 70% cut (purge).  This target is volatility-neutral:
              a big move is equally likely to hit either side, so only the
              *direction* is left to predict.

Run:  python experiments/chart_vision/train_cnn.py [--target barrier]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples" / "data"
OUT = Path(__file__).resolve().parent / "results"

W = 20          # candles per image
HZ = 10         # target horizon (bars)
THR = 0.01      # +1%
IMG_H = 32
PX = 2          # pixels per candle -> width 40
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # of train windows, last chronologically, for early stopping
SEEDS = [0, 1, 2]
MAX_BARRIER = 200  # bars allowed for the +1%/-1% barrier to resolve
TARGET = "fixed"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def render(O, H, L, C):
    lo = L.min(1, keepdims=True)
    hi = H.max(1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    px = lambda p: np.clip(((p - lo) / rng * (IMG_H - 1)).round().astype(int), 0, IMG_H - 1)
    ph, pl = px(H), px(L)
    bt, bb = px(np.maximum(O, C)), px(np.minimum(O, C))
    rows = np.arange(IMG_H)[None, :, None]
    wick = (rows >= pl[:, None]) & (rows <= ph[:, None])
    body = (rows >= bb[:, None]) & (rows <= bt[:, None])
    val = np.where(body, 1.0, np.where(wick, 0.5, 0.0)).astype(np.float32)
    up = (C >= O)[:, None, :]
    img = np.stack([val * up, val * ~up], axis=-1)              # N,H,W,2 (NHWC)
    img = np.repeat(img, PX, axis=2)[:, ::-1]                    # 2 px / candle, top = high
    return np.ascontiguousarray(img)


def windows(df, ends):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]
    O, H, L, C = o[idx], h[idx], l[idx], c[idx]
    ref = C[:, -1:]
    raw = np.concatenate([O / ref - 1, H / ref - 1, L / ref - 1, C / ref - 1], 1)
    logret = np.diff(np.log(C), axis=1)
    vol = np.stack([np.log((H.max(1) - L.min(1)) / C[:, -1]),
                    np.log(logret.std(1) + 1e-9),
                    np.log(((H - L) / C).mean(1) + 1e-9)], 1)
    fwd = c[ends + HZ] / c[ends] - 1
    return render(O, H, L, C), raw.astype(np.float32), vol.astype(np.float32), fwd


def barrier(df, ends):
    """+THR before -THR from close[t]. Returns (label in {1,0,-1 unresolved/ambiguous}, resolve index)."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(c)
    lab = np.full(len(ends), -1, dtype=np.int8)
    res = np.full(len(ends), n + MAX_BARRIER, dtype=np.int64)
    for i, t in enumerate(ends):
        stop = min(n, t + 1 + MAX_BARRIER)
        up = h[t + 1:stop] >= c[t] * (1 + THR)
        dn = l[t + 1:stop] <= c[t] * (1 - THR)
        iu = np.argmax(up) if up.any() else MAX_BARRIER + 1
        idn = np.argmax(dn) if dn.any() else MAX_BARRIER + 1
        if iu == idn:                    # same bar or never: unknown order / unresolved
            res[i] = t + 1 + min(iu, MAX_BARRIER)
            continue
        lab[i] = int(iu < idn)
        res[i] = t + 1 + min(iu, idn)
    return lab, res


def build():
    parts = []
    for f in sorted(glob.glob(str(DATA / "*_1HOUR.csv"))):
        asset = Path(f).name.replace("_1HOUR.csv", "")
        df = pd.read_csv(f, sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        cut = int(TRAIN_FRAC * n)
        # non-overlapping windows; t = last visible candle; label must end before the cut
        tr_ends = np.arange(W - 1, cut - HZ, W)
        te_ends = np.arange(cut + W - 1, n - HZ, W)          # first test image starts at cut
        for split, ends in (("train", tr_ends), ("test", te_ends)):
            if TARGET == "barrier":
                lab, res = barrier(df, ends)
                keep = lab >= 0
                if split == "train":
                    keep &= res < cut                          # purge: resolved before the cut
                ends, lab, res = ends[keep], lab[keep], res[keep]
            img, raw, vol, fwd = windows(df, ends)
            if TARGET == "barrier":
                fwd = np.where(lab == 1, THR, -THR)            # realised P&L of the barrier trade
                bars = (res - ends).astype(np.float32)
            sp = np.full(len(ends), split, dtype=object)
            if split == "train":
                sp[int(len(ends) * (1 - VAL_FRAC)):] = "val"
            parts.append(dict(img=img, raw=raw, vol=vol, fwd=fwd, split=sp,
                              bars=bars if TARGET == "barrier" else np.full(len(ends), HZ, np.float32),
                              asset=np.full(len(ends), asset, dtype=object)))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["y"] = (cat["fwd"] >= THR).astype(np.float32)   # barrier: fwd is exactly +/-THR
    return cat


# ---------------------------------------------------------------------------
# CNN (JAX)
# ---------------------------------------------------------------------------

def conv(x, w, b):
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")


def init(key, n_side):
    ks = jax.random.split(key, 6)
    he = lambda k, shape, fan: jax.random.normal(k, shape) * jnp.sqrt(2.0 / fan)
    return {
        "c1": (he(ks[0], (3, 3, 2, 16), 18), jnp.zeros(16)),
        "c2": (he(ks[1], (3, 3, 16, 32), 144), jnp.zeros(32)),
        "c3": (he(ks[2], (3, 3, 32, 64), 288), jnp.zeros(64)),
        # 32x40 -> pool -> 16x20 -> pool -> 8x10 -> pool -> 4x5 ; keep a coarse
        # time axis (5 columns) so the net can tell "recent" from "old" candles
        "d1": (he(ks[3], (64 * 5 + n_side, 32), 64 * 5 + n_side), jnp.zeros(32)),
        "d2": (he(ks[4], (32, 1), 32) * 0.1, jnp.zeros(1)),
    }


def forward(p, x, side):
    h = pool(jax.nn.relu(conv(x, *p["c1"])))
    h = pool(jax.nn.relu(conv(h, *p["c2"])))
    h = pool(jax.nn.relu(conv(h, *p["c3"])))      # N,4,5,64
    h = h.mean(axis=1).reshape(h.shape[0], -1)     # average over price axis, keep time
    h = jnp.concatenate([h, side], axis=1)
    h = jax.nn.relu(h @ p["d1"][0] + p["d1"][1])
    return (h @ p["d2"][0] + p["d2"][1])[:, 0]


def train_cnn(Xtr, Str, ytr, Xva, Sva, yva, seed, epochs=60, bs=128, patience=8):
    key = jax.random.PRNGKey(seed)
    p = init(key, Str.shape[1])
    opt = optax.adamw(1e-3, weight_decay=1e-3)
    st = opt.init(p)
    pos_w = (1 - ytr.mean()) / ytr.mean()           # balance classes in the loss

    def loss_fn(p, x, s, y):
        z = forward(p, x, s)
        l = optax.sigmoid_binary_cross_entropy(z, y)
        return jnp.mean(l * jnp.where(y > 0, pos_w, 1.0))

    @jax.jit
    def step(p, st, x, s, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, s, y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    pred = jax.jit(forward)
    rng = np.random.default_rng(seed)
    best, best_p, bad, hist = -1.0, p, 0, []
    for ep in range(epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            p, st, _ = step(p, st, Xtr[b], Str[b], ytr[b])
        va = roc_auc_score(yva, np.asarray(pred(p, Xva, Sva)))
        hist.append(va)
        if va > best:
            best, best_p, bad = va, p, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_p, pred, best, hist


def predict(pred, p, X, S, bs=2048):
    return np.concatenate([np.asarray(pred(p, X[i:i + bs], S[i:i + bs])) for i in range(0, len(X), bs)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def within_asset_auc(y, s, assets, min_pos=5):
    aucs, wts = [], []
    for a in np.unique(assets):
        m = assets == a
        if y[m].sum() >= min_pos and (1 - y[m]).sum() >= min_pos:
            aucs.append(roc_auc_score(y[m], s[m]))
            wts.append(m.sum())
    return float(np.average(aucs, weights=wts)), len(aucs)


def within_null(y, s, assets, n=200, seed=0):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yp = y.copy()
        for a in np.unique(assets):
            m = np.where(assets == a)[0]
            yp[m] = y[r.permutation(m)]
        out.append(within_asset_auc(yp, s, assets)[0])
    return float(np.mean(out)), float(np.std(out))


def report(name, y, s, fwd, assets):
    q = np.quantile(s, 0.9)
    top = s >= q
    wa, na = within_asset_auc(y, s, assets)
    nm, ns = within_null(y, s, assets, n=100)
    # top-decile *within each asset* (timing only)
    top_w = np.zeros_like(top)
    for a in np.unique(assets):
        m = np.where(assets == a)[0]
        top_w[m] = s[m] >= np.quantile(s[m], 0.9)
    return dict(model=name, auc_pooled=roc_auc_score(y, s), auc_within=wa, n_assets=na,
                within_null_mean=nm, within_null_sd=ns, z_within=(wa - nm) / ns if ns > 0 else float('nan'),
                prec_top10=y[top].mean(), base_rate=y.mean(),
                prec_top10_within=y[top_w].mean(),
                fwd_bps_top10_within=1e4 * fwd[top_w].mean(), fwd_bps_all=1e4 * fwd.mean())


def main():
    t0 = time.time()
    d = build()
    sp, y, A = d["split"], d["y"], d["asset"]
    tr, va, te = sp == "train", sp == "val", sp == "test"
    trva = tr | va
    print(f"train={tr.sum()} val={va.sum()} test={te.sum()}  "
          f"pos rate train={y[trva].mean():.3f} test={y[te].mean():.3f}  ({time.time()-t0:.0f}s)")
    e = A == "EURUSD"
    print(f"EURUSD alone: train windows={(trva & e).sum()} positives={int(y[trva & e].sum())}; "
          f"test windows={(te & e).sum()} positives={int(y[te & e].sum())}")

    if TARGET == "barrier":
        print(f"median bars to resolve: train={np.median(d['bars'][trva]):.0f} test={np.median(d['bars'][te]):.0f}")
    rows, scores = [], {}
    # asset prior: the train win-rate of each asset (captures drift, zero timing by construction)
    prior = pd.Series(y[trva]).groupby(A[trva]).mean()
    scores["asset prior (train win-rate)"] = pd.Series(A).map(prior).fillna(y[trva].mean()).to_numpy()
    sc_vol = StandardScaler().fit(d["vol"][trva])
    V = sc_vol.transform(d["vol"]).astype(np.float32)

    lr = LogisticRegression(max_iter=1000).fit(V[trva], y[trva])
    scores["vol (range+realised vol)"] = lr.decision_function(V)

    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         early_stopping=True, random_state=0)
    hgb.fit(np.hstack([d["raw"], d["vol"]])[trva], y[trva])
    scores["raw candles + vol (HGB)"] = hgb.predict_proba(np.hstack([d["raw"], d["vol"]]))[:, 1]

    X = d["img"]
    empty = np.zeros((len(y), 1), np.float32)
    logs = {}
    for name, S in (("cnn image only", empty), ("cnn image + y-axis scale", V)):
        preds = []
        for seed in SEEDS:
            p, pred, best, hist = train_cnn(X[tr], S[tr], y[tr], X[va], S[va], y[va], seed)
            preds.append(predict(pred, p, X, S))
            logs[f"{name} seed{seed}"] = dict(best_val_auc=best, epochs=len(hist), val_curve=hist)
            print(f"  {name} seed {seed}: best val AUC={best:.3f} after {len(hist)} epochs ({time.time()-t0:.0f}s)")
        scores[name] = np.mean(preds, axis=0)

    # stacking: does the CNN add information on top of the volatility baseline?
    Z = np.column_stack([scores["vol (range+realised vol)"], scores["cnn image + y-axis scale"]])
    st = LogisticRegression().fit(Z[va], y[va])
    scores["stack vol + cnn"] = st.decision_function(Z)

    sfx = "" if TARGET == "fixed" else f"_{TARGET}"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"train_cnn_scores{sfx}.npz", y=y, fwd=d["fwd"], split=sp.astype(str),
                        asset=A.astype(str), **{k.replace(" ", "_"): v for k, v in scores.items()})
    for name, s in scores.items():
        rows.append(report(name, y[te], s[te], d["fwd"][te], A[te]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"train_cnn_results{sfx}.csv", index=False)
    (OUT / f"train_cnn_training_log{sfx}.json").write_text(json.dumps(logs, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))

    # per-asset AUC of the CNN, for inspection
    s = scores["cnn image + y-axis scale"]
    per = []
    for a in np.unique(A):
        m = te & (A == a)
        if y[m].sum() >= 5 and (1 - y[m]).sum() >= 5:
            per.append(dict(asset=a, n=int(m.sum()), pos=int(y[m].sum()),
                            auc_cnn=roc_auc_score(y[m], s[m]),
                            auc_vol=roc_auc_score(y[m], scores["vol (range+realised vol)"][m])))
    per = pd.DataFrame(per)
    per.to_csv(OUT / f"train_cnn_per_asset{sfx}.csv", index=False)
    print(per.round(3).to_string(index=False))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["fixed", "barrier"], default="fixed")
    TARGET = ap.parse_args().target
    main()
