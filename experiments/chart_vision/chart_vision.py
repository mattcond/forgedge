"""Chart-vision experiment — can "what a trader sees on screen" predict the target?

Pipeline
--------
1. For every bar t, take the last W=50 candles and *render* them as a chart
   image (2 channels: up candles / down candles, 32 px tall, 1 px per candle,
   y-axis auto-scaled to the window's own low/high exactly like a chart
   platform does).
2. Encode the image into a feature vector with an image encoder:
     - ``rconv``  random-kernel CNN (conv 5x5 -> ReLU -> global + per-segment
                  pooling); untrained, so it cannot leak anything.
     - ``pca``    linear autoencoder (PCA) fitted on the train fold only.
     - ``ae``     non-linear MLP autoencoder (3200 -> 256 -> 32 -> 256 -> 3200)
                  fitted on the train fold only; the 32-d bottleneck is the
                  feature space.
3. Use the features to predict the target h bars ahead:
     - ``dir``  sign of the forward log return over h bars  (the real question)
     - ``vol``  |forward return| above its train median   (control: a target
                we *know* is predictable — proves the image carries information)
4. Compare against non-visual baselines built from the very same 50 candles:
     - ``raw``  the 50 candles as numbers (normalised OHLC), no rendering
     - ``kpi``  forgedge ``build_features`` indicators made scale-free
   and against a null distribution obtained by circularly shifting the test
   labels within each asset (keeps autocorrelation, destroys alignment).

Validation: pooled multi-asset, expanding walk-forward on calendar time with a
purge so that no training label overlaps the test period.

Run:  python experiments/chart_vision/chart_vision.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from forgedge import build_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples" / "data"
OUT = Path(__file__).resolve().parent / "results"

ASSETS = ["E_SandP-500", "E_DAAX", "EURUSD", "GBPUSD",
          "E_Brent", "COPPER.CMDUSD", "BTCEUR", "ETHEUR"]
W = 50            # candles in the chart
IMG_H = 32        # chart height in pixels
HORIZONS = [6, 24]
STRIDE = 3        # sample every STRIDE bars (limits overlap/memory)
N_FOLDS = 4
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(asset: str, tf: str = "1HOUR") -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{asset}_{tf}.csv", sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def kpi_features(candles: pd.DataFrame) -> pd.DataFrame:
    """forgedge KPI table, made scale-free so it can be pooled across assets."""
    k = build_features(candles, timestamp_col="timestamp")
    close = k["close"]
    out = {}
    for col in k.columns:
        if col in ("timestamp", "open_dt", "ticker", "open", "high", "low",
                   "close", "volume", "color"):
            continue
        s = k[col].astype(float)
        base = col.split("_")[0]
        if any(t in col for t in ("_sma_", "_ema_", "_min_", "_max_", "bb_mid",
                                    "bb_upper", "bb_lower")):
            if base == "volume":
                continue          # volume is meaningless on FX/CFD feeds
            out[col] = close / s - 1.0
        else:
            out[col] = s
    return pd.DataFrame(out, index=k.index)


def render(o, h, l, c) -> np.ndarray:
    """Render windows of candles into (N, 2, IMG_H, W) uint8-ish float images.

    o/h/l/c: (N, W).  Channel 0 = bullish candles, channel 1 = bearish.
    Wick pixels = 0.5, body pixels = 1.0.  y-axis autoscaled per window.
    """
    lo = l.min(axis=1, keepdims=True)
    hi = h.max(axis=1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)

    def px(p):  # price -> row index (0 = bottom)
        return np.clip(((p - lo) / rng * (IMG_H - 1)).round().astype(int), 0, IMG_H - 1)

    ph, pl = px(h), px(l)
    pb_top, pb_bot = px(np.maximum(o, c)), px(np.minimum(o, c))
    rows = np.arange(IMG_H)[None, :, None]                       # (1, H, 1)
    wick = (rows >= pl[:, None, :]) & (rows <= ph[:, None, :])   # (N, H, W)
    body = (rows >= pb_bot[:, None, :]) & (rows <= pb_top[:, None, :])
    val = np.where(body, 1.0, np.where(wick, 0.5, 0.0)).astype(np.float32)
    up = (c >= o)[:, None, :]
    img = np.stack([val * up, val * ~up], axis=1)               # (N, 2, H, W)
    return img[:, :, ::-1, :]                                     # top row = high


def build_dataset():
    imgs, raws, kpis, scales, meta = [], [], [], [], []
    for a in ASSETS:
        df = load(a)
        k = kpi_features(df)
        o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        n = len(df)
        logc = np.log(c)
        ends = np.arange(W - 1, n - max(HORIZONS), STRIDE)   # t = last visible bar
        ends = ends[ends >= 200]                              # KPI warm-up (168-bar windows)
        idx = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]  # (N, W)
        O, H, L, C = o[idx], h[idx], l[idx], c[idx]
        imgs.append(render(O, H, L, C))
        # "raw" baseline: the same 50 candles as numbers, normalised by last close
        ref = C[:, -1:]
        raws.append(np.concatenate([O / ref - 1, H / ref - 1, L / ref - 1, C / ref - 1],
                                   axis=1).astype(np.float32))
        kpis.append(k.iloc[ends].to_numpy(np.float32))
        scales.append(np.log((H.max(1) - L.min(1)) / C[:, -1]).astype(np.float32)[:, None])
        m = pd.DataFrame({"asset": a, "t": df["timestamp"].to_numpy()[ends]})
        for hz in HORIZONS:
            m[f"ret_{hz}"] = logc[ends + hz] - logc[ends]
            m[f"tend_{hz}"] = df["timestamp"].to_numpy()[ends + hz]
        meta.append(m)
        print(f"  {a:15s} {len(ends):6d} samples")
    meta = pd.concat(meta, ignore_index=True)
    X_img = np.concatenate(imgs)
    X_raw = np.concatenate(raws)
    X_kpi = np.nan_to_num(np.concatenate(kpis), nan=0.0, posinf=0.0, neginf=0.0)
    X_scale = np.concatenate(scales)
    return X_img, X_raw, X_kpi, X_scale, meta


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class RandomConvEncoder:
    """Untrained CNN: K random 5x5x2 kernels, ReLU, mean/max + 5 time-segment means."""

    def __init__(self, k: int = 48, ks: int = 5, n_seg: int = 5, seed: int = 0):
        r = np.random.default_rng(seed)
        w = r.normal(size=(k, 2, ks, ks)).astype(np.float32)
        w -= w.mean(axis=(1, 2, 3), keepdims=True)
        self.w = w.reshape(k, -1)                                   # (K, 2*ks*ks)
        self.b = r.uniform(-1, 1, size=k).astype(np.float32)
        self.ks, self.n_seg = ks, n_seg

    def transform(self, X: np.ndarray, batch: int = 2000) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch):
            x = X[i:i + batch]
            p = sliding_window_view(x, (self.ks, self.ks), axis=(2, 3))  # N,2,H',W',ks,ks
            n, ch, hh, ww = p.shape[:4]
            p = p.transpose(0, 2, 3, 1, 4, 5).reshape(n, hh, ww, -1)
            z = np.maximum(p @ self.w.T + self.b, 0)                     # N,H',W',K
            feats = [z.mean(axis=(1, 2)), z.max(axis=(1, 2))]
            for seg in np.array_split(np.arange(ww), self.n_seg):        # time-localised
                feats.append(z[:, :, seg, :].mean(axis=(1, 2)))
            out.append(np.concatenate(feats, axis=1))
        return np.concatenate(out)


class MLPAutoEncoder:
    """3200 -> 256 -> 32 -> 256 -> 3200 autoencoder (sklearn MLP), bottleneck = features."""

    def __init__(self, bottleneck: int = 32, max_iter: int = 40, n_fit: int = 20000):
        self.m = MLPRegressor(hidden_layer_sizes=(256, bottleneck, 256), activation="relu",
                              max_iter=max_iter, batch_size=256, learning_rate_init=1e-3,
                              early_stopping=False, random_state=0)
        self.n_fit = n_fit

    def fit(self, X):
        sel = RNG.choice(len(X), size=min(self.n_fit, len(X)), replace=False)
        self.m.fit(X[sel], X[sel])
        return self

    def transform(self, X):
        a = X
        for W_, b_ in list(zip(self.m.coefs_, self.m.intercepts_))[:2]:
            a = np.maximum(a @ W_ + b_, 0)
        return a

    def recon_r2(self, X):
        p = self.m.predict(X)
        return 1 - ((X - p) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def null_auc(y, p, assets, n=30):
    """AUC under circular shifts of the label within each asset."""
    res = []
    for _ in range(n):
        ys = y.copy()
        for a in np.unique(assets):
            m = np.where(assets == a)[0]
            if len(m) > 10:
                ys[m] = np.roll(y[m], RNG.integers(len(m) // 5, 4 * len(m) // 5))
        res.append(roc_auc_score(ys, p))
    return np.array(res)


def run(quick: bool):
    t0 = time.time()
    print("Building dataset...")
    X_img, X_raw, X_kpi, X_scale, meta = build_dataset()
    if quick:
        keep = np.sort(RNG.choice(len(meta), 20000, replace=False))
        X_img, X_raw, X_kpi, X_scale = X_img[keep], X_raw[keep], X_kpi[keep], X_scale[keep]
        meta = meta.iloc[keep].reset_index(drop=True)
    X_flat = X_img.reshape(len(X_img), -1)
    print(f"samples={len(meta)}  image={X_img.shape[1:]}  ({time.time()-t0:.0f}s)")

    print("Random-conv features...")
    F_rconv = RandomConvEncoder().transform(X_img)
    print(f"  rconv dim={F_rconv.shape[1]} ({time.time()-t0:.0f}s)")

    # calendar-time expanding walk-forward
    t = meta["t"].to_numpy()
    edges = meta["t"].quantile(np.linspace(0.4, 1.0, N_FOLDS + 1)).to_numpy().astype(t.dtype)
    rows, ae_r2 = [], []
    for f in range(N_FOLDS):
        te = (t >= edges[f]) & (t < edges[f + 1]) if f < N_FOLDS - 1 else (t >= edges[f])
        for hz in HORIZONS:
            tr = meta[f"tend_{hz}"].to_numpy() < edges[f]           # purge: label ends before test
            r = meta[f"ret_{hz}"].to_numpy()
            # encoders fitted on train fold only
            feats = {"rconv": F_rconv, "raw": X_raw, "kpi": X_kpi}
            if hz == HORIZONS[0]:
                pca = PCA(32, random_state=0).fit(X_flat[tr][RNG.choice(tr.sum(), min(30000, tr.sum()), replace=False)])
                F_pca = pca.transform(X_flat)
                ae = MLPAutoEncoder(max_iter=15 if quick else 40).fit(X_flat[tr])
                F_ae = ae.transform(X_flat)
                ae_r2.append(float(ae.recon_r2(X_flat[te][:3000])))
                print(f"  fold {f}: PCA expl.var={pca.explained_variance_ratio_.sum():.2f}  AE recon R2(test)={ae_r2[-1]:.2f}")
            feats["pca"] = F_pca
            feats["ae"] = F_ae
            feats["rconv+kpi"] = np.hstack([F_rconv, X_kpi])
            feats["rconv+scale"] = np.hstack([F_rconv, X_scale])
            feats["ae+scale"] = np.hstack([F_ae, X_scale])
            for target in ("dir", "vol"):
                if target == "dir":
                    y = (r > 0).astype(int)
                else:
                    y = (np.abs(r) > np.median(np.abs(r[tr]))).astype(int)
                for name, F in feats.items():
                    sc = StandardScaler().fit(F[tr])
                    Ftr, Fte = np.clip(sc.transform(F[tr]), -8, 8), np.clip(sc.transform(F[te]), -8, 8)
                    for mdl_name, mdl in (
                        ("logit", LogisticRegression(C=0.01, max_iter=500)),
                        ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                               max_leaf_nodes=15, min_samples_leaf=200,
                                                               l2_regularization=1.0, random_state=0)),
                    ):
                        mdl.fit(Ftr, y[tr])
                        p = mdl.predict_proba(Fte)[:, 1]
                        auc = roc_auc_score(y[te], p)
                        row = dict(fold=f, h=hz, target=target, feat=name, model=mdl_name,
                                   n_train=int(tr.sum()), n_test=int(te.sum()), auc=auc)
                        if target == "dir":
                            nul = null_auc(y[te], p, meta["asset"].to_numpy()[te], n=20)
                            row["null_mean"], row["null_sd"] = nul.mean(), nul.std()
                            q_hi, q_lo = np.quantile(p, [0.7, 0.3])
                            row["ls_bps"] = 1e4 * (r[te][p >= q_hi].mean() - r[te][p <= q_lo].mean())
                            row["hit"] = ((p > 0.5) == y[te]).mean()
                            row["base_rate"] = y[te].mean()
                        rows.append(row)
                # supervised end-to-end "vision" net on the pixels themselves (+ y-axis scale)
                sel = np.where(tr)[0]
                sel = RNG.choice(sel, size=min(len(sel), 25000 if not quick else 10000), replace=False)
                Xp = np.hstack([X_flat, (X_scale - X_scale[tr].mean()) / X_scale[tr].std()])
                net = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-2, max_iter=30,
                                    early_stopping=True, validation_fraction=0.15,
                                    n_iter_no_change=4, random_state=0).fit(Xp[sel], y[sel])
                p = net.predict_proba(Xp[te])[:, 1]
                row = dict(fold=f, h=hz, target=target, feat="pixels+scale", model="mlp_e2e",
                           n_train=len(sel), n_test=int(te.sum()), auc=roc_auc_score(y[te], p))
                if target == "dir":
                    nul = null_auc(y[te], p, meta["asset"].to_numpy()[te], n=20)
                    row["null_mean"], row["null_sd"] = nul.mean(), nul.std()
                    q_hi, q_lo = np.quantile(p, [0.7, 0.3])
                    row["ls_bps"] = 1e4 * (r[te][p >= q_hi].mean() - r[te][p <= q_lo].mean())
                    row["hit"] = ((p > 0.5) == y[te]).mean()
                    row["base_rate"] = y[te].mean()
                rows.append(row)
                print(f"  fold {f} h={hz} {target} done ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / ("results_quick.csv" if quick else "results.csv"), index=False)
    (OUT / "ae_recon_r2.json").write_text(json.dumps(ae_r2))
    return res


def summarise(res: pd.DataFrame):
    g = res.groupby(["target", "h", "feat", "model"])
    s = g.agg(auc=("auc", "mean"), auc_min=("auc", "min"), auc_max=("auc", "max"))
    d = res[res.target == "dir"].groupby(["h", "feat", "model"]).agg(
        null_sd=("null_sd", "mean"), ls_bps=("ls_bps", "mean"), hit=("hit", "mean"),
        base=("base_rate", "mean"))
    pd.set_option("display.width", 200)
    print("\n=== Mean AUC across walk-forward folds ===")
    print(s.round(4).to_string())
    print("\n=== Direction target: null sd (circular shift), long-short (top30% - bottom30%) fwd return in bps ===")
    print(d.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    summarise(run(a.quick))
