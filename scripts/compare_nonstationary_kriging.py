#!/usr/bin/env python3
"""
Methods 6, 7 and 9 of the thermal-predictor benchmark — the comparison that can
kill the stratification regime.

  M6  Kriging with LCZ as external drift  -> models the MEAN per class,
                                             ONE global covariance
  M7  Moving-window (smooth non-stationary) kriging, NO LCZ
                                          -> local adaptivity without partition
  M9  Stratified kriging by LCZ           -> per-class MEAN *and* covariance

The claim under test (regime C in SKELETON_THERMAL_PREDICTOR.md) is that the
residual covariance is *partitioned* by urban morphology. That claim survives
only if M9 beats both M6 and M7:

  * M6 wins  -> only the class MEAN matters; partitioned covariance is unnecessary.
  * M7 wins  -> plain local adaptivity suffices; LCZ adds nothing.
  * M9 wins  -> the partition carries information neither alternative captures.

Run this BEFORE investing in the rest of the paper.

Inputs
------
--residuals  .npy of shape (T, H, W) or (H, W): VMamba2 prediction minus reference.
--lcz        LCZ raster aligned to the same grid (default: the Barcelona 100 m map).

Notes on design decisions that matter for the result
----------------------------------------------------
* Variograms are fitted only up to HALF the maximum lag. Beyond that the
  empirical variogram is unreliable and fits run away — which is exactly how the
  earlier per-class run produced a 41.2 km range inside a 35.6 km domain.
* Cross-validation holds out CONTIGUOUS BLOCKS, never random pixels. With random
  pixels, kriging predicts a point from its immediate neighbours and every method
  looks excellent for the wrong reason.
* Kriging is solved in local neighbourhoods (k nearest training points), so cost
  is linear in the number of targets rather than cubic in the field size.
* Rare classes are pooled: any class below --min-class-px is merged into an
  "other" group, because a variogram cannot be fitted from a handful of pixels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_LCZ = Path.home() / "bcn_lst_exposure/data/processed/lcz_barcelona_100m.tif"


# ----------------------------------------------------------------- variograms
def empirical_variogram(coords, values, n_bins=15, max_frac=0.5):
    """Semivariance in distance bins, truncated at max_frac of the maximum lag."""
    n = len(coords)
    idx = np.random.choice(n, min(n, 1500), replace=False)  # pair subsample
    c, v = coords[idx], values[idx]
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    g = 0.5 * (v[:, None] - v[None, :]) ** 2
    iu = np.triu_indices(len(c), k=1)
    d, g = d[iu], g[iu]
    cutoff = d.max() * max_frac
    keep = d <= cutoff
    d, g = d[keep], g[keep]
    if len(d) < 30:
        return None, None
    edges = np.linspace(0, cutoff, n_bins + 1)
    which = np.digitize(d, edges) - 1
    lag, semi = [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() >= 10:
            lag.append(d[m].mean())
            semi.append(g[m].mean())
    return np.array(lag), np.array(semi)


def fit_spherical(lag, semi):
    """Least-squares fit of (nugget, sill, range) to a spherical model."""
    if lag is None or len(lag) < 4:
        return None
    best, best_err = None, np.inf
    sill0 = semi.max()
    for rng in np.linspace(lag[1], lag[-1], 24):
        for nug in np.linspace(0, sill0 * 0.5, 8):
            pred = spherical(lag, nug, sill0 - nug, rng)
            # closed-form scaling of the partial sill given nugget and range
            base = spherical(lag, 0.0, 1.0, rng)
            denom = (base ** 2).sum()
            psill = max(((semi - nug) * base).sum() / denom, 1e-12) if denom > 0 else sill0
            pred = spherical(lag, nug, psill, rng)
            err = ((pred - semi) ** 2).sum()
            if err < best_err:
                best_err, best = err, (nug, psill, rng)
    return best


def spherical(h, nugget, psill, rng):
    h = np.asarray(h, dtype=float)
    out = np.where(
        h <= rng,
        nugget + psill * (1.5 * h / rng - 0.5 * (h / rng) ** 3),
        nugget + psill,
    )
    return np.where(h == 0, 0.0, out)


# -------------------------------------------------------------------- kriging
def ordinary_kriging_local(train_xy, train_v, target_xy, model, k=48):
    """Ordinary kriging in a k-nearest-neighbour window around each target."""
    nug, psill, rng = model
    tree = cKDTree(train_xy)
    k = min(k, len(train_xy))
    _, nn = tree.query(target_xy, k=k)
    nn = np.atleast_2d(nn)
    out = np.empty(len(target_xy))
    for i, idx in enumerate(nn):
        xy, v = train_xy[idx], train_v[idx]
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        gamma = spherical(d, nug, psill, rng)
        d0 = np.sqrt(((xy - target_xy[i]) ** 2).sum(-1))
        g0 = spherical(d0, nug, psill, rng)
        m = len(idx)
        A = np.ones((m + 1, m + 1))
        A[:m, :m] = gamma
        A[m, m] = 0.0
        b = np.append(g0, 1.0)
        try:
            w = np.linalg.solve(A + np.eye(m + 1) * 1e-10, b)[:m]
        except np.linalg.LinAlgError:
            out[i] = v.mean()
            continue
        out[i] = float(w @ v)
    return out


def moving_window_kriging(train_xy, train_v, target_xy, k=48, refit_every=200):
    """M7: local variogram refitted in windows; no class information used."""
    tree = cKDTree(train_xy)
    k = min(k, len(train_xy))
    _, nn = tree.query(target_xy, k=k)
    nn = np.atleast_2d(nn)
    out = np.empty(len(target_xy))
    model = None
    for i, idx in enumerate(nn):
        if i % refit_every == 0:
            lag, semi = empirical_variogram(train_xy[idx], train_v[idx], n_bins=8)
            fitted = fit_spherical(lag, semi)
            model = fitted or model
        if model is None:
            out[i] = train_v[idx].mean()
            continue
        out[i] = ordinary_kriging_local(
            train_xy[idx], train_v[idx], target_xy[i: i + 1], model, k=k
        )[0]
    return out


# ------------------------------------------------------------------- protocol
def block_folds(xy, n_blocks=4):
    """Contiguous spatial blocks; fold i holds out block i."""
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, n_blocks // 2 + 1))
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, n_blocks // 2 + 1))
    bx = np.clip(np.digitize(xy[:, 0], qx[1:-1]), 0, n_blocks // 2 - 1)
    by = np.clip(np.digitize(xy[:, 1], qy[1:-1]), 0, n_blocks // 2 - 1)
    return bx * (n_blocks // 2) + by


def run(res, lcz, min_class_px, n_blocks, sample, seed):
    rng_ = np.random.default_rng(seed)
    H, W = res.shape
    yy, xx = np.mgrid[0:H, 0:W]
    valid = np.isfinite(res) & (lcz > 0)  # class 0 treated as nodata/water

    xy = np.column_stack([xx[valid], yy[valid]]).astype(float) * 0.1  # km
    v = res[valid].astype(float)
    cls = lcz[valid].astype(int)

    counts = {c: int((cls == c).sum()) for c in np.unique(cls)}
    rare = [c for c, n in counts.items() if n < min_class_px]
    cls = np.where(np.isin(cls, rare), -1, cls)
    print(f"classes kept: {sorted(set(cls.tolist()) - {-1})} | pooled as 'other': {rare}")

    if sample and len(v) > sample:
        pick = rng_.choice(len(v), sample, replace=False)
        xy, v, cls = xy[pick], v[pick], cls[pick]

    folds = block_folds(xy, n_blocks)
    scores = {m: [] for m in ("M6_lcz_drift", "M7_moving_window", "M9_stratified",
                              "OK_global")}

    for f in range(n_blocks):
        te, tr = folds == f, folds != f
        if te.sum() < 20 or tr.sum() < 200:
            continue
        xtr, vtr, ctr = xy[tr], v[tr], cls[tr]
        xte, vte, cte = xy[te], v[te], cls[te]

        # --- baseline: one global mean, one global covariance
        gmod = fit_spherical(*empirical_variogram(xtr, vtr - vtr.mean()))
        if gmod is None:
            continue
        pred_ok = vtr.mean() + ordinary_kriging_local(xtr, vtr - vtr.mean(), xte, gmod)
        scores["OK_global"].append(rmse(pred_ok, vte))

        # --- M6: class MEAN as drift, single global covariance on the residual
        means = {c: vtr[ctr == c].mean() for c in np.unique(ctr)}
        gm = vtr.mean()
        dtr = vtr - np.array([means.get(c, gm) for c in ctr])
        m6mod = fit_spherical(*empirical_variogram(xtr, dtr)) or gmod
        pred_m6 = np.array([means.get(c, gm) for c in cte]) + \
            ordinary_kriging_local(xtr, dtr, xte, m6mod)
        scores["M6_lcz_drift"].append(rmse(pred_m6, vte))

        # --- M7: local variogram, no class information
        pred_m7 = moving_window_kriging(xtr, vtr - gm, xte) + gm
        scores["M7_moving_window"].append(rmse(pred_m7, vte))

        # --- M9: per-class mean AND per-class covariance
        pred_m9 = np.empty(len(xte))
        for c in np.unique(cte):
            sel_te = cte == c
            sel_tr = ctr == c
            if sel_tr.sum() < 60:                      # fall back where support is thin
                pred_m9[sel_te] = pred_m6[sel_te]
                continue
            mc = vtr[sel_tr].mean()
            cmod = fit_spherical(*empirical_variogram(xtr[sel_tr], vtr[sel_tr] - mc))
            if cmod is None:
                pred_m9[sel_te] = pred_m6[sel_te]
                continue
            pred_m9[sel_te] = mc + ordinary_kriging_local(
                xtr[sel_tr], vtr[sel_tr] - mc, xte[sel_te], cmod)
        scores["M9_stratified"].append(rmse(pred_m9, vte))
        print(f"  fold {f}: " + "  ".join(
            f"{k}={vv[-1]:.4f}" for k, vv in scores.items() if vv))

    return scores


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--residuals", required=True,
                   help=".npy (T,H,W) or (H,W) of prediction-minus-reference")
    p.add_argument("--lcz", default=str(DEFAULT_LCZ))
    p.add_argument("--min-class-px", type=int, default=400)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--sample", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--groups", action="store_true",
                   help="collapse raw LCZ codes into 6 morphological groups")
    p.add_argument("--out", default="results/nonstationary_kriging_comparison.json")
    a = p.parse_args()

    res = np.load(a.residuals)
    if res.ndim == 3:
        print(f"residuals {res.shape} -> averaging over {res.shape[0]} time steps")
        res = np.nanmean(res, axis=0)

    try:
        import rasterio
        with rasterio.open(a.lcz) as s:
            lcz = s.read(1)
    except Exception:
        from PIL import Image
        lcz = np.array(Image.open(a.lcz))

    if lcz.shape != res.shape:
        # nearest-neighbour resample onto the residual grid, matching the
        # convention already used by variogram_test.py (scipy.ndimage.zoom, order=0)
        from scipy.ndimage import zoom
        print(f"resampling LCZ {lcz.shape} -> {res.shape} (nearest)")
        lcz = zoom(lcz.astype(float),
                   (res.shape[0] / lcz.shape[0], res.shape[1] / lcz.shape[1]),
                   order=0).astype(int)
        if lcz.shape != res.shape:  # zoom can be off by one
            lcz = lcz[:res.shape[0], :res.shape[1]]

    if a.groups:
        # collapse raw LCZ codes into the morphological groups used elsewhere
        groups = {1: [1, 2, 3], 2: [4, 5, 6], 3: [8, 10], 4: [9],
                  5: [11, 12, 14, 15], 6: [16, 17]}
        names = {1: "Built compact", 2: "Built open", 3: "Industrial",
                 4: "Sparse", 5: "Vegetation", 6: "Dense trees"}
        out = np.zeros_like(lcz)
        for g, codes in groups.items():
            out[np.isin(lcz, codes)] = g
        lcz = out
        print("grouped LCZ:", {names[g]: int((lcz == g).sum()) for g in groups})

    scores = run(res, lcz, a.min_class_px, a.n_blocks, a.sample, a.seed)

    print("\n=== block-CV RMSE (lower is better) ===")
    summary = {}
    for k, vv in scores.items():
        if vv:
            summary[k] = {"mean": float(np.mean(vv)), "std": float(np.std(vv)),
                          "folds": len(vv)}
            print(f"  {k:20s} {np.mean(vv):.5f} +/- {np.std(vv):.5f}  (n={len(vv)})")

    if "M9_stratified" in summary:
        print("\n=== verdict on regime C ===")
        means = {k: summary[k]["mean"] for k in summary}
        spread = max(means.values()) - min(means.values())
        pooled_sd = float(np.mean([summary[k]["std"] for k in summary]))
        print(f"  spread between methods : {spread:.5f}")
        print(f"  between-fold sd (pooled): {pooled_sd:.5f}")
        print(f"  signal / noise          : {spread/pooled_sd:.2f}")

        # Point estimates alone are not evidence: compare paired per-fold results
        # and require the gap to clear the fold-to-fold noise.
        m9_folds = np.array(scores["M9_stratified"])
        verdict_lines = []
        for rival in ("M6_lcz_drift", "M7_moving_window", "OK_global"):
            if not scores.get(rival):
                continue
            d = m9_folds - np.array(scores[rival])
            verdict_lines.append(
                f"  M9 vs {rival:18s} mean diff {d.mean():+.5f} | "
                f"folds favouring M9: {(d < 0).sum()}/{len(d)}")
        print("\n".join(verdict_lines))

        decisive = spread > pooled_sd
        beats_all = all(
            (m9_folds - np.array(scores[r])).mean() < 0 and
            (m9_folds < np.array(scores[r])).sum() > len(m9_folds) / 2
            for r in ("M6_lcz_drift", "M7_moving_window", "OK_global")
            if scores.get(r))
        print()
        if decisive and beats_all:
            print("  -> Regime C SURVIVES: the partition beats every alternative on a "
                  "majority of folds, by more than the fold-to-fold noise.")
        elif not decisive:
            print("  -> NO DETECTABLE EFFECT: the spread between methods is smaller "
                  "than the between-fold noise. Stratification is not distinguishable "
                  "from plain ordinary kriging at this sample size. Regime C is NOT "
                  "supported; do not report a ranking from these point estimates.")
        else:
            print("  -> Regime C NOT supported: an alternative matches or beats the "
                  "partition. Reframe around the mean/drift term.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
