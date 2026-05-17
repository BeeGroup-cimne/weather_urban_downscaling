#!/usr/bin/env python3
"""
CatBoost baseline — CORRECTED.
zarr cache stores NORMALIZED data. Use directly, de-normalize once.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config.config import Config as C

CACHE_PATH = C.PATH_CACHE
STATIC_NPY = C.STATIC_CACHE_PATH
STATS_PATH = C.STATS_PATH

# Time splits
TRAIN_END = "2017-11-01"
VAL_START = "2017-11-01"
VAL_END = "2017-12-01"
TEST_START = "2017-12-01"
TEST_END = "2017-12-31"

# CatBoost hyperparams (CR_BCN_meteo best ERA5-Land)
CB_ITERATIONS = 400
CB_LEARN_RATE = 0.08
CB_DEPTH = 8
CB_L2_REG = 6
CB_N_HARMONICS = 4
CB_EARLY_STOP = 30

GRID_STRIDE = 5
OUTPUT_DIR = PROJECT_ROOT / "model_benchmark" / "results" / "catboost_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Loading data ===")
ds = xr.open_zarr(CACHE_PATH)
print(f"  hr_target: {ds.hr_target.shape}")
print(f"  lr_input:  {ds.lr_input.shape}")

static_grid = np.load(STATIC_NPY)
lat1d = ds.latitude.values
lon1d = ds.longitude.values
stats = np.load(STATS_PATH)

lr_mean = stats["mean_lr"]
lr_std = stats["std_lr"]
# IMPORTANT: zarr stores (raw - mean) / std. These are the de-norm params.
hr_mean = float(stats["mean_hr"])   # 18.748 °C
hr_std = float(stats["std_hr"])     # 6.579 °C
lr_var_names = list(stats["lr_var_names"])
print(f"  HR de-norm: mean={hr_mean:.3f}, std={hr_std:.3f}")
print(f"  LR vars: {lr_var_names}")

# Subsampled grid
y_idxs = np.arange(0, 251, GRID_STRIDE)
x_idxs = np.arange(0, 251, GRID_STRIDE)
n_spatial = len(y_idxs) * len(x_idxs)
print(f"  Grid stride={GRID_STRIDE}: {n_spatial} pts/timestep")

sub_lat = (lat1d[y_idxs][:, None] * np.ones(len(x_idxs))[None, :]).ravel()
sub_lon = (np.ones(len(y_idxs))[:, None] * lon1d[x_idxs][None, :]).ravel()
sub_static = static_grid[y_idxs[:, None], x_idxs[None, :], :].reshape(n_spatial, -1)

# Pre-compute nearest LR cell
lr_lat = ds.latitude_lr.values
lr_lon = ds.longitude_lr.values
lr_near_y = np.argmin(np.abs(lr_lat[None, :] - sub_lat[:, None]), axis=1).astype(np.int32)
lr_near_x = np.argmin(np.abs(lr_lon[None, :] - sub_lon[:, None]), axis=1).astype(np.int32)

STATIC_NAMES = [
    "avg_height", "building_density", "elevation", "height_index",
    "industrial_index", "leisure_index", "max_levels", "ndvi_mean",
    "ndvi_min", "residential_index", "roughness", "services_index", "svf"
]

# Pre-normalize static features (percentile 5-95%, clip to [0,1])
static_norm = np.zeros_like(sub_static)
for c in range(len(STATIC_NAMES)):
    col = sub_static[:, c]
    lo, hi = float(np.percentile(col, 5)), float(np.percentile(col, 95))
    static_norm[:, c] = np.clip((col - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0

FEATURE_NAMES = (
    [f"lr_{v}" for v in lr_var_names]
    + [f"static_{n}" for n in STATIC_NAMES]
    + [f"fourier_h{h}_{t}" for h in range(1, CB_N_HARMONICS + 1) for t in ["sin", "cos"]]
    + [f"fourier_m{h}_{t}" for h in range(1, CB_N_HARMONICS + 1) for t in ["sin", "cos"]]
    + ["lat_norm", "lon_norm"]
)


def add_fourier(hours, months, nh=4):
    feats = []
    for period, vals in [(24.0, hours), (12.0, months)]:
        for h in range(1, nh + 1):
            feats.append(np.sin(2 * np.pi * h * vals / period))
            feats.append(np.cos(2 * np.pi * h * vals / period))
    return np.column_stack(feats)


def build_pool(times_slice, label: str):
    """Build Pool from time slice. hr_target is already normalized in zarr."""
    hr_sub = ds.hr_target.sel(time=times_slice).values  # (t, 251, 251) — already normalized
    lr_sub = ds.lr_input.sel(time=times_slice).values  # (t, 5, 4, 9)
    times = pd.DatetimeIndex(ds.time.sel(time=times_slice).values)
    n_t = len(times)
    print(f"  {label}: {n_t} timesteps")

    hr_flat = hr_sub[:, y_idxs[:, None], x_idxs[None, :]].reshape(n_t, -1)  # (t, n_spatial)

    X_parts, Y_parts = [], []
    for t_idx in range(n_t):
        lr_vals = lr_sub[t_idx, lr_near_y, lr_near_x, :]
        lr_norm = (lr_vals - lr_mean) / np.maximum(lr_std, 1e-6)
        fourier = add_fourier(np.full(n_spatial, times[t_idx].hour),
                              np.full(n_spatial, times[t_idx].month), CB_N_HARMONICS)
        coords = np.column_stack([(sub_lat - 41.2) / 0.4, (sub_lon - 1.9) / 0.5])
        X_t = np.column_stack([lr_norm, static_norm, fourier, coords]).astype(np.float32)
        X_parts.append(X_t)
        Y_parts.append(hr_flat[t_idx])

        if t_idx > 0 and t_idx % 1000 == 0:
            print(f"    ... {t_idx}/{n_t}")

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    Y = np.concatenate(Y_parts, axis=0).astype(np.float32)
    print(f"  → X: {X.shape}, Y: {Y.shape}, mem: {X.nbytes/1e6:.0f} MB")
    pool = Pool(X, Y)
    pool.set_feature_names(FEATURE_NAMES)
    return pool


# ---------------------------------------------------------------------------
print("\n=== Building validation pool ===")
val_pool = build_pool(slice(VAL_START, VAL_END), "Validation")

print("\n=== Building test pool ===")
test_pool = build_pool(slice(TEST_START, TEST_END), "Test")


# ---------------------------------------------------------------------------
# Incremental training
# ---------------------------------------------------------------------------
print("\n=== Training CatBoost (incremental monthly) ===")
model = CatBoostRegressor(
    iterations=CB_ITERATIONS,
    learning_rate=CB_LEARN_RATE,
    depth=CB_DEPTH,
    l2_leaf_reg=CB_L2_REG,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=False,
    early_stopping_rounds=CB_EARLY_STOP,
    random_seed=42,
    allow_writing_files=False,
)

months = [
    ("2017-01-01", "2017-02-01"), ("2017-02-01", "2017-03-01"),
    ("2017-03-01", "2017-04-01"), ("2017-04-01", "2017-05-01"),
    ("2017-05-01", "2017-06-01"), ("2017-06-01", "2017-07-01"),
    ("2017-07-01", "2017-08-01"), ("2017-08-01", "2017-09-01"),
    ("2017-09-01", "2017-10-01"), ("2017-10-01", "2017-11-01"),
]

t0 = time.time()
for i, (start, end) in enumerate(months):
    print(f"\n--- Chunk {i+1}/{len(months)}: {start} to {end} ---")
    train_pool = build_pool(slice(start, end), f"Chunk {i+1}")
    if i == 0:
        model.fit(train_pool, eval_set=val_pool, verbose=50, plot=False)
    else:
        model.fit(train_pool, eval_set=val_pool, verbose=50, plot=False,
                  init_model=model)
    val_score = model.get_best_score()["validation"]["RMSE"]
    print(f"  Val RMSE (normalized): {val_score:.6f}")

elapsed = time.time() - t0
best_val_norm = model.get_best_score()["validation"]["RMSE"]
print(f"\n=== Training done in {elapsed/60:.1f} min ===")
print(f"Best val RMSE (norm): {best_val_norm:.6f} → {best_val_norm*hr_std:.4f} °C")


# ---------------------------------------------------------------------------
# Test evaluation (de-normalize to °C)
# ---------------------------------------------------------------------------
print("\n=== Test evaluation (Dec 2017) ===")
preds_norm = model.predict(test_pool)
actuals_norm = test_pool.get_label()

# De-normalize: zarr stores (raw - mean) / std
preds_c = preds_norm * hr_std + hr_mean
actuals_c = actuals_norm * hr_std + hr_mean

rmse_c = root_mean_squared_error(actuals_c, preds_c)
mae_c = mean_absolute_error(actuals_c, preds_c)
bias_c = float(np.mean(preds_c - actuals_c))
r2 = r2_score(actuals_c, preds_c)

print(f"\n{'='*55}")
print(f"  CatBoost Baseline — Test Set (Dec 2017)")
print(f"{'='*55}")
print(f"  RMSE: {rmse_c:.4f} °C")
print(f"  MAE:  {mae_c:.4f} °C")
print(f"  Bias: {bias_c:.4f} °C")
print(f"  R²:   {r2:.4f}")
print(f"  N:    {len(actuals_c):,}")
print(f"{'='*55}")

# Save
results = {
    "model": "CatBoost_CR_BCN_metodo_CORRECTED",
    "rmse_c": round(float(rmse_c), 4),
    "mae_c": round(float(mae_c), 4),
    "bias_c": round(float(bias_c), 4),
    "r2": round(float(r2), 4),
    "best_val_rmse_norm": round(float(best_val_norm), 6),
    "best_val_rmse_c": round(float(best_val_norm * hr_std), 4),
    "training_minutes": round(elapsed / 60, 1),
    "learning_rate": CB_LEARN_RATE,
    "depth": CB_DEPTH,
    "l2_leaf_reg": CB_L2_REG,
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Feature importance
from catboost import Pool as CBPool
train_pool_last = build_pool(slice("2017-10-01", "2017-11-01"), "for_importance")
imp_data = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": model.get_feature_importance(train_pool_last)
}).sort_values("importance", ascending=False)
imp_data.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
print(f"\nTop 10 features:")
print(imp_data.head(10).to_string(index=False))

model.save_model(str(OUTPUT_DIR / "catboost_baseline.cbm"))
print(f"\nDone! Results in {OUTPUT_DIR}")
print(f"\n(Previous buggy results overwritten)")
