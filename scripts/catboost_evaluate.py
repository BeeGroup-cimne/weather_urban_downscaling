#!/usr/bin/env python3
"""
Compute exp01-compatible metrics from CatBoost predictions.
Run AFTER catboost_benchmark.py completes.

Produces: residual03_mean_c, t03/t14 means, bias, morphology R²
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config.config import Config as C

# Paths
CACHE_PATH = C.PATH_CACHE
STATIC_NPY = C.STATIC_CACHE_PATH
MODEL_DIR = PROJECT_ROOT / "model_benchmark" / "results" / "catboost_baseline"
MODEL_PATH = MODEL_DIR / "catboost_baseline.cbm"
N_HARMONICS = 4

# Load config
GRID_STRIDE = 5
y_idxs = np.arange(0, 251, GRID_STRIDE)
x_idxs = np.arange(0, 251, GRID_STRIDE)
n_spatial = len(y_idxs) * len(x_idxs)

# Load data
ds = xr.open_zarr(CACHE_PATH)
static_grid = np.load(STATIC_NPY)
lat2d = ds.latitude.values
lon2d = ds.longitude.values
stats = np.load(C.STATS_PATH)

lr_mean = stats["mean_lr"]
lr_std = stats["std_lr"]
hr_mean = float(stats["mean_hr"])
hr_std = float(stats["std_hr"])
lr_var_names = list(stats["lr_var_names"])

# Subsampled grid
sub_lat = lat2d[y_idxs][:, None] * np.ones(len(x_idxs))[None, :]
sub_lon = np.ones(len(y_idxs))[:, None] * lon2d[x_idxs][None, :]
sub_lat = sub_lat.ravel()
sub_lon = sub_lon.ravel()
sub_static = static_grid[y_idxs[:, None], x_idxs[None, :], :].reshape(n_spatial, -1)

# Nearest LR
lr_lat = ds.latitude_lr.values
lr_lon = ds.longitude_lr.values
lr_near_y = np.argmin(np.abs(lr_lat[None, :] - sub_lat[:, None]), axis=1).astype(np.int32)
lr_near_x = np.argmin(np.abs(lr_lon[None, :] - sub_lon[:, None]), axis=1).astype(np.int32)

STATIC_NAMES = [
    "avg_height", "building_density", "elevation", "height_index",
    "industrial_index", "leisure_index", "max_levels", "ndvi_mean",
    "ndvi_min", "residential_index", "roughness", "services_index", "svf"
]

# Normalize static
static_norm = np.zeros_like(sub_static)
for c in range(len(STATIC_NAMES)):
    col = sub_static[:, c]
    lo, hi = float(np.percentile(col, 5)), float(np.percentile(col, 95))
    if hi > lo:
        static_norm[:, c] = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    else:
        static_norm[:, c] = 0.0

def add_fourier(hours, months, nh=4):
    feats = []
    for period, vals in [(24.0, hours), (12.0, months)]:
        for h in range(1, nh + 1):
            feats.append(np.sin(2 * np.pi * h * vals / period))
            feats.append(np.cos(2 * np.pi * h * vals / period))
    return np.column_stack(feats)

# Load model
print("Loading model...")
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

# Select test period: Dec 2017
test_da = ds.hr_target.sel(time=slice("2017-12-01", "2017-12-31"))
test_lr = ds.lr_input.sel(time=slice("2017-12-01", "2017-12-31"))
times = pd.DatetimeIndex(test_da.time.values)
n_t = len(times)
print(f"Test period: {n_t} timesteps")

# Filter 03:00 and 14:00 timesteps
night_mask = np.array([t.hour == 3 for t in times])
day_mask = np.array([t.hour == 14 for t in times])
night_indices = np.where(night_mask)[0]
day_indices = np.where(day_mask)[0]
print(f"  Night (03:00): {len(night_indices)} timesteps")
print(f"  Day (14:00): {len(day_indices)} timesteps")

# Predict for ALL timesteps
print("Predicting...")
all_preds = []
all_targets = []

for t_idx in range(n_t):
    hr_sub = test_da.values[t_idx, y_idxs[:, None], x_idxs[None, :]].reshape(-1)
    lr_vals = test_lr.values[t_idx, lr_near_y, lr_near_x, :]
    lr_norm = (lr_vals - lr_mean) / np.maximum(lr_std, 1e-6)
    fourier = add_fourier(np.full(n_spatial, times[t_idx].hour),
                          np.full(n_spatial, times[t_idx].month), N_HARMONICS)
    coords = np.column_stack([(sub_lat - 41.2) / 0.4, (sub_lon - 1.9) / 0.5])
    X_t = np.column_stack([lr_norm, static_norm, fourier, coords]).astype(np.float32)
    pred_norm = model.predict(X_t)
    pred = pred_norm * hr_std + hr_mean
    all_preds.append(pred)
    all_targets.append(hr_sub)

    if t_idx > 0 and t_idx % 500 == 0:
        print(f"  ... {t_idx}/{n_t}")

preds = np.array(all_preds)  # (t, n_spatial)
targets = np.array(all_targets)

# -----------------------------------------------------------------------
# Metrics (exp01-compatible)
# -----------------------------------------------------------------------
# 1. Nighttime residual (03:00) − the key metric
night_pred = preds[night_indices]  # (n_nights, n_spatial)
night_target = targets[night_indices]
residual03 = np.mean(np.abs(night_pred - night_target))
print(f"\n=== Key Metrics ===")
print(f"  residual03_mean_c (MAE @ 03:00): {residual03:.4f} °C")

# 2. Daytime residual (14:00)
day_pred = preds[day_indices]
day_target = targets[day_indices]
residual14 = np.mean(np.abs(day_pred - day_target))
print(f"  residual14_mean_c (MAE @ 14:00): {residual14:.4f} °C")

# 3. Mean temperatures
t03_model = np.mean(night_pred)
t14_model = np.mean(day_pred)
t03_target = np.mean(night_target)
t14_target = np.mean(day_target)
print(f"\n=== Mean Temperatures ===")
print(f"  t03_model: {t03_model:.3f} °C (target: {t03_target:.3f})")
print(f"  t14_model: {t14_model:.3f} °C (target: {t14_target:.3f})")
print(f"  delta (t03-t14) model: {t03_model - t14_model:.3f} °C")
print(f"  delta (t03-t14) target: {t03_target - t14_target:.3f} °C")

# 4. Bias vs HR target
bias_vs_hr = np.mean(preds - targets)
print(f"\n  bias_vs_hr (overall): {bias_vs_hr:.4f} °C")

# 5. Overall pixel-wise metrics
rmse_all = np.sqrt(np.mean((preds - targets)**2))
mae_all = np.mean(np.abs(preds - targets))
print(f"\n=== Overall Pixel-wise ===")
print(f"  RMSE (all hours): {rmse_all:.4f} °C")
print(f"  MAE (all hours):  {mae_all:.4f} °C")

# 6. Morphology gradients − nighttime residual vs building density
bd = sub_static[:, 1]  # building_density is index 1
from sklearn.linear_model import LinearRegression

def morphology_r2(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    lr = LinearRegression().fit(x[mask].reshape(-1, 1), y[mask])
    return lr.score(x[mask].reshape(-1, 1), y[mask])

night_residual_mean = np.mean(np.abs(night_pred - night_target), axis=0)  # (n_spatial,)
r2_bd = morphology_r2(bd, night_residual_mean)

ah = sub_static[:, 0]  # avg_height is index 0
r2_ah = morphology_r2(ah, night_residual_mean)

print(f"\n=== Morphology Gradients ===")
print(f"  R² (residual vs building_density): {r2_bd:.4f}")
print(f"  R² (residual vs avg_height): {r2_ah:.4f}")

# 7. Save
results = {
    "residual03_mean_c": round(residual03, 4),
    "residual14_mean_c": round(residual14, 4),
    "t03_model_mean_c": round(t03_model, 4),
    "t14_model_mean_c": round(t14_model, 4),
    "t03_target_mean_c": round(t03_target, 4),
    "t14_target_mean_c": round(t14_target, 4),
    "delta_model_c": round(t03_model - t14_model, 4),
    "delta_target_c": round(t03_target - t14_target, 4),
    "bias_vs_hr_c": round(bias_vs_hr, 4),
    "rmse_all_hours_c": round(rmse_all, 4),
    "mae_all_hours_c": round(mae_all, 4),
    "r2_building_density": round(r2_bd, 4),
    "r2_avg_height": round(r2_ah, 4),
    "n_test_timesteps": n_t,
    "n_spatial_points": n_spatial,
}

out_path = MODEL_DIR / "benchmark_metrics.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# Comparison table
print(f"\n{'='*70}")
print(f"  Comparison: CatBoost vs Mamba vs ConvLSTM")
print(f"{'='*70}")
print(f"  {'Metric':<30} {'CatBoost':>12} {'Mamba_seq12':>12} {'ConvLSTM_seq6':>12}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'residual03_mean_c (°C)':<30} {residual03:>12.4f} {'1.2361':>12} {'2.8832':>12}")
print(f"  {'bias_vs_hr (°C)':<30} {bias_vs_hr:>12.4f} {'-1.5216':>12} {'-0.1492':>12}")
print(f"  {'bias_vs_era (°C)':<30} {'N/A (no ERA5 bias)':>12} {'0.4528':>12} {'1.8253':>12}")
print(f"  {'t03_model (°C)':<30} {t03_model:>12.3f} {'20.435':>12} {'22.082':>12}")
print(f"  {'t14_model (°C)':<30} {t14_model:>12.3f} {'27.869':>12} {'28.144':>12}")
print(f"  {'R² building_density':<30} {r2_bd:>12.4f} {'0.0091':>12} {'0.1779':>12}")
print(f"  {'R² avg_height':<30} {r2_ah:>12.4f} {'0.0005':>12} {'0.0591':>12}")
print(f"{'='*70}")

print("\nDone!")
