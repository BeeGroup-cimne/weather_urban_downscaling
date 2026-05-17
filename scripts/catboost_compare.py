#!/usr/bin/env python3
"""
Final comparison: CatBoost (corrected) vs Mamba vs ConvLSTM.
Evaluate on same Aug 15 snapshot as exp01 for fairness.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config.config import Config as C

CACHE_PATH = C.PATH_CACHE
STATIC_NPY = C.STATIC_CACHE_PATH
MODEL_DIR = PROJECT_ROOT / "model_benchmark" / "results" / "catboost_baseline"
MODEL_PATH = MODEL_DIR / "catboost_baseline.cbm"
STATS_PATH = C.STATS_PATH

GRID_STRIDE = 5
N_HARMONICS = 4
y_idxs = np.arange(0, 251, GRID_STRIDE)
x_idxs = np.arange(0, 251, GRID_STRIDE)
n_spatial = len(y_idxs) * len(x_idxs)

ds = xr.open_zarr(CACHE_PATH)
static_grid = np.load(STATIC_NPY)
stats = np.load(STATS_PATH)
lr_mean = stats["mean_lr"]
lr_std = stats["std_lr"]
hr_mean = float(stats["mean_hr"])
hr_std = float(stats["std_hr"])
lr_var_names = list(stats["lr_var_names"])

lat1d = ds.latitude.values
lon1d = ds.longitude.values
sub_lat = (lat1d[y_idxs][:, None] * np.ones(len(x_idxs))[None, :]).ravel()
sub_lon = (np.ones(len(y_idxs))[:, None] * lon1d[x_idxs][None, :]).ravel()
sub_static = static_grid[y_idxs[:, None], x_idxs[None, :], :].reshape(n_spatial, -1)

lr_lat = ds.latitude_lr.values
lr_lon = ds.longitude_lr.values
lr_near_y = np.argmin(np.abs(lr_lat[None, :] - sub_lat[:, None]), axis=1).astype(np.int32)
lr_near_x = np.argmin(np.abs(lr_lon[None, :] - sub_lon[:, None]), axis=1).astype(np.int32)

STATIC_NAMES = [
    "avg_height", "building_density", "elevation", "height_index",
    "industrial_index", "leisure_index", "max_levels", "ndvi_mean",
    "ndvi_min", "residential_index", "roughness", "services_index", "svf"
]

static_norm = np.zeros_like(sub_static)
for c in range(len(STATIC_NAMES)):
    col = sub_static[:, c]
    lo, hi = float(np.percentile(col, 5)), float(np.percentile(col, 95))
    static_norm[:, c] = np.clip((col - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0

def add_fourier(hours, months, nh=4):
    feats = []
    for period, vals in [(24.0, hours), (12.0, months)]:
        for h in range(1, nh + 1):
            feats.append(np.sin(2 * np.pi * h * vals / period))
            feats.append(np.cos(2 * np.pi * h * vals / period))
    return np.column_stack(feats)

def predict_timestamp(ts):
    """Predict for timestamp. Returns (pred_c, target_c)."""
    t = pd.Timestamp(ts)
    hr_norm = ds.hr_target.sel(time=t).values[y_idxs[:, None], x_idxs[None, :]].reshape(-1)
    hr_c = hr_norm * hr_std + hr_mean
    lr_vals = ds.lr_input.sel(time=t).values[lr_near_y, lr_near_x, :]
    lr_norm = (lr_vals - lr_mean) / np.maximum(lr_std, 1e-6)
    fourier = add_fourier(np.full(n_spatial, t.hour), np.full(n_spatial, t.month), N_HARMONICS)
    coords = np.column_stack([(sub_lat - 41.2) / 0.4, (sub_lon - 1.9) / 0.5])
    X = np.column_stack([lr_norm, static_norm, fourier, coords]).astype(np.float32)
    pred_norm = model.predict(X)
    return pred_norm * hr_std + hr_mean, hr_c

print("Loading model...")
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

# 1. Summer snapshot (Aug 15, 2017) — same as exp01
print("\n=== Summer snapshot (Aug 15, 2017) ===")
night_p, night_t = predict_timestamp("2017-08-15T03:00:00")
day_p, day_t = predict_timestamp("2017-08-15T14:00:00")

night_mae = float(np.mean(np.abs(night_p - night_t)))
day_mae = float(np.mean(np.abs(day_p - day_t)))
night_bias = float(np.mean(night_p - night_t))

print(f"  Night (03:00): pred={np.mean(night_p):.3f} target={np.mean(night_t):.3f} MAE={night_mae:.4f}°C")
print(f"  Day (14:00):   pred={np.mean(day_p):.3f} target={np.mean(day_t):.3f} MAE={day_mae:.4f}°C")
print(f"  Delta (night-day): pred={np.mean(night_p)-np.mean(day_p):.3f} target={np.mean(night_t)-np.mean(day_t):.3f}")

# Morphology gradient at 03:00
night_res = np.abs(night_p - night_t)
bd = sub_static[:, 1]
ah = sub_static[:, 0]

def r2_morph(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10: return float('nan')
    lr = LinearRegression().fit(x[mask].reshape(-1, 1), y[mask])
    return float(lr.score(x[mask].reshape(-1, 1), y[mask]))

r2_bd = r2_morph(bd, night_res)
r2_ah = r2_morph(ah, night_res)
print(f"  R² residual vs building_density: {r2_bd:.4f}")
print(f"  R² residual vs avg_height: {r2_ah:.4f}")

# 2. Full Dec 2017 evaluation
print("\n=== Winter evaluation (Dec 2017) ===")
test_da = ds.hr_target.sel(time=slice("2017-12-01", "2017-12-31"))
test_lr = ds.lr_input.sel(time=slice("2017-12-01", "2017-12-31"))
times = pd.DatetimeIndex(test_da.time.values)
night_mask = np.array([t.hour == 3 for t in times])
night_idxs = np.where(night_mask)[0]

night_preds, night_targets = [], []
for t_idx in night_idxs:
    hr_n = test_da.values[t_idx, y_idxs[:, None], x_idxs[None, :]].reshape(-1)
    hr_c = hr_n * hr_std + hr_mean
    lr_v = test_lr.values[t_idx, lr_near_y, lr_near_x, :]
    lr_n = (lr_v - lr_mean) / np.maximum(lr_std, 1e-6)
    f = add_fourier(np.full(n_spatial, times[t_idx].hour),
                    np.full(n_spatial, times[t_idx].month), N_HARMONICS)
    c = np.column_stack([(sub_lat - 41.2) / 0.4, (sub_lon - 1.9) / 0.5])
    X = np.column_stack([lr_n, static_norm, f, c]).astype(np.float32)
    pn = model.predict(X)
    night_preds.append(pn * hr_std + hr_mean)
    night_targets.append(hr_c)
night_preds = np.array(night_preds)
night_targets = np.array(night_targets)
winter_night_mae = float(np.mean(np.abs(night_preds - night_targets)))
print(f"  Night MAE @ 03:00: {winter_night_mae:.4f} °C")

# 3. Comparison table
print(f"\n{'='*85}")
print(f"  BENCHMARK: CatBoost vs Mamba vs ConvLSTM")
print(f"{'='*85}")
print(f"  {'Metric':<40} {'CatBoost':>13} {'Mamba_s12':>13} {'ConvLSTM':>13}")
print(f"  {'-'*40} {'-'*13} {'-'*13} {'-'*13}")

print(f"  {'Summer (Aug 15, 2017)':<40} {'':>13} {'':>13} {'':>13}")
print(f"  {'  T03 MAE (°C) ↓':<40} {night_mae:>13.4f} {'1.2361':>13} {'2.8832':>13}")
print(f"  {'  T03 mean (°C)':<40} {np.mean(night_p):>13.3f} {'20.435':>13} {'22.082':>13}")
print(f"  {'  T03 target (°C)':<40} {np.mean(night_t):>13.3f} {'22.419 (UrbClim)':>13} {'':>13}")
print(f"  {'  T03 bias (°C)':<40} {night_bias:>13.3f} {'-1.522':>13} {'-0.149':>13}")
print(f"  {'  R² building_density':<40} {r2_bd:>13.4f} {'0.0091':>13} {'0.1779':>13}")
print(f"  {'  R² avg_height':<40} {r2_ah:>13.4f} {'0.0005':>13} {'0.0591':>13}")

print(f"  {'Winter (Dec 2017)':<40} {'':>13} {'':>13} {'':>13}")
print(f"  {'  T03 MAE (°C)':<40} {winter_night_mae:>13.4f} {'n/a':>13} {'n/a':>13}")
print(f"{'='*85}")
print(f"  ↓ = lower is better")
print(f"  CatBoost uses grid stride={GRID_STRIDE} ({n_spatial} pts)")
print(f"  exp01 metrics from report_model_comparison.md (Aug 15 eval)")
print(f"  Note: CatBoost target is station-interpolated, exp01 uses UrbClim")
print(f"  Direct comparison of MAE is NOT fair — different target data!")

# Save
results = {
    "aug15_summer_night": {
        "mae_c": round(night_mae, 4),
        "pred_mean_c": round(float(np.mean(night_p)), 4),
        "target_mean_c": round(float(np.mean(night_t)), 4),
        "bias_c": round(night_bias, 4),
        "r2_building_density": round(r2_bd, 4),
        "r2_avg_height": round(r2_ah, 4),
    },
    "aug15_summer_day": {
        "mae_c": round(day_mae, 4),
        "pred_mean_c": round(float(np.mean(day_p)), 4),
        "target_mean_c": round(float(np.mean(day_t)), 4),
    },
    "dec2017_winter_night": {
        "mae_c": round(winter_night_mae, 4),
    },
    "exp01_mamba_seq12": {
        "residual03_mae_c": 1.2361,
        "t03_mean_c": 20.435,
        "r2_building_density": 0.0091,
        "r2_avg_height": 0.0005,
    },
    "exp01_convlstm_seq6": {
        "residual03_mae_c": 2.8832,
        "t03_mean_c": 22.082,
        "r2_building_density": 0.1779,
        "r2_avg_height": 0.0591,
    },
}
with open(MODEL_DIR / "benchmark_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {MODEL_DIR / 'benchmark_comparison.json'}")
print("\nDone!")
