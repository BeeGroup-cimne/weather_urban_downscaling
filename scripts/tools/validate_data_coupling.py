#!/usr/bin/env python3
"""
Data Coupling Validation Script
================================
Audits the alignment between the three data sources used by the pipeline:
  1. HR target   (interpolated stations → NetCDF)
  2. LR input    (ERA5-Land → GRIB)
  3. Static data (urban morphology → Zarr)

Reads raw files AND the processed cache to detect coupling problems.
Safe to run alongside active training — read-only on all files.

Output: experiments/data_coupling_report.json  +  console summary.

Usage:
    python scripts/validate_data_coupling.py
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.runtime import Config

# Suppress noisy xarray / cfgrib warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v):
    """Convert numpy/xarray scalar to JSON-safe float."""
    if v is None:
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


def _bbox(lats, lons):
    return {
        "lat_min": _safe_float(np.nanmin(lats)),
        "lat_max": _safe_float(np.nanmax(lats)),
        "lon_min": _safe_float(np.nanmin(lons)),
        "lon_max": _safe_float(np.nanmax(lons)),
    }


def _overlap_pct(bbox_inner, bbox_outer):
    """Fraction of bbox_inner area covered by bbox_outer (0-1, 1 = full)."""
    lat_overlap = max(0, min(bbox_inner["lat_max"], bbox_outer["lat_max"])
                      - max(bbox_inner["lat_min"], bbox_outer["lat_min"]))
    lon_overlap = max(0, min(bbox_inner["lon_max"], bbox_outer["lon_max"])
                      - max(bbox_inner["lon_min"], bbox_outer["lon_min"]))
    inner_area = ((bbox_inner["lat_max"] - bbox_inner["lat_min"])
                  * (bbox_inner["lon_max"] - bbox_inner["lon_min"]))
    if inner_area <= 0:
        return 0.0
    return (lat_overlap * lon_overlap) / inner_area


# ===========================================================================
# 1. SPATIAL OVERLAP AUDIT
# ===========================================================================

def audit_spatial_overlap(report: dict):
    """Measure how well LR covers the HR bounding box after clipping."""
    print("\n" + "=" * 60)
    print("1️⃣  SPATIAL OVERLAP AUDIT")
    print("=" * 60)

    section = {}

    # --- HR bbox (from cache or raw) ---
    cache_path = Config.PATH_CACHE
    if os.path.isdir(cache_path):
        ds = xr.open_zarr(cache_path, consolidated=True)
        da_hr = ds["hr_target"]
        hr_dims = list(da_hr.dims)
        hr_y = next((d for d in hr_dims if d in ["y", "latitude", "lat"]), "y")
        hr_x = next((d for d in hr_dims if d in ["x", "longitude", "lon"]), "x")

        hr_lats = ds[hr_y].values if hr_y in ds.coords else np.arange(da_hr.sizes[hr_y])
        hr_lons = ds[hr_x].values if hr_x in ds.coords else np.arange(da_hr.sizes[hr_x])

        hr_bbox = _bbox(hr_lats, hr_lons)
        section["hr_shape"] = [int(da_hr.sizes[hr_y]), int(da_hr.sizes[hr_x])]
        section["hr_bbox"] = hr_bbox
        print(f"   HR shape : {section['hr_shape']}")
        print(f"   HR bbox  : lat [{hr_bbox['lat_min']:.4f}, {hr_bbox['lat_max']:.4f}]"
              f"  lon [{hr_bbox['lon_min']:.4f}, {hr_bbox['lon_max']:.4f}]")

        # --- LR bbox (from cache) ---
        da_lr = ds["lr_input"]
        lr_dims = list(da_lr.dims)
        lr_lat = next((d for d in lr_dims if "lat" in d or d == "y_lr" or d == "y"), None)
        lr_lon = next((d for d in lr_dims if "lon" in d or d == "x_lr" or d == "x"), None)

        if lr_lat and lr_lon:
            lr_lats = ds[lr_lat].values if lr_lat in ds.coords else np.arange(da_lr.sizes[lr_lat])
            lr_lons = ds[lr_lon].values if lr_lon in ds.coords else np.arange(da_lr.sizes[lr_lon])
            lr_bbox = _bbox(lr_lats, lr_lons)
            section["lr_shape"] = [int(da_lr.sizes[lr_lat]), int(da_lr.sizes[lr_lon])]
            section["lr_bbox"] = lr_bbox
            print(f"   LR shape : {section['lr_shape']}")
            print(f"   LR bbox  : lat [{lr_bbox['lat_min']:.4f}, {lr_bbox['lat_max']:.4f}]"
                  f"  lon [{lr_bbox['lon_min']:.4f}, {lr_bbox['lon_max']:.4f}]")

            # Overlap
            overlap = _overlap_pct(hr_bbox, lr_bbox)
            section["lr_covers_hr_pct"] = _safe_float(overlap * 100)
            status = "✅" if overlap >= 0.99 else ("⚠️" if overlap >= 0.90 else "❌")
            print(f"   {status} LR covers {overlap * 100:.1f}% of HR bbox")
        else:
            section["lr_warning"] = "Could not detect LR lat/lon dims"
            print(f"   ⚠️ Could not detect LR lat/lon dimensions: {lr_dims}")

        ds.close()
    else:
        section["warning"] = f"Cache not found at {cache_path}"
        print(f"   ⚠️ Cache not found at {cache_path}")

    report["spatial_overlap"] = section


# ===========================================================================
# 2. NaN BUDGET REPORT
# ===========================================================================

def audit_nan_budget(report: dict):
    """Report NaN ratios in raw LR vs cleaned cache to quantify 'invented signal'."""
    print("\n" + "=" * 60)
    print("2️⃣  NaN BUDGET REPORT")
    print("=" * 60)

    section = {}

    # --- Raw LR NaN ratio (first 24 timesteps sample) ---
    lr_path = Config.PATH_LR
    if os.path.exists(lr_path):
        try:
            cfgrib_kwargs = {
                "filter_by_keys": {"typeOfLevel": "surface"},
                "errors": "ignore",
                "indexpath": "",
            }
            ds_lr_raw = xr.open_dataset(
                lr_path, engine="cfgrib",
                backend_kwargs=cfgrib_kwargs,
                chunks={"time": 24},
            )
            # Sample first 24 timesteps
            sample = ds_lr_raw.isel(time=slice(0, 24))
            total_cells = 1
            for d in sample.dims:
                total_cells *= sample.sizes[d]

            nan_count = 0
            for var in sample.data_vars:
                vals = sample[var].values
                nan_count += int(np.isnan(vals).sum())

            raw_nan_ratio = nan_count / max(1, total_cells * len(sample.data_vars))
            section["raw_lr_nan_ratio"] = _safe_float(raw_nan_ratio)
            section["raw_lr_sample_timesteps"] = min(24, ds_lr_raw.sizes.get("time", 0))
            print(f"   Raw LR NaN ratio (first 24h): {raw_nan_ratio * 100:.2f}%")
            ds_lr_raw.close()
        except Exception as e:
            section["raw_lr_error"] = str(e)
            print(f"   ⚠️ Could not read raw LR: {e}")
    else:
        section["raw_lr_warning"] = "File not found"
        print(f"   ⚠️ Raw LR not found at {lr_path}")

    # --- Cache NaN ratio (should be 0) ---
    cache_path = Config.PATH_CACHE
    if os.path.isdir(cache_path):
        ds = xr.open_zarr(cache_path, consolidated=True)
        for var_name in ["lr_input", "hr_target"]:
            if var_name in ds:
                da = ds[var_name]
                # Sample 100 timesteps
                n_sample = min(100, da.sizes["time"])
                sample = da.isel(time=slice(0, n_sample))
                nan_ratio = float(sample.isnull().mean().compute().item())
                section[f"cache_{var_name}_nan_ratio"] = _safe_float(nan_ratio)
                status = "✅" if nan_ratio == 0 else "❌"
                print(f"   {status} Cache {var_name} NaN ratio: {nan_ratio * 100:.4f}%")

                # Also check for inf
                inf_ratio = float(np.isinf(sample.values).mean()) if nan_ratio == 0 else None
                if inf_ratio is not None:
                    section[f"cache_{var_name}_inf_ratio"] = _safe_float(inf_ratio)
                    if inf_ratio > 0:
                        print(f"   ❌ Cache {var_name} contains Infs: {inf_ratio * 100:.4f}%")
        ds.close()

    report["nan_budget"] = section


# ===========================================================================
# 3. NORMALIZATION SANITY
# ===========================================================================

def audit_normalization(report: dict):
    """Check that normalization stats are reasonable (no near-zero std)."""
    print("\n" + "=" * 60)
    print("3️⃣  NORMALIZATION SANITY")
    print("=" * 60)

    section = {}

    stats_path = Config.STATS_PATH
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        for key in stats.files:
            vals = stats[key]
            section[key] = {
                "shape": list(vals.shape) if hasattr(vals, "shape") else "scalar",
                "values": vals.tolist() if hasattr(vals, "tolist") else float(vals),
            }

        # Check LR std
        if "std_lr" in stats:
            std_key = "std_lr_raw" if "std_lr_raw" in stats else "std_lr"
            std_lr = stats[std_key]
            std_threshold = float(stats["norm_std_threshold"]) if "norm_std_threshold" in stats else 0.01
            lr_var_names = None
            if "lr_var_names" in stats:
                lr_var_names = [str(v) for v in stats["lr_var_names"].tolist()]
            elif os.path.isdir(Config.PATH_CACHE):
                # Backward compatibility: infer variable names from cache if old stats file lacks metadata.
                try:
                    ds_cache = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
                    if "lr_input" in ds_cache and "variable" in ds_cache["lr_input"].coords:
                        lr_var_names = [str(v) for v in ds_cache["lr_input"]["variable"].values.tolist()]
                    ds_cache.close()
                except Exception:
                    lr_var_names = None

            near_zero = std_lr < std_threshold
            if hasattr(near_zero, "__iter__"):
                n_bad = int(np.sum(near_zero))
                section["lr_near_zero_std_count"] = n_bad
                if n_bad > 0:
                    bad_indices = list(np.where(near_zero)[0])
                    bad_values = [_safe_float(std_lr[i]) for i in bad_indices]
                    section["lr_near_zero_std_vars"] = {
                        "indices": bad_indices,
                        "values": bad_values,
                    }
                    if lr_var_names:
                        bad_names = [lr_var_names[i] if i < len(lr_var_names) else f"var_{i}" for i in bad_indices]
                        section["lr_near_zero_std_vars"]["names"] = bad_names
                        print(f"   ⚠️ {n_bad} LR variable(s) with std < {std_threshold}:")
                        print(f"      Indices: {bad_indices}")
                        print(f"      Names: {bad_names}")
                    else:
                        print(f"   ⚠️ {n_bad} LR variable(s) with std < {std_threshold}: indices {bad_indices}")
                    print(f"      Values: {bad_values}")
                    print(f"      → These channels have near-constant values and will be")
                    print(f"        amplified ~100x after Z-score normalization!")
                else:
                    print(f"   ✅ All LR variables have std >= {std_threshold}")
            else:
                val = float(std_lr)
                section["lr_near_zero_std_count"] = 1 if val < std_threshold else 0
                status = "⚠️" if val < std_threshold else "✅"
                print(f"   {status} LR std = {val:.6f}")

            # If std floor was applied in preprocessing, capture mitigation state
            floor_idx = stats["lr_std_floor_applied_idx"].tolist() if "lr_std_floor_applied_idx" in stats else []
            floor_idx = [int(i) for i in floor_idx]
            section["lr_std_floor_applied_count"] = len(floor_idx)
            if floor_idx:
                section["lr_std_floor_applied_indices"] = floor_idx
                if lr_var_names:
                    section["lr_std_floor_applied_names"] = [
                        lr_var_names[i] if i < len(lr_var_names) else f"var_{i}" for i in floor_idx
                    ]
                print(f"   🛡️ Std floor already applied to {len(floor_idx)} LR variable(s)")

        # Check HR std
        if "std_hr" in stats:
            std_hr = float(stats["std_hr"])
            section["hr_std"] = _safe_float(std_hr)
            status = "⚠️" if std_hr < 0.01 else "✅"
            print(f"   {status} HR std = {std_hr:.6f}")

        # Mean ranges
        if "mean_lr" in stats:
            mean_lr = stats["mean_lr"]
            print(f"   📊 LR mean range: [{np.min(mean_lr):.4f}, {np.max(mean_lr):.4f}]")
        if "mean_hr" in stats:
            print(f"   📊 HR mean: {float(stats['mean_hr']):.4f}")
    else:
        section["warning"] = f"Stats not found at {stats_path}"
        print(f"   ⚠️ Stats not found at {stats_path}")

    report["normalization"] = section


# ===========================================================================
# 4. STATIC DATA COVERAGE
# ===========================================================================

def audit_static_coverage(report: dict):
    """Check how well static data covers the HR grid."""
    print("\n" + "=" * 60)
    print("4️⃣  STATIC DATA COVERAGE")
    print("=" * 60)

    section = {}

    static_cache_path = Config.STATIC_CACHE_PATH
    if os.path.exists(static_cache_path):
        static = np.load(static_cache_path)
        section["shape"] = list(static.shape)
        section["dtype"] = str(static.dtype)
        print(f"   Shape: {static.shape}  dtype: {static.dtype}")

        # NaN / Inf check
        nan_pct = _safe_float(np.isnan(static).mean() * 100)
        inf_pct = _safe_float(np.isinf(static).mean() * 100)
        section["nan_pct"] = nan_pct
        section["inf_pct"] = inf_pct
        status = "✅" if nan_pct == 0 and inf_pct == 0 else "❌"
        print(f"   {status} NaN: {nan_pct}%  Inf: {inf_pct}%")

        # Per-channel stats
        n_channels = static.shape[-1] if static.ndim == 3 else 1
        section["n_channels"] = n_channels
        channel_stats = []
        for c in range(n_channels):
            ch = static[:, :, c] if static.ndim == 3 else static
            ch_info = {
                "channel": c,
                "mean": _safe_float(np.nanmean(ch)),
                "std": _safe_float(np.nanstd(ch)),
                "min": _safe_float(np.nanmin(ch)),
                "max": _safe_float(np.nanmax(ch)),
                "nan_pct": _safe_float(np.isnan(ch).mean() * 100),
                "zero_pct": _safe_float((ch == 0).mean() * 100),
            }
            channel_stats.append(ch_info)
            if ch_info["std"] < 0.001:
                print(f"   ⚠️ Channel {c}: near-constant (std={ch_info['std']})")
            if ch_info["zero_pct"] > 90:
                print(f"   ⚠️ Channel {c}: {ch_info['zero_pct']}% zeros")

        section["channel_stats"] = channel_stats
        print(f"   📊 {n_channels} channels analyzed")

        # Check shape matches HR_SHAPE
        expected_h, expected_w = Config.HR_SHAPE
        if static.shape[0] != expected_h or static.shape[1] != expected_w:
            section["shape_mismatch"] = {
                "expected": [expected_h, expected_w],
                "actual": [static.shape[0], static.shape[1]],
            }
            print(f"   ❌ Shape mismatch! Expected HR ({expected_h}, {expected_w})"
                  f" but static is ({static.shape[0]}, {static.shape[1]})")
        else:
            print(f"   ✅ Shape matches HR_SHAPE ({expected_h}, {expected_w})")
    else:
        section["warning"] = f"Static cache not found at {static_cache_path}"
        print(f"   ⚠️ Static cache not found at {static_cache_path}")

    # Also check source Zarr bbox if available
    static_zarr_path = Config.PATH_STATIC
    if os.path.isdir(static_zarr_path):
        try:
            ds_st = xr.open_zarr(static_zarr_path)
            if "index" in ds_st:
                indices = ds_st["index"].values
                lats = np.array([float(x.split("_")[0]) for x in indices])
                lons = np.array([float(x.split("_")[1]) for x in indices])
                st_bbox = _bbox(lats, lons)
                section["source_zarr_bbox"] = st_bbox
                section["source_zarr_n_points"] = len(indices)
                print(f"   Source Zarr: {len(indices)} points,"
                      f" lat [{st_bbox['lat_min']:.4f}, {st_bbox['lat_max']:.4f}]"
                      f" lon [{st_bbox['lon_min']:.4f}, {st_bbox['lon_max']:.4f}]")
            ds_st.close()
        except Exception as e:
            section["source_zarr_error"] = str(e)
            print(f"   ⚠️ Could not read static Zarr: {e}")

    report["static_coverage"] = section


# ===========================================================================
# 5. TEMPORAL GAP REPORT
# ===========================================================================

def audit_temporal_gaps(report: dict):
    """Detect gaps in the time axis of the cached data."""
    print("\n" + "=" * 60)
    print("5️⃣  TEMPORAL GAP REPORT")
    print("=" * 60)

    section = {}

    cache_path = Config.PATH_CACHE
    if os.path.isdir(cache_path):
        ds = xr.open_zarr(cache_path, consolidated=True)
        times = pd.to_datetime(ds["time"].values)
        section["total_timesteps"] = len(times)
        section["time_range"] = {
            "start": str(times.min()),
            "end": str(times.max()),
        }
        print(f"   Total timesteps: {len(times)}")
        print(f"   Range: {times.min()} → {times.max()}")

        if len(times) > 1:
            diffs = np.diff(times)
            # Expected: 1 hour
            expected = pd.Timedelta("1h")
            gaps = []
            for i, d in enumerate(diffs):
                if d > expected:
                    gap_info = {
                        "index": int(i),
                        "from": str(times[i]),
                        "to": str(times[i + 1]),
                        "gap_hours": _safe_float(d / pd.Timedelta("1h")),
                    }
                    gaps.append(gap_info)

            section["gaps_gt_1h"] = len(gaps)
            if gaps:
                # Show up to 20 biggest
                gaps_sorted = sorted(gaps, key=lambda g: -g["gap_hours"])
                section["largest_gaps"] = gaps_sorted[:20]
                print(f"   ⚠️ {len(gaps)} gaps > 1 hour detected")
                for g in gaps_sorted[:5]:
                    print(f"      {g['from']} → {g['to']}  ({g['gap_hours']:.0f}h)")
                if len(gaps) > 5:
                    print(f"      ... and {len(gaps) - 5} more")
            else:
                print(f"   ✅ No gaps > 1 hour — time axis is contiguous")

            # Also check regularity
            median_diff = np.median([d / pd.Timedelta("1h") for d in diffs])
            section["median_step_hours"] = _safe_float(median_diff)
            print(f"   📊 Median step: {median_diff:.1f}h")

        # Split coverage
        try:
            for split_name, start_attr, end_attr in [
                ("train", "TRAIN_START", "TRAIN_END"),
                ("val", "VAL_START", "VAL_END"),
                ("test", "TEST_START", "TEST_END"),
            ]:
                s = pd.to_datetime(getattr(Config, start_attr))
                e = pd.to_datetime(getattr(Config, end_attr))
                mask = (times >= s) & (times < e)
                n = int(mask.sum())
                expected_hours = int((e - s).total_seconds() / 3600)
                coverage = n / max(1, expected_hours) * 100
                section[f"{split_name}_timesteps"] = n
                section[f"{split_name}_expected_hours"] = expected_hours
                section[f"{split_name}_coverage_pct"] = _safe_float(coverage)
                status = "✅" if coverage >= 95 else ("⚠️" if coverage >= 80 else "❌")
                print(f"   {status} {split_name.upper()}: {n}/{expected_hours} hours"
                      f" ({coverage:.1f}% coverage)")
        except Exception as e:
            section["split_error"] = str(e)

        ds.close()
    else:
        section["warning"] = f"Cache not found at {cache_path}"
        print(f"   ⚠️ Cache not found")

    report["temporal_gaps"] = section


# ===========================================================================
# 6. CROSS-CORRELATION SNAPSHOT
# ===========================================================================

def audit_cross_correlation(report: dict):
    """Compute Pearson(LR_upscaled, HR) on sampled timesteps to detect anomalies."""
    print("\n" + "=" * 60)
    print("6️⃣  CROSS-CORRELATION SNAPSHOT (LR↔HR)")
    print("=" * 60)

    section = {}

    cache_path = Config.PATH_CACHE
    if not os.path.isdir(cache_path):
        section["warning"] = "Cache not found"
        print("   ⚠️ Cache not found")
        report["cross_correlation"] = section
        return

    try:
        # Lazy import TF for resize (optional)
        resize_fn = None
        try:
            import tensorflow as tf
            def _tf_resize(arr_2d, target_shape):
                t = tf.image.resize(arr_2d[..., None], target_shape, method="bilinear")
                return t.numpy()[..., 0]
            resize_fn = _tf_resize
        except ImportError:
            try:
                from scipy.ndimage import zoom
                def _scipy_resize(arr_2d, target_shape):
                    factors = (target_shape[0] / arr_2d.shape[0],
                               target_shape[1] / arr_2d.shape[1])
                    return zoom(arr_2d, factors, order=1)
                resize_fn = _scipy_resize
            except ImportError:
                pass

        if resize_fn is None:
            section["warning"] = "Neither TensorFlow nor scipy available for upscaling"
            print("   ⚠️ Cannot upscale LR (no TF/scipy). Skipping.")
            report["cross_correlation"] = section
            return

        ds = xr.open_zarr(cache_path, consolidated=True)
        da_hr = ds["hr_target"]
        da_lr = ds["lr_input"]

        hr_dims = list(da_hr.dims)
        hr_y = next((d for d in hr_dims if d in ["y", "latitude", "lat"]), "y")
        hr_x = next((d for d in hr_dims if d in ["x", "longitude", "lon"]), "x")
        hr_shape = (da_hr.sizes[hr_y], da_hr.sizes[hr_x])

        lr_dims = list(da_lr.dims)
        lr_lat = next((d for d in lr_dims if "lat" in d or d in ["y_lr", "y"]), None)
        lr_lon = next((d for d in lr_dims if "lon" in d or d in ["x_lr", "x"]), None)
        lr_var_dim = next((d for d in lr_dims if d in ["variable", "channel", "var"]), None)

        if lr_lat is None or lr_lon is None:
            section["warning"] = f"Cannot detect LR dims: {lr_dims}"
            report["cross_correlation"] = section
            return

        # Prefer t2m (or close aliases) as LR reference variable for correlation.
        lr_ref_idx = 0
        lr_ref_name = None
        if lr_var_dim and lr_var_dim in da_lr.coords:
            lr_var_names = [str(v) for v in da_lr[lr_var_dim].values.tolist()]
            lr_var_names_low = [v.lower() for v in lr_var_names]
            candidate_labels = ["t2m", "tas", "2t", "temperature_2m"]
            for cand in candidate_labels:
                if cand in lr_var_names_low:
                    lr_ref_idx = lr_var_names_low.index(cand)
                    break
                partial = [i for i, name in enumerate(lr_var_names_low) if cand in name]
                if partial:
                    lr_ref_idx = partial[0]
                    break
            lr_ref_name = lr_var_names[lr_ref_idx] if lr_ref_idx < len(lr_var_names) else f"{lr_var_dim}[{lr_ref_idx}]"
            print(f"   🔎 LR reference variable for correlation: {lr_ref_name} (index={lr_ref_idx})")
        else:
            print("   ⚠️ LR variable coordinate not found; using first available LR channel for correlation")

        section["lr_reference_variable"] = lr_ref_name if lr_ref_name is not None else "first_channel_fallback"
        section["lr_reference_index"] = int(lr_ref_idx)

        # Sample ~50 evenly-spaced timesteps
        total = ds.sizes["time"]
        n_samples = min(50, total)
        indices = np.linspace(0, total - 1, n_samples, dtype=int)

        correlations = []
        for idx in indices:
            hr_slice = da_hr.isel(time=int(idx)).values
            if hr_slice.ndim > 2:
                hr_slice = hr_slice[..., 0]

            lr_slice_da = da_lr.isel(time=int(idx))
            if lr_var_dim and lr_var_dim in lr_slice_da.dims:
                lr_slice_da = lr_slice_da.isel({lr_var_dim: int(lr_ref_idx)})
            lr_slice = lr_slice_da.values
            if lr_slice.ndim > 2:
                # Defensive fallback for unexpected extra dims.
                while lr_slice.ndim > 2:
                    lr_slice = lr_slice[..., 0]

            if np.all(np.isnan(hr_slice)) or np.all(np.isnan(lr_slice)):
                correlations.append({"index": int(idx), "corr": None})
                continue

            # Upscale LR to HR resolution
            try:
                lr_up = resize_fn(lr_slice, hr_shape)
            except Exception:
                correlations.append({"index": int(idx), "corr": None})
                continue

            # Pearson correlation
            a = hr_slice.flatten()
            b = lr_up.flatten()
            valid = ~(np.isnan(a) | np.isnan(b))
            a, b = a[valid], b[valid]
            if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
                correlations.append({"index": int(idx), "corr": None})
                continue

            corr = float(np.corrcoef(a, b)[0, 1])
            correlations.append({"index": int(idx), "corr": _safe_float(corr)})

        valid_corrs = [c["corr"] for c in correlations if c["corr"] is not None]
        if valid_corrs:
            section["mean_corr"] = _safe_float(np.mean(valid_corrs))
            section["min_corr"] = _safe_float(np.min(valid_corrs))
            section["max_corr"] = _safe_float(np.max(valid_corrs))
            section["std_corr"] = _safe_float(np.std(valid_corrs))
            section["n_samples"] = len(valid_corrs)

            print(f"   📊 Pearson(LR↑, HR) over {len(valid_corrs)} timesteps:")
            print(f"      Mean: {section['mean_corr']:.4f}")
            print(f"      Min:  {section['min_corr']:.4f}")
            print(f"      Max:  {section['max_corr']:.4f}")
            print(f"      Std:  {section['std_corr']:.4f}")

            # Flag anomalous timesteps (corr < mean - 2*std)
            threshold = np.mean(valid_corrs) - 2 * np.std(valid_corrs)
            anomalies = [c for c in correlations
                         if c["corr"] is not None and c["corr"] < threshold]
            section["anomaly_threshold"] = _safe_float(threshold)
            section["n_anomalies"] = len(anomalies)
            if anomalies:
                section["anomalous_timesteps"] = anomalies[:10]
                print(f"   ⚠️ {len(anomalies)} anomalous timesteps (corr < {threshold:.3f})")
                for a in anomalies[:5]:
                    print(f"      index={a['index']}  corr={a['corr']}")
            else:
                print(f"   ✅ No anomalous timesteps detected")
        else:
            section["warning"] = "No valid correlations computed"
            print("   ⚠️ No valid correlations computed")

        ds.close()

    except Exception as e:
        section["error"] = str(e)
        print(f"   ❌ Error: {e}")

    report["cross_correlation"] = section


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("🔍 DATA COUPLING VALIDATION REPORT")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    config_class_name = getattr(Config, "__name__", type(Config).__name__)
    print(f"   Config: {config_class_name}")
    print("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config_class": config_class_name,
        "config_paths": {
            "PATH_HR": Config.PATH_HR,
            "PATH_LR": Config.PATH_LR,
            "PATH_STATIC": Config.PATH_STATIC,
            "PATH_CACHE": Config.PATH_CACHE,
            "STATS_PATH": Config.STATS_PATH,
            "STATIC_CACHE_PATH": Config.STATIC_CACHE_PATH,
        },
    }

    # Run all audits
    audit_spatial_overlap(report)
    audit_nan_budget(report)
    audit_normalization(report)
    audit_static_coverage(report)
    audit_temporal_gaps(report)
    audit_cross_correlation(report)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    issues = []
    warnings_list = []

    # Spatial
    so = report.get("spatial_overlap", {})
    cov = so.get("lr_covers_hr_pct")
    if cov is not None and cov < 99:
        issues.append(f"LR only covers {cov:.1f}% of HR bbox")

    # NaN
    nb = report.get("nan_budget", {})
    for key in ["cache_lr_input_nan_ratio", "cache_hr_target_nan_ratio"]:
        val = nb.get(key, 0)
        if val and val > 0:
            issues.append(f"Cache has NaNs: {key}={val}")

    # Normalization
    norm = report.get("normalization", {})
    n_bad = norm.get("lr_near_zero_std_count", 0)
    n_floor = norm.get("lr_std_floor_applied_count", 0)
    if n_bad > 0:
        if n_floor >= n_bad:
            warnings_list.append(
                f"{n_bad} LR variable(s) had near-zero std but std floor mitigation is active"
            )
        else:
            issues.append(f"{n_bad} LR variable(s) with near-zero std (risk of signal amplification)")

    # Static
    sc = report.get("static_coverage", {})
    if sc.get("nan_pct", 0) > 0:
        issues.append(f"Static data has NaN: {sc['nan_pct']}%")
    if "shape_mismatch" in sc:
        issues.append(f"Static shape mismatch with HR_SHAPE")

    # Temporal
    tg = report.get("temporal_gaps", {})
    n_gaps = tg.get("gaps_gt_1h", 0)
    if n_gaps > 0:
        warnings_list.append(f"{n_gaps} temporal gaps > 1h in cache")
    for split in ["train", "val", "test"]:
        cov = tg.get(f"{split}_coverage_pct")
        if cov is not None and cov < 95:
            issues.append(f"{split.upper()} split: only {cov:.1f}% temporal coverage")

    # Cross-correlation
    cc = report.get("cross_correlation", {})
    mean_corr = cc.get("mean_corr")
    if mean_corr is not None and mean_corr < 0.3:
        issues.append(f"Low LR↔HR correlation: mean={mean_corr:.3f}")
    n_anom = cc.get("n_anomalies", 0)
    if n_anom > 0:
        warnings_list.append(f"{n_anom} timesteps with anomalous LR↔HR correlation")

    report["summary"] = {
        "issues": issues,
        "warnings": warnings_list,
        "status": "PASS" if not issues else "FAIL",
    }

    if issues:
        print(f"\n   ❌ {len(issues)} ISSUE(S) FOUND:")
        for i, iss in enumerate(issues, 1):
            print(f"      {i}. {iss}")
    else:
        print(f"\n   ✅ No critical issues found")

    if warnings_list:
        print(f"\n   ⚠️ {len(warnings_list)} WARNING(S):")
        for w in warnings_list:
            print(f"      - {w}")

    # Save report
    report_dir = os.path.join(Config.EXPERIMENTS_DIR)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "data_coupling_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n   💾 Report saved: {report_path}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
