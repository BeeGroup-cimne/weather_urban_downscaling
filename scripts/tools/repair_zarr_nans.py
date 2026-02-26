#!/usr/bin/env python3
"""
Repair NaNs/Infs in the existing Zarr cache without re-running the full ETL.
Also recomputes stats_config.npz after cleaning.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import zarr
from config.runtime import Config


def _iter_chunk_slices(arr):
    shape = arr.shape
    chunks = arr.chunks
    for idx in np.ndindex(arr.nchunks):
        slices = []
        for i, c in zip(idx, chunks):
            start = i * c
            stop = min(start + c, shape[len(slices)])
            slices.append(slice(start, stop))
        yield tuple(slices)


def _nanmean_chunked(arr):
    if arr.ndim == 4:
        ch = arr.shape[-1]
        sumv = np.zeros(ch, dtype=np.float64)
        count = np.zeros(ch, dtype=np.float64)
        for slc in _iter_chunk_slices(arr):
            data = arr[slc]
            mask = np.isfinite(data)
            sumv += np.nansum(np.where(mask, data, 0.0), axis=(0, 1, 2))
            count += np.sum(mask, axis=(0, 1, 2))
        count = np.where(count == 0, 1.0, count)
        return sumv / count

    sumv = 0.0
    count = 0.0
    for slc in _iter_chunk_slices(arr):
        data = arr[slc]
        mask = np.isfinite(data)
        sumv += np.nansum(np.where(mask, data, 0.0))
        count += np.sum(mask)
    if count == 0:
        return 0.0
    return sumv / count


def _fill_nans(arr, fill):
    for slc in _iter_chunk_slices(arr):
        data = arr[slc]
        if not (np.isnan(data).any() or np.isinf(data).any()):
            continue
        if arr.ndim == 4:
            fill_b = np.asarray(fill).reshape((1, 1, 1, -1))
            data = np.where(np.isnan(data), fill_b, data)
            data = np.where(np.isinf(data), fill_b, data)
        else:
            data = np.where(np.isnan(data), fill, data)
            data = np.where(np.isinf(data), fill, data)
        arr[slc] = data


def _stats_chunked(arr):
    if arr.ndim == 4:
        ch = arr.shape[-1]
        sumv = np.zeros(ch, dtype=np.float64)
        sumsq = np.zeros(ch, dtype=np.float64)
        count = np.zeros(ch, dtype=np.float64)
        for slc in _iter_chunk_slices(arr):
            data = arr[slc]
            mask = np.isfinite(data)
            data = np.where(mask, data, 0.0)
            sumv += np.sum(data, axis=(0, 1, 2))
            sumsq += np.sum(data * data, axis=(0, 1, 2))
            count += np.sum(mask, axis=(0, 1, 2))
        count = np.where(count == 0, 1.0, count)
        mean = sumv / count
        var = sumsq / count - mean * mean
        var = np.where(var < 0, 0.0, var)
        return mean, np.sqrt(var)

    sumv = 0.0
    sumsq = 0.0
    count = 0.0
    for slc in _iter_chunk_slices(arr):
        data = arr[slc]
        mask = np.isfinite(data)
        data = np.where(mask, data, 0.0)
        sumv += np.sum(data)
        sumsq += np.sum(data * data)
        count += np.sum(mask)
    if count == 0:
        return 0.0, 0.0
    mean = sumv / count
    var = sumsq / count - mean * mean
    if var < 0:
        var = 0.0
    return mean, np.sqrt(var)


def main():
    cache_path = Config.PATH_CACHE
    if not os.path.exists(cache_path):
        print(f"❌ Zarr cache not found: {cache_path}")
        return 1

    print(f"🔧 Repairing NaNs in: {cache_path}")
    g = zarr.open_group(cache_path, mode="r+")

    if "lr_input" not in g or "hr_target" not in g:
        print("❌ Missing lr_input or hr_target in cache.")
        return 2

    lr = g["lr_input"]
    hr = g["hr_target"]

    lr_mean = _nanmean_chunked(lr)
    hr_mean = _nanmean_chunked(hr)

    print("🧽 Filling NaNs/Infs...")
    _fill_nans(lr, lr_mean)
    _fill_nans(hr, hr_mean)

    print("📊 Recomputing stats...")
    mean_lr, std_lr = _stats_chunked(lr)
    mean_hr, std_hr = _stats_chunked(hr)

    os.makedirs(os.path.dirname(Config.STATS_PATH), exist_ok=True)
    np.savez(
        Config.STATS_PATH,
        mean_lr=np.asarray(mean_lr),
        std_lr=np.asarray(std_lr),
        mean_hr=float(mean_hr),
        std_hr=float(std_hr),
    )
    print(f"✅ Stats saved to {Config.STATS_PATH}")
    print("✅ Repair complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
