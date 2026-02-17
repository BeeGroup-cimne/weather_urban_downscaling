#!/usr/bin/env python3
"""
A/B comparison of data pipeline settings (main-like vs current).
Generates LR/HR/Static snapshots from the same time index for visual inspection.
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import xarray as xr
from config.runtime import Config
from src.data_loader import BigDataPipeline


def _resize_nearest(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=0)


def _pick_dims(da):
    lat = next((d for d in da.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]), None)
    lon = next((d for d in da.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]), None)
    return lat or da.dims[-2], lon or da.dims[-1]


def _apply_overrides(cfg, overrides):
    for k, v in overrides.items():
        setattr(cfg, k, v)


def _run_variant(tag, cfg, args, overrides, sample_idx):
    out_root = os.path.abspath(os.path.join(cfg.BASE_DIR, "experiments", "ab_compare"))
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    base_static_path = cfg.STATIC_CACHE_PATH

    cfg.PATH_CACHE = os.path.join(out_dir, "weather_cache.zarr")
    cfg.STATS_PATH = os.path.join(out_dir, "stats_config.npz")
    cfg.STATIC_CACHE_PATH = os.path.join(out_dir, "static_processed.npy")
    cfg.EXPERIMENTS_DIR = out_root

    _apply_overrides(cfg, overrides)

    pipeline = BigDataPipeline(cfg)

    static_data = None
    if (not args.force_process_static) and base_static_path and os.path.exists(base_static_path):
        try:
            static_data = np.load(base_static_path)
            print(f"✅ Static loaded from base cache: {base_static_path}")
        except Exception as e:
            print(f"⚠️ No se pudo cargar static base ({base_static_path}): {e}")

    if static_data is None:
        pipeline.process_static_data()
        static_data = np.load(cfg.STATIC_CACHE_PATH)

    if (not args.reuse_cache) or (not os.path.exists(cfg.PATH_CACHE)):
        pipeline.run_etl_process()

    ds = xr.open_zarr(cfg.PATH_CACHE, consolidated=True)
    hr = ds["hr_target"]
    lr = ds["lr_input"]

    total = hr.sizes.get("time", hr.shape[0])
    if sample_idx is None:
        sample_idx = int(np.random.default_rng(args.seed).integers(0, total))

    hr_lat, hr_lon = _pick_dims(hr)
    hr_sample = hr.isel(time=sample_idx).transpose(hr_lat, hr_lon).values
    if hr_sample.ndim > 2:
        hr_sample = hr_sample[..., 0]

    lr_lat, lr_lon = _pick_dims(lr)
    if "variable" in lr.dims:
        lr_sel = lr.isel(time=sample_idx, variable=args.lr_channel)
    else:
        lr_sel = lr.isel(time=sample_idx)
    lr_sample = lr_sel.transpose(lr_lat, lr_lon).values
    if lr_sample.ndim > 2:
        lr_sample = lr_sample[..., 0]
    lr_up = _resize_nearest(lr_sample, hr_sample.shape)

    static_ch = static_data[:, :, args.static_channel]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr_up, cmap="viridis", origin="lower")
    axs[0].set_title(f"{tag} LR (upsampled)")
    axs[1].imshow(hr_sample, cmap="viridis", origin="lower")
    axs[1].set_title(f"{tag} HR")
    axs[2].imshow(static_ch, cmap="viridis", origin="lower")
    axs[2].set_title(f"{tag} Static[{args.static_channel}]")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Sample idx={sample_idx}")
    out_path = os.path.join(out_dir, f"ab_compare_{tag}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved {out_path}")
    return sample_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--force-process-static", action="store_true",
                        help="Force recompute static cache (otherwise reuse base cache if available)")
    parser.add_argument("--lr-channel", type=int, default=0)
    parser.add_argument("--static-channel", type=int, default=12)
    args = parser.parse_args()

    # A = main-like
    cfg = Config
    overrides_a = {
        "SPLIT_MODE": "fraction",
        "STATS_TRAIN_ONLY": False,
        "GAUSSIAN_NOISE_STD": 0.0,
        "L2_WEIGHT_DECAY": 0.0,
    }

    # B = current (time split + regularization)
    overrides_b = {
        "SPLIT_MODE": "time",
        "STATS_TRAIN_ONLY": True,
        "GAUSSIAN_NOISE_STD": 0.01,
        "L2_WEIGHT_DECAY": 1e-4,
    }

    idx = _run_variant("A_main_like", cfg, args, overrides_a, args.sample_index)
    _run_variant("B_current", cfg, args, overrides_b, idx)


if __name__ == "__main__":
    raise SystemExit(main())
