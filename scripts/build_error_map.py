#!/usr/bin/env python3
"""
Build a per-pixel error map (MAE) to drive error-weighted tile sampling.
Saves to experiments/tiles_error_map.npy by default.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

from config.runtime import Config
from src.models_legacy import ModelZoo


def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    return ModelZoo.build_unet()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--out", default=os.path.join("experiments", "tiles_error_map.npy"))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return 2

    z = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    lr = z["lr_input"]
    hr = z["hr_target"]

    lr_time = next((d for d in lr.dims if d in ["time", "valid_time", "t"]), "time")
    lr_lat = next((d for d in lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]), "y")
    lr_lon = next((d for d in lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]), "x")
    lr_var = next((d for d in lr.dims if d in ["variable", "channel", "var"]), None)

    hr_time = next((d for d in hr.dims if d in ["time", "valid_time", "t"]), "time")
    hr_lat = next((d for d in hr.dims if d in ["latitude", "lat", "y"]), "y")
    hr_lon = next((d for d in hr.dims if d in ["longitude", "lon", "x"]), "x")

    hr_h = hr.sizes[hr_lat]
    hr_w = hr.sizes[hr_lon]
    lr_h = lr.sizes[lr_lat]
    lr_w = lr.sizes[lr_lon]
    if lr_var:
        lr_c = lr.sizes[lr_var]
    else:
        lr_c = 1

    patch = args.patch_size
    patch_h, patch_w = patch, patch
    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
    lr_ph = max(1, int(round(patch_h * ratio_y)))
    lr_pw = max(1, int(round(patch_w * ratio_x)))

    # Update Config shapes for model build
    Config.HR_SHAPE = (patch_h, patch_w)
    Config.LR_SHAPE = (lr_ph, lr_pw)
    Config.CHANNELS = lr_c
    Config.TEMPORAL_STRIDE = args.temporal_stride

    # Static normalization
    static = np.load(Config.STATIC_CACHE_PATH)
    if static.ndim == 2:
        static = static[..., np.newaxis]
    if static.shape[0] != hr_h or static.shape[1] != hr_w:
        static = tf.image.resize(static, (hr_h, hr_w), method="bilinear").numpy()
    mean_st = np.mean(static, axis=(0, 1), keepdims=True)
    std_st = np.std(static, axis=(0, 1), keepdims=True)
    static_norm = (static - mean_st) / (std_st + 1e-6)

    # Time splits
    total_len = z.sizes[lr_time]
    def _time_indices(times, start, end):
        times = pd.to_datetime(times).values
        start = np.datetime64(start)
        end = np.datetime64(end)
        start_idx = int(np.searchsorted(times, start, side="left"))
        end_idx = int(np.searchsorted(times, end, side="left"))
        return start_idx, end_idx

    if getattr(Config, "SPLIT_MODE", "fraction") == "time":
        times = z[lr_time].values
        train_start, train_end = _time_indices(times, Config.TRAIN_START, Config.TRAIN_END)
        val_start, val_end = _time_indices(times, Config.VAL_START, Config.VAL_END)
        test_start, test_end = _time_indices(times, Config.TEST_START, Config.TEST_END)
    else:
        split_idx = int(total_len * Config.SPLIT_FRACTION)
        train_start, train_end = 0, split_idx
        val_start, val_end = split_idx, total_len
        test_start, test_end = val_end, total_len

    if args.split == "train":
        start_i, end_i = train_start, train_end
    elif args.split == "val":
        start_i, end_i = val_start, val_end
    else:
        start_i, end_i = test_start, test_end

    # Longitude orientation check
    def _order(vals):
        diffs = np.diff(vals)
        if np.all(diffs > 0):
            return "asc"
        if np.all(diffs < 0):
            return "desc"
        return "unknown"

    flip_lr_lon = False
    try:
        if _order(z[lr_lon].values) != "unknown" and _order(z[hr_lon].values) != "unknown":
            flip_lr_lon = _order(z[lr_lon].values) != _order(z[hr_lon].values)
    except Exception:
        pass

    # Build model
    model = build_model(args.model_type)
    model.load_weights(args.model_path)

    rng = np.random.default_rng(getattr(Config, "SEED", 42))
    seq_len = Config.SEQ_LEN

    err_map = np.zeros((hr_h, hr_w), dtype=np.float32)
    count_map = np.zeros((hr_h, hr_w), dtype=np.float32)

    stride = int(getattr(Config, "TEMPORAL_STRIDE", 1))

    for _ in range(args.samples):
        if end_i - start_i <= seq_len + 1:
            break
        t0 = int(rng.integers(start_i, end_i - seq_len))
        y0 = int(rng.integers(0, max(1, hr_h - patch_h + 1)))
        x0 = int(rng.integers(0, max(1, hr_w - patch_w + 1)))

        lr_y0 = int(round(y0 * ratio_y))
        lr_x0 = int(round(x0 * ratio_x))
        lr_y0 = max(0, min(lr_y0, lr_h - lr_ph))
        lr_x0 = max(0, min(lr_x0, lr_w - lr_pw))

        t_idx = slice(t0, t0 + seq_len * stride, stride)

        if lr_var:
            x_lr = lr.isel({lr_time: t_idx,
                            lr_lat: slice(lr_y0, lr_y0 + lr_ph),
                            lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
                    .transpose(lr_time, lr_lat, lr_lon, lr_var).values
        else:
            x_lr = lr.isel({lr_time: t_idx,
                            lr_lat: slice(lr_y0, lr_y0 + lr_ph),
                            lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
                    .transpose(lr_time, lr_lat, lr_lon).values
            if x_lr.ndim == 3:
                x_lr = x_lr[..., np.newaxis]

        if flip_lr_lon:
            x_lr = x_lr[:, :, ::-1, :]

        y_hr = hr.isel({hr_time: t_idx,
                        hr_lat: slice(y0, y0 + patch_h),
                        hr_lon: slice(x0, x0 + patch_w)}) \
                .transpose(hr_time, hr_lat, hr_lon).values
        if y_hr.ndim == 3:
            y_hr = y_hr[..., np.newaxis]

        st_patch = static_norm[y0:y0 + patch_h, x0:x0 + patch_w, :]
        x_st = np.broadcast_to(st_patch[np.newaxis, ...], (seq_len, *st_patch.shape))

        x_lr_b = x_lr[np.newaxis, ...]
        x_st_b = x_st[np.newaxis, ...]
        y_pred = model((x_lr_b, x_st_b), training=False).numpy()

        # Use last timestep error
        err = np.abs(y_pred[0, -1, :, :, 0] - y_hr[-1, :, :, 0])
        err_map[y0:y0 + patch_h, x0:x0 + patch_w] += err
        count_map[y0:y0 + patch_h, x0:x0 + patch_w] += 1

    count_map[count_map == 0] = 1
    err_map = err_map / count_map

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, err_map)
    print(f"✅ Error map saved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
