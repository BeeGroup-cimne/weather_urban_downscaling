#!/usr/bin/env python3
"""
Overfit a single fixed tile/time sequence to validate pipeline alignment.
If loss doesn't drop quickly, suspect data/target alignment issues.
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.models_legacy import ModelZoo
from src.losses import tf_hybrid_loss


def _order(vals):
    diffs = np.diff(vals)
    if np.all(diffs > 0):
        return "asc"
    if np.all(diffs < 0):
        return "desc"
    return "unknown"


def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    return ModelZoo.build_unet()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--lr-channel", type=int, default=0)
    parser.add_argument("--temporal-stride", type=int, default=None)
    parser.add_argument("--loss", default="hybrid", choices=["hybrid", "mse"])
    args = parser.parse_args()

    ds = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    lr = ds["lr_input"]
    hr = ds["hr_target"]

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
    lr_c = lr.sizes[lr_var] if lr_var else 1

    patch_h = patch_w = args.patch_size
    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
    lr_ph = max(1, int(round(patch_h * ratio_y)))
    lr_pw = max(1, int(round(patch_w * ratio_x)))

    # Pick a centered patch
    y0 = max(0, (hr_h - patch_h) // 2)
    x0 = max(0, (hr_w - patch_w) // 2)
    lr_y0 = int(round(y0 * ratio_y))
    lr_x0 = int(round(x0 * ratio_x))

    # Config override for model build
    Config.HR_SHAPE = (patch_h, patch_w)
    Config.LR_SHAPE = (lr_ph, lr_pw)
    Config.CHANNELS = lr_c

    seq_len = Config.SEQ_LEN
    temporal_stride = args.temporal_stride if args.temporal_stride is not None else getattr(Config, "TEMPORAL_STRIDE", 1)
    temporal_stride = max(1, int(temporal_stride))
    if temporal_stride > seq_len:
        temporal_stride = seq_len

    t0 = max(0, min(int(args.time_index), lr.sizes[lr_time] - seq_len * temporal_stride))
    t_idx = slice(t0, t0 + seq_len * temporal_stride, temporal_stride)

    flip_lr_lon = False
    try:
        if _order(ds[lr_lon].values) != "unknown" and _order(ds[hr_lon].values) != "unknown":
            flip_lr_lon = _order(ds[lr_lon].values) != _order(ds[hr_lon].values)
    except Exception:
        pass

    # Static
    static = np.load(Config.STATIC_CACHE_PATH)
    if static.ndim == 2:
        static = static[..., np.newaxis]
    if static.shape[0] != hr_h or static.shape[1] != hr_w:
        static = tf.image.resize(static, (hr_h, hr_w), method="bilinear").numpy()
    mean_st = np.mean(static, axis=(0, 1), keepdims=True)
    std_st = np.std(static, axis=(0, 1), keepdims=True)
    static_norm = (static - mean_st) / (std_st + 1e-6)

    # Build inputs
    if lr_var:
        x_lr = lr.isel({lr_time: t_idx, lr_lat: slice(lr_y0, lr_y0 + lr_ph), lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
            .transpose(lr_time, lr_lat, lr_lon, lr_var).values
    else:
        x_lr = lr.isel({lr_time: t_idx, lr_lat: slice(lr_y0, lr_y0 + lr_ph), lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
            .transpose(lr_time, lr_lat, lr_lon).values
        if x_lr.ndim == 3:
            x_lr = x_lr[..., np.newaxis]

    if flip_lr_lon:
        x_lr = x_lr[:, :, ::-1, :]

    y_hr = hr.isel({hr_time: t_idx, hr_lat: slice(y0, y0 + patch_h), hr_lon: slice(x0, x0 + patch_w)}) \
        .transpose(hr_time, hr_lat, hr_lon).values
    if y_hr.ndim == 3:
        y_hr = y_hr[..., np.newaxis]

    st_patch = static_norm[y0:y0 + patch_h, x0:x0 + patch_w, :]
    x_st = np.broadcast_to(st_patch[np.newaxis, ...], (seq_len, *st_patch.shape))

    # Dataset (single sample, repeated)
    ds_train = tf.data.Dataset.from_tensors(((x_lr, x_st), y_hr)).repeat().batch(1)

    model = build_model(args.model_type)
    if args.loss == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = tf_hybrid_loss(alpha=0.8, max_val=5.0)
    opt = ModelZoo.get_optimizer(Config.LEARNING_RATE)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["mae", "mse"])

    print("🧪 Fixed-tile overfit sanity")
    print(f"   Patch: {patch_h}x{patch_w}, time_index: {t0}, steps: {args.steps}")
    model.fit(ds_train, epochs=args.epochs, steps_per_epoch=args.steps, verbose=1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
