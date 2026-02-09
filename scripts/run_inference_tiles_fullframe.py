#!/usr/bin/env python3
"""
Full-frame reconstruction using a tile-trained model (sliding window + blending).
Generates a full HR prediction for a single timestep (last step of a sequence by default).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.models_legacy import ModelZoo


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


def _make_weight_window(h, w):
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    w2d = np.outer(wy, wx)
    # avoid all-zero edges
    w2d = np.maximum(w2d, 1e-6)
    return w2d.astype(np.float32)


def _find_time_index(times, time_value):
    times_pd = pd.to_datetime(times)
    target = pd.to_datetime(time_value)
    idx = np.searchsorted(times_pd.values, target.to_datetime64(), side="left")
    idx = min(max(idx, 0), len(times_pd) - 1)
    return int(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48, help="stride for sliding window (in HR pixels)")
    parser.add_argument("--batch-size", type=int, default=8, help="inference batch size")
    parser.add_argument("--time-index", type=int, default=None, help="index in HR time axis")
    parser.add_argument("--time", type=str, default=None, help="timestamp, e.g. 2017-06-15T12:00:00")
    parser.add_argument("--temporal-stride", type=int, default=None, help="override temporal stride")
    parser.add_argument("--use-last", action="store_true", help="use last timestep of sequence for output")
    parser.add_argument("--lr-channel", type=int, default=0, help="LR channel index for visualization")
    parser.add_argument("--experiment-name", default=None, help="label used in outputs and titles")
    parser.add_argument("--out", default=os.path.join("experiments", "figures", "tiles_fullframe_pred.png"))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return 2

    # Load dataset
    ds = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    da_lr = ds["lr_input"]
    da_hr = ds["hr_target"]

    lr_time = next((d for d in da_lr.dims if d in ["time", "valid_time", "t"]), "time")
    lr_lat = next((d for d in da_lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]), "y")
    lr_lon = next((d for d in da_lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]), "x")
    lr_var = next((d for d in da_lr.dims if d in ["variable", "channel", "var"]), None)

    hr_time = next((d for d in da_hr.dims if d in ["time", "valid_time", "t"]), "time")
    hr_lat = next((d for d in da_hr.dims if d in ["latitude", "lat", "y"]), "y")
    hr_lon = next((d for d in da_hr.dims if d in ["longitude", "lon", "x"]), "x")

    hr_h = da_hr.sizes[hr_lat]
    hr_w = da_hr.sizes[hr_lon]
    lr_h = da_lr.sizes[lr_lat]
    lr_w = da_lr.sizes[lr_lon]
    lr_c = da_lr.sizes[lr_var] if lr_var else 1

    patch = args.patch_size
    stride = args.stride
    stride = max(1, stride)
    patch_h, patch_w = patch, patch

    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
    lr_ph = max(1, int(round(patch_h * ratio_y)))
    lr_pw = max(1, int(round(patch_w * ratio_x)))

    # Experiment label
    exp_name = args.experiment_name or f"TilesFullFrame_{args.model_type.upper()}"

    # Update Config for model build
    Config.HR_SHAPE = (patch_h, patch_w)
    Config.LR_SHAPE = (lr_ph, lr_pw)
    Config.CHANNELS = lr_c

    # Time index selection
    times = ds[hr_time].values
    if args.time_index is not None:
        t0 = int(args.time_index)
    elif args.time is not None:
        t0 = _find_time_index(times, args.time)
    else:
        t0 = max(0, len(times) - 1)

    seq_len = Config.SEQ_LEN
    temporal_stride = args.temporal_stride if args.temporal_stride is not None else getattr(Config, "TEMPORAL_STRIDE", 1)
    temporal_stride = max(1, int(temporal_stride))
    if temporal_stride > seq_len:
        temporal_stride = seq_len

    t_idx = slice(t0, t0 + seq_len * temporal_stride, temporal_stride)
    if t0 + seq_len * temporal_stride > len(times):
        t0 = max(0, len(times) - seq_len * temporal_stride)
        t_idx = slice(t0, t0 + seq_len * temporal_stride, temporal_stride)

    # Detect lon order mismatch
    flip_lr_lon = False
    try:
        if _order(ds[lr_lon].values) != "unknown" and _order(ds[hr_lon].values) != "unknown":
            flip_lr_lon = _order(ds[lr_lon].values) != _order(ds[hr_lon].values)
    except Exception:
        pass

    # Load static cache + normalize
    static = np.load(Config.STATIC_CACHE_PATH)
    if static.ndim == 2:
        static = static[..., np.newaxis]
    if static.shape[0] != hr_h or static.shape[1] != hr_w:
        static = tf.image.resize(static, (hr_h, hr_w), method="bilinear").numpy()
    mean_st = np.mean(static, axis=(0, 1), keepdims=True)
    std_st = np.std(static, axis=(0, 1), keepdims=True)
    static_norm = (static - mean_st) / (std_st + 1e-6)

    # Load LR window for the chosen time slice (small, ok to load)
    if lr_var:
        lr_full = da_lr.isel({lr_time: t_idx}).transpose(lr_time, lr_lat, lr_lon, lr_var).values
    else:
        lr_full = da_lr.isel({lr_time: t_idx}).transpose(lr_time, lr_lat, lr_lon).values
        if lr_full.ndim == 3:
            lr_full = lr_full[..., np.newaxis]
    if flip_lr_lon:
        lr_full = lr_full[:, :, ::-1, :]

    # Load HR for reference (last timestep)
    hr_full = da_hr.isel({hr_time: t_idx}).transpose(hr_time, hr_lat, hr_lon).values
    if hr_full.ndim == 3:
        hr_full = hr_full[..., np.newaxis]

    # Build model
    model = build_model(args.model_type)
    model.load_weights(args.model_path)

    weight = _make_weight_window(patch_h, patch_w)
    out_sum = np.zeros((hr_h, hr_w), dtype=np.float32)
    weight_sum = np.zeros((hr_h, hr_w), dtype=np.float32)

    # Generate patch top-lefts
    ys = list(range(0, max(1, hr_h - patch_h + 1), stride))
    xs = list(range(0, max(1, hr_w - patch_w + 1), stride))
    if ys[-1] != hr_h - patch_h:
        ys.append(hr_h - patch_h)
    if xs[-1] != hr_w - patch_w:
        xs.append(hr_w - patch_w)

    coords = [(y0, x0) for y0 in ys for x0 in xs]
    batch_size = max(1, int(args.batch_size))

    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        x_lr_list = []
        x_st_list = []

        for y0, x0 in batch:
            lr_y0 = int(round(y0 * ratio_y))
            lr_x0 = int(round(x0 * ratio_x))
            lr_y0 = max(0, min(lr_y0, lr_h - lr_ph))
            lr_x0 = max(0, min(lr_x0, lr_w - lr_pw))

            x_lr = lr_full[:, lr_y0:lr_y0 + lr_ph, lr_x0:lr_x0 + lr_pw, :]
            st_patch = static_norm[y0:y0 + patch_h, x0:x0 + patch_w, :]
            x_st = np.broadcast_to(st_patch[np.newaxis, ...], (seq_len, *st_patch.shape))
            x_lr_list.append(x_lr)
            x_st_list.append(x_st)

        x_lr_b = np.stack(x_lr_list, axis=0)
        x_st_b = np.stack(x_st_list, axis=0)
        y_pred = model((x_lr_b, x_st_b), training=False).numpy()

        for (y0, x0), pred in zip(batch, y_pred):
            if args.use_last:
                patch_pred = pred[-1, :, :, 0]
            else:
                patch_pred = pred.mean(axis=0)[:, :, 0]
            out_sum[y0:y0 + patch_h, x0:x0 + patch_w] += patch_pred * weight
            weight_sum[y0:y0 + patch_h, x0:x0 + patch_w] += weight

    weight_sum[weight_sum == 0] = 1.0
    full_pred = out_sum / weight_sum

    # Save numpy
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.out))[0]
    out_npy = os.path.join(out_dir, f"{base_name}_{exp_name}.npy")
    out_png = os.path.join(out_dir, f"{base_name}_{exp_name}.png")
    np.save(out_npy, full_pred)

    # Plot comparison (LR upsampled vs Pred vs HR)
    hr_last = hr_full[-1, :, :, 0]
    lr_last = lr_full[-1, :, :, :]
    lr_ch = int(args.lr_channel)
    lr_ch = max(0, min(lr_ch, lr_last.shape[-1] - 1))
    lr_img = lr_last[:, :, lr_ch]
    lr_up = tf.image.resize(lr_img[..., None], (hr_h, hr_w), method="bilinear").numpy()[..., 0]

    vmin = float(np.nanmin(hr_last))
    vmax = float(np.nanmax(hr_last))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(lr_up, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"LR (upsampled) ch={lr_ch}")
    axes[0].axis("off")
    axes[1].imshow(full_pred, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction (tiles->full)")
    axes[1].axis("off")
    axes[2].imshow(hr_last, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[2].set_title("HR Ground Truth")
    axes[2].axis("off")
    fig.suptitle(exp_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_png}")

    # Metrics
    diff = full_pred - hr_last
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))

    ssim_val = None
    try:
        max_val = float(np.nanmax(hr_last) - np.nanmin(hr_last))
        if max_val > 0:
            ssim_val = float(
                tf.image.ssim(
                    hr_last[None, ..., None].astype(np.float32),
                    full_pred[None, ..., None].astype(np.float32),
                    max_val=max_val,
                ).numpy()[0]
            )
    except Exception:
        ssim_val = None

    metrics_path = os.path.join(out_dir, f"{base_name}_{exp_name}_metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("experiment,model,mae,rmse,ssim,patch_size,stride,seq_len,temporal_stride,time_index,time\n")
        f.write(
            f"{exp_name},{args.model_type},{mae:.6f},{rmse:.6f},{'' if ssim_val is None else f'{ssim_val:.6f}'},"
            f"{patch_h}x{patch_w},{stride},{seq_len},{temporal_stride},{t0},"
            f"{pd.to_datetime(times[t0]).isoformat()}\n"
        )
    print(f"Saved {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
