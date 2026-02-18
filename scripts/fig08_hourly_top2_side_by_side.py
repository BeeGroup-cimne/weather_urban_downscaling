#!/usr/bin/env python3

import argparse
import io
import os
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from scripts.fig07_hourly_field_evolution import (
    _build_model,
    _find_dims,
    _order,
    _resolve_lr_channel,
    _run_fullframe_inference,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a-type", default="mamba", choices=["mamba", "unet", "convlstm", "transformer", "baseline_nearest", "baseline_bilinear"])
    parser.add_argument("--model-a-path", default="experiments/models/Tiles_MAMBA_best.h5")
    parser.add_argument("--model-b-type", default="unet", choices=["mamba", "unet", "convlstm", "transformer", "baseline_nearest", "baseline_bilinear"])
    parser.add_argument("--model-b-path", default="experiments/models/Tiles_UNET_best.h5")
    parser.add_argument("--day", default="2017-08-15")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temporal-stride", type=int, default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--lr-channel", type=int, default=None)
    parser.add_argument("--lr-resample", default="nearest", choices=["nearest", "bilinear"])
    parser.add_argument("--temp-cmap", default="inferno")
    parser.add_argument("--out-dir", default=os.path.join("experiments", "figures"))
    parser.add_argument("--tag", default="best2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for model_type, model_path in [
        (args.model_a_type, args.model_a_path),
        (args.model_b_type, args.model_b_path),
    ]:
        is_baseline = str(model_type).startswith("baseline_")
        if not is_baseline and (not model_path or not os.path.exists(model_path)):
            raise SystemExit(f"Model not found: {model_path}")

    ds = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    da_lr = ds["lr_input"]
    da_hr = ds["hr_target"]
    lr_time, lr_lat, lr_lon, lr_var, hr_time, hr_lat, hr_lon = _find_dims(da_lr, da_hr)

    times = pd.to_datetime(ds[hr_time].values)
    day_start = pd.Timestamp(args.day)
    day_end = day_start + pd.Timedelta(days=1)
    target_idx = np.where((times >= day_start) & (times < day_end))[0].tolist()
    if not target_idx:
        raise SystemExit(f"No timestamps found for day {args.day}")

    hr_h = da_hr.sizes[hr_lat]
    hr_w = da_hr.sizes[hr_lon]
    lr_h = da_lr.sizes[lr_lat]
    lr_w = da_lr.sizes[lr_lon]
    lr_c = da_lr.sizes[lr_var] if lr_var else 1

    patch_h = int(args.patch_size)
    patch_w = int(args.patch_size)
    Config.HR_SHAPE = (patch_h, patch_w)
    Config.LR_SHAPE = (
        max(1, int(round(patch_h * (lr_h / float(hr_h))))),
        max(1, int(round(patch_w * (lr_w / float(hr_w))))),
    )
    Config.CHANNELS = lr_c

    temporal_stride = args.temporal_stride if args.temporal_stride is not None else int(getattr(Config, "TEMPORAL_STRIDE", 1))
    temporal_stride = max(1, temporal_stride)
    seq_len = int(getattr(Config, "SEQ_LEN", 6))

    static = np.load(Config.STATIC_CACHE_PATH)
    if static.ndim == 2:
        static = static[..., np.newaxis]
    if static.shape[0] != hr_h or static.shape[1] != hr_w:
        static = tf.image.resize(static, (hr_h, hr_w), method="bilinear").numpy()
    mean_st = np.mean(static, axis=(0, 1), keepdims=True)
    std_st = np.std(static, axis=(0, 1), keepdims=True)
    static_norm = (static - mean_st) / (std_st + 1e-6)

    flip_lr_lon = False
    try:
        if _order(ds[lr_lon].values) != "unknown" and _order(ds[hr_lon].values) != "unknown":
            flip_lr_lon = _order(ds[lr_lon].values) != _order(ds[hr_lon].values)
    except Exception:
        pass

    model_a = _build_model(args.model_a_type)
    model_b = _build_model(args.model_b_type)

    if not str(args.model_a_type).startswith("baseline_"):
        model_a.load_weights(args.model_a_path)
    if not str(args.model_b_type).startswith("baseline_"):
        model_b.load_weights(args.model_b_path)

    lr_ch, _ = _resolve_lr_channel(da_lr, lr_var, args.lr_channel)

    records = []
    lr_for_scale = []
    hr_for_scale = []
    frames_data = []

    for tidx in target_idx:
        start = tidx - (seq_len - 1) * temporal_stride
        if start < 0:
            continue
        t_sel = list(range(start, tidx + 1, temporal_stride))
        if len(t_sel) != seq_len:
            continue

        if lr_var:
            lr_full = da_lr.isel({lr_time: t_sel}).transpose(lr_time, lr_lat, lr_lon, lr_var).values
        else:
            lr_full = da_lr.isel({lr_time: t_sel}).transpose(lr_time, lr_lat, lr_lon).values[..., np.newaxis]

        if flip_lr_lon:
            lr_full = lr_full[:, :, ::-1, :]

        hr_seq = da_hr.isel({hr_time: t_sel}).transpose(hr_time, hr_lat, hr_lon).values
        if hr_seq.ndim == 3:
            hr_seq = hr_seq[..., np.newaxis]
        hr_last = hr_seq[-1, :, :, 0]

        pred_a = _run_fullframe_inference(
            model=model_a,
            model_type=args.model_a_type,
            lr_full=lr_full,
            static_norm=static_norm,
            hr_shape=(hr_h, hr_w),
            lr_shape=(lr_h, lr_w),
            patch=(patch_h, patch_w),
            stride=int(args.stride),
            batch_size=max(1, int(args.batch_size)),
        )
        pred_b = _run_fullframe_inference(
            model=model_b,
            model_type=args.model_b_type,
            lr_full=lr_full,
            static_norm=static_norm,
            hr_shape=(hr_h, hr_w),
            lr_shape=(lr_h, lr_w),
            patch=(patch_h, patch_w),
            stride=int(args.stride),
            batch_size=max(1, int(args.batch_size)),
        )

        lr_img = lr_full[-1, :, :, max(0, min(lr_ch, lr_full.shape[-1] - 1))]
        lr_up = tf.image.resize(lr_img[..., None], (hr_h, hr_w), method=args.lr_resample).numpy()[..., 0]

        mae_a = float(np.nanmean(np.abs(pred_a - hr_last)))
        mae_b = float(np.nanmean(np.abs(pred_b - hr_last)))
        rmse_a = float(np.sqrt(np.nanmean((pred_a - hr_last) ** 2)))
        rmse_b = float(np.sqrt(np.nanmean((pred_b - hr_last) ** 2)))

        timestamp = pd.to_datetime(times[tidx]).isoformat()
        records.append(
            {
                "time": timestamp,
                "hour": int(pd.to_datetime(times[tidx]).hour),
                "mae_model_a": mae_a,
                "mae_model_b": mae_b,
                "rmse_model_a": rmse_a,
                "rmse_model_b": rmse_b,
                "pred_a_mean": float(np.nanmean(pred_a)),
                "pred_b_mean": float(np.nanmean(pred_b)),
                "hr_mean": float(np.nanmean(hr_last)),
            }
        )
        lr_for_scale.append(lr_up)
        hr_for_scale.append(hr_last)
        frames_data.append((timestamp, pred_a, pred_b, hr_last, mae_a, mae_b))

    if not frames_data:
        raise SystemExit(f"No valid hourly frames were generated for {args.day}")

    all_scale = np.concatenate([np.concatenate([x.ravel() for x in lr_for_scale]), np.concatenate([x.ravel() for x in hr_for_scale])])
    all_scale = all_scale[np.isfinite(all_scale)]
    vmin = float(np.percentile(all_scale, 2.0))
    vmax = float(np.percentile(all_scale, 98.0))

    gif_frames = []
    for timestamp, pred_a, pred_b, hr, mae_a, mae_b in frames_data:
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
        im0 = axes[0].imshow(pred_a, cmap=args.temp_cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
        axes[0].set_title(f"{args.model_a_type.upper()} Prediction")
        axes[0].axis("off")
        axes[1].imshow(pred_b, cmap=args.temp_cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
        axes[1].set_title(f"{args.model_b_type.upper()} Prediction")
        axes[1].axis("off")
        axes[2].imshow(hr, cmap=args.temp_cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
        fig.colorbar(im0, ax=axes.tolist(), fraction=0.025, pad=0.02)
        fig.suptitle(
            f"{timestamp} | {args.model_a_type.upper()} MAE={mae_a:.4f} | {args.model_b_type.upper()} MAE={mae_b:.4f}",
            fontsize=11,
        )
        buf = io.BytesIO()
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        gif_frames.append(imageio.imread(buf))
        buf.close()

    prefix = f"{args.model_a_type}_{args.model_b_type}_{args.day}"
    if args.tag:
        prefix = f"{prefix}_{args.tag}"

    gif_path = os.path.join(args.out_dir, f"fig08_hourly_top2_side_by_side_{prefix}.gif")
    imageio.mimsave(gif_path, gif_frames, duration=max(0.05, 1.0 / max(0.1, float(args.fps))))

    metrics_df = pd.DataFrame(records).sort_values("time")
    csv_path = os.path.join(args.out_dir, f"fig08_hourly_top2_side_by_side_{prefix}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10.8, 4.2))
    x = metrics_df["hour"].values
    plt.plot(x, metrics_df["mae_model_a"].values, label=f"{args.model_a_type.upper()} MAE")
    plt.plot(x, metrics_df["mae_model_b"].values, label=f"{args.model_b_type.upper()} MAE")
    plt.xlabel("Hour (dataset local time)")
    plt.ylabel("MAE")
    plt.title(f"Hourly MAE Comparison | {args.model_a_type.upper()} vs {args.model_b_type.upper()} | {args.day}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    summary_path = os.path.join(args.out_dir, f"fig08_hourly_top2_side_by_side_{prefix}_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=180)
    plt.close()

    print(f"Saved GIF: {gif_path}")
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Shared scale (hr_lr): vmin={vmin:.6f}, vmax={vmax:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())

