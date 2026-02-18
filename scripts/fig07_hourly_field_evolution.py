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
from src.models_legacy import ModelZoo


def _order(vals):
    diffs = np.diff(vals)
    if np.all(diffs > 0):
        return "asc"
    if np.all(diffs < 0):
        return "desc"
    return "unknown"


def _find_dims(da_lr, da_hr):
    lr_time = next((d for d in da_lr.dims if d in ["time", "valid_time", "t"]), "time")
    lr_lat = next((d for d in da_lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]), "y")
    lr_lon = next((d for d in da_lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]), "x")
    lr_var = next((d for d in da_lr.dims if d in ["variable", "channel", "var"]), None)
    hr_time = next((d for d in da_hr.dims if d in ["time", "valid_time", "t"]), "time")
    hr_lat = next((d for d in da_hr.dims if d in ["latitude", "lat", "y"]), "y")
    hr_lon = next((d for d in da_hr.dims if d in ["longitude", "lon", "x"]), "x")
    return lr_time, lr_lat, lr_lon, lr_var, hr_time, hr_lat, hr_lon


def _build_model(model_type: str):
    if model_type == "baseline_nearest":
        return ModelZoo.build_lr_upsample_nearest()
    if model_type == "baseline_bilinear":
        return ModelZoo.build_lr_upsample_bilinear()
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    if model_type == "transformer":
        return ModelZoo.build_transformer()
    return ModelZoo.build_unet()


def _make_weight_window(h, w):
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    w2d = np.outer(wy, wx)
    return np.maximum(w2d, 1e-6).astype(np.float32)


def _resolve_lr_channel(da_lr, lr_var_dim, requested_channel):
    if requested_channel is not None and int(requested_channel) >= 0:
        return int(requested_channel), f"ch={int(requested_channel)}"
    if lr_var_dim and lr_var_dim in da_lr.coords:
        try:
            names = [str(v) for v in da_lr.coords[lr_var_dim].values.tolist()]
            if "t2m" in names:
                idx = names.index("t2m")
                return idx, "t2m"
        except Exception:
            pass
    return 0, "ch=0"


def _run_fullframe_inference(
    model,
    model_type,
    lr_full,
    static_norm,
    hr_shape,
    lr_shape,
    patch,
    stride,
    batch_size,
):
    is_baseline = str(model_type).startswith("baseline_")
    hr_h, hr_w = hr_shape
    lr_h, lr_w = lr_shape
    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
    patch_h, patch_w = patch
    lr_ph = max(1, int(round(patch_h * ratio_y)))
    lr_pw = max(1, int(round(patch_w * ratio_x)))

    ys = list(range(0, max(1, hr_h - patch_h + 1), stride))
    xs = list(range(0, max(1, hr_w - patch_w + 1), stride))
    if ys[-1] != hr_h - patch_h:
        ys.append(hr_h - patch_h)
    if xs[-1] != hr_w - patch_w:
        xs.append(hr_w - patch_w)
    coords = [(y0, x0) for y0 in ys for x0 in xs]

    weight = _make_weight_window(patch_h, patch_w)
    out_sum = np.zeros((hr_h, hr_w), dtype=np.float32)
    weight_sum = np.zeros((hr_h, hr_w), dtype=np.float32)

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
            x_st = np.broadcast_to(st_patch[np.newaxis, ...], (lr_full.shape[0], *st_patch.shape))
            x_lr_list.append(x_lr)
            x_st_list.append(x_st)

        x_lr_b = np.stack(x_lr_list, axis=0)
        x_st_b = np.stack(x_st_list, axis=0)
        y_pred = model((x_lr_b, x_st_b), training=False).numpy()

        for (y0, x0), pred in zip(batch, y_pred):
            patch_pred = pred[-1, :, :, 0]
            out_sum[y0:y0 + patch_h, x0:x0 + patch_w] += patch_pred * weight
            weight_sum[y0:y0 + patch_h, x0:x0 + patch_w] += weight

    weight_sum[weight_sum == 0] = 1.0
    return out_sum / weight_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="transformer",
        choices=["mamba", "unet", "convlstm", "transformer", "baseline_nearest", "baseline_bilinear"],
    )
    parser.add_argument("--model-path", default=None)
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
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    is_baseline = str(args.model_type).startswith("baseline_")
    if not is_baseline and (not args.model_path or not os.path.exists(args.model_path)):
        raise SystemExit(f"Model not found: {args.model_path}")

    os.makedirs(args.out_dir, exist_ok=True)

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

    model = _build_model(args.model_type)
    if not is_baseline:
        model.load_weights(args.model_path)

    lr_ch, lr_label = _resolve_lr_channel(da_lr, lr_var, args.lr_channel)
    prefix = f"{args.model_type}_{args.day}"
    if args.tag:
        prefix = f"{prefix}_{args.tag}"

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

        full_pred = _run_fullframe_inference(
            model=model,
            model_type=args.model_type,
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
        abs_err = np.abs(full_pred - hr_last)
        mae = float(np.nanmean(abs_err))
        rmse = float(np.sqrt(np.nanmean((full_pred - hr_last) ** 2)))

        timestamp = pd.to_datetime(times[tidx]).isoformat()
        records.append(
            {
                "time": timestamp,
                "hour": int(pd.to_datetime(times[tidx]).hour),
                "mae": mae,
                "rmse": rmse,
                "pred_mean": float(np.nanmean(full_pred)),
                "hr_mean": float(np.nanmean(hr_last)),
                "pred_std": float(np.nanstd(full_pred)),
                "hr_std": float(np.nanstd(hr_last)),
            }
        )
        lr_for_scale.append(lr_up)
        hr_for_scale.append(hr_last)
        frames_data.append((timestamp, lr_up, full_pred, hr_last, abs_err, mae, rmse))

    if not frames_data:
        raise SystemExit(f"No valid hourly frames were generated for {args.day}")

    all_scale = np.concatenate([np.concatenate([x.ravel() for x in lr_for_scale]), np.concatenate([x.ravel() for x in hr_for_scale])])
    all_scale = all_scale[np.isfinite(all_scale)]
    vmin = float(np.percentile(all_scale, 2.0))
    vmax = float(np.percentile(all_scale, 98.0))

    err_scale = np.concatenate([x[4].ravel() for x in frames_data])
    err_scale = err_scale[np.isfinite(err_scale)]
    err_vmax = float(np.percentile(err_scale, 99.0))

    gif_frames = []
    for timestamp, lr_up, pred, hr, abs_err, mae, rmse in frames_data:
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
        im0 = axes[0].imshow(pred, cmap=args.temp_cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
        axes[0].set_title("Prediction")
        axes[0].axis("off")
        im1 = axes[1].imshow(hr, cmap=args.temp_cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        im2 = axes[2].imshow(abs_err, cmap="magma", origin="lower", vmin=0.0, vmax=err_vmax, interpolation="none")
        axes[2].set_title("Absolute Error")
        axes[2].axis("off")
        fig.colorbar(im0, ax=axes[:2], fraction=0.035, pad=0.02)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
        fig.suptitle(f"{args.model_type.upper()} | {timestamp} | MAE={mae:.4f} RMSE={rmse:.4f} | LR={lr_label}", fontsize=11)
        buf = io.BytesIO()
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        gif_frames.append(imageio.imread(buf))
        buf.close()

    gif_path = os.path.join(args.out_dir, f"fig07_hourly_field_evolution_{prefix}.gif")
    imageio.mimsave(gif_path, gif_frames, duration=max(0.05, 1.0 / max(0.1, float(args.fps))))

    metrics_df = pd.DataFrame(records).sort_values("time")
    csv_path = os.path.join(args.out_dir, f"fig07_hourly_field_evolution_{prefix}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(11, 4.2))
    x = metrics_df["hour"].values
    plt.plot(x, metrics_df["hr_mean"].values, label="HR mean")
    plt.plot(x, metrics_df["pred_mean"].values, label="Pred mean")
    plt.twinx()
    plt.plot(x, metrics_df["mae"].values, color="tab:red", linestyle="--", label="MAE")
    plt.ylabel("MAE")
    ax = plt.gca()
    h1, l1 = ax.get_legend_handles_labels()
    h0, l0 = plt.gcf().axes[0].get_legend_handles_labels()
    plt.gcf().axes[0].legend(h0 + h1, l0 + l1, loc="upper left")
    plt.gcf().axes[0].set_xlabel("Hour (local dataset time)")
    plt.gcf().axes[0].set_ylabel("Domain mean temperature")
    plt.gcf().axes[0].set_title(f"Hourly Evolution Summary | {args.model_type.upper()} | {args.day}")
    plt.gcf().axes[0].grid(True, alpha=0.3)
    summary_path = os.path.join(args.out_dir, f"fig07_hourly_field_evolution_{prefix}_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=180)
    plt.close()

    print(f"Saved GIF: {gif_path}")
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Shared scale (hr_lr): vmin={vmin:.6f}, vmax={vmax:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())

