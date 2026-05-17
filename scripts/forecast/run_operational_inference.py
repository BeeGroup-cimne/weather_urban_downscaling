#!/usr/bin/env python3
"""Run operational downscaling inference from normalized LR forecast tensors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.runtime import Config
from src.models_legacy import ModelZoo


def _order(vals: np.ndarray) -> str:
    diffs = np.diff(vals)
    if np.all(diffs > 0):
        return "asc"
    if np.all(diffs < 0):
        return "desc"
    return "unknown"


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


def _load_stats(path: Path) -> tuple[float, float]:
    stats = np.load(path, allow_pickle=True)
    return float(stats["mean_hr"]), float(stats["std_hr"])


def _load_static(path: Path, hr_shape: tuple[int, int]) -> np.ndarray:
    static = np.load(path).astype(np.float32)
    if static.ndim == 2:
        static = static[..., np.newaxis]
    if static.shape[:2] != hr_shape:
        static = tf.image.resize(static, hr_shape, method="bilinear").numpy()
    mean_st = np.mean(static, axis=(0, 1), keepdims=True)
    std_st = np.std(static, axis=(0, 1), keepdims=True)
    return ((static - mean_st) / (std_st + 1e-6)).astype(np.float32)


def _make_weight_window(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    weight = np.outer(wy, wx)
    return np.maximum(weight, 1e-6).astype(np.float32)


def _coords_for_tiles(hr_h: int, hr_w: int, patch_h: int, patch_w: int, stride: int) -> list[tuple[int, int]]:
    stride = max(1, int(stride))
    ys = list(range(0, max(1, hr_h - patch_h + 1), stride))
    xs = list(range(0, max(1, hr_w - patch_w + 1), stride))
    if ys[-1] != hr_h - patch_h:
        ys.append(hr_h - patch_h)
    if xs[-1] != hr_w - patch_w:
        xs.append(hr_w - patch_w)
    return [(y0, x0) for y0 in ys for x0 in xs]


def _find_dims(da_lr: xr.DataArray) -> tuple[str, str, str, str | None]:
    lr_time = next((d for d in da_lr.dims if d in ["time", "valid_time", "t"]), "time")
    lr_lat = next((d for d in da_lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]), "y")
    lr_lon = next((d for d in da_lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]), "x")
    lr_var = next((d for d in da_lr.dims if d in ["variable", "channel", "var"]), None)
    return lr_time, lr_lat, lr_lon, lr_var


def _target_indices(n_times: int, seq_len: int, temporal_stride: int, max_predictions: int) -> list[int]:
    first = (seq_len - 1) * temporal_stride
    indices = list(range(first, n_times))
    if max_predictions and max_predictions > 0:
        indices = indices[: int(max_predictions)]
    return indices


def _timestamp_name(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y_%m_%d_%H_%M_%S")


def _run_one_timestamp(
    *,
    model,
    lr_seq: np.ndarray,
    static_norm: np.ndarray,
    coords: list[tuple[int, int]],
    full_hr_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    lr_patch_shape: tuple[int, int],
    batch_size: int,
    output_mode: str,
) -> np.ndarray:
    hr_h, hr_w = full_hr_shape
    patch_h, patch_w = patch_shape
    lr_ph, lr_pw = lr_patch_shape
    lr_h, lr_w = lr_seq.shape[1], lr_seq.shape[2]
    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
    weight = _make_weight_window(patch_h, patch_w)
    out_sum = np.zeros((hr_h, hr_w), dtype=np.float32)
    weight_sum = np.zeros((hr_h, hr_w), dtype=np.float32)

    for i in range(0, len(coords), batch_size):
        batch = coords[i : i + batch_size]
        x_lr_list = []
        x_st_list = []
        for y0, x0 in batch:
            lr_y0 = int(round(y0 * ratio_y))
            lr_x0 = int(round(x0 * ratio_x))
            lr_y0 = max(0, min(lr_y0, lr_h - lr_ph))
            lr_x0 = max(0, min(lr_x0, lr_w - lr_pw))
            x_lr = lr_seq[:, lr_y0 : lr_y0 + lr_ph, lr_x0 : lr_x0 + lr_pw, :]
            st_patch = static_norm[y0 : y0 + patch_h, x0 : x0 + patch_w, :]
            x_st = np.broadcast_to(st_patch[np.newaxis, ...], (lr_seq.shape[0], *st_patch.shape))
            x_lr_list.append(x_lr)
            x_st_list.append(x_st)

        y_pred = model((np.stack(x_lr_list, axis=0), np.stack(x_st_list, axis=0)), training=False).numpy()
        for (y0, x0), pred in zip(batch, y_pred):
            if pred.ndim == 4:
                frame = pred[-1, :, :, 0] if output_mode == "last" else pred.mean(axis=0)[:, :, 0]
            elif pred.ndim == 3:
                frame = pred[:, :, 0]
            else:
                raise SystemExit(f"unexpected model output shape for patch: {pred.shape}")
            out_sum[y0 : y0 + patch_h, x0 : x0 + patch_w] += frame.astype(np.float32) * weight
            weight_sum[y0 : y0 + patch_h, x0 : x0 + patch_w] += weight

    weight_sum[weight_sum == 0.0] = 1.0
    return out_sum / weight_sum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-zarr", default="data/forecast/openmeteo_ecmwf_lr.zarr")
    parser.add_argument("--static-cache", default="data/processed/static_processed.npy")
    parser.add_argument("--stats", default="data/processed/stats_config.npz")
    parser.add_argument(
        "--model-type",
        default="mamba",
        choices=["mamba", "unet", "convlstm", "transformer", "baseline_nearest", "baseline_bilinear"],
    )
    parser.add_argument("--model-path", default="experiments/models/Tiles_MAMBA_S42_best.h5")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-mode", choices=["last", "mean"], default="last")
    parser.add_argument("--max-predictions", type=int, default=0)
    parser.add_argument("--out-dir", default="experiments/predictions")
    parser.add_argument("--manifest", default="experiments/predictions/manifest.csv")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.seq_len and args.seq_len > 0:
        Config.SEQ_LEN = int(args.seq_len)
    seq_len = int(getattr(Config, "SEQ_LEN", 6))
    temporal_stride = max(1, int(args.temporal_stride))
    batch_size = max(1, int(args.batch_size))

    forecast_path = Path(args.forecast_zarr)
    if not forecast_path.exists():
        raise SystemExit(f"forecast zarr not found: {forecast_path}")

    ds = xr.open_zarr(forecast_path, consolidated=True)
    da_lr = ds["lr_input"]
    lr_time, lr_lat, lr_lon, lr_var = _find_dims(da_lr)
    lr_h = int(da_lr.sizes[lr_lat])
    lr_w = int(da_lr.sizes[lr_lon])
    lr_c = int(da_lr.sizes[lr_var]) if lr_var else 1

    static_raw = np.load(args.static_cache)
    if static_raw.ndim == 2:
        full_hr_shape = static_raw.shape
    else:
        full_hr_shape = static_raw.shape[:2]
    hr_h, hr_w = int(full_hr_shape[0]), int(full_hr_shape[1])

    patch_h = min(int(args.patch_size), hr_h)
    patch_w = min(int(args.patch_size), hr_w)
    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
    lr_ph = max(1, int(round(patch_h * ratio_y)))
    lr_pw = max(1, int(round(patch_w * ratio_x)))

    Config.HR_SHAPE = (patch_h, patch_w)
    Config.LR_SHAPE = (lr_ph, lr_pw)
    Config.CHANNELS = lr_c
    Config.STATIC_CHANNELS = int(static_raw.shape[-1]) if static_raw.ndim == 3 else 1

    static_norm = _load_static(Path(args.static_cache), full_hr_shape)
    mean_hr, std_hr = _load_stats(Path(args.stats))

    is_baseline = args.model_type.startswith("baseline_")
    model = _build_model(args.model_type)
    if not is_baseline:
        if not args.model_path or not Path(args.model_path).exists():
            raise SystemExit(f"model checkpoint not found: {args.model_path}")
        model.load_weights(args.model_path)

    if lr_var:
        lr_all = da_lr.transpose(lr_time, lr_lat, lr_lon, lr_var).values.astype(np.float32)
    else:
        lr_all = da_lr.transpose(lr_time, lr_lat, lr_lon).values.astype(np.float32)
        lr_all = lr_all[..., np.newaxis]

    try:
        if _order(ds[lr_lon].values) == "desc":
            lr_all = lr_all[:, :, ::-1, :]
    except Exception:
        pass

    times = pd.to_datetime(ds[lr_time].values)
    indices = _target_indices(len(times), seq_len, temporal_stride, int(args.max_predictions))
    if not indices:
        raise SystemExit(f"not enough forecast timesteps for seq_len={seq_len}, temporal_stride={temporal_stride}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    coords = _coords_for_tiles(hr_h, hr_w, patch_h, patch_w, int(args.stride))

    rows = []
    for target_idx in indices:
        start = target_idx - (seq_len - 1) * temporal_stride
        window_idx = list(range(start, target_idx + 1, temporal_stride))
        if len(window_idx) != seq_len:
            continue
        ts = pd.Timestamp(times[target_idx])
        out_path = out_dir / f"pred_{_timestamp_name(ts)}.npy"
        if args.skip_existing and out_path.exists():
            rows.append({"time": ts.isoformat(), "path": str(out_path), "status": "skipped"})
            continue

        pred_norm = _run_one_timestamp(
            model=model,
            lr_seq=lr_all[window_idx],
            static_norm=static_norm,
            coords=coords,
            full_hr_shape=(hr_h, hr_w),
            patch_shape=(patch_h, patch_w),
            lr_patch_shape=(lr_ph, lr_pw),
            batch_size=batch_size,
            output_mode=args.output_mode,
        )
        pred_c = pred_norm * std_hr + mean_hr
        np.save(out_path, pred_c.astype(np.float32))
        rows.append({"time": ts.isoformat(), "path": str(out_path), "status": "written"})
        print(f"wrote {out_path}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "path", "status"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "ok",
        "forecast_zarr": str(forecast_path),
        "model_type": args.model_type,
        "model_path": "" if is_baseline else args.model_path,
        "seq_len": seq_len,
        "temporal_stride": temporal_stride,
        "patch_size": [patch_h, patch_w],
        "lr_patch_size": [lr_ph, lr_pw],
        "full_hr_shape": [hr_h, hr_w],
        "prediction_count": len(rows),
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
    }
    with (out_dir / "operational_inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
