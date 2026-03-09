#!/usr/bin/env python3
"""
Deterministic post-training robustness evaluation (Case Study 2B).

Runs Monte Carlo perturbations on LR input channel(s) and exports:
  - robustness_results.csv
  - report_robustness.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# Keep compatibility with legacy tf.keras model definitions on TF/Keras 2.16+.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.runtime import Config
from src.models_legacy import ModelZoo


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _order(vals) -> str:
    diffs = np.diff(vals)
    if np.all(diffs > 0):
        return "asc"
    if np.all(diffs < 0):
        return "desc"
    return "unknown"


def _find_time_index(times, time_value: str) -> int:
    times_pd = pd.to_datetime(times)
    target = pd.to_datetime(time_value)
    idx = np.searchsorted(times_pd.values, target.to_datetime64(), side="left")
    idx = min(max(idx, 0), len(times_pd) - 1)
    return int(idx)


def _find_dims(da_lr, da_hr):
    lr_time = next((d for d in da_lr.dims if d in ["time", "valid_time", "t"]), "time")
    lr_lat = next((d for d in da_lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]), "y")
    lr_lon = next((d for d in da_lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]), "x")
    lr_var = next((d for d in da_lr.dims if d in ["variable", "channel", "var"]), None)
    hr_time = next((d for d in da_hr.dims if d in ["time", "valid_time", "t"]), "time")
    hr_lat = next((d for d in da_hr.dims if d in ["latitude", "lat", "y"]), "y")
    hr_lon = next((d for d in da_hr.dims if d in ["longitude", "lon", "x"]), "x")
    return lr_time, lr_lat, lr_lon, lr_var, hr_time, hr_lat, hr_lon


def _make_weight_window(h, w):
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    w2d = np.outer(wy, wx)
    return np.maximum(w2d, 1e-6).astype(np.float32)


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


def _resolve_lr_channel(da_lr, lr_var_dim: str | None, spec: str) -> Tuple[int, str]:
    spec = str(spec or "").strip()
    if spec.startswith("ch"):
        digits = "".join([c for c in spec[2:] if c.isdigit()])
        if digits:
            idx = int(digits)
            return idx, spec

    if lr_var_dim and lr_var_dim in da_lr.coords and spec:
        try:
            names = [str(v) for v in da_lr.coords[lr_var_dim].values.tolist()]
            if spec in names:
                return names.index(spec), spec
        except Exception:
            pass

    if lr_var_dim and lr_var_dim in da_lr.coords:
        try:
            names = [str(v) for v in da_lr.coords[lr_var_dim].values.tolist()]
            if "t2m" in names:
                return names.index("t2m"), "t2m"
        except Exception:
            pass

    return 3, "ch3_t2m"


def _ssim_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    max_val = float(np.nanmax(y_true) - np.nanmin(y_true))
    if not np.isfinite(max_val) or max_val <= 0:
        return float("nan")
    try:
        return float(
            tf.image.ssim(
                y_true[None, ..., None].astype(np.float32),
                y_pred[None, ..., None].astype(np.float32),
                max_val=max_val,
            ).numpy()[0]
        )
    except Exception:
        return float("nan")


@dataclass
class PreparedInputs:
    lr_full: np.ndarray
    hr_last: np.ndarray
    timestamp: str


def _run_fullframe_inference(
    model,
    lr_full: np.ndarray,
    static_norm: np.ndarray,
    hr_shape: Tuple[int, int],
    lr_shape: Tuple[int, int],
    patch: Tuple[int, int],
    stride: int,
    batch_size: int,
) -> np.ndarray:
    hr_h, hr_w = hr_shape
    lr_h, lr_w = lr_shape
    patch_h, patch_w = patch

    ratio_y = lr_h / float(hr_h)
    ratio_x = lr_w / float(hr_w)
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


def _prepare_window(
    *,
    ds,
    da_lr,
    da_hr,
    dims,
    t_idx: int,
    seq_len: int,
    temporal_stride: int,
    flip_lr_lon: bool,
) -> PreparedInputs | None:
    lr_time, lr_lat, lr_lon, lr_var, hr_time, hr_lat, hr_lon = dims
    start = t_idx - (seq_len - 1) * temporal_stride
    if start < 0:
        return None
    indices = list(range(start, t_idx + 1, temporal_stride))
    if len(indices) != seq_len:
        return None

    if lr_var:
        lr_full = da_lr.isel({lr_time: indices}).transpose(lr_time, lr_lat, lr_lon, lr_var).values
    else:
        lr_full = da_lr.isel({lr_time: indices}).transpose(lr_time, lr_lat, lr_lon).values
        if lr_full.ndim == 3:
            lr_full = lr_full[..., np.newaxis]
    if flip_lr_lon:
        lr_full = lr_full[:, :, ::-1, :]

    hr_seq = da_hr.isel({hr_time: indices}).transpose(hr_time, hr_lat, hr_lon).values
    if hr_seq.ndim == 3:
        hr_seq = hr_seq[..., np.newaxis]
    hr_last = hr_seq[-1, :, :, 0]

    timestamp = pd.to_datetime(ds[hr_time].values[t_idx]).isoformat()
    return PreparedInputs(lr_full=lr_full, hr_last=hr_last, timestamp=timestamp)


def _write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Monte Carlo robustness experiment for one model/checkpoint.")
    ap.add_argument(
        "--model-type",
        required=True,
        choices=["mamba", "unet", "convlstm", "transformer", "baseline_nearest", "baseline_bilinear"],
    )
    ap.add_argument("--model-path", default="", help="Checkpoint path (.h5). Not required for baselines.")
    ap.add_argument("--seq-len", type=int, default=0)
    ap.add_argument("--patch-size", type=int, default=96)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temporal-stride", type=int, default=1)
    ap.add_argument("--epsilon", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--time", nargs="+", default=[])
    ap.add_argument("--perturbed-var", default="ch3_t2m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--experiment-name", default="cs2_robustness")
    args = ap.parse_args()

    _ensure_dir(args.outdir)
    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    if int(args.seq_len) > 0:
        Config.SEQ_LEN = int(args.seq_len)

    ds = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    da_lr = ds["lr_input"]
    da_hr = ds["hr_target"]
    dims = _find_dims(da_lr, da_hr)
    lr_time, lr_lat, lr_lon, lr_var, hr_time, hr_lat, hr_lon = dims

    hr_h = int(da_hr.sizes[hr_lat])
    hr_w = int(da_hr.sizes[hr_lon])
    lr_h = int(da_lr.sizes[lr_lat])
    lr_w = int(da_lr.sizes[lr_lon])
    lr_c = int(da_lr.sizes[lr_var]) if lr_var else 1

    patch_h = int(args.patch_size)
    patch_w = int(args.patch_size)
    Config.HR_SHAPE = (patch_h, patch_w)
    Config.LR_SHAPE = (
        max(1, int(round(patch_h * (lr_h / float(hr_h))))),
        max(1, int(round(patch_w * (lr_w / float(hr_w))))),
    )
    Config.CHANNELS = lr_c

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

    is_baseline = str(args.model_type).startswith("baseline_")
    model = _build_model(args.model_type)
    if not is_baseline:
        if not args.model_path or not os.path.exists(args.model_path):
            raise SystemExit(f"Model checkpoint not found: {args.model_path}")
        model.load_weights(args.model_path)

    all_times = ds[hr_time].values
    if args.time:
        time_idx = [_find_time_index(all_times, t) for t in args.time]
    else:
        time_idx = [max(0, len(all_times) - 1)]

    ch_idx, pert_label = _resolve_lr_channel(da_lr, lr_var, args.perturbed_var)
    ch_idx = max(0, min(ch_idx, lr_c - 1))

    seq_len = int(getattr(Config, "SEQ_LEN", 6))
    temporal_stride = max(1, int(args.temporal_stride))
    n_trials = max(1, int(args.n_trials))
    epsilons = [float(e) for e in args.epsilon]
    rng = np.random.default_rng(int(args.seed))

    rows: List[dict] = []
    model_name_csv = str(args.model_type)
    checkpoint_name = os.path.basename(args.model_path) if args.model_path else "__baseline__"

    for t_idx in time_idx:
        prepared = _prepare_window(
            ds=ds,
            da_lr=da_lr,
            da_hr=da_hr,
            dims=dims,
            t_idx=t_idx,
            seq_len=seq_len,
            temporal_stride=temporal_stride,
            flip_lr_lon=flip_lr_lon,
        )
        if prepared is None:
            print(f"⚠️ Skipping timestamp index {t_idx}: not enough history for seq_len={seq_len}.")
            continue

        pred_clean = _run_fullframe_inference(
            model=model,
            lr_full=prepared.lr_full,
            static_norm=static_norm,
            hr_shape=(hr_h, hr_w),
            lr_shape=(lr_h, lr_w),
            patch=(patch_h, patch_w),
            stride=max(1, int(args.stride)),
            batch_size=max(1, int(args.batch_size)),
        )
        diff_clean = pred_clean - prepared.hr_last
        mae_clean = float(np.nanmean(np.abs(diff_clean)))
        rmse_clean = float(np.sqrt(np.nanmean(diff_clean ** 2)))
        ssim_clean = _ssim_2d(prepared.hr_last, pred_clean)

        for eps in epsilons:
            trial_preds = []
            for _ in range(n_trials):
                lr_pert = prepared.lr_full.copy()
                noise = rng.normal(loc=0.0, scale=float(eps), size=lr_pert[..., ch_idx].shape).astype(np.float32)
                lr_pert[..., ch_idx] = lr_pert[..., ch_idx] + noise

                pred_pert = _run_fullframe_inference(
                    model=model,
                    lr_full=lr_pert,
                    static_norm=static_norm,
                    hr_shape=(hr_h, hr_w),
                    lr_shape=(lr_h, lr_w),
                    patch=(patch_h, patch_w),
                    stride=max(1, int(args.stride)),
                    batch_size=max(1, int(args.batch_size)),
                )
                trial_preds.append(pred_pert.astype(np.float32))

            pred_trials = np.stack(trial_preds, axis=0)
            range_map = np.nanmax(pred_trials, axis=0) - np.nanmin(pred_trials, axis=0)
            std_map = np.nanstd(pred_trials, axis=0)
            pred_mean = np.nanmean(pred_trials, axis=0)
            dev_map = np.abs(pred_mean - pred_clean)
            rmse_vs_clean = np.sqrt(np.nanmean((pred_mean - pred_clean) ** 2))

            rows.append(
                {
                    "model": model_name_csv,
                    "checkpoint": checkpoint_name,
                    "timestamp": prepared.timestamp,
                    "epsilon_K": float(eps),
                    "n_trials": n_trials,
                    "perturbed_var": pert_label,
                    "pred_max_range_C": float(np.nanmax(range_map)),
                    "pred_mean_std_C": float(np.nanmean(std_map)),
                    "pred_p95_range_C": float(np.nanpercentile(range_map, 95.0)),
                    "pred_mean_abs_dev_vs_clean_C": float(np.nanmean(dev_map)),
                    "pred_rmse_vs_clean_C": float(rmse_vs_clean),
                    "mae_clean_vs_hr": mae_clean,
                    "rmse_clean_vs_hr": rmse_clean,
                    "ssim_clean_vs_hr": ssim_clean,
                }
            )

    if not rows:
        raise SystemExit("No robustness rows produced.")

    fields = [
        "model",
        "checkpoint",
        "timestamp",
        "epsilon_K",
        "n_trials",
        "perturbed_var",
        "pred_max_range_C",
        "pred_mean_std_C",
        "pred_p95_range_C",
        "pred_mean_abs_dev_vs_clean_C",
        "pred_rmse_vs_clean_C",
        "mae_clean_vs_hr",
        "rmse_clean_vs_hr",
        "ssim_clean_vs_hr",
    ]
    csv_path = os.path.join(args.outdir, "robustness_results.csv")
    _write_csv(csv_path, rows, fields)

    with open(os.path.join(args.outdir, "report_robustness.md"), "w", encoding="utf-8") as f:
        f.write("# Robustness Experiment — Monte Carlo Sensitivity Analysis\n\n")
        f.write(f"- Model: `{args.model_type}`\n")
        f.write(f"- Checkpoint: `{args.model_path or '__baseline__'}`\n")
        f.write(f"- Perturbation levels (K): {epsilons}\n")
        f.write(f"- Trials per level: {n_trials}\n")
        f.write(f"- Perturbed variable: {pert_label}\n")
        f.write(f"- Timestamps: {list(args.time)}\n")
        f.write(f"- Seed: {int(args.seed)}\n\n")
        f.write(f"See `{csv_path}` for detailed results.\n")

    # Console summary
    by_eps = {}
    for r in rows:
        e = _safe_float(r.get("epsilon_K"))
        mad = _safe_float(r.get("pred_mean_abs_dev_vs_clean_C"))
        if not np.isfinite(e) or not np.isfinite(mad):
            continue
        by_eps.setdefault(e, []).append(mad)
    print(f"✅ Wrote: {csv_path}")
    for e in sorted(by_eps):
        vals = np.array(by_eps[e], dtype=float)
        print(f"  epsilon={e:.3f}: MAD_vs_clean mean={float(np.mean(vals)):.6f} std={float(np.std(vals)):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
