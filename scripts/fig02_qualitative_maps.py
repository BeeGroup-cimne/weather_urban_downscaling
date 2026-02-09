#!/usr/bin/env python3
"""
Generate Fig02: qualitative maps (LR upsampled vs static vs prediction vs error).
Uses a real batch from the pipeline and a saved model checkpoint.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import xarray as xr

from scripts.fig_utils import ensure_dir, default_fig_dir, safe_import_matplotlib, timestamp
from config.runtime import Config
from src.models_legacy import ModelZoo
from src.data_loader import BigDataPipeline


def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    return ModelZoo.build_unet()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--model-path", default="")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip loading model; use LR upsample as baseline prediction.")
    parser.add_argument("--lr-channel", type=int, default=0,
                        help="Channel index for LR visualization.")
    parser.add_argument("--static-var", default="sky",
                        help="Static variable name to visualize (substring match).")
    parser.add_argument("--static-channel", type=int, default=-1,
                        help="Static channel index (overrides --static-var if >=0).")
    parser.add_argument("--lr-native", action="store_true",
                        help="Show LR at native resolution (no upsample for display).")
    parser.add_argument("--rotate", type=int, default=0,
                        help="Rotate all panels by degrees (0, 90, 180, 270, -90, -180, -270).")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not safe_import_matplotlib():
        return 1
    import matplotlib.pyplot as plt

    out_dir = default_fig_dir()
    ensure_dir(out_dir)
    out_path = args.out or os.path.join(out_dir, "fig02_qualitative_maps.png")

    # Select default model path if not provided (unless skip-model)
    if not args.skip_model:
        if not args.model_path:
            candidates = [
                "experiments/models/Ablation_MAMBA_Legacy_best.h5",
                "experiments/models/UNet_gpu_optimized.h5",
                "experiments/models/ConvLSTM_gpu_optimized.h5",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    args.model_path = cand
                    break
        if not args.model_path or not os.path.exists(args.model_path):
            print("Model checkpoint not found. Provide --model-path or use --skip-model.")
            return 2

    # Resolve static channel name -> index
    static_names = []
    try:
        ds_static = xr.open_zarr(Config.PATH_STATIC)
        static_names = [v for v in ds_static.data_vars if v != "index"]
    except Exception as e:
        print(f"⚠️ No se pudo leer PATH_STATIC para nombres de canales: {e}")

    st_chan = args.static_channel if args.static_channel >= 0 else 0
    if args.static_channel < 0 and static_names:
        target = args.static_var.lower()
        for i, name in enumerate(static_names):
            if target in name.lower():
                st_chan = i
                break
        else:
            for i, name in enumerate(static_names):
                if "svf" in name.lower():
                    st_chan = i
                    break

    if static_names:
        print(f"ℹ️ Static channel selected: {st_chan} ({static_names[st_chan]})")
    else:
        print(f"ℹ️ Static channel selected: {st_chan} (no names available)")

    # Load one batch from validation set (updates Config.LR_SHAPE/HR_SHAPE)
    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    train_ds, val_ds = pipeline.get_tf_datasets()
    (x_lr, x_st), y_hr = next(iter(val_ds.take(1)))

    # Predict (or baseline)
    y_pred = None
    if not args.skip_model:
        # Build and load model after Config shapes are updated
        model = build_model(args.model_type)
        model.load_weights(args.model_path)
        y_pred = model((x_lr, x_st), training=False)

    # Select last timestep for visualization
    lr_last = x_lr[0, -1, ...]
    st_last = x_st[0, -1, ...]
    hr_last = y_hr[0, -1, ..., 0]
    if hasattr(hr_last, "numpy"):
        hr_last = hr_last.numpy()
    pred_last = None if args.skip_model else y_pred[0, -1, ..., 0]

    # Upsample LR for alignment/visualization
    lr_ch = args.lr_channel
    if lr_last.ndim == 3:
        lr_last = lr_last[..., lr_ch]
    if hasattr(lr_last, "numpy"):
        lr_last = lr_last.numpy()
    lr_up = tf.image.resize(lr_last[..., None], Config.HR_SHAPE, method="nearest").numpy()[..., 0]
    st_vis = st_last[..., st_chan]  # selected static channel
    # Baseline prediction if model is skipped
    if args.skip_model:
        pred_last = lr_up
    err = np.abs(pred_last - hr_last)

    def _align_lr_for_plot(lr_up_arr, hr_ref):
        def _corr(a, b):
            a = a.flatten()
            b = b.flatten()
            if np.std(a) == 0 or np.std(b) == 0:
                return -np.inf
            return np.corrcoef(a, b)[0, 1]

        candidates = {
            "as_is": lr_up_arr,
            "rot90": np.rot90(lr_up_arr, 1),
            "rot180": np.rot90(lr_up_arr, 2),
            "rot270": np.rot90(lr_up_arr, 3),
            "flipud": np.flipud(lr_up_arr),
            "fliplr": np.fliplr(lr_up_arr),
            "transpose": lr_up_arr.T,
            "transpose_flipud": np.flipud(lr_up_arr.T),
            "transpose_fliplr": np.fliplr(lr_up_arr.T),
        }

        best_name, best_arr, best_score = "as_is", lr_up_arr, -np.inf
        for name, arr in candidates.items():
            score = _corr(arr, hr_ref)
            if score > best_score:
                best_score = score
                best_name, best_arr = name, arr
        return best_arr, best_name

    lr_disp, lr_tag = _align_lr_for_plot(lr_up, hr_last)
    print(f"ℹ️ LR display alignment: {lr_tag}")

    def _apply_transform(arr, name):
        if name == "as_is":
            return arr
        if name == "rot90":
            return np.rot90(arr, 1)
        if name == "rot180":
            return np.rot90(arr, 2)
        if name == "rot270":
            return np.rot90(arr, 3)
        if name == "flipud":
            return np.flipud(arr)
        if name == "fliplr":
            return np.fliplr(arr)
        if name == "transpose":
            return arr.T
        if name == "transpose_flipud":
            return np.flipud(arr.T)
        if name == "transpose_fliplr":
            return np.fliplr(arr.T)
        return arr

    def _stretch(data):
        p2, p98 = np.nanpercentile(data, [2, 98])
        if p2 == p98:
            p2, p98 = np.nanmin(data), np.nanmax(data)
        return p2, p98

    def _apply_rotate(arr):
        deg = args.rotate % 360
        if deg == 0:
            return arr
        k = deg // 90
        return np.rot90(arr, k)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    pred_title = "Prediction" if not args.skip_model else "Baseline (LR upsample)"
    lr_show = _apply_transform(lr_last, lr_tag) if args.lr_native else lr_disp
    panels = [
        (lr_show, f"LR (ch{lr_ch})"),
        (st_vis, f"Static (ch{st_chan})"),
        (pred_last, pred_title),
        (hr_last, "HR target"),
        (err, "Abs error"),
    ]
    for ax, (data, title) in zip(axes, panels):
        data = _apply_rotate(data)
        vmin, vmax = _stretch(data)
        im = ax.imshow(data, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    mode_tag = "baseline" if args.skip_model else args.model_type
    fig.suptitle(f"Qualitative Example ({mode_tag})", fontsize=10)
    fig.text(0.99, 0.01, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
