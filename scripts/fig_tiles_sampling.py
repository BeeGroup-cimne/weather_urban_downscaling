#!/usr/bin/env python3
"""
Generate a paper-ready figure for tile-based training.
Panels: weight map (optional), LR upsample patch, static patch, HR target, prediction (optional).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import tensorflow as tf
import xarray as xr

from config.runtime import Config
from src.models_legacy import ModelZoo
from src.data_loader_tiles import TileDataPipeline


def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    return ModelZoo.build_unet()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--sampler", default="static_weighted", choices=["static_weighted", "uniform", "error_weighted"])
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--model-path", default="")
    parser.add_argument("--static-channel", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    Config.PATCH_SIZE = (args.patch_size, args.patch_size)
    Config.TILE_SAMPLER = args.sampler

    pipeline = TileDataPipeline(Config)
    train_ds, _ = pipeline.get_tf_datasets()
    (x_lr, x_st), y_hr = next(iter(train_ds.take(1)))

    # Optional prediction
    y_pred = None
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"⚠️ Model not found: {args.model_path}. Skipping prediction.")
        else:
            model = build_model(args.model_type)
            model.load_weights(args.model_path)
            y_pred = model((x_lr, x_st), training=False)

    lr_last = x_lr[0, -1, ...]
    st_last = x_st[0, -1, ..., args.static_channel]
    hr_last = y_hr[0, -1, ..., 0]
    if hasattr(hr_last, "numpy"):
        hr_last = hr_last.numpy()

    lr_up = tf.image.resize(lr_last[..., 0:1], Config.HR_SHAPE, method="bilinear").numpy()[..., 0]

    panels = []
    titles = []

    # Optional weight map
    weight_map = None
    if args.sampler == "static_weighted":
        static = pipeline.static_norm
        weight_map = np.mean(np.abs(static), axis=-1)
    elif args.sampler == "error_weighted":
        err_path = getattr(Config, "TILE_ERROR_MAP_PATH", "")
        if err_path and os.path.exists(err_path):
            weight_map = np.load(err_path)

    if weight_map is not None:
        panels.append(weight_map)
        titles.append(f"Weight map ({args.sampler})")

    panels.extend([lr_up, st_last, hr_last])
    titles.extend(["LR upsample", f"Static ch {args.static_channel}", "HR target"])

    if y_pred is not None:
        pred_last = y_pred[0, -1, ..., 0]
        panels.append(pred_last)
        titles.append("Prediction")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, data, title in zip(axes, panels, titles):
        if hasattr(data, "numpy"):
            data = data.numpy()
        vmin, vmax = np.nanpercentile(data, [2, 98])
        if vmin == vmax:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_dir = os.path.join(Config.EXPERIMENTS_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "tiles_sampling_example.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    main()
