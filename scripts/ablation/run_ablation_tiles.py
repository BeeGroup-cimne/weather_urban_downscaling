#!/usr/bin/env python3
"""
Tile-based ablation study (independent from full-frame pipeline).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import subprocess
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import tensorflow as tf

from config.runtime import Config
from src.data_loader_tiles import TileDataPipeline
from src.models_legacy import ModelZoo
from src.losses import tf_hybrid_loss
from src.utils import run_experiment, plot_comparative_history, notify_completion


EXPERIMENTS_TO_RUN = {
    "unet": ModelZoo.build_unet,
    "lstm": ModelZoo.build_hybrid_unet_lstm,
    "transformer": ModelZoo.build_transformer,
    "mamba": ModelZoo.build_hybrid_unet_mamba,
}


def run_post_inference(model_type: str, model_path: str, patch_size: int, experiment_name: str, preview_time: str | None):
    if not os.path.exists(model_path):
        print(f"⚠️ Post-inference skipped. Model not found: {model_path}")
        return

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "inference", "run_inference_tiles_fullframe.py"),
        "--model-type", model_type,
        "--model-path", model_path,
        "--patch-size", str(patch_size),
        "--stride", str(max(1, patch_size // 2)),
        "--batch-size", "8",
        "--lr-resample", "nearest",
        "--use-last",
        "--experiment-name", experiment_name,
        "--out", os.path.join("experiments", "figures", "tiles_post_train.png"),
    ]
    if preview_time:
        cmd.extend(["--time", preview_time])

    print("🖼️ Generating default post-training full-frame preview...")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"⚠️ Post-training preview failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mamba"],
        choices=["unet", "lstm", "transformer", "mamba"],
        help="Models to run (default: mamba).",
    )
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patches-per-epoch", type=int, default=4000)
    parser.add_argument("--val-patches", type=int, default=400)
    parser.add_argument("--sampler", default="uhi_proxy", choices=["uhi_proxy", "static_weighted", "uniform", "error_weighted"])
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--temporal-sampler", default="p95", choices=["uniform", "weighted", "weighted_station", "p95"])
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--temporal-season-balance", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--tile-weight-alpha", type=float, default=0.7)
    parser.add_argument("--tile-weight-gamma", type=float, default=1.0)
    parser.add_argument("--temporal-weight-gamma", type=float, default=1.0)
    parser.add_argument("--split-mode", default="inherit", choices=["inherit", "time", "fraction"])
    parser.add_argument("--split-fraction", type=float, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--post-inference", default=True, action=argparse.BooleanOptionalAction, help="Generate full-frame preview image after each model")
    parser.add_argument("--preview-time", type=str, default="2017-08-15T15:00:00", help="Timestamp for post-training preview")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for reproducibility")
    parser.add_argument("--experiment-suffix", type=str, default="", help="Suffix appended to experiment names (e.g. S42)")
    args = parser.parse_args()

    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    if args.seed is not None:
        Config.SEED = int(args.seed)
        random.seed(Config.SEED)
        try:
            import numpy as np
            np.random.seed(Config.SEED)
        except Exception:
            pass
        tf.random.set_seed(Config.SEED)

    # Tile config overrides
    Config.SEQ_LEN = args.seq_len
    Config.BATCH_SIZE = args.batch_size
    Config.PREFETCH_BUFFER_SIZE = args.prefetch
    Config.SHUFFLE_BUFFER_SIZE = args.shuffle
    Config.PATCH_SIZE = (args.patch_size, args.patch_size)
    Config.PATCHES_PER_EPOCH = args.patches_per_epoch
    Config.VAL_PATCHES_PER_EPOCH = args.val_patches
    Config.TILE_SAMPLER = args.sampler
    Config.TILE_WEIGHT_ALPHA = args.tile_weight_alpha
    Config.TILE_WEIGHT_GAMMA = args.tile_weight_gamma
    Config.TEMPORAL_STRIDE = args.temporal_stride
    Config.TEMPORAL_SAMPLER = args.temporal_sampler
    Config.TEMPORAL_WEIGHT_GAMMA = args.temporal_weight_gamma
    Config.EPOCHS = args.epochs
    Config.TEMPORAL_SEASON_BALANCE = args.temporal_season_balance
    Config.EARLY_STOPPING_PATIENCE = int(args.early_stopping_patience)
    Config.EARLY_STOPPING_START_EPOCH = int(args.early_stopping_start_epoch)
    Config.EARLY_STOPPING_MIN_DELTA = float(args.early_stopping_min_delta)
    Config.LR_PATIENCE = int(args.lr_patience)
    Config.LR_FACTOR = float(args.lr_factor)
    Config.LR_MIN = float(args.lr_min)
    if args.split_mode != "inherit":
        Config.SPLIT_MODE = args.split_mode
    if args.split_fraction is not None:
        Config.SPLIT_FRACTION = float(args.split_fraction)
    if args.train_start is not None:
        Config.TRAIN_START = args.train_start
    if args.train_end is not None:
        Config.TRAIN_END = args.train_end
    if args.val_start is not None:
        Config.VAL_START = args.val_start
    if args.val_end is not None:
        Config.VAL_END = args.val_end
    if args.test_start is not None:
        Config.TEST_START = args.test_start
    if args.test_end is not None:
        Config.TEST_END = args.test_end

    steps_per_epoch = max(1, args.patches_per_epoch // Config.BATCH_SIZE)
    Config.MAX_STEPS_PER_EPOCH = steps_per_epoch

    print("\n🧩 Tile-based ablation study")
    print(f"   Patch: {args.patch_size}x{args.patch_size}")
    print(f"   Sampler: {args.sampler}")
    print(f"   Steps/epoch: {steps_per_epoch}")
    print(f"   Models: {', '.join(args.models)}")
    print(f"   Seed: {Config.SEED}")
    print(f"   SEQ_LEN: {Config.SEQ_LEN}")
    print(f"   BATCH_SIZE: {Config.BATCH_SIZE}")
    print(f"   Temporal sampler: {Config.TEMPORAL_SAMPLER}")
    print(f"   Season balance: {Config.TEMPORAL_SEASON_BALANCE}")
    print(f"   EARLY_STOPPING_PATIENCE: {Config.EARLY_STOPPING_PATIENCE}")
    print(f"   EARLY_STOPPING_START_EPOCH: {Config.EARLY_STOPPING_START_EPOCH}")
    print(f"   LR_PATIENCE: {Config.LR_PATIENCE}")
    print(f"   Split mode: {Config.SPLIT_MODE}")
    if Config.SPLIT_MODE == "time":
        print(f"   Train: {Config.TRAIN_START} -> {Config.TRAIN_END}")
        print(f"   Val:   {Config.VAL_START} -> {Config.VAL_END}")
    else:
        print(f"   Split fraction: {Config.SPLIT_FRACTION}")

    pipeline = TileDataPipeline(Config)
    train_ds, val_ds = pipeline.get_tf_datasets()
    # Repeat to avoid dataset exhaustion across epochs
    train_ds = train_ds.repeat()
    val_ds = val_ds.repeat()
    steps_per_epoch = max(1, Config.PATCHES_PER_EPOCH // max(1, Config.BATCH_SIZE))
    validation_steps = max(1, Config.VAL_PATCHES_PER_EPOCH // max(1, Config.BATCH_SIZE))

    all_histories = {}
    suffix = args.experiment_suffix.strip()

    for name in args.models:
        builder = EXPERIMENTS_TO_RUN[name]
        tf.keras.backend.clear_session()
        gc.collect()

        exp_name = f"Tiles_{name.upper()}"
        if suffix:
            exp_name = f"{exp_name}_{suffix}"
        print(f"\n{'='*60}")
        print(f"🏗️  MODELO: {name.upper()} (Tiles)")
        print(f"{'='*60}")

        if name == "mamba":
            model = builder(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
        else:
            model = builder()

        loss_fn = tf_hybrid_loss(alpha=0.8, max_val=5.0)
        opt = ModelZoo.get_optimizer(Config.LEARNING_RATE)
        model.compile(optimizer=opt, loss=loss_fn, metrics=["mae", "mse"])

        history = run_experiment(
            model,
            train_ds,
            val_ds,
            exp_name,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
        all_histories[name] = history

        if args.post_inference:
            model_path = os.path.join(Config.EXPERIMENTS_DIR, "models", f"{exp_name}_best.h5")
            run_post_inference(
                model_type=name if name != "lstm" else "convlstm",
                model_path=model_path,
                patch_size=args.patch_size,
                experiment_name=exp_name,
                preview_time=args.preview_time,
            )

    plot_comparative_history(all_histories, save_dir=os.path.join(Config.EXPERIMENTS_DIR, "figures"))
    notify_completion("Ablation tiles completada.")


if __name__ == "__main__":
    main()
