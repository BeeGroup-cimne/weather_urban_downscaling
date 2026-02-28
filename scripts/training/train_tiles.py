#!/usr/bin/env python3
"""
Tile-based training (independent from the full-frame pipeline).
Default sampler: static_weighted.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import tensorflow as tf

from config.runtime import Config
from src.data_loader_tiles import TileDataPipeline
from src.models_legacy import ModelZoo
from src.losses import tf_hybrid_loss


def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    if model_type == "transformer":
        return ModelZoo.build_transformer()
    return ModelZoo.build_unet()


def run_post_inference(model_type: str, model_path: str, patch_size: int, temporal_stride: int, experiment_name: str, preview_time: str | None):
    if not os.path.exists(model_path):
        print(f"⚠️ Post-inference skipped. Model not found: {model_path}")
        return

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "run_inference_tiles_fullframe.py"),
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
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm", "transformer"])
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patches-per-epoch", type=int, default=4000)
    parser.add_argument("--val-patches", type=int, default=400)
    parser.add_argument("--sampler", default="uhi_proxy", choices=["uhi_proxy", "static_weighted", "uniform", "error_weighted"])
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--temporal-sampler", default="p95", choices=["uniform", "weighted", "weighted_station", "p95"])
    parser.add_argument("--seq-len", type=int, default=None, help="Override Config.SEQ_LEN for tile runs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override Config.BATCH_SIZE for tile runs")
    parser.add_argument("--prefetch", type=int, default=None, help="Override Config.PREFETCH_BUFFER_SIZE for tile runs")
    parser.add_argument("--shuffle", type=int, default=None, help="Override Config.SHUFFLE_BUFFER_SIZE for tile runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--temporal-season-balance", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--tile-weight-alpha", type=float, default=0.7, help="Mix weighted vs uniform patch sampling")
    parser.add_argument("--tile-weight-gamma", type=float, default=1.0, help="Emphasize extremes in spatial weight map")
    parser.add_argument("--temporal-weight-gamma", type=float, default=1.0, help="Emphasize extremes in temporal weights")
    parser.add_argument("--split-mode", default="inherit", choices=["inherit", "time", "fraction"])
    parser.add_argument("--split-fraction", type=float, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--post-inference", default=True, action=argparse.BooleanOptionalAction, help="Generate full-frame preview image after training")
    parser.add_argument("--preview-time", type=str, default="2017-08-15T15:00:00", help="Timestamp for post-training preview")
    args = parser.parse_args()

    # Reduce GPU memory fragmentation on NVIDIA (optional)
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    # Tile config overrides (runtime only)
    if args.seq_len is not None:
        Config.SEQ_LEN = int(args.seq_len)
    if args.batch_size is not None:
        Config.BATCH_SIZE = int(args.batch_size)
    if args.prefetch is not None:
        Config.PREFETCH_BUFFER_SIZE = int(args.prefetch)
    if args.shuffle is not None:
        Config.SHUFFLE_BUFFER_SIZE = int(args.shuffle)

    Config.PATCH_SIZE = (args.patch_size, args.patch_size)
    Config.PATCHES_PER_EPOCH = args.patches_per_epoch
    Config.VAL_PATCHES_PER_EPOCH = args.val_patches
    Config.TILE_SAMPLER = args.sampler
    Config.TILE_WEIGHT_ALPHA = float(args.tile_weight_alpha)
    Config.TILE_WEIGHT_GAMMA = float(args.tile_weight_gamma)
    Config.TEMPORAL_STRIDE = args.temporal_stride
    Config.TEMPORAL_SAMPLER = args.temporal_sampler
    Config.TEMPORAL_WEIGHT_GAMMA = float(args.temporal_weight_gamma)
    Config.LEARNING_RATE = args.learning_rate
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

    print("🧩 Tile-based training")
    print(f"   Model: {args.model_type}")
    print(f"   Patch: {args.patch_size}x{args.patch_size}")
    print(f"   Sampler: {args.sampler}")
    print(f"   Patches/epoch: {args.patches_per_epoch}")
    print(f"   SEQ_LEN: {Config.SEQ_LEN}")
    print(f"   BATCH_SIZE: {Config.BATCH_SIZE}")
    print(f"   Temporal sampler: {Config.TEMPORAL_SAMPLER}")
    print(f"   TILE_WEIGHT_ALPHA: {Config.TILE_WEIGHT_ALPHA}")
    print(f"   TILE_WEIGHT_GAMMA: {Config.TILE_WEIGHT_GAMMA}")
    print(f"   TEMPORAL_WEIGHT_GAMMA: {Config.TEMPORAL_WEIGHT_GAMMA}")
    print(f"   TEMPORAL_SEASON_BALANCE: {Config.TEMPORAL_SEASON_BALANCE}")
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

    model = build_model(args.model_type)

    loss_fn = tf_hybrid_loss(alpha=0.8, max_val=5.0)
    opt = ModelZoo.get_optimizer(Config.LEARNING_RATE)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["mae", "mse"])

    # Callbacks
    os.makedirs(os.path.join(Config.EXPERIMENTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.EXPERIMENTS_DIR, "logs"), exist_ok=True)

    experiment_name = f"Tiles_{args.model_type.upper()}"
    model_path = os.path.join(Config.EXPERIMENTS_DIR, "models", f"{experiment_name}_best.h5")
    log_path = os.path.join(Config.EXPERIMENTS_DIR, "logs", f"{experiment_name}_log.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            monitor="val_loss",
            start_from_epoch=Config.EARLY_STOPPING_START_EPOCH,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=Config.LR_FACTOR,
            patience=Config.LR_PATIENCE,
            min_lr=Config.LR_MIN,
        ),
        tf.keras.callbacks.CSVLogger(log_path),
    ]

    steps_per_epoch = max(1, args.patches_per_epoch // Config.BATCH_SIZE)
    val_steps = max(1, args.val_patches // Config.BATCH_SIZE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"✅ Tile training complete. Model: {model_path}")
    if args.post_inference:
        run_post_inference(
            model_type=args.model_type,
            model_path=model_path,
            patch_size=args.patch_size,
            temporal_stride=args.temporal_stride,
            experiment_name=experiment_name,
            preview_time=args.preview_time,
        )


if __name__ == "__main__":
    main()
