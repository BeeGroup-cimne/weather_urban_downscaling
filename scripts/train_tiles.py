#!/usr/bin/env python3
"""
Tile-based training (independent from the full-frame pipeline).
Default sampler: static_weighted.
"""

import argparse
import os
import sys

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
    return ModelZoo.build_unet()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patches-per-epoch", type=int, default=2000)
    parser.add_argument("--val-patches", type=int, default=200)
    parser.add_argument("--sampler", default="static_weighted", choices=["static_weighted", "uniform"])
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--temporal-sampler", default="uniform", choices=["uniform", "weighted"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    # Tile config overrides (runtime only)
    Config.PATCH_SIZE = (args.patch_size, args.patch_size)
    Config.PATCHES_PER_EPOCH = args.patches_per_epoch
    Config.VAL_PATCHES_PER_EPOCH = args.val_patches
    Config.TILE_SAMPLER = args.sampler
    Config.TEMPORAL_STRIDE = args.temporal_stride
    Config.TEMPORAL_SAMPLER = args.temporal_sampler
    Config.LEARNING_RATE = args.learning_rate
    Config.EPOCHS = args.epochs

    print("🧩 Tile-based training")
    print(f"   Model: {args.model_type}")
    print(f"   Patch: {args.patch_size}x{args.patch_size}")
    print(f"   Sampler: {args.sampler}")
    print(f"   Patches/epoch: {args.patches_per_epoch}")

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
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
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


if __name__ == "__main__":
    main()
