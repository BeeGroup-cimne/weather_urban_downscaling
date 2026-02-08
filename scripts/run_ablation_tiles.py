#!/usr/bin/env python3
"""
Tile-based ablation study (independent from full-frame pipeline).
"""

import argparse
import gc
import os
import sys

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
    "mamba": ModelZoo.build_hybrid_unet_mamba,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patches-per-epoch", type=int, default=2000)
    parser.add_argument("--val-patches", type=int, default=200)
    parser.add_argument("--sampler", default="static_weighted", choices=["static_weighted", "uniform", "error_weighted"])
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--temporal-sampler", default="uniform", choices=["uniform", "weighted", "weighted_station"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--temporal-season-balance", action="store_true")
    args = parser.parse_args()

    # Tile config overrides
    Config.PATCH_SIZE = (args.patch_size, args.patch_size)
    Config.PATCHES_PER_EPOCH = args.patches_per_epoch
    Config.VAL_PATCHES_PER_EPOCH = args.val_patches
    Config.TILE_SAMPLER = args.sampler
    Config.TEMPORAL_STRIDE = args.temporal_stride
    Config.TEMPORAL_SAMPLER = args.temporal_sampler
    Config.EPOCHS = args.epochs
    Config.TEMPORAL_SEASON_BALANCE = args.temporal_season_balance

    steps_per_epoch = max(1, args.patches_per_epoch // Config.BATCH_SIZE)
    Config.MAX_STEPS_PER_EPOCH = steps_per_epoch

    print("\n🧩 Tile-based ablation study")
    print(f"   Patch: {args.patch_size}x{args.patch_size}")
    print(f"   Sampler: {args.sampler}")
    print(f"   Steps/epoch: {steps_per_epoch}")

    pipeline = TileDataPipeline(Config)
    train_ds, val_ds = pipeline.get_tf_datasets()
    # Repeat to avoid dataset exhaustion across epochs
    train_ds = train_ds.repeat()
    val_ds = val_ds.repeat()
    steps_per_epoch = max(1, Config.PATCHES_PER_EPOCH // max(1, Config.BATCH_SIZE))
    validation_steps = max(1, Config.VAL_PATCHES_PER_EPOCH // max(1, Config.BATCH_SIZE))

    all_histories = {}

    for name, builder in EXPERIMENTS_TO_RUN.items():
        tf.keras.backend.clear_session()
        gc.collect()

        exp_name = f"Tiles_{name.upper()}"
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

    plot_comparative_history(all_histories, save_dir=os.path.join(Config.EXPERIMENTS_DIR, "figures"))
    notify_completion("Ablation tiles completada.")


if __name__ == "__main__":
    main()
