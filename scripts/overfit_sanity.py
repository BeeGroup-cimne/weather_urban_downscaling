#!/usr/bin/env python3
"""
Overfit sanity check: ensure the model can overfit a tiny subset.
Helps confirm data pipeline alignment and normalization are correct.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import tensorflow as tf

from config.runtime import Config
from src.data_loader import BigDataPipeline
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    print("🧪 Overfit sanity check")
    print(f"   Model: {args.model_type}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Train batches: {args.train_batches}")
    print(f"   Val batches: {args.val_batches}")

    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    train_ds, val_ds = pipeline.get_tf_datasets()

    train_small = train_ds.take(args.train_batches)
    val_small = val_ds.take(args.val_batches)

    model = build_model(args.model_type)

    loss_fn = tf_hybrid_loss(alpha=0.8, max_val=5.0)
    opt = ModelZoo.get_optimizer(args.learning_rate)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["mae", "mse"])

    history = model.fit(
        train_small,
        validation_data=val_small,
        epochs=args.epochs,
        verbose=1
    )

    start_loss = history.history["loss"][0]
    end_loss = history.history["loss"][-1]
    start_val = history.history.get("val_loss", [None])[0]
    end_val = history.history.get("val_loss", [None])[-1]

    print("\n✅ Overfit summary")
    print(f"   Train loss: {start_loss:.4f} -> {end_loss:.4f}")
    if start_val is not None and end_val is not None:
        print(f"   Val loss:   {start_val:.4f} -> {end_val:.4f}")

    if end_loss < start_loss * 0.7:
        print("   ✅ PASS: model can overfit the tiny subset (pipeline likely OK).")
    else:
        print("   ⚠️ WARN: loss did not drop enough; check alignment/normalization.")


if __name__ == "__main__":
    main()
