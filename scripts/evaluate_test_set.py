#!/usr/bin/env python3
"""
Evaluate a trained model on the TEST split with de-normalized metrics.
Uses the same BigDataPipeline and Config time splits.
"""

import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.data_loader import BigDataPipeline
from src.models_legacy import ModelZoo


def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    if model_type == "transformer":
        return ModelZoo.build_transformer()
    return ModelZoo.build_unet()


def load_stats():
    if not os.path.exists(Config.STATS_PATH):
        raise FileNotFoundError(f"Stats not found: {Config.STATS_PATH}")
    stats = np.load(Config.STATS_PATH)
    return float(stats["mean_hr"]), float(stats["std_hr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm", "transformer"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=0, help="0 = all")
    parser.add_argument("--ssim-samples", type=int, default=0, help="0 = skip SSIM")
    parser.add_argument("--out-csv", default="", help="Optional CSV path to save metrics row.")
    parser.add_argument("--split-mode", default="inherit", choices=["inherit", "time", "fraction"])
    parser.add_argument("--split-fraction", type=float, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return 2

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

    # Load stats for de-normalization
    mean_hr, std_hr = load_stats()

    # Build pipeline and dataset (updates Config shapes)
    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    if args.split == "test":
        _, _, eval_ds = pipeline.get_tf_datasets(include_test=True)
    else:
        train_ds, val_ds = pipeline.get_tf_datasets()
        eval_ds = train_ds if args.split == "train" else val_ds

    # Build and load model after Config shapes are updated
    model = build_model(args.model_type)
    model.load_weights(args.model_path)

    # Optional SSIM
    ssim_fn = None
    if args.ssim_samples > 0:
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_fn = ssim
        except Exception as e:
            print(f"⚠️ SSIM no disponible: {e}")

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    ssim_sum = 0.0
    ssim_count = 0

    max_batches = args.max_batches if args.max_batches and args.max_batches > 0 else None

    for i, ((x_lr, x_st), y_true) in enumerate(eval_ds):
        if max_batches is not None and i >= max_batches:
            break

        y_pred = model((x_lr, x_st), training=False)

        y_pred = y_pred.numpy() * std_hr + mean_hr
        y_true = y_true.numpy() * std_hr + mean_hr

        diff = y_pred - y_true
        abs_sum += np.abs(diff).sum()
        sq_sum += (diff ** 2).sum()
        count += diff.size

        if ssim_fn and ssim_count < args.ssim_samples:
            # Compute SSIM on a limited number of frames
            b, t, _, _, _ = y_true.shape
            for bi in range(b):
                for ti in range(t):
                    if ssim_count >= args.ssim_samples:
                        break
                    yt = y_true[bi, ti, :, :, 0]
                    yp = y_pred[bi, ti, :, :, 0]
                    data_range = yt.max() - yt.min() + 1e-6
                    ssim_sum += ssim_fn(yt, yp, data_range=data_range)
                    ssim_count += 1

    if count == 0:
        print("❌ No data to evaluate.")
        return 3

    mae = abs_sum / count
    rmse = (sq_sum / count) ** 0.5
    mse = sq_sum / count

    print("✅ Test Evaluation")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MSE:  {mse:.4f}")
    ssim_mean = float("nan")
    if ssim_count > 0:
        ssim_mean = ssim_sum / ssim_count
        print(f"   SSIM: {ssim_mean:.4f} (samples={ssim_count})")

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model_type", "model_path", "split", "mae", "rmse", "mse", "ssim", "ssim_samples"])
            w.writerow([
                args.model_type,
                args.model_path,
                args.split,
                f"{mae:.6f}",
                f"{rmse:.6f}",
                f"{mse:.6f}",
                "" if np.isnan(ssim_mean) else f"{ssim_mean:.6f}",
                int(ssim_count),
            ])
        print(f"   CSV:  {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
