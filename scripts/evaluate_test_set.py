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
    return ModelZoo.build_unet()


def load_stats():
    if not os.path.exists(Config.STATS_PATH):
        raise FileNotFoundError(f"Stats not found: {Config.STATS_PATH}")
    stats = np.load(Config.STATS_PATH)
    return float(stats["mean_hr"]), float(stats["std_hr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = all")
    parser.add_argument("--ssim-samples", type=int, default=0, help="0 = skip SSIM")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return 2

    # Load stats for de-normalization
    mean_hr, std_hr = load_stats()

    # Build and load model
    model = build_model(args.model_type)
    model.load_weights(args.model_path)

    # Build pipeline and test dataset
    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    _, _, test_ds = pipeline.get_tf_datasets(include_test=True)

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

    for i, ((x_lr, x_st), y_true) in enumerate(test_ds):
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
    if ssim_count > 0:
        print(f"   SSIM: {ssim_sum / ssim_count:.4f} (samples={ssim_count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
