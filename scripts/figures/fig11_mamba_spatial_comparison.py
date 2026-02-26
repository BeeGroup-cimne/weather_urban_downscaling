#!/usr/bin/env python3
"""
Generate a visual spatial comparison between Mamba (SEQ=6), Mamba (SEQ=12) and the Ground Truth (HR).
Loads both models, predicts on the same Test Set frame, and plots a 1x3 grid.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.data_loader import BigDataPipeline
from src.models_legacy import ModelZoo

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "presentation_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.2,
})

def load_stats():
    stats = np.load(Config.STATS_PATH)
    return float(stats["mean_hr"]), float(stats["std_hr"])

def _rotate(img):
    """Rotate image 90 degrees CCW to align North Up for plotting."""
    return np.rot90(img, k=1)

def main():
    print("🚀 Initializing Mamba Spatial Comparison Generator...")
    
    mean_hr, std_hr = load_stats()
    
    # ── 1. Pipeline para SEQ=6 ──
    Config.SEQ_LEN = 6
    pipe6 = BigDataPipeline(Config)
    pipe6.process_static_data()
    pipe6.run_etl_process()
    _, _, test_ds_6 = pipe6.get_tf_datasets(include_test=True)
    
    # IMPORTANTE: Alineación temporal.
    # Dado que SEQ=12 empieza 6 frames "antes" (para llenar su buffer de 12),
    # el primer output target válido corresponde temporalmente al frame +6 del SEQ=6.
    # Usamos .unbatch().skip().batch() para no depender de si BATCH_SIZE es 2, 4 u 8.
    test_ds_6 = test_ds_6.unbatch().skip(6).batch(Config.BATCH_SIZE)
    val_iter_6 = iter(test_ds_6)
    
    try:
        (x_lr_6, x_st_6), y_true_6 = next(val_iter_6)
    except StopIteration:
        print("❌ Dataset vacío o no alineable.")
        return
        
    mamba6 = ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    path6 = os.path.join(PROJECT_ROOT, "experiments", "models", "Tiles_MAMBA_S42_best.h5")
    mamba6.load_weights(path6)
    y_pred_6 = mamba6((x_lr_6, x_st_6), training=False).numpy()
    
    # ── 2. Pipeline para SEQ=12 ──
    Config.SEQ_LEN = 12
    pipe12 = BigDataPipeline(Config)
    pipe12.process_static_data()
    pipe12.run_etl_process()
    _, _, test_ds_12 = pipe12.get_tf_datasets(include_test=True)
    val_iter_12 = iter(test_ds_12)
    (x_lr_12, x_st_12), y_true_12 = next(val_iter_12)
    
    mamba12 = ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    path12 = os.path.join(PROJECT_ROOT, "experiments", "models", "Ablation_MAMBA_Legacy_S42_SEQ12_best.h5")
    mamba12.load_weights(path12)
    y_pred_12 = mamba12((x_lr_12, x_st_12), training=False).numpy()
    
    # ── 3. Extract exact same frame ──
    # Select batch idx 0, time idx -1 (most recent step in sequence)
    # De-normalize temperatures
    target_hr = y_true_12.numpy()[0, -1, :, :, 0] * std_hr + mean_hr
    pred6_hr = y_pred_6[0, -1, :, :, 0] * std_hr + mean_hr
    pred12_hr = y_pred_12[0, -1, :, :, 0] * std_hr + mean_hr
    
    vmin = min(target_hr.min(), pred6_hr.min(), pred12_hr.min())
    vmax = max(target_hr.max(), pred6_hr.max(), pred12_hr.max())
    cmap = "inferno"
    
    mae6 = np.mean(np.abs(target_hr - pred6_hr))
    mae12 = np.mean(np.abs(target_hr - pred12_hr))
    print(f"📊 Frame {target_hr.shape} MAE -> MAMBA(SEQ=6): {mae6:.3f} °C | MAMBA(SEQ=12): {mae12:.3f} °C")

    # ── 4. Plot 1x3 Grid ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MAMBA SEQ=6
    ax = axes[0]
    im = ax.imshow(_rotate(pred6_hr), cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title("MAMBA (SEQ=6)\nlower structural integrity")
    ax.axis("off")
    
    # MAMBA SEQ=12
    ax = axes[1]
    ax.imshow(_rotate(pred12_hr), cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title("MAMBA (SEQ=12)\ndetailed urban features")
    ax.axis("off")
    
    # Ground Truth
    ax = axes[2]
    ax.imshow(_rotate(target_hr), cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth (HR)\nTarget")
    ax.axis("off")
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Temperature T2m (°C)")
    
    plt.subplots_adjust(wspace=0.1)
    
    out_path = os.path.join(OUT_DIR, "fig_mamba_spatial_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Generated {out_path}")

if __name__ == "__main__":
    main()
