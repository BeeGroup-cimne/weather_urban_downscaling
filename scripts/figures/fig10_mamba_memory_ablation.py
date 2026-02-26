#!/usr/bin/env python3
"""
Generate a memory ablation figure comparing MAMBA SEQ=6 vs MAMBA SEQ=12.
Plots Validation MAE over epochs and includes a summary table of the SSIM metric.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

LOGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "logs")
OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "presentation_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.2,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

MAMBA_COLORS = {
    "MAMBA (SEQ=6)": "#1f77b4",   # Base Blue
    "MAMBA (SEQ=12)": "#d62728"   # Red
}

def load_logs():
    seq6_path = os.path.join(LOGS_DIR, "Tiles_MAMBA_S42_log.csv")
    seq12_path = os.path.join(LOGS_DIR, "Ablation_MAMBA_Legacy_S42_SEQ12_log.csv")
    
    df6 = pd.read_csv(seq6_path) if os.path.exists(seq6_path) else None
    df12 = pd.read_csv(seq12_path) if os.path.exists(seq12_path) else None
    
    return df6, df12

def plot_ablation_figure():
    df6, df12 = load_logs()
    
    if df6 is None or df12 is None:
        print("❌ Cannot find both logs. Please ensure both Mamba models have been trained.")
        return
    
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)
    
    # Left subplot: The Validation MAE Curves
    ax_curve = fig.add_subplot(gs[0])
    
    ax_curve.plot(df6['epoch'], df6['val_mae'], label="MAMBA (SEQ=6)", color=MAMBA_COLORS["MAMBA (SEQ=6)"], linewidth=3, alpha=0.85)
    ax_curve.plot(df12['epoch'], df12['val_mae'], label="MAMBA (SEQ=12)", color=MAMBA_COLORS["MAMBA (SEQ=12)"], linewidth=3, alpha=0.85)
    
    ax_curve.set_xlabel("Epochs")
    ax_curve.set_ylabel("Validation MAE (Normalized)")
    ax_curve.set_title("Validation MAE Convergence", pad=15)
    ax_curve.legend()
    ax_curve.set_ylim(0.10, 0.20)
    
    # Highlight the local minima
    min_idx_6 = df6["val_mae"].idxmin()
    min_idx_12 = df12["val_mae"].idxmin()
    
    ax_curve.scatter(df6.loc[min_idx_6, 'epoch'], df6.loc[min_idx_6, 'val_mae'], color=MAMBA_COLORS["MAMBA (SEQ=6)"], s=100, zorder=5)
    ax_curve.scatter(df12.loc[min_idx_12, 'epoch'], df12.loc[min_idx_12, 'val_mae'], color=MAMBA_COLORS["MAMBA (SEQ=12)"], s=100, zorder=5)
    
    # Right subplot: Callout Box for Test Set metrics (SSIM, De-normalized MAE)
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    table_text = (
        "FINAL TEST METRICS\n"
        "(De-normalized)\n\n"
        "MAMBA (SEQ=6)\n"
        "MAE:   1.10 °C\n"
        "SSIM:  0.7355\n\n"
        "MAMBA (SEQ=12)\n"
        "MAE:   0.81 °C\n"
        "SSIM:  0.8099\n\n"
        "------------------\n"
        "Conclusion:\n"
        "Expanding memory\n"
        "context to 12 hours\n"
        "drastically improves\n"
        "structural and \n"
        "spatial integrity\n"
        "(+10% SSIM) at the\n"
        "cost of > epochs."
    )
    
    at = AnchoredText(table_text, loc='center', prop=dict(size=14, family='monospace'), frameon=True)
    at.patch.set_boxstyle("round,pad=0.5")
    at.patch.set_facecolor("#f8f9fa")
    at.patch.set_edgecolor("#ced4da")
    ax_text.add_artist(at)
    
    fig.suptitle("Temporal Memory Ablation: Sequence Length Impact on Spatial Integrity", fontweight='bold', y=0.96)
    
    out_path = os.path.join(OUT_DIR, "fig_mamba_memory_ablation.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Generated {out_path}")

if __name__ == "__main__":
    plot_ablation_figure()
