#!/usr/import/env python3
"""
Generate training vs validation loss curves for the four models.
Reads from experiments/logs/Tiles_{MODEL}_S42_log.csv
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "experiments", "logs")
OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "presentation_figures")

MODELS = {
    "UNET": "Tiles_UNET_S42_log.csv",
    "LSTM": "Tiles_LSTM_S42_log.csv",
    "TRANSFORMER": "Tiles_TRANSFORMER_S42_log.csv",
    "MAMBA": "Tiles_MAMBA_S42_log.csv",
}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fig_h_train_val_curves.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, (model_name, log_file) in zip(axes, MODELS.items()):
        log_path = os.path.join(LOG_DIR, log_file)
        if not os.path.exists(log_path):
            print(f"⚠️ Warning: Log for {model_name} not found at {log_path}. Skipping.")
            ax.set_title(f"{model_name} (No Data)")
            ax.axis("off")
            continue
            
        df = pd.read_csv(log_path)
        
        # Ensure we have the required columns
        if "loss" not in df.columns or "val_loss" not in df.columns:
            print(f"⚠️ Warning: 'loss' or 'val_loss' missing in {log_file}")
            ax.set_title(f"{model_name} (Missing Columns)")
            ax.axis("off")
            continue
            
        epochs = df["epoch"] if "epoch" in df.columns else df.index + 1
        loss = df["loss"]
        val_loss = df["val_loss"]

        # Plot curves
        ax.plot(epochs, loss, label="Train Loss", color="#1f77b4", linewidth=2.5, linestyle="-")
        ax.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.5, linestyle="--")

        # Mark deepest validation point
        best_epoch = val_loss.idxmin()
        best_val = val_loss.min()
        ax.scatter(epochs.iloc[best_epoch], best_val, color="red", zorder=5, s=60)
        ax.annotate(f"Best: {best_val:.4f}\n(Ep {epochs.iloc[best_epoch]})", 
                    (epochs.iloc[best_epoch], best_val),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=9, fontweight='bold', color='darkred')

        ax.set_title(f"Learning Curve: {model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss (MSE)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=11)
        
        # Limit y-axis if huge spikes exist (e.g. initial epochs) to keep plot readable
        y_upper = min(val_loss.max(), np.percentile(val_loss, 95) * 1.5) if len(val_loss) > 5 else val_loss.max()
        if not np.isnan(y_upper):
            ax.set_ylim(bottom=0, top=max(loss.max() * 1.1, val_loss.max() * 1.1))

    fig.suptitle("Training vs Validation Loss (Tiles Models)", fontsize=18, fontweight="bold", y=1.03)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Successfully generated {out_path}")

if __name__ == "__main__":
    import numpy as np # import inside since we just need it to check scale bounds
    main()
