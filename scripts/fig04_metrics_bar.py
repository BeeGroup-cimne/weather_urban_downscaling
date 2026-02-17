#!/usr/bin/env python3
"""
Generate Fig04: bar chart of metrics (MAE/SSIM) per model.
Placeholder values; replace with real metrics from logs.
"""

import os
import numpy as np
from scripts.fig_utils import ensure_dir, default_fig_dir, safe_import_matplotlib, timestamp


def main():
    if not safe_import_matplotlib():
        return 1
    import matplotlib.pyplot as plt

    out_dir = default_fig_dir()
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "fig04_metrics_bar.png")

    models = ["UNet", "ConvLSTM", "Mamba-6", "Mamba-12"]
    mae = [1.0, 0.9, 0.85, 0.8]
    ssim = [0.70, 0.74, 0.78, 0.80]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, mae, width, label="MAE (°C)", color="#4c78a8")
    ax2.bar(x + width / 2, ssim, width, label="SSIM", color="#f58518")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel("MAE (°C)")
    ax2.set_ylabel("SSIM")
    ax1.set_title("Metrics Summary (replace with real values)")

    fig.text(0.99, 0.01, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
