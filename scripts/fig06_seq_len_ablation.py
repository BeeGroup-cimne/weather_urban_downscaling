#!/usr/bin/env python3
"""
Generate Fig06: sequence length ablation (SEQ_LEN=6 vs 12).
Placeholder values; replace with real metrics.
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
    out_path = os.path.join(out_dir, "fig06_seq_len_ablation.png")

    seq = [6, 12]
    mae = [0.85, 0.80]
    ssim = [0.78, 0.80]

    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax2 = ax1.twinx()
    ax1.plot(seq, mae, marker="o", label="MAE (°C)", color="#4c78a8")
    ax2.plot(seq, ssim, marker="o", label="SSIM", color="#f58518")
    ax1.set_xlabel("SEQ_LEN")
    ax1.set_ylabel("MAE (°C)")
    ax2.set_ylabel("SSIM")
    ax1.set_title("Sequence Length Ablation (replace with real values)")

    fig.text(0.99, 0.01, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
