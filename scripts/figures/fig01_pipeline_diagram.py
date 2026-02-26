#!/usr/bin/env python3
"""
Generate Fig01: pipeline diagram (LR + static -> model -> HR).
Placeholder diagram with labeled blocks.
"""

import os
from scripts.fig_utils import ensure_dir, default_fig_dir, safe_import_matplotlib, timestamp


def main():
    if not safe_import_matplotlib():
        return 1
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    out_dir = default_fig_dir()
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "fig01_pipeline.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    blocks = [
        ("LR (ERA5-Land)", (0.05, 0.55)),
        ("Static (Urban)", (0.05, 0.20)),
        ("Fusion + Model\n(U-Net / ConvLSTM / Mamba)", (0.38, 0.35)),
        ("HR Target", (0.72, 0.55)),
        ("Pred HR", (0.72, 0.20)),
    ]

    for label, (x, y) in blocks:
        rect = patches.FancyBboxPatch(
            (x, y), 0.22, 0.25, boxstyle="round,pad=0.02",
            edgecolor="black", facecolor="#f0f0f0", linewidth=1.0
        )
        ax.add_patch(rect)
        ax.text(x + 0.11, y + 0.125, label, ha="center", va="center", fontsize=9)

    # Arrows
    ax.annotate("", xy=(0.38, 0.48), xytext=(0.27, 0.65), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.38, 0.40), xytext=(0.27, 0.28), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.72, 0.60), xytext=(0.60, 0.48), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.72, 0.28), xytext=(0.60, 0.40), arrowprops=dict(arrowstyle="->"))

    ax.text(0.98, 0.02, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
