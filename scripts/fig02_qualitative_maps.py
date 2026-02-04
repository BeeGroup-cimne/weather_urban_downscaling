#!/usr/bin/env python3
"""
Generate Fig02: qualitative maps (LR upsampled vs HR vs prediction vs error).
This script creates placeholders. Replace TODO blocks with real data loading.
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
    out_path = os.path.join(out_dir, "fig02_qualitative_maps.png")

    # TODO: replace with real arrays (H, W)
    rng = np.random.default_rng(0)
    lr = rng.normal(size=(64, 64))
    hr = rng.normal(size=(64, 64))
    pred = hr + rng.normal(scale=0.2, size=(64, 64))
    err = np.abs(pred - hr)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, data, title in zip(
        axes,
        [lr, hr, pred, err],
        ["LR (upsampled)", "HR target", "Prediction", "Abs error"]
    ):
        im = ax.imshow(data, cmap="coolwarm")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Qualitative Example (replace with real fields)", fontsize=10)
    fig.text(0.99, 0.01, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
