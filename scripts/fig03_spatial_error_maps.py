#!/usr/bin/env python3
"""
Generate Fig03: spatial error maps for multiple models.
Placeholder: random fields. Replace with real MAE or bias maps.
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
    out_path = os.path.join(out_dir, "fig03_spatial_error_maps.png")

    rng = np.random.default_rng(1)
    unet = np.abs(rng.normal(size=(64, 64)))
    convlstm = np.abs(rng.normal(scale=0.9, size=(64, 64)))
    mamba = np.abs(rng.normal(scale=0.8, size=(64, 64)))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, data, title in zip(
        axes,
        [unet, convlstm, mamba],
        ["UNet MAE", "ConvLSTM MAE", "Mamba MAE"]
    ):
        im = ax.imshow(data, cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Spatial Error Maps (replace with real metrics)", fontsize=10)
    fig.text(0.99, 0.01, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
