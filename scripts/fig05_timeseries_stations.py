#!/usr/bin/env python3
"""
Generate Fig05: time series at selected stations (HR vs Pred).
Placeholder series; replace with real station time series.
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
    out_path = os.path.join(out_dir, "fig05_timeseries_stations.png")

    t = np.arange(0, 72)
    hr = 20 + 5 * np.sin(2 * np.pi * t / 24)
    pred = hr + np.random.default_rng(2).normal(scale=0.5, size=t.shape)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, hr, label="HR")
    ax.plot(t, pred, label="Pred")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Station Time Series (replace with real stations)")
    ax.legend()

    fig.text(0.99, 0.01, f"Generated {timestamp()}", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✅ Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
