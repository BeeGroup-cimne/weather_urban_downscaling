#!/usr/bin/env python3
"""
Generate 6 presentation-quality figures from Experiment 1 results.

Fig A: Metrics comparison bar chart (MAE, RMSE, SSIM with CI)
Fig B: Qualitative comparison grid (6 models × 2 timestamps)
Fig C: Training curves (val_loss over epochs for 4 models)
Fig D: Ranking stability (tiles vs full-frame slope chart)
# Fig E: Mamba vs Nearest before/after (3 heatwave episodes)
# Fig F: Heatwave temporal MAE profile (per-timestamp grouped bars)
Fig G: Full-frame qualitative comparison (Tiles vs Legacy / Native Full-frame)
"""

import os
import sys
import csv

import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# ── Paths ──────────────────────────────────────────────────────────────
EXP1_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments", "heatwaves", "publish_run_20260220_220458",
)
EXP3_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments", "fullframe", "experiment3_20260221_221940",
)
LOGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "logs")
OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "presentation_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Pretty model names and colors ─────────────────────────────────────
MODEL_DISPLAY = {
    "mamba": "Mamba",
    "lstm": "LSTM",
    "convlstm": "LSTM",
    "unet": "UNet",
    "transformer": "Transformer",
    "baseline_bilinear": "Bilinear",
    "baseline_nearest": "Nearest",
}

MODEL_COLORS = {
    "Mamba": "#2ca02c",
    "LSTM": "#1f77b4",
    "UNet": "#ff7f0e",
    "Transformer": "#9467bd",
    "Bilinear": "#bcbd22",
    "Nearest": "#7f7f7f",
}

DL_ORDER = ["UNet", "LSTM", "Transformer", "Mamba"]
ALL_ORDER = ["UNet", "LSTM", "Transformer", "Mamba", "Bilinear", "Nearest"]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ═══════════════════════════════════════════════════════════════════════
# FIG A: Metrics comparison bar chart
# ═══════════════════════════════════════════════════════════════════════
def fig_a_metrics_bar():
    print("📊 Generating Fig A: Metrics comparison...")
    agg_path = os.path.join(EXP1_DIR, "metrics_aggregate_ci.csv")
    rows = _read_csv(agg_path)

    data = {}
    for r in rows:
        name = MODEL_DISPLAY.get(r["model"], r["model"])
        data[name] = {
            "mae": float(r["mae_mean"]),
            "mae_lo": float(r["mae_ci_low"]),
            "mae_hi": float(r["mae_ci_high"]),
            "rmse": float(r["rmse_mean"]),
            "rmse_lo": float(r["rmse_ci_low"]),
            "rmse_hi": float(r["rmse_ci_high"]),
            "ssim": float(r["ssim_mean"]),
            "ssim_lo": float(r["ssim_ci_low"]),
            "ssim_hi": float(r["ssim_ci_high"]),
        }

    models = [m for m in ALL_ORDER if m in data]
    x = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE
    ax = axes[0]
    vals = [data[m]["mae"] for m in models]
    errs_lo = [data[m]["mae"] - data[m]["mae_lo"] for m in models]
    errs_hi = [data[m]["mae_hi"] - data[m]["mae"] for m in models]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.errorbar(x, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black",
                capsize=4, capthick=1.2, linewidth=1.2)
    ax.set_title("MAE (°C)  ↓ lower is better", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, max(vals) * 1.25)

    # RMSE
    ax = axes[1]
    vals = [data[m]["rmse"] for m in models]
    errs_lo = [data[m]["rmse"] - data[m]["rmse_lo"] for m in models]
    errs_hi = [data[m]["rmse_hi"] - data[m]["rmse"] for m in models]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.errorbar(x, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black",
                capsize=4, capthick=1.2, linewidth=1.2)
    ax.set_title("RMSE (°C)  ↓ lower is better", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, max(vals) * 1.25)

    # SSIM
    ax = axes[2]
    vals = [data[m]["ssim"] for m in models]
    errs_lo = [data[m]["ssim"] - data[m]["ssim_lo"] for m in models]
    errs_hi = [data[m]["ssim_hi"] - data[m]["ssim"] for m in models]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.errorbar(x, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black",
                capsize=4, capthick=1.2, linewidth=1.2)
    ax.set_title("SSIM  ↑ higher is better", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0.65, min(1.0, max(vals) * 1.08))

    fig.suptitle(
        "Experiment 1 — Heatwave Downscaling Metrics (6 timestamps, 95% CI)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_a_metrics_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIG B: Qualitative comparison grid
# ═══════════════════════════════════════════════════════════════════════
def fig_b_qualitative_grid():
    print("🗺️  Generating Fig B: Qualitative comparison grid...")
    figures_dir = os.path.join(EXP1_DIR, "figures")

    # Load MAE from metrics_raw.csv for annotations
    metrics_raw_path = os.path.join(EXP1_DIR, "metrics_raw.csv")
    mae_lookup = {}
    if os.path.exists(metrics_raw_path):
        for r in _read_csv(metrics_raw_path):
            mae_lookup[r["experiment"]] = float(r["mae"])

    # Pick 2 heatwave timestamps: peak day + peak night
    ts_day = "2017-06-28_15_00_00"
    ts_night = "2017-06-28_01_00_00"

    model_keys = [
        ("BASELINE_NEAREST", "Nearest"),
        ("UNET_S42", "UNet"),
        ("LSTM_S42", "LSTM"),
        ("TRANSFORMER_S42", "Transformer"),
        ("MAMBA_S42", "Mamba"),
    ]

    fig, axes = plt.subplots(
        len(model_keys), 2, figsize=(10, len(model_keys) * 2.8),
        gridspec_kw={"wspace": 0.08, "hspace": 0.25},
    )

    # Collect all predictions to compute shared color limits
    all_preds = []
    for ts in [ts_day, ts_night]:
        for key, _ in model_keys:
            npy_name = f"tiles_publish_PUB_{key}_{ts}.npy"
            npy_path = os.path.join(figures_dir, npy_name)
            if os.path.exists(npy_path):
                all_preds.append(np.load(npy_path))

    if all_preds:
        all_vals = np.concatenate([a.ravel() for a in all_preds])
        vmin_shared = float(np.nanpercentile(all_vals, 2))
        vmax_shared = float(np.nanpercentile(all_vals, 98))
    else:
        vmin_shared, vmax_shared = 0, 1

    im = None
    for col_idx, (ts, ts_label) in enumerate(
        [(ts_day, "28 Jun 2017 15:00 (day)"), (ts_night, "28 Jun 2017 01:00 (night)")]
    ):
        for row_idx, (key, label) in enumerate(model_keys):
            npy_name = f"tiles_publish_PUB_{key}_{ts}.npy"
            npy_path = os.path.join(figures_dir, npy_name)
            exp_key = f"PUB_{key}_{ts}"

            ax = axes[row_idx, col_idx]

            if not os.path.exists(npy_path):
                ax.text(0.5, 0.5, f"Missing:\n{npy_name}",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # .npy is a 2D prediction array (H, W)
            pred = np.load(npy_path)
            if pred.ndim == 3:
                pred = pred[:, :, 0]

            im = ax.imshow(pred, cmap="YlOrRd", vmin=vmin_shared, vmax=vmax_shared,
                           aspect="equal", interpolation="nearest", origin="lower")

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0,
                              labelpad=70, va="center")
            if row_idx == 0:
                ax.set_title(ts_label, fontsize=11, fontweight="bold")

            # Add MAE from metrics_raw.csv
            mae_val = mae_lookup.get(exp_key)
            if mae_val is not None:
                ax.text(0.97, 0.03, f"MAE={mae_val:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Temperature (normalized)")

    fig.suptitle(
        "Qualitative Comparison — Heatwave Downscaling Predictions",
        fontsize=14, fontweight="bold", y=0.98,
    )
    out = os.path.join(OUT_DIR, "fig_b_qualitative_grid.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIG C: Training curves
# ═══════════════════════════════════════════════════════════════════════
def fig_c_training_curves():
    print("📈 Generating Fig C: Training curves...")

    models_logs = {
        "UNet": os.path.join(LOGS_DIR, "Tiles_UNET_S42_log.csv"),
        "LSTM": os.path.join(LOGS_DIR, "Tiles_LSTM_S42_log.csv"),
        "Transformer": os.path.join(LOGS_DIR, "Tiles_TRANSFORMER_S42_log.csv"),
        "Mamba": os.path.join(LOGS_DIR, "Tiles_MAMBA_S42_log.csv"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name in DL_ORDER:
        log_path = models_logs[model_name]
        if not os.path.exists(log_path):
            print(f"  ⚠️ Missing: {log_path}")
            continue

        rows = _read_csv(log_path)
        epochs = [int(float(r["epoch"])) for r in rows]
        val_loss = [float(r["val_loss"]) for r in rows]
        val_mae = [float(r["val_mae"]) for r in rows]
        color = MODEL_COLORS[model_name]

        ax1.plot(epochs, val_loss, label=model_name, color=color, linewidth=2, alpha=0.9)
        ax2.plot(epochs, val_mae, label=model_name, color=color, linewidth=2, alpha=0.9)

        # Mark best epoch
        best_idx = np.argmin(val_loss)
        ax1.scatter([epochs[best_idx]], [val_loss[best_idx]],
                    color=color, s=80, zorder=5, edgecolors="white", linewidth=1.5)

        best_idx_mae = np.argmin(val_mae)
        ax2.scatter([epochs[best_idx_mae]], [val_mae[best_idx_mae]],
                    color=color, s=80, zorder=5, edgecolors="white", linewidth=1.5)

    ax1.set_title("Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss (MSE)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)

    ax2.set_title("Validation MAE", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MAE")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)

    # Annotation for Mamba
    ax1.annotate(
        "Mamba converges\n2× lower",
        xy=(35, 0.050), fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
        xytext=(25, 0.12), color="#2ca02c", fontweight="bold",
        ha="center",
    )

    fig.suptitle(
        "Training Convergence — Tiles Training (50 epochs, seed 42)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_c_training_curves.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIG D: Ranking stability (slope chart)
# ═══════════════════════════════════════════════════════════════════════
def fig_d_ranking_stability():
    print("📊 Generating Fig D: Ranking stability...")

    ranking_path = os.path.join(EXP3_DIR, "ranking_stability_vs_exp1.csv")
    rows = _read_csv(ranking_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    x_left = 0.2
    x_right = 0.8

    models_data = []
    for r in rows:
        name = MODEL_DISPLAY.get(r["model"], r["model"])
        r1 = r.get("rank_exp1_tiles", "")
        r3 = r.get("rank_exp3_fullframe", "")
        if r1 and r3:
            models_data.append({
                "name": name,
                "rank_tiles": int(r1),
                "rank_fullframe": int(r3),
                "rmse_tiles": float(r["rmse_exp1_tiles"]),
                "rmse_fullframe": float(r["rmse_exp3_fullframe"]),
            })

    max_rank = max(m["rank_tiles"] for m in models_data)

    for m in models_data:
        color = MODEL_COLORS.get(m["name"], "#333333")
        delta = abs(m["rank_tiles"] - m["rank_fullframe"])
        lw = 3.5 if delta == 0 else 2
        ls = "-" if delta == 0 else "--"
        alpha = 1.0 if delta == 0 else 0.6

        ax.plot(
            [x_left, x_right],
            [m["rank_tiles"], m["rank_fullframe"]],
            color=color, linewidth=lw, linestyle=ls, alpha=alpha,
            marker="o", markersize=12, markeredgecolor="white", markeredgewidth=2,
            zorder=3,
        )

        # Labels
        ax.text(
            x_left - 0.06, m["rank_tiles"], f'{m["name"]}',
            ha="right", va="center", fontsize=11, fontweight="bold", color=color,
        )
        ax.text(
            x_right + 0.06, m["rank_fullframe"], f'{m["name"]}',
            ha="left", va="center", fontsize=11, fontweight="bold", color=color,
        )

        # RMSE values
        ax.text(
            x_left, m["rank_tiles"] + 0.25,
            f'RMSE={m["rmse_tiles"]:.3f}',
            ha="center", va="top", fontsize=8, color=color, alpha=0.8,
        )
        ax.text(
            x_right, m["rank_fullframe"] + 0.25,
            f'RMSE={m["rmse_fullframe"]:.3f}',
            ha="center", va="top", fontsize=8, color=color, alpha=0.8,
        )

        # Stability badge
        if delta == 0:
            mid_y = (m["rank_tiles"] + m["rank_fullframe"]) / 2
            ax.text(
                0.5, mid_y - 0.15, "✓ Stable",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15),
            )

    # Axis styling
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(max_rank + 0.8, 0.2)
    ax.set_yticks(range(1, max_rank + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, max_rank + 1)], fontsize=12)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(
        ["Exp1: Tiles\n(6 heatwave timestamps)", "Exp3: Full-frame\n(test set Sep-Oct)"],
        fontsize=11, fontweight="bold",
    )
    ax.axvline(x_left, color="#ddd", linewidth=0.8, zorder=0)
    ax.axvline(x_right, color="#ddd", linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel("Rank (by RMSE)", fontsize=12)

    ax.set_title(
        "Ranking Stability — Tiles vs Full-Frame Evaluation",
        fontsize=14, fontweight="bold", pad=15,
    )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_d_ranking_stability.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIG E: Mamba vs Nearest — Before / After
# ═══════════════════════════════════════════════════════════════════════
def fig_e_mamba_vs_baseline():
    print("🔄 Generating Fig E: Mamba vs Nearest (before/after)...")
    figures_dir = os.path.join(EXP1_DIR, "figures")

    # Load MAE lookup
    metrics_raw_path = os.path.join(EXP1_DIR, "metrics_raw.csv")
    mae_lookup = {}
    if os.path.exists(metrics_raw_path):
        for r in _read_csv(metrics_raw_path):
            mae_lookup[r["experiment"]] = float(r["mae"])

    # 3 heatwave episodes: Jun 28 (peak), Jul 13 (moderate), Aug 15 (late summer)
    episodes = [
        ("2017-06-28_15_00_00", "28 Jun 15:00\n(Peak heatwave)"),
        ("2017-07-13_16_00_00", "13 Jul 16:00\n(Moderate heat)"),
        ("2017-08-15_15_00_00", "15 Aug 15:00\n(Late summer)"),
    ]

    # Collect all data for shared color limits
    all_vals = []
    for ts, _ in episodes:
        for key in ["BASELINE_NEAREST", "MAMBA_S42"]:
            npy_path = os.path.join(figures_dir, f"tiles_publish_PUB_{key}_{ts}.npy")
            if os.path.exists(npy_path):
                arr = np.load(npy_path)
                all_vals.append(arr.ravel())

    if all_vals:
        combined = np.concatenate(all_vals)
        vmin = float(np.nanpercentile(combined, 2))
        vmax = float(np.nanpercentile(combined, 98))
    else:
        vmin, vmax = 0, 1

    fig, axes = plt.subplots(
        2, len(episodes), figsize=(4.5 * len(episodes), 8),
        gridspec_kw={"wspace": 0.08, "hspace": 0.15},
    )

    row_labels = [("BASELINE_NEAREST", "Nearest\n(baseline)"), ("MAMBA_S42", "Mamba\n(ours)")]
    im = None

    for col_idx, (ts, col_label) in enumerate(episodes):
        for row_idx, (key, row_label) in enumerate(row_labels):
            ax = axes[row_idx, col_idx]
            npy_path = os.path.join(figures_dir, f"tiles_publish_PUB_{key}_{ts}.npy")
            exp_key = f"PUB_{key}_{ts}"

            if not os.path.exists(npy_path):
                ax.text(0.5, 0.5, "Missing", transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            pred = np.load(npy_path)
            if pred.ndim == 3:
                pred = pred[:, :, 0]

            im = ax.imshow(pred, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest", origin="lower")

            mae_val = mae_lookup.get(exp_key)
            if mae_val is not None:
                ax.text(0.97, 0.03, f"MAE={mae_val:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

            if row_idx == 0:
                ax.set_title(col_label, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold",
                              rotation=0, labelpad=55, va="center")
            ax.set_xticks([]); ax.set_yticks([])

    # Add improvement arrows between rows
    for col_idx, (ts, _) in enumerate(episodes):
        mae_near = mae_lookup.get(f"PUB_BASELINE_NEAREST_{ts}")
        mae_mamba = mae_lookup.get(f"PUB_MAMBA_S42_{ts}")
        if mae_near and mae_mamba:
            pct_improvement = (1 - mae_mamba / mae_near) * 100
            ax_top = axes[0, col_idx]
            ax_top.text(
                0.5, -0.04,
                f"▼ {pct_improvement:.0f}% lower MAE",
                transform=ax_top.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold", color="#2ca02c",
            )

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Temperature (normalized)")

    fig.suptitle(
        "Downscaling Quality — Nearest Baseline vs Mamba",
        fontsize=14, fontweight="bold", y=0.98,
    )
    out = os.path.join(OUT_DIR, "fig_e_mamba_vs_baseline.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIG F: Heatwave Temporal MAE Profile
# ═══════════════════════════════════════════════════════════════════════
def fig_f_temporal_mae_profile():
    print("🌡️  Generating Fig F: Heatwave temporal MAE profile...")
    metrics_raw_path = os.path.join(EXP1_DIR, "metrics_raw.csv")
    rows = _read_csv(metrics_raw_path)

    # Parse into structure: {model_display_name: {timestamp_label: mae}}
    # 6 timestamps across 3 heatwave episodes
    ts_order = [
        ("2017-06-28T01:00:00", "Jun 28\n01:00"),
        ("2017-06-28T15:00:00", "Jun 28\n15:00"),
        ("2017-07-13T02:00:00", "Jul 13\n02:00"),
        ("2017-07-13T16:00:00", "Jul 13\n16:00"),
        ("2017-08-15T03:00:00", "Aug 15\n03:00"),
        ("2017-08-15T15:00:00", "Aug 15\n15:00"),
    ]
    ts_labels = [lbl for _, lbl in ts_order]
    ts_isos = [iso for iso, _ in ts_order]

    # Build data: model → list of MAE per timestamp
    model_data = {}
    for r in rows:
        name = MODEL_DISPLAY.get(r["model"], r["model"])
        ts = r["time"]
        if ts in ts_isos:
            model_data.setdefault(name, {})
            model_data[name][ts] = float(r["mae"])

    # Plot order: highlight Mamba vs others
    plot_models = [m for m in ALL_ORDER if m in model_data]
    n_models = len(plot_models)
    n_ts = len(ts_labels)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    bar_width = 0.12
    group_width = bar_width * n_models
    x_base = np.arange(n_ts)

    for m_idx, model in enumerate(plot_models):
        offsets = x_base + (m_idx - n_models / 2 + 0.5) * bar_width
        vals = [model_data[model].get(ts_iso, 0) for ts_iso in ts_isos]
        color = MODEL_COLORS.get(model, "#333")

        is_mamba = model == "Mamba"
        ax.bar(
            offsets, vals, width=bar_width * 0.9,
            color=color, label=model,
            edgecolor="white" if not is_mamba else "#1a7a1a",
            linewidth=0.5 if not is_mamba else 2,
            alpha=0.7 if not is_mamba else 1.0,
            zorder=3 if is_mamba else 2,
        )

        # Add value labels for Mamba
        if is_mamba:
            for x_pos, v in zip(offsets, vals):
                ax.text(x_pos, v + 0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color=color)

    # Heatwave episode background shading
    episodes_bg = [
        ((-0.5, 1.5), "Episode 1: Jun 28", "#fff3e0"),
        ((1.5, 3.5), "Episode 2: Jul 13", "#e8f5e9"),
        ((3.5, 5.5), "Episode 3: Aug 15", "#e3f2fd"),
    ]
    for (x0, x1), ep_label, bg_color in episodes_bg:
        ax.axvspan(x0, x1, alpha=0.3, color=bg_color, zorder=0)
        ax.text((x0 + x1) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.26,
                ep_label, ha="center", va="top", fontsize=9,
                fontstyle="italic", color="#666", alpha=0.8)

    # Day/night icons
    for i, ts_iso in enumerate(ts_isos):
        hour = int(ts_iso.split("T")[1].split(":")[0])
        icon = "☀️" if 6 <= hour <= 20 else "🌙"
        ax.text(i, -0.018, icon, ha="center", va="top",
                fontsize=14, transform=ax.get_xaxis_transform())

    ax.set_xticks(x_base)
    ax.set_xticklabels(ts_labels, fontsize=9)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_xlabel("Heatwave Timestamp", fontsize=12)
    ax.legend(
        loc="upper right", ncol=3, fontsize=9,
        framealpha=0.9, edgecolor="#ccc",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.6, n_ts - 0.4)

    ax.set_title(
        "Per-Timestamp MAE Across Heatwave Episodes — All Models",
        fontsize=14, fontweight="bold", pad=15,
    )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_f_temporal_mae_profile.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIG G: Full-frame qualitative comparison
# ═══════════════════════════════════════════════════════════════════════
def fig_g_fullframe_comparison():
    print("🌍 Generating Fig G: Full-frame qualitative comparison...")
    ff_dir = os.path.join(PROJECT_ROOT, "experiments", "features_fullframe")
    
    # Check if predictions exist
    if not os.path.exists(os.path.join(ff_dir, "fullframe_PUB_TRUTH.npy")):
        print("  ⚠️ Missing full-frame predictions (.npy). Please run generate_fullframe_preds.py first.")
        return

    truth = np.load(os.path.join(ff_dir, "fullframe_PUB_TRUTH.npy"))
    bilinear = np.load(os.path.join(ff_dir, "fullframe_PUB_BASELINE_BILINEAR.npy"))
    tiles_lstm = np.load(os.path.join(ff_dir, "fullframe_PUB_Tiles_LSTM.npy"))
    abl_lstm = np.load(os.path.join(ff_dir, "fullframe_PUB_Ablation_LSTM.npy"))
    tiles_unet = np.load(os.path.join(ff_dir, "fullframe_PUB_Tiles_UNET.npy"))
    abl_unet = np.load(os.path.join(ff_dir, "fullframe_PUB_Ablation_UNET.npy"))
    
    # Create a 2x3 grid
    panels = [
        (truth, "Ground Truth\n(Target)"), (tiles_lstm, "LSTM (Trained by Tiles)"), (abl_lstm, "LSTM (Trained natively Full-Frame)"),
        (bilinear, "Bilinear Baseline"), (tiles_unet, "UNet (Trained by Tiles)"), (abl_unet, "UNet (Trained natively Full-Frame)")
    ]
    
    vmin = float(np.nanpercentile(truth, 1))
    vmax = float(np.nanpercentile(truth, 99))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    
    # Calculate MAE vs Truth for each to display
    maes = []
    for data, label in panels:
        mae = float(np.mean(np.abs(data - truth))) if label != "Ground Truth\n(Target)" else 0.0
        maes.append(mae)

    for i, ((data, label), ax, mae) in enumerate(zip(panels, axes.flatten(), maes)):
        im = ax.imshow(data, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest", origin="lower")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")
        
        if mae > 0:
            ax.text(0.95, 0.05, f"MAE: {mae:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=12, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Temperature (normalized)")
    
    fig.suptitle("Impact of Training Domain on Full-Frame Reconstruction\n(28 Jun 2017 15:00)", 
                 fontsize=16, fontweight="bold", y=0.95)
    
    out = os.path.join(OUT_DIR, "fig_g_fullframe_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print(f"🎨 Generating presentation figures → {OUT_DIR}\n")

    fig_a_metrics_bar()
    fig_b_qualitative_grid()
    fig_c_training_curves()
    fig_d_ranking_stability()
    fig_e_mamba_vs_baseline()
    fig_f_temporal_mae_profile()
    fig_g_fullframe_comparison()

    print(f"\n✅ All figures saved to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
