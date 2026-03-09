#!/usr/bin/env python3
"""
Generate high-quality publication figures from final post-training evaluation outputs.

Outputs:
  experiments/eval_outputs/paper_figures_final/*.png
  experiments/eval_outputs/paper_figures_final/*.pdf
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _resolve_path(root: str, value: str | None) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(root, value))


def _load_config(path: str) -> dict:
    if path.lower().endswith(".json"):
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"YAML config requested but PyYAML is not available ({e}). "
            "Use JSON or install pyyaml."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_fig(fig: plt.Figure, out_base: str, dpi: int = 600) -> None:
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def _plot_exp1(exp1_csv: str, out_dir: str) -> None:
    rows = _read_csv(exp1_csv)
    rows = sorted(rows, key=lambda r: _safe_float(r.get("rmse_mean"), 1e12))

    models = [r["model"] for r in rows]
    rmse = [_safe_float(r["rmse_mean"]) for r in rows]
    mae = [_safe_float(r["mae_mean"]) for r in rows]
    ssim = [_safe_float(r["ssim_mean"]) for r in rows]

    x = np.arange(len(models))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.bar(x - w / 2, rmse, width=w, label="RMSE", color="#1f77b4")
    ax.bar(x + w / 2, mae, width=w, label="MAE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Experiment 1: Spatial Errors")
    ax.set_ylabel("Error")
    ax.legend()

    ax = axes[1]
    ax.bar(x, ssim, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Experiment 1: SSIM")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, max(1.0, max(ssim) * 1.08))

    fig.suptitle("Exp1 Domain-Wide Performance (lower is better for RMSE/MAE)")
    _save_fig(fig, os.path.join(out_dir, "fig_exp1_spatial_performance"))


def _plot_exp2(exp2_summary_csv: str, out_dir: str) -> None:
    rows = _read_csv(exp2_summary_csv)
    segments = ["all", "day", "night"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, seg in zip(axes, segments):
        seg_rows = [r for r in rows if str(r.get("segment", "")) == seg]
        seg_rows = sorted(seg_rows, key=lambda r: _safe_float(r.get("RMSE_model"), 1e12))

        models = [r["model"] for r in seg_rows]
        rmse = [_safe_float(r["RMSE_model"]) for r in seg_rows]
        ax.bar(models, rmse, color="#4c78a8")
        ax.set_title(f"Exp2 Segment: {seg}")
        ax.tick_params(axis="x", rotation=30)
        if seg == "all":
            ax.set_ylabel("RMSE vs stations")

    fig.suptitle("Exp2 Station Validation (RMSE by Segment)")
    _save_fig(fig, os.path.join(out_dir, "fig_exp2_station_rmse_segments"))


def _plot_exp3(exp3_csv: str, out_dir: str) -> None:
    rows = _read_csv(exp3_csv)
    rows = sorted(rows, key=lambda r: _safe_float(r.get("rmse_mean"), 1e12))
    models = [r["model"] for r in rows]
    rmse = [_safe_float(r["rmse_mean"]) for r in rows]
    mae = [_safe_float(r["mae_mean"]) for r in rows]
    ssim = [_safe_float(r["ssim_mean"]) for r in rows]

    x = np.arange(len(models))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    ax.bar(x - w / 2, rmse, width=w, label="RMSE", color="#1f77b4")
    ax.bar(x + w / 2, mae, width=w, label="MAE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title("Exp3 Bottleneck Ablation: Error")
    ax.set_ylabel("Error")
    ax.legend()

    ax = axes[1]
    ax.bar(x, ssim, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title("Exp3 Bottleneck Ablation: SSIM")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, max(1.0, max(ssim) * 1.08))

    _save_fig(fig, os.path.join(out_dir, "fig_exp3_bottleneck_ablation"))


def _plot_cs1(cs1_csv: str, out_dir: str) -> None:
    rows = _read_csv(cs1_csv)
    rows = sorted(rows, key=lambda r: _safe_float(r.get("rmse_mean"), 1e12))
    models = [r["model"] for r in rows]
    rmse = [_safe_float(r["rmse_mean"]) for r in rows]
    mae = [_safe_float(r["mae_mean"]) for r in rows]
    ssim = [_safe_float(r["ssim_mean"]) for r in rows]

    x = np.arange(len(models))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.bar(x - w / 2, rmse, width=w, label="RMSE", color="#9467bd")
    ax.bar(x + w / 2, mae, width=w, label="MAE", color="#8c564b")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Case Study 1: Heatwave Errors")
    ax.set_ylabel("Error")
    ax.legend()

    ax = axes[1]
    ax.bar(x, ssim, color="#17becf")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Case Study 1: Heatwave SSIM")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, max(1.0, max(ssim) * 1.08))

    _save_fig(fig, os.path.join(out_dir, "fig_cs1_heatwave_performance"))


def _plot_cs2(cs2_csv: str, out_dir: str) -> None:
    rows = _read_csv(cs2_csv)
    by_model_eps: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for r in rows:
        model = str(r.get("model_key", "")).strip()
        eps = _safe_float(r.get("epsilon_K"))
        mad = _safe_float(r.get("pred_mean_abs_dev_vs_clean_C"))
        if model and np.isfinite(eps) and np.isfinite(mad):
            by_model_eps[(model, eps)].append(mad)

    models = sorted({k[0] for k in by_model_eps.keys()})
    eps_all = sorted({k[1] for k in by_model_eps.keys()})

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    palette = {
        "mamba": "#1f77b4",
        "lstm": "#ff7f0e",
        "unet": "#2ca02c",
        "transformer": "#d62728",
    }
    for m in models:
        y = []
        yerr = []
        for e in eps_all:
            vals = by_model_eps.get((m, e), [])
            if not vals:
                y.append(np.nan)
                yerr.append(np.nan)
                continue
            arr = np.array(vals, dtype=float)
            y.append(float(np.mean(arr)))
            yerr.append(float(np.std(arr, ddof=0)))
        ax.errorbar(
            eps_all,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=m,
            color=palette.get(m, None),
        )

    ax.set_xlabel("Perturbation amplitude ε (K)")
    ax.set_ylabel("Mean absolute deviation vs clean (C)")
    ax.set_title("Case Study 2: Monte Carlo Robustness")
    ax.legend(title="Model")
    _save_fig(fig, os.path.join(out_dir, "fig_cs2_montecarlo_robustness"))


def _extract_ts_key(path: str) -> Optional[str]:
    name = os.path.basename(path)
    m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.png$", name)
    return m.group(1) if m else None


def _common_timestamp_pair(
    folder: str,
    pattern_a: str,
    pattern_b: str,
) -> Optional[Tuple[str, str]]:
    files_a = sorted(glob.glob(os.path.join(folder, pattern_a)))
    files_b = sorted(glob.glob(os.path.join(folder, pattern_b)))
    map_a = {k: p for p in files_a if (k := _extract_ts_key(p))}
    map_b = {k: p for p in files_b if (k := _extract_ts_key(p))}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common:
        return None
    key = common[len(common) // 2]
    return map_a[key], map_b[key]


def _plot_side_by_side(
    img_a: str,
    img_b: str,
    title_a: str,
    title_b: str,
    suptitle: str,
    out_base: str,
) -> None:
    arr_a = mpimg.imread(img_a)
    arr_b = mpimg.imread(img_b)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    axes[0].imshow(arr_a)
    axes[0].axis("off")
    axes[0].set_title(title_a)
    axes[1].imshow(arr_b)
    axes[1].axis("off")
    axes[1].set_title(title_b)
    fig.suptitle(suptitle)
    _save_fig(fig, out_base, dpi=600)


def _plot_qualitative_panels(cfg: dict, root: str, out_dir: str) -> None:
    n = cfg.get("narrative", {})
    exp1_dir = _resolve_path(root, n.get("exp1", {}).get("output_dir", ""))
    cs1_dir = _resolve_path(root, n.get("case_study_1", {}).get("output_dir", ""))
    exp2_dir = _resolve_path(root, n.get("exp2", {}).get("output_dir", ""))

    exp1_fig = os.path.join(exp1_dir, "figures")
    pair = _common_timestamp_pair(
        exp1_fig,
        "exp1_*_MAMBA_*_*.png",
        "exp1_*_BASELINE_BILINEAR_*_*.png",
    )
    if pair:
        _plot_side_by_side(
            pair[0],
            pair[1],
            "Hybrid-Mamba",
            "Baseline Bilinear",
            "Exp1 qualitative comparison (same timestamp)",
            os.path.join(out_dir, "fig_exp1_qualitative_mamba_vs_bilinear"),
        )

    cs1_fig = os.path.join(cs1_dir, "figures")
    pair = _common_timestamp_pair(
        cs1_fig,
        "cs1_*_MAMBA_*_*.png",
        "cs1_*_BASELINE_NEAREST_*_*.png",
    )
    if pair:
        _plot_side_by_side(
            pair[0],
            pair[1],
            "Hybrid-Mamba",
            "Baseline Nearest",
            "CS1 heatwave qualitative comparison (same timestamp)",
            os.path.join(out_dir, "fig_cs1_qualitative_mamba_vs_baseline"),
        )

    # Exp2 scatter comparison (existing model-level plots).
    mamba_sc = os.path.join(exp2_dir, "mamba", "stations_eval_scatter.png")
    bl_sc = os.path.join(exp2_dir, "baseline_bilinear", "stations_eval_scatter.png")
    if os.path.exists(mamba_sc) and os.path.exists(bl_sc):
        _plot_side_by_side(
            mamba_sc,
            bl_sc,
            "Mamba",
            "Baseline Bilinear",
            "Exp2 station scatter comparison",
            os.path.join(out_dir, "fig_exp2_scatter_mamba_vs_bilinear"),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate high-quality final publication figures.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--out-dir",
        default="experiments/eval_outputs/paper_figures_final",
        help="Output directory for figures.",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)

    out_dir = _resolve_path(root, args.out_dir)
    _ensure_dir(out_dir)
    _setup_style()

    n = cfg.get("narrative", {})
    exp1_dir = _resolve_path(root, n.get("exp1", {}).get("output_dir", ""))
    exp2_dir = _resolve_path(root, n.get("exp2", {}).get("output_dir", ""))
    exp3_dir = _resolve_path(root, n.get("exp3", {}).get("output_dir", ""))
    cs1_dir = _resolve_path(root, n.get("case_study_1", {}).get("output_dir", ""))
    cs2_dir = _resolve_path(root, n.get("case_study_2", {}).get("output_dir", ""))

    _plot_exp1(os.path.join(exp1_dir, "metrics_aggregate.csv"), out_dir)
    _plot_exp2(os.path.join(exp2_dir, "stations_eval_models_summary.csv"), out_dir)
    _plot_exp3(os.path.join(exp3_dir, "fullframe_eval_aggregate.csv"), out_dir)
    _plot_cs1(os.path.join(cs1_dir, "metrics_aggregate.csv"), out_dir)
    _plot_cs2(os.path.join(cs2_dir, "robustness_summary.csv"), out_dir)
    _plot_qualitative_panels(cfg, root, out_dir)

    print(f"✅ Publication figures generated in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

