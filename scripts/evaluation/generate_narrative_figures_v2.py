#!/usr/bin/env python3
"""
Generate reproducible narrative figures (v2) from deterministic eval outputs.

This script does not run training/inference. It only reads artifacts already
produced in experiments/eval_outputs/* and writes publication-ready figures.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_config(path: str) -> dict:
    if path.lower().endswith(".json"):
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


def _resolve_path(root: str, value: str | None) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(root, value))


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


def _slug_time(ts: str) -> str:
    return str(ts).strip().replace("-", "_").replace(":", "_").replace("T", "_").replace(" ", "_")


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


def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.asarray(arr, dtype=float)


def _pick_best(paths: Sequence[str]) -> Optional[str]:
    if not paths:
        return None
    s42 = [p for p in paths if "_S42_" in os.path.basename(p)]
    if s42:
        return sorted(s42)[0]
    return sorted(paths)[0]


def _find_exp1_npy(fig_dir: str, model: str, ts_slug: str) -> Optional[str]:
    mk = model.upper()
    if model.startswith("baseline_"):
        patt = os.path.join(fig_dir, f"exp1_*_{mk}_{ts_slug}.npy")
    else:
        patt = os.path.join(fig_dir, f"exp1_*_{mk}_*_{ts_slug}.npy")
    return _pick_best(glob.glob(patt))


def _find_cs1_npy(fig_dir: str, model: str, ts_slug: str) -> Optional[str]:
    mk = model.upper()
    if model.startswith("baseline_"):
        patt = os.path.join(fig_dir, f"cs1_*_{mk}_{ts_slug}.npy")
    else:
        patt = os.path.join(fig_dir, f"cs1_*_{mk}_*_{ts_slug}.npy")
    return _pick_best(glob.glob(patt))


def _aggregate_exp1_metrics(raw_csv: str) -> Dict[Tuple[str, str], dict]:
    rows = _read_csv(raw_csv)
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        model = str(r.get("model", "")).strip().lower()
        ts = str(r.get("time", "")).strip()
        if model and ts:
            grouped[(model, ts)].append(r)

    out: Dict[Tuple[str, str], dict] = {}
    for key, vals in grouped.items():
        maes = [_safe_float(v.get("mae")) for v in vals]
        rmses = [_safe_float(v.get("rmse")) for v in vals]
        ssims = [_safe_float(v.get("ssim")) for v in vals]
        maes = [x for x in maes if np.isfinite(x)]
        rmses = [x for x in rmses if np.isfinite(x)]
        ssims = [x for x in ssims if np.isfinite(x)]
        out[key] = {
            "mae": float(np.mean(maes)) if maes else float("nan"),
            "rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "ssim": float(np.mean(ssims)) if ssims else float("nan"),
            "n": len(vals),
        }
    return out


def _plot_exp1_day_night_grid(exp1_dir: str, out_dir: str) -> None:
    fig_dir = os.path.join(exp1_dir, "figures")
    ts_day = "2017-06-28T15:00:00"
    ts_night = "2017-06-28T01:00:00"
    ts_pair = [ts_day, ts_night]
    models = ["mamba", "lstm", "unet", "transformer", "baseline_bilinear", "baseline_nearest"]
    labels = ["Mamba", "LSTM", "UNet", "Transformer", "Bilinear", "Nearest"]

    arrays: Dict[Tuple[str, str], np.ndarray] = {}
    for ts in ts_pair:
        s = _slug_time(ts)
        for m in models:
            fp = _find_exp1_npy(fig_dir, m, s)
            if fp:
                arrays[(m, ts)] = _load_npy(fp)

    if not arrays:
        print("⚠️ Exp1 day/night grid skipped: no npy maps found.")
        return

    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays.values()])
    vmin = float(np.nanpercentile(vals, 2.0))
    vmax = float(np.nanpercentile(vals, 98.0))

    fig, axes = plt.subplots(2, len(models), figsize=(2.2 * len(models), 5.3))
    for r, ts in enumerate(ts_pair):
        for c, (m, lbl) in enumerate(zip(models, labels)):
            ax = axes[r, c]
            arr = arrays.get((m, ts))
            if arr is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            im = ax.imshow(arr, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lbl)
            if c == 0:
                ax.set_ylabel("Day" if "T15:00:00" in ts else "Night")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.01)
    cbar.set_label("Temperature (normalized)")
    fig.suptitle("Exp1 Spatial Fields: Day vs Night Across Models")
    _save_fig(fig, os.path.join(out_dir, "fig_v2_exp1_day_night_model_grid"))


def _plot_exp1_mamba_vs_bilinear(exp1_dir: str, out_dir: str) -> None:
    fig_dir = os.path.join(exp1_dir, "figures")
    raw_csv = os.path.join(exp1_dir, "metrics_raw.csv")
    metrics = _aggregate_exp1_metrics(raw_csv) if os.path.exists(raw_csv) else {}

    ts = "2017-06-28T15:00:00"
    s = _slug_time(ts)
    fp_m = _find_exp1_npy(fig_dir, "mamba", s)
    fp_b = _find_exp1_npy(fig_dir, "baseline_bilinear", s)
    if not fp_m or not fp_b:
        print("⚠️ Exp1 Mamba-vs-Bilinear skipped: missing maps.")
        return

    mamba = _load_npy(fp_m)
    bilin = _load_npy(fp_b)
    delta = mamba - bilin

    vals = np.concatenate([mamba[np.isfinite(mamba)].ravel(), bilin[np.isfinite(bilin)].ravel()])
    vmin = float(np.nanpercentile(vals, 2.0))
    vmax = float(np.nanpercentile(vals, 98.0))
    dmax = float(np.nanpercentile(np.abs(delta[np.isfinite(delta)]), 99.0))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    im0 = axes[0].imshow(bilin, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    axes[0].set_title("Baseline Bilinear")
    axes[1].imshow(mamba, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    axes[1].set_title("Hybrid-Mamba")
    im2 = axes[2].imshow(delta, cmap="coolwarm", vmin=-dmax, vmax=dmax, origin="lower", interpolation="nearest")
    axes[2].set_title("Mamba - Bilinear")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    m_stats = metrics.get(("mamba", ts), {})
    b_stats = metrics.get(("baseline_bilinear", ts), {})
    caption = (
        f"{ts} | "
        f"Mamba MAE={_safe_float(m_stats.get('mae')):.3f}, RMSE={_safe_float(m_stats.get('rmse')):.3f} | "
        f"Bilinear MAE={_safe_float(b_stats.get('mae')):.3f}, RMSE={_safe_float(b_stats.get('rmse')):.3f}"
    )
    fig.suptitle("Exp1 Qualitative Contrast and Correction Field")
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)
    fig.colorbar(im0, ax=axes[:2], fraction=0.03, pad=0.02, label="Temperature (normalized)")
    fig.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.02, label="Delta")
    _save_fig(fig, os.path.join(out_dir, "fig_v2_exp1_mamba_vs_bilinear_delta"))


def _plot_exp2_station_heatmap(exp2_dir: str, out_dir: str) -> None:
    fp = os.path.join(exp2_dir, "stations_eval_per_station_all_models.csv")
    if not os.path.exists(fp):
        print("⚠️ Exp2 station heatmap skipped: missing per-station table.")
        return
    rows = _read_csv(fp)
    model_order = ["baseline_bilinear", "baseline_nearest", "mamba", "lstm", "unet", "transformer"]
    model_lbl = {
        "baseline_bilinear": "Bilinear",
        "baseline_nearest": "Nearest",
        "mamba": "Mamba",
        "lstm": "LSTM",
        "unet": "UNet",
        "transformer": "Transformer",
    }
    stations = sorted({str(r.get("station_id", "")).strip() for r in rows if r.get("station_id", "")})
    if not stations:
        print("⚠️ Exp2 station heatmap skipped: no stations found.")
        return
    st_idx = {s: i for i, s in enumerate(stations)}
    mat = np.full((len(stations), len(model_order)), np.nan, dtype=float)

    for r in rows:
        s = str(r.get("station_id", "")).strip()
        m = str(r.get("model", "")).strip().lower()
        if s in st_idx and m in model_order:
            mat[st_idx[s], model_order.index(m)] = _safe_float(r.get("MAE_model"))

    mamba_col = model_order.index("mamba")
    rank_key = np.nan_to_num(mat[:, mamba_col], nan=np.nanmax(np.nan_to_num(mat[:, mamba_col], nan=1e9)))
    order = np.argsort(rank_key)
    mat = mat[order, :]
    stations_sorted = [stations[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.2, 0.45 * len(stations_sorted) + 2.2))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels([model_lbl[m] for m in model_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(stations_sorted)))
    ax.set_yticklabels(stations_sorted)
    ax.set_title("Exp2 Station MAE by Model (lower is better)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Station")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="MAE (C)")
    _save_fig(fig, os.path.join(out_dir, "fig_v2_exp2_station_mae_heatmap"))


def _plot_exp2_segments(exp2_dir: str, out_dir: str) -> None:
    fp = os.path.join(exp2_dir, "stations_eval_rank_by_segment.csv")
    if not os.path.exists(fp):
        print("⚠️ Exp2 segment RMSE skipped: rank table missing.")
        return
    rows = _read_csv(fp)
    segs = ["all", "day", "night"]
    model_order = ["baseline_bilinear", "baseline_nearest", "mamba", "lstm", "unet", "transformer"]
    model_lbl = {
        "baseline_bilinear": "Bilinear",
        "baseline_nearest": "Nearest",
        "mamba": "Mamba",
        "lstm": "LSTM",
        "unet": "UNet",
        "transformer": "Transformer",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    for ax, seg in zip(axes, segs):
        rr = [r for r in rows if str(r.get("segment", "")).strip().lower() == seg]
        vals = {str(r.get("model", "")).strip().lower(): _safe_float(r.get("RMSE_model")) for r in rr}
        y = [vals.get(m, np.nan) for m in model_order]
        ax.bar([model_lbl[m] for m in model_order], y, color="#4c78a8")
        ax.set_title(seg.upper())
        ax.tick_params(axis="x", rotation=28)
        if seg == "all":
            ax.set_ylabel("RMSE vs stations")
    fig.suptitle("Exp2 Ground-Truth RMSE by Segment")
    _save_fig(fig, os.path.join(out_dir, "fig_v2_exp2_segment_rmse"))


def _plot_exp3_seq_ablation(exp3_dir: str, out_dir: str) -> None:
    agg_fp = os.path.join(exp3_dir, "fullframe_eval_aggregate.csv")
    cmp_fp = os.path.join(exp3_dir, "seq_compare.csv")
    if not os.path.exists(agg_fp):
        print("⚠️ Exp3 ablation skipped: aggregate table missing.")
        return
    rows = _read_csv(agg_fp)
    rows = sorted(rows, key=lambda r: str(r.get("model", "")))
    models = [str(r.get("model", "")) for r in rows]
    rmse = [_safe_float(r.get("rmse_mean")) for r in rows]
    mae = [_safe_float(r.get("mae_mean")) for r in rows]
    ssim = [_safe_float(r.get("ssim_mean")) for r in rows]
    x = np.arange(len(models))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    ax.bar(x - w / 2, rmse, width=w, label="RMSE", color="#1f77b4")
    ax.bar(x + w / 2, mae, width=w, label="MAE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Error Metrics")
    ax.legend()

    ax2 = axes[1]
    ax2.bar(x, ssim, color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.set_ylabel("SSIM")
    ax2.set_title("Structural Similarity")
    ax2.set_ylim(0, max(1.0, max(ssim) * 1.08))

    if os.path.exists(cmp_fp):
        crows = _read_csv(cmp_fp)
        if crows:
            cr = crows[0]
            txt = (
                f"Delta (seq12 - seq6): "
                f"RMSE={_safe_float(cr.get('rmse_delta_b_minus_a')):.3f}, "
                f"MAE={_safe_float(cr.get('mae_delta_b_minus_a')):.3f}, "
                f"SSIM={_safe_float(cr.get('ssim_delta_b_minus_a')):.3f}"
            )
            fig.text(0.5, 0.01, txt, ha="center", va="bottom", fontsize=9)
    fig.suptitle("Exp3 Bottleneck Ablation: Sequence Length Effect")
    _save_fig(fig, os.path.join(out_dir, "fig_v2_exp3_seq_ablation"))


def _extract_ts(path: str) -> Optional[str]:
    m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.npy$", os.path.basename(path))
    return m.group(1) if m else None


def _plot_cs1_storyboard(cs1_dir: str, out_dir: str) -> None:
    fig_dir = os.path.join(cs1_dir, "figures")
    m_files = glob.glob(os.path.join(fig_dir, "cs1_*_MAMBA_*_*.npy"))
    b_files = glob.glob(os.path.join(fig_dir, "cs1_*_BASELINE_NEAREST_*_*.npy"))
    map_m = {_extract_ts(p): p for p in m_files if _extract_ts(p)}
    map_b = {_extract_ts(p): p for p in b_files if _extract_ts(p)}
    common = sorted(set(map_m.keys()) & set(map_b.keys()))
    if len(common) < 3:
        print("⚠️ CS1 storyboard skipped: not enough common timestamps.")
        return
    picks = [common[0], common[len(common) // 2], common[-1]]

    arrays = []
    for ts in picks:
        arrays.extend([_load_npy(map_b[ts]), _load_npy(map_m[ts])])
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    vmin = float(np.nanpercentile(vals, 2.0))
    vmax = float(np.nanpercentile(vals, 98.0))

    deltas = []
    for ts in picks:
        deltas.append(_load_npy(map_m[ts]) - _load_npy(map_b[ts]))
    dmax = float(np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in deltas])), 99.0))

    fig, axes = plt.subplots(len(picks), 3, figsize=(11.0, 3.2 * len(picks)))
    for r, ts in enumerate(picks):
        bl = _load_npy(map_b[ts])
        mm = _load_npy(map_m[ts])
        dd = mm - bl
        im0 = axes[r, 0].imshow(bl, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        axes[r, 0].set_title("Baseline Nearest" if r == 0 else "")
        axes[r, 1].imshow(mm, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        axes[r, 1].set_title("Hybrid-Mamba" if r == 0 else "")
        im2 = axes[r, 2].imshow(dd, cmap="coolwarm", vmin=-dmax, vmax=dmax, origin="lower", interpolation="nearest")
        axes[r, 2].set_title("Mamba - Baseline" if r == 0 else "")
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        label = ts.replace("_", "-")
        label = label[:10] + " " + label[11:19]
        axes[r, 0].set_ylabel(label, rotation=90)

    fig.colorbar(im0, ax=axes[:, :2].ravel().tolist(), fraction=0.02, pad=0.01, label="Temperature (normalized)")
    fig.colorbar(im2, ax=axes[:, 2].ravel().tolist(), fraction=0.03, pad=0.02, label="Delta")
    fig.suptitle("Case Study 1: Heatwave Storyboard (Baseline vs Mamba)")
    _save_fig(fig, os.path.join(out_dir, "fig_v2_cs1_heatwave_storyboard"))


def _plot_cs2_robustness(cs2_dir: str, out_dir: str) -> None:
    fp = os.path.join(cs2_dir, "robustness_summary.csv")
    if not os.path.exists(fp):
        print("⚠️ CS2 robustness skipped: summary missing.")
        return
    rows = _read_csv(fp)

    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for r in rows:
        model = str(r.get("model_key", "")).strip().lower()
        eps = _safe_float(r.get("epsilon_K"))
        mad = _safe_float(r.get("pred_mean_abs_dev_vs_clean_C"))
        if model and np.isfinite(eps) and np.isfinite(mad):
            grouped[(model, eps)].append(mad)

    models = sorted({k[0] for k in grouped.keys()})
    eps_all = sorted({k[1] for k in grouped.keys()})
    if not models or not eps_all:
        print("⚠️ CS2 robustness skipped: no valid points.")
        return

    palette = {
        "mamba": "#1f77b4",
        "lstm": "#ff7f0e",
        "unet": "#2ca02c",
        "transformer": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for m in models:
        y = []
        yerr = []
        for e in eps_all:
            vals = np.asarray(grouped.get((m, e), []), dtype=float)
            if vals.size == 0:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                y.append(float(np.mean(vals)))
                yerr.append(float(np.std(vals, ddof=0)))
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
    _save_fig(fig, os.path.join(out_dir, "fig_v2_cs2_robustness_curve"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate reproducible narrative figures v2 from eval outputs.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--out-dir",
        default="figures/repro_v2",
        help="Output directory for v2 figures (PNG+PDF).",
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

    _plot_exp1_day_night_grid(exp1_dir, out_dir)
    _plot_exp1_mamba_vs_bilinear(exp1_dir, out_dir)
    _plot_exp2_station_heatmap(exp2_dir, out_dir)
    _plot_exp2_segments(exp2_dir, out_dir)
    _plot_exp3_seq_ablation(exp3_dir, out_dir)
    _plot_cs1_storyboard(cs1_dir, out_dir)
    _plot_cs2_robustness(cs2_dir, out_dir)

    print(f"✅ Narrative v2 figures generated in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

