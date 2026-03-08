#!/usr/bin/env python3
"""
Consolidate Case Study 2 artifacts:
- cooling metrics from cs2_cooling_*.npy and cs2_persistence_*.npy
- full-frame metrics from Experiment 3 outputs
- rank stability vs Case Study 1
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


def _safe_float(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _model_from_upper(name: str) -> str:
    n = name.strip().upper()
    if n == "UNET":
        return "unet"
    if n == "LSTM":
        return "lstm"
    if n == "TRANSFORMER":
        return "transformer"
    if n == "MAMBA":
        return "mamba"
    return n.lower()


def _load_exp3_metrics(exp3_dir: Optional[str]) -> List[Dict]:
    if not exp3_dir:
        return []

    agg_path = os.path.join(exp3_dir, "fullframe_eval_aggregate.csv")
    raw_path = os.path.join(exp3_dir, "fullframe_eval_raw.csv")

    rows: List[Dict] = []
    if os.path.exists(agg_path):
        with open(agg_path, newline="", encoding="utf-8") as f:
            agg = list(csv.DictReader(f))
        for r in agg:
            model = r.get("model", "").strip().lower()
            if not model:
                continue
            rmse = _safe_float(r.get("rmse_mean", ""))
            mae = _safe_float(r.get("mae_mean", ""))
            ssim = _safe_float(r.get("ssim_mean", ""))
            if np.isnan(rmse) or np.isnan(mae):
                continue
            rows.append(
                {
                    "model": model,
                    "n": int(float(r.get("n", "0"))) if r.get("n") else 0,
                    "mae": mae,
                    "rmse": rmse,
                    "ssim": ssim,
                }
            )

    if rows:
        rows.sort(key=lambda x: x["rmse"])
        return rows

    if not os.path.exists(raw_path):
        return []

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"mae": [], "rmse": [], "ssim": []})
    with open(raw_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            model = r.get("model", "").strip().lower()
            if not model:
                continue
            mae = _safe_float(r.get("mae", ""))
            rmse = _safe_float(r.get("rmse", ""))
            ssim = _safe_float(r.get("ssim", ""))
            if np.isnan(mae) or np.isnan(rmse):
                continue
            grouped[model]["mae"].append(mae)
            grouped[model]["rmse"].append(rmse)
            if not np.isnan(ssim):
                grouped[model]["ssim"].append(ssim)

    out: List[Dict] = []
    for model, vals in grouped.items():
        if not vals["rmse"]:
            continue
        out.append(
            {
                "model": model,
                "n": len(vals["rmse"]),
                "mae": float(np.mean(vals["mae"])),
                "rmse": float(np.mean(vals["rmse"])),
                "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else float("nan"),
            }
        )
    out.sort(key=lambda x: x["rmse"])
    return out


def _load_cs1_rank(cs1_agg_csv: Optional[str]) -> Dict[str, int]:
    if not cs1_agg_csv or not os.path.exists(cs1_agg_csv):
        return {}

    rows = []
    with open(cs1_agg_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            variant = r.get("variant", "").strip().lower()
            ssim = _safe_float(r.get("ssim_mean", ""))
            mae = _safe_float(r.get("mae_mean", ""))
            if np.isnan(ssim):
                continue
            rows.append((variant, ssim, mae))
    rows.sort(key=lambda x: (-x[1], x[2]))

    rank = {}
    for i, (variant, _, _) in enumerate(rows, 1):
        key = variant
        if variant == "mamba_seq6":
            key = "mamba"
        rank[key] = i
    return rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--case2-dir", required=True, help="Explicit Case Study 2 directory.")
    parser.add_argument("--exp3-dir", required=True, help="Explicit Experiment 3 directory.")
    parser.add_argument("--cs1-agg-csv", default="", help="Explicit Case Study 1 aggregate CSV (optional).")
    args = parser.parse_args()

    root = os.path.abspath(args.project_root)
    case2_dir = args.case2_dir
    exp3_dir = args.exp3_dir
    cs1_agg = args.cs1_agg_csv

    if not os.path.isabs(case2_dir):
        case2_dir = os.path.join(root, case2_dir)
    if not os.path.isabs(exp3_dir):
        exp3_dir = os.path.join(root, exp3_dir)
    if cs1_agg and not os.path.isabs(cs1_agg):
        cs1_agg = os.path.join(root, cs1_agg)

    fig_dir = os.path.join(case2_dir, "figures")
    if not os.path.isdir(fig_dir):
        raise SystemExit(f"Missing figures directory: {fig_dir}")

    cooling_pair_rows = []
    cooling_means: Dict[str, List[np.ndarray]] = defaultdict(list)

    for fp in sorted(glob.glob(os.path.join(fig_dir, "cs2_cooling_*_*.npy"))):
        bn = os.path.basename(fp).replace(".npy", "")
        # pattern: cs2_cooling_MODEL_PAIR
        parts = bn.split("_")
        if len(parts) < 4:
            continue
        model_u = parts[2]
        pair = "_".join(parts[3:])
        if model_u == "mean":
            continue
        model = _model_from_upper(model_u)
        arr = np.load(fp)
        cooling_means[model].append(arr)
        cooling_pair_rows.append(
            {
                "model": model,
                "pair": pair,
                "mean_delta_t": float(np.nanmean(arr)),
                "p10_delta_t": float(np.nanpercentile(arr, 10)),
                "p90_delta_t": float(np.nanpercentile(arr, 90)),
                "min_delta_t": float(np.nanmin(arr)),
                "max_delta_t": float(np.nanmax(arr)),
            }
        )

    cooling_summary_rows = []
    for model, arrs in cooling_means.items():
        stack = np.stack(arrs, axis=0)
        mean_map = np.nanmean(stack, axis=0)
        p_path = os.path.join(fig_dir, f"cs2_persistence_{model.upper()}.npy")
        persistent_pct = float("nan")
        if os.path.exists(p_path):
            p = np.load(p_path)
            persistent_pct = float((p >= 3).mean() * 100.0)
        cooling_summary_rows.append(
            {
                "model": model,
                "n_pairs": int(stack.shape[0]),
                "mean_delta_t": float(np.nanmean(mean_map)),
                "p10_delta_t": float(np.nanpercentile(mean_map, 10)),
                "p90_delta_t": float(np.nanpercentile(mean_map, 90)),
                "persistent_hotspots_pct_ge3": persistent_pct,
            }
        )
    cooling_summary_rows.sort(key=lambda x: x["mean_delta_t"])  # more negative cooling first

    exp3_rows = _load_exp3_metrics(exp3_dir)
    cs1_rank = _load_cs1_rank(cs1_agg if os.path.exists(cs1_agg) else None)

    exp3_rank = {r["model"]: i + 1 for i, r in enumerate(exp3_rows)}
    stability_rows = []
    base_models = {r["model"] for r in cooling_summary_rows}
    if "mamba_seq12" in exp3_rank or "mamba_seq12" in cs1_rank:
        base_models.add("mamba_seq12")
    for model in sorted(base_models):
        r1 = cs1_rank.get(model)
        r2 = exp3_rank.get(model)
        delta = (r2 - r1) if (r1 is not None and r2 is not None) else ""
        stability_rows.append(
            {
                "model": model,
                "rank_cs1": "" if r1 is None else r1,
                "rank_cs2_fullframe": "" if r2 is None else r2,
                "delta_rank": delta,
            }
        )

    cooling_pair_csv = os.path.join(case2_dir, "cs2_cooling_by_pair.csv")
    cooling_summary_csv = os.path.join(case2_dir, "cs2_cooling_summary.csv")
    fullframe_csv = os.path.join(case2_dir, "cs2_fullframe_eval_summary.csv")
    stability_csv = os.path.join(case2_dir, "cs2_rank_stability_vs_cs1.csv")
    report_md = os.path.join(case2_dir, "report_casestudy2.md")

    with open(cooling_pair_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "pair", "mean_delta_t", "p10_delta_t", "p90_delta_t", "min_delta_t", "max_delta_t"],
        )
        w.writeheader()
        w.writerows(cooling_pair_rows)

    with open(cooling_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "n_pairs", "mean_delta_t", "p10_delta_t", "p90_delta_t", "persistent_hotspots_pct_ge3"],
        )
        w.writeheader()
        w.writerows(cooling_summary_rows)

    with open(fullframe_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n", "mae", "rmse", "ssim"])
        w.writeheader()
        w.writerows(exp3_rows)

    with open(stability_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "rank_cs1", "rank_cs2_fullframe", "delta_rank"])
        w.writeheader()
        w.writerows(stability_rows)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Case Study 2 (Night Cooling & Persistence)\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Case2 dir: `{case2_dir}`\n")
        f.write(f"- Exp3 dir: `{exp3_dir or 'not found'}`\n")
        f.write(f"- CS1 aggregate: `{cs1_agg if os.path.exists(cs1_agg) else 'not found'}`\n\n")

        f.write("## Full-frame evaluation (from Experiment 3)\n\n")
        if exp3_rows:
            f.write("| rank | model | RMSE | MAE | SSIM |\n")
            f.write("|---:|---|---:|---:|---:|\n")
            for i, r in enumerate(exp3_rows, 1):
                ssim_txt = "" if np.isnan(r["ssim"]) else f"{r['ssim']:.6f}"
                f.write(f"| {i} | {r['model']} | {r['rmse']:.6f} | {r['mae']:.6f} | {ssim_txt} |\n")
        else:
            f.write("No full-frame metrics found.\n")

        f.write("\n## Nighttime cooling summary\n\n")
        if cooling_summary_rows:
            f.write("| model | n_pairs | mean ΔT | P10 ΔT | P90 ΔT | persistent hotspots (>=3 nights) |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for r in cooling_summary_rows:
                f.write(
                    f"| {r['model']} | {r['n_pairs']} | {r['mean_delta_t']:.6f} | "
                    f"{r['p10_delta_t']:.6f} | {r['p90_delta_t']:.6f} | {r['persistent_hotspots_pct_ge3']:.3f}% |\n"
                )
        else:
            f.write("No cooling maps found.\n")

        f.write("\n## Ranking stability vs Case Study 1\n\n")
        if stability_rows:
            f.write("| model | rank_cs1 | rank_cs2_fullframe | delta_rank |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in stability_rows:
                f.write(
                    f"| {r['model']} | {r['rank_cs1']} | {r['rank_cs2_fullframe']} | {r['delta_rank']} |\n"
                )
        else:
            f.write("No ranking data available.\n")

    print(report_md)
    print(cooling_summary_csv)
    print(fullframe_csv)
    print(stability_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
