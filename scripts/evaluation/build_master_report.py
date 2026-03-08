#!/usr/bin/env python3
"""
Build a single markdown report for Exp1/Exp2/Exp3/CS1/CS2 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


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
    return os.path.join(root, value)


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _metric(row: dict, *keys: str) -> float:
    for k in keys:
        if k in row and str(row.get(k, "")).strip() != "":
            return _safe_float(row.get(k))
    return float("nan")


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best_by_rmse(rows: List[dict], segment_key: str = "segment", seg_val: str = "all") -> List[dict]:
    subset = [r for r in rows if str(r.get(segment_key, "all")) == seg_val]
    return sorted(
        subset,
        key=lambda r: (
            _safe_float(r.get("RMSE_model"), 1e12),
            _safe_float(r.get("MAE_model"), 1e12),
            -_safe_float(r.get("Corr_model"), -1e12),
        ),
    )


def _section_status(path: str) -> Tuple[bool, str]:
    if path and os.path.exists(path):
        return True, path
    return False, path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build consolidated master markdown report.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--out",
        default="experiments/eval_outputs/report_master_narrative.md",
        help="Output markdown path.",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)
    out_md = _resolve_path(root, args.out)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    n = cfg.get("narrative", {})
    exp1_dir = _resolve_path(root, n.get("exp1", {}).get("output_dir", ""))
    exp2_dir = _resolve_path(root, n.get("exp2", {}).get("output_dir", ""))
    exp3_dir = _resolve_path(root, n.get("exp3", {}).get("output_dir", ""))
    cs1_dir = _resolve_path(root, n.get("case_study_1", {}).get("output_dir", ""))
    cs2_dir = _resolve_path(root, n.get("case_study_2", {}).get("output_dir", ""))
    exp2_dual_dir = _resolve_path(root, n.get("exp2_dual_protocol", {}).get("output_dir", ""))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Master Evaluation Report (Post-Training)\n\n")
        f.write(f"- Config: `{cfg_path}`\n")
        f.write(f"- Project root: `{root}`\n\n")

        # Exp1
        f.write("## Experiment 1 — Spatial Performance\n\n")
        exp1_csv = os.path.join(exp1_dir, "metrics_aggregate.csv")
        ok, _ = _section_status(exp1_csv)
        if ok:
            rows = _read_csv(exp1_csv)
            rows = sorted(
                rows,
                key=lambda r: (
                    _metric(r, "rmse_mean", "RMSE_model"),
                    _metric(r, "mae_mean", "MAE_model"),
                ),
            )
            f.write(f"- Output dir: `{exp1_dir}`\n\n")
            f.write("| Rank | Model | RMSE | MAE | SSIM |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for i, r in enumerate(rows, 1):
                f.write(
                    f"| {i} | {r.get('model','')} | {_metric(r, 'rmse_mean', 'RMSE_model'):.4f} | "
                    f"{_metric(r, 'mae_mean', 'MAE_model'):.4f} | {_metric(r, 'ssim_mean', 'SSIM_model'):.4f} |\n"
                )
            f.write("\n")
        else:
            f.write(f"- Missing: `{exp1_csv}`\n\n")

        # Exp2
        f.write("## Experiment 2 — Station Ground-Truth Validation\n\n")
        exp2_rank = os.path.join(exp2_dir, "stations_eval_rank_by_segment.csv")
        ok, _ = _section_status(exp2_rank)
        if ok:
            rows = _read_csv(exp2_rank)
            f.write(f"- Output dir: `{exp2_dir}`\n\n")
            for seg in ["all", "day", "night"]:
                seg_rows = [r for r in rows if str(r.get("segment", "")) == seg]
                if not seg_rows:
                    continue
                seg_rows = sorted(seg_rows, key=lambda r: int(float(r.get("rank", "9999"))))
                f.write(f"### Segment: `{seg}`\n\n")
                f.write("| Rank | Model | RMSE | MAE | Corr |\n")
                f.write("|---|---|---:|---:|---:|\n")
                for r in seg_rows:
                    f.write(
                        f"| {r.get('rank','')} | {r.get('model','')} | "
                        f"{_safe_float(r.get('RMSE_model')):.4f} | {_safe_float(r.get('MAE_model')):.4f} | "
                        f"{_safe_float(r.get('Corr_model')):.4f} |\n"
                    )
                f.write("\n")
        else:
            f.write(f"- Missing: `{exp2_rank}`\n\n")

        # Exp2 dual protocol (optional)
        f.write("## Experiment 2 — Dual Protocol Comparison (Optional)\n\n")
        dual_cmp = os.path.join(exp2_dual_dir, "exp2_protocol_comparison_by_model_segment.csv")
        dual_rank = os.path.join(exp2_dual_dir, "exp2_protocol_rankings.csv")
        if exp2_dual_dir and os.path.exists(dual_cmp) and os.path.exists(dual_rank):
            cmp_rows = _read_csv(dual_cmp)
            f.write(f"- Output dir: `{exp2_dual_dir}`\n\n")
            f.write("| Compared protocol | Segment | Model | ΔRMSE | ΔMAE | ΔCorr |\n")
            f.write("|---|---|---|---:|---:|---:|\n")
            for r in sorted(cmp_rows, key=lambda x: (x["compared_protocol"], x["segment"], x["model"])):
                f.write(
                    f"| {r.get('compared_protocol','')} | {r.get('segment','')} | {r.get('model','')} | "
                    f"{_safe_float(r.get('delta_RMSE_cmp_minus_ref')):.4f} | "
                    f"{_safe_float(r.get('delta_MAE_cmp_minus_ref')):.4f} | "
                    f"{_safe_float(r.get('delta_Corr_cmp_minus_ref')):.4f} |\n"
                )
            f.write("\n")
        else:
            if exp2_dual_dir:
                f.write(f"- Missing dual-protocol outputs in `{exp2_dual_dir}`\n\n")
            else:
                f.write("- Not configured in eval_config (`narrative.exp2_dual_protocol`).\n\n")

        # Exp3
        f.write("## Experiment 3 — Bottleneck Ablation\n\n")
        exp3_csv = os.path.join(exp3_dir, "fullframe_eval_aggregate.csv")
        ok, _ = _section_status(exp3_csv)
        if ok:
            rows = _read_csv(exp3_csv)
            rows = sorted(
                rows,
                key=lambda r: (
                    _metric(r, "rmse_mean", "RMSE_model"),
                    _metric(r, "mae_mean", "MAE_model"),
                ),
            )
            f.write(f"- Output dir: `{exp3_dir}`\n\n")
            f.write("| Rank | Model | RMSE | MAE | SSIM |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for i, r in enumerate(rows, 1):
                f.write(
                    f"| {i} | {r.get('model','')} | {_metric(r, 'rmse_mean', 'RMSE_model'):.4f} | "
                    f"{_metric(r, 'mae_mean', 'MAE_model'):.4f} | {_metric(r, 'ssim_mean', 'SSIM_model'):.4f} |\n"
                )
            f.write("\n")
        else:
            f.write(f"- Missing: `{exp3_csv}`\n\n")

        # Case Study 1
        f.write("## Case Study 1 — Extreme Heatwave\n\n")
        cs1_csv = os.path.join(cs1_dir, "metrics_aggregate.csv")
        ok, _ = _section_status(cs1_csv)
        if ok:
            rows = _read_csv(cs1_csv)
            rows = sorted(
                rows,
                key=lambda r: (
                    _metric(r, "rmse_mean", "RMSE_model"),
                    _metric(r, "mae_mean", "MAE_model"),
                ),
            )
            f.write(f"- Output dir: `{cs1_dir}`\n\n")
            f.write("| Rank | Model | RMSE | MAE | SSIM |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for i, r in enumerate(rows, 1):
                f.write(
                    f"| {i} | {r.get('model','')} | {_metric(r, 'rmse_mean', 'RMSE_model'):.4f} | "
                    f"{_metric(r, 'mae_mean', 'MAE_model'):.4f} | {_metric(r, 'ssim_mean', 'SSIM_model'):.4f} |\n"
                )
            f.write("\n")
        else:
            f.write(f"- Missing: `{cs1_csv}`\n\n")

        # Case Study 2
        f.write("## Case Study 2 — Robustness\n\n")
        cs2_csv = os.path.join(cs2_dir, "robustness_summary.csv")
        ok, _ = _section_status(cs2_csv)
        if ok:
            rows = _read_csv(cs2_csv)
            f.write(f"- Output dir: `{cs2_dir}`\n\n")
            f.write("| Model | Epsilon (K) | mean_abs_dev_vs_clean_C | max_range_C |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in rows:
                f.write(
                    f"| {r.get('model_key','')} | {_safe_float(r.get('epsilon_K')):.4f} | "
                    f"{_safe_float(r.get('pred_mean_abs_dev_vs_clean_C')):.4f} | "
                    f"{_safe_float(r.get('pred_max_range_C')):.4f} |\n"
                )
            f.write("\n")
        else:
            f.write(f"- Missing: `{cs2_csv}`\n\n")

    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
