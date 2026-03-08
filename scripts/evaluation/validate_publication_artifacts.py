#!/usr/bin/env python3
"""
Publication gate: validate required post-training artifacts for paper release.
Fails with non-zero exit code when required artifacts are missing or malformed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
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
    return os.path.abspath(os.path.join(root, value))


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _is_finite(v) -> bool:
    return math.isfinite(_safe_float(v))


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_file_exists(path: str, errors: List[str]) -> None:
    if not os.path.exists(path):
        errors.append(f"Missing file: {path}")
        return
    if os.path.isfile(path) and os.path.getsize(path) == 0:
        errors.append(f"Empty file: {path}")


def _check_csv_columns(path: str, required_cols: List[str], errors: List[str]) -> List[dict]:
    try:
        rows = _read_csv(path)
    except Exception as e:
        errors.append(f"Unreadable CSV ({path}): {e}")
        return []
    if not rows:
        errors.append(f"CSV has no rows: {path}")
        return []
    cols = set(rows[0].keys())
    missing = [c for c in required_cols if c not in cols]
    if missing:
        errors.append(f"CSV missing columns ({path}): {', '.join(missing)}")
    return rows


def _check_numeric_rows(
    path: str,
    rows: List[dict],
    numeric_cols: List[str],
    errors: List[str],
) -> None:
    for i, row in enumerate(rows, 2):
        for c in numeric_cols:
            if c not in row:
                continue
            if not _is_finite(row.get(c)):
                errors.append(f"Non-finite value in {path}:{i} column '{c}' -> {row.get(c)}")
                return


def _default_artifact_map(root: str, narrative: dict) -> Dict[str, Dict[str, List[str] | str]]:
    def stage_dir(name: str) -> str:
        return _resolve_path(root, narrative.get(name, {}).get("output_dir", ""))

    exp1 = stage_dir("exp1")
    exp2 = stage_dir("exp2")
    exp3 = stage_dir("exp3")
    cs1 = stage_dir("case_study_1")
    cs2 = stage_dir("case_study_2")
    eval_root = os.path.join(root, "experiments", "eval_outputs")

    return {
        "exp1": {
            "path": os.path.join(exp1, "metrics_aggregate.csv"),
            "required_cols": ["model", "rmse_mean", "mae_mean", "ssim_mean"],
            "numeric_cols": ["rmse_mean", "mae_mean", "ssim_mean"],
        },
        "exp2": {
            "path": os.path.join(exp2, "stations_eval_rank_by_segment.csv"),
            "required_cols": ["segment", "rank", "model", "RMSE_model", "MAE_model", "Corr_model"],
            "numeric_cols": ["rank", "RMSE_model", "MAE_model", "Corr_model"],
        },
        "exp3": {
            "path": os.path.join(exp3, "fullframe_eval_aggregate.csv"),
            "required_cols": ["model", "n", "rmse_mean", "mae_mean", "ssim_mean"],
            "numeric_cols": ["n", "rmse_mean", "mae_mean", "ssim_mean"],
        },
        "cs1": {
            "path": os.path.join(cs1, "metrics_aggregate.csv"),
            "required_cols": ["model", "rmse_mean", "mae_mean", "ssim_mean"],
            "numeric_cols": ["rmse_mean", "mae_mean", "ssim_mean"],
        },
        "cs2": {
            "path": os.path.join(cs2, "robustness_summary.csv"),
            "required_cols": [
                "model_key",
                "epsilon_K",
                "n_trials",
                "pred_mean_abs_dev_vs_clean_C",
                "pred_rmse_vs_clean_C",
                "pred_max_range_C",
            ],
            "numeric_cols": [
                "epsilon_K",
                "n_trials",
                "pred_mean_abs_dev_vs_clean_C",
                "pred_rmse_vs_clean_C",
                "pred_max_range_C",
            ],
        },
        "reports": {
            "path": [
                os.path.join(eval_root, "report_master_narrative.md"),
                os.path.join(eval_root, "report_paper_ready.md"),
                os.path.join(eval_root, "repro_manifest.json"),
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate publication-ready post-training artifacts.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--out",
        default="experiments/eval_outputs/publication_gate_report.json",
        help="Output JSON report path.",
    )
    ap.add_argument("--strict-exp3-n", type=int, default=1, help="Minimum n required in Exp3 aggregate.")
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)
    out_path = _resolve_path(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    narrative = cfg.get("narrative", {})
    amap = _default_artifact_map(root, narrative)

    errors: List[str] = []
    warnings: List[str] = []
    checked_files: List[str] = []
    summary: Dict[str, dict] = {}

    # Stage CSV checks
    for stage in ["exp1", "exp2", "exp3", "cs1", "cs2"]:
        spec = amap[stage]
        path = str(spec["path"])
        checked_files.append(path)
        _check_file_exists(path, errors)
        if errors and any(path in e for e in errors):
            continue
        rows = _check_csv_columns(path, list(spec["required_cols"]), errors)
        if rows:
            _check_numeric_rows(path, rows, list(spec["numeric_cols"]), errors)
        summary[stage] = {
            "path": path,
            "rows": len(rows),
        }
        if stage == "exp3" and rows:
            min_n = min(_safe_float(r.get("n")) for r in rows)
            if min_n < float(args.strict_exp3_n):
                errors.append(
                    f"Exp3 n below threshold ({min_n} < {args.strict_exp3_n}) in {path}."
                )
            if min_n <= 1:
                warnings.append("Exp3 has n=1 per model; statistical confidence is limited.")

    # Reports / manifest
    for p in amap["reports"]["path"]:  # type: ignore[index]
        path = str(p)
        checked_files.append(path)
        _check_file_exists(path, errors)

    # Optional dual-protocol warning
    dual_dir = _resolve_path(root, narrative.get("exp2_dual_protocol", {}).get("output_dir", ""))
    dual_cmp = os.path.join(dual_dir, "exp2_protocol_comparison_by_model_segment.csv")
    if not os.path.exists(dual_cmp):
        warnings.append("Exp2 dual-protocol comparison not found (optional).")

    report = {
        "gate_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg_path,
        "project_root": root,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checked_files": checked_files,
        "summary": summary,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(out_path)
    if errors:
        print("Gate failed with errors:")
        for e in errors:
            print(f"- {e}")
        return 2

    print("Gate passed.")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

