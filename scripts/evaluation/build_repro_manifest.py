#!/usr/bin/env python3
"""
Build a reproducibility manifest with checksums for config, checkpoints and key outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: str) -> dict:
    rec = {"path": path, "exists": os.path.exists(path)}
    if rec["exists"]:
        st = os.stat(path)
        rec["size_bytes"] = st.st_size
        rec["mtime_utc"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        rec["sha256"] = _sha256(path)
    return rec


def _iter_checkpoint_paths(checkpoints_cfg: dict) -> Iterable[str]:
    for _, model_map in checkpoints_cfg.items():
        if not isinstance(model_map, dict):
            continue
        for model_key, value in model_map.items():
            if model_key == "baselines":
                continue
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for _, p in value.items():
                    if isinstance(p, str):
                        yield p


def _read_ckpt_manifest(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ckpt = str(row.get("checkpoint", "")).strip()
            if not ckpt or ckpt.startswith("__"):
                continue
            out.append({"model": row.get("model", ""), "checkpoint": ckpt})
    return out


def _git_head(root: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _collect_stage_outputs(root: str, narrative: dict) -> Dict[str, List[str]]:
    def stage_dir(name: str) -> str:
        return _resolve_path(root, narrative.get(name, {}).get("output_dir", ""))

    exp1 = stage_dir("exp1")
    exp2 = stage_dir("exp2")
    exp3 = stage_dir("exp3")
    cs1 = stage_dir("case_study_1")
    cs2 = stage_dir("case_study_2")
    dual = stage_dir("exp2_dual_protocol")

    return {
        "exp1": [
            os.path.join(exp1, "metrics_raw.csv"),
            os.path.join(exp1, "metrics_aggregate.csv"),
            os.path.join(exp1, "metrics_aggregate_ci.csv"),
            os.path.join(exp1, "metrics_by_model_seed.csv"),
        ],
        "exp2": [
            os.path.join(exp2, "stations_eval_models_summary.csv"),
            os.path.join(exp2, "stations_eval_rank_by_segment.csv"),
            os.path.join(exp2, "stations_eval_per_station_all_models.csv"),
            os.path.join(exp2, "checkpoints_used.csv"),
        ],
        "exp2_dual_protocol": [
            os.path.join(dual, "exp2_protocol_comparison_by_model_segment.csv"),
            os.path.join(dual, "exp2_protocol_rankings.csv"),
            os.path.join(dual, "report_exp2_dual_protocol.md"),
        ],
        "exp3": [
            os.path.join(exp3, "fullframe_eval_raw.csv"),
            os.path.join(exp3, "fullframe_eval_aggregate.csv"),
            os.path.join(exp3, "ranking_stability_vs_exp1.csv"),
            os.path.join(exp3, "seq_compare.csv"),
        ],
        "cs1": [
            os.path.join(cs1, "metrics_raw.csv"),
            os.path.join(cs1, "metrics_aggregate.csv"),
        ],
        "cs2": [
            os.path.join(cs2, "robustness_summary.csv"),
            os.path.join(cs2, "cs2_rank_stability.csv"),
        ],
        "global_reports": [
            os.path.join(root, "experiments", "eval_outputs", "report_master_narrative.md"),
            os.path.join(root, "experiments", "eval_outputs", "report_paper_ready.md"),
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build reproducibility checksum manifest.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--out",
        default="experiments/eval_outputs/repro_manifest.json",
        help="Output manifest JSON path.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required stage outputs are missing.",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)

    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)
    out_path = _resolve_path(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    manifest: dict = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": root,
        "git_commit": _git_head(root),
        "frozen_training": bool(cfg.get("frozen_training", False)),
        "config": _file_record(cfg_path),
        "checkpoints": {},
        "stage_outputs": {},
        "notes": [
            "Checkpoint hashes are taken from config-declared paths and explicit checkpoint manifests when present.",
            "Output file hashes are included for key CSV/MD artifacts used in publication workflow.",
        ],
    }

    # Config-declared checkpoints
    ckpt_cfg = cfg.get("checkpoints", {})
    ckpt_paths = sorted(set(_iter_checkpoint_paths(ckpt_cfg)))
    manifest["checkpoints"]["from_config"] = [
        _file_record(_resolve_path(root, p)) for p in ckpt_paths
    ]

    # Exp3 direct comparison checkpoints (outside checkpoint registry)
    exp3_comparisons = cfg.get("narrative", {}).get("exp3", {}).get("comparisons", [])
    exp3_ckpts = []
    for c in exp3_comparisons:
        p = _resolve_path(root, str(c.get("checkpoint", "")).strip())
        if p:
            exp3_ckpts.append({"name": c.get("name", ""), **_file_record(p)})
    manifest["checkpoints"]["exp3_comparisons"] = exp3_ckpts

    # Explicit checkpoint manifests from stage outputs
    narrative = cfg.get("narrative", {})
    exp2_dir = _resolve_path(root, narrative.get("exp2", {}).get("output_dir", ""))
    ckpt_used_csv = os.path.join(exp2_dir, "checkpoints_used.csv")
    ckpt_used = _read_ckpt_manifest(ckpt_used_csv)
    manifest["checkpoints"]["from_exp2_checkpoints_used"] = [
        {"model": r["model"], **_file_record(_resolve_path(root, r["checkpoint"]))}
        for r in ckpt_used
    ]

    # Stage output records
    outputs = _collect_stage_outputs(root, narrative)
    missing_required: List[str] = []
    required_groups = {"exp1", "exp2", "exp3", "cs1", "cs2"}
    for group, files in outputs.items():
        records = [_file_record(p) for p in files]
        manifest["stage_outputs"][group] = records
        if group in required_groups and any(not r["exists"] for r in records):
            for r in records:
                if not r["exists"]:
                    missing_required.append(r["path"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(out_path)
    if args.strict and missing_required:
        print("Missing required outputs:")
        for p in missing_required:
            print(f"- {p}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

