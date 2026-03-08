#!/usr/bin/env python3
"""
Run Experiment 2 with two deterministic protocols and compare outcomes.

Typical use:
  ./.venv/bin/python scripts/evaluation/run_exp2_dual_protocol.py \
    --config config/eval_config.yaml \
    --reuse-existing
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
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


def _dump_yaml(path: str, obj: dict) -> None:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(f"PyYAML is required to write temporary protocol configs ({e}).")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _resolve_path(root: str, value: str | None) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run(cmd: List[str], cwd: str) -> None:
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _read_summary_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _consolidate_existing_exp2(out_dir: str, models: List[str]) -> None:
    summary_rows: List[dict] = []
    per_station_rows: List[dict] = []
    rank_rows: List[dict] = []

    for model in models:
        model_out = os.path.join(out_dir, model)
        summary_seg = os.path.join(model_out, "stations_eval_summary_by_segment.csv")
        summary_fallback = os.path.join(model_out, "stations_eval_summary.csv")
        summary_path = summary_seg if os.path.exists(summary_seg) else summary_fallback
        if os.path.exists(summary_path):
            with open(summary_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rr = dict(row)
                    rr["model"] = model
                    rr["checkpoint"] = "__from_existing_run__"
                    summary_rows.append(rr)

        ps_path = os.path.join(model_out, "stations_eval_per_station.csv")
        pss_path = os.path.join(model_out, "stations_eval_per_station_by_segment.csv")
        for p in [ps_path, pss_path]:
            if os.path.exists(p):
                with open(p, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        rr = dict(row)
                        rr["model"] = model
                        rr["checkpoint"] = "__from_existing_run__"
                        per_station_rows.append(rr)

    if not summary_rows:
        return

    _write_csv(
        os.path.join(out_dir, "stations_eval_models_summary.csv"),
        summary_rows,
        list(summary_rows[0].keys()),
    )
    if per_station_rows:
        _write_csv(
            os.path.join(out_dir, "stations_eval_per_station_all_models.csv"),
            per_station_rows,
            list(per_station_rows[0].keys()),
        )

    grouped: Dict[str, List[dict]] = {}
    for r in summary_rows:
        seg = str(r.get("segment", "all"))
        grouped.setdefault(seg, []).append(r)
    for seg, vals in grouped.items():
        vals_sorted = sorted(
            vals,
            key=lambda x: (
                _safe_float(x.get("RMSE_model"), 1e12),
                _safe_float(x.get("MAE_model"), 1e12),
                -_safe_float(x.get("Corr_model"), -1e12),
            ),
        )
        for rank, row in enumerate(vals_sorted, 1):
            rank_rows.append(
                {
                    "segment": seg,
                    "rank": rank,
                    "model": row.get("model", ""),
                    "RMSE_model": row.get("RMSE_model", ""),
                    "MAE_model": row.get("MAE_model", ""),
                    "Corr_model": row.get("Corr_model", ""),
                    "N": row.get("N", ""),
                    "checkpoint": row.get("checkpoint", ""),
                }
            )
    if rank_rows:
        _write_csv(
            os.path.join(out_dir, "stations_eval_rank_by_segment.csv"),
            rank_rows,
            list(rank_rows[0].keys()),
        )


def _protocol_overrides(raw: dict) -> Tuple[str, dict]:
    pid = str(raw.get("id", "")).strip()
    if not pid:
        raise SystemExit("Each protocol in narrative.exp2_dual_protocol.protocols must define 'id'.")
    ov = dict(raw.get("overrides", {}))
    for k, v in raw.items():
        if k not in {"id", "name", "overrides"}:
            ov[k] = v
    return pid, ov


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Exp2 dual protocol comparison.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip protocol run when stations_eval_models_summary.csv already exists.",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)

    narrative = cfg.get("narrative", {})
    exp2_base = copy.deepcopy(narrative.get("exp2", {}))
    if not exp2_base:
        raise SystemExit("Config is missing narrative.exp2 (base stage).")

    dual = narrative.get("exp2_dual_protocol", {})
    protocols_raw = dual.get("protocols", [])
    if len(protocols_raw) < 2:
        raise SystemExit("narrative.exp2_dual_protocol.protocols must include at least 2 entries.")

    compare_segments = [str(s) for s in dual.get("segments", ["all", "day", "night"])]
    compare_out_dir = _resolve_path(root, dual.get("output_dir", "experiments/eval_outputs/exp2_dual_protocol"))
    _ensure_dir(compare_out_dir)

    python_bin = cfg.get("python_bin") or sys.executable
    protocol_outputs: Dict[str, str] = {}
    protocol_overrides: Dict[str, dict] = {}
    protocol_order: List[str] = []

    for raw in protocols_raw:
        pid, ov = _protocol_overrides(raw)
        protocol_order.append(pid)
        protocol_overrides[pid] = ov
        exp2_stage = copy.deepcopy(exp2_base)
        exp2_stage.update(ov)
        if not exp2_stage.get("run_id"):
            exp2_stage["run_id"] = f"exp2_{pid}"
        if not exp2_stage.get("output_dir"):
            exp2_stage["output_dir"] = os.path.join(compare_out_dir, exp2_stage["run_id"])
        out_dir = _resolve_path(root, exp2_stage["output_dir"])
        protocol_outputs[pid] = out_dir
        summary_csv = os.path.join(out_dir, "stations_eval_models_summary.csv")

        if args.reuse_existing:
            if not os.path.exists(summary_csv):
                _consolidate_existing_exp2(out_dir, [str(m) for m in exp2_stage.get("models", [])])
            if os.path.exists(summary_csv):
                print(f"✓ Reusing existing Exp2 protocol output for '{pid}': {out_dir}")
                continue

        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.setdefault("narrative", {})
        cfg_tmp["narrative"]["exp2"] = exp2_stage

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"exp2_{pid}_", delete=False
        ) as tf:
            tmp_cfg = tf.name
        _dump_yaml(tmp_cfg, cfg_tmp)
        try:
            _run(
                [
                    python_bin,
                    "scripts/evaluation/run_narrative_eval.py",
                    "--config", tmp_cfg,
                    "--stages", "exp2",
                ],
                cwd=root,
            )
        finally:
            try:
                os.unlink(tmp_cfg)
            except OSError:
                pass

    # Build cross-protocol comparison using protocol[0] as reference.
    ref = protocol_order[0]
    ref_rows = _read_summary_rows(os.path.join(protocol_outputs[ref], "stations_eval_models_summary.csv"))
    ref_map = {(r.get("segment", ""), r.get("model", "")): r for r in ref_rows}

    comp_rows: List[dict] = []
    ranking_rows: List[dict] = []

    for pid in protocol_order:
        rows = _read_summary_rows(os.path.join(protocol_outputs[pid], "stations_eval_models_summary.csv"))
        for seg in compare_segments:
            seg_rows = [r for r in rows if str(r.get("segment", "")) == seg]
            seg_rows = sorted(
                seg_rows,
                key=lambda r: (
                    _safe_float(r.get("RMSE_model"), 1e12),
                    _safe_float(r.get("MAE_model"), 1e12),
                    -_safe_float(r.get("Corr_model"), -1e12),
                ),
            )
            for rank, r in enumerate(seg_rows, 1):
                ranking_rows.append(
                    {
                        "protocol": pid,
                        "segment": seg,
                        "rank": rank,
                        "model": r.get("model", ""),
                        "RMSE_model": r.get("RMSE_model", ""),
                        "MAE_model": r.get("MAE_model", ""),
                        "Corr_model": r.get("Corr_model", ""),
                    }
                )

    for pid in protocol_order[1:]:
        rows = _read_summary_rows(os.path.join(protocol_outputs[pid], "stations_eval_models_summary.csv"))
        for r in rows:
            seg = str(r.get("segment", ""))
            model = str(r.get("model", ""))
            if seg not in compare_segments:
                continue
            rr = ref_map.get((seg, model))
            if not rr:
                continue
            rmse_ref = _safe_float(rr.get("RMSE_model"))
            rmse_cmp = _safe_float(r.get("RMSE_model"))
            mae_ref = _safe_float(rr.get("MAE_model"))
            mae_cmp = _safe_float(r.get("MAE_model"))
            corr_ref = _safe_float(rr.get("Corr_model"))
            corr_cmp = _safe_float(r.get("Corr_model"))
            comp_rows.append(
                {
                    "reference_protocol": ref,
                    "compared_protocol": pid,
                    "segment": seg,
                    "model": model,
                    "RMSE_ref": rmse_ref,
                    "RMSE_cmp": rmse_cmp,
                    "delta_RMSE_cmp_minus_ref": rmse_cmp - rmse_ref,
                    "MAE_ref": mae_ref,
                    "MAE_cmp": mae_cmp,
                    "delta_MAE_cmp_minus_ref": mae_cmp - mae_ref,
                    "Corr_ref": corr_ref,
                    "Corr_cmp": corr_cmp,
                    "delta_Corr_cmp_minus_ref": corr_cmp - corr_ref,
                }
            )

    comp_csv = os.path.join(compare_out_dir, "exp2_protocol_comparison_by_model_segment.csv")
    rank_csv = os.path.join(compare_out_dir, "exp2_protocol_rankings.csv")
    if comp_rows:
        _write_csv(comp_csv, comp_rows, list(comp_rows[0].keys()))
    if ranking_rows:
        _write_csv(rank_csv, ranking_rows, list(ranking_rows[0].keys()))

    report_md = os.path.join(compare_out_dir, "report_exp2_dual_protocol.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Experiment 2 Dual-Protocol Comparison\n\n")
        f.write(f"- Reference protocol: `{ref}`\n")
        f.write(f"- Compared protocols: `{', '.join(protocol_order[1:])}`\n")
        f.write(f"- Segments: `{', '.join(compare_segments)}`\n\n")

        f.write("## Protocol runs\n\n")
        f.write("| Protocol | Output dir |\n")
        f.write("|---|---|\n")
        for pid in protocol_order:
            f.write(f"| {pid} | `{protocol_outputs[pid]}` |\n")
        f.write("\n")

        if comp_rows:
            f.write("## Delta vs reference (negative delta RMSE/MAE is better)\n\n")
            f.write("| Compared | Segment | Model | ΔRMSE | ΔMAE | ΔCorr |\n")
            f.write("|---|---|---|---:|---:|---:|\n")
            for r in sorted(
                comp_rows,
                key=lambda x: (
                    x["compared_protocol"],
                    x["segment"],
                    x["delta_RMSE_cmp_minus_ref"],
                ),
            ):
                f.write(
                    f"| {r['compared_protocol']} | {r['segment']} | {r['model']} | "
                    f"{r['delta_RMSE_cmp_minus_ref']:.4f} | {r['delta_MAE_cmp_minus_ref']:.4f} | "
                    f"{r['delta_Corr_cmp_minus_ref']:.4f} |\n"
                )
            f.write("\n")

        if ranking_rows:
            f.write("## Ranking by protocol (RMSE)\n\n")
            f.write("| Protocol | Segment | Rank | Model | RMSE | MAE | Corr |\n")
            f.write("|---|---|---:|---|---:|---:|---:|\n")
            for r in sorted(
                ranking_rows,
                key=lambda x: (x["protocol"], x["segment"], int(x["rank"])),
            ):
                f.write(
                    f"| {r['protocol']} | {r['segment']} | {r['rank']} | {r['model']} | "
                    f"{_safe_float(r['RMSE_model']):.4f} | {_safe_float(r['MAE_model']):.4f} | "
                    f"{_safe_float(r['Corr_model']):.4f} |\n"
                )
            f.write("\n")

    print(comp_csv if os.path.exists(comp_csv) else "(no comparison CSV produced)")
    print(rank_csv if os.path.exists(rank_csv) else "(no ranking CSV produced)")
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
