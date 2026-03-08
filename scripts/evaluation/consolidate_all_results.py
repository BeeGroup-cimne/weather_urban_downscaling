#!/usr/bin/env python3
"""
Build a unified consolidation report across:
- Experiment 1 (publish run)
- Experiment 2 (stations)
- Case Study 1 (heatwave + adaptation)
- Experiment 3 (full-frame)

Behavior:
1) Use explicit run directories provided via CLI args.
2) Try to reconstruct missing Experiment 1 artifacts from per-figure metrics CSVs.
3) Write consolidated outputs inside Experiment 2 directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional


def _safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _build_metrics_raw_from_figures(exp1_dir: str) -> Optional[str]:
    raw_csv = os.path.join(exp1_dir, "metrics_raw.csv")
    fig_metrics = sorted(glob.glob(os.path.join(exp1_dir, "figures", "*_metrics.csv")))
    if not fig_metrics:
        return None

    header: Optional[List[str]] = None
    rows: List[Dict[str, str]] = []
    for fp in fig_metrics:
        with open(fp, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        if header is None:
            header = list(data[0].keys())
        for rec in data:
            # Keep only known header keys to prevent malformed merges.
            rows.append({k: rec.get(k, "") for k in header})

    if not header or not rows:
        return None

    _write_csv(raw_csv, rows, header)
    return raw_csv


def _ensure_experiment1_artifacts(project_root: str, exp1_dir: Optional[str]) -> Dict[str, str]:
    status = {
        "exp1_dir": exp1_dir or "",
        "metrics_raw": "",
        "metrics_aggregate_ci": "",
        "report_publish": "",
        "report_experiment1": "",
        "note": "",
    }
    if not exp1_dir or not os.path.isdir(exp1_dir):
        status["note"] = "Experiment 1 directory not found."
        return status

    raw_csv = os.path.join(exp1_dir, "metrics_raw.csv")
    agg_ci = os.path.join(exp1_dir, "metrics_aggregate_ci.csv")
    report_pub = os.path.join(exp1_dir, "report_publish.md")
    report_e1 = os.path.join(exp1_dir, "report_experiment1.md")

    # Rebuild metrics_raw.csv from figures if missing.
    if not os.path.exists(raw_csv):
        rebuilt_raw = _build_metrics_raw_from_figures(exp1_dir)
        if rebuilt_raw:
            raw_csv = rebuilt_raw

    # Rebuild aggregate/report with consolidate_experiment1.py if raw exists and aggregate missing.
    consolidate_py = os.path.join(project_root, "scripts", "evaluation", "consolidate_experiment1.py")
    if os.path.exists(raw_csv) and os.path.exists(consolidate_py):
        if not os.path.exists(agg_ci) or not os.path.exists(report_e1):
            subprocess.run(
                [
                    "python3",
                    consolidate_py,
                    "--out-dir",
                    exp1_dir,
                    "--raw-csv",
                    raw_csv,
                    "--bootstrap-samples",
                    "2000",
                    "--bootstrap-seed",
                    "42",
                    "--alpha",
                    "0.05",
                ],
                check=False,
            )

    status["metrics_raw"] = raw_csv if os.path.exists(raw_csv) else ""
    status["metrics_aggregate_ci"] = agg_ci if os.path.exists(agg_ci) else ""
    status["report_publish"] = report_pub if os.path.exists(report_pub) else ""
    status["report_experiment1"] = report_e1 if os.path.exists(report_e1) else ""

    if not status["metrics_raw"] and not status["metrics_aggregate_ci"]:
        status["note"] = (
            "No Experiment 1 consolidated artifacts found. "
            "Sync run folder from server including CSV artifacts."
        )
    return status


def _read_exp2_rank(exp2_dir: str) -> List[Dict]:
    path = os.path.join(exp2_dir, "stations_eval_models_summary.csv")
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("segment") != "all":
                continue
            out.append(
                {
                    "model": r.get("model", ""),
                    "ssim": _safe_float(r.get("Corr_model", "")),
                    "mae": _safe_float(r.get("MAE_model", "")),
                    "rmse": _safe_float(r.get("RMSE_model", "")),
                    "n": _safe_int(r.get("N", "")),
                    "checkpoint": r.get("checkpoint", ""),
                }
            )
    out.sort(key=lambda x: (-x["ssim"], x["mae"]))
    for i, row in enumerate(out, 1):
        row["rank_ssim"] = i
    return out


def _read_cs1_rank(cs1_dir: Optional[str]) -> List[Dict]:
    out: List[Dict] = []
    if not cs1_dir:
        return out
    path = os.path.join(cs1_dir, "metrics_aggregate.csv")
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                {
                    "variant": r.get("variant", ""),
                    "ssim": _safe_float(r.get("ssim_mean", "")),
                    "mae": _safe_float(r.get("mae_mean", "")),
                    "rmse": _safe_float(r.get("rmse_mean", "")),
                    "n": _safe_int(r.get("n", "")),
                }
            )
    out.sort(key=lambda x: (-x["ssim"], x["mae"]))
    for i, row in enumerate(out, 1):
        row["rank_ssim"] = i
    return out


def _read_exp3_status(exp3_dir: str) -> List[Dict]:
    dirs = [exp3_dir] if exp3_dir and os.path.isdir(exp3_dir) else []
    out: List[Dict] = []
    for d in dirs:
        raw = os.path.join(d, "fullframe_eval_raw.csv")
        agg = os.path.join(d, "fullframe_eval_aggregate.csv")
        n_raw = 0
        n_agg = 0
        if os.path.exists(raw):
            with open(raw, newline="", encoding="utf-8") as f:
                n_raw = max(sum(1 for _ in f) - 1, 0)
        if os.path.exists(agg):
            with open(agg, newline="", encoding="utf-8") as f:
                n_agg = max(sum(1 for _ in f) - 1, 0)
        out.append({"dir": d, "rows_raw": n_raw, "rows_agg": n_agg})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--exp2-dir", required=True, help="Explicit experiment2 directory.")
    parser.add_argument("--exp1-dir", required=True, help="Explicit experiment1 publish_run directory.")
    parser.add_argument("--cs1-dir", required=True, help="Explicit case study 1 directory.")
    parser.add_argument("--exp3-dir", required=True, help="Explicit experiment3 directory.")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    exp2_dir = args.exp2_dir
    exp1_dir = args.exp1_dir
    cs1_dir = args.cs1_dir
    exp3_dir = args.exp3_dir

    if not os.path.isabs(exp2_dir):
        exp2_dir = os.path.join(project_root, exp2_dir)
    if not os.path.isabs(exp1_dir):
        exp1_dir = os.path.join(project_root, exp1_dir)
    if not os.path.isabs(cs1_dir):
        cs1_dir = os.path.join(project_root, cs1_dir)
    if not os.path.isabs(exp3_dir):
        exp3_dir = os.path.join(project_root, exp3_dir)

    if not exp2_dir:
        raise SystemExit("No experiment2 directory found.")

    exp1_status = _ensure_experiment1_artifacts(project_root, exp1_dir)
    exp2_rank = _read_exp2_rank(exp2_dir)
    cs1_rank = _read_cs1_rank(cs1_dir)
    exp3_status = _read_exp3_status(exp3_dir)

    csv_out = os.path.join(exp2_dir, "consolidated_all_results.csv")
    csv_rows: List[Dict] = []
    for r in exp2_rank:
        csv_rows.append(
            {
                "section": "experiment2_stations",
                "rank_ssim": r["rank_ssim"],
                "model": r["model"],
                "variant": "",
                "ssim_or_corr": f"{r['ssim']:.6f}",
                "mae": f"{r['mae']:.6f}",
                "rmse": f"{r['rmse']:.6f}",
                "n": r["n"],
                "notes": r["checkpoint"],
            }
        )
    for r in cs1_rank:
        csv_rows.append(
            {
                "section": "casestudy1_heatwave",
                "rank_ssim": r["rank_ssim"],
                "model": "",
                "variant": r["variant"],
                "ssim_or_corr": f"{r['ssim']:.6f}",
                "mae": f"{r['mae']:.6f}",
                "rmse": f"{r['rmse']:.6f}",
                "n": r["n"],
                "notes": "",
            }
        )
    _write_csv(
        csv_out,
        csv_rows,
        ["section", "rank_ssim", "model", "variant", "ssim_or_corr", "mae", "rmse", "n", "notes"],
    )

    md_out = os.path.join(exp2_dir, "consolidated_all_results.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Consolidated Results (Experiment 1 + 2 + Case Study + Experiment 3)\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Base experiment folder: `{exp2_dir}`\n\n")

        f.write("## Source runs used\n\n")
        f.write(f"- Experiment 2 (stations): `{exp2_dir}`\n")
        f.write(f"- Case Study 1: `{cs1_dir or 'not found'}`\n")
        f.write(f"- Experiment 1 publish run: `{exp1_dir or 'not found'}`\n")
        if exp3_status:
            f.write(f"- Experiment 3: `{exp3_status[0]['dir']}`\n")
        f.write("\n")

        f.write("## Experiment 2 (stations, segment=all, sorted by SSIM/Corr)\n\n")
        if exp2_rank:
            f.write("| rank | model | SSIM/Corr | MAE | RMSE | N | checkpoint |\n")
            f.write("|---:|---|---:|---:|---:|---:|---|\n")
            for r in exp2_rank:
                f.write(
                    f"| {r['rank_ssim']} | {r['model']} | {r['ssim']:.6f} | {r['mae']:.6f} | "
                    f"{r['rmse']:.6f} | {r['n']} | `{r['checkpoint']}` |\n"
                )
        else:
            f.write("No valid rows found.\n")
        f.write("\n")

        f.write("## Case Study 1 (heatwave+adaptation, sorted by SSIM mean)\n\n")
        if cs1_rank:
            f.write("| rank | variant | SSIM mean | MAE mean | RMSE mean | n |\n")
            f.write("|---:|---|---:|---:|---:|---:|\n")
            for r in cs1_rank:
                f.write(
                    f"| {r['rank_ssim']} | {r['variant']} | {r['ssim']:.6f} | "
                    f"{r['mae']:.6f} | {r['rmse']:.6f} | {r['n']} |\n"
                )
        else:
            f.write("No valid rows found.\n")
        f.write("\n")

        f.write("## Mamba seq6 vs seq12\n\n")
        e6 = next((r for r in exp2_rank if r["model"] == "mamba"), None)
        e12 = next((r for r in exp2_rank if r["model"] == "mamba_seq12"), None)
        c6 = next((r for r in cs1_rank if r["variant"] == "mamba_seq6"), None)
        c12 = next((r for r in cs1_rank if r["variant"] == "mamba_seq12"), None)
        if e6 and e12:
            f.write(
                f"- Experiment 2: seq12 improves SSIM/Corr by `{(e12['ssim'] - e6['ssim']):.6f}` "
                f"and reduces MAE by `{(e6['mae'] - e12['mae']):.6f}`.\n"
            )
        else:
            f.write("- Experiment 2: comparison not available.\n")
        if c6 and c12:
            f.write(
                f"- Case Study 1: seq6 improves SSIM mean by `{(c6['ssim'] - c12['ssim']):.6f}` "
                f"and MAE mean by `{(c12['mae'] - c6['mae']):.6f}`.\n"
            )
        else:
            f.write("- Case Study 1: comparison not available.\n")
        f.write("\n")

        f.write("## Experiment 1 artifacts\n\n")
        if exp1_status["metrics_aggregate_ci"] or exp1_status["metrics_raw"]:
            if exp1_status["metrics_raw"]:
                f.write(f"- metrics_raw: `{exp1_status['metrics_raw']}`\n")
            if exp1_status["metrics_aggregate_ci"]:
                f.write(f"- metrics_aggregate_ci: `{exp1_status['metrics_aggregate_ci']}`\n")
            if exp1_status["report_experiment1"]:
                f.write(f"- report_experiment1: `{exp1_status['report_experiment1']}`\n")
            if exp1_status["report_publish"]:
                f.write(f"- report_publish: `{exp1_status['report_publish']}`\n")
        else:
            f.write(exp1_status["note"] + "\n")
        f.write("\n")

        f.write("## Experiment 3 status\n\n")
        if exp3_status:
            f.write("| run | eval_raw rows | eval_aggregate rows | status |\n")
            f.write("|---|---:|---:|---|\n")
            for row in exp3_status:
                state = "OK" if (row["rows_raw"] > 0 or row["rows_agg"] > 0) else "empty"
                f.write(
                    f"| `{row['dir']}` | {row['rows_raw']} | {row['rows_agg']} | {state} |\n"
                )
        else:
            f.write("Experiment 3 directory not found.\n")

    print(md_out)
    print(csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
