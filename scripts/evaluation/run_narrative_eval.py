#!/usr/bin/env python3
"""
Deterministic post-training evaluation orchestrator.

Runs the publication narrative in order:
  1) Experiment 1 - Spatial Performance
  2) Experiment 2 - Ground-Truth Validation
  3) Experiment 3 - Bottleneck Ablation
  4) Case Study 1 - Extreme Heatwave
  5) Case Study 2 - Robustness
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


MODEL_ALIAS = {
    "convlstm": "lstm",
    "baseline_nearest": "baseline_nearest",
    "baseline_bilinear": "baseline_bilinear",
    "unet": "unet",
    "transformer": "transformer",
    "mamba": "mamba",
}


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


def _run(cmd: List[str], cwd: str, env: dict | None = None) -> None:
    print("▶", " ".join(cmd))
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env.setdefault("MPLBACKEND", "Agg")
    run_env.setdefault("XDG_CACHE_HOME", os.path.join(cwd, ".cache"))
    run_env.setdefault("MPLCONFIGDIR", os.path.join(run_env["XDG_CACHE_HOME"], "matplotlib"))
    os.makedirs(run_env["XDG_CACHE_HOME"], exist_ok=True)
    os.makedirs(run_env["MPLCONFIGDIR"], exist_ok=True)
    subprocess.run(cmd, check=True, cwd=cwd, env=run_env)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _normalize_timestamp(ts) -> str:
    if hasattr(ts, "isoformat"):
        s = ts.isoformat()
    else:
        s = str(ts)
    s = s.strip()
    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)
    return s


def _slug_time(iso_ts) -> str:
    s = _normalize_timestamp(iso_ts)
    return s.replace(":", "_").replace("-", "_").replace("T", "_")


def _read_times(stage_cfg: dict, root: str) -> List[str]:
    times = list(stage_cfg.get("times", []))
    times_file = stage_cfg.get("times_file", "")
    if times_file:
        times_file = _resolve_path(root, times_file)
        with open(times_file, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row or row.startswith("#"):
                    continue
                times.append(row)
    # Preserve order while deduplicating
    out = []
    seen = set()
    for t in times:
        ts = _normalize_timestamp(t)
        if not ts or ts in seen:
            continue
        seen.add(ts)
        out.append(ts)
    if not out:
        raise SystemExit("No timestamps defined for stage.")
    return out


def _model_type(model_key: str) -> Tuple[str, int | None]:
    mk = model_key.strip().lower()
    if mk == "lstm":
        return "convlstm", None
    if mk == "mamba_seq12":
        return "mamba", 12
    if mk == "mamba_seq6":
        return "mamba", 6
    return mk, None


def _seed_key(seed: int | str | None) -> str | None:
    if seed is None:
        return None
    s = str(seed).strip().lower()
    return s if s.startswith("s") else f"s{s}"


def _resolve_checkpoint_entries(
    checkpoint_set: dict,
    model: str,
    seed: int | str | None = None,
) -> List[Tuple[str, str]]:
    raw = checkpoint_set.get(model)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [("", raw)]
    if isinstance(raw, dict):
        wanted = _seed_key(seed)
        if wanted is not None:
            if wanted in raw:
                return [(wanted, raw[wanted])]
            # fallback: compare by numeric part
            want_num = wanted.lstrip("s")
            for k, v in raw.items():
                if str(k).lower().lstrip("s") == want_num:
                    return [(str(k), v)]
            raise SystemExit(f"Requested seed '{seed}' not found for model '{model}'.")
        return [(str(k), str(v)) for k, v in sorted(raw.items(), key=lambda x: str(x[0]))]
    raise SystemExit(f"Invalid checkpoint spec for model '{model}': {type(raw)}")


def _collect_metrics_rows(metrics_files: Iterable[str]) -> List[dict]:
    rows: List[dict] = []
    for fp in metrics_files:
        if not os.path.exists(fp):
            continue
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = dict(row)
                r["model"] = MODEL_ALIAS.get(r.get("model", "").strip().lower(), r.get("model", "").strip().lower())
                rows.append(r)
    return rows


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _aggregate_metrics(rows: List[dict], out_csv: str, group_keys: List[str]) -> None:
    grouped: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k, "") for k in group_keys)].append(r)

    out: List[dict] = []
    for key, vals in grouped.items():
        maes = [_safe_float(v.get("mae")) for v in vals]
        rmses = [_safe_float(v.get("rmse")) for v in vals]
        ssims = [_safe_float(v.get("ssim")) for v in vals]
        maes = [v for v in maes if v == v]
        rmses = [v for v in rmses if v == v]
        ssims = [v for v in ssims if v == v]
        row = {k: key[i] for i, k in enumerate(group_keys)}
        row["n"] = len(vals)
        row["mae_mean"] = sum(maes) / len(maes) if maes else ""
        row["rmse_mean"] = sum(rmses) / len(rmses) if rmses else ""
        row["ssim_mean"] = sum(ssims) / len(ssims) if ssims else ""
        row["mae_std"] = float((sum((x - row["mae_mean"]) ** 2 for x in maes) / max(1, len(maes) - 1)) ** 0.5) if len(maes) > 1 else 0.0
        row["rmse_std"] = float((sum((x - row["rmse_mean"]) ** 2 for x in rmses) / max(1, len(rmses) - 1)) ** 0.5) if len(rmses) > 1 else 0.0
        row["ssim_std"] = float((sum((x - row["ssim_mean"]) ** 2 for x in ssims) / max(1, len(ssims) - 1)) ** 0.5) if len(ssims) > 1 else 0.0
        out.append(row)

    out.sort(key=lambda r: _safe_float(r.get("rmse_mean"), 1e12))
    fields = group_keys + ["n", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "ssim_mean", "ssim_std"]
    _write_csv(out_csv, out, fields)


def _run_grid_stage(
    *,
    stage_name: str,
    stage_cfg: dict,
    root: str,
    python_bin: str,
    checkpoint_set: dict,
    single_seed: int | str | None = None,
    run_consolidate_exp1: bool = False,
) -> dict:
    out_dir = _resolve_path(root, stage_cfg["output_dir"])
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(fig_dir)
    times = _read_times(stage_cfg, root)
    run_id = stage_cfg.get("run_id", stage_name).upper()
    models = stage_cfg.get("models", [])
    if not models:
        models = [k for k in checkpoint_set.keys() if k != "baselines"]
        models.extend(checkpoint_set.get("baselines", []))

    metrics_files: List[str] = []
    map_basename = stage_cfg.get("map_basename", stage_name)
    map_out = os.path.join(fig_dir, f"{map_basename}.png")
    patch_size = int(stage_cfg.get("patch_size", 96))
    stride = int(stage_cfg.get("stride", 48))
    infer_batch = int(stage_cfg.get("infer_batch", 8))
    scale_from = stage_cfg.get("scale_from", "hr_lr")

    for model in models:
        model_type, forced_seq = _model_type(model)
        is_baseline = model.startswith("baseline_")
        if is_baseline:
            entries = [("", "")]
        else:
            entries = _resolve_checkpoint_entries(checkpoint_set, model, seed=single_seed)
            if not entries:
                raise SystemExit(f"No checkpoints resolved for model '{model}' in stage '{stage_name}'.")

        for seed_label, ckpt in entries:
            ckpt_abs = _resolve_path(root, ckpt) if ckpt else ""
            if ckpt_abs and not os.path.exists(ckpt_abs):
                raise SystemExit(f"Checkpoint not found: {ckpt_abs}")
            for ts in times:
                tag = _slug_time(ts)
                exp_name_parts = [run_id, model.upper()]
                if seed_label:
                    exp_name_parts.append(seed_label.upper())
                exp_name_parts.append(tag)
                exp_name = "_".join(exp_name_parts)

                cmd = [
                    python_bin,
                    "scripts/inference/run_inference_tiles_fullframe.py",
                    "--model-type", model_type,
                    "--patch-size", str(patch_size),
                    "--stride", str(stride),
                    "--batch-size", str(infer_batch),
                    "--time", ts,
                    "--use-last",
                    "--lr-resample", "nearest",
                    "--scale-from", scale_from,
                    "--experiment-name", exp_name,
                    "--out", map_out,
                ]
                if ckpt_abs:
                    cmd.extend(["--model-path", ckpt_abs])
                seq_len = int(stage_cfg.get("seq_len", 0)) or forced_seq
                if seq_len:
                    cmd.extend(["--seq-len", str(seq_len)])
                _run(cmd, cwd=root)

                metric_fp = os.path.join(fig_dir, f"{os.path.splitext(os.path.basename(map_out))[0]}_{exp_name}_metrics.csv")
                metrics_files.append(metric_fp)

    raw_rows = _collect_metrics_rows(metrics_files)
    if not raw_rows:
        raise SystemExit(f"No metrics generated for stage '{stage_name}'.")

    raw_csv = os.path.join(out_dir, "metrics_raw.csv")
    raw_fields = list(raw_rows[0].keys())
    _write_csv(raw_csv, raw_rows, raw_fields)
    _aggregate_metrics(raw_rows, os.path.join(out_dir, "metrics_aggregate.csv"), ["model"])

    # model+seed aggregate (seed parsed from experiment naming)
    for r in raw_rows:
        exp = str(r.get("experiment", ""))
        seed = ""
        parts = exp.split("_")
        for p in parts:
            if p.startswith("S") and p[1:].isdigit():
                seed = p
                break
        r["seed"] = seed
    _aggregate_metrics(raw_rows, os.path.join(out_dir, "metrics_by_model_seed.csv"), ["model", "seed"])

    if run_consolidate_exp1:
        _run(
            [
                python_bin,
                "scripts/evaluation/consolidate_experiment1.py",
                "--out-dir", out_dir,
                "--raw-csv", raw_csv,
                "--bootstrap-samples", str(int(stage_cfg.get("bootstrap_samples", 2000))),
                "--bootstrap-seed", str(int(stage_cfg.get("bootstrap_seed", 42))),
                "--alpha", str(float(stage_cfg.get("alpha", 0.05))),
            ],
            cwd=root,
        )

    return {
        "out_dir": out_dir,
        "raw_csv": raw_csv,
        "aggregate_csv": os.path.join(out_dir, "metrics_aggregate.csv"),
        "aggregate_ci_csv": os.path.join(out_dir, "metrics_aggregate_ci.csv"),
    }


def _run_exp2(
    *,
    stage_cfg: dict,
    root: str,
    python_bin: str,
    checkpoint_set: dict,
    data_cfg: dict,
) -> dict:
    out_dir = _resolve_path(root, stage_cfg["output_dir"])
    _ensure_dir(out_dir)

    models = stage_cfg.get("models", [])
    if not models:
        models = [k for k in checkpoint_set.keys() if k != "baselines"]
        models.extend(checkpoint_set.get("baselines", []))
    seed = stage_cfg.get("seed", 42)

    stations_obs_csv = _resolve_path(root, stage_cfg.get("stations_obs_csv") or data_cfg.get("stations_obs_csv", ""))
    stations_grib = _resolve_path(root, stage_cfg.get("stations_grib") or data_cfg.get("stations_grib", ""))
    stations_meta_csv = _resolve_path(root, stage_cfg.get("stations_meta_csv") or data_cfg.get("stations_meta_csv", ""))
    heatwave_times_file = _resolve_path(root, stage_cfg.get("heatwave_times_file") or data_cfg.get("heatwave_times_file", ""))

    if not stations_obs_csv and not stations_grib:
        raise SystemExit("Experiment 2 requires stations_obs_csv or stations_grib in config.data.")

    ckpt_used: Dict[str, str] = {}
    for model in models:
        model_type, forced_seq = _model_type(model)
        model_out = os.path.join(out_dir, model)
        _ensure_dir(model_out)
        is_baseline = model.startswith("baseline_")
        ckpt = ""
        if not is_baseline:
            entries = _resolve_checkpoint_entries(checkpoint_set, model, seed=seed)
            if not entries:
                raise SystemExit(f"No checkpoint for model '{model}' (seed={seed}) in Exp2.")
            ckpt = _resolve_path(root, entries[0][1])
            if not os.path.exists(ckpt):
                raise SystemExit(f"Checkpoint not found for Exp2: {ckpt}")
        ckpt_used[model] = ckpt if ckpt else "__baseline__"

        cmd = [
            python_bin,
            "scripts/evaluation/evaluate_stations_grib.py",
            "--model-type", model_type,
            "--split", str(stage_cfg.get("split", "test")),
            "--max-samples", str(int(stage_cfg.get("max_samples", 1000))),
            "--stride", str(int(stage_cfg.get("stride", 1))),
            "--extraction-method", str(stage_cfg.get("extraction_method", "bilinear")),
            "--day-start-hour", str(int(stage_cfg.get("day_start_hour", 8))),
            "--day-end-hour", str(int(stage_cfg.get("day_end_hour", 19))),
            "--time-offset-hours", str(float(stage_cfg.get("time_offset_hours", 0.0))),
            "--out-dir", model_out,
        ]
        if "footprint_radius_px" in stage_cfg:
            cmd.extend(["--footprint-radius-px", str(int(stage_cfg.get("footprint_radius_px", 0)))])
        if "footprint_sigma_px" in stage_cfg:
            cmd.extend(["--footprint-sigma-px", str(float(stage_cfg.get("footprint_sigma_px", 1.0)))])
        seq_len = int(stage_cfg.get("seq_len", 0)) or forced_seq
        if seq_len:
            cmd.extend(["--seq-len", str(seq_len)])
        if heatwave_times_file:
            cmd.extend(["--heatwave-times-file", heatwave_times_file])
        if stations_obs_csv:
            cmd.extend(["--stations-obs-csv", stations_obs_csv])
        else:
            cmd.extend(["--stations-grib", stations_grib])
            if stations_meta_csv:
                cmd.extend(["--stations-csv", stations_meta_csv])

        bias_mode = str(stage_cfg.get("bias_correction_mode", "none"))
        if bias_mode != "none":
            cmd.append("--bias-correction")
            cmd.extend(["--bias-correction-mode", bias_mode])
        if stage_cfg.get("save_hourly_traces", False):
            cmd.append("--save-hourly-traces")
        if is_baseline:
            cmd.append("--baseline")
        else:
            cmd.extend(["--model-path", ckpt])
        _run(cmd, cwd=root)

    # Write explicit checkpoints manifest for reproducibility.
    ckpt_rows = [{"model": m, "checkpoint": ckpt_used.get(m, "")} for m in models]
    if ckpt_rows:
        _write_csv(
            os.path.join(out_dir, "checkpoints_used.csv"),
            ckpt_rows,
            ["model", "checkpoint"],
        )

    # Consolidate model-level outputs
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
                    row = dict(row)
                    row["model"] = model
                    row["checkpoint"] = ckpt_used.get(model, "")
                    summary_rows.append(row)

        ps_path = os.path.join(model_out, "stations_eval_per_station.csv")
        pss_path = os.path.join(model_out, "stations_eval_per_station_by_segment.csv")
        for path in [ps_path, pss_path]:
            if os.path.exists(path):
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        rr = dict(row)
                        rr["model"] = model
                        rr["checkpoint"] = ckpt_used.get(model, "")
                        per_station_rows.append(rr)

    if summary_rows:
        fields = list(summary_rows[0].keys())
        _write_csv(os.path.join(out_dir, "stations_eval_models_summary.csv"), summary_rows, fields)

        grouped: Dict[str, List[dict]] = defaultdict(list)
        for r in summary_rows:
            seg = str(r.get("segment", "all"))
            grouped[seg].append(r)
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

    if per_station_rows:
        _write_csv(
            os.path.join(out_dir, "stations_eval_per_station_all_models.csv"),
            per_station_rows,
            list(per_station_rows[0].keys()),
        )
    if rank_rows:
        _write_csv(
            os.path.join(out_dir, "stations_eval_rank_by_segment.csv"),
            rank_rows,
            list(rank_rows[0].keys()),
        )

    return {
        "out_dir": out_dir,
        "summary_csv": os.path.join(out_dir, "stations_eval_models_summary.csv"),
        "rank_csv": os.path.join(out_dir, "stations_eval_rank_by_segment.csv"),
    }


def _run_exp3(
    *,
    stage_cfg: dict,
    root: str,
    python_bin: str,
    exp1_agg_csv: str = "",
) -> dict:
    out_dir = _resolve_path(root, stage_cfg["output_dir"])
    eval_dir = os.path.join(out_dir, "evals")
    _ensure_dir(eval_dir)

    comparisons = stage_cfg.get("comparisons", [])
    if not comparisons:
        raise SystemExit("Exp3 requires narrative.exp3.comparisons in config.")

    raw_rows: List[dict] = []
    for comp in comparisons:
        name = str(comp.get("name") or comp.get("model") or "comparison").strip()
        model_key = str(comp.get("model", name)).strip().lower()
        model_type, forced_seq = _model_type(str(comp.get("model_type", model_key)))
        model_path = _resolve_path(root, str(comp.get("checkpoint", comp.get("model_path", ""))))
        if model_path and not os.path.exists(model_path):
            raise SystemExit(f"Exp3 checkpoint not found: {model_path}")
        seq_len = int(comp.get("seq_len", 0)) or forced_seq
        split = str(comp.get("split", stage_cfg.get("split", "test")))
        out_csv = os.path.join(eval_dir, f"{name}.csv")

        cmd = [
            python_bin,
            "scripts/evaluation/evaluate_test_set.py",
            "--model-type", model_type,
            "--model-path", model_path,
            "--split", split,
            "--ssim-samples", str(int(stage_cfg.get("ssim_samples", 256))),
            "--max-batches", str(int(stage_cfg.get("max_batches", 0))),
            "--log-every", str(int(stage_cfg.get("log_every", 25))),
            "--out-csv", out_csv,
        ]
        if seq_len:
            cmd.extend(["--seq-len", str(seq_len)])
        _run(cmd, cwd=root)

        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r = rows[0]
        raw_rows.append(
            {
                "model": model_key,
                "comparison": name,
                "seed": str(comp.get("seed", "")),
                "split": r.get("split", split),
                "mae": r.get("mae", ""),
                "rmse": r.get("rmse", ""),
                "mse": r.get("mse", ""),
                "ssim": r.get("ssim", ""),
                "ssim_samples": r.get("ssim_samples", ""),
                "model_type": r.get("model_type", model_type),
                "model_path": r.get("model_path", model_path),
                "seq_len": seq_len or "",
            }
        )

    if not raw_rows:
        raise SystemExit("Exp3 generated no evaluation rows.")

    raw_csv = os.path.join(out_dir, "fullframe_eval_raw.csv")
    _write_csv(raw_csv, raw_rows, list(raw_rows[0].keys()))

    cmd = [
        python_bin,
        "scripts/evaluation/consolidate_experiment3.py",
        "--out-dir", out_dir,
        "--eval-raw-csv", raw_csv,
        "--bootstrap-samples", str(int(stage_cfg.get("bootstrap_samples", 2000))),
        "--bootstrap-seed", str(int(stage_cfg.get("bootstrap_seed", 42))),
        "--alpha", str(float(stage_cfg.get("alpha", 0.05))),
    ]
    if exp1_agg_csv and os.path.exists(exp1_agg_csv):
        cmd.extend(["--exp1-agg-csv", exp1_agg_csv])
    _run(cmd, cwd=root)

    # Explicit ablation comparison export (e.g., seq6 vs seq12)
    by_name = {r["comparison"]: r for r in raw_rows}
    seq_compare_rows: List[dict] = []
    for pair in stage_cfg.get("compare_pairs", []):
        a = by_name.get(pair.get("a", ""))
        b = by_name.get(pair.get("b", ""))
        if not a or not b:
            continue
        seq_compare_rows.append(
            {
                "pair": f"{pair.get('a')} vs {pair.get('b')}",
                "rmse_delta_b_minus_a": _safe_float(b.get("rmse")) - _safe_float(a.get("rmse")),
                "mae_delta_b_minus_a": _safe_float(b.get("mae")) - _safe_float(a.get("mae")),
                "ssim_delta_b_minus_a": _safe_float(b.get("ssim")) - _safe_float(a.get("ssim")),
            }
        )
    if seq_compare_rows:
        _write_csv(os.path.join(out_dir, "seq_compare.csv"), seq_compare_rows, list(seq_compare_rows[0].keys()))

    return {
        "out_dir": out_dir,
        "raw_csv": raw_csv,
        "aggregate_csv": os.path.join(out_dir, "fullframe_eval_aggregate.csv"),
    }


def _run_cs2(
    *,
    stage_cfg: dict,
    root: str,
    python_bin: str,
    checkpoint_set: dict,
) -> dict:
    out_dir = _resolve_path(root, stage_cfg["output_dir"])
    _ensure_dir(out_dir)
    models = stage_cfg.get("models", [])
    if not models:
        models = [k for k in checkpoint_set.keys() if k != "baselines"]
        models.extend(checkpoint_set.get("baselines", []))
    seed = stage_cfg.get("seed", 42)
    times = _read_times(stage_cfg, root)
    eps = stage_cfg.get("epsilons", [0.1, 0.25, 0.5, 1.0])
    n_trials = int(stage_cfg.get("n_trials", 50))

    merged_rows: List[dict] = []
    for model in models:
        model_type, forced_seq = _model_type(model)
        is_baseline = model.startswith("baseline_")
        ckpt = ""
        if not is_baseline:
            entries = _resolve_checkpoint_entries(checkpoint_set, model, seed=seed)
            if not entries:
                raise SystemExit(f"No checkpoint for CS2 model '{model}' (seed={seed}).")
            ckpt = _resolve_path(root, entries[0][1])
            if not os.path.exists(ckpt):
                raise SystemExit(f"CS2 checkpoint not found: {ckpt}")

        model_out = os.path.join(out_dir, model)
        _ensure_dir(model_out)
        cmd = [
            python_bin,
            "scripts/evaluation/run_robustness_experiment.py",
            "--model-type", model_type,
            "--patch-size", str(int(stage_cfg.get("patch_size", 96))),
            "--stride", str(int(stage_cfg.get("stride", 48))),
            "--batch-size", str(int(stage_cfg.get("batch_size", 8))),
            "--n-trials", str(n_trials),
            "--seed", str(int(stage_cfg.get("rng_seed", 42))),
            "--outdir", model_out,
            "--experiment-name", f"{stage_cfg.get('run_id', 'cs2')}_{model}",
        ]
        seq_len = int(stage_cfg.get("seq_len", 0)) or forced_seq
        if seq_len:
            cmd.extend(["--seq-len", str(seq_len)])
        if ckpt:
            cmd.extend(["--model-path", ckpt])
        cmd.extend(["--epsilon"] + [str(float(x)) for x in eps])
        cmd.extend(["--time"] + list(times))
        _run(cmd, cwd=root)

        model_csv = os.path.join(model_out, "robustness_results.csv")
        if not os.path.exists(model_csv):
            continue
        with open(model_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rr = dict(row)
                rr["model_key"] = model
                rr["checkpoint_path"] = ckpt if ckpt else "__baseline__"
                merged_rows.append(rr)

    if not merged_rows:
        raise SystemExit("CS2 generated no robustness rows.")

    summary_csv = os.path.join(out_dir, "robustness_summary.csv")
    _write_csv(summary_csv, merged_rows, list(merged_rows[0].keys()))

    # Rank stability by epsilon (lower mean abs deviation vs clean is better)
    by_eps_model: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in merged_rows:
        eps_val = str(r.get("epsilon_K", ""))
        model = str(r.get("model_key", ""))
        mad = _safe_float(r.get("pred_mean_abs_dev_vs_clean_C"))
        if mad == mad:
            by_eps_model[(eps_val, model)].append(mad)

    rank_rows: List[dict] = []
    by_eps: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (eps_val, model), vals in by_eps_model.items():
        by_eps[eps_val].append((model, sum(vals) / len(vals)))
    for eps_val, pairs in by_eps.items():
        pairs.sort(key=lambda x: x[1])
        for rank, (model, score) in enumerate(pairs, 1):
            rank_rows.append(
                {
                    "epsilon_K": eps_val,
                    "rank": rank,
                    "model": model,
                    "mean_abs_dev_vs_clean_C": score,
                }
            )
    if rank_rows:
        _write_csv(os.path.join(out_dir, "cs2_rank_stability.csv"), rank_rows, list(rank_rows[0].keys()))

    return {"out_dir": out_dir, "summary_csv": summary_csv}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run deterministic post-training evaluation narrative.")
    ap.add_argument("--config", required=True, help="Path to eval_config.yaml (or .json)")
    ap.add_argument(
        "--stages",
        default="exp1,exp2,exp3,cs1,cs2",
        help="Comma-separated subset: exp1,exp2,exp3,cs1,cs2",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    root = _resolve_path(os.getcwd(), cfg.get("project_root", ".")) or os.getcwd()
    root = os.path.abspath(root)

    if not cfg.get("frozen_training", False):
        raise SystemExit("Config must set frozen_training: true for this orchestrator.")

    python_bin = cfg.get("python_bin") or sys.executable
    stages = {s.strip().lower() for s in str(args.stages).split(",") if s.strip()}
    data_cfg = cfg.get("data", {})
    checkpoints_cfg = cfg.get("checkpoints", {})
    narrative = cfg.get("narrative", {})

    exp1_out = {}
    if "exp1" in stages and narrative.get("exp1", {}).get("enabled", True):
        s = narrative["exp1"]
        cset = checkpoints_cfg[s["checkpoint_set"]]
        exp1_out = _run_grid_stage(
            stage_name="exp1",
            stage_cfg=s,
            root=root,
            python_bin=python_bin,
            checkpoint_set=cset,
            single_seed=None,
            run_consolidate_exp1=True,
        )

    if "exp2" in stages and narrative.get("exp2", {}).get("enabled", True):
        s = narrative["exp2"]
        cset = checkpoints_cfg[s["checkpoint_set"]]
        _run_exp2(
            stage_cfg=s,
            root=root,
            python_bin=python_bin,
            checkpoint_set=cset,
            data_cfg=data_cfg,
        )

    if "exp3" in stages and narrative.get("exp3", {}).get("enabled", True):
        s = narrative["exp3"]
        exp1_agg = s.get("exp1_agg_csv", "") or exp1_out.get("aggregate_ci_csv", "")
        if exp1_agg:
            exp1_agg = _resolve_path(root, exp1_agg)
        _run_exp3(
            stage_cfg=s,
            root=root,
            python_bin=python_bin,
            exp1_agg_csv=exp1_agg,
        )

    if "cs1" in stages and narrative.get("case_study_1", {}).get("enabled", True):
        s = narrative["case_study_1"]
        cset = checkpoints_cfg[s["checkpoint_set"]]
        _run_grid_stage(
            stage_name="cs1",
            stage_cfg=s,
            root=root,
            python_bin=python_bin,
            checkpoint_set=cset,
            single_seed=s.get("seed", 42),
            run_consolidate_exp1=False,
        )

    if "cs2" in stages and narrative.get("case_study_2", {}).get("enabled", True):
        s = narrative["case_study_2"]
        cset = checkpoints_cfg[s["checkpoint_set"]]
        _run_cs2(
            stage_cfg=s,
            root=root,
            python_bin=python_bin,
            checkpoint_set=cset,
        )

    print("✅ Deterministic evaluation narrative completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
