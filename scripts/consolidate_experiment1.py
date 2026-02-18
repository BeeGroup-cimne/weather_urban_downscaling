#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np


MODEL_MAP = {
    "convlstm": "lstm",
    "baseline_nearest": "baseline_nearest",
    "baseline_bilinear": "baseline_bilinear",
    "unet": "unet",
    "transformer": "transformer",
    "mamba": "mamba",
}


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite(value):
    return not (math.isnan(value) or math.isinf(value))


def _model_label(raw_model):
    return MODEL_MAP.get(str(raw_model).strip().lower(), str(raw_model).strip().lower())


def _extract_seed(experiment):
    m = re.search(r"_S(\d+)_", str(experiment))
    return int(m.group(1)) if m else None


def _bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, seed=42):
    arr = np.asarray([v for v in values if _is_finite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return mean, low, high


def _bootstrap_delta_ci(model_vals, base_vals, n_boot=2000, alpha=0.05, seed=42):
    model_arr = np.asarray([v for v in model_vals if _is_finite(v)], dtype=np.float64)
    base_arr = np.asarray([v for v in base_vals if _is_finite(v)], dtype=np.float64)
    if model_arr.size == 0 or base_arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    delta = float(np.mean(base_arr) - np.mean(model_arr))
    if model_arr.size == 1 and base_arr.size == 1:
        return delta, delta, delta
    rng = np.random.default_rng(seed)
    model_idx = rng.integers(0, model_arr.size, size=(n_boot, model_arr.size))
    base_idx = rng.integers(0, base_arr.size, size=(n_boot, base_arr.size))
    delta_boot = base_arr[base_idx].mean(axis=1) - model_arr[model_idx].mean(axis=1)
    low = float(np.quantile(delta_boot, alpha / 2.0))
    high = float(np.quantile(delta_boot, 1.0 - alpha / 2.0))
    return delta, low, high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--raw-csv", default="")
    parser.add_argument("--training-csv", default="")
    parser.add_argument("--baseline", default="baseline_nearest")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = args.out_dir
    raw_csv = args.raw_csv or os.path.join(out_dir, "metrics_raw.csv")
    training_csv = args.training_csv or os.path.join(out_dir, "training_summary.csv")
    baseline = _model_label(args.baseline)

    if not os.path.exists(raw_csv):
        raise SystemExit(f"metrics_raw not found: {raw_csv}")

    rows = []
    by_model = defaultdict(list)
    by_model_seed = defaultdict(list)
    with open(raw_csv, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            model = _model_label(rec.get("model", ""))
            mae = _to_float(rec.get("mae"))
            rmse = _to_float(rec.get("rmse"))
            ssim = _to_float(rec.get("ssim"))
            if not (_is_finite(mae) and _is_finite(rmse) and _is_finite(ssim)):
                continue
            seed = _extract_seed(rec.get("experiment", ""))
            row = {
                "experiment": rec.get("experiment", ""),
                "model": model,
                "seed": seed,
                "mae": mae,
                "rmse": rmse,
                "ssim": ssim,
                "time": rec.get("time", ""),
            }
            rows.append(row)
            by_model[model].append(row)
            if seed is not None:
                by_model_seed[(model, seed)].append(row)

    if not rows:
        raise SystemExit(f"No valid rows found in: {raw_csv}")

    agg_ci_path = os.path.join(out_dir, "metrics_aggregate_ci.csv")
    with open(agg_ci_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "n",
            "mae_mean",
            "mae_ci_low",
            "mae_ci_high",
            "rmse_mean",
            "rmse_ci_low",
            "rmse_ci_high",
            "ssim_mean",
            "ssim_ci_low",
            "ssim_ci_high",
        ])
        out = []
        for model, vals in by_model.items():
            maes = [v["mae"] for v in vals]
            rmses = [v["rmse"] for v in vals]
            ssims = [v["ssim"] for v in vals]
            mae_mean, mae_low, mae_high = _bootstrap_mean_ci(
                maes, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed
            )
            rmse_mean, rmse_low, rmse_high = _bootstrap_mean_ci(
                rmses, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 1
            )
            ssim_mean, ssim_low, ssim_high = _bootstrap_mean_ci(
                ssims, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 2
            )
            out.append(
                (
                    model,
                    len(vals),
                    mae_mean,
                    mae_low,
                    mae_high,
                    rmse_mean,
                    rmse_low,
                    rmse_high,
                    ssim_mean,
                    ssim_low,
                    ssim_high,
                )
            )
        out.sort(key=lambda x: x[5])
        for row in out:
            w.writerow(row)

    by_seed_path = os.path.join(out_dir, "metrics_by_model_seed.csv")
    with open(by_seed_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "n_events", "mae_mean", "rmse_mean", "ssim_mean"])
        out = []
        for (model, seed), vals in by_model_seed.items():
            out.append(
                (
                    model,
                    seed,
                    len(vals),
                    float(np.mean([v["mae"] for v in vals])),
                    float(np.mean([v["rmse"] for v in vals])),
                    float(np.mean([v["ssim"] for v in vals])),
                )
            )
        out.sort(key=lambda x: (x[0], x[1]))
        for row in out:
            w.writerow(row)

    delta_path = os.path.join(out_dir, "metrics_delta_vs_baseline_ci.csv")
    base_rows = by_model.get(baseline, [])
    with open(delta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "delta_mae",
            "delta_mae_ci_low",
            "delta_mae_ci_high",
            "delta_rmse",
            "delta_rmse_ci_low",
            "delta_rmse_ci_high",
            "delta_ssim",
            "delta_ssim_ci_low",
            "delta_ssim_ci_high",
        ])
        if base_rows:
            base_mae = [v["mae"] for v in base_rows]
            base_rmse = [v["rmse"] for v in base_rows]
            base_ssim = [v["ssim"] for v in base_rows]
            out = []
            for model, vals in by_model.items():
                maes = [v["mae"] for v in vals]
                rmses = [v["rmse"] for v in vals]
                ssims = [v["ssim"] for v in vals]
                d_mae, d_mae_low, d_mae_high = _bootstrap_delta_ci(
                    maes, base_mae, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 11
                )
                d_rmse, d_rmse_low, d_rmse_high = _bootstrap_delta_ci(
                    rmses, base_rmse, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 12
                )
                d_ssim_neg, d_ssim_neg_low, d_ssim_neg_high = _bootstrap_delta_ci(
                    base_ssim, ssims, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 13
                )
                d_ssim = -d_ssim_neg
                d_ssim_low = -d_ssim_neg_high
                d_ssim_high = -d_ssim_neg_low
                out.append((
                    model,
                    d_mae,
                    d_mae_low,
                    d_mae_high,
                    d_rmse,
                    d_rmse_low,
                    d_rmse_high,
                    d_ssim,
                    d_ssim_low,
                    d_ssim_high,
                ))
            out.sort(key=lambda x: x[4], reverse=True)
            for row in out:
                w.writerow(row)

    report_path = os.path.join(out_dir, "report_experiment1.md")
    agg_rows = []
    with open(agg_ci_path, newline="", encoding="utf-8") as f:
        agg_rows = list(csv.DictReader(f))
    delta_rows = []
    if os.path.exists(delta_path):
        with open(delta_path, newline="", encoding="utf-8") as f:
            delta_rows = list(csv.DictReader(f))
    train_rows = []
    if os.path.exists(training_csv):
        with open(training_csv, newline="", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))

    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Experimento 1 consolidado\n\n")
        rf.write(f"- Source metrics: `{raw_csv}`\n")
        rf.write(f"- Bootstrap: n={args.bootstrap_samples}, alpha={args.alpha}, seed={args.bootstrap_seed}\n")
        rf.write(f"- Baseline de referencia: `{baseline}`\n\n")

        rf.write("## Aggregate metrics (95% CI)\n\n")
        rf.write("| model | n | MAE mean [CI] | RMSE mean [CI] | SSIM mean [CI] |\n")
        rf.write("|---|---:|---:|---:|---:|\n")
        for r in agg_rows:
            rf.write(
                f"| {r['model']} | {r['n']} | "
                f"{float(r['mae_mean']):.6f} [{float(r['mae_ci_low']):.6f}, {float(r['mae_ci_high']):.6f}] | "
                f"{float(r['rmse_mean']):.6f} [{float(r['rmse_ci_low']):.6f}, {float(r['rmse_ci_high']):.6f}] | "
                f"{float(r['ssim_mean']):.6f} [{float(r['ssim_ci_low']):.6f}, {float(r['ssim_ci_high']):.6f}] |\n"
            )

        if delta_rows:
            rf.write("\n## Delta vs baseline\n\n")
            rf.write("| model | ΔMAE [CI] | ΔRMSE [CI] | ΔSSIM [CI] |\n")
            rf.write("|---|---:|---:|---:|\n")
            for r in delta_rows:
                rf.write(
                    f"| {r['model']} | "
                    f"{float(r['delta_mae']):.6f} [{float(r['delta_mae_ci_low']):.6f}, {float(r['delta_mae_ci_high']):.6f}] | "
                    f"{float(r['delta_rmse']):.6f} [{float(r['delta_rmse_ci_low']):.6f}, {float(r['delta_rmse_ci_high']):.6f}] | "
                    f"{float(r['delta_ssim']):.6f} [{float(r['delta_ssim_ci_low']):.6f}, {float(r['delta_ssim_ci_high']):.6f}] |\n"
                )

        if train_rows:
            rf.write("\n## Training summary\n\n")
            rf.write("| model | seed | epochs_logged | best_val_loss | best_val_mae | final_val_loss | final_val_mae |\n")
            rf.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in train_rows:
                rf.write(
                    f"| {r.get('model','')} | {r.get('seed','')} | {r.get('epochs_logged','')} | "
                    f"{r.get('best_val_loss','')} | {r.get('best_val_mae','')} | "
                    f"{r.get('final_val_loss','')} | {r.get('final_val_mae','')} |\n"
                )

    print(f"Aggregate CI: {agg_ci_path}")
    print(f"By seed: {by_seed_path}")
    print(f"Delta CI: {delta_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
