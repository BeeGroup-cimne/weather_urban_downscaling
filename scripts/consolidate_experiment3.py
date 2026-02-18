#!/usr/bin/env python3

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite(value):
    return not (math.isnan(value) or math.isinf(value))


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


def _load_exp1_rmse(path):
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        model = str(row.get("model", "")).strip().lower()
        rmse = _to_float(row.get("rmse_mean"))
        if model and _is_finite(rmse):
            out[model] = rmse
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-raw-csv", default="")
    parser.add_argument("--training-summary-csv", default="")
    parser.add_argument("--exp1-agg-csv", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = args.out_dir
    raw_csv = args.eval_raw_csv or os.path.join(out_dir, "fullframe_eval_raw.csv")
    training_csv = args.training_summary_csv or os.path.join(out_dir, "fullframe_training_summary.csv")
    exp1_agg = args.exp1_agg_csv

    if not os.path.exists(raw_csv):
        raise SystemExit(f"fullframe eval raw csv not found: {raw_csv}")

    by_model = defaultdict(list)
    with open(raw_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = str(row.get("model", "")).strip().lower()
            mae = _to_float(row.get("mae"))
            rmse = _to_float(row.get("rmse"))
            ssim = _to_float(row.get("ssim"))
            if not model or not (_is_finite(mae) and _is_finite(rmse)):
                continue
            by_model[model].append(
                {
                    "seed": row.get("seed", ""),
                    "split": row.get("split", ""),
                    "mae": mae,
                    "rmse": rmse,
                    "ssim": ssim,
                }
            )

    agg_out = os.path.join(out_dir, "fullframe_eval_aggregate.csv")
    with open(agg_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "n",
            "mae_mean",
            "mae_std",
            "mae_ci_low",
            "mae_ci_high",
            "rmse_mean",
            "rmse_std",
            "rmse_ci_low",
            "rmse_ci_high",
            "ssim_mean",
            "ssim_std",
            "ssim_ci_low",
            "ssim_ci_high",
        ])
        rows = []
        for model, vals in by_model.items():
            maes = [v["mae"] for v in vals if _is_finite(v["mae"])]
            rmses = [v["rmse"] for v in vals if _is_finite(v["rmse"])]
            ssims = [v["ssim"] for v in vals if _is_finite(v["ssim"])]
            if not maes or not rmses:
                continue
            mae_mean, mae_low, mae_high = _bootstrap_mean_ci(
                maes, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed
            )
            rmse_mean, rmse_low, rmse_high = _bootstrap_mean_ci(
                rmses, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 1
            )
            ssim_mean, ssim_low, ssim_high = _bootstrap_mean_ci(
                ssims, n_boot=args.bootstrap_samples, alpha=args.alpha, seed=args.bootstrap_seed + 2
            )
            rows.append(
                (
                    model,
                    len(vals),
                    mae_mean,
                    float(np.std(maes)) if len(maes) > 1 else 0.0,
                    mae_low,
                    mae_high,
                    rmse_mean,
                    float(np.std(rmses)) if len(rmses) > 1 else 0.0,
                    rmse_low,
                    rmse_high,
                    ssim_mean,
                    float(np.std(ssims)) if len(ssims) > 1 else 0.0,
                    ssim_low,
                    ssim_high,
                )
            )
        rows.sort(key=lambda x: x[6])  # rmse asc
        for row in rows:
            w.writerow(row)

    exp1_rmse = _load_exp1_rmse(exp1_agg)
    ranking_out = os.path.join(out_dir, "ranking_stability_vs_exp1.csv")
    if os.path.exists(agg_out):
        with open(agg_out, newline="", encoding="utf-8") as f:
            exp3_rows = list(csv.DictReader(f))
    else:
        exp3_rows = []

    exp3_rank = {r["model"]: idx + 1 for idx, r in enumerate(sorted(exp3_rows, key=lambda x: _to_float(x.get("rmse_mean"))))}
    exp1_rank = {m: idx + 1 for idx, (m, _) in enumerate(sorted(exp1_rmse.items(), key=lambda x: x[1]))}

    with open(ranking_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "rank_exp1_tiles", "rank_exp3_fullframe", "delta_rank", "rmse_exp1_tiles", "rmse_exp3_fullframe"])
        models = sorted(set(list(exp3_rank.keys()) + list(exp1_rank.keys())))
        for model in models:
            r1 = exp1_rank.get(model)
            r3 = exp3_rank.get(model)
            rmse1 = exp1_rmse.get(model, float("nan"))
            rmse3 = next((_to_float(r.get("rmse_mean")) for r in exp3_rows if r.get("model") == model), float("nan"))
            delta = (r3 - r1) if (r1 is not None and r3 is not None) else ""
            w.writerow([model, "" if r1 is None else r1, "" if r3 is None else r3, delta, rmse1, rmse3])

    report_out = os.path.join(out_dir, "report_experiment3.md")
    with open(report_out, "w", encoding="utf-8") as rf:
        rf.write("# Experimento 3: réplica full-frame top modelos\n\n")
        rf.write(f"- Eval raw: `{raw_csv}`\n")
        if exp1_agg:
            rf.write(f"- Exp1 aggregate (tiles): `{exp1_agg}`\n")
        rf.write(f"- Bootstrap: n={args.bootstrap_samples}, alpha={args.alpha}, seed={args.bootstrap_seed}\n\n")

        rf.write("## Full-frame aggregate\n\n")
        rf.write("| model | n | MAE mean [CI] | RMSE mean [CI] | SSIM mean [CI] |\n")
        rf.write("|---|---:|---:|---:|---:|\n")
        for r in sorted(exp3_rows, key=lambda x: _to_float(x.get("rmse_mean"))):
            rf.write(
                f"| {r.get('model','')} | {r.get('n','')} | "
                f"{_to_float(r.get('mae_mean')):.6f} [{_to_float(r.get('mae_ci_low')):.6f}, {_to_float(r.get('mae_ci_high')):.6f}] | "
                f"{_to_float(r.get('rmse_mean')):.6f} [{_to_float(r.get('rmse_ci_low')):.6f}, {_to_float(r.get('rmse_ci_high')):.6f}] | "
                f"{_to_float(r.get('ssim_mean')):.6f} [{_to_float(r.get('ssim_ci_low')):.6f}, {_to_float(r.get('ssim_ci_high')):.6f}] |\n"
            )

        if exp1_rmse:
            rf.write("\n## Ranking stability (tiles vs full-frame)\n\n")
            rf.write("| model | rank_exp1_tiles | rank_exp3_fullframe | delta_rank | rmse_exp1_tiles | rmse_exp3_fullframe |\n")
            rf.write("|---|---:|---:|---:|---:|---:|\n")
            with open(ranking_out, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rf.write(
                        f"| {row.get('model','')} | {row.get('rank_exp1_tiles','')} | {row.get('rank_exp3_fullframe','')} | "
                        f"{row.get('delta_rank','')} | {row.get('rmse_exp1_tiles','')} | {row.get('rmse_exp3_fullframe','')} |\n"
                    )

        if os.path.exists(training_csv):
            rf.write("\n## Training summary\n\n")
            with open(training_csv, newline="", encoding="utf-8") as f:
                train_rows = list(csv.DictReader(f))
            rf.write("| model | seed | epochs_logged | best_val_loss | best_val_mae | final_val_loss | final_val_mae |\n")
            rf.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in train_rows:
                rf.write(
                    f"| {r.get('model','')} | {r.get('seed','')} | {r.get('epochs_logged','')} | "
                    f"{r.get('best_val_loss','')} | {r.get('best_val_mae','')} | "
                    f"{r.get('final_val_loss','')} | {r.get('final_val_mae','')} |\n"
                )

    print(f"Full-frame aggregate: {agg_out}")
    print(f"Ranking stability: {ranking_out}")
    print(f"Experiment 3 report: {report_out}")


if __name__ == "__main__":
    main()
