#!/usr/bin/env python3
"""Generate heatwave alerts from model prediction maps."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEVEL_RANK = {
    "normal": 0,
    "watch": 1,
    "warning": 2,
    "severe": 3,
}


@dataclass(frozen=True)
class PredictionItem:
    time: pd.Timestamp
    path: Path


def _to_naive_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(pd.to_datetime(value))
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts.tz_localize(None)


def _parse_time_from_name(path: Path, pattern: str) -> pd.Timestamp:
    match = re.search(pattern, path.name)
    if not match:
        raise ValueError(f"could not parse timestamp from filename: {path}")
    raw = match.group(1)
    normalized = raw.replace("_", "-")
    parts = normalized.split("-")
    if len(parts) >= 6:
        normalized = "-".join(parts[:3]) + "T" + ":".join(parts[3:6])
    elif len(parts) == 5:
        normalized = "-".join(parts[:3]) + "T" + ":".join(parts[3:5])
    return _to_naive_timestamp(normalized)


def _load_manifest(path: Path) -> list[PredictionItem]:
    df = pd.read_csv(path)
    required = {"time", "path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    items: list[PredictionItem] = []
    for _, row in df.iterrows():
        pred_path = Path(str(row["path"]))
        if not pred_path.is_absolute():
            pred_path = (path.parent / pred_path).resolve()
        items.append(PredictionItem(_to_naive_timestamp(row["time"]), pred_path))
    return items


def _discover_predictions(args: argparse.Namespace) -> list[PredictionItem]:
    if args.manifest:
        items = _load_manifest(Path(args.manifest))
    else:
        pred_dir = Path(args.prediction_dir)
        files = sorted(pred_dir.glob(args.pattern))
        items = [
            PredictionItem(_parse_time_from_name(path, args.time_regex), path.resolve())
            for path in files
        ]
    if not items:
        raise SystemExit("no prediction files found")
    missing = [str(item.path) for item in items if not item.path.exists()]
    if missing:
        raise SystemExit(f"prediction files not found: {missing[:5]}")
    return sorted(items, key=lambda item: item.time)


def _load_prediction_stack(items: Iterable[PredictionItem]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times = []
    arrays = []
    expected_shape = None
    for item in items:
        arr = np.load(item.path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise SystemExit(f"prediction must be 2D or HxWx1: {item.path} shape={arr.shape}")
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise SystemExit(f"prediction shape mismatch: {item.path} shape={arr.shape}, expected={expected_shape}")
        times.append(item.time)
        arrays.append(arr)
    return pd.DatetimeIndex(times), np.stack(arrays, axis=0)


def _load_threshold_map(args: argparse.Namespace, target_shape: tuple[int, int]) -> np.ndarray:
    if args.threshold_celsius is not None:
        return np.full(target_shape, float(args.threshold_celsius), dtype=np.float32)

    if args.threshold_map:
        path = Path(args.threshold_map)
        if path.suffix == ".npy":
            thr = np.load(path).astype(np.float32)
        else:
            ds = xr.open_dataset(path)
            var = args.threshold_var or next(iter(ds.data_vars))
            if var not in ds:
                raise SystemExit(f"threshold variable not found: {var}. Available: {list(ds.data_vars)}")
            thr = ds[var].values.astype(np.float32)
        if thr.ndim == 3 and thr.shape[0] == 1:
            thr = thr[0]
        if thr.ndim == 3 and thr.shape[-1] == 1:
            thr = thr[..., 0]
        if thr.shape != target_shape:
            raise SystemExit(f"threshold map shape mismatch: {thr.shape}, expected={target_shape}")
        return thr

    if args.derive_threshold_from_cache:
        return _derive_threshold_from_cache(args, target_shape)

    raise SystemExit("provide --threshold-celsius, --threshold-map, or --derive-threshold-from-cache")


def _derive_threshold_from_cache(args: argparse.Namespace, target_shape: tuple[int, int]) -> np.ndarray:
    from config.runtime import Config

    ds = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    if "hr_target" not in ds:
        raise SystemExit(f"hr_target not found in cache: {Config.PATH_CACHE}")
    da = ds["hr_target"]
    stats = np.load(Config.STATS_PATH, allow_pickle=True)
    mean_hr = float(stats["mean_hr"])
    std_hr = float(stats["std_hr"])
    da = (da * std_hr) + mean_hr
    time_dim = next((d for d in da.dims if d in {"time", "valid_time", "t"}), "time")
    lat_dim = next((d for d in da.dims if d in {"latitude", "lat", "y"}), da.dims[-2])
    lon_dim = next((d for d in da.dims if d in {"longitude", "lon", "x"}), da.dims[-1])

    start = pd.Timestamp(args.base_start)
    end = pd.Timestamp(args.base_end)
    months = [int(m.strip()) for m in str(args.months).split(",") if m.strip()]
    base = da.sel({time_dim: (da[time_dim] >= start) & (da[time_dim] < end)})
    daily_tmax = base.resample({time_dim: "1D"}).max()
    daily_tmax = daily_tmax.sel({time_dim: daily_tmax[time_dim].dt.month.isin(months)})
    if hasattr(daily_tmax.data, "chunks"):
        daily_tmax = daily_tmax.chunk({time_dim: -1})
    thr = daily_tmax.quantile(float(args.pctl), dim=time_dim, skipna=True)
    thr = thr.transpose(lat_dim, lon_dim).values.astype(np.float32)
    if thr.ndim == 3 and thr.shape[-1] == 1:
        thr = thr[..., 0]
    if thr.shape != target_shape:
        raise SystemExit(f"derived threshold shape mismatch: {thr.shape}, expected={target_shape}")
    return thr


def _level_from_fraction(
    fraction: float,
    *,
    watch_fraction: float,
    warning_fraction: float,
    severe_fraction: float,
) -> str:
    if fraction >= severe_fraction:
        return "severe"
    if fraction >= warning_fraction:
        return "warning"
    if fraction >= watch_fraction:
        return "watch"
    return "normal"


def _events_from_daily(daily: pd.DataFrame, min_duration_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = daily["alert_level"].map(LEVEL_RANK).to_numpy() >= LEVEL_RANK["warning"]
    dates = pd.to_datetime(daily["date"]).to_numpy()
    event_ids = np.zeros(len(daily), dtype=np.int32)
    rows = []
    event_id = 0
    start = None

    def finish(end_idx: int) -> None:
        nonlocal event_id, start
        if start is None:
            return
        duration = end_idx - start + 1
        if duration >= min_duration_days:
            event_id += 1
            event_ids[start : end_idx + 1] = event_id
            chunk = daily.iloc[start : end_idx + 1]
            peak_idx = chunk["max_exceedance_c"].astype(float).idxmax()
            rows.append(
                {
                    "event_id": event_id,
                    "start_date": str(pd.Timestamp(dates[start]).date()),
                    "end_date": str(pd.Timestamp(dates[end_idx]).date()),
                    "duration_days": int(duration),
                    "peak_date": str(pd.to_datetime(daily.loc[peak_idx, "date"]).date()),
                    "peak_alert_level": max(chunk["alert_level"], key=lambda v: LEVEL_RANK[str(v)]),
                    "peak_exceed_fraction": float(chunk["exceed_fraction"].max()),
                    "peak_max_pred_c": float(chunk["max_pred_c"].max()),
                    "peak_max_exceedance_c": float(chunk["max_exceedance_c"].max()),
                }
            )
        start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if start is not None and ((not flag) or i == len(flags) - 1):
            finish(i if flag and i == len(flags) - 1 else i - 1)

    daily = daily.copy()
    daily["event_id"] = event_ids
    return daily, pd.DataFrame(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="CSV with columns: time,path")
    parser.add_argument("--prediction-dir", default="experiments/predictions", help="Directory with .npy maps")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern used with --prediction-dir")
    parser.add_argument(
        "--time-regex",
        default=r"(\d{4}[-_]\d{2}[-_]\d{2}[T_]\d{2}[_:]\d{2}(?:[_:]\d{2})?)",
        help="Regex with one timestamp capture group for filenames.",
    )
    parser.add_argument("--threshold-celsius", type=float, default=None)
    parser.add_argument("--threshold-map", default="", help=".npy or NetCDF threshold map")
    parser.add_argument("--threshold-var", default="", help="Variable name for NetCDF threshold map")
    parser.add_argument("--derive-threshold-from-cache", action="store_true")
    parser.add_argument("--base-start", default="2017-01-01")
    parser.add_argument("--base-end", default="2018-01-01")
    parser.add_argument("--months", default="6,7,8,9")
    parser.add_argument("--pctl", type=float, default=0.95)
    parser.add_argument("--watch-fraction", type=float, default=0.05)
    parser.add_argument("--warning-fraction", type=float, default=0.10)
    parser.add_argument("--severe-fraction", type=float, default=0.25)
    parser.add_argument("--min-duration-days", type=int, default=3)
    parser.add_argument("--out-dir", default="experiments/alerts/latest")
    args = parser.parse_args(argv)

    items = _discover_predictions(args)
    times, preds = _load_prediction_stack(items)
    threshold = _load_threshold_map(args, target_shape=preds.shape[1:])

    exceedance = preds - threshold[None, :, :]
    exceed_mask = exceedance > 0.0
    n_cells = int(np.prod(preds.shape[1:]))

    hourly_rows = []
    for i, time in enumerate(times):
        fraction = float(np.nanmean(exceed_mask[i]))
        hourly_rows.append(
            {
                "time": time.isoformat(),
                "date": time.date().isoformat(),
                "alert_level": _level_from_fraction(
                    fraction,
                    watch_fraction=args.watch_fraction,
                    warning_fraction=args.warning_fraction,
                    severe_fraction=args.severe_fraction,
                ),
                "exceed_fraction": fraction,
                "affected_cells": int(np.nansum(exceed_mask[i])),
                "total_cells": n_cells,
                "mean_pred_c": float(np.nanmean(preds[i])),
                "max_pred_c": float(np.nanmax(preds[i])),
                "mean_exceedance_c": float(np.nanmean(np.maximum(exceedance[i], 0.0))),
                "max_exceedance_c": float(np.nanmax(exceedance[i])),
            }
        )
    hourly = pd.DataFrame(hourly_rows)

    daily_rows = []
    for date, group in hourly.groupby("date", sort=True):
        idx = group.index.to_numpy()
        daily_max = np.nanmax(preds[idx], axis=0)
        daily_exceedance = daily_max - threshold
        daily_mask = daily_exceedance > 0.0
        fraction = float(np.nanmean(daily_mask))
        daily_rows.append(
            {
                "date": date,
                "alert_level": _level_from_fraction(
                    fraction,
                    watch_fraction=args.watch_fraction,
                    warning_fraction=args.warning_fraction,
                    severe_fraction=args.severe_fraction,
                ),
                "exceed_fraction": fraction,
                "affected_cells": int(np.nansum(daily_mask)),
                "total_cells": n_cells,
                "mean_pred_c": float(np.nanmean(daily_max)),
                "max_pred_c": float(np.nanmax(daily_max)),
                "mean_exceedance_c": float(np.nanmean(np.maximum(daily_exceedance, 0.0))),
                "max_exceedance_c": float(np.nanmax(daily_exceedance)),
            }
        )
    daily = pd.DataFrame(daily_rows)
    daily, events = _events_from_daily(daily, min_duration_days=int(args.min_duration_days))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = out_dir / "alerts_hourly.csv"
    daily_path = out_dir / "alerts_daily.csv"
    events_path = out_dir / "heatwave_events.csv"
    summary_path = out_dir / "alerts_summary.json"
    latest_path = out_dir / "latest_alert.json"
    latest_exceedance_path = out_dir / "latest_exceedance.npy"

    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)
    events.to_csv(events_path, index=False)
    np.save(latest_exceedance_path, exceedance[-1].astype(np.float32))

    highest_hourly = max(hourly["alert_level"], key=lambda v: LEVEL_RANK[str(v)])
    highest_daily = max(daily["alert_level"], key=lambda v: LEVEL_RANK[str(v)])
    latest = hourly.iloc[-1].to_dict()
    latest_date = times[-1].normalize()
    latest["is_active_heatwave_event"] = bool(
        any(
            pd.Timestamp(row["start_date"]) <= latest_date <= pd.Timestamp(row["end_date"])
            for _, row in events.iterrows()
        )
        if not events.empty
        else False
    )

    summary = {
        "status": "ok",
        "prediction_count": int(len(times)),
        "start_time": times[0].isoformat(),
        "end_time": times[-1].isoformat(),
        "highest_hourly_alert_level": highest_hourly,
        "highest_daily_alert_level": highest_daily,
        "event_count": int(len(events)),
        "threshold": {
            "mode": (
                "scalar"
                if args.threshold_celsius is not None
                else "map"
                if args.threshold_map
                else "derived_from_cache"
            ),
            "pctl": float(args.pctl) if args.derive_threshold_from_cache else None,
            "months": args.months if args.derive_threshold_from_cache else None,
        },
        "outputs": {
            "hourly": str(hourly_path),
            "daily": str(daily_path),
            "events": str(events_path),
            "latest": str(latest_path),
            "latest_exceedance": str(latest_exceedance_path),
        },
    }
    _write_json(summary_path, summary)
    _write_json(latest_path, latest)

    print(f"alerts summary: {summary_path}")
    print(f"highest daily level: {highest_daily}")
    print(f"events: {len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
