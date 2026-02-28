#!/usr/bin/env python3
"""
Experiment 2 sanity checks (stations external validation).

Purpose:
  - Detect gross issues behind surprising station scores (units, de-norm, time shift).
  - Quantify whether a checkpoint matches its training target (UrbClim HR) at all.

This script is intentionally lightweight (small sample counts) and prints a compact report.
It uses the same cache + normalization assumptions as scripts/evaluation/evaluate_stations_grib.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config  # noqa: E402
from src.models_legacy import ModelZoo  # noqa: E402


def _time_index(values) -> pd.Index:
    return pd.to_datetime(values).floor("h")


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[ok]
    obs = obs[ok]
    if pred.size == 0:
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Corr": np.nan}
    err = pred - obs
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if pred.size < 3:
        corr = np.nan
    else:
        with np.errstate(all="ignore"):
            corr = float(np.corrcoef(pred, obs)[0, 1])
    return {"N": int(pred.size), "MAE": mae, "RMSE": rmse, "Bias": bias, "Corr": corr}


def _load_station_obs_csv(path: str):
    df = pd.read_csv(path)
    required = {"time", "station_id", "lat", "lon"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"stations-obs-csv missing columns: {', '.join(missing)}")

    value_col = None
    for candidate in ["obs_c", "t2m_c", "air_temperature_c", "temperature_c", "value_c"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        raise ValueError("stations-obs-csv missing temperature column (obs_c or alias).")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.floor("h")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time", "station_id", "lat", "lon", value_col])
    df["station_id"] = df["station_id"].astype(str)
    if df.empty:
        raise ValueError("stations-obs-csv has no valid rows after parsing.")

    # collapse duplicates per station-hour
    df = (
        df.groupby(["time", "station_id", "lat", "lon"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "obs_c"})
    )

    # Kelvin guard
    if float(df["obs_c"].mean()) > 200.0:
        df["obs_c"] = df["obs_c"] - 273.15

    st_meta = df.groupby("station_id", as_index=False)[["lat", "lon"]].first()
    st_meta = st_meta.sort_values("station_id").reset_index(drop=True)
    station_ids = st_meta["station_id"].values
    lat = st_meta["lat"].values
    lon = st_meta["lon"].values

    times = pd.Index(sorted(df["time"].unique()))
    time_pos = {t: i for i, t in enumerate(times)}
    st_pos = {sid: i for i, sid in enumerate(station_ids)}
    obs = np.full((len(times), len(station_ids)), np.nan, dtype=np.float32)

    for row in df.itertuples(index=False):
        i = time_pos[row.time]
        j = st_pos[row.station_id]
        obs[i, j] = float(row.obs_c)

    return times, station_ids, lat, lon, obs


def _build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    if model_type == "transformer":
        return ModelZoo.build_transformer()
    return ModelZoo.build_unet()


def _nearest_idx(arr_1d: np.ndarray, val: float) -> int:
    arr_1d = np.asarray(arr_1d)
    return int(np.argmin(np.abs(arr_1d - val)))


def _split_bounds(times: pd.Index, split: str) -> tuple[int, int]:
    def _time_indices(times_idx, start, end):
        arr = np.asarray(times_idx.values).astype("datetime64[ns]")
        start_dt = np.datetime64(pd.to_datetime(start).to_datetime64())
        end_dt = np.datetime64(pd.to_datetime(end).to_datetime64())
        start_i = int(np.searchsorted(arr, start_dt, side="left"))
        end_i = int(np.searchsorted(arr, end_dt, side="left"))
        return start_i, end_i

    if split == "train":
        return _time_indices(times, Config.TRAIN_START, Config.TRAIN_END)
    if split == "val":
        return _time_indices(times, Config.VAL_START, Config.VAL_END)
    return _time_indices(times, Config.TEST_START, Config.TEST_END)


def _iter_eval_starts(
    start_i: int,
    end_i: int,
    seq_len: int,
    stride: int,
    max_samples: int,
) -> Iterable[int]:
    # iterate i such that last_idx = i + seq_len - 1 is within [start_i, end_i)
    used = 0
    for i in range(start_i, max(start_i, end_i - seq_len), stride):
        yield i
        used += 1
        if used >= max_samples:
            return


@dataclass
class SanityResult:
    n_times_used: int
    n_station_points: int
    out_norm_mean: float
    out_norm_std: float
    out_c_mean: float
    out_c_std: float
    model_vs_hr_stations: dict
    model_vs_hr_grid: dict
    model_vs_obs_stations: dict
    hr_vs_obs_stations: dict
    lag_table: list[dict]


def run_sanity(
    *,
    model_path: str,
    model_type: str,
    seq_len: int,
    split: str,
    stations_obs_csv: str,
    max_samples: int,
    stride: int,
    lags: list[int],
    grid_sample: int,
) -> SanityResult:
    # Load stations
    st_times, st_ids, st_lat, st_lon, st_obs = _load_station_obs_csv(stations_obs_csv)
    st_time_index = pd.Index(st_times)
    st_time_map = {t: i for i, t in enumerate(st_time_index)}

    # Cache
    z = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    lr = z["lr_input"]
    hr = z["hr_target"]
    times = _time_index(z["time"].values)

    # Sync shapes from cache
    lr_lat_dim = next(d for d in lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"])
    lr_lon_dim = next(d for d in lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"])
    hr_lat_dim = next(d for d in hr.dims if d in ["latitude", "lat", "y"])
    hr_lon_dim = next(d for d in hr.dims if d in ["longitude", "lon", "x"])
    Config.LR_SHAPE = (lr.sizes[lr_lat_dim], lr.sizes[lr_lon_dim])
    Config.HR_SHAPE = (hr.sizes[hr_lat_dim], hr.sizes[hr_lon_dim])
    Config.SEQ_LEN = int(seq_len)

    # Station -> HR indices (nearest); if UrbClim vs stations is sane, this mapping is sane.
    hr_lat_vals = np.asarray(z[hr_lat_dim].values)
    hr_lon_vals = np.asarray(z[hr_lon_dim].values)
    if hr_lat_vals.ndim != 1 or hr_lon_vals.ndim != 1:
        raise ValueError("Sanity check expects 1D HR lat/lon coords in cache (got 2D).")
    st_i = np.array([_nearest_idx(hr_lat_vals, float(v)) for v in st_lat], dtype=np.int32)
    st_j = np.array([_nearest_idx(hr_lon_vals, float(v)) for v in st_lon], dtype=np.int32)

    # Static normalization (same logic as evaluation)
    static = np.load(Config.STATIC_CACHE_PATH)
    if static.ndim == 2:
        static = static[..., np.newaxis]
    mean_st = np.mean(static, axis=(0, 1), keepdims=True)
    std_st = np.std(static, axis=(0, 1), keepdims=True)
    static_norm = (static - mean_st) / (std_st + 1e-6)

    # Stats for de-normalization
    stats = np.load(Config.STATS_PATH)
    mean_hr = float(stats["mean_hr"])
    std_hr = float(stats["std_hr"])

    # Build model and load weights
    model = _build_model(model_type)
    model.load_weights(model_path)

    start_i, end_i = _split_bounds(times, split)

    out_norm_vals = []
    out_c_vals = []

    pred_st_all = []
    hr_st_all = []
    obs_st_all = []

    pred_grid_all = []
    hr_grid_all = []

    # for lag checks (collect per-time mean across stations)
    time_used = []
    pred_mean = []
    obs_mean = []

    n_times_used = 0
    rng = np.random.default_rng(42)

    for i in _iter_eval_starts(start_i, end_i, Config.SEQ_LEN, stride, max_samples):
        last_idx = i + Config.SEQ_LEN - 1
        t = times[last_idx]
        if t not in st_time_map:
            continue

        x_lr = lr.isel(time=slice(i, i + Config.SEQ_LEN)).values.astype(np.float32)
        x_st = np.broadcast_to(static_norm[np.newaxis, ...], (Config.SEQ_LEN, *static_norm.shape)).astype(np.float32)
        x_lr_b = np.expand_dims(x_lr, 0)
        x_st_b = np.expand_dims(x_st, 0)

        y_pred_norm = model((x_lr_b, x_st_b), training=False).numpy()[0, -1, :, :, 0]
        y_pred_c = y_pred_norm * std_hr + mean_hr

        y_hr_norm = hr.isel(time=last_idx).values
        if y_hr_norm.ndim > 2:
            y_hr_norm = y_hr_norm[..., 0]
        y_hr_c = y_hr_norm * std_hr + mean_hr

        obs_idx = st_time_map[t]
        obs_t = st_obs[obs_idx]

        pred_t = y_pred_c[st_i, st_j]
        hr_t = y_hr_c[st_i, st_j]

        if grid_sample and grid_sample > 0:
            h, w = y_pred_c.shape
            n = int(min(grid_sample, h * w))
            flat_idx = rng.choice(h * w, size=n, replace=False)
            ii = flat_idx // w
            jj = flat_idx % w
            pred_grid_all.append(y_pred_c[ii, jj].astype(np.float32))
            hr_grid_all.append(y_hr_c[ii, jj].astype(np.float32))

        ok = np.isfinite(obs_t)
        if not np.any(ok):
            continue

        pred_st_all.append(pred_t[ok].astype(np.float32))
        hr_st_all.append(hr_t[ok].astype(np.float32))
        obs_st_all.append(obs_t[ok].astype(np.float32))

        out_norm_vals.append(float(np.mean(y_pred_norm)))
        out_c_vals.append(float(np.mean(y_pred_c)))

        time_used.append(pd.to_datetime(t))
        pred_mean.append(float(np.nanmean(pred_t[ok])))
        obs_mean.append(float(np.nanmean(obs_t[ok])))

        n_times_used += 1

    if n_times_used == 0:
        raise RuntimeError("No aligned times found between cache and station observations.")

    pred_st_all = np.concatenate(pred_st_all, axis=0)
    hr_st_all = np.concatenate(hr_st_all, axis=0)
    obs_st_all = np.concatenate(obs_st_all, axis=0)

    # Metrics at station points
    m_model_hr = _metrics(pred_st_all, hr_st_all)
    m_model_obs = _metrics(pred_st_all, obs_st_all)
    m_hr_obs = _metrics(hr_st_all, obs_st_all)

    # Metrics over random grid points (model vs HR only)
    if pred_grid_all:
        pred_grid_all = np.concatenate(pred_grid_all, axis=0)
        hr_grid_all = np.concatenate(hr_grid_all, axis=0)
        m_model_hr_grid = _metrics(pred_grid_all, hr_grid_all)
    else:
        m_model_hr_grid = {"N": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Corr": np.nan}

    # Lag diagnostics using the mean over stations per time step.
    # This is a blunt tool, but it catches obvious off-by-one/hour shifts.
    t_series = pd.Index(time_used)
    pred_series = pd.Series(pred_mean, index=t_series).sort_index()
    obs_series = pd.Series(obs_mean, index=t_series).sort_index()

    lag_rows = []
    for lag in lags:
        if lag == 0:
            aligned_pred = pred_series
        else:
            aligned_pred = pred_series.shift(lag, freq="h")
        df = pd.DataFrame({"pred": aligned_pred, "obs": obs_series}).dropna()
        lag_rows.append(
            {
                "lag_hours": int(lag),
                **_metrics(df["pred"].to_numpy(), df["obs"].to_numpy()),
            }
        )

    return SanityResult(
        n_times_used=n_times_used,
        n_station_points=int(pred_st_all.size),
        out_norm_mean=float(np.mean(out_norm_vals)),
        out_norm_std=float(np.std(out_norm_vals)),
        out_c_mean=float(np.mean(out_c_vals)),
        out_c_std=float(np.std(out_c_vals)),
        model_vs_hr_stations=m_model_hr,
        model_vs_hr_grid=m_model_hr_grid,
        model_vs_obs_stations=m_model_obs,
        hr_vs_obs_stations=m_hr_obs,
        lag_table=lag_rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm", "transformer"])
    ap.add_argument("--seq-len", type=int, default=6)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--stations-obs-csv", required=True)
    ap.add_argument("--max-samples", type=int, default=50, help="Max time windows to try (not station-points).")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--lags", default="-3,-2,-1,0,1,2,3", help="Comma-separated lag hours for offset check.")
    ap.add_argument("--grid-sample", type=int, default=3000, help="Random grid pixels per time for model-vs-HR check (0 disables).")
    args = ap.parse_args()

    lags = [int(x.strip()) for x in str(args.lags).split(",") if x.strip()]

    r = run_sanity(
        model_path=args.model_path,
        model_type=args.model_type,
        seq_len=int(args.seq_len),
        split=args.split,
        stations_obs_csv=args.stations_obs_csv,
        max_samples=int(args.max_samples),
        stride=int(args.stride),
        lags=lags,
        grid_sample=int(args.grid_sample),
    )

    def _fmt(m: dict) -> str:
        return (
            f"N={m['N']} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
            f"Bias={m['Bias']:.3f} Corr={m['Corr']:.3f}"
        )

    print("== Exp2 Sanity Check ==")
    print(f"model_path: {args.model_path}")
    print(f"model_type: {args.model_type}  seq_len: {args.seq_len}  split: {args.split}")
    print(f"times_used: {r.n_times_used}  station_points_used: {r.n_station_points}")
    print(f"pred_norm(mean over maps): mean={r.out_norm_mean:.3f} std={r.out_norm_std:.3f}")
    print(f"pred_C(mean over maps):    mean={r.out_c_mean:.3f} std={r.out_c_std:.3f}")
    print(f"model vs HR (stations):    {_fmt(r.model_vs_hr_stations)}")
    print(f"model vs HR (grid):        {_fmt(r.model_vs_hr_grid)}")
    print(f"model vs OBS (stations):   {_fmt(r.model_vs_obs_stations)}")
    print(f"HR vs OBS (stations):      {_fmt(r.hr_vs_obs_stations)}")
    print("lag_check (mean over stations):")
    for row in r.lag_table:
        print(f"  lag={row['lag_hours']:+d}h  " + _fmt(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
