#!/usr/bin/env python3
"""
Evaluate model vs station observations (GRIB, t2m) and UrbClim HR.
Generates per-station and summary tables + basic figures for paper.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.models_legacy import ModelZoo


def _time_index(values):
    return pd.to_datetime(values).floor("H")


def _find_var(ds, preferred):
    for name in preferred:
        if name in ds.data_vars:
            return name
    # fallback to first variable
    return list(ds.data_vars)[0]


def _load_station_grib(path: str, stations_csv: str = ""):
    cfgrib_kwargs = {
        "filter_by_keys": {"typeOfLevel": "surface"},
        "errors": "ignore",
        "indexpath": "",
    }
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=cfgrib_kwargs)

    var = _find_var(ds, ["t2m", "2t", "tas", "airTemperature"])
    da = ds[var]

    # convert to Celsius if needed
    sample_mean = float(da.isel({da.dims[0]: slice(0, min(3, da.sizes[da.dims[0]]))}).mean().values)
    if sample_mean > 200:
        da = da - 273.15

    time_dim = next((d for d in da.dims if d in ["time", "valid_time"]), da.dims[0])

    # Try station dimension
    station_dim = None
    for d in da.dims:
        if d not in {time_dim, "latitude", "longitude", "lat", "lon", "y", "x"}:
            station_dim = d
            break

    if station_dim is not None:
        # Expect lat/lon coords per station_dim
        lat_name = next((c for c in ds.coords if "lat" in c.lower()), None)
        lon_name = next((c for c in ds.coords if "lon" in c.lower()), None)
        if lat_name is None or lon_name is None:
            raise ValueError("Station GRIB missing latitude/longitude coords.")

        lat = ds[lat_name]
        lon = ds[lon_name]
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("Station GRIB lat/lon are not 1D per station.")

        # Data to (time, station)
        obs = da.transpose(time_dim, station_dim)
        times = _time_index(obs[time_dim].values)
        station_ids = obs[station_dim].values
        return times, station_ids, lat.values, lon.values, obs.values

    # Otherwise, treat as gridded and require stations CSV
    if not stations_csv:
        raise ValueError("GRIB looks gridded. Provide --stations-csv with lat/lon.")

    stations_df = pd.read_csv(stations_csv)
    required = {"station_id", "lat", "lon"}
    if not required.issubset(stations_df.columns):
        raise ValueError("stations.csv must include station_id, lat, lon")

    lat_name = next((c for c in ds.coords if "lat" in c.lower()), None)
    lon_name = next((c for c in ds.coords if "lon" in c.lower()), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Gridded GRIB missing lat/lon coords.")

    obs = da
    obs = obs.interp(
        {lat_name: xr.DataArray(stations_df["lat"].values, dims="station"),
         lon_name: xr.DataArray(stations_df["lon"].values, dims="station")},
        method="nearest"
    )
    times = _time_index(obs[time_dim].values)
    station_ids = stations_df["station_id"].values
    return times, station_ids, stations_df["lat"].values, stations_df["lon"].values, obs.values


def _build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    if model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    return ModelZoo.build_unet()


def _order(vals):
    diffs = np.diff(vals)
    if np.all(diffs > 0):
        return "asc"
    if np.all(diffs < 0):
        return "desc"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations-grib", required=True, help="GRIB file with station t2m")
    parser.add_argument("--stations-csv", default="", help="CSV with station_id,lat,lon (if GRIB is gridded)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="mamba", choices=["mamba", "unet", "convlstm"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out-dir", default=os.path.join("experiments", "stations_eval"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load stations data
    st_times, st_ids, st_lat, st_lon, st_obs = _load_station_grib(
        args.stations_grib, args.stations_csv
    )
    st_time_index = pd.Index(st_times)
    st_time_map = {t: i for i, t in enumerate(st_time_index)}

    # Load cache
    z = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    lr = z["lr_input"]
    hr = z["hr_target"]

    # Sync Config shapes from cache before building model
    lr_lat_dim = next(d for d in lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"])
    lr_lon_dim = next(d for d in lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"])
    hr_lat_dim = next(d for d in hr.dims if d in ["latitude", "lat", "y"])
    hr_lon_dim = next(d for d in hr.dims if d in ["longitude", "lon", "x"])
    Config.LR_SHAPE = (lr.sizes[lr_lat_dim], lr.sizes[lr_lon_dim])
    Config.HR_SHAPE = (hr.sizes[hr_lat_dim], hr.sizes[hr_lon_dim])
    times = _time_index(z["time"].values)

    # Detect lon order mismatch for LR
    lr_lon = next(d for d in lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"])
    hr_lon = next(d for d in hr.dims if d in ["longitude", "lon", "x"])
    flip_lr_lon = False
    try:
        lr_order = _order(z[lr_lon].values)
        hr_order = _order(z[hr_lon].values)
        if lr_order != "unknown" and hr_order != "unknown" and lr_order != hr_order:
            flip_lr_lon = True
    except Exception:
        pass

    # Map stations to HR grid indices (nearest)
    hr_lat_dim = next(d for d in hr.dims if d in ["latitude", "lat", "y"])
    hr_lon_dim = next(d for d in hr.dims if d in ["longitude", "lon", "x"])
    hr_lat_vals = z[hr_lat_dim].values
    hr_lon_vals = z[hr_lon_dim].values

    def _nearest_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    st_i = np.array([_nearest_idx(hr_lat_vals, v) for v in st_lat])
    st_j = np.array([_nearest_idx(hr_lon_vals, v) for v in st_lon])

    # Split indices
    def _time_indices(times_idx, start, end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        start_i = int(np.searchsorted(times_idx.values, start, side="left"))
        end_i = int(np.searchsorted(times_idx.values, end, side="left"))
        return start_i, end_i

    if args.split == "train":
        start_i, end_i = _time_indices(times, Config.TRAIN_START, Config.TRAIN_END)
    elif args.split == "val":
        start_i, end_i = _time_indices(times, Config.VAL_START, Config.VAL_END)
    else:
        start_i, end_i = _time_indices(times, Config.TEST_START, Config.TEST_END)

    # Static normalization (as in pipeline)
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

    # Build model
    model = _build_model(args.model_type)
    model.load_weights(args.model_path)

    # Accumulators
    pred_vals: Dict[str, List[float]] = {str(s): [] for s in st_ids}
    hr_vals: Dict[str, List[float]] = {str(s): [] for s in st_ids}
    obs_vals: Dict[str, List[float]] = {str(s): [] for s in st_ids}

    seq_len = Config.SEQ_LEN
    used = 0

    for i in range(start_i, end_i - seq_len, args.stride):
        last_idx = i + seq_len - 1
        t = times[last_idx]
        if t not in st_time_map:
            continue

        # Build inputs
        x_lr = lr.isel(time=slice(i, i + seq_len)).values
        if flip_lr_lon:
            x_lr = x_lr[:, :, ::-1, :]
        x_st = np.broadcast_to(static_norm[np.newaxis, ...], (seq_len, *static_norm.shape))

        x_lr_b = np.expand_dims(x_lr, 0).astype(np.float32)
        x_st_b = np.expand_dims(x_st, 0).astype(np.float32)

        y_pred = model((x_lr_b, x_st_b), training=False).numpy()[0, -1, :, :, 0]
        y_pred = y_pred * std_hr + mean_hr

        y_hr = hr.isel(time=last_idx).values
        if y_hr.ndim > 2:
            y_hr = y_hr[..., 0]
        y_hr = y_hr * std_hr + mean_hr

        obs_idx = st_time_map[t]
        obs_t = st_obs[obs_idx]

        for k, sid in enumerate(st_ids):
            sid = str(sid)
            obs_val = float(obs_t[k])
            pred_val = float(y_pred[st_i[k], st_j[k]])
            hr_val = float(y_hr[st_i[k], st_j[k]])
            if np.isfinite(obs_val):
                obs_vals[sid].append(obs_val)
                pred_vals[sid].append(pred_val)
                hr_vals[sid].append(hr_val)

        used += 1
        if args.max_samples and used >= args.max_samples:
            break

    def _metrics(pred, obs):
        pred = np.array(pred)
        obs = np.array(obs)
        if len(obs) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Corr": np.nan, "N": 0}
        err = pred - obs
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        corr = np.corrcoef(pred, obs)[0, 1] if len(obs) > 1 else np.nan
        return {"MAE": mae, "RMSE": rmse, "Bias": bias, "Corr": corr, "N": len(obs)}

    # Per-station metrics
    rows = []
    for sid in st_ids:
        sid = str(sid)
        m_pred = _metrics(pred_vals[sid], obs_vals[sid])
        m_hr = _metrics(hr_vals[sid], obs_vals[sid])
        rows.append({
            "station_id": sid,
            "MAE_model": m_pred["MAE"],
            "RMSE_model": m_pred["RMSE"],
            "Bias_model": m_pred["Bias"],
            "Corr_model": m_pred["Corr"],
            "MAE_urbclim": m_hr["MAE"],
            "RMSE_urbclim": m_hr["RMSE"],
            "Bias_urbclim": m_hr["Bias"],
            "Corr_urbclim": m_hr["Corr"],
            "N": m_pred["N"],
        })

    df = pd.DataFrame(rows)
    per_station_path = os.path.join(args.out_dir, "stations_eval_per_station.csv")
    df.to_csv(per_station_path, index=False)

    # Summary (global)
    all_obs = np.concatenate([obs_vals[str(s)] for s in st_ids if len(obs_vals[str(s)]) > 0])
    all_pred = np.concatenate([pred_vals[str(s)] for s in st_ids if len(pred_vals[str(s)]) > 0])
    all_hr = np.concatenate([hr_vals[str(s)] for s in st_ids if len(hr_vals[str(s)]) > 0])

    sum_pred = _metrics(all_pred, all_obs)
    sum_hr = _metrics(all_hr, all_obs)

    summary = pd.DataFrame([{
        "split": args.split,
        "MAE_model": sum_pred["MAE"],
        "RMSE_model": sum_pred["RMSE"],
        "Bias_model": sum_pred["Bias"],
        "Corr_model": sum_pred["Corr"],
        "MAE_urbclim": sum_hr["MAE"],
        "RMSE_urbclim": sum_hr["RMSE"],
        "Bias_urbclim": sum_hr["Bias"],
        "Corr_urbclim": sum_hr["Corr"],
        "N": sum_pred["N"],
        "samples_used": used,
    }])
    summary_path = os.path.join(args.out_dir, "stations_eval_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(all_obs, all_pred, s=5, alpha=0.4)
    ax[0].set_title("Model vs Stations")
    ax[0].set_xlabel("Stations (°C)")
    ax[0].set_ylabel("Model (°C)")

    ax[1].scatter(all_obs, all_hr, s=5, alpha=0.4, color="orange")
    ax[1].set_title("UrbClim vs Stations")
    ax[1].set_xlabel("Stations (°C)")
    ax[1].set_ylabel("UrbClim (°C)")

    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "stations_eval_scatter.png")
    fig.savefig(fig_path, dpi=200)

    print(f"✅ Per-station table: {per_station_path}")
    print(f"✅ Summary table: {summary_path}")
    print(f"✅ Scatter: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
