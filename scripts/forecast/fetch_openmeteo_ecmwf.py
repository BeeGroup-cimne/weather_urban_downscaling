#!/usr/bin/env python3
"""Fetch Open-Meteo ECMWF forecast data and align it to the trained LR tensor."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OPENMETEO_ECMWF_ENDPOINT = "https://api.open-meteo.com/v1/ecmwf"
DEFAULT_HOURLY = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "shortwave_radiation",
]
TRAINED_VARIABLES = ["u10", "v10", "d2m", "t2m", "lai_hv", "lai_lv", "tp", "ssrd", "fal"]


def _load_stats(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise SystemExit(f"stats file not found: {path}")
    stats = np.load(path, allow_pickle=True)
    mean_lr = stats["mean_lr"].astype(np.float32)
    std_lr = stats["std_lr"].astype(np.float32)
    names = [str(v) for v in stats["lr_var_names"].tolist()]
    return mean_lr, std_lr, names


def _read_centroids(path: Path, lat_order: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"grid_y", "grid_x", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"centroid CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["grid_y"] = df["grid_y"].astype(int)
    df["grid_x"] = df["grid_x"].astype(int)
    ascending_lat = lat_order == "south_to_north"
    return df.sort_values(["latitude", "longitude"], ascending=[ascending_lat, True]).reset_index(drop=True)


def _build_url(args: argparse.Namespace, centroids: pd.DataFrame) -> str:
    params = {
        "latitude": ",".join(f"{v:.6f}" for v in centroids["latitude"].to_numpy()),
        "longitude": ",".join(f"{v:.6f}" for v in centroids["longitude"].to_numpy()),
        "hourly": ",".join(DEFAULT_HOURLY),
        "forecast_days": str(args.forecast_days),
        "timezone": "UTC",
        "cell_selection": "nearest",
        "elevation": ",".join("nan" for _ in range(len(centroids))),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    if args.past_days:
        params["past_days"] = str(args.past_days)
    return f"{args.endpoint}?{urllib.parse.urlencode(params)}"


def _fetch_json(url: str, timeout: float) -> list[dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "weather-urban-downscaling/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("error"):
        raise SystemExit(f"Open-Meteo error: {payload.get('reason', payload)}")
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise SystemExit("Unexpected Open-Meteo response shape")


def _as_hourly_frame(location: dict) -> pd.DataFrame:
    hourly = location.get("hourly")
    if not hourly:
        raise SystemExit(f"missing hourly payload for location: {location}")
    df = pd.DataFrame(hourly)
    if "time" not in df:
        raise SystemExit("hourly payload does not include time")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    return df


def _wind_components(speed_ms: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction_rad = np.deg2rad(direction_deg.astype(np.float32))
    speed_ms = speed_ms.astype(np.float32)
    u = -speed_ms * np.sin(direction_rad)
    v = -speed_ms * np.cos(direction_rad)
    return u.astype(np.float32), v.astype(np.float32)


def _to_trained_channels(
    frames: list[pd.DataFrame],
    centroids: pd.DataFrame,
    mean_lr: np.ndarray,
    std_lr: np.ndarray,
    var_names: list[str],
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if var_names != TRAINED_VARIABLES:
        raise SystemExit(f"unexpected LR variable order: {var_names}; expected {TRAINED_VARIABLES}")

    times = pd.DatetimeIndex(frames[0]["time"])
    y_size = int(centroids["grid_y"].max()) + 1
    x_size = int(centroids["grid_x"].max()) + 1
    raw = np.zeros((len(times), y_size, x_size, len(var_names)), dtype=np.float32)

    neutral = {name: float(mean_lr[i]) for i, name in enumerate(var_names)}
    for i, frame in enumerate(frames):
        if not pd.DatetimeIndex(frame["time"]).equals(times):
            raise SystemExit("Open-Meteo returned inconsistent time axes across centroids")
        row = centroids.iloc[i]
        y = int(row["grid_y"])
        x = int(row["grid_x"])

        wind_u, wind_v = _wind_components(
            frame["wind_speed_10m"].to_numpy(dtype=np.float32),
            frame["wind_direction_10m"].to_numpy(dtype=np.float32),
        )

        raw[:, y, x, var_names.index("u10")] = wind_u
        raw[:, y, x, var_names.index("v10")] = wind_v
        raw[:, y, x, var_names.index("d2m")] = frame["dew_point_2m"].to_numpy(dtype=np.float32)
        raw[:, y, x, var_names.index("t2m")] = frame["temperature_2m"].to_numpy(dtype=np.float32)
        raw[:, y, x, var_names.index("tp")] = frame["precipitation"].to_numpy(dtype=np.float32) / 1000.0
        raw[:, y, x, var_names.index("ssrd")] = frame["shortwave_radiation"].to_numpy(dtype=np.float32) * 3600.0

        for name in ["lai_hv", "lai_lv", "fal"]:
            raw[:, y, x, var_names.index(name)] = neutral[name]

    std = np.where(std_lr == 0.0, 1.0, std_lr).astype(np.float32)
    normalized = (raw - mean_lr.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    return times, normalized.astype(np.float32)


def _write_outputs(
    out_zarr: Path,
    out_manifest: Path,
    centroids: pd.DataFrame,
    times: pd.DatetimeIndex,
    lr_input: np.ndarray,
    var_names: list[str],
    request_url: str,
) -> None:
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    y_coords = (
        centroids.sort_values("grid_y")
        .drop_duplicates("grid_y")
        .sort_values("grid_y")["latitude"]
        .to_numpy(dtype=np.float32)
    )
    x_coords = (
        centroids.sort_values("grid_x")
        .drop_duplicates("grid_x")
        .sort_values("grid_x")["longitude"]
        .to_numpy(dtype=np.float32)
    )

    ds = xr.Dataset(
        data_vars={
            "lr_input": (
                ("time", "latitude_lr", "longitude_lr", "variable"),
                lr_input,
            )
        },
        coords={
            "time": times.to_numpy(),
            "latitude_lr": y_coords,
            "longitude_lr": x_coords,
            "variable": var_names,
        },
        attrs={
            "source": "Open-Meteo ECMWF API",
            "endpoint": OPENMETEO_ECMWF_ENDPOINT,
            "cell_selection": "nearest",
            "elevation": "nan",
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "normalization": "training mean/std from stats_config.npz",
            "request_url": request_url,
        },
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=True)

    manifest = {
        "status": "ok",
        "source": "open-meteo-ecmwf",
        "out_zarr": str(out_zarr),
        "time_start": times[0].isoformat(),
        "time_end": times[-1].isoformat(),
        "n_times": int(len(times)),
        "shape": list(lr_input.shape),
        "variables": var_names,
        "filled_neutral_channels": ["lai_hv", "lai_lv", "fal"],
    }
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", default="config/forecast/amb_era5land_centroids.csv")
    parser.add_argument("--stats", default="data/processed/stats_config.npz")
    parser.add_argument("--out-zarr", default="data/forecast/openmeteo_ecmwf_lr.zarr")
    parser.add_argument("--out-manifest", default="data/forecast/openmeteo_ecmwf_manifest.json")
    parser.add_argument("--endpoint", default=OPENMETEO_ECMWF_ENDPOINT)
    parser.add_argument("--forecast-days", type=int, default=7)
    parser.add_argument("--past-days", type=int, default=0)
    parser.add_argument("--lat-order", choices=["south_to_north", "north_to_south"], default="south_to_north")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    centroids = _read_centroids(Path(args.centroids), args.lat_order)
    mean_lr, std_lr, var_names = _load_stats(Path(args.stats))
    url = _build_url(args, centroids)

    if args.dry_run:
        print(url)
        return 0

    payloads = _fetch_json(url, args.timeout_seconds)
    if len(payloads) != len(centroids):
        raise SystemExit(f"expected {len(centroids)} location payloads, got {len(payloads)}")

    frames = [_as_hourly_frame(location) for location in payloads]
    times, lr_input = _to_trained_channels(frames, centroids, mean_lr, std_lr, var_names)
    _write_outputs(Path(args.out_zarr), Path(args.out_manifest), centroids, times, lr_input, var_names, url)

    print(f"forecast zarr: {args.out_zarr}")
    print(f"manifest: {args.out_manifest}")
    print(f"shape: {lr_input.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
