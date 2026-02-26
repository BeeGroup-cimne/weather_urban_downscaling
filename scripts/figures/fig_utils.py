#!/usr/bin/env python3
"""
Small utilities for paper figure scripts.
Keeps outputs consistent and avoids duplicated boilerplate.
"""

import os
from datetime import datetime
from typing import Iterable, Optional, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def default_fig_dir() -> str:
    return os.path.join("experiments", "figures")


def safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception as exc:
        print(f"❌ matplotlib not available: {exc}")
        print("Install it with: pip install matplotlib")
        return False


def safe_import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception as exc:
        print(f"⚠️ cartopy not available: {exc}")
        print("Install it with: pip install cartopy")
        return None, None


def parse_percentile_range(text: str, default: Tuple[float, float] = (2.0, 98.0)) -> Tuple[float, float]:
    try:
        parts = [float(p.strip()) for p in str(text).split(",")]
        if len(parts) != 2:
            raise ValueError("Need two values")
        low, high = parts
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("Invalid percentile range")
        return low, high
    except Exception:
        print(f"⚠️ Invalid percentile range '{text}'. Using default {default[0]},{default[1]}.")
        return default


def robust_limits(
    arrays: Iterable[np.ndarray],
    pct_low: float = 2.0,
    pct_high: float = 98.0,
    hard_min: Optional[float] = None,
    hard_max: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    values = []
    for arr in arrays:
        if arr is None:
            continue
        vec = np.asarray(arr).astype(np.float32).ravel()
        vec = vec[np.isfinite(vec)]
        if vec.size:
            values.append(vec)

    if not values:
        return hard_min, hard_max

    merged = np.concatenate(values)

    vmin = hard_min
    vmax = hard_max
    if vmin is None:
        vmin = float(np.nanpercentile(merged, pct_low))
    if vmax is None:
        vmax = float(np.nanpercentile(merged, pct_high))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(merged))
        vmax = float(np.nanmax(merged))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None, None
    return vmin, vmax


def extract_lat_lon_2d_from_da(da):
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "x"]

    lat_name = next((d for d in da.dims if d in lat_candidates), None)
    lon_name = next((d for d in da.dims if d in lon_candidates), None)
    if lat_name is None or lon_name is None:
        return None, None

    if lat_name not in da.coords or lon_name not in da.coords:
        return None, None

    lat = np.asarray(da.coords[lat_name].values)
    lon = np.asarray(da.coords[lon_name].values)

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d

    if lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
        return lat, lon

    return None, None
