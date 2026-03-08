#!/usr/bin/env python3
"""Shared plotting helpers for evaluation/inference scripts."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def parse_percentile_range(raw: str, default: Tuple[float, float] = (2.0, 98.0)) -> Tuple[float, float]:
    """Parse a `low,high` percentile range with validation and fallback."""
    try:
        low_s, high_s = str(raw).split(",", 1)
        low = float(low_s.strip())
        high = float(high_s.strip())
        if 0.0 <= low < high <= 100.0:
            return low, high
    except Exception:
        pass
    return default


def robust_limits(
    arrays: Iterable[np.ndarray],
    pct_low: float = 2.0,
    pct_high: float = 98.0,
    hard_min: float | None = None,
    hard_max: float | None = None,
) -> Tuple[float, float]:
    """Compute robust display limits from finite values across arrays."""
    values = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue
        fin = a[np.isfinite(a)]
        if fin.size:
            values.append(fin.ravel())

    if values:
        stacked = np.concatenate(values, axis=0)
        vmin = float(np.nanpercentile(stacked, pct_low))
        vmax = float(np.nanpercentile(stacked, pct_high))
    else:
        vmin, vmax = 0.0, 1.0

    if hard_min is not None:
        vmin = float(hard_min)
    if hard_max is not None:
        vmax = float(hard_max)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = vmin + 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def extract_lat_lon_2d_from_da(da) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Return 2D latitude/longitude arrays aligned to the HR grid when available.

    Supports:
    - 2D coords: latitude/longitude (or lat/lon, latitude_2d/longitude_2d)
    - 1D coords: converted via meshgrid
    """
    coord_names_lat = ("latitude", "lat", "latitude_2d")
    coord_names_lon = ("longitude", "lon", "longitude_2d")

    lat = next((da.coords[n] for n in coord_names_lat if n in da.coords), None)
    lon = next((da.coords[n] for n in coord_names_lon if n in da.coords), None)

    if lat is None or lon is None:
        return None, None

    lat_v = np.asarray(lat.values)
    lon_v = np.asarray(lon.values)

    if lat_v.ndim == 2 and lon_v.ndim == 2:
        return lat_v, lon_v
    if lat_v.ndim == 1 and lon_v.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_v, lat_v)
        return lat2d, lon2d
    if lat_v.ndim == 2 and lon_v.ndim == 1:
        lon2d = np.broadcast_to(lon_v[None, :], lat_v.shape)
        return lat_v, lon2d
    if lon_v.ndim == 2 and lat_v.ndim == 1:
        lat2d = np.broadcast_to(lat_v[:, None], lon_v.shape)
        return lat2d, lon_v
    return None, None


def safe_import_cartopy():
    """Import cartopy lazily and safely for optional geo overlays."""
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore

        return ccrs, cfeature
    except Exception:
        return None, None

