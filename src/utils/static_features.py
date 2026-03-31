"""Utilities for static feature schema, ordering and cache metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_index_lat_lon(index_values: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse index keys formatted as '<lat>_<lon>' into numeric arrays."""
    lats: List[float] = []
    lons: List[float] = []
    for raw in index_values:
        lat_s, lon_s = str(raw).split("_", 1)
        lats.append(float(lat_s))
        lons.append(float(lon_s))
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def select_static_feature_names(ds_static, cfg) -> Tuple[List[str], List[str]]:
    """
    Return ordered static feature names according to config schema.

    If requested features are missing:
    - strict mode: raises ValueError
    - non-strict mode: keeps available requested features and reports missing
    """
    available = [str(v) for v in ds_static.data_vars.keys()]
    requested = [str(v) for v in getattr(cfg, "STATIC_FEATURES", available)]
    missing = [v for v in requested if v not in available]
    selected = [v for v in requested if v in available]

    strict = bool(getattr(cfg, "STATIC_SCHEMA_STRICT", False))
    if missing and strict:
        raise ValueError(
            f"STATIC_SCHEMA_STRICT=1 and required static features are missing: {missing}. "
            f"Available: {available}"
        )
    return selected, missing


def static_cache_meta_path(static_cache_path: str) -> Path:
    return Path(static_cache_path).with_suffix(".meta.json")


def write_static_cache_meta(static_cache_path: str, payload: Dict[str, Any]) -> None:
    meta_path = static_cache_meta_path(static_cache_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def read_static_cache_meta(static_cache_path: str) -> Dict[str, Any] | None:
    meta_path = static_cache_meta_path(static_cache_path)
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def robust_unit_scale(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Scale a numeric array to [0, 1] robustly using percentiles."""
    x = np.asarray(arr, dtype=np.float32)
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(fin, p_low))
    hi = float(np.percentile(fin, p_high))
    if hi <= lo:
        lo = float(np.min(fin))
        hi = float(np.max(fin))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return y.astype(np.float32)


def to_json_str(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=True)
