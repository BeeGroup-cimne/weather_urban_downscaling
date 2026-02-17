#!/usr/bin/env python3
"""
Quick data health check before training.
Validates required files, shapes, and NaN presence in the Zarr cache.
"""

import os
import sys
import numpy as np
import xarray as xr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config


def check_exists(path, is_dir=False):
    if is_dir:
        return os.path.isdir(path)
    return os.path.isfile(path)

def latest_mtime(path):
    if not path:
        return 0
    if os.path.isfile(path):
        return os.path.getmtime(path)
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, ".zmetadata"),
            os.path.join(path, ".zgroup"),
            os.path.join(path, ".zattrs"),
        ]
        mtimes = [os.path.getmtime(path)]
        for c in candidates:
            if os.path.exists(c):
                mtimes.append(os.path.getmtime(c))
        return max(mtimes)
    return 0


def main():
    required_files = [
        (Config.PATH_HR, False),
        (Config.PATH_LR, False),
        (Config.PATH_STATIC, True),
        (Config.PATH_CACHE, True),
        (Config.STATS_PATH, False),
        (Config.STATIC_CACHE_PATH, False),
    ]
    
    missing = []
    for path, is_dir in required_files:
        if path and not check_exists(path, is_dir=is_dir):
            missing.append(path)
    
    if missing:
        print("❌ Faltan archivos requeridos:")
        for p in missing:
            print(f"   - {p}")
        sys.exit(1)

    # Cache freshness check
    cache_mtime = latest_mtime(Config.PATH_CACHE)
    raw_mtime = max(
        latest_mtime(Config.PATH_HR),
        latest_mtime(Config.PATH_LR),
        latest_mtime(Config.PATH_STATIC),
    )
    stats_mtime = latest_mtime(Config.STATS_PATH)
    static_cache_mtime = latest_mtime(Config.STATIC_CACHE_PATH)

    if raw_mtime > cache_mtime:
        print("⚠️ Cache Zarr parece desactualizado respecto a datos crudos.")
        print("   Sugerencia: borrar cache y re-ejecutar ETL.")
    if stats_mtime < cache_mtime:
        print("⚠️ Stats parecen más antiguos que el cache.")
        print("   Sugerencia: recomputar stats (o regenerar cache).")
    if static_cache_mtime < latest_mtime(Config.PATH_STATIC):
        print("⚠️ Cache estático es más antiguo que el Zarr estático.")
        print("   Sugerencia: borrar static_processed.npy y regenerar.")
    
    # Load Zarr and check NaNs
    print("✅ Archivos básicos presentes. Revisando cache Zarr...")
    z = xr.open_zarr(Config.PATH_CACHE, consolidated=True)
    
    hr_nan = z["hr_target"].isnull().any().compute().item()
    lr_nan = z["lr_input"].isnull().any().compute().item()
    
    print(f"   HR NaN: {hr_nan}")
    print(f"   LR NaN: {lr_nan}")
    
    if hr_nan or lr_nan:
        print("❌ Cache contiene NaNs. Reprocesa el ETL.")
        sys.exit(2)

    # Static cache NaN check
    try:
        static = np.load(Config.STATIC_CACHE_PATH)
        st_nan = np.isnan(static).any() or np.isinf(static).any()
        print(f"   Static NaN/Inf: {st_nan}")
        if st_nan:
            print("❌ Static cache contiene NaNs/Infs. Reprocesa estáticos.")
            sys.exit(3)
    except Exception as e:
        print(f"⚠️ No se pudo verificar static cache: {e}")

    # Orientation sanity check
    try:
        hr = z["hr_target"].isel(time=0)
        lr = z["lr_input"].isel(time=0)

        if hr.ndim > 2:
            hr = hr[..., 0]
        if lr.ndim > 2:
            lr = lr[..., 0]

        hr_np = hr.values
        lr_np = lr.values

        import tensorflow as tf
        lr_up = tf.image.resize(lr_np[..., None], hr_np.shape, method="bilinear").numpy()[..., 0]

        def _corr(a, b):
            a = a.flatten()
            b = b.flatten()
            if np.std(a) == 0 or np.std(b) == 0:
                return -np.inf
            return np.corrcoef(a, b)[0, 1]

        candidates = {
            "as_is": lr_up,
            "rot90": np.rot90(lr_up, 1),
            "rot180": np.rot90(lr_up, 2),
            "rot270": np.rot90(lr_up, 3),
            "flipud": np.flipud(lr_up),
            "fliplr": np.fliplr(lr_up),
            "transpose": lr_up.T,
            "transpose_flipud": np.flipud(lr_up.T),
            "transpose_fliplr": np.fliplr(lr_up.T),
        }

        best_name, best_score = "as_is", -np.inf
        for name, arr in candidates.items():
            score = _corr(arr, hr_np)
            if score > best_score:
                best_score = score
                best_name = name

        print(f"🧭 Orientation check: best LR->HR alignment = {best_name} (corr={best_score:.3f})")
        if best_name != "as_is":
            print("⚠️ Posible rotación/flip detectado. Revisa alineación espacial.")
    except Exception as e:
        print(f"⚠️ No se pudo verificar orientación: {e}")

    print("✅ Cache limpio. Todo listo para entrenar.")
    sys.exit(0)


if __name__ == "__main__":
    main()
