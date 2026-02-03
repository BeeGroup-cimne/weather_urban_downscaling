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
    
    print("✅ Cache limpio. Todo listo para entrenar.")
    sys.exit(0)


if __name__ == "__main__":
    main()
