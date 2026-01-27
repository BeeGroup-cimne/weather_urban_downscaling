#!/usr/bin/env python
"""
Diagnostic script to verify temporal alignment between datasets.
Run this to troubleshoot "Sin coincidencia temporal" errors.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import xarray as xr
import pandas as pd
from config.config import Config

# Verificar rango temporal de todos los datasets
print("=" * 60)
print("DIAGNÓSTICO TEMPORAL COMPLETO")
print("=" * 60)
print(f"📂 Directorio base: {Config.BASE_DIR}")

# Path for station data (from environment or default)
PATH_STATIONS = os.environ.get(
    'WEATHER_STATIONS_PATH', 
    os.path.join(PROJECT_ROOT, 'data', 'raw', 'weather_stations.zarr')
)

# 1. Datos de estaciones (fuente para HR)
print("\n📍 ESTACIONES METEOROLÓGICAS:")
times = None
try:
    if os.path.exists(PATH_STATIONS):
        ds_stations = xr.open_dataset(PATH_STATIONS)
        times = pd.to_datetime(ds_stations.time.values)
        print(f"   Archivo: {PATH_STATIONS}")
        print(f"   Rango: {times.min()} -> {times.max()}")
        print(f"   Total timesteps: {len(times)}")
    else:
        print(f"   ⚠️ Archivo no encontrado: {PATH_STATIONS}")
        print(f"   Define WEATHER_STATIONS_PATH o copia los datos a data/raw/")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Datos LR (ERA5)
print("\n🌍 ERA5 (Low Resolution):")
times_lr = None
try:
    if os.path.exists(Config.PATH_LR):
        ds_lr = xr.open_dataset(
            Config.PATH_LR,
            engine='cfgrib',
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}, "errors": "ignore"}
        )
        
        # Manejar step si existe
        if 'step' in ds_lr.coords:
            ds_lr = ds_lr.stack(combined_time=('time', 'step'))
            ds_lr = ds_lr.swap_dims({'combined_time': 'valid_time'})
            ds_lr = ds_lr.rename({'valid_time': 'time'})
        elif 'valid_time' in ds_lr.coords:
            ds_lr = ds_lr.rename({'valid_time': 'time'})
        
        times_lr = pd.to_datetime(ds_lr.time.values)
        print(f"   Archivo: {Config.PATH_LR}")
        print(f"   Rango: {times_lr.min()} -> {times_lr.max()}")
        print(f"   Total timesteps: {len(times_lr)}")
    else:
        print(f"   ⚠️ Archivo no encontrado: {Config.PATH_LR}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Datos HR actuales
print("\n🎯 HR ACTUAL (estaciones_interpoladas_final.nc):")
times_hr = None
try:
    if os.path.exists(Config.PATH_HR):
        ds_hr = xr.open_dataset(Config.PATH_HR)
        times_hr = pd.to_datetime(ds_hr.time.values)
        print(f"   Archivo: {Config.PATH_HR}")
        print(f"   Rango: {times_hr.min()} -> {times_hr.max()}")
        print(f"   Total timesteps: {len(times_hr)}")
    else:
        print(f"   ⚠️ Archivo no encontrado: {Config.PATH_HR}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("DIAGNÓSTICO DE SOLAPAMIENTO:")
print("=" * 60)

# Verificar intersección
if times_hr is not None and times_lr is not None:
    intersection = set(times_hr.floor('h')) & set(times_lr.floor('h'))
    print(f"   Timesteps HR: {len(times_hr)}")
    print(f"   Timesteps LR: {len(times_lr)}")
    print(f"   Intersección: {len(intersection)} timesteps")
    
    if len(intersection) == 0:
        print("\n   ⚠️  NO HAY SOLAPAMIENTO TEMPORAL")
        
        # Verificar si estaciones cubren el período LR
        if times is not None:
            stations_intersect = set(times.floor('h')) & set(times_lr.floor('h'))
            print(f"\n   📍 Intersección Estaciones ∩ ERA5: {len(stations_intersect)} timesteps")
            
            if len(stations_intersect) > 0:
                print(f"   ✅ Las estaciones SÍ cubren parte del período ERA5")
                print(f"      Regenera el archivo HR con: python src/data/hourly_generator.py")
            else:
                print(f"   ❌ Las estaciones NO cubren el período ERA5")
                print(f"      Necesitas datos HR para el período 2010-2015")
    else:
        print(f"\n   ✅ HAY SOLAPAMIENTO TEMPORAL")
        sorted_intersection = sorted(intersection)
        print(f"      Desde: {sorted_intersection[0]}")
        print(f"      Hasta: {sorted_intersection[-1]}")
else:
    print("   ❌ No se pudo calcular intersección (faltan datos)")
