import xarray as xr
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

def inspect_times():
    print("Iniciando inspección de tiempos...")
    
    # 1. Load HR Data
    print(f"\nCargando HR: {Config.PATH_HR}")
    try:
        ds_hr = xr.open_dataset(Config.PATH_HR)
        print("✅ HR cargado exitosamente.")
        print(f"Dimensiones HR: {ds_hr.dims}")
        if 'time' in ds_hr.coords:
            times_hr = ds_hr.time.values
            print(f"Tipo de datos tiempo HR: {times_hr.dtype}")
            print(f"Rango HR: {pd.to_datetime(times_hr.min())} - {pd.to_datetime(times_hr.max())}")
            print(f"Primeros 5 HR: {times_hr[:5]}")
            print(f"Últimos 5 HR: {times_hr[-5:]}")
            print(f"Total steps HR: {len(times_hr)}")
        else:
            print("❌ No se encontró la coordenada 'time' en HR.")
            print(f"Coords disponibles: {list(ds_hr.coords)}")
    except Exception as e:
        print(f"❌ Error cargando HR: {e}")

    # 2. Load LR Data
    print(f"\nCargando LR: {Config.PATH_LR}")
    try:
        # Use simple open_dataset or open_mfdataset depending on file type (Grib often needs engine='cfgrib')
        # Config uses grib, so we might need cfgrib.
        # However, data_loader.py uses open_mfdataset with engine="cfgrib".
        # Let's try basic open_dataset first, if it fails, try cfgrib.
        try:
            ds_lr = xr.open_dataset(Config.PATH_LR, engine="cfgrib", backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}, "errors": "ignore"})
        except:
             ds_lr = xr.open_dataset(Config.PATH_LR, engine="cfgrib")

        print("✅ LR cargado exitosamente.")
        
        # Logic from data_loader to potential fix time dims
        if 'step' in ds_lr.coords:
            print("Aplicando fix de 'step' a LR para ver tiempos reales...")
            ds_lr = ds_lr.stack(combined_time=('time', 'step'))
            ds_lr = ds_lr.dropna('combined_time', how='all')
            ds_lr = ds_lr.swap_dims({'combined_time': 'valid_time'})
            ds_lr = ds_lr.drop_vars(['time', 'step', 'combined_time'], errors='ignore')
            ds_lr = ds_lr.rename({'valid_time': 'time'})
            ds_lr = ds_lr.sortby('time')
        elif 'valid_time' in ds_lr.coords:
             if 'time' in ds_lr.dims and ds_lr.coords['valid_time'].ndim == 1:
                ds_lr = ds_lr.swap_dims({'time': 'valid_time'})
                ds_lr = ds_lr.rename({'valid_time': 'time'})
             elif 'valid_time' not in ds_lr.dims:
                 ds_lr = ds_lr.rename({'valid_time': 'time'})
        
        # Deduplicate
        _, index = np.unique(ds_lr['time'], return_index=True)
        ds_lr = ds_lr.isel(time=index)

        print(f"Dimensiones LR: {ds_lr.dims}")
        if 'time' in ds_lr.coords:
            times_lr = ds_lr.time.values
            print(f"Tipo de datos tiempo LR: {times_lr.dtype}")
            print(f"Rango LR: {pd.to_datetime(times_lr.min())} - {pd.to_datetime(times_lr.max())}")
            print(f"Primeros 5 LR: {times_lr[:5]}")
            print(f"Total steps LR: {len(times_lr)}")
        else:
            print("❌ No se encontró la coordenada 'time' en LR.")
    except Exception as e:
        print(f"❌ Error cargando LR: {e}")

if __name__ == "__main__":
    inspect_times()
