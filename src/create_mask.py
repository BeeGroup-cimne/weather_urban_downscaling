import numpy as np
import xarray as xr
import pandas as pd
import sys

def create_station_mask_from_csv(csv_path, reference_nc_path, output_path):
    """Genera máscara binaria desde CSV de estaciones"""
    print(f"📍 Creando máscara desde {csv_path}...")
    try:
        ds = xr.open_dataset(reference_nc_path)
        grid_lat = ds.latitude.values
        grid_lon = ds.longitude.values
        H, W = grid_lat.shape
        
        df = pd.read_csv(csv_path) # Requiere cols 'lat', 'lon'
        mask = np.zeros((H, W), dtype=np.float32)
        
        for _, row in df.iterrows():
            dist = (grid_lat - row['lat'])**2 + (grid_lon - row['lon'])**2
            idx = np.unravel_index(np.argmin(dist), (H, W))
            mask[idx] = 1.0
            
        np.save(output_path, mask)
        print(f"✅ Máscara guardada en {output_path}. Estaciones activas: {int(mask.sum())}")
    except Exception as e:
        print(f"❌ Error creando máscara: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python src/create_mask.py estaciones.csv referencia.nc salida.npy")
    else:
        create_station_mask_from_csv(sys.argv[1], sys.argv[2], sys.argv[3])
