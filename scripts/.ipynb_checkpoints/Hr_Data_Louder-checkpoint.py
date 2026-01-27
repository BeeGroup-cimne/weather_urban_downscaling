import xarray as xr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
import os

# CONFIGURACIÓN

PATH_STATIONS = '/Users/kerincardona/weather_urbclim_200801-201712.zarr'
PATH_GRID_REF = "/Users/kerincardona/Documents/weather_urban_downscaling/urbclim/tas_Barcelona_UrbClim_2008_03_v1.0.nc"
OUTPUT_FILE = "estaciones_interpoladas_final.nc"

# Configuración IDW
K_NEIGHBORS = 12
POWER = 2.0

def process_stations_final():
    print(f"🚀 Iniciando Pipeline Final (Conversión + IDW)...")
    
    # 1. Cargar Datos
    ds = xr.open_dataset(PATH_STATIONS, chunks={'time': 100})
    
    # Filtro temporal (3 meses)
    start_date = pd.to_datetime(ds.time.values[0])
    end_date = start_date + pd.DateOffset(months=3)
    ds_subset = ds.sel(time=slice(start_date, end_date))
    print(f"   ⏱️ Pasos de tiempo: {len(ds_subset.time)}")

    # 2. TRANSFORMACIÓN DE COORDENADAS
    print("🌍 Transformando coordenadas de estaciones (EPSG:3035 -> EPSG:4326)...")
    
    # Coordenadas originales (Metros)
    src_x = ds_subset['x'].isel(time=0).compute().values
    src_y = ds_subset['y'].isel(time=0).compute().values
    
    # Definir transformación: ETRS89-LAEA (3035) -> WGS84 (4326)
    # UrbClim suele usar 3035. Si fallara, probar EPSG:25831 (UTM31N).
    transformer = Transformer.from_crs("epsg:3035", "epsg:4326", always_xy=True)
    
    # Transformar (X, Y) -> (Lon, Lat)
    st_lon, st_lat = transformer.transform(src_x, src_y)
    
    print(f"   Coordenadas transformadas (Ejemplo):")
    print(f"   Original (m): {src_x[0]:.1f}, {src_y[0]:.1f}")
    print(f"   Final (deg) : {st_lon[0]:.4f}, {st_lat[0]:.4f}")

    # 3. CARGAR GRID TARGET
    print("🗺️  Leyendo rejilla objetivo...")
    ds_ref = xr.open_dataset(PATH_GRID_REF)
    
    target_lat = ds_ref['latitude'].values if 'latitude' in ds_ref else ds_ref['y'].values
    target_lon = ds_ref['longitude'].values if 'longitude' in ds_ref else ds_ref['x'].values
    
    if target_lat.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(target_lon, target_lat)
    else:
        grid_lat_2d, grid_lon_2d = target_lat, target_lon
        
    # Verificar solapamiento
    overlap_lat = (st_lat.min() < grid_lat_2d.max()) and (st_lat.max() > grid_lat_2d.min())
    overlap_lon = (st_lon.min() < grid_lon_2d.max()) and (st_lon.max() > grid_lon_2d.min())
    
    if not overlap_lat or not overlap_lon:
        print("❌ ERROR CRÍTICO: Incluso tras la conversión, no solapan.")
        print(f"   Estaciones Lat: {st_lat.min():.4f}-{st_lat.max():.4f}")
        print(f"   Grid Lat      : {grid_lat_2d.min():.4f}-{grid_lat_2d.max():.4f}")
        return
    else:
        print("✅ ¡Solapamiento exitoso! Las coordenadas ahora coinciden.")

    # 4. PREPARAR PUNTOS PARA IDW
    # KDTree necesita puntos (x, y) = (Lon, Lat) o (Lat, Lon)
    # IMPORTANTE: Usar el mismo orden. Usaremos (Lat, Lon)
    station_points = np.column_stack((st_lat, st_lon))
    grid_points = np.column_stack((grid_lat_2d.ravel(), grid_lon_2d.ravel()))

    # 5. CÁLCULO DE PESOS
    print("📐 Calculando pesos espaciales (KDTree)...")
    tree = cKDTree(station_points)
    dists, idxs = tree.query(grid_points, k=K_NEIGHBORS)
    
    # Evitar div/0
    dists = np.maximum(dists, 1e-9)
    weights = 1.0 / (dists ** POWER)
    # Normalizar
    weights_norm = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)

    # 6. INTERPOLACIÓN
    print("🔄 Interpolando series temporales...")
    
    BATCH_SIZE = 100 
    n_total_times = len(ds_subset.time)
    temp_dask = ds_subset['airTemperature'].chunk({'time': 100, 'weatherStation': -1})
    
    result_list = []
    
    for i in range(0, n_total_times, BATCH_SIZE):
        if i % 500 == 0: print(f"   Procesando batch {i}/{n_total_times}...", end='\r')
        
        t_slice = slice(i, min(i+BATCH_SIZE, n_total_times))
        
        # Cargar valores
        temps_batch = temp_dask[t_slice].compute().values
        
        # Rellenar NaNs en origen por si acaso
        row_means = np.nanmean(temps_batch, axis=1, keepdims=True)
        row_means = np.nan_to_num(row_means, nan=0.0)
        inds = np.where(np.isnan(temps_batch))
        temps_batch[inds] = np.take(row_means, inds[0])
        
        # Interpolación IDW
        neighbor_temps = temps_batch[:, idxs] 
        interpolated_flat = np.sum(neighbor_temps * weights_norm, axis=2)
        
        batch_out_3d = interpolated_flat.reshape((-1, grid_lat_2d.shape[0], grid_lat_2d.shape[1]))
        result_list.append(batch_out_3d)

    print("\n📦 Uniendo resultados...")
    full_array = np.concatenate(result_list, axis=0)

    # 7. GUARDAR
    local_output = OUTPUT_FILE
    if os.path.exists(local_output):
        try: os.remove(local_output)
        except: local_output = local_output.replace(".nc", "_v_final.nc")

    ds_final = xr.Dataset(
        data_vars={"t2m": (("time", "y", "x"), full_array.astype('float32'))},
        coords={
            "time": ds_subset.time.values,
            "latitude": (("y", "x"), grid_lat_2d),
            "longitude": (("y", "x"), grid_lon_2d)
        }
    )
    
    # Conversión final de unidades si fuera necesario (Kelvin -> Celsius ya parece estar hecho en input -0.6 a 5.8)
    # Si los datos de entrada son Celsius, perfecto. Si el modelo espera Kelvin, sumar 273.15 aquí.
    
    print(f"💾 Guardando: {local_output}")
    ds_final.to_netcdf(local_output)
    print("✅ ¡Pipeline Final Completado!")

if __name__ == "__main__":
    process_stations_final()