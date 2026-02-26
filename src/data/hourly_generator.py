import xarray as xr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
import os
import sys

# Añadir ruta del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config

# CONFIGURACIÓN - Rutas portables del proyecto
# 
# Archivo fuente de estaciones meteorológicas:
# - Se busca primero en data/raw/weather_stations.zarr
# - Si no existe, se usa la variable de entorno WEATHER_STATIONS_PATH
# - Fallback a un path por defecto (solo para desarrollo local)


PATH_STATIONS_DEFAULT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'weather_stations.zarr')
PATH_STATIONS = os.environ.get('WEATHER_STATIONS_PATH', PATH_STATIONS_DEFAULT)

# Archivo de referencia para la grilla (usamos el HR existente como plantilla)
PATH_GRID_REF = Config.PATH_HR
# Archivo de salida
OUTPUT_FILE = Config.PATH_HR

# Configuración IDW
K_NEIGHBORS = 12
POWER = 2.0

def process_stations_final(start_date_str=None, end_date_str=None):
    """
    Pipeline para interpolar estaciones meteorológicas a una grilla regular.
    
    Args:
        start_date_str: Fecha de inicio en formato 'YYYY-MM-DD' (opcional, usa todo si no se especifica)
        end_date_str: Fecha de fin en formato 'YYYY-MM-DD' (opcional)
    """
    print(f"🚀 Iniciando Pipeline Final (Conversión + IDW)...")
    
    # 1. Cargar Datos
    if not os.path.exists(PATH_STATIONS):
        print(f"❌ ERROR: No se encontró el archivo de estaciones: {PATH_STATIONS}")
        print("   Por favor, ajusta PATH_STATIONS a la ubicación correcta de tus datos de estaciones.")
        return
    
    # Intentar abrir con zarr primero, luego fallback
    try:
        ds = xr.open_zarr(PATH_STATIONS, chunks={'time': 100})
    except:
        ds = xr.open_dataset(PATH_STATIONS, chunks={'time': 100})
    
    # Filtro temporal - AHORA CONFIGURABLE
    if start_date_str and end_date_str:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        print(f"   ⏱️ Filtrando período: {start_date} -> {end_date}")
    else:
        # Usar todo el rango disponible
        start_date = pd.to_datetime(ds.time.values[0])
        end_date = pd.to_datetime(ds.time.values[-1])
        print(f"   ⏱️ Usando rango completo: {start_date} -> {end_date}")
    
    ds_subset = ds.sel(time=slice(start_date, end_date))
    print(f"   ⏱️ Pasos de tiempo a procesar: {len(ds_subset.time)}")

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
    ds_ref = None
    
    if os.path.exists(PATH_GRID_REF):
        ds_ref = xr.open_dataset(PATH_GRID_REF)
    else:
        # Fallback 1: usar Zarr estático si existe
        if os.path.exists(Config.PATH_STATIC):
            print(f"   ⚠️  {PATH_GRID_REF} no existe. Usando {Config.PATH_STATIC} como referencia.")
            ds_ref = xr.open_zarr(Config.PATH_STATIC)
        
        # Fallback 2: usar grilla LR (ERA5-Land GRIB)
        if ds_ref is None:
            lr_path = Config.PATH_LR
            if not os.path.exists(lr_path):
                # Buscar cualquier .grib en data/processed/era5land
                era5_dir = os.path.join(PROJECT_ROOT, "data", "processed", "era5land")
                if os.path.isdir(era5_dir):
                    gribs = [f for f in os.listdir(era5_dir) if f.endswith(".grib")]
                    if gribs:
                        lr_path = os.path.join(era5_dir, gribs[0])
            
            if os.path.exists(lr_path):
                print(f"   ⚠️  Usando grilla LR: {lr_path}")
                try:
                    ds_ref = xr.open_dataset(
                        lr_path,
                        engine="cfgrib",
                        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}, "errors": "ignore"}
                    )
                except Exception as e:
                    print(f"❌ ERROR abriendo GRIB LR: {e}")
                    ds_ref = None
    
    if ds_ref is None:
        print("❌ ERROR: No se encontró archivo de referencia NetCDF, GRIB LR ni Zarr Estático.")
        return

    # Detectar nombres de coordenadas en la referencia
    if 'latitude' in ds_ref.coords:
        target_lat = ds_ref['latitude'].values
        target_lon = ds_ref['longitude'].values
    elif 'lat' in ds_ref.coords and 'lon' in ds_ref.coords:
        target_lat = ds_ref['lat'].values
        target_lon = ds_ref['lon'].values
    elif 'y' in ds_ref.coords and 'x' in ds_ref.coords:
        target_lat = ds_ref['y'].values
        target_lon = ds_ref['x'].values
    elif 'index' in ds_ref:
        # Zarr estático con index plano "lat_lon"
        print("   ⚠️  Referencia estática sin coords 2D. Generando grilla regular 251x251 desde el bounding box.")
        indices = ds_ref['index'].values
        lats_idx = np.array([float(x.split('_')[0]) for x in indices])
        lons_idx = np.array([float(x.split('_')[1]) for x in indices])
        
        lat_min, lat_max = lats_idx.min(), lats_idx.max()
        lon_min, lon_max = lons_idx.min(), lons_idx.max()
        
        h, w = Config.HR_SHAPE
        lat_1d = np.linspace(lat_min, lat_max, h)
        lon_1d = np.linspace(lon_min, lon_max, w)
        grid_lon_2d, grid_lat_2d = np.meshgrid(lon_1d, lat_1d)
        target_lat, target_lon = grid_lat_2d, grid_lon_2d
    else:
        # Fallback a variables si no están en coords
        if 'latitude' in ds_ref:
            target_lat = ds_ref['latitude'].values
            target_lon = ds_ref['longitude'].values
        elif 'lat' in ds_ref:
            target_lat = ds_ref['lat'].values
            target_lon = ds_ref['lon'].values
        else:
            target_lat = ds_ref['y'].values
            target_lon = ds_ref['x'].values
    
    if isinstance(target_lat, np.ndarray) and target_lat.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(target_lon, target_lat)
    elif isinstance(target_lat, np.ndarray) and target_lat.ndim == 2:
        grid_lat_2d, grid_lon_2d = target_lat, target_lon
    else:
        # En caso extremo, forzamos a grilla regular con HR_SHAPE
        h, w = Config.HR_SHAPE
        lat_1d = np.linspace(target_lat.min(), target_lat.max(), h)
        lon_1d = np.linspace(target_lon.min(), target_lon.max(), w)
        grid_lon_2d, grid_lat_2d = np.meshgrid(lon_1d, lat_1d)
        
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
    
    # Detectar variable de temperatura
    hr_var = None
    for v in ['airTemperature', 'tas', 't2m', 'temp']:
        if v in ds_subset:
            hr_var = v
            break
    if hr_var is None:
        hr_var = list(ds_subset.data_vars)[0]
        print(f"   ⚠️ Variable estándar no encontrada. Usando '{hr_var}'")
    
    print(f"   🌡️ Usando variable: {hr_var}")
    temp_dask = ds_subset[hr_var].chunk({'time': 100})
    
    # Detectar nombre de dimensión de estaciones
    st_dim = 'weatherStation' if 'weatherStation' in temp_dask.dims else 'station'
    if st_dim not in temp_dask.dims:
        st_dim = temp_dask.dims[-1] # Asumimos que la última es la de estaciones
    
    temp_dask = temp_dask.chunk({st_dim: -1})
    
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
    import argparse
    parser = argparse.ArgumentParser(description='Interpola estaciones meteorológicas a grilla regular')
    parser.add_argument('--start', type=str, help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Fecha fin (YYYY-MM-DD)')
    args = parser.parse_args()
    
    process_stations_final(args.start, args.end)
