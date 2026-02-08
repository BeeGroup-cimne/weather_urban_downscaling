
# MOTOR DE DATOS BIG DATA (Scalable Pipeline)

import xarray as xr
import numpy as np
import glob
import shutil
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.interpolate import NearestNDInterpolator
import os
import pandas as pd
from config.config import Config


class BigDataPipeline:
    def __init__(self, config):
        self.cfg = config
        self.cache_dir = self.cfg.PATH_CACHE
        self.stats_path = self.cfg.STATS_PATH
        
        self.ds_static_single = None 
        print(f"📂 Cache configurado en: {self.cache_dir}")
        print(f"📊 Stats configurado en: {self.stats_path}")
        
    def process_static_data(self):
        """
        Procesa datos estáticos (Elevación, etc.).
        CORREGIDO: Detecta automáticamente si las dimensiones son y/x (UrbClim) o lat/lon.
        """
        print("🗺️ Processing Static Data...")
        
        # Definimos ruta del caché estático
        static_cache_path = self.cfg.STATIC_CACHE_PATH

        # Si ya existe cache Zarr, preferimos su grilla HR (es la que usa el entrenamiento)
        cache_hr = None
        if os.path.exists(self.cache_dir):
            try:
                z = xr.open_zarr(self.cache_dir, consolidated=True)
                if "hr_target" in z:
                    cache_hr = z["hr_target"]
            except Exception:
                cache_hr = None

        # --- FIX: DETECCIÓN ROBUSTA DE DIMENSIONES ---
        # Intentar obtener dimensiones desde el cache primero (sin abrir PATH_HR)
        dim_y = dim_x = None
        if cache_hr is not None and 'latitude' in cache_hr.coords and 'longitude' in cache_hr.coords:
            lat_c = cache_hr['latitude'].values
            lon_c = cache_hr['longitude'].values
            if getattr(lat_c, 'ndim', 1) == 1 and getattr(lon_c, 'ndim', 1) == 1:
                dim_y, dim_x = lat_c.shape[0], lon_c.shape[0]
            else:
                dim_y, dim_x = lat_c.shape
        elif cache_hr is not None and 'y' in cache_hr.sizes and 'x' in cache_hr.sizes:
            dim_y = cache_hr.sizes['y']
            dim_x = cache_hr.sizes['x']

        # Si el cache estático ya coincide, salir temprano sin abrir PATH_HR
        if dim_y is not None and dim_x is not None and os.path.exists(static_cache_path):
            print("🗺️ Cargando datos estáticos desde caché (.npy)...")
            cached = np.load(static_cache_path)
            if cached.shape[0] == dim_y and cached.shape[1] == dim_x:
                self.ds_static_single = cached
                print(f"   ✅ Static Data Shape Final: {self.ds_static_single.shape}")
                return
            print(f"   ⚠️ Cache estático desalineado {cached.shape[:2]} != ({dim_y}, {dim_x}). Recalculando...")

        print("🗺️ Procesando e Interpolando datos estáticos (Esto se hará solo una vez)...")
        
        # 1. Abrimos el dataset HR para leer dimensiones y grilla
        # Usamos open_dataset normal para asegurar acceso a todas las variables
        ds_hr = None
        try:
            ds_hr = xr.open_dataset(self.cfg.PATH_HR)
        except Exception as e:
            if cache_hr is None:
                raise
            print(f"⚠️ No se pudo abrir PATH_HR ({self.cfg.PATH_HR}). Usando grilla del cache. Error: {e}")

        # --- FIX: DETECCIÓN ROBUSTA DE DIMENSIONES ---
        # Preferir tamaño real del grid lat/lon (si existe y es 2D)
        if cache_hr is not None and 'latitude' in cache_hr.coords and 'longitude' in cache_hr.coords:
            lat_c = cache_hr['latitude'].values
            lon_c = cache_hr['longitude'].values
            if getattr(lat_c, 'ndim', 1) == 1 and getattr(lon_c, 'ndim', 1) == 1:
                dim_y, dim_x = lat_c.shape[0], lon_c.shape[0]
            else:
                dim_y, dim_x = lat_c.shape
        elif ds_hr is not None and 'latitude' in ds_hr.coords and hasattr(ds_hr['latitude'], 'ndim') and ds_hr['latitude'].ndim == 2:
            dim_y, dim_x = ds_hr['latitude'].shape
        elif ds_hr is not None and 'y' in ds_hr.sizes and 'x' in ds_hr.sizes:
            dim_y = ds_hr.sizes['y']
            dim_x = ds_hr.sizes['x']
        elif ds_hr is not None and 'latitude' in ds_hr.sizes:
            dim_y = ds_hr.sizes['latitude']
            dim_x = ds_hr.sizes['longitude']
        else:
            # Fallback por si acaso
            if ds_hr is None:
                raise RuntimeError("❌ No se pudo determinar la grilla HR (cache y PATH_HR no disponibles).")
            dim_y, dim_x = ds_hr.shape[-2], ds_hr.shape[-1]
            
        print(f" 📏 Dimensiones detectadas HR: {dim_y}x{dim_x}")

        if os.path.exists(static_cache_path):
            print("🗺️ Cargando datos estáticos desde caché (.npy)...")
            cached = np.load(static_cache_path)
            if cached.shape[0] == dim_y and cached.shape[1] == dim_x:
                self.ds_static_single = cached
                print(f"   ✅ Static Data Shape Final: {self.ds_static_single.shape}")
                return
            print(f"   ⚠️ Cache estático desalineado {cached.shape[:2]} != ({dim_y}, {dim_x}). Recalculando...")

        # --- GRID PARA INTERPOLACIÓN ---
        # Necesitamos los valores reales de latitud/longitud de cada pixel para saber dónde interpolar
        # Aunque las dimensiones se llamen 'y'/'x', las variables 'latitude'/'longitude' existen
        if cache_hr is not None and 'latitude' in cache_hr.coords and 'longitude' in cache_hr.coords:
            grid_y = cache_hr['latitude'].values
            grid_x = cache_hr['longitude'].values
        else:
            if ds_hr is None:
                raise RuntimeError("❌ No se pudo leer latitude/longitude para interpolación.")
            grid_y = ds_hr['latitude'].values
            grid_x = ds_hr['longitude'].values
        
        # Aplanar para interpolar
        if getattr(grid_y, 'ndim', 1) == 1 and getattr(grid_x, 'ndim', 1) == 1:
            yy, xx = np.meshgrid(grid_y, grid_x, indexing='ij')
            target_points = np.column_stack((yy.ravel(), xx.ravel()))
        else:
            target_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))
        
        # 2. Cargar Static Zarr (tus datos de elevación/landuse)
        ds_static = xr.open_zarr(self.cfg.PATH_STATIC)
        indices = ds_static['index'].values
        
        # Parsear coordenadas del Zarr estático
        lats = np.array([float(x.split('_')[0]) for x in indices])
        lons = np.array([float(x.split('_')[1]) for x in indices])
        points = np.column_stack((lats, lons))
        
        # 3. Interpolación
        static_layers = []
        for var in ds_static.data_vars:
            if var == 'index': continue
            values = ds_static[var].values
            interpolator = NearestNDInterpolator(points, values)
            interp_flat = interpolator(target_points)
            static_layers.append(interp_flat.reshape(dim_y, dim_x))
            
        self.ds_static_single = np.stack(static_layers, axis=-1).astype('float32')
        if np.isnan(self.ds_static_single).any():
            print("⚠️ Static data contiene NaNs. Rellenando con media por canal.")
            ch_mean = np.nanmean(self.ds_static_single, axis=(0, 1), keepdims=True)
            self.ds_static_single = np.where(
                np.isnan(self.ds_static_single), ch_mean, self.ds_static_single
            )
            self.ds_static_single = np.nan_to_num(self.ds_static_single, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"   ✅ Static Data Shape Final: {self.ds_static_single.shape}")

        print(f"💾 Guardando caché estático en {static_cache_path}...")
        # Aseguramos que la carpeta exista
        os.makedirs(os.path.dirname(static_cache_path), exist_ok=True)
        np.save(static_cache_path, self.ds_static_single)
        
        print(f"   ✅ Static Data Shape Final: {self.ds_static_single.shape}")

    def compute_global_stats(self, ds_hr_lazy, ds_lr_lazy):
        """Calcula estadísticas globales usando Dask (Sin RAM excesiva)"""
        if os.path.exists(self.stats_path):
            print("📊 Cargando estadísticas pre-calculadas...")
            stats = np.load(self.stats_path)
            return stats['mean_lr'], stats['std_lr'], stats['mean_hr'], stats['std_hr']
        
        print("🧮 Calculando estadísticas globales (Streamed)...")
        with ProgressBar():
            # Compute real values triggers calculation
            mean_hr = ds_hr_lazy["tas_C"].mean().compute().item()
            std_hr = ds_hr_lazy["tas_C"].std().compute().item()
            mean_lr = ds_lr_lazy["t2m"].mean().compute().item()
            std_lr = ds_lr_lazy["t2m"].std().compute().item()

        np.savez(self.stats_path, mean_lr=mean_lr, std_lr=std_lr, mean_hr=mean_hr, std_hr=std_hr)
        return mean_lr, std_lr, mean_hr, std_hr

    def run_etl_process(self):
        """
        Fase 1: Transforma RAW -> ZARR (Soporte Multi-Canal + Interpolación 'Nearest' Marítima).
        """
        if os.path.exists(self.cache_dir):
            print(f"⚡ Cache Zarr detectado. (IMPORTANTE: BORRA {self.cache_dir} si cambiaste datos).")
            try:
                ds_zarr = xr.open_zarr(self.cache_dir)
                # Intentamos leer dimensiones estándar, fallback a nombres detectados
                lat_dim = next((d for d in ds_zarr.dims if d in ['latitude', 'lat', 'y']), 'latitude')
                lon_dim = next((d for d in ds_zarr.dims if d in ['longitude', 'lon', 'x']), 'longitude')
                self.cfg.LR_SHAPE = (ds_zarr.dims[lat_dim], ds_zarr.dims[lon_dim])
                return
            except:
                shutil.rmtree(self.cache_dir)

        print("🚀 Iniciando ETL Multi-Canal...")
        
        # 1. Carga Lazy
        ds_hr = xr.open_mfdataset(self.cfg.PATH_HR, chunks={'time': 100}, parallel=True)
        # LR puede ser un archivo único o un patrón con múltiples GRIB
        lr_path = self.cfg.PATH_LR
        cfgrib_kwargs = {
            "filter_by_keys": {"typeOfLevel": "surface"},
            "errors": "ignore",
            # Evitar crear índices .idx persistentes (pueden corromperse)
            "indexpath": ""
        }
        
        if os.path.isfile(lr_path):
            ds_lr = xr.open_dataset(
                lr_path,
                engine="cfgrib",
                backend_kwargs=cfgrib_kwargs,
                chunks={'time': 100}
            )
        else:
            ds_lr = xr.open_mfdataset(
                lr_path, 
                engine="cfgrib", 
                backend_kwargs=cfgrib_kwargs,
                chunks={'time': 100}, 
                parallel=True
            )

        # 2. Detección Inteligente de Variable HR (Target)
        hr_var = None
        for v in ['tas', 't2m', 'airTemperature', 'temp']:
            if v in ds_hr:
                hr_var = v
                break
        
        if hr_var is None:
            vars_list = list(ds_hr.data_vars)
            if vars_list:
                hr_var = vars_list[0]
                print(f"⚠️ No se encontró nombre estándar. Usando '{hr_var}' como target.")
            else:
                raise ValueError("❌ El dataset HR no tiene variables.")

        print(f"   🎯 Target HR detectado: '{hr_var}'")

        # 3. Conversión de Unidades Automática
        sample_mean = ds_hr[hr_var].isel(time=slice(0, 10)).mean().compute().item()
        
        if sample_mean > 200:
            print(f"   🌡️ Unidades detectadas: Kelvin (Media ~{sample_mean:.0f}). Convirtiendo a °C...")
            ds_hr["tas_C"] = ds_hr[hr_var] - 273.15
        else:
            print(f"   🌡️ Unidades detectadas: Celsius (Media ~{sample_mean:.0f}). Manteniendo valores.")
            ds_hr["tas_C"] = ds_hr[hr_var]

        # Conversiones LR
        for var in ['t2m', 'd2m', 'skt']:
            if var in ds_lr: ds_lr[var] = ds_lr[var] - 273.15

        # 4. Fix Temporal (Time/Step stack)
        if 'step' in ds_lr.coords:
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

        _, index = np.unique(ds_lr['time'], return_index=True)
        ds_lr = ds_lr.isel(time=index)

        # 5. Sincronización Robusta (Fix Temporal)
        print(f"   🔍 DEBUG TEMPORAL:")
        print(f"      HR Time Type: {ds_hr.time.dtype}")
        print(f"      HR Range: {ds_hr.time.min().values} -> {ds_hr.time.max().values}")
        print(f"      HR Samples: {ds_hr.time.values[:3]}")
        print(f"      LR Time Type: {ds_lr.time.dtype}")
        print(f"      LR Range: {ds_lr.time.min().values} -> {ds_lr.time.max().values}") 
        print(f"      LR Samples: {ds_lr.time.values[:3]}")

        # --- FIX: NORMALIZACIÓN DE TIEMPOS ---
        # Convertimos ambos ejes temporales a pandas.DatetimeIndex con precisión horaria
        # para evitar problemas de nanosegunos, cftime, zonas horarias, etc.
        import pandas as pd
        
        # Normalizar HR
        times_hr_raw = ds_hr.time.values
        if hasattr(times_hr_raw[0], 'strftime'):  # cftime objects
            times_hr_pd = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in times_hr_raw])
        else:
            times_hr_pd = pd.to_datetime(times_hr_raw)
        
        # Reducir a precisión horaria (elimina microsegundos, ns)
        times_hr_normalized = times_hr_pd.floor('H')
        
        # Normalizar LR
        times_lr_raw = ds_lr.time.values
        if hasattr(times_lr_raw[0], 'strftime'):  # cftime objects
            times_lr_pd = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in times_lr_raw])
        else:
            times_lr_pd = pd.to_datetime(times_lr_raw)
        
        times_lr_normalized = times_lr_pd.floor('H')
        
        print(f"   ✅ Tiempos normalizados (Precisión horaria)")
        print(f"      HR: {times_hr_normalized.min()} -> {times_hr_normalized.max()}")
        print(f"      LR: {times_lr_normalized.min()} -> {times_lr_normalized.max()}")
        
        # Asignar los nuevos índices normalizados
        ds_hr['time'] = times_hr_normalized
        ds_lr['time'] = times_lr_normalized
        
        # Ahora sí, intersección
        common_times = np.intersect1d(ds_hr.time.values, ds_lr.time.values)
        
        if len(common_times) == 0: 
            print("      ❌ ERROR: No hay intersección temporal incluso después de normalizar.")
            print(f"      HR disponible: {len(times_hr_normalized)} timesteps")
            print(f"      LR disponible: {len(times_lr_normalized)} timesteps")
            raise ValueError("❌ Sin coincidencia temporal. Verifica que los datasets cubran el mismo período.")
        
        print(f"   ✅ Intersección exitosa: {len(common_times)} timesteps comunes")
        ds_hr = ds_hr.sel(time=common_times)
        ds_lr = ds_lr.sel(time=common_times)

        # 6. Normalizar coordenadas HR a lat/lon 1D (si es posible)
        hr_lat_dim = 'latitude' if 'latitude' in ds_hr.dims else 'y'
        hr_lon_dim = 'longitude' if 'longitude' in ds_hr.dims else 'x'

        if hr_lat_dim == 'y' and 'latitude' in ds_hr.coords and 'longitude' in ds_hr.coords:
            lat = ds_hr['latitude']
            lon = ds_hr['longitude']
            if lat.ndim == 2 and lon.ndim == 2:
                lat1d = lat[:, 0]
                lon1d = lon[0, :]
                try:
                    is_rect_lat = np.allclose(lat.values, lat1d.values[:, None], atol=1e-6)
                    is_rect_lon = np.allclose(lon.values, lon1d.values[None, :], atol=1e-6)
                except Exception:
                    is_rect_lat = is_rect_lon = False

                if is_rect_lat and is_rect_lon:
                    print("✅ HR: lat/lon 2D rectilíneo detectado. Convirtiendo a 1D.")
                    ds_hr = ds_hr.rename({'latitude': 'latitude_2d', 'longitude': 'longitude_2d'})
                    ds_hr = ds_hr.assign_coords(latitude=('y', lat1d.values),
                                                longitude=('x', lon1d.values))
                    ds_hr = ds_hr.swap_dims({'y': 'latitude', 'x': 'longitude'})
                    hr_lat_dim = 'latitude'
                    hr_lon_dim = 'longitude'
                else:
                    print("⚠️ HR lat/lon 2D no es rectilíneo. Se mantiene y/x.")
            else:
                print("ℹ️ HR ya contiene lat/lon 1D o no-2D; se mantiene.")

        # 6. Recorte y Tensores
        hr_lat_coord = hr_lat_dim if hr_lat_dim in ds_hr.coords else 'y'
        hr_lon_coord = hr_lon_dim if hr_lon_dim in ds_hr.coords else 'x'
        
        min_lat = ds_hr[hr_lat_coord].min().compute().item()
        max_lat = ds_hr[hr_lat_coord].max().compute().item()
        min_lon = ds_hr[hr_lon_coord].min().compute().item()
        max_lon = ds_hr[hr_lon_coord].max().compute().item()
        
        buffer = 0.15 # Buffer pequeño para asegurar cobertura
        ds_lr_clipped = ds_lr.sel(
            latitude=slice(max_lat + buffer, min_lat - buffer), 
            longitude=slice(min_lon - buffer, max_lon + buffer)
        ).sortby('latitude', ascending=True).sortby('longitude', ascending=True)

        print("   📦 Empaquetando variables LR...")
        # to_array crea dims: (variable, time, latitude, longitude)
        ds_lr_arr = ds_lr_clipped.to_array(dim='variable', name='lr_input')
        
        # --- 🧹 FIX: variables LR completamente vacías (todo NaN) ---
        try:
            all_nan = ds_lr_arr.isnull().all(dim=['time', 'latitude', 'longitude'])
            if bool(all_nan.any()):
                nan_vars = ds_lr_arr['variable'].values[all_nan.values]
                print(f"   ⚠️ Variables LR con todo NaN: {list(nan_vars)} -> se rellenan con 0")
                ds_lr_arr = ds_lr_arr.where(~all_nan, other=0)
        except Exception as e:
            print(f"   ⚠️ No se pudo filtrar variables NaN: {e}")
        
        # --- 🛠️ FIX DE INTERPOLACIÓN MARÍTIMA 🛠️ ---
        print("   🌊 Rellenando NaNs (Estrategia Nearest/Extrapolate)...")
        # Usamos interpolate_na con 'nearest' y extrapolación.
        # Esto rellena el mar con el valor del píxel costero más cercano (escalón plano),
        # evitando gradientes falsos.
        
        # 1. Relleno Longitudinal (Este-Oeste)
        ds_lr_clean = ds_lr_arr.interpolate_na(dim='longitude', method='nearest', fill_value="extrapolate")
        # 2. Relleno Latitudinal (Norte-Sur) - Para esquinas rebeldes
        ds_lr_clean = ds_lr_clean.interpolate_na(dim='latitude', method='nearest', fill_value="extrapolate")
        
        # 3. Fallback Temporal (Por si queda algún hueco raro)
        ds_lr_clean = ds_lr_clean.ffill(dim='time').bfill(dim='time')
        # -----------------------------------------------
        
        # HR Clean
        ds_hr_clean = ds_hr["tas_C"].ffill(dim=hr_lon_dim).bfill(dim=hr_lon_dim) \
                                    .ffill(dim=hr_lat_dim).bfill(dim=hr_lat_dim)

        # Rellenar NaNs restantes (LR y HR) si existen
        try:
            lr_has_nan = bool(ds_lr_clean.isnull().any().compute().item())
        except Exception:
            lr_has_nan = True
        if lr_has_nan:
            print("⚠️ LR aún contiene NaNs. Rellenando con media por variable...")
            mean_lr_fill = ds_lr_clean.mean(dim=['time', 'latitude', 'longitude'], skipna=True)
            ds_lr_clean = ds_lr_clean.fillna(mean_lr_fill)

        try:
            hr_has_nan = bool(ds_hr_clean.isnull().any().compute().item())
        except Exception:
            hr_has_nan = True
        if hr_has_nan:
            print("⚠️ HR aún contiene NaNs. Rellenando con media global...")
            mean_hr_fill = ds_hr_clean.mean(skipna=True)
            ds_hr_clean = ds_hr_clean.fillna(mean_hr_fill)

        # 7. Stats Vectoriales
        print("   🧮 Calculando estadísticas...")
        with ProgressBar():
            # Especificamos dims explícitas para asegurar promedio correcto
            mean_lr = ds_lr_clean.mean(dim=['time', 'latitude', 'longitude']).compute()
            std_lr = ds_lr_clean.std(dim=['time', 'latitude', 'longitude']).compute()
            mean_hr = ds_hr_clean.mean().compute().item()
            std_hr = ds_hr_clean.std().compute().item()
        
        # Guardar estadísticas para visualizaciones
        try:
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            np.savez(
                self.stats_path,
                mean_lr=mean_lr.values,
                std_lr=std_lr.values,
                mean_hr=mean_hr,
                std_hr=std_hr
            )
            print(f"   💾 Stats guardadas en: {self.stats_path}")
        except Exception as e:
            print(f"   ⚠️ No se pudieron guardar stats: {e}")

        ds_lr_norm = (ds_lr_clean - mean_lr) / (std_lr + 1e-6)
        ds_hr_norm = (ds_hr_clean - mean_hr) / (std_hr + 1e-6)
        
        # --- 🔄 FIX ROTACIÓN: TRANSPOSICIÓN ANTES DE GUARDAR 🔄 ---
        # Aseguramos el orden canónico para Tensorflow: (Time, Lat, Lon, Variable)
        # to_array pone variable primero, así que lo movemos al final.
        print("   🔄 Transponiendo a (Time, Lat, Lon, Variable)...")
        ds_lr_norm = ds_lr_norm.transpose('time', 'latitude', 'longitude', 'variable')
        
        # Para HR: (Time, Lat, Lon)
        ds_hr_norm = ds_hr_norm.transpose('time', hr_lat_dim, hr_lon_dim)

        # 8. Guardar
        ds_final = xr.Dataset({"hr_target": ds_hr_norm, "lr_input": ds_lr_norm})
        
        # Actualizamos la config con las dimensiones reales lat/lon
        self.cfg.LR_SHAPE = (ds_lr_clipped.sizes['latitude'], ds_lr_clipped.sizes['longitude'])
        
        if os.path.exists(self.cache_dir): shutil.rmtree(self.cache_dir)
        
        encoding = {k: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for k in ds_final.data_vars}
        print("💾 Guardando Zarr...")
        with ProgressBar():
            ds_final.chunk({'time': 100}).to_zarr(self.cache_dir, mode='w', encoding=encoding, consolidated=True)
            
        print("✅ ETL Finalizado.")
        
    def get_tf_datasets(self, include_test: bool = False):
        """Generadores Multi-Canal (Corrige conflicto de nombres Lat/Lon vs Y/X)"""

        

        print("🔌 Conectando generadores a Zarr...")
        ds = xr.open_zarr(self.cache_dir, consolidated=True)

        # ✅ IMPORT LAZY: Solo importa TF si realmente llamamos a esta función
        import tensorflow as tf
        
        # A) Analizar Input de Baja Resolución (LR)
        # Buscamos qué nombres usa Específicamente la variable 'lr_input'
        da_lr = ds['lr_input']
        lr_dims_list = list(da_lr.dims)
        lr_lat = next((d for d in lr_dims_list if d in ['latitude', 'lat', 'y']), 'y')
        lr_lon = next((d for d in lr_dims_list if d in ['longitude', 'lon', 'x']), 'x')
        
        # Tamaño REAL del input (debe ser pequeño, ej: 5x9)
        real_lr_h = da_lr.sizes[lr_lat]
        real_lr_w = da_lr.sizes[lr_lon]
        
        print(f"   📐 Input LR detectado: {lr_lat}={real_lr_h}, {lr_lon}={real_lr_w}")

        # B) Analizar Target de Alta Resolución (HR)
        # Buscamos nombres para 'hr_target'
        da_hr = ds['hr_target']
        hr_dims_list = list(da_hr.dims)
        hr_y = next((d for d in hr_dims_list if d in ['y', 'latitude', 'lat']), 'y')
        hr_x = next((d for d in hr_dims_list if d in ['x', 'longitude', 'lon']), 'x')
        real_hr_h = da_hr.sizes[hr_y]
        real_hr_w = da_hr.sizes[hr_x]
        if self.cfg.HR_SHAPE != (real_hr_h, real_hr_w):
            print(f"   ⚠️ CORRECCIÓN HR_SHAPE: {self.cfg.HR_SHAPE} -> ({real_hr_h}, {real_hr_w}).")
            self.cfg.HR_SHAPE = (real_hr_h, real_hr_w)

        # Detectar orientación de longitudes y alinear LR con HR si es necesario
        def _order(vals):
            diffs = np.diff(vals)
            if np.all(diffs > 0):
                return "asc"
            if np.all(diffs < 0):
                return "desc"
            return "unknown"

        try:
            lr_lon_vals = ds[lr_lon].values
            hr_lon_vals = ds[hr_x].values
            lr_order = _order(lr_lon_vals)
            hr_order = _order(hr_lon_vals)
        except Exception:
            lr_order = hr_order = "unknown"

        flip_lr_lon = False
        if lr_order != "unknown" and hr_order != "unknown" and lr_order != hr_order:
            flip_lr_lon = True
            print(f"   ⚠️ Longitud order mismatch (LR={lr_order}, HR={hr_order}). Aplicando flip LR en lon.")

        # --- 🛑 2. ACTUALIZACIÓN DE CONFIGURACIÓN 🛑 ---
        # Ahora actualizamos Config usando las dimensiones del LR, no las del HR
        if self.cfg.LR_SHAPE != (real_lr_h, real_lr_w):
            print(f"   ⚠️ CORRECCIÓN DE SHAPE: Config decía {self.cfg.LR_SHAPE}, pero LR real es ({real_lr_h}, {real_lr_w}).")
            print(f"   🔧 Actualizando Config.LR_SHAPE a ({real_lr_h}, {real_lr_w}).")
            self.cfg.LR_SHAPE = (real_lr_h, real_lr_w)
        # ---------------------------------------------------------

        # 3. Preparar Estáticos (Normalización)
        # Aseguramos que static sea numpy para evitar líos de dims
        if isinstance(self.ds_static_single, (xr.DataArray, xr.Dataset)):
            static_data = self.ds_static_single.values
        else:
            static_data = self.ds_static_single

        # Asegurar forma (H, W, Chan)
        if static_data.ndim == 2:
             static_data = static_data[..., np.newaxis]
             
        mean_st = np.mean(static_data, axis=(0, 1), keepdims=True)
        std_st = np.std(static_data, axis=(0, 1), keepdims=True)
        static_norm = (static_data - mean_st) / (std_st + 1e-6)
        
        total_len = ds.sizes['time']
        seq_len = self.cfg.SEQ_LEN
        stride = int(getattr(self.cfg, "TEMPORAL_STRIDE", 1))
        if stride < 1:
            stride = 1
        if stride > seq_len:
            print(f"   ⚠️ TEMPORAL_STRIDE ({stride}) > SEQ_LEN ({seq_len}). Usando stride={seq_len}.")
            stride = seq_len

        temporal_sampler = getattr(self.cfg, "TEMPORAL_SAMPLER", "uniform")
        season_balance = bool(getattr(self.cfg, "TEMPORAL_SEASON_BALANCE", False))
        rng = np.random.default_rng(getattr(self.cfg, "SEED", 42))

        def _build_time_weights():
            if temporal_sampler not in ("weighted", "weighted_station"):
                return None
            # Base series: use HR mean over space
            series = None
            if temporal_sampler == "weighted_station":
                station_path = getattr(self.cfg, "STATION_GRIB_PATH", "")
                if station_path and os.path.exists(station_path):
                    try:
                        cfgrib_kwargs = {
                            "filter_by_keys": {"typeOfLevel": "surface"},
                            "errors": "ignore",
                            "indexpath": "",
                        }
                        ds_st = xr.open_dataset(station_path, engine="cfgrib", backend_kwargs=cfgrib_kwargs)
                        # pick variable
                        for v in ["t2m", "2t", "tas", "airTemperature"]:
                            if v in ds_st:
                                var = v
                                break
                        else:
                            var = list(ds_st.data_vars)[0]
                        da = ds_st[var]
                        # Celsius if needed
                        if float(da.isel({da.dims[0]: slice(0, min(3, da.sizes[da.dims[0]]))}).mean().values) > 200:
                            da = da - 273.15
                        time_dim = next((d for d in da.dims if d in ["time", "valid_time"]), da.dims[0])
                        # mean over all non-time dims
                        reduce_dims = [d for d in da.dims if d != time_dim]
                        series = da.mean(dim=reduce_dims).values
                        station_times = pd.to_datetime(da[time_dim].values).floor("H")
                        ds_times = pd.to_datetime(ds["time"].values).floor("H")
                        # align to ds times
                        time_map = {t: i for i, t in enumerate(station_times)}
                        aligned = np.zeros(ds_times.shape[0], dtype=np.float32)
                        for i, t in enumerate(ds_times):
                            j = time_map.get(t)
                            if j is not None:
                                aligned[i] = series[j]
                        series = aligned
                    except Exception:
                        series = None
                if series is None:
                    print("⚠️ No se pudo usar estaciones para pesos temporales, fallback a HR.")

            if series is None:
                try:
                    hr_mean = da_hr.mean(dim=[d for d in da_hr.dims if d != 'time']).values
                    series = hr_mean
                except Exception:
                    series = None

            if series is None:
                return None

            # Temporal gradient as importance
            series = np.asarray(series, dtype=np.float32)
            grad = np.abs(np.diff(series, prepend=series[0]))
            grad = grad - np.nanmin(grad)
            gamma = float(getattr(self.cfg, "TEMPORAL_WEIGHT_GAMMA", 1.0))
            if gamma != 1.0:
                grad = np.power(grad, gamma)
            min_prob = float(getattr(self.cfg, "TEMPORAL_MIN_PROB", 1e-6))
            grad = grad + min_prob
            grad = grad / np.sum(grad)
            return grad

        time_weights = _build_time_weights()

        def _time_indices(times, start, end):
            times = pd.to_datetime(times).values
            start = np.datetime64(start)
            end = np.datetime64(end)
            start_idx = int(np.searchsorted(times, start, side="left"))
            end_idx = int(np.searchsorted(times, end, side="left"))
            return start_idx, end_idx

        test_start = test_end = None
        if getattr(self.cfg, "SPLIT_MODE", "fraction") == "time":
            try:
                times = ds['time'].values
                train_start, train_end = _time_indices(times, self.cfg.TRAIN_START, self.cfg.TRAIN_END)
                val_start, val_end = _time_indices(times, self.cfg.VAL_START, self.cfg.VAL_END)
                if include_test:
                    test_start, test_end = _time_indices(times, self.cfg.TEST_START, self.cfg.TEST_END)
                print(f"   📂 Dataset (train) range: {train_start} -> {train_end}")
                print(f"   📂 Dataset (val) range: {val_start} -> {val_end}")
                if include_test:
                    print(f"   📂 Dataset (test) range: {test_start} -> {test_end}")
            except Exception as e:
                print(f"⚠️ Time split fallback to fraction due to: {e}")
                split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
                train_start, train_end = 0, split_idx
                val_start, val_end = split_idx, total_len
        else:
            split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
            train_start, train_end = 0, split_idx
            if include_test:
                val_end = int(total_len * 0.9)
                val_start = split_idx
                test_start, test_end = val_end, total_len
            else:
                val_start, val_end = split_idx, total_len

        # Precompute valid start indices
        max_start_train = train_end - seq_len * stride
        max_start_val = val_end - seq_len * stride
        max_start_test = test_end - seq_len * stride if include_test else None

        def _season_index(times_idx):
            # DJF=0, MAM=1, JJA=2, SON=3
            months = times_idx.month
            seasons = np.zeros_like(months)
            seasons[(months >= 3) & (months <= 5)] = 1
            seasons[(months >= 6) & (months <= 8)] = 2
            seasons[(months >= 9) & (months <= 11)] = 3
            return seasons

        times_pd = pd.to_datetime(ds["time"].values)
        seasons = _season_index(times_pd)

        def _sample_time(start_i, end_i, max_start):
            if max_start is None or max_start <= start_i:
                return start_i
            if temporal_sampler == "uniform" and not season_balance:
                return int(rng.integers(start_i, max_start))

            # Build candidate indices
            candidates = np.arange(start_i, max_start)
            if candidates.size == 0:
                return int(start_i)

            # Seasonal balance
            if season_balance:
                season_ids = [0, 1, 2, 3]
                available = [s for s in season_ids if np.any(seasons[candidates] == s)]
                if available:
                    chosen_season = rng.choice(available)
                    candidates = candidates[seasons[candidates] == chosen_season]

            if time_weights is None:
                return int(rng.choice(candidates))

            weights = time_weights[candidates]
            weights = weights / np.sum(weights)
            return int(rng.choice(candidates, p=weights))
        
        # Detectar canales LR
        if 'variable' in da_lr.sizes:
            n_channels = da_lr.sizes['variable']
        elif 'channel' in da_lr.sizes:
            n_channels = da_lr.sizes['channel']
        else:
            n_channels = 1
            
        print(f"   ℹ️ Input Channels Detectados: {n_channels}")

        def generator(start_i, end_i, max_start):
            # For weighted sampling, we draw with replacement for a fixed count
            if temporal_sampler != "uniform" or season_balance:
                sample_count = max(1, max_start - start_i)
                for _ in range(sample_count):
                    i = _sample_time(start_i, end_i, max_start)
                    t_idx = slice(i, i + seq_len * stride, stride)
                    yield _yield_at(i, t_idx)
                return

            for i in range(start_i, end_i - seq_len * stride):
                t_idx = slice(i, i + seq_len * stride, stride)
                yield _yield_at(i, t_idx)

        def _yield_at(i, t_idx):
            # 1. LR INPUT
            # Usamos los nombres detectados para LR (lr_lat, lr_lon)
            # Transpose: (Time, Alto, Ancho, Var)
            if 'variable' in da_lr.dims:
                x_lr = da_lr.isel(time=t_idx) \
                            .transpose('time', lr_lat, lr_lon, 'variable') \
                            .values
            else:
                x_lr = da_lr.isel(time=t_idx) \
                            .transpose('time', lr_lat, lr_lon) \
                            .values
                if x_lr.ndim == 3:
                    x_lr = x_lr[..., np.newaxis]
            
            if flip_lr_lon:
                x_lr = x_lr[:, :, ::-1, :]
            
            # 2. HR TARGET
            # Usamos los nombres detectados para HR (hr_y, hr_x)
            y_hr = da_hr.isel(time=t_idx) \
                        .transpose('time', hr_y, hr_x) \
                        .values
            y_hr = y_hr[..., np.newaxis] 
            
            # 3. STATIC
            x_st = np.repeat(static_norm[np.newaxis, ...], seq_len, axis=0)
            
            return (x_lr, x_st), y_hr

        # Output Signatures
        lr_h, lr_w = self.cfg.LR_SHAPE
        st_h, st_w = self.cfg.HR_SHAPE
        
        spec_lr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, lr_h, lr_w, n_channels), dtype=tf.float32)
        spec_st = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, self.cfg.STATIC_CHANNELS), dtype=tf.float32)
        spec_hr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, 1), dtype=tf.float32)

        shuffle_buf = getattr(self.cfg, "SHUFFLE_BUFFER_SIZE", 500)
        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_start, train_end, max_start_train), output_signature=((spec_lr, spec_st), spec_hr)
        ).shuffle(shuffle_buf).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_start, val_end, max_start_val), output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        if include_test:
            test_ds = tf.data.Dataset.from_generator(
                lambda: generator(test_start, test_end, max_start_test), output_signature=((spec_lr, spec_st), spec_hr)
            ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            return train_ds, val_ds, test_ds
        
        return train_ds, val_ds
