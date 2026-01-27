
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

        if os.path.exists(static_cache_path):
            print("🗺️ Cargando datos estáticos desde caché (.npy)...")
            self.ds_static_single = np.load(static_cache_path)
            print(f"   ✅ Static Data Shape Final: {self.ds_static_single.shape}")
            return

        print("🗺️ Procesando e Interpolando datos estáticos (Esto se hará solo una vez)...")
        
        # 1. Abrimos el dataset HR para leer dimensiones y grilla
        # Usamos open_dataset normal para asegurar acceso a todas las variables
        ds_hr = xr.open_dataset(self.cfg.PATH_HR)
        
        # --- FIX: DETECCIÓN ROBUSTA DE DIMENSIONES ---
        # UrbClim usa 'y' y 'x' como dimensiones, no 'latitude'/'longitude'
        if 'y' in ds_hr.sizes and 'x' in ds_hr.sizes:
            dim_y = ds_hr.sizes['y']
            dim_x = ds_hr.sizes['x']
        elif 'latitude' in ds_hr.sizes:
            dim_y = ds_hr.sizes['latitude']
            dim_x = ds_hr.sizes['longitude']
        else:
            # Fallback por si acaso
            dim_y, dim_x = ds_hr.shape[-2], ds_hr.shape[-1]
            
        print(f" 📏 Dimensiones detectadas HR: {dim_y}x{dim_x}")

        # --- GRID PARA INTERPOLACIÓN ---
        # Necesitamos los valores reales de latitud/longitud de cada pixel para saber dónde interpolar
        # Aunque las dimensiones se llamen 'y'/'x', las variables 'latitude'/'longitude' existen
        grid_y = ds_hr['latitude'].values
        grid_x = ds_hr['longitude'].values
        
        # Aplanar para interpolar
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
        ds_lr = xr.open_mfdataset(
            self.cfg.PATH_LR, 
            engine="cfgrib", 
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}, "errors": "ignore"},
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

        # 6. Recorte y Tensores
        hr_lat_coord = 'latitude' if 'latitude' in ds_hr.coords else 'y'
        hr_lon_coord = 'longitude' if 'longitude' in ds_hr.coords else 'x'
        
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
        hr_lat_dim = 'latitude' if 'latitude' in ds_hr.dims else 'y'
        hr_lon_dim = 'longitude' if 'longitude' in ds_hr.dims else 'x'
        ds_hr_clean = ds_hr["tas_C"].ffill(dim=hr_lon_dim).bfill(dim=hr_lon_dim) \
                                    .ffill(dim=hr_lat_dim).bfill(dim=hr_lat_dim)

        # 7. Stats Vectoriales
        print("   🧮 Calculando estadísticas...")
        with ProgressBar():
            # Especificamos dims explícitas para asegurar promedio correcto
            mean_lr = ds_lr_clean.mean(dim=['time', 'latitude', 'longitude']).compute()
            std_lr = ds_lr_clean.std(dim=['time', 'latitude', 'longitude']).compute()
            mean_hr = ds_hr_clean.mean().compute().item()
            std_hr = ds_hr_clean.std().compute().item()

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
        
    def get_tf_datasets(self):
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
        split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
        
        # Detectar canales LR
        if 'variable' in da_lr.sizes:
            n_channels = da_lr.sizes['variable']
        elif 'channel' in da_lr.sizes:
            n_channels = da_lr.sizes['channel']
        else:
            n_channels = 1
            
        print(f"   ℹ️ Input Channels Detectados: {n_channels}")

        def generator(start_i, end_i):
            seq_len = self.cfg.SEQ_LEN
            
            for i in range(start_i, end_i - seq_len):
                # 1. LR INPUT
                # Usamos los nombres detectados para LR (lr_lat, lr_lon)
                # Transpose: (Time, Alto, Ancho, Var)
                if 'variable' in da_lr.dims:
                     x_lr = da_lr.isel(time=slice(i, i+seq_len)) \
                                 .transpose('time', lr_lat, lr_lon, 'variable') \
                                 .values
                else:
                     x_lr = da_lr.isel(time=slice(i, i+seq_len)) \
                                 .transpose('time', lr_lat, lr_lon) \
                                 .values
                     if x_lr.ndim == 3: x_lr = x_lr[..., np.newaxis]
                
                # 2. HR TARGET
                # Usamos los nombres detectados para HR (hr_y, hr_x)
                y_hr = da_hr.isel(time=slice(i, i+seq_len)) \
                            .transpose('time', hr_y, hr_x) \
                            .values
                y_hr = y_hr[..., np.newaxis] 
                
                # 3. STATIC
                x_st = np.repeat(static_norm[np.newaxis, ...], seq_len, axis=0)
                
                yield (x_lr, x_st), y_hr

        # Output Signatures
        lr_h, lr_w = self.cfg.LR_SHAPE
        st_h, st_w = self.cfg.HR_SHAPE
        
        spec_lr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, lr_h, lr_w, n_channels), dtype=tf.float32)
        spec_st = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, Config.STATIC_CHANNELS), dtype=tf.float32)
        spec_hr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, 1), dtype=tf.float32)

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(0, split_idx), output_signature=((spec_lr, spec_st), spec_hr)
        ).shuffle(500).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(split_idx, total_len), output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds