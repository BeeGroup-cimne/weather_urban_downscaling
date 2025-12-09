# ==========================================
# 2. MOTOR DE DATOS BIG DATA (Scalable Pipeline)
# ==========================================
import xarray as xr
import numpy as np
import glob
import shutil
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.interpolate import NearestNDInterpolator
import os
import tensorflow as tf


class BigDataPipeline:
    def __init__(self, config):
        self.cfg = config
        self.cache_dir = self.cfg.PATH_CACHE
        self.stats_path = "./stats_config.npz"
        self.ds_static_single = None 

        processed_dir = os.path.dirname(self.cache_dir)
        os.makedirs(processed_dir, exist_ok=True)

        self.stats_path = os.path.join(processed_dir, "stats_config.npz")
        
        self.ds_static_single = None 
        print(f"📂 Cache configurado en: {self.cache_dir}")
        print(f"📊 Stats configurado en: {self.stats_path}")
        
    def process_static_data(self):
        """
        Procesa datos estáticos (Elevación, etc.).
        CORREGIDO: Detecta automáticamente si las dimensiones son y/x (UrbClim) o lat/lon.
        """
        print("🗺️ Processing Static Data...")
        
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
            
        print(f"   📏 Dimensiones detectadas HR: {dim_y}x{dim_x}")

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
        Fase 1: Transforma RAW -> ZARR (Soporte Multi-Canal + Auto-Detect HR).
        """
        if os.path.exists(self.cache_dir):
            print(f"⚡ Cache Zarr detectado. (IMPORTANTE: BORRA {self.cache_dir} si cambiaste datos).")
            try:
                ds_zarr = xr.open_zarr(self.cache_dir)
                self.cfg.LR_SHAPE = (ds_zarr.dims['latitude'], ds_zarr.dims['longitude'])
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
        # Buscamos 'tas', 't2m' o 'airTemperature'
        hr_var = None
        for v in ['tas', 't2m', 'airTemperature', 'temp']:
            if v in ds_hr:
                hr_var = v
                break
        
        if hr_var is None:
            # Fallback: Usar la primera variable de datos que encuentre
            vars_list = list(ds_hr.data_vars)
            if vars_list:
                hr_var = vars_list[0]
                print(f"⚠️ No se encontró nombre estándar. Usando '{hr_var}' como target.")
            else:
                raise ValueError("❌ El dataset HR no tiene variables.")

        print(f"   🎯 Target HR detectado: '{hr_var}'")

        # 3. Conversión de Unidades Automática (Kelvin/Celsius)
        # Calculamos la media del primer chunk para saber si es K o C
        sample_mean = ds_hr[hr_var].isel(time=slice(0, 10)).mean().compute().item()
        
        if sample_mean > 200:
            print(f"   🌡️ Unidades detectadas: Kelvin (Media ~{sample_mean:.0f}). Convirtiendo a °C...")
            ds_hr["tas_C"] = ds_hr[hr_var] - 273.15
        else:
            print(f"   🌡️ Unidades detectadas: Celsius (Media ~{sample_mean:.0f}). Manteniendo valores.")
            ds_hr["tas_C"] = ds_hr[hr_var]

        # Conversiones LR (ERA5 siempre viene en Kelvin si es GRIB)
        if 't2m' in ds_lr: ds_lr["t2m"] = ds_lr["t2m"] - 273.15
        if 'd2m' in ds_lr: ds_lr["d2m"] = ds_lr["d2m"] - 273.15
        if 'skt' in ds_lr: ds_lr["skt"] = ds_lr["skt"] - 273.15

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

        # 5. Sincronización
        common_times = np.intersect1d(ds_hr.time.values, ds_lr.time.values)
        if len(common_times) == 0: raise ValueError("❌ Sin coincidencia temporal.")
        ds_hr = ds_hr.sel(time=common_times)
        ds_lr = ds_lr.sel(time=common_times)

        # 6. Recorte y Tensores
        hr_lat_coord = 'latitude' if 'latitude' in ds_hr.coords else 'y'
        hr_lon_coord = 'longitude' if 'longitude' in ds_hr.coords else 'x'
        
        min_lat = ds_hr[hr_lat_coord].min().compute().item()
        max_lat = ds_hr[hr_lat_coord].max().compute().item()
        min_lon = ds_hr[hr_lon_coord].min().compute().item()
        max_lon = ds_hr[hr_lon_coord].max().compute().item()
        
        buffer = 0.15
        ds_lr_clipped = ds_lr.sel(
            latitude=slice(max_lat + buffer, min_lat - buffer), 
            longitude=slice(min_lon - buffer, max_lon + buffer)
        ).sortby('latitude', ascending=False).sortby('longitude', ascending=True)

        print("   📦 Empaquetando variables LR...")
        ds_lr_arr = ds_lr_clipped.to_array(dim='variable', name='lr_input')
        
        print("   🌊 Rellenando NaNs (Spatial Fill)...")
        ds_lr_clean = ds_lr_arr.ffill(dim='longitude').bfill(dim='longitude') \
                               .ffill(dim='latitude').bfill(dim='latitude')
        
        # HR Clean (Usando dims correctas)
        hr_lat_dim = 'latitude' if 'latitude' in ds_hr.dims else 'y'
        hr_lon_dim = 'longitude' if 'longitude' in ds_hr.dims else 'x'
        ds_hr_clean = ds_hr["tas_C"].ffill(dim=hr_lon_dim).bfill(dim=hr_lon_dim) \
                                    .ffill(dim=hr_lat_dim).bfill(dim=hr_lat_dim)

        # 7. Stats Vectoriales
        print("   🧮 Calculando estadísticas...")
        with ProgressBar():
            mean_lr = ds_lr_clean.mean(dim=['time', 'latitude', 'longitude']).compute()
            std_lr = ds_lr_clean.std(dim=['time', 'latitude', 'longitude']).compute()
            mean_hr = ds_hr_clean.mean().compute().item()
            std_hr = ds_hr_clean.std().compute().item()

        ds_lr_norm = (ds_lr_clean - mean_lr) / (std_lr + 1e-6)
        ds_hr_norm = (ds_hr_clean - mean_hr) / (std_hr + 1e-6)
        
        # 8. Guardar
        ds_final = xr.Dataset({"hr_target": ds_hr_norm, "lr_input": ds_lr_norm})
        self.cfg.LR_SHAPE = (ds_lr_clipped.sizes['latitude'], ds_lr_clipped.sizes['longitude'])
        
        if os.path.exists(self.cache_dir): shutil.rmtree(self.cache_dir)
        
        encoding = {k: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} for k in ds_final.data_vars}
        print("💾 Guardando Zarr...")
        with ProgressBar():
            ds_final.chunk({'time': 100}).to_zarr(self.cache_dir, mode='w', encoding=encoding, consolidated=True)
            
        print("✅ ETL Finalizado.")
        
    def get_tf_datasets(self):
        """Generadores Multi-Canal (Soporta N variables dinámicamente)"""
        print("🔌 Conectando generadores a Zarr...")
        ds = xr.open_zarr(self.cache_dir, consolidated=True)
        
        # Estáticos
        mean_st = self.ds_static_single.mean(axis=(0, 1), keepdims=True)
        std_st = self.ds_static_single.std(axis=(0, 1), keepdims=True)
        static_norm = (self.ds_static_single - mean_st) / (std_st + 1e-6)
        
        total_len = ds.sizes['time']
        split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
        
        # Detectar número real de canales en el archivo procesado
        n_channels = ds.sizes['variable'] if 'variable' in ds.sizes else 1
        print(f"   ℹ️ Input Channels Detectados: {n_channels}")

        def generator(start_i, end_i):
            seq_len = self.cfg.SEQ_LEN
            lat_dim = 'latitude' if 'latitude' in ds.dims else 'y'
            lon_dim = 'longitude' if 'longitude' in ds.dims else 'x'
            
            for i in range(start_i, end_i - seq_len):
                # 1. LR INPUT (Multi-Channel)
                # Zarr shape: (variable, time, lat, lon) o similar
                # Transpose Target: (time, lat, lon, variable) -> Formato TF
                x_lr = ds['lr_input'].isel(time=slice(i, i+seq_len)) \
                                     .transpose('time', lat_dim, lon_dim, 'variable') \
                                     .values
                
                # NO agregamos np.newaxis, porque 'variable' ya es el último eje
                
                # 2. HR TARGET (Single Channel)
                y_hr = ds['hr_target'].isel(time=slice(i, i+seq_len)) \
                                      .transpose('time', 'y', 'x') \
                                      .values
                y_hr = y_hr[..., np.newaxis] # HR sí necesita canal extra (es 1 sola variable)
                
                # 3. STATIC
                x_st = np.repeat(static_norm[np.newaxis, ...], seq_len, axis=0)
                
                yield (x_lr, x_st), y_hr

        # Output Signatures
        lr_h, lr_w = self.cfg.LR_SHAPE
        st_h, st_w = self.cfg.HR_SHAPE
        
        # shape=(Seq, H, W, N_Channels)
        spec_lr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, lr_h, lr_w, n_channels), dtype=tf.float32)
        spec_st = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, 18), dtype=tf.float32)
        spec_hr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, 1), dtype=tf.float32)

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(0, split_idx), output_signature=((spec_lr, spec_st), spec_hr)
        ).shuffle(500).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(split_idx, total_len), output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds