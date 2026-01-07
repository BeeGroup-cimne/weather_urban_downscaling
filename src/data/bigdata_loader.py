# MOTOR DE DATOS BIG DATA (Versión Corregida: Dimensiones Explícitas)

# MOTOR DE DATOS BIG DATA (Versión Final: Auto-Shape Correction)

import xarray as xr
import numpy as np
import os
import tensorflow as tf
from config.config import Config

class BigDataPipeline:
    def __init__(self, config):
        self.cfg = config
        self.cache_dir = self.cfg.PATH_CACHE
        
        project_root = os.getcwd()
        if os.path.basename(project_root) in ['src', 'scripts', 'experiments']:
             project_root = os.path.dirname(project_root)
             
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.join(project_root, self.cache_dir)
        
        self.ds_static_single = None 
        static_cache_path = os.path.join(project_root, "processed_cache_zarr/static_processed.npy")
        if os.path.exists(static_cache_path):
             self.ds_static_single = np.load(static_cache_path)
        else:
             print("⚠️ No se encontró static_processed.npy. Usaremos ceros.")

    def get_tf_datasets(self):
        """Generadores que corrigen la configuración automáticamente"""
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"❌ No se encuentra el caché Zarr en {self.cache_dir}")

        ds = xr.open_zarr(self.cache_dir, consolidated=True)

        # --- 1. DETECCIÓN ROBUSTA DE DIMENSIONES ---
        da_lr = ds['lr_input']
        da_hr = ds['hr_target']
        
        # Detectar nombres de dimensiones
        lr_dims = list(da_lr.dims)
        lr_lat = next((d for d in lr_dims if d in ['latitude', 'lat', 'y']), 'y')
        lr_lon = next((d for d in lr_dims if d in ['longitude', 'lon', 'x']), 'x')
        lr_time = next((d for d in lr_dims if d in ['time', 'valid_time', 't']), 'time')
        lr_var_dim = next((d for d in lr_dims if d in ['variable', 'channel', 'var']), None)
        
        hr_dims = list(da_hr.dims)
        hr_y = next((d for d in hr_dims if d in ['y', 'latitude', 'lat']), 'y')
        hr_x = next((d for d in hr_dims if d in ['x', 'longitude', 'lon']), 'x')

        # --- 🛠️ AUTO-FIX SHAPE (La corrección clave) 🛠️ ---
        # Leemos el tamaño real del Zarr
        real_h = da_lr.sizes[lr_lat]
        real_w = da_lr.sizes[lr_lon]
        
        print(f"   📏 Shape LR Real detectado en Zarr: ({real_h}, {real_w})")
        
        # Si la config está mal, la corregimos en caliente
        if self.cfg.LR_SHAPE != (real_h, real_w):
            print(f"   ⚠️ CORRIGIENDO CONFIGURACIÓN: {self.cfg.LR_SHAPE} -> ({real_h}, {real_w})")
            # Actualizamos la clase Config globalmente para que el Modelo también se entere
            self.cfg.LR_SHAPE = (real_h, real_w)
        
        # --- 2. CÁLCULO DE CANALES ---
        if lr_var_dim:
            n_channels = da_lr.sizes[lr_var_dim]
        else:
            n_channels = 1
            
        total_len = da_lr.sizes[lr_time]
        print(f"   ✅ Input Channels: {n_channels}")

        # --- 3. PREPARAR ESTÁTICOS ---
        if self.ds_static_single is None:
            st_h, st_w = self.cfg.HR_SHAPE
            static_norm = np.zeros((st_h, st_w, self.cfg.STATIC_CHANNELS), dtype='float32')
        else:
            static_data = self.ds_static_single
            if static_data.ndim == 2: static_data = static_data[..., np.newaxis]
            mean_st = np.mean(static_data, axis=(0, 1), keepdims=True)
            std_st = np.std(static_data, axis=(0, 1), keepdims=True)
            static_norm = (static_data - mean_st) / (std_st + 1e-6)

        split_idx = int(total_len * self.cfg.SPLIT_FRACTION)

        def generator(start_i, end_i):
            seq_len = self.cfg.SEQ_LEN
            for i in range(start_i, end_i - seq_len):
                # LR INPUT
                if lr_var_dim:
                    x_lr = da_lr.isel({lr_time: slice(i, i+seq_len)}) \
                                .transpose(lr_time, lr_lat, lr_lon, lr_var_dim) \
                                .values
                else:
                    x_lr = da_lr.isel({lr_time: slice(i, i+seq_len)}) \
                                .transpose(lr_time, lr_lat, lr_lon) \
                                .values
                    if x_lr.ndim == 3: x_lr = x_lr[..., np.newaxis]
                
                # HR TARGET
                y_hr = da_hr.isel({lr_time: slice(i, i+seq_len)}) \
                            .transpose(lr_time, hr_y, hr_x) \
                            .values
                if y_hr.ndim == 3: y_hr = y_hr[..., np.newaxis]

                # STATIC
                x_st = np.repeat(static_norm[np.newaxis, ...], seq_len, axis=0)
                
                yield (x_lr, x_st), y_hr

        # Output Signatures con las dimensiones REALES (self.cfg.LR_SHAPE actualizado)
        lr_h, lr_w = self.cfg.LR_SHAPE # Ahora valdrá (5, 6)
        st_h, st_w = self.cfg.HR_SHAPE
        
        spec_lr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, lr_h, lr_w, n_channels), dtype=tf.float32)
        spec_st = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, self.cfg.STATIC_CHANNELS), dtype=tf.float32)
        spec_hr = tf.TensorSpec(shape=(self.cfg.SEQ_LEN, st_h, st_w, 1), dtype=tf.float32)

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(0, split_idx), output_signature=((spec_lr, spec_st), spec_hr)
        ).shuffle(100).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(split_idx, total_len), output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds

def get_dataloaders():
    pipeline = BigDataPipeline(Config)
    return pipeline.get_tf_datasets()