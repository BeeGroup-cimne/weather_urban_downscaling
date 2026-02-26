#!/usr/bin/env python
# coding: utf-8

"""
Urban Weather Downscaling Pipeline
----------------------------------
Este script orquesta la carga de datos meteorológicos, el preprocesamiento,
y la comparación de tres modelos de Deep Learning (ConvLSTM, U-Net, Transformer).

Autor: Kerin Cardona

NOTA: Este es un script legacy de experimentación.
Para producción, usar scripts/run_ablation.py
"""

import os
import sys
import platform
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tensorflow as tf
from scipy.interpolate import griddata
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D,
    Cropping2D, TimeDistributed, Concatenate, BatchNormalization,
    Activation, Add, LeakyReLU, Resizing, Dropout, MultiHeadAttention,
    Lambda, Permute, Dense, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import central config
from config.runtime import Config as CentralConfig

# ==========================================
# 1. CONFIGURACIÓN GLOBAL (Extiende Config central)
# ==========================================
class Config:
    # Rutas de archivos - Heredadas de configuración central
    PATH_HR = CentralConfig.PATH_HR
    PATH_LR = CentralConfig.PATH_LR
    PATH_STATIC = CentralConfig.PATH_STATIC

    # Hiperparámetros (Basados en las necesidades del Transformer - Exp 3)
    SEED = 42
    BATCH_SIZE = 8  # Reducido para acomodar el Transformer en memoria
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    SEQ_LEN = 2  # Ventana de tiempo
    SPLIT_FRACTION = 0.9  # 90% Train, 10% Val

    # Dimensiones (Detectadas dinámicamente, pero definimos defaults)
    HR_SHAPE = (251, 251)
    LR_SHAPE = (4, 3)  # Ajustado a tu recorte final

    # Configuración de Hardware
    IS_MAC_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# ==========================================
# 2. MOTOR DE DATOS BIG DATA (Scalable Pipeline)
# ==========================================
import glob
import shutil
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.interpolate import NearestNDInterpolator

class BigDataPipeline:
    def __init__(self, config):
        self.cfg = config
        self.cache_dir = "./processed_cache_zarr"
        self.stats_path = "./stats_config.npz"
        self.ds_static_single = None 
        
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

# ==========================================
# 3. ZOOLÓGICO DE MODELOS (Arquitecturas)
# ==========================================
class ModelZoo:
    @staticmethod
    def get_optimizer(lr):
        """Selecciona el optimizador adecuado según el Hardware"""
        if Config.IS_MAC_SILICON:
            from tensorflow.keras.optimizers.legacy import Adam
            print("🚀 Using Legacy Adam (Metal/M-Series)")
            return Adam(learning_rate=lr)
        else:
            from tensorflow.keras.optimizers import Adam
            print("⚙️ Using Standard Adam")
            return Adam(learning_rate=lr)

    @staticmethod
    def res_block(x, filters):
        skip = x
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Add()([x, skip])
        return x

    @staticmethod
    def conv_block(x, filters):
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(0.1))(x)
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(0.1))(x)
        return x

    @staticmethod
    def temporal_transformer_block(x_input, embed_dim, num_heads=4, ff_dim=512):
        """Bloque Transformer para datos Espacio-Temporales"""
        # Shape: (Batch, Time, H, W, C)
        x = Permute((2, 3, 1, 4))(x_input)  # -> (Batch, H, W, Time, C)

        def flatten_spatial(x):
            s = tf.shape(x)
            return tf.reshape(x, (-1, s[3], s[4]))  # (Batch*H*W, Time, C)

        x_reshaped = Lambda(flatten_spatial)(x)
        x_norm = LayerNormalization(epsilon=1e-6)(x_reshaped)
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x_norm, x_norm)
        attn_out = Dropout(0.1)(attn_out)
        out1 = Add()([x_reshaped, attn_out])

        x_norm2 = LayerNormalization(epsilon=1e-6)(out1)
        ffn = Dense(ff_dim, activation="gelu")(x_norm2)
        ffn = Dense(embed_dim)(ffn)
        out2 = Add()([out1, ffn])

        def restore_spatial(args):
            x_proc, x_orig = args
            s = tf.shape(x_orig)  # (Batch, Time, H, W, C)
            return tf.reshape(x_proc, (s[0], s[2], s[3], s[1], s[4]))

        out_restored = Lambda(restore_spatial)([out2, x_input])
        return Permute((3, 1, 2, 4))(out_restored)

    @classmethod
    def build_convlstm(cls):
        """Experimento 1: ConvLSTM + Upsampling"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        x = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)(inp_dyn)

        # Upsampling progresivo
        for _ in range(6):
            x = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(x)
            x = TimeDistributed(Conv2D(64, (3, 3), padding="same"))(x)
            x = TimeDistributed(LeakyReLU(0.2))(x)

        x = TimeDistributed(Resizing(*Config.HR_SHAPE))(x)
        merged = Concatenate()([x, inp_st])

        x = TimeDistributed(Conv2D(64, (3, 3), padding="same"))(merged)
        x = cls.res_block(x, 64)
        out = TimeDistributed(Conv2D(1, (1, 1), activation="linear"))(x)

        model = Model([inp_dyn, inp_st], out, name="Exp1_ConvLSTM")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model

    @classmethod
    def build_unet(cls):
        """Experimento 2: U-Net Standard"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        # Bridge
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(inp_dyn)
        x = Concatenate()([x_up, inp_st])

        # Encoder
        c1 = cls.conv_block(x, 32)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        c2 = cls.conv_block(p1, 64)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        c3 = cls.conv_block(p2, 128)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # Bottleneck
        b = cls.conv_block(p3, 256)
        b = TimeDistributed(Dropout(0.3))(b)

        # Decoder
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(b)
        u3 = Concatenate()([u3, c3])
        c4 = cls.conv_block(u3, 128)

        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2])
        c5 = cls.conv_block(u2, 64)

        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1])
        c6 = cls.conv_block(u1, 32)

        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)

        model = Model([inp_dyn, inp_st], out, name="Exp2_UNet")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model

    @classmethod
    def build_hybrid_unet_lstm(cls):
        """
        ARQUITECTURA HÍBRIDA: U-Net + ConvLSTM
        --------------------------------------
        1. Encoder Espacial (TimeDistributed Conv2D): Comprime cada frame de la secuencia.
        2. Bottleneck Temporal (ConvLSTM): Aprende la evolución temporal en el espacio latente.
        3. Decoder Espacial (TimeDistributed UpSampling): Reconstruye la alta resolución.
        """
        # Inputs Dinámicos (LR) y Estáticos (HR)
        # Nota: Usamos '9' canales o 'None' si queremos flexibilidad total
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9)) 
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        # --- 1. BRIDGE & FUSION ---
        # Escalamos la entrada LR al tamaño HR para concatenarla con los datos estáticos
        # Esto permite que la red vea la topografía (HR) desde la primera capa.
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(inp_dyn)
        x = Concatenate()([x_up, inp_st])

        # --- 2. ENCODER (Espacial / Frame a Frame) ---
        # Usamos TimeDistributed para aplicar las mismas Convoluciones a cada paso de tiempo t
        
        # Bloque 1
        c1 = cls.conv_block(x, 32)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        
        # Bloque 2
        c2 = cls.conv_block(p1, 64)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        
        # Bloque 3
        c3 = cls.conv_block(p2, 128)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # --- 3. BOTTLENECK (Temporal / ConvLSTM) ---
        # Aquí sustituimos la convolución normal por una ConvLSTM.
        # Esto procesa la secuencia temporal en el espacio latente (comprimido).
        # return_sequences=True es vital para mantener la dimensión de tiempo para el decoder.
        
        lstm_out = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=True)(p3)
        lstm_out = TimeDistributed(BatchNormalization())(lstm_out)
        lstm_out = TimeDistributed(LeakyReLU(0.1))(lstm_out)
        
        # Podemos añadir una segunda capa LSTM si hay memoria suficiente (Opcional)
        # lstm_out = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=True)(lstm_out)
        # lstm_out = TimeDistributed(BatchNormalization())(lstm_out)

        # --- 4. DECODER (Reconstrucción Espacial) ---
        # Usamos las Skip Connections (c3, c2, c1) del Encoder original
        
        # Upsample 1
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(lstm_out)
        u3 = Concatenate()([u3, c3]) # Skip Connection
        c4 = cls.conv_block(u3, 128)

        # Upsample 2
        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2]) # Skip Connection
        c5 = cls.conv_block(u2, 64)

        # Upsample 3
        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1]) # Skip Connection
        c6 = cls.conv_block(u1, 32)

        # --- 5. OUTPUT ---
        # Conv 1x1 para colapsar canales a 1 (Temperatura HR)
        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)

        model = Model([inp_dyn, inp_st], out, name="Exp4_Hybrid_UNet_LSTM")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model    

    @classmethod
    def build_transformer(cls):
        """Experimento 3: U-Net con Transformer Bottleneck"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        # Bridge & Encoder (Igual a U-Net)
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(inp_dyn)
        x = Concatenate()([x_up, inp_st])

        c1 = cls.conv_block(x, 32)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        c2 = cls.conv_block(p1, 64)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        c3 = cls.conv_block(p2, 128)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # --- TRANSFORMER BOTTLENECK ---
        x_neck = TimeDistributed(Conv2D(256, (1, 1), padding="same"))(p3)
        x_trans = cls.temporal_transformer_block(x_neck, embed_dim=256, num_heads=4)

        # Decoder
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(x_trans)
        u3 = Concatenate()([u3, c3])
        c4 = cls.conv_block(u3, 128)

        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2])
        c5 = cls.conv_block(u2, 64)

        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1])
        c6 = cls.conv_block(u1, 32)

        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)

        model = Model([inp_dyn, inp_st], out, name="Exp3_TransformerUNet")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model
        
# ==========================================
# 4. ENTRENAMIENTO Y VISUALIZACIÓN
# ==========================================
def run_experiment(model, train_ds, val_ds, experiment_name):
    """Ejecuta el ciclo de entrenamiento estandarizado"""
    print(f"\n🧪 Iniciando {experiment_name}...")

    # Callbacks
    callbacks = [
        ModelCheckpoint(f"{experiment_name}_best.h5", save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        CSVLogger(f"{experiment_name}_log.csv")
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    return history

def visualize_results(model, val_ds, title):
    """Genera gráfica comparativa Input/Pred/Target y la guarda"""
    try:
        # Extraer un batch para visualizar
        (x_lr, x_st), y_true = next(iter(val_ds))
        
        # Predecir
        y_pred = model.predict([x_lr, x_st], verbose=0)

        # Visualizar primer sample, último frame de la secuencia
        idx = 0
        t = Config.SEQ_LEN - 1 

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{title} - Sample {idx} Frame {t}")

        # 1. Input LR
        plt.subplot(1, 3, 1)
        # x_lr shape: (Batch, Time, Lat, Lon, Chan)
        plt.imshow(x_lr[idx, t, :, :, 0], cmap='viridis') 
        plt.title("Input Low Res (LR)")
        plt.axis('off')

        # 2. Predicción HR
        plt.subplot(1, 3, 2)
        plt.imshow(y_pred[idx, t, :, :, 0], cmap='viridis')
        plt.title("Prediction (HR)")
        plt.axis('off')

        # 3. Ground Truth HR
        plt.subplot(1, 3, 3)
        plt.imshow(y_true[idx, t, :, :, 0], cmap='viridis')
        plt.title("Ground Truth (HR)")
        plt.axis('off')
        
        # Guardar imagen para no bloquear la ejecución
        filename = f"result_{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"📸 Visualización guardada: {filename}")
        
    except Exception as e:
        print(f"⚠️ Error al generar visualización: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Limpiar sesiones anteriores para liberar memoria
    tf.keras.backend.clear_session()
    
    print("🚀 Iniciando Pipeline de Big Data...")

    # 1. Inicializar Pipeline
    pipeline = BigDataPipeline(Config)
    
    # 2. Procesar Datos Estáticos (En memoria, son ligeros)
    pipeline.process_static_data()
    
    # 3. Ejecutar ETL (Crea carpeta ./processed_cache_zarr si no existe)
    pipeline.run_etl_process()
    
    # 4. OBTENER DATASETS (¡Paso crítico faltante!)
    # Esto conecta los generadores a los archivos Zarr
    train_ds, val_ds = pipeline.get_tf_datasets()
    
    # Verificación rápida (opcional)
    try:
        x_samp, y_samp = next(iter(train_ds))
        print(f"✅ Datos cargados correctamente. Input Shape: {x_samp[0].shape}")
    except Exception as e:
        print(f"⚠️ Error verificando datasets: {e}")

    # 5. Definir Modelos a Probar
    experiments = [
        #("UNet", ModelZoo.build_unet),
         #s("ConvLSTM", ModelZoo.build_convlstm), # Descomentar para probar otros
        ("Transformer", ModelZoo.build_transformer)
        #("Hybrid_UNet_LSTM", ModelZoo.build_hybrid_unet_lstm),
    ]

    histories = {}

    # 6. Bucle de Experimentos
    for name, builder in experiments:
        print(f"\n🏗️ Construyendo modelo: {name}...")
        
        # Limpiar sesión entre modelos para evitar fugas de memoria
        if name != experiments[0][0]:
            tf.keras.backend.clear_session()
            
        model = builder()
        
        # Opcional: Imprimir resumen
        # model.summary() 

        # Entrenar
        hist = run_experiment(model, train_ds, val_ds, name)
        histories[name] = hist

        # Visualizar resultados
        visualize_results(model, val_ds, f"Resultados: {name}")

    print("\n✅ Todos los experimentos finalizados correctamente.")