#!/usr/bin/env python
# coding: utf-8

"""
Urban Weather Downscaling Pipeline - Versión Simplificada
---------------------------------------------------------
Usa la clase DataPipeline original que ya funciona

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
import tensorflow as tf
from scipy.interpolate import NearestNDInterpolator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D,
    TimeDistributed, Concatenate, BatchNormalization,
    Activation, Add, LeakyReLU, Resizing, Dropout
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
    # Rutas - Heredadas de la configuración central
    PATH_HR = CentralConfig.PATH_HR
    PATH_LR = CentralConfig.PATH_LR
    PATH_STATIC = CentralConfig.PATH_STATIC
    
    # Hiperparámetros (locales para este experimento)
    SEED = 42
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    SEQ_LEN = 2
    SPLIT_FRACTION = 0.9
    
    # Dimensiones
    HR_SHAPE = (251, 251)
    LR_SHAPE = (4, 3)
    
    # Hardware
    IS_MAC_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# Configurar semillas
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

print(f"🔧 Configuración cargada. Mac Silicon: {Config.IS_MAC_SILICON}")

# ==========================================
# 2. DATA PIPELINE ORIGINAL (QUE SÍ FUNCIONA)
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.cfg = config
        self.ds_hr = None
        self.ds_lr = None
        self.ds_static_single = None 
        
    def load_hr_data(self):
        """Carga datos HR (Target)"""
        print("💾 Loading HR Data...")
        ds = xr.open_dataset(self.cfg.PATH_HR)
        ds = ds.astype('float32') 
        ds["tas_C"] = ds["tas"] - 273.15
        ds["tas_C"].attrs.update({"units": "°C"})
        self.ds_hr = ds
        return ds

    def load_lr_data(self):
        """Carga datos LR (Input)"""
        print("💾 Loading LR Data...")
        
        # Cargar ERA5
        ds = xr.open_dataset(
            self.cfg.PATH_LR,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
                "filter_by_keys": {"typeOfLevel": "surface"},
                "errors": "ignore"
            }
        )
        ds = ds.astype('float32')
        ds["t2m"] = ds["t2m"] - 273.15
        
        # Sincronización Temporal
        ds_stacked = ds.stack(idx=('time', 'step'))
        ds_flat = ds_stacked.swap_dims({'idx': 'valid_time'}).sortby('valid_time')
        ds_flat = ds_flat.drop_vars(['time', 'step'], errors='ignore')
        ds_flat = ds_flat.rename({'valid_time': 'time'})
        ds_flat = ds_flat.sel(time=self.ds_hr.time)
        
        # Recorte Geográfico
        min_lat, max_lat = self.ds_hr.latitude.min().item(), self.ds_hr.latitude.max().item()
        min_lon, max_lon = self.ds_hr.longitude.min().item(), self.ds_hr.longitude.max().item()
        
        buffer = 0.15
        ds_clipped = ds_flat.sel(
            latitude=slice(max_lat + buffer, min_lat - buffer), 
            longitude=slice(min_lon - buffer, max_lon + buffer)
        ).compute()
        
        # Ordenar coordenadas
        ds_clipped = ds_clipped.sortby('latitude', ascending=False)
        ds_clipped = ds_clipped.sortby('longitude', ascending=True)

        self.ds_lr = ds_clipped
        print(f"✅ LR Data recortada.")
        return self.ds_lr

    def process_static_data(self):
        """Interpolación eficiente (Low RAM)"""
        print("🗺️ Processing Static Data...")
        ds_static = xr.open_zarr(self.cfg.PATH_STATIC)
        
        indices = ds_static['index'].values
        lats = np.array([float(x.split('_')[0]) for x in indices])
        lons = np.array([float(x.split('_')[1]) for x in indices])
        points = np.column_stack((lats, lons))
        
        grid_y = self.ds_hr['latitude'].values
        grid_x = self.ds_hr['longitude'].values
        target_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))
        
        dim_y, dim_x = self.ds_hr.sizes['y'], self.ds_hr.sizes['x']
        static_layers = []
        
        for var in ds_static.data_vars:
            if var == 'index': continue
            values = ds_static[var].values
            interpolator = NearestNDInterpolator(points, values)
            interp_flat = interpolator(target_points)
            static_layers.append(interp_flat.reshape(dim_y, dim_x))
            
        self.ds_static_single = np.stack(static_layers, axis=-1).astype('float32')

    def get_prepared_datasets(self):
        """
        Generador final: 
        1. Dimensiones corregidas.
        2. Estandarización Dinámica por canal.
        3. Estandarización Estática por canal.
        """
        print("⚙️ Preparando tensores...")
        
        # 1. Transpose Correcto
        X_lr = self.ds_lr.to_array(dim='feature') \
                   .transpose('time', 'latitude', 'longitude', 'feature').values
                   
        y_hr = self.ds_hr.drop_vars('tas', errors='ignore').to_array(dim='feature') \
                   .transpose('time', 'y', 'x', 'feature').values
        
        print(f"   Input Shape: {X_lr.shape}")
        
        # 2. Split
        split_idx = int(len(X_lr) * self.cfg.SPLIT_FRACTION)
        X_train, X_val = X_lr[:split_idx], X_lr[split_idx:]
        y_train, y_val = y_hr[:split_idx], y_hr[split_idx:]
        
        # 3. SCALING DINÁMICO (Por canal)
        print("📊 Estandarizando variables dinámicas...")
        mean_X = X_train.mean(axis=(0, 1, 2), keepdims=True)
        std_X = X_train.std(axis=(0, 1, 2), keepdims=True)
        
        mean_y = y_train.mean(axis=(0, 1, 2), keepdims=True)
        std_y = y_train.std(axis=(0, 1, 2), keepdims=True)
        
        std_X = np.where(std_X == 0, 1.0, std_X)
        std_y = np.where(std_y == 0, 1.0, std_y)
        
        X_train_sc = (X_train - mean_X) / std_X
        X_val_sc = (X_val - mean_X) / std_X
        y_train_sc = (y_train - mean_y) / std_y
        y_val_sc = (y_val - mean_y) / std_y
        
        # 4. SCALING ESTÁTICO
        print("🏔️ Estandarizando variables estáticas...")
        mean_st = self.ds_static_single.mean(axis=(0, 1), keepdims=True)
        std_st = self.ds_static_single.std(axis=(0, 1), keepdims=True)
        std_st = np.where(std_st == 0, 1.0, std_st)
        
        static_scaled = (self.ds_static_single - mean_st) / std_st
        
        # 5. Secuencias
        def create_sequences(x_data, y_data, seq_len):
            X_seq, y_seq = [], []
            for i in range(len(x_data) - seq_len + 1):
                X_seq.append(x_data[i : i + seq_len])
                y_seq.append(y_data[i : i + seq_len])
            return np.array(X_seq), np.array(y_seq)
        
        X_tr_seq, y_tr_seq = create_sequences(X_train_sc, y_train_sc, self.cfg.SEQ_LEN)
        X_val_seq, y_val_seq = create_sequences(X_val_sc, y_val_sc, self.cfg.SEQ_LEN)
        
        # 6. Bloque estático
        static_block = np.repeat(static_scaled[np.newaxis, ...], self.cfg.SEQ_LEN, axis=0)
        
        def make_tf_dataset(x_seq, y_seq, static_block):
            ds_dyn = tf.data.Dataset.from_tensor_slices(x_seq)
            ds_tar = tf.data.Dataset.from_tensor_slices(y_seq)
            ds_stat = tf.data.Dataset.from_tensors(static_block).repeat()
            return tf.data.Dataset.zip(((ds_dyn, ds_stat), ds_tar))

        print("🚀 Construyendo Datasets finales...")
        with tf.device('/CPU:0'):
            train_ds = make_tf_dataset(X_tr_seq, y_tr_seq, static_block)
            train_ds = train_ds.shuffle(100).batch(self.cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            val_ds = make_tf_dataset(X_val_seq, y_val_seq, static_block)
            val_ds = val_ds.batch(self.cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
        return train_ds, val_ds

# ==========================================
# 3. MODELOS (Versión simplificada)
# ==========================================
class ModelZoo:
    @staticmethod
    def get_optimizer(lr):
        if Config.IS_MAC_SILICON:
            from tensorflow.keras.optimizers.legacy import Adam
            return Adam(learning_rate=lr)
        else:
            from tensorflow.keras.optimizers import Adam
            return Adam(learning_rate=lr)

    @staticmethod
    def build_convlstm():
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
        out = TimeDistributed(Conv2D(1, (1, 1), activation="linear"))(x)

        model = Model([inp_dyn, inp_st], out, name="ConvLSTM")
        model.compile(optimizer=ModelZoo.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
def run_experiment(model, train_ds, val_ds, experiment_name):
    """Ejecuta el ciclo de entrenamiento"""
    print(f"\n🧪 Iniciando {experiment_name}...")

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

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("URBAN WEATHER DOWNSCALING - VERSIÓN SIMPLIFICADA")
    print("=" * 60)
    
    try:
        # 1. Preparar Datos (Usando la clase original que SÍ funciona)
        pipeline = DataPipeline(Config)  # ¡IMPORTANTE! Usa DataPipeline, no ScalableDataPipeline
        pipeline.load_hr_data()
        pipeline.load_lr_data()
        pipeline.process_static_data()

        train_ds, val_ds = pipeline.get_prepared_datasets()

        # 2. Definir Modelo Simple (solo ConvLSTM para probar)
        tf.keras.backend.clear_session()
        model = ModelZoo.build_convlstm()
        model.summary()

        # 3. Entrenar
        history = run_experiment(model, train_ds, val_ds, "ConvLSTM_Simple")

        print("\n✅ Entrenamiento completado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()