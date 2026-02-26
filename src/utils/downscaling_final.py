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
    SEQ_LEN = 1  # Ventana de tiempo
    SPLIT_FRACTION = 0.9  # 90% Train, 10% Val

    # Dimensiones (Detectadas dinámicamente, pero definimos defaults)
    HR_SHAPE = (251, 251)
    LR_SHAPE = (4, 3)  # Ajustado a tu recorte final

    # Configuración de Hardware
    IS_MAC_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"


# Configurar semillas
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

print(f"🔧 Configuración cargada. Mac Silicon detectado: {Config.IS_MAC_SILICON}")


# ==========================================
# 2. MOTOR DE DATOS (Data Engine)
# ==========================================
from scipy.interpolate import NearestNDInterpolator

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
        """
        MÉTODO PROFESIONAL: Bounding Box Intersection con Buffer.
        Garantiza cobertura total sin adivinar índices.
        """
        print("💾 Loading LR Data (Geospatial Bounding Box)...")
        
        # 1. Cargar ERA5 (LR)
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
        
        # 2. Sincronización Temporal (Standard)
        ds_stacked = ds.stack(idx=('time', 'step'))
        ds_flat = ds_stacked.swap_dims({'idx': 'valid_time'}).sortby('valid_time')
        ds_flat = ds_flat.drop_vars(['time', 'step'], errors='ignore')
        ds_flat = ds_flat.rename({'valid_time': 'time'})
        ds_flat = ds_flat.sel(time=self.ds_hr.time)
        
        # --- 3. EL RECORTE DEFINITIVO (GEOSPATIAL CROP) ---
        
        # A) Obtener límites exactos de HR (Target)
        min_lat, max_lat = self.ds_hr.latitude.min().item(), self.ds_hr.latitude.max().item()
        min_lon, max_lon = self.ds_hr.longitude.min().item(), self.ds_hr.longitude.max().item()
        
        # B) Definir un BUFFER de seguridad
        # ERA5 tiene ~0.1° de resolución. Añadimos 0.15° extra a cada lado
        # para asegurar que atrapamos los píxeles vecinos necesarios para la interpolación.
        buffer = 0.15
        
        target_lat_max = max_lat + buffer
        target_lat_min = min_lat - buffer
        target_lon_min = min_lon - buffer
        target_lon_max = max_lon + buffer
        
        print(f"   🎯 HR Bounds: Lat [{min_lat:.2f}, {max_lat:.2f}], Lon [{min_lon:.2f}, {max_lon:.2f}]")
        print(f"   🛡️ LR Buffer: Lat [{target_lat_min:.2f}, {target_lat_max:.2f}], Lon [{target_lon_min:.2f}, {target_lon_max:.2f}]")

        # C) Recorte Geográfico (Slice)
        # NOTA: En archivos GRIB/NetCDF climáticos, la latitud suele venir de Norte a Sur (Decreciente).
        # Por eso el slice va de (Max -> Min).
        ds_clipped = ds_flat.sel(
            latitude=slice(target_lat_max, target_lat_min), 
            longitude=slice(target_lon_min, target_lon_max)
        ).compute()
        
        # D) Comprobación de seguridad
        if ds_clipped.sizes['latitude'] == 0 or ds_clipped.sizes['longitude'] == 0:
            raise ValueError("❌ El recorte resultó vacío. Revisa si las longitudes están en formato 0-360 vs -180/180.")

        # E) Ordenar coordenadas estrictamente (Latitud descendente, Longitud ascendente)
        # Esto elimina el riesgo de matrices invertidas o 'flipped'.
        ds_clipped = ds_clipped.sortby('latitude', ascending=False)
        ds_clipped = ds_clipped.sortby('longitude', ascending=True)

        # 4. Asignación y Actualización de Config
        self.ds_lr = (
            ds_clipped
            .ffill(dim='longitude').bfill(dim='longitude')
            .ffill(dim='latitude').bfill(dim='latitude')
        )
        
        # Actualizar la configuración con la realidad del terreno
        real_h = self.ds_lr.sizes['latitude']
        real_w = self.ds_lr.sizes['longitude']
        self.cfg.LR_SHAPE = (real_h, real_w)
        
        print(f"✅ LR Data recortada profesionalmente.")
        print(f"   Shape Final (Lat x Lon): {self.cfg.LR_SHAPE}")
        
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

    def create_sequences_dynamic(self, x_data, y_data, seq_len):
        """Secuencias dinámicas"""
        X_seq, y_seq = [], []
        for i in range(len(x_data) - seq_len + 1):
            X_seq.append(x_data[i : i + seq_len])
            y_seq.append(y_data[i : i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    def get_prepared_datasets(self):
        """
        Generador final V3: 
        1. Dimensiones corregidas.
        2. Estandarización Dinámica por canal.
        3. Estandarización Estática por canal (NUEVO).
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
        print("📊 Estandarizando variables dinámicas (Temp, Presión)...")
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
        
        # --- 4. SCALING ESTÁTICO (NUEVO: Elevación, etc.) ---
        print("🏔️ Estandarizando variables estáticas (Elevación, LandUse)...")
        # ds_static_single shape: (251, 251, 18) -> (Alto, Ancho, Canales)
        # Calculamos media sobre ejes 0 y 1 (Alto y Ancho), manteniendo Canal (2)
        
        mean_st = self.ds_static_single.mean(axis=(0, 1), keepdims=True)
        std_st = self.ds_static_single.std(axis=(0, 1), keepdims=True)
        
        # Seguridad
        std_st = np.where(std_st == 0, 1.0, std_st)
        
        # Aplicamos la transformación y guardamos en una variable temporal
        # (No sobreescribimos self.ds_static_single por si quieres re-usar la clase)
        static_scaled = (self.ds_static_single - mean_st) / std_st
        
        print(f"   Elevación media (normalizada): {static_scaled[..., 0].mean():.4f}") # Debería ser ~0.0
        
        # 5. Secuencias
        X_tr_seq, y_tr_seq = self.create_sequences_dynamic(X_train_sc, y_train_sc, self.cfg.SEQ_LEN)
        X_val_seq, y_val_seq = self.create_sequences_dynamic(X_val_sc, y_val_sc, self.cfg.SEQ_LEN)
        
        # 6. Bloque estático (Usando el YA ESCALADO)
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
# 4. ENTRENAMIENTO Y EJECUCIÓN
# ==========================================
def run_experiment(model, train_ds, val_ds, experiment_name):
    """Ejecuta el ciclo de entrenamiento estandarizado"""
    print(f"\n🧪 Iniciando {experiment_name}...")

    # Callbacks armonizados (Todos usan la configuración del Exp 3)
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
    """Genera gráfica comparativa Input/Pred/Target"""
    (x_dyn, x_st), y_true = next(iter(val_ds))
    y_pred = model.predict([x_dyn, x_st])

    # Visualizar primer sample, frame 1
    idx, t = 0, 1

    plt.figure(figsize=(15, 5))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(x_dyn[idx, t, :, :, 0], cmap='viridis')
    plt.title("Input (LR)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y_pred[idx, t, :, :, 0], cmap='viridis')
    plt.title("Prediction (HR)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(y_true[idx, t, :, :, 0], cmap='viridis')
    plt.title("Ground Truth (HR)")
    plt.axis('off')
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Preparar Datos (Una sola vez para todos)
    pipeline = DataPipeline(Config)
    pipeline.load_hr_data()
    pipeline.load_lr_data()
    pipeline.process_static_data()

    train_ds, val_ds = pipeline.get_prepared_datasets()

    # 2. Definir Experimentos
    experiments = [
        ("ConvLSTM", ModelZoo.build_convlstm),
        ("UNet", ModelZoo.build_unet),
        ("Transformer", ModelZoo.build_transformer)
    ]

    histories = {}

    # 3. Correr Bucle de Experimentos
    for name, builder in experiments:
        tf.keras.backend.clear_session()  # Limpiar memoria entre modelos
        model = builder()
        model.summary()

        hist = run_experiment(model, train_ds, val_ds, name)
        histories[name] = hist

        visualize_results(model, val_ds, f"Resultados: {name}")

    print("\n✅ Todos los experimentos finalizados correctamente.")


