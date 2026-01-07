import os
import platform
import tensorflow as tf
import numpy as np

class Config:
    # ---------------------------------------------------------
    # 1. DETECCIÓN DE HARDWARE (NUEVO)
    # ---------------------------------------------------------
    # Detectamos robustamente si es Mac Apple Silicon
    IS_MAC_SILICON = platform.system() == "Darwin" and (platform.machine() == 'arm64' or platform.processor() == 'arm')

    # Variable 'DEVICE' requerida por run_ablation.py
    if IS_MAC_SILICON:
        DEVICE = "MPS (Metal Performance Shaders)"
    elif len(tf.config.list_physical_devices('GPU')) > 0:
        DEVICE = "CUDA (NVIDIA)"
    else:
        DEVICE = "CPU Standard"

    # ---------------------------------------------------------
    # 2. RUTAS DE ARCHIVOS (Tus rutas originales)
    # ---------------------------------------------------------
    PATH_HR = "/Users/kerincardona/weather_urban_downscaling/data/processed/estaciones_interpoladas_final.nc"
    PATH_LR = "/Users/kerincardona/Documents/weather_urban_downscaling/era5land/era5_combined_01_03.grib"
    PATH_STATIC = '/Users/kerincardona/weather_urban_downscaling/data/processed/weather_static_FINAL_stations.zarr'
    PATH_CACHE = "data/processed/weather_cache.zarr"

    # ---------------------------------------------------------
    # 3. HIPERPARÁMETROS
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 8  
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    SEQ_LEN = 3 
    SPLIT_FRACTION = 0.8 

    # ---------------------------------------------------------
    # 4. DIMENSIONES
    # ---------------------------------------------------------
    HR_SHAPE = (251, 251)
    LR_SHAPE = (4, 3) 
    CHANNELS = 9
    STATIC_CHANNELS = 13

# Configuración Global de Semillas
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

print(f"🔧 Configuración cargada. Hardware: {Config.DEVICE}")