import os
import platform
import tensorflow as tf
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN GLOBAL (Armonizada)
# ==========================================
class Config:
    # Rutas de archivos
    PATH_HR = "/Users/kerincardona/weather_urban_downscaling_project/data/processed/estaciones_interpoladas_final.nc"
    PATH_LR = "/Users/kerincardona/Documents/weather_urban_downscaling/era5land/era5_combined_01_03.grib"
    PATH_STATIC = '/Users/kerincardona/extracted_zarr_data/weather_static_features.zarr'

    # Guardamos el Zarr en la carpeta de procesados
    PATH_CACHE = "data/processed/weather_cache.zarr"

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


# Configurar semillas
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

print(f"🔧 Configuración cargada. Mac Silicon detectado: {Config.IS_MAC_SILICON}")