import os
import platform
import numpy as np
from pathlib import Path

class Config:
    # ---------------------------------------------------------
    # 1. DETECCIÓN DE HARDWARE 
    # ---------------------------------------------------------
    IS_MAC_SILICON = platform.system() == "Darwin" and (platform.machine() == 'arm64' or platform.processor() == 'arm')

    # Detectar dispositivo disponible
    DEVICE = "CPU Standard"
    
    try:
        import tensorflow as tf
        if IS_MAC_SILICON:
            DEVICE = "MPS (Metal Performance Shaders)"
        elif len(tf.config.list_physical_devices('GPU')) > 0:
            DEVICE = "CUDA (NVIDIA)"
    except ImportError:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            DEVICE = "CUDA (NVIDIA)"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            DEVICE = "MPS (Metal Performance Shaders)"
    except ImportError:
        pass

    # ---------------------------------------------------------
    # 2. RUTAS DE ARCHIVOS (Dinámicas - Relativas al proyecto)
    # ---------------------------------------------------------
    
    # Directorio raíz del proyecto (carpeta que contiene 'config' y 'data')
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Rutas de datos de entrada
    PATH_HR = str(BASE_DIR / "data" / "processed" / "estaciones_interpoladas_final.nc")
    PATH_LR = str(BASE_DIR / "data" / "processed" / "era5land" / "lr_2010_2025.grib")
    PATH_STATIC = str(BASE_DIR / "data" / "processed" / "weather_static_FINAL_stations.zarr")
    PATH_CACHE = str(BASE_DIR / "data" / "processed" / "weather_cache.zarr")

    # Rutas de salida y cache procesado
    EXPERIMENTS_DIR = str(BASE_DIR / "experiments")
    STATS_PATH = str(BASE_DIR / "data" / "processed" / "stats_config.npz")
    STATIC_CACHE_PATH = str(BASE_DIR / "data" / "processed" / "static_processed.npy")

    # ---------------------------------------------------------
    # 3. HIPERPARÁMETROS
    # ---------------------------------------------------------

    SEED = 42
    BATCH_SIZE = 4  
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    SEQ_LEN = 6 
    SPLIT_FRACTION = 0.8 

    # ---------------------------------------------------------
    # 4. DIMENSIONES
    # ---------------------------------------------------------
    
    HR_SHAPE = (251, 251)
    LR_SHAPE = (4, 3) 
    CHANNELS = 9
    STATIC_CHANNELS = 13


# Configuración Global de Semillas (Solo si las librerías están disponibles)
np.random.seed(Config.SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(Config.SEED)
except ImportError:
    pass

try:
    import torch
    torch.manual_seed(Config.SEED)
except ImportError:
    pass

print(f"🔧 Configuración cargada. Hardware: {Config.DEVICE}")
print(f"📂 Directorio base: {Config.BASE_DIR}")