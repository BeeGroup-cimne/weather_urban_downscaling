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
    PATH_LR = str(BASE_DIR / "data" / "processed" / "era5land" / "lr_2017.grib")
    PATH_STATIC = str(BASE_DIR / "data" / "processed" / "weather_static_FINAL_stations.zarr")
    PATH_CACHE = str(BASE_DIR / "data" / "processed" / "weather_cache.zarr")

    # Rutas de salida y cache procesado
    EXPERIMENTS_DIR = str(BASE_DIR / "experiments")
    STATS_PATH = str(BASE_DIR / "data" / "processed" / "stats_config.npz")
    STATIC_CACHE_PATH = str(BASE_DIR / "data" / "processed" / "static_processed.npy")

    # ---------------------------------------------------------
    # 2.1 SPLITS TEMPORALES (para paper)
    # ---------------------------------------------------------
    SPLIT_MODE = "time"  # "time" o "fraction"
    TRAIN_START = "2017-01-01"
    TRAIN_END = "2017-11-01"
    VAL_START = "2017-11-01"
    VAL_END = "2017-12-01"
    TEST_START = "2017-12-01"
    TEST_END = "2018-01-01"

    # ---------------------------------------------------------
    # 3. HIPERPARÁMETROS
    # ---------------------------------------------------------

    SEED = 42
    BATCH_SIZE = 4  
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    SEQ_LEN = 6 
    SPLIT_FRACTION = 0.8 
    # Limitar pasos por época para runs locales rápidos (None = usar todo)
    MAX_STEPS_PER_EPOCH = 200
    SHUFFLE_BUFFER_SIZE = 50
    PREFETCH_BUFFER_SIZE = 1

    # ---------------------------------------------------------
    # 4. DIMENSIONES
    # ---------------------------------------------------------
    
    HR_SHAPE = (251, 251)
    LR_SHAPE = (4, 3) 
    CHANNELS = 9
    STATIC_CHANNELS = 13
    UNET_BASE_FILTERS = 32

    # Regularización / generalización
    L2_WEIGHT_DECAY = 1e-4
    GAUSSIAN_NOISE_STD = 0.01
    STATS_TRAIN_ONLY = True
    # Robustez de normalización: evita amplificación extrema cuando std ~= 0
    NORM_NEAR_ZERO_STD_THRESHOLD = 0.01
    NORM_STD_FLOOR = 0.01

    # ---------------------------------------------------------
    # 5. TILE-BASED TRAINING (Optional)
    # ---------------------------------------------------------
    PATCH_SIZE = (96, 96)
    PATCHES_PER_EPOCH = 2000
    VAL_PATCHES_PER_EPOCH = 200
    TILE_SAMPLER = "static_weighted"  # "static_weighted" | "uniform" | "error_weighted"
    TILE_WEIGHT_ALPHA = 0.85          # mix of weighted vs uniform sampling
    TILE_WEIGHT_GAMMA = 1.0           # emphasize extremes in weight map
    TILE_MIN_PROB = 1e-6
    TILE_ERROR_MAP_PATH = str(BASE_DIR / "experiments" / "tiles_error_map.npy")
    TEMPORAL_STRIDE = 1
    TEMPORAL_SAMPLER = "uniform"  # "uniform" | "weighted" | "weighted_station"
    TEMPORAL_WEIGHT_GAMMA = 1.0
    TEMPORAL_MIN_PROB = 1e-6
    TEMPORAL_SEASON_BALANCE = False
    STATION_GRIB_PATH = str(BASE_DIR / "data" / "processed" / "stations_t2m.grib")


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
