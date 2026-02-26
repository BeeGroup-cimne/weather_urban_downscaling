"""
Configuración optimizada para entrenamiento en GPU server con alta memoria.
Hereda de Config base y sobreescribe solo lo necesario.
"""

import os
import platform
import numpy as np
from .config import Config

class GPUServerConfig(Config):
    # ==========================================================
    # DETECCIÓN DE HARDWARE Y MEMORIA
    # ==========================================================
    
    # Detectar hardware con soporte para múltiples GPUs
    IS_MAC_SILICON = platform.system() == "Darwin" and (platform.machine() == 'arm64' or platform.processor() == 'arm')
    
    DEVICE = "CPU Standard"
    GPU_MEMORY_GB = None
    
    try:
        import tensorflow as tf
        
        # Detectar GPUs disponibles
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            DEVICE = f"CUDA ({len(gpus)} GPUs)"
            
            # Intentar obtener memoria GPU (NVIDIA específico)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    GPU_MEMORY_GB = int(result.stdout.strip().split('\n')[0]) // 1024
                    print(f"🎮 GPU Memory detectada: {GPU_MEMORY_GB}GB")
            except:
                GPU_MEMORY_GB = 24  # Default asumpción
                
        elif IS_MAC_SILICON:
            DEVICE = "MPS (Metal Performance Shaders)"
            GPU_MEMORY_GB = 16  # Unified memory assumption
            
    except ImportError:
        pass
    
    # ==========================================================
    # CONFIGURACIÓN ADAPTATIVA SEGÚN MEMORIA
    # ==========================================================
    
    # Default parameters (overwritten by GPU detection logic below)
    EFFECTIVE_BATCH_SIZE = Config.BATCH_SIZE
    GRADIENT_ACCUMULATION_STEPS = 1
    MIXED_PRECISION = False
    MODEL_DIM = 64 # Valor base

    if GPU_MEMORY_GB is None:
        # Fallback para CPU
        BATCH_SIZE = 1
        SEQ_LEN = 3
        MODEL_DIM = 64
        MIXED_PRECISION = False
        GRADIENT_ACCUMULATION_STEPS = 1
        
    elif GPU_MEMORY_GB < 16:
        # GPUs con poca memoria (RTX 3060, etc)
        BATCH_SIZE = 1
        SEQ_LEN = 4
        MODEL_DIM = 64
        MIXED_PRECISION = True
        GRADIENT_ACCUMULATION_STEPS = 4
        
    elif GPU_MEMORY_GB < 24:
        # GPUs medianas (RTX 3080, etc)
        BATCH_SIZE = 2
        SEQ_LEN = 12
        MODEL_DIM = 128
        MIXED_PRECISION = True
        GRADIENT_ACCUMULATION_STEPS = 2
        
    elif GPU_MEMORY_GB < 50:
         # L40S / RTX 6000 Ada / A6000 (48GB VRAM)
         # Optimized for 48GB: Can fit bigger batches without full A100 scale
        BATCH_SIZE = 8
        SEQ_LEN = 6
        MODEL_DIM = 128
        MIXED_PRECISION = True 
        GRADIENT_ACCUMULATION_STEPS = 1
         
    else:
        # GPUs grandes (A100, H100, H200 > 80GB)
        BATCH_SIZE = 16
        SEQ_LEN = 6
        MODEL_DIM = 128
        MIXED_PRECISION = True
        GRADIENT_ACCUMULATION_STEPS = 1

    # Override: desactivar mixed precision para evitar loss scaling en loop custom
    MIXED_PRECISION = False
    
    # Effective batch size update
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    # ==========================================================
    # HIPERPARÁMETROS OPTIMIZADOS (Overrides)
    # ==========================================================
    
    EPOCHS = 100  # Más épocas con batches más pequeños
    LEARNING_RATE = 5e-5  # Learning rate más pequeño para estabilidad
    
    # Dimensiones de modelo adaptativas
    UNET_FILTERS = 64  # Reducido de 128
    CONVLSTM_FILTERS = 64  # Reducido de 128
    MAMBA_MODEL_DIM = MODEL_DIM
    MAMBA_STATE_DIM = MODEL_DIM // 2
    
    # ==========================================================
    # CONFIGURACIÓN DE MEMORIA Y PERFORMANCE
    # ==========================================================
    
    # Configuración de datos optimizada
    ZARR_CHUNK_SIZE = 50  # Chunks más pequeños
    STATIC_BROADCASTING = True  # Evitar np.repeat
    PREFETCH_BUFFER_SIZE = 1  # Buffer mínimo para evitar acumulación
    SHUFFLE_BUFFER_SIZE = 50  # Shuffle buffer reducido
    
     # Memory monitoring
    MEMORY_MONITORING = True
    GPU_MEMORY_FRACTION = 0.9  # Usar 90% de GPU memory
    
    # Checkpointing y recuperación
    ENABLE_CHECKPOINTING = True
    SAVE_FREQUENCY = 5  # Guardar cada 5 épocas
    
    # Early stopping más agresivo
    EARLY_STOPPING_PATIENCE = 10
    
    # Limitar pasos por época (None = usar todo el dataset)
    MAX_STEPS_PER_EPOCH = None

    # ==========================================================
    # CONFIGURACIÓN DE DOCKER Y ENTORNO
    # ==========================================================
    
    # Variables de entorno para Docker
    DOCKER_MEMORY_LIMIT = "32g"
    DOCKER_SHM_SIZE = "8g"
    CUDA_VISIBLE_DEVICES = "0"  # Primera GPU por defecto
    
    # ==========================================================
    # MÉTODOS ÚTILES
    # ==========================================================
    
    @classmethod
    def print_memory_info(cls):
        """Imprimir información de memoria configurada"""
        print(f"🧠 Configuración de Memoria:")
        print(f"   Hardware detectado: {cls.DEVICE}")
        print(f"   GPU Memory: {cls.GPU_MEMORY_GB}GB")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Gradient Accumulation: {cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Effective Batch Size: {cls.EFFECTIVE_BATCH_SIZE}")
        print(f"   Sequence Length: {cls.SEQ_LEN}")
        print(f"   Model Dimension: {cls.MAMBA_MODEL_DIM}")
        print(f"   Mixed Precision: {cls.MIXED_PRECISION}")
        print(f"   Memory Fraction: {cls.GPU_MEMORY_FRACTION}")
    
    @classmethod
    def setup_gpu_memory(cls):
        """Configurar memory growth de TensorFlow"""
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    if cls.GPU_MEMORY_GB is not None:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=int(cls.GPU_MEMORY_FRACTION * cls.GPU_MEMORY_GB * 1024)
                            )]
                        )
                print(f"✅ GPU memory configurada para {cls.GPU_MEMORY_FRACTION * 100}% de {cls.GPU_MEMORY_GB}GB")
        except Exception as e:
            print(f"⚠️ Error configurando GPU memory: {e}")
    
    @classmethod
    def get_optimizer_config(cls):
        """Configuración de optimizador adaptativa"""
        return {
            'learning_rate': cls.LEARNING_RATE,
            'gradient_accumulation_steps': cls.GRADIENT_ACCUMULATION_STEPS,
            'mixed_precision': cls.MIXED_PRECISION
        }

# Configurar memoria GPU si es posible
GPUServerConfig.setup_gpu_memory()

# Imprimir configuración
GPUServerConfig.print_memory_info()

print(f"🔧 Configuración GPU Server cargada")
print(f"📂 Directorio base: {GPUServerConfig.BASE_DIR}")
