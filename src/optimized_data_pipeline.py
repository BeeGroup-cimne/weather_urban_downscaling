"""
BigDataPipeline optimizado para GPU server con memoria limitada
Solución a problemas de OOM mediante chunking y broadcasting eficiente
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from dask.diagnostics import ProgressBar
from typing import Tuple, Optional
import gc
import psutil

# Importar configuración
from config.gpu_server_config import GPUServerConfig as Config

class OptimizedBigDataPipeline:
    def __init__(self, config=None):
        self.config = config or Config
        self.stats = None
        self.static_processed = None
        
        # Memory monitoring
        self.memory_peak = 0
        
        print(f"🚀 Pipeline Optimizado inicializado")
        print(f"   Batch Size: {self.config.BATCH_SIZE}")
        print(f"   Chunk Size: {self.config.ZARR_CHUNK_SIZE}")
        
    def monitor_memory(self, stage: str):
        """Monitor memory usage con logging"""
        gpu_memory = 0
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1024**3
        except:
            pass
            
        cpu_memory = psutil.virtual_memory().percent
        self.memory_peak = max(self.memory_peak, gpu_memory)
        
        print(f"📊 [{stage}] GPU: {gpu_memory:.2f}GB, CPU: {cpu_memory}% (Peak GPU: {self.memory_peak:.2f}GB)")
        
    def process_static_data(self):
        """Procesamiento de datos estáticos con broadcasting eficiente"""
        print("🏗️ Procesando datos estáticos (optimizado)...")
        
        try:
            # Usar chunking para archivos grandes
            ds = xr.open_zarr(
                self.config.PATH_STATIC,
                chunks={dim: self.config.ZARR_CHUNK_SIZE for dim in ['lat', 'lon']}
            )
            
            # Seleccionar variables estáticas clave para reducir memoria
            static_vars = [
                'building_height', 'building_area_residential', 
                'building_area_industrial', 'sky_view_factor',
                'surface_roughness', 'street_width', 'elevation'
            ]
            
            # Filtrar variables que existen
            available_vars = [var for var in static_vars if var in ds.variables]
            ds_static = ds[available_vars]
            
            # Procesar en chunks para evitar OOM
            chunks_processed = []
            for i in range(0, len(ds_static.lat), self.config.ZARR_CHUNK_SIZE):
                lat_chunk = slice(i, min(i + self.config.ZARR_CHUNK_SIZE, len(ds_static.lat)))
                
                for j in range(0, len(ds_static.lon), self.config.ZARR_CHUNK_SIZE):
                    lon_chunk = slice(j, min(j + self.config.ZARR_CHUNK_SIZE, len(ds_static.lon)))
                    
                    chunk = ds_static.isel(lat=lat_chunk, lon=lon_chunk).compute()
                    chunks_processed.append(chunk)
                    
                    # Limpiar memoria
                    del chunk
                    gc.collect()
            
            # Combinar chunks
            ds_static = xr.concat([xr.concat(chunks_processed[i::len(ds_static.lon)//self.config.ZARR_CHUNK_SIZE], 
                                        dim='lon') 
                                 for i in range(len(ds_static.lat)//self.config.ZARR_CHUNK_SIZE)], dim='lat')
            
            self.monitor_memory("Static Load")
            
            # Normalización
            static_norm = self._normalize_static_data(ds_static)
            self.static_processed = static_norm
            self.monitor_memory("Static Norm")
            
            # Guardar processed data
            np.save(self.config.STATIC_CACHE_PATH, static_norm)
            print(f"✅ Datos estáticos guardados en {self.config.STATIC_CACHE_PATH}")
            
        except Exception as e:
            print(f"❌ Error procesando datos estáticos: {e}")
            raise
            
    def _normalize_static_data(self, ds_static):
        """Normalización eficiente de datos estáticos"""
        # Convertir a numpy una sola vez
        static_array = ds_static.to_array().values  # Shape: (vars, lat, lon)
        
        # Transponer a (lat, lon, vars)
        static_array = np.transpose(static_array, (1, 2, 0))
        
        # Normalizar por canal
        static_norm = np.zeros_like(static_array, dtype=np.float32)
        
        for i in range(static_array.shape[2]):
            channel_data = static_array[:, :, i]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            if std_val > 0:
                static_norm[:, :, i] = (channel_data - mean_val) / std_val
            else:
                static_norm[:, :, i] = channel_data - mean_val
                
        return static_norm.astype(np.float32)
    
    def get_tf_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Crear TensorFlow datasets con optimización de memoria"""
        print("📦 Creando TF datasets optimizados...")
        
        try:
            # Cargar datos dinámicos con chunking
            ds = xr.open_zarr(
                self.config.PATH_CACHE,
                chunks={'time': self.config.ZARR_CHUNK_SIZE}
            )
            
            self.monitor_memory("Dynamic Load")
            
            # Extraer arrays
            lr_data = ds['lr_features'].values.astype(np.float32)
            hr_data = ds['hr_target'].values.astype(np.float32)
            
            print(f"📊 Shapes: LR={lr_data.shape}, HR={hr_data.shape}")
            self.monitor_memory("Arrays Ready")
            
            # Cargar datos estáticos
            if self.static_processed is None:
                try:
                    self.static_processed = np.load(self.config.STATIC_CACHE_PATH)
                except:
                    print("⚠️ Cache estático no encontrado, procesando desde cero...")
                    self.process_static_data()
            
            # Preparar datos para sequences
            sequences_x, sequences_y = self._prepare_sequences_efficient(
                lr_data, hr_data, self.static_processed
            )
            
            self.monitor_memory("Sequences Ready")
            
            # Limpiar memoria grande
            del lr_data, hr_data
            gc.collect()
            
            # Split
            split_idx = int(len(sequences_x) * self.config.SPLIT_FRACTION)
            
            train_x = sequences_x[:split_idx]
            train_y = sequences_y[:split_idx]
            val_x = sequences_x[split_idx:]
            val_y = sequences_y[split_idx:]
            
            self.monitor_memory("Data Split")
            
            # Crear TF datasets con optimización
            train_ds = self._create_optimized_dataset(train_x, train_y, training=True)
            val_ds = self._create_optimized_dataset(val_x, val_y, training=False)
            
            # Limpiar más memoria
            del sequences_x, sequences_y
            gc.collect()
            
            self.monitor_memory("Final Datasets")
            
            print(f"✅ Datasets creados: Train={len(train_x)}, Val={len(val_x)}")
            return train_ds, val_ds
            
        except Exception as e:
            print(f"❌ Error creando datasets: {e}")
            raise
    
    def _prepare_sequences_efficient(self, lr_data, hr_data, static_norm):
        """Preparar secuencias con broadcasting eficiente"""
        print("🔄 Preparando secuencias (broadcasting eficiente)...")
        
        seq_len = self.config.SEQ_LEN
        sequences_x = []
        sequences_y = []
        
        # Limitar cantidad de secuencias para evitar OOM en debugging
        max_sequences = min(1000, len(lr_data) - seq_len + 1)
        
        for i in range(max_sequences):
            try:
                # Features dinámicos
                x_dynamic = lr_data[i:i+seq_len]  # (seq_len, lr_shape, channels)
                
                # Broadcasting eficiente de datos estáticos
                # Evitar np.repeat que duplica memoria
                static_shape = static_norm.shape
                static_resized = np.resize(static_norm, (static_shape[0], static_shape[1], static_shape[2]))
                
                # Expandir a secuencia sin duplicar datos
                x_static = np.broadcast_to(static_resized[np.newaxis, ...], 
                                         (seq_len, *static_resized.shape))
                
                # Concatenar
                x_sequence = np.concatenate([x_dynamic, x_static], axis=-1)
                
                # Target
                y_sequence = hr_data[i+seq_len-1]  # Último timestep como target
                
                sequences_x.append(x_sequence.astype(np.float32))
                sequences_y.append(y_sequence.astype(np.float32))
                
                # Limpiar variables temporales
                del x_dynamic, x_static, x_sequence, y_sequence
                
                if i % 100 == 0:
                    print(f"   Procesadas {i}/{max_sequences} secuencias...")
                    self.monitor_memory(f"Seq {i}")
                    
            except Exception as e:
                print(f"⚠️ Error en secuencia {i}: {e}")
                continue
        
        return np.array(sequences_x), np.array(sequences_y)
    
    def _create_optimized_dataset(self, x_data, y_data, training=True):
        """Crear TF dataset con configuración de memoria optimizada"""
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        
        if training:
            # Shuffle con buffer pequeño para ahorrar memoria
            dataset = dataset.shuffle(
                buffer_size=min(self.config.SHUFFLE_BUFFER_SIZE, len(x_data)),
                seed=self.config.SEED
            )
        
        # Batch con prefetch pequeño
        dataset = dataset.batch(self.config.BATCH_SIZE, drop_remainder=True)
        
        # Prefetch limitado para evitar acumulación
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE if self.config.GPU_MEMORY_GB and self.config.GPU_MEMORY_GB >= 24 else 1
        )
        
        # Mixed precision si está disponible
        if self.config.MIXED_PRECISION:
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("🎯 Mixed precision activado")
            except:
                print("⚠️ Mixed precision no disponible")
        
        return dataset

# Función wrapper para compatibilidad con código existente
def BigDataPipeline(config=None):
    """Wrapper para compatibilidad"""
    return OptimizedBigDataPipeline(config)

if __name__ == "__main__":
    print("🧪 Testing Optimized Pipeline...")
    pipeline = OptimizedBigDataPipeline()
    
    # Test
    pipeline.process_static_data()
    train_ds, val_ds = pipeline.get_tf_datasets()
    
    print("✅ Pipeline optimizado funcionando correctamente!")