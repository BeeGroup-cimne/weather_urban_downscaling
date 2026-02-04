#!/usr/bin/env python3
"""
Entrenamiento optimizado para GPU server con gestión de memoria
Solución a problemas de OOM y entrenamiento estable
"""

import os
import sys
import gc
import time
import numpy as np
import tensorflow as tf
from typing import Dict, Any

# Agregar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, os.path.join(parent_dir, 'src')])

from config.runtime import Config
from src.optimized_data_pipeline import OptimizedBigDataPipeline
from src.models_legacy import ModelZoo
from src.models.transformer_optimized import build_lightweight_transformer_unet
from src.utils_legacy import run_experiment, notify_completion
from src.losses import tf_hybrid_loss

class GPUOptimizedTrainer:
    def __init__(self):
        self.config = Config
        self.pipeline = None
        self.models = {}
        
        if not hasattr(self.config, "GPU_MEMORY_GB"):
            raise RuntimeError("GPU config requerido. Exporta USE_GPU_CONFIG=1 antes de ejecutar.")
        
        print(f"🚀 GPU Optimized Trainer inicializado")
        self.config.print_memory_info()
        
    def setup_environment(self):
        """Configurar entorno de entrenamiento optimizado"""
        print(f"🔧 Configurando entorno de GPU...")
        
        # Configurar TensorFlow para GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth (only if no virtual device config is set)
                for gpu in gpus:
                    if tf.config.experimental.get_virtual_device_configuration(gpu):
                        continue
                    if self.config.GPU_MEMORY_GB:
                        continue
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Limitar memoria si se especifica
                if self.config.GPU_MEMORY_GB:
                    if not tf.config.experimental.get_virtual_device_configuration(gpus[0]):
                        memory_limit = int(self.config.GPU_MEMORY_FRACTION * self.config.GPU_MEMORY_GB * 1024)
                        tf.config.experimental.set_virtual_device_configuration(
                            gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                        )
                        print(f"✅ GPU memory limitada a {memory_limit/1024:.1f}GB")
                
                # Mixed precision
                if self.config.MIXED_PRECISION:
                    try:
                        from tensorflow.keras import mixed_precision
                        policy = mixed_precision.Policy('mixed_float16')
                        mixed_precision.set_global_policy(policy)
                        print(f"✅ Mixed precision activado")
                    except Exception as e:
                        print(f"⚠️ Mixed precision error: {e}")
                        
            except Exception as e:
                print(f"❌ Error configurando GPU: {e}")
                raise
        
        # Configurar seeds
        tf.random.set_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        
    def build_models(self) -> Dict[str, Any]:
        """Construir modelos con dimensiones optimizadas"""
        print(f"🏗️ Construyendo modelos optimizados...")
        
        input_shape = (
            self.config.SEQ_LEN,
            *self.config.HR_SHAPE,
            self.config.CHANNELS + self.config.STATIC_CHANNELS
        )
        
        models = {}
        
        try:
            # UNet Optimizado
            print(f"   📦 Construyendo UNet optimizado...")
            models['UNet'] = ModelZoo.build_unet()
            
            # ConvLSTM Optimizado  
            print(f"   📦 Construyendo ConvLSTM optimizado...")
            models['ConvLSTM'] = ModelZoo.build_hybrid_unet_lstm()
            
            # Transformer Optimizado (desactivado por OOM en HR con atención completa)
            print(f"⚠️ Omitiendo Transformer (OOM en A10 con embedding grande)")
            
            # Mamba Optimizado (si hay suficiente memoria)
            if self.config.GPU_MEMORY_GB and self.config.GPU_MEMORY_GB >= 24:
                print(f"   📦 Construyendo Mamba optimizado...")
                models['Mamba'] = ModelZoo.build_hybrid_unet_mamba()
            else:
                print(f"⚠️ Omitiendo Mamba (memoria GPU insuficiente)")
            
            # Imprimir resumen de modelos
            for name, model in models.items():
                params = model.count_params() / 1e6
                print(f"   ✅ {name}: {params:.2f}M parámetros")
                
        except Exception as e:
            print(f"❌ Error construyendo modelos: {e}")
            raise
        
        return models
    
    def train_model_with_memory_management(self, model, model_name: str, train_ds, val_ds):
        """Entrenar modelo con gestión de memoria activa"""
        print(f"\n🎯 Entrenando {model_name}...")
        
        try:
            # Compilar con optimizador adaptativo
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
            
            # Loss function con SSIM
            loss_fn = tf_hybrid_loss(alpha=0.8, max_val=1.0)
            
            # Compilar
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['mae', 'mse']
            )
            
            # Callbacks con memory management
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"experiments/models/{model_name}_gpu_optimized.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                ),
                MemoryMonitorCallback()  # Callback personalizado
            ]
            
            # Entrenamiento con gradient accumulation
            history = self._train_with_gradient_accumulation(
                model, train_ds, val_ds, callbacks, model_name
            )
            
            print(f"✅ {model_name} entrenado exitosamente")
            return history
            
        except tf.errors.ResourceExhaustedError as e:
            print(f"❌ OOM Error en {model_name}: {e}")
            print(f"🧹 Limpiando memoria y reintentando...")
            
            # Limpiar memoria
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reintentar con batch size más pequeño
            if self.config.BATCH_SIZE > 1:
                original_batch_size = self.config.BATCH_SIZE
                self.config.BATCH_SIZE = 1
                self.config.GRADIENT_ACCUMULATION_STEPS *= original_batch_size
                
                print(f"🔄 Reintentando {model_name} con batch_size={self.config.BATCH_SIZE}")
                return self.train_model_with_memory_management(model, model_name, train_ds, val_ds)
            else:
                raise
                
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {e}")
            raise
    
    def _train_with_gradient_accumulation(self, model, train_ds, val_ds, callbacks, model_name):
        """Implementar gradient accumulation para batches efectivos más grandes"""
        
        # Crear datasets más pequeños para accumulation
        train_ds_small = train_ds.unbatch().batch(self.config.BATCH_SIZE)
        
        # Custom training loop con gradient accumulation
        epochs = self.config.EPOCHS
        accumulation_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # Training con accumulation
            train_losses = []
            train_maes = []
            accumulated_gradients = None
            step_count = 0
            
            for step, (x_batch, y_batch) in enumerate(train_ds_small):
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    
                    # Loss calculation
                    total_loss = loss_fn(y_batch, y_pred)
                    
                    # Normalize by accumulation steps
                    scaled_loss = total_loss / accumulation_steps
                
                # Calculate gradients
                gradients = tape.gradient(scaled_loss, model.trainable_variables)
                
                # Accumulate gradients
                if accumulated_gradients is None:
                    accumulated_gradients = gradients
                else:
                    accumulated_gradients = [
                        acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
                    ]
                
                train_losses.append(total_loss.numpy())
                train_maes.append(tf.reduce_mean(tf.keras.metrics.mae(y_batch, y_pred)).numpy())
                
                step_count += 1
                
                # Apply accumulated gradients
                if step_count % accumulation_steps == 0:
                    model.optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                    accumulated_gradients = None
                
                # Memory cleanup
                if step_count % 10 == 0:
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                # Limitar steps si se configuró un máximo
                if self.config.MAX_STEPS_PER_EPOCH is not None and step_count >= self.config.MAX_STEPS_PER_EPOCH:
                    break
            
            # Validation
            val_loss, val_mae = self._validate_model(model, val_ds)
            
            # Store history
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)
            
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['mae'].append(avg_train_mae)
            history['val_mae'].append(val_mae)
            
            print(f"   📊 Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}")
            print(f"   📊 Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
            
            # Early stopping manual
            if len(history['val_loss']) >= self.config.EARLY_STOPPING_PATIENCE:
                recent_losses = history['val_loss'][-self.config.EARLY_STOPPING_PATIENCE:]
                if all(loss >= min(history['val_loss'][:-self.config.EARLY_STOPPING_PATIENCE]) for loss in recent_losses):
                    print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                    break
        
        return history
    
    def _validate_model(self, model, val_ds):
        """Validación eficiente de modelo"""
        val_losses = []
        val_maes = []
        
        for x_batch, y_batch in val_ds.take(10):  # Limitar para velocidad
            y_pred = model(x_batch, training=False)
            
            loss = tf.reduce_mean(tf.keras.losses.mse(y_batch, y_pred))
            mae = tf.reduce_mean(tf.keras.metrics.mae(y_batch, y_pred))
            
            val_losses.append(loss.numpy())
            val_maes.append(mae.numpy())
        
        return np.mean(val_losses), np.mean(val_maes)
    
    def run_training_pipeline(self):
        """Pipeline completo de entrenamiento optimizado"""
        try:
            print(f"\n🚀 Iniciando pipeline de entrenamiento optimizado")
            
            # Setup
            self.setup_environment()
            
            # Data pipeline
            print(f"\n📊 Preparando datos...")
            self.pipeline = OptimizedBigDataPipeline(self.config)
            train_ds, val_ds = self.pipeline.get_tf_datasets()
            
            # Models
            models = self.build_models()
            
            # Training loop
            histories = {}
            for model_name, model in models.items():
                print(f"\n{'='*50}")
                
                # Ajustar targets si el modelo solo predice el último timestep
                model_train_ds = train_ds
                model_val_ds = val_ds
                
                try:
                    if isinstance(model.output_shape, tuple) and len(model.output_shape) == 4:
                        def select_last_timestep(x, y):
                            return x, y[:, -1, ...]
                        model_train_ds = train_ds.map(select_last_timestep)
                        model_val_ds = val_ds.map(select_last_timestep)
                except Exception:
                    pass
                
                history = self.train_model_with_memory_management(
                    model, model_name, model_train_ds, model_val_ds
                )
                histories[model_name] = history
                
                # Guardar historia
                import pandas as pd
                df = pd.DataFrame(history)
                df.to_csv(f"experiments/logs/{model_name}_gpu_optimized_log.csv", index=False)
                
                # Memory cleanup entre modelos
                tf.keras.backend.clear_session()
                gc.collect()
            
            print(f"\n🎉 Entrenamiento completado!")
            notify_completion("Entrenamiento GPU optimizado finalizado")
            
            return histories
            
        except Exception as e:
            print(f"❌ Error en pipeline: {e}")
            raise

class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """Callback para monitorear memoria durante entrenamiento"""
    
    def on_epoch_end(self, epoch, logs=None):
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                memory_mb = int(result.stdout.strip())
                print(f"   🧠 GPU Memory: {memory_mb}MB")
        except:
            pass

def train_gpu_optimized():
    """Función principal para Docker"""
    trainer = GPUOptimizedTrainer()
    return trainer.run_training_pipeline()

if __name__ == "__main__":
    train_gpu_optimized()
