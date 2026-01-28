#!/usr/bin/env python3
"""
Entrenamiento de Transformer optimizado para Weather Downscaling
Script específico para comparar con otros modelos
"""

import os
import sys
import time
import gc
import numpy as np
import tensorflow as tf
from typing import Dict, Any

# Agregar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, os.path.join(parent_dir, 'src')])

from config.gpu_server_config import GPUServerConfig as Config
from src.optimized_data_pipeline import OptimizedBigDataPipeline
from src.models.transformer_optimized import build_lightweight_transformer_unet

class TransformerTrainer:
    """Trainer especializado para Transformer models"""
    
    def __init__(self):
        self.config = Config
        self.pipeline = None
        
        print(f"🤖 Transformer Trainer inicializado")
        print(f"   GPU Memory: {self.config.GPU_MEMORY_GB}GB")
        print(f"   Batch Size: {self.config.BATCH_SIZE}")
        print(f"   Seq Length: {self.config.SEQ_LEN}")
        
    def setup_environment(self):
        """Configurar TensorFlow para Transformer"""
        print(f"🔧 Configurando entorno Transformer...")
        
        # Configurar GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                if self.config.GPU_MEMORY_GB:
                    memory_limit = int(self.config.GPU_MEMORY_FRACTION * self.config.GPU_MEMORY_GB * 1024)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
                
                # Mixed precision para Transformers (muy efectivo)
                if self.config.MIXED_PRECISION:
                    from tensorflow.keras import mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    print(f"✅ Mixed precision activado para Transformer")
                
            except Exception as e:
                print(f"⚠️ Error configurando GPU: {e}")
    
    def build_transformer_model(self):
        """Construir Transformer optimizado"""
        print(f"🏗️ Construyendo Transformer optimizado...")
        
        input_shape = (
            self.config.SEQ_LEN,
            *self.config.HR_SHAPE,
            self.config.CHANNELS + self.config.STATIC_CHANNELS
        )
        
        try:
            model = build_lightweight_transformer_unet(
                input_shape=input_shape,
                max_memory_gb=self.config.GPU_MEMORY_GB or 8
            )
            
            params = model.count_params() / 1e6
            print(f"   ✅ Transformer: {params:.2f}M parámetros")
            
            # Calcular estimación de memoria
            batch_memory = self._estimate_memory_usage(model, input_shape)
            print(f"   🧠 Memory estimada por batch: {batch_memory:.2f}GB")
            
            return model
            
        except Exception as e:
            print(f"❌ Error construyendo Transformer: {e}")
            raise
    
    def _estimate_memory_usage(self, model, input_shape):
        """Estimar uso de memoria del Transformer"""
        batch_size = self.config.BATCH_SIZE
        seq_len, H, W, C = input_shape
        
        # Input memory
        input_memory = batch_size * seq_len * H * W * C * 4 / 1024**3  # float32
        
        # Attention memory (quadratic complexity)
        embed_dim = H * W * C // 8  # Aproximación
        attention_memory = batch_size * seq_len**2 * embed_dim * 4 / 1024**3
        
        # Model parameters
        param_memory = model.count_params() * 4 / 1024**3  # float32
        
        # Activationes (aproximación)
        activation_memory = input_memory * 3  # Hidden states, intermediates, outputs
        
        total_memory = input_memory + attention_memory + param_memory + activation_memory
        
        return min(total_memory, self.config.GPU_MEMORY_GB or 8) * 0.8  # 80% de disponibilidad
    
    def train_transformer(self, train_ds, val_ds):
        """Entrenar Transformer con optimizaciones"""
        print(f"\n🎯 Entrenando Transformer...")
        
        # Construir modelo
        model = self.build_transformer_model()
        
        # Compilar con loss híbrido
        def hybrid_loss(y_true, y_pred):
            mse_loss = tf.keras.losses.mse(y_true, y_pred)
            ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
            return (1 - 0.8) * mse_loss + 0.8 * ssim_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE,
                clipnorm=1.0  # Gradient clipping para estabilidad
            ),
            loss=hybrid_loss,
            metrics=['mae', 'mse']
        )
        
        # Callbacks optimizados para Transformers
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Más paciencia para Transformers
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"experiments/models/Transformer_gpu_optimized.h5",
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            TransformerMemoryCallback(self.config)
        ]
        
        # Entrenamiento con gradient accumulation
        history = self._train_with_gradient_accumulation(
            model, train_ds, val_ds, callbacks
        )
        
        print(f"✅ Transformer entrenado exitosamente")
        return history
    
    def _train_with_gradient_accumulation(self, model, train_ds, val_ds, callbacks):
        """Training loop con gradient accumulation"""
        
        # Preparar datasets
        train_ds_small = train_ds.unbatch().batch(self.config.BATCH_SIZE)
        epochs = self.config.EPOCHS
        accumulation_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
        print(f"📈 Usando gradient accumulation:")
        print(f"   Batch size: {self.config.BATCH_SIZE}")
        print(f"   Accumulation steps: {accumulation_steps}")
        print(f"   Effective batch size: {self.config.BATCH_SIZE * accumulation_steps}")
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_losses = []
            train_maes = []
            accumulated_gradients = None
            step_count = 0
            
            for step, (x_batch, y_batch) in enumerate(train_ds_small):
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    
                    # Loss calculation
                    mse_loss = tf.keras.losses.mse(y_batch, y_pred)
                    ssim_loss = 1 - tf.image.ssim(y_batch, y_pred, max_val=1.0)
                    total_loss = (1 - 0.8) * mse_loss + 0.8 * ssim_loss
                    
                    # Normalize por accumulation steps
                    scaled_loss = total_loss / accumulation_steps
                
                # Calcular gradientes
                gradients = tape.gradient(scaled_loss, model.trainable_variables)
                
                # Acumular gradientes
                if accumulated_gradients is None:
                    accumulated_gradients = gradients
                else:
                    accumulated_gradients = [
                        acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
                    ]
                
                train_losses.append(total_loss.numpy())
                train_maes.append(tf.keras.metrics.mae(y_batch, y_pred).numpy())
                
                step_count += 1
                
                # Aplicar gradientes acumulados
                if step_count % accumulation_steps == 0:
                    # Gradient clipping
                    gradients, _ = tf.clip_by_global_norm(accumulated_gradients, 1.0)
                    
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    accumulated_gradients = None
                
                # Memory cleanup
                if step_count % 5 == 0:
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                # Limitar pasos para debugging
                if step_count >= 30:  # Limitar para pruebas rápidas
                    break
            
            # Validation
            val_loss, val_mae = self._validate_model(model, val_ds)
            
            # Guardar historia
            avg_train_loss = np.mean(train_losses) if train_losses else 0
            avg_train_mae = np.mean(train_maes) if train_maes else 0
            
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['mae'].append(avg_train_mae)
            history['val_mae'].append(val_mae)
            
            print(f"   📊 Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}")
            print(f"   📊 Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
            
            # Early stopping
            if len(history['val_loss']) >= 15:
                recent_losses = history['val_loss'][-15:]
                if all(loss >= min(history['val_loss'][:-15]) for loss in recent_losses):
                    print(f"🛑 Early stopping en epoch {epoch + 1}")
                    break
        
        return history
    
    def _validate_model(self, model, val_ds):
        """Validación eficiente"""
        val_losses = []
        val_maes = []
        
        for x_batch, y_batch in val_ds.take(10):  # Limitar para velocidad
            y_pred = model(x_batch, training=False)
            
            loss = tf.keras.losses.mse(y_batch, y_pred)
            mae = tf.keras.metrics.mae(y_batch, y_pred)
            
            val_losses.append(loss.numpy())
            val_maes.append(mae.numpy())
        
        return np.mean(val_losses), np.mean(val_maes)
    
    def run_training_pipeline(self):
        """Pipeline completo de entrenamiento Transformer"""
        try:
            print(f"\n🚀 Iniciando pipeline Transformer...")
            
            # Setup
            self.setup_environment()
            
            # Data pipeline
            print(f"\n📊 Preparando datos...")
            self.pipeline = OptimizedBigDataPipeline(self.config)
            train_ds, val_ds = self.pipeline.get_tf_datasets()
            
            # Training
            history = self.train_transformer(train_ds, val_ds)
            
            # Guardar resultados
            self._save_results(history)
            
            print(f"\n🎉 Transformer training completado!")
            return True
            
        except Exception as e:
            print(f"❌ Error en pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_results(self, history):
        """Guardar resultados del Transformer"""
        import pandas as pd
        
        # Guardar historia
        df = pd.DataFrame(history)
        log_path = f"experiments/logs/Transformer_gpu_optimized_log.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        df.to_csv(log_path, index=False)
        print(f"   💾 Logs guardados: {log_path}")

class TransformerMemoryCallback(tf.keras.callbacks.Callback):
    """Callback especializado para Transformers"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def on_epoch_end(self, epoch, logs=None):
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                mem_used, mem_total, temp = result.stdout.strip().split(', ')
                print(f"   🧠 GPU: {mem_used}MB/{mem_total}MB ({int(mem_used)/int(mem_total)*100:.1f}%), Temp: {temp}°C")
        except:
            pass

def main():
    """Ejecución principal"""
    trainer = TransformerTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 ¡Transformer entrenado exitosamente!")
        print("📝 Resultados guardados en experiments/")
    else:
        print("\n❌ Falló entrenamiento del Transformer")
    
    return success

if __name__ == "__main__":
    main()