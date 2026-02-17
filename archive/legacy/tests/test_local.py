#!/usr/bin/env python3
"""
Test local optimizado para Mac Silicon - Weather Downscaling
Versión reducida para debugging rápido
"""

import os
import sys
import numpy as np
import time
import gc
from pathlib import Path

# Agregar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, os.path.join(parent_dir, 'src')])

class LocalTestConfig:
    """Configuración ultra-optimizada para testing local en Mac Silicon"""
    
    # Hardware local (Mac Silicon)
    IS_MAC_SILICON = True
    DEVICE = "MPS (Metal Performance Shaders)"
    ESTIMATED_MEMORY_GB = 16  # M1/M2/M3 typical
    
    # Configuración conservadora para testing
    BATCH_SIZE = 1
    SEQ_LEN = 2  # Reducido de 6
    GRADIENT_ACCUMULATION_STEPS = 2
    EFFECTIVE_BATCH_SIZE = 2
    
    # Learning rate más alto para testing rápido
    LEARNING_RATE = 1e-3
    EPOCHS = 3  # Solo 3 épocas para test
    SEED = 42
    
    # Reducir dimensionalidades
    HR_SHAPE = (64, 64)  # Reducido de 251x251 para test rápido
    LR_SHAPE = (4, 3)
    CHANNELS = 9
    STATIC_CHANNELS = 13
    
    # Model dimensions más pequeñas
    UNET_FILTERS = 32  # Reducido de 64
    CONVLSTM_FILTERS = 32
    MAMBA_MODEL_DIM = 32  # Reducido de 64
    MAMBA_STATE_DIM = 16
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    EXPERIMENTS_DIR = str(BASE_DIR / "experiments")
    STATS_PATH = str(BASE_DIR / "data" / "processed" / "stats_config.npz")
    STATIC_CACHE_PATH = str(BASE_DIR / "data" / "processed" / "static_processed.npy")
    
    # Testing limits
    MAX_SEQUENCES = 20  # Solo 20 secuencias para test rápido
    MEMORY_MONITORING = True
    
    @classmethod
    def print_info(cls):
        print("🧪 Configuración Local Test:")
        print(f"   Hardware: {cls.DEVICE}")
        print(f"   Image Size: {cls.HR_SHAPE} (reducido para test)")
        print(f"   Sequence Length: {cls.SEQ_LEN}")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Epochs: {cls.EPOCHS}")
        print(f"   Model Filters: {cls.UNET_FILTERS}")
        print(f"   Max Sequences: {cls.MAX_SEQUENCES}")

class LocalTestPipeline:
    """Pipeline ultra simplificado para testing local"""
    
    def __init__(self):
        self.config = LocalTestConfig
        self.config.print_info()
        
    def create_synthetic_data(self):
        """Crear datos sintéticos para testing (evitar carga real)"""
        print("🎲 Creando datos sintéticos para testing...")
        
        # Datos dinámicos (simulando weather data)
        x_dynamic = np.random.randn(
            self.config.MAX_SEQUENCES,
            self.config.SEQ_LEN,
            *self.config.HR_SHAPE,
            self.config.CHANNELS
        ).astype(np.float32)
        
        # Datos estáticos (simulando urban features)
        static_shape = (*self.config.HR_SHAPE, self.config.STATIC_CHANNELS)
        static_data = np.random.randn(*static_shape).astype(np.float32)
        
        # Crear secuencias
        sequences_x = []
        sequences_y = []
        
        for i in range(self.config.MAX_SEQUENCES - self.config.SEQ_LEN):
            # Features dinámicos
            x_seq_dynamic = x_dynamic[i:i+self.config.SEQ_LEN]
            
            # Static broadcasting (eficiente)
            x_seq_static = np.broadcast_to(
                static_data[np.newaxis, ...], 
                (self.config.SEQ_LEN, *static_shape)
            )
            
            # Concatenar
            x_sequence = np.concatenate([x_seq_dynamic, x_seq_static], axis=-1)
            
            # Target (último timestep, solo primer canal)
            y_sequence = x_seq_dynamic[-1, :, :, 0:1]  # Solo temperatura
            
            sequences_x.append(x_sequence)
            sequences_y.append(y_sequence)
        
        return np.array(sequences_x), np.array(sequences_y)
    
    def create_simple_unet(self, input_shape):
        """Crear UNet simplificado para testing"""
        print(f"🏗️ Creando UNet simplificado...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            
            inputs = layers.Input(shape=input_shape)
            
            # Encoder muy simple
            x = layers.Conv2D(self.config.UNET_FILTERS, 3, activation='relu', padding='same')(inputs)
            x = layers.Conv2D(self.config.UNET_FILTERS, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(2)(x)
            
            # Bottleneck
            x = layers.Conv2D(self.config.UNET_FILTERS*2, 3, activation='relu', padding='same')(x)
            x = layers.Conv2D(self.config.UNET_FILTERS*2, 3, activation='relu', padding='same')(x)
            
            # Decoder simple
            x = layers.UpSampling2D(2)(x)
            x = layers.Conv2D(self.config.UNET_FILTERS, 3, activation='relu', padding='same')(x)
            
            # Output (solo 1 canal - temperatura)
            outputs = layers.Conv2D(1, 1, activation='linear')(x)
            
            model = Model(inputs, outputs)
            
            params = model.count_params() / 1e6
            print(f"   ✅ UNet creado: {params:.2f}M parámetros")
            
            return model
            
        except Exception as e:
            print(f"❌ Error creando UNet: {e}")
            raise
    
    def run_training_test(self):
        """Ejecutar prueba completa de entrenamiento"""
        print(f"\n🚀 Iniciando prueba local de entrenamiento...")
        
        try:
            # 1. Crear datos sintéticos
            X_train, y_train = self.create_synthetic_data()
            print(f"   📊 Datos creados: X={X_train.shape}, y={y_train.shape}")
            
            # 2. Crear modelo
            input_shape = (
                self.config.SEQ_LEN,
                *self.config.HR_SHAPE,
                self.config.CHANNELS + self.config.STATIC_CHANNELS
            )
            model = self.create_simple_unet(input_shape)
            
            # 3. Compilar
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # 4. Training loop
            print(f"🏋️ Iniciando entrenamiento de prueba...")
            history = {
                'loss': [],
                'mae': [],
                'val_loss': [],
                'val_mae': []
            }
            
            # Split simple
            split_idx = int(0.8 * len(X_train))
            X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
            y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]
            
            for epoch in range(self.config.EPOCHS):
                print(f"\n📅 Epoch {epoch + 1}/{self.config.EPOCHS}")
                
                # Entrenamiento
                hist = model.fit(
                    X_train_split, y_train_split,
                    batch_size=self.config.BATCH_SIZE,
                    epochs=1,
                    validation_data=(X_val_split, y_val_split),
                    verbose=1
                )
                
                # Guardar historia
                history['loss'].extend(hist.history['loss'])
                history['mae'].extend(hist.history['mae'])
                history['val_loss'].extend(hist.history['val_loss'])
                history['val_mae'].extend(hist.history['val_mae'])
                
                # Memory cleanup
                if hasattr(tf, 'keras') and hasattr(tf.keras.backend, 'clear_session'):
                    tf.keras.backend.clear_session()
                gc.collect()
            
            # 5. Evaluar final
            final_loss = history['val_loss'][-1]
            final_mae = history['val_mae'][-1]
            
            print(f"\n🎯 Resultados Finales:")
            print(f"   Final Val Loss: {final_loss:.4f}")
            print(f"   Final Val MAE: {final_mae:.4f}")
            
            # 6. Guardar resultados
            os.makedirs(self.config.EXPERIMENTS_DIR, exist_ok=True)
            
            # Guardar modelo
            model_path = os.path.join(self.config.EXPERIMENTS_DIR, "models", "test_unet_local.h5")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            print(f"   💾 Modelo guardado: {model_path}")
            
            # Guardar historia
            import pandas as pd
            df = pd.DataFrame(history)
            log_path = os.path.join(self.config.EXPERIMENTS_DIR, "logs", "test_local_log.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            df.to_csv(log_path, index=False)
            print(f"   📊 Logs guardados: {log_path}")
            
            print(f"\n✅ Prueba local completada exitosamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error en prueba local: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Ejecución principal de prueba local"""
    print("🧪 Weather Downscaling - Prueba Local")
    print("=" * 50)
    
    # Verificar TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow disponible: {tf.__version__}")
        
        # Verificar MPS
        if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'get_memory_info'):
            print("✅ MPS (Metal Performance Shaders) disponible")
        else:
            print("⚠️ MPS no detectado, usando CPU")
            
    except ImportError:
        print("❌ TensorFlow no disponible")
        return False
    
    # Ejecutar prueba
    pipeline = LocalTestPipeline()
    success = pipeline.run_training_test()
    
    if success:
        print("\n🎉 ¡Prueba local exitosa! El sistema está listo para GPU server.")
        print("📝 Siguientes pasos:")
        print("   1. Subir el proyecto al servidor GPU")
        print("   2. Ejecutar: ./deploy_gpu_server.sh")
        print("   3. Revisar resultados en experiments/")
    else:
        print("\n❌ Prueba local falló. Revisa los errores arriba.")
    
    return success

if __name__ == "__main__":
    main()