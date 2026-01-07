import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import xarray as xr
from config.config import Config
from src.data_loader import BigDataPipeline
from src.models import ModelZoo
from src.utils import notify_completion, plot_comparative_history , run_experiment , visualize_results

tf.random.set_seed(Config.SEED)

def combined_loss(y_true, y_pred):
    # 1. Error de Valores (MSE) - Para precisión numérica
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # 2. Error de Estructura (SSIM) - Para nitidez visual
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=5.0))
    
    # 3. Combinación (Ajusta el peso alfa)
    alpha = 0.8
    return (1 - alpha) * mse + alpha * ssim_loss


# MAIN EXECUTION

if __name__ == "__main__":
    
    # Limpiar sesiones anteriores para liberar memoria
    tf.keras.backend.clear_session()
    
    print("🚀 Iniciando Pipeline de Big Data...")

    # 1. Inicializar Pipeline
    pipeline = BigDataPipeline(Config)
    
    # 2. Procesar Datos Estáticos (En memoria, son ligeros)
    pipeline.process_static_data()
    
    # 3. Ejecutar ETL (Crea carpeta ./processed_cache_zarr si no existe)
    pipeline.run_etl_process()
    
    # 4. OBTENER DATASETS (¡Paso crítico faltante!)
    # Esto conecta los generadores a los archivos Zarr
    train_ds, val_ds = pipeline.get_tf_datasets()
    
    # Verificación rápida (opcional)
    try:
        x_samp, y_samp = next(iter(train_ds))
        print(f"✅ Datos cargados correctamente. Input Shape: {x_samp[0].shape}")
    except Exception as e:
        print(f"⚠️ Error verificando datasets: {e}")

    # 5. Definir Modelos a Probar
    experiments = [
        #("UNet", ModelZoo.build_unet),
        ("ConvLSTM", ModelZoo.build_convlstm), # Descomentar para probar otros
        #("Transformer", ModelZoo.build_transformer),
        #("Hybrid_UNet_LSTM", ModelZoo.build_hybrid_unet_lstm),
        #("UNet_Mamba", ModelZoo.build_hybrid_unet_mamba)
    ]

    histories = {}

    # 6. Bucle de Experimentos
    for name, builder in experiments:
        print(f"\n🏗️ Construyendo modelo: {name}...")
        
        # Limpiar sesión entre modelos para evitar fugas de memoria
        if name != experiments[0][0]:
            tf.keras.backend.clear_session()
            
        model = builder(
            lr_shape=Config.LR_SHAPE, 
            hr_shape=Config.HR_SHAPE
        )

        # ---------------------------------------------------------
        # AQUÍ es donde "inyectamos" la nueva Loss Function al modelo
        print(f"⚙️ Compilando {name} con Loss Híbrida (MSE + SSIM)...")
        
        model.compile(
            # Usamos la versión Legacy que está optimizada para Mac Silicon (M1/M2/M3/M4)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=Config.LEARNING_RATE),
            loss=combined_loss,      # <--- 
            metrics=['mae', 'mse']   # Mantenemos MAE/MSE solo para monitorizar en logs
        )
        # ---------------------------------------------------------
        
        # Opcional: Imprimir resumen
        # model.summary() 

        # Entrenar
        hist = run_experiment(model, train_ds, val_ds, name)
        histories[name] = hist

        # Visualizar resultados
        visualize_results(model, val_ds, f"Resultados: {name}")

    print("\n✅ Todos los experimentos finalizados correctamente.")
