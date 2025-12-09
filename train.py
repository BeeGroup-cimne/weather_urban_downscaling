import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import xarray as xr
from config.config import Config
from src.data_loader import BigDataPipeline
from src.models import ModelZoo
from src.utils import notify_completion, plot_comparative_history , run_experiment , visualize_results

tf.random.set_seed(Config.SEED)

# ==========================================
# MAIN EXECUTION
# ==========================================
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
         #s("ConvLSTM", ModelZoo.build_convlstm), # Descomentar para probar otros
        ("Transformer", ModelZoo.build_transformer)
        #("Hybrid_UNet_LSTM", ModelZoo.build_hybrid_unet_lstm),
    ]

    histories = {}

    # 6. Bucle de Experimentos
    for name, builder in experiments:
        print(f"\n🏗️ Construyendo modelo: {name}...")
        
        # Limpiar sesión entre modelos para evitar fugas de memoria
        if name != experiments[0][0]:
            tf.keras.backend.clear_session()
            
        model = builder()
        
        # Opcional: Imprimir resumen
        # model.summary() 

        # Entrenar
        hist = run_experiment(model, train_ds, val_ds, name)
        histories[name] = hist

        # Visualizar resultados
        visualize_results(model, val_ds, f"Resultados: {name}")

    print("\n✅ Todos los experimentos finalizados correctamente.")
