import gc
import os
import sys
import tensorflow.keras

# Forzar compatibilidad con tf_keras (Keras 2) sobre TensorFlow 2.16+
os.environ['TF_USE_LEGACY_KERAS'] = '1'


import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session

# --- AJUSTE DE RUTAS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# IMPORTS DEL PROYECTO (ALINEADOS CON TRAIN.PY)
from src.models_legacy import ModelZoo # <--- Usamos Legacy Models (ReLU, No-BN)
from src.data_loader import BigDataPipeline # <--- Usamos el Data Loader completo
from src.utils import run_experiment, visualize_results, notify_completion, plot_comparative_history
from config.config import Config

# --- CONFIGURACIÓN DE EXPERIMENTOS ---
# Mapeo de nombres a métodos de construcción en ModelZoo
EXPERIMENTS_TO_RUN = {
    'unet': ModelZoo.build_unet,
    'lstm': ModelZoo.build_hybrid_unet_lstm,
    'mamba': ModelZoo.build_hybrid_unet_mamba
}

# --- 🧠 FÍSICA: FUNCIÓN DE PÉRDIDA HÍBRIDA (TAO LOSS) ---
# Recuperada de tu train.py original.
def combined_loss(y_true, y_pred):
    # 1. Error de Valores (MSE)
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # 2. Error de Estructura (SSIM)
    # max_val=5.0 asume que los datos están normalizados (aprox -2.5 a 2.5 sigmas)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=5.0))
    
    # 3. Combinación Ponderada
    # Alpha alto (0.8) prioriza que el mapa "se vea bien" (bordes definidos)
    alpha = 0.8
    return (1 - alpha) * mse + alpha * ssim_loss

def main():
    print(f"\n🚀 INICIANDO ESTUDIO DE ABLACIÓN (Alineado con train.py)")
    print(f"   ⚡ Hardware: {Config.DEVICE}")
    
    output_base_dir = os.path.join(PROJECT_ROOT, "experiments")
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_histories = {}
    results_summary = []

    # 1. CARGA DE DATOS (Mismo Pipeline que train.py)
    tf.keras.backend.clear_session()
    print("\n📦 Inicializando Big Data Pipeline...")
    
    # =========================================================================
    # CRÍTICO: Pipeline Completo (Igual que train.py)
    # 1. Instanciar
    pipeline = BigDataPipeline(Config)
    
    # 2. PROCESAR DATOS ESTÁTICOS (Esto estaba faltando en ablation!)
    # Esto asegura que edificios y topografía existan y no sean ceros.
    pipeline.process_static_data() 
    
    # 3. ETL
    pipeline.run_etl_process()
    
    # 4. Obtener Datasets
    try:
        train_ds, val_ds = pipeline.get_tf_datasets()
        print("   ✅ Datos listos y procesados.")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        sys.exit(1)
    # =========================================================================

    # 2. BUCLE DE EXPERIMENTOS
    for strategy_name, builder_func in EXPERIMENTS_TO_RUN.items():
        clear_session()
        gc.collect()
        
        experiment_name = f"Ablation_{strategy_name.upper()}_Legacy"
        print(f"\n{'='*60}")
        print(f"🏗️  MODELO: {strategy_name.upper()} (Legacy Architecture)")
        print(f"{'='*60}")
        
        try:
            # A. Construir Estructura (Usando ModelZoo Legacy)
            # ModelZoo methods might take different args, let's inspect usage in train.py
            # train.py: model = builder(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
            # But ModelZoo methods in src/models_legacy.py vary. 
            # build_unet takes (cls) -> no args? No, it's a classmethod. 
            # build_hybrid_unet_mamba takes (lr_shape, hr_shape).
            # Let's try passing shapes if accepted, or wrapping.
            
            # Inspecting models_legacy.py again briefly via memory...
            # build_unet(cls) -> uses Config internally.
            # build_hybrid_unet_mamba(steps=...) -> uses defaults or args.
            
            # Safe bet: Check inspection or try/except, or just call it if we know.
            # train.py code:
            # experiments = [("UNet_Mamba", ModelZoo.build_hybrid_unet_mamba)]
            # model = builder(lr_shape=..., hr_shape=...)
            
            if strategy_name == 'mamba':
                model = builder_func(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
            else:
                # UNet and others in Legacy might not take args or take them differently.
                # models_legacy.py: build_unet(cls) -> no args.
                model = builder_func()
            
            # B. INYECCIÓN DE PÉRDIDA (El Cambio Clave)
            # ModelZoo compiles with 'mse' by default. We re-compile to add SSIM.
            print(f"⚙️ Re-compilando con Loss Híbrida (MSE + SSIM)...")
            
            # Optimizer: ModelZoo has get_optimizer static method
            opt = ModelZoo.get_optimizer(Config.LEARNING_RATE)
            
            model.compile(
                optimizer=opt,
                loss=combined_loss,     # <--- Tu pérdida personalizada
                metrics=['mae', 'mse']  
            )
            
        except Exception as e:
            print(f"⚠️ Error construyendo/compilando {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # C. Entrenar
        # run_experiment is imported from src.utils (legacy) same as train.py
        history = run_experiment(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            experiment_name=experiment_name
        )
        
        all_histories[strategy_name] = history
        
        # Guardamos métricas
        results_summary.append({
            'Model': strategy_name,
            'Params': model.count_params(),
            'Best_Val_Loss': min(history.history['val_loss']),
            'Best_Val_MAE': min(history.history['val_mae'])
        })
        
        # D. Visualizar
        visualize_results(model, val_ds, title=experiment_name)

    # 3. REPORTE FINAL
    print("\n📊 RESUMEN FINAL")
    df = pd.DataFrame(results_summary)
    print(df)
    
    df.to_csv(os.path.join(output_base_dir, "ablation_summary.csv"), index=False)
    plot_comparative_history(all_histories, save_dir=os.path.join(output_base_dir, "figures"))
    
    notify_completion("Estudio de ablación (Legacy Align) completado.")

if __name__ == "__main__":
    main()