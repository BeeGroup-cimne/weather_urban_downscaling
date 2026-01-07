import os
import sys
import gc
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session

# --- AJUSTE DE RUTAS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# IMPORTS DEL PROYECTO
from src.models.downsr_unet import DownsrUNet
from src.data.bigdata_loader import get_dataloaders
from src.training.engine import run_experiment, visualize_results, notify_completion, plot_comparative_history
from config.config import Config

# --- CONFIGURACIÓN DE EXPERIMENTOS ---
EXPERIMENTS_TO_RUN = ['unet', 'lstm', 'mamba'] 

# --- 🧠 FÍSICA: FUNCIÓN DE PÉRDIDA HÍBRIDA (TAO LOSS) ---
# Recuperada de tu train.py original.
# Balancea precisión numérica (MSE) con fidelidad estructural (SSIM).
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

def get_optimized_optimizer():
    """Retorna el optimizador adecuado para M4 o Cloud"""
    if Config.IS_MAC_SILICON:
        # print("   ⚡ Usando Optimizer Legacy (Optimizado para Metal/M-Series)")
        return tf.keras.optimizers.legacy.Adam(learning_rate=Config.LEARNING_RATE)
    else:
        return tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)

def main():
    print(f"\n🚀 INICIANDO ESTUDIO DE ABLACIÓN (Modo Físico Híbrido)")
    print(f"   ⚡ Hardware: {Config.DEVICE}")
    
    output_base_dir = os.path.join(PROJECT_ROOT, "experiments")
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_histories = {}
    results_summary = []

    # 1. CARGA DE DATOS
    print("\n📦 Inicializando Big Data Pipeline...")
    try:
        train_ds, val_ds = get_dataloaders()
        print("   ✅ Datos listos.")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        sys.exit(1)

    # 2. BUCLE DE EXPERIMENTOS
    for strategy in EXPERIMENTS_TO_RUN:
        clear_session()
        gc.collect()
        
        experiment_name = f"Ablation_{strategy.upper()}"
        print(f"\n{'='*60}")
        print(f"🏗️  MODELO: {strategy.upper()} + Hybrid Loss")
        print(f"{'='*60}")
        
        try:
            # A. Construir Estructura
            builder = DownsrUNet(bottleneck_type=strategy)
            model = builder.build()
            
            # B. INYECCIÓN DE PÉRDIDA (El Cambio Clave)
            # Re-compilamos el modelo para usar combined_loss en lugar de MSE simple
            model.compile(
                optimizer=get_optimized_optimizer(),
                loss=combined_loss,     # <--- Tu pérdida personalizada
                metrics=['mae', 'mse']  # Métricas legibles para humanos
            )
            
        except Exception as e:
            print(f"⚠️ Error construyendo/compilando {strategy}: {e}")
            continue
        
        # C. Entrenar
        history = run_experiment(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            experiment_name=experiment_name
        )
        
        all_histories[strategy] = history
        
        # Guardamos métricas clave (usamos val_mae que es fácil de interpretar en ºC)
        results_summary.append({
            'Model': strategy,
            'Params': model.count_params(),
            'Best_Val_Loss (Hybrid)': min(history.history['val_loss']),
            'Best_Val_MAE (deg C)': min(history.history['val_mae'])
        })
        
        # D. Visualizar
        visualize_results(model, val_ds, title=experiment_name)

    # 3. REPORTE FINAL
    print("\n📊 RESUMEN FINAL")
    df = pd.DataFrame(results_summary)
    print(df)
    
    df.to_csv(os.path.join(output_base_dir, "ablation_summary.csv"), index=False)
    plot_comparative_history(all_histories, save_dir=os.path.join(output_base_dir, "figures"))
    
    notify_completion("Estudio de ablación con Loss Híbrida completado.")

if __name__ == "__main__":
    main()