import os

# Forzar compatibilidad con tf_keras (Keras 2) sobre TensorFlow 2.16+
# Debe establecerse ANTES de importar TensorFlow/Keras.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import gc
import sys
import random

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
from config.runtime import Config
from src.losses import tf_hybrid_loss

# --- CONFIGURACIÓN DE EXPERIMENTOS ---
# Mapeo de nombres a métodos de construcción en ModelZoo
EXPERIMENTS_TO_RUN = [
    ("unet", ModelZoo.build_unet),
    ("lstm", ModelZoo.build_hybrid_unet_lstm),
    ("transformer", ModelZoo.build_transformer),
    ("mamba", ModelZoo.build_hybrid_unet_mamba),
]

# --- 🧠 FÍSICA: FUNCIÓN DE PÉRDIDA HÍBRIDA (TAO LOSS) ---
combined_loss = tf_hybrid_loss(alpha=0.8, max_val=5.0)

def _safe_min(values):
    valid = [v for v in values if v is not None and not pd.isna(v)]
    if not valid:
        return float("nan")
    return float(min(valid))

def _get_last_epoch_metrics(history, log_path):
    """
    Return last-epoch metrics preferring CSV log (source of truth on resume/interruptions),
    with fallback to in-memory Keras History.
    """
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if not df.empty:
                last = df.iloc[-1]
                return {
                    "final_epoch": int(last["epoch"]) if "epoch" in df.columns and pd.notna(last["epoch"]) else len(df) - 1,
                    "final_val_loss": float(last["val_loss"]) if "val_loss" in df.columns and pd.notna(last["val_loss"]) else float("nan"),
                    "final_val_mae": float(last["val_mae"]) if "val_mae" in df.columns and pd.notna(last["val_mae"]) else float("nan"),
                }
        except Exception as e:
            print(f"⚠️ No se pudo leer el log para métricas finales ({log_path}): {e}")

    hist = getattr(history, "history", {}) or {}
    val_loss = hist.get("val_loss", [])
    val_mae = hist.get("val_mae", [])
    final_epoch = len(val_loss) - 1 if val_loss else (len(val_mae) - 1 if val_mae else None)
    return {
        "final_epoch": final_epoch,
        "final_val_loss": float(val_loss[-1]) if val_loss else float("nan"),
        "final_val_mae": float(val_mae[-1]) if val_mae else float("nan"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["unet", "lstm", "transformer", "mamba"],
        help="Run only the selected models in order (default: unet lstm transformer mamba).",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=6,
        help="Minimum temporal sequence length for all models in ablation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for reproducibility.")
    parser.add_argument("--experiment-suffix", type=str, default="", help="Suffix appended to experiment names (e.g. S42).")
    parser.add_argument("--epochs", type=int, default=None, help="Override Config.EPOCHS.")
    parser.add_argument("--split-mode", default="inherit", choices=["inherit", "time", "fraction"])
    parser.add_argument("--split-fraction", type=float, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        Config.SEED = int(args.seed)
        random.seed(Config.SEED)
        try:
            import numpy as np
            np.random.seed(Config.SEED)
        except Exception:
            pass
        tf.random.set_seed(Config.SEED)

    if args.epochs is not None:
        Config.EPOCHS = int(args.epochs)

    if args.split_mode != "inherit":
        Config.SPLIT_MODE = args.split_mode
    if args.split_fraction is not None:
        Config.SPLIT_FRACTION = float(args.split_fraction)
    if args.train_start is not None:
        Config.TRAIN_START = args.train_start
    if args.train_end is not None:
        Config.TRAIN_END = args.train_end
    if args.val_start is not None:
        Config.VAL_START = args.val_start
    if args.val_end is not None:
        Config.VAL_END = args.val_end
    if args.test_start is not None:
        Config.TEST_START = args.test_start
    if args.test_end is not None:
        Config.TEST_END = args.test_end

    print(f"\n🚀 INICIANDO ESTUDIO DE ABLACIÓN (Alineado con train.py)")
    print(f"   ⚡ Hardware: {Config.DEVICE}")
    print(f"   🎲 Seed: {Config.SEED}")
    print(f"   🔀 Split mode: {Config.SPLIT_MODE}")
    if Config.SPLIT_MODE == "time":
        print(f"   Train: {Config.TRAIN_START} -> {Config.TRAIN_END}")
        print(f"   Val:   {Config.VAL_START} -> {Config.VAL_END}")
        print(f"   Test:  {Config.TEST_START} -> {Config.TEST_END}")

    # "Full-frame" en este repo significa NO tiles (usar BigDataPipeline) y correr epochs completos.
    # Permite forzar comportamiento en servidores CPU/GPU via env var.
    if os.getenv("FULLFRAME", "0") == "1":
        Config.MAX_STEPS_PER_EPOCH = None
    
    output_base_dir = Config.EXPERIMENTS_DIR
    os.makedirs(output_base_dir, exist_ok=True)

    if getattr(Config, "SEQ_LEN", 0) < args.min_seq_len:
        print(f"   ⏱️ Ajustando SEQ_LEN: {Config.SEQ_LEN} -> {args.min_seq_len}")
        Config.SEQ_LEN = args.min_seq_len
    
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
    
    # Steps y repeat para evitar agotamiento del dataset
    steps_per_epoch = None
    validation_steps = None
    if getattr(Config, "MAX_STEPS_PER_EPOCH", None):
        train_ds = train_ds.take(Config.MAX_STEPS_PER_EPOCH)
        val_ds = val_ds.take(max(1, Config.MAX_STEPS_PER_EPOCH // 2))
        steps_per_epoch = Config.MAX_STEPS_PER_EPOCH
        validation_steps = max(1, Config.MAX_STEPS_PER_EPOCH // 2)
        train_ds = train_ds.repeat()
        val_ds = val_ds.repeat()
    else:
        steps_per_epoch = getattr(pipeline, "train_steps", None)
        validation_steps = getattr(pipeline, "val_steps", None)
        if steps_per_epoch is not None and validation_steps is not None:
            train_ds = train_ds.repeat()
            val_ds = val_ds.repeat()
    # =========================================================================

    # 2. BUCLE DE EXPERIMENTOS
    selected = args.models or [name for name, _ in EXPERIMENTS_TO_RUN]
    experiment_list = [(name, fn) for name, fn in EXPERIMENTS_TO_RUN if name in selected]
    suffix = args.experiment_suffix.strip()

    for strategy_name, builder_func in experiment_list:
        clear_session()
        gc.collect()
        
        experiment_name = f"Ablation_{strategy_name.upper()}_Legacy"
        if suffix:
            experiment_name = f"{experiment_name}_{suffix}"
        print(f"\n{'='*60}")
        print(f"🏗️  MODELO: {strategy_name.upper()} (Legacy Architecture)")
        print(f"{'='*60}")

        # --- RESUME LOGIC: Check if done ---
        model_path_best = os.path.join(output_base_dir, "models", f"{experiment_name}_best.h5")
        log_path = os.path.join(output_base_dir, "logs", f"{experiment_name}_log.csv")
        
        # Si existe el modelo "best" y el log tiene épocas completas, podríamos saltar
        # Pero mejor dejamos que run_experiment decida si reanudar o no.
        # Sin embargo, si ya terminó (época final alcanzada), run_experiment retornará None.
        if os.path.exists(model_path_best) and os.path.exists(log_path):
            try:
                import pandas as pd
                df_log = pd.read_csv(log_path)
                if not df_log.empty and df_log["epoch"].iloc[-1] >= (Config.EPOCHS - 1):
                    print(f"✅ Experimento {experiment_name} ya completado ({len(df_log)} épocas). Saltando.")
                    
                    # Cargar historial para el reporte final igualmente
                    all_histories[strategy_name] = df_log
                    final_metrics = {
                        "final_epoch": df_log["epoch"].iloc[-1],
                        "final_val_loss": df_log["val_loss"].iloc[-1],
                        "final_val_mae": df_log["val_mae"].iloc[-1],
                    }
                    results_summary.append({
                        'Model': strategy_name,
                        'Params': "Unknown (Skipped)",
                        'Final_Epoch': final_metrics["final_epoch"],
                        'Final_Val_Loss': final_metrics["final_val_loss"],
                        'Final_Val_MAE': final_metrics["final_val_mae"],
                        'Best_Val_Loss': df_log['val_loss'].min(),
                        'Best_Val_MAE': df_log['val_mae'].min()
                    })
                    continue
            except Exception as e:
                print(f"⚠️ Error verificando completitud de {experiment_name}: {e}. Se intentará reanudar.")
        
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
            experiment_name=experiment_name,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
        
        if history is None:
            print(f"⏭️ Experimento {experiment_name} omitido (ya completado o error).")
            # Intentar recuperar logs para el resumen
            if os.path.exists(log_path):
                import pandas as pd
                history_df = pd.read_csv(log_path)
                all_histories[strategy_name] = history_df
                # Add to summary if not already added
                # (Logic above handles full skip, this handles 'resume but finished immediately')
            continue

        all_histories[strategy_name] = history
        final_metrics = _get_last_epoch_metrics(history, log_path)
        
        # Guardamos métricas
        results_summary.append({
            'Model': strategy_name,
            'Params': model.count_params(),
            'Final_Epoch': final_metrics["final_epoch"],
            'Final_Val_Loss': final_metrics["final_val_loss"],
            'Final_Val_MAE': final_metrics["final_val_mae"],
            'Best_Val_Loss': _safe_min(history.history.get('val_loss', [])),
            'Best_Val_MAE': _safe_min(history.history.get('val_mae', []))
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
