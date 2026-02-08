import os
import platform
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from config.runtime import Config
from tensorflow.keras.utils import plot_model

def resolve_stats_path():
    """Resolver ruta de stats de forma robusta."""
    candidates = [
        getattr(Config, "STATS_PATH", None),
        os.path.join("data", "processed", "stats_config.npz"),
        os.path.join("scripts", "stats_config.npz"),
        "stats_config.npz"
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def notify_completion(message="Proceso finalizado"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_msg = f"{message} a las {timestamp}"
    print(f"\n🔔 {full_msg}")
    if platform.system() == "Darwin":
        os.system(f'say "{message}" &')
    elif platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 1000)
    else:
        print('\a')

def plot_comparative_history(histories, save_dir):
    if not histories: return
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    metrics = [('loss', 'Loss'), ('mae', 'MAE')]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        for name, history in histories.items():
            ax.plot(history.history[metric], label=f'{name} Train')
            val_key = f'val_{metric}'
            if val_key in history.history:
                ax.plot(history.history[val_key], '--', label=f'{name} Val')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    
    filepath = os.path.join(save_dir, "comparativa_final.png")
    plt.savefig(filepath)
    print(f"📊 Gráfica guardada en: {filepath}")


# 4. ENTRENAMIENTO Y VISUALIZACIÓN

def run_experiment(model, train_ds, val_ds, experiment_name, steps_per_epoch=None, validation_steps=None):
    """Ejecuta el ciclo de entrenamiento estandarizado"""
    print(f"\n🧪 Iniciando {experiment_name}...")

    base_dir = Config.EXPERIMENTS_DIR
    log_dir = os.path.join(base_dir, "logs")
    model_dir = os.path.join(base_dir, "models")
    fig_dir = os.path.join(base_dir, "figures")

    # Crear carpetas si no existen
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{experiment_name}_log.csv")
    model_path = os.path.join(model_dir, f"{experiment_name}_best.h5")
    plot_path = os.path.join(fig_dir, f"{experiment_name}_architecture.png")   

    # -- GUARDAR IMAGEN DE LA ARQUITECTURA (OPCIONAL) ---
    try:
        print(f"   📸 Guardando diagrama del modelo en {plot_path}...")
        plot_model(
            model, 
            to_file=plot_path, 
            show_shapes=True, 
            show_layer_names=True,
            expand_nested=True
        )
    except Exception as e:
        print(f"   ⚠️ No se pudo graficar el modelo (quizás falta graphviz): {e}")

        

    # Callbacks
    callbacks = [
        # Guardar solo el mejor modelo en la carpeta models/
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1),
        
        # Detener si no mejora
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        
        # Reducir Learning Rate
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        
        # Guardar log CSV en la carpeta logs/
        CSVLogger(log_path)
    ]

    # --- 4. ENTRENAR ---
    fit_kwargs = {
        "epochs": Config.EPOCHS,
        "callbacks": callbacks,
        "verbose": 1
    }
    
    if steps_per_epoch is None and getattr(Config, "MAX_STEPS_PER_EPOCH", None):
        steps_per_epoch = Config.MAX_STEPS_PER_EPOCH
        validation_steps = max(1, Config.MAX_STEPS_PER_EPOCH // 2)

    if steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = steps_per_epoch
    if validation_steps is not None:
        fit_kwargs["validation_steps"] = validation_steps
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        **fit_kwargs
    )
    
    print(f"✅ Experimento {experiment_name} finalizado.")
    print(f"   📄 Log guardado en: {log_path}")
    print(f"   💾 Modelo guardado en: {model_path}")
    
    return history

def visualize_results(model, val_ds, title):
    """Genera gráfica comparativa Input/Pred/Target y la guarda"""
    try:

        # Definir carpeta de destino
        # Usamos os.path.join para compatibilidad de rutas
        output_dir = os.path.join(Config.EXPERIMENTS_DIR, "figures")
        
        #Crear la carpeta si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer un batch para visualizar
        (x_lr, x_st), y_true = next(iter(val_ds))

        stats_path = resolve_stats_path()
        if not stats_path:
            print("⚠️ No se encontró stats_config.npz. Saltando visualización.")
            return
        
        print(f"   📊 Cargando estadísticas desde: {stats_path}")
        stats = np.load(stats_path)

        
        # Predecir
        mean_hr, std_hr = stats['mean_hr'], stats['std_hr']
    
        y_pred = model.predict([x_lr, x_st], verbose=0)

        y_pred_real = (y_pred * std_hr) + mean_hr
        y_true_real = (y_true * std_hr) + mean_hr

        def _align_lr_for_plot(lr_up, hr_ref):
            def _corr(a, b):
                a = a.flatten()
                b = b.flatten()
                if np.std(a) == 0 or np.std(b) == 0:
                    return -np.inf
                return np.corrcoef(a, b)[0, 1]

            candidates = {
                "as_is": lr_up,
                "rot90": np.rot90(lr_up, 1),
                "rot180": np.rot90(lr_up, 2),
                "rot270": np.rot90(lr_up, 3),
                "flipud": np.flipud(lr_up),
                "fliplr": np.fliplr(lr_up),
                "transpose": lr_up.T,
                "transpose_flipud": np.flipud(lr_up.T),
                "transpose_fliplr": np.fliplr(lr_up.T),
            }

            best_name, best_arr, best_score = "as_is", lr_up, -np.inf
            for name, arr in candidates.items():
                score = _corr(arr, hr_ref)
                if score > best_score:
                    best_score = score
                    best_name, best_arr = name, arr
            return best_arr, best_name

        # Visualizar primer sample, último frame de la secuencia
        idx = 0
        t = Config.SEQ_LEN - 1 

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{title} - Sample {idx} Frame {t}")

        # 1. Input LR (align for display)
        def _rotate(arr):
            deg = int(os.getenv("PLOT_ROTATE", "0")) % 360
            if deg == 0:
                return arr
            return np.rot90(arr, deg // 90)

        plt.subplot(1, 3, 1)
        # x_lr shape: (Batch, Time, Lat, Lon, Chan)
        lr_raw = x_lr[idx, t, :, :, 0]
        # Upsample LR for display using nearest-neighbor to keep pixelated look
        lr_up = tf.image.resize(lr_raw[..., None], Config.HR_SHAPE, method="nearest").numpy()[..., 0]
        hr_ref = y_true_real[idx, t, :, :, 0]
        if hasattr(hr_ref, "numpy"):
            hr_ref = hr_ref.numpy()
        lr_disp, lr_tag = _align_lr_for_plot(lr_up, hr_ref)
        print(f"   ℹ️ LR display alignment: {lr_tag}")
        plt.imshow(_rotate(lr_disp), cmap='viridis', origin='lower', interpolation='nearest') 
        plt.title("Input Low Res (LR)")
        plt.axis('off')

        # 2. Predicción HR
        plt.subplot(1, 3, 2)
        plt.imshow(_rotate(y_pred_real[idx, t, :, :, 0]), cmap='viridis',origin='lower')
        plt.title("Prediction (HR)")
        plt.axis('off')

        # 3. Ground Truth HR
        plt.subplot(1, 3, 3)
        plt.imshow(_rotate(y_true_real[idx, t, :, :, 0]), cmap='viridis',origin='lower')
        plt.title("Ground Truth (HR)")
        plt.axis('off')
        
        # 3. Guardar imagen
        # Usamos f-string y aseguramos que sea un string válido
        safe_title = title.replace(' ', '_')
        filename = f"result_{safe_title}.png"
        
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path) # <--- AQUÍ ESTABA EL ERROR
        plt.close()
        print(f"📸 Visualización guardada en: {save_path}")
        
    except Exception as e:
        print(f"⚠️ Error al generar visualización: {e}")
