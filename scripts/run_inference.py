import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import tensorflow as tf
from config.config import Config
from train import BigDataPipeline
from src.utils import visualize_results

val_ds_cache = None

def cargar_datos_validacion():
    """Carga los datos una sola vez y los guarda en memoria global"""
    global val_ds_cache
    if val_ds_cache is None:
        print("🌊 Cargando Pipeline de Datos (Validación)...")
        pipeline = BigDataPipeline(Config)
        pipeline.process_static_data()
        _, val_ds = pipeline.get_tf_datasets()
        val_ds_cache = val_ds
    return val_ds_cache

def evaluar_modelo(nombre_archivo_modelo, etiqueta_personalizada=None):
    """
    Carga un modelo específico y genera su visualización.
    
    Args:
        nombre_archivo_modelo (str): Nombre del archivo .h5 (ej: 'Transformer_best.h5')
        etiqueta_personalizada (str): Título para la gráfica (Opcional)
    """
    # 1. Construir ruta (asumiendo estructura de carpeta experiments/models)
    ruta_modelo = os.path.join("experiments", "models", nombre_archivo_modelo)
    
    # Fallback: Si no está en la carpeta experiments, buscar en raíz
    if not os.path.exists(ruta_modelo):
        if os.path.exists(nombre_archivo_modelo):
            ruta_modelo = nombre_archivo_modelo
        else:
            print(f"❌ ERROR: No encuentro el archivo '{ruta_modelo}' ni en raíz.")
            return

    # 2. Cargar Modelo
    print(f"\n📂 Cargando modelo: {nombre_archivo_modelo}...")
    try:
        model = tf.keras.models.load_model(ruta_modelo)
    except Exception as e:
        print(f"⚠️ Error cargando {nombre_archivo_modelo}: {e}")
        return

    # 3. Obtener Datos
    val_ds = cargar_datos_validacion()

    # 4. Visualizar
    # Si no pasamos etiqueta, usamos el nombre del archivo sin extensión
    if etiqueta_personalizada is None:
        etiqueta_personalizada = os.path.splitext(nombre_archivo_modelo)[0]
        
    print(f"🎨 Generando gráfica para: {etiqueta_personalizada}")
    
    # Llamamos a tu función de utils.py
    # Se guardará en experiments/figures/result_Inferencia_{etiqueta}.png
    visualize_results(model, val_ds, title=f"Inferencia_{etiqueta_personalizada}")


# EJECUCIÓN MAESTRA

if __name__ == "__main__":
    
    # 📝 LISTA DE MODELOS A PROBAR
    # Pon aquí los nombres exactos de tus archivos .h5
    mis_modelos = [
        "UNet_best.h5",
        "ConvLSTM_best.h5", 
        "Transformer_best.h5",
        # Agrega modelos híbridos si existen:
        # "Hybrid_LSTM_best.h5",
        # "Hybrid_Mamba_best.h5"
    ]

    print(f"🚀 Iniciando evaluación masiva de {len(mis_modelos)} modelos...\n")

    for modelo in mis_modelos:
        evaluar_modelo(modelo)

    print("\n✅ ¡Proceso terminado! Revisa la carpeta 'experiments/figures'")