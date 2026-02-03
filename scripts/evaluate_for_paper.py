#!/usr/bin/env python3
"""
Evaluación sistemática de modelos para paper académico
Genera métricas completas y visualizaciones comparativas
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

# Agregar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, os.path.join(parent_dir, 'src')])

from config.runtime import Config
from train import BigDataPipeline

class ModelEvaluator:
    def __init__(self):
        self.config = Config
        self.pipeline = BigDataPipeline(self.config)
        self.results = {}
        
    def setup_data(self):
        """Configurar datos de evaluación"""
        print("🔧 Configurando pipeline de datos...")
        self.pipeline.process_static_data()
        _, val_ds = self.pipeline.get_tf_datasets()
        
        # Convertir a numpy para evaluación detallada
        val_data = list(val_ds.take(50))  # 50 batches para evaluación
        self.X_val = np.concatenate([x[0] for x in val_data])
        self.y_val = np.concatenate([x[1] for x in val_data])
        
        print(f"📊 Datos cargados: {self.X_val.shape} -> {self.y_val.shape}")
        
    def load_model(self, model_path):
        """Cargar modelo desde experiments/models"""
        full_path = os.path.join("experiments", "models", model_path)
        if not os.path.exists(full_path):
            # Buscar en raíz si no está en experiments
            if os.path.exists(model_path):
                full_path = model_path
            else:
                raise FileNotFoundError(f"No se encuentra: {model_path}")
                
        return tf.keras.models.load_model(full_path)
        
    def calculate_metrics(self, y_true, y_pred):
        """Calcular métricas completas"""
        # Asegurar shapes correctos para SSIM
        if len(y_true.shape) == 4:
            # Para imágenes: (batch, height, width, channels)
            ssim_scores = []
            for i in range(y_true.shape[0]):
                # Usar canal 0 (temperatura principal) para SSIM
                ssim_score = ssim(
                    y_true[i, :, :, 0], 
                    y_pred[i, :, :, 0],
                    data_range=y_true[i, :, :, 0].max() - y_true[i, :, :, 0].min()
                )
                ssim_scores.append(ssim_score)
            avg_ssim = np.mean(ssim_scores)
        else:
            avg_ssim = 0
            
        # Métricas estándar
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        
        # MAE específico para temperatura (primer canal)
        temp_mae = mean_absolute_error(
            y_true[..., 0].flatten(), 
            y_pred[..., 0].flatten()
        )
        
        return {
            'MSE': mse,
            'MAE': mae, 
            'RMSE': rmse,
            'Temp_MAE': temp_mae,
            'SSIM': avg_ssim,
            'MAE_Reduction': f"{((5.0 - temp_mae) / 5.0) * 100:.1f}%"  # vs baseline
        }
        
    def evaluate_model(self, model_name):
        """Evaluar un modelo específico"""
        print(f"\n🔍 Evaluando {model_name}...")
        
        try:
            model = self.load_model(model_name)
            
            # Predicciones
            y_pred = model.predict(self.X_val, verbose=0)
            
            # Métricas
            metrics = self.calculate_metrics(self.y_val, y_pred)
            metrics['Model'] = model_name
            metrics['Parameters'] = self.count_parameters(model)
            
            self.results[model_name] = metrics
            print(f"✅ {model_name}: MAE={metrics['MAE']:.4f}, SSIM={metrics['SSIM']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluando {model_name}: {e}")
            
    def count_parameters(self, model):
        """Contar parámetros del modelo"""
        return model.count_params() / 1e6  # En millones
        
    def generate_comparison_table(self):
        """Generar tabla comparativa para paper"""
        if not self.results:
            print("❌ No hay resultados para comparar")
            return
            
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.round(4)
        
        print("\n📊 TABLA COMPARATIVA PARA PAPER")
        print("=" * 80)
        print(df.to_string())
        
        # Guardar tabla
        os.makedirs("experiments", exist_ok=True)
        df.to_csv("experiments/paper_results_table.csv")
        print(f"\n💾 Tabla guardada en: experiments/paper_results_table.csv")
        
    def generate_comparison_plots(self):
        """Generar gráficas comparativas"""
        if not self.results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Modelos - Weather Downscaling', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        
        # Plot 1: MAE Comparison
        mae_values = [self.results[m]['MAE'] for m in models]
        axes[0, 0].bar(models, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: SSIM Comparison
        ssim_values = [self.results[m]['SSIM'] for m in models]
        axes[0, 1].bar(models, ssim_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('Structural Similarity (SSIM)')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Parameter Efficiency
        params = [self.results[m]['Parameters'] for m in models]
        mae_per_param = [self.results[m]['MAE'] / self.results[m]['Parameters'] for m in models]
        axes[1, 0].bar(models, mae_per_param, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_title('MAE por Millón de Parámetros')
        axes[1, 0].set_ylabel('MAE/Params')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Temporal MAE
        temp_mae = [self.results[m]['Temp_MAE'] for m in models]
        axes[1, 1].bar(models, temp_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('MAE Temperatura (°C)')
        axes[1, 1].set_ylabel('MAE (°C)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("experiments/paper_model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📈 Gráfica guardada en: experiments/paper_model_comparison.png")

def main():
    """Ejecución principal"""
    print("🚀 Iniciando evaluación sistemática para paper...")
    
    evaluator = ModelEvaluator()
    
    # Configurar datos
    evaluator.setup_data()
    
    # Lista de modelos a evaluar
    models_to_evaluate = [
        "UNet_best.h5",
        "ConvLSTM_best.h5", 
        "Transformer_best.h5",
        "Transformer_gpu_optimized.h5",  # Nuevo Transformer optimizado
        # Agrega más modelos si existen
    ]
    
    # Evaluar cada modelo
    for model in models_to_evaluate:
        evaluator.evaluate_model(model)
    
    # Generar resultados
    evaluator.generate_comparison_table()
    evaluator.generate_comparison_plots()
    
    print("\n🎯 Evaluación completada! Revisa la carpeta 'experiments/'")

if __name__ == "__main__":
    main()
