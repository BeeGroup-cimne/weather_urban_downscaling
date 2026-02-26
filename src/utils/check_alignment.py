#!/usr/bin/env python
"""
Utility to verify spatial alignment between HR and LR data using overlay visualization.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cv2
from config.runtime import Config

# Use project-relative path
CACHE_DIR = Config.PATH_CACHE

def verificar_alineacion_overlay():
    print(f"🔍 Cargando Zarr: {CACHE_DIR}")
    
    if not os.path.exists(CACHE_DIR):
        print("❌ No se encuentra el archivo Zarr.")
        print(f"   Ejecuta primero el pipeline de datos.")
        return
        
    ds = xr.open_zarr(CACHE_DIR, consolidated=True)
    
    # 1. Obtener muestras (Primer tiempo, primer canal/variable)
    hr_data = ds['hr_target'].isel(time=0)
    lr_data = ds['lr_input'].isel(time=0)
    
    # Asegurar que son 2D (Lat, Lon) seleccionando canal 0 si existe
    if hr_data.ndim > 2: hr_data = hr_data[..., 0]
    if lr_data.ndim > 2: lr_data = lr_data[..., 0]
    
    # Convertir a numpy
    hr_img = hr_data.values
    lr_img = lr_data.values
    
    print(f"📐 Shape Original HR: {hr_img.shape}")
    print(f"📐 Shape Original LR: {lr_img.shape}")

    # 2. Re-escalar LR al tamaño de HR para superponer
    lr_resized = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 3. Normalizar para visualizar mejor (0 a 1)
    def normalize(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    
    hr_norm = normalize(hr_img)
    lr_norm = normalize(lr_resized)

    # 4. PLOT DE SUPERPOSICIÓN
    plt.figure(figsize=(12, 10))
    
    # Capa base: HR (Escala de grises)
    plt.imshow(hr_norm, cmap='gray', origin='lower', extent=[0, 10, 0, 10], label='HR (Ground Truth)')
    
    # Capa superior: LR (Mapa de calor semitransparente)
    plt.imshow(lr_norm, cmap='jet', alpha=0.5, origin='lower', extent=[0, 10, 0, 10], label='LR (ERA5 Input)')
    
    plt.title(f"Prueba de Alineación: HR (Gris) vs LR (Color)\nLR Shape: {lr_img.shape} -> HR Shape: {hr_img.shape}")
    plt.colorbar(label="Intensidad Relativa")
    
    # Guardar en experiments/figures
    output_path = os.path.join(Config.EXPERIMENTS_DIR, 'figures', 'prueba_alineacion_overlay.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"📸 Imagen guardada: {output_path}")
    plt.show()

if __name__ == "__main__":
    verificar_alineacion_overlay()