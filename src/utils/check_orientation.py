#!/usr/bin/env python
"""
Utility to verify data orientation and visualize HR/LR alignment.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from config.config import Config

# Use project-relative path
CACHE_DIR = Config.PATH_CACHE

def verificar_orientacion():
    print(f"🔍 Inspeccionando archivo: {CACHE_DIR}")
    
    if not os.path.exists(CACHE_DIR):
        print("❌ No se encuentra el archivo Zarr.")
        print(f"   Ejecuta primero el pipeline de datos.")
        return

    ds = xr.open_zarr(CACHE_DIR, consolidated=True)
    
    # 1. IDENTIFICAR VARIABLES
    print("\n--- 1. Análisis de Dimensiones ---")
    print(f"Dimensiones encontradas: {list(ds.dims)}")
    
    # Detectar cuál es el Target (HR) y cuál el Input (LR)
    if 'hr_target' not in ds or 'lr_input' not in ds:
        print("⚠️ No encuentro las variables 'hr_target' o 'lr_input'.")
        print(f"Variables disponibles: {list(ds.data_vars)}")
        return

    hr_var = ds['hr_target']
    lr_var = ds['lr_input']

    print(f"Shape HR (Target): {hr_var.shape} -> Nombres: {hr_var.dims}")
    print(f"Shape LR (Input):  {lr_var.shape} -> Nombres: {lr_var.dims}")

    # 2. VERIFICACIÓN DE ORDEN (Lat vs Lon)
    print("\n--- 2. Verificación de Coordenadas ---")
    
    # Chequeo HR
    dims_hr = hr_var.dims
    spatial_dims_hr = dims_hr[-2:]
    print(f"Dimensiones espaciales HR: {spatial_dims_hr}")
    
    # Chequeo LR
    dims_lr = lr_var.dims
    spatial_dims_lr = dims_lr[-2:]
    print(f"Dimensiones espaciales LR: {spatial_dims_lr}")

    # 3. VISUALIZACIÓN COMPARATIVA
    print("\n--- 3. Generando Prueba Visual (Matplotlib vs Xarray) ---")
    
    # Tomamos el primer tiempo
    data_hr = hr_var.isel(time=0)
    data_lr = lr_var.isel(time=0)
    
    # Si tiene canales, tomamos el primero (Temp)
    if 'channel' in data_hr.dims:
        data_hr = data_hr.isel({d:0 for d in data_hr.dims if d not in spatial_dims_hr})
    elif data_hr.ndim > 2:
        data_hr = data_hr[..., 0]

    if 'channel' in data_lr.dims:
        data_lr = data_lr.isel({d:0 for d in data_lr.dims if d not in spatial_dims_lr})
    elif data_lr.ndim > 2:
        data_lr = data_lr[..., 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- FILA 1: IMSHOW (RAW NUMPY) ---
    ax1 = axes[0, 0]
    ax1.imshow(data_lr.values, origin='lower')
    ax1.set_title(f"Input LR (Numpy/Imshow)\nShape: {data_lr.shape}")
    
    ax2 = axes[0, 1]
    ax2.imshow(data_hr.values, origin='lower')
    ax2.set_title(f"Target HR (Numpy/Imshow)\nShape: {data_hr.shape}")

    # --- FILA 2: XARRAY PLOT (GEO-CONSCIENTE) ---
    ax3 = axes[1, 0]
    try:
        data_lr.plot(ax=ax3)
        ax3.set_title("Input LR (Xarray - Coordenadas Reales)")
    except:
        ax3.text(0.5, 0.5, "Error ploteando Xarray LR")

    ax4 = axes[1, 1]
    try:
        data_hr.plot(ax=ax4)
        ax4.set_title("Target HR (Xarray - Coordenadas Reales)")
    except:
        ax4.text(0.5, 0.5, "Error ploteando Xarray HR")

    plt.tight_layout()
    
    # Save to experiments/figures
    output_path = os.path.join(Config.EXPERIMENTS_DIR, 'figures', 'diagnostico_rotacion.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"📸 Diagnóstico guardado en: {output_path}")
    plt.show()

if __name__ == "__main__":
    verificar_orientacion()