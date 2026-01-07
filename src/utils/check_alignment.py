import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

CACHE_DIR = "/Users/kerincardona/weather_urban_downscaling/data/processed/weather_cache.zarr"

def verificar_alineacion_overlay():
    print(f"🔍 Cargando Zarr: {CACHE_DIR}")
    ds = xr.open_zarr(CACHE_DIR, consolidated=True)
    
    # 1. Obtener muestras (Primer tiempo, primer canal/variable)
    # Asumimos orden (Time, Lat, Lon, Chan) o (Time, Lat, Lon)
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
    # Usamos interpolación 'nearest' para ver los píxeles gordos originales
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
    # alpha=0.5 hace la magia de la transparencia
    plt.imshow(lr_norm, cmap='jet', alpha=0.5, origin='lower', extent=[0, 10, 0, 10], label='LR (ERA5 Input)')
    
    plt.title(f"Prueba de Alineación: HR (Gris) vs LR (Color)\nLR Shape: {lr_img.shape} -> HR Shape: {hr_img.shape}")
    plt.colorbar(label="Intensidad Relativa")
    
    # Guardar
    plt.savefig("prueba_alineacion_overlay.png", dpi=150)
    print("📸 Imagen guardada: prueba_alineacion_overlay.png")
    plt.show()

if __name__ == "__main__":
    verificar_alineacion_overlay()