import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIG ---
CACHE_DIR = "/Users/kerincardona/weather_urban_downscaling/data/processed/weather_cache.zarr"

def verificar_orientacion():
    print(f"🔍 Inspeccionando archivo: {CACHE_DIR}")
    
    if not os.path.exists(CACHE_DIR):
        print("❌ No se encuentra el archivo Zarr.")
        return

    ds = xr.open_zarr(CACHE_DIR, consolidated=True)
    
    # 1. IDENTIFICAR VARIABLES
    # Buscamos nombres típicos de latitud/longitud
    print("\n--- 1. Análisis de Dimensiones ---")
    print(f"Dimensiones encontradas: {list(ds.dims)}")
    
    # Detectar cuál es el Target (HR) y cuál el Input (LR)
    # Asumimos que 'hr_target' y 'lr_input' son las claves
    if 'hr_target' not in ds or 'lr_input' not in ds:
        print("⚠️ No encuentro las variables 'hr_target' o 'lr_input'.")
        print(f"Variables disponibles: {list(ds.data_vars)}")
        return

    hr_var = ds['hr_target']
    lr_var = ds['lr_input']

    print(f"Shape HR (Target): {hr_var.shape} -> Nombres: {hr_var.dims}")
    print(f"Shape LR (Input):  {lr_var.shape} -> Nombres: {lr_var.dims}")

    # 2. VERIFICACIÓN DE ORDEN (Lat vs Lon)
    # En meteorología/imágenes, el orden ESTÁNDAR debe ser: (Time, Latitud, Longitud)
    # Si es (Time, Longitud, Latitud), saldrá rotado en imshow.
    
    print("\n--- 2. Verificación de Coordenadas ---")
    
    # Función auxiliar para chequear si una dimensión parece Latitud o Longitud
    def es_latitud(dim_name, coords):
        # Latitud varía menos en Barcelona (41.x) que Longitud (2.x)? No, al revés.
        # Mejor: Latitud suele llamarse 'lat', 'y', 'latitude'
        return any(x in dim_name.lower() for x in ['lat', 'y'])

    # Chequeo HR
    dims_hr = hr_var.dims # Ejemplo: ('time', 'y', 'x') o ('time', 'lat', 'lon')
    spatial_dims_hr = dims_hr[-2:] # Las últimas dos suelen ser las espaciales
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
    if 'channel' in data_hr.dims: # O la dimensión que sea canales
        data_hr = data_hr.isel({d:0 for d in data_hr.dims if d not in spatial_dims_hr})
    elif data_hr.ndim > 2:
        # Asumimos último eje son canales
        data_hr = data_hr[..., 0]

    if 'channel' in data_lr.dims:
        data_lr = data_lr.isel({d:0 for d in data_lr.dims if d not in spatial_dims_lr})
    elif data_lr.ndim > 2:
        data_lr = data_lr[..., 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- FILA 1: IMSHOW (RAW NUMPY) ---
    # Esto muestra cómo lo ve la Red Neuronal (Matriz pura)
    ax1 = axes[0, 0]
    ax1.imshow(data_lr.values, origin='lower') # origin='lower' pone (0,0) abajo a la izq
    ax1.set_title(f"Input LR (Numpy/Imshow)\nShape: {data_lr.shape}")
    
    ax2 = axes[0, 1]
    ax2.imshow(data_hr.values, origin='lower')
    ax2.set_title(f"Target HR (Numpy/Imshow)\nShape: {data_hr.shape}")

    # --- FILA 2: XARRAY PLOT (GEO-CONSCIENTE) ---
    # Esto usa las coordenadas reales. Si esto sale bien pero lo de arriba sale mal,
    # significa que los datos están transpuestos.
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
    plt.savefig("diagnostico_rotacion.png")
    print("📸 Diagnóstico guardado en: diagnostico_rotacion.png")
    plt.show()

if __name__ == "__main__":
    verificar_orientacion()