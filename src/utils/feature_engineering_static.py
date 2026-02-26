import xarray as xr
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.runtime import Config

# ================= CONFIGURACIÓN =================
# Ruta a tu archivo de datos estáticos actual
INPUT_PATH = os.environ.get('STATIC_INPUT_PATH', Config.PATH_STATIC)

# Ruta donde guardaremos el dataset mejorado
OUTPUT_PATH = os.path.join(Config.BASE_DIR, 'data', 'static', 'weather_static_engineered.zarr')

def engineer_static_features():
    print(f"🔧 Iniciando Feature Engineering sobre: {INPUT_PATH}")
    
    # 1. Cargar el dataset original
    ds = xr.open_zarr(INPUT_PATH)
    print(f"   Variables originales: {len(ds.data_vars)}")
    
    # =========================================================
    # 2. CREACIÓN DE INDICES COMPUESTOS (Agrupación Térmica)
    # =========================================================
    # Usamos la MEDIA de los percentiles para mantener la escala (0-1 o 0-100)
    # Si sumáramos, podríamos tener valores desorbitados.
    
    print("   ⚗️  Fusionando categorías de edificios...")

    # A. INDICE RESIDENCIAL (Calor nocturno, calefacción)
    # Combina área residencial + número de viviendas
    ds['residential_index'] = (
        ds['building_area_residential_percentile'] + 
        ds['n_dwellings_percentile']
    ) / 2.0

    # B. INDICE INDUSTRIAL/LOGÍSTICO (Techos grandes, calor diurno)
    ds['industrial_index'] = (
        ds['building_area_industrial_percentile'] + 
        ds['building_area_warehouse_parking_percentile']
    ) / 2.0

    # C. INDICE SERVICIOS Y OFICINAS (Actividad diurna, AC)
    ds['services_index'] = (
        ds['building_area_offices_percentile'] + 
        ds['building_area_commercial_percentile'] + 
        ds['building_area_healthcare_and_charity_percentile'] + 
        ds['building_area_religious_percentile'] + 
        ds['building_area_cultural_percentile']
    ) / 5.0

    # D. INDICE OCIO Y GRANDES INSTALACIONES (Espacios singulares)
    ds['leisure_index'] = (
        ds['building_area_sports_facilities_percentile'] + 
        ds['building_area_entertainment_venues_percentile'] + 
        ds['building_area_leisure_and_hospitality_percentile'] + 
        ds['building_area_singular_building_percentile']
    ) / 4.0

    # =========================================================
    # 3. VARIABLES FÍSICAS (Geometría y Terreno)
    # =========================================================
    print("   📐 Procesando variables físicas...")

    # A. PROXY DE ALTURA (Vital para la sombra)
    # Renombramos para que tenga sentido físico
    ds['height_index'] = ds['n_floors_above_ground_percentile']

    # B. TOPOGRAFÍA
    # Mantenemos elevación tal cual (es la más importante)
    # ds['elevation'] ya existe

    # C. VEGETACIÓN (Refrigeración)
    # Mantenemos NDVI Mean y Min (para detectar agua/asfalto puro)
    # ndvi_max suele ser redundante con mean en ciudad
    # ds['ndvi_mean'] ya existe
    # ds['ndvi_min'] ya existe

    # =========================================================
    # 4. LIMPIEZA Y SELECCIÓN FINAL
    # =========================================================
    # Lista de las variables "Campeonas" que nos quedamos
    keep_vars = [
        'elevation',
        'height_index',
        'ndvi_mean',
        'ndvi_min',
        'residential_index',
        'industrial_index',
        'services_index',
        'leisure_index'
    ]

    # Crear nuevo dataset solo con lo que importa
    ds_final = ds[keep_vars]
    
    # Convertir a float32 para ahorrar memoria (GPU Friendly)
    ds_final = ds_final.astype(np.float32)
    

    # =========================================================
    # 🚑 FIX PARA EL ERROR DE ZARR (TypeError: found 238)
    # =========================================================
    print("   🚑 Reparando tipos de datos en coordenadas...")
    # El error dice que el 'index' tiene objetos no-string. Lo forzamos a string.
    if 'index' in ds_final.coords:
        ds_final.coords['index'] = ds_final.coords['index'].astype(str)
    
    # Asegurarnos de limpiar cualquier codificación previa que pudiera molestar
    for var in ds_final.coords:
        if 'chunks' in ds_final.coords[var].encoding:
            del ds_final.coords[var].encoding['chunks']

            
    # =========================================================
    
    

    print("\n📊 Dataset Final Generado:")
    print(ds_final)
    
    # =========================================================
    # 5. GUARDADO
    # =========================================================
    if os.path.exists(OUTPUT_PATH):
        import shutil
        print(f"   ⚠️ Borrando versión anterior en {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)

    print(f"💾 Guardando nuevo Zarr en: {OUTPUT_PATH}")
    ds_final.to_zarr(OUTPUT_PATH, mode='w', consolidated=True)
    print("✅ ¡Feature Engineering completado con éxito!")

if __name__ == "__main__":
    engineer_static_features()