import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
# 1. Tu archivo de estaciones (Zarr)
RUTA_ESTACIONES_ZARR = 'data/static/weather_static_engineered.zarr'

# 2. Tus edificios 3D
RUTA_EDIFICIOS = "scripts/Scripts_Utils/barcelona_infalible.geojson"

# 3. Salida
RUTA_OUTPUT = "data/processed/weather_static_FINAL_stations.zarr"

# 4. RADIO DE INFLUENCIA (CRÍTICO)
# ¿Qué tanto afecta el entorno a la estación?
# 150m es el estándar para "Local Climate Zones" (LCZ).
RADIO_BUFFER = 150 # Metros

def enriquecer_estaciones():
    print("🚀 Iniciando Enriquecimiento de Estaciones...")
    
    # 1. CARGAR ESTACIONES (Zarr)
    ds = xr.open_zarr(RUTA_ESTACIONES_ZARR)
    indices = ds.index.values 
    print(f"   - Total de estaciones/puntos: {len(indices):,}")
    
    # 2. CONVERTIR TEXTO A GEOMETRÍA
    print("📍 Decodificando coordenadas...")
    lat_lons = [x.split('_') for x in indices]
    lats = [float(x[0]) for x in lat_lons]
    lons = [float(x[1]) for x in lat_lons]
    
    df_estaciones = pd.DataFrame({'index_zarr': indices, 'lat': lats, 'lon': lons})
    gdf_estaciones = gpd.GeoDataFrame(
        df_estaciones, 
        geometry=[Point(xy) for xy in zip(lons, lats)],
        crs="EPSG:4326" 
    )
    
    # 3. CARGAR EDIFICIOS (CON AUTO-REPARACIÓN)
    print("🏠 Cargando edificios...")
    gdf_edificios = gpd.read_file(RUTA_EDIFICIOS)
    
    # --- 🛑 BLOQUE DE REPARACIÓN DE DATOS 🛑 ---
    # Si falta la columna 'height_m', la calculamos ahora mismo
    if 'height_m' not in gdf_edificios.columns:
        print("⚠️ No se encontró 'height_m'. Calculándola al vuelo...")
        
        # Intentamos usar niveles si existen
        if 'building:levels' in gdf_edificios.columns:
            # Convertir a numérico forzando errores a NaN
            gdf_edificios['levels_clean'] = pd.to_numeric(gdf_edificios['building:levels'], errors='coerce')
            # Rellenar nulos con 3 pisos (estándar promedio)
            gdf_edificios['levels_clean'] = gdf_edificios['levels_clean'].fillna(3)
        else:
            # Si ni siquiera hay niveles, asumimos 3 pisos para todos
            print("   ⚠️ Tampoco hay 'building:levels'. Asumiendo 3 pisos por defecto.")
            gdf_edificios['levels_clean'] = 3
            
        # Fórmula: Pisos * 3 metros
        gdf_edificios['height_m'] = gdf_edificios['levels_clean'] * 3
        
        # También aseguramos que exista 'levels_final' para el cálculo de max_levels
        if 'levels_final' not in gdf_edificios.columns:
             gdf_edificios['levels_final'] = gdf_edificios['levels_clean']
             
    print("   ✅ Columna 'height_m' verificada.")
    # ----------------------------------------------
    
    # Reproyectar a Metros
    print("📏 Reproyectando a UTM 31N...")
    gdf_estaciones = gdf_estaciones.to_crs("EPSG:25831")
    gdf_edificios = gdf_edificios.to_crs("EPSG:25831")
    
    # Calcular huella
    gdf_edificios['footprint_m2'] = gdf_edificios.geometry.area
    
    # 4. CREAR BURBUJAS
    print(f"⭕ Creando zonas de influencia de {RADIO_BUFFER}m...")
    gdf_buffers = gdf_estaciones.copy()
    gdf_buffers.geometry = gdf_estaciones.geometry.buffer(RADIO_BUFFER)
    AREA_BURBUJA = np.pi * (RADIO_BUFFER**2)
    
    # 5. CRUCE ESPACIAL
    print("🔗 Analizando intersecciones...")
    join = gpd.sjoin(gdf_buffers, gdf_edificios, how="left", predicate="intersects")
    
    # 6. CÁLCULO DE MÉTRICAS
    print("∑ Calculando física urbana...")
    
    # Rellenar nulos (ahora sí funcionará porque height_m existe seguro)
    join['footprint_m2'] = join['footprint_m2'].fillna(0)
    join['height_m'] = join['height_m'].fillna(0)
    
    # Si levels_final no existe (caso raro), lo creamos
    if 'levels_final' not in join.columns:
        join['levels_final'] = 0
    else:
        join['levels_final'] = join['levels_final'].fillna(0)

    stats = join.groupby('index_zarr').agg({
        'footprint_m2': 'sum',
        'height_m': 'mean', 
        'levels_final': 'max'
    }).rename(columns={
        'footprint_m2': 'total_built',
        'height_m': 'avg_height',
        'levels_final': 'max_levels'
    })
    
    df_final = df_estaciones.set_index('index_zarr').join(stats).fillna(0)
    
    # 7. DERIVAR VARIABLES FÍSICAS
    print("⚗️ Derivando Sky View Factor (SVF)...")
    
    # Densidad
    df_final['building_density'] = df_final['total_built'] / AREA_BURBUJA
    df_final['building_density'] = df_final['building_density'].clip(upper=1.0)
    
    # Ancho Calle
    L_ref = RADIO_BUFFER * 2 
    densidad_safe = df_final['building_density'].clip(lower=0.0001)
    df_final['street_width'] = L_ref * (1 - np.sqrt(densidad_safe))
    
    # SVF
    aspect_ratio = df_final['avg_height'] / (0.5 * df_final['street_width'])
    df_final['svf'] = np.cos(np.arctan(aspect_ratio))
    df_final.loc[df_final['building_density'] < 0.01, 'svf'] = 1.0
    
    # Rugosidad
    df_final['roughness'] = df_final['avg_height'] * 0.1 * df_final['building_density']
    df_final.loc[df_final['roughness'] == 0, 'roughness'] = 0.001

    # 8. GUARDAR
    print(f"💾 Inyectando variables en: {RUTA_OUTPUT}")
    
    vars_df = df_final[['avg_height', 'building_density', 'svf', 'roughness', 'max_levels']]
    ds_vars = xr.Dataset.from_dataframe(vars_df)
    ds_vars = ds_vars.rename({'index_zarr': 'index'})
    
    ds_final = xr.merge([ds, ds_vars])
    ds_final.to_zarr(RUTA_OUTPUT, mode='w', consolidated=True)
    
    print("✅ ¡Integración completada con éxito!")
    return df_final

def visualizar_validacion(df):
    """Muestra un mapa rápido para verificar que tiene sentido"""
    print("\n🎨 Generando mapa de validación (SVF)...")
    plt.figure(figsize=(10, 8))
    plt.scatter(df.lon, df.lat, c=df.svf, cmap='magma', s=5, alpha=0.7)
    plt.colorbar(label='Sky View Factor (0=Cerrado, 1=Abierto)')
    plt.title("Validación: SVF en Estaciones")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()

if __name__ == "__main__":
    df_res = enriquecer_estaciones()
    visualizar_validacion(df_res)