import osmnx as ox
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from shapely.geometry import box

# --- 1. CONFIGURACIÓN DEL SERVIDOR OFICIAL ---
# Usamos el servidor principal de Alemania. Es el más fiable.
ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"
ox.settings.use_cache = True
ox.settings.log_console = False
# Timeout alto por si el servidor va lento
ox.settings.requests_timeout = 180 

# --- 2. PARÁMETROS ---
NORTH, SOUTH = 41.5084, 41.2603
EAST, WEST = 2.2949, 1.9698
TAGS = {'building': True} 

# Usamos 4x4. 16 sectores es un equilibrio seguro para el servidor oficial.
CHUNKS = 4

def descargar_infalible():
    print(f"🚀 Iniciando Protocolo Infalible (OSMnx v{ox.__version__})")
    print("📡 Conectando a: overpass-api.de (Alemania)")

    # Grid matemático
    lats = np.linspace(SOUTH, NORTH, CHUNKS + 1)
    lons = np.linspace(WEST, EAST, CHUNKS + 1)
    
    resultados = []
    total = CHUNKS * CHUNKS
    count = 0
    start_time = time.time()

    # --- BUCLE ---
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            count += 1
            
            # Definimos coordenadas (minx, miny, maxx, maxy)
            # Aseguramos el orden correcto para Shapely
            s, n = lats[i], lats[i+1]
            w, e = lons[j], lons[j+1]
            
            # CREAMOS UN POLÍGONO FÍSICO
            # Esto elimina cualquier duda sobre el orden de los parámetros
            polygon_sector = box(w, s, e, n)
            centro = polygon_sector.centroid
            
            print(f"   📍 Sector {count}/{total} (Centro: {centro.y:.2f}, {centro.x:.2f})...", end=" ")

            try:
                # Usamos features_from_polygon que es más robusto que bbox
                gdf = ox.features_from_polygon(polygon_sector, tags=TAGS)
                
                if not gdf.empty:
                    resultados.append(gdf)
                    print(f"✅ RECIBIDO: {len(gdf)} edificios.")
                else:
                    print("⚠️ Vacío (¿Parque/Mar?)")
                
                # Pausa OBLIGATORIA para el servidor oficial
                time.sleep(2) 

            except Exception as e:
                # Si es un error de conexión real, lo veremos
                print(f"❌ Error: {e}")

    # --- UNIFICACIÓN ---
    if resultados:
        print("\n🧩 Ensamblando mapa...")
        gdf_final = pd.concat(resultados)
        
        # Limpieza de IDs duplicados
        gdf_final = gdf_final[~gdf_final.index.duplicated(keep='first')]
        
        minutes = (time.time() - start_time) / 60
        print(f"🎉 ¡CONSEGUIDO! {len(gdf_final)} edificios en {minutes:.1f} minutos.")
        return gdf_final
    else:
        print("\n💀 Fallo: El servidor no devolvió datos. Posible bloqueo de Firewall/VPN.")
        return None

def visualizar(gdf):
    if gdf is None: return
    print("\n🎨 Renderizando visualización...")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='black', alpha=1, markersize=0.5)
    ax.set_axis_off()
    plt.title("Barcelona Extraccion Completa")
    plt.show()

if __name__ == "__main__":
    datos = descargar_infalible()
    
    if datos is not None:
        datos.to_file("barcelona_infalible.geojson", driver='GeoJSON')
        visualizar(datos)