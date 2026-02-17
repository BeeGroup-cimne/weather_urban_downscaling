# Hoja de ruta (Servidor) — Full-frame + `scripts/run_ablation.py`

## Objetivo

Preparar el repositorio para subirlo a un servidor (idealmente con GPU), **sin incluir**:
- Imágenes (figuras, diagramas, PNG/JPG, etc.)
- Modelos entrenados (pesos `.h5`, checkpoints, etc.)
- Outputs de `experiments/` (resultados previos)

…pero **conservando la estructura** de carpetas para que el servidor pueda generar nuevos outputs al ejecutar.

El **único entrypoint “definitivo”** de ejecución es:
- `scripts/run_ablation.py` (4 modelos: `unet`, `lstm`, `transformer`, `mamba`)

## Estructura del repo (cómo se usa en servidor)

- `config/`: configuración runtime (`config/runtime.py`) + defaults (`config/config.py`) + perfil GPU (`config/gpu_server_config.py`)
- `src/`: pipeline full-frame (`src/data_loader.py`), modelos TF legacy (`src/models_legacy.py`), utilidades de entrenamiento (`src/utils/training.py`)
- `scripts/`: entrypoints y utilidades; en servidor solo se necesita `scripts/run_ablation.py`
- `docker/`: `docker/Dockerfile.tf` construye la imagen de entrenamiento TF
- `data/`: **se monta en el servidor** (datasets/caches fuera del bundle)
- `experiments/`: **se monta en el servidor** (logs/modelos/figuras generadas en runtime)

## Roadmap recomendado (orden eficiente)

1) Congelar el entrypoint de producción  
   - Usar solo `scripts/run_ablation.py` y ejecutar los 4 modelos en orden.
   - Evitar rutas alternativas (tiles, scripts de paper) en el flujo del servidor.

2) Full-frame (sin tiles) + epochs completos  
   - Full-frame en este repo significa usar `BigDataPipeline` (no `TileDataPipeline`).
   - Para evitar runs “capados”, forzar `FULLFRAME=1` en servidor.

3) Separar “código” vs “datos/resultados”  
   - `data/` y `experiments/` se gestionan como volúmenes en el servidor.
   - El repositorio que se sube no debe contener datasets ni outputs históricos.

4) Empaquetado limpio para subir al servidor  
   - Generar un bundle sin imágenes/modelos/experimentos previos:
     - `scripts/make_server_bundle.sh`
   - Alternativa: `git clone` en servidor (y mantener `.gitignore` / `.dockerignore` como barrera).

5) Ejecución en servidor (Docker Compose)  
   - Usar `docker-compose.server-fullframe.yml` para correr el ablation full-frame con GPU config y sin generar figuras por defecto.

## Comandos (server-ready)

### 1) Crear bundle para subir (sin imágenes/modelos/experiments)
```bash
./scripts/make_server_bundle.sh
```
Salida por defecto:
- `dist/weather_urban_downscaling_server_bundle.tar.gz`

### 2) En el servidor: descomprimir y montar datos
Estructura mínima esperada (archivos reales van por fuera del bundle):
- `data/processed/estaciones_interpoladas_final.nc`
- `data/processed/era5land/*.grib`
- `data/processed/weather_static_FINAL_stations.zarr/`

### 3) Ejecutar ablation full-frame (4 modelos)
```bash
docker compose -f docker-compose.server-fullframe.yml up --build
```

Por defecto, el compose setea:
- `FULLFRAME=1`
- `USE_GPU_CONFIG=1`
- `SAVE_MODEL_DIAGRAM=0`, `SAVE_VISUALIZATIONS=0`, `SAVE_COMPARATIVE_HISTORY=0`

Si quieres figuras en servidor, setea a `1` esas variables.

