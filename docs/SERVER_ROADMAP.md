# Server Roadmap — Full-frame + `scripts/run_ablation.py`

## Goal

Prepare the repository for server deployment (ideally GPU), **excluding**:
- images (figures, diagrams, PNG/JPG, etc.)
- trained models (weights `.h5`, checkpoints, etc.)
- prior `experiments/` outputs

...while **preserving folder structure** so the server can generate new outputs at runtime.

The only definitive execution entrypoint in this roadmap is:
- `scripts/run_ablation.py` (4 models: `unet`, `lstm`, `transformer`, `mamba`)

## Repository structure (server usage)

- `config/`: runtime configuration (`config/runtime.py`) + defaults (`config/config.py`) + GPU profile (`config/gpu_server_config.py`)
- `src/`: full-frame pipeline (`src/data_loader.py`), legacy TF models (`src/models_legacy.py`), training utilities (`src/utils/training.py`)
- `scripts/`: entrypoints and utilities; in this server flow only `scripts/run_ablation.py` is required
- `docker/`: `docker/Dockerfile.tf` builds the TF training image
- `data/`: **mounted on server** (datasets/caches outside bundle)
- `experiments/`: **mounted on server** (logs/models/figures generated at runtime)

## Recommended roadmap (efficient order)

1) Freeze production entrypoint  
   - Use only `scripts/run_ablation.py` and run the 4 models in sequence.
   - Avoid alternative paths (tiles, paper scripts) in this server flow.

2) Full-frame (no tiles) + full epochs  
   - In this repository, full-frame means using `BigDataPipeline` (not `TileDataPipeline`).
   - To avoid capped runs, enforce `FULLFRAME=1` on server.

3) Separate code vs data/results  
   - Manage `data/` and `experiments/` as server volumes.
   - The uploaded repository should not include datasets or historical outputs.

4) Clean packaging for server upload  
   - Build a bundle without images/models/previous experiments:
     - `scripts/make_server_bundle.sh`
   - Alternative: `git clone` on server (with `.gitignore` / `.dockerignore` barriers).

5) Server execution (Docker Compose)  
   - Use `docker-compose.server-fullframe.yml` to run full-frame ablation with GPU config and figures disabled by default.

## Commands (server-ready)

### 1) Create upload bundle (without images/models/experiments)
```bash
./scripts/make_server_bundle.sh
```
Default output:
- `dist/weather_urban_downscaling_server_bundle.tar.gz`

### 2) On server: extract and mount data
Minimum expected structure (actual files are outside the bundle):
- `data/processed/estaciones_interpoladas_final.nc`
- `data/processed/era5land/*.grib`
- `data/processed/weather_static_FINAL_stations.zarr/`

### 3) Run full-frame ablation (4 models)
```bash
docker compose -f docker-compose.server-fullframe.yml up --build
```

By default, compose sets:
- `FULLFRAME=1`
- `USE_GPU_CONFIG=1`
- `SAVE_MODEL_DIAGRAM=0`, `SAVE_VISUALIZATIONS=0`, `SAVE_COMPARATIVE_HISTORY=0`

If you need figures on server, set those variables to `1`.

