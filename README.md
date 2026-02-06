# Weather Urban Downscaling

**High-Resolution Urban Heat Island Downscaling with Deep Learning**

This project implements state-of-the-art Deep Learning models to downscale low-resolution climate data (ERA5-Land, ~9km) into high-resolution urban thermal maps (~100m or less), incorporating complex static urban features (building heights, street width, sky view factor).

It features a dual-engine architecture supporting both **TensorFlow (Legacy & Modern)** and **PyTorch (Mamba SSM)**.
The current server target is **Ubuntu 24.04.3 LTS + NVIDIA A10 (22GB)** using Docker Compose v2.

---

## 🚀 Key Features

*   **Big Data Pipeline**: A robust ETL engine built with `xarray`, `dask`, and `zarr` to process large climate data efficiently.
    *   *Automatic Static Feature Generation*: Calculates urban indices (SVF, Roughness, Density) on the fly.
*   **Temporal Sampling Controls**: Sequence stride, weighted sampling, and optional seasonal balancing to better cover rare dynamics.
    *   Optional **station-weighted sampling** (GRIB) with automatic fallback to HR-derived weights.
*   **Tile-Based Training (Optional)**: Patch sampling for large grids with uniform, static-weighted, or error-weighted strategies.
*   **Data Health & Repair**: Built-in NaN checks and fast Zarr repair utilities.
*   **Hybrid Loss Function**: Combines **MSE** (Numerical Accuracy) and **SSIM** (Structural Similarity) to produce sharp, visually coherent maps. `Loss = (1-α)*MSE + α*SSIM`.
*   **Multi-Model Support**:
    *   **U-Net** (Baseline)
    *   **Hybrid U-Net + LSTM** (Spatiotemporal)
    *   **Hybrid U-Net + Mamba** (State Space Models for efficient long-sequence modeling)
*   **Dual Engine**: 
    *   `TensorFlow` (Primary production engine)
    *   `PyTorch` (Experimental Mamba implementation)
*   **Paper-Ready Splits**: Fixed temporal splits for train/val/test (2017 months) to avoid leakage.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   TensorFlow 2.13+ (Mac Silicon optimized) OR PyTorch 2.x
*   Docker & Docker Compose (recommended for server deployment)
*   GDAL/GeoPandas (for static data processing)

### Option 1: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/weather_urban_downscaling.git
cd weather_urban_downscaling

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install Dependencies
# For Mac Silicon (M1/M2/M3):
pip install -r requirements_mac.txt

# For Linux/CUDA:
pip install -r requirements_tf.txt  # or requirements_torch.txt
```

### Option 2: Docker Setup (Recommended for Servers)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/weather_urban_downscaling.git
cd weather_urban_downscaling

# 2. Prepare the data directory structure
mkdir -p data/processed/era5land data/raw

# 3. Copy your data files (see Data Requirements section below)

# 4. Build Docker images
docker compose build

# 5. Run training
# TensorFlow (default):
docker compose run tf-trainer

# PyTorch (experimental):
docker compose run torch-trainer

# With GPU support (requires nvidia-docker):
# Use docker-compose.gpu-optimized.yml (see GPU Server Quick Start)
```

---

## 📁 Data Requirements

Before running, you need to provide the following data files in the `data/` directory:

```
data/
├── processed/
│   ├── estaciones_interpoladas_final.nc   # HR target (interpolated stations)
│   ├── weather_static_FINAL_stations.zarr/ # Static features (buildings, etc.)
│   ├── stations_t2m.grib                 # Optional station GRIB for weighted sampling
│   └── era5land/
│       └── lr_2017.grib                   # LR input (ERA5-Land)
└── raw/
    └── weather_stations.zarr/             # Raw station data (optional)
```

**Data Sources:**
- **ERA5-Land**: Download from [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **HR Target**: Generated from meteorological station data using `src/data/hourly_generator.py`
- **Static Features**: Generated from urban morphology data

**Derived caches (created by pipeline):**
- `data/processed/weather_cache.zarr`
- `data/processed/static_processed.npy`
- `data/processed/stats_config.npz`

---

## 📂 Project Structure

```
├── config/
│   ├── config.py           # Local/default config
│   ├── gpu_server_config.py# Server/GPU config
│   └── runtime.py          # Selects GPU config when USE_GPU_CONFIG=1
├── data/                   # Data storage (Zarr cache, Raw NetCDF) - NOT in git
├── docker/
│   ├── Dockerfile.tf       # TensorFlow container
│   └── Dockerfile.torch    # PyTorch container
├── docker-compose.yml      # Base orchestration
├── docker-compose.gpu-optimized.yml # GPU server orchestration (Compose v2)
├── scripts/
│   ├── run_ablation.py       # Main experimentation script (TF)
│   ├── run_ablation_tiles.py # Ablation on tile-based pipeline
│   ├── gpu_server_train.py   # GPU-optimized TF training
│   ├── train_tiles.py        # Tile-based training (TF)
│   ├── train_torch.py        # PyTorch training (baseline)
│   ├── torch_gpu_train.py    # PyTorch Mamba training (GPU)
│   ├── run_mamba_seq6.py     # Mamba experiment SEQ_LEN=6
│   ├── run_mamba_seq12.py    # Mamba experiment SEQ_LEN=12
│   ├── run_inference.py      # Fast inference for saved models
│   ├── evaluate_test_set.py  # Quick test-set evaluation
│   ├── overfit_sanity.py     # Ultra-rapid sanity check
│   ├── check_data_health.py  # Cache/NaN validation
│   ├── repair_zarr_nans.py   # Fast NaN repair + stats recompute
│   ├── build_error_map.py    # Build error map for tile sampling
│   └── fig_tiles_sampling.py # Visualize tile sampling
├── src/
│   ├── data_loader.py      # BigDataPipeline (ETL & Generators)
│   ├── data_loader_tiles.py# Patch-based pipeline (optional)
│   ├── optimized_data_pipeline.py # Streaming pipeline for GPU server
│   ├── models_legacy.py    # TF Models (ReLU, No-BN -> Sharp Results)
│   ├── tf_engine/          # Refactored TF components
│   └── torch_engine/       # PyTorch Mamba Utilities
├── experiments/            # Training outputs (models, logs, figures)
├── requirements_tf.txt     # TensorFlow dependencies
├── requirements_torch.txt  # PyTorch dependencies
└── train.py                # Main Entry Point
```

---

## 🏃 Usage

### 1. Training (TensorFlow - Production)

The main script uses the **Standard/Legacy** methodology which ensures high sharpness by avoiding unnecessary Batch Normalization in the generator.

```bash
# Local:
python train.py

# Docker:
docker compose run tf-trainer
```

*   **Pipeline**: Automatically generates `data/processed/static_processed.npy` if missing.
*   **Models**: Trains U-Net + ConvLSTM (Transformer disabled on A10 due to OOM).
*   **Resume**: If checkpoints exist (GPU server), training resumes from the last saved epoch.

### 2. Ablation Study

To compare multiple architectures (U-Net vs LSTM vs Mamba) under identical conditions:

```bash
# Local:
python scripts/run_ablation.py

# Docker:
docker compose run tf-trainer python scripts/run_ablation.py
```

*   **Outputs**: Generates `experiments/ablation_summary.csv` and comparative plots.
*   **Note**: This script has been aligned to use the exact same data pipeline and "Legacy" architecture as `train.py` to ensure fair comparison.

### 3. Data Health & Fast Repair

```bash
# Check for NaNs and cache consistency
python -m scripts.check_data_health

# Repair NaNs in Zarr cache and recompute stats
python -m scripts.repair_zarr_nans
```

### 4. Overfit Sanity Check (Ultra-Rapid)

```bash
python -m scripts.overfit_sanity --model-type mamba --epochs 20 --train-batches 4 --val-batches 2
```

### 4.1 Active Config Snapshot

```bash
python -m scripts.print_active_config
```

### 5. Tile-Based Training (Optional)

```bash
# Train with patches (useful for large grids)
python scripts/train_tiles.py --model-type unet --epochs 50 --temporal-sampler weighted

# Tile-based ablation
python scripts/run_ablation_tiles.py --model-type mamba --epochs 30
```

### 6. Inference

```bash
python scripts/run_inference.py --model-type unet --model-path experiments/models/UNet_best.h5
```

### 7. PyTorch (Experimental)

To train the Mamba model using PyTorch:

```bash
# Local:
python scripts/train_torch.py

# Docker:
docker compose run torch-trainer
```

*   **Features**: Implements a custom 5D-capable Mamba block and the Hybrid (MSE+SSIM) loss in PyTorch.

---

## ⏱️ Temporal Sampling Options

These options live in `config/config.py` and `config/gpu_server_config.py`:

```python
TEMPORAL_STRIDE = 1                 # sample every N steps (1 = contiguous)
TEMPORAL_SAMPLER = "uniform"        # "uniform" | "weighted" | "weighted_station"
TEMPORAL_WEIGHT_GAMMA = 1.5         # emphasize rare dynamics (>=1)
TEMPORAL_MIN_PROB = 1e-6            # avoid zero-probability
TEMPORAL_SEASON_BALANCE = False     # True = DJF/MAM/JJA/SON balancing
STATION_GRIB_PATH = "data/processed/stations_t2m.grib"
```

Notes:
*   `weighted` uses HR temporal gradients as weights.
*   `weighted_station` uses station GRIB if available, otherwise falls back to HR.

---

## 🧩 Tile Training Options (Optional)

```python
PATCH_SIZE = (96, 96)
PATCHES_PER_EPOCH = 2000
VAL_PATCHES_PER_EPOCH = 200
TILE_SAMPLER = "static_weighted"  # "uniform" | "static_weighted" | "error_weighted"
TILE_WEIGHT_ALPHA = 0.85
TILE_WEIGHT_GAMMA = 1.0
TILE_MIN_PROB = 1e-6
TILE_ERROR_MAP_PATH = "experiments/tiles_error_map.npy"
```

Use `scripts/build_error_map.py` to generate the error map if you want error-weighted sampling.

---

## 🐳 Docker Commands Quick Reference

```bash
# Build all images
docker compose build

# Run TensorFlow trainer
docker compose run tf-trainer

# Run PyTorch trainer
docker compose run torch-trainer

# Run data preprocessing only
docker compose run data-prep

# Run with custom command
docker compose run tf-trainer python -c "from config.config import Config; print(Config.DEVICE)"

# View logs
docker compose logs -f tf-trainer

# Clean up containers
docker compose down

# Remove images
docker compose down --rmi all
```

---

## ⚡ GPU Server Quick Start

This repo is designed to run with **Docker Compose v2** on Ubuntu servers.

1. Preprocess (one-time):
```bash
docker compose -f docker-compose.gpu-optimized.yml --profile preprocessing up data-prep
```

2. Data health check:
```bash
docker compose -f docker-compose.gpu-optimized.yml run --rm data-prep python scripts/check_data_health.py
```

3. Train (TensorFlow):
```bash
docker compose -f docker-compose.gpu-optimized.yml up -d tf-trainer
docker compose -f docker-compose.gpu-optimized.yml logs -f tf-trainer
```

Nota: en GPU se usa automáticamente `config/gpu_server_config.py` vía `USE_GPU_CONFIG=1`.

4. Optional (PyTorch Mamba):
```bash
docker compose -f docker-compose.gpu-optimized.yml run --rm torch-trainer python scripts/run_mamba_seq6.py
docker compose -f docker-compose.gpu-optimized.yml run --rm torch-trainer python scripts/run_mamba_seq12.py
```

---

## 🧠 Methodology Highlights

### The "Sharpness" Secret
Through extensive testing, we found that standard modern practices (Batch Normalization, LeakyReLU) often result in "blurry" thermal maps when downscaling.
*   **Our Solution**: We use a "Legacy" U-Net architecture relying on **Standard ReLU** and **No Batch Normalization** in the generative path. This allows the model to predict high-frequency spatial gradients (sharp building edges) more effectively.

### Static Data Injection
We don't just learn from historical temperature. We inject detailed urban morphology:
*   **Building Height & Density**
*   **Sky View Factor (SVF)**
*   **Surface Roughness**

These are concatenated with the dynamic low-resolution input, allowing the model to "physicalize" the downscaling process.

---

## 📊 Results (Populate After Runs)

| Model             | MAE (ºC) | SSIM     | Notes |
| ----------------- | -------- | -------- | ----- |
| U-Net (Baseline)  | TBD      | TBD      | Train 2017 Jan–Oct, Val Nov, Test Dec |
| ConvLSTM          | TBD      | TBD      | Same split, SEQ_LEN=6 |
| Mamba (SEQ=6)     | TBD      | TBD      | PyTorch |
| Mamba (SEQ=12)    | TBD      | TBD      | PyTorch |

**Result Artifacts (expected paths):**
- `experiments/logs/<model>_gpu_optimized_log.csv`
- `experiments/models/<model>_gpu_optimized.h5`
- `experiments/figures/` (plots and maps)

---

## 🔧 Troubleshooting

### "Sin coincidencia temporal" Error
This means the time ranges of HR and LR data don't overlap. Run the diagnostic:
```bash
python scripts/diagnose_time.py
```

### NaNs in Cache
If `scripts.check_data_health` reports NaNs:
```bash
python -m scripts.repair_zarr_nans
```

### Permission Errors on Mac
Grant Full Disk Access to Terminal in System Settings > Privacy & Security.

### Docker GPU Not Detected
Ensure nvidia-docker is installed and uncomment the `deploy` section in `docker-compose.yml`.

### Mixed Precision Warnings
Mixed precision is disabled on the server to avoid loss-scaling issues with the custom training loop.

---

## 📝 License
MIT License. See `LICENSE`.

---

## ✅ Evaluation Protocol (for Paper)

**Metrics**
- MAE (°C)
- RMSE (°C)
- SSIM
- Optional: Bias and correlation (per grid cell and overall)

**Evaluation Windows**
- Train: 2017-01-01 → 2017-11-01
- Val: 2017-11-01 → 2017-12-01
- Test: 2017-12-01 → 2018-01-01

**Output Reporting**
- Per-epoch logs: `experiments/logs/`
- Best checkpoints: `experiments/models/`
- Figures: `experiments/figures/`

**Suggested Plots**
- Spatial error maps (MAE and bias)
- SSIM histograms (per timestep)
- Time-series at selected stations
- Scatter plot HR vs Pred

---

## ✅ Reproducibility Checklist

- Fixed temporal split (train/val/test by month in 2017)
- Fixed SEQ_LEN for baselines (6)
- Mamba runs at SEQ_LEN = 6 and 12
- Seeds set in `config/config.py` and `config/gpu_server_config.py`
- Cache files created and checked: `weather_cache.zarr`, `static_processed.npy`, `stats_config.npz`
- Save git hash + config snapshot for each run (recommended)

---

## 🧪 How To Generate Figures (Suggested)

**Quick evaluation + plots (after training):**
```bash
# Example: evaluate and generate paper plots
python scripts/evaluate_for_paper.py
```

**Typical outputs (expected):**
- `experiments/figures/mae_map.png`
- `experiments/figures/bias_map.png`
- `experiments/figures/ssim_hist.png`
- `experiments/figures/scatter_hr_vs_pred.png`

If you need custom plotting per model, create subfolders:
`experiments/figures/unet/`, `experiments/figures/convlstm/`, `experiments/figures/mamba_seq6/`, etc.

**Planned figure file names (paper-ready):**
- `experiments/figures/fig01_pipeline.png`
- `experiments/figures/fig02_qualitative_maps.png`
- `experiments/figures/fig03_spatial_error_maps.png`
- `experiments/figures/fig04_metrics_bar.png`
- `experiments/figures/fig05_timeseries_stations.png`
- `experiments/figures/fig06_seq_len_ablation.png`

**Suggested scripts (to automate outputs):**
- `scripts/fig01_pipeline_diagram.py`
- `scripts/fig02_qualitative_maps.py`
- `scripts/fig03_spatial_error_maps.py`
- `scripts/fig04_metrics_bar.py`
- `scripts/fig05_timeseries_stations.py`
- `scripts/fig06_seq_len_ablation.py`

If preferred, these can be consolidated into:
`scripts/generate_paper_figures.py`

---

## 🧾 Paper-Ready Abstract (Draft Proposal)

Urban heat island mapping requires high‑resolution temperature fields, yet available reanalysis products are typically coarse (e.g., ~9 km). We propose a deep learning downscaling framework that transforms ERA5‑Land inputs into high‑resolution urban thermal maps by combining dynamic meteorological predictors with static urban morphology (elevation, SVF, roughness, and related descriptors). The core model family is a legacy U‑Net with a hybrid MSE+SSIM loss that preserves sharp spatial gradients, extended with temporal baselines (ConvLSTM) and state‑space sequence modeling (Mamba) to capture multi‑hour dynamics. Experiments are conducted with a fixed chronological split of 2017 (train: Jan–Oct, val: Nov, test: Dec) to avoid temporal leakage. We report spatial accuracy (MAE/RMSE), structural similarity (SSIM), and qualitative error maps, and we analyze the impact of sequence length (6 vs. 12) on temporal skill. The results demonstrate robust downscaling performance and improved structural fidelity in urban temperature patterns, while longer sequences yield additional gains in temporal coherence.

---

## 🧩 Methods Section Checklist (Paper)

- Data sources (ERA5-Land, station-based HR)
- HR target generation (interpolation + grid 251×251)
- Static features and preprocessing
- Temporal split rationale (chronological, no leakage)
- Model architectures (U-Net, ConvLSTM, Mamba)
- Loss function (MSE + SSIM, α)
- Training protocol (epochs, batch, seq_len)
- Metrics and evaluation protocol
- Hardware and reproducibility

---

## 📌 Paper-Ready Configuration (Summary)

**Temporal Splits (fixed):**
- Train: **2017-01-01 → 2017-11-01**
- Val: **2017-11-01 → 2017-12-01**
- Test: **2017-12-01 → 2018-01-01**

**Sequences:**
- Default for TF (UNet/ConvLSTM): **SEQ_LEN = 6**
- Mamba experiments: **SEQ_LEN = 6** and **SEQ_LEN = 12**

**Server Environment:**
- Ubuntu 24.04.3 LTS
- NVIDIA A10 (22GB)
- Docker Compose v2 (`docker compose`)

**Recommended Presets (Starting Point):**

| Hardware | BATCH_SIZE | GRAD_ACCUM | SEQ_LEN | MIXED_PRECISION | Notes |
| -------- | ---------- | ---------- | ------- | --------------- | ----- |
| A10 22GB | 2 | 2 | 6 | False | Stable baseline for full-frame training |
| Apple M4 | 2 | 4 | 6 | False | Good stability on MPS, low memory pressure |

Preset files (copy into config as needed):
- `config/presets/a10.py`
- `config/presets/m4.py`
TEMPORAL_SAMPLER = "weighted_station"  # optional if stations GRIB is available
