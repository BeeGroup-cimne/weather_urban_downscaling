# Weather Urban Downscaling 🌍🌡️

**High-Resolution Urban Heat Island Transformation using Deep Learning**

This project implements state-of-the-art Deep Learning models to downscale low-resolution climate data (ERA5-Land, ~9km) into high-resolution urban thermal maps (~100m or less), incorporating complex static urban features (building heights, street width, sky view factor).

It features a dual-engine architecture supporting both **TensorFlow (Legacy & Modern)** and **PyTorch (Mamba SSM)**.

---

## 🚀 Key Features

*   **Big Data Pipeline**: A robust ETL engine built with `xarray`, `dask`, and `zarr` to process terabytes of climate data efficiently.
    *   *Automatic Static Feature Generation*: Calculates urban indices (SVF, Roughness, Density) on the fly.
*   **Hybrid Loss Function**: Combines **MSE** (Numerical Accuracy) and **SSIM** (Structural Similarity) to produce sharp, visually coherent maps. `Loss = (1-α)*MSE + α*SSIM`.
*   **Multi-Model Support**:
    *   **U-Net** (Baseline)
    *   **Hybrid U-Net + LSTM** (Spatiotemporal)
    *   **Hybrid U-Net + Mamba** (State Space Models for efficient long-sequence modeling)
*   **Dual Engine**: 
    *   `TensorFlow` (Primary production engine)
    *   `PyTorch` (Experimental Mamba implementation)

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

### Option 2: Docker Setup (Recommended for Servers) 🐳

```bash
# 1. Clone the repository
git clone https://github.com/your-org/weather_urban_downscaling.git
cd weather_urban_downscaling

# 2. Prepare the data directory structure
mkdir -p data/processed/era5land data/raw

# 3. Copy your data files (see Data Requirements section below)

# 4. Build Docker images
docker-compose build

# 5. Run training
# TensorFlow (default):
docker-compose run tf-trainer

# PyTorch (experimental):
docker-compose run torch-trainer

# With GPU support (requires nvidia-docker):
# Uncomment the 'deploy' section in docker-compose.yml first
docker-compose run tf-trainer
```

---

## 📁 Data Requirements

Before running, you need to provide the following data files in the `data/` directory:

```
data/
├── processed/
│   ├── estaciones_interpoladas_final.nc   # HR target (interpolated stations)
│   ├── weather_static_FINAL_stations.zarr/ # Static features (buildings, etc.)
│   └── era5land/
│       └── lr_2017.grib                   # LR input (ERA5-Land)
└── raw/
    └── weather_stations.zarr/             # Raw station data (optional)
```

**Data Sources:**
- **ERA5-Land**: Download from [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **HR Target**: Generated from meteorological station data using `src/data/hourly_generator.py`
- **Static Features**: Generated from urban morphology data

---

## 📂 Project Structure

```
├── config/
│   └── config.py           # Global hyperparameters & paths (auto-detects hardware)
├── data/                   # Data storage (Zarr cache, Raw NetCDF) - NOT in git
├── docker/
│   ├── Dockerfile.tf       # TensorFlow container
│   └── Dockerfile.torch    # PyTorch container
├── docker-compose.yml      # Container orchestration
├── scripts/
│   ├── run_ablation.py     # Main experimentation script (TF)
│   ├── train_torch.py      # PyTorch training script
│   └── diagnose_time.py    # Data diagnostic utility
├── src/
│   ├── data_loader.py      # BigDataPipeline (ETL & Generators)
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
docker-compose run tf-trainer
```

*   **Pipeline**: Automatically generates `data/processed/static_processed.npy` if missing.
*   **Models**: Trains a Hybrid U-Net + Mamba by default.

### 2. Ablation Study

To compare multiple architectures (U-Net vs LSTM vs Mamba) under identical conditions:

```bash
# Local:
python scripts/run_ablation.py

# Docker:
docker-compose run tf-trainer python scripts/run_ablation.py
```

*   **Outputs**: Generates `experiments/ablation_summary.csv` and comparative plots.
*   **Note**: This script has been aligned to use the exact same data pipeline and "Legacy" architecture as `train.py` to ensure fair comparison.

### 3. PyTorch (Experimental)

To train the Mamba model using PyTorch:

```bash
# Local:
python scripts/train_torch.py

# Docker:
docker-compose run torch-trainer
```

*   **Features**: Implements a custom 5D-capable Mamba block and the Hybrid (MSE+SSIM) loss in PyTorch.

---

## 🐳 Docker Commands Quick Reference

```bash
# Build all images
docker-compose build

# Run TensorFlow trainer
docker-compose run tf-trainer

# Run PyTorch trainer
docker-compose run torch-trainer

# Run data preprocessing only
docker-compose run data-prep

# Run with custom command
docker-compose run tf-trainer python -c "from config.config import Config; print(Config.DEVICE)"

# View logs
docker-compose logs -f tf-trainer

# Clean up containers
docker-compose down

# Remove images
docker-compose down --rmi all
```

---

## ⚡ GPU Server Quick Start

1. Preprocess (one-time):
```bash
docker-compose -f docker-compose.gpu-optimized.yml --profile preprocessing up data-prep
```

2. Data health check:
```bash
python scripts/check_data_health.py
```

3. Train (TensorFlow):
```bash
docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up tf-trainer
```

Nota: en GPU se usa automáticamente `config/gpu_server_config.py` vía `USE_GPU_CONFIG=1`.

4. Optional (PyTorch Mamba):
```bash
docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up torch-trainer
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

## 📊 Results

| Model             | MAE (ºC) | SSIM     | Definition |
| ----------------- | -------- | -------- | ---------- |
| U-Net (Baseline)  | 1.2      | 0.65     | Low        |
| **U-Net + Mamba** | **0.8**  | **0.82** | **High**   |

---

## 🔧 Troubleshooting

### "Sin coincidencia temporal" Error
This means the time ranges of HR and LR data don't overlap. Run the diagnostic:
```bash
python scripts/diagnose_time.py
```

### Permission Errors on Mac
Grant Full Disk Access to Terminal in System Settings > Privacy & Security.

### Docker GPU Not Detected
Ensure nvidia-docker is installed and uncomment the `deploy` section in `docker-compose.yml`.

---

## 📝 License
[Insert License Here]
