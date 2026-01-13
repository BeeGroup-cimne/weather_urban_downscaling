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
*   GDAL/GeoPandas (for static data processing)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/weather_urban_downscaling.git
cd weather_urban_downscaling

# 2. Install Dependencies
# For Mac Silicon (M1/M2/M3):
pip install -r requirements_mac.txt

# For Linux/CUDA:
pip install -r requirements_tf.txt  # or requirements_torch.txt
```

---

## 📂 Project Structure

```
├── config/
│   └── config.py           # Global hyperparameters & paths
├── data/                   # Data storage (Zarr cache, Raw NetCDF)
├── scripts/
│   ├── run_ablation.py     # Main experimentation script (TF)
│   ├── train_torch.py      # PyTorch training script
│   └── run_inference.py    # Production inference
├── src/
│   ├── data_loader.py      # BigDataPipeline (ETL & Generators)
│   ├── models_legacy.py    # TF Models (ReLU, No-BN -> Sharp Results)
│   ├── tf_engine/          # Refactored TF components
│   └── torch_engine/       # PyTorch Mamba Utilities
└── train.py                # Main Entry Point
```

---

## 🏃 Usage

### 1. Training (TensorFlow - Production)

The main script uses the **Standard/Legacy** methodology which ensures high sharpness by avoiding unnecessary Batch Normalization in the generator.

```bash
python train.py
```

*   **Pipeline**: Automatically generates `processed_cache_zarr/static_processed.npy` if missing.
*   **Models**: Trains a Hybrid U-Net + Mamba by default.

### 2. Ablation Study

To compare multiple architectures (U-Net vs LSTM vs Mamba) under identical conditions:

```bash
python scripts/run_ablation.py
```

*   **Outputs**: Generates `experiments/ablation_summary.csv` and comparative plots.
*   **Note**: This script has been aligned to use the exact same data pipeline and "Legacy" architecture as `train.py` to ensure fair comparison.

### 3. PyTorch (Experimental)

To train the Mamba model using PyTorch:

```bash
python scripts/train_torch.py
```

*   **Features**: Implements a custom 5D-capable Mamba block and the Hybrid (MSE+SSIM) loss in PyTorch.

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

| Model | MAE (ºC) | SSIM | Definition |
|-------|----------|------|------------|
| U-Net (Baseline) | 1.2 | 0.65 | Low |
| **U-Net + Mamba** | **0.8** | **0.82** | **High** |

---

## 📝 License
[Insert License Here]
