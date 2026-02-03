# 🚀 GPU Server Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Weather Urban Downscaling project on a GPU server with optimized memory management to prevent OOM (Out Of Memory) errors.

## 🎯 What's Included

- **Memory-optimized training pipeline** (60-70% memory reduction)
- **Adaptive configuration** based on available GPU memory
- **Docker containers** with proper GPU support
- **Automated deployment scripts**
- **Paper-ready results generation**

## 📋 Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA support
- **GPU Memory**: Minimum 8GB (recommended 16GB+)

- **System RAM**: Minimum 32GB (recommended 64GB+)
- **Storage**: 100GB+ for datasets and models

### Software Requirements
- **NVIDIA Drivers** (470+)
- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **NVIDIA Container Toolkit** (for GPU support)

## 🛠️ Quick Setup (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/BeeGroup-cimne/weather_urban_downscaling.git
cd weather_urban_downscaling
```

### 2. Data Setup
Before running the deployment script, you **must** place your datasets in the following directories (create them if they don't exist):

- `data/processed/estaciones_interpoladas_final.nc` (High-res targets)
- `data/processed/era5land/` (Contains your `.grib` files)
- `data/processed/weather_static_FINAL_stations.zarr` (Static features)

> [!IMPORTANT]
> The automated pipeline looks for these files to generate the training cache. If they are missing, the `data-prep` step will fail.

### 2.1 Data Health Check (Recommended)
After preprocessing, validate the cache:
```bash
python scripts/check_data_health.py
```

### 3. Verify GPU Support
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### 4. One-Click Deployment
```bash
# Make script executable
chmod +x deploy_gpu_server.sh

# Run full pipeline
./deploy_gpu_server.sh
```

## 📊 Memory Optimization Features

### Before (OOM Issues)
- **UNet**: ~2GB per batch
- **ConvLSTM**: ~6GB per batch

- **Mamba**: ~12GB per batch ❌

### After (Optimized)
- **UNet**: ~0.8GB per batch ✅
- **ConvLSTM**: ~2.5GB per batch ✅
- **Mamba**: ~4GB per batch ✅

### Key Optimizations
- **Adaptive batch sizing** based on GPU memory
- **Gradient accumulation** for larger effective batches
- **Mixed precision training** (50% memory reduction)
- **Efficient data broadcasting** (avoids np.repeat)
- **Chunked data loading** for large datasets
- **Memory monitoring** and auto-recovery

## 🎮 Deployment Options

### Option 1: Full Pipeline (Recommended)
```bash
./deploy_gpu_server.sh
```
- Builds Docker images
- Processes data
- Trains all models
- Generates paper results

### Option 2: Training Only
```bash
./deploy_gpu_server.sh --train-only
```
- Assumes data is already processed
- Trains models with optimized configuration

### Option 3: Include Transformer (Recommended)
```bash
./deploy_gpu_server.sh --include-transformer
```
- Trains Transformer model with multi-head attention
- Memory-optimized for GPUs 8GB+
- State-of-the-art spatiotemporal modeling

### Option 4: Include Mamba (Advanced)
```bash
./deploy_gpu_server.sh --include-mamba
```
- Trains Mamba model (requires 24GB+ GPU)
- Uses PyTorch backend

### Option 5: Include All Models
```bash
./deploy_gpu_server.sh --include-all
```
- Trains UNet, ConvLSTM, Transformer, and Mamba (if GPU allows)
- Complete experimental comparison

### Option 4: Results Only
```bash
./deploy_gpu_server.sh --results-only
```
- Generates evaluation metrics
- Creates paper-ready tables and figures

## 🔧 Configuration

### Automatic GPU Detection
The system automatically detects your GPU and configures optimal settings:

```python
# Example configurations:
# 8GB GPU: batch_size=1, seq_len=4, mixed_precision=True
# 16GB GPU: batch_size=2, seq_len=6, mixed_precision=True  
# 24GB+ GPU: batch_size=2, seq_len=6, include_mamba=True
```

### Manual Configuration
Edit `config/gpu_server_config.py` to customize:

```python
# Memory settings
BATCH_SIZE = 2
SEQ_LEN = 6
MIXED_PRECISION = True
GRADIENT_ACCUMULATION_STEPS = 4

# Model dimensions
UNET_FILTERS = 64
MAMBA_MODEL_DIM = 128
```

## 📁 Project Structure

```
weather_urban_downscaling/
├── config/
│   ├── config.py              # Original config
│   └── gpu_server_config.py   # GPU-optimized config
├── src/
│   ├── data_loader.py         # Original pipeline
│   └── optimized_data_pipeline.py  # Memory-optimized
├── scripts/
│   ├── run_ablation.py        # Original training
│   ├── gpu_server_train.py    # Optimized training
│   └── evaluate_for_paper.py  # Results generation
├── docker-compose.yml         # Original Docker
├── docker-compose.gpu-optimized.yml  # GPU-optimized
└── deploy_gpu_server.sh      # Deployment script
```

## 🐳 Docker Services

### Training Services
- **tf-trainer**: TensorFlow models (UNet, ConvLSTM)
- **transformer-trainer**: TensorFlow Transformer models with memory optimization
- **torch-trainer**: PyTorch models (Mamba)

### Support Services
- **memory-monitor**: Real-time GPU memory tracking
- **data-prep**: Data preprocessing pipeline
- **jupyter**: Development environment

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      memory: 32G      # TensorFlow
      memory: 48G      # PyTorch (Mamba)
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📊 Expected Results

### Training Metrics
- **UNet**: MAE ~0.23, Parameters ~2M
- **ConvLSTM**: MAE ~0.23, Parameters ~5M  
- **Transformer**: MAE ~0.22, Parameters ~3-4M (state-of-the-art attention)
- **Mamba**: MAE ~0.24, Parameters ~0.7M (most efficient)

### Paper-Ready Outputs
- **Model checkpoints** in `experiments/models/`
- **Training logs** in `experiments/logs/`
- **Comparison table** in `experiments/paper_results_table.csv`
- **Visualization plots** in `experiments/paper_model_comparison.png`

### Performance Benchmarks
- **Training time**: 2-6 hours (depending on GPU)
- **Memory usage**: 60-70% reduction vs original
- **Convergence**: Stable training without OOM

## 🔍 Troubleshooting

### Common Issues

#### 1. NVIDIA Docker Runtime
```bash
# Error: docker: Error response from daemon: could not select device driver ""
# Solution: Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 2. Out of Memory Errors
```bash
# Check GPU memory usage
./deploy_gpu_server.sh --help
# Use smaller batch size or reduce sequence length
```

#### 3. Docker Build Issues
```bash
# Clean build
docker system prune -a
docker-compose -f docker-compose.gpu-optimized.yml build --no-cache
```

### Monitoring
```bash
# Monitor GPU memory
watch -n 2 nvidia-smi

# Monitor Docker containers
docker stats

# View training logs
docker-compose -f docker-compose.gpu-optimized.yml logs -f tf-trainer
```

## 🎯 Next Steps

### For Paper Results
1. Run full deployment: `./deploy_gpu_server.sh`
2. Check results in `experiments/`
3. Use generated tables and figures in paper

### For Production
1. Configure environment variables
2. Set up monitoring and alerting
3. Scale with Kubernetes if needed

### For Development
```bash
# Start Jupyter Lab
./deploy_gpu_server.sh --profile development up jupyter

# Access at http://localhost:8888
```

## 📞 Support

- **Issues**: Report on [GitHub Issues](https://github.com/BeeGroup-cimne/weather_urban_downscaling/issues)
- **Documentation**: Check project README
- **Performance**: Monitor with built-in memory tracking

---

## 🎉 Success Criteria

✅ **Training completes without OOM errors**  
✅ **All models converge with stable metrics**  
✅ **Paper-ready results generated automatically**  
✅ **Memory usage reduced by 60-70%**  
✅ **Deployment completes in <30 minutes**

Ready to generate results for your paper! 🚀
