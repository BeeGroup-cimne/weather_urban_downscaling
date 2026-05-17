FROM tensorflow/tensorflow:2.15.0-gpu AS runtime

# Metadata
LABEL org.opencontainers.image.title="weather-urban-downscaling"
LABEL org.opencontainers.image.description="TensorFlow GPU runtime for reproducible weather urban downscaling jobs"
LABEL org.opencontainers.image.source="https://github.com/BeeGroup-cimne/weather_urban_downscaling"

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    libeccodes-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# IMPORTANT: The base image ships TF 2.15.0 + numpy 1.26.2 which are
# ABI-compatible. We MUST pin both so pip doesn't upgrade TF to 2.19+
# (compiled against numpy 2.x ABI) or numpy to 2.x.
COPY requirements_tf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir 'numpy>=1.24,<2.0' 'tensorflow>=2.15.0,<2.16' && \
    pip install --no-cache-dir -r requirements_tf.txt

# Copy project code (only what's needed for run_ablation.py)
COPY config/ ./config/
COPY src/__init__.py ./src/
COPY src/data_loader.py ./src/
COPY src/data_loader_tiles.py ./src/
COPY src/models_legacy.py ./src/
COPY src/models/ ./src/models/
COPY src/losses.py ./src/
COPY src/data/ ./src/data/
COPY src/utils/__init__.py ./src/utils/
COPY src/utils/training.py ./src/utils/

# Scripts needed for server training, evaluation, and figures
COPY scripts/ ./scripts/

# Environment variables
ENV TF_USE_LEGACY_KERAS=1
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# Create necessary directories (data & experiments mounted as volumes)
RUN mkdir -p /app/data/processed/era5land \
    /app/data/raw \
    /app/data/static \
    /app/experiments/models \
    /app/experiments/logs \
    /app/experiments/figures \
    /tmp/matplotlib

HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD python scripts/ops/healthcheck.py --runtime

# Default command
CMD ["python", "scripts/ablation/run_ablation.py", "--models", "unet", "lstm", "transformer", "mamba", "--min-seq-len", "6"]
