FROM tensorflow/tensorflow:2.15.0-gpu

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="TensorFlow environment for Weather Urban Downscaling"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    libeccodes-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements_tf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_tf.txt

# Copy project code (only what's needed for run_ablation.py)
COPY config/ ./config/
COPY src/__init__.py ./src/
COPY src/data_loader.py ./src/
COPY src/models_legacy.py ./src/
COPY src/losses.py ./src/
COPY src/data/ ./src/data/
COPY src/utils/__init__.py ./src/utils/
COPY src/utils/training.py ./src/utils/

# Only the scripts needed for server training
COPY scripts/run_ablation.py ./scripts/
COPY scripts/run_server_fullframe.sh ./scripts/
COPY scripts/print_active_config.py ./scripts/

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
    /app/experiments/figures

# Default command
CMD ["python", "scripts/run_ablation.py", "--models", "unet", "lstm", "transformer", "mamba", "--min-seq-len", "6"]
