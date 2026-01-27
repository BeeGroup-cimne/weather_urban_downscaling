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

# Copy project code
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY train.py .

# Environment variables
ENV TF_USE_LEGACY_KERAS=1
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/data/processed/era5land \
    /app/data/raw \
    /app/experiments/models \
    /app/experiments/logs \
    /app/experiments/figures

# Default command
CMD ["python", "scripts/run_ablation.py"]
