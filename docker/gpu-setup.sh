#!/bin/bash

# Script de setup para Docker en servidor GPU
echo "🔧 Configurando Docker para servidor GPU..."

# 1. Verificar NVIDIA Docker
if ! command -v nvidia-docker &> /dev/null; then
    echo "⚠️ nvidia-docker no encontrado. Instalando NVIDIA Container Toolkit..."
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
fi

# 2. Verificar GPU
echo "🎮 Verificando GPU..."
nvidia-smi

# 3. Crear directorios necesarios
echo "📂 Creando directorios..."
mkdir -p data/processed/era5land
mkdir -p data/raw
mkdir -p experiments/models
mkdir -p experiments/logs
mkdir -p experiments/figures

# 4. Permisos
echo "🔐 Configurando permisos..."
chmod -R 755 data/
chmod -R 755 experiments/

# 5. Variables de entorno
echo "🌍 Configurando variables de entorno..."
export CUDA_VISIBLE_DEVICES=0
export TF_USE_LEGACY_KERAS=1
export PYTHONUNBUFFERED=1

echo "✅ Setup completado. Ahora ejecuta: docker-compose up --build"