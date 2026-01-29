#!/bin/bash

# Script para probar localmente sin Docker
echo "🚀 Probando proyecto localmente..."

# 1. Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no encontrado. Instala Python 3.8+"
    exit 1
fi

# 2. Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
echo "📚 Instalando dependencias..."
pip install --upgrade pip

# Instalar TensorFlow o PyTorch según prefieras
echo "🤖 Instalando TensorFlow..."
pip install tensorflow==2.15.0 pandas numpy xarray cfgrib eccodes

# O si prefieres PyTorch:
# pip install torch torchvision torchaudio pandas numpy xarray cfgrib eccodes

# 4. Crear directorios
echo "📂 Creando directorios..."
mkdir -p data/processed/era5land
mkdir -p data/raw
mkdir -p experiments/models
mkdir -p experiments/logs
mkdir -p experiments/figures

# 5. Ejecutar prueba
echo "🧪 Ejecutando prueba..."
python scripts/run_ablation.py

echo "✅ Prueba local completada"