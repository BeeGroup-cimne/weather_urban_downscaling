#!/bin/bash
set -e

echo "=========================================================="
echo "🐳 Reconstruyendo la imagen de Docker (TF)"
echo "=========================================================="
docker build -f docker/Dockerfile.tf -t weather_thesis:tf .

echo ""
echo "=========================================================="
echo "🎯 Ejecutando pipeline completo de Fixes para Exp 3 en Docker"
echo "=========================================================="
# Montamos el volumen actual y ejecutamos el script de fixes que ya tenemos preparado
docker run --rm \
    --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/experiments:/app/experiments \
    -w /app \
    --env TF_USE_LEGACY_KERAS=1 \
    --env PYTHONPATH=/app \
    weather_thesis:tf \
    bash -c "bash ./run_all_experiment3_fixes.sh"

echo ""
echo "✅ Todo completado vía Docker!"
