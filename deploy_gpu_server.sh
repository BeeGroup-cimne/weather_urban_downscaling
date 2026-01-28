#!/bin/bash

# ==========================================================
# Script de Despliegue para GPU Server - Weather Downscaling
# Solución completa para problemas de memoria y OOM
# ==========================================================

set -e  # Exit on error

echo "🚀 Desplegando Weather Downscaling en GPU Server..."
echo "================================================"

# Función para coloured output
print_status() {
    echo -e "\033[1;34m📋 $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✅ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠️  $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m❌ $1\033[0m"
}

# Verificar requisitos
check_requirements() {
    print_status "Verificando requisitos del sistema..."
    
    # Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado"
        exit 1
    fi
    
    # Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose no está instalado"
        exit 1
    fi
    
    # NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime no está configurado"
        print_warning "Instala nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    
    # GPU disponible
    if ! nvidia-smi &> /dev/null; then
        print_error "NVIDIA GPU o drivers no detectados"
        exit 1
    fi
    
    # Mostrar info GPU
    print_success "GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

# Preparar directorios
prepare_directories() {
    print_status "Preparando directorios..."
    
    mkdir -p experiments/models
    mkdir -p experiments/logs
    mkdir -p experiments/figures
    mkdir -p data/processed
    
    print_success "Directorios creados"
}

# Build imágenes Docker
build_docker_images() {
    print_status "Construyendo imágenes Docker optimizadas..."
    
    # Build imagen TensorFlow
    print_status "   🔨 Construyendo imagen TensorFlow..."
    docker build -f docker/Dockerfile.tf -t weather_thesis:tf .
    
    # Build imagen PyTorch
    print_status "   🔨 Construyendo imagen PyTorch..."
    docker build -f docker/Dockerfile.torch -t weather_thesis:torch .
    
    print_success "Imágenes Docker construidas"
}

# Preparar datos (si no existen)
prepare_data() {
    print_status "Verificando datos de entrada..."
    
    # Verificar archivos crudos
    if [[ ! -f "data/processed/weather_cache.zarr" ]]; then
        print_warning "Datos procesados no encontrados. Ejecutando preprocessing..."
        
        docker-compose -f docker-compose.gpu-optimized.yml --profile preprocessing up data-prep
        docker-compose -f docker-compose.gpu-optimized.yml --profile preprocessing down
        
        print_success "Preprocessing completado"
    else
        print_success "Datos ya procesados"
    fi
}

# Entrenamiento principal
run_training() {
    print_status "Iniciando entrenamiento optimizado..."
    
    # Iniciar monitor de memoria
    print_status "   🧠 Iniciando monitor de memoria..."
    docker-compose -f docker-compose.gpu-optimized.yml --profile monitoring up -d memory-monitor
    
    # Entrenamiento TensorFlow
    print_status "   🎯 Iniciando entrenamiento TensorFlow..."
    docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up tf-trainer
    
    # Esperar finalización
    print_status "⏳ Esperando finalización del entrenamiento..."
    docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up tf-trainer
    
    # Limpiar
    docker-compose -f docker-compose.gpu-optimized.yml down
    
    print_success "Entrenamiento completado"
}

# Entrenamiento Transformer - opcional
run_transformer_training() {
    if [[ $1 == "--include-transformer" ]]; then
        print_status "🤖 Iniciando entrenamiento Transformer..."
        
        docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up transformer-trainer
        
        print_success "Entrenamiento Transformer completado"
    else
        print_warning "Omitiendo entrenamiento Transformer (usar --include-transformer para activar)"
    fi
}

# Entrenamiento Transformer - opcional
run_transformer_training() {
    if [[ $1 == "--include-transformer" ]]; then
        print_status "🤖 Iniciando entrenamiento Transformer..."
        
        docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up transformer-trainer
        
        print_success "Entrenamiento Transformer completado"
    else
        print_warning "Omitiendo entrenamiento Transformer (usar --include-transformer para activar)"
    fi
}

# Entrenamiento PyTorch (Mamba) - opcional
run_pytorch_training() {
    if [[ $1 == "--include-mamba" ]]; then
        print_status "🔥 Iniciando entrenamiento PyTorch Mamba..."
        
        docker-compose -f docker-compose.gpu-optimized.yml --profile training --profile gpu up torch-trainer
        
        print_success "Entrenamiento Mamba completado"
    else
        print_warning "Omitiendo entrenamiento Mamba (usar --include-mamba para activar)"
    fi
}

# Generar resultados para paper
generate_results() {
    print_status "Generando resultados para paper..."
    
    docker run --rm \
        -v $(pwd):/app \
        --gpus all \
        weather_thesis:tf \
        python scripts/evaluate_for_paper.py
    
    print_success "Resultados generados en experiments/"
}

# Limpiar recursos
cleanup() {
    print_status "Limpiando recursos Docker..."
    
    docker-compose -f docker-compose.gpu-optimized.yml down --remove-orphans
    docker system prune -f
    
    print_success "Limpieza completada"
}

# Mostrar resultados
show_results() {
    print_status "Resumen de resultados generados:"
    
    echo "📊 Modelos entrenados:"
    ls -la experiments/models/ | grep -E "\.h5$"
    
    echo ""
    echo "📈 Logs de entrenamiento:"
    ls -la experiments/logs/ | grep -E "\.csv$"
    
    echo ""
    echo "📊 Resultados para paper:"
    if [[ -f "experiments/paper_results_table.csv" ]]; then
        cat experiments/paper_results_table.csv
    fi
    
    if [[ -f "experiments/paper_model_comparison.png" ]]; then
        print_success "Gráfica comparativa generada: experiments/paper_model_comparison.png"
    fi
}

# Función de ayuda
show_help() {
    echo "Uso: $0 [OPCIONES]"
    echo ""
    echo "Opciones:"
    echo "  --help                 Mostrar esta ayuda"
    echo "  --check-only           Solo verificar requisitos"
    echo "  --build-only           Solo construir imágenes Docker"
    echo "  --preprocess-only      Solo procesar datos"
    echo "  --train-only           Solo entrenar modelos"
    echo "  --include-transformer Incluir entrenamiento Transformer"
    echo "  --include-mamba        Incluir entrenamiento Mamba (PyTorch)"
    echo "  --include-all          Incluir todos los modelos (Transformer + Mamba)"
    echo "  --results-only         Solo generar resultados para paper"
    echo "  --cleanup              Limpiar recursos Docker"
    echo ""
    echo "Ejemplos:"
    echo "  $0                     Pipeline completo"
    echo "  $0 --train-only       Solo entrenamiento"
    echo "  $0 --include-transformer Pipeline completo con Transformer"
    echo "  $0 --include-mamba    Pipeline completo con Mamba"
    echo "  $0 --include-all      Pipeline completo con todos los modelos"
    echo "  $0 --results-only      Generar resultados para paper"
}

# Main execution
main() {
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --check-only)
            check_requirements
            exit 0
            ;;
        --build-only)
            check_requirements
            prepare_directories
            build_docker_images
            exit 0
            ;;
        --preprocess-only)
            check_requirements
            prepare_directories
            prepare_data
            exit 0
            ;;
        --train-only)
            check_requirements
            prepare_directories
            build_docker_images
            run_training
            run_pytorch_training $1
            exit 0
            ;;
        --results-only)
            generate_results
            show_results
            exit 0
            ;;
        --cleanup)
            cleanup
            exit 0
            ;;
        --include-mamba)
            INCLUDE_MAMBA=true
            ;;
        --include-transformer)
            INCLUDE_TRANSFORMER=true
            ;;
        --include-all)
            INCLUDE_TRANSFORMER=true
            INCLUDE_MAMBA=true
            ;;
        "")
            INCLUDE_TRANSFORMER=false
            INCLUDE_MAMBA=false
            ;;
        *)
            print_error "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
    
    # Pipeline completo
    print_status "Iniciando pipeline completo de despliegue..."
    
    check_requirements
    prepare_directories
    build_docker_images
    prepare_data
    run_training
    run_transformer_training $1
    run_pytorch_training $1
    generate_results
    show_results
    
    print_success "🎉 Despliegue completado exitosamente!"
    print_status "Revisa la carpeta experiments/ para resultados y modelos entrenados"
}

# Trap para limpieza
trap cleanup EXIT

# Ejecutar main
main "$@"