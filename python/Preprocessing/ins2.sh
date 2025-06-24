#!/bin/bash

set -e  # Exit on error

echo "🚧 Preparando estructura de carpetas..."
mkdir -p DQ-DETR/models/dqdetr/ops/src/cpu
mkdir -p DQ-DETR/models/dqdetr/ops/src/cuda

cd DQ-DETR/models/dqdetr/ops

echo "🌐 Descargando archivos fuente..."
wget -O src/cpu/ms_deform_attn_cpu.cpp https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/models/ops/src/cpu/ms_deform_attn_cpu.cpp
wget -O src/cuda/ms_deform_attn_cuda.cu https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/models/ops/src/cuda/ms_deform_attn_cuda.cu

echo "🔧 Compilando e instalando MultiScaleDeformableAttention..."
python3 setup.py build install

echo "✅ ¡Extensión compilada e instalada correctamente!"
