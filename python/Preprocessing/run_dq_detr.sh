#!/bin/bash

# === Configuración base ===
PROJECT_DIR="/home/jman/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing"
#"$HOME/small_object_detection"
DQ_DIR="$PROJECT_DIR/DQ-DETR"
THRESHOLDS=("t10" "t20" "t30" "t40" "t50")
NUM_FOLDS=5
echo "🚀 Iniciando entrenamiento + evaluación DQ-DETR"

# === Activar entorno virtual (ajusta si usas conda o venv) ===
source ~/dqenv/bin/activate  # ❗️CAMBIA esto a tu entorno

# === Instalar dependencias necesarias (solo una vez) ===
pip install addict yapf pycocotools

# === Compilar MultiScaleDeformableAttention (solo una vez) ===
cd "$DQ_DIR" #/models/dqdetr/ops"
python3 setup.py build install

# === Volver al directorio raíz del proyecto ===
cd "$DQ_DIR"

# === Loop de entrenamiento y evaluación ===
for threshold in "${THRESHOLDS[@]}"; do
  for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    echo "======================================="
    echo "🚀 Entrenando $threshold fold$fold"
    
    # Entrenamiento
    python3 main_dual_ann_train_val.py \
      --config_file configs/DQ-DETR.yaml \
      --output_dir "$PROJECT_DIR/runs/DQ_DETR/$threshold/fold$fold" \
      --train_ann "$PROJECT_DIR/threshold_labels/$threshold/folds/fold$fold/instances_train.json" \
      --val_ann "$PROJECT_DIR/threshold_labels/$threshold/folds/fold$fold/instances_val.json" \
      --image_folder "$PROJECT_DIR/threshold_labels/$threshold/folds/fold$fold/images"

    echo "📊 Evaluando $threshold fold$fold"

    # Evaluación
    python3 evaluate_and_save_dual_ann.py \
      --config_file configs/DQ-DETR.yaml \
      --resume "$PROJECT_DIR/runs/DQ_DETR/$threshold/fold$fold/checkpoint.pth" \
      --val_ann "$PROJECT_DIR/threshold_labels/$threshold/folds/fold$fold/instances_val.json" \
      --image_folder "$PROJECT_DIR/threshold_labels/$threshold/folds/fold$fold/images" \
      --eval
  done
done

echo "✅ Entrenamiento y evaluación completados para todos los folds y thresholds"
