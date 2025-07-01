import os
import shutil
import json
import re
from sklearn.model_selection import StratifiedKFold

# === CONFIGURACIÓN ===
BASE_DIR     = './'  # Carpeta con subdirs labels_*_t<threshold>
IMAGE_DIR    = os.path.expanduser('~/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive_golden/Small Object dataset/train/images')
OUTPUT_DIR   = os.path.join(BASE_DIR, '5_fold_splits')
INDICES_DIR  = os.path.join(BASE_DIR, 'split_indices')
N_SPLITS     = 5
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDICES_DIR, exist_ok=True)

# 1) Detectar thresholds disponibles
label_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith('labels_')]
thresholds = sorted({d.rsplit('_', 1)[1] for d in label_dirs})

for thr in thresholds:
    # 2) Reunir todas las anotaciones .txt de todas las clases para este umbral
    txt_paths = []
    for d in label_dirs:
        if d.endswith(f'_{thr}'):
            folder = os.path.join(BASE_DIR, d)
            for fn in os.listdir(folder):
                if fn.endswith('.txt'):
                    txt_paths.append(os.path.join(folder, fn))
    txt_paths.sort()
    filenames = [os.path.basename(p) for p in txt_paths]

    # Extraer etiquetas de clase desde el nombre de archivo (p.ej. 'fish123.txt' → 'fish')
    classes = [re.match(r'([a-zA-Z]+)', os.path.splitext(f)[0]).group(1) for f in filenames]

    # Comprobar suficiencia de muestras por clase
    from collections import Counter
    class_counts = Counter(classes)
    if any(count < N_SPLITS for count in class_counts.values()):
        print(f"Umbral {thr}: clases con <{N_SPLITS} muestras, no se puede estratificar. Se omite.")
        continue

    # 3) Crear splits estratificados con StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits_log = {'files': filenames, 'splits': {}}

    for fold, (train_idx, val_idx) in enumerate(skf.split(filenames, classes), start=1):
        splits_log['splits'][f'fold{fold}'] = {
            'train': [filenames[i] for i in train_idx],
            'val':   [filenames[i] for i in val_idx]
        }

        # 4) Copiar archivos .txt e imágenes a la estructura de carpeta
        for split_name, indices in [('train', train_idx), ('val', val_idx)]:
            labels_out = os.path.join(OUTPUT_DIR, thr, f'fold{fold}', split_name, 'labels')
            images_out = os.path.join(OUTPUT_DIR, thr, f'fold{fold}', split_name, 'images')
            os.makedirs(labels_out, exist_ok=True)
            os.makedirs(images_out, exist_ok=True)

            for i in indices:
                txt_src = txt_paths[i]
                txt_dst = os.path.join(labels_out, filenames[i])
                shutil.copy2(txt_src, txt_dst)

                base = os.path.splitext(filenames[i])[0]
                img_name = base + '.jpg'
                img_src = os.path.join(IMAGE_DIR, img_name)
                if os.path.exists(img_src):
                    shutil.copy2(img_src, os.path.join(images_out, img_name))

        print(f"Fold {fold} para umbral {thr}: train={len(train_idx)}, val={len(val_idx)}")

    # 5) Guardar JSON de índices
    json_path = os.path.join(INDICES_DIR, f'{thr}_indices.json')
    with open(json_path, 'w') as jf:
        json.dump(splits_log, jf, indent=2)
    print(f"Índices para umbral {thr} guardados en {json_path}")

print("==> Splits 5-fold estratificados generados para todos los umbrales.")
