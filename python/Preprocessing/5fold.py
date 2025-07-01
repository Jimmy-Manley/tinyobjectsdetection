import os
import shutil
import json
from sklearn.model_selection import KFold

# === CONFIGURACIÓN ===
# Carpeta que contiene subdirectorios 'labels_<species>_t<threshold>'
BASE_DIR = './'      # <- Ajusta esta ruta a donde están tus folders labels_fly_t0, etc.
# Carpeta donde están las imágenes correspondientes (opcional)
IMAGE_DIR = '~/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive_golden/Small Object dataset/train/images'
N_SPLITS = 5
RANDOM_STATE = 42

# Directorio para guardar los archivos JSON con los índices de cada split
INDICES_DIR = os.path.join(BASE_DIR, 'split_indices')
os.makedirs(INDICES_DIR, exist_ok=True)

def split_annotation_folder(label_folder):
    """
    Divide los .txt de label_folder en N_SPLITS folds,
    crea train/val y guarda un JSON con los índices de cada split.
    """
    files = sorted(f for f in os.listdir(label_folder) if f.endswith('.txt'))
    if not files:
        print(f"No hay .txt en {label_folder}, se omite.")
        return

    splits_data = {'files': files, 'splits': {}}
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(kf.split(files), start=1):
        splits_data['splits'][f'fold{fold}'] = {
            'train': [files[i] for i in train_idx],
            'val':   [files[i] for i in val_idx]
        }
        for split_name, indices in [('train', train_idx), ('val', val_idx)]:
            out_dir = os.path.join(label_folder, f'fold{fold}', split_name)
            labels_out = os.path.join(out_dir, 'labels')
            images_out = os.path.join(out_dir, 'images')
            os.makedirs(labels_out, exist_ok=True)
            os.makedirs(images_out, exist_ok=True)

            for i in indices:
                label_file = files[i]
                # Copiar anotación
                shutil.copy2(
                    os.path.join(label_folder, label_file),
                    os.path.join(labels_out, label_file)
                )
                # Copiar imagen si IMAGE_DIR está definido
                if IMAGE_DIR:
                    base = os.path.splitext(label_file)[0]
                    img_file = base + '.jpg'
                    src_img = os.path.join(IMAGE_DIR, img_file)
                    if os.path.exists(src_img):
                        shutil.copy2(src_img, os.path.join(images_out, img_file))

        print(f"Split creado: {label_folder}/fold{fold} "
              f"(train={len(train_idx)}, val={len(val_idx)})")

    # Guardar JSON con índices
    folder_name = os.path.basename(label_folder)
    indices_path = os.path.join(INDICES_DIR, f'{folder_name}_indices.json')
    with open(indices_path, 'w') as jf:
        json.dump(splits_data, jf, indent=2)
    print(f"Índices guardados en {indices_path}")

if __name__ == "__main__":
    for sub in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, sub)
        if os.path.isdir(path) and sub.startswith('labels_'):
            print(f"\nProcesando directorio de anotaciones: {sub}")
            split_annotation_folder(path)
