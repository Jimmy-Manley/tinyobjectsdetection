import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
thresholds = ['t10', 't20', 't30', 't40', 't50']
num_folds = 5

# === PATHS ===
base_input = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/datasets/threshold_labels').expanduser()
combined_jsons = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/combined_jsons').expanduser()
base_output = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/labels_all').expanduser()

# === MAIN LOOP ===
for threshold in thresholds:
    # Load the full JSON for this threshold
    coco_json_path = combined_jsons / f"instances_train_{threshold}.json"
    if not coco_json_path.exists():
        print(f"❌ Missing JSON: {coco_json_path}")
        continue

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    original_images = coco_data['images']
    original_annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Map for fast lookup
    filename_to_id = {img['file_name']: img['id'] for img in original_images}
    id_to_annotations = {}
    for ann in original_annotations:
        id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    for val_fold in range(num_folds):
        print(f"\n🔄 Processing {threshold} fold {val_fold} (val)")

        # Paths
        fold_base = base_output / threshold / f'fold{val_fold}'
        train_img_dir = fold_base / 'images' / 'train'
        val_img_dir = fold_base / 'images' / 'val'
        ann_dir = fold_base / 'annotations'
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        train_ids, val_ids = [], []

        # === Process folds
        for fold_idx in range(num_folds):
            fold_path = base_input / threshold / 'folds' / f'fold{fold_idx}' / 'images' / 'train'
            if not fold_path.exists():
                print(f"⚠️ Missing: {fold_path}")
                continue

            image_list = list(fold_path.glob('*.jpg'))

            if fold_idx == val_fold:
                print(f"📁 Copying {len(image_list)} val images from fold{fold_idx}")
                for img_path in tqdm(image_list, desc=f"{threshold} val"):
                    shutil.copy(img_path, val_img_dir)
                    name = img_path.name
                    if name not in filename_to_id:
                        print(f"⚠️ {name} not in COCO JSON — skipped")
                        continue
                    val_ids.append(filename_to_id[name])
            else:
                print(f"📁 Copying {len(image_list)} train images from fold{fold_idx}")
                for img_path in tqdm(image_list, desc=f"{threshold} train"):
                    shutil.copy(img_path, train_img_dir)
                    name = img_path.name
                    if name not in filename_to_id:
                        print(f"⚠️ {name} not in COCO JSON — skipped")
                        continue
                    train_ids.append(filename_to_id[name])

        # === Build filtered COCO JSONs
        train_json = {
            'images': [img for img in original_images if img['id'] in train_ids],
            'annotations': [ann for img_id in train_ids for ann in id_to_annotations.get(img_id, [])],
            'categories': categories
        }

        val_json = {
            'images': [img for img in original_images if img['id'] in val_ids],
            'annotations': [ann for img_id in val_ids for ann in id_to_annotations.get(img_id, [])],
            'categories': categories
        }

        # === Save filtered JSONs
        with open(ann_dir / 'instances_train.json', 'w') as f:
            json.dump(train_json, f, indent=2)

        with open(ann_dir / 'instances_val.json', 'w') as f:
            json.dump(val_json, f, indent=2)

        print(f"✅ Done: fold{val_fold} — {len(train_ids)} train / {len(val_ids)} val images")