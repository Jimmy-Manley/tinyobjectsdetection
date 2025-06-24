import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
thresholds = ['t0','t10', 't20', 't30', 't40', 't50']
num_folds = 5

# === PATHS ===
src_base = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/datasets/threshold_labels').expanduser()
json_base = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/combined_jsons_thresh_clip').expanduser()
output_base = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/replicated_folds').expanduser()

for threshold in thresholds:
    print(f"\n📂 Processing threshold: {threshold}")

    # Load full COCO JSON
    json_path = json_base / f"instances_train_{threshold}.json"
    if not json_path.exists():
        print(f"❌ Missing JSON: {json_path}")
        continue

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_all = coco_data['images']
    anns_all = coco_data['annotations']
    categories = coco_data['categories']

    # Lookup tables
    fname_to_id = {img["file_name"]: img["id"] for img in images_all}
    id_to_anns = {}
    for ann in anns_all:
        id_to_anns.setdefault(ann['image_id'], []).append(ann)

    for fold in range(num_folds):
        print(f"🔄 Fold {fold} for {threshold}")

        train_filenames = []
        val_filenames = []

        # Build train/val lists based on folder structure
        for i in range(num_folds):
            folder = src_base / threshold / 'folds' / f'fold{i}' / 'images' / 'train'
            if not folder.exists():
                print(f"⚠️ Missing: {folder}")
                continue

            files = [f.name for f in folder.glob('*.jpg')]
            if i == fold:
                train_filenames.extend(files)
                fold_folder = folder  # keep to copy from
            else:
                val_filenames.extend(files)

        # Output paths
        fold_base = output_base / threshold / f'fold{fold}'
        train_dir = fold_base / 'images' / 'train'
        val_dir = fold_base / 'images' / 'val'
        ann_dir = fold_base / 'annotations'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        # Copy train images
        print(f"📁 Copying train images...")
        train_ids = []
        for fname in tqdm(train_filenames, desc="Train"):
            src = fold_folder / fname
            dst = train_dir / fname
            if src.exists():
                shutil.copy(src, dst)
                if fname in fname_to_id:
                    train_ids.append(fname_to_id[fname])

        # Copy val images (from all other folds)
        print(f"📁 Copying val images...")
        val_ids = []
        for i in range(num_folds):
            if i == fold:
                continue
            val_folder = src_base / threshold / 'folds' / f'fold{i}' / 'images' / 'train'
            for fname in val_filenames:
                src = val_folder / fname
                dst = val_dir / fname
                if src.exists():
                    shutil.copy(src, dst)
                    if fname in fname_to_id:
                        val_ids.append(fname_to_id[fname])

        # Generate train JSON
        train_json = {
            "images": [img for img in images_all if img["id"] in train_ids],
            "annotations": [ann for img_id in train_ids for ann in id_to_anns.get(img_id, [])],
            "categories": categories
        }

        # Generate val JSON
        val_json = {
            "images": [img for img in images_all if img["id"] in val_ids],
            "annotations": [ann for img_id in val_ids for ann in id_to_anns.get(img_id, [])],
            "categories": categories
        }

        with open(ann_dir / 'instances_train.json', 'w') as f:
            json.dump(train_json, f, indent=2)

        with open(ann_dir / 'instances_val.json', 'w') as f:
            json.dump(val_json, f, indent=2)

        print(f"✅ Fold {fold} done: {len(train_ids)} train, {len(val_ids)} val")
