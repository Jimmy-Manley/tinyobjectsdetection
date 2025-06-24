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
    # Load full JSON for this threshold
    coco_json_path = combined_jsons / f"instances_train_{threshold}.json"
    if not coco_json_path.exists():
        print(f"❌ Missing JSON: {coco_json_path}")
        continue

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    original_images = coco_data['images']
    original_annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Build lookup tables
    filename_to_id = {img['file_name']: img['id'] for img in original_images}
    id_to_annotations = {}
    for ann in original_annotations:
        id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    for fold in range(num_folds):
        print(f"\n🔄 Processing {threshold} fold {fold}")

        # Define source and destination paths
        src_img_dir = base_input / threshold / 'folds' / f'fold{fold}' / 'images' / 'train'
        dst_img_dir = base_output / threshold / f'fold{fold}' / 'images' / 'train'
        dst_ann_dir = base_output / threshold / f'fold{fold}' / 'annotations'
        dst_json_path = dst_ann_dir / 'instances_train.json'

        if not src_img_dir.exists():
            print(f"⚠️ Skipping missing folder: {src_img_dir}")
            continue

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        # Copy image files
        image_files = list(src_img_dir.glob('*.jpg'))
        selected_image_ids = []

        print(f"📂 Copying {len(image_files)} images...")
        for img_path in tqdm(image_files, desc=f"{threshold} fold{fold}"):
            shutil.copy(img_path, dst_img_dir)
            fname = img_path.name
            if fname in filename_to_id:
                selected_image_ids.append(filename_to_id[fname])

        # Filter images and annotations
        filtered_images = [img for img in original_images if img['id'] in selected_image_ids]
        filtered_annotations = [
            ann for img_id in selected_image_ids for ann in id_to_annotations.get(img_id, [])
        ]

        # Save filtered JSON
        fold_json = {
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': categories
        }

        with open(dst_json_path, 'w') as f:
            json.dump(fold_json, f, indent=2)

        print(f"✅ Saved COCO JSON: {dst_json_path} ({len(filtered_images)} images, {len(filtered_annotations)} annotations)")
