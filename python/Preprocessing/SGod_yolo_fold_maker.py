import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
thresholds = ['t0', 't10', 't20', 't30', 't40', 't50']
num_folds = 5

# Paths
threshold_base_dir = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/datasets/threshold_labels/').expanduser()
coco_base_dir = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/combined_jsons_thresh_clip/').expanduser()
output_base = Path('~/Documents/PROYECTO/Detección_particulas/train/fold_coco_replicated_full/').expanduser()

# === MAIN LOOP ===
for threshold in thresholds:
    print(f"\n🔄 Processing threshold: {threshold}")
    coco_json_path = coco_base_dir / f"instances_train_{threshold}.json"
    if not coco_json_path.exists():
        print(f"❌ Missing COCO file: {coco_json_path}")
        continue

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    all_images = coco_data['images']
    all_annotations = coco_data['annotations']
    categories = coco_data['categories']

    filename_to_id = {img['file_name']: img['id'] for img in all_images}
    id_to_annotations = {}
    for ann in all_annotations:
        id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    for fold in range(num_folds):
        base_fold_dir = threshold_base_dir / threshold / 'folds' / f'fold{fold}'
        for split in ['train', 'val']:
            split_dir = base_fold_dir / 'images' / split
            if not split_dir.exists():
                print(f"⚠️ Skipping: {split_dir}")
                continue

            image_files = sorted(split_dir.glob('*.jpg'))
            selected_filenames = [f.name for f in image_files if f.name in filename_to_id]
            selected_ids = [filename_to_id[f] for f in selected_filenames]

            # Filter images and annotations
            filtered_images = [img for img in all_images if img['id'] in selected_ids]
            filtered_annotations = [ann for img_id in selected_ids for ann in id_to_annotations.get(img_id, [])]

            # Output directories
            out_dir = output_base / threshold / f'fold{fold}' / split
            out_img_dir = out_dir / 'images'
            out_label_dir = out_dir / 'labels'
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_label_dir.mkdir(parents=True, exist_ok=True)

            # Copy image files
            for img_file in image_files:
                dst = out_img_dir / img_file.name
                shutil.copy(img_file, dst)

            # Save COCO JSON
            out_json = {
                "images": filtered_images,
                "annotations": filtered_annotations,
                "categories": categories
            }
            json_path = output_base / threshold / f'fold{fold}' / f"instances_{split}.json"
            with open(json_path, 'w') as f:
                json.dump(out_json, f, indent=2)

            print(f"✅ {threshold} fold{fold} {split}: {len(filtered_images)} images, {len(filtered_annotations)} annotations → {json_path.name}")
