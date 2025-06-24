import os
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
thresholds = ['t0', 't10', 't20', 't30', 't40', 't50']
num_folds = 5

# Paths
threshold_base_dir = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/datasets/threshold_labels/').expanduser()
coco_base_dir = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/combined_jsons_thresh_clip/').expanduser()
output_base = Path('~/Documents/PROYECTO/Detección_particulas/train/annotations/fold_coco_replicated').expanduser()

output_base.mkdir(parents=True, exist_ok=True)

# === MAIN LOOP ===
for threshold in thresholds:
    coco_json_path = coco_base_dir / f"instances_train_{threshold}.json"

    if not coco_json_path.exists():
        print(f"❌ Missing COCO file: {coco_json_path}")
        continue

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    all_images = coco_data['images']
    all_annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Map filename → image_id
    filename_to_id = {img['file_name']: img['id'] for img in all_images}
    # Map image_id → annotations
    id_to_anns = {}
    for ann in all_annotations:
        id_to_anns.setdefault(ann['image_id'], []).append(ann)

    for fold in range(num_folds):
        print(f"\n🔄 Processing {threshold} fold{fold}")

        base_fold_path = threshold_base_dir / threshold / 'folds' / f'fold{fold}'

        for split in ['train', 'val']:
            split_dir = base_fold_path / 'images' / split
            if not split_dir.exists():
                print(f"⚠️ Skipping missing folder: {split_dir}")
                continue

            image_files = sorted(f.name for f in split_dir.glob('*.jpg'))
            selected_ids = [filename_to_id[f] for f in image_files if f in filename_to_id]

            filtered_images = [img for img in all_images if img['id'] in selected_ids]
            filtered_annotations = [ann for img_id in selected_ids for ann in id_to_anns.get(img_id, [])]

            output_json = {
                "images": filtered_images,
                "annotations": filtered_annotations,
                "categories": categories
            }

            output_path = output_base / f"instances_{threshold}_fold{fold}_{split}.json"
            with open(output_path, 'w') as f:
                json.dump(output_json, f, indent=2)

            print(f"✅ Saved: {output_path.name} | {split.capitalize()} Images: {len(filtered_images)}, Annotations: {len(filtered_annotations)}")
