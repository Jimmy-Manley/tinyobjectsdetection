import os
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
thresholds = ['t0','t10', 't20', 't30', 't40', 't50']  # Adjust if needed
num_folds = 5

# Set these to your actual paths
threshold_base_dir = os.path.expanduser('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/datasets/threshold_labels/')
coco_json_path = os.path.expanduser('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/combined_jsons/instances_train_t0.json')
output_base = os.path.expanduser('~/Documents/PROYECTO/Detección_particulas/train/annotations/fold_coco')

os.makedirs(output_base, exist_ok=True)

# === Load original COCO dataset ===
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

original_images = coco_data['images']
original_annotations = coco_data['annotations']
categories = coco_data['categories']

# Map image filename to ID for quick lookup
filename_to_id = {img['file_name']: img['id'] for img in original_images}

# Map image ID to annotations
id_to_anns = {}
for ann in original_annotations:
    id_to_anns.setdefault(ann['image_id'], []).append(ann)

# === MAIN LOOP ===
for threshold in thresholds:
    for fold in range(num_folds):
        print(f"\n🔄 Processing {threshold} fold {fold}")
        
        fold_dir = Path(threshold_base_dir) / threshold / 'folds' / f'fold{fold}'  / 'images' / 'train'  
        print(f"Checking folder: {fold_dir}")
        if not fold_dir.exists():
            print(f"⚠️ Skipping missing folder: {fold_dir}")
            continue

        image_files = sorted(f.name for f in fold_dir.glob('*.jpg'))
        selected_image_ids = [filename_to_id[f] for f in image_files if f in filename_to_id]

        # Filter images and annotations
        filtered_images = [img for img in original_images if img['id'] in selected_image_ids]
        filtered_annotations = [ann for img_id in selected_image_ids for ann in id_to_anns.get(img_id, [])]

        coco_split = {
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': categories
        }

        # Save
        output_path = Path(output_base) / f"instances_{threshold}_fold{fold}.json"
        with open(output_path, 'w') as f:
            json.dump(coco_split, f, indent=2)

        print(f"✅ Saved: {output_path.name} | Images: {len(filtered_images)}, Annotations: {len(filtered_annotations)}")
