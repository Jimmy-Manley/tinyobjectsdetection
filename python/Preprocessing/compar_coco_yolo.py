import os
import json
from pathlib import Path

def convert_coco_to_yolo_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]

def read_yolo_labels(yolo_label_path):
    with open(yolo_label_path, 'r') as f:
        lines = f.read().strip().splitlines()
    return [list(map(float, line.split())) for line in lines]

def compare_labels(yolo_labels, coco_labels, epsilon=1e-3):
    if len(yolo_labels) != len(coco_labels):
        return False, f"Count mismatch: YOLO={len(yolo_labels)}, COCO={len(coco_labels)}"

    for yolo_ann, coco_ann in zip(sorted(yolo_labels), sorted(coco_labels)):
        if int(yolo_ann[0]) != int(coco_ann[0]):
            return False, f"Class mismatch: YOLO={int(yolo_ann[0])}, COCO={int(coco_ann[0])}"
        for i in range(1, 5):
            if abs(yolo_ann[i] - coco_ann[i]) > epsilon:
                return False, f"Coord mismatch: {yolo_ann[1:]} vs {coco_ann[1:]}"
    return True, "OK"

# === CONFIGURATION ===
thresholds = ['t0', 't10', 't20', 't30', 't40', 't50']
num_folds = 5
splits = ['train', 'val']

# Base paths
yolo_base = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/datasets/threshold_labels/').expanduser()
coco_base = Path('~/Documents/PROYECTO/Detección_particulas/train/fold_coco_replicated_full/').expanduser()

results = []

for threshold in thresholds:
    for fold in range(num_folds):
        for split in splits:
            coco_json_path = coco_base / threshold / f'fold{fold}' / f'instances_{split}.json'
            yolo_labels_dir = yolo_base / threshold / 'folds' / f'fold{fold}' / 'labels' / split

            if not coco_json_path.exists() or not yolo_labels_dir.exists():
                continue

            with open(coco_json_path) as f:
                coco_data = json.load(f)

            # Build lookup structures
            images_by_id = {img['id']: img for img in coco_data['images']}
            annotations_by_image = {}
            for ann in coco_data['annotations']:
                annotations_by_image.setdefault(ann['image_id'], []).append(ann)

            for image in coco_data['images']:
                img_file = image['file_name']
                img_id = image['id']
                img_width = image['width']
                img_height = image['height']

                yolo_file = yolo_labels_dir / Path(img_file).with_suffix('.txt').name
                if not yolo_file.exists():
                    results.append({
                        "threshold": threshold,
                        "fold": fold,
                        "split": split,
                        "image": img_file,
                        "status": "YOLO file missing"
                    })
                    continue

                yolo_labels = read_yolo_labels(yolo_file)
                coco_labels = []
                for ann in annotations_by_image.get(img_id, []):
                    yolo_bbox = convert_coco_to_yolo_bbox(ann['bbox'], img_width, img_height)
                    coco_labels.append([ann['category_id']] + yolo_bbox)

                match, reason = compare_labels(yolo_labels, coco_labels)
                results.append({
                    "threshold": threshold,
                    "fold": fold,
                    "split": split,
                    "image": img_file,
                    "status": reason
                })

# Optionally: save or display result
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("coco_vs_yolo_verification_results.csv", index=False)
print("✅ Verification completed. Results saved to 'coco_vs_yolo_verification_results.csv'")
