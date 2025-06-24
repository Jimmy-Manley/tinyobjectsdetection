import os
import json
from pathlib import Path

# === CONFIGURATION ===
thresholds = ['t10', 't20', 't30', 't40', 't50']
num_folds = 5
tolerance = 1e-3  # Allow for small float rounding difference

# === PATHS ===
coco_base = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/replicated_folds').expanduser()
yolo_base = Path('~/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/labels_all').expanduser()

def load_yolo_labels(label_path):
    """Load YOLO labels from a .txt file"""
    if not label_path.exists():
        return []
    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)
        labels.append((int(cls), x, y, w, h))
    return labels

for threshold in thresholds:
    for fold in range(num_folds):
        print(f"\n🔍 Verifying: {threshold} fold{fold}")

        # === Paths ===
        coco_json_path = coco_base / threshold / f'fold{fold}' / 'annotations' / 'instances_train.json'
        images_dir = coco_base / threshold / f'fold{fold}' / 'images' / 'train'
        yolo_labels_dir = yolo_base / threshold / f'fold{fold}' / 'labels' / 'train'

        if not coco_json_path.exists():
            print(f"❌ Missing: {coco_json_path}")
            continue

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']
            cls_id = ann['category_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append((cls_id, bbox))

        mismatched = 0
        total_checked = 0

        for img_id, file_name in image_id_to_name.items():
            image_path = images_dir / file_name
            label_path = yolo_labels_dir / file_name.replace('.jpg', '.txt')

            if not label_path.exists():
                print(f"⚠️  Missing YOLO label for {file_name}")
                continue

            img_annotations = annotations_by_image.get(img_id, [])
            yolo_labels = load_yolo_labels(label_path)

            if len(img_annotations) != len(yolo_labels):
                print(f"❌ Mismatch count: {file_name} | COCO: {len(img_annotations)} vs YOLO: {len(yolo_labels)}")
                mismatched += 1
                continue

            # Optional: match annotations by class and position
            for (cls_coco, bbox), (cls_yolo, x, y, w, h) in zip(img_annotations, yolo_labels):
                if cls_coco != cls_yolo:
                    print(f"❌ Class mismatch in {file_name}: COCO={cls_coco}, YOLO={cls_yolo}")
                    mismatched += 1
                    break
                # Optionally: validate bbox (after normalizing COCO)
                img_width = next(img['width'] for img in coco_data['images'] if img['id'] == img_id)
                img_height = next(img['height'] for img in coco_data['images'] if img['id'] == img_id)

                x_coco = (bbox[0] + bbox[2] / 2) / img_width
                y_coco = (bbox[1] + bbox[3] / 2) / img_height
                w_coco = bbox[2] / img_width
                h_coco = bbox[3] / img_height

                if not all(abs(a - b) < tolerance for a, b in zip((x_coco, y_coco, w_coco, h_coco), (x, y, w, h))):
                    print(f"❌ BBox mismatch in {file_name}: COCO vs YOLO")
                    mismatched += 1
                    break

            total_checked += 1

        print(f"✅ Done: Checked {total_checked} files, {mismatched} mismatches")
