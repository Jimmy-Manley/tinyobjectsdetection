import os
import json
import numpy as np
import scipy.io
from PIL import Image
import re
from tqdm import tqdm

# === CLASS MAPPING ===
class_map = {"fly": 0, "fish": 1, "honeybee": 2, "seagull": 3}

# === THRESHOLDS ===
thresholds = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

# === DATASET CONFIGURATION ===
datasets = {
    "fly": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fly/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fly/img/"
    },
    "honeybee": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/honeybee/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/honeybee/img/"
    },
    "seagull": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/seagull/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/seagull/img/"
    },
    "fish": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fish/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fish/img/"
    },
}

# === UTILITY ===
def extract_id(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

# === MAIN CONVERSION ===
os.makedirs("combined_jsons_thresh_clip", exist_ok=True)

for threshold in thresholds:
    suffix = f"t{int(threshold * 100)}" if threshold > 0 else "t0"
    print(f"\n🔄 Processing threshold: {suffix}")

    images = []
    annotations = []
    categories = [{"id": v, "name": k} for k, v in class_map.items()]
    ann_id = 0
    img_id = 0

    for class_name, paths in datasets.items():
        input_mat_dir = paths["annotations"]
        input_img_dir = paths["images"]
        class_id = class_map[class_name]

        if not os.path.exists(input_mat_dir):
            print(f"⚠️ Skipping: {input_mat_dir} not found.")
            continue

        for mat_file in tqdm(os.listdir(input_mat_dir), desc=f"{class_name} @ {suffix}"):
            if not mat_file.endswith(".mat"):
                continue

            file_id = extract_id(mat_file)
            if not file_id:
                continue

            image_name = f"{class_name}{file_id.zfill(3)}.jpg"
            mat_path = os.path.join(input_mat_dir, mat_file)
            image_path = os.path.join(input_img_dir, image_name)

            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path)
                width, height = image.size
            except:
                continue

            images.append({
                "id": img_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })

            try:
                mat = scipy.io.loadmat(mat_path)
                if "bbox_all" not in mat:
                    continue
                boxes = mat["bbox_all"]
            except:
                continue

            for box in boxes:
                x, y, w, h = box
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, width)
                y2 = min(y + h, height)
                new_w = x2 - x1
                new_h = y2 - y1

                if new_w <= 0 or new_h <= 0:
                    continue

                visible_area = new_w * new_h
                original_area = w * h

                if original_area <= 0 or visible_area / original_area < threshold:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [float(x1), float(y1), float(new_w), float(new_h)],
                    "area": float(new_w * new_h),
                    "iscrowd": 0
                })
                ann_id += 1

            img_id += 1

    # === SAVE COCO JSON ===
    out_path = os.path.join("combined_jsons_thresh_clip", f"instances_train_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

    print(f"✅ Saved: {out_path} | Images: {len(images)}, Annotations: {len(annotations)}")
