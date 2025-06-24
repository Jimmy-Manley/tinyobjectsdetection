import os
import json
import numpy as np
import scipy.io
import cv2
import re
from tqdm import tqdm

# === CLASS MAPPING ===
class_map = {"fly": 0, "fish": 1, "honeybee": 2, "seagull": 3}

# === THRESHOLDS ===
thresholds = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

# === DATASET CONFIGURATION ===


# === DATASET CONFIGURATION ===

# === DATASET CONFIGURATION ===
datasets = {
    "fly": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fly/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fly/img/",
        "output_base": "labels_fly"
    },
    "honeybee": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/honeybee/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/honeybee/img/",
        "output_base": "labels_honeybee"
    },
    "seagull": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/seagull/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/seagull/img/",
        "output_base": "labels_seagull"
    },
    "fish": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fish/gt-bbox/",
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small_Object_dataset/train/fish/img/",
        "output_base": "labels_fish"
    },
}
# === MAIN LOOP PER THRESHOLD === 
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

        for mat_file in tqdm(os.listdir(input_mat_dir), desc=f"{class_name} @ {suffix}"):
            if not mat_file.endswith(".mat"):
                continue

            match = re.search(r'(\d+)', mat_file)
            if not match:
                continue

            id_str = match.group(1).zfill(3)
            image_name = f"{class_name}{id_str}.jpg"
            mat_path = os.path.join(input_mat_dir, mat_file)
            image_path = os.path.join(input_img_dir, image_name)

            if not os.path.exists(image_path):
                continue

            img = cv2.imread(image_path)
            if img is None:
                continue

            height, width = img.shape[:2]

            images.append({
                "id": img_id,
                "file_name": image_name,
                "height": height,
                "width": width
            })

            try:
                data = scipy.io.loadmat(mat_path)
                keys = [k for k in data.keys() if not k.startswith("__")]
                if not keys:
                    continue
                boxes = data[keys[-1]]
            except Exception:
                continue

            for i in range(boxes.shape[0]):
                x, y, w, h = boxes[i]
                vis = 1.0  # assumed full visibility

                if vis < threshold:
                    continue

                category_id = class_map.get(class_name.lower(), 0)
                bbox = [float(x), float(y), float(w), float(h)]
                area = float(w * h)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1

            img_id += 1

    # === SAVE COMBINED JSON FOR THIS THRESHOLD ===
    os.makedirs("combined_jsons", exist_ok=True)
    out_path = os.path.join("combined_jsons", f"instances_train_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

    print(f"✅ Saved {out_path} with {len(images)} images and {len(annotations)} annotations")
