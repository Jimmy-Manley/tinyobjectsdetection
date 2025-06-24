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

# === CONVERSION FUNCTION ===
def convert_mat_to_coco(threshold, suffix, input_img_dir, input_mat_dir, output_json_path, class_prefix):
    images = []
    annotations = []
    categories = [{"id": v, "name": k} for k, v in class_map.items()]
    ann_id = 0
    img_id = 0

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    for mat_file in tqdm(os.listdir(input_mat_dir), desc=f"{class_prefix} @ Threshold {threshold}"):
        if not mat_file.endswith(".mat"):
            continue

        match = re.search(r'(\d+)', mat_file)
        if not match:
            continue

        id_str = match.group(1).zfill(3)
        image_name = f"{class_prefix}{id_str}.jpg"
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
            boxes = data[list(data.keys())[-1]]  # assume last key is the matrix
        except Exception:
            continue

        for i in range(boxes.shape[0]):
            x, y, w, h = boxes[i]
            vis = 1.0  # assumed full visibility

            if vis < threshold:
                continue

            category_id = class_map.get(class_prefix.lower(), 0)
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

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    return len(images), len(annotations)

# === VALIDATION FUNCTION ===
def validate_conversion(class_name, suffix, sample_id="001"):
    try:
        json_path = os.path.join(datasets[class_name]["output_base"], f"instances_train_{suffix}.json")
        mat_path = os.path.join(datasets[class_name]["annotations"], f"bbox_all_{sample_id}.mat")
        image_file = f"{class_name}{sample_id}.jpg"

        with open(json_path) as f:
            coco_data = json.load(f)

        matched_img = [img for img in coco_data["images"] if img["file_name"] == image_file]
        if not matched_img:
            return f"⚠️ {class_name} {suffix}: {image_file} not in JSON."

        matched_img = matched_img[0]
        matched_boxes = [ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == matched_img["id"]]

        mat = scipy.io.loadmat(mat_path)
        mat_key = [k for k in mat.keys() if not k.startswith("__")][0]
        boxes = mat[mat_key]

        return f"{class_name} {suffix} ✅ JSON: {len(matched_boxes)} boxes, MAT: {boxes.shape[0]} boxes"

    except Exception as e:
        return f"❌ {class_name} {suffix} ERROR: {str(e)}"

# === MAIN LOOP WITH LOGGING ===
log_lines = []

for class_name, paths in datasets.items():
    for t in thresholds:
        suffix = f"t{int(t * 100)}" if t > 0 else "original"
        output_json_path = os.path.join(paths["output_base"], f"instances_train_{suffix}.json")

        num_imgs, num_anns = convert_mat_to_coco(
            threshold=t,
            suffix=suffix,
            input_img_dir=paths["images"],
            input_mat_dir=paths["annotations"],
            output_json_path=output_json_path,
            class_prefix=class_name
        )

        val_result = validate_conversion(class_name=class_name, suffix=suffix)
        log_lines.append(f"{class_name.upper()} [{suffix}] - IMGs: {num_imgs}, ANNs: {num_anns} | {val_result}")

# === SAVE SUMMARY LOG ===
with open("conversion_validation_summary.txt", "w") as log_file:
    for line in log_lines:
        log_file.write(line + "\n")
