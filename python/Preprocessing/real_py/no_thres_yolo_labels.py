import os
import scipy.io
from PIL import Image
import re

# Class ID mapping
class_map = {
    "fly": 0,
    "fish": 1,
    "honeybee": 2,
    "seagull": 3
}

# Configuration per category
datasets = {
    "fly": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/img/",
        "output_base": "labels_fly_no_threshold"
    },
    "honeybee": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/honeybee/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/honeybee/img/",        
        "output_base": "labels_honeybee_no_threshold"
    },
    "seagull": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/seagull/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/seagull/img/",      
        "output_base": "labels_seagull_no_threshold"
    },
    "fish": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/img/",   
        "output_base": "labels_fish_no_threshold"
    },
}

def extract_id(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

for name, paths in datasets.items():
    ann_folder = paths["annotations"]
    img_folder = paths["images"]
    out_folder = paths["output_base"]
    class_id = class_map[name]
    os.makedirs(out_folder, exist_ok=True)

    for mat_file in os.listdir(ann_folder):
        if not mat_file.endswith(".mat"):
            continue

        file_id = extract_id(mat_file)
        if not file_id:
            continue

        mat_path = os.path.join(ann_folder, mat_file)
        matching_imgs = [img for img in os.listdir(img_folder) if img.endswith(f"{file_id}.jpg")]
        if not matching_imgs:
            continue

        img_file = matching_imgs[0]
        img_path = os.path.join(img_folder, img_file)

        try:
            image = Image.open(img_path)
            width, height = image.size
        except:
            continue

        mat = scipy.io.loadmat(mat_path)
        if "bbox_all" not in mat:
            continue

        bbox_all = mat["bbox_all"]
        yolo_annotations = []

        for box in bbox_all:
            x, y, w, h = box
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            new_w = x2 - x1
            new_h = y2 - y1

            if new_w <= 0 or new_h <= 0:
                continue  # Skip invalid

            # YOLO format with normalization
            x_center = (x1 + new_w / 2) / width
            y_center = (y1 + new_h / 2) / height
            w_norm = new_w / width
            h_norm = new_h / height
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if yolo_annotations:
            base_name = os.path.splitext(img_file)[0]
            out_path = os.path.join(out_folder, base_name + ".txt")
            with open(out_path, "w") as f:
                f.write("\n".join(yolo_annotations))

print("✅ Label conversion (no threshold) complete.")
