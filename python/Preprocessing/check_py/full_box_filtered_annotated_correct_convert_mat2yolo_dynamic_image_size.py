import os
import scipy.io
from PIL import Image, ImageDraw

# Configuration per category
datasets = {
    "fly": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/img/",
        "output": "labels_fly"
    },
    "honeybee": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/honeybee/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/honeybee/img/",        
        "output": "labels_honeybee"
    },
    "seagull": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/seagull/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/seagull/img/",      
        "output": "labels_seagull"
    },
    "fish": {
        "annotations": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/gt-bbox/", 
        "images": "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/img/",   
        "output": "labels_fish"
    },
}

visible_ratio_threshold = 0.3
log_summary_file = "bbox_conversion_summary.log"

clipped_count = 0
skipped_invalid_count = 0
converted_count = 0
clipped_log = []
skipped_log = []

def extract_id(filename):
    import re
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

for name, paths in datasets.items():
    ann_folder = paths["annotations"]
    img_folder = paths["images"]
    out_folder = paths["output"]
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
                skipped_invalid_count += 1
                skipped_log.append(f"{img_file}: invalid area {x, y, w, h} -> {x1, y1, new_w, new_h}")
                continue

            visible_area = new_w * new_h
            original_area = w * h
            if visible_area / original_area < visible_ratio_threshold:
                skipped_invalid_count += 1
                skipped_log.append(f"{img_file}: <30% visible {visible_area / original_area:.2f}")
                continue

            if w != new_w or h != new_h:
                clipped_count += 1
                clipped_log.append(f"{img_file}: clipped {x, y, w, h} -> {x1, y1, new_w, new_h}")

            x_center = (x1 + new_w / 2) / width
            y_center = (y1 + new_h / 2) / height
            w_norm = new_w / width
            h_norm = new_h / height
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if yolo_annotations:
            base_name = os.path.splitext(img_file)[0]
            out_path = os.path.join(out_folder, base_name + ".txt")
            with open(out_path, "w") as f:
                f.write("\n".join(yolo_annotations))
            converted_count += 1

# Write summary
with open(log_summary_file, "w") as f:
    f.write(f"Converted files: {converted_count}\n")
    f.write(f"Clipped boxes: {clipped_count}\n")
    f.write(f"Skipped boxes: {skipped_invalid_count}\n\n")
    f.write("Clipped Log:\n")
    for entry in clipped_log:
        f.write(entry + "\n")
    f.write("\nSkipped Log:\n")
    for entry in skipped_log:
        f.write(entry + "\n")

print("✅ Conversion complete. Logs written to", log_summary_file)
