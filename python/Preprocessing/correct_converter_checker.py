import os
import scipy.io
from PIL import Image

# Paths
annotations_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/seagull/gt-bbox/"
images_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/seagull/img/"
labels_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/labels_seagull/"

tolerance = 1e-6  # Use stricter tolerance if desired, e.g., 1e-8

# Loop through all .mat files
for mat_file in os.listdir(annotations_folder):
    if not mat_file.startswith("bbox_all_") or not mat_file.endswith(".mat"):
        continue

    # Extract numeric ID
    img_id = mat_file.replace("bbox_all_", "").replace(".mat", "")
    image_name = f"seagull{img_id}.jpg"
    label_name = f"seagull{img_id}.txt"

    image_path = os.path.join(images_folder, image_name)
    label_path = os.path.join(labels_folder, label_name)
    mat_path = os.path.join(annotations_folder, mat_file)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue

    if not os.path.exists(label_path):
        print(f"❌ Label not found: {label_path}")
        continue

    # Load image size
    with Image.open(image_path) as img:
        width, height = img.size

    # Load .mat data
    mat = scipy.io.loadmat(mat_path)
    if "bbox_all" not in mat:
        print(f"⚠️ 'bbox_all' missing in {mat_file}")
        continue

    mat_boxes = mat["bbox_all"]
    converted = []
    for box in mat_boxes:
        x, y, w, h = box
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height
        converted.append([x_center, y_center, w_norm, h_norm])

    # Load .txt annotations
    with open(label_path, "r") as f:
        txt_lines = [line.strip().split() for line in f.readlines()]
        txt_boxes = [[float(x) for x in line[1:]] for line in txt_lines if len(line) == 5]

    # Compare
    if len(converted) != len(txt_boxes):
        print(f"❌ Mismatch in box count: {image_name} | .mat: {len(converted)} vs .txt: {len(txt_boxes)}")
        continue

    mismatch = False
    for i in range(len(converted)):
        for j in range(4):
            tol = converted[i][j] - txt_boxes[i][j]
            # print(tol)
            if abs(converted[i][j] - txt_boxes[i][j]) > tolerance:
                mismatch = True
                break
        if mismatch:
            print(f"❌ Box mismatch in {image_name} at index {i}")
            break

    if not mismatch:
        print(f"✅ Match: {image_name}")
