import scipy.io
import os
from PIL import Image

# Config
annotations_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/gt-bbox/"
images_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/img/"
labels_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/fish_labels_txt/"
tolerance = 1.0  # Acceptable pixel difference

def compare_boxes(mat_box, txt_box, img_width, img_height):
    # Convert YOLO normalized to absolute
    class_id, x_c, y_c, w, h = map(float, txt_box.split())
    x = (x_c * img_width) - (w * img_width) / 2
    y = (y_c * img_height) - (h * img_height) / 2
    w_abs = w * img_width
    h_abs = h * img_height
    return abs(mat_box[0] - x) < tolerance and abs(mat_box[1] - y) < tolerance and abs(mat_box[2] - w_abs) < tolerance and abs(mat_box[3] - h_abs) < tolerance

# Loop over .mat files
for file in os.listdir(annotations_folder):
    if file.endswith(".mat"):
        base = os.path.splitext(file)[0]
        mat_path = os.path.join(annotations_folder, file)
        txt_path = os.path.join(labels_folder, base + ".txt")
        image_path = os.path.join(images_folder, base + ".jpg")

        if not os.path.exists(txt_path) or not os.path.exists(image_path):
            print(f"Missing file for {base}, skipping.")
            continue

        # Load image size
        with Image.open(image_path) as img:
            w, h = img.size

        # Load original boxes
        mat = scipy.io.loadmat(mat_path)
        if "bbox_all" not in mat:
            print(f"Missing bbox_all in {file}, skipping.")
            continue
        mat_boxes = mat["bbox_all"]

        # Load YOLO boxes
        with open(txt_path, "r") as f:
            txt_boxes = f.read().strip().split("\n")

        if len(mat_boxes) != len(txt_boxes):
            print(f"Mismatch count in {base}: {len(mat_boxes)} vs {len(txt_boxes)}")
            continue

        # Compare each box
        mismatch = False
        for mb, tb in zip(mat_boxes, txt_boxes):
            if not compare_boxes(mb, tb, w, h):
                print(f"❌ Mismatch in {base}:")
                print(f"  .mat: {mb}")
                print(f"  .txt (reversed): {tb}")
                mismatch = True
                break

        if not mismatch:
            print(f"✅ {base} matches.")

