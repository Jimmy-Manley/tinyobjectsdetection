import scipy.io
import os
from PIL import Image

# Configuration
annotations_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/gt-bbox/"  # Folder with .mat files
images_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/img/"            # Folder with images
output_folder = "labels_true"            # Folder to store YOLO .txt files

os.makedirs(output_folder, exist_ok=True)

# Loop through all .mat files
for filename in os.listdir(annotations_folder):
    if filename.endswith(".mat"):
        mat_path = os.path.join(annotations_folder, filename)
        mat = scipy.io.loadmat(mat_path)
        
        if "bbox_all" not in mat:
            print(f"Skipped {filename}: 'bbox_all' not found.")
            continue

        # Derive image filename (assumes same name, different extension)
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(images_folder, base_name + ".jpg")

        if not os.path.exists(image_path):
            print(f"Image not found for {filename}, skipping.")
            continue

        # Load image and get size
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        bbox_all = mat["bbox_all"]
        yolo_annotations = []

        for box in bbox_all:
            if len(box) != 4:
                print(f"Invalid box in {filename}, skipping box: {box}")
                continue

            x, y, w, h = box
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w_norm = w / image_width
            h_norm = h / image_height
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save to .txt
        txt_filename = os.path.join(output_folder, base_name + ".txt")
        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_annotations))

        print(f"✅ Saved: {txt_filename}")
