import scipy.io
import os

# Define the root dataset path
root_path = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train"

# Subfolders inside 'train' (e.g., 'fish', 'fly', etc.)
categories = ['fish', 'fly', 'honeybee', 'seagull']

# Image dimensions (set according to your dataset or read dynamically)
image_width = 640
image_height = 480

# Output folder for YOLO labels
output_base = "/home/jman/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/yolo_labels"

for category in categories:
    mat_folder = os.path.join(root_path, category, "gt-bbox")
    output_folder = os.path.join(output_base, category)
    os.makedirs(output_folder, exist_ok=True)

    # List all .mat files in current category folder
    for file_name in os.listdir(mat_folder):
        if file_name.endswith(".mat"):
            mat_path = os.path.join(mat_folder, file_name)
            mat = scipy.io.loadmat(mat_path)

            # Extract annotation
            if "bbox_all" not in mat:
                print(f"⚠️ 'bbox_all' not found in {mat_path}")
                continue

            bbox_all = mat['bbox_all']
            yolo_annotations = []

            for box in bbox_all:
                x, y, w, h = box
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                w_norm = w / image_width
                h_norm = h / image_height
                yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # Save .txt file with same name as .mat file
            txt_name = file_name.replace(".mat", ".txt")
            txt_path = os.path.join(output_folder, txt_name)

            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_annotations))

            print(f"✅ Saved: {txt_path}")
