import scipy.io
import os

# Configuration
annotations_folder = "/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fish/gt-bbox/"  # Folder with .mat files
output_folder = "labels"            # Folder to store YOLO .txt files
image_width = 640
image_height = 480

os.makedirs(output_folder, exist_ok=True)

# Loop through all .mat files
for filename in os.listdir(annotations_folder):
    if filename.endswith(".mat"):
        mat_path = os.path.join(annotations_folder, filename)
        mat = scipy.io.loadmat(mat_path)
        
        if "bbox_all" not in mat:
            print(f"Skipped {filename}: 'bbox_all' not found.")
            continue
        
        bbox_all = mat["bbox_all"]
        yolo_annotations = []

        for box in bbox_all:
            x, y, w, h = box
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w_norm = w / image_width
            h_norm = h / image_height
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save to .txt
        base_name = os.path.splitext(filename)[0]
        txt_filename = os.path.join(output_folder, base_name + ".txt")
        
        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_annotations))
        
        print(f"Saved: {txt_filename}")
