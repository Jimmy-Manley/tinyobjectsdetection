import scipy.io
import os
from PIL import Image
import re

# Configuration
annotations_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/gt-bbox/"  # Folder with .mat files like bbox_all_070.mat
images_path = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/img/"              # Folder with images like fish070.jpg
output_folder = "labels_fly"            # Output YOLO .txt files
log_summary_file = "fly_conversion_summary.log"

os.makedirs(output_folder, exist_ok=True)

# Regex to extract the numeric ID from the .mat filename
def extract_id(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

# Summary counters
clipped_count = 0
skipped_invalid_count = 0
converted_count = 0
clipped_log = []
skipped_log = []

# Loop through all .mat files
for filename in os.listdir(annotations_folder):
    if filename.endswith(".mat"):
        mat_path = os.path.join(annotations_folder, filename)
        file_id = extract_id(filename)

        if not file_id:
            print(f"⚠️ Skipped {filename}: no numeric ID found.")
            continue

        # Find the image file that ends with the same ID
        matching_images = [img for img in os.listdir(images_path) if img.endswith(f"{file_id}.jpg")]
        if not matching_images:
            print(f"❌ No matching image for {filename}")
            continue

        image_file = matching_images[0]
        image_path = os.path.join(images_path, image_file)

        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except Exception as e:
            print(f"❌ Failed to open image {image_file}: {e}")
            continue

        mat = scipy.io.loadmat(mat_path)
        if "bbox_all" not in mat:
            print(f"⚠️ Skipped {filename}: 'bbox_all' not found.")
            continue

        bbox_all = mat["bbox_all"]
        yolo_annotations = []

        for box in bbox_all:
            x, y, w, h = box

            # Clip bounding box to image bounds (Option 2)
            original_box = (x, y, w, h)
            x = max(0, x)
            y = max(0, y)
            x2 = min(x + w, image_width)
            y2 = min(y + h, image_height)
            new_w = x2 - x
            new_h = y2 - y

            if new_w <= 0 or new_h <= 0:
                print(f"❌ Skipped invalid clipped box in {image_file}: original={original_box} new={(x, y, new_w, new_h)}")
                skipped_invalid_count += 1
                skipped_log.append(f"{image_file}: original={original_box} new={(x, y, new_w, new_h)}")
                continue

            if (w != new_w or h != new_h):
                print(f"⚠️ Clipped box in {image_file}: original={original_box} new={(x, y, new_w, new_h)}")
                clipped_count += 1
                clipped_log.append(f"{image_file}: original={original_box} new={(x, y, new_w, new_h)}")

            x_center = (x + new_w / 2) / image_width
            y_center = (y + new_h / 2) / image_height
            w_norm = new_w / image_width
            h_norm = new_h / image_height
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save annotations with the same base name as the image file (without extension)
        base_name = os.path.splitext(image_file)[0]
        txt_filename = os.path.join(output_folder, base_name + ".txt")

        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_annotations))

        print(f"✅ Saved: {txt_filename}")
        converted_count += 1

# Write summary log
with open(log_summary_file, "w") as log_file:
    log_file.write(f"Total files processed: {converted_count}\n")
    log_file.write(f"Clipped boxes: {clipped_count}\n")
    log_file.write(f"Skipped invalid boxes: {skipped_invalid_count}\n\n")

    if clipped_log:
        log_file.write("Clipped boxes detail:\n")
        for entry in clipped_log:
            log_file.write(entry + "\n")

    if skipped_log:
        log_file.write("\nSkipped invalid boxes detail:\n")
        for entry in skipped_log:
            log_file.write(entry + "\n")

print(f"\n✉️ Summary written to {log_summary_file}")
