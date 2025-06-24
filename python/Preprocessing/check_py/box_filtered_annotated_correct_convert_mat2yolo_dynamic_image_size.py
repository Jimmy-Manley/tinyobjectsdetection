import scipy.io
import os
from PIL import Image
import re

# Configuration
annotations_folder = "/path/to/gt-bbox/"  # Replace with your path
images_path = "/path/to/img/"

output_folder = "labels_fly"
log_summary_file = "fly_conversion_summary.log"

# Threshold to ignore boxes with small visible area
visible_ratio_threshold = 0.3

os.makedirs(output_folder, exist_ok=True)

# Extract ID from file name
def extract_id(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

# Summary counters and logs
clipped_count = 0
skipped_invalid_count = 0
converted_count = 0
clipped_log = []
skipped_log = []

# Loop through all .mat files
for filename in os.listdir(annotations_folder):
    if not filename.endswith(".mat"):
        continue

    file_id = extract_id(filename)
    if not file_id:
        print(f"⚠️ Skipped {filename}: no numeric ID found.")
        continue

    mat_path = os.path.join(annotations_folder, filename)
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
        original_box = (x, y, w, h)

        # Clip bounding box
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, image_width)
        y2 = min(y + h, image_height)
        new_w = x2 - x1
        new_h = y2 - y1

        if new_w <= 0 or new_h <= 0:
            skipped_invalid_count += 1
            skipped_log.append(f"{image_file}: invalid area - original={original_box} new={(x1, y1, new_w, new_h)}")
            continue

        visible_area = new_w * new_h
        original_area = w * h
        if visible_area / original_area < visible_ratio_threshold:
            skipped_invalid_count += 1
            skipped_log.append(f"{image_file}: <30% visible - original={original_box} visible_ratio={visible_area / original_area:.2f}")
            continue

        if (w != new_w or h != new_h):
            clipped_count += 1
            clipped_log.append(f"{image_file}: clipped - original={original_box} new={(x1, y1, new_w, new_h)}")

        x_center = (x1 + new_w / 2) / image_width
        y_center = (y1 + new_h / 2) / image_height
        w_norm = new_w / image_width
        h_norm = new_h / image_height
        yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Save YOLO annotation file
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
