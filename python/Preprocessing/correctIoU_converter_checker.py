import os
import scipy.io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Paths
annotations_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/gt-bbox/"
images_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/train/fly/img/"
labels_folder = "/home/jman/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/labels_fly/"

viz_output_folder = "bbox_viz"
os.makedirs(viz_output_folder, exist_ok=True)

log_file = open("bbox_matching_log.txt", "w")

# IOU threshold for visual marking (low values mean potential issues)
iou_threshold = 0.99

def calculate_iou(box1, box2):
    # box = [x_center, y_center, w, h]
    def to_corners(b):
        x1 = b[0] - b[2] / 2
        y1 = b[1] - b[3] / 2
        x2 = b[0] + b[2] / 2
        y2 = b[1] + b[3] / 2
        return x1, y1, x2, y2

    b1 = to_corners(box1)
    b2 = to_corners(box2)

    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Loop through all .mat files
for mat_file in os.listdir(annotations_folder):
    if not mat_file.startswith("bbox_all_") or not mat_file.endswith(".mat"):
        continue

    img_id = mat_file.replace("bbox_all_", "").replace(".mat", "")
    image_name = f"fly{img_id}.jpg"
    label_name = f"fly{img_id}.txt"

    image_path = os.path.join(images_folder, image_name)
    label_path = os.path.join(labels_folder, label_name)
    mat_path = os.path.join(annotations_folder, mat_file)

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    mat = scipy.io.loadmat(mat_path)
    if "bbox_all" not in mat:
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

    with open(label_path, "r") as f:
        txt_lines = [line.strip().split() for line in f.readlines()]
        txt_boxes = [[float(x) for x in line[1:]] for line in txt_lines if len(line) == 5]

    if len(converted) != len(txt_boxes):
        print(f"❌ Mismatch in box count: {image_name} | .mat: {len(converted)} vs .txt: {len(txt_boxes)}")
        log_file.write(f"[COUNT] {image_name}: .mat={len(converted)} vs .txt={len(txt_boxes)}\n")
        continue

    mismatch = False
    mismatched_indices = []
    for i in range(len(converted)):
        iou = calculate_iou(converted[i], txt_boxes[i])
        if iou < iou_threshold:
            mismatch = True
            mismatched_indices.append((i, iou))

    if not mismatch:
        print(f"✅ Match: {image_name}")
        log_file.write(f"[OK] {image_name}\n")
    else:
        print(f"❌ Mismatch in {image_name}: {len(mismatched_indices)} boxes below IoU {iou_threshold}")
        log_file.write(f"[MISMATCH] {image_name}: {len(mismatched_indices)} box mismatches\n")

        # Draw mismatches
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, iou in mismatched_indices:
            # Convert YOLO to box corners
            for color, box in [("red", converted[i]), ("blue", txt_boxes[i])]:
                x_c, y_c, w, h = box
                x1 = (x_c - w/2) * width
                y1 = (y_c - h/2) * height
                x2 = (x_c + w/2) * width
                y2 = (y_c + h/2) * height
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        img.save(os.path.join(viz_output_folder, f"{image_name.replace('.jpg', '_viz.jpg')}"))

log_file.close()
print("\n📝 Log written to bbox_matching_log.txt")
