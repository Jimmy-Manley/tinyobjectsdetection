import scipy.io
import os

# Load the MATLAB file
mat = scipy.io.loadmat('bbox_all_079.mat')
bbox_all = mat['bbox_all']  # shape: (30, 4)

# Image size (update if needed)
image_width = 640
image_height = 480

# YOLO formatted annotations
yolo_annotations = []
for box in bbox_all:
    x, y, w, h = box
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height
    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

# Save as .txt file
output_path = "labels/bbox_all_079.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    f.write("\n".join(yolo_annotations))

print(f"Saved YOLO annotations to: {output_path}")
