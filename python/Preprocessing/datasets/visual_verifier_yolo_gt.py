import os
import cv2

# === CONFIGURATION ===
images_dir = "./images"        # e.g., /home/user/images
labels_dir = "./labels"        # e.g., /home/user/labels
output_dir = "./annotated_images"     # Output for visual verification

# Image settings
image_ext = ".jpg"
label_ext = ".txt"
class_names = ["class_0", "class_1", "class_2"]  # Replace with your actual classes if needed

# === MAIN FUNCTION ===
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(images_dir):
    if not filename.endswith(image_ext):
        continue

    image_path = os.path.join(images_dir, filename)
    label_path = os.path.join(labels_dir, filename.replace(image_ext, label_ext))

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load {image_path}")
        continue

    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        print(f"⚠️ No label found for {filename}")
        continue

    with open(label_path, 'r') as f:
        for line in f:
            cls, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert from YOLO format (relative) to pixel values
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{int(cls)}"
            cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save or show
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image)
    print(f"✅ Saved: {out_path}")
