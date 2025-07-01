import os
import cv2
import scipy.io

# === CONFIGURATION ===
images_dir = "/.img"           # e.g., /home/user/dataset/img
labels_dir = "./gt-bbox"       # e.g., /home/user/dataset/gt-bbox
output_dir = "./annotated_images_mat" # Output folder

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(images_dir):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(images_dir, filename)
    label_path = os.path.join(labels_dir, filename.replace(".jpg", ".mat"))

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load {image_path}")
        continue

    if not os.path.exists(label_path):
        print(f"⚠️ No label found for {filename}")
        continue

    try:
        mat = scipy.io.loadmat(label_path)

        # === Adjust this line to match your .mat structure ===
        bboxes = mat.get("bbox")  # replace 'bbox' with correct key if needed

        if bboxes is None:
            print(f"⚠️ No 'bbox' key found in {label_path}")
            continue

        for box in bboxes:
            # Assuming box format is [x, y, width, height]
            x, y, w, h = map(int, box)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "obj", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, image)
        print(f"✅ Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error processing {label_path}: {e}")
