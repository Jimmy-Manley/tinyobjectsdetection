import os
import csv
from PIL import Image
import numpy as np

def compute_stats(image_path):
    """
    Returns (mean, std) of grayscale intensities for the image.
    - mean = brightness
    - std  = RMS contrast
    """
    with Image.open(image_path) as img:
        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

def scan_and_write_stats(root_images_dir, output_csv='brightness_contrast.csv'):
    splits = ['train', 'val', 'test']
    exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    rows   = []

    for split in splits:
        split_dir = os.path.join(root_images_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if os.path.splitext(fname)[1].lower() in exts:
                full = os.path.join(split_dir, fname)
                rel  = os.path.join(split, fname)
                try:
                    mean, std = compute_stats(full)
                    rows.append((split, rel, mean, std))
                except Exception as e:
                    print(f"⚠️  Failed {rel}: {e}")

    # Write CSV
    csv_path = os.path.join(root_images_dir, output_csv)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'image_path', 'brightness', 'contrast'])
        for split, rel, mean, std in sorted(rows, key=lambda x: (x[0], x[3])):
            w.writerow([split, rel, f"{mean:.2f}", f"{std:.2f}"])

    print(f"✨ Wrote {len(rows)} entries to {csv_path}")

if __name__ == '__main__':
    import os
    # If this script lives in the images/ folder, auto-detect it:
    ROOT = '/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/yolo_dataset/images'
    scan_and_write_stats(ROOT)
