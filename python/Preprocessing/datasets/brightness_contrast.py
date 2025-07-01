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

def scan_and_write_stats(images_dir, output_csv='brightness_contrast.csv'):
    """
    Scans all image files in `images_dir`, computes brightness & contrast,
    and writes CSV with columns: image_path, brightness, contrast
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    rows = []

    for fname in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in exts:
            continue
        full_path = os.path.join(images_dir, fname)
        try:
            mean, std = compute_stats(full_path)
            rows.append((fname, mean, std))
        except Exception as e:
            print(f"⚠️  Failed {fname}: {e}")

    # Write CSV
    csv_path = os.path.join(images_dir, output_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'brightness', 'contrast'])
        for fname, mean, std in rows:
            writer.writerow([fname, f"{mean:.2f}", f"{std:.2f}"])

    print(f"✨ Wrote {len(rows)} entries to {csv_path}")

if __name__ == '__main__':
    # Adjust this to point at your single folder of images
    IMAGES_ROOT = './images'
    scan_and_write_stats(IMAGES_ROOT)
