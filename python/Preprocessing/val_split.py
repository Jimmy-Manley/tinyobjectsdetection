import os
import random
import shutil
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
base = Path('datasets/small_objects')
img_dir = base / 'images/train'
lbl_dir = base / 'labels/train'

val_img_dir = base / 'images/val'
val_lbl_dir = base / 'labels/val'

# Make val folders
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# List all image files
images = list(img_dir.glob("*.jpg"))
val_size = int(len(images) * 0.2)

# Randomly select validation images
val_images = random.sample(images, val_size)

# Move files
for img_path in val_images:
    lbl_path = lbl_dir / (img_path.stem + ".txt")

    shutil.move(str(img_path), str(val_img_dir / img_path.name))
    shutil.move(str(lbl_path), str(val_lbl_dir / lbl_path.name))

print(f"Moved {val_size} image-label pairs to validation set.")
