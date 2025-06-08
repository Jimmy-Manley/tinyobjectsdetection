import os
import shutil
from sklearn.model_selection import KFold

# === CONFIGURATION ===
base_dir = 'threshold_labels'  # Base path (reuse folder name for consistency)
source = 'original'  # This folder must contain images/train and labels/train
num_folds = 5
random_seed = 42

# === GET MASTER IMAGE LIST FROM t10 ===
reference_dir = os.path.join(base_dir, 't10', 'images', 'train')
images = [f for f in os.listdir(reference_dir) if f.endswith('.jpg') or f.endswith('.png')]
images.sort()

kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
fold_indices = list(kf.split(images))  # Save once for reuse

# === APPLY TO ORIGINAL DATASET ===
print(f"\n🔄 Processing original (non-thresholded) dataset")
image_dir = os.path.join(base_dir, source, 'images', 'train')
label_dir = os.path.join(base_dir, source, 'labels', 'train')
output_base = os.path.join(base_dir, source, 'folds')

for fold, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"  Creating Fold {fold}...")
    fold_dir = os.path.join(output_base, f'fold{fold}')
    for split in ['train', 'val']:
        os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)

    for split_name, indices in zip(['train', 'val'], [train_idx, val_idx]):
        for i in indices:
            img_name = images[i]
            lbl_name = os.path.splitext(img_name)[0] + '.txt'

            # Copy image and label
            shutil.copy(os.path.join(image_dir, img_name), os.path.join(fold_dir, 'images', split_name, img_name))
            shutil.copy(os.path.join(label_dir, lbl_name), os.path.join(fold_dir, 'labels', split_name, lbl_name))

print("\n✅ 5-fold splits created for the original dataset with matching folds.")
