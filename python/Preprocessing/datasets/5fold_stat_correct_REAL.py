import os
import shutil
from sklearn.model_selection import KFold

# === CONFIGURATION ===
base_dir = 'no_threshold_labels'  # Path where t10, t20, ..., t50 are located
thresholds = ['t0']# ['t10', 't20', 't30', 't40', 't50']
num_folds = 5
random_seed = 42

# === LOOP THROUGH EACH THRESHOLD AND DO SPLITTING ===
for threshold in thresholds:
    print(f"\n🔄 Processing threshold: {threshold}")
    
    image_dir = os.path.join(base_dir, threshold, 'images', 'train')
    label_dir = os.path.join(base_dir, threshold, 'labels', 'train')
    output_base = os.path.join(base_dir, threshold, 'folds')

    # Get only valid image filenames for this threshold
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    images.sort()

    # New fold indices per threshold (Option 2: fresh KFold)
    #kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed + int(threshold[1:]))
    fold_indices = list(kf.split(images))

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

print("\n✅ All threshold splits created with statistically correct 5-folds.")
