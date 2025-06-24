import os
import shutil
from sklearn.model_selection import KFold

# === CONFIGURATION ===
base_dir = 'threshold_labels'  # Path where t10, t20, ..., t50 are located
thresholds = ['t10', 't20', 't30', 't40', 't50']
num_folds = 5
random_seed = 42

# === GET MASTER IMAGE LIST FROM t10 ===
t10_images_dir = os.path.join(base_dir, 't10', 'images', 'train')
#images = [f for f in os.listdir(t10_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
images = [f for f in os.listdir() if f.endswith('.jpg') or f.endswith('.png')]

images.sort()

#kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
#fold_indices = list(kf.split(images))  # Save once for reuse

# === APPLY TO EACH THRESHOLD FOLDER ===
for threshold in thresholds:
    print(f"\n🔄 Processing threshold: {threshold}")
    image_dir = os.path.join(base_dir, threshold, 'images', 'train')
    label_dir = os.path.join(base_dir, threshold, 'labels', 'train')
    output_base = os.path.join(base_dir, threshold, 'folds')

    # Option2 statistically correct
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    fold_indices = list(kf.split(images))  # Save every time for statistically correct option 


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

print("\n✅ All threshold splits created with consistent 5-folds.")
