import os

# === CONFIGURATION ===
thresholds = [0, 10, 20, 30, 40, 50]
base_threshold_dir = "/content/drive/MyDrive/small_object_detection/threshold_labels"
output_yaml_dir = "/content/drive/MyDrive/small_object_detection/configs"
os.makedirs(output_yaml_dir, exist_ok=True)

# === GENERATE YAMLs ===
for t in thresholds:
    for fold in range(5):
        yaml_filename = f"threshold_t{t}_fold{fold}.yaml"
        yaml_path = os.path.join(output_yaml_dir, yaml_filename)

        train_path = os.path.join(base_threshold_dir, f"t{t}", "folds", f"fold{fold}", "images", "train")
        val_path = os.path.join(base_threshold_dir, f"t{t}", "folds", f"fold{fold}", "images", "val")

        yaml_content = f"""train: {train_path}
val: {val_path}

nc: 4
names: ["fly", "fish", "honeybee", "seagull"]
"""

        with open('/home/jman/Documents/PROYECTO/Detección_particulas/código/python/Preprocessing/configs/'+yaml_filename, "w") as f:
            f.write(yaml_content)

print(f"✅ YAML generation complete. Saved in: {output_yaml_dir}")
