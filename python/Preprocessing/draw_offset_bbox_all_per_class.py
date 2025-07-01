import os
import cv2
import scipy.io
import numpy as np
from pathlib import Path

# === CONFIGURACIÓN ===
base_dir = Path("/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive (1)/Small Object dataset/train")
output_dir = Path("./annotated_offset_per_class")
output_dir.mkdir(parents=True, exist_ok=True)

# === OFFSET INDIVIDUAL POR CLASE ===
offsets_per_class = {
    "fly": (-7, -7),
    "fish": (-10, -10),
    "seagull": (-12, -12),
    "honeybee": (-17, -17)
}

# === FUNCIONES ===
def draw_box(img, x, y, w, h, color, label):
    x, y, w, h = map(int, [round(x), round(y), round(w), round(h)])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def process_class(class_name, offset_x, offset_y):
    class_path = base_dir / class_name
    img_dir = class_path / "img"
    mat_dir = class_path / "gt-bbox"
    
    for mat_file in sorted(mat_dir.glob("bbox_all_*.mat")):
        index = mat_file.stem.split("_")[-1]
        img_file = img_dir / f"{class_name}{index}.jpg"
        
        if not img_file.exists():
            print(f"⚠️ Imagen no encontrada: {img_file}")
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            print(f"❌ Error al leer imagen: {img_file}")
            continue

        try:
            mat = scipy.io.loadmat(str(mat_file))
            if "bbox_all" not in mat:
                print(f"⚠️ 'bbox_all' no encontrado en {mat_file.name}")
                continue

            bboxes = mat["bbox_all"]
            for i, box in enumerate(bboxes):
                x, y, w, h = map(float, box)
                x -= 1  # MATLAB 1-based a 0-based
                y -= 1
                x += offset_x
                y += offset_y
                draw_box(image, x, y, w, h, (255, 0, 255), f"{class_name}_GT")

            out_class_dir = output_dir / class_name
            out_class_dir.mkdir(exist_ok=True)
            out_path = out_class_dir / img_file.name
            cv2.imwrite(str(out_path), image)
            print(f"✅ Guardado: {out_path}")

        except Exception as e:
            print(f"❌ Error en {mat_file.name}: {e}")

# === PROCESAR TODAS LAS CLASES CON SUS OFFSETS ===
for cls in ["fly", "fish", "seagull", "honeybee"]:
    ox, oy = offsets_per_class.get(cls, (0, 0))
    process_class(cls, ox, oy)
