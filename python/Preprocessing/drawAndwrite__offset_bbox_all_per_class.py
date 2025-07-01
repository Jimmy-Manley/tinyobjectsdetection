import os
import cv2
import scipy.io
import numpy as np
from pathlib import Path

# === CONFIGURACIÓN ===
base_dir = Path("/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive (3)/Small Object dataset/train")
output_dir = Path("./annotated_offset_per_class")
output_dir.mkdir(parents=True, exist_ok=True)

offsets_per_class = {
    "fly":      (-7,  -7),
    "fish":     (-10, -10),
    "seagull":  (-12, -12),
    "honeybee": (-17, -17)
}

def draw_box(img, x, y, w, h, color, label):
    x, y, w, h = map(int, [round(x), round(y), round(w), round(h)])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(10, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def process_class(class_name, offset_x, offset_y):
    class_path = base_dir / class_name
    img_dir = class_path / "img"
    mat_dir = class_path / "gt-bbox"
    
    for mat_file in sorted(mat_dir.glob("bbox_all_*.mat")):
        index = mat_file.stem.split("_")[-1]
        img_file = img_dir / f"{class_name}{index}.jpg"
        if not img_file.exists():
            print(f"⚠️  Imagen no encontrada: {img_file}")
            continue

        # 1) Leer mat y aplicar offset
        try:
            data = scipy.io.loadmat(str(mat_file))
            if "bbox_all" not in data:
                print(f"⚠️  'bbox_all' no encontrado en {mat_file.name}")
                continue
            bboxes = data["bbox_all"].astype(float)
        except Exception as e:
            print(f"❌ Error cargando {mat_file.name}: {e}")
            continue

        # Ajuste de base MATLAB → base 0 y offset
        bboxes[:, 0] = bboxes[:, 0] - 1 + offset_x
        bboxes[:, 1] = bboxes[:, 1] - 1 + offset_y

        # 2) Sobrescribir el .mat con las nuevas cajas
        scipy.io.savemat(str(mat_file), {"bbox_all": bboxes})
        print(f"💾 Mat actualizado: {mat_file.name}")

        # 3) Cargar imagen y dibujar cajas para verificación
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"❌ Error al leer imagen: {img_file}")
            continue
        h_img, w_img = image.shape[:2]

        for i, (x, y, w_box, h_box) in enumerate(bboxes):
            # recorte a límites de la imagen
            x1 = max(0, x)
            y1 = max(0, y)
            draw_box(image, x1, y1, w_box, h_box,
                     color=(255, 0, 255),
                     label=f"{class_name}_GT")

        # 4) Guardar imagen anotada
        out_class_dir = output_dir / class_name
        out_class_dir.mkdir(exist_ok=True)
        out_path = out_class_dir / img_file.name
        cv2.imwrite(str(out_path), image)
        print(f"✅ Imagen anotada guardada: {out_path}")

# === PROCESAR TODAS LAS CLASES ===
for cls_name, (ox, oy) in offsets_per_class.items():
    process_class(cls_name, ox, oy)
