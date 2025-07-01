import os
import cv2
import scipy.io
from pathlib import Path

# === CONFIGURACIÓN GENERAL ===
base_dir = Path("/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive (1)/Small Object dataset/train")
output_txt_dir = Path("./yolo_labels_per_class_offset")
output_viz_dir = Path("./visual_yolo_labels_per_class_offset")
output_txt_dir.mkdir(parents=True, exist_ok=True)
output_viz_dir.mkdir(parents=True, exist_ok=True)

# === OFFSETS POR CLASE ===
offsets_per_class = {
    "fly":      (-7, -7),
    "fish":     (-12, -12),
    "seagull":  (-12, -12),
    "honeybee": (-17, -17)
}

# === IDs por clase para YOLO
class_id_map = {
    "fly": 0,
    "fish": 1,
    "seagull": 2,
    "honeybee": 3
}

# === FUNCIONES ===
def convert_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def draw_yolo_box(img, xc, yc, w, h, img_w, img_h, color, label):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def process_class_to_yolo_and_visualize(class_name, offset_x, offset_y, class_id):
    class_path = base_dir / class_name
    img_dir = class_path / "img"
    mat_dir = class_path / "gt-bbox"

    txt_out_class = output_txt_dir / class_name
    viz_out_class = output_viz_dir / class_name
    txt_out_class.mkdir(exist_ok=True)
    viz_out_class.mkdir(exist_ok=True)

    for mat_file in sorted(mat_dir.glob("bbox_all_*.mat")):
        index = mat_file.stem.split("_")[-1]
        img_file = img_dir / f"{class_name}{index}.jpg"
        txt_file = txt_out_class / f"{class_name}{index}.txt"
        viz_file = viz_out_class / img_file.name

        if not img_file.exists():
            print(f" Imagen no encontrada: {img_file}")
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            print(f"❌ Error al leer imagen: {img_file}")
            continue

        h_img, w_img = image.shape[:2]
        lines = []

        try:
            mat = scipy.io.loadmat(str(mat_file))
            if "bbox_all" not in mat:
                print(f" 'bbox_all' no encontrado en {mat_file.name}")
                continue

            bboxes = mat["bbox_all"]
            for box in bboxes:
                x, y, w, h = map(float, box)
                x -= 1  # 1-based a 0-based
                y -= 1
                x += offset_x
                y += offset_y

                xc, yc, wn, hn = convert_to_yolo(x, y, w, h, w_img, h_img)
                lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

                # Visualización
                draw_yolo_box(image, xc, yc, wn, hn, w_img, h_img, (0, 255, 0), f"{class_name}")

            # Guardar etiquetas YOLO
            with open(txt_file, 'w') as f:
                f.write("\n".join(lines))

            # Guardar imagen con cajas dibujadas
            cv2.imwrite(str(viz_file), image)
            print(f"✅ {class_name} -> {txt_file.name}")

        except Exception as e:
            print(f"❌ Error en {mat_file.name}: {e}")

# === EJECUTAR PARA TODAS LAS CLASES ===
for cls in offsets_per_class.keys():
    ox, oy = offsets_per_class[cls]
    cid = class_id_map[cls]
    process_class_to_yolo_and_visualize(cls, ox, oy, cid)
