import os
import shutil
from PIL import Image, ImageDraw

# === CONFIGURACIÓN ===
BASE_DIR     = './'           # Carpeta que contiene 'img' y 'labels'
IMG_DIR      = os.path.join(BASE_DIR, 'img')
LABELS_DIR   = os.path.join(BASE_DIR, 'labels')
OUTPUT_BASE  = os.path.join(BASE_DIR, '5_fold_vis_thresh')
THRESHOLDS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Umbrales de visibilidad

for thr in THRESHOLDS:
    thr_str = f"t{int(thr*100)}"
    out_dir = os.path.join(OUTPUT_BASE, thr_str)
    img_out  = os.path.join(out_dir, 'img')
    lbl_out  = os.path.join(out_dir, 'labels')
    discarded_out = os.path.join(out_dir, 'discarded')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    os.makedirs(discarded_out, exist_ok=True)

    samples = []
    for img_fn in os.listdir(IMG_DIR):
        if not img_fn.lower().endswith(('.jpg','jpeg','png')):
            continue
        label_fn  = os.path.splitext(img_fn)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_fn)
        if not os.path.exists(label_path):
            continue

        # Leer todas las anotaciones YOLO
        passed_boxes = []
        discarded_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, w_n, h_n = parts
                x_c, y_c, w_n, h_n = map(float, (x_c, y_c, w_n, h_n))
                area_norm = w_n * h_n
                if thr == 0.0 or area_norm >= thr:
                    passed_boxes.append((cls, x_c, y_c, w_n, h_n))
                else:
                    discarded_boxes.append((cls, x_c, y_c, w_n, h_n))

        # Decidir inclusión del ejemplo
        if not passed_boxes:
            # Si no hay ninguna caja válida, se descarta todo el ejemplo
            continue

        # Copiar imagen y etiqueta al dataset filtrado
        shutil.copy2(os.path.join(IMG_DIR, img_fn),
                     os.path.join(img_out, img_fn))
        shutil.copy2(label_path,
                     os.path.join(lbl_out, label_fn))
        samples.append(img_fn)

        # Si hay cajas descartadas, generar imagen de verificación
        if discarded_boxes:
            img_path = os.path.join(IMG_DIR, img_fn)
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            w_img, h_img = img.size

            # Dibujar en rojo las cajas descartadas
            for cls, x_c, y_c, w_n, h_n in discarded_boxes:
                # convertir a coordenadas píxel
                w_px = w_n * w_img
                h_px = h_n * h_img
                x_px = x_c * w_img - w_px/2
                y_px = y_c * h_img - h_px/2
                rect = [x_px, y_px, x_px + w_px, y_px + h_px]
                draw.rectangle(rect, outline='red', width=2)

            discarded_path = os.path.join(discarded_out, img_fn)
            img.save(discarded_path)
            print(f"📌 Cajas descartadas en {thr_str}/{img_fn} guardadas en {discarded_path}")

    print(f"Dataset {thr_str}: {len(samples)} muestras creadas en {out_dir}")


