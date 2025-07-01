import os
import shutil

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
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    samples = []
    for img_fn in os.listdir(IMG_DIR):
        if not img_fn.lower().endswith(('.jpg','jpeg','png')):
            continue
        label_fn  = os.path.splitext(img_fn)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_fn)
        if not os.path.exists(label_path):
            continue

        include = (thr == 0.0)
        if thr > 0.0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, x_c, y_c, w_n, h_n = parts
                    area_norm = float(w_n) * float(h_n)
                    if area_norm >= thr:
                        include = True
                        break

        if include:
            # Copiar imagen y etiqueta al dataset filtrado
            shutil.copy2(os.path.join(IMG_DIR, img_fn),
                         os.path.join(img_out, img_fn))
            shutil.copy2(label_path,
                         os.path.join(lbl_out, label_fn))
            samples.append(img_fn)

    print(f"Dataset {thr_str}: {len(samples)} muestras creadas en {out_dir}")

