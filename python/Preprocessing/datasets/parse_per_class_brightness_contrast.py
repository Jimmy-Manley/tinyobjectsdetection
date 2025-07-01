import os
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
import re

# Patrón para extraer clase: letras antes de los dígitos
CLASS_PATTERN = re.compile(r"^([a-zA-Z]+)")
# Extensiones de imagen válidas
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def compute_stats(image_path):
    """
    Retorna (brillo, contraste RMS) de la imagen en monocromo.
    - brillo: media de intensidades
    - contraste: desviación estándar
    """
    with Image.open(image_path) as img:
        gray = img.convert('L')      # escala de grises ('L')
        arr = np.array(gray, dtype=np.float32)
        return float(arr.mean()), float(arr.std())


def scan_by_filename_class(images_dir, output_csv='stats_by_class.csv'):
    """
    Escanea todas las imágenes en images_dir, extrae la clase del nombre,
    calcula brillo y contraste, y agrupa resultados por clase.
    """
    stats_per_class = defaultdict(list)

    for fname in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in EXTS:
            continue
        m = CLASS_PATTERN.match(fname)
        if not m:
            continue  # omitir si no coincide el prefijo de clase
        cls = m.group(1)  # e.g., 'fish', 'fly'
        full_path = os.path.join(images_dir, fname)
        try:
            mean, std = compute_stats(full_path)
            stats_per_class[cls].append((mean, std))
        except Exception as e:
            print(f"⚠️  Error en {fname}: {e}")

    # Escribir CSV con estadísticas por clase
    csv_path = os.path.join(images_dir, output_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'count',
                         'brightness_mean', 'brightness_std',
                         'contrast_mean', 'contrast_std'])
        for cls, vals in stats_per_class.items():
            arr = np.array(vals, dtype=np.float32)
            b_mean, b_std = arr[:, 0].mean(), arr[:, 0].std()
            c_mean, c_std = arr[:, 1].mean(), arr[:, 1].std()
            writer.writerow([cls,
                             len(vals),
                             f"{b_mean:.2f}",
                             f"{b_std:.2f}",
                             f"{c_mean:.2f}",
                             f"{c_std:.2f}"])

    print(f"✨ Estadísticas por clase guardadas en {csv_path}")


if __name__ == '__main__':
    # Ajusta esta ruta al directorio que contiene todas las imágenes
    IMAGES_ROOT = './images'
    scan_by_filename_class(IMAGES_ROOT)
