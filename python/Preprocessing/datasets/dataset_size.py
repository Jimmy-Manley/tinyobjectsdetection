#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
PROJECT_ROOT = os.getcwd()                         # Ejecuta este script desde la raíz del proyecto
IMAGE_DIR    = os.path.join(PROJECT_ROOT, 'images')  # Carpeta con todas las imágenes
LABEL_DIR    = os.path.join(PROJECT_ROOT, 'labels')  # Carpeta con todas las anotaciones YOLO (.txt)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeo de IDs de clase a nombres (ajusta si cambian)
CLASS_MAP = {
    0: 'fly',
    1: 'fish',
    2: 'honeybee',
    3: 'seagull'
}

# ---------------- Recolectar métricas ----------------
records = []
for txt_path in glob.glob(os.path.join(LABEL_DIR, '*.txt')):
    name = os.path.splitext(os.path.basename(txt_path))[0]
    # Encontrar la imagen asociada
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = os.path.join(IMAGE_DIR, name + ext)
        if os.path.isfile(candidate):
            img_path = candidate
            break
    if img_path is None:
        continue
    
    # Dimensiones de la imagen
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    img_area = img_w * img_h
    
    # Leer anotaciones YOLO y calcular métricas por instancia
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, _, _, w_norm, h_norm = parts
            cls_id = int(cls_id)
            w_px = float(w_norm) * img_w
            h_px = float(h_norm) * img_h
            area_px = w_px * h_px
            area_rel = area_px / img_area
            records.append({
                'image': name + os.path.splitext(img_path)[1],
                'species': CLASS_MAP.get(cls_id, str(cls_id)),
                'width_px': w_px,
                'height_px': h_px,
                'area_px': area_px,
                'area_rel': area_rel
            })

# ---------------- DataFrame y CSV ----------------
df = pd.DataFrame(records)

# CSV detallado
csv_detailed = os.path.join(OUTPUT_DIR, 'instance_size_detailed.csv')
df.to_csv(csv_detailed, index=False)
print(f"🚀 Detalles guardados en: {csv_detailed}")

# Resumen por especie
summary = df.groupby('species').agg(
    count         = ('area_px', 'size'),
    avg_width_px  = ('width_px', 'mean'),
    avg_height_px = ('height_px', 'mean'),
    avg_area_px   = ('area_px', 'mean'),
    avg_area_rel  = ('area_rel', 'mean')
).reset_index()

csv_summary = os.path.join(OUTPUT_DIR, 'instance_size_summary.csv')
summary.to_csv(csv_summary, index=False)
print(f"📊 Resumen guardado en: {csv_summary}")

# Mostrar tabla de resumen
print("\n=== Resumen por especie ===")
print(summary)

# ---------------- Gráficos ----------------
# 1) Histograma de área relativa
plt.figure()
df['area_rel'].hist(bins=30)
plt.title('Distribución de Área Relativa')
plt.xlabel('Área inst. / Área imagen')
plt.ylabel('Número de instancias')
plt.savefig(os.path.join(OUTPUT_DIR, 'hist_area_rel.png'))
plt.close()

# 2) Barras de recuento por especie
plt.figure()
summary.set_index('species')['count'].plot(kind='bar')
plt.title('Instancias totales por especie')
plt.xlabel('Especie')
plt.ylabel('Recuento de instancias')
plt.savefig(os.path.join(OUTPUT_DIR, 'bar_count_species.png'))
plt.close()

# 3) Scatter ancho vs alto
plt.figure()
plt.scatter(df['width_px'], df['height_px'], alpha=0.5)
plt.title('Ancho vs Alto de Bounding Boxes')
plt.xlabel('Width (px)')
plt.ylabel('Height (px)')
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_wh_px.png'))
plt.close()

# ---------------- Estadísticas globales ----------------
overall = df[['width_px', 'height_px', 'area_px', 'area_rel']].mean().round(3)
print("\n🔍 Estadísticas generales promedio:")
print(f"  Ancho medio (px):        {overall['width_px']}")
print(f"  Alto medio (px):         {overall['height_px']}")
print(f"  Área media (px):         {overall['area_px']}")
print(f"  Área media relativa:     {overall['area_rel']}")

print(f"\n📈 Gráficos guardados en: {OUTPUT_DIR}")
