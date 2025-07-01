import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
PROJECT_ROOT = os.getcwd()                  # Ejecuta el script desde la carpeta raíz del proyecto
IMAGE_ROOT   = os.path.join(PROJECT_ROOT, 'images')  # images/<species>/*.jpg
LABEL_ROOT   = os.path.join(PROJECT_ROOT, 'labels')  # labels/<species>/*.txt
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPECIES_FOLDERS = [d for d in os.listdir(LABEL_ROOT)
                   if os.path.isdir(os.path.join(LABEL_ROOT, d))]

# ---------------- Recolectar métricas ----------------
records = []

for species in SPECIES_FOLDERS:
    species_label_dir = os.path.join(LABEL_ROOT, species)
    species_image_dir = os.path.join(IMAGE_ROOT, species)
    for txt_file in glob.glob(os.path.join(species_label_dir, '*.txt')):
        name = os.path.splitext(os.path.basename(txt_file))[0]
        
        # Buscar la imagen asociada
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            candidate = os.path.join(species_image_dir, name + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        
        # Dimensiones de la imagen
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        # Leer anotaciones YOLO: cls x_center y_center w_norm h_norm
        widths, heights, areas = [], [], []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, _, _, w_norm, h_norm = parts
                w_norm = float(w_norm)
                h_norm = float(h_norm)
                widths.append(w_norm)
                heights.append(h_norm)
                areas.append(w_norm * h_norm)
        
        if not areas:
            continue
        
        # Calcular métricas por imagen
        records.append({
            'species': species,
            'image': name,
            'instances': len(areas),
            'avg_width_norm':  np.mean(widths),
            'avg_height_norm': np.mean(heights),
            'avg_area_norm':   np.mean(areas),
            'min_area_norm':   np.min(areas),
            'max_area_norm':   np.max(areas),
        })

# ---------------- Crear DataFrame y guardar CSV ----------------
df = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_DIR, 'instance_size_metrics.csv')
df.to_csv(csv_path, index=False)
print(f"🚀 Saved detailed log to {csv_path}")

# Resumen por especie
summary = df.groupby('species').agg(
    image_count    = ('image',         'count'),
    avg_instances  = ('instances',     'mean'),
    avg_width_norm = ('avg_width_norm','mean'),
    avg_height_norm= ('avg_height_norm','mean'),
    avg_area_norm  = ('avg_area_norm', 'mean'),
).reset_index()
summary_csv = os.path.join(OUTPUT_DIR, 'instance_size_summary.csv')
summary.to_csv(summary_csv, index=False)
print(f"📊 Saved summary log to {summary_csv}")

# ---------------- Visualizaciones y guardar gráficos ----------------
# 1) Histograma de áreas normalizadas
plt.figure()
df['avg_area_norm'].hist(bins=30)
plt.title('Distribución Área Promedio Normalizada')
plt.xlabel('Área promedio (w*h normalizado)')
plt.ylabel('Número de imágenes')
hist_path = os.path.join(OUTPUT_DIR, 'hist_area_norm.png')
plt.savefig(hist_path)
plt.close()
print(f"📈 Saved histogram to {hist_path}")

# 2) Barras de instancias promedio por especie
plt.figure()
summary.set_index('species')['avg_instances'].plot(kind='bar')
plt.title('Instancias promedio por imagen (por especie)')
plt.xlabel('Especie')
plt.ylabel('Número promedio de instancias')
bar_path = os.path.join(OUTPUT_DIR, 'bar_avg_instances.png')
plt.savefig(bar_path)
plt.close()
print(f"📈 Saved bar chart to {bar_path}")

# 3) Scatter ancho vs alto normalizado
plt.figure()
plt.scatter(df['avg_width_norm'], df['avg_height_norm'], alpha=0.6)
plt.title('Ancho vs Alto Promedio (normalizado)')
plt.xlabel('Ancho medio (w_norm)')
plt.ylabel('Alto medio (h_norm)')
scatter_path = os.path.join(OUTPUT_DIR, 'scatter_wh_norm.png')
plt.savefig(scatter_path)
plt.close()
print(f"📈 Saved scatter plot to {scatter_path}")

# ---------------- Estadísticas globales ----------------
overall = df[['instances','avg_width_norm','avg_height_norm','avg_area_norm']].mean().round(3)
print("🔍 Estadísticas generales:")
print(f"  • Instancias promedio por imagen: {overall['instances']}")
print(f"  • Ancho medio (normalizado):       {overall['avg_width_norm']}")
print(f"  • Alto medio (normalizado):        {overall['avg_height_norm']}")
print(f"  • Área promedio (normalizado):     {overall['avg_area_norm']}")

# Fin del script
