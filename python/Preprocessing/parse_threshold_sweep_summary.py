import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
# Ruta a tu CSV de resumen (asegúrate de que existe en este directorio)
CSV_PATH = 'threshold_sweep_summary.csv'

# === 1) Cargar datos ===
df = pd.read_csv(CSV_PATH)

# === 2) Gráfico: total de cajas recortadas por clase ===
clipped_per_class = df.groupby('dataset')['clipped_boxes'].sum().sort_values()

plt.figure(figsize=(6,4))
clipped_per_class.plot(kind='bar')
plt.title('Total de cajas recortadas por clase')
plt.xlabel('Clase')
plt.ylabel('Número de cajas recortadas')
plt.tight_layout()
plt.savefig('clipped_per_class.png')
plt.show()

# === 3) Gráfico: total de cajas recortadas por umbral ===
clipped_per_threshold = df.groupby('threshold')['clipped_boxes'].sum().sort_index()

plt.figure(figsize=(6,4))
clipped_per_threshold.plot(kind='bar')
plt.title('Total de cajas recortadas por umbral de visibilidad')
plt.xlabel('Umbral de visibilidad')
plt.ylabel('Número de cajas recortadas')
plt.tight_layout()
plt.savefig('clipped_per_threshold.png')
plt.show()
