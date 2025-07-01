import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV
CSV_PATH = os.path.join(os.getcwd(), 'brightness_contrast.csv')
df = pd.read_csv(CSV_PATH)

# 2. Quick look at your data
print(df.head())
print(df.describe())

# 3. Compute percentiles for contrast
pcts = df['contrast'].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
print("\nContraste percentiles:")
print(pcts)

# 4. Plot histograms
plt.figure()
df['brightness'].hist(bins=30)
plt.title('Distribución del brillo')
plt.xlabel('Intensidad Media (0 - 255)')
plt.ylabel('Cuenta de imágenes')
plt.show()

plt.figure()
df['contrast'].hist(bins=30)
plt.title('Distribución del contraste')
plt.xlabel('RMS Contraste (Std Dev) (Desviación estándar por píxel)')
plt.ylabel('Cuenta de imágenes')
plt.show()

# 5. Scatter: brightness vs. contrast
plt.figure()
plt.scatter(df['brightness'], df['contrast'], alpha=0.6)
plt.title('Brillo vs. Contraste')
plt.xlabel('Intensidad Media (0 -255)')
plt.ylabel('RMS Contraste (0 - 127 aprox.)')
plt.show()

# 6. Normalización y clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans

scaler = StandardScaler()
X      = scaler.fit_transform(df[['brightness', 'contrast']])
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 7. Scatter coloreado por cluster
plt.figure()
plt.scatter(
    df['brightness'],
    df['contrast'],
    c=df['cluster'],
    cmap='viridis',
    alpha=0.6
)
plt.title('Brillo vs. Contraste (coloreado por cluster)')
plt.xlabel('Intensidad Media (0–255)')
plt.ylabel('RMS Contraste (0–127 aprox.)')
plt.show()

# Centroides en unidades originales
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
# Dibujar centroides sobre el scatter coloreado
plt.figure()
plt.scatter(df['brightness'], df['contrast'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='red', label='Centroides')
plt.title('Brillo vs. Contraste con Centroides de Cluster')
plt.xlabel('Intensidad Media (0–255)')
plt.ylabel('RMS Contraste (0–127 aprox.)')
plt.legend()
plt.show()


counts  = df['cluster'].value_counts().sort_index()
percent = (counts / len(df) * 100).round(1)
print("Tamaños por cluster:\n", counts)
print("Porcentajes por cluster:\n", percent)
