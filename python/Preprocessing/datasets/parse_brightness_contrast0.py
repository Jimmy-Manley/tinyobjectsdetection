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
