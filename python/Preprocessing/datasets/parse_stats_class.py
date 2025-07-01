import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV de estadísticas por clase
df = pd.read_csv('stats_by_class.csv')

# Barra de brillo medio por clase
plt.figure()
df.plot(x='class', y='brightness_mean', kind='bar', legend=False)
plt.ylabel('Brillo medio')
plt.title('Brillo medio por clase')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Barra de contraste medio por clase
plt.figure()
df.plot(x='class', y='contrast_mean', kind='bar', legend=False)
plt.ylabel('Contraste RMS medio')
plt.title('Contraste RMS medio por clase')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Histograma de brillo (todas las imágenes)
indiv = pd.read_csv('brightness_contrast.csv')
plt.figure()
indiv['brightness'].hist(bins=20)
plt.xlabel('Brillo')
plt.ylabel('Frecuencia')
plt.title('Distribución de brillo')
plt.tight_layout()
plt.show()

# Histograma de contraste (todas las imágenes)
plt.figure()
indiv['contrast'].hist(bins=20)
plt.xlabel('Contraste RMS')
plt.ylabel('Frecuencia')
plt.title('Distribución de contraste')
plt.tight_layout()
plt.show()
