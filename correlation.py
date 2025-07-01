import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load your CSV ===
file_path = "brightness_contrast.csv"  # Update path if needed
df = pd.read_csv(file_path)

# === Step 2: Calculate correlation ===
correlation_matrix = df[['brightness', 'contrast']].corr()
correlation_value = correlation_matrix.loc['brightness', 'contrast']

# === Step 3: Print correlation value ===
print(f"📊 Pearson correlation coefficient (brightness vs contrast): {correlation_value:.4f}")

# === Step 4: Plot heatmap ===
plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation between Brightness and Contrast")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Optional: save to file
plt.show()
