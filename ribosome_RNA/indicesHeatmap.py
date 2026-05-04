import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

cluster_csv_file = OUTPUTS_DIR / "16S_clusters_hdbscan.csv"
heatmap_output = OUTPUTS_DIR / "cluster_abundance_heatmap.png"

df = pd.read_csv(cluster_csv_file)
print("CSV loaded. Number of sequences:", df.shape[0])

cluster_counts = df['Cluster'].value_counts().sort_index()
print("\nNumber of sequences per cluster:")
print(cluster_counts)

counts = cluster_counts.values
shannon_index = entropy(counts)
print(f"\nShannon diversity index: {shannon_index:.4f}")

simpson_index = 1 - np.sum((counts / counts.sum())**2)
print(f"Simpson diversity index: {simpson_index:.4f}")

cluster_id = 0
cluster_sequences = df[df['Cluster'] == cluster_id]['Header'].tolist()
print(f"\nFirst 10 sequences in cluster {cluster_id}:")
print(cluster_sequences[:10])

noise_sequences = df[df['Cluster'] == -1]['Header'].tolist()
print(f"\nNumber of noise sequences: {len(noise_sequences)}")
print("First 10 noise sequences:")
print(noise_sequences[:10])

heatmap_data = pd.DataFrame(cluster_counts).T
heatmap_data.index = ["Abundance"]

plt.figure(figsize=(12,4))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Cluster Abundance Heatmap")
plt.xlabel("Cluster ID")
plt.ylabel("")
plt.tight_layout()
plt.savefig(heatmap_output)
plt.show()
print(f"Heatmap saved to {heatmap_output}")
