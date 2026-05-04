import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

embeddings_file = DATA_DIR / "16S_embeddings_2k.npy"
clusters_csv = OUTPUTS_DIR / "16S_clusters_hdbscan.csv"

embeddings = np.load(embeddings_file)
cluster_df = pd.read_csv(clusters_csv)
cluster_labels = cluster_df["Cluster"].to_numpy()

mask = cluster_labels != -1
filtered_embeddings = embeddings[mask]
filtered_labels = cluster_labels[mask]

if len(np.unique(filtered_labels)) > 1:
    score = silhouette_score(filtered_embeddings, filtered_labels, metric='cosine')
    print("Silhouette score (excluding noise):", score)
else:
    print("Not enough clusters to compute silhouette score")
