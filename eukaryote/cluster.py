import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd

embeddings_file = "/eukaryote/data/SSU_eukaryote_embeddings.npy"
kmer_file = "/eukaryote/data/SSU_eukaryote_rRNA.txt"
output_clusters_csv = "/eukaryote/output/SSU_eukaryote_hdbscan.csv"
output_novel_csv = "/eukaryote/output/SSU_potential_novel.csv"

min_cluster_size = 35
min_samples = 8
umap_neighbors = 30
umap_min_dist = 0.05

embeddings = np.load(embeddings_file)
print("Embeddings loaded:", embeddings.shape)

headers = [line.strip()[1:] for line in open(kmer_file) if line.startswith(">")]
headers = headers[:embeddings.shape[0]]

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    prediction_data=True
)
cluster_labels = clusterer.fit_predict(embeddings)
probabilities = clusterer.probabilities_

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = sum(cluster_labels == -1)
print(f"Clusters found: {n_clusters}")
print(f"Noise points (potential novel taxa): {n_noise}")

cluster_df = pd.DataFrame({
    "Header": headers,
    "Cluster": cluster_labels,
    "Confidence": probabilities
})
cluster_df.to_csv(output_clusters_csv, index=False)
print(f"Cluster assignments saved to {output_clusters_csv}")

novel_df = cluster_df[cluster_df["Cluster"] == -1]
novel_df.to_csv(output_novel_csv, index=False)
print(f"Potential novel taxa saved to {output_novel_csv}")

reducer = umap.UMAP(
    n_neighbors=umap_neighbors,
    min_dist=umap_min_dist,
    n_components=2,
    random_state=42
)
embeddings_2d = reducer.fit_transform(embeddings)
print("UMAP reduced embeddings shape:", embeddings_2d.shape)

plt.figure(figsize=(12, 8))
palette = plt.cm.tab20
unique_labels = set(cluster_labels)

for label in unique_labels:
    idx = cluster_labels == label
    color = "k" if label == -1 else palette(label % 20)
    label_name = f"Cluster {label}" if label != -1 else "Potential Novel Taxa"
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=6, c=[color], label=label_name)

plt.title("HDBSCAN Clusters (Reference Space) – Potential Novel Taxa Highlighted")
plt.xlabel("UMAP 1 (nonlinear projection)")
plt.ylabel("UMAP 2 (nonlinear projection)")
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()