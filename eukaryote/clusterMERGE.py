import numpy as np
import pandas as pd
import hdbscan
import umap
import matplotlib.pyplot as plt

embeddings_16S_file = "/ribosome_RNA/data/16S_embeddings_2k.npy"
kmer_16S_file = "/ribosome_RNA/data/16S_kmers_fullheader.txt"

embeddings_euk_file = "/eukaryote/data/SSU_eukaryote_embeddings.npy"
kmer_euk_file = "/eukaryote/data/SSU_eukaryote_rRNA.txt"
output_csv = "/eukaryote/output/combined_clusters.csv"

min_cluster_size = 35
min_samples = 8

umap_neighbors = 30
umap_min_dist = 0.05

embeddings_16S = np.load(embeddings_16S_file)
headers_16S = [line.strip()[1:] for line in open(kmer_16S_file) if line.startswith(">")]
headers_16S = headers_16S[:embeddings_16S.shape[0]]

embeddings_euk = np.load(embeddings_euk_file)
headers_euk = [line.strip()[1:] for line in open(kmer_euk_file) if line.startswith(">")]
headers_euk = headers_euk[:embeddings_euk.shape[0]]

embeddings = np.vstack([embeddings_16S, embeddings_euk])
headers = headers_16S + headers_euk
sources = ["16S"]*len(headers_16S) + ["Eukaryote"]*len(headers_euk)

print("Combined embeddings shape:", embeddings.shape)

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
    "Source": sources,
    "Cluster": cluster_labels,
    "Confidence": probabilities
})
cluster_df.to_csv(output_csv, index=False)
print(f"Combined cluster assignments saved to {output_csv}")

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

source_markers = {"16S": "o", "Eukaryote": "^"}

for label in unique_labels:
    for source in source_markers:
        idx = (cluster_labels == label) & (np.array(sources) == source)
        if sum(idx) == 0:
            continue
        color = "k" if label == -1 else palette(label % 20)
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            s=30,
            c=[color],
            marker=source_markers[source],
            label=f"{source} Cluster {label}" if label != -1 else f"{source} Potential Novel"
        )

plt.title("HDBSCAN Clusters – Sources Highlighted")
plt.xlabel("UMAP 1 (nonlinear projection)")
plt.ylabel("UMAP 2 (nonlinear projection)")
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()