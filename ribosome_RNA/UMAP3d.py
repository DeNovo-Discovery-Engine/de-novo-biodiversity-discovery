import numpy as np
import pandas as pd
import umap
import hdbscan
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

embeddings_file = DATA_DIR / "16S_embeddings_2k.npy"
kmer_file = DATA_DIR / "16S_kmers_fullheader.txt"
output_csv = OUTPUTS_DIR / "16S_clusters_hdbscan.csv"

min_cluster_size = 30
min_samples = 5
umap_neighbors = 30
umap_min_dist = 0.1
umap_components = 3

embeddings = np.load(embeddings_file)
print("Embeddings loaded:", embeddings.shape)

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples)
cluster_labels = clusterer.fit_predict(embeddings)

print("Number of clusters found:", len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
print("Number of noise points (-1 labels):", sum(cluster_labels == -1))

headers = [line.strip()[1:] for line in open(kmer_file) if line.startswith(">")]
headers = headers[:embeddings.shape[0]]

cluster_df = pd.DataFrame({
    "Header": headers,
    "Cluster": cluster_labels
})
cluster_df.to_csv(output_csv, index=False)
print(f"Cluster assignments saved to {output_csv}")

reducer = umap.UMAP(
    n_neighbors=umap_neighbors,
    min_dist=umap_min_dist,
    n_components=3,
    random_state=42
)
embeddings_3d = reducer.fit_transform(embeddings)
print("UMAP reduced embeddings shape:", embeddings_3d.shape)

cluster_df["UMAP_1"] = embeddings_3d[:, 0]
cluster_df["UMAP_2"] = embeddings_3d[:, 1]
cluster_df["UMAP_3"] = embeddings_3d[:, 2]

fig = px.scatter_3d(
    cluster_df,
    x="UMAP_1",
    y="UMAP_2",
    z="UMAP_3",
    color=cluster_df["Cluster"].astype(str),
    hover_name="Header",
    title="3D UMAP Visualization of 16S Embeddings (HDBSCAN Clusters)",
    opacity=0.8
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(scene=dict(
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2',
    zaxis_title='UMAP 3'
))

fig.show()
