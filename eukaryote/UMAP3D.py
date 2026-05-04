import numpy as np
import pandas as pd
import hdbscan
import umap
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

embeddings_16S_file = "/ribosome_RNA/data/16S_embeddings_2k.npy"
kmer_16S_file = "/ribosome_RNA/data/16S_kmers_fullheader.txt"
 
embeddings_euk_file ="/eukaryote/data/SSU_eukaryote_embeddings.npy"
kmer_euk_file ="/eukaryote/data/SSU_eukaryote_rRNA.txt"

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

clusterer = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=8, prediction_data=True)
cluster_labels = clusterer.fit_predict(embeddings)
probabilities = clusterer.probabilities_

reducer_3d = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=3, random_state=42)
embeddings_3d = reducer_3d.fit_transform(embeddings)
print("3D UMAP shape:", embeddings_3d.shape)

df = pd.DataFrame({
    "UMAP1": embeddings_3d[:, 0],
    "UMAP2": embeddings_3d[:, 1],
    "UMAP3": embeddings_3d[:, 2],
    "Cluster": cluster_labels,
    "Source": sources,
    "Header": headers,
    "Confidence": probabilities
})

num_clusters = len(set(cluster_labels))
palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel  # good mix
cluster_color_map = {label: palette[i % len(palette)] if label != -1 else "#000000" 
                     for i, label in enumerate(set(cluster_labels))}

source_marker_map = {"16S": "circle", "Eukaryote": "diamond"}

fig = go.Figure()

for label in set(df["Cluster"]):
    for source in ["16S", "Eukaryote"]:
        subset = df[(df["Cluster"]==label) & (df["Source"]==source)]
        if subset.shape[0] == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=subset["UMAP1"],
            y=subset["UMAP2"],
            z=subset["UMAP3"],
            mode="markers",
            marker=dict(
                size=6,
                color=cluster_color_map[label],
                symbol=source_marker_map[source],
                opacity=0.85,   
            ),
            name=f"{source} Cluster {label}" if label!=-1 else f"{source} Outliner",
            text=subset["Header"],
            hovertemplate="<b>%{text}</b><br>Cluster: %{name}<br>X: %{x:.2f}, Y: %{y:.2f}, Z: %{z:.2f}<extra></extra>"
        ))

fig.update_layout(
    scene=dict(
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        zaxis_title="UMAP3"
    ),
    legend=dict(itemsizing='constant'),
    title="3D UMAP – HDBSCAN Clusters",
    width=1000,
    height=800
)

fig.show()

cluster_df = pd.DataFrame({
    "Header": headers,
    "Source": sources,
    "Cluster": cluster_labels,
    "Confidence": probabilities
})

heatmap_data = cluster_df.pivot_table(index="Cluster", columns="Header", values="Confidence", fill_value=0)

plt.figure(figsize=(18, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Cluster Confidence'})
plt.title("HDBSCAN Cluster Confidence Heatmap")
plt.xlabel("Samples")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()