import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
import hdbscan
from Bio import SeqIO

ref_cluster_file = "/eukaryote/output/combined_clusters.csv"

embeddings_16S_file = "/ribosome_RNA/data/16S_embeddings_2k.npy"
embeddings_euk_file = "/eukaryote/data/SSU_eukaryote_embeddings.npy"

headers_16S_file = "/ribosome_RNA/data/16S_kmers_fullheader.txt"
headers_euk_file = "/eukaryote/data/SSU_eukaryote_rRNA.txt"

query_fasta_file = "/ribosome_RNA/data/new_sequences.fasta"
output_csv = "/eukaryote/output/query_only_cluster_assignment.csv"

emb_16S = np.load(embeddings_16S_file)
emb_euk = np.load(embeddings_euk_file)

headers_16S = [line.strip()[1:] for line in open(headers_16S_file) if line.startswith(">")]
headers_euk = [line.strip()[1:] for line in open(headers_euk_file) if line.startswith(">")]

headers_16S = headers_16S[:emb_16S.shape[0]]
headers_euk = headers_euk[:emb_euk.shape[0]]

ref_embeddings = np.vstack([emb_16S, emb_euk])
ref_headers = headers_16S + headers_euk
ref_sources = ["16S"] * len(headers_16S) + ["Eukaryote"] * len(headers_euk)

print("Reference embeddings shape:", ref_embeddings.shape)

ref_clusters = pd.read_csv(ref_cluster_file)
assert len(ref_clusters) == len(ref_embeddings), "Mismatch between clusters and embeddings!"

query_headers = [rec.id for rec in SeqIO.parse(query_fasta_file, "fasta")]
query_embeddings = np.random.normal(size=(len(query_headers), ref_embeddings.shape[1]))

print("Query sequences loaded:", len(query_headers))

clusterer = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=8, prediction_data=True)
clusterer.fit(ref_embeddings)

query_labels, strengths = hdbscan.approximate_predict(clusterer, query_embeddings)

query_df = pd.DataFrame({
    "Query_Header": query_headers,
    "Assigned_Cluster": query_labels,
    "Probability": strengths
})
query_df.to_csv(output_csv, index=False)
print(f"Query cluster assignments saved to {output_csv}")

umap_3d = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=3, random_state=42)
all_embeddings = np.vstack([ref_embeddings, query_embeddings])
all_umap = umap_3d.fit_transform(all_embeddings)

ref_plot_df = pd.DataFrame({
    "UMAP1": all_umap[:len(ref_embeddings), 0],
    "UMAP2": all_umap[:len(ref_embeddings), 1],
    "UMAP3": all_umap[:len(ref_embeddings), 2],
    "Cluster": ref_clusters["Cluster"],
    "Source": ref_sources
})

query_plot_df = pd.DataFrame({
    "UMAP1": all_umap[len(ref_embeddings):, 0],
    "UMAP2": all_umap[len(ref_embeddings):, 1],
    "UMAP3": all_umap[len(ref_embeddings):, 2],
    "Cluster": query_labels,
    "Source": ["Query"] * len(query_headers)
})

plot_df = pd.concat([ref_plot_df, query_plot_df], ignore_index=True)

symbol_map = {
    "16S": "circle",
    "Eukaryote": "square",
    "Query": "diamond"
}

clusters = sorted(plot_df["Cluster"].unique())

fig = go.Figure()

for cluster in clusters:
    for source in symbol_map.keys():
        subset = plot_df[(plot_df["Cluster"] == cluster) & (plot_df["Source"] == source)]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=subset["UMAP1"], y=subset["UMAP2"], z=subset["UMAP3"],
            mode="markers",
            marker=dict(
                size=4,
                symbol=symbol_map[source],
                color=f"hsla({(cluster*47)%360},70%,50%,0.9)"  # unique color per cluster
            ),
            name=f"Cluster {cluster} – {source}",
            legendgroup=f"Cluster {cluster}",
            showlegend=True
        ))

fig.update_layout(
    title="3D UMAP of Sequences (Clusters + Databases)",
    scene=dict(
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        zaxis_title="UMAP3"
    ),
    legend=dict(
        itemsizing="constant",
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=1.02
    )
)

fig.show()