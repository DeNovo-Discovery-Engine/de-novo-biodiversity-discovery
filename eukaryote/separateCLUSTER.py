import pandas as pd
import numpy as np
import umap
import plotly.express as px

csv_16S = "/ribosome_RNA/outputs/16S_clusters_hdbscan.csv"
csv_euk = "/eukaryote/output/SSU_eukaryote_hdbscan.csv"

df_16S = pd.read_csv(csv_16S)
df_16S["Source"] = "16S"

df_euk = pd.read_csv(csv_euk)
df_euk["Source"] = "Eukaryote"

combined_df = pd.concat([df_16S, df_euk], ignore_index=True)

embeddings_16S = np.load("/ribosome_RNA/data/16S_embeddings_2k.npy")
embeddings_euk = np.load("/eukaryote/data/SSU_eukaryote_embeddings.npy")
combined_embeddings = np.vstack([embeddings_16S, embeddings_euk])

reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=3, random_state=42)
embeddings_3d = reducer.fit_transform(combined_embeddings)

combined_df["UMAP1"] = embeddings_3d[:, 0]
combined_df["UMAP2"] = embeddings_3d[:, 1]
combined_df["UMAP3"] = embeddings_3d[:, 2]

fig = px.scatter_3d(
    combined_df,
    x="UMAP1",
    y="UMAP2",
    z="UMAP3",
    color="Cluster",               
    symbol="Source",               
    hover_data=["Header", "Source", "Cluster", "Confidence"],
    opacity=0.9,                   
    color_continuous_scale=px.colors.qualitative.Set2  
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    title="Combined 3D Clusters of 16S and Eukaryotes",
    legend=dict(itemsizing="constant", traceorder="normal"),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
