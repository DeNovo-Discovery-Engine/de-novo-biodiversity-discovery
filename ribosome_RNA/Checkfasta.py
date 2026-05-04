import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import KNeighborsClassifier
import umap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

reference_clusters_csv = OUTPUTS_DIR / "16S_clusters_hdbscan.csv"
reference_embeddings_file = DATA_DIR / "16S_embeddings_2k.npy"
new_fasta_file = DATA_DIR / "new_sequences.fasta"
output_csv = OUTPUTS_DIR / "new_sequences_knn_clustered.csv"

model_name = "armheb/DNA_bert_6"
device = "cuda" if torch.cuda.is_available() else "cpu"

n_neighbors = 6

umap_neighbors = 12
umap_min_dist = 0.05
umap_components = 3

ref_cluster_df = pd.read_csv(reference_clusters_csv)
ref_embeddings = np.load(reference_embeddings_file)
ref_labels = ref_cluster_df['Cluster'].to_numpy()

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name)
model.eval()
model.to(device)

def embed_sequence(seq):
    tokens = tokenizer(seq, return_tensors='pt', truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        out = model(**tokens)
    emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

new_headers = []
new_sequences = []

for record in SeqIO.parse(new_fasta_file, "fasta"):
    seq = str(record.seq).upper()
    if set(seq).issubset({'A','C','G','T','N'}):
        new_sequences.append(seq)
        new_headers.append(record.id)

new_embeddings = np.array([embed_sequence(seq) for seq in new_sequences])

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(ref_embeddings, ref_labels)

predicted_clusters = knn.predict(new_embeddings)
cluster_probabilities = knn.predict_proba(new_embeddings).max(axis=1)

output_df = pd.DataFrame({
    "Header": new_headers,
    "Predicted_Cluster": predicted_clusters,
    "Membership_Probability": cluster_probabilities
})
output_df.to_csv(output_csv, index=False)
print(f"New sequences clustered and saved to: {output_csv}")

all_embeddings = np.vstack([ref_embeddings, new_embeddings])
all_labels = np.concatenate([ref_labels, predicted_clusters])
all_headers = list(ref_cluster_df['Header']) + new_headers
source = ['Reference']*len(ref_embeddings) + ['New']*len(new_embeddings)

reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                    n_components=3, random_state=42)
embedding_3d = reducer.fit_transform(all_embeddings)

vis_df = pd.DataFrame({
    "UMAP_1": embedding_3d[:,0],
    "UMAP_2": embedding_3d[:,1],
    "UMAP_3": embedding_3d[:,2],
    "Cluster": all_labels.astype(str),
    "Header": all_headers,
    "Source": source
})

fig = px.scatter_3d(
    vis_df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
    color="Cluster", symbol="Source",
    hover_name="Header",
    title="3D UMAP of Reference + New Sequences (KNN)"
)
fig.update_traces(marker=dict(size=4))
fig.show()

cluster_counts = pd.Series(predicted_clusters).value_counts().sort_index()
heatmap_df = pd.DataFrame(cluster_counts).T

plt.figure(figsize=(12,2))
sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Number of New Sequences per Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Count")
plt.show()

novel_threshold = 0.5
novel_sequences = output_df[output_df['Membership_Probability'] < novel_threshold]
print(f"Number of potential novel sequences: {len(novel_sequences)}")
