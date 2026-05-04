import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import umap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

new_fasta_file = DATA_DIR / "new_sequences.fasta"
k = 6
window_size = 512
step = 256
batch_size = 32

reference_embeddings_file = DATA_DIR / "16S_embeddings_2k.npy"
reference_labels_file = OUTPUTS_DIR / "16S_clusters_hdbscan.csv"

output_csv = OUTPUTS_DIR / "new_sequences_clustered.csv"
novel_csv = OUTPUTS_DIR / "new_sequences_novel.csv"

novelty_threshold = 0.75

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ref_embeddings = np.load(reference_embeddings_file)
ref_labels_df = pd.read_csv(reference_labels_file)
ref_labels = ref_labels_df['Cluster'].values

mask = ref_labels != -1
ref_embeddings_clean = ref_embeddings[mask]
ref_labels_clean = ref_labels[mask]

scaler = StandardScaler()
ref_embeddings_scaled = scaler.fit_transform(ref_embeddings_clean)

def get_kmers(sequence, k=6):
    sequence = sequence.upper()
    return [sequence[i:i+k] for i in range(len(sequence)-k+1) if "N" not in sequence[i:i+k]]

def chunk_kmers(kmer_list, window_size=512, step=256):
    return [kmer_list[i:i+window_size] for i in range(0, len(kmer_list), step)]

new_sequences = []
new_headers = []

for record in SeqIO.parse(new_fasta_file, "fasta"):
    kmers = get_kmers(str(record.seq), k)
    new_sequences.append(kmers)
    new_headers.append(record.description)

tokenizer = BertTokenizer.from_pretrained("armheb/DNA_bert_6")
model = BertModel.from_pretrained("armheb/DNA_bert_6")
model.eval()
model.to(device)

all_embeddings = []

for idx, seq_kmers in enumerate(new_sequences, start=1):
    windows = chunk_kmers(seq_kmers, window_size, step)
    window_embeddings = []

    for w in windows:
        seq_text = " ".join(w)
        inputs = tokenizer(seq_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze()
            window_embeddings.append(emb.cpu())

    seq_embedding = torch.stack(window_embeddings).mean(dim=0)
    all_embeddings.append(seq_embedding.numpy())

new_embeddings = np.stack(all_embeddings)
new_embeddings_scaled = scaler.transform(new_embeddings)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(ref_embeddings_scaled)

distances, indices = knn.kneighbors(new_embeddings_scaled)
similarities = 1 - distances

pred_labels = []
membership_probs = []

for i, neighbor_idxs in enumerate(indices):
    neighbor_clusters = ref_labels_clean[neighbor_idxs]
    counts = np.bincount(neighbor_clusters)
    cluster = np.argmax(counts)
    pred_labels.append(cluster)
    membership_probs.append(similarities[i].mean())

pred_labels_array = np.array(pred_labels)
membership_probs_array = np.array(membership_probs)
novel_mask = membership_probs_array < novelty_threshold

cluster_df = pd.DataFrame({
    "Header": new_headers,
    "Predicted_Cluster": pred_labels_array,
    "Membership_Probability": membership_probs_array
})
cluster_df.to_csv(output_csv, index=False)

novel_df = cluster_df[novel_mask]
novel_df.to_csv(novel_csv, index=False)

print(f"Cluster assignments saved to {output_csv}")
print(f"Novel / low-similarity sequences saved to {novel_csv}")

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=3, random_state=42)
emb_3d = reducer.fit_transform(np.vstack([ref_embeddings_scaled, new_embeddings_scaled]))

labels_combined = np.concatenate([ref_labels_clean, pred_labels_array])
df_umap = pd.DataFrame(emb_3d, columns=['UMAP1','UMAP2','UMAP3'])
df_umap['Cluster'] = labels_combined
df_umap['Type'] = ['Reference']*len(ref_embeddings_scaled) + ['New']*len(new_embeddings_scaled)

fig = px.scatter_3d(df_umap, x='UMAP1', y='UMAP2', z='UMAP3',
                    color='Cluster', symbol='Type',
                    hover_data=['Cluster','Type'], opacity=0.7)
fig.update_layout(title="3D UMAP of Reference + New Sequences", legend=dict(x=1.05, y=1))
fig.show()

heatmap_data = cluster_df.groupby("Predicted_Cluster").size().reset_index(name="Count")
plt.figure(figsize=(10,5))
sns.barplot(x="Predicted_Cluster", y="Count", data=heatmap_data, palette="tab20")
plt.title("Number of New Sequences per Predicted Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of sequences")
plt.show()
