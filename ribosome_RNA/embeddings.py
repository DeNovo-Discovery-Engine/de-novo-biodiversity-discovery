'''import pandas as pd
import numpy as np
import umap
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

csv_16S = OUTPUTS_DIR / "16S_clusters_hdbscan.csv"
csv_euk = DATA_DIR / "SSU_eukaryote_hdbscan.csv"

df_16S = pd.read_csv(csv_16S)
df_16S["Source"] = "16S"

df_euk = pd.read_csv(csv_euk)
df_euk["Source"] = "Eukaryote"


combined_df = pd.concat([df_16S, df_euk], ignore_index=True)


embeddings_16S = np.load(DATA_DIR / "16S_embeddings_2k.npy")
embeddings_euk = np.load(DATA_DIR / "SSU_eukaryote_embeddings.npy")
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
    opacity=0.95,                  
    color_continuous_scale=px.colors.qualitative.Set2  
)


fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))


fig.update_layout(
    title="Combined 3D Clusters of 16S and Eukaryotes",
    legend=dict(
        title="Cluster / Source",
        traceorder="grouped",
        itemsizing="constant",
        x=1.05,   
        y=1,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="Black",
        borderwidth=1
    ),
    margin=dict(l=0, r=250, b=0, t=50)  
)

fig.show()'''


import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

kmer_file = DATA_DIR / "16S_kmers_fullheader.txt"
num_sequences = 5000
batch_size = 64
window_size = 512
step = 256
output_file = DATA_DIR / "16S_embeddings_2k.npy"

tokenizer = BertTokenizer.from_pretrained("armheb/DNA_bert_6")
model = BertModel.from_pretrained("armheb/DNA_bert_6")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sequences = []
headers = []

with open(kmer_file, "r") as f:
    current_header = ""
    count = 0
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            current_header = line[1:]
            headers.append(current_header)
            count += 1
            if count > num_sequences:
                break
        else:
            sequences.append(line.split())

sequences = sequences[:num_sequences]
headers = headers[:num_sequences]

def chunk_kmers(kmer_list, window_size=512, step=256):
    return [kmer_list[i:i+window_size] for i in range(0, len(kmer_list), step)]

all_embeddings = []

for i in range(0, len(sequences), batch_size):
    batch_seqs = sequences[i:i+batch_size]
    batch_headers = headers[i:i+batch_size]

    for seq_kmers, header in zip(batch_seqs, batch_headers):
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

        sequence_embedding = torch.stack(window_embeddings).mean(dim=0)
        all_embeddings.append(sequence_embedding.numpy())

embeddings_array = np.stack(all_embeddings)
np.save(output_file, embeddings_array)
print(f"Embeddings saved to {output_file}")
print(f"Total sequences embedded: {len(all_embeddings)}")
