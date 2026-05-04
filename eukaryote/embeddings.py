import torch
from transformers import BertTokenizer, BertModel
import numpy as np

kmer_file = "/eukaryote/data/SSU_eukaryote_rRNA.txt"
num_sequences = 5000
batch_size = 64
window_size = 512
step = 256
output_file = "/eukaryote/data/SSU_eukaryote_embeddings.npy"

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

for idx, (seq_kmers, header) in enumerate(zip(sequences, headers), start=1):
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

    if idx % 100 == 0:
        print(f"{idx} embeddings generated...")

embeddings_array = np.stack(all_embeddings)
np.save(output_file, embeddings_array)
print(f"Embeddings saved to {output_file}")
print(f"Total sequences embedded: {len(all_embeddings)}")
