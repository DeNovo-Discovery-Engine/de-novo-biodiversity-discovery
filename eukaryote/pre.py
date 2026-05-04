from Bio import SeqIO
import os

fasta_file = "/eukaryote/data/SSU_eukaryote_rRNA.fasta"
k = 6
output_file = "/eukaryote/data/SSU_eukaryote_rRNA.txt"

def get_kmers(sequence, k=6):
    sequence = sequence.upper()
    return [sequence[i:i+k] for i in range(len(sequence)-k+1) if "N" not in sequence[i:i+k]]

all_kmers_per_seq = []

for record in SeqIO.parse(fasta_file, "fasta"):
    seq = str(record.seq)
    kmers = get_kmers(seq, k)
    all_kmers_per_seq.append((record.description, kmers))

print(f"Processed {len(all_kmers_per_seq)} sequences")
total_kmers = sum(len(km[1]) for km in all_kmers_per_seq)
print(f"Total k-mers generated: {total_kmers}")

with open(output_file, "w") as f:
    for full_header, kmers in all_kmers_per_seq:
        if kmers:
            f.write(f">{full_header}\n")
            f.write(" ".join(kmers) + "\n")

print(f"K-mers with full headers saved to {output_file}")
