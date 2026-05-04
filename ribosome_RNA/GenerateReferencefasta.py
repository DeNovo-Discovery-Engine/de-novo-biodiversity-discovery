import random
from pathlib import Path

from Bio import SeqIO

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

original_fasta = DATA_DIR / "16S_sequences.fasta"
new_fasta = DATA_DIR / "new_sequences.fasta"
num_sequences = 100

sequences = list(SeqIO.parse(original_fasta, "fasta"))
total_sequences = len(sequences)
print(f"Total sequences in original FASTA: {total_sequences}")

if num_sequences > total_sequences:
    raise ValueError(f"num_sequences ({num_sequences}) is greater than total sequences ({total_sequences})")

sampled_sequences = random.sample(sequences, num_sequences)

SeqIO.write(sampled_sequences, new_fasta, "fasta")
print(f"Saved {num_sequences} random sequences to {new_fasta}")
