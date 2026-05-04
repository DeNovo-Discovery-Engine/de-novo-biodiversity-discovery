# De Novo Biodiversity Discovery

A machine learning pipeline for discovering novel microbial taxa through sequence clustering and analysis. This tool uses DNA-BERT embeddings and HDBSCAN clustering to identify previously unknown ribosomal RNA sequences from both prokaryotic (16S) and eukaryotic (18S) samples.

---

## 📋 Table of Contents

1. [Features](#features)
2. [Project Overview](#project-overview)
3. [Installation & Setup](#installation--setup)
4. [Downloading Models](#downloading-models)
5. [Project Structure](#project-structure)
6. [Quick Start Guide](#quick-start-guide)
7. [Detailed Workflow](#detailed-workflow)
8. [Configuration](#configuration)
9. [Output Files](#output-files)
10. [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Dual Pipeline Support**: Analyze both 16S (prokaryotic) and 18S/SSU (eukaryotic) rRNA sequences
- **Advanced Embeddings**: Uses DNA-BERT (armheb/DNA_bert_6) for high-quality sequence representations
- **Novel Taxa Detection**: Identifies potential novel organisms through noise point detection in HDBSCAN clustering
- **3D Visualization**: Interactive 3D UMAP plots for cluster exploration
- **Reference-based Classification**: Assign new sequences to existing reference clusters
- **GPU Support**: Optional GPU acceleration for faster embedding generation

---

## 🎯 Project Overview

This project implements a complete de novo biodiversity discovery pipeline:

1. **Preprocessing**: Convert FASTA sequences to k-mers for BERT tokenization
2. **Embedding Generation**: Generate DNA sequence embeddings using DNA-BERT model
3. **Clustering**: Apply HDBSCAN clustering to identify groups and novel taxa
4. **Visualization**: Create interactive 3D plots for cluster exploration
5. **Classification**: Assign new query sequences to reference clusters with confidence scores

**Key Concept**: Points classified as "noise" (-1) by HDBSCAN are treated as potential novel taxa that don't fit into established groups.

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+** (tested with Python 3.9-3.11)
- **Conda** (recommended) or **pip**
- **CUDA 11.8+** (optional, for GPU acceleration)
- **4GB+ RAM** (8GB+ recommended, more for large datasets)
- **Disk space**: ~2GB for models and data

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/de-novo-biodiversity-discovery.git
cd de-novo-biodiversity-discovery
```

### Step 2: Create Virtual Environment

#### Using Conda (Recommended)

```bash
# Create environment
conda create -n biodiversity python=3.10

# Activate environment
conda activate biodiversity

# Install CUDA-enabled PyTorch (if you have CUDA)
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# Or CPU-only PyTorch
conda install pytorch-cpu -c pytorch
```

#### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Package Details:**
- `torch` - Deep learning framework
- `transformers` - Hugging Face model library
- `biopython` - Bioinformatics toolkit
- `hdbscan` - Clustering algorithm
- `scikit-learn` - ML utilities
- `numpy`, `pandas` - Data manipulation
- `umap-learn` - Dimensionality reduction
- `matplotlib`, `plotly`, `seaborn` - Visualization

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from transformers import BertModel; print('Transformers: OK')"
python -c "import hdbscan; print('HDBSCAN: OK')"
```

---

## 📦 Downloading Models

### DNA-BERT Model

The pipeline uses **DNA_bert_6** from Hugging Face. This model is automatically downloaded on first use.

#### Automatic Download (Recommended)

The model downloads automatically when you first run an embedding script:

```bash
# First run - model downloads (~400MB)
python eukaryote/embeddings.py
```

#### Manual Pre-download

To download the model before running scripts:

```python
from transformers import BertTokenizer, BertModel

# Download tokenizer
tokenizer = BertTokenizer.from_pretrained("armheb/DNA_bert_6")

# Download model
model = BertModel.from_pretrained("armheb/DNA_bert_6")

print("Model downloaded successfully!")
```

Or use the command line:

```bash
# Download to cache
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('armheb/DNA_bert_6'); BertModel.from_pretrained('armheb/DNA_bert_6'); print('Downloaded!')"
```

#### Model Cache Location

The model is cached at:
- **Linux/Mac**: `~/.cache/huggingface/hub/models--armheb--DNA_bert_6/`
- **Windows**: `C:\Users\{username}\.cache\huggingface\hub\models--armheb--DNA_bert_6\`

To change the cache directory:

```bash
# Set environment variable before running
export HF_HOME=/path/to/cache  # Linux/Mac
set HF_HOME=D:\cache            # Windows CMD
```

---

## 📁 Project Structure

```
de-novo-biodiversity-discovery/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── eukaryote/                         # 18S rRNA (Eukaryotic) Pipeline
│   ├── checkFASTA.py                 # Validate FASTA format
│   ├── pre.py                        # Convert sequences to k-mers
│   ├── embeddings.py                 # Generate DNA-BERT embeddings
│   ├── cluster.py                    # HDBSCAN clustering
│   ├── clusterMERGE.py              # Merge clustering results
│   ├── separateCLUSTER.py           # Separate clusters by criteria
│   ├── Heatmap.py                   # Generate heatmap visualizations
│   ├── UMAP3D.py                    # 3D UMAP visualization
│   │
│   ├── data/                        # Input data directory
│   │   ├── SSU_eukaryote_rRNA.fasta       # Input sequences
│   │   ├── SSU_eukaryote_rRNA.txt         # K-mers (generated by pre.py)
│   │   └── SSU_eukaryote_embeddings.npy   # Embeddings (generated by embeddings.py)
│   │
│   └── output/                      # Output directory
│       ├── SSU_eukaryote_hdbscan.csv              # Cluster assignments
│       ├── SSU_potential_novel.csv                # Novel taxa candidates
│       ├── query_cluster_assignment.csv           # Query classifications
│       ├── query_only_cluster_assignment.csv      # Query-only classifications
│       └── combined_clusters.csv                  # Combined results
│
├── ribosome_RNA/                     # 16S rRNA (Prokaryotic) Pipeline
│   ├── Checkfasta.py                # Validate FASTA format
│   ├── newCLUSTER.py                # HDBSCAN clustering
│   ├── newCHECKfasta.py             # FASTA validation
│   ├── embeddings.py                # Generate DNA-BERT embeddings
│   ├── GenerateReferencefasta.py    # Generate reference sequences
│   ├── heatmaplatest.py             # Heatmap visualization
│   ├── indicesHeatmap.py            # Heatmap with indices
│   ├── Silhoutte.py                 # Silhouette score analysis
│   ├── UMAP3d.py                    # 3D UMAP visualization
│   │
│   ├── data/                        # Input data directory
│   │   ├── 16S_sequences.fasta            # Input sequences
│   │   ├── 16S_kmers_fullheader.txt       # K-mers (generated by pre-processing)
│   │   ├── 16S_embeddings_2k.npy          # Embeddings (generated by embeddings.py)
│   │   └── new_sequences.fasta            # Query sequences
│   │
│   └── outputs/                     # Output directory
│       ├── 16S_clusters_hdbscan.csv              # Cluster assignments
│       ├── 16S_potential_novel.csv              # Novel taxa candidates
│       ├── new_sequences_clustered.csv          # Query classifications
│       ├── new_sequences_knn_clustered.csv      # KNN-based classifications
│       └── new_sequences_novel.csv              # Novel sequence predictions
│
└── images/                          # Visualization outputs
    ├── clusters_3d.html             # Interactive 3D plots
    └── heatmaps.png                 # Heatmap visualizations
```

---

## 🎓 Quick Start Guide

### For First-Time Users (Complete Walkthrough)

#### ⚠️ IMPORTANT: Data Files Required

Before running the pipeline, prepare these data files in the correct directories:

**For 16S rRNA Pipeline:**
- Place your sequences in: `ribosome_RNA/data/16S_sequences.fasta`

**For 18S rRNA (Eukaryotic) Pipeline:**
- Place your sequences in: `eukaryote/data/SSU_eukaryote_rRNA.fasta`

**For Query/New Sequences Classification:**
- Place new sequences in: `ribosome_RNA/data/new_sequences.fasta`

#### Step 1: Prepare Your Data

```bash
# For 16S rRNA data
cp your_16S_sequences.fasta ribosome_RNA/data/16S_sequences.fasta

# For 18S rRNA data  
cp your_18S_sequences.fasta eukaryote/data/SSU_eukaryote_rRNA.fasta

# (Optional) For query sequences
cp your_query_sequences.fasta ribosome_RNA/data/new_sequences.fasta
```

#### Step 2: Run the Pipeline (Example: 16S)

**IMPORTANT**: Follow this exact order! Each step depends on the previous one.

```bash
cd ribosome_RNA

# Step 1: Validate your FASTA file
python Checkfasta.py

# Step 2: Convert sequences to k-mers (REQUIRED before embeddings)
# If there's a file like "16S_kmers_fullheader.txt", this step is already done
# Otherwise, you may need to run pre-processing if your embeddings.py script requires it

# Step 3: Generate embeddings (this downloads DNA-BERT automatically)
# ⏱️  TAKES TIME: 10-30 minutes depending on your data size and hardware
# 🚀 GPU RECOMMENDED: 10x faster with CUDA
python embeddings.py

# Step 4: Cluster sequences and identify novel taxa
python newCLUSTER.py

# Step 5: Create 3D visualization
python UMAP3d.py
```

#### Step 3: View Results

1. **Cluster Results**: Open `outputs/16S_clusters_hdbscan.csv` in Excel or Python pandas
2. **Novel Taxa**: Check `outputs/16S_potential_novel.csv` for candidates
3. **Visualization**: Open the generated HTML file in browser for interactive 3D plot

---

## � Understanding the Data Flow

**Dependency Chain** - Each step depends on the previous one:

```
Your FASTA File
    ↓
pre.py (Preprocessing) 
    ↓ generates K-mer file (.txt)
    ↓
embeddings.py (Generate Embeddings)
    ↓ generates Embedding array (.npy)
    ↓
cluster.py or newCLUSTER.py (Clustering)
    ↓ generates Cluster assignments (.csv)
    ↓
UMAP3d.py or similar (Visualization)
    ↓ generates 3D Plot (HTML)
```

**Important**: You CANNOT skip steps. Each script needs the output of the previous step.

---

### Pipeline Stages

#### 1. **Preprocessing** (pre.py / Checkfasta.py)

**Purpose**: Convert DNA sequences to k-mers for BERT processing

**Prerequisites**:
- FASTA file with sequences (e.g., `SSU_eukaryote_rRNA.fasta`)

**What it does**:
- Reads FASTA sequences
- Generates 6-bp k-mers (default)
- Removes k-mers containing 'N' (ambiguous bases)
- Stores both original headers and k-mers

**Input**: FASTA file (DNA sequences)
**Output**: Text file with headers and space-separated k-mers (e.g., `SSU_eukaryote_rRNA.txt`)

**Example**:
```
Original: ATCGATCGATCG...
K-mers (k=6): ATCGAT TCGATC CGATCG ...
```

**⚠️ REQUIRED FOR**: Embedding generation (embeddings.py requires the k-mer `.txt` file as input)

#### 2. **Embedding Generation** (embeddings.py)

**Purpose**: Convert sequences to numerical vectors using DNA-BERT

**Prerequisites**:
- K-mer file (generated by preprocessing step)
- Hugging Face transformers library (auto-downloads model on first run)

**What it does**:
- Loads pre-trained DNA_bert_6 model from Hugging Face
- Tokenizes k-mers using BERT tokenizer
- Processes sequences in windows (512 k-mers/window)
- Generates 768-dimensional embeddings per sequence
- Handles GPU acceleration if available

**Important Parameters**:
- `num_sequences`: **Number of sequences to process** (must match or be less than actual count in your file)
  - Default: 5000 (MUST ADJUST if your file has different count)
  - Set to actual sequence count in your k-mer file
  - Example: If your FASTA has 2000 sequences → set `num_sequences = 2000`

**Input**: K-mer file + model (auto-downloaded on first run)
**Output**: NumPy array (.npy) of embeddings with shape (num_sequences, 768)

**Runtime**:
- **GPU (CUDA)**: ~1 minute per 1000 sequences
- **CPU**: ~5-10 minutes per 1000 sequences

**Tips**:
- First run downloads model (~400MB) - this is normal
- GPU strongly recommended for large datasets
- Adjust `window_size` if memory issues occur
- **Verify your actual sequence count and update `num_sequences` parameter**

#### 3. **Clustering** (newCLUSTER.py / cluster.py)

**Purpose**: Group sequences into taxonomic clusters

**What it does**:
- Uses HDBSCAN algorithm on embeddings
- Identifies clusters with min_cluster_size = 35
- Classifies "noise" points as potential novel taxa (label = -1)
- Calculates cluster confidence scores

**Parameters**:
- `min_cluster_size`: Minimum cluster size (default: 35)
- `min_samples`: Points in core region (default: 8)

**Output**:
- Cluster labels for each sequence
- Confidence scores (0-1)
- Noise points flagged for novelty

**Understanding Results**:
- Label 0, 1, 2, etc. = Known taxonomic groups
- Label -1 = Noise = Potential novel organism

#### 4. **Dimensionality Reduction** (UMAP3d.py)

**Purpose**: Project embeddings to 3D for visualization

**What it does**:
- Applies UMAP algorithm to embeddings
- Reduces 768-dimensional space to 3D
- Preserves local structure and cluster relationships
- Generates interactive plotly visualization

**Output**: Interactive HTML file for 3D exploration

#### 5. **Classification** (checkFASTA.py - query assignment)

**Purpose**: Assign new sequences to reference clusters

**What it does**:
- Takes known reference clusters
- Generates embeddings for new/query sequences
- Uses approximate_predict to assign to nearest cluster
- Provides probability scores

---

## ⚙️ Configuration

### Adjusting Clustering Parameters

Edit the clustering scripts to change sensitivity:

**File**: `ribosome_RNA/newCLUSTER.py` (or `eukaryote/cluster.py`)

```python
# More strict clustering (larger clusters, fewer novel)
min_cluster_size = 50      # Increase to ~50-100
min_samples = 10           # Increase correspondingly

# More sensitive clustering (more novel taxa detected)
min_cluster_size = 15      # Decrease to 10-20
min_samples = 5            # Decrease correspondingly
```

**Effect of parameters**:
- **Larger min_cluster_size**: Conservative, fewer clusters, more noise (potential novel)
- **Smaller min_cluster_size**: Liberal, more clusters, fewer noise
- **Increase min_samples**: Stricter cluster membership requirements

### Adjusting Embedding Generation Parameters

**Important**: If you have more or fewer sequences than the default setting, adjust this:

**File**: `eukaryote/embeddings.py` or `ribosome_RNA/embeddings.py`

```python
# Default: processes 5000 sequences
num_sequences = 5000

# Adjust to match your actual dataset size:
# - Set to actual number of sequences in your FASTA file
# - If your file has 2000 sequences, set to 2000
# - If you want to process only a subset, set to that number
num_sequences = 2000  # Your dataset size
```

**Memory & Speed Adjustments**:
```python
# If running out of memory, reduce these:
window_size = 256       # Default: 512 (k-mers per window)
step = 128             # Default: 256 (stride between windows)
batch_size = 32        # Default: 64

# Increasing these speeds up but uses more memory (only if you have GPU with enough VRAM)
batch_size = 128       # Larger batch size
```

### Adjusting UMAP Parameters

**File**: `eukaryote/UMAP3D.py` or `ribosome_RNA/UMAP3d.py`

```python
reducer = umap.UMAP(
    n_neighbors=30,        # Local neighborhood size (default: 30)
    min_dist=0.05,         # Minimum distance between points (default: 0.05)
    n_components=3,        # 3 for 3D, 2 for 2D
    random_state=42        # For reproducibility
)
```

### GPU Configuration

**Check GPU availability**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Enable/disable GPU in embeddings.py**:
```python
# Automatic (uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force CPU only
device = torch.device("cpu")

# Force GPU
device = torch.device("cuda:0")
```

---

## 📈 Output Files

### Cluster Assignment Files

**Format**: CSV with columns:
- `Header`: Sequence identifier
- `Cluster`: Cluster label (-1 = potential novel)
- `Confidence`: Confidence score (0-1)

**Example**:
```
Header,Cluster,Confidence
seq_001,5,0.92
seq_002,5,0.88
seq_003,-1,0.15    ← Potential novel taxon
seq_004,12,0.87
```

### Novel Taxa Files

**Format**: CSV subset of cluster assignments where Cluster == -1

**Use case**: Review sequences that don't fit established groups

### Query Classification Files

**Format**: CSV with query sequence assignments

**Columns**:
- `Query_Header`: New sequence ID
- `Assigned_Cluster`: Reference cluster it matches
- `Probability`: Confidence of assignment (0-1)

### Visualization Files

**3D UMAP HTML**:
- Interactive 3D scatter plot
- Hover over points for details
- Zoom, rotate, pan controls
- Color-coded by cluster
- Shape-coded by source (16S, Eukaryote, Query)

---

## 🔧 Troubleshooting

### Issue: Model Download Fails

**Error**: `ConnectionError` when downloading DNA_bert_6

**Solutions**:
```bash
# Option 1: Manually download
python -c "from transformers import BertModel; BertModel.from_pretrained('armheb/DNA_bert_6')"

# Option 2: Specify cache directory
export HF_HOME=/path/to/cache
python embeddings.py

# Option 3: Download with wget/curl
# Visit https://huggingface.co/armheb/DNA_bert_6 for manual download
```

### Issue: Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory` or system freezes

**Solutions**:
```python
# Reduce window size in embeddings.py
window_size = 256       # Decrease from 512
step = 128             # Decrease from 256

# Process in smaller batches
batch_size = 16        # Decrease from 64

# Use CPU instead of GPU
device = torch.device("cpu")
```

### Issue: GPU Not Found

**Error**: `cuda not available` or GPU not detected

**Solutions**:
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: No Clusters Found

**Error**: HDBSCAN returns very few or no clusters

**Solutions**:
1. **Check data quality**: Ensure sequences are similar enough to cluster
2. **Reduce min_cluster_size**:
   ```python
   min_cluster_size = 15  # Instead of 35
   ```
3. **Verify embeddings**: Check that embeddings.npy contains valid data
4. **Inspect sequences**: Make sure FASTA file isn't corrupted

### Issue: Slow Embedding Generation

**On CPU**: Very slow (30+ minutes for 1000 sequences)

**Solutions**:
1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Reduce sequence count**: Process in batches
3. **Reduce window size**: Trade-off between quality and speed
4. **Use smaller model**: DNA_bert_3 (but lower quality)

### Issue: "FileNotFoundError" when running embeddings.py

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '.../16S_kmers_fullheader.txt'`

**Cause**: The k-mer file doesn't exist. The embeddings script expects pre-processed k-mer files.

**Solution**: Check if there's a preprocessing script that needs to run first:

```bash
# Look for a preprocessing or k-mer generation script in your directory
# Common names: pre.py, preprocess.py, GenerateReferencefasta.py

# For 16S:
# The embeddings.py already expects the k-mer file
# You may need to generate k-mers from your FASTA first

# For Eukaryote:
# First run the preprocessing to generate k-mers:
python pre.py  # Generates SSU_eukaryote_rRNA.txt from FASTA
python embeddings.py  # Then generate embeddings from k-mers
```

### Issue: Embeddings Script Skips Sequences or Gets Wrong Count

**Error**: `num_sequences = 5000` but your FASTA has 2000 sequences

**Cause**: The `num_sequences` parameter is hardcoded and limits how many sequences are processed

**Solution**: Edit `num_sequences` to match your actual dataset:

```bash
# Before running embeddings.py, edit it:
nano embeddings.py  # Or use your editor

# Change this line to match your actual sequence count:
num_sequences = 5000  # ← Change to your actual count
```

### Issue: Script Can't Find Files (Windows Users)

**Error**: `FileNotFoundError` or `No such file or directory`

**Cause**: Scripts use relative paths that depend on working directory

**Solution**: Always change to the correct directory before running:

```bash
# Navigate to the correct directory FIRST
cd ribosome_RNA
python embeddings.py

# OR cd eukaryote first for eukaryote scripts
cd eukaryote
python embeddings.py
```

---

## 🧬 Understanding Results

### What Do Cluster Numbers Mean?

- **Cluster 0, 1, 2, ...**: Established taxonomic groups in your reference data
- **Cluster -1**: Noise points = Potential novel organisms

### How Many Novel Sequences Are Normal?

Depends on your data:
- **Well-known organisms**: 0-5% novel taxa
- **Mixed environments**: 5-15% novel taxa
- **Extreme environments**: 15-40% novel taxa

### Why Aren't All Sequences in Clusters?

HDBSCAN creates clusters only where density is high. Singletons and low-density regions become noise, which is valuable for novelty detection!

---

## 📚 Citation

If you use this tool in research, please cite:

```bibtex
@software{biodiversity_discovery,
  title={De Novo Biodiversity Discovery},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/de-novo-biodiversity-discovery}
}
```

**Related Papers**:
- DNA-BERT: Yanrong Ji et al., "DNA-BERT: pre-trained bidirectional encoder representations from transformers model for DNA-sequence based prediction tasks"
- HDBSCAN: Leland McInnes, John Healy, Steve Astels. "hdbscan: hierarchical density based clustering"
- UMAP: Leland McInnes et al., "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"

---