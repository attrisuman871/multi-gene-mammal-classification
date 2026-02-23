> Note: This project emphasizes reproducibility, modularity, and phylogenetic supervision rather than maximizing predictive accuracy.


# Multi-Gene Mammalian Species Classification

A modular machine learning pipeline that classifies mammalian species using multi-gene DNA sequences. The pipeline fetches sequences from NCBI, processes and aligns them, extracts multiple feature representations, and trains a multi-input CNN classifier fused with species metadata.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Sections](#pipeline-sections)
- [Dependencies](#dependencies)
- [External Tools](#external-tools)
- [Configuration](#configuration)
- [Input Files](#input-files)
- [Output Files](#output-files)
- [Genes Used](#genes-used)
- [Species Covered](#species-covered)
- [Model Architecture](#model-architecture)
- [Usage](#usage)

---

## Overview

The notebook `multi-gene-species-modular.ipynb` implements an end-to-end pipeline:

1. Fetches DNA sequences for ~80 mammalian species and 6 genes from NCBI
2. Cleans and filters sequences, then aligns them with MAFFT + TrimAl
3. Encodes sequences using three feature methods: k-mer frequencies, one-hot encoding, and BioVec (Word2Vec) embeddings
4. Downloads a phylogenetic tree from the Open Tree of Life API and computes pairwise evolutionary distances
5. Enriches species with geographic and conservation metadata
6. Performs unsupervised hierarchical clustering and visualises dendrograms
7. Trains a multi-input CNN that fuses all five gene branches with species metadata for supervised classification

**Original notebook:** 196 cells → **Modular notebook:** 40 cells (23 code, 17 markdown)

---

## Project Structure

```
Downloads/
├── multi-gene-species-modular.ipynb   # Main modular notebook
├── README.md                          # This file
├── mammal_metadata.csv                # Input: species metadata (see Input Files)
├── dna_sequences.json                 # Generated: raw NCBI sequences
├── complete_dna_sequences.json        # Generated: filtered complete sequences
├── phylogenetic_tree.nwk              # Generated: Newick tree from Open Tree of Life
├── pairwise_distances.csv             # Generated: pairwise evolutionary distances
├── square_distance_matrix.csv         # Generated: square distance matrix
├── filtered_mammal_metadata.csv       # Generated: metadata for 24 final species
├── updated_species_with_latlong.csv   # Generated: metadata + geocoordinates + IUCN
├── gene_fastas/                       # Raw per-gene FASTA files
├── filtered/                          # Quality-filtered FASTA files
├── extracted_sequences/               # Common-species FASTA files
├── aligned/                           # MAFFT-aligned + TrimAl-trimmed FASTAs + PHYLIPs
├── kmers/                             # K-mer frequency CSVs per gene
├── onehot/                            # One-hot encoded .npy arrays per gene
└── biovec/                            # BioVec embedding .npy arrays per gene
```

All directory paths are controlled by variables in the **Configuration** cell — change `BASE_DIR` to move the entire project.

---

## Pipeline Sections

| # | Section | Key Output |
|---|---------|------------|
| 0 | Installation | — |
| 1 | Imports & Configuration | All constants, paths, gene/species lists |
| 2 | Data Collection (NCBI API) | `dna_sequences.json` |
| 3 | Data Cleaning & Filtering | `complete_dna_sequences.json` |
| 4 | Per-Gene FASTA Export | `gene_fastas/<GENE>.fasta` |
| 5 | Sequence Quality Filtering | `filtered/<GENE>.filtered.fasta` |
| 6 | Species Intersection | `extracted_sequences/<gene>_common_species.fasta` |
| 7 | Sequence Alignment | `aligned/<gene>_aligned.trimmed.fasta`, `.phy` |
| 8a | K-mer Frequencies | `kmers/<gene>_kmer_frequencies.csv` |
| 8b | One-Hot Encoding | `onehot/<gene>_onehot.npy` |
| 8c | BioVec Embeddings | `biovec/<gene>_biovec.npy` |
| 9 | Phylogenetic Tree & Distance Matrix | `phylogenetic_tree.nwk`, `square_distance_matrix.csv` |
| 10 | Metadata Preparation | `updated_species_with_latlong.csv` |
| 11 | Hierarchical Clustering | Dendrograms, silhouette score |
| 12 | Multi-Input CNN Classification | Trained model, accuracy / F1 metrics |

---

## Dependencies

Install all Python dependencies with:

```bash
pip install biopython ete3 gensim geopy torch torchvision
```

Full list:

| Package | Purpose |
|---------|---------|
| `biopython` | NCBI Entrez API, FASTA parsing, sequence alignment I/O |
| `ete3` | Phylogenetic tree loading and pairwise distance computation |
| `gensim` | Word2Vec model for BioVec embeddings |
| `geopy` | Geocoding species distribution locations |
| `torch` / `torchvision` | CNN model definition and training |
| `numpy` | Array operations |
| `pandas` | DataFrame operations |
| `scikit-learn` | Scaling, KMeans, silhouette score, classification metrics |
| `scipy` | Hierarchical clustering (linkage, dendrogram) |
| `matplotlib` / `seaborn` | Plotting |
| `requests` | Open Tree of Life API calls |

---

## External Tools

Two external bioinformatics tools are required for Section 7 (Sequence Alignment):

| Tool | Purpose | Install |
|------|---------|---------|
| **MAFFT** | Multiple sequence alignment | https://mafft.cbrc.jp/alignment/software/ |
| **TrimAl** | Alignment trimming | http://trimal.cgenomics.org/ |

By default the notebook expects both to be on your `PATH`. If they are installed elsewhere, update the two variables in the Configuration cell:

```python
MAFFT_PATH  = "mafft"    # e.g. "/usr/local/bin/mafft" or r"C:\mafft-win\mafft.bat"
TRIMAL_PATH = "trimal"   # e.g. "/usr/local/bin/trimal" or r"C:\tools\trimal\trimal.exe"
```

---

## Configuration

All tunable settings live in a single **Configuration** cell (cell 5):

```python
# Root directory — all subdirectories are derived from this
BASE_DIR = "."

# External tool paths
MAFFT_PATH  = "mafft"
TRIMAL_PATH = "trimal"

# Gene lists
SELECTED_GENES = ["COI", "CytB", "16S rRNA", "RAG1", "BRCA1", "APOB"]
GENES          = ["apob", "brca1", "coi", "cytb", "rag1"]  # post-filtering

# Model hyperparameters
NUM_CLASSES   = 5
BATCH_SIZE    = 8
EPOCHS        = 15
LEARNING_RATE = 1e-3
KMER_K        = 3      # k-mer size
BIOVEC_DIM    = 100    # Word2Vec embedding dimensions
```

---

## Input Files

| File | Description | Required by |
|------|-------------|-------------|
| NCBI credentials | Set `Entrez.email` and `Entrez.api_key` in Section 2 | Section 2 |
| `mammal_metadata.csv` | Species metadata with columns: `Scientific_Name`, `iucnStatus`, `countryDistribution`, `biogeographicRealm`, `family` | Section 10 |

The `mammal_metadata.csv` file should contain at minimum these columns:

```
Scientific_Name, iucnStatus, countryDistribution, biogeographicRealm, family
```

---

## Output Files

| File | Description |
|------|-------------|
| `dna_sequences.json` | Raw sequences fetched from NCBI for all species and genes |
| `complete_dna_sequences.json` | Sequences for species that have all five genes present |
| `phylogenetic_tree.nwk` | Newick-format phylogenetic tree from Open Tree of Life |
| `pairwise_distances.csv` | Long-format table of pairwise evolutionary distances |
| `square_distance_matrix.csv` | Square symmetric distance matrix (species × species) |
| `filtered_mammal_metadata.csv` | Metadata filtered to the 24 final species |
| `updated_species_with_latlong.csv` | Metadata enriched with latitude, longitude, and IUCN ordinal |

---

## Genes Used

| Gene | Type | Role |
|------|------|------|
| **COI** | Mitochondrial | Cytochrome c oxidase subunit I — standard DNA barcoding gene |
| **CytB** | Mitochondrial | Cytochrome b — widely used phylogenetic marker |
| **RAG1** | Nuclear | Recombination activating gene — robust phylogenetic marker |
| **BRCA1** | Nuclear | Breast cancer gene 1 — used in mammalian phylogenomics |
| **APOB** | Nuclear | Apolipoprotein B — long gene used in higher-level phylogenies |

*16S rRNA was downloaded but removed due to high missing-data rates.*

---

## Species Covered

The pipeline starts with ~80 candidate species spanning five mammalian orders and narrows to the **24 species** that have high-quality sequences in all five genes:

| Group | Example Species |
|-------|----------------|
| Primates | *Gorilla gorilla*, *Macaca mulatta*, *Lemur catta*, *Callithrix jacchus* |
| Rodents | *Mus musculus*, *Rattus rattus*, *Castor canadensis*, *Chinchilla lanigera* |
| Carnivores | *Panthera leo*, *Panthera tigris*, *Ursus maritimus*, *Hyaena hyaena* |
| Ungulates | *Capra hircus*, *Ovis aries*, *Hippopotamus amphibius*, *Equus przewalskii* |
| Bats | *Eptesicus fuscus*, *Rhinolophus ferrumequinum* |

---

## Model Architecture

```
Gene 1 (APOB)  ──► GeneCNN ──┐
Gene 2 (BRCA1) ──► GeneCNN ──┤
Gene 3 (COI)   ──► GeneCNN ──┼──► Concatenate ──► Classifier ──► Species Class
Gene 4 (CytB)  ──► GeneCNN ──┤         ▲
Gene 5 (RAG1)  ──► GeneCNN ──┘         │
                                        │
Metadata (lat, lon, IUCN) ──► MetadataEncoder ──┘
```

**GeneCNN** — two Conv1d layers (32 → 64 filters, kernel=5) + AdaptiveMaxPool + Linear
**MetadataEncoder** — two-layer MLP (3 → 64 → 64)
**Classifier** — Linear(5×64 + 64 → 128) → ReLU → Linear(128 → NUM_CLASSES)

Labels are derived by running KMeans (`k=5`) on the phylogenetic distance matrix.

---

## Results

### Hierarchical Clustering
- Distance Metric: Cosine
- Silhouette Score: **0.4511**

Clustering revealed biologically coherent groupings aligned with taxonomic structure and geographic proximity.

### Multi-Input CNN Classification
- Accuracy: **0.50**
- Precision: **0.39**
- Recall: **0.50**
- F1 Score: **0.44**

Performance is constrained by limited species coverage (n=24) and phylogenetically derived multi-class structure, yet the model captures non-trivial cross-gene genomic patterns.

Despite limited sample size, the model successfully learned non-trivial genomic structure across five gene inputs.

---

## Key Insights

- Multi-gene input improves biological signal consistency.
- One-hot encoding outperformed BioVec embeddings for supervised learning due to preserved positional alignment.
- Phylogenetic supervision provides biologically meaningful label generation.
- Evolutionary distance correlates partially with geographic distribution and conservation status.

This demonstrates that genomic ML models can capture taxonomic structure beyond simple sequence similarity.

---

## Why This Matters

- Enables scalable ML-assisted taxonomy.
- Bridges machine learning and evolutionary biology.
- Demonstrates integration of genomic data with ecological metadata.
- Provides a foundation for eDNA species detection and conservation analytics.

This pipeline can serve as a template for large-scale genomic classification systems.

---

## Limitations

- Small final species set (24 species) due to multi-gene completeness constraints.
- CNN performance limited by dataset size.
- Clustering labels derived from KMeans rather than curated taxonomy.
- No branch-length regression or evolutionary time modeling.

Future work will expand dataset size and incorporate transformer-based sequence modeling.

---

## Usage

1. **Clone / download** the notebook and place `mammal_metadata.csv` in the same directory.

2. **Set your NCBI credentials** in Section 2:
   ```python
   Entrez.email   = "your_email@example.com"
   Entrez.api_key = "your_api_key"
   ```
   Register for a free API key at https://www.ncbi.nlm.nih.gov/account/

3. **Set your project root** in the Configuration cell:
   ```python
   BASE_DIR = "/path/to/your/project"
   ```

4. **Run all cells top to bottom.** Each section is self-contained — it loads its inputs, processes them, and saves its outputs before the next section begins.

5. If you already have intermediate files (e.g., aligned FASTAs), you can skip earlier sections and start from any point in the pipeline.
