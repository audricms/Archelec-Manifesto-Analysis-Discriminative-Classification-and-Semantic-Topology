# Archelec Manifesto Analysis: Discriminative Classification & Semantic Topology

*Course: Machine Learning for NLP, ENSAE (2026).*  
*A comprehensive scientific report detailing the methodology, empirical results, and ideological analysis is available at the root of this repository: `Project_Report.pdf`.*

## 1. Project Context & Objectives

This repository contains the computational pipeline for analyzing historical electoral manifestos from the French legislative elections (1981, 1988, 1993) via the CEVIPOF Archelec corpus. 

The analytical objectives are strictly bifurcated into two domains:
1. **Discriminative Classification:** Optimizing political faction prediction ($y \in \mathcal{C}$) using high-dimensional sparse baselines (TF-IDF + Linear SVM) and contextualized representations (CamemBERT Linear Probes / mDeBERTa Zero-Shot), evaluated via macro-$F_1$.
2. **Topological & Error Analysis:** Mapping the ideological manifold. This includes diagnosing structural anomalies via Local Outlier Factor ($LOF_k$) in $\mathbb{R}^{768}$, extracting SVM boundary weights ($\mathbf{W} \in \mathbb{R}^{C \times V}$), and computing the generative semantic proximity between factions using zero-mean L2-normalized class centroids ($\cos(\hat{\mu}_i, \hat{\mu}_j)$).

## 2. Repository Architecture
```text
├── Project_Report.pdf                # Formal empirical manuscript
├── README.md
├── main.py                           # 14-step pipeline orchestrator
├── pyproject.toml / uv.lock          # Dependency management
├── data/                             
│   ├── gazetteers/                   # INSEE COG administrative data (spatial masking)
│   ├── manifestos/                   # OCR text files and metadata.csv
│   └── processed/                    # Split data, TF-IDF matrices, and .npz embeddings
├── models/                           # Serialized SVMs, vectorizers, and CamemBERT checkpoints
├── reports/                        
│   ├── figures/                      # Serialized UMAPs, heatmaps, and EDA distributions
│   └── metrics/                      # CSV ledgers (Outliers, SVM features, Proximity matrices)
└── src/                              # Executable pipeline modules
    ├── data/                         # SQLite parsing and Drive fetching
    ├── preprocessing/                # spaCy cleaning and D_train/D_test splitting
    └── models/                       # Inference, baseline, zero-shot, and topology modules
```

## 3. Environment Setup

This project utilizes `uv` for deterministic dependency resolution. 
```bash
# Instantiate the virtual environment and synchronize dependencies from uv.lock
uv sync

# Activate the environment
source .venv/bin/activate
```

## 4. Data Ingestion Protocol

Raw corpus data ($\approx 3$GB) is excluded from version control. You must hydrate the `data/` directory by executing the following pipeline.

### 4.1 Fetching Auxiliary Data 
Geographical masking requires official French administrative gazetteers sourced from the INSEE Code Officiel Géographique (COG).
```bash
uv run main.py --download_auxiliary_data
```
*Hydrates the `data/gazetteers/` and `data/manifestos/metadata.csv` paths from the secure remote storage.*

### 4.2 Extracting the Archelec Transcriptions (Arkindex)
Raw OCR manifestos are extracted from a relational database dump.

1. Register on the [Arkindex Demo instance](https://demo.arkindex.org).
2. Navigate to the [Archelec corpus](https://demo.arkindex.org/browse/1bc39ca6-399b-47ca-9de1-ab2ef481cabb?top_level=true&folder=true). Click **Import/Export -> Manage database exports**, and download the latest SQLite archive.
3. Place the downloaded `.sqlite` file in the root directory of this repository.
4. Open `src/data/extract_manifestos.py` and ensure `DB_PATH` matches your specific timestamped filename (e.g., `sciencespo-archelec-20260217-121320.sqlite`).
```bash
uv run main.py --extract_manifestos
```

## 5. Pipeline Orchestration

The experimental pipeline is governed by `main.py`, structured as a directed acyclic execution graph. 

To execute the pipeline end-to-end (Preprocessing $\to$ Baselines $\to$ Transformers $\to$ Topology $\to$ Metrics):
```bash
uv run main.py --all
```

Alternatively, individual experimental phases can be isolated:

**Data Engineering:**
```bash
uv run main.py --preprocess_data --analyze_preprocessing --split_data
```

**Discriminative Baselines & Feature Extraction:**
```bash
uv run main.py --build_tfidf --run_svm --run_svm_features_analysis
```

**Contextual Optimization & Inference:**
```bash
uv run main.py --run_zero_shot --run_linear_probe
```

**Topological Mapping & Anomaly Detection:**
```bash
uv run main.py --run_outlier_scoring --run_umap --run_outliers_inspection --run_semantic_analysis
```