import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Core Utilities
from src.lock_seeds import lock_seeds

# Data Engineering
from src.data.extract_auxiliary_data import fetch_auxiliary_data
from src.data.extract_manifestos import extract_manifestos
from src.preprocessing.preprocessing import run_preprocessing
from src.preprocessing.preprocessing_analysis import run_preprocessing_analysis
from src.preprocessing.split_data import split_data

# Feature Engineering & Baselines
from src.models.tfidf import build_tfidf_features
from src.models.svm import run_svm
from src.svm_features_analysis import run_features_analysis

# Transformer Models & Outlier Detection
from src.models.zero_shot import run_zero_shot
from src.models.linear_probe import run_linear_probe
from src.models.outlier_scoring import run_outlier_scoring
from src.outliers_inspection import run_outliers_inspection
from src.camembert_semantic_proximity import run_semantic_analysis

# Manifold Projection
from src.models.umap import run_umap


def main():
    lock_seeds(42)

    parser = argparse.ArgumentParser(description="Archelec Pipeline Orchestrator")

    parser.add_argument(
        "--download_auxiliary_data",
        action="store_true",
        help="Download auxiliary data (Google Drive)",
    )
    parser.add_argument(
        "--extract_manifestos",
        action="store_true",
        help="Extract manifestos from Arkindex SQLite",
    )
    parser.add_argument(
        "--preprocess_data",
        action="store_true",
        help="Run the spaCy NLP cleaning pipeline",
    )
    parser.add_argument(
        "--analyze_preprocessing",
        action="store_true",
        help="Run EDA and lexical analysis on cleaned texts",
    )
    parser.add_argument(
        "--split_data", action="store_true", help="Split data into train/test sets"
    )
    parser.add_argument(
        "--build_tfidf",
        action="store_true",
        help="Build TF-IDF feature matrices using the split data",
    )
    parser.add_argument(
        "--run_svm",
        action="store_true",
        help="Train and evaluate the baseline TF-IDF SVM model",
    )
    parser.add_argument(
        "--run_zero_shot",
        action="store_true",
        help="Run and evaluate the Zero-Shot LLM classification",
    )
    parser.add_argument(
        "--run_linear_probe",
        action="store_true",
        help="Train and evaluate the CamemBERT Linear Probe",
    )
    parser.add_argument(
        "--run_outlier_scoring",
        action="store_true",
        help="Extract CamemBERT embeddings and compute LOF outliers",
    )
    parser.add_argument(
        "--run_umap",
        action="store_true",
        help="Run and evaluate the UMAP dimensionality reduction",
    )
    parser.add_argument(
        "--run_outliers_inspection",
        action="store_true",
        help="Cross-examine visual and statistical anomalies",
    )
    parser.add_argument(
        "--run_svm_features_analysis",
        action="store_true",
        help="Extract and serialize discriminative SVM features",
    )
    parser.add_argument(
        "--run_semantic_analysis",
        action="store_true",
        help="ComputeCamemBERT semantic proximity",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run the entire pipeline end-to-end"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.download_auxiliary_data or args.all:
        print("\n=== STEP 1: Downloading Auxiliary Data ===")
        fetch_auxiliary_data()

    if args.extract_manifestos or args.all:
        print("\n=== STEP 2: Extracting Manifestos from SQLite ===")
        extract_manifestos()

    if args.preprocess_data or args.all:
        print("\n=== STEP 3: NLP Preprocessing ===")
        run_preprocessing()

    if args.analyze_preprocessing or args.all:
        print("\n=== STEP 4: Preprocessing Analysis (EDA) ===")
        run_preprocessing_analysis()

    if args.split_data or args.all:
        print("\n=== STEP 5: Splitting Data (D_train / D_test) ===")
        split_data()

    if args.build_tfidf or args.all:
        print("\n=== STEP 6: Feature Engineering (TF-IDF Space) ===")
        build_tfidf_features()

    if args.run_svm or args.all:
        print("\n=== STEP 7: Baseline Optimization (Linear SVM) ===")
        run_svm()

    if args.run_zero_shot or args.all:
        print("\n=== STEP 8: Inference Evaluation (mDeBERTa Zero-Shot) ===")
        run_zero_shot()

    if args.run_linear_probe or args.all:
        print("\n=== STEP 9: Optimization (CamemBERT Linear Probe) ===")
        run_linear_probe(limit=None)

    if args.run_outlier_scoring or args.all:
        print("\n=== STEP 10: Contextual Embedding & Anomaly Detection (LOF) ===")
        run_outlier_scoring(n_neighbors=15)

    if args.run_umap or args.all:
        print("\n=== STEP 11: Topological Projection & Diagnostic Mapping (UMAP) ===")
        run_umap()

    if args.run_outliers_inspection or args.all:
        print("\n=== STEP 12: Outliers Inspection ===")
        run_outliers_inspection()

    if args.run_svm_features_analysis or args.all:
        print("\n=== STEP 13: Discriminative Feature Extraction (SVM) ===")
        run_features_analysis()

    if args.run_semantic_analysis or args.all:
        print("\n=== STEP 14: Semantic Proximity (CamemBERT) ===")
        run_semantic_analysis()


if __name__ == "__main__":
    main()
