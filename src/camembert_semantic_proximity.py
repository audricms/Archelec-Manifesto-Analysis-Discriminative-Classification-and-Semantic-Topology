import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_normalized_centroids(embeddings: np.ndarray, labels: np.ndarray):
    """
    Computes the L2-normalized class centroids after applying an isotropic
    correction (global mean subtraction) to the target embedding space.
    """
    unique_classes = np.unique(labels)
    centroids = []

    global_mean = np.mean(embeddings, axis=0)
    centered_embeddings = embeddings - global_mean

    norm_embeddings = normalize(centered_embeddings, norm="l2", axis=1)

    for target_class in unique_classes:
        class_mask = labels == target_class
        class_embeddings = norm_embeddings[class_mask]

        centroid = np.mean(class_embeddings, axis=0)
        centroids.append(centroid)

    centroids_matrix = np.vstack(centroids)
    return normalize(centroids_matrix, norm="l2", axis=1), unique_classes


def render_semantic_heatmap(df_proximity, output_path):
    """
    Renders a publication-ready heatmap of the generative semantic proximity.
    """
    mask = np.triu(np.ones_like(df_proximity, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        df_proximity,
        mask=mask,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".3f",
        cbar_kws={
            "shrink": 0.8,
            "label": r"Semantic Similarity ($\cos(\hat{\mu}_i, \hat{\mu}_j)$)",
        },
    )

    ax.set_title(
        "Ideological Proximity via Mean-Centered CamemBERT Centroids",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_semantic_analysis():
    print("Loading global distribution arrays (D_train U D_test)...")
    try:
        X_train_emb = np.load(PROCESSED_DIR / "embeddings_camembert_train.npy")
        X_test_emb = np.load(PROCESSED_DIR / "embeddings_camembert_test.npy")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["target_label"].values
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["target_label"].values
    except FileNotFoundError as e:
        print(f"Fatal: Required embedding matrices missing. {e}")
        sys.exit(1)

    # Reconstruct global spatial density
    X_global = np.vstack((X_train_emb, X_test_emb))
    y_global = np.concatenate((y_train, y_test))

    print("Computing zero-mean L2-normalized class centroids...")
    centroids, classes = compute_normalized_centroids(X_global, y_global)

    print("Computing generative semantic proximity matrix...")
    similarity_matrix = cosine_similarity(centroids)
    df_proximity = pd.DataFrame(
        similarity_matrix, index=classes, columns=classes
    ).round(4)

    proximity_csv_path = METRICS_DIR / "camembert_semantic_proximity_matrix.csv"
    proximity_fig_path = FIGURES_DIR / "camembert_semantic_proximity_heatmap.png"

    df_proximity.to_csv(proximity_csv_path, index=True, encoding="utf-8")

    print("Rendering structural proximity heatmap...")
    render_semantic_heatmap(df_proximity, proximity_fig_path)

    print(f"\n[SUCCESS] Semantic proximity matrix serialized to: {proximity_csv_path}")
    print(f"[SUCCESS] Semantic heatmap exported to: {proximity_fig_path}")
