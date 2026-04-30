import sys
import re
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import umap
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "umap_projections"
METRICS_DIR = REPORTS_DIR / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_document_subset(
    doc_ids: list,
    df_corpus: pd.DataFrame,
    lof_ledger: pd.DataFrame,
    svm_model,
    tfidf,
    global_vocab: set,
    subset_name="SUBSET",
):
    """
    Executes a rigorous diagnostic extraction for a specific subset of documents,
    computing OOV degradation, decision boundary margins, and test-set LOF rank.
    Returns a list of dictionaries for tabular serialization.
    """
    print(f"\n{'=' * 80}")
    print(f"=== INITIATING DIAGNOSTIC AUDIT: {subset_name} ===")
    print(f"{'=' * 80}")

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    metrics_accumulator = []

    # Filter strictly for requested IDs and sort by descending anomaly severity
    valid_ids = [
        doc_id
        for doc_id in doc_ids
        if doc_id in lof_ledger.index and doc_id in df_corpus.index
    ]
    subset_lof_ledger = lof_ledger.loc[valid_ids].sort_values(
        by="LOF_Score", ascending=False
    )

    if subset_lof_ledger.empty:
        print("No valid documents found in the test set for this subset.")
        return metrics_accumulator

    for doc_id, row in subset_lof_ledger.iterrows():
        doc = df_corpus.loc[doc_id]
        if isinstance(doc, pd.DataFrame):
            doc = doc.iloc[0]

        cleaned_text = str(doc["cleaned_text"])

        # OOV Computation
        tokens = token_pattern.findall(cleaned_text.lower())
        total_tokens = len(tokens)
        oov_tokens = (
            [t for t in tokens if t not in global_vocab] if total_tokens > 0 else []
        )
        oov_rate = len(oov_tokens) / total_tokens if total_tokens > 0 else 1.0

        # SVM Decision Boundary Prediction
        vectorized_text = tfidf.transform([cleaned_text])
        y_pred = svm_model.predict(vectorized_text)[0]
        y_true = doc["target_label"]

        # Extract source text for contextual proof
        year = str(doc.get("date", "XXXX"))[:4]
        source_txt_path = (
            DATA_DIR / "manifestos" / year / "legislatives" / f"{doc_id}.txt"
        )
        try:
            with open(source_txt_path, "r", encoding="utf-8") as f:
                source_text = f.read()
        except (FileNotFoundError, UnicodeDecodeError):
            source_text = "[FATAL: SOURCE FILE UNREADABLE OR MISSING]"

        margin_status = "CORRECT" if y_true == y_pred else "MISCLASSIFIED"

        if oov_rate > 0.15:
            diagnostic_heuristic = "Topological isolation driven by OCR degradation."
        elif y_true != y_pred:
            diagnostic_heuristic = "True Anomaly. Boundary misclassification corroborated by spatial isolation."
        else:
            diagnostic_heuristic = "Intra-class Outlier. Visually isolated but semantically aligned to decision boundary."

        # Console Output
        print(f"\n--- DOCUMENT ID: {doc_id} ---")
        print(f"Date : {doc.get('date', 'N/A')}")
        print(f"Official Party      : {doc.get('titulaire-soutien', 'N/A')}")
        print(f"Assigned Bloc (y)      : {y_true}")
        print(f"SVM Bloc Prediction (y_hat) : {y_pred}")
        print(f"Classification Status  : {margin_status}")
        print(f"LOF Anomaly Score      : {row['LOF_Score']:.4f}")
        print(f"Test Set Rank          : {int(row['Rank'])} / {len(lof_ledger)}")
        print(f"Test Set Percentile    : Top {row['Percentile']:.2f}%")
        print(
            f"OOV Rate               : {oov_rate:.4f} ({len(oov_tokens)} / {total_tokens} tokens)"
        )
        print(f">>> STATUS: {diagnostic_heuristic}")
        print("-" * 80)

        # Accumulate strict metrics for serialization
        metrics_accumulator.append(
            {
                "Document_ID": doc_id,
                "Subset_Origin": subset_name,
                "Date": doc.get("date", "Unknown"),
                "Official_Party": doc.get("titulaire-soutien", "Unknown"),
                "True_Label_y": y_true,
                "Predicted_Label_y_hat": y_pred,
                "Margin_Status": margin_status,
                "LOF_Score": round(row["LOF_Score"], 4),
                "Test_Set_Rank": int(row["Rank"]),
                "Test_Set_Percentile": round(row["Percentile"], 4),
                "OOV_Rate": round(oov_rate, 4),
                "Diagnostic_Heuristic": diagnostic_heuristic,
                "Snippet_Processed": cleaned_text[:500].replace("\n", " ") + "...",
            }
        )

    return metrics_accumulator


def generate_highlighted_topology(
    top_10_ids: list, df_test: pd.DataFrame, lof_ledger: pd.DataFrame
):
    """
    Projects the dataset via UMAP and visually isolates the Top 10 structural anomalies
    strictly against the test background, mapped by categorical political bloc.
    """
    print("\n--- Generating Highlighted Topological Projection ---")
    try:
        X_train_emb = np.load(PROCESSED_DIR / "embeddings_camembert_train.npy")
        X_test_emb = np.load(PROCESSED_DIR / "embeddings_camembert_test.npy")
    except FileNotFoundError:
        print("Fatal: Embeddings missing. Cannot render topological map.")
        return

    # Reconstruct Global Space strictly to guarantee consistent UMAP coordinates
    X_global_emb = np.vstack((X_train_emb, X_test_emb))
    n_train = len(X_train_emb)

    print("Optimizing R^768 -> R^2 manifold for consistent visualization...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
    )
    global_coords_2d = reducer.fit_transform(X_global_emb)

    # Isolate Test Set Coordinates
    test_coords_2d = global_coords_2d[n_train:]

    df_plot = pd.DataFrame(
        {
            "x": test_coords_2d[:, 0],
            "y": test_coords_2d[:, 1],
            "Political_Bloc": lof_ledger.loc[df_test.index, "target_label"],
            "Document_ID": df_test.index,
        },
        index=df_test.index,
    )

    df_plot["Is_Top_10"] = df_plot.index.isin(top_10_ids)

    fig = go.Figure()

    all_blocs = sorted(df_plot["Political_Bloc"].dropna().unique().tolist())
    colors = px.colors.qualitative.Vivid
    color_map = {bloc: colors[i % len(colors)] for i, bloc in enumerate(all_blocs)}

    for bloc in all_blocs:
        df_bloc = df_plot[df_plot["Political_Bloc"] == bloc]
        if df_bloc.empty:
            continue

        # Layer 1: Background Density
        df_bg = df_bloc[~df_bloc["Is_Top_10"]]
        if not df_bg.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_bg["x"],
                    y=df_bg["y"],
                    mode="markers",
                    marker=dict(size=5, color=color_map[bloc], opacity=0.3),
                    name=bloc,
                    legendgroup=bloc,
                    hoverinfo="none",
                )
            )

        # Layer 2: Top 10 Anomalies
        df_fg = df_bloc[df_bloc["Is_Top_10"]]
        if not df_fg.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_fg["x"],
                    y=df_fg["y"],
                    mode="markers+text",
                    text=df_fg["Political_Bloc"],
                    textposition="top center",
                    marker=dict(
                        size=9,
                        color=color_map[bloc],
                        symbol="star",
                        line=dict(width=1, color="black"),
                    ),
                    name=f"{bloc} (Outlier)",
                    legendgroup=bloc,
                    showlegend=False,
                    hovertext=df_fg["Document_ID"],
                )
            )

    fig.update_layout(
        title="<b>Test Set Semantic Topology</b><br><sup>Highlighting Top 10 Statistical Outliers</sup>",
        template="plotly_white",
        width=1200,
        height=900,
        showlegend=True,
    )

    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    html_path = FIGURES_DIR / "test_set_top_10_statistical_outliers.html"
    png_path = FIGURES_DIR / "top_10_statistical_outliers.png"

    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=3)
    print(f"Topological artifacts strictly exported to {FIGURES_DIR}")


def run_outliers_inspection():
    print("Loading test-set baselines and global outlier ledger...")
    try:
        df_corpus = pd.read_csv(PROCESSED_DIR / "cleaned_manifestos.csv", index_col=0)
        df_test = pd.read_csv(PROCESSED_DIR / "X_test_raw.csv", index_col=0)
        ledger_global = pd.read_csv(
            METRICS_DIR / "camembert_lof_outlier_scores.csv", index_col=0
        )
        svm_model = joblib.load(MODELS_DIR / "svm.joblib")
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    except FileNotFoundError as e:
        print(f"Fatal: Required prerequisites missing. {e}")
        sys.exit(1)

    # Enforce strict Test-Set isolation to prevent transductive evaluation leakage
    lof_ledger = ledger_global[ledger_global["split"] == "test"].copy()

    # Compute rank and percentile mathematically relative to N_test
    lof_ledger["Rank"] = lof_ledger["LOF_Score"].rank(ascending=False, method="min")
    lof_ledger["Percentile"] = (lof_ledger["Rank"] / len(lof_ledger)) * 100

    global_vocab = set(tfidf.get_feature_names_out())
    global_metrics_accumulator = []

    # --- PART 1: VISUAL OUTLIERS (O_vis) ---
    visual_anomalies = [
        "EL135_L_1981_06_059_20_1_PF_01",
        "EL195_L_1993_03_075_03_1_PF_04",
        "EL197_L_1993_03_091_06_1_PF_04",
        "EL174_L_1988_06_015_02_1_PF_04",
        "EL191_L_1993_03_033_11_2_PF_01",
        "EL135_L_1981_06_050_03_1_PF_02",
    ]
    vis_metrics = analyze_document_subset(
        visual_anomalies,
        df_corpus,
        lof_ledger,
        svm_model,
        tfidf,
        global_vocab,
        subset_name="VISUAL ANOMALIES",
    )
    global_metrics_accumulator.extend(vis_metrics)

    # --- PART 2: TOP 10 STATISTICAL OUTLIERS (O_stat) ---
    top_10_ledger = lof_ledger.sort_values(by="LOF_Score", ascending=False).head(10)
    top_10_ids = top_10_ledger.index.tolist()

    stat_metrics = analyze_document_subset(
        top_10_ids,
        df_corpus,
        lof_ledger,
        svm_model,
        tfidf,
        global_vocab,
        subset_name="TOP 10 STATISTICAL OUTLIERS",
    )
    global_metrics_accumulator.extend(stat_metrics)

    # --- PART 3: SERIALIZATION ---
    if global_metrics_accumulator:
        df_metrics = pd.DataFrame(global_metrics_accumulator)
        # Drop duplicates in case a document belongs to both O_vis and O_stat
        df_metrics = df_metrics.drop_duplicates(subset=["Document_ID"])
        metrics_csv_path = METRICS_DIR / "outliers_analysis.csv"
        df_metrics.to_csv(metrics_csv_path, index=False, encoding="utf-8")
        print(
            f"\n[SUCCESS] Structural anomaly matrix strictly serialized to: {metrics_csv_path}"
        )

    # --- PART 4: TOPOLOGICAL RENDERING ---
    generate_highlighted_topology(top_10_ids, df_test, lof_ledger)
