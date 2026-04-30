import sys
import joblib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def generate_features_ledger(svm_model, tfidf_vec, top_n=20):
    """
    Extracts the weight matrix W \in R^{C \times V} and constructs
    a structured ledger of top polarizing tokens per class.
    """
    feature_names = tfidf_vec.get_feature_names_out()
    classes = svm_model.classes_
    data = []

    for idx, target_class in enumerate(classes):
        coef = svm_model.coef_[idx]

        top_pos_idx = coef.argsort()[-top_n:][::-1]
        top_neg_idx = coef.argsort()[:top_n]

        for rank in range(top_n):
            data.append(
                {
                    "Political_Bloc": target_class,
                    "Rank": rank + 1,
                    "Positive_Feature": feature_names[top_pos_idx[rank]],
                    "Pos_Weight": round(coef[top_pos_idx[rank]], 4),
                    "Negative_Feature": feature_names[top_neg_idx[rank]],
                    "Neg_Weight": round(coef[top_neg_idx[rank]], 4),
                }
            )

    return pd.DataFrame(data)


def run_features_analysis():
    print("Loading serialized baseline models...")
    try:
        svm_model = joblib.load(MODELS_DIR / "svm.joblib")
        tfidf_vec = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    except FileNotFoundError as e:
        print(f"Fatal: Required serialized artifacts missing. {e}")
        sys.exit(1)

    print("Extracting discriminative feature weights...")
    df_importance = generate_features_ledger(svm_model, tfidf_vec, top_n=20)

    importance_path = METRICS_DIR / "svm_feature_importance.csv"

    # --- Serialization ---
    df_importance.to_csv(importance_path, index=False, encoding="utf-8")

    print(f"\n[SUCCESS] Feature ledger strictly serialized to: {importance_path}")
