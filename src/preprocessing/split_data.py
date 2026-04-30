import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def split_data():
    """Splits the cleaned data into train/test while preserving indices and metadata."""
    print("\n--- 1. Loading Cleaned Data ---")
    cleaned_csv_path = PROCESSED_DIR / "cleaned_manifestos.csv"

    if not cleaned_csv_path.exists():
        print(f"Fatal: {cleaned_csv_path} not found. Execute preprocessing step first.")
        sys.exit(1)

    df_clean = pd.read_csv(cleaned_csv_path, index_col=0)
    df_clean["cleaned_text"] = df_clean["cleaned_text"].fillna("")

    required_cols = ["date", "cleaned_text", "titulaire-soutien", "target_label"]
    missing = [col for col in required_cols if col not in df_clean.columns]
    if missing:
        print(f"Fatal: Missing required metadata columns in dataset: {missing}")
        sys.exit(1)

    print("\n--- 2. Train/Test Split (Stratified) ---")

    X = df_clean[["date", "cleaned_text", "titulaire-soutien"]]
    y = df_clean["target_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Set: {len(X_train)} documents")
    print(f"Testing Set:  {len(X_test)} documents")

    print("\n--- 3. Serializing Splits ---")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=True)
    X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=True)

    pd.DataFrame(y_train).to_csv(PROCESSED_DIR / "y_train.csv", index=True)
    pd.DataFrame(y_test).to_csv(PROCESSED_DIR / "y_test.csv", index=True)

    print(f"Split data serialized successfully to {PROCESSED_DIR}")
