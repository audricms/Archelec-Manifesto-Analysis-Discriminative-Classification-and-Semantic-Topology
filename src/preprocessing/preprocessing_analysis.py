import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MANIFESTOS_DIR = DATA_DIR / "manifestos"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures" / "preprocessing_analysis"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def analyze_compression(df_clean: pd.DataFrame):
    """Calculates noise ratio by comparing raw OCR text to cleaned tokens."""
    print("\n--- 1. Calculating Compression Metrics ---")

    raw_texts = []
    for doc_id, row in tqdm(
        df_clean.iterrows(), total=len(df_clean), desc="Fetching Raw Text"
    ):
        year = str(pd.to_datetime(row["date"]).year)
        file_path = MANIFESTOS_DIR / year / "legislatives" / f"{doc_id}.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                raw_texts.append(f.read())
        else:
            raw_texts.append("")

    df_clean["raw_text"] = raw_texts

    df_clean["cleaned_text"] = df_clean["cleaned_text"].fillna("")

    df_clean["raw_length"] = df_clean["raw_text"].apply(lambda x: len(str(x).split()))
    df_clean["clean_length"] = df_clean["cleaned_text"].apply(
        lambda x: len(str(x).split())
    )

    df_clean["noise_ratio"] = np.where(
        df_clean["raw_length"] > 0,
        1 - (df_clean["clean_length"] / df_clean["raw_length"]),
        0,
    )

    print(f"Average Raw Words per Manifesto: {df_clean['raw_length'].mean():.0f}")
    print(f"Average Clean Tokens per Manifesto: {df_clean['clean_length'].mean():.0f}")
    print(
        f"Average Noise Removed: {df_clean['noise_ratio'].mean() * 100:.1f}% per document"
    )

    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df_clean,
        x="raw_length",
        fill=True,
        label="Raw OCR Words",
        color="crimson",
        alpha=0.5,
    )
    sns.kdeplot(
        data=df_clean,
        x="clean_length",
        fill=True,
        label="Cleaned Tokens",
        color="mediumseagreen",
        alpha=0.5,
    )
    plt.title("Document Length Distribution: Before vs. After Preprocessing")
    plt.xlabel("Number of Words / Tokens")
    plt.ylabel("Density")
    plt.xlim(0, 1500)
    plt.legend()
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / "compression_distribution.png", dpi=300)
    print(f"Saved plot to {REPORTS_DIR / 'compression_distribution.png'}")

    print("\n--- Visual Text Verification ---")
    if not df_clean.empty:
        sample_idx = df_clean.index[0]
        print("RAW OCR TEXT (First 300 chars):")
        print("-" * 50)
        print(str(df_clean.loc[sample_idx, "raw_text"])[:300] + "...")
        print("\nPREPROCESSED TEXT (First 300 chars):")
        print("-" * 50)
        print(str(df_clean.loc[sample_idx, "cleaned_text"])[:300] + "...")


def plot_corpus_geometry(
    df: pd.DataFrame,
    label_col: str = "target_label",
    date_col: str = "date",
):
    """Plots the macroscopic class balance and diachronic temporal distribution."""
    print("\n--- 2. Corpus Geometry & Class Balance ---")

    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df[date_col]).dt.year

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    order = df[label_col].value_counts().index

    sns.countplot(
        data=df,
        y=label_col,
        order=order,
        hue=label_col,
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Distribution of Political Blocs (Target Labels)")
    axes[0].set_xlabel("Number of Manifestos")
    axes[0].set_ylabel("Political Bloc")

    temporal_counts = pd.crosstab(df["year"], df[label_col])
    temporal_counts.plot(kind="bar", stacked=True, colormap="tab20", ax=axes[1])
    axes[1].set_title("Temporal Distribution by Political Bloc")
    axes[1].set_xlabel("Election Year")
    axes[1].set_ylabel("Number of Manifestos")
    axes[1].legend(title="Political Bloc", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "corpus_geometry.png", dpi=300)


def plot_lexical_statistics(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "target_label",
):
    """Computes document lengths, vocabulary size |V|, and plots lexical distributions."""
    print("\n--- 3. Lexical Statistics & Geometry ---")

    df[text_col] = df[text_col].fillna("")

    df["doc_length"] = df[text_col].apply(lambda x: len(x.split()))
    df["unique_tokens"] = df[text_col].apply(lambda x: len(set(x.split())))
    order = df[label_col].value_counts().index

    global_vec = CountVectorizer()
    global_vec.fit(df[text_col])
    vocab_size = len(global_vec.vocabulary_)

    print(f"Total documents (N): {len(df)}")
    print(f"Global Vocabulary Size (|V|): {vocab_size:,} unique tokens")
    print(
        f"Average document length: {df['doc_length'].mean():.0f} tokens (Median: {df['doc_length'].median():.0f})"
    )
    print(f"Average UNIQUE tokens per doc: {df['unique_tokens'].mean():.0f}")

    mean_length = df["doc_length"].mean()
    richness = df["unique_tokens"].mean() / mean_length if mean_length > 0 else 0
    print(f"Lexical Richness Ratio: {richness:.2f} (Unique/Total)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    sns.boxplot(
        data=df,
        x="doc_length",
        y=label_col,
        order=order,
        hue=label_col,
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Document Lengths by Political Bloc")
    axes[0].set_xlabel("Number of Tokens")
    axes[0].set_ylabel("Political Bloc")

    sns.histplot(data=df, x="doc_length", bins=50, kde=True, color="indigo", ax=axes[1])
    axes[1].set_title("Global Distribution of Manifesto Lengths")
    axes[1].set_xlabel("Number of Tokens")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "lexical_statistics.png", dpi=300)


def plot_top_ngrams_per_class(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "target_label",
    ngram_range: tuple = (1, 1),
    top_n: int = 10,
    num_classes: int = None,
):
    """Dynamically extracts and plots the top n-grams for all available classes."""
    ngram_name = (
        "Unigrams"
        if ngram_range == (1, 1)
        else "Bigrams"
        if ngram_range == (2, 2)
        else f"{ngram_range}-grams"
    )
    print(f"\n--- 4. Semantic Signatures: Top {top_n} {ngram_name} ---")

    df[text_col] = df[text_col].fillna("")

    # If num_classes is defined, limit it. Otherwise, plot everything sorted by frequency.
    if num_classes is not None:
        major_classes = df[label_col].value_counts().nlargest(num_classes).index
    else:
        major_classes = df[label_col].value_counts().index

    fig, axes = plt.subplots(
        nrows=len(major_classes), ncols=1, figsize=(10, 4 * len(major_classes))
    )

    if len(major_classes) == 1:
        axes = [axes]

    for ax, target_class in zip(axes, major_classes):
        corpus = df[df[label_col] == target_class][text_col]

        if corpus.str.strip().eq("").all():
            print(f"Skipping {target_class} - all documents are empty.")
            continue

        vec = CountVectorizer(ngram_range=ngram_range)
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0)

        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

        if not words_freq:
            continue

        words, freqs = zip(*words_freq)

        sns.barplot(
            x=list(freqs),
            y=list(words),
            ax=ax,
            hue=list(words),
            palette="Blues_r" if ngram_range == (1, 1) else "flare",
            legend=False,
        )
        ax.set_title(f"Top {top_n} {ngram_name} for {target_class}")
        ax.set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"top_{ngram_name.lower()}.png", dpi=300)


def run_preprocessing_analysis():
    """Executes the EDA pipeline on the cleaned dataset."""
    print("\n--- Starting EDA on Preprocessed Manifestos ---")
    cleaned_csv_path = PROCESSED_DIR / "cleaned_manifestos.csv"

    if not cleaned_csv_path.exists():
        print(
            f"Error: {cleaned_csv_path} not found. Please run preprocessing step first."
        )
        sys.exit(1)

    print(f"Loading cleaned dataset from {cleaned_csv_path}...")
    df_clean = pd.read_csv(cleaned_csv_path, index_col=0)

    analyze_compression(df_clean)
    plot_corpus_geometry(df_clean)
    plot_lexical_statistics(df_clean)

    plot_top_ngrams_per_class(
        df_clean,
        ngram_range=(1, 1),
        top_n=10,
    )
    plot_top_ngrams_per_class(
        df_clean,
        ngram_range=(2, 2),
        top_n=10,
    )

    print("\nAnalysis complete.")
