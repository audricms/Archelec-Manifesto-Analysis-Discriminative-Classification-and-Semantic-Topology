import re
import unicodedata
import pandas as pd
import spacy
from typing import List, Set
from pathlib import Path
from tqdm import tqdm


def map_political_blocs_temporal(row: pd.Series) -> str:
    """Maps raw political affiliations to stable macro-blocs diachronically."""
    label = str(row["titulaire-soutien"]).lower()
    year = row["date"].year

    if pd.isna(label) or label == "non mentionné":
        return "UNKNOWN"

    # Left and Far-Right
    if "communiste français" in label:
        return "LEFT_PCF"
    if "front national" in label:
        return "FAR_RIGHT_FN"
    if any(
        x in label for x in ["parti socialiste", "socialiste", "radicaux de gauche"]
    ):
        return "LEFT_PS"
    if any(x in label for x in ["verts", "écologie", "écologiste"]):
        return "ECO"
    if any(x in label for x in ["lutte ouvrière", "ligue communiste"]):
        return "FAR_LEFT"

    # The Right-Wing Bloc (Time-Conditional)
    is_rpr = "rassemblement pour la république" in label
    is_udf = "union pour la démocratie française" in label

    if is_rpr and is_udf:
        # Explicit alliances
        if year == 1981:
            return "RIGHT_UNM"
        if year == 1988:
            return "RIGHT_URC"
        if year == 1993:
            return "RIGHT_UPF"
    elif is_rpr:
        return "RIGHT_RPR"
    elif is_udf:
        return "RIGHT_UDF"

    return "OTHER"


def load_and_filter_metadata(metadata_path: Path) -> pd.DataFrame:
    """Loads metadata, parses dates, and filters for valid legislative targets."""
    df = pd.read_csv(metadata_path, index_col="id")
    df["date"] = pd.to_datetime(df["date"])
    df["target_label"] = df.apply(map_political_blocs_temporal, axis=1)

    main_parties = df[~df["target_label"].str.contains("UNKNOWN|OTHER")].copy()
    main_parties = main_parties[main_parties["contexte-election"] == "législatives"]
    return main_parties


def strip_accents(text: str) -> str:
    """Normalize diacritics to mitigate OCR-induced accent-dropping."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if unicodedata.category(c) != "Mn"
    )


def build_gazetteer(file_paths: List[Path]) -> Set[str]:
    """Constructs a deterministic geographical exclusion set."""
    POLITICAL_SAFE_WORDS = {
        "nouvelle",
        "nouveau",
        "grand",
        "grande",
        "centre",
        "pays",
        "nord",
        "sud",
        "est",
        "ouest",
        "union",
        "val",
        "haute",
        "hauts",
        "bas",
    }

    series_list = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, usecols=["LIBELLE"])
            series_list.append(df["LIBELLE"])
        except ValueError:
            raise RuntimeError(f"Column 'LIBELLE' not found in {path}")

    raw_gazetteer = pd.concat(series_list, ignore_index=True).dropna()
    gazetteer = set()
    tokenizer = re.compile(r"\b[a-z]{3,}\b")

    for name in raw_gazetteer:
        norm_name = strip_accents(name.lower())
        tokens = tokenizer.findall(norm_name)
        for token in tokens:
            if token not in POLITICAL_SAFE_WORDS:
                gazetteer.add(token)

    return gazetteer


class ArchelecPreprocessor:
    def __init__(self, gazetteer_paths: List[Path]):
        print("Initializing spaCy pipeline and compiling gazetteer...")
        try:
            self.nlp = spacy.load("fr_core_news_lg", disable=["parser"])
        except OSError:
            raise RuntimeError("Run: uv run python -m spacy download fr_core_news_lg")

        self.gazetteer = build_gazetteer(gazetteer_paths)

        discourse_markers = {
            "madame",
            "monsieur",
            "mademoiselle",
            "électrice",
            "électeur",
            "concitoyen",
            "cher",
            "chère",
            "salutation",
            "dévoué",
            "fidèle",
            "candidat",
            "circonscription",
            "vote",
            "voter",
            "élection",
            "scrutin",
            "tour",
            "premier",
            "deuxième",
            "suppléant",
            "bulletin",
            "suffrage",
            "député",
            "canton",
            "maire",
            "sortant",
            "mandat",
            "assemblée",
            "vouloir",
            "pouvoir",
            "devoir",
            "faire",
            "falloir",
            "mettre",
            "aller",
            "agir",
            "savoir",
            "répondre",
            "attendre",
            "permettre",
            "croire",
            "ici",
            "oui",
            "non",
            "beaucoup",
            "ensemble",
            "bien",
            "toujours",
            "aujourd'hui",
            "maintenant",
            "ainsi",
            "plus",
            "moins",
            "année",
            "suite",
            "jour",
            "nouveau",
            "nouvelle",
            "fois",
            "moment",
            "cadre",
            "objet",
            "point",
            "particulier",
            "notamment",
            "janvier",
            "février",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre",
            "lundi",
            "mardi",
            "mercredi",
            "jeudi",
            "vendredi",
            "samedi",
            "dimanche",
            "printemps",
            "été",
            "automne",
            "hiver",
        }
        self.discourse_markers_norm = {
            strip_accents(w).lower() for w in discourse_markers
        }

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = re.sub(r"\bP\.?\s*S\.?\b", "parti_socialiste", text, flags=re.IGNORECASE)
        text = re.sub(r"\bP\.?\s*C\.?\b", "parti_communiste", text, flags=re.IGNORECASE)
        text = re.sub(r"\bR\.?\s*P\.?\s*R\.?\b", "rpr", text, flags=re.IGNORECASE)
        text = re.sub(r"\bU\.?\s*D\.?\s*F\.?\b", "udf", text, flags=re.IGNORECASE)

        text = re.sub(r"-\s+", "", text)
        text = re.sub(r"(?i)sciences\s*po\s*/\s*fonds\s*cevipof", "", text)
        text = re.sub(r"(?i)imprimer[il]e\s+[^\n]+", "", text)
        text = re.sub(r"(?i)vu\s+les\s+candidats?", "", text)

        doc = self.nlp(text)
        sanitized_tokens = []
        negation_active = False

        for token in doc:
            if token.is_punct:
                negation_active = False
                continue

            if token.lemma_.lower() in ["ne", "n'"]:
                negation_active = True
                continue

            if token.ent_type_ in ["LOC", "PER"]:
                continue

            if token.pos_ not in ["NOUN", "ADJ", "VERB", "PROPN"]:
                continue

            lemma_norm = strip_accents(token.lemma_.lower())

            if lemma_norm in ["etre", "avoir"]:
                continue

            if (
                lemma_norm in self.gazetteer
                or lemma_norm in self.discourse_markers_norm
            ):
                continue

            if token.is_stop or token.is_space:
                continue

            if len(lemma_norm) < 3 or not lemma_norm.isalpha():
                continue

            if negation_active:
                lemma_norm = "ne_" + lemma_norm
                negation_active = False

            sanitized_tokens.append(lemma_norm)

        return " ".join(sanitized_tokens)


def run_preprocessing():
    """Runs the NLP pipeline on extracted text and saves the final dataframe."""
    print("\n--- Starting NLP Preprocessing Pipeline ---")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    MANIFESTOS_DIR = DATA_DIR / "manifestos"
    GAZETTEERS_DIR = DATA_DIR / "gazetteers"
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = MANIFESTOS_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. Run extraction first."
        )

    print("Loading and filtering metadata...")
    df_clean = load_and_filter_metadata(metadata_path)

    insee_files = [
        GAZETTEERS_DIR / "v_arrondissement_2025.csv",
        GAZETTEERS_DIR / "v_commune_2025.csv",
        GAZETTEERS_DIR / "v_commune_comer_2025.csv",
        GAZETTEERS_DIR / "v_departement_2025.csv",
        GAZETTEERS_DIR / "v_region_2025.csv",
        GAZETTEERS_DIR / "v_comer_2025.csv",
    ]
    preprocessor = ArchelecPreprocessor(insee_files)

    print("\nReading and cleaning manifesto texts...")
    cleaned_texts = []
    valid_indices = []

    for doc_id, row in tqdm(
        df_clean.iterrows(), total=len(df_clean), desc="Processing Documents"
    ):
        year = str(row["date"].year)
        file_path = MANIFESTOS_DIR / year / "legislatives" / f"{doc_id}.txt"

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            cleaned_texts.append(preprocessor.clean_text(raw_text))
            valid_indices.append(doc_id)
        else:
            continue

    df_final = df_clean.loc[valid_indices].copy()
    df_final["cleaned_text"] = cleaned_texts

    output_path = PROCESSED_DIR / "cleaned_manifestos.csv"
    df_final.to_csv(output_path, index=True)
    print(f"\nPipeline complete! Cleaned dataset saved to {output_path}")
