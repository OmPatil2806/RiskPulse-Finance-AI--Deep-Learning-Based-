"""
RiskPulse Finance AI — Data Preprocessing Module
Handles cleaning, label encoding, and train/test splitting.
"""

import re
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#  Label mapping 
# We collapse 7 raw classes into 4 clinically meaningful fintech risk categories
LABEL_MAP = {
    "Normal": "Normal",
    "Anxiety": "Anxiety",
    "Depression": "Depression",
    "Stress": "Depression",          # Stress → Depression spectrum
    "Bipolar": "Depression",         # Bipolar → Depression spectrum
    "Suicidal": "Depression",        # Suicidal → Depression spectrum (highest risk)
    "Personality disorder": "Other", # Personality disorder → Other
}

CLASSES = ["Normal", "Anxiety", "Depression", "Other"]


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, special chars, extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s']", " ", text)               # keep letters, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, drop nulls, clean text, map labels."""
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    df.drop(columns=[c for c in df.columns if "unnamed" in c.lower()], inplace=True)

    # Rename for clarity
    df.rename(columns={"statement": "text", "status": "label"}, inplace=True)

    # Drop rows with missing text
    before = len(df)
    df.dropna(subset=["text"], inplace=True)
    logger.info("Dropped %d rows with missing text (%d remain)", before - len(df), len(df))

    # Deduplicate
    df.drop_duplicates(subset=["text"], inplace=True)
    logger.info("After dedup: %d rows", len(df))

    # Map labels to 4 classes
    df["label"] = df["label"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty cleaned texts
    df = df[df["clean_text"].str.len() > 2].copy()

    logger.info("Final dataset shape: %s", df.shape)
    logger.info("Class distribution:\n%s", df["label"].value_counts().to_string())
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame):
    """Encode string labels to integers. Returns df, encoder, and class array."""
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    df["label_enc"] = le.transform(df["label"])
    return df, le


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified 80/20 train-test split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    logger.info("Train size: %d  |  Test size: %d", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict:
    """Compute inverse-frequency class weights for imbalanced training."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = {i: float(w) for i, w in enumerate(weights)}
    logger.info("Class weights: %s", cw)
    return cw


if __name__ == "__main__":
    df = load_and_clean("data\dataset.csv")
    df, le = encode_labels(df)
    train_df, test_df = split_data(df)
    cw = compute_class_weights(train_df["label_enc"].values, num_classes=4)
    print("Preprocessing complete.")
    print("Classes:", le.classes_)
