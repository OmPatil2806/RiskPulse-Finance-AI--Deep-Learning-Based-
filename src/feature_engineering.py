"""
RiskPulse Finance AI — Feature Engineering Module
Handles tokenization + padding (LSTM) and TF-IDF (baseline).
IMPORTANT: Tokenizer and TF-IDF are fit ONLY on training data.
"""

import logging
import pickle
import numpy as np
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#  Hyperparameters 
VOCAB_SIZE    = 20_000
MAX_SEQ_LEN   = 150
OOV_TOKEN     = "<OOV>"
TFIDF_MAX_FEATURES = 15_000


class TextFeatureEngineer:
    """
    Wraps tokenizer (LSTM path) and TF-IDF vectoriser (baseline path).
    Fit only on training data; transform can be called on any split.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        tfidf_max_features: int = TFIDF_MAX_FEATURES,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tfidf_max_features = tfidf_max_features

        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=OOV_TOKEN)
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )
        self._fitted = False

    #  Fit 

    def fit(self, train_texts: list[str]) -> None:
        """Fit tokenizer and TF-IDF on TRAINING data only."""
        logger.info("Fitting tokenizer on %d training samples…", len(train_texts))
        self.tokenizer.fit_on_texts(train_texts)
        logger.info("Vocab size (actual): %d", len(self.tokenizer.word_index))

        logger.info("Fitting TF-IDF on %d training samples…", len(train_texts))
        self.tfidf.fit(train_texts)
        self._fitted = True
        logger.info("Feature engineering fit complete.")

    #  Transform 

    def texts_to_sequences(self, texts: list[str]) -> np.ndarray:
        """Tokenize + pad → (N, max_seq_len) int array for LSTM."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_seq_len, padding="post", truncating="post")
        return padded

    def texts_to_tfidf(self, texts: list[str]) -> np.ndarray:
        """TF-IDF transform → sparse matrix for baseline models."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self.tfidf.transform(texts)

    #  Persist 

    def save(self, path: str = "../models") -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open(f"{path}/tfidf.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        logger.info("Saved tokenizer and TF-IDF to %s", path)

    @classmethod
    def load(cls, path: str = "../models") -> "TextFeatureEngineer":
        eng = cls()
        with open(f"{path}/tokenizer.pkl", "rb") as f:
            eng.tokenizer = pickle.load(f)
        with open(f"{path}/tfidf.pkl", "rb") as f:
            eng.tfidf = pickle.load(f)
        eng._fitted = True
        logger.info("Loaded tokenizer and TF-IDF from %s", path)
        return eng

    #  Properties 
    
    

    @property
    def actual_vocab_size(self) -> int:
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)


def prepare_lstm_inputs(texts: list[str], engineer: TextFeatureEngineer) -> np.ndarray:
    """Convenience: return padded sequences."""
    return engineer.texts_to_sequences(texts)


def prepare_tfidf_inputs(texts: list[str], engineer: TextFeatureEngineer):
    """Convenience: return TF-IDF sparse matrix."""
    return engineer.texts_to_tfidf(texts)

if __name__ == "__main__":
    from data_preprocessing import load_and_clean, encode_labels, split_data

    df = load_and_clean("data\dataset.csv")
    df, le = encode_labels(df)
    train_df, test_df = split_data(df)

    engineer = TextFeatureEngineer()

    # Fit only on train data
    engineer.fit(train_df["clean_text"].tolist())

    # Transform
    X_train = engineer.texts_to_sequences(train_df["clean_text"].tolist())
    X_test = engineer.texts_to_sequences(test_df["clean_text"].tolist())

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    engineer.save()