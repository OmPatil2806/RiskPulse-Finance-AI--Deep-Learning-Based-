"""
RiskPulse Finance AI — Training Pipeline
Orchestrates: preprocessing → feature engineering → model training → saving.
"""

import os
import sys
import json
import logging
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_preprocessing import load_and_clean, encode_labels, split_data, compute_class_weights
from feature_engineering import TextFeatureEngineer
from model import build_bilstm_model, build_baseline_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path(__file__).parent.parent / "data" / "dataset.csv"
MODELS_DIR   = Path(__file__).parent.parent / "models"
EPOCHS       = 15
BATCH_SIZE   = 64
VOCAB_SIZE   = 20_000
EMBED_DIM    = 128
LSTM_UNITS   = 128
MAX_SEQ_LEN  = 150
DROPOUT      = 0.3
LR           = 1e-3
NUM_CLASSES  = 4
RANDOM_STATE = 42


def set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train() -> None:
    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    df = load_and_clean(str(DATA_PATH))
    df, le = encode_labels(df)
    train_df, test_df = split_data(df, test_size=0.2, random_state=RANDOM_STATE)

    # Save label encoder
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.info("Label encoder saved.")

    # Save class names for inference
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)

    # ── 2. Feature engineering (fit ONLY on train) ────────────────────────────
    fe = TextFeatureEngineer(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)
    fe.fit(train_df["clean_text"].tolist())
    fe.save(str(MODELS_DIR))

    X_train = fe.texts_to_sequences(train_df["clean_text"].tolist())
    X_test  = fe.texts_to_sequences(test_df["clean_text"].tolist())

    y_train = to_categorical(train_df["label_enc"].values, num_classes=NUM_CLASSES)
    y_test  = to_categorical(test_df["label_enc"].values,  num_classes=NUM_CLASSES)

    y_train_int = train_df["label_enc"].values

    # ── 3. Class weights ──────────────────────────────────────────────────────
    cw = compute_class_weights(y_train_int, num_classes=NUM_CLASSES)
    with open(MODELS_DIR / "class_weights.json", "w") as f:
        json.dump(cw, f)

    # ── 4. Build BiLSTM ───────────────────────────────────────────────────────
    model = build_bilstm_model(
        vocab_size=fe.actual_vocab_size,
        embedding_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT,
        num_classes=NUM_CLASSES,
        learning_rate=LR,
    )

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_bilstm.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(str(MODELS_DIR / "training_log.csv")),
    ]

    # ── 6. Train ──────────────────────────────────────────────────────────────
    logger.info("Starting BiLSTM training…")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Save final model ───────────────────────────────────────────────────
    model.save(str(MODELS_DIR / "trained_model.keras"))
    # Also save .h5 for compatibility
    model.save(str(MODELS_DIR / "trained_model.h5"))
    logger.info("Model saved.")

    # ── 8. Quick eval on test set ─────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test Loss: %.4f  |  Test Accuracy: %.4f", loss, acc)

    # ── 9. Baseline (Logistic Regression on TF-IDF) ───────────────────────────
    logger.info("Training baseline Logistic Regression on TF-IDF…")
    X_train_tfidf = fe.texts_to_tfidf(train_df["clean_text"].tolist())
    X_test_tfidf  = fe.texts_to_tfidf(test_df["clean_text"].tolist())
    y_train_int   = train_df["label_enc"].values
    y_test_int    = test_df["label_enc"].values

    baseline = build_baseline_model()
    baseline.fit(X_train_tfidf, y_train_int)
    bl_acc = baseline.score(X_test_tfidf, y_test_int)
    logger.info("Baseline LR Test Accuracy: %.4f", bl_acc)

    with open(MODELS_DIR / "baseline_lr.pkl", "wb") as f:
        pickle.dump(baseline, f)
    logger.info("Baseline model saved.")

    # ── 10. Save test data for evaluation ─────────────────────────────────────
    test_df.to_csv(MODELS_DIR / "test_data.csv", index=False)
    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    train()
