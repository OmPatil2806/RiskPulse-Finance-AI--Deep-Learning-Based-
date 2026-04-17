"""
RiskPulse Finance AI — Model Architecture Module
Defines BiLSTM deep learning model and baseline LogisticRegression model.
"""

import logging
from tensorflow import keras
from tensorflow.keras import layers, regularizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 4


def build_bilstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    max_seq_len: int = 150,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    BiLSTM text classifier with:
      Embedding → SpatialDropout → BiLSTM (x2, stacked) → GlobalMaxPool
      → Dense(ReLU) → Dropout → Dense(Softmax)
    """
    inputs = keras.Input(shape=(max_seq_len,), name="token_ids")

    #  Embedding 
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_seq_len,
        name="embedding",
    )(inputs)
    x = layers.SpatialDropout1D(dropout_rate, name="spatial_dropout")(x)

    #  BiLSTM stack 
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_1",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units // 2,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_2",
    )(x)

    #  Pooling 
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # ── Dense head 
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(32, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RiskPulse_BiLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("BiLSTM model built. Total params: %s", model.count_params())
    model.summary(print_fn=logger.info)
    return model


def build_baseline_model():
    """Lightweight logistic regression baseline over TF-IDF features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
    n_jobs=-1,
    )
    logger.info("Baseline Logistic Regression model created.")
    return model


def load_bilstm(path: str) -> keras.Model:
    """Load a saved BiLSTM model from disk."""
    model = keras.models.load_model(path)
    logger.info("Loaded BiLSTM model from %s", path)
    return model

if __name__ == "__main__":
    model = build_bilstm_model(
        vocab_size=20000,
        embedding_dim=128,
        max_seq_len=150
    )