"""
RiskPulse Finance AI — Evaluation Module
Generates full classification report, confusion matrix, and risk score summary.
"""

import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import TextFeatureEngineer
from risk_score import compute_risk_score, risk_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR  = Path(__file__).parent.parent / "notebooks" / "eval_plots"
NUM_CLASSES = 4


def load_artifacts():
    model    = load_model(str(MODELS_DIR / "trained_model.keras"))
    fe       = TextFeatureEngineer.load(str(MODELS_DIR))
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    X_test   = np.load(MODELS_DIR / "X_test.npy")
    y_test   = np.load(MODELS_DIR / "y_test.npy")
    test_df  = pd.read_csv(MODELS_DIR / "test_data.csv")
    return model, fe, le, X_test, y_test, test_df


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("RiskPulse — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_training_history(save_path: Path) -> None:
    log_path = MODELS_DIR / "training_log.csv"
    if not log_path.exists():
        logger.warning("Training log not found; skipping history plot.")
        return
    log = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(log["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(log["val_accuracy"], label="Val Acc",    linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(log["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(log["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("RiskPulse Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training history plot saved to %s", save_path)


def evaluate() -> dict:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, fe, le, X_test, y_test, test_df = load_artifacts()
    class_names = le.classes_.tolist()

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred_proba = model.predict(X_test, batch_size=256, verbose=0)
    y_pred_int   = np.argmax(y_pred_proba, axis=1)
    y_true_int   = np.argmax(y_test, axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true_int, y_pred_int)
    report = classification_report(y_true_int, y_pred_int, target_names=class_names, digits=4)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, average="weighted"
    )
    cm = confusion_matrix(y_true_int, y_pred_int)

    logger.info("=== EVALUATION RESULTS ===")
    logger.info("Accuracy : %.4f", acc)
    logger.info("Precision: %.4f  Recall: %.4f  F1: %.4f", p, r, f1)
    logger.info("\n%s", report)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, class_names, PLOTS_DIR / "confusion_matrix.png")
    plot_training_history(PLOTS_DIR / "training_history.png")

    # ── Risk score summary ────────────────────────────────────────────────────
    test_df = test_df.copy()
    test_df["predicted_class"] = le.inverse_transform(y_pred_int)
    test_df["risk_score"]  = test_df["predicted_class"].apply(
        lambda c: compute_risk_score(c, y_pred_proba[test_df.index[test_df["predicted_class"] == c][0]])
        if (test_df["predicted_class"] == c).any() else 0
    )
    # Vectorised risk scores
    scores, levels = [], []
    for i, (cls, proba) in enumerate(zip(
        le.inverse_transform(y_pred_int), y_pred_proba
    )):
        s = compute_risk_score(cls, proba)
        scores.append(s)
        levels.append(risk_level(s))
    test_df["risk_score"] = scores
    test_df["risk_level"] = levels

    logger.info("Risk level distribution:\n%s", test_df["risk_level"].value_counts().to_string())
    test_df.to_csv(MODELS_DIR / "evaluation_results.csv", index=False)

    # Save metrics JSON
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(p), 4),
        "recall_weighted": round(float(r), 4),
        "f1_weighted": round(float(f1), 4),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete. Artifacts saved to %s", MODELS_DIR)
    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print("\n=== FINAL METRICS ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
