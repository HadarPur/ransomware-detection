import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from logger import setup_logging, get_logger

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

def plot_confidence_distribution(predictions_df):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=predictions_df, x='confidence_score', hue='verdict', multiple="stack", bins=10)
    plt.title('Distribution of Model Confidence Scores')
    plt.xlabel('Confidence (0.0 to 1.0)')
    plt.ylabel('Number of Files')
    plt.show()

def plot_model_evaluation(y_true, y_pred, out_dir="model_plots",
                          title="Confusion Matrix", file_prefix="confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    logger.info(f"[{file_prefix}] TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    logger.info(f"[{file_prefix}] Classification report:\n{classification_report(y_true, y_pred, digits=4, zero_division=0)}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Clean', 'Encrypted'],
        yticklabels=['Clean', 'Encrypted']
    )
    plt.title(title)
    plt.ylabel('Actual State')
    plt.xlabel('Predicted State')

    out_path = os.path.join(out_dir, f"{file_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Evaluation Plot to: {out_path}\n")

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"[{file_prefix}] Accuracy: {acc:.4f}")
    logger.info(f"[{file_prefix}] Precision: {precision:.4f}")
    logger.info(f"[{file_prefix}] Recall: {recall:.4f}")
    logger.info(f"[{file_prefix}] F1-score: {f1:.4f}\n")
