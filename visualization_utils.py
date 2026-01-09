import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from logger import setup_logging, get_logger
import logging

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

def plot_model_evaluation(y_true, y_pred, out_dir="model_plots"):
    """
    Evaluate performance using confusion matrix.
    """
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    # 3. Print the TP, FP, TN, FN breakdown
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"True Positives (Caught Ransomware): {tp}")
    logger.info(f"False Positives (Clean files flagged): {fp}")
    logger.info(f"True Negatives (Correctly identified clean): {tn}")
    logger.info(f"False Negatives (Missed Ransomware): {fn}")

    # 4. Print detailed report (Precision, Recall, F1)
    logger.info(f"Detailed Classification Report:\n{classification_report(y_true, y_pred)}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Encrypted'],
                yticklabels=['Clean', 'Encrypted'])

    plt.title('Final Verdict Confusion Matrix')
    plt.ylabel('Actual State')
    plt.xlabel('Predicted State')

    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved Evaluation Plot to: {out_path}")

