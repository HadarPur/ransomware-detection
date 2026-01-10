import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger import setup_logging, get_logger, logging
from sklearn.metrics import confusion_matrix

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

def run_false_positive_assessment(detector, validation_df, out_dir="model_plots/validation"):
    """
    Assesses the False Positive Rate (FPR) on the 'More_Clean_Files' dataset.
    Since all files are known to be clean, any 'Encrypted' prediction is a False Positive.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # We must use the exact same columns used during detector.train()
    X_val = validation_df[detector.feature_cols]

    # Since these are all clean files, the true labels are all 0 (Not Encrypted)
    y_true_val = validation_df["is_encrypted"]

    # 4. Generate Predictions using the detector's evaluate method
    # This returns: (ensemble_pred, svc_pred, lr_pred, knn_pred)
    val_results = detector.evaluate(X_val, y_true_val, False, dataset_name="More_Clean_Files")
    ensemble_predictions = val_results[0]

    # 5. Calculate Metrics
    total_files = len(X_val)
    # sum() works because Encrypted = 1 and Clean = 0
    fp_count = sum(ensemble_predictions)
    fpr = (fp_count / total_files) * 100 if total_files > 0 else 0

    # 6. Report Results
    logger.info(f"Total Benign Files Tested: {total_files}")
    logger.info(f"Number of False Positives: {fp_count}")
    logger.info(f"False Positive Rate (FPR): {fpr:.2f}%")

    # 7. Visualize for Deliverables
    # Using your existing plotting function to show the Confusion Matrix
    plot_model_assessment(
        y_true_val,
        ensemble_predictions,
        out_dir=out_dir,
        title="Validation - False Positive Analysis",
        file_prefix="Ensemble"
    )

    return fpr

def plot_model_assessment(y_true, y_pred, out_dir="model_plots",
                          title="Confusion Matrix", file_prefix="confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    cm = cm[:1, :]

    plt.figure(figsize=(8, 3))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Clean', 'Encrypted'],
        yticklabels=['Clean']
    )
    plt.title(title)
    plt.ylabel('Actual State')
    plt.xlabel('Predicted State')

    out_path = os.path.join(out_dir, f"{file_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Evaluation Plot to: {out_path}\n")