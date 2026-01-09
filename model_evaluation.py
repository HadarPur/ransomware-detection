import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from logger import setup_logging, get_logger
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

def evaluate_model_performance(detector, test_df):
    """
    Implementation for Task 7: Evaluation metrics and per-variant results.
    """
    # Generate predictions
    y_true = test_df['is_encrypted']
    y_pred_labels = detector.evaluate_batch(test_df)

    # Map 'ENCRYPTED' back to 1 and 'NOT ENCRYPTED' to 0 for metrics
    y_pred = y_pred_labels.map({'ENCRYPTED': 1, 'NOT ENCRYPTED': 0})

    # 1. Overall Performance
    print("")
    logger.info("--- Overall Performance ---")
    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = cm.ravel()
    logger.info(f"TP: {TP}")
    logger.info(f"TN: {TN}")
    logger.info(f"FP: {FP}")
    logger.info(f"FN: {FN}")

    logger.info(f"Detailed Report:\n{classification_report(y_true, y_pred)}")

    # # 2. Per-Variant Results (if variant column exists)
    # if 'variant' in test_df.columns:
    #     print("\n--- Per-Ransomware-Variant Accuracy ---")
    #     for variant in test_df['variant'].unique():
    #         variant_data = test_df[test_df['variant'] == variant]
    #         v_true = variant_data['is_encrypted']
    #         v_pred_labels = detector.evaluate_batch(variant_data)
    #         v_pred = v_pred_labels.map({'ENCRYPTED': 1, 'NOT ENCRYPTED': 0})
    #
    #         acc = accuracy_score(v_true, v_pred)
    #         print(f"Variant {variant}: {acc * 100:.1f}% detection rate")

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
    logger.info(f"Saved Evaluation Plot to: {out_path}\n")

    # Calculate overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)  # by default, for positive class (ENCRYPTED)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Log metrics
    logger.info(f"-------- Overall Performance --------")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    print("")


