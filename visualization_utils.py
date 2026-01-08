import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_feature_distribution(df, feature, target, title, y_label, out_dir="plots"):
    """
    Visualizes the relationship between a feature and a target variable.

    Parameters:
    feature (pd.Series): The feature to visualize.
    target (pd.Series): The target variable to compare against.
    title (str): The title of the plot.
    y_label (str): The label for the y-axis.
    """

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(20, 12))

    df.boxplot(column=feature, by=target)
    plt.title(title)
    plt.suptitle("")
    plt.ylabel(y_label)
    plt.grid(True)

    plt.text(
        0.5, 0.95,
        "Box = IQR (25–75%)\nGreen line = Median\nDots = Outliers",
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # Construct filename
    filename = f"{feature}_by_{target}.png"
    out_path = os.path.join(out_dir, filename)

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved plot to: {out_path}")

def run_exploratory_visualizations(df):
    plot_feature_distribution(df, "compression_ratio", "label", "Compression Ratio by Class", "Compressed / Original Size")
    plot_feature_distribution(df, "entropy_std", "label", "Entropy Variance Across Chunks", "Entropy Std")
    plot_feature_distribution(df, "entropy", "label", "Global Entropy by Class", "Entropy")
    plot_feature_distribution(df, "zero_byte_ratio", "label", "Zero-Byte Ratio by Class", "Zero Byte Ratio")

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
    print(f"True Positives (Caught Ransomware): {tp}")
    print(f"False Positives (Clean files flagged): {fp}")
    print(f"True Negatives (Correctly identified clean): {tn}")
    print(f"False Negatives (Missed Ransomware): {fn}")

    # 4. Print detailed report (Precision, Recall, F1)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))

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
    print(f"Saved Evaluation Plot to: {out_path}")

