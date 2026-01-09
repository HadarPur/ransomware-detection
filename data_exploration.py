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

    logger.info(f"Saved plot to: {out_path}")

def run_exploratory_visualizations(df):
    logger.info(f"----------- Generating Exploratory DataFrame -----------")
    logger.info(f'{df.head(40)}\n')

    logger.info(f"----------- Generating Exploratory Visualizations -----------")
    plot_feature_distribution(df, "compression_ratio", "label", "Compression Ratio by Class", "Compressed / Original Size")
    plot_feature_distribution(df, "entropy_std", "label", "Entropy Variance Across Chunks", "Entropy Std")
    plot_feature_distribution(df, "entropy", "label", "Global Entropy by Class", "Entropy")
    plot_feature_distribution(df, "zero_byte_ratio", "label", "Zero-Byte Ratio by Class", "Zero Byte Ratio")
