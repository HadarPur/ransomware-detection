import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from logger import setup_logging, get_logger
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup logging configuration to print to console
setup_logging(level=logging.INFO, log_to_file=False)
logger = get_logger(__name__)

def plot_feature_distribution(df, feature, target, title, y_label, out_dir="data_exploration_plots"):
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


def plot_correlation_matrix(df, out_dir="data_exploration_plots"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=['float64', 'int64']).corr()  # numeric features only
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Matrix (Pearson)")

    out_path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved correlation matrix to: {out_path}")

def plot_pca(df, features=None, hue="label", out_dir="data_exploration_plots"):
    os.makedirs(out_dir, exist_ok=True)

    if features is None:
        features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)  # scale for PCA

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    df_pca = df.copy()
    df_pca['PC1'] = components[:, 0]
    df_pca['PC2'] = components[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue=hue, s=60, alpha=0.7)
    plt.title(f"PCA Projection (2D) of Features")

    out_path = os.path.join(out_dir, "pca_2d.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved PCA plot to: {out_path}")



def plot_pairplot(df, features=None, hue="label", out_dir="data_exploration_plots"):
    os.makedirs(out_dir, exist_ok=True)

    if features is None:
        features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    sns.pairplot(df[features + [hue]], hue=hue, diag_kind="kde", corner=True)

    out_path = os.path.join(out_dir, "pairplot.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved pairplot to: {out_path}")



def run_exploratory_visualizations(df):
    logger.info(f"----------- Generating Exploratory DataFrame -----------")
    logger.info(f'Data head:\n{df.head(40)}\n')

    logger.info(f"----------- Generating Features Distribution -----------")
    plot_feature_distribution(df, "compression_ratio", "label", "Compression Ratio by Class", "Compressed / Original Size")
    plot_feature_distribution(df, "entropy_std", "label", "Entropy Variance Across Chunks", "Entropy Std")
    plot_feature_distribution(df, "entropy", "label", "Global Entropy by Class", "Entropy")
    plot_feature_distribution(df, "zero_byte_ratio", "label", "Zero-Byte Ratio by Class", "Zero Byte Ratio")
    plot_feature_distribution(df, "chi_square_normalized", "label", "Chi Square by Class", "Chi Square Normalized")
    plot_feature_distribution(df, "serial_byte_correlation", "label", "Serial Byte Correlation by Class", "Serial Byte Correlation")
    print("")

    logger.info(f"----------- Generating Features Correlation -----------")
    plot_correlation_matrix(df)
    plot_pairplot(df)
    plot_pca(df)

    print("")
