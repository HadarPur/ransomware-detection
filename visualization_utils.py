import matplotlib.pyplot as plt
import os

def visualization(df, feature, target, title, y_label, out_dir="plots"):
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

def feature_visualization(df):
    visualization(df, "compression_ratio", "label", "Compression Ratio by Class", "Compressed / Original Size")
    visualization(df, "entropy_std", "label", "Entropy Variance Across Chunks", "Entropy Std")
    visualization(df, "entropy", "label", "Global Entropy by Class", "Entropy")
    visualization(df, "zero_byte_ratio", "label", "Zero-Byte Ratio by Class", "Zero Byte Ratio")