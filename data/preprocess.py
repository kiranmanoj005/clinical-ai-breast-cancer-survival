import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pathlib import Path


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag outliers using the IQR method.
    Returns a boolean DataFrame — True where a value is an outlier.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    n_outliers = outlier_mask.sum().sum()
    print(f"[Outlier Detection] {n_outliers} outlier cells detected across {df.shape[1]} features.")
    return outlier_mask


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, output_dir: str = "outputs") -> None:
    """
    Plot per-feature KDE distributions split by class and save to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features = X.columns.tolist()
    n_cols = 5
    n_rows = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    palette = {0: "#e74c3c", 1: "#2ecc71"}
    labels = {0: "Malignant", 1: "Benign"}

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for cls in [0, 1]:
            subset = X[y == cls][feat]
            sns.kdeplot(subset, ax=ax, label=labels[cls],
                        color=palette[cls], fill=True, alpha=0.35)
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.legend(fontsize=6)

    # Hide unused axes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class (Benign vs Malignant)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = Path(output_dir) / "feature_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Feature Distributions] Saved to {out_path}")


def plot_outlier_heatmap(outlier_mask: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Save a heatmap showing which samples and features contain outliers.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(
        outlier_mask.T.astype(int),
        cmap="Reds",
        cbar_kws={"label": "Outlier (1=Yes)"},
        ax=ax,
        yticklabels=True,
        xticklabels=False
    )
    ax.set_title("Outlier Map (IQR method) — Features × Samples",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    out_path = Path(output_dir) / "outlier_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Outlier Heatmap] Saved to {out_path}")


def load_and_preprocess(
    test_size: float = 0.2,
    random_state: int = 42,
    run_outlier_detection: bool = False,
    plot_distributions: bool = False,
    output_dir: str = "outputs"
):
    """
    Load and preprocess the Wisconsin Breast Cancer dataset.
    Optionally runs outlier detection and saves feature distribution plots.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Basic sanity checks
    assert X.isnull().sum().sum() == 0, "Unexpected nulls in dataset"
    assert len(X) == len(y), "Feature/label length mismatch"

    if run_outlier_detection:
        outlier_mask = detect_outliers_iqr(X)
        plot_outlier_heatmap(outlier_mask, output_dir=output_dir)

    if plot_distributions:
        plot_feature_distributions(X, y, output_dir=output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset: {len(X)} samples | {X.shape[1]} features")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, data.feature_names


if __name__ == "__main__":
    load_and_preprocess(
        run_outlier_detection=True,
        plot_distributions=True
    )
