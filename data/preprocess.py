import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_and_preprocess(test_size: float = 0.2, random_state: int = 42):
    """
    Load and preprocess the Wisconsin Breast Cancer dataset.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Basic sanity checks
    assert X.isnull().sum().sum() == 0, "Unexpected nulls in dataset"
    assert len(X) == len(y), "Feature/label length mismatch"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset: {len(X)} samples | {X.shape[1]} features")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, data.feature_names


if __name__ == "__main__":
    load_and_preprocess()
