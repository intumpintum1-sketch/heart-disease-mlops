from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_FEATURES_PATH = RAW_DATA_DIR / "heart_disease_features.csv"
RAW_TARGET_PATH = RAW_DATA_DIR / "heart_disease_target.csv"


def download_dataset():
    """Download the UCI Heart Disease dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    X.to_csv(RAW_FEATURES_PATH, index=False)
    y.to_csv(RAW_TARGET_PATH, index=False)

    print("Dataset downloaded successfully.")
    print(f"Features saved to: {RAW_FEATURES_PATH}")
    print(f"Target saved to: {RAW_TARGET_PATH}")

    return X, y


def validate_dataset(X, y):
    """Perform basic validation."""

    print("\nDataset validation")
    print("------------------")

    print(f"Feature rows: {X.shape[0]}")
    print(f"Feature columns: {X.shape[1]}")
    print(f"Target rows: {y.shape[0]}")

    print("\nFeature columns:")
    print(list(X.columns))

    print("\nMissing values:")
    print(X.isnull().sum())

    print("\nTarget distribution:")
    print(y.iloc[:, 0].value_counts().sort_index())


if __name__ == "__main__":
    X, y = download_dataset()
    validate_dataset(X, y)