
import pandas as pd
import pytest

from src.preprocessing import FEATURES, validate_dataset


def make_valid_data():
    X = pd.DataFrame(
        [{feature: 1 for feature in FEATURES} for _ in range(4)]
    )
    y = pd.Series([0, 1, 0, 1], name="target")
    return X, y


def test_valid_dataset_passes():
    X, y = make_valid_data()
    validate_dataset(X, y)


def test_missing_feature_is_rejected():
    X, y = make_valid_data()
    X = X.drop(columns=["age"])

    with pytest.raises(ValueError, match="Missing required features"):
        validate_dataset(X, y)


def test_mismatched_lengths_are_rejected():
    X, y = make_valid_data()

    with pytest.raises(ValueError, match="different lengths"):
        validate_dataset(X, y.iloc[:-1])


def test_non_binary_target_is_rejected():
    X, y = make_valid_data()
    y.iloc[0] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dataset(X, y)


def test_missing_target_is_rejected():
    X, y = make_valid_data()
    y.iloc[0] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(X, y)


def test_empty_dataset_is_rejected():
    X, y = make_valid_data()
    X = X.iloc[0:0]
    y = y.iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        validate_dataset(X, y)
