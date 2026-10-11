
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURES,
    build_model_pipeline,
    validate_dataset,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 80

    X = pd.DataFrame(index=range(n))

    # Numerical features
    X["age"] = rng.integers(30, 80, n)
    X["trestbps"] = rng.integers(90, 190, n)
    X["chol"] = rng.integers(150, 350, n)
    X["thalach"] = rng.integers(80, 200, n)
    X["oldpeak"] = rng.uniform(0, 4, n)

    # Categorical/discrete features
    X["sex"] = rng.integers(0, 2, n)
    X["cp"] = rng.integers(1, 5, n)
    X["fbs"] = rng.integers(0, 2, n)
    X["restecg"] = rng.integers(0, 3, n)
    X["exang"] = rng.integers(0, 2, n)
    X["slope"] = rng.integers(1, 4, n)
    X["ca"] = rng.integers(0, 4, n)
    X["thal"] = rng.integers(3, 8, n)

    # Include missing categorical values to test imputation
    X.loc[0, "ca"] = np.nan
    X.loc[1, "thal"] = np.nan

    # Balanced synthetic binary target
    y = pd.Series([0, 1] * (n // 2), name="target")

    return X[FEATURES], y


@pytest.mark.parametrize(
    "model_name",
    ["Logistic Regression", "Random Forest"],
)
def test_model_pipeline_predictions(model_name, sample_data):
    X, y = sample_data
    validate_dataset(X, y)

    model = build_model_pipeline(model_name)
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(y)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(y), 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model_pipeline("Unknown Model")
