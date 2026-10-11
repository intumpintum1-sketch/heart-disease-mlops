
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_dataset(X, y):
    """Validate the feature schema and binary target."""
    if X.empty:
        raise ValueError("Feature dataset is empty.")

    missing_columns = set(FEATURES) - set(X.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required features: {sorted(missing_columns)}"
        )

    if len(X) != len(y):
        raise ValueError("Features and target have different lengths.")

    if y.isnull().any():
        raise ValueError("Target contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")


def build_preprocessor():
    """Build leakage-safe numerical and categorical preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_model_pipeline(model_name, random_state=42):
    """Return a complete preprocessing and classification pipeline."""
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
