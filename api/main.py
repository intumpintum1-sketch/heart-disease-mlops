
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import HeartDiseaseInput, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_PATH}. "
        "Complete Phase 3 and save the fitted pipeline first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational MLOps project demonstrating "
        "model inference through a REST API."
    ),
    version="1.0.0",
)

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)


@app.get("/health")
def health_check():
    """Report whether the API process and model are available."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HeartDiseaseInput):
    """Return a binary model prediction and estimated probability."""
    try:
        features = pd.DataFrame(
            [input_data.model_dump()],
            columns=[
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ],
        )

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        probability_disease = float(probabilities[1])

        return PredictionResponse(
            prediction=prediction,
            prediction_label=(
                "Disease" if prediction == 1 else "No Disease"
            ),
            probability_disease=probability_disease,
            model="Logistic Regression",
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process input: {exc}",
        ) from exc
