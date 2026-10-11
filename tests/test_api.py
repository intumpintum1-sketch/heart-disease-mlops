
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def valid_payload():
    return {
        "age": 58,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["prediction_label"] in ["Disease", "No Disease"]
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["model"] == "Logistic Regression"


def test_prediction_with_missing_optional_values():
    payload = valid_payload()
    payload["ca"] = None
    payload["thal"] = None

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_invalid_input_is_rejected():
    payload = valid_payload()
    payload["age"] = 150

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_required_feature_is_rejected():
    payload = valid_payload()
    del payload["chol"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
