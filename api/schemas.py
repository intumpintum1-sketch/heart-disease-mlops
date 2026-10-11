

from pydantic import BaseModel, ConfigDict, Field


class HeartDiseaseInput(BaseModel):
    """Validated input features for the prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(gt=0, le=100)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    thalach: float = Field(gt=0)
    oldpeak: float = Field(ge=0)

    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    exang: int = Field(ge=0, le=1)
    slope: int = Field(ge=1, le=3)

    # Optional because these features had missing values in the dataset.
    ca: float | None = Field(default=None, ge=0, le=3)
    thal: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    prediction: int = Field(ge=0, le=1)
    prediction_label: str
    probability_disease: float = Field(ge=0, le=1)
    model: str
