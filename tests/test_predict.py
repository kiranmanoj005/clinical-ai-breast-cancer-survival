import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from model.train import build_and_train_pipeline
from app.predict import predict_survival


@pytest.fixture(scope="module")
def pipeline():
    return build_and_train_pipeline(save=False)


@pytest.fixture(scope="module")
def sample_patient():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    return X.iloc[0].to_dict()


def test_pipeline_accuracy(pipeline):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.93, f"Accuracy too low: {acc:.2f}"


def test_predict_survival_output_keys(pipeline, sample_patient):
    result = predict_survival(pipeline, sample_patient)
    assert "prediction" in result
    assert "survival_probability" in result
    assert "risk_level" in result
    assert "probabilities" in result


def test_survival_probability_range(pipeline, sample_patient):
    result = predict_survival(pipeline, sample_patient)
    assert 0 <= result["survival_probability"] <= 100


def test_risk_level_valid(pipeline, sample_patient):
    result = predict_survival(pipeline, sample_patient)
    assert result["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]


def test_prediction_binary(pipeline, sample_patient):
    result = predict_survival(pipeline, sample_patient)
    assert result["prediction"] in [0, 1]
