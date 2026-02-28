import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from model.train import build_and_train_pipeline

MODEL_PATH = Path("model/pipeline.pkl")


def load_pipeline():
    """Load trained pipeline from disk, training if not yet saved."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    # Auto-train on first run
    pipeline = build_and_train_pipeline()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def predict_survival(pipeline, patient_data: dict) -> dict:
    """
    Run survival prediction for a single patient.

    Args:
        pipeline: Trained sklearn pipeline with predict_proba support.
        patient_data: Dict of feature_name -> value.

    Returns:
        Dict with keys: prediction, probability, confidence, risk_level, feature_vector
    """
    feature_vector = pd.DataFrame([patient_data])
    proba = pipeline.predict_proba(feature_vector)[0]
    pred = pipeline.predict(feature_vector)[0]

    survival_prob = proba[0] * 100  # class 0 = malignant => lower survival
    # Remap: class 1 = benign = higher survival
    survival_prob = proba[1] * 100

    risk_level = (
        "Low Risk" if survival_prob >= 75
        else "Moderate Risk" if survival_prob >= 50
        else "High Risk"
    )

    return {
        "prediction": int(pred),
        "survival_probability": round(survival_prob, 2),
        "risk_level": risk_level,
        "probabilities": proba,
        "feature_vector": feature_vector,
    }
