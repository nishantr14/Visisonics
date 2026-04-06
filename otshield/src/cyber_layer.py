"""OTShield Cyber Layer — anomaly detection inference."""

import os, json
import joblib
import numpy as np
import pandas as pd
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel):
    return os.path.join(BASE, rel)

# Load artifacts
with open(_path("models/feature_names_cyber.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(_path("models/if_score_bounds.json")) as f:
    IF_BOUNDS = json.load(f)

with open(_path("models/autoencoder_threshold.json")) as f:
    AE_THRESHOLD = json.load(f)["threshold"]

model_if    = joblib.load(_path("models/isolation_forest.pkl"))
model_ocsvm = joblib.load(_path("models/ocsvm_cyber.pkl"))
scaler      = joblib.load(_path("models/scaler_cyber.pkl"))

# Load autoencoder
from tensorflow import keras
model_ae = keras.models.load_model(_path("models/autoencoder_cyber.h5"))

# Best model selected during training
BEST_MODEL = "isolation_forest"


def get_cyber_score(sensor_dict):
    """
    Input:  sensor_dict — keys must match feature_names_cyber.json
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    df = pd.DataFrame([sensor_dict])[FEATURE_NAMES]
    X  = scaler.transform(df)

    if BEST_MODEL == "isolation_forest":
        raw   = model_if.decision_function(X)[0]
        score = (raw - IF_BOUNDS["min"]) / (IF_BOUNDS["max"] - IF_BOUNDS["min"])
        score = 1 - score
    elif BEST_MODEL == "ocsvm":
        pred  = model_ocsvm.decision_function(X)[0]
        score = 1 / (1 + np.exp(pred))
    else:
        recon = model_ae.predict(X, verbose=0)
        recon_error = float(np.mean((X - recon) ** 2))
        score = recon_error / (AE_THRESHOLD * 2)

    score = max(0.0, min(1.0, score))
    score = round(score * 100, 1)

    explainer   = shap.TreeExplainer(model_if)
    shap_vals   = explainer.shap_values(X)
    top_feature = FEATURE_NAMES[int(np.abs(shap_vals[0]).argmax())]

    if score > 70:
        explanation = "High anomaly detected in sensor pattern"
    elif score > 40:
        explanation = "Moderate anomaly detected"
    else:
        explanation = "Normal behavior"

    return score, explanation, top_feature
