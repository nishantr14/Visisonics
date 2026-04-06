"""OTShield Physical Layer — sensor telemetry anomaly detection inference."""

import os, json
import joblib
import numpy as np
import pandas as pd
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel):
    return os.path.join(BASE, rel)

with open(_path("models/feature_names_physical.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(_path("models/optimized_thresholds_physical.json")) as f:
    OPT_THRESHOLDS = json.load(f)

with open(_path("models/ocsvm_score_bounds_physical.json")) as f:
    OCSVM_BOUNDS = json.load(f)

with open(_path("models/if_score_bounds_physical.json")) as f:
    IF_BOUNDS = json.load(f)

model_ocsvm = joblib.load(_path("models/ocsvm_physical.pkl"))
model_if    = joblib.load(_path("models/isolation_forest_physical.pkl"))
scaler      = joblib.load(_path("models/scaler_physical.pkl"))

BEST_MODEL = "isolation_forest"


def get_physical_score(sensor_dict):
    """
    Input:  sensor_dict — keys must match feature_names_physical.json
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    df = pd.DataFrame([sensor_dict])[FEATURE_NAMES]
    X  = scaler.transform(df)

    if BEST_MODEL == "ocsvm":
        raw    = model_ocsvm.decision_function(X)[0]
        thresh = OPT_THRESHOLDS["ocsvm"]
        score  = 0.5 + (thresh - raw)
    else:
        raw    = model_if.decision_function(X)[0]
        thresh = OPT_THRESHOLDS["isolation_forest"]
        score  = 0.5 + (thresh - raw)

    score = max(0.0, min(1.0, score))
    score = round(score * 100, 1)

    try:
        explainer  = shap.KernelExplainer(model_ocsvm.decision_function, X)
        shap_vals  = explainer.shap_values(X)
        top_feature = FEATURE_NAMES[int(np.abs(shap_vals[0]).argmax())]
    except Exception:
        top_feature = FEATURE_NAMES[0]

    if score > 70:
        explanation = "Physical anomaly detected in sensor telemetry"
    elif score > 40:
        explanation = "Moderate physical deviation detected"
    else:
        explanation = "Physical sensors nominal"

    return score, explanation, top_feature
