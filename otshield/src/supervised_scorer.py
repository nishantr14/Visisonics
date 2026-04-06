"""
OTShield Unified Supervised Scorer
Uses the ensemble RF+GBM trained on all 43 BATADAL features + static engineering.
No temporal features = no history buffer needed = no leakage.
"""

import os, sys, json
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel):
    return os.path.join(BASE, rel)

# Add parent to path so we can import engineer_features
sys.path.insert(0, BASE)
from train_supervised import engineer_features

# Load artifacts
with open(_path("models/feature_names_supervised.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(_path("models/supervised_meta.json")) as f:
    META = json.load(f)

SENSOR_COLS = META["sensor_cols"]
OPTIMAL_THRESHOLD = META["optimal_threshold"]

model = joblib.load(_path("models/supervised_model.pkl"))
scaler = joblib.load(_path("models/scaler_supervised.pkl"))


def get_supervised_score(sensor_dict):
    """
    Input:  sensor_dict with keys matching BATADAL sensor columns
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    df = pd.DataFrame([{col: sensor_dict.get(col, 0) for col in SENSOR_COLS}])
    X_eng = engineer_features(df, SENSOR_COLS)
    X = scaler.transform(X_eng[FEATURE_NAMES])

    proba = model.predict_proba(X)[0, 1]
    score = round(proba * 100, 1)

    # Top feature by deviation from scaler mean
    raw_vals = X_eng[FEATURE_NAMES].values[0]
    deviations = np.abs((raw_vals - scaler.mean_) / np.clip(scaler.scale_, 1e-8, None))
    sensor_idx = [i for i, name in enumerate(FEATURE_NAMES) if name in SENSOR_COLS]
    if sensor_idx:
        best_idx = sensor_idx[int(np.argmax(deviations[sensor_idx]))]
    else:
        best_idx = int(np.argmax(deviations))
    top_feature = FEATURE_NAMES[best_idx]

    if score > 70:
        explanation = "High anomaly detected in sensor pattern"
    elif score > 40:
        explanation = "Moderate anomaly detected"
    else:
        explanation = "Normal behavior"

    return score, explanation, top_feature


def reset_history():
    """No-op. Kept for backward compatibility."""
    pass
