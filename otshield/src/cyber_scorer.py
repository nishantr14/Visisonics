"""
OTShield Cyber Scorer
Random Forest trained on TON_IoT network flow data.
Completely independent from the physical layer — separate model, features, scaler.
"""

import os, sys, json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(rel):
    return os.path.join(BASE, rel)


# Add parent to path so we can import preprocess_cyber
sys.path.insert(0, BASE)
from train_cyber import preprocess_cyber, NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS

# Load artifacts
with open(_path("models/feature_names_cyber.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(_path("models/cyber_meta.json")) as f:
    META = json.load(f)

with open(_path("models/cyber_label_encoders.json")) as f:
    le_raw = json.load(f)

# Reconstruct LabelEncoders from saved classes
LABEL_ENCODERS = {}
for col, classes in le_raw.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    LABEL_ENCODERS[col] = le

ATTACK_TYPES = META.get("attack_types", [])

model = joblib.load(_path("models/cyber_model.pkl"))
scaler = joblib.load(_path("models/scaler_cyber.pkl"))


def get_cyber_score(network_dict):
    """
    Input:  network_dict with TON_IoT network flow keys
            (duration, src_bytes, dst_bytes, proto, etc.)
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    df = pd.DataFrame([network_dict])
    X = preprocess_cyber(df, label_encoders=LABEL_ENCODERS, fit=False)

    # Ensure column order matches training
    for col in FEATURE_NAMES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_NAMES]

    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0, 1]
    score = round(proba * 100, 1)

    # Top feature by deviation from scaler mean
    raw_vals = X.values[0]
    deviations = np.abs((raw_vals - scaler.mean_) / np.clip(scaler.scale_, 1e-8, None))
    best_idx = int(np.argmax(deviations))
    top_feature = FEATURE_NAMES[best_idx]

    if score > 70:
        explanation = "High network anomaly detected"
    elif score > 40:
        explanation = "Moderate network anomaly"
    else:
        explanation = "Network traffic normal"

    return score, explanation, top_feature
