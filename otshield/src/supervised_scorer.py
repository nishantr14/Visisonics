"""
OTShield Unified Supervised Scorer
Shared by cyber_layer.py and physical_layer.py.
Uses the supervised Random Forest trained on all 43 BATADAL features + engineering.
Maintains a history buffer for temporal features (rolling stats, diffs).
"""

import os, json, collections
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel):
    return os.path.join(BASE, rel)

# Load artifacts
with open(_path("models/feature_names_supervised.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(_path("models/supervised_meta.json")) as f:
    META = json.load(f)

SENSOR_COLS = META["sensor_cols"]
OPTIMAL_THRESHOLD = META["optimal_threshold"]

model = joblib.load(_path("models/supervised_model.pkl"))
scaler = joblib.load(_path("models/scaler_supervised.pkl"))

# History buffer for temporal features (last 6 readings)
HISTORY_SIZE = 6
_history = collections.deque(maxlen=HISTORY_SIZE)


def _engineer_from_history(current_dict):
    """Build feature vector including temporal features from history buffer."""
    # Add current reading to history
    _history.append({col: current_dict.get(col, 0) for col in SENSOR_COLS})

    # Build a small DataFrame from history for rolling calculations
    hist_df = pd.DataFrame(list(_history))

    # We only need the LAST row's features, but rolling needs the full history
    full_df = hist_df.copy()

    level_cols = [c for c in SENSOR_COLS if c.startswith("L_T")]
    pressure_cols = [c for c in SENSOR_COLS if c.startswith("P_J")]
    flow_cols = [c for c in SENSOR_COLS if c.startswith("F_PU") or c.startswith("F_V")]
    status_cols = [c for c in SENSOR_COLS if c.startswith("S_")]
    continuous_cols = level_cols + pressure_cols + flow_cols

    X = full_df[SENSOR_COLS].copy()

    # Aggregate statistics
    X["level_mean"] = full_df[level_cols].mean(axis=1)
    X["level_std"] = full_df[level_cols].std(axis=1)
    X["level_range"] = full_df[level_cols].max(axis=1) - full_df[level_cols].min(axis=1)
    X["pressure_mean"] = full_df[pressure_cols].mean(axis=1)
    X["pressure_std"] = full_df[pressure_cols].std(axis=1)
    X["pressure_range"] = full_df[pressure_cols].max(axis=1) - full_df[pressure_cols].min(axis=1)
    X["flow_mean"] = full_df[flow_cols].mean(axis=1)
    X["flow_std"] = full_df[flow_cols].std(axis=1)
    X["total_active_pumps"] = full_df[status_cols].sum(axis=1)
    X["total_flow"] = full_df[flow_cols].sum(axis=1)

    # Level differences
    for i in range(len(level_cols) - 1):
        X[f"{level_cols[i]}_diff_{level_cols[i+1]}"] = X[level_cols[i]] - X[level_cols[i+1]]

    # Status-flow mismatch
    for sc in status_cols:
        fc = sc.replace("S_", "F_")
        if fc in SENSOR_COLS:
            X[f"{sc}_flow_mismatch"] = X[sc] * (1 - (X[fc] > 0.1).astype(float))

    # Temporal features
    for col in continuous_cols:
        X[f"{col}_diff1"] = X[col].diff().fillna(0)

    for col in continuous_cols:
        X[f"{col}_roll3_mean"] = X[col].rolling(3, min_periods=1).mean()
        X[f"{col}_roll3_std"] = X[col].rolling(3, min_periods=1).std().fillna(0)
        X[f"{col}_roll6_mean"] = X[col].rolling(6, min_periods=1).mean()

    for sc in status_cols:
        X[f"{sc}_changed"] = X[sc].diff().abs().fillna(0)
    X["any_status_change"] = sum(X[f"{sc}_changed"] for sc in status_cols)

    for col in level_cols + pressure_cols[:4]:
        roll_mean = X[col].rolling(6, min_periods=1).mean()
        roll_std = X[col].rolling(6, min_periods=1).std().fillna(1)
        X[f"{col}_zscore6"] = ((X[col] - roll_mean) / roll_std.clip(lower=0.001))

    # Return only the last row, with features in the correct order
    last_row = X.iloc[[-1]][FEATURE_NAMES]
    return last_row


def get_supervised_score(sensor_dict):
    """
    Input:  sensor_dict with keys matching BATADAL sensor columns
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    df = _engineer_from_history(sensor_dict)
    X = scaler.transform(df)

    # Probability of being an attack
    proba = model.predict_proba(X)[0, 1]
    score = round(proba * 100, 1)

    # Top feature by deviation from scaler mean
    raw_vals = df.values[0]
    means = scaler.mean_
    stds = scaler.scale_
    deviations = np.abs((raw_vals - means) / np.clip(stds, 1e-8, None))
    # Only consider original sensor columns for interpretability
    sensor_feature_indices = [i for i, name in enumerate(FEATURE_NAMES) if name in SENSOR_COLS]
    if sensor_feature_indices:
        best_idx = sensor_feature_indices[int(np.argmax(deviations[sensor_feature_indices]))]
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
    """Clear the history buffer (e.g., on restart)."""
    _history.clear()
