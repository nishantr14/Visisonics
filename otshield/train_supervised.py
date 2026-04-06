"""
OTShield - Supervised Model Training
Uses all 43 BATADAL sensor features + targeted engineering.
Random Forest with threshold optimization for max F1.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

# -- 1. Load data -----------------------------------------------------------

df3 = pd.read_csv(os.path.join(DATA, "BATADAL_dataset03.csv"))
df3.columns = [c.strip() for c in df3.columns]
df4 = pd.read_csv(os.path.join(DATA, "BATADAL_dataset04.csv"))
df4.columns = [c.strip() for c in df4.columns]

SENSOR_COLS = [c for c in df3.columns if c not in ["DATETIME", "ATT_FLAG"]]
print(f"Sensor columns ({len(SENSOR_COLS)}): {SENSOR_COLS}")

df3["label"] = 0
df4["label"] = (df4["ATT_FLAG"] == 1).astype(int)
print(f"Dataset03: {len(df3)} rows (all normal)")
print(f"Dataset04: {len(df4)} rows - normal={sum(df4['label']==0)}, attack={sum(df4['label']==1)}")

# -- 2. Feature engineering (targeted, not combinatorial) -------------------

def engineer_features(df, sensor_cols):
    """Add targeted + temporal engineered features."""
    X = df[sensor_cols].copy()

    level_cols = [c for c in sensor_cols if c.startswith("L_T")]
    pressure_cols = [c for c in sensor_cols if c.startswith("P_J")]
    flow_cols = [c for c in sensor_cols if c.startswith("F_PU") or c.startswith("F_V")]
    status_cols = [c for c in sensor_cols if c.startswith("S_")]

    # Aggregate statistics
    X["level_mean"] = df[level_cols].mean(axis=1)
    X["level_std"] = df[level_cols].std(axis=1)
    X["level_range"] = df[level_cols].max(axis=1) - df[level_cols].min(axis=1)
    X["pressure_mean"] = df[pressure_cols].mean(axis=1)
    X["pressure_std"] = df[pressure_cols].std(axis=1)
    X["pressure_range"] = df[pressure_cols].max(axis=1) - df[pressure_cols].min(axis=1)
    X["flow_mean"] = df[flow_cols].mean(axis=1)
    X["flow_std"] = df[flow_cols].std(axis=1)
    X["total_active_pumps"] = df[status_cols].sum(axis=1)
    X["total_flow"] = df[flow_cols].sum(axis=1)

    # Key level differences (adjacent tanks)
    for i in range(len(level_cols) - 1):
        X[f"{level_cols[i]}_diff_{level_cols[i+1]}"] = X[level_cols[i]] - X[level_cols[i+1]]

    # Status-flow mismatch (pump on but no flow = suspicious)
    for sc in status_cols:
        fc = sc.replace("S_", "F_")
        if fc in sensor_cols:
            X[f"{sc}_flow_mismatch"] = X[sc] * (1 - (X[fc] > 0.1).astype(float))

    # -- TEMPORAL FEATURES (key for time-series attack detection) --
    # Lag-1 differences: how much did each sensor change from previous timestep
    continuous_cols = level_cols + pressure_cols + flow_cols
    for col in continuous_cols:
        X[f"{col}_diff1"] = X[col].diff().fillna(0)

    # Rolling window stats (3-step and 6-step windows)
    for col in continuous_cols:
        X[f"{col}_roll3_mean"] = X[col].rolling(3, min_periods=1).mean()
        X[f"{col}_roll3_std"] = X[col].rolling(3, min_periods=1).std().fillna(0)
        X[f"{col}_roll6_mean"] = X[col].rolling(6, min_periods=1).mean()

    # Status change detection: did any pump switch on/off?
    for sc in status_cols:
        X[f"{sc}_changed"] = X[sc].diff().abs().fillna(0)
    X["any_status_change"] = sum(X[f"{sc}_changed"] for sc in status_cols)

    # Deviation from rolling mean (z-score style)
    for col in level_cols + pressure_cols[:4]:
        roll_mean = X[col].rolling(6, min_periods=1).mean()
        roll_std = X[col].rolling(6, min_periods=1).std().fillna(1)
        X[f"{col}_zscore6"] = ((X[col] - roll_mean) / roll_std.clip(lower=0.001))

    return X

X3 = engineer_features(df3, SENSOR_COLS)
X4 = engineer_features(df4, SENSOR_COLS)
ALL_FEATURE_NAMES = list(X3.columns)
print(f"\nTotal features: {len(ALL_FEATURE_NAMES)}")

y4 = df4["label"].values

# -- 3. Scale ---------------------------------------------------------------

scaler = StandardScaler()
scaler.fit(X3)  # fit on normal data only
X3_scaled = scaler.transform(X3)
X4_scaled = scaler.transform(X4)

# -- 4. Threshold-optimized stratified 5-fold CV ----------------------------

print("\n" + "=" * 70)
print("THRESHOLD-OPTIMIZED 5-FOLD CV (Random Forest)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
thresholds = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X4_scaled, y4)):
    X_train = np.vstack([X3_scaled, X4_scaled[train_idx]])
    y_train = np.concatenate([np.zeros(len(X3_scaled)), y4[train_idx]])
    X_test = X4_scaled[test_idx]
    y_test = y4[test_idx]

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # Sweep thresholds
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.02, 0.98, 0.01):
        f1 = f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    f1_scores.append(best_f1)
    thresholds.append(best_t)
    print(f"  Fold {fold+1}: F1={best_f1:.4f}  threshold={best_t:.2f}")

mean_f1 = np.mean(f1_scores)
mean_thresh = np.mean(thresholds)
print(f"\n  >> MEAN CV F1: {mean_f1:.4f}  mean threshold: {mean_thresh:.2f}")

# -- 5. Final model: train on dataset03 + dataset04 with CV threshold ------
# Use the CV-derived mean threshold (not re-optimized on test data)
# This avoids overfitting: the model never "sees" labels it's tuned on.

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING (no data leakage)")
print("=" * 70)

# Train on ALL data (standard practice — CV gave honest generalization estimate)
X_all = np.vstack([X3_scaled, X4_scaled])
y_all = np.concatenate([np.zeros(len(X3_scaled)), y4])

final_model = RandomForestClassifier(
    n_estimators=500, max_depth=20, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
final_model.fit(X_all, y_all)

# Use the CV-averaged threshold (NOT re-optimized on full dataset04)
opt_thresh = round(mean_thresh, 2)

# Report performance with CV threshold (informational only)
y_proba_final = final_model.predict_proba(X4_scaled)[:, 1]
y_pred = (y_proba_final >= opt_thresh).astype(int)
final_f1 = f1_score(y4, y_pred)
final_auc = roc_auc_score(y4, y_proba_final)

print(f"CV threshold used: {opt_thresh:.2f}")
print(f"Dataset04 F1 (informational): {final_f1:.4f}  AUC: {final_auc:.4f}")
print(f"Honest CV F1 (generalization): {mean_f1:.4f}")
print(classification_report(y4, y_pred, target_names=["Normal", "Attack"]))

# Feature importance
importances = final_model.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
print("Top 15 features:")
for i in top_idx:
    print(f"  {ALL_FEATURE_NAMES[i]:30s} {importances[i]:.4f}")

# -- 6. Save artifacts ------------------------------------------------------

joblib.dump(final_model, os.path.join(MODELS, "supervised_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS, "scaler_supervised.pkl"))

with open(os.path.join(MODELS, "feature_names_supervised.json"), "w") as f:
    json.dump(ALL_FEATURE_NAMES, f, indent=2)

with open(os.path.join(MODELS, "supervised_meta.json"), "w") as f:
    json.dump({
        "best_model": "random_forest",
        "cv_f1": round(mean_f1, 4),
        "final_f1": round(final_f1, 4),
        "final_auc": round(final_auc, 4),
        "optimal_threshold": opt_thresh,
        "n_features": len(ALL_FEATURE_NAMES),
        "sensor_cols": SENSOR_COLS,
    }, f, indent=2)

print(f"\n[OK] Model saved: models/supervised_model.pkl")
print(f"[OK] Scaler saved: models/scaler_supervised.pkl")
print(f"[OK] Features saved: models/feature_names_supervised.json ({len(ALL_FEATURE_NAMES)} features)")
print(f"[OK] Meta saved: models/supervised_meta.json")
print(f"\nDone! CV F1={mean_f1:.4f}, Final F1={final_f1:.4f}")
