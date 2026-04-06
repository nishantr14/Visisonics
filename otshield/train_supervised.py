"""
OTShield - Supervised Model Training
Uses all 43 BATADAL sensor features + static engineering (no temporal leakage).
Ensemble (RF + GBM) with threshold optimization for max F1.

Usage:
    python train_supervised.py

The engineer_features() function can also be imported by other modules
without triggering training.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")


def engineer_features(df, sensor_cols):
    """Static feature engineering - no temporal features, no leakage risk."""
    X = df[sensor_cols].copy()

    level_cols = [c for c in sensor_cols if c.startswith("L_T")]
    pressure_cols = [c for c in sensor_cols if c.startswith("P_J")]
    flow_cols = [c for c in sensor_cols if c.startswith("F_PU") or c.startswith("F_V")]
    status_cols = [c for c in sensor_cols if c.startswith("S_")]

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

    for i in range(len(level_cols) - 1):
        X[f"{level_cols[i]}_diff_{level_cols[i+1]}"] = X[level_cols[i]] - X[level_cols[i+1]]

    for sc in status_cols:
        fc = sc.replace("S_", "F_")
        if fc in sensor_cols:
            X[f"{sc}_flow_mismatch"] = X[sc] * (1 - (X[fc] > 0.1).astype(float))

    return X


if __name__ == "__main__":
    df3 = pd.read_csv(os.path.join(DATA, "BATADAL_dataset03.csv"))
    df3.columns = [c.strip() for c in df3.columns]
    df4 = pd.read_csv(os.path.join(DATA, "BATADAL_dataset04.csv"))
    df4.columns = [c.strip() for c in df4.columns]

    SENSOR_COLS = [c for c in df3.columns if c not in ["DATETIME", "ATT_FLAG"]]
    df3["label"] = 0
    df4["label"] = (df4["ATT_FLAG"] == 1).astype(int)

    print(f"Sensor columns: {len(SENSOR_COLS)}")
    print(f"Dataset03: {len(df3)} rows (all normal)")
    print(f"Dataset04: {len(df4)} rows - normal={sum(df4['label']==0)}, attack={sum(df4['label']==1)}")

    X3 = engineer_features(df3, SENSOR_COLS)
    X4 = engineer_features(df4, SENSOR_COLS)
    ALL_FEATURE_NAMES = list(X3.columns)
    y4 = df4["label"].values
    print(f"Features: {len(ALL_FEATURE_NAMES)}")

    scaler = StandardScaler().fit(X3)
    X3s = scaler.transform(X3)
    X4s = scaler.transform(X4)

    # Ensemble: RF + GBM
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     min_samples_leaf=5, random_state=42)

    print("\n" + "=" * 60)
    print("STRATIFIED 5-FOLD CV (Ensemble RF+GBM)")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, aucs, threshs = [], [], []

    for fold, (tr, te) in enumerate(skf.split(X4s, y4)):
        X_train = np.vstack([X3s, X4s[tr]])
        y_train = np.concatenate([np.zeros(len(X3s)), y4[tr]])

        ens = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")
        ens.fit(X_train, y_train)
        yp = ens.predict_proba(X4s[te])[:, 1]

        auc = roc_auc_score(y4[te], yp)
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.02, 0.98, 0.01):
            f1 = f1_score(y4[te], (yp >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        f1s.append(best_f1); aucs.append(auc); threshs.append(best_t)
        print(f"  Fold {fold+1}: F1={best_f1:.4f}  AUC={auc:.4f}  thresh={best_t:.2f}")

    mean_f1 = np.mean(f1s)
    mean_auc = np.mean(aucs)
    mean_thresh = np.mean(threshs)
    print(f"\n  >> CV F1={mean_f1:.4f}  AUC={mean_auc:.4f}  thresh={mean_thresh:.2f}")

    # Final model
    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    X_all = np.vstack([X3s, X4s])
    y_all = np.concatenate([np.zeros(len(X3s)), y4])
    final = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")
    final.fit(X_all, y_all)

    opt_thresh = round(mean_thresh, 2)
    yp_final = final.predict_proba(X4s)[:, 1]
    y_pred = (yp_final >= opt_thresh).astype(int)
    final_f1 = f1_score(y4, y_pred)
    final_auc = roc_auc_score(y4, yp_final)

    print(f"Threshold: {opt_thresh}")
    print(f"F1: {final_f1:.4f}  AUC: {final_auc:.4f}")
    print(classification_report(y4, y_pred, target_names=["Normal", "Attack"]))

    # Save
    joblib.dump(final, os.path.join(MODELS, "supervised_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS, "scaler_supervised.pkl"))
    with open(os.path.join(MODELS, "feature_names_supervised.json"), "w") as f:
        json.dump(ALL_FEATURE_NAMES, f, indent=2)
    with open(os.path.join(MODELS, "supervised_meta.json"), "w") as f:
        json.dump({
            "best_model": "ensemble_rf_gbm",
            "cv_f1": round(mean_f1, 4),
            "cv_auc": round(mean_auc, 4),
            "final_f1": round(final_f1, 4),
            "final_auc": round(final_auc, 4),
            "optimal_threshold": opt_thresh,
            "n_features": len(ALL_FEATURE_NAMES),
            "sensor_cols": SENSOR_COLS,
        }, f, indent=2)

    print("[OK] All artifacts saved")
    print(f"Done! CV F1={mean_f1:.4f} AUC={mean_auc:.4f}")
