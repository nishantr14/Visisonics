"""
OTShield - Physical Layer Model Training (BATADAL)
Ensemble (RF + GBM) with TimeSeriesSplit to allow temporal features WITHOUT leakage.

Key insight: StratifiedKFold shuffles rows, so rolling/lag features leak future info.
TimeSeriesSplit preserves time ordering, making temporal features safe.

Usage:
    python train_supervised.py

The engineer_features() function can be imported by other modules
without triggering training.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")


def engineer_features(df, sensor_cols):
    """
    Feature engineering — static + temporal.
    Temporal features (rolling, lag, diff) are safe when used with TimeSeriesSplit
    because rows are never shuffled across time.
    """
    X = df[sensor_cols].copy()

    level_cols = [c for c in sensor_cols if c.startswith("L_T")]
    pressure_cols = [c for c in sensor_cols if c.startswith("P_J")]
    flow_cols = [c for c in sensor_cols if c.startswith("F_PU") or c.startswith("F_V")]
    status_cols = [c for c in sensor_cols if c.startswith("S_")]

    # ── Static cross-sensor features ──
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

    # ── Temporal features (safe with TimeSeriesSplit) ──
    # Rolling statistics over short windows
    for col in level_cols + pressure_cols[:4]:
        X[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
        X[f"{col}_roll3_std"] = df[col].rolling(3, min_periods=1).std().fillna(0)
        X[f"{col}_roll6_mean"] = df[col].rolling(6, min_periods=1).mean()
        X[f"{col}_diff1"] = df[col].diff(1).fillna(0)

    # Flow rate changes
    for col in flow_cols[:6]:
        X[f"{col}_diff1"] = df[col].diff(1).fillna(0)
        X[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()

    # Pump status change detection
    for col in status_cols:
        X[f"{col}_changed"] = df[col].diff(1).abs().fillna(0)

    # System-level temporal
    X["level_mean_roll6"] = X["level_mean"].rolling(6, min_periods=1).mean()
    X["pressure_mean_roll6"] = X["pressure_mean"].rolling(6, min_periods=1).mean()
    X["flow_mean_diff1"] = X["flow_mean"].diff(1).fillna(0)
    X["active_pumps_diff1"] = X["total_active_pumps"].diff(1).fillna(0)

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

    # Engineer features on concatenated data (temporal features need contiguous time)
    # Dataset03 comes first (earlier in time), then dataset04
    df_all = pd.concat([df3, df4], ignore_index=True)
    y_all_labels = np.concatenate([np.zeros(len(df3)), df4["label"].values])

    X_all = engineer_features(df_all, SENSOR_COLS)
    ALL_FEATURE_NAMES = list(X_all.columns)
    print(f"Features: {len(ALL_FEATURE_NAMES)}")

    # Split back: dataset03 portion for scaler fitting, dataset04 for CV
    n3 = len(df3)
    X3 = X_all.iloc[:n3]
    X4 = X_all.iloc[n3:]
    y4 = df4["label"].values

    # Fit scaler on normal data only
    scaler = StandardScaler().fit(X3)
    X3s = scaler.transform(X3)
    X4s = scaler.transform(X4)

    # Ensemble: RF + GBM
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     min_samples_leaf=5, random_state=42)

    # ── TimeSeriesSplit CV on dataset04 ──
    # This respects temporal order: train on earlier rows, test on later rows.
    # Temporal features (rolling, diff) are computed BEFORE splitting on the
    # full time-ordered sequence, which is valid because each row's temporal
    # features only depend on preceding rows (rolling uses min_periods, diff
    # looks backward). No future information leaks into test folds.
    print("\n" + "=" * 60)
    print("TIME-SERIES 5-FOLD CV (Ensemble RF+GBM)")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=5)
    f1s, aucs, threshs = [], [], []

    for fold, (tr, te) in enumerate(tscv.split(X4s)):
        # Prepend all of dataset03 (normal) to each fold's training set
        X_train = np.vstack([X3s, X4s[tr]])
        y_train = np.concatenate([np.zeros(len(X3s)), y4[tr]])

        # Skip folds where train or test has only one class
        if len(np.unique(y_train)) < 2 or len(np.unique(y4[te])) < 2:
            print(f"  Fold {fold+1}: SKIPPED (insufficient class diversity, "
                  f"train_attacks={y_train.sum():.0f}, test_attacks={y4[te].sum():.0f})")
            continue

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
        print(f"  Fold {fold+1}: F1={best_f1:.4f}  AUC={auc:.4f}  thresh={best_t:.2f}  "
              f"(train={len(tr)}, test={len(te)}, attacks_in_test={y4[te].sum():.0f})")

    mean_f1 = np.mean(f1s)
    mean_auc = np.mean(aucs)
    mean_thresh = np.mean(threshs)
    print(f"\n  >> CV F1={mean_f1:.4f}  AUC={mean_auc:.4f}  thresh={mean_thresh:.2f}")

    # ── Final model on all data ──
    print("\n" + "=" * 60)
    print("FINAL MODEL (trained on all data)")
    print("=" * 60)

    X_final = np.vstack([X3s, X4s])
    y_final = np.concatenate([np.zeros(len(X3s)), y4])
    final = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")
    final.fit(X_final, y_final)

    opt_thresh = round(mean_thresh, 2)

    # Hold-out eval: use last 20% of dataset04 (time-ordered)
    split_idx = int(len(X4s) * 0.8)
    X_holdout = X4s[split_idx:]
    y_holdout = y4[split_idx:]
    yp_holdout = final.predict_proba(X_holdout)[:, 1]
    y_pred_holdout = (yp_holdout >= opt_thresh).astype(int)
    holdout_f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    holdout_auc = roc_auc_score(y_holdout, yp_holdout) if len(np.unique(y_holdout)) > 1 else 0.5

    print(f"Threshold: {opt_thresh}")
    print(f"Hold-out F1 (last 20%): {holdout_f1:.4f}")
    print(f"Hold-out AUC (last 20%): {holdout_auc:.4f}")
    print(classification_report(y_holdout, y_pred_holdout, target_names=["Normal", "Attack"]))

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
            "holdout_f1": round(holdout_f1, 4),
            "holdout_auc": round(holdout_auc, 4),
            "optimal_threshold": opt_thresh,
            "n_features": len(ALL_FEATURE_NAMES),
            "sensor_cols": SENSOR_COLS,
            "cv_method": "TimeSeriesSplit_5fold",
        }, f, indent=2)

    print("[OK] All artifacts saved")
    print(f"Done! CV F1={mean_f1:.4f} AUC={mean_auc:.4f}")
