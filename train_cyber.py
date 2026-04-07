from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def find_best_threshold(probs, y_true):
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 50):
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def main():
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = [str(c).strip() for c in df.columns]

    label_col = "label"
    y = (df[label_col] > 0).astype(int).to_numpy()

    print("\nLABEL DISTRIBUTION:\n", df[label_col].value_counts())

    # ---------- FEATURE SELECTION ----------
    X = df.drop(columns=[label_col]).copy()

    SAFE_FEATURES = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "src_pkts",
        "dst_pkts",
        "missed_bytes",
    ]

    available = [c for c in SAFE_FEATURES if c in X.columns]

    if len(available) == 0:
        raise ValueError("SAFE_FEATURES not found in dataset")

    X = X[available]

    # numeric safety
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    feature_names = X.columns.tolist()

    # ---------- SPLIT ----------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # ---------- IMPUTATION ----------
    med = X_train.median()
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    # ---------- SCALING ----------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ---------- MODEL ----------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ---------- CALIBRATION ----------
    val_probs_raw = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, y_val)

    val_probs = calibrator.transform(val_probs_raw)

    # ---------- THRESHOLD TUNING ----------
    best_threshold = find_best_threshold(val_probs, y_val)
    print("\nBEST THRESHOLD:", best_threshold)

    # ---------- TEST ----------
    test_probs = calibrator.transform(model.predict_proba(X_test)[:, 1])
    pred = (test_probs >= best_threshold).astype(int)

    print("\nPRED DISTRIBUTION:", np.bincount(pred))

    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    brier = brier_score_loss(y_test, test_probs)

    # ---------- SAVE ----------
    joblib.dump(model, MODELS_DIR / "random_forest_cyber.pkl")
    joblib.dump(calibrator, MODELS_DIR / "probability_calibrator_cyber.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler_cyber.pkl")

    with open(MODELS_DIR / "feature_names_cyber.json", "w") as f:
        json.dump(feature_names, f)

    print("\nFINAL RESULTS")
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Brier:", brier)


if __name__ == "__main__":
    main()