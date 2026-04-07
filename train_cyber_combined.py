import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


# -------------------------------
# FEATURE STANDARDIZATION
# -------------------------------
def standardize_columns(df):
    df = df.copy()

    mapping = {
        "dur": "duration",
        "sbytes": "src_bytes",
        "dbytes": "dst_bytes",
        "spkts": "src_pkts",
        "dpkts": "dst_pkts",
    }

    df = df.rename(columns=mapping)
    return df


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    ton = pd.read_csv(ROOT / "data/train_test_network.csv")
    unsw_train = pd.read_csv(ROOT / "data/UNSW_NB15_training-set.csv")
    unsw_test = pd.read_csv(ROOT / "data/UNSW_NB15_testing-set.csv")

    unsw = pd.concat([unsw_train, unsw_test], ignore_index=True)

    return ton, unsw


# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(ton, unsw):
    ton = standardize_columns(ton)
    unsw = standardize_columns(unsw)

    # unify label
    ton["label"] = (ton["label"] > 0).astype(int)
    unsw["label"] = (unsw["label"] > 0).astype(int)

    SAFE_FEATURES = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "src_pkts",
        "dst_pkts",
    ]

    ton = ton[[col for col in SAFE_FEATURES if col in ton.columns] + ["label"]]
    unsw = unsw[[col for col in SAFE_FEATURES if col in unsw.columns] + ["label"]]

    df = pd.concat([ton, unsw], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df, SAFE_FEATURES


# -------------------------------
# TRAIN
# -------------------------------
def train(df, feature_names):
    X = df[feature_names]
    y = df["label"]

    # numeric safety
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    med = X_train.median()
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # calibration
    val_probs_raw = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, y_val)

    val_probs = calibrator.transform(val_probs_raw)

    # threshold tuning
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 50):
        pred = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("\nBEST THRESHOLD:", best_t)

    # test
    test_probs = calibrator.transform(model.predict_proba(X_test)[:, 1])
    pred = (test_probs >= best_t).astype(int)

    print("\nPRED DISTRIBUTION:", np.bincount(pred))

    print("\nFINAL RESULTS")
    print("F1:", f1_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))

    # save
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODELS_DIR / "rf_combined.pkl")
    joblib.dump(calibrator, MODELS_DIR / "calibrator_combined.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler_combined.pkl")


# -------------------------------
# MAIN
# -------------------------------
def main():
    ton, unsw = load_data()
    df, features = preprocess(ton, unsw)

    print("\nDATASET SIZE:", len(df))
    print("ATTACK RATIO:", df["label"].mean())

    train(df, features)


if __name__ == "__main__":
    main()