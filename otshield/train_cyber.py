"""
OTShield - Cyber Layer Model Training
Random Forest on TON_IoT Network dataset (211K flows, 9 attack types).
Separate pipeline from the physical layer — no shared features or model.

Usage:
    python train_cyber.py

The preprocess_cyber() function can be imported by other modules
without triggering training.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(BASE), "data", "ton",
                    "Train_Test_datasets", "Train_Test_Network_dataset")
MODELS = os.path.join(BASE, "models")

# ── Feature definitions ──────────────────────────────────────────────────────

NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "missed_bytes",
    "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes",
    "src_port", "dst_port",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]

CATEGORICAL_COLS = ["proto", "service", "conn_state"]

BINARY_COLS = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established",
]

# Columns to drop (leaky identifiers, free-text, not generalizable)
DROP_COLS = [
    "src_ip", "dst_ip", "dns_query", "http_uri", "http_user_agent",
    "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer",
    "http_orig_mime_types", "http_resp_mime_types",
    "http_trans_depth", "http_method", "http_version",
    "weird_name", "weird_addl", "weird_notice",
    "label", "type",
]


def preprocess_cyber(df, label_encoders=None, fit=False):
    """
    Preprocess TON_IoT network data into model-ready features.

    Args:
        df: Raw DataFrame from TON_IoT CSV
        label_encoders: dict of fitted LabelEncoders (None if fit=True)
        fit: If True, fit new LabelEncoders and return them

    Returns:
        X: DataFrame of processed features
        label_encoders: dict of LabelEncoders (only if fit=True)
    """
    X = pd.DataFrame()

    # Numeric columns — replace '-' with 0
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(df[col].replace("-", 0), errors="coerce").fillna(0)

    # Binary columns — map 'T' -> 1, else -> 0
    for col in BINARY_COLS:
        X[col] = (df[col].astype(str).str.strip().str.upper() == "T").astype(int)

    # Categorical columns — LabelEncode
    if fit:
        label_encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            vals = df[col].astype(str).replace("-", "none").fillna("none")
            le.fit(vals)
            X[col] = le.transform(vals)
            label_encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            le = label_encoders[col]
            vals = df[col].astype(str).replace("-", "none").fillna("none")
            # Handle unseen labels at inference
            known = set(le.classes_)
            vals = vals.apply(lambda v: v if v in known else "none")
            X[col] = le.transform(vals)

    # Engineered features
    X["bytes_ratio"] = X["src_bytes"] / (X["dst_bytes"] + 1)
    X["pkts_ratio"] = X["src_pkts"] / (X["dst_pkts"] + 1)
    X["bytes_per_pkt_src"] = X["src_bytes"] / (X["src_pkts"] + 1)
    X["bytes_per_pkt_dst"] = X["dst_bytes"] / (X["dst_pkts"] + 1)
    X["total_bytes"] = X["src_bytes"] + X["dst_bytes"]
    X["total_pkts"] = X["src_pkts"] + X["dst_pkts"]

    if fit:
        return X, label_encoders
    return X


if __name__ == "__main__":
    print("=" * 60)
    print("OTShield - Cyber Layer Training (TON_IoT Network)")
    print("=" * 60)

    # Load data
    csv_path = os.path.join(DATA, "train_test_network.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    y = df["label"].values
    attack_types = df["type"].values
    print(f"Normal: {(y == 0).sum()}, Attack: {(y == 1).sum()}")
    print(f"Attack types: {pd.Series(attack_types).value_counts().to_dict()}")

    # Preprocess
    X, label_encoders = preprocess_cyber(df, fit=True)
    FEATURE_NAMES = list(X.columns)
    print(f"Features: {len(FEATURE_NAMES)}")

    # Scale
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Stratified 5-fold CV
    print("\n" + "=" * 60)
    print("STRATIFIED 5-FOLD CV (Random Forest)")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, aucs = [], []

    for fold, (tr, te) in enumerate(skf.split(Xs, y)):
        rf_fold = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf_fold.fit(Xs[tr], y[tr])
        yp = rf_fold.predict_proba(Xs[te])[:, 1]
        y_pred = (yp >= 0.5).astype(int)

        f1 = f1_score(y[te], y_pred)
        auc = roc_auc_score(y[te], yp)
        f1s.append(f1)
        aucs.append(auc)
        print(f"  Fold {fold+1}: F1={f1:.4f}  AUC={auc:.4f}")

    mean_f1 = np.mean(f1s)
    mean_auc = np.mean(aucs)
    print(f"\n  >> CV F1={mean_f1:.4f}  AUC={mean_auc:.4f}")

    # Final model on all data
    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    final_rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    final_rf.fit(Xs, y)

    yp_final = final_rf.predict_proba(Xs)[:, 1]
    y_pred_final = (yp_final >= 0.5).astype(int)
    final_f1 = f1_score(y, y_pred_final)
    final_auc = roc_auc_score(y, yp_final)

    print(f"F1: {final_f1:.4f}  AUC: {final_auc:.4f}")
    print(classification_report(y, y_pred_final, target_names=["Normal", "Attack"]))

    # Save artifacts
    joblib.dump(final_rf, os.path.join(MODELS, "cyber_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS, "scaler_cyber.pkl"))

    # Save label encoders as JSON-serializable dict
    le_dict = {col: list(le.classes_) for col, le in label_encoders.items()}
    with open(os.path.join(MODELS, "cyber_label_encoders.json"), "w") as f:
        json.dump(le_dict, f, indent=2)

    with open(os.path.join(MODELS, "feature_names_cyber.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    with open(os.path.join(MODELS, "cyber_meta.json"), "w") as f:
        json.dump({
            "best_model": "random_forest",
            "cv_f1": round(mean_f1, 4),
            "cv_auc": round(mean_auc, 4),
            "final_f1": round(final_f1, 4),
            "final_auc": round(final_auc, 4),
            "n_features": len(FEATURE_NAMES),
            "n_samples": len(df),
            "attack_types": sorted(df[df["label"] == 1]["type"].unique().tolist()),
        }, f, indent=2)

    print("[OK] All cyber artifacts saved to models/")
    print(f"Done! CV F1={mean_f1:.4f} AUC={mean_auc:.4f}")
