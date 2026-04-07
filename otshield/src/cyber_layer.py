"""OTShield cyber layer anomaly inference (TON data model only)."""

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(rel_path: str) -> str:
    return os.path.join(BASE, rel_path)


with open(_path("models/feature_names_cyber.json"), encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)

with open(_path("models/if_score_bounds.json"), encoding="utf-8") as f:
    IF_BOUNDS = json.load(f)

with open(_path("models/optimized_thresholds_cyber.json"), encoding="utf-8") as f:
    OPT_THRESHOLDS = json.load(f)

with open(_path("models/autoencoder_threshold.json"), encoding="utf-8") as f:
    AE_THRESHOLD = json.load(f)["threshold"]

BEST_MODEL = "isolation_forest"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_from_bounds(raw: float, bounds: dict) -> float:
    minimum = float(bounds.get("min", -1.0))
    maximum = float(bounds.get("max", 1.0))
    if maximum <= minimum:
        return _clip01(raw)
    return _clip01((raw - minimum) / (maximum - minimum))


class _CyberArtifacts:
    """Cyber-only model/scaler artifacts trained off TON data."""

    def __init__(self) -> None:
        self.model_if = joblib.load(_path("models/isolation_forest.pkl"))
        self.model_ocsvm = joblib.load(_path("models/ocsvm_cyber.pkl"))
        self.scaler = joblib.load(_path("models/scaler_cyber.pkl"))
        from tensorflow import keras

        self.model_ae = keras.models.load_model(_path("models/autoencoder_cyber.h5"))
        self.explainer = shap.TreeExplainer(self.model_if)


_ARTIFACTS = None


def _artifacts() -> _CyberArtifacts:
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _CyberArtifacts()
    return _ARTIFACTS


def get_cyber_score(sensor_dict: dict) -> tuple[float, str, str]:
    """Return TON cyber anomaly score in 0-1 with explanation and top feature."""
    artifacts = _artifacts()
    row = {name: sensor_dict.get(name, 0.0) for name in FEATURE_NAMES}
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    X = artifacts.scaler.transform(df)

    if BEST_MODEL == "isolation_forest":
        raw = float(artifacts.model_if.decision_function(X)[0])
        threshold = float(OPT_THRESHOLDS.get("isolation_forest", 0.0))
        threshold_n = _normalize_from_bounds(threshold, IF_BOUNDS)
        raw_n = _normalize_from_bounds(raw, IF_BOUNDS)
        score = _clip01(threshold_n - raw_n + 0.5)
    elif BEST_MODEL == "ocsvm":
        raw = float(artifacts.model_ocsvm.decision_function(X)[0])
        threshold = float(OPT_THRESHOLDS.get("ocsvm", 0.0))
        score = _clip01(0.5 + (threshold - raw))
    else:
        recon = artifacts.model_ae.predict(X, verbose=0)
        recon_error = float(np.mean((X - recon) ** 2))
        threshold = float(OPT_THRESHOLDS.get("autoencoder", AE_THRESHOLD))
        score = _clip01(recon_error / (threshold * 2.0)) if threshold > 0 else _clip01(recon_error)

    shap_vals = artifacts.explainer.shap_values(X)
    top_feature = FEATURE_NAMES[int(np.abs(shap_vals[0]).argmax())]

    if score > 0.7:
        explanation = "High cyber anomaly detected"
    elif score > 0.4:
        explanation = "Moderate cyber anomaly detected"
    else:
        explanation = "Cyber behavior within baseline"

    return round(score, 4), explanation, top_feature
