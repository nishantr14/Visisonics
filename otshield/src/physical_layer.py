"""OTShield Physical Layer -- sensor telemetry anomaly detection (supervised)."""

from supervised_scorer import get_supervised_score, SENSOR_COLS

# Re-export for backward compatibility
FEATURE_NAMES = SENSOR_COLS


def get_physical_score(sensor_dict):
    """
    Input:  sensor_dict with BATADAL sensor keys
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    score, explanation, top_feature = get_supervised_score(sensor_dict)

    # Physical-specific explanation text
    if score > 70:
        explanation = "Physical anomaly detected in sensor telemetry"
    elif score > 40:
        explanation = "Moderate physical deviation detected"
    else:
        explanation = "Physical sensors nominal"

    return score, explanation, top_feature
