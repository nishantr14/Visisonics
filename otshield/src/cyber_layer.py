"""OTShield Cyber Layer -- anomaly detection inference (supervised)."""

from otshield.src.supervised_scorer import get_supervised_score, SENSOR_COLS

# Re-export for backward compatibility
FEATURE_NAMES = SENSOR_COLS


def get_cyber_score(sensor_dict):
    """
    Input:  sensor_dict with BATADAL sensor keys
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    score, explanation, top_feature = get_supervised_score(sensor_dict)

    # Cyber-specific explanation text
    if score > 70:
        explanation = "High anomaly detected in sensor pattern"
    elif score > 40:
        explanation = "Moderate anomaly detected"
    else:
        explanation = "Normal behavior"

    return score, explanation, top_feature
