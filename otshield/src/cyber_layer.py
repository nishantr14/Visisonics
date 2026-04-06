"""OTShield Cyber Layer -- network traffic anomaly detection (TON_IoT RF model)."""

from cyber_scorer import get_cyber_score


def get_cyber_layer_score(network_dict):
    """
    Input:  network_dict with TON_IoT flow keys
    Output: (score: float 0-100, explanation: str, top_feature: str)
    """
    return get_cyber_score(network_dict)
