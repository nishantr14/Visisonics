"""
Physical Layer: Rule-based anomaly detection for OT systems
"""

import random

# ───────────────────────────────────────────────
# THRESHOLDS (tuneable)
# ───────────────────────────────────────────────
TANK_DIFF_THRESHOLD = 2.0
FLOW_MIN = 10
FLOW_MAX = 50
PRESSURE_MIN = 1
PRESSURE_MAX = 4


# ───────────────────────────────────────────────
# MAIN FUNCTION
# ───────────────────────────────────────────────
def get_physical_score(reading: dict):
    """
    Computes anomaly score (0–1) based on physical system behavior.

    Returns:
        score (float): anomaly score
        features (dict): contributing values
        info (dict): explanation metadata
    """

    # ── Extract sensor values ──
    t1 = reading.get("L_T1", 5)
    t2 = reading.get("L_T2", 5)
    t3 = reading.get("L_T3", 5)

    flow = reading.get("FLOW", 30)
    pressure = reading.get("PRESSURE", 2)

    # ── Initialize ──
    score = 0.0
    reasons = []

    # ─────────────────────────────
    # 1. Tank imbalance
    # ─────────────────────────────
    tank_diff = max(t1, t2, t3) - min(t1, t2, t3)

    if tank_diff > TANK_DIFF_THRESHOLD:
        score += 0.4
        reasons.append(f"Tank imbalance detected (Δ={round(tank_diff,2)})")

    # ─────────────────────────────
    # 2. Flow anomaly
    # ─────────────────────────────
    if flow < FLOW_MIN or flow > FLOW_MAX:
        score += 0.3
        reasons.append(f"Flow out of range ({round(flow,2)})")

    # ─────────────────────────────
    # 3. Pressure anomaly
    # ─────────────────────────────
    if pressure < PRESSURE_MIN or pressure > PRESSURE_MAX:
        score += 0.3
        reasons.append(f"Pressure anomaly ({round(pressure,2)})")

    # ─────────────────────────────
    # 4. Correlation anomaly
    # ─────────────────────────────
    if flow > 45 and pressure < 1.2:
        score += 0.2
        reasons.append("Flow-pressure mismatch")

    # ─────────────────────────────
    # Clamp score
    # ─────────────────────────────
    score = min(score, 1.0)

    # ─────────────────────────────
    # Add slight noise (realism)
    # ─────────────────────────────
    score = min(1.0, max(0.0, score + random.uniform(-0.05, 0.05)))

    # ─────────────────────────────
    # Explanation
    # ─────────────────────────────
    if not reasons:
        explanation = "Physical system operating normally"
    else:
        explanation = "; ".join(reasons)

    # ─────────────────────────────
    # Feature output (for debugging/UI)
    # ─────────────────────────────
    features = {
        "tank_levels": [round(t1, 2), round(t2, 2), round(t3, 2)],
        "flow": round(flow, 2),
        "pressure": round(pressure, 2),
        "tank_diff": round(tank_diff, 2),
    }

    info = {
        "explanation": explanation
    }

    return round(score, 4), features, info
