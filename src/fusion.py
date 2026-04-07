"""
Advanced Fusion Engine for OTShield
"""

def _clip01(x):
    return max(0.0, min(1.0, float(x)))


# ───────────────────────────────────────────────
# CORE FUSION LOGIC
# ───────────────────────────────────────────────
def fuse_scores(cyber, physical):
    cyber = _clip01(cyber)
    physical = _clip01(physical)

    # 🔥 Agreement-based confidence
    agreement = 1 - abs(cyber - physical)

    # 🔥 Dynamic weighting
    if physical > 0.7:
        # trust physical more (real-world faults matter more)
        w_cyber, w_physical = 0.35, 0.65
    elif cyber > 0.7:
        # cyber attack scenario
        w_cyber, w_physical = 0.6, 0.4
    else:
        # balanced
        w_cyber, w_physical = 0.5, 0.5

    # 🔥 Fusion score
    fused = (w_cyber * cyber) + (w_physical * physical)

    # 🔥 Boost if both high (coordinated attack)
    if cyber > 0.7 and physical > 0.7:
        fused += 0.1

    fused = _clip01(fused)

    return round(fused, 4), agreement, w_cyber, w_physical


# ───────────────────────────────────────────────
# RISK SCORE (EXTENDED)
# ───────────────────────────────────────────────
def get_risk_score(cyber, physical, audio=None, visual=None):
    base_score, agreement, wc, wp = fuse_scores(cyber, physical)

    # optional modalities
    if audio is not None:
        base_score += 0.1 * audio
    if visual is not None:
        base_score += 0.1 * visual

    return round(_clip01(base_score), 4), {
        "agreement": round(agreement, 4),
        "cyber_weight": wc,
        "physical_weight": wp
    }


# ───────────────────────────────────────────────
# RISK CLASSIFICATION (SMART)
# ───────────────────────────────────────────────
def classify_risk(cyber, physical, fusion_score):
    if cyber > 0.7 and physical > 0.7:
        return "CRITICAL"

    if cyber > 0.7 and physical < 0.5:
        return "CYBER THREAT"

    if physical > 0.7 and cyber < 0.5:
        return "PHYSICAL FAULT"

    if fusion_score > 0.6:
        return "WARNING"

    return "SAFE"


# ───────────────────────────────────────────────
# EXPLANATION ENGINE (UPGRADED)
# ───────────────────────────────────────────────
def generate_explanation(cyber, physical):
    if cyber > 0.7 and physical > 0.7:
        return "Coordinated cyber-physical anomaly detected"

    if cyber > 0.7:
        return "Suspicious network activity detected"

    if physical > 0.7:
        return "Abnormal physical system behavior detected"

    if cyber > 0.4 or physical > 0.4:
        return "Minor deviations observed in system behavior"

    return "System operating within normal parameters"