"""Tri-modal fusion engine for OTShield risk scoring."""


def get_risk_score(cyber, physical, audio=None, visual=None):
    """Compute weighted fusion score and return (score, mode_string)."""
    if audio is not None and visual is not None:
        score = 0.35 * cyber + 0.30 * physical + 0.20 * audio + 0.15 * visual
        mode = "Quad-Modal"
    elif audio is not None:
        score = 0.40 * cyber + 0.35 * physical + 0.25 * audio
        mode = "Tri-Modal"
    elif visual is not None:
        score = 0.40 * cyber + 0.35 * physical + 0.25 * visual
        mode = "Tri-Modal"
    else:
        score = 0.60 * cyber + 0.40 * physical
        mode = "Bi-Modal (Core)"

    return round(score * 100, 1), mode


def _level(val):
    """Classify a single 0-1 score as HIGH/MEDIUM/LOW."""
    if val > 0.6:
        return "HIGH"
    elif val >= 0.3:
        return "MEDIUM"
    return "LOW"


def classify_risk(cyber, physical, audio=None, visual=None):
    """Return risk label based on individual layer scores (0-1 scale)."""
    c = _level(cyber)
    p = _level(physical)
    a = _level(audio) if audio is not None else "LOW"
    v = _level(visual) if visual is not None else "LOW"

    if c == "HIGH" and p == "HIGH":
        return "CRITICAL"
    if c == "HIGH":
        return "CYBER THREAT"
    if p == "HIGH" and (a == "HIGH" or v == "HIGH"):
        return "PHYSICAL FAULT"
    if p == "HIGH" or a == "HIGH" or v == "HIGH":
        return "WARNING"
    return "SAFE"


def get_recommended_action(risk_level, top_feature=""):
    """Return operator-facing recommendation string."""
    actions = {
        "CRITICAL":       f"ISOLATE affected PLC segment \u2014 Dispatch inspection \u2014 {top_feature}",
        "CYBER THREAT":   f"Verify network commands \u2014 Check PLC access logs \u2014 {top_feature}",
        "PHYSICAL FAULT": f"Dispatch maintenance to flagged equipment \u2014 {top_feature}",
        "WARNING":        f"Monitor closely \u2014 {top_feature}",
        "SAFE":           "All systems nominal",
    }
    return actions.get(risk_level, "All systems nominal")
