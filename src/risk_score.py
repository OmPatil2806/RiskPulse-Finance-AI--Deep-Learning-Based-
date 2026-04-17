"""
RiskPulse Finance AI — Financial Stress Risk Score Module
Converts model predictions into actionable fintech risk scores.
"""

import numpy as np

# Risk band definitions 
RISK_BANDS = {
    "Normal":     (0,   25),
    "Anxiety":    (25,  60),
    "Depression": (60, 100),
    "Other":      (40,  70),
}

RISK_LEVEL_THRESHOLDS = [
    (0,  35,  "Low Risk",    "#2ecc71"),   # green
    (35, 65,  "Medium Risk", "#f39c12"),   # orange
    (65, 100, "High Risk",   "#e74c3c"),   # red
]

RISK_ADVICE = {
    "Low Risk":    "Financial health appears stable. Regular monitoring recommended.",
    "Medium Risk": "Moderate stress indicators detected. Consider proactive financial counselling.",
    "High Risk":   "High stress indicators detected. Immediate financial and mental health support advised.",
}


def compute_risk_score(predicted_class: str, probabilities: np.ndarray) -> float:
    """
    Map predicted class + confidence → scalar risk score in [0, 100].

    Formula:
        base  = midpoint of class band
        delta = (confidence - 0.5) * band_width   (confidence in [0, 1])
        score = clip(base + delta, band_lo, band_hi)
    """
    lo, hi   = RISK_BANDS.get(predicted_class, (40, 70))
    mid      = (lo + hi) / 2
    width    = (hi - lo) / 2
    conf     = float(np.max(probabilities))
    delta    = (conf - 0.5) * width
    score    = float(np.clip(mid + delta, lo, hi))
    return round(score, 2)


def risk_level(score: float) -> str:
    for lo, hi, label, _ in RISK_LEVEL_THRESHOLDS:
        if lo <= score < hi:
            return label
    return "High Risk"


def risk_color(score: float) -> str:
    for lo, hi, _, color in RISK_LEVEL_THRESHOLDS:
        if lo <= score < hi:
            return color
    return "#e74c3c"


def risk_advice(level: str) -> str:
    return RISK_ADVICE.get(level, "Consult a financial wellness advisor.")


def full_risk_report(predicted_class: str, probabilities: np.ndarray, class_names: list) -> dict:
    """Return a complete risk report dict for UI rendering."""
    score  = compute_risk_score(predicted_class, probabilities)
    level  = risk_level(score)
    color  = risk_color(score)
    advice = risk_advice(level)

    prob_dict = {cls: round(float(p) * 100, 2)
                 for cls, p in zip(class_names, probabilities)}

    return {
        "predicted_class": predicted_class,
        "risk_score":      score,
        "risk_level":      level,
        "risk_color":      color,
        "advice":          advice,
        "probabilities":   prob_dict,
    }


if __name__ == "__main__":
    import numpy as np

    # fake model output (softmax probabilities)
    probabilities = np.array([0.1, 0.2, 0.6, 0.1])

    class_names = ["Normal", "Anxiety", "Depression", "Other"]

    predicted_class = class_names[np.argmax(probabilities)]

    report = full_risk_report(predicted_class, probabilities, class_names)

    print("\n=== RISK REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")