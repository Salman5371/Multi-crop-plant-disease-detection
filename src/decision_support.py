def estimate_preliminary_severity(predicted_class, confidence):
    """
    Preliminary image-based severity/risk indication.
    This is not ground-truth biological severity estimation.
    """
    label = predicted_class.lower()

    if "healthy" in label:
        return "No visible disease detected", "No immediate treatment required."

    if confidence >= 0.90:
        severity = "High attention required"
        note = "The model is highly confident about disease presence. Immediate monitoring is recommended."
    elif confidence >= 0.70:
        severity = "Moderate attention required"
        note = "The model indicates possible disease presence. Further field inspection is recommended."
    else:
        severity = "Low confidence / uncertain"
        note = "The prediction is uncertain. Manual inspection or another image is recommended."

    return severity, note


def get_management_recommendation(predicted_class):
    """
    Rule-based disease management recommendation.
    """
    label = predicted_class.lower()

    if "healthy" in label:
        return [
            "No disease symptoms were detected.",
            "Continue regular field monitoring.",
            "Maintain proper irrigation, spacing, and sanitation.",
            "Avoid unnecessary pesticide application."
        ]

    recommendations = [
        "Remove and destroy severely infected leaves or plant parts where appropriate.",
        "Avoid unnecessary overhead irrigation to reduce leaf wetness.",
        "Improve field sanitation and remove crop residues that may carry pathogens.",
        "Maintain proper plant spacing to improve air circulation.",
        "Monitor nearby plants regularly for similar symptoms."
    ]

    if "blight" in label:
        recommendations.append("For blight-like symptoms, improve air movement and avoid prolonged moisture on leaves.")
        recommendations.append("Use locally recommended fungicide only after expert or extension-service confirmation.")

    if "rot" in label:
        recommendations.append("For rot-like symptoms, avoid waterlogging and improve drainage conditions.")
        recommendations.append("Remove infected plant parts carefully to reduce disease spread.")

    if "rust" in label:
        recommendations.append("For rust-like symptoms, remove infected leaves and avoid dense planting.")

    if "mold" in label or "mildew" in label:
        recommendations.append("For mold or mildew symptoms, reduce humidity around the crop canopy.")

    if "spot" in label:
        recommendations.append("For leaf spot symptoms, avoid leaf wetness and remove heavily affected leaves.")

    if "bacterial" in label:
        recommendations.append("For suspected bacterial disease, avoid working in the field when leaves are wet.")
        recommendations.append("Use clean tools and avoid spreading infection mechanically.")

    if "virus" in label or "viral" in label:
        recommendations.append("For viral symptoms, remove infected plants and control insect vectors where applicable.")

    return recommendations


def generate_alert(crop, disease, confidence, severity):
    confidence_percent = confidence * 100

    if "healthy" in disease.lower():
        return f"""
SMART FARMING ALERT

Crop: {crop}
Status: Healthy
Confidence: {confidence_percent:.2f}%

Message:
No visible disease symptoms were detected. Continue regular monitoring and maintain good crop management practices.
"""

    return f"""
SMART FARMING ALERT

Crop: {crop}
Detected Problem: {disease}
Confidence: {confidence_percent:.2f}%
Severity / Risk Level: {severity}

Recommended Action:
Please inspect the affected plant area carefully and follow the suggested management recommendations.
"""