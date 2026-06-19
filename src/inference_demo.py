import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import build_efficientnetv2s
from src.decision_support import (
    estimate_preliminary_severity,
    get_management_recommendation,
    generate_alert
)


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "efficientnetv2s_best.pth"
CLASS_NAMES_PATH = ROOT_DIR / "Data" / "class_names.json"

DEFAULT_IMAGE_PATH = ROOT_DIR / "Data" / "sample" / "test_leaf.jpg"

OUTPUT_DIR = ROOT_DIR / "outputs" / "prediction_reports"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Utility functions
# =========================

def load_class_names(path):
    """
    Load class names from JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path, num_classes):
    """
    Load trained EfficientNetV2-S model.
    """
    model = build_efficientnetv2s(num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def preprocess_image(image_path):
    """
    Preprocess one image for EfficientNetV2-S inference.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def extract_crop_and_disease(class_name):
    """
    Extract crop and disease name from predicted class label.
    Example:
    Tomato_Early_Blight -> Tomato, Early Blight
    """
    cleaned = class_name.replace("__", "_")
    parts = cleaned.split("_")

    crop = parts[0].title()

    if len(parts) > 1:
        disease = " ".join(parts[1:]).title()
    else:
        disease = "Unknown"

    return crop, disease


def predict(model, image_path, class_names):
    """
    Predict disease class and confidence score.
    """
    image_tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = predicted_index.item()
    confidence = confidence.item()
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence, predicted_index


def save_prediction_report(
    image_path,
    predicted_class,
    crop,
    disease,
    confidence,
    severity,
    severity_note,
    recommendations,
    alert
):
    """
    Save prediction output as a text report.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_stem = Path(image_path).stem
    report_path = OUTPUT_DIR / f"{image_stem}_prediction_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SMART FARMING DISEASE DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input Information\n")
        f.write("-" * 60 + "\n")
        f.write(f"Image Path       : {image_path}\n\n")

        f.write("Prediction Result\n")
        f.write("-" * 60 + "\n")
        f.write(f"Predicted Class  : {predicted_class}\n")
        f.write(f"Crop             : {crop}\n")
        f.write(f"Disease/Status   : {disease}\n")
        f.write(f"Confidence       : {confidence * 100:.2f}%\n\n")

        f.write("Severity / Risk Indication\n")
        f.write("-" * 60 + "\n")
        f.write(f"Level            : {severity}\n")
        f.write(f"Note             : {severity_note}\n\n")

        f.write("Management Recommendation\n")
        f.write("-" * 60 + "\n")
        for i, rec in enumerate(recommendations, start=1):
            f.write(f"{i}. {rec}\n")

        f.write("\nAlert Message\n")
        f.write("-" * 60 + "\n")
        f.write(alert)

    return report_path


def run_inference(image_path):
    """
    Complete inference + decision support + report saving pipeline.
    """
    print("Using device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    class_names = load_class_names(CLASS_NAMES_PATH)
    print("Number of classes:", len(class_names))

    model = load_model(MODEL_PATH, num_classes=len(class_names))
    print("Model loaded successfully.")

    predicted_class, confidence, predicted_index = predict(
        model=model,
        image_path=image_path,
        class_names=class_names
    )

    crop, disease = extract_crop_and_disease(predicted_class)

    severity, severity_note = estimate_preliminary_severity(
        predicted_class=predicted_class,
        confidence=confidence
    )

    recommendations = get_management_recommendation(predicted_class)

    alert = generate_alert(
        crop=crop,
        disease=disease,
        confidence=confidence,
        severity=severity
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Image           : {image_path.name}")
    print(f"Predicted Index : {predicted_index}")
    print(f"Predicted Class : {predicted_class}")
    print(f"Crop            : {crop}")
    print(f"Disease/Status  : {disease}")
    print(f"Confidence      : {confidence * 100:.2f}%")

    print("\n" + "=" * 60)
    print("SEVERITY / RISK INDICATION")
    print("=" * 60)
    print(f"Level : {severity}")
    print(f"Note  : {severity_note}")

    print("\n" + "=" * 60)
    print("MANAGEMENT RECOMMENDATION")
    print("=" * 60)
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec}")

    print("\n" + "=" * 60)
    print("ALERT MESSAGE")
    print("=" * 60)
    print(alert)

    report_path = save_prediction_report(
        image_path=image_path,
        predicted_class=predicted_class,
        crop=crop,
        disease=disease,
        confidence=confidence,
        severity=severity,
        severity_note=severity_note,
        recommendations=recommendations,
        alert=alert
    )

    print("\n" + "=" * 60)
    print("REPORT SAVED")
    print("=" * 60)
    print(f"Report saved at: {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Run plant disease prediction and save decision support report."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to input leaf image."
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    run_inference(image_path)


if __name__ == "__main__":
    main()