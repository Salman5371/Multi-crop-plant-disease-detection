import json
from pathlib import Path

import pandas as pd
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

SAMPLE_DIR = ROOT_DIR / "Data" / "sample"

OUTPUT_DIR = ROOT_DIR / "outputs" / "prediction_reports"
OUTPUT_CSV = OUTPUT_DIR / "batch_prediction_results.csv"

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
    Preprocess image for inference.
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
    Extract crop and disease/status from predicted class label.
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
    Predict disease class for one image.
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


def get_image_files(folder_path):
    """
    Collect all image files from sample folder.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    image_paths = [
        file for file in folder_path.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    return sorted(image_paths)


def save_individual_report(
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
    Save individual text report for each image.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / f"{image_path.stem}_prediction_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SMART FARMING DISEASE DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input Information\n")
        f.write("-" * 60 + "\n")
        f.write(f"Image Name       : {image_path.name}\n")
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


# =========================
# Main batch inference
# =========================

def main():
    print("Using device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(f"Sample image folder not found: {SAMPLE_DIR}")

    class_names = load_class_names(CLASS_NAMES_PATH)
    print("Number of classes:", len(class_names))

    model = load_model(MODEL_PATH, num_classes=len(class_names))
    print("Model loaded successfully.")

    image_paths = get_image_files(SAMPLE_DIR)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in: {SAMPLE_DIR}")

    print(f"Total sample images found: {len(image_paths)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_results = []

    for image_path in image_paths:
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

        report_path = save_individual_report(
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

        batch_results.append({
            "image_name": image_path.name,
            "predicted_index": predicted_index,
            "predicted_class": predicted_class,
            "crop": crop,
            "disease_status": disease,
            "confidence_percent": round(confidence * 100, 2),
            "severity_or_risk": severity,
            "severity_note": severity_note,
            "recommendation_summary": " | ".join(recommendations[:3]),
            "individual_report_path": str(report_path)
        })

        print("\n" + "=" * 60)
        print(f"Image           : {image_path.name}")
        print(f"Predicted Class : {predicted_class}")
        print(f"Crop            : {crop}")
        print(f"Disease/Status  : {disease}")
        print(f"Confidence      : {confidence * 100:.2f}%")
        print(f"Severity/Risk   : {severity}")
        print(f"Report Saved    : {report_path}")

    df = pd.DataFrame(batch_results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 60)
    print(f"CSV report saved at: {OUTPUT_CSV}")
    print(f"Total images processed: {len(batch_results)}")


if __name__ == "__main__":
    main()