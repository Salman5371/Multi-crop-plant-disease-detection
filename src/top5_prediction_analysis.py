import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import build_efficientnetv2s


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "efficientnetv2s_best.pth"
CLASS_NAMES_PATH = ROOT_DIR / "Data" / "class_names.json"

SAMPLE_DIR = ROOT_DIR / "Data" / "sample"

OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
TOP_K = 5


# =========================
# Helper functions
# =========================

def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(num_classes):
    model = build_efficientnetv2s(num_classes=num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

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


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_crop_and_disease(class_name):
    cleaned = class_name.replace("__", "_")
    parts = cleaned.split("_")

    crop = parts[0].title()

    if len(parts) > 1:
        disease = " ".join(parts[1:]).title()
    else:
        disease = "Unknown"

    return crop, disease


def predict_top5(model, image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=TOP_K, dim=1)

    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()

    records = []

    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        class_name = class_names[int(idx)]
        crop, disease = extract_crop_and_disease(class_name)

        records.append({
            "rank": rank,
            "predicted_index": int(idx),
            "predicted_class": class_name,
            "crop": crop,
            "disease_status": disease,
            "confidence_percent": round(float(prob) * 100, 2)
        })

    return records


def classify_prediction_case(top5_records):
    """
    Classify uncertainty based on top-1 confidence and gap between top-1 and top-2.
    """
    top1 = top5_records[0]["confidence_percent"]
    top2 = top5_records[1]["confidence_percent"] if len(top5_records) > 1 else 0
    gap = top1 - top2

    if top1 >= 90 and gap >= 30:
        return "Confident prediction"
    elif top1 < 70:
        return "Low-confidence / uncertain prediction"
    elif gap < 10:
        return "Ambiguous prediction"
    else:
        return "Moderate-confidence prediction"


def get_image_files(folder_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    return sorted([
        p for p in folder_path.iterdir()
        if p.suffix.lower() in image_extensions
    ])


# =========================
# Main analysis
# =========================

def main():
    print("Using device:", DEVICE)

    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(f"Sample folder not found: {SAMPLE_DIR}")

    class_names = load_class_names(CLASS_NAMES_PATH)
    model = load_model(num_classes=len(class_names))

    print("Model loaded successfully.")
    print("Number of classes:", len(class_names))

    image_paths = get_image_files(SAMPLE_DIR)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in sample folder: {SAMPLE_DIR}")

    all_top5_records = []
    summary_records = []

    for image_path in image_paths:
        top5_records = predict_top5(model, image_path, class_names)
        case_type = classify_prediction_case(top5_records)

        top1 = top5_records[0]
        top2 = top5_records[1] if len(top5_records) > 1 else None

        summary_records.append({
            "image_name": image_path.name,
            "top1_class": top1["predicted_class"],
            "top1_crop": top1["crop"],
            "top1_disease_status": top1["disease_status"],
            "top1_confidence_percent": top1["confidence_percent"],
            "top2_class": top2["predicted_class"] if top2 else None,
            "top2_confidence_percent": top2["confidence_percent"] if top2 else None,
            "confidence_gap_top1_top2": round(
                top1["confidence_percent"] - top2["confidence_percent"], 2
            ) if top2 else None,
            "case_type": case_type
        })

        for record in top5_records:
            record["image_name"] = image_path.name
            record["case_type"] = case_type
            all_top5_records.append(record)

        print("\n" + "=" * 70)
        print(f"Image: {image_path.name}")
        print(f"Case type: {case_type}")
        for record in top5_records:
            print(
                f"Rank {record['rank']}: "
                f"{record['predicted_class']} "
                f"({record['confidence_percent']}%)"
            )

    top5_df = pd.DataFrame(all_top5_records)
    summary_df = pd.DataFrame(summary_records)

    top5_path = OUTPUT_TABLE_DIR / "top5_prediction_details.csv"
    summary_path = OUTPUT_TABLE_DIR / "top5_prediction_summary.csv"

    top5_df.to_csv(top5_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report_text = f"""
TOP-5 PREDICTION ANALYSIS REPORT
============================================================

Total sample images analyzed: {len(image_paths)}

Summary:
{summary_df.to_string(index=False)}

Interpretation:
Top-5 prediction analysis was conducted to better understand model uncertainty.
When the top-1 confidence is low or the confidence gap between the top-1 and top-2
predictions is small, the model output should be interpreted cautiously. This is
particularly important for smart farming decision-support systems because a wrong
high-level decision based on an uncertain image may mislead farmers.

This analysis also helps explain cases where noisy, low-quality, watermarked, or
out-of-distribution images produce incorrect or unstable predictions. Therefore,
the system should recommend manual verification when the model confidence is low
or when multiple disease classes receive similar confidence scores.
"""

    report_path = OUTPUT_REPORT_DIR / "top5_prediction_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nTop-5 prediction analysis completed.")
    print("Saved files:")
    print(top5_path)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()