import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from sklearn.metrics import classification_report, confusion_matrix

from src.models import build_efficientnetv2s


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DISK_PATH = ROOT_DIR / "Data" / "processed"
HF_DATASET_NAME = "Saon110/bd-crop-vegetable-plant-disease-dataset"

MODEL_PATH = ROOT_DIR / "models" / "efficientnetv2s_best.pth"
CLASS_NAMES_PATH = ROOT_DIR / "Data" / "class_names.json"
LABEL2ID_PATH = ROOT_DIR / "Data" / "label2id.json"

OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_FIGURE_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# =========================
# Dataset loading
# =========================

def load_dataset_safely():
    try:
        print("Trying to load dataset from local disk:", DATA_DISK_PATH)
        ds = load_from_disk(str(DATA_DISK_PATH))
        print("Dataset loaded from local disk.")
        return ds

    except Exception as e:
        print("Local dataset loading failed.")
        print("Reason:", e)
        print("\nTrying to load dataset from Hugging Face...")
        ds = load_dataset(HF_DATASET_NAME)
        print("Dataset loaded from Hugging Face.")
        return ds


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Test dataset wrapper
# =========================

class HFTestDataset(Dataset):
    def __init__(self, hf_dataset, label2id, transform=None):
        self.hf_dataset = hf_dataset
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        image = item["image"].convert("RGB")
        label_name = item["label_name"]
        label = self.label2id[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label, label_name


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================
# Model loading
# =========================

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


# =========================
# Evaluation
# =========================

def evaluate_model(model, test_loader, class_names):
    all_true = []
    all_pred = []
    all_confidence = []
    all_true_names = []
    all_pred_names = []

    with torch.no_grad():
        for images, labels, label_names in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            confidence, preds = torch.max(probs, dim=1)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_confidence.extend(confidence.cpu().numpy().tolist())

            all_true_names.extend(list(label_names))
            all_pred_names.extend([class_names[p] for p in preds.cpu().numpy().tolist()])

    prediction_df = pd.DataFrame({
        "true_label_id": all_true,
        "predicted_label_id": all_pred,
        "true_label_name": all_true_names,
        "predicted_label_name": all_pred_names,
        "confidence": all_confidence,
        "correct": np.array(all_true) == np.array(all_pred)
    })

    return prediction_df, all_true, all_pred


# =========================
# Analysis functions
# =========================

def create_classification_report_df(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class_name"})

    class_report_df = report_df[
        report_df["class_name"].isin(class_names)
    ].copy()

    class_report_df = class_report_df.sort_values("f1-score", ascending=True)

    return report_df, class_report_df


def create_top_confusion_pairs(y_true, y_pred, class_names, top_n=30):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    records = []

    for true_idx in range(len(class_names)):
        true_total = cm[true_idx, :].sum()

        for pred_idx in range(len(class_names)):
            if true_idx == pred_idx:
                continue

            count = cm[true_idx, pred_idx]

            if count > 0:
                confusion_rate = count / true_total if true_total > 0 else 0

                records.append({
                    "true_class": class_names[true_idx],
                    "predicted_class": class_names[pred_idx],
                    "confusion_count": int(count),
                    "true_class_support": int(true_total),
                    "confusion_rate_percent": round(confusion_rate * 100, 2)
                })

    confusion_df = pd.DataFrame(records)

    if len(confusion_df) > 0:
        confusion_df = confusion_df.sort_values(
            ["confusion_count", "confusion_rate_percent"],
            ascending=False
        )

    top_confusions = confusion_df.head(top_n).copy()

    return confusion_df, top_confusions


def plot_top_confusions(top_confusions, save_path):
    if len(top_confusions) == 0:
        print("No confusion pairs found.")
        return

    plot_df = top_confusions.head(15).copy()
    plot_df["pair"] = (
        "True: " + plot_df["true_class"] +
        "\nPred: " + plot_df["predicted_class"]
    )

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df["pair"], plot_df["confusion_count"])
    plt.xlabel("Number of Misclassifications")
    plt.ylabel("Confused Class Pair")
    plt.title("Top Confused Class Pairs")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_low_f1_classes(low_classes, save_path):
    plot_df = low_classes.head(15).copy()

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["class_name"], plot_df["f1-score"])
    plt.xlabel("F1-score")
    plt.ylabel("Class Name")
    plt.title("Lowest Performing Classes by F1-score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================

def main():
    print("Using device:", DEVICE)
    print("Starting error analysis...")

    class_names = load_json(CLASS_NAMES_PATH)
    label2id = load_json(LABEL2ID_PATH)

    ds = load_dataset_safely()

    if "test" not in ds:
        raise ValueError("Test split not found in dataset.")

    test_dataset = HFTestDataset(
        hf_dataset=ds["test"],
        label2id=label2id,
        transform=get_test_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = load_model(num_classes=len(class_names))
    print("Model loaded successfully.")

    prediction_df, y_true, y_pred = evaluate_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names
    )

    prediction_path = OUTPUT_TABLE_DIR / "efficientnetv2s_test_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    print("Prediction file saved:", prediction_path)

    report_df, class_report_df = create_classification_report_df(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names
    )

    full_report_path = OUTPUT_TABLE_DIR / "efficientnetv2s_classification_report.csv"
    low10_path = OUTPUT_TABLE_DIR / "low10_performing_classes.csv"
    high10_path = OUTPUT_TABLE_DIR / "high10_performing_classes.csv"

    report_df.to_csv(full_report_path, index=False, encoding="utf-8-sig")
    class_report_df.head(10).to_csv(low10_path, index=False, encoding="utf-8-sig")
    class_report_df.tail(10).sort_values("f1-score", ascending=False).to_csv(
        high10_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Classification report saved:", full_report_path)

    confusion_df, top_confusions = create_top_confusion_pairs(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        top_n=30
    )

    all_confusions_path = OUTPUT_TABLE_DIR / "all_confusion_pairs.csv"
    top_confusions_path = OUTPUT_TABLE_DIR / "top30_confused_class_pairs.csv"

    confusion_df.to_csv(all_confusions_path, index=False, encoding="utf-8-sig")
    top_confusions.to_csv(top_confusions_path, index=False, encoding="utf-8-sig")

    print("Top confusion pairs saved:", top_confusions_path)

    # Save figures
    top_confusion_fig = OUTPUT_FIGURE_DIR / "top_confused_class_pairs.png"
    low_f1_fig = OUTPUT_FIGURE_DIR / "low_performing_classes_f1.png"

    plot_top_confusions(top_confusions, top_confusion_fig)
    plot_low_f1_classes(class_report_df, low_f1_fig)

    # Summary report
    accuracy = prediction_df["correct"].mean()
    total_test = len(prediction_df)
    total_wrong = (~prediction_df["correct"]).sum()

    report_text = f"""
ERROR ANALYSIS REPORT
============================================================

Model: EfficientNetV2-S
Test samples: {total_test}
Correct predictions: {total_test - total_wrong}
Wrong predictions: {total_wrong}
Test accuracy from prediction file: {accuracy:.4f}

Lowest performing classes by F1-score:
{class_report_df.head(10)[['class_name', 'precision', 'recall', 'f1-score', 'support']].to_string(index=False)}

Top confused class pairs:
{top_confusions.head(10).to_string(index=False)}

Interpretation:
The error analysis identifies classes where the model has lower performance and
class pairs where misclassification occurs frequently. Low-performing classes may
be affected by limited sample size, visual similarity between disease symptoms,
or noisy input patterns. This analysis is important because overall accuracy alone
does not fully explain the strengths and weaknesses of the model across all 94 classes.
"""

    report_path = OUTPUT_REPORT_DIR / "error_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Error analysis report saved:", report_path)
    print("Figures saved:")
    print(top_confusion_fig)
    print(low_f1_fig)

    print("\nError analysis completed successfully.")


if __name__ == "__main__":
    main()