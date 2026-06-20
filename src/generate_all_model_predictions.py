import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk

import src.models as model_zoo


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DISK_PATH = ROOT_DIR / "Data" / "processed"
HF_DATASET_NAME = "Saon110/bd-crop-vegetable-plant-disease-dataset"

CLASS_NAMES_PATH = ROOT_DIR / "Data" / "class_names.json"
LABEL2ID_PATH = ROOT_DIR / "Data" / "label2id.json"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


MODEL_CONFIGS = {
    "mobilenetv2": {
        "builder_aliases": ["build_mobilenetv2", "build_mobile_net_v2"],
        "weight_path": MODEL_DIR / "mobilenetv2_best.pth",
        "output_path": OUTPUT_TABLE_DIR / "mobilenetv2_test_predictions.csv",
    },
    "resnet50": {
        "builder_aliases": ["build_resnet50", "build_resnet_50"],
        "weight_path": MODEL_DIR / "resnet50_best.pth",
        "output_path": OUTPUT_TABLE_DIR / "resnet50_test_predictions.csv",
    },
    "efficientnetv2s": {
        "builder_aliases": ["build_efficientnetv2s", "build_efficientnet_v2_s", "build_efficientnet2s"],
        "weight_path": MODEL_DIR / "efficientnetv2s_best.pth",
        "output_path": OUTPUT_TABLE_DIR / "efficientnetv2s_test_predictions_for_stats.csv",
    },
}


# =========================
# Dataset
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

        return image, label, label_name, idx


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
# Model functions
# =========================

def find_model_builder(builder_aliases):
    for name in builder_aliases:
        if hasattr(model_zoo, name):
            return getattr(model_zoo, name), name

    raise AttributeError(f"No matching builder found. Tried: {builder_aliases}")


def load_model(weight_path, builder_aliases, num_classes):
    builder_fn, builder_name = find_model_builder(builder_aliases)
    print("Using builder:", builder_name)

    model = builder_fn(num_classes=num_classes)

    checkpoint = torch.load(weight_path, map_location=DEVICE)

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


def predict_model(model, test_loader, class_names):
    records = []

    with torch.no_grad():
        for images, labels, label_names, indices in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, dim=1)

            labels_cpu = labels.cpu().numpy().tolist()
            preds_cpu = preds.cpu().numpy().tolist()
            conf_cpu = confidence.cpu().numpy().tolist()
            indices_cpu = indices.cpu().numpy().tolist()

            for idx, true_id, pred_id, conf, true_name in zip(
                indices_cpu, labels_cpu, preds_cpu, conf_cpu, label_names
            ):
                records.append({
                    "sample_index": idx,
                    "true_label_id": true_id,
                    "predicted_label_id": pred_id,
                    "true_label_name": true_name,
                    "predicted_label_name": class_names[pred_id],
                    "confidence": conf,
                    "correct": true_id == pred_id
                })

    return pd.DataFrame(records)


# =========================
# Main
# =========================

def main():
    print("Using device:", DEVICE)

    class_names = load_json(CLASS_NAMES_PATH)
    label2id = load_json(LABEL2ID_PATH)

    ds = load_dataset_safely()

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

    for model_key, config in MODEL_CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"Generating test predictions for: {model_key}")
        print("=" * 70)

        if not config["weight_path"].exists():
            print(f"Weight file not found: {config['weight_path']}")
            continue

        model = load_model(
            weight_path=config["weight_path"],
            builder_aliases=config["builder_aliases"],
            num_classes=len(class_names)
        )

        pred_df = predict_model(
            model=model,
            test_loader=test_loader,
            class_names=class_names
        )

        pred_df.to_csv(config["output_path"], index=False, encoding="utf-8-sig")

        accuracy = pred_df["correct"].mean()

        print(f"Saved predictions: {config['output_path']}")
        print(f"Accuracy: {accuracy:.4f}")

    print("\nAll model prediction files generated.")


if __name__ == "__main__":
    main()