import json
from pathlib import Path
from datasets import load_dataset, load_from_disk


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DISK_PATH = ROOT_DIR / "Data" / "processed"

CLASS_NAMES_PATH = ROOT_DIR / "Data" / "class_names.json"
LABEL2ID_PATH = ROOT_DIR / "Data" / "label2id.json"
ID2LABEL_PATH = ROOT_DIR / "Data" / "id2label.json"

HF_DATASET_NAME = "Saon110/bd-crop-vegetable-plant-disease-dataset"


def create_label_mappings(dataset_split):
    """
    Create label mappings from Hugging Face dataset split.
    Dataset must contain 'label_name' column.
    """
    label_names = dataset_split["label_name"]
    class_names = sorted(list(set(label_names)))

    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {str(idx): name for name, idx in label2id.items()}

    return class_names, label2id, id2label


def load_dataset_safely():
    """
    First try loading from local disk.
    If local disk dataset is not valid, load from Hugging Face.
    """
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


def main():
    ds = load_dataset_safely()

    print("\nDataset structure:")
    print(ds)

    if "train" not in ds:
        raise ValueError("Train split not found in dataset.")

    if "label_name" not in ds["train"].column_names:
        raise ValueError("label_name column not found in train split.")

    class_names, label2id, id2label = create_label_mappings(ds["train"])

    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)

    with open(LABEL2ID_PATH, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)

    with open(ID2LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=4, ensure_ascii=False)

    print("\nSaved files:")
    print("Class names:", CLASS_NAMES_PATH)
    print("Label2ID:", LABEL2ID_PATH)
    print("ID2Label:", ID2LABEL_PATH)

    print("\nNumber of classes:", len(class_names))
    print("First 10 classes:")
    print(class_names[:10])


if __name__ == "__main__":
    main()