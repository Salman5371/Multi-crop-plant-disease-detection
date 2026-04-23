from typing import Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, label2id: Dict[str, int], transform=None):
        self.hf_dataset = hf_dataset
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label_name = item["label_name"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        label = self.label2id[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


class HFDatasetWrapperGradCAM(Dataset):
    def __init__(self, hf_dataset, label2id: Dict[str, int], transform=None):
        self.hf_dataset = hf_dataset
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label_name = item["label_name"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        original_image = image.copy()
        label = self.label2id[label_name]

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return image_tensor, label, original_image, label_name


def create_label_mappings(train_split) -> Tuple[list, Dict[str, int], Dict[int, str]]:
    class_names = sorted(list(set(train_split["label_name"])))
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for name, i in label2id.items()}
    return class_names, label2id, id2label


def create_dataloaders(
    ds,
    label2id,
    train_transform,
    val_test_transform,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    train_dataset = HFDatasetWrapper(ds["train"], label2id=label2id, transform=train_transform)
    val_dataset = HFDatasetWrapper(ds["valid"], label2id=label2id, transform=val_test_transform)
    test_dataset = HFDatasetWrapper(ds["test"], label2id=label2id, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader