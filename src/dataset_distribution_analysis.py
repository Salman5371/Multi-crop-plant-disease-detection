import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DISK_PATH = ROOT_DIR / "Data" / "processed"
HF_DATASET_NAME = "Saon110/bd-crop-vegetable-plant-disease-dataset"

OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_FIGURE_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Dataset loading
# =========================

def load_dataset_safely():
    """
    Try to load dataset from local disk.
    If local loading fails, load from Hugging Face.
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


# =========================
# Helper functions
# =========================

def extract_crop_name(label_name):
    """
    Extract crop name from label name.
    Example:
    Tomato_Early_Blight -> Tomato
    Banana_leaf_Panama_Disease -> Banana
    """
    label_name = str(label_name).replace("__", "_")
    crop = label_name.split("_")[0]
    return crop.title()


def split_to_dataframe(ds, split_name):
    """
    Convert one dataset split into a pandas DataFrame with label information.
    """
    label_names = ds[split_name]["label_name"]

    df = pd.DataFrame({
        "split": split_name,
        "label_name": label_names
    })

    df["crop"] = df["label_name"].apply(extract_crop_name)

    return df


def save_bar_plot(df, x_col, y_col, title, xlabel, ylabel, save_path, horizontal=False):
    """
    Save a clean bar plot using matplotlib.
    """
    plt.figure(figsize=(10, 6))

    if horizontal:
        plt.barh(df[x_col], df[y_col])
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.gca().invert_yaxis()
    else:
        plt.bar(df[x_col], df[y_col])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main analysis
# =========================

def main():
    ds = load_dataset_safely()

    print("\nDataset structure:")
    print(ds)

    required_splits = ["train", "valid", "test"]

    for split in required_splits:
        if split not in ds:
            raise ValueError(f"Missing dataset split: {split}")

        if "label_name" not in ds[split].column_names:
            raise ValueError(f"'label_name' column not found in {split} split.")

    # Convert all splits to dataframe
    split_dfs = []

    for split in required_splits:
        split_df = split_to_dataframe(ds, split)
        split_dfs.append(split_df)

    full_df = pd.concat(split_dfs, ignore_index=True)

    # =========================
    # 1. Dataset split summary
    # =========================

    split_summary = (
        full_df
        .groupby("split")
        .size()
        .reset_index(name="image_count")
    )

    split_summary["percentage"] = (
        split_summary["image_count"] / split_summary["image_count"].sum() * 100
    ).round(2)

    split_summary_path = OUTPUT_TABLE_DIR / "dataset_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False, encoding="utf-8-sig")

    print("\nDataset split summary:")
    print(split_summary)

    # =========================
    # 2. Crop-wise distribution
    # =========================

    crop_distribution = (
        full_df
        .groupby("crop")
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
    )

    crop_distribution["percentage"] = (
        crop_distribution["image_count"] / crop_distribution["image_count"].sum() * 100
    ).round(2)

    crop_distribution_path = OUTPUT_TABLE_DIR / "crop_wise_distribution.csv"
    crop_distribution.to_csv(crop_distribution_path, index=False, encoding="utf-8-sig")

    print("\nCrop-wise distribution:")
    print(crop_distribution)

    # =========================
    # 3. Class-wise distribution
    # =========================

    class_distribution = (
        full_df
        .groupby(["crop", "label_name"])
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
    )

    class_distribution["percentage"] = (
        class_distribution["image_count"] / class_distribution["image_count"].sum() * 100
    ).round(4)

    class_distribution_path = OUTPUT_TABLE_DIR / "class_wise_distribution.csv"
    class_distribution.to_csv(class_distribution_path, index=False, encoding="utf-8-sig")

    print("\nClass-wise distribution saved.")

    # =========================
    # 4. Class-wise split distribution
    # =========================

    class_split_distribution = (
        full_df
        .groupby(["label_name", "split"])
        .size()
        .reset_index(name="image_count")
    )

    class_split_pivot = class_split_distribution.pivot_table(
        index="label_name",
        columns="split",
        values="image_count",
        fill_value=0
    ).reset_index()

    for col in required_splits:
        if col not in class_split_pivot.columns:
            class_split_pivot[col] = 0

    class_split_pivot["total"] = (
        class_split_pivot["train"] +
        class_split_pivot["valid"] +
        class_split_pivot["test"]
    )

    class_split_pivot = class_split_pivot.sort_values("total", ascending=False)

    class_split_path = OUTPUT_TABLE_DIR / "class_wise_split_distribution.csv"
    class_split_pivot.to_csv(class_split_path, index=False, encoding="utf-8-sig")

    # =========================
    # 5. Top 10 and bottom 10 classes
    # =========================

    top10_classes = class_distribution.head(10).copy()
    bottom10_classes = class_distribution.tail(10).copy()

    top10_path = OUTPUT_TABLE_DIR / "top10_largest_classes.csv"
    bottom10_path = OUTPUT_TABLE_DIR / "bottom10_smallest_classes.csv"

    top10_classes.to_csv(top10_path, index=False, encoding="utf-8-sig")
    bottom10_classes.to_csv(bottom10_path, index=False, encoding="utf-8-sig")

    print("\nTop 10 largest classes:")
    print(top10_classes)

    print("\nBottom 10 smallest classes:")
    print(bottom10_classes)

    # =========================
    # 6. Imbalance summary
    # =========================

    total_images = len(full_df)
    total_classes = full_df["label_name"].nunique()
    total_crops = full_df["crop"].nunique()

    largest_class = class_distribution.iloc[0]
    smallest_class = class_distribution.iloc[-1]

    imbalance_ratio = round(
        largest_class["image_count"] / smallest_class["image_count"], 2
    )

    report_text = f"""
DATASET DISTRIBUTION ANALYSIS REPORT
============================================================

Total images  : {total_images}
Total crops   : {total_crops}
Total classes : {total_classes}

Dataset split:
{split_summary.to_string(index=False)}

Largest class:
{largest_class['label_name']} = {largest_class['image_count']} images

Smallest class:
{smallest_class['label_name']} = {smallest_class['image_count']} images

Class imbalance ratio:
Largest class / smallest class = {imbalance_ratio}

Interpretation:
The dataset contains multiple crop groups and 94 disease or healthy classes.
However, the distribution of images is not uniform across all classes.
Some classes have substantially higher sample counts, while some classes
have very limited samples. This class imbalance may affect model learning,
especially for minority disease classes. Therefore, class-wise performance
analysis is important for interpreting the model results beyond overall accuracy.
"""

    report_path = OUTPUT_REPORT_DIR / "dataset_distribution_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nImbalance report saved at:", report_path)

    # =========================
    # 7. Save figures
    # =========================

    split_fig_path = OUTPUT_FIGURE_DIR / "dataset_split_distribution.png"
    save_bar_plot(
        df=split_summary,
        x_col="split",
        y_col="image_count",
        title="Dataset Split Distribution",
        xlabel="Dataset Split",
        ylabel="Number of Images",
        save_path=split_fig_path,
        horizontal=False
    )

    crop_fig_path = OUTPUT_FIGURE_DIR / "crop_wise_distribution.png"
    save_bar_plot(
        df=crop_distribution,
        x_col="crop",
        y_col="image_count",
        title="Crop-wise Image Distribution",
        xlabel="Crop",
        ylabel="Number of Images",
        save_path=crop_fig_path,
        horizontal=True
    )

    top10_fig_path = OUTPUT_FIGURE_DIR / "top10_largest_classes.png"
    save_bar_plot(
        df=top10_classes,
        x_col="label_name",
        y_col="image_count",
        title="Top 10 Largest Classes",
        xlabel="Class Name",
        ylabel="Number of Images",
        save_path=top10_fig_path,
        horizontal=True
    )

    bottom10_fig_path = OUTPUT_FIGURE_DIR / "bottom10_smallest_classes.png"
    save_bar_plot(
        df=bottom10_classes.sort_values("image_count", ascending=True),
        x_col="label_name",
        y_col="image_count",
        title="Bottom 10 Smallest Classes",
        xlabel="Class Name",
        ylabel="Number of Images",
        save_path=bottom10_fig_path,
        horizontal=True
    )

    print("\nFigures saved:")
    print(split_fig_path)
    print(crop_fig_path)
    print(top10_fig_path)
    print(bottom10_fig_path)

    print("\nAll dataset distribution analysis outputs generated successfully.")


if __name__ == "__main__":
    main()