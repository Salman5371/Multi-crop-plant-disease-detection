import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from datasets import load_dataset, load_from_disk


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DISK_PATH = ROOT_DIR / "Data" / "processed"
HF_DATASET_NAME = "Saon110/bd-crop-vegetable-plant-disease-dataset"

OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "valid", "test"]


# =========================
# Dataset loading
# =========================

def load_dataset_safely():
    """
    Try local Hugging Face disk dataset first.
    If unavailable, load from Hugging Face Hub.
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
# Hashing functions
# =========================

def exact_image_hash(image):
    """
    Create a stable exact visual hash from RGB image pixels.
    This checks exact pixel-level duplicates after RGB conversion.
    """
    image = image.convert("RGB")
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()


def canonical_png_hash(image):
    """
    Create hash after saving image into a canonical PNG buffer.
    This is useful when original file encoding differs.
    """
    image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.md5(buffer.getvalue()).hexdigest()


def average_hash(image, hash_size=16):
    """
    Simple perceptual average hash.
    This helps identify potential visual duplicates.
    It is not a perfect near-duplicate method, but useful as a screening tool.
    """
    image = image.convert("L").resize((hash_size, hash_size))
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)

    bits = "".join(["1" if p > avg else "0" for p in pixels])
    hex_hash = hex(int(bits, 2))[2:].zfill((hash_size * hash_size) // 4)

    return hex_hash


# =========================
# Main processing
# =========================

def process_split(ds, split_name):
    """
    Process one split and return hash records.
    """
    records = []
    total = len(ds[split_name])

    print(f"\nProcessing split: {split_name} | Total images: {total}")

    for idx, item in enumerate(ds[split_name]):
        image = item["image"]
        label_name = item["label_name"]

        exact_hash = exact_image_hash(image)
        png_hash = canonical_png_hash(image)
        perceptual_hash = average_hash(image, hash_size=16)

        records.append({
            "split": split_name,
            "index": idx,
            "label_name": label_name,
            "exact_hash": exact_hash,
            "png_hash": png_hash,
            "perceptual_hash": perceptual_hash
        })

        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1}/{total} images from {split_name}")

    return records


def find_cross_split_duplicates(df, hash_col):
    """
    Find duplicate hash groups that appear in more than one split.
    """
    grouped = (
        df.groupby(hash_col)
        .agg(
            duplicate_count=("index", "count"),
            split_count=("split", "nunique"),
            splits=("split", lambda x: ", ".join(sorted(set(x)))),
            labels=("label_name", lambda x: " | ".join(sorted(set(x)))[:500])
        )
        .reset_index()
    )

    cross_split_duplicates = grouped[
        (grouped["duplicate_count"] > 1) &
        (grouped["split_count"] > 1)
    ].sort_values("duplicate_count", ascending=False)

    return cross_split_duplicates


def find_all_duplicate_groups(df, hash_col):
    """
    Find all duplicate groups, including within the same split.
    """
    grouped = (
        df.groupby(hash_col)
        .agg(
            duplicate_count=("index", "count"),
            split_count=("split", "nunique"),
            splits=("split", lambda x: ", ".join(sorted(set(x)))),
            labels=("label_name", lambda x: " | ".join(sorted(set(x)))[:500])
        )
        .reset_index()
    )

    duplicate_groups = grouped[
        grouped["duplicate_count"] > 1
    ].sort_values("duplicate_count", ascending=False)

    return duplicate_groups


# =========================
# Main analysis
# =========================

def main():
    print("Starting data leakage / duplicate check...")

    ds = load_dataset_safely()

    for split in SPLITS:
        if split not in ds:
            raise ValueError(f"Missing split: {split}")

        if "image" not in ds[split].column_names:
            raise ValueError(f"'image' column missing in {split}")

        if "label_name" not in ds[split].column_names:
            raise ValueError(f"'label_name' column missing in {split}")

    all_records = []

    for split in SPLITS:
        split_records = process_split(ds, split)
        all_records.extend(split_records)

    hash_df = pd.DataFrame(all_records)

    hash_table_path = OUTPUT_TABLE_DIR / "image_hash_records.csv"
    hash_df.to_csv(hash_table_path, index=False, encoding="utf-8-sig")

    print("\nImage hash records saved:", hash_table_path)

    # Exact RGB hash duplicates
    exact_all_duplicates = find_all_duplicate_groups(hash_df, "exact_hash")
    exact_cross_split_duplicates = find_cross_split_duplicates(hash_df, "exact_hash")

    # Canonical PNG hash duplicates
    png_all_duplicates = find_all_duplicate_groups(hash_df, "png_hash")
    png_cross_split_duplicates = find_cross_split_duplicates(hash_df, "png_hash")

    # Perceptual hash duplicate groups
    perceptual_all_duplicates = find_all_duplicate_groups(hash_df, "perceptual_hash")
    perceptual_cross_split_duplicates = find_cross_split_duplicates(hash_df, "perceptual_hash")

    # Save tables
    exact_all_path = OUTPUT_TABLE_DIR / "exact_duplicate_groups_all_splits.csv"
    exact_cross_path = OUTPUT_TABLE_DIR / "exact_cross_split_duplicates.csv"

    png_all_path = OUTPUT_TABLE_DIR / "png_duplicate_groups_all_splits.csv"
    png_cross_path = OUTPUT_TABLE_DIR / "png_cross_split_duplicates.csv"

    perceptual_all_path = OUTPUT_TABLE_DIR / "perceptual_duplicate_groups_all_splits.csv"
    perceptual_cross_path = OUTPUT_TABLE_DIR / "perceptual_cross_split_duplicates.csv"

    exact_all_duplicates.to_csv(exact_all_path, index=False, encoding="utf-8-sig")
    exact_cross_split_duplicates.to_csv(exact_cross_path, index=False, encoding="utf-8-sig")

    png_all_duplicates.to_csv(png_all_path, index=False, encoding="utf-8-sig")
    png_cross_split_duplicates.to_csv(png_cross_path, index=False, encoding="utf-8-sig")

    perceptual_all_duplicates.to_csv(perceptual_all_path, index=False, encoding="utf-8-sig")
    perceptual_cross_split_duplicates.to_csv(perceptual_cross_path, index=False, encoding="utf-8-sig")

    total_images = len(hash_df)

    exact_cross_count = len(exact_cross_split_duplicates)
    png_cross_count = len(png_cross_split_duplicates)
    perceptual_cross_count = len(perceptual_cross_split_duplicates)

    report_text = f"""
DATA LEAKAGE AND DUPLICATE CHECK REPORT
============================================================

Total images checked: {total_images}

Exact RGB duplicate groups across train/valid/test:
{exact_cross_count}

Canonical PNG duplicate groups across train/valid/test:
{png_cross_count}

Potential perceptual duplicate groups across train/valid/test:
{perceptual_cross_count}

Saved output files:
1. {hash_table_path}
2. {exact_all_path}
3. {exact_cross_path}
4. {png_all_path}
5. {png_cross_path}
6. {perceptual_all_path}
7. {perceptual_cross_path}

Interpretation:
This analysis checks whether identical or potentially visually similar images are
shared across train, validation, and test splits. Cross-split duplicates may cause
data leakage and can artificially increase model performance. Exact hash matching
detects identical RGB image content, while canonical PNG hashing reduces the effect
of file encoding differences. Perceptual hashing is used as a screening method for
potential visual duplicates, but it should be interpreted cautiously because it may
also group visually similar but non-identical images.

If no or very few cross-split duplicate groups are found, the evaluation is less
likely to be affected by duplicate-image leakage. If many cross-split duplicates are
found, this should be reported as a limitation and the dataset should be cleaned
before claiming strong generalization.
"""

    report_path = OUTPUT_REPORT_DIR / "data_leakage_duplicate_check_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nData leakage / duplicate check completed.")
    print("Report saved at:", report_path)

    print("\nSummary:")
    print(f"Total images checked: {total_images}")
    print(f"Exact cross-split duplicate groups: {exact_cross_count}")
    print(f"PNG cross-split duplicate groups: {png_cross_count}")
    print(f"Perceptual cross-split duplicate groups: {perceptual_cross_count}")


if __name__ == "__main__":
    main()