import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

import src.models as model_zoo


# =========================
# Paths and configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 94
IMAGE_SIZE = 224
BATCH_SIZE = 1
WARMUP_RUNS = 10
TIMING_RUNS = 50


# =========================
# Model configuration
# =========================

MODEL_CONFIGS = {
    "MobileNetV2": {
        "builder_aliases": ["build_mobilenetv2", "build_mobile_net_v2"],
        "weight_path": MODEL_DIR / "mobilenetv2_best.pth",
        "accuracy": 0.9453,
        "f1_score": 0.9444
    },
    "ResNet50": {
        "builder_aliases": ["build_resnet50", "build_resnet_50"],
        "weight_path": MODEL_DIR / "resnet50_best.pth",
        "accuracy": 0.9555,
        "f1_score": 0.9550
    },
    "EfficientNetV2-S": {
        "builder_aliases": ["build_efficientnetv2s", "build_efficientnet_v2_s", "build_efficientnet2s"],
        "weight_path": MODEL_DIR / "efficientnetv2s_best.pth",
        "accuracy": 0.9575,
        "f1_score": 0.9571
    }
}


# =========================
# Helper functions
# =========================

def find_model_builder(builder_aliases):
    """
    Find model builder function from src.models using possible function names.
    """
    for name in builder_aliases:
        if hasattr(model_zoo, name):
            return getattr(model_zoo, name), name

    raise AttributeError(
        f"No matching model builder found in src.models. Tried: {builder_aliases}"
    )


def load_checkpoint_if_available(model, weight_path):
    """
    Load model checkpoint if available.
    If loading fails, continue with architecture only for parameter and speed analysis.
    """
    if not weight_path.exists():
        print(f"Warning: Weight file not found: {weight_path}")
        return model, "Weight file not found"

    try:
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
        return model, "Loaded successfully"

    except Exception as e:
        print(f"Warning: Could not load weights for {weight_path.name}")
        print("Reason:", e)
        return model, f"Weight loading failed: {e}"


def count_parameters(model):
    """
    Count total and trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_file_size_mb(path):
    """
    Get model weight file size in MB.
    """
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return round(size_mb, 2)

    return None


def measure_inference_time(model):
    """
    Measure average inference time per image in milliseconds.
    """
    model.to(DEVICE)
    model.eval()

    dummy_input = torch.randn(
        BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE
    ).to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            _ = model(dummy_input)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    avg_time_sec = (end_time - start_time) / TIMING_RUNS
    avg_time_ms = avg_time_sec * 1000

    return round(avg_time_ms, 3)


def estimate_fps(inference_time_ms):
    """
    Estimate frames/images per second.
    """
    if inference_time_ms <= 0:
        return None

    fps = 1000 / inference_time_ms
    return round(fps, 2)


# =========================
# Main analysis
# =========================

def main():
    print("Using device:", DEVICE)
    print("Running model efficiency analysis...")

    results = []

    for model_name, config in MODEL_CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"Analyzing model: {model_name}")
        print("=" * 70)

        builder_fn, builder_name = find_model_builder(config["builder_aliases"])
        print(f"Using builder function: {builder_name}")

        model = builder_fn(num_classes=NUM_CLASSES)

        model, loading_status = load_checkpoint_if_available(
            model=model,
            weight_path=config["weight_path"]
        )

        model.to(DEVICE)

        total_params, trainable_params = count_parameters(model)
        model_size_mb = get_file_size_mb(config["weight_path"])
        inference_time_ms = measure_inference_time(model)
        fps = estimate_fps(inference_time_ms)

        result = {
            "model": model_name,
            "builder_function": builder_name,
            "total_parameters": total_params,
            "total_parameters_million": round(total_params / 1_000_000, 3),
            "trainable_parameters": trainable_params,
            "trainable_parameters_million": round(trainable_params / 1_000_000, 3),
            "model_size_mb": model_size_mb,
            "inference_time_ms_per_image": inference_time_ms,
            "estimated_fps": fps,
            "accuracy": config["accuracy"],
            "f1_score": config["f1_score"],
            "weight_loading_status": loading_status
        }

        results.append(result)

        print(f"Total parameters        : {result['total_parameters_million']} M")
        print(f"Trainable parameters    : {result['trainable_parameters_million']} M")
        print(f"Model size              : {model_size_mb} MB")
        print(f"Inference time/image    : {inference_time_ms} ms")
        print(f"Estimated FPS           : {fps}")
        print(f"Accuracy                : {config['accuracy']}")
        print(f"F1-score                : {config['f1_score']}")
        print(f"Weight loading status   : {loading_status}")

    df = pd.DataFrame(results)

    output_csv = OUTPUT_TABLE_DIR / "model_efficiency_analysis.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("MODEL EFFICIENCY ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"CSV saved at: {output_csv}")

    # Save text report
    report_path = OUTPUT_REPORT_DIR / "model_efficiency_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODEL EFFICIENCY ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Device used: {DEVICE}\n")
        f.write(f"Input image size: {IMAGE_SIZE} x {IMAGE_SIZE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Timing runs: {TIMING_RUNS}\n\n")

        f.write(df.to_string(index=False))

        f.write("\n\nInterpretation:\n")
        f.write(
            "The efficiency analysis compares the three trained models in terms of "
            "parameter count, model size, inference time, accuracy, and F1-score. "
            "This analysis is important for smart farming applications because a model "
            "should not only be accurate but also computationally practical for real-world "
            "decision-support systems.\n"
        )

    print(f"Report saved at: {report_path}")


if __name__ == "__main__":
    main()