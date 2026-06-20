from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


PREDICTION_FILES = {
    "MobileNetV2": OUTPUT_TABLE_DIR / "mobilenetv2_test_predictions.csv",
    "ResNet50": OUTPUT_TABLE_DIR / "resnet50_test_predictions.csv",
    "EfficientNetV2-S": OUTPUT_TABLE_DIR / "efficientnetv2s_test_predictions_for_stats.csv",
}


# =========================
# Helper functions
# =========================

def load_prediction_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["sample_index", "true_label_id", "predicted_label_id", "correct"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")

    df = df.sort_values("sample_index").reset_index(drop=True)
    df["correct"] = df["correct"].astype(bool)

    return df


def run_mcnemar_test(df_a, df_b, model_a, model_b):
    """
    McNemar test on paired model predictions.

    Table:
    b00: both correct
    b01: model A correct, model B wrong
    b10: model A wrong, model B correct
    b11: both wrong
    """
    if len(df_a) != len(df_b):
        raise ValueError("Prediction files have different number of samples.")

    if not (df_a["sample_index"].values == df_b["sample_index"].values).all():
        raise ValueError("Sample indices do not match between prediction files.")

    a_correct = df_a["correct"].values
    b_correct = df_b["correct"].values

    both_correct = ((a_correct == True) & (b_correct == True)).sum()
    a_correct_b_wrong = ((a_correct == True) & (b_correct == False)).sum()
    a_wrong_b_correct = ((a_correct == False) & (b_correct == True)).sum()
    both_wrong = ((a_correct == False) & (b_correct == False)).sum()

    table = [
        [both_correct, a_correct_b_wrong],
        [a_wrong_b_correct, both_wrong]
    ]

    # exact=True is safer when discordant pairs are small
    result = mcnemar(table, exact=True)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "both_correct": int(both_correct),
        "model_a_correct_model_b_wrong": int(a_correct_b_wrong),
        "model_a_wrong_model_b_correct": int(a_wrong_b_correct),
        "both_wrong": int(both_wrong),
        "mcnemar_statistic": result.statistic,
        "p_value": result.pvalue,
        "significant_at_0_05": result.pvalue < 0.05
    }


def main():
    prediction_dfs = {}

    for model_name, path in PREDICTION_FILES.items():
        prediction_dfs[model_name] = load_prediction_file(path)
        acc = prediction_dfs[model_name]["correct"].mean()
        print(f"{model_name} accuracy from prediction file: {acc:.4f}")

    comparisons = [
        ("EfficientNetV2-S", "ResNet50"),
        ("EfficientNetV2-S", "MobileNetV2"),
        ("ResNet50", "MobileNetV2"),
    ]

    results = []

    for model_a, model_b in comparisons:
        result = run_mcnemar_test(
            df_a=prediction_dfs[model_a],
            df_b=prediction_dfs[model_b],
            model_a=model_a,
            model_b=model_b
        )

        results.append(result)

        print("\n" + "=" * 70)
        print(f"Comparison: {model_a} vs {model_b}")
        print("=" * 70)
        print(f"{model_a} correct, {model_b} wrong: {result['model_a_correct_model_b_wrong']}")
        print(f"{model_a} wrong, {model_b} correct: {result['model_a_wrong_model_b_correct']}")
        print(f"p-value: {result['p_value']}")
        print(f"Significant at 0.05: {result['significant_at_0_05']}")

    results_df = pd.DataFrame(results)

    output_csv = OUTPUT_TABLE_DIR / "statistical_significance_mcnemar_results.csv"
    results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report_path = OUTPUT_REPORT_DIR / "statistical_significance_test_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STATISTICAL SIGNIFICANCE TEST REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Test used: McNemar's test\n")
        f.write("Purpose: Paired comparison of model correctness on the same test samples.\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write(
            "McNemar's test was used to assess whether the difference in prediction "
            "correctness between two models is statistically significant on the same "
            "test set. A p-value below 0.05 indicates that the difference between the "
            "two models is statistically significant.\n"
        )

    print("\nStatistical significance results saved:")
    print(output_csv)
    print(report_path)


if __name__ == "__main__":
    main()