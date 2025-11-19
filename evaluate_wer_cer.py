import csv
from jiwer import wer, cer
import pandas as pd

GROUND_CSV = "ground_truth.csv"
PRED_CSV = "predictions.csv"

OUTPUT_CSV = "wer_cer_results.csv"


def load_column_from_csv(path, column_name):
    df = pd.read_csv(path)
    return df[column_name].astype(str).tolist()


def compute_metrics(gt_list, pr_list):
    results = []

    for i, (gt, pr) in enumerate(zip(gt_list, pr_list)):
        sample_wer = wer(gt, pr)
        sample_cer = cer(gt, pr)

        results.append({
            "index": i,
            "ground_truth": gt,
            "prediction": pr,
            "wer": sample_wer,
            "cer": sample_cer
        })

    avg_wer = sum([r["wer"] for r in results]) / len(results)
    avg_cer = sum([r["cer"] for r in results]) / len(results)

    return results, avg_wer, avg_cer


def save_results(results, avg_wer, avg_cer):
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved per-sample results to:", OUTPUT_CSV)
    print(f"\nAverage WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}\n")


if __name__ == "__main__":

    print("Loading CSV files...")

    gt_list = load_column_from_csv(GROUND_CSV, "ground_truth")
    pr_list = load_column_from_csv(PRED_CSV, "prediction")

    if len(gt_list) != len(pr_list):
        raise ValueError("Ground truth and prediction list lengths do not match!")

    print("Computing WER and CER...")

    results, avg_wer, avg_cer = compute_metrics(gt_list, pr_list)

    save_results(results, avg_wer, avg_cer)

    print("DONE.")
