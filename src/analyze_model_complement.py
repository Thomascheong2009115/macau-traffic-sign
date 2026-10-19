import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth


DETAIL_FIELDS = [
    "IMG",
    "true_class",
    "final_pred",
    "wavelet_pred",
    "final_correct",
    "wavelet_correct",
    "result",
]
SUMMARY_FIELDS = ["section", "key", "value"]


def _field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], mapping: dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return row.get(mapping[key], "").strip()
    return default


def read_debug_predictions(path: str | Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            pred = _get(row, mapping, ("final_class", "Class", "class", "top1_class", "main_top1_class", "top1", "pred", "prediction"))
            if img:
                predictions[img] = pred
    return predictions


def analyze_model_complement(
    final_debug_csv: str | Path,
    wavelet_debug_csv: str | Path,
    gt_csv: str | Path,
    output_csv: str | Path = "output/model_complement_final_vs_wavelet_high.csv",
    summary_csv: str | Path = "output/model_complement_summary.csv",
) -> dict[str, int | float]:
    truth = read_ground_truth(gt_csv)
    final_predictions = read_debug_predictions(final_debug_csv)
    wavelet_predictions = read_debug_predictions(wavelet_debug_csv)

    metrics: dict[str, int | float] = {
        "both_correct": 0,
        "final_only_correct": 0,
        "wavelet_only_correct": 0,
        "both_wrong": 0,
        "union_correct": 0,
        "total": 0,
    }
    rescue_classes: Counter[str] = Counter()
    damage_classes: Counter[str] = Counter()
    detail_rows: list[dict[str, str]] = []

    for img, true_label in truth.items():
        if img not in final_predictions or img not in wavelet_predictions:
            continue
        final_pred = final_predictions[img]
        wavelet_pred = wavelet_predictions[img]
        final_correct = final_pred == true_label
        wavelet_correct = wavelet_pred == true_label
        metrics["total"] += 1
        if final_correct and wavelet_correct:
            result = "both_correct"
            metrics["both_correct"] += 1
        elif final_correct and not wavelet_correct:
            result = "final_only_correct"
            metrics["final_only_correct"] += 1
            damage_classes[true_label] += 1
        elif wavelet_correct and not final_correct:
            result = "wavelet_only_correct"
            metrics["wavelet_only_correct"] += 1
            rescue_classes[true_label] += 1
        else:
            result = "both_wrong"
            metrics["both_wrong"] += 1
        if final_correct or wavelet_correct:
            metrics["union_correct"] += 1
        detail_rows.append(
            {
                "IMG": img,
                "true_class": true_label,
                "final_pred": final_pred,
                "wavelet_pred": wavelet_pred,
                "final_correct": str(final_correct),
                "wavelet_correct": str(wavelet_correct),
                "result": result,
            }
        )

    total = int(metrics["total"])
    metrics["union_accuracy"] = float(metrics["union_correct"]) / total if total else 0.0

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows: list[dict[str, str]] = []
    for key in ("both_correct", "final_only_correct", "wavelet_only_correct", "both_wrong", "union_correct", "union_accuracy", "total"):
        summary_rows.append({"section": "metric", "key": key, "value": str(metrics[key])})
    for label, count in sorted(rescue_classes.items()):
        summary_rows.append({"section": "wavelet_rescue_class", "key": label, "value": str(count)})
    for label, count in sorted(damage_classes.items()):
        summary_rows.append({"section": "wavelet_damage_class", "key": label, "value": str(count)})

    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="analyze final vs wavelet-high complementarity")
    parser.add_argument("--final_debug", default="output/debug_latest.csv")
    parser.add_argument("--wavelet_debug", default="output/wavelet_ablation/high/debug_predictions.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/model_complement_final_vs_wavelet_high.csv")
    parser.add_argument("--summary_csv", default="output/model_complement_summary.csv")
    args = parser.parse_args()
    metrics = analyze_model_complement(args.final_debug, args.wavelet_debug, args.gt, args.output_csv, args.summary_csv)
    for key in ("both_correct", "final_only_correct", "wavelet_only_correct", "both_wrong", "union_correct", "union_accuracy"):
        print(f"{key}={metrics[key]}")


if __name__ == "__main__":
    main()
